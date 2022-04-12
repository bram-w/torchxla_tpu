import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.metrics as met
import torch_xla
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import sys
import os
import webdataset as wds
import datetime
import time
from itertools import islice
import torch_xla.debug.profiler as xp
from google.cloud import storage
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.blob import Blob

from orig_clip import clip
import random


for extra in ('/usr/share/torch-xla-1.8/pytorch/xla/test', '/pytorch/xla/test', '/usr/share/pytorch/xla/test'):
    if os.path.exists(extra):
        sys.path.insert(0, extra)

import schedulers
import args_parse 


MODEL_OPTS = {
    '--model': {
        'type': str,
        'default': 'ViT-B/32',
    },
    '--test_set_batch_size': {
        'type': int,
    },
    '--lr_scheduler_type': {
        'type': str,
        'default' : 'WarmupAndExponentialDecayScheduler'
    },
    '--lr_scheduler_divide_every_n_epochs': {
        'type': int,
        'default' : 20,
    },
    '--lr_scheduler_divisor': {
        'type': int,
        'default' : 5,
    },
    '--dataset': {
        'choices': ['gcsdataset', 'torchdataset'],
        'default': 'gcsdataset',
        'type': str,
    },
    '--wds_traindir': {
        'type': str,
        'default':'/tmp/cifar',
    },
    '--wds_testdir': {
        'type': str,
        'default': '/tmp/cifar',
    },
    '--trainsize': {
        'type': int,
        'default': 1280000,
    },
    '--testsize': {
        'type': int,
        'default': 50000,
    },
    '--wd': {
        'type': float,
        'default': 0.0,
    },
    '--dropout': {
        'type': float,
        'default': 0.0,
    },
    '--optim': {
        'type': str,
        'default': "SGD",
    },
    '--save_model': {
        'type': str,
        'default': "",
    },
    '--load_chkpt_file': {
        'type': str,
        'default': "",
    },
    '--load_chkpt_dir': {
        'type': str,
        'default': "",
    },
    '--model_bucket': {
        'type': str,
        'default': "",
    },
    '--upload_chkpt': {
        'action': 'store_true',
    },
    '--pretrained': {
        'action': 'store_true',
    },
}

        
FLAGS = args_parse.parse_common_options(
    datadir='/tmp/imagenet',
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    opts=MODEL_OPTS.items(),
    profiler_port=9012,
)



DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=64,
    num_epochs=18,
    momentum=0.9,
    lr=0.001,
    wd=0.00,
    target_accuracy=0.0,
)
MODEL_SPECIFIC_DEFAULTS = {
    # Override some of the args in DEFAULT_KWARGS, or add them to the dict
    # if they don't exist.
}

# Set any args that were not explicitly given by the user.
default_value_dict = MODEL_SPECIFIC_DEFAULTS.get(FLAGS.model, DEFAULT_KWARGS)
for arg, value in default_value_dict.items():
    if getattr(FLAGS, arg) is None:
        setattr(FLAGS, arg, value)


def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer)

trainsize = FLAGS.trainsize # 1280000
testsize = FLAGS.testsize # 50000 

def _upload_blob_gcs(gcs_uri, source_file_name, destination_blob_name):
    """Uploads a file to GCS bucket"""
    client = storage.Client()
    blob = Blob.from_string(os.path.join(gcs_uri, destination_blob_name))
    blob.bucket._client = client
    blob.upload_from_filename(source_file_name)
    
    xm.master_print("Saved Model Checkpoint file {} and uploaded to {}.".format(source_file_name, os.path.join(gcs_uri, destination_blob_name)))
    
def _read_blob_gcs(BUCKET, CHKPT_FILE, DESTINATION):
    """Downloads a file from GCS to local directory"""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET)
    blob = bucket.get_blob(CHKPT_FILE)
    blob.download_to_filename(DESTINATION)
    
def identity(x):
    return x   

def my_worker_splitter(urls):
    """Split urls per worker
    Selects a subset of urls based on Torch get_worker_info.
    Used as a shard selection function in Dataset.
    replaces wds.split_by_worker"""
    # import torch

    urls = [url for url in urls]

    assert isinstance(urls, list)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers

        return urls[wid::num_workers]
    else:
        return urls

def my_node_splitter(urls):
    """Split urls_ correctly per accelerator node
    :param urls:
    :return: slice of urls_
    """
    rank=xm.get_ordinal()
    num_replicas=xm.xrt_world_size()

    urls_this = urls[rank::num_replicas]
    
    return urls_this



def make_train_loader(image_transform,
                      shuffle=10000, batch_size=FLAGS.batch_size):
    num_dataset_instances = xm.xrt_world_size() * FLAGS.num_workers
    epoch_size = trainsize // num_dataset_instances

    dataset = (
        wds.WebDataset(FLAGS.wds_traindir, 
                       splitter=my_worker_splitter, 
                       nodesplitter=my_node_splitter, 
                       shardshuffle=True, length=epoch_size)
        .shuffle(shuffle)
        .decode("pil")
        .to_tuple("ppm;jpg;jpeg;png", "txt")
        .map_tuple(image_transform, random.choice)
        .batched(batch_size, partial=True)
        )

    loader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, drop_last=False, num_workers=FLAGS.num_workers)
    return loader
  
def make_val_loader(image_transform, batch_size=FLAGS.test_set_batch_size):
    num_dataset_instances = xm.xrt_world_size() * FLAGS.num_workers
    epoch_test_size = testsize // num_dataset_instances

    val_dataset = (
        wds.WebDataset(FLAGS.wds_testdir, # FLAGS.wds_testdir, 
                       splitter=my_worker_splitter, 
                       nodesplitter=my_node_splitter, 
                       shardshuffle=False, length=epoch_test_size) 
        .decode("pil")
        .to_tuple("ppm;jpg;jpeg;png", "txt")
        .map_tuple(image_transform, random.choice)
        .batched(batch_size, partial=True)
    )

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=FLAGS.num_workers) 
    return val_loader

    
def train_imagenet():
    print("TODO: Check 'shuffle' args")
    print("TODO: Sort out SGD (mom) vs Adam optimizer")
    print("TODO: Base saving off of loss instead of acc?")
    print('==> Preparing data..')
    

    torch.manual_seed(42)
    server = xp.start_server(FLAGS.profiler_port)

    device = xm.xla_device()
    model, preprocess = clip.load(FLAGS.model, 
                                  load_pretrained_weights=FLAGS.pretrained)
    model = model.to(device)
    # if 'freq' in FLAGS.model: model.conv_proj = PatchDCT(16, 3)
        
        
    train_loader = make_train_loader(preprocess,
                                     batch_size=FLAGS.batch_size,
                                     shuffle=10000)
    test_loader = make_val_loader(preprocess,
                                  batch_size=FLAGS.test_set_batch_size)
    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(FLAGS.logdir)
    optim_call = eval(f"optim.{FLAGS.optim}")
    
    # Below is ugly, should just make kwarg dict
    # and delete momentum if feeding to Adam
    if FLAGS.optim == 'SGD':
        optimizer = optim_call(
            model.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
            weight_decay=FLAGS.wd)
    elif FLAGS.optim == 'Adam':
        optimizer = optim_call(
            model.parameters(),
            lr=FLAGS.lr,
            weight_decay=FLAGS.wd)
    else:
        raise NotImplementedERror
        
    num_training_steps_per_epoch = trainsize // (
        FLAGS.batch_size * xm.xrt_world_size())
    lr_scheduler = schedulers.wrap_optimizer_with_scheduler(
        optimizer,
        scheduler_type=getattr(FLAGS, 'lr_scheduler_type', None),
        scheduler_divisor=getattr(FLAGS, 'lr_scheduler_divisor', None),
        scheduler_divide_every_n_epochs=getattr(
            FLAGS, 'lr_scheduler_divide_every_n_epochs', None),
        num_steps_per_epoch=num_training_steps_per_epoch,
        summary_writer=writer)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint = None
    if FLAGS.load_chkpt_file != "":
        xm.master_print("Attempting Restart from {}".format(FLAGS.load_chkpt_file))
        if FLAGS.model_bucket:
            _read_blob_gcs(FLAGS.model_bucket, FLAGS.load_chkpt_file, FLAGS.load_chkpt_dir)
            checkpoint = torch.load(FLAGS.load_chkpt_dir)
            xm.master_print("Loading saved model {}".format(FLAGS.load_chkpt_file))
        elif os.path.exists(FLAGS.load_chkpt_file):
            checkpoint = torch.load(FLAGS.load_chkpt_file)
        if checkpoint is not None:
            xm.master_print("FOUND: Restarting from {}".format(FLAGS.load_chkpt_file))
            model.load_state_dict(checkpoint['model_state_dict']) #.to(device)
            model = model.to(device)
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
        else:
            xm.master_print("No restart checkpoint found")
 
          

          
          
    def train_loop_fn(loader, epoch):
        train_steps = trainsize // (FLAGS.batch_size * xm.xrt_world_size())
        tracker = xm.RateTracker()
        total_samples = 0
        model.train()
        for step, (imgs, txts) in enumerate(loader):
            txts = clip.tokenize(txts).to(device)
            imgs = imgs.to(device)
            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(imgs, txts.squeeze())
            target = torch.arange(txts.shape[0]).to(device)
            img_loss = F.cross_entropy(logits_per_image, target)
            txt_loss = F.cross_entropy(logits_per_text, target)
            loss = (img_loss + txt_loss ) / 2
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS.batch_size)
            total_samples += imgs.size()[0]
            if lr_scheduler:
                lr_scheduler.step()
            if step % FLAGS.log_steps == 0:
                xm.add_step_closure(
                    _train_update, args=(device, step, loss, tracker, epoch, writer))
                test_utils.write_to_summary(writer, step, dict_to_write={'Rate_step': tracker.rate()}, write_xla_metrics=False)
            if step == train_steps:
                break   
        
        reduced_global = xm.mesh_reduce('reduced_global', tracker.global_rate(), np.mean)

        return total_samples, reduced_global                                   


          
    def test_loop_fn(loader, epoch):
        test_steps = testsize // (FLAGS.test_set_batch_size * xm.xrt_world_size())
        total_local_samples, correct = 0, 0
        model.eval()
        for step, (imgs, txts) in enumerate(loader):
            txts = clip.tokenize(txts).to(device)
            imgs = imgs.to(device)
            logits_per_image, logits_per_text = model(imgs, txts.squeeze())
            target = torch.arange(txts.shape[0]).to(device)
            img_loss = F.cross_entropy(logits_per_image, target)
            txt_loss = F.cross_entropy(logits_per_text, target)
            loss = (img_loss + txt_loss ) / 2
          
            pred = logits_per_image.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
          
            total_local_samples += imgs.size()[0]
            if step % FLAGS.log_steps == 0:
                xm.add_step_closure(
                    test_utils.print_test_update, args=(device, None, epoch, step))
            if step == test_steps:
                break
        accuracy_replica = 100.0 * correct / total_local_samples
          
        # Modified this so can pass empty urls
        # accuracy = xm.mesh_reduce('test_accuracy', accuracy_replica, np.mean)
        # Have weird naming here b/c don't know if collisions are OK
        total_correct = xm.mesh_reduce('all_num_correct', correct, np.sum)
        total_preds = xm.mesh_reduce('all_num_samples', total_local_samples, np.sum)
        accuracy = 100.0 * total_correct / total_preds
        return accuracy, accuracy_replica, total_preds

    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    test_device_loader = pl.MpDeviceLoader(test_loader, device)
    accuracy, max_accuracy = 0.0, 0.0
    training_start_time = time.time()
    
    if checkpoint is not None:
        best_valid_acc = checkpoint['best_valid_acc']
        start_epoch = checkpoint['epoch'] + 1
        xm.master_print('Loaded Model CheckPoint: Epoch={}/{}, Val Accuracy={:.2f}%'.format(
            start_epoch, FLAGS.num_epochs, best_valid_acc))
    else:
        best_valid_acc = 0.0
        start_epoch = 1
    
    for epoch in range(start_epoch, FLAGS.num_epochs + 1):
        xm.master_print('Epoch {} train begin {}'.format(
            epoch, test_utils.now()))
        replica_epoch_start = time.time()
        
        xm.master_print("Surprise val loop!!!!")
        accuracy, accuracy_replica, replica_test_samples = test_loop_fn(test_device_loader, epoch)
        
        replica_train_samples, reduced_global = train_loop_fn(train_device_loader, epoch)
        xm.master_print("Done with train loop")
        replica_epoch_time = time.time() - replica_epoch_start
        avg_epoch_time_mesh = xm.mesh_reduce('epoch_time', replica_epoch_time, np.mean)
        reduced_global = reduced_global * xm.xrt_world_size()
        xm.master_print('Epoch {} train end {}, Epoch Time={}, Replica Train Samples={}, Reduced GlobalRate={:.2f}'.format(
            epoch, test_utils.now(), 
            str(datetime.timedelta(seconds=avg_epoch_time_mesh)).split('.')[0], 
            replica_train_samples, 
            reduced_global))
        
        accuracy, accuracy_replica, replica_test_samples = test_loop_fn(test_device_loader, epoch)
        xm.master_print('Epoch {} test end {}, Reduced Accuracy={:.2f}%, Replica Accuracy={:.2f}%, Replica Test Samples={}'.format(
            epoch, test_utils.now(), 
            accuracy, accuracy_replica, 
            replica_test_samples))
        
        if FLAGS.save_model != "":
            if accuracy > best_valid_acc:
                xm.master_print('Epoch {} validation accuracy improved from {:.2f}% to {:.2f}% - saving model...'.format(epoch, best_valid_acc, accuracy))
                best_valid_acc = accuracy
                xm.save(
                    {
                        "epoch": epoch,
                        "nepochs": FLAGS.num_epochs,
                        "model_state_dict": model.state_dict(),
                        "best_valid_acc": best_valid_acc,
                        "opt_state_dict": optimizer.state_dict(),
                    },
                    FLAGS.save_model,
                )
                if xm.is_master_ordinal() and FLAGS.upload_chkpt:
                    _upload_blob_gcs(FLAGS.logdir, FLAGS.save_model, 'model-chkpt.pt')
            xm.save(
                {
                    "epoch": epoch,
                    "nepochs": FLAGS.num_epochs,
                    "model_state_dict": model.state_dict(),
                    "curr_valid_acc": accuracy,
                    "best_valid_acc": best_valid_acc,
                    "opt_state_dict": optimizer.state_dict(),
                },
                'CURRENT_' + FLAGS.save_model,
                 )
        
        max_accuracy = max(accuracy, max_accuracy)
        test_utils.write_to_summary(
            writer,
            epoch,
            dict_to_write={'Accuracy/test': accuracy,
                           'Global Rate': reduced_global},
            write_xla_metrics=False)
        if FLAGS.metrics_debug:
            xm.master_print(met.metrics_report())
    test_utils.close_summary_writer(writer)
    total_train_time = time.time() - training_start_time
    xm.master_print('Total Train Time: {}'.format(str(datetime.timedelta(seconds=total_train_time)).split('.')[0]))    
    xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
    xm.master_print('Avg. Global Rate: {:.2f} examples per second'.format(reduced_global))
    return max_accuracy


def _mp_fn(index, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    accuracy = train_imagenet()
    if accuracy < FLAGS.target_accuracy:
        print('Accuracy {} is below target {}'.format(accuracy,
                                                      FLAGS.target_accuracy))
        sys.exit(21)


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores, start_method='fork') # , start_method='spawn'
