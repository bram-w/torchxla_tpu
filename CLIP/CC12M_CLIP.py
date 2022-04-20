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
from orig_clip import model as clip_model_lib
import random

from arch_settings import model_to_settings

"""
Orig  CLIP paper is 400M images, 32 epochs, batchsize ~32k
So about 400k steps
(400M / 32k steps/epoch ~ 13k steps /epoch, or just cancel 32s and see)



CLIP id across 256-592 GPUs with subsets of pairwise similarities
So <128 examples per core. Diverse dataset though

So I'm going to try 512 cores with batch size 128/core
Over 64 workers would be 1024/worker, hitting 65536, double that of CLIP


ADDED:
- Temperature CLIPPING
- Letting FLAGS handle script-administration but putting arch-specific params in arch_settings.py
- Handling/understanding of scaling with global batch size
"""


for extra in ('/usr/share/torch-xla-1.8/pytorch/xla/test', '/pytorch/xla/test', '/usr/share/pytorch/xla/test'):
    if os.path.exists(extra):
        sys.path.insert(0, extra)

from my_lr_scheduler import LinearWarmupCosineAnnealingLR
import args_parse 

batch_size = 128
num_workers = 8


MODEL_OPTS = {
    '--model': {
        'type': str,
        'default': 'ViT-B/32',
    },
    '--wds_traindir': {
        'type': str,
        'default':'/tmp/cifar',
    },
    '--wds_testdir': {
        'type': str,
        'default': '/tmp/cifar',
    },
    '--save_model': {
        'type': str,
        'default': "",
    },
    '--load_ckpt_file': {
        'type': str,
        'default': "current.ckpt",
    },
    '--load_ckpt_dir': {
        'type': str,
        'default': "",
    },
    '--model_bucket': {
        'type': str,
        'default': "",
    },
    '--upload_ckpt': {
        'action': 'store_true',
    },
}

        
FLAGS = args_parse.parse_common_options(
    datadir='/tmp/imagenet',
    batch_size=None,
    num_epochs=3906,
    momentum=None,
    lr=None,
    target_accuracy=None,
    opts=MODEL_OPTS.items(),
    profiler_port=9012,
)




DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=None,
    num_epochs=3906,
    momentum=None,
    lr=None,
    wd=None,
    target_accuracy=0.0,
)
MODEL_SPECIFIC_DEFAULTS = {
    # Override some of the args in DEFAULT_KWARGS, or add them to the dict
    # if they don't exist.
}



def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer)

trainsize = int((400*1e6) * (32)) # This is 250k steps at 1024 --> 256 million
assert 1000 % FLAGS.log_steps == 0 # need to hit below logic

def _upload_blob_gcs(gcs_uri, source_file_name, destination_blob_name):
    """Uploads a file to GCS bucket"""
    client = storage.Client()
    blob = Blob.from_string(os.path.join(gcs_uri, destination_blob_name))
    blob.bucket._client = client
    blob.upload_from_filename(source_file_name)
    
    xm.master_print("Saved Model Checkpoint file {} and uploaded to {}.".format(source_file_name, os.path.join(gcs_uri, destination_blob_name)))
    
def _read_blob_gcs(BUCKET, ckpt_FILE, DESTINATION):
    """Downloads a file from GCS to local directory"""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET)
    blob = bucket.get_blob(ckpt_FILE)
    blob.download_to_filename(DESTINATION)
    
def identity(x):
    return x   

def split_and_choose(x):
    return random.choice(x.split('CAPTIONBREAK'))

def make_train_loader(image_transform,
                      shuffle=10000, batch_size=batch_size):
    num_dataset_instances = xm.xrt_world_size() * num_workers
    epoch_size = trainsize // num_dataset_instances

    dataset = wds.DataPipeline(
        wds.ResampledShards(FLAGS.wds_traindir),
        # we now have an iterator over all shards
        wds.tarfile_to_samples(),
        wds.shuffle(10000),
        wds.decode("pil"),
        # we now have a list of decompressed train samples from each shard in this worker, in sequence
        wds.to_tuple("ppm;jpg;jpeg;png", "txt"),
        wds.map_tuple(image_transform, identity),
        wds.batched(batch_size)
        ).with_epoch(epoch_size).with_length(epoch_size) # adds `__len__` method to dataset

    loader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)
    loader = loader.with_length(epoch_size) # adds `__len__` method to dataloader

    return loader      
    

    
def train_imagenet():
    print('==> Preparing data..')
    

    torch.manual_seed(42)
    server = xp.start_server(FLAGS.profiler_port)

    device = xm.xla_device()
    rank=xm.get_ordinal()
    model = clip_model_lib.CLIP(**model_to_settings[FLAGS.model][0]).to(device)
    preprocess_train = transforms.Compose([transforms.RandomResizedCrop(224, (0.9, 1)), # 0.9 from mlfoundations (transorm.py)
                                         clip._convert_image_to_rgb,
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                              std=(0.26862954, 0.26130258, 0.27577711))
                                        ])
    preprocess_val = transforms.Compose([transforms.Resize(224),
                                         transforms.CenterCrop(size=(224,224)),
                                         clip._convert_image_to_rgb,
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                              std=(0.26862954, 0.26130258, 0.27577711)),
                                        ])
                                           
    
    # if 'freq' in FLAGS.model: model.conv_proj = PatchDCT(16, 3)

    train_loader = make_train_loader(preprocess_train,
                                     batch_size=batch_size,
                                     shuffle=10000)
    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(FLAGS.logdir)
        
    full_batch_size = batch_size * xm.xrt_world_size()
    num_training_steps_per_epoch = trainsize // full_batch_size
    
    opt_hparam_dict = model_to_settings[FLAGS.model][1]
    optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_hparam_dict['lr'] * (full_batch_size / 32768),
            weight_decay=2e-2,
            betas=(0.9, opt_hparam_dict['adam_beta2']),
            eps=opt_hparam_dict['adam_eps']
            )
    
    
    lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                 num_training_steps_per_epoch,
                                                 int(2000 * (32768/full_batch_size)))
    loss_fn = nn.CrossEntropyLoss()
    checkpoint = None
    start_step = 0
    if FLAGS.load_ckpt_file != "":
        xm.master_print("Attempting Restart from {}".format(FLAGS.load_ckpt_file))
        if FLAGS.model_bucket:
            raise NotImplementedError
            _read_blob_gcs(FLAGS.model_bucket, FLAGS.load_ckpt_file, FLAGS.load_ckpt_dir)
            checkpoint = torch.load(FLAGS.load_ckpt_dir)
            xm.master_print("Loading saved model {}".format(FLAGS.load_ckpt_file))
        elif os.path.exists(FLAGS.load_ckpt_file):
            xm.master_print("FOUND LOCAL FILE")
            checkpoint = torch.load(FLAGS.load_ckpt_file)
            
        if checkpoint is not None:
            xm.master_print("FOUND: Restarting from {}".format(FLAGS.load_ckpt_file))
            model.load_state_dict(checkpoint['model_state_dict']) #.to(device)
            model = model.to(device)
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
            lr_scheduler_state_dict = checkpoint['lr_scheduler_state_dict']
            lr_scheduler.load_state_dict(lr_scheduler_state_dict)
            start_step = lr_scheduler_state_dict['_step_count']
        else:
            xm.master_print("No restart checkpoint found")
 
          

          
          
    def train_loop_fn(loader, epoch):
        train_steps = trainsize // (batch_size * xm.xrt_world_size())
        if start_step >= train_steps: return 0, 0
        tracker = xm.RateTracker()
        total_samples = 0
        model.train()
        for raw_step, (imgs, txts_raw) in enumerate(loader):
            # Below is needed to maintain good CLIP stability and isn't directly in model
            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            step = raw_step + start_step
            optimizer.zero_grad()
            txts = clip.tokenize(txts_raw).to(xm.xla_device())
            logits_per_image, logits_per_text = model(imgs, txts.squeeze())
            target = torch.arange(txts.shape[0], device=xm.xla_device())
            img_loss = F.cross_entropy(logits_per_image, target)
            txt_loss = F.cross_entropy(logits_per_text, target)
            loss = (img_loss + txt_loss ) / 2
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(batch_size)
            total_samples += imgs.size()[0]
            if lr_scheduler:
                lr_scheduler.step()
            if step % FLAGS.log_steps == 0:
                xm.add_step_closure(
                    _train_update, args=(device, step, loss, tracker, epoch, writer))
                test_utils.write_to_summary(writer, step, dict_to_write={'Rate_step': tracker.rate()}, write_xla_metrics=False)
                if step % 1000 == 0:
                    xm.master_print("Saving model...")
                    xm.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "opt_state_dict": optimizer.state_dict(),
                                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                            },
                            'current.ckpt',
                            )
            if step == train_steps:
                break   
        
        reduced_global = xm.mesh_reduce('reduced_global', tracker.global_rate(), np.mean)

        return total_samples, reduced_global                                   


    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    accuracy, max_accuracy = 0.0, 0.0
    training_start_time = time.time()
    
    if checkpoint is not None:
        # best_valid_acc = checkpoint['best_valid_acc']
        start_epoch = checkpoint['epoch'] + 1
        xm.master_print('Loaded Model CheckPoint: Epoch={}/{}'.format(
                           start_epoch, FLAGS.num_epochs))
        # xm.master_print('Loaded Model CheckPoint: Epoch={}/{}, Val Accuracy={:.2f}%'.format(
        #     start_epoch, FLAGS.num_epochs, best_valid_acc))
    else:
        best_valid_acc = 0.0
        start_epoch = 1
    
    for epoch in range(start_epoch, FLAGS.num_epochs + 1):
        xm.master_print('Epoch {} train begin {}'.format(
            epoch, test_utils.now()))
        replica_epoch_start = time.time()
        
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


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores, start_method='fork') # , start_method='spawn'
