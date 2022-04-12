# pip3 install ftfy regex notebook torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html einops


import torch
from torchvision.datasets import CocoCaptions
import torchvision.transforms as transforms
import sys
import random
import torchvision.models as models

from orig_clip import clip

num_gpu = torch.cuda.device_count()
num_workers = 8 * num_gpu
batch_size = 16 # 256 * num_gpu# can be 512 I thought?
def random_select_text_and_tokenize_text(text_choices):
    # print(text_choices[0])
    return random.choice(text_choices)


# viz_enc = models.vit_b_32()
# viz_enc.heads = torch.nn.Identity()
# viz_enc = models.resnet50(num_classes=512)
# dummy_text = torch.randint(0, 10000, (4, 76))
# dummy_images = torch.randn(4, 3, 256, 256)
# print(viz_enc(dummy_images).shape) # returns 2048

model, preprocess = clip.load("ViT-B/32", load_pretrained_weights=True)
print(preprocess)
# mock data
coco_train_ds = CocoCaptions(root="/export/share/datasets/vision/coco/images/train2014/",
                             annFile="/export/share/datasets/vision/coco/annotations/annotations/captions_train2014.json",
                            transform=preprocess,
                            target_transform=random_select_text_and_tokenize_text)

loader = torch.utils.data.DataLoader(coco_train_ds, shuffle=True,
                                    batch_size=batch_size)

# opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
opt = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
# opt = torch.optim.Adam(model.parameters(), lr=1e-3)
if num_gpu > 1: model = torch.nn.DataParallel(model)
model = model.train().cuda()
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(100):
    for imgs, txts in loader:
        # print(txts); asdf
        print(imgs.min(), imgs.max())
        # print(imgs)
        # print(txts)
        # txts = txts.cuda()
        txts = clip.tokenize(txts)
        if num_gpu == 1:
            txts = txts.cuda()
            imgs = imgs.cuda()
        # print(txts.shape, imgs.shape)
        opt.zero_grad()
        # loss = model(dummy_text.cuda(), dummy_images.cuda(),
        #         return_loss=True)
        # loss = model(txts, imgs,
        #          return_loss=True)
        logits_per_image, logits_per_text = model(imgs, txts.squeeze())
        target = torch.arange(txts.shape[0], 
                              device=logits_per_image.device)
        target_size = batch_size // num_gpu
        target = target % target_size
        
        print(logits_per_image.shape, logits_per_text.shape, target.shape)
        img_loss = criterion(logits_per_image, target)
        txt_loss = criterion(logits_per_text, target)
        loss = (img_loss + txt_loss ) / 2
        probs = logits_per_image.softmax(dim=-1)
        # print(probs)
        loss.backward()
        opt.step()
        print(loss.item())

