import torch
import torch.nn as nn
from models.backbone import DeepLabV2_ResNet101_MSC
from torchsummary import summary
from libs.utils import get_device
from collections import OrderedDict
import os
import h5py
import cv2
from typing import Dict, Any, Iterable
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import torch
import torch.backends.cudnn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from libs.datasets import get_dataset
from models.backbone import DeepLabV2_ResNet101_MSC

dir = '/nfs/liujiaxuan/data/pascalvoc2012/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg'
image = cv2.imread(dir)
plt.imshow(image)
plt.savefig('111.png')

config_path = 'configs/voc12.yaml'
CONFIG = OmegaConf.load(config_path)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# make deterministic
# if args.seed is not None:
#     cv_utils.make_deterministic(args.seed)
dataset = get_dataset(CONFIG.DATASET.NAME)(
    root=CONFIG.DATASET.ROOT,
    split=CONFIG.DATASET.SPLIT.TRAIN,
    ignore_label=CONFIG.DATASET.IGNORE_LABEL,
    mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
    augment=False,
    # base_size=CONFIG.IMAGE.SIZE.BASE,
    # crop_size=CONFIG.IMAGE.SIZE.TRAIN,
    # scales=CONFIG.DATASET.SCALES,
    # flip=False,
)

backbone = DeepLabV2_ResNet101_MSC(2048)
state_dict = torch.load('data/deeplabv2_without_aspp.pth', map_location=lambda storage, loc: storage)
backbone.load_state_dict(state_dict)
# backbone = torch.nn.DataParallel(backbone)
backbone.eval()
backbone.to(device)
# [n, dim]
codebook_tensor: torch.Tensor = torch.load('data/cluster_128_from_1000000.pth')
codebook_tensor = codebook_tensor.to(device)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
    num_workers=CONFIG.DATALOADER.NUM_WORKERS,
    shuffle=False,
)
i = 0
for img_id,img, target in tqdm(dataloader, total=len(dataloader), desc="loading"):
    # img_id = target["image_id"]
    _,_,H,W = img.shape
    img1 = img[0]
    dataset_mean = torch.Tensor([CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R])
    img2 = img1 + dataset_mean.unsqueeze(-1).unsqueeze(-1)
    img2 = img2.numpy().transpose(1,2,0)
    img2 = np.round(img2).astype(np.uint8)
    # img2 = np.clip(img2,0,1)
    plt.imshow(img2)
    plt.savefig('{}.png'.format(img_id[0]))
    # processed_tensors = []  
    # for item in img_id:  
    #     processed_item = item.replace('_', '')  
    #     processed_tensors.append(int(processed_item)) 
    # img_id = torch.as_tensor(processed_tensors).to(device)
    img = img.to(device)
    with torch.no_grad():
        feat = backbone(img)    #[bs,2048,41,41]
    feat= feat.permute(0,2,3,1)
    dist = torch.cdist(feat, codebook_tensor)
    min_dist, match = dist.min(dim=-1)
    np.save('data/npy/{}.npy'.format(img_id),match.squeeze(0).cpu().numpy())
    match = match.unsqueeze(0).float()
    match = F.interpolate(
            match, size=(H, W), mode="nearest"
        )
    arr = match.squeeze(0).squeeze(0).cpu().numpy()
    np.save('data/npy/{}_in.npy'.format(img_id),arr)
    arr = arr.astype(np.uint8)

    color_image = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)  
  
    # 可以选择将灰度图的值映射到RGB颜色空间的不同范围，这里将灰度值映射到0-255的范围  
    color_image = cv2.applyColorMap(color_image, cv2.COLORMAP_JET)  
    
    plt.imshow(color_image)  
    plt.axis('off')  # 不显示坐标轴  
    plt.savefig('data/nearest_rgb/{}_codes.png'.format(img_id[0]))  