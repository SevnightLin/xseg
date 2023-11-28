import os
import argparse
import collections
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
from libs.utils import resize_labels
from models.backbone import DeepLabV2_ResNet101_MSC

config_path = 'configs/voc12.yaml'
CONFIG = OmegaConf.load(config_path)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # dataset_mean = torch.Tensor([CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R])
    # img2 = img + dataset_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    # processed_tensors = []  
    # for item in img_id:  
    #     processed_item = item.replace('_', '')  
    #     processed_tensors.append(int(processed_item)) 
    # img_id = torch.as_tensor(processed_tensors).to(device)
    img = img.to(device)
    with torch.no_grad():
        feat = backbone(img)    #[bs,2048,41,41]
    _,_,h,w, = feat.shape
    feat= feat.permute(0,2,3,1)
    dist = torch.cdist(feat, codebook_tensor)
    min_dist, match = dist.min(dim=-1)
    # drow match
    # arr = match.squeeze(0).cpu().numpy()
    # resized_array = np.repeat(np.repeat(arr, 8, axis=0), 8, axis=1)  
  
    # fig, ax = plt.subplots(figsize=(8, 8))  
    # ax.imshow(resized_array, cmap='cool',alpha=0.5)  
    # for i in range(arr.shape[0]):  
    #     for j in range(arr.shape[1]):  
    #         ax.text(8*j+4, 8*i+4, str(arr[i, j]), ha='center', va='center', color='black', fontsize=4)  
    # ax.axis('off')
    # plt.savefig('data/image_code/{}.png'.format(img_id[0]), dpi=300) 
    label = resize_labels(target,size=(w,h))
    # drow match
    arr = label.squeeze(0).numpy()
    resized_array = np.repeat(np.repeat(arr, 8, axis=0), 8, axis=1)  
  
    fig, ax = plt.subplots(figsize=(8, 8))  
    ax.imshow(resized_array, cmap='cool',alpha=0.5)  
    for i in range(arr.shape[0]):  
        for j in range(arr.shape[1]):  
            ax.text(8*j+4, 8*i+4, str(arr[i, j]), ha='center', va='center', color='black', fontsize=4)  
    ax.axis('off')
    plt.savefig('data/image_code_label/{}.png'.format(img_id[0]), dpi=300) 
    i = i + 1
    if i == 100:
        break