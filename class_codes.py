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
from models.backbone import DeepLabV2_ResNet101_MSC

config_path = 'configs/voc12.yaml'
CONFIG = OmegaConf.load(config_path)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = get_dataset(CONFIG.DATASET.NAME)(
    root=CONFIG.DATASET.ROOT,
    split=CONFIG.DATASET.SPLIT.TRAIN,
    ignore_label=CONFIG.DATASET.IGNORE_LABEL,
    mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
    augment=False,
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
num_classes = 21 
code_counts = torch.zeros(len(dataloader), num_classes, 128, device=device)
for img_id, img, label in tqdm(dataloader, total=len(dataloader), desc="statistics"):
    # img_id = target["image_id"]
    _,_,H,W = img.shape
    img = img.to(device)
    with torch.no_grad():
        feat = backbone(img)    #[bs,2048,41,41]
    feat = F.interpolate(feat, size=(H,W), mode='bilinear', align_corners=False)
    feat = feat.permute(0,2,3,1)
    ingredients = torch.cdist(feat, codebook_tensor).argmin(dim = 3) #[bs,h,w]
    label = label.to(device)
     
    for c in range(num_classes):  
        mask = label == c  
        code_counts[i, c] += torch.bincount(ingredients[mask].flatten(), minlength=128)  

    i = i + 1 
    
code_counts = code_counts.sum(dim=0)  # 所有循环的统计结果累加  
top20_codes_num = torch.topk(code_counts, k=20, dim=1).values.cpu().numpy()
top20_codes = torch.topk(code_counts, k=20, dim=1).indices.cpu().numpy()
np.save('data/top20_codes.npy', top20_codes)
np.save('data/top20_codes_num.npy', top20_codes_num)
