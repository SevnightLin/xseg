from __future__ import absolute_import, division, print_function

import json
import multiprocessing
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from libs.datasets import get_dataset
from models.backbone import DeepLabV2_ResNet101_MSC
from models.xseg import Discretization, Graph ,Predictor ,Matcher
from libs.utils import scores

def get_patch(code:torch.Tensor,patch:torch.Tensor,device):
    eye = torch.eye(128).to(device)
    patch = eye[code]
    shifted = torch.zeros_like(code).to(device)
    # right
    shifted[:,:,1:] = code[:,:,:-1]
    shifted[:,:,:1] = code[:,:,:1]
    patch = patch + eye[shifted]
    # right-bottom
    temp = shifted.clone()
    shifted[:,1:,:] = temp[:,:-1,:]
    shifted[:,:1,:] = temp[:,:1,:]
    patch = patch + eye[shifted]
    # bottom
    shifted[:,1:,:] = code[:,:-1,:]
    shifted[:,:1,:] = code[:,:1,:]
    patch = patch + eye[shifted]
    # bottom-left
    temp = shifted.clone()
    shifted[:,:,:-1] = temp[:,:,1:]
    shifted[:,:,-1] = temp[:,:,-1]
    patch = patch + eye[shifted]
    # left
    shifted[:,:,:-1] = code[:,:,1:]
    shifted[:,:,-1] = code[:,:,-1]
    patch = patch + eye[shifted]
    # left-top
    temp = shifted.clone()
    shifted[:,:-1,:] = temp[:,1:,:]
    shifted[:,-1,:] = temp[:,-1,:]
    patch = patch + eye[shifted]
    # top
    shifted[:,:-1,:] = code[:,1:,:]
    shifted[:,-1,:] = code[:,-1,:]
    patch = patch + eye[shifted]
    # top-right
    temp = shifted.clone()
    shifted[:,:,1:] = temp[:,:,:-1]
    shifted[:,:,:1] = temp[:,:,:1]
    patch = patch + eye[shifted]

    return patch

def test():
    config_path = 'configs/voc12.yaml'
    save_path = 'data/scores/graph_5000.json'
    state_dict_ = torch.load('data/checkpoints/graph/checkpoint_5000.pth')
    CONFIG = OmegaConf.load(config_path)
    device = torch.device('cuda:1')
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
    )

    backbone = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load('data/deeplabv2_without_aspp.pth', map_location=lambda storage, loc: storage)
    backbone.load_state_dict(state_dict)
    backbone.eval()
    backbone.to(device)
    
    discretization = Discretization(128,2048,True,[0, 1])
    # discretization.to(device)

    discretization.initial_vocabulary('data/cluster_128_from_1000000.pth')

    graph = Graph()
    # graph.to(device)
    init : torch.Tensor = torch.load('data/init.pth')
    init = torch.round(init)
    graph.vertex.copy_(init)
    matcher = Matcher()
    # matcher.to(device)
    predictor = Predictor(
        # backbone=backbone,
        discretization=discretization,
        graph=graph,
        matcher=matcher
    )
    predictor.load_state_dict(state_dict_)
    predictor.to(device)
    predictor.eval()

    preds, gts = [], []
    for image_ids, images, gt_labels in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        # Image
        images = images.to(device)  #[1,3,366,500]

        # Forward propagation
        with torch.no_grad():
            feat = backbone(images)
            # feat = feat.permute(0,2,3,1)
            # ingredients = torch.cdist(feat, codebook_tensor).argmin(dim = 3) #[bs,h,w]
            # _, h, w = ingredients.shape
            # feat_patch = torch.zeros(CONFIG.SOLVER.BATCH_SIZE.TEST,h,w,128).to(device)
            # feat_patch = get_patch(ingredients,feat_patch,device)
            # feat_norm = F.normalize(feat_patch,dim=-1)
            # feat_norm = feat_norm.permute(0,3,1,2)
            # logits = model(feat_norm)
            logits = predictor(feat)
        # Pixel-wise labeling
        logits = logits.permute(0,3,1,2)
        _, H, W = gt_labels.shape
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )   #[1,21,366,500]
        probs = F.softmax(logits, dim=1)    #[1,21,366,500]
        labels = torch.argmax(probs, dim=1) #[1,1,366,500]

        preds += list(labels.cpu().numpy())
        gts += list(gt_labels.numpy())

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    test()