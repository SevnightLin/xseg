from __future__ import absolute_import, division, print_function

import json
import multiprocessing
import os

import click
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from libs.datasets import get_dataset
from models.backbone import DeepLabV2_ResNet101_MSC
from libs.utils import get_device, resize_labels

# from models.xseg import Discretization, Graph ,Predictor ,Matcher
import time

def get_patch(code:torch.Tensor,patch:torch.Tensor,device):
    eye = torch.eye(128).to(device)
    patch = eye[code]
    shifted = torch.zeros_like(code).to(device)
    # right
    shifted[:,:,1:] = code[:,:,:-1]
    shifted[:,:,:1] = code[:,:,:1]
    patch = patch + eye[shifted]
    # right-bottom
    shifted[:,1:,:] = shifted[:,:-1,:]
    shifted[:,:1,:] = shifted[:,:1,:]
    patch = patch + eye[shifted]
    # bottom
    shifted[:,1:,:] = code[:,:-1,:]
    shifted[:,:1,:] = code[:,:1,:]
    patch = patch + eye[shifted]
    # bottom-left
    shifted[:,:,:-1] = shifted[:,:,1:]
    shifted[:,:,-1] = shifted[:,:,-1]
    patch = patch + eye[shifted]
    # left
    shifted[:,:,:-1] = code[:,:,1:]
    shifted[:,:,-1] = code[:,:,-1]
    patch = patch + eye[shifted]
    # left-top
    shifted[:,:-1,:] = shifted[:,1:,:]
    shifted[:,-1,:] = shifted[:,-1,:]
    patch = patch + eye[shifted]
    # top
    shifted[:,:-1,:] = code[:,1:,:]
    shifted[:,-1,:] = code[:,-1,:]
    patch = patch + eye[shifted]
    # top-right
    shifted[:,:,1:] = shifted[:,:,:-1]
    shifted[:,:,:1] = shifted[:,:,:1]
    patch = patch + eye[shifted]

    return patch

def train():
    config_path = 'configs/voc12.yaml'
    checkpoint_dir = 'data/checkpoints_nopatch'
    writer = SummaryWriter('logs')
    CONFIG = OmegaConf.load(config_path)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # device = get_device('cuda')
    device = torch.device('cuda:1')
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=True,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
    )
    print(len(loader))
    loader_iter = iter(loader)

    backbone = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load('data/deeplabv2_without_aspp.pth', map_location=lambda storage, loc: storage)
    backbone.load_state_dict(state_dict)
    backbone.eval()
    backbone.to(device)

    codebook_tensor: torch.Tensor = torch.load('data/cluster_128_from_1000000.pth')
    codebook_tensor = codebook_tensor.to(device)
    model = nn.Conv2d(2048,21,1)
    # state_dict_ = torch.load('data/checkpoints/checkpoint_10000.pth')
    # model.load_state_dict(state_dict_)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.SOLVER.LR, weight_decay=CONFIG.SOLVER.WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL)
    criterion.to(device)

    average_loss = MovingAverageValueMeter(CONFIG.SOLVER.AVERAGE_LOSS)
    model.train()

    for iteration in tqdm(
        range(1, CONFIG.SOLVER.ITER_MAX + 1),
        total=CONFIG.SOLVER.ITER_MAX,
        dynamic_ncols=True,
    ):

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        loss = 0
        for _ in range(CONFIG.SOLVER.ITER_SIZE):
            try:
                _, images, labels = next(loader_iter)
            except:
                loader_iter = iter(loader)
                _, images, labels = next(loader_iter)
            # images [1,3,321,321]
            # Propagate forward
            images = images.to(device)
            with torch.no_grad():
                feat = backbone(images)
            # feat = feat.permute(0,2,3,1)
            # ingredients = torch.cdist(feat, codebook_tensor).argmin(dim = 3) #[bs,h,w]
            # feat_patch = torch.zeros(5,41,41,128).to(device)
            # feat_patch = get_patch(ingredients,feat_patch,device)
            # feat_norm = F.normalize(feat_patch,dim=-1)
            # feat_norm = feat_norm.permute(0,3,1,2)
            # logits = model(feat_norm)
            logits = model(feat)
            # Loss
            iter_loss = 0
            # for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
            _, _, H, W = logits.shape
            labels_ = resize_labels(labels, size=(H, W))
            iter_loss = criterion(logits, labels_.to(device))

            # Propagate backward (just compute gradients)
            iter_loss /= CONFIG.SOLVER.ITER_SIZE
            iter_loss.backward()
            loss += float(iter_loss)

        average_loss.add(loss)

        # Update weights with accumulated gradients
        optimizer.step()

        # Update learning rate
        #scheduler.step(epoch=iteration)

        # TensorBoard
        if iteration % CONFIG.SOLVER.ITER_TB == 0:
            writer.add_scalar("loss/train", average_loss.value()[0], iteration)
            for i, o in enumerate(optimizer.param_groups):
                writer.add_scalar("lr/group_{}".format(i), o["lr"], iteration)
            for i in range(torch.cuda.device_count()):
                writer.add_scalar(
                    "gpu/device_{}/memory_cached".format(i),
                    torch.cuda.memory_cached(i) / 1024 ** 3,
                    iteration,
                )

            if False:
                for name, param in model.module.base.named_parameters():
                    name = name.replace(".", "/")
                    # Weight/gradient distribution
                    writer.add_histogram(name, param, iteration, bins="auto")
                    if param.requires_grad:
                        writer.add_histogram(
                            name + "/grad", param.grad, iteration, bins="auto"
                        )

        # Save a model
        if iteration % CONFIG.SOLVER.ITER_SAVE == 0:
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(iteration)),
            )
    # torch.save(
    #     model.state_dict(), os.path.join(checkpoint_dir, "checkpoint_final.pth")
    # )

if __name__ == '__main__':
    train()