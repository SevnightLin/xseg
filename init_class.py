#增加极小值
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from libs.datasets import get_dataset
from models.backbone import DeepLabV2_ResNet101_MSC
from libs.utils import get_device,resize_labels
from models.xseg import Discretization


def train():
    config_path = 'configs/voc12.yaml'
    CONFIG = OmegaConf.load(config_path)
    device = get_device('cuda')
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
    loader_iter = iter(loader)

    backbone = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load('data/deeplabv2_without_aspp.pth', map_location=lambda storage, loc: storage)
    backbone.load_state_dict(state_dict)
    backbone.eval()
    backbone.to(device)

    discretization = Discretization(128,2048,True,[0, 1])

    discretization.initial_vocabulary('data/cluster_128_from_1000000.pth')
    discretization.to(device)

    class_nodes = torch.zeros(21,5).to(device)
    num_per_class = torch.zeros(21).to(device)
    for image_ids, images, gt_labels in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        with torch.no_grad():
            out = backbone(images.to(device))
            ingredient = discretization(out)
        bs, h, w = ingredient.shape
        labels = resize_labels(gt_labels, size=(h, w))
        for i in range(bs):
            for j in range(h):
                for k in range(w):
                    label = labels[i][j][k] if labels[i][j][k] != 255 else 0
                    center = ingredient[i][j][k]
                    class_nodes[label][0] += center
                    class_nodes[label][1] += ingredient[i][j-1][k] if j > 0 else center
                    class_nodes[label][2] += ingredient[i][j][k+1] if k < w-1 else center
                    class_nodes[label][3] += ingredient[i][j+1][k] if j < h-1 else center
                    class_nodes[label][4] += ingredient[i][j][k-1] if k > 0 else center
                    num_per_class[label] = num_per_class[label] + 1

    class_nodes /= num_per_class[:, None]
    print(class_nodes)
    torch.save(class_nodes,'data/init.pth')

if __name__ == '__main__':
    train()