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
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm
from libs.utils import makedirs,get_device

from libs.datasets import get_dataset
from models.backbone import DeepLabV2_ResNet101_MSC
from libs.utils import DenseCRF, PolynomialLR, scores

from typing import Dict, Any, List
import h5py

class Adapter:
    def __init__(self):
        self.shape: torch.Size = None

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        self.cls_token = x[:1]
        return x[1:]

    def reconstruct(self, x: torch.Tensor, match: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.cls_token, x), dim=0)
        return x, match

class KMeansClustering:
    def __init__(self, num_clusters: int, method: str):
        self.num_clusters = num_clusters
        self.method = method

    def scipy_kmeans(self, x: np.ndarray) -> np.ndarray:
        from scipy.cluster.vq import kmeans
        centers, _ = kmeans(x, self.num_clusters)
        return centers

    def minibatch_kmeans(self, x: np.ndarray) -> np.ndarray:
        from sklearn.cluster import MiniBatchKMeans
        k_means = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            batch_size=1024,
            verbose=True,
            compute_labels=False,
            n_init="auto"
        )
        k_means.fit(x)
        centers = k_means.cluster_centers_
        return centers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        methods = {
            "cpu_kmeans": self.scipy_kmeans,
            "minibatch": self.minibatch_kmeans
        }
        return methods[self.method](x)
        
def collect_features():
    # Configuration
    CONFIG = OmegaConf.load('configs/voc12.yaml')
    device = get_device(True)
    torch.set_grad_enabled(False)

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

    # Model
    model = DeepLabV2_ResNet101_MSC(2048)
    state_dict = torch.load('data/deeplabv2_without_aspp.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    features: List[torch.Tensor] = list()
    with torch.no_grad():
        for image_ids, images, gt_labels in tqdm(
            loader, total=len(loader), dynamic_ncols=True
        ):
            # Image
            images = images.to(device)  #[5,3,321,321]

            # Forward propagation
            logits = model(images)  #[5,21,41,41]
            # adapt feature: [bs, dim, h, w] -> [h * w, bs, dim]    [1681,5,21]
            feat = logits.permute(2, 3, 0, 1).reshape(logits.shape[2] * logits.shape[3], logits.shape[0], logits.shape[1])
            #feat = adaptor.adapt(logits)    
            # [h * w, bs, dim] -> [h * w * bs, dim]
            feat = feat.flatten(0, 1).cpu() #[8450,21]
            # old version # [bs, dim, h, w] -> [bs, dim, h * w] -> [bs, h * w, dim] -> [bs * h * w, dim]
            # feat = feat.cpu().flatten(2).permute(0, 2, 1).flatten(0, 1)
            features += feat.unbind(0)
            if len(features) >= 1000000:
                print('collect more than 1000000 features')
                break
    features = features[:1000000]
    features = torch.stack(features).numpy()    #[17788342,21]
    with h5py.File(os.path.join('data', "saved_features.h5"), "w") as file:
        file["features"] = features
    print(f"Collected {len(features)} features.")
    return features

def clustering(features: np.ndarray):
    num_features = features.shape[0]
    clustering = KMeansClustering(128, "minibatch")
    cluster_centers = clustering(features)  #[128,21]
    save_fp = os.path.join("data", f"cluster_128_from_{num_features}.pth")
    cluster_centers = torch.from_numpy(cluster_centers).to(torch.float32)
    torch.save(cluster_centers, save_fp)
    print("Done")

def main():
    #args.num_clusters = cv_utils.get_cfg(args.cfg_fp)["discretization"]["vocabulary"]["size"]

    features: np.ndarray
    print("Generating new features")
    features = collect_features()
    clustering(features)

if __name__ == "__main__":
    main()