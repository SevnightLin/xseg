import os
import argparse
import collections
from typing import Dict, Any, Iterable

import matplotlib
# matplotlib.use('TkAgg')

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
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

# import cv_lib.utils as cv_utils

# from dark_kg.data import build_train_dataset


class Accumulator:
    def __init__(self):
        self.values = list()

    def update(self, value: torch.Tensor):
        """
        value: [bs, ...]
        """
        self.values.append(value.cpu())

    def accumulate(self) -> torch.Tensor:
        return torch.cat(self.values)


class   DictAccumulator:
    def __init__(self):
        self.values = collections.defaultdict(Accumulator)

    def update(self, value: Dict[str, torch.Tensor]):
        for k, v in value.items():
            self.values[k].update(v)

    def accumulate(self) -> Dict[str, torch.Tensor]:
        ret = dict()
        for k, v in tqdm(self.values.items(), desc="accumulating"):
            ret[k] = v.accumulate()
        return ret


def run_encode(
    args,
    dataset: data.Dataset,
    device: torch.device
):
    # create backbone
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
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
    )
    i = 0
    accumulator = DictAccumulator()
    for img_id,img, target in tqdm(dataloader, total=len(dataloader), desc="loading"):
        # img1 = img[0]
        # dataset_mean = torch.Tensor([CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R])
        # img2 = img1 + dataset_mean.unsqueeze(-1).unsqueeze(-1)
        # img2 = img2.numpy().transpose(1,2,0)
        # img2 = np.round(img2).astype(np.uint8)
        # # img2 = np.clip(img2,0,1)
        # plt.imshow(img2)
        # plt.savefig('{}.png'.format(img_id[0]))
        processed_tensors = []  
        for item in img_id:  
            processed_item = item.replace('_', '')  
            processed_tensors.append(int(processed_item)) 
        img_id = torch.as_tensor(processed_tensors).to(device)
        img = img.to(device)
        with torch.no_grad():
            feat = backbone(img)    #[bs,2048,41,41]
        # distance matrix: [bs, n, n_codes]
        feat= feat.permute(0,2,3,1)
        bs, H, W, dim = feat.shape
        feat = feat.reshape(bs, H * W, dim)
        dist = torch.cdist(feat, codebook_tensor)
        # dist = dist.transpose(1,2)
        min_dist, match = dist.min(dim=1)
        accumulator.update({
            "img_id": img_id,
            "img": img,
            "feat": feat,
            "dist": min_dist,
            "match": match
        })
        i = i + 1
        if i == 750:
            break
    all_values = accumulator.accumulate()
    return all_values


def crop(
    img: torch.Tensor,
    position: torch.Tensor,
    feat_h: int,
    feat_w: int,
    crop_h: int,
    crop_w: int
) -> torch.Tensor:
    """
    Args:
        img: [3, H, W]
        position: [2] (i, j)
    """
    assert crop_h % 2 == 1 and crop_w % 2 == 1
    H, W = img.shape[-2:]
    c_y = torch.round(position[0] * (H - 1) / (feat_h - 1)).int().item()
    c_x = torch.round(position[1] * (W - 1) / (feat_w - 1)).int().item()
    top = c_y - (crop_h - 1) // 2
    left = c_x - (crop_w - 1) // 2
    return TF.crop(img, top, left, crop_h, crop_w)


def run_match(
    args,
    all_values: Dict[str, torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    dist = all_values["dist"].to(device)
    K = args.topk
    N = dist.shape[1]
    # image id for top k and code n, shape: [K * N]
    topk_img_id = dist.topk(K, dim=0, largest=False)[1].flatten().cpu()
    # raw image for top k and code n, shape: [K * N, 3, H, W]
    topk_img = all_values["img"][topk_img_id]
    # feature position for top k and code n, [K * N]
    topk_match = all_values["match"][topk_img_id].unflatten(0, (K, N)).diagonal(dim1=1, dim2=2).flatten()
    # feature map position
    gird = torch.meshgrid(
        torch.arange(args.feat_h),
        torch.arange(args.feat_w),
        indexing="ij"
    )
    # [H * W, 2]
    gird = torch.stack(gird, dim=-1).flatten(0, 1)
    # [K * N, 2]
    topk_match_pos = gird[topk_match]

    crop_imgs = list()
    for id in tqdm(range(K * N), desc="cropping"):
        img = topk_img[id]
        pos = topk_match_pos[id]
        img = crop(img, pos, args.feat_h, args.feat_w, args.crop_h, args.crop_w)
        crop_imgs.append(img)
    # [K, N, 3, H, W]
    crop_imgs = torch.stack(crop_imgs).unflatten(0, (K, N))
    return crop_imgs


def vis_imgs(
    args,
    imgs: torch.Tensor,
    # dataset_mean: Iterable[float],
    # dataset_std: Iterable[float],
    scale_factor: int = 2,
    dpi: float = 240,
    padding: int = 2
):
    """
    Args:
        imgs: [K, N, 3, H, W]
    """
    c_begin = 0 if args.c_begin is None else args.c_begin
    c_end = imgs.shape[1] if args.c_end is None else args.c_end
    imgs = imgs[:, c_begin:c_end, ...]
    K, N = imgs.shape[:2]
    # [K, N, 3, H, W] -> [K*N, 3, H, W]
    imgs = imgs.flatten(0, 1)
    dataset_mean = torch.Tensor([CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R])
    mean_bgr = dataset_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) 
    imgs = imgs + mean_bgr
    # imgs = torch.round(imgs).to(torch.uint8)
    # un-normalize
    # _imgs = list()
    # for channel_id,  m in enumerate(dataset_mean):
    #     channel = imgs[:, channel_id] + m
    #     _imgs.append(channel)
    # imgs = torch.stack(_imgs, dim=1)
    # _imgs = list()
    # for channel_id, (s, m) in enumerate(zip(dataset_std, dataset_mean)):
    #     channel = imgs[:, channel_id] * s + m
    #     _imgs.append(channel)
    # imgs = torch.stack(_imgs, dim=1)
    imgs = make_grid(imgs, nrow=N, padding=padding)
    imgs = F.interpolate(
        imgs[None, ...],
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=True
    )[0]
    # imgs.clamp_(0, 1)
    imgs = torch.round(imgs).to(torch.uint8)
    imgs = TF.to_pil_image(imgs)

    figsize = (imgs.width / dpi, imgs.height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(imgs)
    w = args.crop_w * scale_factor + scale_factor * padding
    h = args.crop_h * scale_factor + scale_factor * padding
    ax.set_xticks([(w - 1) // 2 + i * w for i in range(N)])
    ax.set_xticklabels(range(c_begin, c_end), rotation=args.rotate_x)
    ax.set_yticks([(h - 1) // 2 + i * h for i in range(K)])
    ax.set_yticklabels(range(1, 1 + K))
    ax.set_xlabel("Code")
    ax.set_ylabel(f"Top {K}")
    fp = os.path.join(args.save_path, f"vis_codes-{c_begin}_{c_end}_750.pdf")
    fig.savefig(fp, bbox_inches="tight")


def main(args):
    # set cuda
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
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=False,
    )
    # split configs
    # data_cfg: Dict[str, Any] = cv_utils.get_cfg(args.data_cfg)
    # get dataloader
    print("Building dataset...")
    # data_cfg["make_partial"] = args.make_partial
    # dataset, _, _, _ = build_train_dataset(data_cfg)
    # dataset.augmentations = None

    all_values = run_encode(args, dataset, device)
    # [K * N, 3, H, W]
    img_crops = run_match(args, all_values, device)
    print("Plotting...")
    # vis_imgs(args, img_crops, dataset.dataset_mean, dataset.dataset_std, scale_factor=args.scale_factor)
    vis_imgs(args, img_crops, scale_factor=args.scale_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str)
    parser.add_argument("--backbone_jit", type=str)
    parser.add_argument("--codebook_jit", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--feat_h", type=int, default=41)
    parser.add_argument("--feat_w", type=int, default=41)
    parser.add_argument("--crop_h", type=int, default=51)
    parser.add_argument("--crop_w", type=int, default=51)
    parser.add_argument("--c_begin", type=int, default=None)
    parser.add_argument("--c_end", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scale_factor", type=float, default=2)
    parser.add_argument("--rotate_x", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--make_partial", type=float, default=None)
    args = parser.parse_args()
    args.save_path = 'data'
    os.makedirs(args.save_path, exist_ok=True)
    main(args)

