import torch
import torch.nn as nn
from torchsummary import summary
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

a = torch.randint(0,128,(5,))
b = torch.zeros(128,128,128,128,128)
c = b[a]
print(c)