from typing import Tuple, Dict, List

import logging

import torch
import torch.nn as nn

from .discretization import Discretization
from .graph import Graph
from .matcher import Matcher
from .parameter import MyParameter, normalize_sum_clamp
from torch_geometric.data.data import Data
from torch_geometric.transforms import ToSparseTensor 
import collections
import time


class Predictor(nn.Module):
    """
    Procedure:
        1. use ingredient model to predict sequence of ingredient
        2. use SchemaNet to predict
    Prediction:
        "pred": SchemaNet prediction, shape: [bs, num_classes];
        "origin_pred": origin model prediction, shape: [bs, num_classes];
        "codes": codes predicted by origin model, shape: [bs, H, W];
        "attribution": attribution to codes w.r.t. each class, shape: [bs, num_classes, H, W]
    """
    def __init__(
        self,
        # backbone: DeepLabV2_ResNet101_MSC,
        discretization:Discretization,
        graph: Graph,
        matcher: Matcher
    ):
        super().__init__()
        # self.backbone = backbone
        self.discretization = discretization
        self.graph = graph
        self.matcher = matcher
        self.num_classes = 21

    def get_device(cuda):
        cuda = cuda and torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        if cuda:
            print("Device:")
            for i in range(torch.cuda.device_count()):
                print("    {}:".format(i), torch.cuda.get_device_name(i))
        else:
            print("Device: CPU")
        return device

    def forward(self, x: torch.Tensor, requires_graph: bool = False):
        ret = collections.OrderedDict()
        with torch.no_grad():
            # output = self.backbone(x)
            output = self.discretization(x)
        instance_nodes = self.graph(output)
        # instance_nodes = instance_nodes.to(torch.device('cuda'))
        class_nodes = self.graph.get_atlas()
        pred = self.matcher(
            instance_nodes= instance_nodes,
            class_nodes=class_nodes
        )
        # ret["pred"] = pred
        # ret.update(class_dict)
        # if requires_graph:
        #     ret.update(instance_dict)
        #     ret["ingredients"] = output["ingredients"]
        #     ret["attn_cls"] = output["attn_cls"]
        return pred