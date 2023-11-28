from typing import Tuple, Dict, List

import logging

import torch
import torch.nn as nn
import sys


from .parameter import MyParameter, normalize_sum_clamp

import time

class Graph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # parameters
        class_max_vertices = 5
        num_classes = 21
        self.vertex = MyParameter(
            shape=(num_classes, class_max_vertices),
            as_buffer=False
        )

    def get_class_vertices(self, detach: bool = False) -> torch.Tensor:
        vertex = self.vertex.tensor
        if detach:
            vertex = vertex.detach()
        # normalize
        # vertex = normalize_sum_clamp(vertex, detach_sum=True, min_val=1.0e-5)
        return vertex

    def get_atlas(self, detach: bool = False) -> Dict[str, torch.Tensor]:
        class_vertices = self.get_class_vertices(detach)
        # class_edges = self.get_class_edges(detach)
        return class_vertices
        # return {
        #     "class_vertices": class_vertices,
        #     "class_edges": class_edges,
        #     "class_ingredients": self.class_ingredients.tensor
        # }

    def forward(self,ingredients):
        bs, h , w = ingredients.shape
        size = (bs, h, w, 5)
        instance_nodes = torch.zeros(bs,h,w,5)
        instance_nodes = instance_nodes.to(ingredients.device)
        for i in range(bs):
            for j in range(h):
                for k in range(w):
                    padding = ingredients[i][j][k]
                    instance_nodes[i][j][k][0] = ingredients[i][j][k].item()
                    instance_nodes[i][j][k][1] = ingredients[i][j-1][k].item() if j > 0 else padding
                    instance_nodes[i][j][k][2] = ingredients[i][j][k+1].item() if k < w-1 else padding
                    instance_nodes[i][j][k][3] = ingredients[i][j+1][k].item() if j < h-1 else padding
                    instance_nodes[i][j][k][4] = ingredients[i][j][k-1].item() if k > 0 else padding
        return instance_nodes