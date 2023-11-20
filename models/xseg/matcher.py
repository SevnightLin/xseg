from typing import Tuple, Dict, List

import logging

import torch
import torch.nn as nn

from gnn import GCN, SAGE
from torch_geometric.data.data import Data
from torch_geometric.transforms import ToSparseTensor 
import time

class Matcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_codes = 128
        self.embedding = nn.Embedding(
            num_embeddings= 128 + 1,
            embedding_dim=21,
            padding_idx=128
        )
        self.gnn = GCN(21,128,21,2,0.0,False)
        # self.gnn = SAGE(21,128,21,2,0.0,False)
        # SUPPORTED_SIM = {
        #     "cosine": self._cosine_sim,
        #     "euclidean": self._euclidean_sim,
        #     "inner_product": self._inner_product
        # }
        # self.similarity = SUPPORTED_SIM[similarity]
        self._reset_parameters()

    def _reset_parameters(self):
        # nn.init.normal_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)
        nn.init.trunc_normal_(self.embedding.weight[:self.num_codes])

    def _cosine_sim(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        sim = torch.cosine_similarity(feat_1, feat_2, dim=-1)
        return (sim + 1) / 2

    def _euclidean_sim(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        dist = torch.linalg.vector_norm(feat_1 - feat_2, dim=-1)
        return 1 / (1 + dist)

    def _inner_product(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        dist = (feat_1 * feat_2).sum(-1)
        return dist

    def forward(
        self,
        # instance_dict: Dict[str, List[torch.Tensor]],
        # class_dict: Dict[str, torch.Tensor]
        instance_nodes, #[bs,h,w,5]     [1,41,41,5]
        class_nodes     #[21,5]
    ):
        transform = ToSparseTensor()
        edge_index = torch.Tensor([[0,0,0,0],
                                   [1,2,3,4]]).long()
        edge_index = edge_index.to(instance_nodes.device)
        bs , h, w, num = instance_nodes.shape
        sim = torch.zeros(bs,h,w,21)
        sim = sim.to(instance_nodes.device)
        out_class = torch.zeros(21,105)
        out_class = out_class.to(instance_nodes.device)
        for t in range(21):
            class_ingredient = class_nodes[t][:]
            class_ingredient = class_ingredient.long()
            x_class = self.embedding(class_ingredient)
            data_class = Data(x = x_class,edge_index=edge_index)
            data_class = transform(data_class)
            out = self.gnn(data_class.x,data_class.adj_t)
            # out = torch.exp(out)
            out_class[t] = out.reshape(105)

        for i in range(bs):
            for j in range(h):
                for k in range(w):
                    node_ingredient = instance_nodes[i][j][k][:]
                    node_ingredient = node_ingredient.long()
                    x = self.embedding(node_ingredient)
                    data_instance = Data(x = x, edge_index=edge_index)
                    data_instance = transform(data_instance)
                    out_instance = self.gnn(data_instance.x,data_instance.adj_t)
                    # out_instance = torch.exp(out_instance)
                    out_instance = out_instance.reshape(105)
                    sim[i][j][k] = self._inner_product(out_instance,out_class)
        return sim