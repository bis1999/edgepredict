"""
Fixed GNN models - removed double sigmoid bug.

Key Fix:
- _MLPPredictor now returns raw logits (no sigmoid)
- BCEWithLogitsLoss handles sigmoid internally
- Sigmoid only applied during inference in predict()

This module matches the exact naming convention from your original gnn_models.py
"""

import torch
import torch.nn.functional as F
import random
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Literal, Union
import itertools
from tqdm import tqdm
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from logger import get_logger
    log = get_logger(__name__)
except ImportError:
    import logging
    log = logging.getLogger("gnn")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)


# Core GNN Components
class _GCN(torch.nn.Module):
    def __init__(self, dims, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.dropout = dropout

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, adj_t)


class _SAGE(torch.nn.Module):
    def __init__(self, dims, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.dropout = dropout

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, adj_t)


class _MLPPredictor(torch.nn.Module):
    """
    MLP predictor that outputs raw logits (FIXED VERSION).
    """
    def __init__(self, dim, hidden, layers=2, dropout=0.2):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(2 * dim, hidden))
        for _ in range(layers - 2):
            self.layers.append(torch.nn.Linear(hidden, hidden))
        self.layers.append(torch.nn.Linear(hidden, 1))
        self.dropout = dropout

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # IMPORTANT: Return raw logits, NOT sigmoid(logits)
        return self.layers[-1](x).squeeze()


def generate_node_features(num_nodes, dim, feature_type='random', seed=42):
    """Generate node features for GNN input"""
    torch.manual_seed(seed)
    if feature_type == 'random':
        x = torch.empty(num_nodes, dim)
        torch.nn.init.xavier_uniform_(x)
        return x
    elif feature_type == 'identity':
        # Use min(num_nodes, dim) to handle dimension mismatch
        effective_dim = min(num_nodes, dim)
        if num_nodes <= dim:
            x = torch.zeros(num_nodes, dim)
            x[:, :num_nodes] = torch.eye(num_nodes)
        else:
            x = torch.eye(num_nodes, dim)
        return x
    else:
        raise ValueError(f"Unknown node feature type: {feature_type}")


#