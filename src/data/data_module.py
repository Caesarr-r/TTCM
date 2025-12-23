"""data_module.py
Lightweight facade for data loading and basic few-shot splits.
Used by training / evaluation; advanced S1/S2 augmentation and LLM-based
retrieval are delegated to `ssda_backend.py`.
"""
from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Dict, List

__all__ = [
    "load_data_from_pt",
    "preprocess_data_for_fewshot",
    "create_fixed_graph_split",
    "prepare_training_data",  # delegated to backend (S1/S2 + OOD retrieval)
    "DataHelper",
    "InBatchDataset",
]

# -----------------------------------------------------------------------------
# Dataset helpers (kept consistent with legacy backend)
# -----------------------------------------------------------------------------

class DataHelper(Dataset):
    def __init__(self, edge_index, args, directed=False, transform=None):
        self.transform = transform
        self.args = args
        self.neighs = dict()
        nodes = np.unique(edge_index)
        self.node_dim = nodes.shape[0]
        self.all_nodes = set(nodes)
        self.all_nodes_array = np.array(list(self.all_nodes))
        for i in range(edge_index.shape[1]):
            s_node = edge_index[0, i].item()
            t_node = edge_index[1, i].item()
            if s_node not in self.neighs:
                self.neighs[s_node] = []
            if t_node not in self.neighs:
                self.neighs[t_node] = []
            self.neighs[s_node].append(t_node)
            if not directed:
                self.neighs[t_node].append(s_node)
        for node in self.neighs:
            self.neighs[node] = np.array(list(self.neighs[node]))
        self.idx = nodes

    def __len__(self):
        return self.node_dim

    def __getitem__(self, idx):
        s_n = self.idx[idx].item()
        positive_neighbors = list(self.neighs.get(s_n, []))
        if not positive_neighbors:
            positive_neighbors = [s_n]
        t_n = [np.random.choice(positive_neighbors).item() for _ in range(self.args.neigh_num)]
        if self.args.sampling_mode == 'explicit_no_check':
            neg_n = np.random.choice(self.all_nodes_array, size=self.args.neg_num, replace=True).tolist()
        else:
            s_n_neighbors = self.neighs.get(s_n, np.array([], dtype=np.int64))
            neg_n = []
            while len(neg_n) < self.args.neg_num:
                needed = self.args.neg_num - len(neg_n)
                sample_size = max(needed * 2, 10)
                potential_negs = np.random.choice(self.all_nodes_array, size=sample_size, replace=True)
                mask_not_self = (potential_negs != s_n)
                if s_n_neighbors.size > 0:
                    mask_not_neighbor = ~np.isin(potential_negs, s_n_neighbors, assume_unique=True)
                    valid_mask = mask_not_self & mask_not_neighbor
                else:
                    valid_mask = mask_not_self
                valid_negs = potential_negs[valid_mask]
                if valid_negs.size > 0:
                    neg_n.extend(valid_negs.tolist())
            neg_n = neg_n[:self.args.neg_num]
        sample = {'s_n': s_n, 't_n': np.array(t_n), 'neg_n': np.array(neg_n)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class InBatchDataset(Dataset):
    def __init__(self, edge_index):
        nodes = np.unique(edge_index.cpu().numpy())
        self.nodes = nodes
    def __len__(self):
        return len(self.nodes)
    def __getitem__(self, idx):
        return self.nodes[idx]

# -----------------------------------------------------------------------------
# High-level APIs (kept consistent with legacy backend)
# -----------------------------------------------------------------------------
def load_data_from_pt(device_param, dataset_name, dataset_pt_path):
    if dataset_name == 'ogbn_arxiv' and 'ogbn_arxiv_processed.pt' in dataset_pt_path:
        dataset_pt_path = dataset_pt_path.replace('ogbn_arxiv_processed.pt', 'arxiv_processed.pt')
    try:
        data = torch.load(dataset_pt_path)
    except Exception as e:
        print(f"Error while loading {dataset_pt_path} .pt file: {e}")
        raise
    target_device = device_param
    if isinstance(device_param, torch.device) and device_param.type == 'cuda':
        visible = torch.cuda.device_count()
        idx = device_param.index if device_param.index is not None else 0
        if visible == 0:
            print("Warning: no visible GPU detected, falling back to CPU.")
            target_device = torch.device('cpu')
        elif idx < 0 or idx >= visible:
            print(f"Warning: target GPU index {idx} is out of visible range [0, {visible-1}], falling back to cuda:0.")
            target_device = torch.device('cuda:0')
    try:
        data.x = data.x.to(target_device)
        data.edge_index = data.edge_index.to(target_device)
    except RuntimeError as e:
        if 'invalid device ordinal' in str(e).lower():
            print(f"Warning: caught invalid device ordinal, forcing fallback to cuda:0/CPU. Error: {e}")
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                target_device = torch.device('cuda:0')
            else:
                target_device = torch.device('cpu')
            data.x = data.x.to(target_device)
            data.edge_index = data.edge_index.to(target_device)
        else:
            raise
    return data


def preprocess_data_for_fewshot(data_obj):
    labels = data_obj.y.cpu().numpy()
    class_to_indices = {}
    for i, label in enumerate(labels):
        class_to_indices.setdefault(label, []).append(i)
    return class_to_indices


def create_fixed_graph_split(class_to_indices_map, id_class_indices, k_shot, test_size=0.8, seed=42, edge_index=None):
    random.seed(seed)
    np.random.seed(seed)
    all_classes_indices = sorted(list(class_to_indices_map.keys()))
    ood_classes_indices = sorted([c for c in all_classes_indices if c not in id_class_indices])
    support_indices, support_labels, remaining_indices = [], [], []
    class_mapping = {original: i for i, original in enumerate(id_class_indices)}
    for original_class_idx in id_class_indices:
        all_nodes_for_class = class_to_indices_map[original_class_idx]
        if len(all_nodes_for_class) < k_shot:
            print(f"Warning: class {original_class_idx} has only {len(all_nodes_for_class)} nodes (< k_shot={k_shot}).")
            sampled = all_nodes_for_class
        else:
            sampled = random.sample(all_nodes_for_class, k_shot)
        support_indices.extend(sampled)
        support_labels.extend([class_mapping[original_class_idx]] * len(sampled))
        remaining_nodes = [node for node in all_nodes_for_class if node not in sampled]
        remaining_indices.extend(remaining_nodes)
    for ood_class_idx in ood_classes_indices:
        remaining_indices.extend(class_to_indices_map[ood_class_idx])
    remaining_labels = []
    all_labels_map = {node: label for label, nodes in class_to_indices_map.items() for node in nodes}
    for idx in remaining_indices:
        remaining_labels.append(all_labels_map[idx])
    train_indices_pool, test_indices, test_labels_original = [], [], []
    if len(remaining_indices) > 0:
        try:
            train_indices_pool, test_indices, _, test_labels_original = train_test_split(
                remaining_indices, remaining_labels, test_size=test_size, random_state=seed, stratify=remaining_labels
            )
        except ValueError:
            print("Warning: stratified split failed, falling back to non-stratified split.")
            train_indices_pool, test_indices, _, test_labels_original = train_test_split(
                remaining_indices, remaining_labels, test_size=test_size, random_state=seed
            )
    return {
        "support_indices": np.array(support_indices),
        "support_labels": np.array(support_labels),
        "test_indices": np.array(test_indices),
        "test_multiclass_labels": np.array(test_labels_original),
        "class_mapping": class_mapping,
        "train_indices_pool": np.array(train_indices_pool),
    }

# -----------------------------------------------------------------------------
# High-level prepare_training_data (delegated to backend to avoid circular deps)
# -----------------------------------------------------------------------------
from src.data import ssda_backend as _backend  # late import to avoid circular dependency

def prepare_training_data(*args, **kwargs):
    return _backend.prepare_training_data(*args, **kwargs)
