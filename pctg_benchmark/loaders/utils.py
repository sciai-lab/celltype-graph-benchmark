import numpy as np
import torch

from pctg_benchmark import gt_mapping
from pctg_benchmark.transforms.transforms import setup_transforms
from pctg_benchmark.transforms.graph import filter_label_from_edges_feature, filter_label_from_edges_ids
from pctg_benchmark.transforms.graph import rectify_graph, remove_edges, remove_node
from typing import Tuple


def concatenate_features(list_feat):
    list_feat = [feat[:, None] if feat.ndim == 1 else feat for feat in list_feat]
    counts_np, counts_torch = 0, 0
    for _feat in list_feat:
        if isinstance(_feat, np.ndarray):
            counts_np += 1
        elif isinstance(_feat, torch.Tensor):
            counts_torch += 1

    if len(list_feat) == counts_np:
        return np.concatenate(list_feat, axis=1)
    elif len(list_feat) == counts_torch:
        return torch.cat(list_feat, 1)
    else:
        raise ValueError("Features must be either all numpy arrays or all torch tensors")


def collect_features(features_dict, list_configs, transfrom_factory=None):
    list_feat = []
    for item in list_configs:
        feat = features_dict[item['name']]
        if 'pre_transform' in item:
            transform = setup_transforms(item['pre_transform'], transfrom_factory=transfrom_factory)
            feat = transform(feat)
        list_feat.append(feat)

    return concatenate_features(list_feat)


def remove_edge_full(edges_ids,
                     edges_label,
                     edges_features,
                     ids_to_remove: Tuple[tuple]):
    for id_to_rm in ids_to_remove:
        edges_features = remove_edges(edges_ids, edges_features, id_to_rm)
        edges_label = remove_edges(edges_ids, edges_label, id_to_rm)
        edges_ids = remove_edges(edges_ids, edges_ids, id_to_rm)
    return edges_ids, edges_label, edges_features


def remove_node_full(nodes_ids, edges_ids,
                     nodes_label, edges_label,
                     nodes_features, edges_features,
                     ids_to_remove: Tuple[int]):

    for offset, id_to_rm in enumerate(ids_to_remove):
        original_id_to_rm = id_to_rm - offset
        _, _, nodes_label, edges_label = remove_node(nodes_ids,
                                                     edges_ids,
                                                     nodes_label,
                                                     edges_label,
                                                     original_id_to_rm)

        nodes_ids, edges_ids, nodes_features, edges_features = remove_node(nodes_ids,
                                                                           edges_ids,
                                                                           nodes_features,
                                                                           edges_features,
                                                                           original_id_to_rm)

    return nodes_ids, edges_ids, nodes_label, edges_label, nodes_features, edges_features


def graph_preprocessing(nodes_ids, edges_ids, edges_label, edges_features):
    edges_features = filter_label_from_edges_feature(edges_ids, edges_features)
    edges_label = filter_label_from_edges_feature(edges_ids, edges_label)
    edges_ids = filter_label_from_edges_ids(edges_ids)

    nodes_ids, edges_ids = rectify_graph(nodes_ids, edges_ids)
    return nodes_ids, edges_ids, edges_label, edges_features


def map_nodes_labels(nodes_label):
    mapped_labels = np.zeros_like(nodes_label)
    nodes_to_mask = []
    for i in range(mapped_labels.shape[0]):
        new_label = gt_mapping[nodes_label[i]]
        if new_label is None:
            nodes_to_mask.append(i)
        else:
            mapped_labels[i] = new_label
    return mapped_labels, nodes_to_mask


