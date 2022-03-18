import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj
from ctg_benchmark.utils.utils import create_cell_mapping


def filter_label_from_edges_feature(edges_ids, edges_features, label=0):
    mask = np.logical_and(edges_ids[:, 0] != label, edges_ids[:, 1] != label)
    mask = np.where(mask)[0]
    return edges_features[mask]


def filter_label_from_edges_ids(edges_ids, label=0):
    return filter_label_from_edges_feature(edges_ids, edges_ids, label=label)


def rectify_graph(cell_ids, edges_ids):
    cell_ids_mapping = create_cell_mapping(cell_ids, np.arange(cell_ids.shape[0]))
    new_edges_ids = [[cell_ids_mapping[e1], cell_ids_mapping[e2]] for e1, e2 in edges_ids]
    return np.arange(cell_ids.shape[0]), np.array(new_edges_ids)


def remove_edges(edges_ids, edges_features, ids_to_remove=(0, 0)):
    ids_rm_min, ids_rm_max = min(ids_to_remove), max(ids_to_remove)
    for i, (e1, e2) in enumerate(edges_ids):
        e_min, e_max = min(e1, e2), max(e1, e2)
        if e_min == ids_rm_min and e_max == ids_rm_max:
            return np.delete(edges_features, i, axis=0)
    else:
        print(f"No edges has been deleted, {ids_to_remove} not in graph")

    return edges_features


def _remove_ids(ids, feat, id_to_remove):
    ids = np.delete(ids, id_to_remove, axis=0)
    feat = np.delete(feat, id_to_remove, axis=0)
    return ids, feat


def remove_node(cell_ids, edges_ids,
                cell_features, edges_features,
                id_to_remove):
    cell_ids, cell_features = _remove_ids(cell_ids,
                                          cell_features,
                                          id_to_remove)

    edges_ids_to_remove = [i for i, e in enumerate(edges_ids) if id_to_remove in e]
    edges_ids, edges_features = _remove_ids(edges_ids,
                                            edges_features,
                                            edges_ids_to_remove)

    cell_ids, edges_ids = rectify_graph(cell_ids, edges_ids)
    return cell_ids, edges_ids, cell_features, edges_features


class RandomAdjDropout:
    def __init__(self, p: float = 0.05):
        self.p = p

    def __call__(self, data: Data) -> Data:
        data.edge_index, data.edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.p)
        return data
