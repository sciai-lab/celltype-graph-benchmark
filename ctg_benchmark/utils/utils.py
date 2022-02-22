import numpy as np
from numba import njit
from ctg_benchmark.utils.io import load_yaml
from ctg_benchmark import ctg_basic_loader_config


def cantor_sym_depair(z):
    w = np.floor((np.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = int(z - t)
    x = int(w - y)
    return min(x, y), max(x, y)


@njit
def cantor_sym_pair(k1, k2):
    _k1, _k2 = min(k1, k2), max(k1, k2)
    z = int(0.5 * (_k1 + _k2) * (_k1 + _k2 + 1) + _k2)
    return z


def check_safe_cast(array, types=('int64', 'float64')):
    for _types in types:
        if np.can_cast(array.dtype, _types):
            out_type = _types
            return out_type
    else:
        raise RuntimeError


def edges_ids2cantor_ids(edges_ids):
    return np.array([cantor_sym_pair(e1, e2) for e1, e2 in edges_ids])


def create_features_mapping(features_ids, features, safe_cast=True):
    if safe_cast:
        out_type = check_safe_cast(features)
        features = features.astype(out_type)

    mapping = {}
    for key, value in zip(features_ids, features):
        mapping[key] = value
    return mapping


def create_cell_mapping(cell_ids, cell_feature, safe_cast=True):
    return create_features_mapping(cell_ids, cell_feature, safe_cast=safe_cast)


def create_edge_mapping(edges_ids, edges_features, safe_cast=True):
    cantor_edges_ids = edges_ids2cantor_ids(edges_ids)
    return create_features_mapping(cantor_edges_ids, edges_features, safe_cast=safe_cast)


def get_basic_loader_config(key: str = None):
    config = load_yaml(ctg_basic_loader_config)
    config = config.get(key) if key is not None else config
    return config
