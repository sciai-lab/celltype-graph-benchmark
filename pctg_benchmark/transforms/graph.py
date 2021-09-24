import numpy as np


def filter_bg_from_edges(edges_ids, features, bg=0):
    mask = np.where(np.min(edges_ids, axis=1) != bg)[0]
    return features[mask]
