import numpy as np
from typing import Optional


def compute_to_onehot(feat: np.ndarray,
                      max_channel: Optional[int] = None,
                      extreme: Optional[tuple[float, float]] = (0, 1)) -> np.ndarray:
    """Create a one hot encoding of an integer array"""
    assert max_channel > 1, f'onehot feature must have at least 1 channel. given: {max_channel}'

    # scale data between 0-max
    feat = feat - np.min(feat)

    # clip max channel
    max_channel = np.max(feat) + 1 if max_channel is None else max_channel
    max_channel = int(max_channel)
    feat = np.clip(feat, 0, max_channel - 1)

    # create onehot encoding
    feat_onehot = np.zeros((feat.shape[0], max_channel))
    feat_onehot += extreme[0]

    feat_onehot[range(feat.shape[0]), feat] = extreme[1]
    return feat_onehot
