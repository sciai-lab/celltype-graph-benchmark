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


def compute_random_shuffle(feat: np.ndarray, seed: int = 0) -> np.ndarray:
    """Shuffle features randomly - to be used only for sanity check"""
    np.random.seed(seed)
    np.random.shuffle(feat)
    return feat


def compute_set_to_value(feat: np.ndarray, value: int = 0) -> np.ndarray:
    """Set all features to a single value - to be used only for sanity check"""
    return value * np.ones_like(feat)


def compute_set_to_random(feat: np.ndarray, mode: str = 'rand') -> np.ndarray:
    """Replace features with random features - to be used only for sanity check"""
    if mode == 'rand':
        return np.random.rand(*feat.shape)
    elif mode == 'normal':
        return np.random.randn(*feat.shape)
    else:
        raise NotImplementedError
