import numpy as np
from scipy.stats import median_abs_deviation
from typing import Optional
from pctg_benchmark.transforms.utils import class_from_func


def clip_quantile(feat: np.ndarray,
                  q: tuple[float, float] = (0.01, 0.95)) -> np.ndarray:
    """Clip array values between certain parameters"""
    if q[0] > 0. and q[1] < 1.:
        q_min, q_max = (np.quantile(feat, q=q[0]), np.quantile(feat, q=q[1]))
        return np.clip(feat, q_min, q_max)
    else:
        return feat


ClipQuantile = class_from_func(clip_quantile)

def quantile_zscore(feat, q: tuple[float, float] = (0.01, 0.95), std: Optional[int] = None) -> np.ndarray:
    """Apply z-norm to an array"""
    feat = clip_quantile(feat, q)
    std = np.std(feat) if std is None else 1
    feat = (feat - np.mean(feat)) / std
    return feat


def quantile_robust_zscore(feat: np.ndarray,
                           q: tuple[float, float] = (0.01, 0.95),
                           mad: Optional[int] = None) -> np.ndarray:
    """Apply robust z-norm an array"""
    feat = clip_quantile(feat, q)
    mad = 1 if mad is None else median_abs_deviation(feat)
    feat = (feat - np.median(feat)) / mad
    return feat


def quantile_norm(feat: np.ndarray,
                  data_range: tuple[float, float] = (0., 1.),
                  q: tuple[float, float] = (0.01, 0.95)) -> np.ndarray:
    """Normalize an array to a given data range"""
    feat = clip_quantile(feat, q)
    feat = (feat - np.min(feat)) / (np.max(feat) - np.min(feat))
    feat = feat * (data_range[1] - data_range[0]) + data_range[0]
    return feat


def to_onehot(feat: np.ndarray,
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


def random_shuffle(feat: np.ndarray, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    np.random.shuffle(feat)
    return feat


def set_to_value(feat: np.ndarray, value: int = 0) -> np.ndarray:
    return value * np.ones_like(feat)


def set_to_random(feat: np.ndarray, mode='rand') -> np.ndarray:
    if mode == 'rand':
        return np.random.rand(*feat.shape)
    elif mode == 'normal':
        return np.random.randn(*feat.shape)
    else:
        raise NotImplementedError
