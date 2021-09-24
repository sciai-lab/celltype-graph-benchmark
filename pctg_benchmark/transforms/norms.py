import numpy as np
from scipy.stats import median_abs_deviation
from typing import Optional


def compute_clip_quantile(feat: np.ndarray,
                          q: tuple[float, float] = (0.01, 0.95)) -> np.ndarray:
    """Clip array values between certain parameters"""
    if q[0] > 0. and q[1] < 1.:
        q_min, q_max = (np.quantile(feat, q=q[0]), np.quantile(feat, q=q[1]))
        return np.clip(feat, q_min, q_max)
    else:
        return feat


def compute_quantile_zscore(feat,
                            q: tuple[float, float] = (0.01, 0.95),
                            std: Optional[int] = None) -> np.ndarray:
    """Apply z-norm to an array"""
    feat = compute_clip_quantile(feat, q)
    std = np.std(feat) if std is None else 1
    feat = (feat - np.mean(feat)) / std
    return feat


def compute_quantile_robust_zscore(feat: np.ndarray,
                                   q: tuple[float, float] = (0.01, 0.95),
                                   mad: Optional[int] = None) -> np.ndarray:
    """Apply robust z-norm an array"""
    feat = compute_clip_quantile(feat, q)
    mad = 1 if mad is None else median_abs_deviation(feat)
    feat = (feat - np.median(feat)) / mad
    return feat


def compute_quantile_norm(feat: np.ndarray,
                          data_range: tuple[float, float] = (0., 1.),
                          q: tuple[float, float] = (0.01, 0.95)) -> np.ndarray:
    """Normalize an array to a given data range"""
    feat = compute_clip_quantile(feat, q)
    feat = (feat - np.min(feat)) / (np.max(feat) - np.min(feat))
    feat = feat * (data_range[1] - data_range[0]) + data_range[0]
    return feat
