import numpy as np


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
