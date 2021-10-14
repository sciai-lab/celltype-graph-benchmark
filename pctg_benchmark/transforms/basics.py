import numpy as np
from typing import Optional
import torch
from typing import Tuple


def compute_to_onehot(feat: np.ndarray,
                      max_channel: int = None,
                      extreme: Tuple[float, float] = (0, 1)) -> np.ndarray:
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


def compute_abs(feat: np.ndarray) -> np.ndarray:
    return np.abs(feat)


def compute_to_torch_tensor(feat: np.ndarray,
                            data_type: str = 'float') -> torch.tensor:
    if data_type == 'int':
        feat = feat.astype('int64')
        tensor_feat = torch.from_numpy(feat)
        tensor_feat = tensor_feat.long()

    elif data_type == 'float':
        tensor_feat = torch.from_numpy(feat)
        tensor_feat = tensor_feat.float()

    else:
        raise NotImplementedError
    return tensor_feat
