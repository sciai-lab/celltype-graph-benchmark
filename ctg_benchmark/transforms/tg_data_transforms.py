import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj


class RandomNormalNoise:
    """
    Add gaussian Noise to node and edges features
    """
    def __init__(self, noise_sigma: float = 0.1):
        """
        noise_sigma: define the standard deviation of the sampled noise
        """
        self.noise_sigma = noise_sigma

    def __call__(self, data: Data) -> Data:
        noise = self.noise_sigma * torch.randn_like(data.x)
        data.x = data.x + noise

        noise = self.noise_sigma * torch.randn_like(data.edge_attr)
        data.edge_attr = data.edge_attr + noise
        return data


class RandomAdjDropout:
    def __init__(self, p: float = 0.05):
        self.p = p

    def __call__(self, data: Data) -> Data:
        data.edge_index, data.edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.p)
        return data
