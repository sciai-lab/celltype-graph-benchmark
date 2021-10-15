import os.path
from typing import Tuple
from abc import ABC, abstractmethod
import torch
from torch_geometric.data import InMemoryDataset, download_url
from pctg_benchmark.loaders.build_dataset import default_build_torch_geometric_data
from pctg_benchmark.loaders.build_dataset import build_cv_splits, build_std_splits
from pctg_benchmark.utils.io import save_yaml, load_yaml
from pctg_benchmark.utils.utils import get_basic_loader_config
import tqdm


class PCTG(InMemoryDataset, ABC):

    def __init__(self, root,
                 raw_file_metas: list,
                 processed_dir: str,
                 transform=None,
                 pre_transform=None,
                 raw_transform_config: dict = None,
                 force_process: bool = False) -> None:

        self.raw_file_metas = raw_file_metas
        self._raw_file_names = [meta['path'] for meta in raw_file_metas]
        self._processed_dir = processed_dir
        self.in_edges_attr: int = None
        self.in_features: int = None
        self.raw_transform_config = raw_transform_config
        self.raw_transform_config_path = os.path.join(processed_dir, 'raw_transform_config.yaml')
        os.makedirs(processed_dir, exist_ok=True)

        super().__init__(root, transform, pre_transform)
        if force_process or not os.path.isfile(self.processed_paths[0]) or self._check_raw_config():
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.in_edges_attr = int(self.data.in_edges_attr[0])
        self.in_features = int(self.data.in_features[0])

    def _check_raw_config(self):
        if not os.path.isfile(self.raw_transform_config_path):
            # if file does not exist default was used
            return True

        if self.raw_transform_config is None:
            # load default
            new_config = get_basic_loader_config('dataset')
        else:
            new_config = self.raw_transform_config

        # check if configs match if not re-run process
        old_config = load_yaml(self.raw_transform_config_path)
        return old_config != new_config

    @abstractmethod
    def get_raw_file_metas(self, *args):
        pass

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self):
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        """ To be implemented upon release"""
        # Download to `self.raw_dir`.
        pass

    def build_data_list(self):
        data_list, config = [], None
        for meta in tqdm.tqdm(self.raw_file_metas, desc='Processing RawFiles'):
            path, unique_idx = meta['path'], meta['unique_idx']
            data, config = default_build_torch_geometric_data(path, self.raw_transform_config, meta)
            data_list.append(data)

        return data_list, config

    def process(self):
        data_list, config = self.build_data_list()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        save_yaml(config, self.raw_transform_config_path)


class PCTGCrossValidationSplit(PCTG):
    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 split: int = 0,
                 phase: str = 'test',
                 raw_transform_config: dict = None,
                 grs: Tuple[str] = ('es_pca_grs',),
                 number_splits: int = 5,
                 force_process: bool = False,
                 file_list_path: str = None) -> None:

        name_grs = '_'.join([_grs for _grs in grs])
        processed_dir = os.path.join(root, f'processed_{name_grs}_{phase}_split{split}')
        raw_file_metas = self.get_raw_file_metas(root, split, phase, grs, file_list_path, number_splits)
        super().__init__(root=root,
                         raw_file_metas=raw_file_metas,
                         processed_dir=processed_dir,
                         transform=transform,
                         pre_transform=pre_transform,
                         raw_transform_config=raw_transform_config,
                         force_process=force_process)

    @staticmethod
    def get_raw_file_metas(root, split, phase, grs, file_list_path, number_splits):
        raw_data_paths = [os.path.join(root, 'raw', _grs) for _grs in grs]
        splits = build_cv_splits(raw_data_paths,
                                 file_list_path=file_list_path,
                                 number_splits=number_splits)
        raw_file_metas = splits[split][phase]
        return raw_file_metas


class PCTGSimpleSplit(PCTG):
    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 ratio: tuple = (0.6, 0.1, 0.3),
                 seed: int = 0,
                 phase: str = 'test',
                 raw_transform_config: dict = None,
                 grs: Tuple[str] = ('es_pca_grs',),
                 force_process: bool = False,
                 file_list_path: str = None) -> None:

        name_grs = '_'.join([_grs for _grs in grs])
        ratio_name = '_'.join([str(_r).replace('0.', '') for _r in ratio])

        processed_dir = os.path.join(root, f'processed_{name_grs}_{ratio_name}_{phase}_seeds{seed}')
        raw_file_metas = self.get_raw_file_metas(root, ratio, seed, phase, grs, file_list_path)
        super().__init__(root=root,
                         raw_file_metas=raw_file_metas,
                         processed_dir=processed_dir,
                         transform=transform,
                         pre_transform=pre_transform,
                         raw_transform_config=raw_transform_config,
                         force_process=force_process)

    @staticmethod
    def get_raw_file_metas(root, ratio, seed, phase, grs, file_list_path):
        raw_data_paths = [os.path.join(root, 'raw', _grs) for _grs in grs]
        splits = build_std_splits(raw_data_paths,
                                  ratio,
                                  seed=seed,
                                  file_list_path=file_list_path)
        raw_file_metas = splits[phase]
        return raw_file_metas
