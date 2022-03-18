import os.path
from typing import Tuple
from abc import ABC, abstractmethod
import torch
from torch_geometric.data import InMemoryDataset
from ctg_benchmark.loaders.build_dataset import default_build_torch_geometric_data
from ctg_benchmark.loaders.build_dataset import build_dataset
from ctg_benchmark.loaders.build_dataset import build_cv_splits, build_std_splits
from ctg_benchmark.utils.io import save_yaml, load_yaml
from torch_geometric.loader import DataLoader
from ctg_benchmark.utils.utils import get_basic_loader_config
import tqdm


class CTG(InMemoryDataset, ABC):

    def __init__(self, root,
                 raw_file_metas: list,
                 processed_dir: str,
                 transform=None,
                 pre_transform=None,
                 raw_transform_config: dict = None,
                 force_process: bool = False, ) -> None:

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

        self.in_edges_attr = int(self.data.edge_attr.shape[1])
        self.in_features = int(self.data.x.shape[1])

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

    @staticmethod
    def custom_download(grs, raw_path):
        for _grs in grs:
            grs_raw_dir = os.path.join(raw_path, _grs)
            if not os.path.isdir(grs_raw_dir):
                build_dataset(raw_path, dataset_name=_grs)

    def download(self):
        pass

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


class CTGCrossValidationSplit(CTG):
    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 split: int = 0,
                 phase: str = 'val',
                 seed: int = 0,
                 raw_transform_config: dict = None,
                 grs: Tuple[str] = ('label_grs_surface',),
                 number_splits: int = 5,
                 force_process: bool = False,
                 file_list_path: str = None,
                 directory_structure: tuple = ('ovules-celltype-dataset', 'raw')) -> None:
        """
        torch geometric InMemory DataSet
        Parameters
        ----------
        root: root containing the dataset (if not present the dataset will be downloaded)
        transform: train time data augmentation
        pre_transform: build time data processing
        split: split in use, must be between 0 and number_splits
        phase: must be on of [train, val, test]
        seed: random seed, to ensure reproducibility of the splits
        raw_transform_config: dictionary detailing what features to load and what processing applying to the them
        grs: Global reference systems
        number_splits: number of cross validation splits, default 5
        force_process: force re-processing of features
        file_list_path: list of dataset raw stacks
        directory_structure: default dataset directory structure
        """

        name_grs = '_'.join([_grs for _grs in grs])
        processed_dir = os.path.join(root, f'processed_{name_grs}_{phase}_split{split}_num_splits{number_splits}')
        raw_path = os.path.join(root, directory_structure[0], directory_structure[1])
        self.custom_download(grs, raw_path)
        raw_file_metas = self.get_raw_file_metas(raw_path, split, phase, grs, file_list_path, number_splits, seed)
        super().__init__(root=root,
                         raw_file_metas=raw_file_metas,
                         processed_dir=processed_dir,
                         transform=transform,
                         pre_transform=pre_transform,
                         raw_transform_config=raw_transform_config,
                         force_process=force_process)

    @staticmethod
    def get_raw_file_metas(raw_path, split, phase, grs, file_list_path, number_splits, seed):
        raw_paths_grs = [os.path.join(raw_path, _grs) for _grs in grs]
        splits = build_cv_splits(raw_paths_grs,
                                 file_list_path=file_list_path,
                                 number_splits=number_splits,
                                 seed=seed)
        raw_file_metas = splits[split][phase]
        return raw_file_metas


class CTGSimpleSplit(CTG):
    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 ratio: tuple = (0.6, 0.1, 0.3),
                 seed: int = 0,
                 phase: str = 'test',
                 raw_transform_config: dict = None,
                 grs: Tuple[str] = ('label_grs_surface',),
                 force_process: bool = False,
                 file_list_path: str = None,
                 directory_structure: tuple = ('ovules-celltype-dataset', 'raw')) -> None:
        """
        torch geometric InMemory DataSet
        Parameters
        ----------
        root: root containing the dataset (if not present the dataset will be downloaded)
        transform: train time transformation
        pre_transform: build time data processing
        ratio: ratio between train, test, and validation
        seed: random seed, to ensure reproducibility of the splits
        phase: define if the dataset is the test, train or val split
        raw_transform_config: single features transform to be applied before creating the torch_geometric data object
        grs: Global reference systems
        force_process: force re-processing of features
        file_list_path: list of dataset raw stacks
        directory_structure: default dataset directory structure
        """
        name_grs = '_'.join([_grs for _grs in grs])
        ratio_name = '_'.join([str(_r).replace('0.', '') for _r in ratio])

        processed_dir = os.path.join(root, f'processed_{name_grs}_{ratio_name}_{phase}_seeds{seed}')
        raw_path = os.path.join(root, directory_structure[0], directory_structure[1])
        self.custom_download(grs, raw_path)
        raw_file_metas = self.get_raw_file_metas(raw_path, ratio, seed, phase, grs, file_list_path)
        super().__init__(root=root,
                         raw_file_metas=raw_file_metas,
                         processed_dir=processed_dir,
                         transform=transform,
                         pre_transform=pre_transform,
                         raw_transform_config=raw_transform_config,
                         force_process=force_process)

    @staticmethod
    def get_raw_file_metas(raw_path, ratio, seed, phase, grs, file_list_path):
        raw_paths_grs = [os.path.join(raw_path, _grs) for _grs in grs]
        splits = build_std_splits(raw_paths_grs,
                                  ratio,
                                  seed=seed,
                                  file_list_path=file_list_path)
        raw_file_metas = splits[phase]
        return raw_file_metas


def get_split_loaders(root,
                      batch_size=1,
                      shuffle=True,
                      transform=None,
                      pre_transform=None,
                      ratio: tuple = (0.6, 0.1, 0.3),
                      seed: int = 0,
                      raw_transform_config: dict = None,
                      grs: Tuple[str] = ('label_grs_surface',),
                      force_process: bool = False,
                      **loaderkwargs) -> dict:
    """
    Returns a simple split DataLoader for the CellTypeGraph Benchmark
    Parameters
    ----------
    root: root containing the dataset (if not present the dataset will be downloaded)
    batch_size: default 1
    shuffle: if true train data loader is going to be shuffled
    transform: train time transformation
    pre_transform: build time data processing
    ratio: ratio between train, test, and validation
    seed: random seed, to ensure reproducibility of the splits
    raw_transform_config: single features transfrom to be applied before creating the torch_geometric data object
    grs: Global reference systems
    force_process: force re-processing of features
    loaderkwargs: torch_geometric.loader.DataLoader arguments
    """
    loaders = {'train': None, 'val': None, 'test': None}
    for mode in loaders:
        transform = transform if mode == 'train' else None
        mode_dataset = CTGSimpleSplit(root=root,
                                      phase=mode,
                                      transform=transform,
                                      pre_transform=pre_transform,
                                      ratio=ratio,
                                      seed=seed,
                                      raw_transform_config=raw_transform_config,
                                      grs=grs,
                                      force_process=force_process)

        shuffle = shuffle if mode == 'train' else False
        loaders[mode] = DataLoader(mode_dataset,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   **loaderkwargs)
    return loaders


def get_cross_validation_loaders(root,
                                 batch_size=1,
                                 shuffle=True,
                                 transform=None,
                                 pre_transform=None,
                                 number_splits: int = 5,
                                 seed: int = 0,
                                 raw_transform_config: dict = None,
                                 grs: Tuple[str] = ('label_grs_surface',),
                                 force_process: bool = False,
                                 **loaderkwargs) -> dict:
    """
    Returns a n-fold cross validation split DataLoader for the CellTypeGraph Benchmark
    Parameters
    ----------
    root: root containing the dataset (if not present the dataset will be downloaded)
    batch_size: default 1
    shuffle: if true train data loader is going to be shuffled
    transform: train time transformation
    pre_transform: build time data processing
    seed: random seed, to ensure reproducibility of the splits
    raw_transform_config: dictionary detailing what features to load and what processing applying to each feature
    grs: Global reference systems
    number_splits: number of cross validation splits, default 5
    force_process: force re-processing of features
    loaderkwargs: torch_geometric.loader.DataLoader arguments
    """

    cv_loaders = {}
    for split in range(number_splits):
        loaders = {'train': None, 'val': None}
        for mode in loaders:
            transform = transform if mode == 'train' else None
            mode_dataset = CTGCrossValidationSplit(root=root,
                                                   split=split,
                                                   phase=mode,
                                                   transform=transform,
                                                   pre_transform=pre_transform,
                                                   number_splits=number_splits,
                                                   seed=seed,
                                                   raw_transform_config=raw_transform_config,
                                                   grs=grs,
                                                   force_process=force_process)

            shuffle = shuffle if mode == 'train' else False
            loaders[mode] = DataLoader(mode_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       **loaderkwargs)

        cv_loaders[split] = loaders
    return cv_loaders

