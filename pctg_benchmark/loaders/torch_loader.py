import os.path

import torch
from torch_geometric.data import InMemoryDataset, download_url
from pctg_benchmark.loaders.build_dataset import build_cv_splits, default_build_torch_geometric_data


class PCTG(InMemoryDataset):
    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 split: int = 0,
                 phase: str = 'test',
                 raw_transform_config: dict = None,
                 grs: tuple[str] = ('es_pca_grs',),
                 number_splits: int = 5,
                 force_process: bool = False,
                 file_list_path: str = None) -> None:

        raw_data_paths = [os.path.join(root, 'raw', _grs) for _grs in grs]
        self.name_grs = '_'.join([_grs for _grs in grs])

        self.splits = build_cv_splits(raw_data_paths,
                                      file_list_path=file_list_path,
                                      number_splits=number_splits,
                                      )
        self.split, self.phase = split, phase
        self.raw_file_metas = self.splits[split][phase]
        self._raw_file_names = [meta['path'] for meta in self.raw_file_metas]
        self.raw_transform_config = raw_transform_config

        super().__init__(root, transform, pre_transform)
        if force_process:
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root,
                            f'processed_{self.name_grs}_{self.phase}_split{self.split}')

    @property
    def raw_file_names(self):
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def build_data_list(self):
        data_list = []
        for meta in self.raw_file_metas:
            path, unique_idx = meta['path'], meta['unique_idx']
            data = default_build_torch_geometric_data(path,
                                                      self.raw_transform_config,
                                                      meta)
            data_list.append(data)

        return data_list

    def process(self):
        data_list = self.build_data_list()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
