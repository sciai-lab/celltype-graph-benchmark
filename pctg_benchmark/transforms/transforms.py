from pctg_benchmark.transforms.utils import class_from_func, to_camel_case
from dataclasses import dataclass
import importlib
import copy
from torch_geometric.transforms import Compose
from typing import List


@dataclass
class TransformFactory:
    transform_modules: List[str] = None
    default_functional_name_key = 'compute_'

    def __post_init__(self):
        transform_modules_default: List[str] = ['pctg_benchmark.transforms.basics',
                                                'pctg_benchmark.transforms.norms',
                                                'pctg_benchmark.transforms.sanity_check']
        self.transform_modules = self.transform_modules if self.transform_modules is not None else []
        self.transform_modules += transform_modules_default

        for module_name in self.transform_modules:
            module = importlib.import_module(module_name)
            functions = [function.replace(self.default_functional_name_key, '') for function in dir(module)
                         if function.find(self.default_functional_name_key) == 0]
            for function in functions:
                func = getattr(module, function)
                self.__setattr__(to_camel_case(function), class_from_func(func))


all_transforms = TransformFactory()


def setup_transforms(transforms_list, transfrom_factory=None):
    transfrom_factory = transfrom_factory if transfrom_factory is not None else TransformFactory()
    transforms = []
    for feat_config in transforms_list:
        _feat_config = copy.copy(feat_config)
        name = _feat_config['name']
        if isinstance(name, str) and hasattr(all_transforms, name):
            transform_class = transfrom_factory.__getattribute__(name)

        elif hasattr(name, '__call__'):
            transform_class = name

        else:
            raise NotImplementedError

        _feat_config.pop('name')

        transform_instance = transform_class(**_feat_config)
        transforms.append(transform_instance)
    return Compose(transforms)
