from ctg_benchmark.transforms.utils import class_from_func, to_camel_case
from dataclasses import dataclass
import importlib
import copy
from torch_geometric.transforms import Compose
from typing import List


@dataclass
class TransformFactory:
    transform_modules: List[str] = None
    default_functional_name_key: str = 'compute_'

    def __post_init__(self):
        transform_modules_default: List[str] = ['ctg_benchmark.transforms.basics',
                                                'ctg_benchmark.transforms.norms',
                                                'ctg_benchmark.transforms.sanity_check']
        self.transform_modules = self.transform_modules if self.transform_modules is not None else []
        self.transform_modules += transform_modules_default

        for module_name in self.transform_modules:
            module = importlib.import_module(module_name)
            functions = [function for function in dir(module) if function.find(self.default_functional_name_key) == 0]
            for function in functions:
                func = getattr(module, function)
                function = function.replace(self.default_functional_name_key, '')
                self.__setattr__(to_camel_case(function), class_from_func(func))


default_factory = TransformFactory()


def setup_transforms(transforms_list, transform_factory: TransformFactory = None) -> Compose:
    transform_factory = transform_factory if transform_factory is not None else default_factory
    transforms = []
    for feat_config in transforms_list:
        _feat_config = copy.copy(feat_config)
        name = _feat_config['name']
        if isinstance(name, str) and hasattr(transform_factory, name):
            transform_class = transform_factory.__getattribute__(name)

        elif hasattr(name, '__call__'):
            transform_class = name

        else:
            raise NotImplementedError

        _feat_config.pop('name')

        transform_instance = transform_class(**_feat_config)
        transforms.append(transform_instance)
    return Compose(transforms)
