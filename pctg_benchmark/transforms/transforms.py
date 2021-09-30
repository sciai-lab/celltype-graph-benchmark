from pctg_benchmark.transforms.utils import class_from_func, to_camel_case
from dataclasses import dataclass
import importlib
import copy
from torch_geometric.transforms import Compose


@dataclass
class TransformFactory:
    transform_modules: list[str]

    def __post_init__(self):
        for module_name in self.transform_modules:
            module = importlib.import_module(module_name)
            functions = [function for function in dir(module) if function.find('compute_') == 0]
            for function in functions:
                func = getattr(module, function)
                self.__setattr__(to_camel_case(function), class_from_func(func))


all_transforms = TransformFactory(['pctg_benchmark.transforms.basics',
                                   'pctg_benchmark.transforms.norms',
                                   'pctg_benchmark.transforms.sanity_check']
                                  )


def setup_transforms(transforms_list):
    transforms = []
    for feat_config in transforms_list:
        feat_config = copy.copy(feat_config)
        name = feat_config['name']
        if isinstance(name, str) and hasattr(all_transforms, name):
            transform_class = all_transforms.__getattribute__(name)

        elif hasattr(name, '__call__'):
            transform_class = name

        else:
            raise NotImplementedError

        feat_config.pop('name')

        transform_instance = transform_class(**feat_config)
        transforms.append(transform_instance)
    return Compose(transforms)





