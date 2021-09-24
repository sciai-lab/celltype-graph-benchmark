from pctg_benchmark.transforms.utils import class_from_func, to_camel_case
from dataclasses import dataclass
import torch
import importlib


@dataclass
class TransformFactory:
    transform_module: str

    def __post_init__(self):
        module = importlib.import_module(self.transform_module)
        functions = [function for function in dir(module) if function.find('compute_') == 0]
        for function in functions:
            func = getattr(module, function)
            self.__setattr__(to_camel_case(function), class_from_func(func))


normalization_transforms = TransformFactory('pctg_benchmark.transforms.norms')


def configure_transforms(transform_list):
    for key, value in config.items():
        print(key, value)
