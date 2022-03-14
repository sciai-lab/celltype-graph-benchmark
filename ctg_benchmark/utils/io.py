import os
import numpy as np
import h5py
import yaml


def open_full_stack(path, keys=None):
    with h5py.File(path, 'r') as f:
        stacks = {'attributes': {}}
        for _key, _value in f.attrs.items():
            stacks['attributes'][_key] = _value

        if keys is None:
            keys = f.keys()

        for _key in keys:
            if isinstance(f[_key], h5py.Group):
                stacks[_key] = {}
                for __keys in f[_key].keys():
                    stacks[_key][__keys] = f[_key][__keys][...]

            elif isinstance(f[_key], h5py.Dataset):
                stacks[_key] = f[_key][...]

    return stacks


def _update_dict(template_dict, up_dict):
    for key, value in up_dict.items():
        if isinstance(up_dict[key], dict) and key in template_dict:
            template_dict[key] = _update_dict(template_dict[key], up_dict[key])
        else:
            template_dict[key] = up_dict[key]

    return template_dict


def load_yaml(config_path):
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    def load(loader, node):
        with open(node.value, 'r') as _f:
            return yaml.full_load(_f)

    def home_path(loader, node):
        return os.path.expanduser('~')

    yaml.add_constructor('!join', join)
    yaml.add_constructor('!home_path', home_path)
    yaml.add_constructor('!update', load)
    yaml.add_constructor('!load', load)
    with open(config_path, 'r') as f:
        config = yaml.full_load(f)

    if '_internal_variables' in config and 'template' in config['_internal_variables']:
        template_config = config['_internal_variables']['template']
        del template_config['_internal_variables']
        del config['_internal_variables']
        config = _update_dict(template_config, config)
    else:
        if '_internal_variables' in config:
            del config['_internal_variables']

    return config


def save_yaml(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f)


def export_full_stack(path, stack):
    for key, value in stack.items():
        if isinstance(value, dict):
            for group_key, group_value in value.items():
                if key == "attributes":
                    create_h5_attrs(path, group_value, group_key)
                else:
                    create_h5(path, group_value, key=f"{key}/{group_key}", voxel_size=None)

        elif isinstance(value, np.ndarray):
            if value.ndim == 3:
                voxel_size = stack['attributes'].get('element_size_um', [1.0, 1.0, 1.0])
            else:
                voxel_size = None
            create_h5(path, value, key=key, voxel_size=voxel_size)


def create_h5(path, stack, key, voxel_size=(1.0, 1.0, 1.0), mode='a'):
    del_h5_key(path, key)
    with h5py.File(path, mode) as f:
        f.create_dataset(key, data=stack, compression='gzip')
        # save voxel_size
        if voxel_size is not None:
            f[key].attrs['element_size_um'] = voxel_size


def create_h5_attrs(path, value, key, mode='a'):
    with h5py.File(path, mode) as f:
        f.attrs[key] = value


def del_h5_key(path, key, mode='a'):
    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]
