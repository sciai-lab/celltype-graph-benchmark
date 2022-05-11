from ctg_benchmark.utils.io import open_full_stack, export_full_stack
from ctg_benchmark.transforms.transform_grs import change_fullstack_basis
from pathlib import Path
import os

# class ids in the original dataset
# 1: ab oi/oi2 / L1
# 2: ad oi/oi1 / L2
# 3: ab ii/ii2 / L3
# 4: ad ii/ii1 / L4
# 5: nu
# 5: pc
# 7: fu / Funiculus
# 8: es
# 10: ac
# 14: L5 - merged with L4
# 0: to be ignored

gt_mapping = {0: None, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 14: 3}
inv_gt_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10}

original_name_mapping = {0: 'ignore',
                         1: 'ab oi/oi2 - L1', 2: 'ad oi/oi1 - L2', 3: 'ab ii/ii2 - L3', 4: 'ad ii/ii1 - L4',
                         5: 'nu', 6: 'pc', 7: 'fu / Funiculus', 8: 'es', 10: 'ac',
                         14: 'L5 - merged with L4'}
net_name_mapping = {key: original_name_mapping[value] for key, value in inv_gt_mapping.items()}

# Find the global path
ctg_global_path = Path(__file__).parent.absolute()
resources_dir = 'resources'

ctg_basic_loader_config = os.path.join(ctg_global_path, resources_dir, 'loader_base_config.yaml')
default_dataset_file_list = os.path.join(ctg_global_path, resources_dir, 'list_data.csv')

urls = {'label_grs_surface': {'url': 'https://heibox.uni-heidelberg.de/f/23c79badbc9e46f59242/?dl=1',
                              'md5sum': '51e21dc509a9205b9ec47f38a307b156'}
        }
