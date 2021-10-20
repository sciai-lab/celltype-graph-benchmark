import os
from pathlib import Path

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
pctg_global_path = Path(__file__).parent.absolute()
resources_dir = 'resources'

pctg_basic_loader_config = os.path.join(pctg_global_path, resources_dir, 'loader_base_config.yaml')
default_dataset_file_list = os.path.join(pctg_global_path, resources_dir, 'list_data.csv')

anonymous_url = b'x\x9c\xcb())(\xb6\xd2\xd7\xcfH\xcdL\xca\xaf\xd0+\xcd\xcb\xd4\x052SRs\x92R\x8b\xd2\xf5RR\xf5\xd3\xf4S-\xcd\r\x92S\x8d,\x13\x8d\xcc\x92M,\x8d\x12-R\xd2R\xf5\xedSrl\r\x01\x98\xe4\x14/'
