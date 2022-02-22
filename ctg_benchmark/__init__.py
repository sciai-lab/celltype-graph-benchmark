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

# TODO path need to be decoded upon acceptance
anonymous_urls = {'trivial_grs': b'x\x9c\xcb())(\xb6\xd2\xd7\xcfH\xcdL\xca\xaf\xd0+\xcd\xcb\xd4\x052SRs\x92R\x8b\xd2\xf5RR\xf5\xd3\xf4\x93\x93\x13-\xcc\x0c-\x93\xcd,M\xd2LLL\x12\x93\xccM,\xf5\xedSrl\r\x01\x97\x9a\x13\xd2',
                  'label_grs_surface': b'x\x9c\xcb())(\xb6\xd2\xd7\xcfH\xcdL\xca\xaf\xd0+\xcd\xcb\xd4\x052SRs\x92R\x8b\xd2\xf5RR\xf5\xd3\xf4\x8d\x8c\x93\xcd-\x93\x12S\x92\x92-SM\xcc\xd2L-\x8dL\x8c\xf4\xedSrl\r\x01\x9a<\x13\xfc',
                  'label_grs_funiculum': b'x\x9c\xcb())(\xb6\xd2\xd7\xcfH\xcdL\xca\xaf\xd0+\xcd\xcb\xd4\x052SRs\x92R\x8b\xd2\xf5RR\xf5\xd3\xf4\x13\x8d\xd2\x8c-\r\xcc\x93SSSLL\x12-\xcd,L,\x0c\xf4\xedSrl\r\x01\x97\xc1\x13\xcd',
                  'es_trivial_grs': b'x\x9c\xcb())(\xb6\xd2\xd7\xcfH\xcdL\xca\xaf\xd0+\xcd\xcb\xd4\x052SRs\x92R\x8b\xd2\xf5RR\xf5\xd3\xf4-\x12M\x8d\xcd\x0c\x8d\x8c\x92MM\x92L\x12\x8d\xd2\x12\r\xcd\x93\xf4\xedSrl\r\x01\x90Q\x13\xb6',
                  'es_pca_grs': b'x\x9c\xcb())(\xb6\xd2\xd7\xcfH\xcdL\xca\xaf\xd0+\xcd\xcb\xd4\x052SRs\x92R\x8b\xd2\xf5RR\xf5\xd3\xf4\r\xcd\x92\r\x13\x8dS-\xd3RL\x0cLRS\xcc-\x93L,\xf4\xedSrl\r\x01\x97\xc0\x13\xfa'
                  }