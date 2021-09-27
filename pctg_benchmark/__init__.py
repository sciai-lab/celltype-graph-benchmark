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

original_name_mapping = {0: 'ignore', 1: 'ab oi/oi2 / L1', 2: 'ad oi/oi1 / L2', 3: 'ab ii/ii2 / L3', 4: 'ad ii/ii1 / L4',
                         5: 'nu', 6: 'pc', 7: 'fu / Funiculus', 8: 'es', 10: 'ac', 14: 'L5 - merged with L4'}
net_name_mapping = {key: original_name_mapping[value] for key, value in inv_gt_mapping.items()}