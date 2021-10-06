import numpy as np
from pctg_benchmark.transforms.transforms import setup_transforms
from pctg_benchmark.transforms.graph import filter_label_from_edges_feature, filter_label_from_edges_ids
from pctg_benchmark.transforms.graph import rectify_graph, remove_edges, remove_node


def collect_features(features_dict, list_configs, as_array=True):
    list_feat = []
    for item in list_configs:
        feat = features_dict[item['name']]
        if 'pre_transform' in item:
            transform = setup_transforms(item['pre_transform'])
            feat = transform(feat)
        list_feat.append(feat)

    list_feat = [feat[:, None] if feat.ndim == 1 else feat for feat in list_feat]
    array_feat = np.concatenate(list_feat, axis=1) if as_array else list_feat
    return array_feat


def remove_edge_full(edges_ids,
                     edges_label,
                     edges_features,
                     ids_to_remove: tuple[tuple]):
    for id_to_rm in ids_to_remove:
        edges_features = remove_edges(edges_ids, edges_features, id_to_rm)
        edges_label = remove_edges(edges_ids, edges_label, id_to_rm)
        edges_ids = remove_edges(edges_ids, edges_ids, id_to_rm)
    return edges_ids, edges_label, edges_features


def remove_node_full(nodes_ids, edges_ids,
                     nodes_label, edges_label,
                     nodes_features, edges_features,
                     ids_to_remove: tuple[int]):

    for offset, id_to_rm in enumerate(ids_to_remove):
        original_id_to_rm = id_to_rm - offset
        _, _, nodes_label, edges_label = remove_node(nodes_ids,
                                                     edges_ids,
                                                     nodes_label,
                                                     edges_label,
                                                     original_id_to_rm)

        nodes_ids, edges_ids, nodes_features, edges_features = remove_node(nodes_ids,
                                                                           edges_ids,
                                                                           nodes_features,
                                                                           edges_features,
                                                                           original_id_to_rm)

    return nodes_ids, edges_ids, nodes_label, edges_label, nodes_features, edges_features


def graph_preprocessing(nodes_ids, edges_ids, edges_label, edges_features):
    edges_features = filter_label_from_edges_feature(edges_ids, edges_features)
    edges_label = filter_label_from_edges_feature(edges_ids, edges_label)
    edges_ids = filter_label_from_edges_ids(edges_ids)

    nodes_ids, edges_ids = rectify_graph(nodes_ids, edges_ids)
    return nodes_ids, edges_ids, edges_label, edges_features

