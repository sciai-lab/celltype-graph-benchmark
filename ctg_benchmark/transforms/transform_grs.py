import numpy as np
import numpy.typing as npt
from plantcelltype.features.cell_vector_features import to_unit_vector


def transform_coordinates(points_coo: npt.ArrayLike,
                          origin: npt.ArrayLike,
                          axis: npt.ArrayLike):
    """ NewCoo = NewAxis dot (TrivialCoo - NewOrigin)"""
    points_coo = points_coo - origin
    points_coo = np.dot(points_coo, axis.T)
    return points_coo


def inv_transform_coordinates(points_coo: npt.ArrayLike,
                              origin: npt.ArrayLike,
                              axis: npt.ArrayLike) -> npt.ArrayLike:
    """ TrivialCoo = NewAxis^-1 dot OldCoo + NewOrigin)"""
    inv_axis = np.linalg.inv(axis)
    points_coo = np.dot(points_coo, inv_axis.T)
    points_coo = points_coo + origin
    return points_coo


def scale_points(points_coo: npt.ArrayLike,
                 scaling: npt.ArrayLike,
                 reverse: bool = False) -> npt.ArrayLike:
    scaling = 1 / scaling if reverse else scaling
    return points_coo * scaling


def vectors_to_orientation(vectors_array: npt.ArrayLike) -> npt.ArrayLike:
    out_orientation_vectors_array = np.zeros((vectors_array.shape[0], 6))

    out_orientation_vectors_array[:, :3] = vectors_array ** 2
    out_orientation_vectors_array[:, 3] = vectors_array[:, 0] * vectors_array[:, 1]
    out_orientation_vectors_array[:, 4] = vectors_array[:, 1] * vectors_array[:, 2]
    out_orientation_vectors_array[:, 5] = vectors_array[:, 2] * vectors_array[:, 0]
    return out_orientation_vectors_array


def orientations_to_vectors(orientation_vectors_array: npt.ArrayLike) -> npt.ArrayLike:
    out_vectors_array = np.zeros((orientation_vectors_array.shape[0], 3))
    a = np.sqrt(orientation_vectors_array[:, 0])
    b = np.sqrt(orientation_vectors_array[:, 1])
    c = np.sqrt(orientation_vectors_array[:, 2])
    out_vectors_array[:, 0] = a
    out_vectors_array[:, 1] = b * np.sign(orientation_vectors_array[:, 3])
    out_vectors_array[:, 2] = c * np.sign(orientation_vectors_array[:, 5])
    return out_vectors_array


class BasisTransformer:
    def __init__(self, origin, axis, new_origin, new_axis):
        self.origin = origin
        self.axis = axis
        self.new_origin = new_origin
        self.new_axis = new_axis

    def _change_coo_basis(self, points_coo: npt.ArrayLike, new_axis, new_origin) -> npt.ArrayLike:
        points_coo_trivial = inv_transform_coordinates(points_coo, axis=self.axis, origin=self.origin)
        return transform_coordinates(points_coo_trivial, axis=new_axis, origin=new_origin)

    def change_coo_basis(self, points_coo: npt.ArrayLike) -> npt.ArrayLike:
        return self._change_coo_basis(points_coo, new_axis=self.new_axis, new_origin=self.new_origin)

    def change_vector_basis(self, points_coo: npt.ArrayLike) -> npt.ArrayLike:
        return self._change_coo_basis(points_coo, new_origin=self.origin, new_axis=self.new_axis)

    def change_orientation_basis(self,
                                 points_coo: npt.ArrayLike) -> npt.ArrayLike:
        new_points = orientations_to_vectors(orientation_vectors_array=points_coo)

        new_points = self.change_vector_basis(new_points)

        new_orientations = vectors_to_orientation(new_points)
        return new_orientations


def change_basis_cell_features(cell_features, base_transformer):
    # adapt cell_features
    features = {'invariant': ['bg_edt_um',
                              'hops_to_bg',
                              'com_voxels',
                              'degree_centrality',
                              'hops_to_es',
                              'lrs_axis2_angle_grs',
                              'rw_centrality',
                              'surface_um',
                              'surface_voxels',
                              'volume_um',
                              'volume_voxels',
                              'length_axis1_grs',
                              'length_axis2_grs',
                              'length_axis3_grs',
                              'lrs_axis12_dot_grs',
                              'lrs_proj_axis1_grs',
                              'lrs_proj_axis2_grs',
                              'lrs_proj_axis3_grs',
                              'pca_explained_variance_grs',
                              'pca_proj_axis1_grs',
                              'pca_proj_axis2_grs',
                              'pca_proj_axis3_grs',
                              'proj_length_unit_sphere', ],
                'coordinates': ['com_grs'],
                'vectors': ['lrs_axis1_grs',
                            'lrs_axis2_grs',
                            'lrs_axis3_grs',
                            'pca_axis1_grs',
                            'pca_axis2_grs',
                            'pca_axis3_grs'],
                'orientations': ['lrs_orientation_axis1_grs',
                                 'lrs_orientation_axis2_grs',
                                 'lrs_orientation_axis3_grs',
                                 'pca_orientation_axis1_grs',
                                 'pca_orientation_axis2_grs',
                                 'pca_orientation_axis3_grs', ]
                }

    new_cell_features = {}
    for key, feat in cell_features.items():
        if key in features['invariant']:
            new_feat = feat.copy()

        elif key in features['coordinates']:
            new_feat = base_transformer.change_coo_basis(feat)

        elif key in features['vectors']:
            new_feat = base_transformer.change_vector_basis(feat)

        elif key in features['orientations']:
            new_feat = base_transformer.change_orientation_basis(feat)

        elif key == 'com_proj_grs':
            new_feat = base_transformer.change_coo_basis(cell_features['com_grs'])
            new_feat = to_unit_vector(new_feat)
            new_feat = new_feat.dot(base_transformer.new_axis)
        else:
            raise ValueError

        new_cell_features[key] = new_feat
    return new_cell_features


def change_basis_edges_features(edges_features, base_transformer):
    # adapt edges_features
    features = {'invariant': ['com_distance_um',
                              'lrs1e1_dot_ev_grs',
                              'lrs1e2_dot_ev_grs',
                              'lrs2e1_dot_ev_grs',
                              'lrs2e2_dot_ev_grs',
                              'lrs3e1_dot_ev_grs',
                              'lrs3e2_dot_ev_grs',
                              'lrs_dot_axis1_grs',
                              'lrs_dot_axis2_grs',
                              'lrs_dot_axis3_grs',
                              'lrs_proj_grs',
                              'com_voxels',
                              'surface_um',
                              'surface_voxels'],
                'coordinates': ['com_grs'],
                }

    new_edges_features = {}
    for key, feat in edges_features.items():
        if key in features['invariant']:
            new_feat = feat.copy()

        elif key in features['coordinates']:
            new_feat = base_transformer.change_coo_basis(feat)

        elif key == 'plane_vectors_grs':
            new_coo = base_transformer.change_coo_basis(feat[:, 0])
            new_dir = base_transformer.change_vector_basis(feat[:, 1])
            new_feat = np.stack([new_coo, new_dir], axis=1)
        else:
            raise ValueError

        new_edges_features[key] = new_feat
    return new_edges_features


def change_samples(sample_dict, base_transformer):
    new_sample_dict = {}
    for s_key, feat in sample_dict.items():
        if s_key.find('grs') != -1:
            new_sample_dict[s_key] = base_transformer.change_coo_basis(feat)
        else:
            new_sample_dict[s_key] = feat.copy()
    return new_sample_dict


def change_fullstack_basis(stack, new_axis, new_origin):
    axis = stack['attributes']['global_reference_system_axis']
    origin = stack['attributes']['global_reference_system_origin']
    bt = BasisTransformer(origin=origin, axis=axis, new_origin=new_origin, new_axis=new_axis)

    new_stack = {key: stack[key].copy() for key in ['attributes',
                                                    'cell_ids',
                                                    'cell_labels',
                                                    'edges_ids',
                                                    'edges_labels',
                                                    'grs',
                                                    'labels',
                                                    'rag_boundaries',
                                                    'segmentation']}
    new_stack['attributes']['global_reference_system_axis'] = new_axis.copy()
    new_stack['attributes']['global_reference_system_origin'] = new_origin.copy()

    new_stack['cell_features'] = change_basis_cell_features(stack['cell_features'], bt)
    new_stack['edges_features'] = change_basis_edges_features(stack['edges_features'], bt)
    for key in ['cell_samples', 'edges_samples']:
        new_stack[key] = change_samples(stack[key], bt)

    return new_stack


def generate_all_grs_stacks(stack):
    dict_new_stacks = {}
    for axis_key, origin_key in [('es_pca_grs_axis', 'es_pca_grs_origin'),
                                 ('es_trivial_grs_axis', 'es_trivial_grs_origin'),
                                 ('label_grs_funiculus_axis', 'label_grs_funiculus_origin'),
                                 ('label_grs_surface_axis', 'label_grs_surface_origin')]:
        new_axis = stack['grs'][axis_key]
        new_origin = stack['grs'][origin_key]
        name = axis_key.replace('_axis', '')
        dict_new_stacks[name] = change_fullstack_basis(stack, new_axis=new_axis, new_origin=new_origin)

    return dict_new_stacks


