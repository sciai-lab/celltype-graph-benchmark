import torch


def torch_transform_coordinates(points_coo: torch.Tensor,
                                origin: torch.Tensor,
                                axis: torch.Tensor):
    """ NewCoo = NewAxis dot (TrivialCoo - NewOrigin)"""
    points_coo = points_coo - origin
    points_coo = torch.matmul(points_coo, axis.T)
    return points_coo


def torch_inv_transform_coordinates(points_coo: torch.Tensor,
                                    origin: torch.Tensor,
                                    axis: torch.Tensor) -> torch.Tensor:
    """ TrivialCoo = NewAxis^-1 dot OldCoo + NewOrigin)"""
    inv_axis = torch.linalg.inv(axis)
    points_coo = torch.matmul(points_coo, inv_axis.T)
    points_coo = points_coo + origin
    return points_coo


def torch_scale_points(points_coo: torch.Tensor,
                       scaling: torch.Tensor,
                       reverse: bool = False) -> torch.Tensor:
    scaling = 1 / scaling if reverse else scaling
    return points_coo * scaling


def torch_vectors_to_orientation(vectors_array: torch.Tensor) -> torch.Tensor:
    out_orientation_vectors_array = torch.zeros(vectors_array.shape[0], 6)

    out_orientation_vectors_array[:, :3] = vectors_array ** 2
    out_orientation_vectors_array[:, 3] = vectors_array[:, 0] * vectors_array[:, 1]
    out_orientation_vectors_array[:, 4] = vectors_array[:, 1] * vectors_array[:, 2]
    out_orientation_vectors_array[:, 5] = vectors_array[:, 2] * vectors_array[:, 0]
    return out_orientation_vectors_array


def torch_orientations_to_vectors(orientation_vectors_array: torch.Tensor) -> torch.Tensor:
    out_vectors_array = torch.zeros((orientation_vectors_array.shape[0], 3))
    a = torch.sqrt(orientation_vectors_array[:, 0])
    b = torch.sqrt(orientation_vectors_array[:, 1])
    c = torch.sqrt(orientation_vectors_array[:, 2])
    out_vectors_array[:, 0] = a
    out_vectors_array[:, 1] = b * torch.sign(orientation_vectors_array[:, 3])
    out_vectors_array[:, 2] = c * torch.sign(orientation_vectors_array[:, 5])
    return out_vectors_array


class TorchBasisTransformer:
    def __init__(self, origin, axis, new_origin, new_axis, slices=None):
        self.origin = origin
        self.axis = axis

        self.new_origin = new_origin
        self.new_axis = new_axis
        self.slices = slices

    def change_coo_basis(self, points_coo: torch.Tensor, *args, **kwarg) -> torch.Tensor:
        points_coo_trivial = torch_inv_transform_coordinates(points_coo, axis=self.axis, origin=self.origin)
        return torch_transform_coordinates(points_coo_trivial, axis=self.new_axis, origin=self.new_origin)

    def change_vector_basis(self, points_coo: torch.Tensor, *args, **kwarg) -> torch.Tensor:
        return self.change_coo_basis(points_coo, self.new_axis, new_origin=self.origin)

    def change_orientation_basis(self, points_coo: torch.Tensor, *args, **kwarg) -> torch.Tensor:
        new_points = torch_orientations_to_vectors(orientation_vectors_array=points_coo)

        new_points = self.change_coo_basis(new_points, self.new_axis, new_origin=self.origin)

        new_orientations = torch_vectors_to_orientation(new_points)
        return new_orientations

    def __call__(self, data):
        return data
