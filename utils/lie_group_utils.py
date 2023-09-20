import torch

from utils.constant_utils import TENSOR_EYE3


def is_tensor(variable):
    return torch.is_tensor(variable)


def is_close(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if is_tensor(a):
        return torch.isclose(a, b, rtol, atol, equal_nan)
    else:
        return None


def outer(a, b):
    if is_tensor(a):
        return torch.outer(a, b)
    else:
        return None


def skew(vector):
    if is_tensor(vector):
        matrix = torch.tensor(
            [[0, -vector[2], vector[1]],
             [vector[2], 0, -vector[0]],
             [-vector[1], vector[0], 0]],
            dtype=torch.float64
        )
        return matrix.to(vector.device)
    else:
        return None


def so3exp(lie_algebra):
    if is_tensor(lie_algebra):
        lie_algebra_norm = lie_algebra.norm()

        if is_close(lie_algebra_norm, torch.tensor(0.0, dtype=torch.float64)):
            lie_algebra_skew = skew(lie_algebra)
            lie_group = TENSOR_EYE3.to(lie_algebra.device) + lie_algebra_skew
            return lie_group

        lie_algebra_unit = lie_algebra / lie_algebra_norm
        lie_algebra_unit_skew = skew(lie_algebra_unit)

        c = lie_algebra_norm.cos()
        s = lie_algebra_norm.sin()
        lie_group = c * TENSOR_EYE3.to(lie_algebra.device) + (1 - c) * outer(lie_algebra_unit, lie_algebra_unit) + s * lie_algebra_unit_skew
        return lie_group
    else:
        return None


def sen3exp(lie_algebra):
    if is_tensor(lie_algebra):
        lie_algebra_so = lie_algebra[:3]
        lie_algebra_so_norm = torch.norm(lie_algebra_so)

        if is_close(lie_algebra_so_norm, torch.tensor(0.0, dtype=torch.float64)):
            lie_algebra_so_skew = skew(lie_algebra_so)
            lie_group_so = TENSOR_EYE3.to(lie_algebra.device) + lie_algebra_so_skew
            jacobian = TENSOR_EYE3.to(lie_algebra.device) + 0.5 * lie_algebra_so_skew
        else:
            lie_algebra_so_unit = lie_algebra_so / lie_algebra_so_norm
            lie_algebra_so_unit_skew = skew(lie_algebra_so_unit)

            s = torch.sin(lie_algebra_so_norm)
            c = torch.cos(lie_algebra_so_norm)
            jacobian = (s / lie_algebra_so_norm) * TENSOR_EYE3.to(lie_algebra.device) + (1 - s / lie_algebra_so_norm) * outer(lie_algebra_so_unit, lie_algebra_so_unit) \
                + ((1 - c) / lie_algebra_so_norm) * lie_algebra_so_unit_skew
            lie_group_so = c * TENSOR_EYE3.to(lie_algebra.device) + (1 - c) * outer(lie_algebra_so_unit, lie_algebra_so_unit) + s * lie_algebra_so_unit_skew

        lie_algebra_vectors = jacobian.mm(lie_algebra[3:].view(-1, 3).t())
        return lie_group_so, lie_algebra_vectors[:, 0], lie_algebra_vectors[:, 1]
    else:
        return None

