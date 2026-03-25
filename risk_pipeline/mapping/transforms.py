from __future__ import annotations

from typing import Iterable

import numpy as np


ArrayLike = np.ndarray | Iterable[float] | Iterable[Iterable[float]]


def to_numpy(array: ArrayLike) -> np.ndarray:
    """Convert array-like input (including torch tensors) to numpy."""
    if hasattr(array, "detach") and hasattr(array, "cpu"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def invert_transform(T_ab: ArrayLike) -> np.ndarray:
    """Return inverse of a 4x4 homogeneous transform matrix."""
    T = to_numpy(T_ab).astype(np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected shape (4, 4), got {T.shape}")

    R = T[:3, :3]
    t = T[:3, 3]

    T_ba = np.eye(4, dtype=np.float64)
    T_ba[:3, :3] = R.T
    T_ba[:3, 3] = -R.T @ t
    return T_ba


def transform_points(T_ab: ArrayLike, points_b: ArrayLike) -> np.ndarray:
    """Transform points from frame b to frame a with T_ab."""
    T = to_numpy(T_ab).astype(np.float64)
    points = to_numpy(points_b).astype(np.float64)

    if T.shape != (4, 4):
        raise ValueError(f"Expected shape (4, 4), got {T.shape}")
    if points.ndim == 1:
        points = points[None, :]
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points with last dim 3, got {points.shape}")

    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([points, ones], axis=1)
    transformed = (T @ homo.T).T
    return transformed[:, :3]


def object_to_world(T_world_object: ArrayLike, points_object: ArrayLike) -> np.ndarray:
    """Convert object-frame points into world frame."""
    return transform_points(T_world_object, points_object)


def world_to_object(T_world_object: ArrayLike, points_world: ArrayLike) -> np.ndarray:
    """Convert world-frame points into object frame."""
    return transform_points(invert_transform(T_world_object), points_world)
