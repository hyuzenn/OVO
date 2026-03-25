from __future__ import annotations

import numpy as np
import torch

from risk_pipeline.mapping.mapper import MappingConfig, RiskFeatureMapper


def _pose_with_translation(tx: float, ty: float, tz: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def test_mapper_integrate_with_positions() -> None:
    mapper = RiskFeatureMapper(MappingConfig(voxel_size=1.0))
    mapper.initialize_map(feature_dim=4)

    z_i_prime = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    positions_local = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [1.2, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )
    T_t = _pose_with_translation(1.0, 0.0, 0.0)

    indices = mapper.integrate(z_i_prime=z_i_prime, bbox_or_position=positions_local, T_t=T_t)
    assert indices == [(1, 0, 0), (2, 0, 0)]

    state = mapper.export_state()
    assert len(state["voxels"]) == 2
    np.testing.assert_allclose(state["voxels"][(1, 0, 0)]["mean_feature"], np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(state["voxels"][(2, 0, 0)]["mean_feature"], np.array([0.0, 1.0, 0.0, 0.0]))


def test_mapper_integrate_with_bbox_and_average_accumulation() -> None:
    mapper = RiskFeatureMapper(MappingConfig(voxel_size=1.0))
    mapper.initialize_map(feature_dim=2)

    z_i_prime = np.array(
        [
            [1.0, 1.0],
            [3.0, 3.0],
        ],
        dtype=np.float64,
    )
    bboxes = np.array(
        [
            [0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
            [0.4, 0.2, 0.2, 0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )

    mapper.integrate(z_i_prime=z_i_prime, bbox_or_position=bboxes, T_t=np.eye(4), assume_bbox=True)
    state = mapper.export_state()

    voxel = state["voxels"][(0, 0, 0)]
    assert voxel["count"] == 2
    np.testing.assert_allclose(voxel["mean_feature"], np.array([2.0, 2.0]))
