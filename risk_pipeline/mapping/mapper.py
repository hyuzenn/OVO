from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transforms import to_numpy, transform_points
from .voxel_map import SparseVoxelMap


@dataclass(slots=True)
class MappingConfig:
    voxel_size: float = 0.5


class RiskFeatureMapper:
    """Independent mapping branch for Integrate(V, z_i', T_t).

    V is represented by `bbox_or_position` (per-node 3D points in sensor/local frame),
    z_i' are modulation-updated node features, and T_t maps local frame -> world frame.
    """

    def __init__(self, config: MappingConfig | None = None) -> None:
        self.config = config or MappingConfig()
        self.map: SparseVoxelMap | None = None

    def initialize_map(self, feature_dim: int) -> None:
        self.map = SparseVoxelMap(voxel_size=self.config.voxel_size, feature_dim=feature_dim)

    def integrate(self, z_i_prime, bbox_or_position, T_t, *, assume_bbox: bool = False) -> list[tuple[int, int, int]]:
        """Integrate(V, z_i', T_t) into sparse voxel map.

        Args:
            z_i_prime: [N, D] features after modulation.
            bbox_or_position: [N, 3] positions or [N, 6] boxes (center in first 3 values).
            T_t: [4, 4] local/sensor to world pose.
            assume_bbox: if True, force interpreting input as bounding boxes.
        """
        if self.map is None:
            raise RuntimeError("Map is not initialized. Call initialize_map(feature_dim) first.")

        features = to_numpy(z_i_prime).astype(np.float64)
        positions = self._extract_positions(bbox_or_position, assume_bbox=assume_bbox)

        if features.ndim != 2:
            raise ValueError(f"Expected z_i_prime shape [N, D], got {features.shape}")
        if positions.shape[0] != features.shape[0]:
            raise ValueError(
                f"Batch mismatch between features ({features.shape[0]}) and positions ({positions.shape[0]})"
            )
        if features.shape[1] != self.map.feature_dim:
            raise ValueError(f"Feature dim mismatch: {features.shape[1]} vs {self.map.feature_dim}")

        positions_world = transform_points(T_t, positions)
        integrated_indices = []
        for feature, position_world in zip(features, positions_world):
            idx = self.map.integrate(position_world=position_world, feature=feature)
            integrated_indices.append(idx)
        return integrated_indices

    def export_state(self) -> dict:
        if self.map is None:
            raise RuntimeError("Map is not initialized.")
        return self.map.export()

    @staticmethod
    def _extract_positions(bbox_or_position, *, assume_bbox: bool) -> np.ndarray:
        arr = to_numpy(bbox_or_position).astype(np.float64)
        print(
            "[mapper] _extract_positions input "
            f"type={type(bbox_or_position).__name__} shape={arr.shape} assume_bbox={assume_bbox}"
        )
        if arr.ndim != 2:
            raise ValueError(f"Expected shape [N, 3] or [N, 6], got {arr.shape}")

        if arr.shape[1] == 3 and not assume_bbox:
            return arr
        if arr.shape[1] >= 3:
            return arr[:, :3]
        raise ValueError(f"Cannot extract positions from shape {arr.shape}")
