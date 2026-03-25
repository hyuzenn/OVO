from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


VoxelIndex = Tuple[int, int, int]


@dataclass(slots=True)
class VoxelFeature:
    feature_sum: np.ndarray
    count: int = 0

    def update(self, feature: np.ndarray) -> None:
        if self.feature_sum.shape != feature.shape:
            raise ValueError(f"Feature shape mismatch: {self.feature_sum.shape} vs {feature.shape}")
        self.feature_sum += feature
        self.count += 1

    @property
    def mean_feature(self) -> np.ndarray:
        if self.count == 0:
            return self.feature_sum
        return self.feature_sum / float(self.count)


@dataclass(slots=True)
class SparseVoxelMap:
    voxel_size: float
    feature_dim: int
    voxels: Dict[VoxelIndex, VoxelFeature] = field(default_factory=dict)

    def position_to_index(self, position_world: np.ndarray) -> VoxelIndex:
        coord = np.floor(position_world / self.voxel_size).astype(int)
        return int(coord[0]), int(coord[1]), int(coord[2])

    def integrate(self, position_world: np.ndarray, feature: np.ndarray) -> VoxelIndex:
        idx = self.position_to_index(position_world)
        if idx not in self.voxels:
            self.voxels[idx] = VoxelFeature(feature_sum=np.zeros(self.feature_dim, dtype=np.float64), count=0)
        self.voxels[idx].update(feature.astype(np.float64))
        return idx

    def export(self) -> dict:
        return {
            "voxel_size": self.voxel_size,
            "feature_dim": self.feature_dim,
            "voxels": {
                idx: {"count": voxel.count, "mean_feature": voxel.mean_feature.copy()}
                for idx, voxel in self.voxels.items()
            },
        }
