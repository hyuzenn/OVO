"""Dummy end-to-end risk-aware semantic mapping pipeline.

Important design note:
- This pipeline starts from Scene Graph inputs (nodes + relation triplets + bbox features).
- It does NOT consume raw RGB-D frames directly, unlike the default OVO-SLAM ingestion path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .failure_retrieval import FailureEpisodeRetrieval
from .triplet_gcn import TripletGCNEncoder
from .failure_modulation import FailureConditionedModulation


@dataclass(slots=True)
class DummyBatch:
    """Synthetic scene-graph batch.

    Shapes:
      z_i:       [B, N, D]   semantic embeddings
      bbox_geom: [B, N, G]   bbox-derived geometry features
      edge_idx:  [B, E, 2]   directed edges (src, dst)
      edge_type: [B, E]      relation type ids
      pose_t:    [B, 4, 4]   global camera/SLAM pose (optional for voxel integration)
    """

    z_i: torch.Tensor
    bbox_geom: torch.Tensor
    edge_idx: torch.Tensor
    edge_type: torch.Tensor
    pose_t: torch.Tensor


class RiskAwareTSDFUpdater(torch.nn.Module):
    """Step-4 map writer (dummy voxel fusion).

    Input:
      z_i_prime [B, N, D]
      pose_t    [B, 4, 4]
    Output:
      map_v     [B, Vx, Vy, Vz, D]
    """

    def __init__(self, dim: int, voxel_size: Tuple[int, int, int] = (8, 8, 8)):
        super().__init__()
        self.dim = dim
        self.voxel_size = voxel_size

    def forward(self, z_i_prime: torch.Tensor, pose_t: torch.Tensor) -> torch.Tensor:
        bsz = z_i_prime.shape[0]
        vx, vy, vz = self.voxel_size
        scene_feat = z_i_prime.mean(dim=1)  # [B, D]
        map_v = scene_feat[:, None, None, None, :].expand(bsz, vx, vy, vz, self.dim).contiguous()
        _ = pose_t  # kept to document interface with SLAM/global frame transformation
        return map_v


@dataclass(slots=True)
class RiskAwarePipeline:
    """Composed E2E dummy pipeline.

    Note: entrypoint consumes Scene Graph tensor batch, not RGB-D image tensors.
    """

    retriever: FailureEpisodeRetrieval
    encoder: TripletGCNEncoder
    modulator: FailureConditionedModulation
    tsdf_updater: RiskAwareTSDFUpdater

    def __call__(self, batch: DummyBatch) -> dict[str, torch.Tensor]:
        r_i_retr, p_i_retr = self.retriever(batch.z_i, batch.bbox_geom)
        r_i_rel = self.encoder(batch.z_i, batch.edge_idx, batch.edge_type)
        p_f = 0.5 * p_i_retr + 0.5 * torch.sigmoid(r_i_rel.norm(dim=-1, keepdim=True))
        z_i_prime = self.modulator(batch.z_i, r_i_retr, r_i_rel, p_f)
        map_v = self.tsdf_updater(z_i_prime, batch.pose_t)
        return {
            "r_i_retr": r_i_retr,
            "p_i_retr": p_i_retr,
            "r_i_rel": r_i_rel,
            "p_f": p_f,
            "z_i_prime": z_i_prime,
            "map_v": map_v,
        }


def build_dummy_batch(
    batch_size: int = 2,
    num_nodes: int = 6,
    num_edges: int = 10,
    dim: int = 32,
    geom_dim: int = 8,
    num_edge_types: int = 16,
    device: str = "cpu",
) -> DummyBatch:
    """Create random scene-graph tensors for quick E2E validation."""

    z_i = torch.randn(batch_size, num_nodes, dim, device=device)
    bbox_geom = torch.randn(batch_size, num_nodes, geom_dim, device=device)
    edge_idx = torch.randint(0, num_nodes, (batch_size, num_edges, 2), device=device)
    edge_type = torch.randint(0, num_edge_types, (batch_size, num_edges), device=device)
    pose_t = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    return DummyBatch(z_i=z_i, bbox_geom=bbox_geom, edge_idx=edge_idx, edge_type=edge_type, pose_t=pose_t)


def run_dummy_pipeline(
    device: str = "cpu",
    retrieval_cls: type[torch.nn.Module] = FailureEpisodeRetrieval,
    retrieval_kwargs: dict | None = None,
    encoder_cls: type[torch.nn.Module] = TripletGCNEncoder,
    encoder_kwargs: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Run end-to-end with random noise inputs.

    Returns a dictionary of intermediate/final tensors so callers can verify shape compatibility.
    """

    torch.manual_seed(7)
    dim = 32
    geom_dim = 8
    num_edge_types = 16

    batch = build_dummy_batch(dim=dim, geom_dim=geom_dim, num_edge_types=num_edge_types, device=device)

    retrieval_kwargs = retrieval_kwargs or {}
    encoder_kwargs = encoder_kwargs or {}

    pipeline = RiskAwarePipeline(
        retriever=retrieval_cls(dim=dim, geom_dim=geom_dim, **retrieval_kwargs),
        encoder=encoder_cls(dim=dim, num_edge_types=num_edge_types, **encoder_kwargs),
        modulator=FailureConditionedModulation(dim=dim),
        tsdf_updater=RiskAwareTSDFUpdater(dim=dim),
    )

    outputs = pipeline(batch)
    return outputs


if __name__ == "__main__":
    out = run_dummy_pipeline()
    for k, v in out.items():
        print(f"{k:>10s}: {tuple(v.shape)}")
