"""Dummy end-to-end risk-aware semantic mapping pipeline.

Important design note:
- This pipeline starts from Scene Graph inputs (nodes + relation triplets + bbox features).
- It does NOT consume raw RGB-D frames directly, unlike the default OVO-SLAM ingestion path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


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


class FailureEpisodeRetrieval(torch.nn.Module):
    """Step-1 retrieval block.

    Input:
      z_i      [B, N, D]
      bbox_geom[B, N, G]
    Output:
      r_i_retr [B, N, D]
      p_i_retr [B, N, 1]
    """

    def __init__(self, dim: int, geom_dim: int):
        super().__init__()
        self.fuser = torch.nn.Linear(dim + geom_dim, dim)
        self.prior = torch.nn.Linear(dim + geom_dim, 1)

    def forward(self, z_i: torch.Tensor, bbox_geom: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_i, bbox_geom], dim=-1)
        r_i_retr = torch.tanh(self.fuser(x))
        p_i_retr = torch.sigmoid(self.prior(x))
        return r_i_retr, p_i_retr


class TripletGCNEncoder(torch.nn.Module):
    """Step-2 context encoder.

    Input:
      node_feat [B, N, D]
      edge_idx  [B, E, 2]
      edge_type [B, E]
    Output:
      r_i_rel   [B, N, D]
    """

    def __init__(self, dim: int, num_edge_types: int):
        super().__init__()
        self.edge_emb = torch.nn.Embedding(num_edge_types, dim)
        self.msg_mlp = torch.nn.Linear(dim * 2, dim)

    def forward(self, node_feat: torch.Tensor, edge_idx: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, dim = node_feat.shape
        n_edges = edge_idx.shape[1]
        out = torch.zeros_like(node_feat)

        for b in range(bsz):
            src = edge_idx[b, :, 0].long()  # [E]
            dst = edge_idx[b, :, 1].long()  # [E]
            e_emb = self.edge_emb(edge_type[b].long())  # [E, D]
            msg = torch.cat([node_feat[b, src], e_emb], dim=-1)  # [E, 2D]
            msg = torch.tanh(self.msg_mlp(msg))  # [E, D]
            out[b].index_add_(0, dst, msg)

        deg = torch.zeros((bsz, n_nodes, 1), device=node_feat.device)
        one = torch.ones((bsz, n_edges, 1), device=node_feat.device)
        for b in range(bsz):
            deg[b].index_add_(0, edge_idx[b, :, 1].long(), one[b])

        r_i_rel = out / deg.clamp_min(1.0)
        return r_i_rel


class FailureConditionedModulation(torch.nn.Module):
    """Step-3 gated residual modulation.

    Input:
      z_i       [B, N, D]
      r_i_retr  [B, N, D]
      r_i_rel   [B, N, D]
      p_f       [B, N, 1]
    Output:
      z_i_prime [B, N, D]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = torch.nn.Linear(dim * 2 + 1, dim)
        self.delta = torch.nn.Linear(dim * 3, dim)

    def forward(
        self,
        z_i: torch.Tensor,
        r_i_retr: torch.Tensor,
        r_i_rel: torch.Tensor,
        p_f: torch.Tensor,
    ) -> torch.Tensor:
        gate_in = torch.cat([p_f, r_i_retr, r_i_rel], dim=-1)
        g_i = torch.sigmoid(self.gate(gate_in))

        delta_in = torch.cat([z_i, r_i_retr, r_i_rel], dim=-1)
        delta_i = torch.tanh(self.delta(delta_in))

        z_i_prime = z_i + g_i * delta_i
        return z_i_prime


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


def run_dummy_pipeline(device: str = "cpu") -> dict[str, torch.Tensor]:
    """Run end-to-end with random noise inputs.

    Returns a dictionary of intermediate/final tensors so callers can verify shape compatibility.
    """

    torch.manual_seed(7)
    dim = 32
    geom_dim = 8
    num_edge_types = 16

    batch = build_dummy_batch(dim=dim, geom_dim=geom_dim, num_edge_types=num_edge_types, device=device)

    pipeline = RiskAwarePipeline(
        retriever=FailureEpisodeRetrieval(dim=dim, geom_dim=geom_dim),
        encoder=TripletGCNEncoder(dim=dim, num_edge_types=num_edge_types),
        modulator=FailureConditionedModulation(dim=dim),
        tsdf_updater=RiskAwareTSDFUpdater(dim=dim),
    )

    outputs = pipeline(batch)
    return outputs


if __name__ == "__main__":
    out = run_dummy_pipeline()
    for k, v in out.items():
        print(f"{k:>10s}: {tuple(v.shape)}")
