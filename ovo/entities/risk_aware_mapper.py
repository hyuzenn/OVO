"""End-to-end RiskAwareSystem integration.

Pipeline:
  data input (SG-FRONT JSON)
    -> retrieval risk cue r_i^retr
    -> relation risk cue r_i^rel
    -> risk score P_f
    -> residual gated modulation z_i'
    -> TSDF update interface
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import numpy as np

from .failure_modulation import FailureConditionedModulation
from .failure_retrieval import FailureEpisodeRetrieval
from .risk_dummy_pipeline import DummyBatch, RiskAwareTSDFUpdater
from .sgfront_dataloader import SGFrontDataLoader
from .triplet_gcn import TripletGCNEncoder


@dataclass(slots=True)
class RiskAwareMapperConfig:
    embedding_dim: int = 32
    geom_dim: int = 8
    num_edge_types: int = 16
    retrieval_k: int = 3
    message_passing_steps: int = 2
    lambda_1: float = 0.5  # weight for P_obj
    lambda_2: float = 0.5  # weight for P_ctx
    device: str = "cpu"
    failure_memory_json: str | None = None


class RiskAwareMapper:
    """Integrated mapper for Step1~Step4 interface.

    Shapes through the pipeline:
      z_i       [B, N, D]
      r_i_retr  [B, N, D]
      r_i_rel   [B, N, D]
      p_obj     [B, N, 1]
      p_ctx     [B, N, 1]
      p_f       [B, N, 1]
      z_i_prime [B, N, D]
    """

    def __init__(self, cfg: RiskAwareMapperConfig) -> None:
        self.cfg = cfg

        self.loader = SGFrontDataLoader(
            embedding_dim=cfg.embedding_dim,
            embedding_mode="random",
            device=cfg.device,
        )
        self.retriever = FailureEpisodeRetrieval(
            dim=cfg.embedding_dim,
            geom_dim=cfg.geom_dim,
            k=cfg.retrieval_k,
            memory_json=cfg.failure_memory_json,
            device=cfg.device,
        )
        self.encoder = TripletGCNEncoder(
            dim=cfg.embedding_dim,
            num_edge_types=cfg.num_edge_types,
            message_passing_steps=cfg.message_passing_steps,
        )
        self.modulator = FailureConditionedModulation(dim=cfg.embedding_dim)
        self.tsdf_updater = RiskAwareTSDFUpdater(dim=cfg.embedding_dim)

    def run(self, relationships_json: str | Path, obj_boxes_json: str | Path) -> Dict[str, Any]:
        obj_dict, graph = self.loader.load(relationships_json, obj_boxes_json)
        batch = self.loader.to_tensor_batch(graph)

        outputs = self.forward_batch(batch)
        self._write_back_to_graph(graph, outputs)

        return {
            "object_dict": obj_dict,
            "graph": graph,
            "batch": batch,
            **outputs,
        }

    def forward_batch(self, batch: DummyBatch) -> Dict[str, torch.Tensor]:
        # Step-1 Retrieval
        r_i_retr, p_i_retr = self.retriever(batch.z_i, batch.bbox_geom)  # [B,N,D], [B,N,1]

        # Step-2 Relation Context
        r_i_rel = self.encoder(batch.z_i, batch.edge_idx, batch.edge_type)  # [B,N,D]

        # Risk decomposition
        p_obj = torch.clamp(p_i_retr, 0.0, 1.0)  # [B,N,1]
        p_ctx = torch.sigmoid(r_i_rel.norm(dim=-1, keepdim=True))  # [B,N,1]
        p_f = self.cfg.lambda_1 * p_obj + self.cfg.lambda_2 * p_ctx  # [B,N,1]
        p_f = torch.clamp(p_f, 0.0, 1.0)

        # Step-3 Modulation
        z_i_prime = self.modulator(batch.z_i, r_i_retr, r_i_rel, p_f)  # [B,N,D]

        # Step-4 placeholder interface (TSDF update)
        map_v = self.update_map(z_i_prime, batch.pose_t)  # [B,Vx,Vy,Vz,D]

        return {
            "r_i_retr": r_i_retr,
            "p_i_retr": p_i_retr,
            "r_i_rel": r_i_rel,
            "p_obj": p_obj,
            "p_ctx": p_ctx,
            "p_f": p_f,
            "z_i_prime": z_i_prime,
            "map_v": map_v,
        }

    def update_map(self, z_i_prime: torch.Tensor, pose_t: torch.Tensor) -> torch.Tensor:
        """Step-4 TSDF update interface placeholder.

        Replace this function body with real TSDF/voxel fusion integration.
        """

        return self.tsdf_updater(z_i_prime, pose_t)

    @staticmethod
    def summarize_outputs(outputs: Dict[str, torch.Tensor], topk: int = 5) -> str:
        p_f = outputs["p_f"][0, :, 0].detach()  # [N]
        lines = [
            f"nodes={p_f.numel()}",
            f"p_f mean={float(p_f.mean()):.4f}, max={float(p_f.max()):.4f}",
        ]
        vals, idx = torch.topk(p_f, k=min(topk, p_f.numel()))
        for rank, (i, v) in enumerate(zip(idx.tolist(), vals.tolist()), start=1):
            lines.append(f"top{rank}: node={i}, p_f={v:.4f}")
        return "\n".join(lines)

    @staticmethod
    def embedding_delta_norm(z_i: torch.Tensor, z_i_prime: torch.Tensor) -> torch.Tensor:
        """Return per-node ||z_i' - z_i||_2, shape [B, N]."""

        return (z_i_prime - z_i).norm(dim=-1)

    @staticmethod
    def _write_back_to_graph(graph, outputs: Dict[str, torch.Tensor]) -> None:
        """Write key outputs back into graph nodes for downstream inspection."""

        node_ids = sorted(graph.nodes.keys())
        for i, nid in enumerate(node_ids):
            node = graph.nodes[nid]
            node.r_i_retr = outputs["r_i_retr"][0, i].detach().cpu().tolist()
            node.r_i_rel = outputs["r_i_rel"][0, i].detach().cpu().tolist()
            node.z_i_prime = outputs["z_i_prime"][0, i].detach().cpu().tolist()
            node.risk.p_obj = float(outputs["p_obj"][0, i, 0].item())
            node.risk.p_ctx = float(outputs["p_ctx"][0, i, 0].item())
            node.risk.p_f = float(outputs["p_f"][0, i, 0].item())

    @staticmethod
    def export_scene_arrays(graph, batch: DummyBatch, outputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Export numpy arrays for evaluation/visualization."""

        node_ids = sorted(graph.nodes.keys())
        centers = np.array([graph.nodes[nid].bbox.center for nid in node_ids], dtype=np.float32)
        p_f = outputs["p_f"][0, :, 0].detach().cpu().numpy()
        delta_norm = (outputs["z_i_prime"] - batch.z_i).norm(dim=-1)[0].detach().cpu().numpy()
        return {"centers": centers, "p_f": p_f, "delta_norm": delta_norm}
