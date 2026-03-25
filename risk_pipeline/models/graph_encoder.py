from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from risk_pipeline.core.structures import Object3D, SceneBundle


class RelationAwareGraphEncoder(nn.Module):
    """Lightweight relation-aware encoder for SceneBundle graphs.

    Input:
        scene_graph: SceneBundle

    Output:
        r_i_rel: torch.Tensor with shape [N, D]
            - N: number of objects in scene_graph.objects
            - D: hidden_dim

    Notes:
        - This module builds initial node features from object geometry + label hash.
        - One relation-aware message passing step is applied over directed edges.
        - The module intentionally keeps boundaries simple for Phase B.
    """

    def __init__(self, hidden_dim: int = 32, relation_vocab_size: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relation_vocab_size = relation_vocab_size

        # [center(3), size(3), yaw(1), label_scalar(1)] -> D
        self.node_proj = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rel_embedding = nn.Embedding(relation_vocab_size, hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _label_scalar(label: str) -> float:
        # Stable scalar in [-1, 1] to avoid introducing a text encoder dependency.
        value = sum(ord(ch) for ch in label) % 997
        return (value / 498.5) - 1.0

    def _build_node_features(self, objects: Dict[int, Object3D], node_order: List[int]) -> torch.Tensor:
        rows = []
        for object_id in node_order:
            obj = objects[object_id]
            label_scalar = self._label_scalar(obj.label)
            rows.append(
                [
                    float(obj.center[0]),
                    float(obj.center[1]),
                    float(obj.center[2]),
                    float(obj.size[0]),
                    float(obj.size[1]),
                    float(obj.size[2]),
                    float(obj.yaw),
                    float(label_scalar),
                ]
            )
        if not rows:
            return torch.zeros((0, 8), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def forward(self, scene_graph: SceneBundle) -> torch.Tensor:
        node_order = sorted(scene_graph.objects.keys())
        n_nodes = len(node_order)
        if n_nodes == 0:
            return torch.zeros((0, self.hidden_dim), dtype=torch.float32)

        id_to_index = {node_id: idx for idx, node_id in enumerate(node_order)}
        x0 = self.node_proj(self._build_node_features(scene_graph.objects, node_order))  # [N, D]

        agg = torch.zeros_like(x0)
        deg = torch.zeros((n_nodes, 1), dtype=x0.dtype, device=x0.device)

        for rel in scene_graph.graph.relations:
            if rel.subject_id not in id_to_index or rel.object_id not in id_to_index:
                continue
            src = id_to_index[rel.subject_id]
            dst = id_to_index[rel.object_id]
            rel_idx = int(rel.relation_id) % self.relation_vocab_size
            rel_emb = self.rel_embedding.weight[rel_idx]  # [D]

            msg_input = torch.cat([x0[src], rel_emb], dim=-1)
            msg = self.msg_mlp(msg_input)
            agg[dst] = agg[dst] + msg
            deg[dst] = deg[dst] + 1.0

        agg = agg / deg.clamp_min(1.0)
        updated = self.update(torch.cat([x0, agg], dim=-1))
        return self.norm(x0 + updated)
