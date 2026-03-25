"""Triplet-based context risk encoder (Step-2).

Encodes SG-FRONT relation triplets (subject, predicate, object) into contextual
node embeddings r_i^rel via multi-step message passing.
"""

from __future__ import annotations

from typing import Tuple

import torch


class TripletMessageLayer(torch.nn.Module):
    """One message-passing hop using (src node, predicate embedding) messages.

    Shapes:
      x         [B, N, D]
      edge_idx  [B, E, 2]  (src, dst)
      edge_type [B, E]
      edge_emb  [R, De]

    Message:
      m_ij = MLP([x_src || e_pred])

    Aggregation:
      mean over incoming edges for each destination node.
    """

    def __init__(self, node_dim: int, edge_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.msg_mlp = torch.nn.Sequential(
            torch.nn.Linear(node_dim + edge_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, node_dim),
        )
        self.upd_mlp = torch.nn.Sequential(
            torch.nn.Linear(node_dim * 2, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, node_dim),
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, edge_type: torch.Tensor, edge_table: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, dim = x.shape
        n_edges = edge_idx.shape[1]

        agg = torch.zeros_like(x)                              # [B,N,D]
        deg = torch.zeros((bsz, n_nodes, 1), device=x.device)  # [B,N,1]

        one = torch.ones((n_edges, 1), device=x.device)
        for b in range(bsz):
            src = edge_idx[b, :, 0].long()  # [E]
            dst = edge_idx[b, :, 1].long()  # [E]
            pred = edge_table[edge_type[b].long()]            # [E,De]
            msg_in = torch.cat([x[b, src], pred], dim=-1)     # [E,D+De]
            msg = self.msg_mlp(msg_in)                         # [E,D]

            agg[b].index_add_(0, dst, msg)
            deg[b].index_add_(0, dst, one)

        agg = agg / deg.clamp_min(1.0)
        upd = self.upd_mlp(torch.cat([x, agg], dim=-1))
        out = self.norm(x + self.dropout(upd))
        return out


class TripletGCNEncoder(torch.nn.Module):
    """Multi-hop triplet context encoder.

    Inputs:
      node_feat [B, N, D] : z_i or current risk state
      edge_idx  [B, E, 2]
      edge_type [B, E]

    Outputs:
      r_i_rel   [B, N, D] : relation/context-aware risk embedding
      optional updated node state [B, N, D]
    """

    def __init__(
        self,
        dim: int,
        num_edge_types: int,
        edge_dim: int | None = None,
        message_passing_steps: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.edge_dim = edge_dim or dim
        self.message_passing_steps = message_passing_steps

        self.edge_embedding = torch.nn.Embedding(num_edge_types, self.edge_dim)
        self.layers = torch.nn.ModuleList(
            [TripletMessageLayer(node_dim=dim, edge_dim=self.edge_dim, dropout=dropout) for _ in range(message_passing_steps)]
        )
        self.rel_head = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_idx: torch.Tensor,
        edge_type: torch.Tensor,
        return_node_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x0 = node_feat
        x = node_feat

        for layer in self.layers:
            x = layer(x, edge_idx, edge_type, self.edge_embedding.weight)

        r_i_rel = self.rel_head(torch.cat([x0, x], dim=-1))
        if return_node_state:
            return r_i_rel, x
        return r_i_rel
