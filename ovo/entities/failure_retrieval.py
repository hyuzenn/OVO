"""Failure episode memory and retrieval for risk-aware mapping.

Step-1 module of the architecture:
- Query input: semantic embedding z_i + bbox geometry features.
- K-NN retrieval: cosine-similarity search over failure prototypes.
- Output: retrieved risk cue r_i^retr and prior risk p_i^retr.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import torch


def _unit(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)


@dataclass(slots=True)
class FailurePrototype:
    """One stored failure episode prototype."""

    episode_id: str
    scenario: str
    semantic_proto: torch.Tensor   # [D]
    geom_proto: torch.Tensor       # [G]
    risk_embedding: torch.Tensor   # [D]
    prior_risk: float              # scalar in [0,1]
    failure_type: str = "other"
    tags: List[str] = field(default_factory=list)

    @property
    def key(self) -> torch.Tensor:
        return torch.cat([self.semantic_proto, self.geom_proto], dim=-1)  # [D+G]


class FailureMemory:
    """Container and initializer for failure prototypes."""

    def __init__(self, prototypes: Sequence[FailurePrototype] | None = None) -> None:
        self.prototypes: List[FailurePrototype] = list(prototypes) if prototypes is not None else []

    def add(self, proto: FailurePrototype) -> None:
        self.prototypes.append(proto)

    def extend(self, protos: Iterable[FailurePrototype]) -> None:
        self.prototypes.extend(protos)

    def __len__(self) -> int:
        return len(self.prototypes)

    @staticmethod
    def _fit_dim(vec: torch.Tensor, dim: int) -> torch.Tensor:
        if vec.numel() == dim:
            return vec
        if vec.numel() > dim:
            return vec[:dim]
        pad = torch.zeros(dim - vec.numel(), dtype=vec.dtype)
        return torch.cat([vec, pad], dim=0)

    def as_tensors(self, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stacked tensors for vectorized retrieval.

        Returns:
          keys       [M, D+G]
          risk_emb   [M, D]
          prior_risk [M, 1]
        """

        if len(self.prototypes) == 0:
            return (
                torch.zeros((0, 1), dtype=torch.float32, device=device),
                torch.zeros((0, 1), dtype=torch.float32, device=device),
                torch.zeros((0, 1), dtype=torch.float32, device=device),
            )

        keys = torch.stack([p.key for p in self.prototypes], dim=0).to(device)
        risk_emb = torch.stack([p.risk_embedding for p in self.prototypes], dim=0).to(device)
        prior = torch.tensor([p.prior_risk for p in self.prototypes], dtype=torch.float32, device=device).unsqueeze(-1)
        return keys, risk_emb, prior

    def to_json(self, path: str | Path) -> None:
        payload = {"prototypes": []}
        for p in self.prototypes:
            payload["prototypes"].append(
                {
                    "episode_id": p.episode_id,
                    "scenario": p.scenario,
                    "semantic_proto": p.semantic_proto.detach().cpu().tolist(),
                    "geom_proto": p.geom_proto.detach().cpu().tolist(),
                    "risk_embedding": p.risk_embedding.detach().cpu().tolist(),
                    "prior_risk": float(p.prior_risk),
                    "failure_type": p.failure_type,
                    "tags": p.tags,
                }
            )
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def from_json(
        path: str | Path,
        dim: int,
        geom_dim: int,
        device: str | torch.device = "cpu",
    ) -> "FailureMemory":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)

        protos: List[FailurePrototype] = []
        for item in payload.get("prototypes", []):
            s = torch.tensor(item.get("semantic_proto", []), dtype=torch.float32)
            g = torch.tensor(item.get("geom_proto", []), dtype=torch.float32)
            r = torch.tensor(item.get("risk_embedding", []), dtype=torch.float32)
            s = _unit(FailureMemory._fit_dim(s, dim))
            g = _unit(FailureMemory._fit_dim(g, geom_dim))
            r = _unit(FailureMemory._fit_dim(r, dim))
            protos.append(
                FailurePrototype(
                    episode_id=str(item.get("episode_id", f"episode_{len(protos)}")),
                    scenario=str(item.get("scenario", "unknown scenario")),
                    semantic_proto=s.to(device),
                    geom_proto=g.to(device),
                    risk_embedding=r.to(device),
                    prior_risk=float(item.get("prior_risk", 0.5)),
                    failure_type=str(item.get("failure_type", "other")),
                    tags=list(item.get("tags", [])),
                )
            )
        return FailureMemory(prototypes=protos)

    @staticmethod
    def build_dummy_memory(dim: int, geom_dim: int, device: str | torch.device = "cpu") -> "FailureMemory":
        """Initialize with curated dummy failure episodes.

        Examples include:
        - narrow doorway passage risk
        - unstable table-top placement risk
        - cluttered corridor collision risk
        - slippery floor near sink
        """

        def vec(seed: int, n: int) -> torch.Tensor:
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            return _unit(torch.randn(n, generator=g))

        protos = [
            FailurePrototype(
                episode_id="ep_narrow_door_01",
                scenario="narrow doorway with chair arms",
                semantic_proto=vec(11, dim),
                geom_proto=vec(101, geom_dim),
                risk_embedding=vec(201, dim),
                prior_risk=0.82,
                failure_type="collision",
                tags=["door", "narrow", "passage"],
            ),
            FailurePrototype(
                episode_id="ep_unstable_table_01",
                scenario="object near unstable table edge",
                semantic_proto=vec(12, dim),
                geom_proto=vec(102, geom_dim),
                risk_embedding=vec(202, dim),
                prior_risk=0.74,
                failure_type="fall",
                tags=["table", "unstable", "edge"],
            ),
            FailurePrototype(
                episode_id="ep_cluttered_corridor_01",
                scenario="cluttered corridor turning point",
                semantic_proto=vec(13, dim),
                geom_proto=vec(103, geom_dim),
                risk_embedding=vec(203, dim),
                prior_risk=0.88,
                failure_type="collision",
                tags=["corridor", "clutter", "turn"],
            ),
            FailurePrototype(
                episode_id="ep_sink_slip_01",
                scenario="wet floor near sink cabinet",
                semantic_proto=vec(14, dim),
                geom_proto=vec(104, geom_dim),
                risk_embedding=vec(204, dim),
                prior_risk=0.67,
                failure_type="stuck",
                tags=["kitchen", "wet", "slippery"],
            ),
        ]
        mem = FailureMemory(prototypes=protos)
        for p in mem.prototypes:
            p.semantic_proto = p.semantic_proto.to(device)
            p.geom_proto = p.geom_proto.to(device)
            p.risk_embedding = p.risk_embedding.to(device)
        return mem


class FailureEpisodeRetrieval(torch.nn.Module):
    """Memory-based KNN retriever used by risk pipeline.

    Input:
      z_i       [B, N, D]
      bbox_geom [B, N, G]
    Output:
      r_i_retr  [B, N, D]
      p_i_retr  [B, N, 1]

    Notes:
      - Query vector q := concat(z_i, bbox_geom) in R^(D+G)
      - Similarity: cosine(q, prototype_key)
      - Top-K weighted aggregation via softmax(sim / temperature)
    """

    def __init__(
        self,
        dim: int,
        geom_dim: int,
        k: int = 3,
        temperature: float = 0.07,
        memory: FailureMemory | None = None,
        memory_json: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.geom_dim = geom_dim
        self.k = k
        self.temperature = temperature
        self.device = device

        if memory is None and memory_json is not None:
            memory = FailureMemory.from_json(path=memory_json, dim=dim, geom_dim=geom_dim, device=device)
        if memory is None:
            memory = FailureMemory.build_dummy_memory(dim=dim, geom_dim=geom_dim, device=device)
        self.memory = memory

    def forward(self, z_i: torch.Tensor, bbox_geom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_nodes, d = z_i.shape
        if d != self.dim:
            raise ValueError(f"z_i dim mismatch: expected {self.dim}, got {d}")
        if bbox_geom.shape[-1] != self.geom_dim:
            raise ValueError(f"bbox geom dim mismatch: expected {self.geom_dim}, got {bbox_geom.shape[-1]}")

        keys, risk_emb, prior = self.memory.as_tensors(device=z_i.device)
        m = keys.shape[0]
        if m == 0:
            return torch.zeros_like(z_i), torch.zeros((bsz, n_nodes, 1), dtype=z_i.dtype, device=z_i.device)

        q = torch.cat([z_i, bbox_geom], dim=-1)            # [B,N,D+G]
        qn = _unit(q.reshape(-1, self.dim + self.geom_dim))  # [B*N,D+G]
        kn = _unit(keys)                                      # [M,D+G]

        sim = qn @ kn.T                                       # [B*N,M]
        k = min(self.k, m)
        topv, topi = torch.topk(sim, k=k, dim=-1)            # [B*N,K]

        w = torch.softmax(topv / self.temperature, dim=-1)   # [B*N,K]
        picked_risk = risk_emb[topi]                         # [B*N,K,D]
        picked_prior = prior[topi]                           # [B*N,K,1]

        r = (w.unsqueeze(-1) * picked_risk).sum(dim=1)       # [B*N,D]
        p = (w.unsqueeze(-1) * picked_prior).sum(dim=1)      # [B*N,1]

        r_i_retr = r.reshape(bsz, n_nodes, self.dim)
        p_i_retr = p.reshape(bsz, n_nodes, 1)
        return r_i_retr, p_i_retr
