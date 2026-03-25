from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


@dataclass(slots=True)
class FailurePrototype:
    """Single failure prototype.

    Fields:
        prototype_id: unique id
        embedding: tensor-like vector of shape [D]
        metadata: optional descriptive fields
    """

    prototype_id: str
    embedding: torch.Tensor
    metadata: dict[str, Any]


class FailurePrototypeMemory:
    """Prototype memory for retrieval modules.

    Tensor contract:
        as_tensor() -> torch.Tensor of shape [M, D]
        - M: number of prototypes
        - D: embedding dimension (fixed at memory creation)
    """

    def __init__(self, prototypes: Sequence[FailurePrototype] | None = None) -> None:
        self._prototypes = list(prototypes or [])

    @property
    def prototypes(self) -> list[FailurePrototype]:
        return self._prototypes

    def __len__(self) -> int:
        return len(self._prototypes)

    def add(self, prototype: FailurePrototype) -> None:
        self._prototypes.append(prototype)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FailurePrototypeMemory":
        items = payload.get("prototypes", [])
        prototypes: list[FailurePrototype] = []
        for idx, item in enumerate(items):
            pid = str(item.get("prototype_id", f"proto_{idx}"))
            emb = torch.tensor(item.get("embedding", []), dtype=torch.float32)
            if emb.ndim != 1:
                raise ValueError(f"Prototype embedding must be 1D, got shape {tuple(emb.shape)}")
            meta = dict(item.get("metadata", {}))
            prototypes.append(FailurePrototype(prototype_id=pid, embedding=emb, metadata=meta))

        memory = cls(prototypes)
        memory.validate()
        return memory

    @classmethod
    def from_json(cls, path: str | Path) -> "FailurePrototypeMemory":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def extend(self, prototypes: Iterable[FailurePrototype]) -> None:
        self._prototypes.extend(prototypes)
        self.validate()

    def validate(self) -> None:
        if not self._prototypes:
            return
        dim = self._prototypes[0].embedding.numel()
        for proto in self._prototypes:
            if proto.embedding.ndim != 1:
                raise ValueError("All prototype embeddings must be 1D vectors.")
            if proto.embedding.numel() != dim:
                raise ValueError(
                    f"Prototype embedding dim mismatch: expected {dim}, got {proto.embedding.numel()}"
                )

    def as_tensor(self, device: torch.device | str | None = None) -> torch.Tensor:
        if not self._prototypes:
            return torch.zeros((0, 0), dtype=torch.float32, device=device)
        tensor = torch.stack([p.embedding for p in self._prototypes], dim=0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
