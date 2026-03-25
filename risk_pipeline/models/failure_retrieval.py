from __future__ import annotations

import torch
from torch import nn

from risk_pipeline.data.failure_memory import FailurePrototypeMemory


class FailureRetriever(nn.Module):
    """Retrieve failure context vectors from prototype memory.

    Input:
        z_i: torch.Tensor [N, D]
        memory: FailurePrototypeMemory with tensorized prototypes [M, D]

    Output:
        r_i_retr: torch.Tensor [N, D]

    Method:
        cosine similarity + top-k weighted aggregation over prototypes.
    """

    def __init__(self, top_k: int = 3, temperature: float = 0.2) -> None:
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, memory: FailurePrototypeMemory) -> torch.Tensor:
        if z_i.ndim != 2:
            raise ValueError(f"z_i must be [N, D], got {tuple(z_i.shape)}")

        _, dim = z_i.shape
        prototypes = memory.as_tensor(device=z_i.device)
        if len(memory) == 0:
            return torch.zeros_like(z_i)
        if prototypes.shape[1] != dim:
            raise ValueError(f"Prototype dim mismatch: z_i has D={dim}, memory has D={prototypes.shape[1]}")

        z_norm = torch.nn.functional.normalize(z_i, dim=-1)
        p_norm = torch.nn.functional.normalize(prototypes, dim=-1)
        sim = z_norm @ p_norm.T  # [N, M]

        k = min(self.top_k, prototypes.shape[0])
        top_values, top_indices = torch.topk(sim, k=k, dim=-1)
        weights = torch.softmax(top_values / self.temperature, dim=-1)  # [N, K]

        gathered = p_norm[top_indices]  # [N, K, D]
        r_i_retr = (weights.unsqueeze(-1) * gathered).sum(dim=1)  # [N, D]
        return r_i_retr
