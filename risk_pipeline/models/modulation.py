from __future__ import annotations

import torch
from torch import nn


class GatedResidualModulation(nn.Module):
    """Gated residual modulation for node representations.

    Input:
        z_i: torch.Tensor [N, D]
        r_i_rel: torch.Tensor [N, D]
        r_i_retr: torch.Tensor [N, D]

    Output:
        z_i_prime: torch.Tensor [N, D]

    Form:
        gate = sigmoid(MLP([r_i_rel || r_i_retr]))
        delta = MLP([z_i || r_i_rel || r_i_retr])
        z_i' = z_i + gate * delta

    Stability:
        The final gate layer bias is initialized to a negative value so
        early-phase gate outputs start small.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, init_gate_bias: float = -4.0) -> None:
        super().__init__()
        hidden = hidden_dim or dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

        nn.init.constant_(self.gate_mlp[-1].bias, init_gate_bias)

    @staticmethod
    def _summary_stats(values: torch.Tensor) -> dict[str, float]:
        if values.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }

    def modulate_with_stats(
        self, z_i: torch.Tensor, r_i_rel: torch.Tensor, r_i_retr: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
        if z_i.shape != r_i_rel.shape or z_i.shape != r_i_retr.shape:
            raise ValueError(
                f"Input shape mismatch: z_i={tuple(z_i.shape)}, r_i_rel={tuple(r_i_rel.shape)}, r_i_retr={tuple(r_i_retr.shape)}"
            )

        gate_input = torch.cat([r_i_rel, r_i_retr], dim=-1)
        delta_input = torch.cat([z_i, r_i_rel, r_i_retr], dim=-1)

        gate = torch.sigmoid(self.gate_mlp(gate_input))
        delta = self.delta_mlp(delta_input)
        modulation_delta = gate * delta
        z_i_prime = z_i + modulation_delta

        cosine = torch.nn.functional.cosine_similarity(z_i, z_i_prime, dim=-1)
        stats = {
            "gate": self._summary_stats(gate),
            "delta_l2": self._summary_stats(torch.linalg.norm(modulation_delta, dim=-1)),
            "cosine_z_z_prime": self._summary_stats(cosine),
        }
        return z_i_prime, stats

    def forward(self, z_i: torch.Tensor, r_i_rel: torch.Tensor, r_i_retr: torch.Tensor) -> torch.Tensor:
        z_i_prime, _ = self.modulate_with_stats(z_i, r_i_rel, r_i_retr)
        return z_i_prime
