"""Step-3: Failure-conditioned residual gated modulation."""

from __future__ import annotations

import torch


class FailureConditionedModulation(torch.nn.Module):
    """Residual Gated Modulation.

    Formula:
      g_i      = sigmoid(MLP([P_f(v_i) || r_i^retr || r_i^rel]))
      Delta_i  = MLP([z_i || r_i^retr || r_i^rel])
      z_i'     = z_i + g_i ⊙ Delta_i

    Input shapes:
      z_i       [B, N, D]
      r_i_retr  [B, N, D]
      r_i_rel   [B, N, D]
      p_f       [B, N, 1]

    Output:
      z_i_prime [B, N, D]
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, risk_preserve: bool = True) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.risk_preserve = risk_preserve

        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim * 2 + 1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, dim),
        )
        self.delta_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim * 3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        z_i: torch.Tensor,
        r_i_retr: torch.Tensor,
        r_i_rel: torch.Tensor,
        p_f: torch.Tensor,
    ) -> torch.Tensor:
        # gate_in: [B, N, 2D+1]
        gate_in = torch.cat([p_f, r_i_retr, r_i_rel], dim=-1)
        # g_raw: [B, N, D]
        g_raw = torch.sigmoid(self.gate_mlp(gate_in))

        # Optional risk-preserving scaling:
        # p_f near 0 -> g_i near 0 to preserve z_i; high p_f -> stronger modulation.
        if self.risk_preserve:
            # p_scale: [B, N, D]
            p_scale = p_f.expand_as(g_raw)
            # g_i: [B, N, D]
            g_i = g_raw * p_scale
        else:
            # g_i: [B, N, D]
            g_i = g_raw

        # delta_in: [B, N, 3D]
        delta_in = torch.cat([z_i, r_i_retr, r_i_rel], dim=-1)
        # delta_i: [B, N, D]
        delta_i = self.delta_mlp(delta_in)

        # z_i_prime: [B, N, D]
        z_i_prime = z_i + g_i * delta_i
        return z_i_prime
