"""3D visualization helpers for risk-aware scene graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _color_from_risk(risk: float) -> tuple[float, float, float]:
    # blue (safe) -> red (risky)
    r = float(np.clip(risk, 0.0, 1.0))
    return (r, 0.0, 1.0 - r)


def visualize_risk_scene(
    centers: np.ndarray,
    p_f: np.ndarray,
    delta_norm: np.ndarray | None = None,
    out_path: str | Path = "risk_scene.png",
    title: str = "Risk-aware Scene",
) -> str:
    """Render object centers with risk coloring and optional delta intensity.

    Args:
      centers: [N,3]
      p_f: [N]
      delta_norm: [N] optional embedding delta norm
    """

    centers = np.asarray(centers, dtype=np.float32)
    p_f = np.asarray(p_f, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"centers must be [N,3], got {centers.shape}")
    if p_f.shape[0] != centers.shape[0]:
        raise ValueError(f"p_f size mismatch: {p_f.shape[0]} vs {centers.shape[0]}")

    colors = np.array([_color_from_risk(v) for v in p_f], dtype=np.float32)
    if delta_norm is None:
        sizes = np.full((centers.shape[0],), 120.0, dtype=np.float32)
    else:
        d = np.asarray(delta_norm, dtype=np.float32)
        d = d / (d.max() + 1e-6)
        sizes = 80.0 + 220.0 * d

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=colors, s=sizes, alpha=0.85)

    for i, (x, y, z) in enumerate(centers):
        ax.text(x, y, z, f"{i}:{p_f[i]:.2f}", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # color bar (blue->red)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm)
    sm.set_array(np.clip(p_f, 0.0, 1.0))
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Risk P_f")

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path
