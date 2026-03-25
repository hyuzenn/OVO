"""Evaluation utilities for risk-aware semantic mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(slots=True)
class RiskMetrics:
    chamfer_distance: float | None
    risk_precision: float | None


def _as_points(x: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected Nx3 points, got shape={arr.shape}")
    return arr


def chamfer_distance(points_a: Sequence[Sequence[float]], points_b: Sequence[Sequence[float]]) -> float:
    """Symmetric Chamfer distance between two point sets (Nx3 and Mx3)."""

    a = _as_points(points_a)
    b = _as_points(points_b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    # pairwise squared distances: [N, M]
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)

    a_to_b = np.min(d2, axis=1)
    b_to_a = np.min(d2, axis=0)
    return float(np.mean(a_to_b) + np.mean(b_to_a))


def risk_precision(
    p_f: Sequence[float],
    inconsistency_mask: Sequence[bool],
    risk_threshold: float = 0.7,
) -> float:
    """Precision over predicted high-risk regions.

    Precision = TP / PredictedPositive
      - PredictedPositive: p_f > risk_threshold
      - TP: predicted positive and actual inconsistency=True
    """

    p = np.asarray(p_f, dtype=np.float32)
    y = np.asarray(inconsistency_mask, dtype=bool)
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"size mismatch: len(p_f)={p.shape[0]} len(mask)={y.shape[0]}")

    pred = p > risk_threshold
    pred_count = int(pred.sum())
    if pred_count == 0:
        return 0.0
    tp = int(np.logical_and(pred, y).sum())
    return float(tp / pred_count)


def evaluate_metrics(
    p_f: Sequence[float],
    inconsistency_mask: Sequence[bool] | None = None,
    robot_failure_points: Sequence[Sequence[float]] | None = None,
    predicted_risk_points: Sequence[Sequence[float]] | None = None,
    risk_threshold: float = 0.7,
) -> RiskMetrics:
    chamfer = None
    precision = None

    if robot_failure_points is not None and predicted_risk_points is not None:
        chamfer = chamfer_distance(robot_failure_points, predicted_risk_points)

    if inconsistency_mask is not None:
        precision = risk_precision(p_f=p_f, inconsistency_mask=inconsistency_mask, risk_threshold=risk_threshold)

    return RiskMetrics(chamfer_distance=chamfer, risk_precision=precision)
