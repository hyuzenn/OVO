"""Run RiskAwareMapper end-to-end on SG-FRONT JSON files.

Usage example:
  python main.py \
      --relationships path/to/relationships_0.json \
      --obj-boxes path/to/obj_boxes_0.json \
      --lambda1 0.6 --lambda2 0.4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from ovo.entities.risk_evaluator import evaluate_metrics
from ovo.entities.risk_aware_mapper import RiskAwareMapper, RiskAwareMapperConfig
from ovo.entities.risk_visualizer import visualize_risk_scene


def _find_first(pattern: str, root: Path) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Risk-aware SG-FRONT mapper")
    p.add_argument("--relationships", type=Path, default=None, help="relationships_*.json path")
    p.add_argument("--obj-boxes", type=Path, default=None, help="obj_boxes_*.json path")
    p.add_argument("--lambda1", type=float, default=0.5, help="weight for P_obj")
    p.add_argument("--lambda2", type=float, default=0.5, help="weight for P_ctx")
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--geom-dim", type=int, default=8)
    p.add_argument("--message-passing-steps", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--failure-memory-json", type=Path, default=None, help="persistent failure memory JSON")
    p.add_argument("--risk-threshold", type=float, default=0.7)
    p.add_argument("--inconsistency-json", type=Path, default=None, help="JSON list[bool] or list[int] mask")
    p.add_argument("--failure-points-json", type=Path, default=None, help="JSON Nx3 robot failure points")
    p.add_argument("--report-json", type=Path, default=Path("outputs/risk_report.json"))
    p.add_argument("--report-csv", type=Path, default=Path("outputs/risk_scores.csv"))
    p.add_argument("--vis-png", type=Path, default=Path("outputs/risk_scene.png"))
    return p.parse_args()


def _load_optional_json(path: Path | None):
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    root = Path("/workspace/OVO")

    relationships = args.relationships or _find_first("relationships_*.json", root)
    obj_boxes = args.obj_boxes or _find_first("obj_boxes_*.json", root)

    if relationships is None or obj_boxes is None:
        print("[ERROR] SG-FRONT JSON files not found in /workspace/OVO.")
        print("        Provide --relationships and --obj-boxes explicitly.")
        return 1

    cfg = RiskAwareMapperConfig(
        embedding_dim=args.embedding_dim,
        geom_dim=args.geom_dim,
        message_passing_steps=args.message_passing_steps,
        lambda_1=args.lambda1,
        lambda_2=args.lambda2,
        device=args.device,
        failure_memory_json=str(args.failure_memory_json) if args.failure_memory_json else None,
    )
    mapper = RiskAwareMapper(cfg)
    result = mapper.run(relationships_json=relationships, obj_boxes_json=obj_boxes)

    outputs = {k: v for k, v in result.items() if isinstance(v, torch.Tensor)}
    scene = mapper.export_scene_arrays(result["graph"], result["batch"], outputs)
    p_f = scene["p_f"]

    print("=== RiskAwareMapper Summary ===")
    print(f"relationships: {relationships}")
    print(f"obj_boxes:     {obj_boxes}")
    print(f"lambda1={cfg.lambda_1:.3f}, lambda2={cfg.lambda_2:.3f}")
    print(mapper.summarize_outputs(outputs, topk=5))

    delta = mapper.embedding_delta_norm(result["batch"].z_i, outputs["z_i_prime"])[0]  # [N]
    print("\n=== Embedding Delta ||z_i' - z_i|| ===")
    for i, v in enumerate(delta.tolist()):
        print(f"node={i:03d} delta_norm={v:.4f}")

    print("\n=== Tensor Shapes ===")
    for k in ["r_i_retr", "p_i_retr", "r_i_rel", "p_obj", "p_ctx", "p_f", "z_i_prime", "map_v"]:
        print(f"{k:>10s}: {tuple(outputs[k].shape)}")

    inconsistency_raw = _load_optional_json(args.inconsistency_json)
    failure_points = _load_optional_json(args.failure_points_json)
    inconsistency_mask = None
    if inconsistency_raw is not None:
        # Accept either list[bool] mask or list[int] indices.
        if len(inconsistency_raw) > 0 and isinstance(inconsistency_raw[0], bool):
            inconsistency_mask = inconsistency_raw
        else:
            inconsistency_mask = [False] * len(p_f)
            for idx in inconsistency_raw:
                if 0 <= int(idx) < len(inconsistency_mask):
                    inconsistency_mask[int(idx)] = True

    predicted_risk_points = scene["centers"][p_f > args.risk_threshold].tolist()
    metrics = evaluate_metrics(
        p_f=p_f.tolist(),
        inconsistency_mask=inconsistency_mask,
        robot_failure_points=failure_points,
        predicted_risk_points=predicted_risk_points if len(predicted_risk_points) > 0 else None,
        risk_threshold=args.risk_threshold,
    )

    vis_path = visualize_risk_scene(
        centers=scene["centers"],
        p_f=scene["p_f"],
        delta_norm=scene["delta_norm"],
        out_path=args.vis_png,
        title=f"Risk Scene (lambda1={cfg.lambda_1:.2f}, lambda2={cfg.lambda_2:.2f})",
    )

    report = {
        "relationships": str(relationships),
        "obj_boxes": str(obj_boxes),
        "lambda1": cfg.lambda_1,
        "lambda2": cfg.lambda_2,
        "risk_threshold": args.risk_threshold,
        "metrics": {
            "chamfer_distance": metrics.chamfer_distance,
            "risk_precision": metrics.risk_precision,
        },
        "visualization": str(vis_path),
        "nodes": [
            {
                "node_index": i,
                "p_f": float(scene["p_f"][i]),
                "delta_norm": float(scene["delta_norm"][i]),
                "center": scene["centers"][i].tolist(),
            }
            for i in range(len(scene["p_f"]))
        ],
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["node_index", "p_f", "delta_norm", "x", "y", "z"])
        w.writeheader()
        for row in report["nodes"]:
            x, y, z = row["center"]
            w.writerow(
                {
                    "node_index": row["node_index"],
                    "p_f": row["p_f"],
                    "delta_norm": row["delta_norm"],
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

    print("\n=== Metrics ===")
    print(f"chamfer_distance={metrics.chamfer_distance}")
    print(f"risk_precision={metrics.risk_precision}")
    print(f"saved report json: {args.report_json}")
    print(f"saved report csv:  {args.report_csv}")
    print(f"saved figure:      {vis_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
