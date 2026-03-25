from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from risk_pipeline.core.runner import PipelineConfig, RiskPipelineRunner
from risk_pipeline.data.failure_memory import FailurePrototypeMemory
from risk_pipeline.data.sgfront_loader import SGFrontLoader


def _serialize_map_state(map_state: dict) -> dict:
    serializable = {
        "voxel_size": float(map_state["voxel_size"]),
        "feature_dim": int(map_state["feature_dim"]),
        "voxels": {},
    }
    for key, value in map_state["voxels"].items():
        str_key = f"{key[0]},{key[1]},{key[2]}"
        serializable["voxels"][str_key] = {
            "count": int(value["count"]),
            "mean_feature": np.asarray(value["mean_feature"]).tolist(),
        }
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone risk pipeline MVP end-to-end")
    parser.add_argument("--relationships-json", required=True)
    parser.add_argument("--obj-boxes-json", required=True)
    parser.add_argument("--scan-id")
    parser.add_argument("--memory-json", required=True)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("risk_pipeline_output.json"))
    args = parser.parse_args()

    loader = SGFrontLoader()
    scene = loader.load(args.relationships_json, args.obj_boxes_json, scan_id=args.scan_id)
    if loader.last_scene_stats:
        stats = loader.last_scene_stats
        print(f"[run_pipeline] selected scan id: {stats['selected_scan_id']}")
        print(f"[run_pipeline] num objects: {stats['num_objects']}")
        print(f"[run_pipeline] num relationships: {stats['num_relationships']}")
        print(f"[run_pipeline] num valid boxes: {stats['num_valid_boxes']}")
    memory = FailurePrototypeMemory.from_json(args.memory_json)

    cfg = PipelineConfig(
        hidden_dim=args.hidden_dim,
        retrieval_top_k=args.top_k,
        voxel_size=args.voxel_size,
    )
    runner = RiskPipelineRunner(config=cfg)
    outputs = runner.run(scene_graph=scene, memory=memory)
    selected_scan_id = loader.last_scene_stats["selected_scan_id"] if loader.last_scene_stats else args.scan_id

    result = {
        "run": {
            "selected_scan_id": selected_scan_id,
            "config": {
                "top_k": int(args.top_k),
                "hidden_dim": int(args.hidden_dim),
                "voxel_size": float(args.voxel_size),
            },
        },
        "summary": {
            "num_objects": len(scene.objects),
            "num_relations": len(scene.graph.relations),
            "num_voxels": len(outputs["map_state"]["voxels"]),
        },
        "retrieval": {
            "top_k": int(outputs["retrieval_stats"]["top_k"]),
            "selected_prototype_indices": outputs["retrieval_stats"]["selected_prototype_indices"],
            "similarity_scores_summary": outputs["retrieval_stats"]["similarity_summary"],
        },
        "modulation": {
            "gate_statistics": outputs["modulation_stats"]["gate"],
            "delta_statistics": {
                "mean_l2": outputs["modulation_stats"]["delta_mean_l2"],
                "l2_summary": outputs["modulation_stats"]["delta_l2"],
                "cosine_mean": outputs["modulation_stats"]["cosine_mean"],
                "cosine_summary": outputs["modulation_stats"]["cosine_z_z_prime"],
            },
        },
        "artifacts": {
            "node_order": outputs["node_order"],
            "integrated_voxels": outputs["integrated_voxels"],
            "map_state": _serialize_map_state(outputs["map_state"]),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        "[run_pipeline] objects="
        f"{result['summary']['num_objects']} relations={result['summary']['num_relations']}"
    )
    print(f"[run_pipeline] voxels={result['summary']['num_voxels']}")
    print(f"[run_pipeline] wrote {args.output}")


if __name__ == "__main__":
    main()
