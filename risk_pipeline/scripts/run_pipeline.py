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

    result = {
        "num_objects": len(scene.objects),
        "num_relations": len(scene.graph.relations),
        "num_voxels": len(outputs["map_state"]["voxels"]),
        "node_order": outputs["node_order"],
        "integrated_voxels": outputs["integrated_voxels"],
        "map_state": _serialize_map_state(outputs["map_state"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[run_pipeline] objects={result['num_objects']} relations={result['num_relations']}")
    print(f"[run_pipeline] voxels={result['num_voxels']}")
    print(f"[run_pipeline] wrote {args.output}")


if __name__ == "__main__":
    main()
