from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from risk_pipeline.core.runner import PipelineConfig, RiskPipelineRunner
from risk_pipeline.data.failure_memory import FailurePrototypeMemory
from risk_pipeline.data.sgfront_loader import SGFrontLoader


def _write_scene_files(tmp_path: Path) -> tuple[Path, Path]:
    relationships = {
        "objects": {"1": "chair", "2": "table"},
        "relationships": [[1, 2, 0, "left of"]],
    }
    obj_boxes = {
        "1": {"param7": [0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0]},
        "2": {"param7": [1.0, 0.0, 0.0, 1.2, 0.7, 0.8, 0.1]},
    }

    rel_path = tmp_path / "relationships.json"
    box_path = tmp_path / "obj_boxes.json"
    rel_path.write_text(json.dumps(relationships), encoding="utf-8")
    box_path.write_text(json.dumps(obj_boxes), encoding="utf-8")
    return rel_path, box_path


def _memory(dim: int) -> FailurePrototypeMemory:
    payload = {
        "prototypes": [
            {"prototype_id": "p0", "embedding": [0.2] * dim, "metadata": {"type": "collision"}},
            {"prototype_id": "p1", "embedding": [0.4] * dim, "metadata": {"type": "stuck"}},
        ]
    }
    return FailurePrototypeMemory.from_dict(payload)


def test_runner_end_to_end(tmp_path: Path) -> None:
    rel_path, box_path = _write_scene_files(tmp_path)
    scene = SGFrontLoader().load(rel_path, box_path)

    cfg = PipelineConfig(hidden_dim=8, retrieval_top_k=2, voxel_size=1.0)
    runner = RiskPipelineRunner(config=cfg)
    outputs = runner.run(scene_graph=scene, memory=_memory(dim=8), T_t=np.eye(4))

    assert outputs["z_i"].shape == (2, 8)
    assert outputs["r_i_rel"].shape == (2, 8)
    assert outputs["r_i_retr"].shape == (2, 8)
    assert outputs["z_i_prime"].shape == (2, 8)
    assert len(outputs["integrated_voxels"]) == 2
    assert outputs["map_state"]["feature_dim"] == 8
