from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from risk_pipeline.core.runner import PipelineConfig, RiskPipelineRunner
from risk_pipeline.data.failure_memory import FailurePrototypeMemory
from risk_pipeline.data.sgfront_loader import SGFrontLoader
from risk_pipeline.models.failure_retrieval import FailureRetriever


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


def _write_multi_scan_files(tmp_path: Path) -> tuple[Path, Path]:
    relationships = {
        "MasterBedroom-111": {
            "objects": {"1": "bed"},
            "relationships": [[1, 1, 0, "intersect"]],
        },
        "MasterBedroom-33296": {
            "objects": {"1": "chair", "2": "table"},
            "relationships": [[1, 2, 0, "left of"]],
        },
    }
    obj_boxes = {
        "MasterBedroom-111": {
            "1": {"param7": [2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]},
            "scene_center": [0.0, 0.0, 0.0],
        },
        "MasterBedroom-33296": {
            "1": {"param7": [0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0]},
            "2": {"param7": [1.0, 0.0, 0.0, 1.2, 0.7, 0.8, 0.1]},
            "scene_center": [0.0, 0.0, 0.0],
        },
    }
    rel_path = tmp_path / "relationships_multi.json"
    box_path = tmp_path / "obj_boxes_multi.json"
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


def test_runner_supports_scan_id_selection(tmp_path: Path) -> None:
    rel_path, box_path = _write_multi_scan_files(tmp_path)
    loader = SGFrontLoader()
    scene = loader.load(rel_path, box_path, scan_id="MasterBedroom-33296")

    cfg = PipelineConfig(hidden_dim=8, retrieval_top_k=2, voxel_size=1.0)
    runner = RiskPipelineRunner(config=cfg)
    outputs = runner.run(scene_graph=scene, memory=_memory(dim=8), T_t=np.eye(4))

    assert loader.last_scene_stats is not None
    assert loader.last_scene_stats["selected_scan_id"] == "MasterBedroom-33296"
    assert len(outputs["node_order"]) == 2


def test_scene_center_does_not_make_valid_boxes_empty(tmp_path: Path) -> None:
    rel_path, box_path = _write_multi_scan_files(tmp_path)
    loader = SGFrontLoader()
    scene = loader.load(rel_path, box_path, scan_id="MasterBedroom-33296")

    assert loader.last_scene_stats is not None
    assert loader.last_scene_stats["num_valid_boxes"] > 0
    assert len(scene.objects) == 2


def test_loader_runner_mapper_bbox_smoke(tmp_path: Path) -> None:
    rel_path, box_path = _write_multi_scan_files(tmp_path)
    scene = SGFrontLoader().load(rel_path, box_path, scan_id="MasterBedroom-33296")

    cfg = PipelineConfig(hidden_dim=8, retrieval_top_k=2, voxel_size=1.0)
    runner = RiskPipelineRunner(config=cfg)
    outputs = runner.run(scene_graph=scene, memory=_memory(dim=8), T_t=np.eye(4))

    assert len(outputs["integrated_voxels"]) == len(outputs["node_order"])
    assert len(outputs["integrated_voxels"]) > 0


def test_retrieval_output_can_change_with_top_k() -> None:
    z_i = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    memory = FailurePrototypeMemory.from_dict(
        {
            "prototypes": [
                {"prototype_id": "p0", "embedding": [1.0, 0.0, 0.0], "metadata": {}},
                {"prototype_id": "p1", "embedding": [0.8, 0.6, 0.0], "metadata": {}},
                {"prototype_id": "p2", "embedding": [0.0, 1.0, 0.0], "metadata": {}},
            ]
        }
    )

    r_top1 = FailureRetriever(top_k=1, temperature=0.2)(z_i, memory)
    r_top2 = FailureRetriever(top_k=2, temperature=0.2)(z_i, memory)

    assert not torch.allclose(r_top1, r_top2)


def test_runner_modulation_changes_node_representation(tmp_path: Path) -> None:
    rel_path, box_path = _write_scene_files(tmp_path)
    scene = SGFrontLoader().load(rel_path, box_path)

    cfg = PipelineConfig(hidden_dim=8, retrieval_top_k=2, voxel_size=1.0)
    runner = RiskPipelineRunner(config=cfg)
    outputs = runner.run(scene_graph=scene, memory=_memory(dim=8), T_t=np.eye(4))

    assert not torch.allclose(outputs["z_i"], outputs["z_i_prime"])
