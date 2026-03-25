from __future__ import annotations

import json
from pathlib import Path

from risk_pipeline.data.sgfront_loader import SGFrontLoader


def test_loader_selects_requested_scan_and_ignores_scene_center(tmp_path: Path) -> None:
    rel_payload = {
        "MasterBedroom-111": {
            "objects": {"1": "bed"},
            "relationships": [[1, 1, 0, "intersect"]],
        },
        "MasterBedroom-33296": {
            "objects": {"1": "chair", "2": "table"},
            "relationships": [[1, 2, 0, "left of"]],
        },
    }
    box_payload = {
        "MasterBedroom-111": {
            "1": {"param7": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]},
            "scene_center": [0.0, 0.0, 0.0],
        },
        "MasterBedroom-33296": {
            "1": {"param7": [0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0]},
            "2": {"param7": [1.0, 0.0, 0.0, 1.2, 0.7, 0.8, 0.1]},
            "scene_center": [0.0, 0.0, 0.0],
        },
    }

    rel_path = tmp_path / "relationships.json"
    box_path = tmp_path / "obj_boxes.json"
    rel_path.write_text(json.dumps(rel_payload), encoding="utf-8")
    box_path.write_text(json.dumps(box_payload), encoding="utf-8")

    loader = SGFrontLoader()
    bundle = loader.load(rel_path, box_path, scan_id="MasterBedroom-33296")

    assert len(bundle.objects) == 2
    assert len(bundle.graph.relations) == 1
    assert loader.last_scene_stats is not None
    assert loader.last_scene_stats["selected_scan_id"] == "MasterBedroom-33296"
    assert loader.last_scene_stats["num_valid_boxes"] == 2
