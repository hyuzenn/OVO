import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ovo.entities.sgfront_dataloader import SGFrontDataLoader, SceneGraph


def test_list_scan_ids_nonempty(tmp_path: Path):
    relationships_index = {
        "scan_001": {
            "objects": {"1": "chair"},
            "relationships": [],
        },
        "scan_002": {
            "objects": {"2": "table"},
            "relationships": [],
        },
    }
    rel_path = tmp_path / "relationships_index.json"
    rel_path.write_text(json.dumps(relationships_index), encoding="utf-8")

    scan_ids = SGFrontDataLoader.list_scan_ids(rel_path)

    assert scan_ids
    assert set(scan_ids) == {"scan_001", "scan_002"}


def test_load_scan_to_scenegraph(tmp_path: Path):
    relationships_index = {
        "scan_001": {
            "objects": {
                "1": "chair",
                "2": "table",
            },
            "relationships": [
                [1, 2, 0, "left of"],
            ],
        }
    }
    obj_boxes_index = {
        "scan_001": {
            "1": {"param7": [0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0]},
            "2": {"param7": [1.0, 0.0, 0.0, 1.2, 0.7, 0.8, 0.2]},
        }
    }

    rel_path = tmp_path / "relationships_index.json"
    box_path = tmp_path / "obj_boxes_index.json"
    rel_path.write_text(json.dumps(relationships_index), encoding="utf-8")
    box_path.write_text(json.dumps(obj_boxes_index), encoding="utf-8")

    loader = SGFrontDataLoader(embedding_dim=8)
    graph = loader.load_scan("scan_001", rel_path, box_path)

    assert isinstance(graph, SceneGraph)
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1


def test_bbox_parse_param7():
    center, size, yaw = SGFrontDataLoader._parse_param7([1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 0.5])

    volume = size[0] * size[1] * size[2]

    assert center == (1.0, 2.0, 3.0)
    assert size == (2.0, 4.0, 5.0)
    assert yaw == 0.5
    assert volume == 40.0
