from pathlib import Path

from risk_pipeline.data.sgfront_loader import SGFrontLoader


def test_loader_reads_sample_data() -> None:
    root = Path(__file__).resolve().parents[2]
    rel_json = root / "sample_data" / "relationships_0.json"
    boxes_json = root / "sample_data" / "obj_boxes_0.json"

    bundle = SGFrontLoader().load(rel_json, boxes_json)

    assert bundle.objects
    assert bundle.graph.relations
    assert len(bundle.graph.adjacency) > 0
