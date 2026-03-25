from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ovo.entities.sgfront_dataloader import SGFrontDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview SG-FRONT scene graph.")
    parser.add_argument("--relationships", type=Path, required=True, help="Path to relationships JSON.")
    parser.add_argument("--obj-boxes", type=Path, required=True, help="Path to obj_boxes JSON.")
    parser.add_argument("--scan-id", type=str, default=None, help="Optional scan id to select.")
    parser.add_argument("--scan-index", type=int, default=0, help="Fallback scan index if --scan-id is not set.")
    parser.add_argument("--max-edges-preview", type=int, default=10, help="Number of edges to preview.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    loader = SGFrontDataLoader()
    scene_graph = loader.load_scene_graph(
        relationships_json=args.relationships,
        obj_boxes_json=args.obj_boxes,
        scan_id=args.scan_id,
        scan_index=args.scan_index,
    )

    graph = scene_graph.graph
    node_items = sorted(graph.nodes.items(), key=lambda item: item[0])

    print(scene_graph.summary())
    print("\n[Nodes Preview]")
    for node_id, node in node_items[: args.max_edges_preview]:
        cx, cy, cz = node.bbox.center
        sx, sy, sz = node.bbox.size
        print(
            f"- id={node_id} label={node.class_label} "
            f"center=({cx:.3f}, {cy:.3f}, {cz:.3f}) size=({sx:.3f}, {sy:.3f}, {sz:.3f})"
        )

    print("\n[Edges Preview]")
    for edge in graph.edges[: args.max_edges_preview]:
        print(
            f"- ({edge.subject_id}) -[{edge.relation_name}:{edge.relation_id}]-> ({edge.object_id})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
