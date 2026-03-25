from __future__ import annotations

import argparse

from risk_pipeline.data.sgfront_loader import SGFrontLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SG-FRONT JSON files")
    parser.add_argument("--relationships-json", required=True)
    parser.add_argument("--obj-boxes-json", required=True)
    parser.add_argument("--scan-id")
    args = parser.parse_args()

    loader = SGFrontLoader()
    bundle = loader.load(args.relationships_json, args.obj_boxes_json, scan_id=args.scan_id)

    if loader.last_scene_stats:
        stats = loader.last_scene_stats
        print(f"selected scan id: {stats['selected_scan_id']}")
        print(f"num objects: {stats['num_objects']}")
        print(f"num relationships: {stats['num_relationships']}")
        print(f"num valid boxes: {stats['num_valid_boxes']}")
    print(f"objects: {len(bundle.objects)}")
    print(f"relations: {len(bundle.graph.relations)}")


if __name__ == "__main__":
    main()
