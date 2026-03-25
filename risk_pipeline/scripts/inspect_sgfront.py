from __future__ import annotations

import argparse

from risk_pipeline.data.sgfront_loader import SGFrontLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SG-FRONT JSON files")
    parser.add_argument("relationships_json")
    parser.add_argument("obj_boxes_json")
    args = parser.parse_args()

    loader = SGFrontLoader()
    bundle = loader.load(args.relationships_json, args.obj_boxes_json)

    print(f"objects: {len(bundle.objects)}")
    print(f"relations: {len(bundle.graph.relations)}")


if __name__ == "__main__":
    main()
