from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple prototype failure memory JSON")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num-prototypes", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("failure_memory.json"))
    args = parser.parse_args()

    prototypes = []
    for i in range(args.num_prototypes):
        value = (i + 1) / max(args.num_prototypes, 1)
        prototypes.append(
            {
                "prototype_id": f"proto_{i}",
                "embedding": [float(value)] * args.dim,
                "metadata": {
                    "failure_type": ["collision", "stuck", "fall", "near_miss"][i % 4],
                    "source": "prototype_script",
                },
            }
        )

    payload = {
        "description": "Simple deterministic prototype memory for MVP pipeline runs",
        "dim": args.dim,
        "prototypes": prototypes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[build_failure_memory] wrote {len(prototypes)} prototypes (dim={args.dim}) to {args.output}")


if __name__ == "__main__":
    main()
