from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

from risk_pipeline.core.structures import GraphData, Object3D, Relation, SceneBundle


class SGFrontLoader:
    """Load SG-FRONT relationship/box JSON files into lightweight structures."""

    def load(self, relationships_json: str | Path, obj_boxes_json: str | Path) -> SceneBundle:
        rel_data = self._read_json(relationships_json)
        box_data = self._read_json(obj_boxes_json)

        objects = self._parse_objects(rel_data, box_data)
        graph = self._parse_graph(rel_data)
        return SceneBundle(objects=objects, graph=graph)

    @staticmethod
    def _read_json(path: str | Path) -> dict:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _to_int(value) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit() or ch == "-")
            if digits:
                return int(digits)
        return int(value)

    def _parse_objects(self, rel_data: dict, box_data: dict) -> Dict[int, Object3D]:
        objects: Dict[int, Object3D] = {}
        raw_objects = rel_data.get("objects", {})

        for raw_id, raw_label in raw_objects.items():
            obj_id = self._to_int(raw_id)
            box_payload = box_data.get(str(raw_id), box_data.get(obj_id, {}))
            center, size, yaw = self._parse_param7(box_payload.get("param7", []))
            objects[obj_id] = Object3D(
                object_id=obj_id,
                label=str(raw_label),
                center=center,
                size=size,
                yaw=yaw,
            )

        return objects

    def _parse_graph(self, rel_data: dict) -> GraphData:
        relations = []
        adjacency: Dict[int, list[int]] = {}
        for item in rel_data.get("relationships", []):
            if not isinstance(item, Sequence) or len(item) < 4:
                continue
            subject_id = self._to_int(item[0])
            object_id = self._to_int(item[1])
            relation_id = self._to_int(item[2])
            relation_name = str(item[3])
            relations.append(
                Relation(
                    subject_id=subject_id,
                    object_id=object_id,
                    relation_id=relation_id,
                    relation_name=relation_name,
                )
            )
            adjacency.setdefault(subject_id, []).append(object_id)
            adjacency.setdefault(object_id, [])

        return GraphData(relations=relations, adjacency=adjacency)

    @staticmethod
    def _parse_param7(param7: Sequence[float]) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        if len(param7) >= 7:
            return (
                (float(param7[0]), float(param7[1]), float(param7[2])),
                (float(param7[3]), float(param7[4]), float(param7[5])),
                float(param7[6]),
            )
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0
