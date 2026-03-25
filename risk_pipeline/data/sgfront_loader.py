from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

from risk_pipeline.core.structures import GraphData, Object3D, Relation, SceneBundle


class SGFrontLoader:
    """Load SG-FRONT relationship/box JSON files into lightweight structures."""

    def __init__(self) -> None:
        self.last_scene_stats: dict | None = None

    def load(
        self,
        relationships_json: str | Path,
        obj_boxes_json: str | Path,
        *,
        scan_id: str | None = None,
    ) -> SceneBundle:
        rel_data = self._read_json(relationships_json)
        box_data = self._read_json(obj_boxes_json)

        rel_scene, selected_scan_id = self._select_relationship_scene(rel_data, scan_id=scan_id)
        box_scene = self._select_box_scene(box_data, selected_scan_id=selected_scan_id)

        objects, valid_box_count = self._parse_objects(rel_scene, box_scene)
        graph = self._parse_graph(rel_scene)
        self.last_scene_stats = {
            "selected_scan_id": selected_scan_id,
            "num_objects": len(objects),
            "num_relationships": len(graph.relations),
            "num_valid_boxes": valid_box_count,
        }
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

    def _parse_objects(self, rel_data: dict, box_data: dict) -> tuple[Dict[int, Object3D], int]:
        objects: Dict[int, Object3D] = {}
        valid_box_count = 0
        raw_objects = rel_data.get("objects", {})

        for raw_id, raw_label in raw_objects.items():
            obj_id = self._to_int(raw_id)
            box_payload = box_data.get(str(raw_id), box_data.get(obj_id, {}))
            center, size, yaw = self._parse_param7(box_payload.get("param7", []))
            if len(box_payload.get("param7", [])) >= 7:
                valid_box_count += 1
            objects[obj_id] = Object3D(
                object_id=obj_id,
                label=str(raw_label),
                center=center,
                size=size,
                yaw=yaw,
            )

        return objects, valid_box_count

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

    @staticmethod
    def _is_scene_payload(payload: object) -> bool:
        return isinstance(payload, dict) and ("objects" in payload or "relationships" in payload)

    def _select_relationship_scene(self, rel_data: dict, *, scan_id: str | None) -> tuple[dict, str]:
        if self._is_scene_payload(rel_data):
            return rel_data, scan_id or "__single_scene__"

        if isinstance(rel_data, dict) and "scans" in rel_data and isinstance(rel_data["scans"], list):
            scan_map = {
                str(item.get("scan") or item.get("scan_id")): item
                for item in rel_data["scans"]
                if isinstance(item, dict)
            }
            selected = scan_id or next(iter(scan_map), None)
            if selected is None or selected not in scan_map:
                raise ValueError(f"Scan id '{scan_id}' not found in relationships JSON")
            return scan_map[selected], selected

        if isinstance(rel_data, dict):
            scan_candidates = {str(k): v for k, v in rel_data.items() if self._is_scene_payload(v)}
            selected = scan_id or next(iter(scan_candidates), None)
            if selected is None or selected not in scan_candidates:
                raise ValueError(f"Scan id '{scan_id}' not found in relationships JSON")
            return scan_candidates[selected], selected

        raise ValueError("Unsupported relationships JSON structure")

    def _select_box_scene(self, box_data: dict, *, selected_scan_id: str) -> dict:
        if self._is_object_box_payload(box_data):
            return self._strip_scene_center(box_data)

        if isinstance(box_data, dict) and "scans" in box_data and isinstance(box_data["scans"], list):
            scan_map = {
                str(item.get("scan") or item.get("scan_id")): item.get("boxes", item)
                for item in box_data["scans"]
                if isinstance(item, dict)
            }
            resolved_scan_id = selected_scan_id
            if selected_scan_id == "__single_scene__":
                resolved_scan_id = next(iter(scan_map), None)
            if resolved_scan_id is None or resolved_scan_id not in scan_map:
                raise ValueError(
                    f"Selected scan '{selected_scan_id}' is not present in obj_boxes JSON"
                )
            return self._strip_scene_center(scan_map[resolved_scan_id])

        if isinstance(box_data, dict):
            resolved_scan_id = selected_scan_id
            if selected_scan_id == "__single_scene__":
                resolved_scan_id = next(iter(box_data), None)
            if resolved_scan_id is None or resolved_scan_id not in box_data:
                raise ValueError(
                    f"Selected scan '{selected_scan_id}' is not present in obj_boxes JSON"
                )
            payload = box_data[resolved_scan_id]
            if not isinstance(payload, dict):
                raise ValueError(f"obj_boxes payload for scan '{selected_scan_id}' must be an object")
            return self._strip_scene_center(payload)

        raise ValueError("Unsupported obj_boxes JSON structure")

    @staticmethod
    def _is_object_box_payload(payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        if "scene_center" in payload:
            return True
        if not payload:
            return False
        first_val = next(iter(payload.values()))
        return isinstance(first_val, dict) and "param7" in first_val

    @staticmethod
    def _strip_scene_center(payload: dict) -> dict:
        return {key: value for key, value in payload.items() if key != "scene_center"}
