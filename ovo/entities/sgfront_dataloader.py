"""SG-FRONT JSON parser for risk-aware scene-graph pipeline.

This loader parses SG-FRONT relationships/obj_boxes JSON files into:
- RiskAugmentedObjectDictionary
- GraphDictionary
and converts a parsed graph into the tensor `DummyBatch` format used by
`ovo.entities.risk_dummy_pipeline`.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from .risk_dummy_pipeline import DummyBatch
from .risk_schema import (
    BBox3D,
    GraphDictionary,
    Instance3D,
    RelationTriplet,
    RiskAugmentedObjectDictionary,
    SceneGraph,
)


def _to_int(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit() or ch == "-")
        if digits:
            return int(digits)
    return int(value)


def _normalize_id(value: Any) -> int | str:
    """Normalize mixed int/str ids.

    Policy:
    - int-like values are converted to int
    - otherwise keep a trimmed string representation
    """

    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            return text
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _stable_seed(text: str, base_seed: int = 17) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) + base_seed) % (2**31 - 1)


class SGFrontDataLoader:
    """Parser and tensorizer for SG-FRONT scene-graph JSON files."""

    def __init__(
        self,
        embedding_dim: int = 32,
        embedding_mode: str = "random",  # random | text
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.embedding_mode = embedding_mode
        self.device = device

    def load(
        self,
        relationships_json: str | Path,
        obj_boxes_json: str | Path,
    ) -> Tuple[RiskAugmentedObjectDictionary, GraphDictionary]:
        scene_graph = self.load_scene_graph(
            relationships_json=relationships_json,
            obj_boxes_json=obj_boxes_json,
            scan_id=None,
            scan_index=0,
        )
        obj_dict = RiskAugmentedObjectDictionary(objects=scene_graph.graph.nodes)
        return obj_dict, scene_graph.graph

    def load_scene_graph(
        self,
        relationships_json: str | Path,
        obj_boxes_json: str | Path,
        scan_id: str | None = None,
        scan_index: int = 0,
    ) -> SceneGraph:
        rel_data = self._read_json(relationships_json)
        box_data = self._read_json(obj_boxes_json)

        rel_scans = self._extract_relationship_scans(rel_data)
        box_scans = self._extract_box_scans(box_data)
        selected_scan_id, rel_scan, box_scan = self._select_scan(
            rel_scans=rel_scans,
            box_scans=box_scans,
            scan_id=scan_id,
            scan_index=scan_index,
        )

        boxes = self._parse_obj_boxes(box_scan)
        nodes = self._parse_objects(rel_scan, boxes)
        edges, adjacency, edge_vocab = self._parse_relationships(rel_scan)

        graph = GraphDictionary(
            nodes=nodes,
            edges=edges,
            edge_type_vocab=edge_vocab,
            adjacency=adjacency,
        )
        return SceneGraph(scan_id=selected_scan_id, graph=graph)

    def to_tensor_batch(
        self,
        graph: GraphDictionary,
        pose_t: torch.Tensor | None = None,
    ) -> DummyBatch:
        """Convert parsed graph to tensor batch format.

        Output shapes:
          z_i       [B=1, N, D]
          bbox_geom [B=1, N, G=8]
          edge_idx  [B=1, E, 2]
          edge_type [B=1, E]
          pose_t    [B=1, 4, 4]
        """

        node_ids = sorted(graph.nodes.keys())
        id2idx = {nid: i for i, nid in enumerate(node_ids)}

        z_i = []
        bbox_geom = []
        for nid in node_ids:
            node = graph.nodes[nid]
            z_i.append(torch.tensor(node.z_i, dtype=torch.float32))
            b = node.bbox
            sx, sy, sz = b.size
            cx, cy, cz = b.center
            volume = float(b.volume if b.volume is not None else sx * sy * sz)
            aspect = float(b.aspect_ratio if b.aspect_ratio is not None else sx / max(sy, 1e-6))
            geom = torch.tensor([sx, sy, sz, volume, aspect, cx, cy, cz], dtype=torch.float32)
            bbox_geom.append(geom)

        z_i_t = torch.stack(z_i, dim=0).unsqueeze(0).to(self.device)
        bbox_t = torch.stack(bbox_geom, dim=0).unsqueeze(0).to(self.device)

        if graph.edges:
            e_idx = []
            e_typ = []
            for e in graph.edges:
                if e.subject_id not in id2idx or e.object_id not in id2idx:
                    continue
                e_idx.append([id2idx[e.subject_id], id2idx[e.object_id]])
                e_typ.append(graph.edge_type_vocab.get(e.relation_name, 0))
            edge_idx = torch.tensor(e_idx, dtype=torch.long, device=self.device).unsqueeze(0)
            edge_type = torch.tensor(e_typ, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            edge_idx = torch.zeros((1, 0, 2), dtype=torch.long, device=self.device)
            edge_type = torch.zeros((1, 0), dtype=torch.long, device=self.device)

        if pose_t is None:
            pose_t = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            pose_t = pose_t.to(self.device)
            if pose_t.dim() == 2:
                pose_t = pose_t.unsqueeze(0)

        return DummyBatch(
            z_i=z_i_t,
            bbox_geom=bbox_t,
            edge_idx=edge_idx,
            edge_type=edge_type,
            pose_t=pose_t,
        )

    @staticmethod
    def _read_json(path: str | Path) -> dict:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _extract_relationship_scans(rel_data: dict) -> Dict[str, dict]:
        if isinstance(rel_data, dict) and "objects" in rel_data and "relationships" in rel_data:
            scan_key = str(rel_data.get("scan") or rel_data.get("scan_id") or "scan_0")
            return {scan_key: rel_data}

        scans = rel_data.get("scans", []) if isinstance(rel_data, dict) else []
        if isinstance(scans, list):
            out: Dict[str, dict] = {}
            for i, item in enumerate(scans):
                if not isinstance(item, dict):
                    continue
                sid = str(item.get("scan") or item.get("scan_id") or item.get("id") or f"scan_{i}")
                out[sid] = item
            if out:
                return out
        raise ValueError("Invalid relationships JSON format: expected single scan or `scans` list.")

    @staticmethod
    def _looks_like_box_payload(value: dict) -> bool:
        return "param7" in value or "8points" in value or "8_points" in value

    def _extract_box_scans(self, box_data: dict) -> Dict[str, dict]:
        if not isinstance(box_data, dict):
            raise ValueError("Invalid obj_boxes JSON format: expected dictionary.")

        if box_data and all(isinstance(v, dict) and self._looks_like_box_payload(v) for v in box_data.values()):
            return {"scan_0": box_data}

        scans = box_data.get("scans", [])
        if isinstance(scans, list):
            out: Dict[str, dict] = {}
            for i, item in enumerate(scans):
                if not isinstance(item, dict):
                    continue
                sid = str(item.get("scan") or item.get("scan_id") or item.get("id") or f"scan_{i}")
                boxes = item.get("obj_boxes") or item.get("boxes") or item.get("objects")
                if isinstance(boxes, dict):
                    out[sid] = boxes
            if out:
                return out

        scan_mapped = {
            str(k): v
            for k, v in box_data.items()
            if isinstance(v, dict)
            and v
            and all(isinstance(obj_payload, dict) and self._looks_like_box_payload(obj_payload) for obj_payload in v.values())
        }
        if scan_mapped:
            return scan_mapped

        raise ValueError("Invalid obj_boxes JSON format: could not parse scan/object boxes.")

    @staticmethod
    def _select_scan(
        rel_scans: Dict[str, dict],
        box_scans: Dict[str, dict],
        scan_id: str | None,
        scan_index: int,
    ) -> Tuple[str, dict, dict]:
        common_scan_ids = sorted(set(rel_scans.keys()) & set(box_scans.keys()))
        if not common_scan_ids:
            if len(rel_scans) == 1 and len(box_scans) == 1:
                rel_id = next(iter(rel_scans))
                box_id = next(iter(box_scans))
                return rel_id, rel_scans[rel_id], box_scans[box_id]
            raise ValueError("No matching scan IDs between relationships and obj_boxes JSON files.")

        if scan_id is not None:
            if scan_id not in common_scan_ids:
                raise ValueError(f"Requested scan_id '{scan_id}' not found. Available: {common_scan_ids[:10]}")
            selected_id = scan_id
        else:
            if scan_index < 0 or scan_index >= len(common_scan_ids):
                raise IndexError(f"scan_index={scan_index} out of range for {len(common_scan_ids)} available scans.")
            selected_id = common_scan_ids[scan_index]

        return selected_id, rel_scans[selected_id], box_scans[selected_id]

    def _parse_objects(self, rel_data: dict, boxes: Dict[int, BBox3D]) -> Dict[int, Instance3D]:
        objects = rel_data.get("objects", {})
        nodes: Dict[int, Instance3D] = {}

        for raw_id, raw_label in objects.items():
            obj_id = _to_int(raw_id)
            class_label = str(raw_label)
            bbox = boxes.get(obj_id, BBox3D(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), yaw=0.0))
            nodes[obj_id] = Instance3D(
                instance_id=obj_id,
                class_label=class_label,
                bbox=bbox,
                z_i=self._init_embedding(class_label),
            )

        return nodes

    @staticmethod
    def _parse_relationships(rel_data: dict) -> Tuple[List[RelationTriplet], Dict[int, List[int]], Dict[str, int]]:
        raw_rels = rel_data.get("relationships", [])
        edges: List[RelationTriplet] = []
        adjacency: Dict[int, List[int]] = {}
        edge_vocab: Dict[str, int] = {}

        for item in raw_rels:
            if not isinstance(item, Sequence) or len(item) < 4:
                continue
            subject_id = _to_int(item[0])
            object_id = _to_int(item[1])
            relation_id = _to_int(item[2])
            relation_name = str(item[3])

            edge_vocab.setdefault(relation_name, len(edge_vocab))
            edges.append(
                RelationTriplet(
                    subject_id=subject_id,
                    object_id=object_id,
                    relation_name=relation_name,
                    relation_id=relation_id,
                )
            )
            adjacency.setdefault(subject_id, []).append(object_id)
            adjacency.setdefault(object_id, [])

        return edges, adjacency, edge_vocab

    @staticmethod
    def _parse_obj_boxes(box_data: dict) -> Dict[int, BBox3D]:
        boxes: Dict[int, BBox3D] = {}

        for raw_id, payload in box_data.items():
            if not isinstance(payload, dict):
                continue

            obj_id = _to_int(raw_id)
            param7 = payload.get("param7", [])
            points = payload.get("8points", payload.get("8_points", []))

            center, size, yaw = SGFrontDataLoader._parse_param7(param7)
            corners = SGFrontDataLoader._parse_8points(points)
            volume = float(size[0] * size[1] * size[2])
            aspect = float(size[0] / max(size[1], 1e-6))

            boxes[obj_id] = BBox3D(
                center=center,
                size=size,
                yaw=yaw,
                corners_8=corners,
                volume=volume,
                aspect_ratio=aspect,
            )

        return boxes

    @staticmethod
    def _parse_param7(param7: Iterable[float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
        values = list(param7)
        if len(values) >= 7:
            x, y, z, sx, sy, sz, yaw = [float(v) for v in values[:7]]
            return (x, y, z), (abs(sx), abs(sy), abs(sz)), yaw
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0

    @staticmethod
    def _parse_8points(points: Iterable[Iterable[float]]) -> List[Tuple[float, float, float]]:
        out: List[Tuple[float, float, float]] = []
        for p in points:
            vals = list(p)
            if len(vals) >= 3:
                out.append((float(vals[0]), float(vals[1]), float(vals[2])))
        return out

    def _init_embedding(self, class_label: str) -> List[float]:
        if self.embedding_mode == "text":
            text_emb = self._sentence_embedding(class_label)
            if text_emb is not None:
                return text_emb
        return self._random_embedding(class_label)

    def _random_embedding(self, class_label: str) -> List[float]:
        g = torch.Generator(device="cpu")
        g.manual_seed(_stable_seed(class_label))
        z = torch.randn(self.embedding_dim, generator=g)
        z = torch.nn.functional.normalize(z, dim=0)
        return z.tolist()

    def _sentence_embedding(self, class_label: str) -> List[float] | None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode([class_label], normalize_embeddings=True)[0]
        out = torch.tensor(emb, dtype=torch.float32)
        if out.numel() != self.embedding_dim:
            out = self._fit_dim(out, self.embedding_dim)
        return out.tolist()

    @staticmethod
    def _fit_dim(vec: torch.Tensor, dim: int) -> torch.Tensor:
        if vec.numel() == dim:
            return vec
        if vec.numel() > dim:
            return vec[:dim]
        pad = torch.zeros(dim - vec.numel(), dtype=vec.dtype)
        return torch.cat([vec, pad], dim=0)


@dataclass(slots=True)
class SGObject:
    object_id: int | str
    class_name: str
    bbox: Tuple[float, float, float, float, float, float]  # cx, cy, cz, sx, sy, sz


@dataclass(slots=True)
class SGRelation:
    subject_id: int | str
    object_id: int | str
    relation_id: int | str | None
    relation_name: str


@dataclass(slots=True)
class SceneGraph:
    scan_id: str
    objects: Dict[int | str, SGObject] = field(default_factory=dict)
    relationships: List[SGRelation] = field(default_factory=list)


class SGFrontLoader:
    """Scan-aware SG-FRONT loader that converts raw JSON into SceneGraph."""

    _DEFAULT_SCAN_ID = "__default__"
    _DEFAULT_BBOX = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    def __init__(self, relationships_path: str | Path, obj_boxes_path: str | Path) -> None:
        self.relationships_path = Path(relationships_path)
        self.obj_boxes_path = Path(obj_boxes_path)
        self._relationships_data = self._read_json(self.relationships_path)
        self._obj_boxes_data = self._read_json(self.obj_boxes_path)
        self._scan_to_rel = self._index_relationship_scans(self._relationships_data)
        self._scan_to_boxes = self._index_obj_box_scans(self._obj_boxes_data)
        self._scan_ids = sorted(set(self._scan_to_rel) | set(self._scan_to_boxes))

    def list_scan_ids(self) -> list[str]:
        return list(self._scan_ids)

    def infer_scan_id(self, index: int = 0) -> str:
        if not self._scan_ids:
            raise ValueError("No scan_id available in relationships/obj_boxes JSON.")
        if index < 0 or index >= len(self._scan_ids):
            raise IndexError(f"scan index out of range: {index} (num_scans={len(self._scan_ids)})")
        return self._scan_ids[index]

    def load_scan(self, scan_id: str) -> SceneGraph:
        if scan_id not in self._scan_ids:
            raise KeyError(f"scan_id '{scan_id}' not found. available={self._scan_ids}")

        rel_scan = self._scan_to_rel.get(scan_id, {})
        box_scan = self._scan_to_boxes.get(scan_id, {})

        objects_raw = rel_scan.get("objects", {})
        if not isinstance(objects_raw, dict):
            warnings.warn(f"[SGFrontLoader] scan '{scan_id}': invalid objects payload; using empty dict.")
            objects_raw = {}

        boxes = self._parse_box_map(scan_id, box_scan)
        objects = self._parse_objects(scan_id, objects_raw, boxes)
        relationships = self._parse_relationships(scan_id, rel_scan.get("relationships", []), objects)
        return SceneGraph(scan_id=scan_id, objects=objects, relationships=relationships)

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _index_relationship_scans(self, rel_data: dict) -> Dict[str, dict]:
        # Canonical SG-FRONT format: {"scans": [{"scan": "...", "objects": ..., "relationships": ...}, ...]}
        scans = rel_data.get("scans")
        if isinstance(scans, list):
            out: Dict[str, dict] = {}
            for scan_item in scans:
                if not isinstance(scan_item, dict):
                    continue
                raw_scan_id = scan_item.get("scan") or scan_item.get("scan_id")
                if raw_scan_id is None:
                    warnings.warn("[SGFrontLoader] Encountered scan entry without scan id; skipped.")
                    continue
                scan_id = str(raw_scan_id)
                out[scan_id] = scan_item
            return out

        # Backward compatibility: single-scene format with top-level objects/relationships.
        if "objects" in rel_data or "relationships" in rel_data:
            warnings.warn(
                "[SGFrontLoader] relationships JSON has no 'scans'; treating as single-scene data."
            )
            return {self._DEFAULT_SCAN_ID: rel_data}

        return {}

    def _index_obj_box_scans(self, box_data: dict) -> Dict[str, dict]:
        # Expected: {"scan_id_a": {...}, "scan_id_b": {...}} OR single-scene {"1": {...}, ...}
        if not box_data:
            return {}

        first_val = next(iter(box_data.values()))
        if isinstance(first_val, dict) and ("param7" in first_val or "8points" in first_val or "8_points" in first_val):
            warnings.warn("[SGFrontLoader] obj_boxes JSON appears single-scene; mapped to default scan id.")
            return {self._DEFAULT_SCAN_ID: box_data}

        out: Dict[str, dict] = {}
        for raw_scan_id, payload in box_data.items():
            if isinstance(payload, dict):
                out[str(raw_scan_id)] = payload
        return out

    def _parse_box_map(self, scan_id: str, box_scan_data: dict) -> Dict[int | str, Tuple[float, float, float, float, float, float]]:
        out: Dict[int | str, Tuple[float, float, float, float, float, float]] = {}
        for raw_obj_id, payload in box_scan_data.items():
            if not isinstance(payload, dict):
                warnings.warn(f"[SGFrontLoader] scan '{scan_id}': invalid box payload for id={raw_obj_id}; skipped.")
                continue
            obj_id = _normalize_id(raw_obj_id)
            param7 = payload.get("param7")
            if isinstance(param7, Sequence) and len(param7) >= 6:
                cx, cy, cz = float(param7[0]), float(param7[1]), float(param7[2])
                sx, sy, sz = abs(float(param7[3])), abs(float(param7[4])), abs(float(param7[5]))
                out[obj_id] = (cx, cy, cz, sx, sy, sz)
            else:
                warnings.warn(
                    f"[SGFrontLoader] scan '{scan_id}': missing/invalid param7 for object {obj_id}; using default bbox."
                )
                out[obj_id] = self._DEFAULT_BBOX
        return out

    def _parse_objects(
        self,
        scan_id: str,
        objects_raw: dict,
        boxes: Dict[int | str, Tuple[float, float, float, float, float, float]],
    ) -> Dict[int | str, SGObject]:
        objects: Dict[int | str, SGObject] = {}
        for raw_obj_id, class_name in objects_raw.items():
            obj_id = _normalize_id(raw_obj_id)
            bbox = boxes.get(obj_id, self._DEFAULT_BBOX)
            if obj_id not in boxes:
                warnings.warn(
                    f"[SGFrontLoader] scan '{scan_id}': bbox missing for object {obj_id}; using default bbox."
                )
            objects[obj_id] = SGObject(object_id=obj_id, class_name=str(class_name), bbox=bbox)
        return objects

    def _parse_relationships(
        self,
        scan_id: str,
        relationships_raw: Any,
        objects: Dict[int | str, SGObject],
    ) -> List[SGRelation]:
        if not isinstance(relationships_raw, list):
            warnings.warn(f"[SGFrontLoader] scan '{scan_id}': invalid relationships payload; using empty list.")
            return []

        relations: List[SGRelation] = []
        for idx, item in enumerate(relationships_raw):
            if not isinstance(item, Sequence) or len(item) < 4:
                warnings.warn(f"[SGFrontLoader] scan '{scan_id}': malformed relationship at index {idx}; skipped.")
                continue
            subj_id = _normalize_id(item[0])
            obj_id = _normalize_id(item[1])
            rel_id = _normalize_id(item[2]) if item[2] is not None else None
            rel_name = str(item[3])
            if subj_id not in objects or obj_id not in objects:
                warnings.warn(
                    f"[SGFrontLoader] scan '{scan_id}': relationship {item} references missing object(s); skipped."
                )
                continue
            relations.append(
                SGRelation(subject_id=subj_id, object_id=obj_id, relation_id=rel_id, relation_name=rel_name)
            )
        return relations
