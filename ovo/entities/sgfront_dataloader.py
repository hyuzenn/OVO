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
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from .risk_dummy_pipeline import DummyBatch
from .risk_schema import (
    BBox3D,
    GraphDictionary,
    Instance3D,
    RelationTriplet,
    RiskAugmentedObjectDictionary,
)


def _to_int(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit() or ch == "-")
        if digits:
            return int(digits)
    return int(value)


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
        rel_data = self._read_json(relationships_json)
        box_data = self._read_json(obj_boxes_json)

        boxes = self._parse_obj_boxes(box_data)
        nodes = self._parse_objects(rel_data, boxes)
        edges, adjacency, edge_vocab = self._parse_relationships(rel_data)

        graph = GraphDictionary(
            nodes=nodes,
            edges=edges,
            edge_type_vocab=edge_vocab,
            adjacency=adjacency,
        )
        obj_dict = RiskAugmentedObjectDictionary(objects=nodes)
        return obj_dict, graph

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
