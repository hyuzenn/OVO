"""Data structures for risk-aware semantic mapping.

This module defines the canonical object/graph dictionaries used by the
SG-FRONT-based failure-aware pipeline.

Note:
    These schemas assume Scene Graph input (nodes/edges/bbox features),
    not raw RGB-D frame tensors from the default OVO-SLAM perception path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, TypedDict


Vector = List[float]  # expected per-instance shape: [D]
NodeId = int
EdgeType = str
Point3D = Tuple[float, float, float]


@dataclass(slots=True)
class BBox3D:
    """3D bounding-box summary from SG-FRONT obj_boxes files."""

    center: Point3D
    size: Point3D
    yaw: float
    corners_8: Optional[List[Point3D]] = None
    volume: Optional[float] = None
    aspect_ratio: Optional[float] = None


@dataclass(slots=True)
class RiskScores:
    """Scalar risk decomposition for one object instance."""

    p_obj: float = 0.0
    p_ctx: float = 0.0
    p_f: float = 0.0
    failure_type: Optional[Literal["collision", "stuck", "fall", "other"]] = None


@dataclass(slots=True)
class Instance3D:
    """Risk-augmented object record.

    Required fields from the design document:
      - z_i: original semantic embedding
      - r_i_retr: retrieved risk embedding from failure memory
      - r_i_rel: context/relation risk embedding from Triplet-GCN
      - z_i_prime: final modulated embedding written to the map
    """

    instance_id: NodeId
    class_label: str
    bbox: BBox3D
    points_ids: List[int] = field(default_factory=list)

    z_i: Vector = field(default_factory=list)  # [D] semantic embedding
    r_i_retr: Vector = field(default_factory=list)  # [D] retrieval risk embedding
    r_i_rel: Vector = field(default_factory=list)  # [D] relation/context risk embedding
    z_i_prime: Vector = field(default_factory=list)  # [D] final modulated embedding

    gate_g_i: Optional[Vector] = None
    delta_i: Optional[Vector] = None

    risk: RiskScores = field(default_factory=RiskScores)
    source_room: Optional[str] = None
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RelationTriplet:
    """Directed relation edge (subject, predicate, object)."""

    subject_id: NodeId
    object_id: NodeId
    relation_name: str
    relation_id: Optional[int] = None
    confidence: float = 1.0
    edge_type_embedding: Optional[Vector] = None


@dataclass(slots=True)
class GraphDictionary:
    """Scene graph storage used by Triplet-GCN and modulation stages."""

    nodes: Dict[NodeId, Instance3D] = field(default_factory=dict)
    edges: List[RelationTriplet] = field(default_factory=list)
    edge_type_vocab: Dict[EdgeType, int] = field(default_factory=dict)
    adjacency: Dict[NodeId, List[NodeId]] = field(default_factory=dict)


@dataclass(slots=True)
class SceneGraph:
    """Selected scene graph with a convenience summary."""

    scan_id: str
    graph: GraphDictionary

    def summary(self) -> str:
        node_count = len(self.graph.nodes)
        edge_count = len(self.graph.edges)
        relation_types = len(self.graph.edge_type_vocab)
        avg_out_degree = 0.0
        if node_count > 0:
            avg_out_degree = edge_count / node_count
        return (
            f"SceneGraph(scan_id='{self.scan_id}', "
            f"nodes={node_count}, edges={edge_count}, "
            f"relation_types={relation_types}, avg_out_degree={avg_out_degree:.2f})"
        )


@dataclass(slots=True)
class RiskAugmentedObjectDictionary:
    """Object dictionary keyed by instance id."""

    objects: Dict[NodeId, Instance3D] = field(default_factory=dict)


# TypedDict variants for JSON I/O (parser / serializer boundaries)
class BBox3DDict(TypedDict, total=False):
    center: Point3D
    size: Point3D
    yaw: float
    corners_8: List[Point3D]
    volume: float
    aspect_ratio: float


class Instance3DDict(TypedDict, total=False):
    instance_id: int
    class_label: str
    bbox: BBox3DDict
    points_ids: List[int]

    z_i: Vector
    r_i_retr: Vector
    r_i_rel: Vector
    z_i_prime: Vector

    gate_g_i: Vector
    delta_i: Vector

    p_obj: float
    p_ctx: float
    p_f: float
    failure_type: str


class RelationTripletDict(TypedDict, total=False):
    subject_id: int
    object_id: int
    relation_name: str
    relation_id: int
    confidence: float


class GraphDictionaryDict(TypedDict, total=False):
    nodes: Dict[int, Instance3DDict]
    edges: List[RelationTripletDict]
    edge_type_vocab: Dict[str, int]
    adjacency: Dict[int, List[int]]


class RiskAugmentedObjectDictionaryDict(TypedDict, total=False):
    objects: Dict[int, Instance3DDict]
