from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


Point3D = Tuple[float, float, float]


@dataclass(slots=True)
class Object3D:
    object_id: int
    label: str
    center: Point3D
    size: Point3D
    yaw: float = 0.0


@dataclass(slots=True)
class Relation:
    subject_id: int
    object_id: int
    relation_id: int
    relation_name: str


@dataclass(slots=True)
class GraphData:
    relations: List[Relation] = field(default_factory=list)
    adjacency: Dict[int, List[int]] = field(default_factory=dict)


@dataclass(slots=True)
class SceneBundle:
    objects: Dict[int, Object3D] = field(default_factory=dict)
    graph: GraphData = field(default_factory=GraphData)
