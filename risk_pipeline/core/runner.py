from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from risk_pipeline.core.structures import SceneBundle
from risk_pipeline.data.failure_memory import FailurePrototypeMemory
from risk_pipeline.mapping.mapper import MappingConfig, RiskFeatureMapper
from risk_pipeline.models.failure_retrieval import FailureRetriever
from risk_pipeline.models.graph_encoder import RelationAwareGraphEncoder
from risk_pipeline.models.modulation import GatedResidualModulation


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for the standalone MVP runner."""

    hidden_dim: int = 32
    relation_vocab_size: int = 256
    retrieval_top_k: int = 3
    retrieval_temperature: float = 0.2
    modulation_hidden_dim: int | None = None
    modulation_init_gate_bias: float = -4.0
    voxel_size: float = 0.5


class BaseNodeRepresentationBuilder(nn.Module):
    """Build base node representations z_i from object geometry + label hash."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _label_scalar(label: str) -> float:
        value = sum(ord(ch) for ch in label) % 997
        return (value / 498.5) - 1.0

    def forward(self, scene_graph: SceneBundle) -> tuple[torch.Tensor, list[int]]:
        node_order = sorted(scene_graph.objects.keys())
        if not node_order:
            return torch.zeros((0, self.hidden_dim), dtype=torch.float32), []

        rows = []
        for object_id in node_order:
            obj = scene_graph.objects[object_id]
            rows.append(
                [
                    float(obj.center[0]),
                    float(obj.center[1]),
                    float(obj.center[2]),
                    float(obj.size[0]),
                    float(obj.size[1]),
                    float(obj.size[2]),
                    float(obj.yaw),
                    float(self._label_scalar(obj.label)),
                ]
            )

        features = torch.tensor(rows, dtype=torch.float32)
        return self.proj(features), node_order


class RiskPipelineRunner:
    """Standalone MVP execution path.

    Order:
      1) load scene graph (outside this class)
      2) build base node reps z_i
      3) graph encode -> r_i_rel
      4) retrieval -> r_i_retr
      5) modulation -> z_i'
      6) mapping integrate
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

        self.base_builder = BaseNodeRepresentationBuilder(hidden_dim=self.config.hidden_dim)
        self.graph_encoder = RelationAwareGraphEncoder(
            hidden_dim=self.config.hidden_dim,
            relation_vocab_size=self.config.relation_vocab_size,
        )
        self.retriever = FailureRetriever(
            top_k=self.config.retrieval_top_k,
            temperature=self.config.retrieval_temperature,
        )
        self.modulation = GatedResidualModulation(
            dim=self.config.hidden_dim,
            hidden_dim=self.config.modulation_hidden_dim,
            init_gate_bias=self.config.modulation_init_gate_bias,
        )
        self.mapper = RiskFeatureMapper(MappingConfig(voxel_size=self.config.voxel_size))
        self.mapper.initialize_map(feature_dim=self.config.hidden_dim)

    def run(
        self,
        scene_graph: SceneBundle,
        memory: FailurePrototypeMemory,
        *,
        T_t: np.ndarray | torch.Tensor | None = None,
    ) -> dict:
        # 2) z_i
        z_i, node_order = self.base_builder(scene_graph)

        # 3) r_i_rel
        r_i_rel = self.graph_encoder(scene_graph)

        # 4) r_i_retr
        r_i_retr = self.retriever(z_i, memory)

        # 5) z_i'
        z_i_prime = self.modulation(z_i=z_i, r_i_rel=r_i_rel, r_i_retr=r_i_retr)

        # 6) mapping integrate
        positions = torch.tensor(
            [scene_graph.objects[node_id].center for node_id in node_order],
            dtype=torch.float32,
        )
        if T_t is None:
            T_t = np.eye(4, dtype=np.float64)
        integrated_voxels = self.mapper.integrate(
            z_i_prime=z_i_prime,
            bbox_or_position=positions,
            T_t=T_t,
        )

        return {
            "node_order": node_order,
            "z_i": z_i,
            "r_i_rel": r_i_rel,
            "r_i_retr": r_i_retr,
            "z_i_prime": z_i_prime,
            "integrated_voxels": integrated_voxels,
            "map_state": self.mapper.export_state(),
        }
