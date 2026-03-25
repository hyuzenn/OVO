from __future__ import annotations

import torch

from risk_pipeline.core.structures import GraphData, Object3D, Relation, SceneBundle
from risk_pipeline.data.failure_memory import FailurePrototypeMemory
from risk_pipeline.models.failure_retrieval import FailureRetriever
from risk_pipeline.models.graph_encoder import RelationAwareGraphEncoder
from risk_pipeline.models.modulation import GatedResidualModulation


def _dummy_scene() -> SceneBundle:
    objects = {
        1: Object3D(object_id=1, label="chair", center=(0.0, 0.0, 0.0), size=(0.5, 1.0, 0.5), yaw=0.0),
        2: Object3D(object_id=2, label="table", center=(1.0, 0.0, 0.0), size=(1.2, 0.7, 0.8), yaw=0.1),
        3: Object3D(object_id=3, label="cabinet", center=(2.0, 0.0, 0.0), size=(0.8, 1.8, 0.5), yaw=-0.1),
    }
    relations = [
        Relation(subject_id=1, object_id=2, relation_id=0, relation_name="left of"),
        Relation(subject_id=2, object_id=3, relation_id=1, relation_name="in front of"),
    ]
    return SceneBundle(objects=objects, graph=GraphData(relations=relations, adjacency={1: [2], 2: [3], 3: []}))


def _dummy_memory(dim: int) -> FailurePrototypeMemory:
    payload = {
        "prototypes": [
            {"prototype_id": "p0", "embedding": [0.1] * dim, "metadata": {"type": "collision"}},
            {"prototype_id": "p1", "embedding": [0.2] * dim, "metadata": {"type": "stuck"}},
            {"prototype_id": "p2", "embedding": [0.3] * dim, "metadata": {"type": "fall"}},
        ]
    }
    return FailurePrototypeMemory.from_dict(payload)


def test_phase_b_modules_forward() -> None:
    dim = 16
    scene = _dummy_scene()
    graph_encoder = RelationAwareGraphEncoder(hidden_dim=dim)

    z_i = graph_encoder(scene)
    assert z_i.shape == (3, dim)

    memory = _dummy_memory(dim=dim)
    retriever = FailureRetriever(top_k=2)
    r_i_retr = retriever(z_i, memory)
    assert r_i_retr.shape == (3, dim)

    r_i_rel = graph_encoder(scene)
    mod = GatedResidualModulation(dim=dim)
    z_i_prime = mod(z_i=z_i, r_i_rel=r_i_rel, r_i_retr=r_i_retr)

    assert z_i_prime.shape == (3, dim)
    assert torch.isfinite(z_i_prime).all()


def test_gate_starts_small() -> None:
    dim = 8
    z_i = torch.randn(4, dim)
    r_i_rel = torch.randn(4, dim)
    r_i_retr = torch.randn(4, dim)

    mod = GatedResidualModulation(dim=dim, init_gate_bias=-6.0)
    z_i_prime = mod(z_i, r_i_rel, r_i_retr)

    mean_shift = (z_i_prime - z_i).abs().mean().item()
    assert mean_shift < 0.05
