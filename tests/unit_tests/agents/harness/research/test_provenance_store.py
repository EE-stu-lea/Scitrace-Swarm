from __future__ import annotations

import json

from jiuwenswarm.agents.harness.research.provenance.graph import ScientificEvidenceGraph
from jiuwenswarm.agents.harness.research.provenance.models import (
    Claim,
    EdgeRelation,
    EvidenceEdge,
    ExperimentArtifact,
    LiteratureEvidence,
)
from jiuwenswarm.agents.harness.research.provenance.store import ScientificProvenanceStore


def test_sqlite_and_jsonl_store_round_trip(tmp_path) -> None:
    store = ScientificProvenanceStore(tmp_path)
    evidence = LiteratureEvidence(
        title="Reliable Evidence",
        authors=["Alice"],
        year=2025,
        source_url="https://example.org/evidence",
        summary="Seed evidence",
    )
    artifact = ExperimentArtifact(
        experiment_id="exp-1",
        file_path="outputs/exp-1.json",
        sha256="deadbeef",
        command="python train.py --seed 42",
        seed=42,
        dataset="mockset",
        metric_name="f1",
        metric_value=0.77,
    )
    claim = Claim(
        text="The experiment improves F1.",
        evidence_ids=[evidence.id],
        artifact_ids=[artifact.id],
        section="results",
    )

    for node in (evidence, artifact, claim):
        store.upsert_node(node)
    store.upsert_edge(EvidenceEdge(source_id=claim.id, target_id=evidence.id, relation=EdgeRelation.SUPPORTS))
    store.upsert_edge(EvidenceEdge(source_id=claim.id, target_id=artifact.id, relation=EdgeRelation.TESTS))

    graph = store.load_graph()

    assert graph.get_node(evidence.id) is not None
    assert graph.get_node(artifact.id) is not None
    assert graph.get_node(claim.id) is not None
    assert len(graph.edges) == 2

    audit_lines = store.paths.audit_jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(audit_lines) == 5
    first_event = json.loads(audit_lines[0])
    assert first_event["event_type"] == "upsert_node"


def test_graph_loaded_from_store_retains_support_links(tmp_path) -> None:
    store = ScientificProvenanceStore(tmp_path)
    graph = ScientificEvidenceGraph()
    evidence = LiteratureEvidence(
        title="Tracing Claims",
        authors=["Bob"],
        year=2024,
        source_url="https://example.org/tracing",
    )
    claim = Claim(text="Claim linked to literature.", evidence_ids=[evidence.id])
    graph.add_node(evidence)
    graph.add_node(claim)
    graph.add_edge(EvidenceEdge(source_id=claim.id, target_id=evidence.id, relation=EdgeRelation.SUPPORTS))

    for node in graph.nodes.values():
        store.upsert_node(node)
    for edge in graph.edges.values():
        store.upsert_edge(edge)

    loaded = store.load_graph()

    assert loaded.claim_support_ids(claim) == {evidence.id}
