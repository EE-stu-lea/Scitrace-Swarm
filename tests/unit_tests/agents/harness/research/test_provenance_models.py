from __future__ import annotations

from jiuwenswarm.agents.harness.research.provenance.graph import ScientificEvidenceGraph
from jiuwenswarm.agents.harness.research.provenance.models import (
    Claim,
    ClaimStatus,
    ExperimentArtifact,
    LiteratureEvidence,
)
from jiuwenswarm.agents.harness.research.provenance.validator import ScientificValidator


def test_literature_evidence_stable_id_for_same_payload() -> None:
    first = LiteratureEvidence(
        title="Graph-guided Multi-Agent Research",
        authors=["Alice", "Bob"],
        year=2026,
        source_url="https://example.org/paper",
        doi="10.1000/example",
        summary="A paper about evidence-grounded writing.",
    )
    second = LiteratureEvidence(
        title="Graph-guided Multi-Agent Research",
        authors=["Alice", "Bob"],
        year=2026,
        source_url="https://example.org/paper",
        doi="10.1000/example",
        summary="Different summary should not affect stable identity.",
    )

    assert first.id == second.id
    assert first.source_checksum == second.source_checksum


def test_claim_and_artifact_models_expose_required_fields() -> None:
    artifact = ExperimentArtifact(
        experiment_id="exp-1",
        file_path="results/metrics.json",
        sha256="abc123",
        command="python run.py --seed 7",
        seed=7,
        dataset="cifar10",
        metric_name="accuracy",
        metric_value=0.91,
    )
    claim = Claim(
        text="Our method improves accuracy on CIFAR-10.",
        status=ClaimStatus.SUPPORTED,
        artifact_ids=[artifact.id],
        confidence=0.88,
        section="results",
    )

    assert artifact.metric_name == "accuracy"
    assert artifact.metric_value == 0.91
    assert claim.status is ClaimStatus.SUPPORTED
    assert claim.artifact_ids == (artifact.id,)
    assert claim.section == "results"


def test_validator_rejects_claim_without_support() -> None:
    graph = ScientificEvidenceGraph()
    claim = Claim(text="This claim has no evidence.")
    graph.add_node(claim)

    errors = ScientificValidator().validate_graph(graph)

    assert any(error.code == "claim_missing_support" for error in errors)
