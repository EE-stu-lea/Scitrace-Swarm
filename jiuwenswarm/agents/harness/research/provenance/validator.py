from __future__ import annotations

from dataclasses import dataclass

from .graph import ScientificEvidenceGraph
from .models import Claim, ExperimentArtifact


@dataclass(frozen=True)
class ScientificValidationError:
    code: str
    message: str
    node_id: str | None = None
    severity: str = "error"


class ScientificValidator:
    """Minimal provenance validation rules for the research workflow."""

    def validate_claim(self, claim: Claim, graph: ScientificEvidenceGraph) -> list[ScientificValidationError]:
        errors: list[ScientificValidationError] = []
        support_ids = graph.claim_support_ids(claim)
        if not support_ids:
            errors.append(
                ScientificValidationError(
                    code="claim_missing_support",
                    message="Claim must reference at least one evidence node or experiment artifact.",
                    node_id=claim.id,
                )
            )

        missing_ids = [
            node_id
            for node_id in list(claim.evidence_ids) + list(claim.artifact_ids)
            if graph.get_node(node_id) is None
        ]
        if missing_ids:
            errors.append(
                ScientificValidationError(
                    code="claim_missing_reference",
                    message=f"Claim references unknown provenance nodes: {', '.join(sorted(missing_ids))}",
                    node_id=claim.id,
                )
            )
        return errors

    def validate_artifact(self, artifact: ExperimentArtifact) -> list[ScientificValidationError]:
        errors: list[ScientificValidationError] = []
        if not artifact.sha256:
            errors.append(
                ScientificValidationError(
                    code="artifact_missing_sha256",
                    message="Experiment artifact must record a SHA-256 checksum.",
                    node_id=artifact.id,
                )
            )
        if not artifact.command:
            errors.append(
                ScientificValidationError(
                    code="artifact_missing_command",
                    message="Experiment artifact must record the execution command.",
                    node_id=artifact.id,
                )
            )
        return errors

    def validate_graph(self, graph: ScientificEvidenceGraph) -> list[ScientificValidationError]:
        errors: list[ScientificValidationError] = []
        for node in graph.nodes.values():
            if isinstance(node, Claim):
                errors.extend(self.validate_claim(node, graph))
            if isinstance(node, ExperimentArtifact):
                errors.extend(self.validate_artifact(node))
        for edge in graph.edges.values():
            if graph.get_node(edge.source_id) is None or graph.get_node(edge.target_id) is None:
                errors.append(
                    ScientificValidationError(
                        code="edge_missing_endpoint",
                        message="Evidence graph edge points to a missing node.",
                        node_id=edge.id,
                    )
                )
        return errors
