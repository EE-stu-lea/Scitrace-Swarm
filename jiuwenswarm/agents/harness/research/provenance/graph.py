from __future__ import annotations

from dataclasses import dataclass, field

from .models import Claim, EdgeRelation, EvidenceEdge, ExperimentArtifact, ProvenanceNode


@dataclass
class ScientificEvidenceGraph:
    """In-memory evidence graph used by the research workflow."""

    nodes: dict[str, ProvenanceNode] = field(default_factory=dict)
    edges: dict[str, EvidenceEdge] = field(default_factory=dict)

    def add_node(self, node: ProvenanceNode) -> ProvenanceNode:
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: EvidenceEdge) -> EvidenceEdge:
        if edge.source_id not in self.nodes:
            raise KeyError(f"Unknown source node: {edge.source_id}")
        if edge.target_id not in self.nodes:
            raise KeyError(f"Unknown target node: {edge.target_id}")
        self.edges[edge.id] = edge
        return edge

    def get_node(self, node_id: str) -> ProvenanceNode | None:
        return self.nodes.get(node_id)

    def iter_nodes(self, kind: str | None = None) -> list[ProvenanceNode]:
        if kind is None:
            return list(self.nodes.values())
        return [node for node in self.nodes.values() if node.kind.value == kind]

    def iter_edges(self, relation: EdgeRelation | None = None) -> list[EvidenceEdge]:
        if relation is None:
            return list(self.edges.values())
        return [edge for edge in self.edges.values() if edge.relation == relation]

    def outgoing(self, node_id: str, relation: EdgeRelation | None = None) -> list[EvidenceEdge]:
        return [
            edge
            for edge in self.edges.values()
            if edge.source_id == node_id and (relation is None or edge.relation == relation)
        ]

    def incoming(self, node_id: str, relation: EdgeRelation | None = None) -> list[EvidenceEdge]:
        return [
            edge
            for edge in self.edges.values()
            if edge.target_id == node_id and (relation is None or edge.relation == relation)
        ]

    def claim_support_ids(self, claim: Claim) -> set[str]:
        support_ids = set(claim.evidence_ids) | set(claim.artifact_ids)
        for edge in self.outgoing(claim.id):
            if edge.relation in {EdgeRelation.SUPPORTS, EdgeRelation.DERIVED_FROM, EdgeRelation.TESTS}:
                support_ids.add(edge.target_id)
        return support_ids

    def artifact_nodes(self) -> list[ExperimentArtifact]:
        return [
            node
            for node in self.nodes.values()
            if isinstance(node, ExperimentArtifact)
        ]
