# SciTrace-Swarm Architecture

SciTrace-Swarm is a native JiuwenSwarm Team workflow with five constrained roles: Research Leader, Literature & Ideation Agent, Experiment Agent, Writer Agent, and Auditor Agent. Every handoff carries the same `research_id`, an allowed write scope, and schema-validated stage output.

The workflow is `topic intake -> live literature retrieval -> experiment design review -> restricted experiment execution -> claim compilation -> paper drafting -> audit -> deterministic consistency gate -> targeted revision`. The Scientific Evidence Graph stores research questions, hypotheses, evidence, plans, artifacts, claims, and revision decisions as stable hashed nodes linked by typed edges. SQLite provides queryable state and JSONL provides an append-only audit trail.

Four optional JiuwenSwarm rails provide lifecycle controls: `ScientificProvenanceRail`, `StageAwareContextRail`, `ScientificConsistencyGateRail`, and `ResourceAuditRail`. They are registered in the framework and disabled by default, preserving compatibility for existing users.
