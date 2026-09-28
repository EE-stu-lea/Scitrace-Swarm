# openJiuwen Framework Contribution

SciTrace-Swarm adds reusable scientific-research capabilities inside `jiuwenswarm/agents/harness/research` and registers four optional rails through the framework configuration and provider registry. The additions include typed provenance models, SQLite/JSONL storage, stage-aware context selection, live literature providers, restricted experiment execution, a claim compiler, a consistency gate, resource accounting, and native Team stage contracts.

All new rails are disabled by default. Existing JiuwenSwarm workflows therefore retain their prior behavior. Unit and integration tests cover models, storage, rail registration, real-mode fallback rules, Team handoff validation, experiment execution, resume, and end-to-end workflow behavior.

{{PR_SECTION}}
