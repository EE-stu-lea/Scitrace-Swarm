# openJiuwen / JiuwenSwarm Module Calls

The real Team path loads a native `TeamAgentSpec`, enriches it through JiuwenSwarm, and invokes `Runner.run_agent_team_streaming`. A fresh Team session is used per stage because each stage is an independently validated handoff; shared scientific state is persisted in the Evidence Graph.

Model configuration is resolved through JiuwenSwarm's normal default-model configuration. Literature retrieval uses OpenAlex and Crossref providers. Experiment execution uses `SafeExperimentExecutor`, which accepts only approved executables, enforces a timeout, hashes outputs, and requires structured `results.json`. The consistency gate compares every reported metric, claim ID, citation ID, and artifact ID with persisted records before accepting the paper.
