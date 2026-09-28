"""Research automation extensions for JiuwenSwarm."""

from .claim_compiler import ClaimCompilerInput, sanitize_claim_payload
from .experiment_executor import SafeExperimentExecutor
from .literature import (
    CrossrefLiteratureProvider,
    FixtureLiteratureProvider,
    OpenAlexLiteratureProvider,
    ResilientLiteratureProvider,
)
from .run_store import ResearchRunStore
from .runtime import ResearchStageHarness, build_runtime
from .team_contract import TEAM_TEMPLATE_ID
from .team_runtime import JiuwenSwarmResearchTeamRuntime

__all__ = [
    "ClaimCompilerInput",
    "CrossrefLiteratureProvider",
    "FixtureLiteratureProvider",
    "JiuwenSwarmResearchTeamRuntime",
    "OpenAlexLiteratureProvider",
    "ResearchRunStore",
    "ResearchStageHarness",
    "ResilientLiteratureProvider",
    "SafeExperimentExecutor",
    "TEAM_TEMPLATE_ID",
    "build_runtime",
    "sanitize_claim_payload",
]
