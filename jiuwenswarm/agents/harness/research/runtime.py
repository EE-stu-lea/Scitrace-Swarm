from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Protocol

from jiuwenswarm.agents.harness.research.claim_compiler import (
    ClaimCompilerError,
    ClaimCompilerInput,
    conservative_claim_text,
    infer_claim_status,
)
from jiuwenswarm.agents.harness.research.team_runtime import (
    DeterministicTeamBackend,
    JiuwenSwarmResearchTeamRuntime,
    OpenJiuwenTeamBackend,
)
from jiuwenswarm.agents.harness.research.provenance.models import (
    ClaimStatus,
    ExperimentArtifact,
    LiteratureEvidence,
)
from jiuwenswarm.symphony.llm import (
    LLMConfig,
    create_model_response_observer,
    extract_message_content,
    get_llm_token_usage_summary,
    llm_usage_context,
    reset_llm_token_usage,
)


class ResearchRuntimeError(RuntimeError):
    pass


class ResearchModelRuntime(Protocol):
    async def complete_json(
        self,
        *,
        stage: str,
        operation: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def complete_text(
        self,
        *,
        stage: str,
        operation: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise NotImplementedError

    def usage_summary(self) -> dict[str, Any]:
        raise NotImplementedError

    def runtime_info(self) -> dict[str, Any]:
        raise NotImplementedError

    def call_count(self) -> int:
        raise NotImplementedError


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _exception_chain(exc: Exception) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "message": str(current),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


@dataclass
class MockResearchModelRuntime:
    """Deterministic model runtime for stable tests."""

    force_metric_drift: bool = False
    force_persistent_gate_failure: bool = False
    call_log_path: str | Path | None = None
    _request_count: int = 0

    def __post_init__(self) -> None:
        self._call_log = Path(self.call_log_path) if self.call_log_path else None

    def _record_call(
        self,
        *,
        stage: str,
        operation: str,
        status: str,
        error: Exception | None = None,
    ) -> None:
        _append_jsonl(
            self._call_log,
            {
                "timestamp": _utc_now_iso(),
                "backend": "mock",
                "provider": "mock",
                "model_name": "mock-research",
                "stage": stage,
                "operation": operation,
                "request_index": self._request_count,
                "status": status,
                "error_chain": _exception_chain(error) if error is not None else [],
            },
        )

    async def complete_json(
        self,
        *,
        stage: str,
        operation: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        _ = system_prompt
        self._request_count += 1
        try:
            if stage == "topic_intake":
                payload = {
                    "research_question": user_prompt.splitlines()[0].strip() or "Research question",
                    "hypothesis": "Evidence-aware context governance improves claim grounding.",
                }
            elif stage == "claim_compiler":
                payload = {
                    "text": "Evidence-aware context governance improves claim grounding for autonomous research agents.",
                    "confidence": 0.83,
                    "section": "results",
                    "compiler_source": "mock",
                }
            elif stage == "experiment_plan":
                payload = {
                    "hypothesis": "Evidence-aware retrieval improves top-1 support selection.",
                    "controls": ["fixed benchmark", "fixed scorer parameters", "deterministic seed"],
                    "metrics": ["citation_support_rate"],
                    "limitations": ["controlled benchmark; external validity is not established"],
                }
            elif stage == "review":
                payload = {
                    "summary": "Adjust the metric values and keep all claim and citation references aligned with the structured appendix.",
                    "severity": "error",
                }
            else:
                raise ResearchRuntimeError(f"unsupported mock JSON stage: {stage}.{operation}")
            self._record_call(stage=stage, operation=operation, status="success")
            return payload
        except Exception as exc:
            self._record_call(stage=stage, operation=operation, status="error", error=exc)
            raise

    async def complete_text(
        self,
        *,
        stage: str,
        operation: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        _ = system_prompt
        self._request_count += 1
        try:
            if stage == "draft":
                payload = json.loads(user_prompt)
                text = _render_mock_draft(
                    payload,
                    drift=self.force_metric_drift or self.force_persistent_gate_failure,
                )
            elif stage == "revision":
                draft = user_prompt.split("# Structured Errors", 1)[0].rstrip()
                text = _replace_metric_value(
                    draft,
                    drift=self.force_persistent_gate_failure or self.force_metric_drift,
                )
            else:
                raise ResearchRuntimeError(f"unsupported mock text stage: {stage}.{operation}")
            self._record_call(stage=stage, operation=operation, status="success")
            return text
        except Exception as exc:
            self._record_call(stage=stage, operation=operation, status="error", error=exc)
            raise

    def usage_summary(self) -> dict[str, Any]:
        return {
            "total": {
                "request_count": self._request_count,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "by_stage": {},
            "by_operation": {},
            "records": [],
        }

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": "mock",
            "provider": "mock",
            "model_name": "mock-research",
            "api_base": "",
        }

    def call_count(self) -> int:
        return self._request_count


class JiuwenSwarmResearchModelRuntime:
    """Reuse JiuwenSwarm's configured model stack for workflow stages."""

    def __init__(self, llm_config: LLMConfig | None = None, *, call_log_path: str | Path | None = None) -> None:
        self.llm_config = llm_config or LLMConfig.from_default_model()
        self._model = None
        self._call_log = Path(call_log_path) if call_log_path else None
        self._request_count = 0
        self._observe_response = create_model_response_observer(self.llm_config)
        reset_llm_token_usage()

    def _record_call(
        self,
        *,
        stage: str,
        operation: str,
        status: str,
        error: Exception | None = None,
    ) -> None:
        info = self.runtime_info()
        _append_jsonl(
            self._call_log,
            {
                "timestamp": _utc_now_iso(),
                "backend": "jiuwenswarm",
                "provider": info["provider"],
                "model_name": info["model_name"],
                "api_base": info["api_base"],
                "stage": stage,
                "operation": operation,
                "request_index": self._request_count,
                "status": status,
                "error_chain": _exception_chain(error) if error is not None else [],
            },
        )

    async def complete_json(
        self,
        *,
        stage: str,
        operation: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        content = await self.complete_text(
            stage=stage,
            operation=operation,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ResearchRuntimeError(f"model returned invalid JSON for {stage}.{operation}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ResearchRuntimeError(f"model must return a JSON object for {stage}.{operation}")
        return payload

    async def complete_text(
        self,
        *,
        stage: str,
        operation: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self._request_count += 1
        if self._model is None:
            try:
                self._model = self.llm_config.create_model()
            except Exception as exc:
                self._record_call(stage=stage, operation=operation, status="error", error=exc)
                raise ResearchRuntimeError(
                    f"failed to create JiuwenSwarm model runtime: {exc}"
                ) from exc
        try:
            with llm_usage_context(stage, operation):
                response = await self._model.invoke(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                )
            self._observe_response(response, stage, operation)
            content = extract_message_content(response)
            if not content:
                raise ResearchRuntimeError(f"empty model content for {stage}.{operation}")
            self._record_call(stage=stage, operation=operation, status="success")
            return content
        except Exception as exc:
            self._record_call(stage=stage, operation=operation, status="error", error=exc)
            raise

    def usage_summary(self) -> dict[str, Any]:
        return get_llm_token_usage_summary()

    def runtime_info(self) -> dict[str, Any]:
        client_config = self.llm_config.model_client_config or {}
        return {
            "backend": "jiuwenswarm",
            "provider": str(client_config.get("client_provider") or ""),
            "model_name": self.llm_config.model,
            "api_base": str(client_config.get("api_base") or "").strip().rstrip("/"),
        }

    def call_count(self) -> int:
        return self._request_count


class ResearchStageHarness:
    """Stage-level harness wrapper for single-flow SciTrace MVP."""

    def __init__(
        self,
        runtime: ResearchModelRuntime,
        *,
        allow_claim_fallback: bool = True,
        max_retries: int = 3,
    ) -> None:
        self.runtime = runtime
        self.allow_claim_fallback = bool(allow_claim_fallback)
        self.claim_fallback_used = False
        self.max_retries = max(1, int(max_retries))
        self._runner: asyncio.Runner | None = None

    def _run_async(self, awaitable):
        if self._runner is None:
            self._runner = asyncio.Runner()
        return self._runner.run(awaitable)

    def _retry_json(
        self,
        *,
        stage: str,
        operation: str,
        system_prompt: str,
        user_prompt: str,
        validator=None,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for _ in range(self.max_retries):
            try:
                payload = self._run_async(
                    self.runtime.complete_json(
                        stage=stage,
                        operation=operation,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                )
                return validator(payload) if validator is not None else payload
            except TimeoutError:
                # Native Team timeouts can leave background work in flight and
                # should not trigger a second long invocation automatically.
                raise
            except Exception as exc:  # noqa: BLE001 - retry transient model-output failure
                last_exc = exc
        raise last_exc  # type: ignore[misc]

    def _retry_text(self, *, stage: str, operation: str, system_prompt: str, user_prompt: str) -> str:
        last_exc: Exception | None = None
        for _ in range(self.max_retries):
            try:
                return self._run_async(
                    self.runtime.complete_text(
                        stage=stage,
                        operation=operation,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                )
            except TimeoutError:
                # A native Team timeout is expensive and may leave background
                # work in flight. Let the caller resume from its durable stage
                # checkpoint instead of immediately starting a second Team.
                raise
            except Exception as exc:  # noqa: BLE001 - retry transient model-output failure
                last_exc = exc
        raise last_exc  # type: ignore[misc]

    def close(self, *, timeout_seconds: float = 10.0) -> None:
        """Close the async runner without hanging a completed CLI workflow.

        Native persistent teams may leave cancellation-sensitive background
        tasks behind. ``asyncio.Runner.close`` waits for those tasks and the
        default executor without a deadline, so perform cleanup on a daemon
        thread and let the CLI's hard-exit boundary handle a rare timeout.
        """
        runner = self._runner
        if runner is None:
            return
        self._runner = None
        cleanup = threading.Thread(target=runner.close, name="scitrace-runner-close", daemon=True)
        cleanup.start()
        cleanup.join(timeout=max(0.1, float(timeout_seconds)))

    def generate_research_brief(self, *, topic: str) -> dict[str, Any]:
        system_prompt = (
            "You are a research planning assistant. Return strict JSON with keys "
            "`research_question` and `hypothesis`. Keep them evidence-aware and conservative."
        )
        user_prompt = topic.strip()
        def validate(payload: dict[str, Any]) -> dict[str, Any]:
            if not str(payload.get("research_question") or "").strip():
                raise ResearchRuntimeError("research_question is empty")
            if not str(payload.get("hypothesis") or "").strip():
                raise ResearchRuntimeError("hypothesis is empty")
            return payload

        return self._retry_json(
            stage="topic_intake",
            operation="research_question",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            validator=validate,
        )

    def design_experiment(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "You are an experiment-design reviewer. Return exactly one JSON object with keys "
            "`hypothesis`, `controls`, `metrics`, and `limitations`. Do not return code and do not "
            "invent results. The executor will run a trusted, fixed script after this review."
        )

        def validate(result: dict[str, Any]) -> dict[str, Any]:
            normalized = dict(result)
            hypothesis = normalized.get("hypothesis")
            if isinstance(hypothesis, dict):
                hypothesis = next(
                    (
                        hypothesis.get(key)
                        for key in ("text", "statement", "hypothesis", "rationale")
                        if str(hypothesis.get(key) or "").strip()
                    ),
                    json.dumps(hypothesis, ensure_ascii=False, sort_keys=True),
                )
                normalized["hypothesis"] = str(hypothesis)
            if not str(normalized.get("hypothesis") or "").strip():
                raise ResearchRuntimeError("experiment plan review hypothesis is empty")
            for key in ("controls", "metrics", "limitations"):
                value = normalized.get(key)
                if isinstance(value, str) and value.strip():
                    normalized[key] = [value.strip()]
                    value = normalized[key]
                elif isinstance(value, dict):
                    # Team members commonly return a structured object such as
                    # {"design": "...", "conditions": [...]}. Preserve that
                    # information as one list entry instead of spending another
                    # model call solely to change the JSON container shape.
                    normalized[key] = [value]
                    value = normalized[key]
                elif isinstance(value, (tuple, set)):
                    normalized[key] = list(value)
                    value = normalized[key]
                if not isinstance(value, list):
                    raise ResearchRuntimeError(f"experiment plan review `{key}` must be a list")
            return normalized

        return self._retry_json(
            stage="experiment_plan",
            operation="design_experiment",
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            validator=validate,
        )

    def compile_claim(
        self,
        compiler_input: ClaimCompilerInput,
    ) -> dict[str, Any]:
        status = infer_claim_status(
            compiler_input.artifacts,
            supported_threshold=compiler_input.status_threshold_supported,
            refuted_threshold=compiler_input.status_threshold_refuted,
        )
        evidence_payload = [item.to_dict() for item in compiler_input.evidence]
        artifact_payload = [item.to_dict() for item in compiler_input.artifacts]
        system_prompt = (
            "You are a scientific claim compiler. Return strict JSON with keys "
            "`text`, `confidence`, and optional `section`. The claim must stay conservative "
            "and must not invent evidence ids or metrics."
        )
        user_prompt = json.dumps(
            {
                "topic": compiler_input.topic,
                "status": status.value,
                "evidence": evidence_payload,
                "artifacts": artifact_payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        def normalize(result: dict[str, Any]) -> dict[str, Any]:
            if str(result.get("text") or "").strip():
                return result
            nested_claim = result.get("claim")
            if isinstance(nested_claim, dict) and str(nested_claim.get("text") or "").strip():
                normalized = dict(nested_claim)
                normalized.setdefault("section", compiler_input.section)
                return normalized
            candidates = result.get("claims")
            if isinstance(candidates, list):
                for candidate in candidates:
                    if isinstance(candidate, dict) and str(candidate.get("text") or "").strip():
                        normalized = dict(candidate)
                        normalized.setdefault("section", compiler_input.section)
                        return normalized
            raise ClaimCompilerError("claim text is empty")

        payload = None
        try:
            payload = self._retry_json(
                stage="claim_compiler",
                operation="compile_claim",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                validator=normalize,
            )
        except Exception:
            if not self.allow_claim_fallback:
                raise
        if not str((payload or {}).get("text") or "").strip():
            if not self.allow_claim_fallback:
                raise ClaimCompilerError("claim text is empty")
            self.claim_fallback_used = True
            return {
                "text": conservative_claim_text(
                    topic=compiler_input.topic,
                    status=status,
                    evidence=compiler_input.evidence,
                    artifacts=compiler_input.artifacts,
                ),
                "confidence": 0.5 if status is ClaimStatus.INCONCLUSIVE else 0.75,
                "section": compiler_input.section,
                "compiler_source": "fallback",
            }
        return payload

    def write_draft(self, *, prompt: str) -> str:
        system_prompt = (
            "You are an academic short-paper writer. Produce an English ICLR-style markdown paper with "
            "a title, Abstract, Introduction, Related Work, Method, Experimental Setup, Results, "
            "Limitations, Conclusion, and References. Treat the provided experiment result as the sole "
            "source of quantitative claims. Respect confidence intervals, claim boundaries, train/test "
            "separation, and any inconclusive status. Describe a benchmark as controlled only when the "
            "artifact says it is controlled. Do not present the orchestration framework as the paper's "
            "scientific contribution. Follow the paper_requirements in the user payload as method constraints. "
            "When the supplied research question or hypothesis describes an untested mechanism, narrow the paper's "
            "question and claim to what the supplied experiment actually tests; retain its provenance identifier "
            "only in the final Structured References JSON block. Do not mention research-question, hypothesis, "
            "claim, or artifact IDs in the visible title, abstract, or body. "
            "Distinguish the exact optimized budget from computation or latency, and distinguish a batch "
            "allocator from an online single-query policy. "
            "The predicted recall scores may be real-valued, but allocated document depths are integers; "
            "never describe the allocated depth as a continuous function. "
            "Compare directly with the closest relevant prior work. Cite at least four relevant verified sources "
            "in the paper body using backtick-delimited evidence IDs, "
            "and include matching bibliography entries. Cite only literature that supports a concrete statement in Related Work or Method; "
            "omit irrelevant retrieved items instead of listing them for provenance. State what was not "
            "measured, including downstream answer quality when only retrieval metrics are provided. "
            "Do not invent bibliography entries or experimental results. "
            "For readable prose and tables, round displayed secondary metrics and confidence intervals "
            "consistently to 4-6 decimal places; keep all exact reported metric values in the final "
            "Structured References JSON block. Do not imply preregistration without a timestamped protocol. "
            "In Results, write a Markdown table with SciFact, NFCorpus, Macro, and FiQA lexical rows, "
            "and columns for candidate recall, fixed-baseline recall, difference, and paired-bootstrap 95% interval. "
            "Use required_results_table_rows as the sole source for the table; state that FiQA is a separate stress test. "
            "Embed the provided result_figure exactly once as a standalone Markdown image using its path, "
            "with a factual alt-text caption using its caption_hint, including the separate panel scales. "
            "Discuss the budget sweep and do not imply that its plotted points "
            "have confidence intervals when none were supplied. Describe the allocator objective, feasible "
            "integer depths, input features, training/test boundary, selected hyperparameters, and baselines "
            "sufficiently for another researcher to reproduce the comparison. Discuss the supplied feature "
            "ablation and negative results without inventing missing measurements. "
            "Preserve every claim ID and structured metric value exactly as provided. Preserve the exact ID of every source you actually cite. "
            "Include a final fenced JSON block under `## Structured References`."
        )
        return self._retry_text(
            stage="draft",
            operation="write_draft",
            system_prompt=system_prompt,
            user_prompt=prompt,
        )

    def review_and_revise(self, *, draft: str, errors: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
        review = self.review_draft(draft=draft, errors=errors)
        revision_system = (
            "You are revising a scientific markdown draft. Fix only the reported scientific, "
            "method-scope, literature-relevance, and formatting errors. Replace off-topic references "
            "only with verified literature supplied in the original draft context; never invent sources. "
            "Preserve the exact claim ids, citation ids, and metric values from the provided structured block. "
            "Keep research-question, hypothesis, claim, and artifact IDs only in that final structured block, "
            "not in visible prose. Document depth is integer-valued even when predicted recall scores are continuous."
        )
        revision_prompt = draft + "\n\n# Structured Errors\n" + json.dumps(errors, ensure_ascii=False, indent=2)
        revised = self._retry_text(
            stage="revision",
            operation="targeted_revision",
            system_prompt=revision_system,
            user_prompt=revision_prompt,
        )
        return review, revised

    def review_draft(self, *, draft: str, errors: list[dict[str, Any]]) -> dict[str, Any]:
        review_system = (
            "You are a scientific reviewer. Return strict JSON with a concise `summary` "
            "and `severity`. Focus only on the structured errors."
        )
        review_payload = json.dumps(
            {"errors": errors, "draft_excerpt": draft[:4000]},
            ensure_ascii=False,
            sort_keys=True,
        )
        review = self._retry_json(
            stage="review",
            operation="review_draft",
            system_prompt=review_system,
            user_prompt=review_payload,
        )
        return review


def _replace_metric_value(text: str, *, drift: bool) -> str:
    if not drift:
        return text
    marker = "## Structured References"
    if marker not in text or "```json" not in text:
        return text
    before, tail = text.split(marker, 1)
    _, after_json = tail.split("```json", 1)
    raw_json, suffix = after_json.split("```", 1)
    payload = json.loads(raw_json.strip())
    reported = payload.get("reported_metrics") or []
    if not reported:
        return text
    metric_item = reported[0]
    try:
        original = float(metric_item.get("claimed_value"))
    except (TypeError, ValueError):
        return text
    updated = round(original + 0.05, 4)
    metric_item["claimed_value"] = updated
    visible_original = str(original)
    visible_updated = str(updated)
    patched_before = before.replace(visible_original, visible_updated)
    return (
        patched_before
        + marker
        + "\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```"
        + suffix
    )


def _render_mock_draft(payload: dict[str, Any], *, drift: bool) -> str:
    topic = str(payload.get("topic") or "").strip()
    research_question = payload.get("research_question") or {}
    evidence = payload.get("evidence") or []
    artifact = payload.get("artifact") or {}
    claims = payload.get("claims") or []
    metric_value = artifact.get("metric_value")
    if drift and isinstance(metric_value, (int, float)):
        metric_value = round(float(metric_value) + 0.05, 4)
    lines = [
        "# SciTrace-Swarm MVP Draft",
        "",
        "## Topic",
        topic,
        "",
        "## Research Question",
        f"- `{research_question.get('id', '')}` {research_question.get('text', '')}",
        "",
        "## Results",
    ]
    for claim in claims:
        lines.append(f"- `{claim['id']}` [{claim['status']}] {claim['text']}")
    lines.extend(["", "## Evidence"])
    for item in evidence:
        authors = ", ".join(item.get("authors", []))
        lines.append(f"- `{item['id']}` {item['title']} ({item.get('year', 'n/a')}), {authors}")
    lines.extend(
        [
            "",
            "## Experiment Result",
            f"- `{artifact.get('id', '')}` {artifact.get('metric_name', '')} = {metric_value}",
            "",
            "## Structured References",
            "```json",
            json.dumps(
                {
                    "citations": [item["id"] for item in evidence],
                    "claims": [{"claim_id": item["id"]} for item in claims],
                    "reported_metrics": [
                        {
                            "artifact_id": artifact.get("id", ""),
                            "metric_name": artifact.get("metric_name", ""),
                            "claimed_value": metric_value,
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_runtime(
    mode: str,
    *,
    config: dict[str, Any] | None = None,
    output_root: str | Path | None = None,
    force_metric_drift: bool = False,
    force_persistent_gate_failure: bool = False,
    call_log_path: str | Path | None = None,
):
    normalized = str(mode or "mock").strip().lower()
    if normalized == "mock":
        return MockResearchModelRuntime(
            force_metric_drift=force_metric_drift,
            force_persistent_gate_failure=force_persistent_gate_failure,
            call_log_path=call_log_path,
        )
    if normalized == "real":
        return JiuwenSwarmResearchModelRuntime(call_log_path=call_log_path)
    if normalized == "mock_team":
        root = Path(output_root) if output_root is not None else None
        if root is None:
            raise ResearchRuntimeError("mock_team runtime requires output_root")
        return JiuwenSwarmResearchTeamRuntime(
            backend=DeterministicTeamBackend(
                config=config or {},
                output_root=root,
                research_id=root.name,
            ),
            research_id=root.name,
            call_log_path=call_log_path,
            output_root=root,
        )
    if normalized == "real_team":
        root = Path(output_root) if output_root is not None else None
        if root is None:
            raise ResearchRuntimeError("real_team runtime requires output_root")
        return JiuwenSwarmResearchTeamRuntime(
            backend=OpenJiuwenTeamBackend(
                config=config or {},
                output_root=root,
                project_dir=str(Path.cwd()),
            ),
            research_id=root.name,
            call_log_path=call_log_path,
            output_root=root,
        )
    raise ResearchRuntimeError(f"unsupported runtime mode: {mode}")


__all__ = [
    "JiuwenSwarmResearchModelRuntime",
    "MockResearchModelRuntime",
    "ResearchModelRuntime",
    "ResearchRuntimeError",
    "ResearchStageHarness",
    "build_runtime",
]
