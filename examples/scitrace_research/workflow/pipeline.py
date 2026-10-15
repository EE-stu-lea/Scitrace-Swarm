from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from jiuwenswarm.agents.harness.research.citation_expansion import (
    expansion_work_key,
    run_citation_expansion,
)
from jiuwenswarm.agents.harness.research.claim_compiler import (
    ClaimCompilerInput,
    sanitize_claim_payload,
)
from jiuwenswarm.agents.harness.research.experiment_executor import (
    ExperimentExecutorError,
    SafeExperimentExecutor,
)
from jiuwenswarm.agents.harness.research.literature import (
    CrossrefLiteratureProvider,
    FixtureLiteratureProvider,
    OpenAlexLiteratureProvider,
    ResilientLiteratureProvider,
)
from jiuwenswarm.agents.harness.research.model_diagnostics import (
    sanitize_text,
    write_model_diagnostics,
)
from jiuwenswarm.agents.harness.research.provenance.models import (
    EdgeRelation,
    EvidenceEdge,
    ExperimentArtifact,
    ExperimentPlan,
    Hypothesis,
    LiteratureEvidence,
    ResearchQuestion,
    RevisionDecision,
    node_from_dict,
)
from jiuwenswarm.agents.harness.research.reporting.resource_reporter import (
    ResourceAuditSnapshot,
    ResourceReporter,
)
from jiuwenswarm.agents.harness.research.rails import (
    ScientificConsistencyGateRail,
    ScientificProvenanceRail,
    StageAwareContextRail,
)
from jiuwenswarm.agents.harness.research.run_store import ResearchRunStore, utc_now_iso
from jiuwenswarm.agents.harness.research.runtime import ResearchStageHarness, build_runtime

from .paper_quality import audit_draft, feature_ablation_rows, select_literature, title_covers_groups
from .method_contract import ROUTE_B2_METHOD_CONTRACT
from .result_figures import write_budget_curve
from .budget_effect_figure import render_budget_effect_figure, FIGURE_PATH as EFFECT_FIGURE_PATH
from .result_rows import _primary_results_table_rows, _adaptive_baseline_rows, _train_subsampling_rows
from .paper_narrative import narrative_context
from .method_overview import render_method_overview
from .query_diagnostics import build_query_diagnostics, render_query_diagnostics


class WorkflowInterrupted(RuntimeError):
    pass


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    loaded = yaml.load(Path(path).read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def dump_yaml_config(path: str | Path, payload: dict[str, Any]) -> None:
    from io import StringIO

    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.default_flow_style = False
    buffer = StringIO()
    yaml.dump(payload, buffer)
    Path(path).write_text(buffer.getvalue(), encoding="utf-8")


def run_pipeline(
    *,
    topic: str,
    config: dict[str, Any],
    output_dir: str | Path,
    mode: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    strict_real: bool = False,
    no_mock_fallback: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = utc_now_iso()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_root / "evidence"
    experiments_dir = output_root / "experiments"
    model_calls_path = output_root / "model_calls.jsonl"
    diagnostics_path = output_root / "model_diagnostics.json"
    manifest_path = output_root / "run_manifest.json"
    previous_manifest = {}
    previous_runtime_seconds = 0.0
    previous_resource_usage: dict[str, Any] = {}
    if resume and manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_resource_usage = (
                json.loads((output_root / "resource_usage.json").read_text(encoding="utf-8"))
                if (output_root / "resource_usage.json").exists() else {}
            )
            previous_runtime_seconds = float(previous_resource_usage.get("total_runtime_seconds", 0.0))
        except (OSError, ValueError, TypeError):
            previous_manifest = {}
            previous_runtime_seconds = 0.0
            previous_resource_usage = {}
    team_manifest_path = output_root / "team_manifest.json"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    runtime_mode = str(mode or config.get("mode") or "mock").strip().lower()
    real_mode = runtime_mode in {"real", "real_team"}
    team_mode = runtime_mode in {"mock_team", "real_team"}
    strict_real = bool(strict_real or config.get("strict_real", False))
    no_mock_fallback = bool(no_mock_fallback or config.get("no_mock_fallback", False) or strict_real)
    debug = _config_section(config, "debug")
    run_store = ResearchRunStore(output_root)
    run_config = dict(config)
    run_config["topic"] = topic
    run_config["mode"] = runtime_mode
    run_config["strict_real"] = strict_real
    run_config["no_mock_fallback"] = no_mock_fallback
    dump_yaml_config(output_root / "run_config.yaml", run_config)
    run_store.set_metadata("topic", topic)
    run_store.set_metadata("mode", runtime_mode)
    run_store.set_metadata("status", "running")
    run_store.set_metadata("strict_real", strict_real)
    run_store.set_metadata("no_mock_fallback", no_mock_fallback)
    manifest = {
        "run_id": output_root.name,
        "mode": runtime_mode,
        "strict_real": strict_real,
        "provider": "",
        "model_name": "",
        "literature_provider": "",
        "mock_fallback_used": False,
        "fixture_used": False,
        "model_call_count": 0,
        "retrieval_call_count": 0,
        "paper_quality_enabled": bool(_config_section(config, "paper_quality").get("enabled", False)),
        "start_time": previous_manifest.get("start_time") or started_at,
        "end_time": "",
        "status": "running",
        "failure_stage": "",
    }
    _write_json(manifest_path, manifest)
    if team_mode:
        team_config = _config_section(config, "team")
        _write_json(
            team_manifest_path,
            {
                "research_id": output_root.name,
                "mode": runtime_mode,
                "template_id": str(team_config.get("template_id") or "scitrace_phase4"),
                "session_id": f"scitrace-team-{output_root.name}",
                "shared_run_root": str(output_root),
                "direct_specialist_stages": list(team_config.get("direct_specialist_stages") or []),
                "execution_note": (
                    "Configured specialist stages use one native openJiuwen model request per "
                    "attempt; other model stages use native Team streaming."
                ),
                "roles": [
                    "research_leader",
                    "literature_ideation_agent",
                    "experiment_agent",
                    "writer_agent",
                    "auditor_agent",
                ],
            },
        )

    if dry_run:
        result = {
            "status": "dry_run",
            "topic": topic,
            "mode": runtime_mode,
            "output_dir": str(output_root),
            "stages": [
                "topic_intake",
                "literature_research",
                "experiment_plan",
                "experiment_execute",
                "claim_compile",
                "draft_generate",
                "consistency_loop",
            ],
        }
        run_store.set_metadata("status", "dry_run")
        run_store.set_stage("dry_run", status="completed", payload=result)
        manifest["status"] = "dry_run"
        manifest["end_time"] = utc_now_iso()
        _write_json(manifest_path, manifest)
        return result

    try:
        diagnostics_payload = None
        if real_mode:
            diagnostics_payload = write_model_diagnostics(diagnostics_path)
            _append_diagnostics_model_calls(model_calls_path, diagnostics_payload)
            resolved = diagnostics_payload.get("resolved_model") or {}
            manifest["provider"] = str(resolved.get("provider") or "")
            manifest["model_name"] = str(resolved.get("model_name") or "")
            _write_json(manifest_path, manifest)
            diagnostic_error = _model_diagnostics_error(diagnostics_payload)
            if diagnostic_error:
                raise RuntimeError(diagnostic_error)

        runtime = build_runtime(
            runtime_mode,
            config=config,
            output_root=output_root,
            force_metric_drift=bool(_config_section(config, "writer").get("force_metric_drift", False)),
            force_persistent_gate_failure=bool(debug.get("force_persistent_gate_failure", False)),
            call_log_path=model_calls_path,
        )
        runtime_info = runtime.runtime_info()
        manifest["provider"] = str(runtime_info.get("provider") or "")
        manifest["model_name"] = str(runtime_info.get("model_name") or "")
        _write_json(manifest_path, manifest)
        harness = ResearchStageHarness(
            runtime,
            allow_claim_fallback=not (real_mode and no_mock_fallback),
            max_retries=int(_config_section(config, "team").get("max_stage_attempts", 3) or 3),
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_stage"] = "runtime_init"
        manifest["model_call_count"] = _count_model_call_records(model_calls_path, include_diagnostics=False)
        manifest["end_time"] = utc_now_iso()
        _write_json(manifest_path, manifest)
        run_store.set_metadata("status", "failed")
        run_store.set_metadata("failure_stage", "runtime_init")
        raise
    provenance = ScientificProvenanceRail(enabled=True, output_root=output_root, capture_tool_audits=False)
    stage_context = StageAwareContextRail(
        enabled=True,
        output_root=output_root,
        token_budget=int(_config_section(config, "context").get("token_budget", 1800) or 1800),
        include_conflicts=bool(_config_section(config, "context").get("include_conflicts", True)),
    )
    gate = ScientificConsistencyGateRail(enabled=True, output_root=output_root, block_on_error=False)
    team_metadata: dict[str, Any] = {}
    team_manifest_path = output_root / "team_manifest.json"
    if team_manifest_path.exists():
        try:
            team_metadata = json.loads(team_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            team_metadata = {}
    roles = team_metadata.get("roles") or []
    resource_snapshot = ResourceAuditSnapshot(
        session_id=str(team_metadata.get("session_id") or manifest["run_id"]),
        role="native_team" if runtime_mode in {"mock_team", "real_team"} else "single_agent",
        member_name=", ".join(str(role) for role in roles) if roles else "research_workflow",
    )
    resource_snapshot.extra["workflow"] = "scitrace_mvp"
    if resume:
        resource_snapshot.extra["resumed_run"] = True
        resource_snapshot.total_runtime_seconds = previous_runtime_seconds
        _restore_prior_tool_usage(resource_snapshot, previous_resource_usage)
    current_stage = ""

    try:
        current_stage = "topic_intake"
        research_question_payload = _resume_or_compute(
            run_store,
            "topic_intake",
            resume=resume,
            compute=lambda: _run_topic_intake(topic, harness, provenance, config=config),
        )
        _enforce_runtime_budget(config, runtime, stage=current_stage, call_log_path=model_calls_path)
        _maybe_interrupt(debug, current_stage)

        current_stage = "literature_research"
        literature_payload = _resume_or_compute(
            run_store,
            "literature_research",
            resume=resume,
            compute=lambda: _run_literature_research(
                topic=topic,
                config=config,
                evidence_dir=evidence_dir,
                provenance=provenance,
                resource_snapshot=resource_snapshot,
                strict_real=real_mode and strict_real,
                no_mock_fallback=real_mode and no_mock_fallback,
            ),
        )
        manifest["literature_provider"] = str(literature_payload.get("requested_provider") or "")
        manifest["fixture_used"] = bool(literature_payload.get("fixture_used", False))
        manifest["mock_fallback_used"] = bool(manifest["mock_fallback_used"] or literature_payload.get("fallback_used", False))
        manifest["retrieval_call_count"] = int(literature_payload.get("retrieval_call_count", 0) or 0)
        _write_json(manifest_path, manifest)
        _maybe_interrupt(debug, current_stage)

        current_stage = "experiment_plan"
        plan_payload = _resume_or_compute(
            run_store,
            "experiment_plan",
            resume=resume,
            compute=lambda: _run_experiment_plan(
                topic=topic,
                config=config,
                harness=harness,
                experiments_dir=experiments_dir,
                provenance=provenance,
                research_question_payload=research_question_payload,
                literature_payload=literature_payload,
            ),
        )
        _maybe_interrupt(debug, current_stage)

        current_stage = "experiment_execute"
        execution_payload = _resume_or_compute(
            run_store,
            "experiment_execute",
            resume=resume,
            compute=lambda: _run_experiment_execute(
                config=config,
                experiments_dir=experiments_dir,
                plan_payload=plan_payload,
                provenance=provenance,
                resource_snapshot=resource_snapshot,
            ),
        )
        _maybe_interrupt(debug, current_stage)

        current_stage = "claim_compile"
        claim_payload = _resume_or_compute(
            run_store,
            "claim_compile",
            resume=resume,
            compute=lambda: _run_claim_compile(
                topic=topic,
                config=config,
                harness=harness,
                provenance=provenance,
                literature_payload=literature_payload,
                execution_payload=execution_payload,
            ),
        )
        manifest["mock_fallback_used"] = bool(manifest["mock_fallback_used"] or harness.claim_fallback_used)
        _write_json(output_root / "claims.json", claim_payload)
        _enforce_runtime_budget(config, runtime, stage=current_stage, call_log_path=model_calls_path)
        _write_json(manifest_path, manifest)
        _maybe_interrupt(debug, current_stage)

        current_stage = "draft_generate"
        draft_payload = _resume_or_compute(
            run_store,
            "draft_generate",
            resume=resume,
            compute=lambda: _run_draft_generate(
                topic=topic,
                config=config,
                harness=harness,
                stage_context=stage_context,
                output_root=output_root,
                research_question_payload=research_question_payload,
                literature_payload=literature_payload,
                execution_payload=execution_payload,
                claim_payload=claim_payload,
            ),
        )
        draft_path = output_root / "draft.md"
        draft_path.write_text(str(draft_payload["draft"]), encoding="utf-8")
        _enforce_runtime_budget(config, runtime, stage=current_stage, call_log_path=model_calls_path)
        _maybe_interrupt(debug, current_stage)

        current_stage = "consistency_loop"
        validation_payload = _resume_or_compute(
            run_store,
            "consistency_loop",
            resume=resume,
            compute=lambda: _run_consistency_loop(
                config=config,
                harness=harness,
                gate=gate,
                provenance=provenance,
                output_root=output_root,
                literature_payload=literature_payload,
                claim_payload=claim_payload,
                execution_payload=execution_payload,
                initial_draft=str(draft_payload["draft"]),
            ),
        )
        draft_path.write_text(str(validation_payload["draft"]), encoding="utf-8")
        draft_sha256 = hashlib.sha256(draft_path.read_bytes()).hexdigest()
        validation_payload["validation_report"]["draft_sha256"] = draft_sha256
        _write_json(output_root / "validation_report.json", validation_payload["validation_report"])
        budget_overrun = ""
        try:
            _enforce_runtime_budget(config, runtime, stage=current_stage, call_log_path=model_calls_path)
        except RuntimeError as exc:
            # The consistency checkpoint is already durable and no further
            # model calls occur below. Preserve a valid, gate-passing result
            # and report the overrun instead of converting completed work into
            # a failed run that repeatedly re-enters resume/finalization.
            if not validation_payload["validation_report"]["ok"]:
                raise
            budget_overrun = sanitize_text(exc)
            manifest["budget_overrun"] = budget_overrun
            run_store.append_audit_event(
                "budget_overrun_after_validation",
                {"stage": current_stage, "message": budget_overrun},
            )

        usage_summary = _durable_usage_summary(model_calls_path, runtime)
        resource_snapshot.total_runtime_seconds = previous_runtime_seconds + (time.perf_counter() - started)
        workflow_model_calls = _count_model_call_records(model_calls_path, include_diagnostics=False)
        diagnostic_model_calls = _count_model_call_records(model_calls_path, include_diagnostics=True) - workflow_model_calls
        _fill_resource_snapshot(
            resource_snapshot,
            usage_summary=usage_summary,
            topic=topic,
            claim_count=len(claim_payload["claims"]),
            evidence_count=len(literature_payload["evidence"]),
            artifact_count=1,
            started=started,
            workflow_model_calls=workflow_model_calls,
            diagnostic_model_calls=max(0, diagnostic_model_calls),
            retries=_count_failed_model_call_records(model_calls_path),
            model_name=str(runtime_info.get("model_name") or ""),
        )
        # A resumed process may have no prior resource_usage.json when the
        # first process was interrupted in a stage. Report the traceable
        # first-start-to-finalization wall time rather than just this process.
        run_start = datetime.fromisoformat(str(manifest["start_time"]))
        elapsed_wall = max(0.0, (datetime.now(timezone.utc) - run_start).total_seconds())
        resource_snapshot.total_runtime_seconds = max(
            resource_snapshot.total_runtime_seconds, elapsed_wall
        )
        resource_snapshot.extra["runtime_accounting"] = (
            "elapsed wall time from run_manifest.start_time; includes any resume gap"
            if resume else "elapsed wall time from run_manifest.start_time"
        )
        if budget_overrun:
            resource_snapshot.extra["budget_overrun"] = budget_overrun
        resource_reporter = ResourceReporter(
            output_root,
            json_filename="resource_usage.json",
            markdown_filename="resource_report.md",
            nest_by_session=False,
        )
        usage_json_path, usage_md_path = resource_reporter.write_snapshot(resource_snapshot)

        status = "success" if validation_payload["validation_report"]["ok"] else "failed"
        manifest["status"] = status
        manifest["failure_stage"] = "" if status == "success" else current_stage
        manifest["model_call_count"] = workflow_model_calls
        manifest["draft_sha256"] = draft_sha256
        manifest["end_time"] = utc_now_iso()
        _write_json(manifest_path, manifest)

        result = {
            "status": status,
            "topic": topic,
            "mode": runtime_mode,
            "strict_real": strict_real,
            "output_dir": str(output_root),
            "draft_path": str(draft_path),
            "claims_path": str(output_root / "claims.json"),
            "validation_report_path": str(output_root / "validation_report.json"),
            "resource_usage_path": str(usage_json_path),
            "resource_report_path": str(usage_md_path),
            "research_db_path": str(run_store.sqlite_path),
            "audit_path": str(run_store.audit_path),
            "manifest_path": str(manifest_path),
            "model_diagnostics_path": str(diagnostics_path) if diagnostics_path.exists() else "",
            "model_calls_path": str(model_calls_path),
            "budget_overrun": budget_overrun,
        }
        run_store.set_metadata("status", status)
        run_store.set_metadata("result", result)
        run_store.append_audit_event("workflow_complete", result)
        return result
    except Exception as exc:
        if current_stage:
            # A post-stage guard (for example the resource budget check) can
            # fail after _resume_or_compute has durably stored a valid result.
            # Preserve that completed checkpoint so --resume does not repeat an
            # expensive real-model call. Failures raised while computing the
            # stage still replace its running checkpoint with a failed record.
            checkpoint = run_store.get_stage(current_stage)
            if checkpoint is None or checkpoint.status != "completed":
                run_store.set_stage(
                    current_stage,
                    status="failed",
                    payload={"error": sanitize_text(exc)},
                )
        # A failed run still needs a traceable resource report. Use durable
        # model-call records so a resumed process includes prior paid usage;
        # never let telemetry replace the stage's original exception.
        try:
            usage_summary = _durable_usage_summary(model_calls_path, runtime)
            workflow_calls = _count_model_call_records(model_calls_path, include_diagnostics=False)
            diagnostic_calls = _count_model_call_records(model_calls_path, include_diagnostics=True) - workflow_calls
            _fill_resource_snapshot(
                resource_snapshot, usage_summary=usage_summary, topic=topic,
                claim_count=0, evidence_count=0, artifact_count=0,
                started=started, workflow_model_calls=workflow_calls,
                diagnostic_model_calls=max(0, diagnostic_calls),
                retries=_count_failed_model_call_records(model_calls_path),
                model_name=str(runtime_info.get("model_name") or ""),
            )
            resource_snapshot.total_runtime_seconds = previous_runtime_seconds + (time.perf_counter() - started)
            for incomplete_count in ("claim_count", "evidence_count", "artifact_count"):
                resource_snapshot.extra.pop(incomplete_count, None)
            resource_snapshot.extra.update({
                "run_status": "failed", "failure_stage": current_stage,
                "partial_report": True,
            })
            ResourceReporter(
                output_root, json_filename="resource_usage.json",
                markdown_filename="resource_report.md", nest_by_session=False,
            ).write_snapshot(resource_snapshot)
        except Exception as report_error:
            try:
                run_store.append_audit_event("resource_report_write_failed", {
                    "error_type": type(report_error).__name__,
                })
            except OSError:
                pass
        manifest["status"] = "failed"
        manifest["failure_stage"] = current_stage
        manifest["mock_fallback_used"] = bool(manifest["mock_fallback_used"] or harness.claim_fallback_used)
        manifest["model_call_count"] = _count_model_call_records(model_calls_path, include_diagnostics=False)
        manifest["end_time"] = utc_now_iso()
        _write_json(manifest_path, manifest)
        run_store.set_metadata("status", "failed")
        run_store.set_metadata("failure_stage", current_stage)
        raise
    finally:
        harness.close()


def _run_topic_intake(
    topic: str,
    harness: ResearchStageHarness,
    provenance: ScientificProvenanceRail,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scope = ({"implementation_contract": ROUTE_B2_METHOD_CONTRACT, **narrative_context()}
             if _config_section(config or {}, "paper_quality").get("require_research_narrative") else None)
    brief = (harness.generate_research_brief(topic=topic, research_scope=scope) if scope else
             harness.generate_research_brief(topic=topic))
    research_question = ResearchQuestion(text=str(brief.get("research_question") or topic).strip())
    hypothesis = Hypothesis(text=str(brief.get("hypothesis") or "").strip() or topic)
    provenance.record_node(research_question, session_id="")
    provenance.record_node(hypothesis, session_id="")
    provenance.record_edge(
        EvidenceEdge(source_id=hypothesis.id, target_id=research_question.id, relation=EdgeRelation.DERIVED_FROM),
        session_id="",
    )
    return {
        "research_question": research_question.to_dict(),
        "hypothesis": hypothesis.to_dict(),
    }


def _append_literature_event(evidence_dir: Path, payload: dict[str, Any]) -> None:
    path = evidence_dir / "literature_events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": utc_now_iso(), **payload}, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _run_literature_research(
    *,
    topic: str,
    config: dict[str, Any],
    evidence_dir: Path,
    provenance: ScientificProvenanceRail,
    resource_snapshot: ResourceAuditSnapshot,
    strict_real: bool,
    no_mock_fallback: bool,
) -> dict[str, Any]:
    literature_cfg = _config_section(config, "literature")
    fixture_path = literature_cfg.get("fixture_path") or (
        Path(__file__).resolve().parents[1] / "fixtures" / "literature_fixture.json"
    )
    fallback_provider = FixtureLiteratureProvider(fixture_path, cache_root=evidence_dir / "raw")
    provider_name = str(literature_cfg.get("provider") or "mock").strip().lower()
    if strict_real and provider_name in {"mock", "fixture"}:
        raise RuntimeError("strict-real requires a real literature provider; fixture/mock providers are disabled")
    provider = _build_literature_provider(
        literature_cfg,
        evidence_dir=evidence_dir,
        fallback_provider=fallback_provider,
        allow_fallback=not no_mock_fallback,
    )
    configured_queries = literature_cfg.get("search_queries")
    search_queries = (
        [str(query).strip() for query in configured_queries if str(query).strip()]
        if isinstance(configured_queries, list) else [topic]
    )
    if not search_queries:
        search_queries = [topic]
    candidates = []
    for query in search_queries:
        started = time.perf_counter()
        try:
            candidates.extend(provider.search(query))
        except Exception:
            resource_snapshot.tool_failures += 1
            raise
        elapsed = max(0.0, time.perf_counter() - started)
        resource_snapshot.tool_calls += 1
        resource_snapshot.tools_by_name["literature_search"] = resource_snapshot.tools_by_name.get("literature_search", 0) + 1
        resource_snapshot.total_tool_seconds += elapsed
        resource_snapshot.tool_seconds_by_name["literature_search"] = (
            resource_snapshot.tool_seconds_by_name.get("literature_search", 0.0) + elapsed
        )
    fixture_used = any(result.provider == "fixture" for result in candidates)
    fallback_used = fixture_used and provider_name not in {"fixture", "mock"}
    if strict_real and (fixture_used or fallback_used):
        raise RuntimeError("strict-real forbids literature fixture/mock fallback")
    results, selection_audit = select_literature(
        candidates,
        focus_terms=[str(term) for term in literature_cfg.get("focus_terms", [])],
        minimum_focus_terms=int(literature_cfg.get("minimum_focus_terms", 0) or 0),
        max_results=int(literature_cfg.get("max_results", 5) or 5),
        priority_title_term_groups=literature_cfg.get("priority_title_term_groups") or [],
        minimum_priority_results=int(literature_cfg.get("minimum_priority_results", 0) or 0),
        reserved_title_phrases=literature_cfg.get("reserved_title_phrases") or [],
    )
    expansion_outcome = None
    if bool(literature_cfg.get("enable_citation_expansion", False)):
        # One-hop citation-graph expansion over real OpenAlex calls. The runner
        # degrades any failure to `skipped`, so base results always survive.
        started = time.perf_counter()
        expansion_outcome = run_citation_expansion(
            results,
            focus_terms=[str(term) for term in literature_cfg.get("focus_terms", [])],
            minimum_focus_terms=int(literature_cfg.get("minimum_focus_terms", 0) or 0),
            max_expansion_results=int(literature_cfg.get("max_expansion_results", 20) or 20),
            timeout_seconds=float(literature_cfg.get("timeout_seconds", 20) or 20),
            retries=int(literature_cfg.get("retries", 2) or 2),
        )
        elapsed = max(0.0, time.perf_counter() - started)
        resource_snapshot.tool_calls += expansion_outcome.api_call_count
        resource_snapshot.tools_by_name["literature_citation_expansion"] = (
            resource_snapshot.tools_by_name.get("literature_citation_expansion", 0)
            + expansion_outcome.api_call_count
        )
        resource_snapshot.total_tool_seconds += elapsed
        resource_snapshot.tool_seconds_by_name["literature_citation_expansion"] = (
            resource_snapshot.tool_seconds_by_name.get("literature_citation_expansion", 0.0) + elapsed
        )
        _append_literature_event(evidence_dir, expansion_outcome.to_audit_event())
        results = results + list(expansion_outcome.kept)
    min_results = int(literature_cfg.get("min_relevant_results", 0) or 0)
    selection_path = evidence_dir / "literature_selection.json"
    _write_json(selection_path, {
        "topic": topic,
        "search_queries": search_queries,
        "focus_terms": literature_cfg.get("focus_terms", []),
        "minimum_focus_terms": int(literature_cfg.get("minimum_focus_terms", 0) or 0),
        "priority_title_term_groups": literature_cfg.get("priority_title_term_groups", []),
        "candidates": selection_audit,
        "selected_dois": [result.doi for result in results],
        **({"citation_expansion": expansion_outcome.to_audit_event()}
           if expansion_outcome is not None else {}),
    })
    if len(results) < min_results:
        raise RuntimeError(
            f"Only {len(results)} relevant literature sources found; require {min_results}. "
            f"Inspect {selection_path} and rerun live retrieval."
        )
    priority_groups = literature_cfg.get("priority_title_term_groups") or []
    minimum_priority = int(literature_cfg.get("minimum_priority_results", 0) or 0)
    priority_count = sum(title_covers_groups(item.title, priority_groups) for item in results)
    if priority_count < minimum_priority:
        raise RuntimeError(
            f"Only {priority_count} directly comparable literature sources found; require "
            f"{minimum_priority}. Inspect {selection_path} and rerun live retrieval."
        )

    expansion_seed_by_key = dict(expansion_outcome.provenance) if expansion_outcome else {}
    evidence_nodes: list[dict[str, Any]] = []
    for result in results:
        metadata = {
            "provider": result.provider,
            "verified_doi": result.verified_doi,
            "verified_url": result.verified_url,
        }
        expansion_seed = expansion_seed_by_key.get(expansion_work_key(result))
        if expansion_seed:
            metadata["expansion_seed_openalex_id"] = expansion_seed
        node = LiteratureEvidence(
            title=result.title,
            authors=result.authors,
            year=result.year,
            source_url=result.source_url,
            doi=result.doi,
            summary=result.summary,
            reliability_score=result.reliability_score,
            metadata=metadata,
        )
        provenance.record_node(node, session_id="")
        evidence_nodes.append(node.to_dict())
    _write_json(evidence_dir / "literature.json", {"topic": topic, "evidence": evidence_nodes})
    # Expansion requests are real retrieval API calls and share the same
    # strict-real accounting as the keyword search queries.
    retrieval_call_count = len(search_queries) + (
        expansion_outcome.api_call_count if expansion_outcome else 0
    )
    payload = {
        "evidence": evidence_nodes,
        "requested_provider": provider_name,
        "resolved_providers": sorted({result.provider for result in results}),
        "fixture_used": fixture_used,
        "fallback_used": fallback_used,
        "retrieval_call_count": retrieval_call_count,
    }
    if expansion_outcome is not None:
        payload["citation_expansion"] = {
            "status": expansion_outcome.status,
            "kept_count": len(expansion_outcome.kept),
            "api_call_count": expansion_outcome.api_call_count,
        }
    return payload


def _run_experiment_plan(
    *,
    topic: str,
    config: dict[str, Any],
    harness: ResearchStageHarness,
    experiments_dir: Path,
    provenance: ScientificProvenanceRail,
    research_question_payload: dict[str, Any],
    literature_payload: dict[str, Any],
) -> dict[str, Any]:
    experiment_cfg = _config_section(config, "experiment")
    script_path = experiments_dir / "run_experiment.py"
    seed = int(experiment_cfg.get("seed", 42) or 42)
    metric_name = str(experiment_cfg.get("metric_name") or "citation_support_rate").strip()
    dataset = str(experiment_cfg.get("dataset") or "SciTraceBench").strip()
    implementation = str(experiment_cfg.get("implementation") or "citation_support").strip().lower()
    script_path.write_text(_experiment_script(implementation), encoding="utf-8")
    experiment_args = [
        "--metric-name",
        metric_name,
        "--dataset",
        dataset,
    ]
    if implementation == "daber_multidataset":
        dependency_path = experiments_dir / "daber_experiment.py"
        dependency_path.write_text(_experiment_script("daber"), encoding="utf-8")
        dataset_sources = experiment_cfg.get("dataset_dirs") or []
        if not isinstance(dataset_sources, list) or len(dataset_sources) < 2:
            raise ValueError("daber_multidataset requires at least two experiment.dataset_dirs")
        dataset_workspace = experiments_dir / "datasets"
        for source_text in dataset_sources:
            source = Path(str(source_text)).expanduser()
            if not source.is_absolute():
                source = (Path.cwd() / source).resolve()
            if not (source / "corpus.jsonl").is_file() or not (source / "queries.jsonl").is_file():
                raise FileNotFoundError(f"invalid BEIR dataset directory: {source}")
            destination = dataset_workspace / source.name
            shutil.copytree(source, destination, dirs_exist_ok=True)
            experiment_args.extend(["--dataset-dir", str(destination.resolve())])
        experiment_args.extend(
            [
                "--split",
                str(experiment_cfg.get("split") or "test"),
                "--budgets",
                str(experiment_cfg.get("budgets") or "2,3,5,8"),
                "--bootstrap-samples",
                str(int(experiment_cfg.get("bootstrap_samples", 1000) or 1000)),
            ]
        )
    elif implementation == "route_b2_artifact":
        source = Path(str(experiment_cfg.get("artifact_dir") or "")).expanduser()
        if not source.is_absolute():
            source = (Path.cwd() / source).resolve()
        if not (source / "aggregate.json").is_file():
            raise FileNotFoundError(f"invalid Route B2 artifact directory: {source}")
        destination = experiments_dir / "route_b2_artifacts"
        shutil.copytree(source, destination, dirs_exist_ok=True)
        experiment_args.extend(["--artifact-dir", str(destination.resolve())])
    plan = {
        "kind": "python_inline",
        "experiment_id": f"exp_{hashlib.sha256(topic.encode('utf-8')).hexdigest()[:12]}",
        "script_path": str(script_path),
        "dataset": dataset,
        "metric_name": metric_name,
        "seed": seed,
        "args": experiment_args,
        "implementation": implementation,
    }
    plan_review = harness.design_experiment(
        payload={
            "topic": topic,
            "research_question": research_question_payload["research_question"],
            "hypothesis": research_question_payload["hypothesis"],
            "evidence_count": len(literature_payload.get("evidence", [])),
            "trusted_plan": plan,
        }
    )
    plan["agent_review"] = plan_review
    plan_node = ExperimentPlan(
        experiment_id=plan["experiment_id"],
        hypothesis_id=research_question_payload["hypothesis"]["id"],
        command=f"{script_path.name} --metric-name {metric_name}",
        dataset=dataset,
        metric_names=[metric_name],
        seed=seed,
    )
    provenance.record_node(plan_node, session_id="")
    _write_json(experiments_dir / "plan.json", plan)
    return {"plan": plan, "plan_node": plan_node.to_dict(), "agent_review": plan_review}


def _run_experiment_execute(
    *,
    config: dict[str, Any],
    experiments_dir: Path,
    plan_payload: dict[str, Any],
    provenance: ScientificProvenanceRail,
    resource_snapshot: ResourceAuditSnapshot,
) -> dict[str, Any]:
    experiment_cfg = _config_section(config, "experiment")
    executor = SafeExperimentExecutor(
        workspace=experiments_dir,
        timeout_seconds=int(experiment_cfg.get("timeout_seconds", 120) or 120),
        allowed_executables=experiment_cfg.get("allowed_executables"),
    )
    started = time.perf_counter()
    try:
        result = executor.execute(plan_payload["plan"], output_dir=experiments_dir)
    except Exception:
        resource_snapshot.tool_failures += 1
        raise
    elapsed = max(0.0, time.perf_counter() - started)
    resource_snapshot.tool_calls += 1
    resource_snapshot.tools_by_name["experiment_execute"] = resource_snapshot.tools_by_name.get("experiment_execute", 0) + 1
    resource_snapshot.total_experiment_seconds += elapsed
    resource_snapshot.total_tool_seconds += elapsed
    resource_snapshot.tool_seconds_by_name["experiment_execute"] = (
        resource_snapshot.tool_seconds_by_name.get("experiment_execute", 0.0) + elapsed
    )

    artifact = ExperimentArtifact(
        experiment_id=str(result.payload.get("experiment_id") or plan_payload["plan"]["experiment_id"]),
        file_path=str(result.results_path),
        sha256=result.output_sha256,
        command=" ".join(result.command),
        seed=result.seed,
        dataset=str(result.payload.get("dataset") or plan_payload["plan"].get("dataset") or ""),
        metric_name=str(result.payload.get("metric_name") or plan_payload["plan"].get("metric_name") or ""),
        metric_value=result.payload.get("metric_value"),
        metadata={
            "claim_status": str((result.payload.get("claim_assessment") or {}).get("status") or "").strip(),
            "claim_reason": str((result.payload.get("claim_assessment") or {}).get("reason") or "").strip(),
            "implementation": str(plan_payload["plan"].get("implementation") or ""),
        },
    )
    provenance.record_node(artifact, session_id="")
    provenance.record_edge(
        EvidenceEdge(
            source_id=artifact.id,
            target_id=plan_payload["plan_node"]["id"],
            relation=EdgeRelation.DERIVED_FROM,
        ),
        session_id="",
    )
    return {"artifact": artifact.to_dict(), "results": result.payload}


def _run_claim_compile(
    *,
    topic: str,
    config: dict[str, Any],
    harness: ResearchStageHarness,
    provenance: ScientificProvenanceRail,
    literature_payload: dict[str, Any],
    execution_payload: dict[str, Any],
) -> dict[str, Any]:
    compiler_cfg = _config_section(config, "claim_compiler")
    evidence_nodes = [
        node_from_dict(node)
        for node in literature_payload["evidence"]
    ]
    artifact = node_from_dict(execution_payload["artifact"])
    compiler_input = ClaimCompilerInput(
        topic=topic,
        evidence=evidence_nodes,
        artifacts=[artifact],
        status_threshold_supported=float(compiler_cfg.get("supported_threshold", 0.75) or 0.75),
        status_threshold_refuted=float(compiler_cfg.get("refuted_threshold", 0.45) or 0.45),
    )
    raw_payload = harness.compile_claim(compiler_input)
    claim = sanitize_claim_payload(raw_payload, compiler_input=compiler_input)
    provenance.record_node(claim, session_id="")
    for evidence in evidence_nodes:
        provenance.record_edge(
            EvidenceEdge(source_id=claim.id, target_id=evidence.id, relation=EdgeRelation.SUPPORTS),
            session_id="",
        )
    provenance.record_edge(
        EvidenceEdge(source_id=claim.id, target_id=artifact.id, relation=EdgeRelation.TESTS),
        session_id="",
    )
    return {"claims": [claim.to_dict()]}


def _paper_results_view(results: dict[str, Any]) -> dict[str, Any]:
    """Send the writer exact paper-relevant results without full trial matrices.

    The complete imported artifact remains in experiments/results.json. This
    deterministic view removes repeated per-budget secondary metrics and train
    trial grids from the Team prompt, which otherwise get copied through the
    leader-to-writer handoff and can exhaust a long-running Team invocation.
    """
    def dataset_view(data: dict[str, Any]) -> dict[str, Any]:
        methods = data.get("methods_by_budget") or {}
        paired = data.get("paired_bootstrap_by_budget") or {}
        return {
            key: data[key] for key in (
                "dataset", "split", "retrievers", "dense_model", "calibration",
                "config", "dataset_statistics", "data_files", "claim_assessment",
                "limitations", "primary_budget", "primary_comparison", "feature_ablation",
            ) if key in data
        } | {
            "fusion_selected_weights": (data.get("fusion") or {}).get("selected_weights", {}),
            "recall_by_budget": {
                budget: {name: metrics.get("recall") for name, metrics in comparisons.items()}
                for budget, comparisons in methods.items()
            },
            "paired_recall_by_budget": {
                budget: {name: metrics.get("recall") for name, metrics in comparisons.items()}
                for budget, comparisons in paired.items()
            },
            "train_subsampling_stability": {
                key: value for key, value in (data.get("train_subsampling_stability") or {}).items()
                if key in {"recall_mean", "recall_std", "runs"}
            },
        }

    return {
        key: results[key] for key in (
            "experiment_id", "dataset", "split", "seed", "primary_budget", "metric_name",
            "metric_value", "primary_comparison", "claim_assessment", "claim_boundary",
            "artifact_bundle",
        ) if key in results
    } | {
        "datasets": {name: dataset_view(data) for name, data in (results.get("datasets") or {}).items()},
        "boundary_results": {
            name: dataset_view(data) for name, data in (results.get("boundary_results") or {}).items()
        },
    }


def _run_draft_generate(
    *,
    topic: str,
    config: dict[str, Any],
    harness: ResearchStageHarness,
    stage_context: StageAwareContextRail,
    output_root: Path,
    research_question_payload: dict[str, Any],
    literature_payload: dict[str, Any],
    execution_payload: dict[str, Any],
    claim_payload: dict[str, Any],
) -> dict[str, Any]:
    context_bundle = stage_context.build_stage_context("", stage="writing")
    quality_cfg = _config_section(config, "paper_quality")
    results = execution_payload["results"]
    figure_manifest = (
        write_budget_curve(results, output_root)
        if bool(quality_cfg.get("require_results_figure", False)) else None
    )
    effect_figure = (
        render_budget_effect_figure(output_root, output_root / "figures")
        if bool(quality_cfg.get("require_budget_effect_figure", False)) else None
    )
    if quality_cfg.get("require_budget_effect_figure") and effect_figure is None:
        raise ValueError("The verified Route B2 budget-effect figure cannot be built")
    if quality_cfg.get("require_research_narrative"):
        render_method_overview(output_root)
        render_query_diagnostics(output_root / "experiments" / "route_b2_artifacts", output_root / "figures")
    table_rows = _primary_results_table_rows(results) if bool(quality_cfg.get("require_results_table", False)) else []
    adaptive_rows = _adaptive_baseline_rows(results) if bool(quality_cfg.get("require_adaptive_baseline_table", False)) else []
    stability_rows = _train_subsampling_rows(results) if bool(quality_cfg.get("require_train_subsampling_summary", False)) else []
    manuscript_context = _manuscript_context(config, literature_payload, execution_payload, claim_payload, output_root)
    _write_json(output_root / "manuscript_context.json", manuscript_context)
    harness.set_manuscript_context(manuscript_context)
    prompt = json.dumps(
        manuscript_context | {
            "topic": topic,
            "research_question": research_question_payload["research_question"],
            "hypothesis": research_question_payload["hypothesis"],
            "evidence": literature_payload["evidence"],
            "artifact": execution_payload["artifact"],
            "results": _paper_results_view(results),
            "required_results_table_rows": table_rows,
            "required_adaptive_baseline_rows": adaptive_rows,
            "required_train_subsampling_summary": stability_rows,
            "result_figure": figure_manifest,
            "budget_effect_figure": {
                "path": EFFECT_FIGURE_PATH,
                "data_path": "figures/budget_effect_data.json",
                "source_sha256": effect_figure["sources"],
                "caption_requirement": "Describe change in evidence recall for SciFact, NFCorpus, and separately trained finance-domain lexical FiQA. Pointwise paired-bootstrap intervals appear at every budget; they condition on realized allocations and are not simultaneous intervals or zero-shot transfer results.",
            } if effect_figure else None,
            "claims": claim_payload["claims"],
            "stage_context": context_bundle,
            "paper_requirements": quality_cfg.get("method_requirements", []),
            "implementation_contract": ROUTE_B2_METHOD_CONTRACT if quality_cfg.get("require_route_b2_method_contract") else None,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    draft = harness.write_draft(prompt=prompt)
    return {"draft": draft, "stage_context": context_bundle, "result_figure": figure_manifest}


def _manuscript_context(config, literature_payload, execution_payload, claim_payload, output_root=None) -> dict[str, Any]:
    """Share compact, verified evidence with all authoring stages, including resume."""
    quality = _config_section(config, "paper_quality")
    results = execution_payload["results"]
    narrative = {}
    if quality.get("require_research_narrative"):
        if output_root is None:
            raise ValueError("The final manuscript requires its verified per-query artifact directory")
        diagnostics = build_query_diagnostics(Path(output_root) / "experiments" / "route_b2_artifacts")
        narrative = narrative_context() | {
            "query_diagnostics": {"primary_budget": diagnostics["primary_budget"],
                "summaries": [row for row in diagnostics["summaries"] if row["budget"] == diagnostics["primary_budget"]],
                "limitations": diagnostics["limitations"],
                "budget_transitions": [{key: value for key, value in row.items() if key != "regression_query_ids"}
                                       for row in diagnostics["budget_transitions"]]},
            "required_query_outcome_rows": query_outcome_rows(diagnostics),
            "method_figure": "figures/method_overview.pdf",
            "query_risk_figure": "figures/query_risk.pdf",
        }
    return narrative | {
        "results": _paper_results_view(results),
        "artifact": execution_payload.get("artifact", {}),
        "evidence": literature_payload["evidence"],
        "claims": claim_payload["claims"],
        "paper_requirements": quality.get("method_requirements", []),
        "manuscript_plan": quality.get("manuscript_plan", {}),
        "minimum_section_words": quality.get("minimum_section_words", {}),
        "implementation_contract": ROUTE_B2_METHOD_CONTRACT
        if quality.get("require_route_b2_method_contract") else None,
        "required_results_table_rows": _primary_results_table_rows(results)
        if quality.get("require_results_table") else [],
        "required_adaptive_baseline_rows": _adaptive_baseline_rows(results)
        if quality.get("require_adaptive_baseline_table") else [],
        "required_train_subsampling_summary": _train_subsampling_rows(results)
        if quality.get("require_train_subsampling_summary") else [],
        "required_feature_ablation_rows": feature_ablation_rows(results)
        if quality.get("require_feature_ablation_table") else [],
    }


def query_outcome_rows(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in ("dataset", "queries", "wins", "ties", "losses",
             "rescued_from_no_evidence", "newly_no_evidence")}
            for row in diagnostics["summaries"] if row["budget"] == diagnostics["primary_budget"]]


def _review_errors(review: dict[str, Any]) -> list[dict[str, str]]:
    return [{
        "code": "paper_reviewer_finding", "severity": "error", "node_id": "",
        "message": f"{finding['section']}: {finding['message']} Fix: {finding['suggested_fix']}",
        "quote": finding["quote"],
    } for finding in review.get("findings", []) if str(finding.get("severity", "")).lower() == "error"]


def _run_consistency_loop(
    *,
    config: dict[str, Any],
    harness: ResearchStageHarness,
    gate: ScientificConsistencyGateRail,
    provenance: ScientificProvenanceRail,
    output_root: Path,
    literature_payload: dict[str, Any],
    claim_payload: dict[str, Any],
    execution_payload: dict[str, Any],
    initial_draft: str,
) -> dict[str, Any]:
    revision_cfg = _config_section(config, "revision")
    max_rounds = max(0, int(revision_cfg.get("max_rounds", 2)))
    draft = initial_draft
    attempts: list[dict[str, Any]] = []
    harness.set_manuscript_context(_manuscript_context(
        config, literature_payload, execution_payload, claim_payload, output_root,
    ))
    reviews_dir = output_root / "reviews"
    reviews_dir.mkdir(exist_ok=True)

    for round_index in range(max_rounds + 1):
        # #region debug-point A:consistency-loop-draft
        _debug_emit_claimed_value_none(
            "A",
            "pipeline.py:_run_consistency_loop",
            "[DEBUG] consistency loop received draft",
            {
                "round": round_index,
                "draft_has_structured_refs": "## Structured References" in draft,
                "draft_preview": draft[:400],
            },
        )
        # #endregion
        structured = _extract_structured_references(draft)
        structured = _complete_structured_references(
            structured,
            literature_payload=literature_payload,
            claim_payload=claim_payload,
            execution_payload=execution_payload,
        )
        # #region debug-point A:consistency-loop-structured
        _debug_emit_claimed_value_none(
            "A",
            "pipeline.py:_run_consistency_loop",
            "[DEBUG] extracted structured references",
            {
                "round": round_index,
                "structured_is_none": structured is None,
                "structured_keys": sorted(structured.keys()) if isinstance(structured, dict) else [],
                "reported_metrics": structured.get("reported_metrics", []) if isinstance(structured, dict) else [],
            },
        )
        # #endregion
        structured["draft_text"] = draft
        # #region debug-point C:consistency-loop-gate-payload
        _debug_emit_claimed_value_none(
            "C",
            "pipeline.py:_run_consistency_loop",
            "[DEBUG] gate payload prepared",
            {
                "round": round_index,
                "reported_metrics": structured.get("reported_metrics", []),
                "claim_ids": [item.get("claim_id") for item in structured.get("claims", []) if isinstance(item, dict)],
                "citation_count": len(structured.get("citations", [])),
            },
        )
        # #endregion
        validation_report = gate.check_payload(structured, session_id="")
        quality_report = audit_draft(
            draft,
            evidence=literature_payload["evidence"],
            config=_config_section(config, "paper_quality"),
            table_rows=_primary_results_table_rows(execution_payload["results"])
            if _config_section(config, "paper_quality").get("require_results_table") else None,
            adaptive_rows=_adaptive_baseline_rows(execution_payload["results"])
            if _config_section(config, "paper_quality").get("require_adaptive_baseline_table") else None,
            stability_rows=_train_subsampling_rows(execution_payload["results"])
            if _config_section(config, "paper_quality").get("require_train_subsampling_summary") else None,
            results=execution_payload["results"],
            query_diagnostics=build_query_diagnostics(output_root / "experiments" / "route_b2_artifacts")
            if _config_section(config, "paper_quality").get("require_query_outcome_table") else None,
        )
        _write_json(output_root / "paper_quality_report.json", quality_report)
        if not quality_report["ok"]:
            validation_report["errors"].extend(quality_report["errors"])
        _enforce_runtime_budget(config, harness.runtime, stage="review", before_call=True,
                                call_log_path=output_root / "model_calls.jsonl")
        audit_review = harness.review_draft(draft=draft, errors=validation_report["errors"])
        audit_review = {**audit_review, "draft_sha256": hashlib.sha256(draft.encode("utf-8")).hexdigest(),
                        "round": round_index, "scope": "full_public_manuscript"}
        _write_json(output_root / "review_report.json", audit_review)
        _write_json(reviews_dir / f"round_{round_index}.json", audit_review)
        validation_report["errors"].extend(_review_errors(audit_review))
        validation_report["error_count"] = len(validation_report["errors"])
        validation_report["ok"] = not validation_report["errors"]
        validation_report["paper_quality"] = quality_report
        validation_report["review"] = audit_review
        attempts.append({"round": round_index, "validation_report": validation_report})
        if validation_report["ok"]:
            return {
                "draft": draft,
                "review_report": audit_review,
                "validation_report": {"ok": True, "attempts": attempts, "errors": []},
            }
        if round_index >= max_rounds:
            return {
                "draft": draft,
                "validation_report": {
                    "ok": False,
                    "attempts": attempts,
                    "errors": validation_report["errors"],
                    "failed_reason": "max_revision_rounds_exceeded",
                },
            }

        _enforce_runtime_budget(config, harness.runtime, stage="revision", before_call=True,
                                call_log_path=output_root / "model_calls.jsonl")
        review, revised_draft = harness.review_and_revise(
            draft=draft, errors=validation_report["errors"], review=audit_review,
        )
        audit_review = review
        _write_json(output_root / "review_report.json", audit_review)
        for claim_dict in claim_payload["claims"]:
            provenance.record_node(
                RevisionDecision(
                    claim_id=claim_dict["id"],
                    decision="targeted_revision",
                    rationale=str(review.get("summary") or ""),
                    metadata={"round": round_index + 1},
                ),
                session_id="",
            )
        draft = revised_draft
        (output_root / "draft.md").write_text(draft, encoding="utf-8")

    return {"draft": draft, "validation_report": {"ok": False, "attempts": attempts, "errors": []}}


def _build_literature_provider(
    cfg: dict[str, Any],
    *,
    evidence_dir: Path,
    fallback_provider: FixtureLiteratureProvider,
    allow_fallback: bool,
):
    provider_name = str(cfg.get("provider") or "mock").strip().lower()
    max_results = int(cfg.get("max_results", 5) or 5)
    timeout_seconds = float(cfg.get("timeout_seconds", 20.0) or 20.0)
    retries = int(cfg.get("retries", 2) or 2)
    verify_links = bool(cfg.get("verify_links", True))

    if provider_name in {"mock", "fixture"}:
        return fallback_provider

    providers = []
    if provider_name in {"openalex", "auto", "real"}:
        providers.append(
            OpenAlexLiteratureProvider(
                cache_root=evidence_dir / "raw",
                max_results=max_results,
                timeout_seconds=timeout_seconds,
                retries=retries,
                verify_links=verify_links,
            )
        )
    if provider_name in {"crossref", "auto", "real"}:
        providers.append(
            CrossrefLiteratureProvider(
                cache_root=evidence_dir / "raw",
                max_results=max_results,
                timeout_seconds=timeout_seconds,
                retries=retries,
                verify_links=verify_links,
            )
        )
    return ResilientLiteratureProvider(providers=providers, fallback=fallback_provider if allow_fallback else None)


def _restore_prior_tool_usage(snapshot: ResourceAuditSnapshot, previous: dict[str, Any]) -> None:
    """Carry forward completed tool work when a Team run resumes."""
    if not previous:
        return
    snapshot.started_at = str(previous.get("started_at") or snapshot.started_at)
    snapshot.tool_calls = int(previous.get("tool_calls", 0) or 0)
    snapshot.tool_failures = int(previous.get("tool_failures", 0) or 0)
    snapshot.total_tool_seconds = float(previous.get("total_tool_seconds", 0.0) or 0.0)
    snapshot.total_experiment_seconds = float(previous.get("total_experiment_seconds", 0.0) or 0.0)
    snapshot.tools_by_name = {
        str(name): int(count) for name, count in (previous.get("tools_by_name") or {}).items()
    }
    snapshot.tool_seconds_by_name = {
        str(name): float(seconds)
        for name, seconds in (previous.get("tool_seconds_by_name") or {}).items()
    }


def _fill_resource_snapshot(
    snapshot: ResourceAuditSnapshot,
    *,
    usage_summary: dict[str, Any],
    topic: str,
    claim_count: int,
    evidence_count: int,
    artifact_count: int,
    started: float,
    workflow_model_calls: int = 0,
    diagnostic_model_calls: int = 0,
    retries: int = 0,
    model_name: str = "",
) -> None:
    total = usage_summary.get("total") or {}
    usage_model_calls = int(total.get("request_count", 0) or 0)
    snapshot.model_calls = max(usage_model_calls, int(workflow_model_calls or 0))
    snapshot.invoke_count = snapshot.model_calls
    snapshot.retries = max(snapshot.retries, int(retries or 0))
    snapshot.input_tokens = int(total.get("prompt_tokens", 0) or 0)
    snapshot.output_tokens = int(total.get("completion_tokens", 0) or 0)
    snapshot.total_tokens = int(total.get("total_tokens", 0) or 0)
    snapshot.total_runtime_seconds = max(snapshot.total_runtime_seconds, time.perf_counter() - started)
    for record in usage_summary.get("records", []) or []:
        model = str(record.get("model") or "unknown")
        snapshot.model_calls_by_name[model] = snapshot.model_calls_by_name.get(model, 0) + 1
        snapshot.model_tokens_by_name[model] = snapshot.model_tokens_by_name.get(model, 0) + int(
            record.get("total_tokens", 0) or 0
        )
    if snapshot.model_calls and not snapshot.model_calls_by_name and model_name:
        snapshot.model_calls_by_name[model_name] = snapshot.model_calls
    snapshot.extra["topic"] = topic
    snapshot.extra["claim_count"] = claim_count
    snapshot.extra["evidence_count"] = evidence_count
    snapshot.extra["artifact_count"] = artifact_count
    if diagnostic_model_calls:
        snapshot.extra["diagnostic_model_calls"] = diagnostic_model_calls
    snapshot.extra["workflow_model_calls"] = snapshot.model_calls
    # Team request_count is the number of stage invocations. Preserve this
    # historical field while exposing separately observed provider receipts.
    snapshot.extra["model_call_scope"] = "workflow_stage_invocations"
    records = usage_summary.get("records") or []
    snapshot.extra["observed_provider_calls"] = sum(int(r.get("provider_calls", 0) or 0) for r in records)
    snapshot.extra["missing_usage_calls"] = sum(int(r.get("missing_usage_calls", 0) or 0) for r in records)
    snapshot.extra["usage_sources"] = sorted({str(r.get("usage_source") or "legacy_unspecified") for r in records})
    snapshot.extra["usage_statuses"] = sorted({str(r.get("usage_status") or "legacy_unspecified") for r in records})
    snapshot.extra["legacy_usage_records"] = sum("usage_source" not in r for r in records)
    snapshot.extra["provider_count_scope"] = "observed llm_usage events; not guaranteed to include unreported or in-flight calls"


def _resume_or_compute(
    run_store: ResearchRunStore,
    stage: str,
    *,
    resume: bool,
    compute,
) -> dict[str, Any]:
    if resume:
        record = run_store.get_stage(stage)
        if record is not None and record.status == "completed":
            return record.payload
    run_store.set_stage(stage, status="running", payload={"stage": stage})
    payload = compute()
    run_store.set_stage(stage, status="completed", payload=payload)
    return payload


def _count_model_call_records(path: Path, *, include_diagnostics: bool = True) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if include_diagnostics or str(payload.get("stage") or "") != "diagnostics":
                count += 1
    return count


def _model_diagnostics_error(payload: dict[str, Any]) -> str:
    """Return a safe fail-fast message when real model probes did not work."""
    level1 = payload.get("level1_sdk_call") or {}
    checks = [
        ("synchronous SDK", level1.get("sync_client") or {}),
        ("asynchronous SDK", level1.get("async_client") or {}),
        ("JiuwenSwarm", payload.get("level2_jiuwenswarm_call") or {}),
    ]
    failed = [(name, result) for name, result in checks if result.get("success") is not True]
    if not failed:
        return ""
    status_codes = {
        int(item["status_code"])
        for _, result in failed
        for item in (result.get("error_chain") or [])
        if isinstance(item, dict) and isinstance(item.get("status_code"), int)
    }
    messages = " ".join(
        str(item.get("message") or "")
        for _, result in failed
        for item in (result.get("error_chain") or [])
        if isinstance(item, dict)
    ).casefold()
    failed_names = ", ".join(name for name, _ in failed)
    if 402 in status_codes or "insufficient balance" in messages:
        return f"Model diagnostics failed ({failed_names}): provider reported insufficient balance (HTTP 402)."
    if 401 in status_codes or 403 in status_codes:
        return f"Model diagnostics failed ({failed_names}): provider rejected the configured credentials."
    return f"Model diagnostics failed ({failed_names}); inspect model_diagnostics.json before running the Team."


def _count_failed_model_call_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("stage") or "") != "diagnostics" and payload.get("status") == "error":
                count += 1
    return count


def _append_diagnostics_model_calls(path: Path, diagnostics: dict[str, Any]) -> None:
    timestamp = utc_now_iso()
    level1 = diagnostics.get("level1_sdk_call") or {}
    for operation in ("sync_client", "async_client"):
        payload = level1.get(operation) or {}
        _append_jsonl(
            path,
            {
                "timestamp": timestamp,
                "backend": "sdk_direct",
                "provider": diagnostics.get("resolved_model", {}).get("provider", ""),
                "model_name": diagnostics.get("resolved_model", {}).get("model_name", ""),
                "stage": "diagnostics",
                "operation": f"level1_{operation}",
                "status": "success" if payload.get("success") else "error",
                "error_chain": payload.get("error_chain", []),
            },
        )
    level2 = diagnostics.get("level2_jiuwenswarm_call") or {}
    _append_jsonl(
        path,
        {
            "timestamp": timestamp,
            "backend": "jiuwenswarm",
            "provider": diagnostics.get("resolved_model", {}).get("provider", ""),
            "model_name": diagnostics.get("resolved_model", {}).get("model_name", ""),
            "stage": "diagnostics",
            "operation": "level2_jiuwenswarm",
            "status": "success" if level2.get("success") else "error",
            "error_chain": level2.get("error_chain", []),
        },
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _durable_usage_summary(path: Path, runtime) -> dict[str, Any]:
    """Count every recorded workflow call, including calls before a resume."""
    if not path.exists():
        return runtime.usage_summary()
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("stage") == "diagnostics":
            continue
        if not all(isinstance(item.get(key), int) for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            return runtime.usage_summary()
        records.append({**item, "model": item.get("model_name") or item.get("model") or "unknown"})
    if not records:
        return runtime.usage_summary()
    return {
        "total": {
            "request_count": len(records),
            "provider_calls": sum(int(item.get("provider_calls", 0) or 0) for item in records),
            "missing_usage_calls": sum(int(item.get("missing_usage_calls", 0) or 0) for item in records),
            "prompt_tokens": sum(item["prompt_tokens"] for item in records),
            "completion_tokens": sum(item["completion_tokens"] for item in records),
            "total_tokens": sum(item["total_tokens"] for item in records),
        },
        "records": records,
    }


def _enforce_runtime_budget(
    config: dict[str, Any], runtime, *, stage: str, call_log_path: Path | None = None,
    before_call: bool = False,
) -> None:
    budget = _config_section(config, "budget")
    usage = _durable_usage_summary(call_log_path, runtime) if call_log_path else runtime.usage_summary()
    total = usage.get("total") or {}
    total_tokens = int(total.get("total_tokens", 0) or 0)
    request_count = int(total.get("request_count", 0) or 0)
    max_total_tokens = budget.get("max_total_tokens")
    max_model_calls = budget.get("max_model_calls")
    cost_per_1k_tokens = float(budget.get("estimated_cost_per_1k_tokens", 0.0) or 0.0)
    max_estimated_cost = budget.get("max_estimated_cost")
    estimated_cost = (total_tokens / 1000.0) * cost_per_1k_tokens if cost_per_1k_tokens > 0 else 0.0
    if max_total_tokens is not None and total_tokens > int(max_total_tokens):
        raise RuntimeError(f"token budget exceeded at stage {stage}: {total_tokens} > {int(max_total_tokens)}")
    if before_call and max_total_tokens is not None and total_tokens == int(max_total_tokens):
        raise RuntimeError(f"token budget exhausted before stage {stage}: {total_tokens}")
    if max_model_calls is not None and request_count > int(max_model_calls):
        raise RuntimeError(f"model call budget exceeded at stage {stage}: {request_count} > {int(max_model_calls)}")
    if before_call and max_model_calls is not None and request_count == int(max_model_calls):
        raise RuntimeError(f"model call budget exhausted before stage {stage}: {request_count}")
    if max_estimated_cost is not None and estimated_cost > float(max_estimated_cost):
        raise RuntimeError(
            f"estimated cost budget exceeded at stage {stage}: {estimated_cost:.6f} > {float(max_estimated_cost):.6f}"
        )
    if before_call and cost_per_1k_tokens > 0 and max_estimated_cost is not None and estimated_cost == float(max_estimated_cost):
        raise RuntimeError(f"estimated cost budget exhausted before stage {stage}: {estimated_cost:.6f}")


def _config_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key) or {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _experiment_script(implementation: str = "citation_support") -> str:
    """Return a real experiment script suitable for the restricted workspace."""
    scripts = {
        "citation_support": "citation_support_experiment.py",
        "daber": "daber_experiment.py",
        "daber_multidataset": "daber_multidataset_experiment.py",
        "route_b2_artifact": "route_b2_artifact_experiment.py",
    }
    try:
        filename = scripts[implementation]
    except KeyError as exc:
        raise ValueError(f"unknown experiment implementation: {implementation}") from exc
    script_path = Path(__file__).resolve().parent / filename
    return script_path.read_text(encoding="utf-8")


def _extract_structured_references(draft: str) -> dict[str, Any] | None:
    marker = "## Structured References"
    if marker not in draft:
        return None
    tail = draft.split(marker, 1)[1]
    if "```json" not in tail:
        return None
    after = tail.split("```json", 1)[1]
    if "```" not in after:
        return None
    raw_json = after.split("```", 1)[0].strip()
    # #region debug-point B:extract-structured-raw-json
    _debug_emit_claimed_value_none(
        "B",
        "pipeline.py:_extract_structured_references",
        "[DEBUG] extracted raw structured json",
        {
            "raw_json_preview": raw_json[:500],
        },
    )
    # #endregion
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        # #region debug-point B:extract-structured-json-error
        _debug_emit_claimed_value_none(
            "B",
            "pipeline.py:_extract_structured_references",
            "[DEBUG] structured json decode failed",
            {
                "raw_json_preview": raw_json[:500],
            },
        )
        # #endregion
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        citations: list[str] = []
        claims: list[dict[str, Any]] = []
        reported_metrics: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            item_type = str(item.get("type") or item.get("kind") or "").strip().lower()
            if item_type == "literatureevidence" and item_id:
                citations.append(item_id)
            elif item_type == "claim" and item_id:
                claims.append({"claim_id": item_id})
            elif item_type == "experimentartifact" and item_id:
                reported_metrics.append(
                    {
                        "artifact_id": item_id,
                        "metric_name": item.get("metric_name"),
                        "claimed_value": item.get("metric_value"),
                    }
                )
        if citations or claims or reported_metrics:
            return {
                "citations": citations,
                "claims": claims,
                "reported_metrics": reported_metrics,
            }
    return None


def _complete_structured_references(
    structured: dict[str, Any] | None,
    *,
    literature_payload: dict[str, Any],
    claim_payload: dict[str, Any],
    execution_payload: dict[str, Any],
) -> dict[str, Any]:
    completed = dict(structured or {})

    citations = completed.get("citations")
    if not isinstance(citations, list) or not any(str(item).strip() for item in citations):
        completed["citations"] = [
            str(item.get("id") or "").strip()
            for item in literature_payload.get("evidence", [])
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        ]

    claims = completed.get("claims")
    if not isinstance(claims, list) or not claims:
        completed["claims"] = [
            {"claim_id": item["id"]}
            for item in claim_payload.get("claims", [])
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        ]

    artifact = execution_payload.get("artifact") if isinstance(execution_payload, dict) else None
    artifact_id = str((artifact or {}).get("id") or "").strip()
    metric_name = (artifact or {}).get("metric_name")
    metric_value = (artifact or {}).get("metric_value")
    reported_metrics = completed.get("reported_metrics")
    if not isinstance(reported_metrics, list):
        reported_metrics = []
    normalized_metrics: list[dict[str, Any]] = []
    for item in reported_metrics:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        if artifact_id and not str(normalized.get("artifact_id") or "").strip():
            normalized["artifact_id"] = artifact_id
        if metric_name and not normalized.get("metric_name"):
            normalized["metric_name"] = metric_name
        if normalized.get("claimed_value") is None and artifact_id and str(normalized.get("artifact_id") or "").strip() == artifact_id:
            normalized["claimed_value"] = metric_value
        normalized_metrics.append(normalized)
    if not normalized_metrics and artifact_id:
        normalized_metrics = [
            {
                "artifact_id": artifact_id,
                "metric_name": metric_name,
                "claimed_value": metric_value,
            }
        ]
    completed["reported_metrics"] = normalized_metrics
    return completed


def _debug_emit_claimed_value_none(hypothesis_id: str, location: str, msg: str, data: dict[str, Any]) -> None:
    import json as _json
    import urllib.request as _urlreq

    _p = ".dbg/claimed-value-none.env"
    _u = "http://127.0.0.1:7777/event"
    _s = "claimed-value-none"
    try:
        with open(_p, encoding="utf-8") as _handle:
            _content = _handle.read()
        for _line in _content.splitlines():
            if _line.startswith("DEBUG_SERVER_URL="):
                _u = _line.split("=", 1)[1].strip() or _u
            elif _line.startswith("DEBUG_SESSION_ID="):
                _s = _line.split("=", 1)[1].strip() or _s
    except Exception:
        pass
    try:
        _urlreq.urlopen(
            _urlreq.Request(
                _u,
                data=_json.dumps(
                    {
                        "sessionId": _s,
                        "runId": "pre-fix",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": msg,
                        "data": data,
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=2.0,
        ).read()
    except Exception:
        pass


def _maybe_interrupt(debug: dict[str, Any], stage: str) -> None:
    if str(debug.get("stop_after_stage") or "").strip() == stage:
        raise WorkflowInterrupted(f"workflow interrupted after stage: {stage}")


__all__ = ["WorkflowInterrupted", "dump_yaml_config", "load_yaml_config", "run_pipeline"]
