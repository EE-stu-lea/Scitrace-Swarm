"""Runtime-only recovery must preserve scientific and model configuration."""
from __future__ import annotations

from copy import deepcopy
import json
import sqlite3

import pytest

from ._stubs import install_openjiuwen_stubs

install_openjiuwen_stubs()

from examples.scitrace_research.resume_policy import inspect_resume, snapshot_resume, validate_resume_config
from examples.scitrace_research.build_submission import _copy_sqlite_consistent


def _old():
    return {
        "topic": "research question", "mode": "real_team",
        "models": {"defaults": [{"model_client_config": {"model_name": "same"}}]},
        "team": {"max_invocation_total_tokens": 100,
                 "stage_resource_limits": {"topic_intake": {"max_total_tokens": 40}}},
    }


def test_resume_accepts_profile_opt_in_and_higher_ceiling_only():
    old = _old()
    new = deepcopy(old)
    new.pop("topic")  # source config is read before pipeline injects topic
    new["team"]["stage_capability_profile"] = "prompt_only"
    new["team"]["direct_specialist_stages"] = ["experiment_plan", "draft", "review", "revision"]
    new["team"]["stage_resource_limits"]["topic_intake"]["max_total_tokens"] = 60
    changes = validate_resume_config(old, new, topic="research question")
    assert {item["path"] for item in changes} == {
        "team.stage_capability_profile", "team.direct_specialist_stages",
        "team.stage_resource_limits.topic_intake.max_total_tokens",
    }


def test_resume_rejects_arbitrary_specialist_routing_change():
    old, new = _old(), _old()
    new["team"]["direct_specialist_stages"] = ["draft"]
    with pytest.raises(ValueError, match="direct_specialist_stages"):
        validate_resume_config(old, new, topic="research question")


@pytest.mark.parametrize("change", ["model", "topic", "reduced_ceiling", "new_scientific_input"])
def test_resume_rejects_scientific_or_model_drift(change):
    old, new = _old(), _old()
    if change == "model":
        new["models"]["defaults"][0]["model_client_config"]["model_name"] = "other"
    elif change == "topic":
        new["topic"] = "other question"
    elif change == "reduced_ceiling":
        new["team"]["max_invocation_total_tokens"] = 99
    else:
        new["experiment"] = {"artifact_dir": "different evidence"}
    with pytest.raises(ValueError):
        validate_resume_config(old, new, topic="research question")


def test_resume_check_is_read_only_and_snapshot_includes_uncheckpointed_wal(tmp_path):
    run = tmp_path / "failed_run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"status": "failed", "mode": "real_team"}))
    (run / "run_config.yaml").write_text('topic: research question\nmode: real_team\nteam: {}\n')
    next_config = tmp_path / "next.yaml"
    next_config.write_text('mode: real_team\nteam:\n  stage_capability_profile: prompt_only\n')
    (run / "provider_usage.jsonl").write_text('{"total_tokens": 10}\n')
    conn = sqlite3.connect(run / "research.db")
    try:
        conn.execute('pragma journal_mode=WAL')
        conn.execute('create table run_stage(stage text, status text)')
        conn.execute("insert into run_stage values('topic_intake','failed')")
        conn.commit()
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes())
                  for p in run.glob('research.db*')}
        inspected = inspect_resume(run, next_config, topic="research question")
        assert inspected["checkpoints"] == [{"stage": "topic_intake", "status": "failed"}]
        assert before == {p.name: (p.stat().st_mtime_ns, p.read_bytes())
                          for p in run.glob('research.db*')}
        snapshot = run / 'resume_history' / 'first'
        audit = snapshot_resume(run, next_config, snapshot, topic="research question")
        assert 'provider_usage.jsonl' in audit['preserved_files_sha256']
        restored = sqlite3.connect(snapshot / 'research.db')
        try:
            assert restored.execute('select stage,status from run_stage').fetchone() == ('topic_intake', 'failed')
        finally:
            restored.close()
        packaged = tmp_path / 'submission.db'
        _copy_sqlite_consistent(run / 'research.db', packaged)
        restored = sqlite3.connect(packaged)
        try:
            assert restored.execute('select stage,status from run_stage').fetchone() == ('topic_intake', 'failed')
        finally:
            restored.close()
    finally:
        conn.close()
