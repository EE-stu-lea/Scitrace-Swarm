#!/usr/bin/env bash
# Run the current strict-real Team workflow; credentials stay in the process.
set -euo pipefail
umask 077
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
research_topic='Predicting recall curves for budgeted batch allocation in hybrid scientific evidence retrieval'
run_config_path="$project_root/examples/scitrace_research/configs/team_real_route_b2.yaml"
check_only=false
resume_dir=""
while [[ $# -gt 0 ]]; do
case "$1" in
  --check) check_only=true ;;
  --resume)
    if [[ $# -lt 2 ]]; then echo '--resume requires an existing run directory.' >&2; exit 2; fi
    resume_dir=$(realpath -- "$2")
    shift ;;
  --help|-h)
    echo 'Usage: bash examples/scitrace_research/launch_paper.sh [--check] [--resume RUN_DIRECTORY]'
    echo '--check: validate local dependencies and configuration; no API calls or paper generation.'
    exit 0 ;;
  *) echo 'Unknown argument. Use --help.' >&2; exit 2 ;;
esac
shift
done
python_bin=${SCITRACE_PYTHON:-}
if [[ -z "$python_bin" ]]; then
  if [[ -x "$project_root/.venv/bin/python" ]]; then
    python_bin="$project_root/.venv/bin/python"
  else
    python_bin=$(command -v python3 || true)
  fi
fi
if [[ ! -x "$python_bin" ]]; then
  echo 'Set SCITRACE_PYTHON to the Python executable with the project dependencies.' >&2
  exit 2
fi
if ! command -v pdflatex >/dev/null 2>&1; then
  echo 'pdflatex is required for PDF export. Install the project TeX dependencies before running.' >&2
  exit 2
fi
for tex_package in fontenc hyperref array graphicx float placeins amsmath amssymb eso-pic fancyhdr natbib url; do
  if ! kpsewhich "$tex_package.sty" >/dev/null 2>&1; then
    echo "Missing TeX package: $tex_package.sty. Install it before starting paid model calls." >&2
    exit 2
  fi
done
export API_BASE=${API_BASE:-https://api.deepseek.com}
export MODEL_NAME=${MODEL_NAME:-deepseek-v4-pro}
export MODEL_PROVIDER=${MODEL_PROVIDER:-DeepSeek}
export ENDPOINT_PROFILE=${ENDPOINT_PROFILE:-deepseek}
# Desktop proxy managers sometimes export the ambiguous socks:// scheme.
# Both the SDK and native Team HTTP clients require an explicit SOCKS version.
# Change only this launcher's child environment, preserving host/port/auth.
for proxy_variable in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  proxy_value=${!proxy_variable:-}
  if [[ "${proxy_value,,}" == socks://* ]]; then
    export "$proxy_variable=socks5://${proxy_value:8}"
    printf 'Normalized %s proxy scheme: socks -> socks5\n' "$proxy_variable"
  fi
done
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n ${SCITRACE_CORE_SRC:-} ]]; then
  [[ -d "$SCITRACE_CORE_SRC/openjiuwen" ]] || { echo 'SCITRACE_CORE_SRC must contain openjiuwen/' >&2; exit 2; }
  export PYTHONPATH="$project_root:$SCITRACE_CORE_SRC${PYTHONPATH:+:$PYTHONPATH}"
fi
export MPLCONFIGDIR="$project_root/.runtime_cache/matplotlib"
export JIUWENSWARM_DATA_DIR="$project_root/.runtime_cache/preflight"
cd "$project_root"
"$python_bin" - <<'PY'
import sys
import os
from pathlib import Path
if sys.version_info < (3, 11):
    raise SystemExit('Python 3.11 or newer is required.')
import openai, matplotlib, numpy, markdown_it
# Client construction validates proxy syntax and optional SOCKS dependencies
# without sending a request or requiring a real credential.
with openai.OpenAI(api_key='local-preflight-only', base_url='https://api.deepseek.com', max_retries=0):
    pass
from examples.scitrace_research.workflow.pipeline import load_yaml_config
from examples.scitrace_research.paper_export import draft_tex
cfg = load_yaml_config('examples/scitrace_research/configs/team_real_route_b2.yaml')
assert cfg['paper_quality']['require_research_narrative']
assert cfg['team']['stage_scoped_team']
assert cfg['team']['stage_capability_profile'] == 'prompt_only'
assert cfg['team']['direct_specialist_stages'] == ['experiment_plan', 'draft', 'review', 'revision']
from examples.scitrace_research.workflow.query_diagnostics import build_query_diagnostics
from jiuwenswarm.agents.harness.research.stage_policy import stage_resource_limits
build_query_diagnostics(Path(cfg['experiment']['artifact_dir']))
for stage in cfg['team']['stage_resource_limits']:
    stage_resource_limits(cfg, stage)
from openjiuwen.agent_teams.schema.blueprint import TeamAgentSpec
from jiuwenswarm.agents.harness.team.config_loader import load_team_spec_dict
from jiuwenswarm.agents.harness.research.stage_policy import scope_team_spec
from jiuwenswarm.agents.harness.research.team_contract import TEAM_STAGE_CONTRACTS
from jiuwenswarm.common.config import resolve_env_vars
os.environ.setdefault('API_KEY', 'local-preflight-only')
resolved = resolve_env_vars(cfg)
spec = load_team_spec_dict(config_base=resolved, template_id=cfg['team']['template_id'], strict_template=True)
for contract in TEAM_STAGE_CONTRACTS.values():
    TeamAgentSpec.model_validate(scope_team_spec(spec, contract))
from jiuwenswarm.agents.harness.research.team_runtime import OpenJiuwenTeamBackend
backend = OpenJiuwenTeamBackend(config=resolved, output_root=Path(os.environ['JIUWENSWARM_DATA_DIR']))
backend._ensure_direct_model()  # construct only; no paid request
for contract in TEAM_STAGE_CONTRACTS.values():
    stage_spec = backend._ensure_team_spec(contract.stage, contract.operation)
    assert stage_spec.lifecycle == 'temporary'
    for member in stage_spec.agents.values():
        assert not member.tools and not member.rails and not member.mcps
        assert not member.enable_sys_operation
        assert member.enable_read_image_multimodal is False
assert not backend._ensure_team_spec('topic_intake', 'research_question').predefined_members
backend._ensure_team_spec('topic_intake', 'research_question').build()
assert Path(cfg['experiment']['artifact_dir'], 'aggregate.json').is_file()
print('Local preflight passed: runtime, plotting, exporter, config and experiment artifacts.')
PY
if [[ -n "$resume_dir" ]]; then
  "$python_bin" -m examples.scitrace_research.resume_policy \
    --run-dir "$resume_dir" --config "$run_config_path" --topic "$research_topic"
fi
if $check_only; then
  echo 'No API calls made. No paper generated.'
  exit 0
fi
if [[ -z ${API_KEY:-} ]]; then
  read -r -s -p 'DeepSeek API key (hidden): ' API_KEY
  printf '\n'
fi
export API_KEY
stamp=$(date +%Y%m%d_%H%M%S)_$$
mkdir -p "$project_root/runs"
resume_args=()
if [[ -n "$resume_dir" ]]; then
  run_dir="$resume_dir"
  resume_args=(--resume)
else
  run_dir="$project_root/runs/final_$stamp"
  [[ ! -e "$run_dir" ]]
fi
# A resumed workflow restores scientific checkpoints, with a fresh Team runtime.
export JIUWENSWARM_DATA_DIR="$project_root/.runtime_cache/final_${stamp}_$$"
[[ ! -e "$JIUWENSWARM_DATA_DIR" ]]
mkdir -p "$JIUWENSWARM_DATA_DIR/config" "$JIUWENSWARM_DATA_DIR/agent/workspace"
cp "$project_root/jiuwenswarm/resources/config.yaml" "$JIUWENSWARM_DATA_DIR/config/config.yaml"
printf 'Run directory: %s\n' "$run_dir"
printf 'Live log command: tail -f "%s/console.log"\n' "$run_dir"
# Verify credentials and the exact requested model without generating tokens.
"$python_bin" -c 'import os, sys; from openai import OpenAI; c=OpenAI(api_key=os.environ["API_KEY"],base_url=os.environ["API_BASE"],timeout=20,max_retries=0); names={m.id for m in c.models.list()}; m=os.environ["MODEL_NAME"]; print("Requested model available:",m in names); sys.exit(0 if m in names else 2)'
if [[ -n "$resume_dir" ]]; then
  snapshot_dir="$run_dir/resume_history/${stamp}_$$"
  "$python_bin" -m examples.scitrace_research.resume_policy \
    --run-dir "$run_dir" --config "$run_config_path" --topic "$research_topic" \
    --snapshot-dir "$snapshot_dir"
else
  mkdir "$run_dir"
fi
event_offset=0
if [[ -f "$run_dir/team_events.jsonl" ]]; then
  event_offset=$(wc -c < "$run_dir/team_events.jsonl")
fi
report_failure() {
  # Only inspect events appended by this invocation, not prior resume errors.
  "$python_bin" - "$run_dir/team_events.jsonl" "$event_offset" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if path.exists():
    with path.open('rb') as handle:
        handle.seek(int(sys.argv[2]))
        for line in handle:
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if event.get('type') == 'provider_fatal_error':
                # These messages are fixed diagnostics, never raw provider text.
                print(event['error'], file=sys.stderr)
                break
PY
  echo "Run failed; inspect $run_dir/console.log and $run_dir/paper_export.log. Existing runs are preserved." >&2
}
trap report_failure ERR
# Keep long native model calls visible without parsing or exposing prompt content.
"$python_bin" -u -m examples.scitrace_research.watch_run --run-dir "$run_dir" --parent-pid "$$" &
watch_pid=$!
cleanup_watch() { kill "$watch_pid" 2>/dev/null || true; wait "$watch_pid" 2>/dev/null || true; }
trap cleanup_watch EXIT
"$python_bin" -u -m examples.scitrace_research.run_team \
  --topic "$research_topic" \
  --config "$run_config_path" \
  --output "$run_dir" --mode real_team --strict-real --no-mock-fallback "${resume_args[@]}" 2>&1 | tee -a "$run_dir/console.log"
"$python_bin" -u -m examples.scitrace_research.paper_export --run-dir "$run_dir" \
  > "$run_dir/paper_export.log" 2>&1
printf 'Generated paper: %s/paper/paper.pdf\n' "$run_dir"
