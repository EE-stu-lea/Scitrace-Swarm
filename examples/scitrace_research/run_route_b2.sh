#!/usr/bin/env bash
set -euo pipefail

for variable_name in API_BASE API_KEY MODEL_NAME MODEL_PROVIDER; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "Missing required environment variable: ${variable_name}" >&2
    exit 2
  fi
done

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex is required to export the ICLR PDF." >&2
  exit 2
fi

run_dir="${1:-runs/route_b2_submission_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "${run_dir}" ]]; then
  echo "Refusing to overwrite an existing run directory: ${run_dir}" >&2
  exit 2
fi

python -m examples.scitrace_research.run_team \
  --topic "Continuous difficulty calibration for budgeted hybrid scientific evidence retrieval" \
  --config examples/scitrace_research/configs/team_real_route_b2.yaml \
  --output "${run_dir}" \
  --mode real_team --strict-real --no-mock-fallback

python -m examples.scitrace_research.paper_export --run-dir "${run_dir}"
echo "Generated Agent paper: ${run_dir}/paper/paper.pdf"
