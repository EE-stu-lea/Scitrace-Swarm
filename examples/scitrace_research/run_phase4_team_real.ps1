$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$venvActivate = Join-Path $repoRoot ".venv_real\Scripts\Activate.ps1"
$configPath = Join-Path $repoRoot "examples\scitrace_research\configs\team_real.yaml"

$requiredEnv = @("API_BASE", "API_KEY", "MODEL_NAME", "MODEL_PROVIDER")
$missing = @()
foreach ($name in $requiredEnv) {
    if (-not (Get-Item "Env:$name" -ErrorAction SilentlyContinue)) {
        $missing += $name
    }
}

if ($missing.Count -gt 0) {
    throw "Missing required environment variables: $($missing -join ', ')"
}

if (-not (Test-Path $venvActivate)) {
    throw "Virtual environment activation script not found: $venvActivate"
}

Set-Location $repoRoot
. $venvActivate

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = Join-Path $repoRoot "runs\team_real_accept_$timestamp"

Write-Host "Running Phase 4 strict-real Team workflow..." -ForegroundColor Cyan
Write-Host "Output directory: $outputDir" -ForegroundColor Cyan

python -m examples.scitrace_research.run_team `
  --topic "Evidence-aware context governance for scientific research agents" `
  --config $configPath `
  --output $outputDir `
  --mode real_team `
  --strict-real `
  --no-mock-fallback

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Phase 4 strict-real Team run finished." -ForegroundColor Green
Write-Host "Artifacts: $outputDir" -ForegroundColor Green
