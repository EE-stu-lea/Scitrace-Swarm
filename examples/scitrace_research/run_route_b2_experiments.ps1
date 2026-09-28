param(
    [string]$DenseModel = "BAAI/bge-small-en-v1.5",
    [int]$BootstrapSamples = 1000,
    [int]$Seed = 42,
    [switch]$IncludeFiqa
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv_real\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing virtual environment: $Python"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $RepoRoot "runs\route_b2_final_$Stamp"
$CacheRoot = Join-Path $RepoRoot "runs\route_b2_cache"
New-Item -ItemType Directory -Force -Path $RunRoot, $CacheRoot | Out-Null

$Datasets = @(
    @{ Name = "scifact"; Path = "examples\scitrace_research\data\beir\scifact" },
    @{ Name = "nfcorpus"; Path = "examples\scitrace_research\data\beir\nfcorpus\nfcorpus" }
)
if ($IncludeFiqa) {
    $Datasets += @{ Name = "fiqa"; Path = "examples\scitrace_research\data\beir\fiqa" }
}

$CompletedRuns = @()
foreach ($Dataset in $Datasets) {
    $OutputDir = Join-Path $RunRoot $Dataset.Name
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $CachePath = Join-Path $CacheRoot ($Dataset.Name + "_bge_small_en_v1_5.npz")
    $MaxTrainQueries = if ($Dataset.Name -eq "fiqa") { 1000 } else { 0 }
    & $Python -m examples.scitrace_research.workflow.route_b2_experiment `
        --dataset-dir $Dataset.Path `
        --split test `
        --budgets 2,3,5,8 `
        --bootstrap-samples $BootstrapSamples `
        --seed $Seed `
        --dense-model $DenseModel `
        --dense-batch-size 16 `
        --dense-cache $CachePath `
        --max-train-queries $MaxTrainQueries `
        --output (Join-Path $OutputDir "results.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Route B2 failed for $($Dataset.Name). Inspect $OutputDir"
    }
    $CompletedRuns += $OutputDir
}

& $Python -m examples.scitrace_research.workflow.route_b2_report `
    @CompletedRuns `
    --samples $BootstrapSamples `
    --seed $Seed `
    --output (Join-Path $RunRoot "aggregate.json")
if ($LASTEXITCODE -ne 0) {
    throw "Route B2 aggregation failed. Inspect $RunRoot"
}

Write-Host "Route B2 artifacts ready: $RunRoot"
Write-Host "Aggregate report: $(Join-Path $RunRoot 'aggregate.md')"
