param(
    [Parameter(Mandatory=$true)][string]$TeamName,
    [string]$AccessToken = "",
    [string]$PrUrl = "",
    [string]$ResumeRunDir = "",
    [string]$SubmissionOutputRoot = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRequired = $true
if ([string]::IsNullOrWhiteSpace($ResumeRunDir)) {
    $RunDir = Join-Path $RepoRoot "runs\route_b_team_real_$Timestamp"
    if (Test-Path -LiteralPath $RunDir) {
        throw "Refusing to overwrite an existing Route B run directory: $RunDir"
    }
    $ResumeArgs = @()
} else {
    if (-not (Test-Path -LiteralPath $ResumeRunDir -PathType Container)) {
        throw "Resume run directory does not exist: $ResumeRunDir"
    }
    $RunDir = (Resolve-Path -LiteralPath $ResumeRunDir).Path
    $ResumeArgs = @("--resume")
    $ManifestPath = Join-Path $RunDir "run_manifest.json"
    $GatePath = Join-Path $RunDir "validation_report.json"
    if ((Test-Path -LiteralPath $ManifestPath) -and (Test-Path -LiteralPath $GatePath)) {
        $Manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
        $Gate = Get-Content -Raw -LiteralPath $GatePath | ConvertFrom-Json
        if ($Manifest.status -eq "success" -and $Gate.ok -eq $true) {
            $RunRequired = $false
            Write-Host "Using completed Route B run without model calls: $RunDir" -ForegroundColor Cyan
        }
    }
    if ($RunRequired) {
        Write-Host "Resuming incomplete Route B run: $RunDir" -ForegroundColor Cyan
    }
}
$SubmissionOutputRoot = if ([string]::IsNullOrWhiteSpace($SubmissionOutputRoot)) {
    Join-Path $RepoRoot "submission_build_route_b_$Timestamp"
} else {
    $SubmissionOutputRoot
}
if (Test-Path -LiteralPath $SubmissionOutputRoot) {
    throw "Refusing to overwrite an existing Route B submission directory: $SubmissionOutputRoot"
}
$Topic = "Calibrating retriever disagreement for budgeted scientific evidence retrieval across SciFact and NFCorpus"

if ($RunRequired) {
    foreach ($Name in @("API_BASE", "API_KEY", "MODEL_NAME", "MODEL_PROVIDER")) {
        $Value = Get-Item "Env:$Name" -ErrorAction SilentlyContinue
        if ($null -eq $Value -or [string]::IsNullOrWhiteSpace($Value.Value)) {
            throw "Environment variable $Name is required to run the Team workflow."
        }
    }
    $RunArgs = @(
        "-m", "examples.scitrace_research.run_team",
        "--topic", $Topic,
        "--config", ".\examples\scitrace_research\configs\team_real_route_b.yaml",
        "--output", $RunDir,
        "--mode", "real_team",
        "--strict-real",
        "--no-mock-fallback"
    ) + $ResumeArgs
    & ".\.venv_real\Scripts\python.exe" @RunArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Route B strict-real Team workflow failed. Inspect $RunDir"
    }
}

$PaperPath = Join-Path $RunDir "paper\paper.pdf"
$PaperManifestPath = Join-Path $RunDir "paper\paper_manifest.json"
if ((Test-Path -LiteralPath $PaperPath) -and (Test-Path -LiteralPath $PaperManifestPath)) {
    & ".\.venv_real\Scripts\python.exe" -c "import sys; from pathlib import Path; from examples.scitrace_research.paper_export import verify; verify(Path(sys.argv[1]))" $RunDir
    if ($LASTEXITCODE -ne 0) {
        throw "Existing paper binding is invalid. Inspect $RunDir before regenerating the PDF."
    }
    Write-Host "Using existing verified PDF; its bytes are unchanged." -ForegroundColor Cyan
} else {
    & ".\.venv_real\Scripts\python.exe" -m examples.scitrace_research.paper_export --run-dir $RunDir
    if ($LASTEXITCODE -ne 0) {
        throw "Paper export failed. Inspect $RunDir"
    }
}

$PaperHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PaperPath).Hash
Write-Host "Route B paper ready: $PaperPath" -ForegroundColor Green
Write-Host "Paper SHA-256: $PaperHash" -ForegroundColor Green

if ([string]::IsNullOrWhiteSpace($AccessToken)) {
    Write-Host "Upload this exact PDF to Stanford Agentic Reviewer, wait for its report, and retain the token issued for this PDF."
    Write-Host "Then build the submission from this same run directory; do not rerun the Team workflow."
    Write-Host "Run directory: $RunDir"
    exit 0
}

$BuildArgs = @(
    "-m", "examples.scitrace_research.build_submission",
    "--run-dir", $RunDir,
    "--output-root", $SubmissionOutputRoot,
    "--team-name", $TeamName,
    "--access-token", $AccessToken
)
if (-not [string]::IsNullOrWhiteSpace($PrUrl)) {
    $BuildArgs += @("--pr-url", $PrUrl)
}
& ".\.venv_real\Scripts\python.exe" @BuildArgs
if ($LASTEXITCODE -ne 0) {
    throw "Submission package build failed. Inspect $RunDir"
}
Write-Host "Submission ZIP ready: $SubmissionOutputRoot\$TeamName.zip" -ForegroundColor Green
