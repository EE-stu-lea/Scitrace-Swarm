param(
    [Parameter(Mandatory=$true)][string]$TeamName,
    [string]$AccessToken = "",
    [string]$PrUrl = "",
    [string]$Topic = "Evidence-aware context governance for autonomous scientific research agents",
    [string]$ResumeRunDir = "",
    [string]$SubmissionOutputRoot = ".\submission_build"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

foreach ($Name in @("API_BASE", "API_KEY", "MODEL_NAME", "MODEL_PROVIDER")) {
    if (-not (Test-Path "Env:$Name") -or [string]::IsNullOrWhiteSpace((Get-Item "Env:$Name").Value)) {
        throw "Environment variable $Name is required."
    }
}

if ([string]::IsNullOrWhiteSpace($ResumeRunDir)) {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunDir = Join-Path $RepoRoot "runs\team_real_submission_$Timestamp"
    $ResumeArgs = @()
} else {
    if (-not (Test-Path -LiteralPath $ResumeRunDir -PathType Container)) {
        throw "Resume run directory does not exist: $ResumeRunDir"
    }
    $RunDir = (Resolve-Path -LiteralPath $ResumeRunDir).Path
    $ResumeArgs = @("--resume")
    Write-Host "Resuming existing run: $RunDir"
}

$RunArgs = @(
    "-m", "examples.scitrace_research.run_team",
    "--topic", $Topic,
    "--config", ".\examples\scitrace_research\configs\team_real.yaml",
    "--output", $RunDir,
    "--mode", "real_team",
    "--strict-real",
    "--no-mock-fallback"
) + $ResumeArgs
& ".\.venv_real\Scripts\python.exe" @RunArgs
if ($LASTEXITCODE -ne 0) { throw "strict-real Team workflow failed with exit code $LASTEXITCODE. Inspect $RunDir" }

& ".\.venv_real\Scripts\python.exe" -m examples.scitrace_research.paper_export --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "Paper export failed: $RunDir" }
Write-Host "Paper ready: $RunDir\paper\paper.pdf"

$HasToken = -not [string]::IsNullOrWhiteSpace($AccessToken)
$HasPr = -not [string]::IsNullOrWhiteSpace($PrUrl)
if ($HasToken) {
    $BuildArgs = @(
        "-m", "examples.scitrace_research.build_submission",
        "--run-dir", $RunDir,
        "--output-root", $SubmissionOutputRoot,
        "--team-name", $TeamName,
        "--access-token", $AccessToken
    )
    if ($HasPr) {
        $BuildArgs += @("--pr-url", $PrUrl)
    }
    & ".\.venv_real\Scripts\python.exe" @BuildArgs
    if ($LASTEXITCODE -ne 0) { throw "Submission package build failed: $RunDir" }
    Write-Host "Submission ZIP ready: $SubmissionOutputRoot\$TeamName.zip"
} else {
    Write-Host "Upload this exact PDF to Stanford Agentic Reviewer and create the official JiuwenSwarm PR."
    Write-Host "Then rerun build_submission with --access-token. Add --pr-url only if a public PR is available."
}
