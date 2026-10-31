# The Gemini few-shot baseline for lora-vs-prompting: 300 test utterances x 3 pool sizes = 900 calls
# on the free tier. Safe to run again: finished answers are kept, so if the daily quota stops it,
# run it again after the reset (5 a.m. in Argentina).
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_gemini.ps1 -Limit 5   # a first look
#   powershell -ExecutionPolicy Bypass -File scripts\run_gemini.ps1            # everything
param([int]$Limit = 0)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Failed: $exe $($argv -join ' ')" }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python not found. Install Python 3.11 or newer from python.org (tick 'Add to PATH')." }
if (-not (Test-Path ".venv")) { Write-Host "Creating the virtual environment..."; Run python @("-m", "venv", ".venv") }
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path ".venv\.installed")) {
    Run $py @("-m", "pip", "install", "--quiet", "--upgrade", "pip")
    Run $py @("-m", "pip", "install", "--quiet", "-e", ".[gemini,dev]")
    Run $py @("-m", "pytest", "-q")
    New-Item ".venv\.installed" -ItemType File | Out-Null
}
if (-not (Test-Path "data\test.jsonl")) { Run $py @("-m", "lvp.cli", "prepare") }

if (-not ((Test-Path ".env") -and (Select-String -Path ".env" -Pattern "^GEMINI_API_KEY=." -Quiet))) {
    $src = Get-ChildItem "..\agent-benchmark\.env", "..\agent-injection-lab\.env" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($src) { $line = Select-String -Path $src.FullName -Pattern "^GEMINI_API_KEY=." | Select-Object -First 1; if ($line) { Add-Content ".env" $line.Line; Write-Host "Reusing the Gemini key from $($src.FullName)" } }
}
if (-not ((Test-Path ".env") -and (Select-String -Path ".env" -Pattern "^GEMINI_API_KEY=." -Quiet))) {
    $key = Read-Host "Paste your Gemini API key (aistudio.google.com/apikey)"
    Add-Content ".env" "GEMINI_API_KEY=$key"
}

$argv = @("-m", "lvp.cli", "gemini")
if ($Limit -gt 0) { $argv += @("--limit", "$Limit") }
Run $py $argv
Write-Host "`nWhen it says 'Gemini runs complete', send the files in results\preds."
