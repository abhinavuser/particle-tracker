#!/usr/bin/env powershell
# DRONE TRACKER - EASY RUN SCRIPT
# Just double-click this file to run!

# Deactivate any active conda/venv environment
Write-Host "Resetting environment..." -ForegroundColor Gray
$env:CONDA_PREFIX = ""
$env:VIRTUAL_ENV = ""

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "        DRONE TRACKING SYSTEM" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$conda = "C:\Users\chipn\Miniconda3\Scripts\conda.exe"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$videoDir = Join-Path $projectDir "videos"
$outputDir = Join-Path $projectDir "outputs"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Project:  $projectDir" -ForegroundColor White
Write-Host "  Videos:   $videoDir" -ForegroundColor White
Write-Host "  Outputs:  $outputDir" -ForegroundColor White
Write-Host "  Conda:    $conda" -ForegroundColor White
Write-Host ""

# Check if conda exists
if (!(Test-Path $conda)) {
    Write-Host "ERROR: Conda not found at: $conda" -ForegroundColor Red
    pause
    exit 1
}

# Check videos
$videoCount = (Get-ChildItem -Path $videoDir -Filter "*.mp4" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($videoCount -eq 0) {
    Write-Host "ERROR: No MP4 videos found in videos folder!" -ForegroundColor Red
    Write-Host "Please add video files to: $videoDir" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "Videos to process: $videoCount" -ForegroundColor Green
Write-Host ""

# Create output directory
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
    Write-Host "Created output directory" -ForegroundColor Green
}

# Run tracker directly with full Python path
Write-Host "Starting tracker..." -ForegroundColor Yellow
Write-Host "Processing will take 3-10 minutes depending on video length..." -ForegroundColor Yellow
Write-Host ""

Set-Location $projectDir

# Use conda run with explicit environment
& "$conda" run -n drone39 python -m src.main `
    --input-dir videos `
    --output-dir outputs `
    --weights yolov8n.pt `
    --conf 0.15 `
    --scale 0.6

# Result check
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=========== SUCCESS ===========" -ForegroundColor Green
    Write-Host "Results saved to outputs folder" -ForegroundColor Green
} else {
    Write-Host "ERROR: Tracker failed!" -ForegroundColor Red
}

Write-Host ""
pause
