<#
Run one or more steps of a Milestone 2 notebook headlessly (Windows PowerShell).

  .\scripts\run_step.ps1 -Notebook M2_Boxes.ipynb  -Steps stage1_test
  .\scripts\run_step.ps1 -Notebook M2_Detect.ipynb -Steps "compose,check"
  .\scripts\run_step.ps1 -Notebook M2_Boxes.ipynb  -Reviewer SC -Mode verify -Splits test     # opens the review window

Uses "python -m jupyter" so it works even when the jupyter command is not on PATH.
Every review variable is set explicitly and removed afterwards (an empty value in PowerShell
removes a variable, which would silently fall back to the notebook's default).
#>
param(
  [Parameter(Mandatory = $true)][string]$Notebook,
  [string]$Steps = "",
  [string]$Reviewer = "", [string]$Mode = "", [string]$Classes = "", [string]$Splits = "",
  [int]$Sample = 0, [switch]$Redo,
  [string]$DetectSource = "", [string]$DetectConf = "", [string]$DetectImgsz = ""
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$vars = @{
  M2_RUN = $Steps; M2_REVIEWER = $Reviewer; M2_REVIEW_MODE = $Mode; M2_REVIEW_CLASSES = $Classes;
  M2_REVIEW_SPLITS = $Splits; M2_REVIEW_SAMPLE = "$Sample"; M2_REVIEW_REDO = $(if ($Redo) { "1" } else { "0" });
  M2_DETECT_SOURCE = $DetectSource; M2_DETECT_CONF = $DetectConf; M2_DETECT_IMGSZ = $DetectImgsz
}
foreach ($k in $vars.Keys) {
  if ($vars[$k] -ne "") { Set-Item -Path "Env:$k" -Value $vars[$k] }
  else { Remove-Item -Path "Env:$k" -ErrorAction SilentlyContinue }
}
if (-not $Reviewer) { foreach ($k in "M2_REVIEW_MODE", "M2_REVIEW_SAMPLE", "M2_REVIEW_REDO") { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } }

$out = [IO.Path]::GetFileNameWithoutExtension($Notebook) + "_run.ipynb"
Write-Host "[$(Get-Date -Format T)] $Notebook | steps: '$Steps' | reviewer: '$Reviewer' $Mode"
try {
  python -m jupyter nbconvert --to notebook --execute $Notebook --output $out --ExecutePreprocessor.timeout=-1
  if ($LASTEXITCODE -ne 0) { throw "nbconvert exited with code $LASTEXITCODE (see milestone2_outputs\logs\)" }
  Write-Host "[$(Get-Date -Format T)] done"
}
finally {
  foreach ($k in $vars.Keys) { Remove-Item -Path "Env:$k" -ErrorAction SilentlyContinue }
}
