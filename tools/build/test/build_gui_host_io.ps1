# Isolated GUI compatibility build for shared TOML and P95 pending display.
param([Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Force -Path $Build|Out-Null
[IO.File]::WriteAllText((Join-Path $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build) '.gitignore'),"*`n")
$reconfigure=-not (Test-Path -LiteralPath (Join-Path $Build 'Makefile'))
& (Join-Path $PSScriptRoot '../dev/build_gui.ps1') -BuildDir $Build -Reconfigure:$reconfigure -Targets @('ecm_gui','ecm_gui_gen_test','ecm_gui_results_test','ecm_gui_log_parse_test')
if($LASTEXITCODE){throw 'GUI host I/O compatibility build failed'}
