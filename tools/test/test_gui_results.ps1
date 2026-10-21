#Requires -Version 5.1
param([string]$Exe='build_gui/ecm_gui_results_test.exe',[string]$Output='data/experiments/gui_result')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$Exe=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Exe)
$Output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$run=Join-Path $Output ('run_'+[Guid]::NewGuid().ToString('N'))
& $Exe $run
if($LASTEXITCODE -ne 0){throw 'GUI Result reader/filter/copy tests failed'}
