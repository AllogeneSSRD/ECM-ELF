#Requires -Version 5.1
param([string]$BuildDir='build_gui_v2_local',[Parameter(Mandatory=$true)][string]$Output,[string]$Python='python')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
& $Python (Join-Path $repo 'tools/packaging/package_gui_v2.py') --exe (Join-Path $BuildDir 'ecm_gui_v2.exe') --out $Output
if($LASTEXITCODE){throw 'GUI v2 packaging failed'}
