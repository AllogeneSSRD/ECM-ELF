#Requires -Version 5.1
param([string]$Build='build_diagnostics_test',[string]$GmpRoot='third_party/gmp-zen3/dist',[string]$VcVars='', [switch]$WithoutLocalPari)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if(-not [IO.Path]::IsPathRooted($Build)){$Build=Join-Path $repo $Build}
if(-not [IO.Path]::IsPathRooted($GmpRoot)){$GmpRoot=Join-Path $repo $GmpRoot}
if(-not $VcVars){$VcVars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' | Select-Object -First 1).FullName}
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$toggle=if($WithoutLocalPari){'OFF'}else{'ON'}
$line="call `"$VcVars`" >nul 2>&1 && cmake -S `"$repo/src/ecm/diagnostics`" -B `"$Build`" -G `"NMake Makefiles`" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DECM_DIAGNOSTICS_GMP_ROOT=`"$GmpRoot`" -DECM_DIAGNOSTICS_LOCAL_PARI=$toggle && cmake --build `"$Build`""
& cmd.exe /d /c $line
if($LASTEXITCODE){throw 'Shared diagnostics build failed'}
