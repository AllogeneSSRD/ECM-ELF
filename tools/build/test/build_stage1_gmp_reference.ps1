#Requires -Version 5.1
# Host-only independent Stage1 point oracle. Use a fresh build directory.
param([string]$Build='build_cuda_cmake/stage1_gmp_reference', [string]$Gmp='', [string]$VcVars='')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/gmp_dependency.ps1')
$dependency=Resolve-EcmGmp -Repo $repo -Gmp $Gmp
if(-not $VcVars){$VcVars=(Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName}
if(-not $VcVars -or -not (Test-Path -LiteralPath $VcVars)){throw 'vcvars64.bat not found; pass -VcVars'}
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $Build){throw 'Use a fresh build directory to preserve reference evidence'}
New-Item -ItemType Directory -Path $Build | Out-Null
$sources=@('tools/bench/stage1_gmp_reference.cpp','tools/build/test/build_stage1_gmp_reference.ps1')
$hashes=[ordered]@{}
foreach($source in $sources){$hashes[$source]=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash}
$exe=Join-Path $Build 'stage1_gmp_reference.exe';$obj=Join-Path $Build 'stage1_gmp_reference.obj'
$log=Join-Path $Build 'build.log';$script=Join-Path $Build 'build.cmd'
$line="@echo off`r`ncall `"$VcVars`" >nul 2>&1`r`nif errorlevel 1 exit /b 1`r`n"+
    "cl /nologo /O2 /std:c++17 /EHsc /I `"$($dependency.Root)/include`" tools/bench/stage1_gmp_reference.cpp /Fe:`"$exe`" /Fo:`"$obj`" /link /LIBPATH:`"$($dependency.Root)/lib`" gmp.lib`r`nexit /b %errorlevel%`r`n"
[IO.File]::WriteAllText($script,$line,[Text.UTF8Encoding]::new($false))
$timer=[Diagnostics.Stopwatch]::StartNew()
& cmd.exe /d /c "`"$script`"" > $log 2>&1
if($LASTEXITCODE -ne 0){Get-Content -LiteralPath $log -Tail 25;throw 'Stage1 reference compile/link failed'}
foreach($source in $sources){
    if((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $hashes[$source]){throw "Source changed during build: $source"}
    $dest=Join-Path $Build "sources/$source";New-Item -ItemType Directory -Force (Split-Path -Parent $dest) | Out-Null
    Copy-Item -LiteralPath $source -Destination $dest
}
Copy-Item -LiteralPath (Join-Path $dependency.Root 'bin/gmp-10.dll') -Destination $Build
[ordered]@{binary_sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash;sources=$hashes;
    gmp=$dependency.Hashes;build_seconds=$timer.Elapsed.TotalSeconds} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Build 'manifest.json') -Encoding UTF8
Write-Host "Built independent Stage1 GMP reference ($([math]::Round($timer.Elapsed.TotalSeconds,1)) s): $exe"
