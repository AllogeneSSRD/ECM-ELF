#Requires -Version 5.1
param([string]$Build = 'build_dev/cpu_stage1_weighted_ntt_probe', [string]$Gmp = '',
      [string]$GwNum = '', [string]$VcVars = '')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/gmp_dependency.ps1')
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
$dependency = Resolve-EcmGmp -Repo $repo -Gmp $Gmp
if (-not $GwNum) { $GwNum = Join-Path $repo '.refactor/p95v3106b01.source/gwnum' }
if (-not $VcVars) {
    $VcVars = Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat' |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $VcVars -or -not (Test-Path -LiteralPath $VcVars)) { throw 'MSVC x64 environment not found' }
if (-not (Test-Path -LiteralPath (Join-Path $GwNum 'gwnum64.lib'))) { throw 'gwnum64.lib not found' }
$Build = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
Initialize-EcmBuildDirectory $Build
$exe = Join-Path $Build 'cpu_stage1_weighted_ntt.exe'
$obj = Join-Path $Build 'cpu_stage1_weighted_ntt.obj'
$batch = Join-Path $Build 'build.cmd'
$commands = @('@echo off',
    ('call "{0}" >nul 2>&1' -f $VcVars), 'if errorlevel 1 exit /b 1',
    ('cl /nologo /O2 /MT /std:c++17 /EHsc /utf-8 /arch:AVX512 /I "{0}/include" /I "{1}" tools/bench/cpu_stage1_weighted_ntt.cpp /Fo"{2}" /Fe"{3}" /link /LIBPATH:"{0}/lib" gmp.lib "{1}/gwnum64.lib" advapi32.lib' -f $dependency.Root,$GwNum,$obj,$exe),
    'exit /b %errorlevel%')
[IO.File]::WriteAllText($batch, ($commands -join "`r`n") + "`r`n", [Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch > (Join-Path $Build 'build.log') 2>&1
if ($LASTEXITCODE) {
    Get-Content -LiteralPath (Join-Path $Build 'build.log') -Tail 25
    throw 'CPU weighted NTT probe build failed'
}
Copy-Item -LiteralPath (Join-Path $dependency.Root 'bin/gmp-10.dll') -Destination $Build -Force
Write-Host "Built CPU weighted NTT probe at $exe"
