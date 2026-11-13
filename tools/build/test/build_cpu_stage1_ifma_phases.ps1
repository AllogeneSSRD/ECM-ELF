#Requires -Version 5.1
param([string]$Build = 'build_dev/cpu_stage1_ifma_phases', [string]$Gmp = '',
      [string]$VcVars = '',
      [string]$Source = 'src/ecm/cpu/arithmetic/ifma/modular_ifma.cpp')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/gmp_dependency.ps1')
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
$dependency = Resolve-EcmGmp -Repo $repo -Gmp $Gmp
if (-not $VcVars) {
    $VcVars = Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat' |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $VcVars -or -not (Test-Path -LiteralPath $VcVars)) { throw 'MSVC x64 environment not found' }
$Build = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
Initialize-EcmBuildDirectory $Build
$exe = Join-Path $Build 'mers_sqr_phases.exe'
$obj = Join-Path $Build 'mers_sqr_phases.obj'
$batch = Join-Path $Build 'build.cmd'
$generated = Join-Path $Build 'ifma_phase_kernels.inc'
Copy-Item -LiteralPath $Source -Destination (Join-Path $Build 'ifma_phase_source.inc') -Force
& python tools/bench/instrument_ifma_phases.py $Source $generated
if ($LASTEXITCODE) { throw 'Cannot instrument current IFMA phases' }
$commands = @('@echo off',
    ('call "{0}" >nul 2>&1' -f $VcVars), 'if errorlevel 1 exit /b 1',
    ('cl /nologo /O2 /MT /std:c++17 /EHsc /utf-8 /arch:AVX512 /I "{0}/include" /I "{1}" /I src/ecm/cpu/arithmetic/ifma tools/bench/mers_sqr_phases.cpp /Fo"{2}" /Fe"{3}" /link /LIBPATH:"{0}/lib" gmp.lib' -f $dependency.Root,$Build,$obj,$exe),
    'exit /b %errorlevel%')
[IO.File]::WriteAllText($batch, ($commands -join "`r`n") + "`r`n", [Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch > (Join-Path $Build 'build.log') 2>&1
if ($LASTEXITCODE) {
    Get-Content -LiteralPath (Join-Path $Build 'build.log') -Tail 25
    throw 'CPU IFMA phase probe build failed'
}
Copy-Item -LiteralPath (Join-Path $dependency.Root 'bin/gmp-10.dll') -Destination $Build -Force
Write-Host "Built CPU IFMA phase probe at $exe"
