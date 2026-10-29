#Requires -Version 5.1
param([string]$Build='build_cuda_stage2_tail_probe',
    [string]$Cuda='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3',
    [string]$VcVars='')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if(-not [IO.Path]::IsPathRooted($Build)){$Build=Join-Path $repo $Build}
$Build=[IO.Path]::GetFullPath($Build).TrimEnd('\','/')
if($Build -ieq $repo -or $Build -eq [IO.Path]::GetPathRoot($Build).TrimEnd('\','/')){throw 'Invalid build directory'}
if(Test-Path -LiteralPath $Build){throw 'Choose a new build directory to preserve previous evidence'}
New-Item -ItemType Directory -Path $Build | Out-Null
[IO.File]::WriteAllText((Join-Path $Build '.gitignore'),"*`n")
if(-not $VcVars){$VcVars=Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat' |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName}
$paths=@('tools/test/cuda_stage2_tail_probe.cu','tools/test/opencl_stage2_oracle.h',
    'src/ecm/stage2/cuda/device/point.cuh','src/ecm/stage2/cuda/device/point_mersenne.cuh',
    'src/ecm/stage2/cuda/device/baby_giant.cuh','src/ecm/stage2/cuda/launch.h',
    'src/ecm/cuda/runtime/launch.h','src/ecm/cuda/dispatch/tiers.def','tools/build/test/build_cuda_stage2_tail_probe.ps1')
$hashes=[ordered]@{};foreach($p in $paths){$hashes[$p]=(Get-FileHash -LiteralPath (Join-Path $repo $p)).Hash}
$gmp=Join-Path $repo 'third_party/gmp-zen3/dist';$nvcc=Join-Path $Cuda 'bin/nvcc.exe';$exe=Join-Path $Build 'cuda_stage2_tail_probe.exe'
$commands=@('@echo off',('call "{0}" >nul 2>&1' -f $VcVars),'if errorlevel 1 exit /b 1',
    ('"{0}" -O3 --std=c++17 --allow-unsupported-compiler -arch=sm_89 -Xcompiler=/MD,/utf-8 -I"{1}/src" -I"{2}/include" "{1}/tools/test/cuda_stage2_tail_probe.cu" "{2}/lib/gmp.lib" -o "{3}"' -f $nvcc,$repo.Replace('\','/'),$gmp.Replace('\','/'),$exe),'exit /b %errorlevel%')
$cmd=Join-Path $Build 'build.cmd';[IO.File]::WriteAllText($cmd,($commands -join "`r`n")+"`r`n")
Push-Location $Build
try{& cmd.exe /d /c $cmd 2>&1 | Tee-Object -FilePath (Join-Path $Build 'build.log');if($LASTEXITCODE){throw 'CUDA tail probe build failed'}}finally{Pop-Location}
foreach($p in $paths){if((Get-FileHash -LiteralPath (Join-Path $repo $p)).Hash -ne $hashes[$p]){throw "Source changed during build: $p"}}
Copy-Item -LiteralPath (Join-Path $gmp 'bin/gmp-10.dll') -Destination $Build
[ordered]@{complete=$true;architecture='sm89';nvcc=$nvcc;nvcc_sha256=(Get-FileHash $nvcc).Hash;
    executable_sha256=(Get-FileHash $exe).Hash;gmp_sha256=(Get-FileHash (Join-Path $Build 'gmp-10.dll')).Hash;
    gmp_header_sha256=(Get-FileHash (Join-Path $gmp 'include/gmp.h')).Hash;
    gmp_import_lib_sha256=(Get-FileHash (Join-Path $gmp 'lib/gmp.lib')).Hash;sources=$hashes} |
    ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Build 'build_manifest.json') -Encoding utf8
