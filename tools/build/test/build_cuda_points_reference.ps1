#Requires -Version 5.1
param([string]$Build='build_cuda_points_reference',
    [string]$Cuda='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3',
    [string]$VcVars='')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if(-not [IO.Path]::IsPathRooted($Build)){$Build=Join-Path $repo $Build}
$Build=[IO.Path]::GetFullPath($Build).TrimEnd('\','/')
if($Build -ieq $repo -or $Build -eq [IO.Path]::GetPathRoot($Build).TrimEnd('\','/')){throw 'Invalid build directory'}
$marker=Join-Path $Build 'cuda_points_profile.txt'
if((Test-Path -LiteralPath $Build) -and -not (Test-Path -LiteralPath $marker) -and
    @(Get-ChildItem -LiteralPath $Build -Force | Where-Object {$_.Name -ne '.gitignore'}).Count){throw 'Choose an empty CUDA reference build directory'}
New-Item -ItemType Directory -Path $Build -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $Build '.gitignore'),"*`n")
if(-not $VcVars){$VcVars=Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat' |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName}
$profile="$repo`n$Cuda`n$VcVars"
if((Test-Path -LiteralPath $marker) -and (Get-Content -LiteralPath $marker -Raw) -ne $profile){throw 'CUDA reference build profile mismatch'}
[IO.File]::WriteAllText($marker,$profile)
$paths=@('tools/test/cuda_points_reference.cu','tools/test/cuda_points_reference.h',
    'src/ecm/stage2/cuda/device/point.cuh','src/ecm/stage2/cuda/device/point_mersenne.cuh',
    'src/ecm/stage2/cuda/launch.h','src/ecm/cuda/runtime/launch.h','src/ecm/cuda/dispatch/tiers.def',
    'tools/build/test/build_cuda_points_reference.ps1')
$hashes=[ordered]@{};foreach($p in $paths){$hashes[$p]=(Get-FileHash -LiteralPath (Join-Path $repo $p)).Hash}
$nvcc=Join-Path $Cuda 'bin/nvcc.exe';$dll=Join-Path $Build 'cuda_points_reference.dll'
$commands=@('@echo off',('call "{0}" >nul 2>&1' -f $VcVars),'if errorlevel 1 exit /b 1',
    ('"{0}" --shared -O3 --std=c++17 --allow-unsupported-compiler -arch=sm_89 -Xcompiler=/MD,/utf-8 -I"{1}/src" "{1}/tools/test/cuda_points_reference.cu" -o "{2}"' -f $nvcc,$repo.Replace('\','/'),$dll),'exit /b %errorlevel%')
$cmd=Join-Path $Build 'build.cmd';[IO.File]::WriteAllText($cmd,($commands -join "`r`n")+"`r`n")
Push-Location $Build
try{& cmd.exe /d /c $cmd 2>&1 | Tee-Object -FilePath (Join-Path $Build 'build.log');if($LASTEXITCODE){throw 'CUDA reference build failed'}}finally{Pop-Location}
foreach($p in $paths){if((Get-FileHash -LiteralPath (Join-Path $repo $p)).Hash -ne $hashes[$p]){throw "Source changed during build: $p"}}
[ordered]@{complete=$true;architecture='sm89';nvcc=$nvcc;nvcc_sha256=(Get-FileHash $nvcc).Hash;
    dll_sha256=(Get-FileHash $dll).Hash;sources=$hashes} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Build 'build_manifest.json') -Encoding utf8
