#Requires -Version 5.1
# Standalone native OpenCL field/integer/S4 correctness probes. No CUDA compiler or production worker.
param([string]$Build='build_opencl_stage2_ntt', [string]$OpenClRoot='',
      [string]$Gmp='', [string]$VcVars='', [switch]$WorkerOnly)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
. (Join-Path $PSScriptRoot '../internal/gmp_dependency.ps1')
$dependency=Resolve-EcmGmp $repo $Gmp
if(-not $OpenClRoot){
    $candidates=@($env:OPENCL_ROOT,$env:CUDA_PATH,'D:/code/vcpkg/installed/x64-windows')
    $candidates+=@(Get-ChildItem 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA' -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -ExpandProperty FullName)
    $OpenClRoot=$candidates | Where-Object {$_ -and (Test-Path -LiteralPath (Join-Path $_ 'include/CL/cl.h')) -and
        ((Test-Path -LiteralPath (Join-Path $_ 'lib/x64/OpenCL.lib')) -or (Test-Path -LiteralPath (Join-Path $_ 'lib/OpenCL.lib')))} |
        Select-Object -First 1
}
if(-not $OpenClRoot){throw 'Pass -OpenClRoot <OpenCL headers/import library prefix>'}
$OpenClRoot=[IO.Path]::GetFullPath($OpenClRoot)
$openclLibrary=Join-Path $OpenClRoot 'lib/x64/OpenCL.lib'
if(-not (Test-Path -LiteralPath $openclLibrary)){$openclLibrary=Join-Path $OpenClRoot 'lib/OpenCL.lib'}
if(-not $VcVars){$VcVars=Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat' |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName}
if(-not $VcVars -or -not (Test-Path -LiteralPath $VcVars)){throw 'MSVC x64 environment not found'}
if(-not [IO.Path]::IsPathRooted($Build)){$Build=Join-Path $repo $Build}
$Build=[IO.Path]::GetFullPath($Build).TrimEnd('\','/')
if($Build -ieq $repo -or $Build -eq [IO.Path]::GetPathRoot($Build).TrimEnd('\','/')){throw 'Build directory cannot be a repository/filesystem root'}
$marker=Join-Path $Build 'opencl_ntt_profile.json'
$profile=[ordered]@{backend='native-opencl';role=$(if($WorkerOnly){'production-worker'}else{'probes-and-worker'});source_root=$repo;vcvars=$VcVars;
    opencl_root=$OpenClRoot;gmp_root=$dependency.Root}
if(Test-Path -LiteralPath $marker){
    $old=Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
    foreach($key in $profile.Keys){if($old.$key -ne $profile[$key]){throw "Build profile mismatch: $key"}}
} elseif((Test-Path -LiteralPath $Build) -and @(Get-ChildItem -LiteralPath $Build -Force |
    Where-Object {$_.Name -ne '.gitignore' -or $_.PSIsContainer}).Count){throw 'Choose an empty or matching OpenCL NTT build directory'}
New-Item -ItemType Directory -Path $Build -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $Build '.gitignore'),"*`n")
[IO.File]::WriteAllText($marker,($profile | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
$inputs=@('src/ecm/opencl/ntt/runtime.h','src/ecm/opencl/ntt/runtime.cpp','src/ecm/opencl/ntt/native.h',
    'src/ecm/stage2/opencl/polynomial.h','src/ecm/stage2/opencl/polynomial.cpp',
    'src/ecm/stage2/opencl/arithmetic.h','src/ecm/stage2/opencl/arithmetic.cpp',
    'src/ecm/stage2/opencl/tree.h','src/ecm/stage2/opencl/tree.cpp',
    'src/ecm/stage2/opencl/points.h','src/ecm/stage2/opencl/points.cpp',
    'src/ecm/opencl/resources/stage2/points.cl','tools/test/opencl_points_probe.cpp',
    'tools/test/cuda_points_reference.h',
    'src/ecm/opencl/resources/stage2/coefficients.cl',
    'tools/test/opencl_stage2_oracle.h','tools/test/opencl_stage2_probe.cpp',
    'src/ecm/opencl/resources/stage2/polynomial.cl','tools/test/opencl_s4_probe.cpp',
    'src/ecm/opencl/resources/ntt/goldilocks.clh','src/ecm/opencl/resources/ntt/transform.cl',
    'tools/test/opencl_ntt_probe.cpp','tools/experiments/opencl/ntt/CMakeLists.txt',
    'tools/build/test/build_opencl_stage2_ntt.ps1','include/ecm/platform/program_version.h')
# The production worker shares host input, queue, configuration and result code.
# Freeze their headers as well as translation units, including new local files.
foreach($relative in @('src/ecm/stage2','src/ecm/curve','src/ecm/platform','src/io',
    'src/ecm/application/configuration','src/ecm/group_order','src/ecm/diagnostics','include/ecm','config')){
    $inputs+=@(Get-ChildItem -LiteralPath (Join-Path $repo $relative) -Recurse -File |
        Where-Object {$_.Extension -in '.h','.cpp','.json','.def','.toml'} |
        ForEach-Object {$_.FullName.Substring($repo.Length+1).Replace('\','/')})
}
$inputs+=@('tools/build/internal/opencl_stage2.cmake','tools/build/internal/cuda_sources.cmake','src/ecm/stage1/selection.cpp')
$inputs=@($inputs | Sort-Object -Unique)
$hashes=[ordered]@{}
foreach($inputPath in $inputs){$hashes[$inputPath]=(Get-FileHash -LiteralPath (Join-Path $repo $inputPath)).Hash}
$cmake=(Get-Command cmake -ErrorAction Stop).Source
$source=Join-Path $repo 'tools/experiments/opencl/ntt'
$targets=if($WorkerOnly){'ecm_opencl_stage2'}else{'opencl_ntt_probe opencl_s4_probe opencl_stage2_probe opencl_points_probe ecm_opencl_stage2'}
$slash={param($p) $p.Replace('\','/')}
$configure='"{0}" -S "{1}" -B "{2}" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DECM_GMP_ROOT="{3}" -DOpenCL_INCLUDE_DIR="{4}/include" -DOpenCL_LIBRARY="{5}"' -f
    $cmake,(&$slash $source),(&$slash $Build),(&$slash $dependency.Root),(&$slash $OpenClRoot),(&$slash $openclLibrary)
$commands=@('@echo off',('call "{0}" >nul 2>&1' -f $VcVars),'if errorlevel 1 exit /b 1',
    $configure,'if errorlevel 1 exit /b 1',('"{0}" --build "{1}" --target {2}' -f $cmake,$Build,$targets),
    'exit /b %errorlevel%')
$batch=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($batch,($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
Write-Host '== Native OpenCL Stage2: compile NTT / integer / S4 / points / GCD probes =='
$saved=$ErrorActionPreference;$ErrorActionPreference='Continue'
$attemptLog=Join-Path $Build ('build_'+[DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_fffffff')+'.log')
try {& cmd.exe /d /c $batch 2>&1 | ForEach-Object {$line=$_.ToString();Write-Host $line;$line} |
    Out-File -LiteralPath $attemptLog -Encoding utf8;$code=$LASTEXITCODE}
finally {$ErrorActionPreference=$saved}
Copy-Item -LiteralPath $attemptLog -Destination (Join-Path $Build 'build.log') -Force
if($code){throw "OpenCL NTT build failed ($code); see $Build/build.log"}
foreach($inputPath in $inputs){if((Get-FileHash -LiteralPath (Join-Path $repo $inputPath)).Hash -ne $hashes[$inputPath]){
    throw "Source changed during OpenCL build: $inputPath"}}
$compilerEntry=Get-Content -LiteralPath (Join-Path $Build 'CMakeCache.txt') |
    Where-Object {$_ -match '^CMAKE_CXX_COMPILER:[^=]+='} | Select-Object -First 1
$compilerPath=$compilerEntry -replace '^CMAKE_CXX_COMPILER:[^=]+=',''
$dependencies=[ordered]@{compiler=(Get-FileHash -LiteralPath $compilerPath).Hash;
    cmake=(Get-FileHash -LiteralPath $cmake).Hash;opencl_library=(Get-FileHash -LiteralPath $openclLibrary).Hash}
foreach($name in $dependency.Hashes.Keys){$dependencies["gmp/$name"]=$dependency.Hashes[$name]}
foreach($header in Get-ChildItem -LiteralPath (Join-Path $OpenClRoot 'include/CL') -File){
    $dependencies["opencl_header/$($header.Name)"]=(Get-FileHash -LiteralPath $header.FullName).Hash}
$executables=[ordered]@{};foreach($target in $targets.Split(' ')){$executables[$target]=(Get-FileHash -LiteralPath (Join-Path $Build ($target+'.exe'))).Hash}
[ordered]@{complete=$true;backend='native-opencl';cuda_compiler_invocations=0;gpu_runs=0;
    compiler=$compilerPath;dependencies=$dependencies;
    executable_sha256=$executables['opencl_ntt_probe'];s4_executable_sha256=$executables['opencl_s4_probe'];
    stage2_executable_sha256=$executables['opencl_stage2_probe'];points_executable_sha256=$executables['opencl_points_probe'];
    worker_executable_sha256=$executables['ecm_opencl_stage2'];executables=$executables;sources=$hashes} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Build 'build_manifest.json') -Encoding utf8
Write-Host "done: $Build/ecm_opencl_stage2.exe (native OpenCL; shared production queue)"
