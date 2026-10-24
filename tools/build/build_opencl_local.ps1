#Requires -Version 5.1
<#
.SYNOPSIS
    Build the desktop OpenCL Stage1 worker in an isolated local profile.
.DESCRIPTION
    Uses ordinary CXX, the existing /MD optimized local policy, and native
    dependency tracking. Does not enable CUDA, GUI, tools or CPU Stage2.
    No release packaging is performed. The complete release orchestrator uses
    BuildRole=release for a separate portable worker with generic vcpkg GMP.
    BuildDir, when supplied, is a final leaf.
#>
param([string]$BuildDir='', [string]$Gmp='', [string]$OpenSslRoot='',
      [string]$OpenClRoot='', [string]$VcVars='',
      [ValidateSet('local','release')][string]$BuildRole='local', [switch]$SkipBuild)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $PSScriptRoot 'internal/gmp_dependency.ps1')
. (Join-Path $PSScriptRoot 'internal/cuda_build_profile.ps1')
$gmpDependency=Resolve-EcmGmp $repo $Gmp -Release:($BuildRole -eq 'release')
$Gmp=$gmpDependency.Root
if(-not $OpenSslRoot){$OpenSslRoot=if($BuildRole -eq 'release'){$Gmp}else{'D:/code/vcpkg/installed/x64-windows'}}
if(-not $OpenClRoot){
    $candidates=@($env:CUDA_PATH,$OpenSslRoot)
    $candidates+=@(Get-ChildItem 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -ExpandProperty FullName)
    $OpenClRoot=$candidates | Where-Object { $_ -and (Test-Path -LiteralPath (Join-Path $_ 'include/CL/cl.h')) -and ((Test-Path -LiteralPath (Join-Path $_ 'lib/x64/OpenCL.lib')) -or (Test-Path -LiteralPath (Join-Path $_ 'lib/OpenCL.lib'))) } | Select-Object -First 1
}
if(-not $OpenClRoot){throw 'OpenCL headers/import library not found; pass -OpenClRoot'}
$openClLibrary=Join-Path $OpenClRoot 'lib/x64/OpenCL.lib'
if(-not (Test-Path -LiteralPath $openClLibrary)){$openClLibrary=Join-Path $OpenClRoot 'lib/OpenCL.lib'}
if(-not $VcVars){
    $VcVars=Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat' | Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if(-not $VcVars -or -not (Test-Path -LiteralPath $VcVars)){throw 'MSVC x64 environment not found'}
$vcRoot=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $VcVars))
$tools=Get-ChildItem (Join-Path $vcRoot 'Tools/MSVC') -Directory | Where-Object {$_.Name -match '^14\.[0-9]+\.[0-9]+$'} | Sort-Object {[version]$_.Name} -Descending | Select-Object -First 1
$compiler=Join-Path $tools.FullName 'bin/Hostx64/x64/cl.exe'
$cmake=(Get-Command cmake -ErrorAction Stop).Source
$fields=[ordered]@{role=$BuildRole;backend='opencl';stage='stage1';source_root=$repo;generator='NMake Makefiles';configuration='optimized';crt='MD';vcvars=$VcVars;toolset=$tools.Name;compiler=$compiler;compiler_sha256=(Get-FileHash -LiteralPath $compiler).Hash;cmake=$cmake;cmake_sha256=(Get-FileHash -LiteralPath $cmake).Hash;gmp_root=$Gmp;openssl_root=[IO.Path]::GetFullPath($OpenSslRoot);opencl_root=[IO.Path]::GetFullPath($OpenClRoot)}
if($BuildRole -eq 'release'){$fields['portable_resources']='ON'}
foreach($name in $gmpDependency.Hashes.Keys){$fields["gmp/$name"]=$gmpDependency.Hashes[$name]}
foreach($path in @($openClLibrary,(Join-Path $OpenClRoot 'include/CL/cl.h'),(Join-Path $OpenSslRoot 'lib/libcrypto.lib'),(Join-Path $OpenSslRoot 'lib/libssl.lib'),(Join-Path $OpenSslRoot 'include/openssl/ssl.h'))){$fields["dependency/$path"]=(Get-FileHash -LiteralPath $path).Hash}
if($BuildRole -eq 'release'){
    foreach($name in @('bin/libcrypto-3-x64.dll','bin/libssl-3-x64.dll','share/openssl/copyright')){
        $path=Join-Path $OpenSslRoot $name;$fields["dependency/$path"]=(Get-FileHash -LiteralPath $path).Hash
    }
}
$hash=Get-EcmProfileHash $fields
if(-not $BuildDir){$BuildDir="build_opencl_$BuildRole/stage1/$($hash.Substring(0,16))"}
if(-not [IO.Path]::IsPathRooted($BuildDir)){$BuildDir=Join-Path $repo $BuildDir}
$BuildDir=[IO.Path]::GetFullPath($BuildDir).TrimEnd('\','/')
if($BuildDir -ieq $repo -or $BuildDir -eq [IO.Path]::GetPathRoot($BuildDir).TrimEnd('\','/')){throw 'Repository/filesystem roots cannot be build directories'}
foreach($part in $BuildDir.Split([char[]]'\/')){
    if(($part -match '^build_(?:cuda|cpu|gui)_' -and -not ($BuildRole -eq 'release' -and $part -eq 'build_cuda_release')) -or
       ($part -match '^build_opencl_(local|dev|release)$' -and $Matches[1] -ne $BuildRole)){throw 'Build directory belongs to another backend or role'}
}
$buildProfile=[pscustomobject]@{Hash=$hash;Fields=$fields}
$marker=Join-Path $BuildDir 'profile.json'
if(-not (Test-Path -LiteralPath $marker) -and (Test-Path -LiteralPath $BuildDir) -and @(Get-ChildItem -LiteralPath $BuildDir -Force).Count){throw 'Unmarked build directory; choose a fresh leaf'}
Assert-EcmBuildProfile $BuildDir $buildProfile -ReadOnly:$SkipBuild
$payloadTool=Join-Path $repo 'tools/packaging/opencl_release_payload.py'
if($SkipBuild){
    if($BuildRole -ne 'release'){throw 'OpenCL SkipBuild requires BuildRole release'}
    & python $payloadTool --build $BuildDir --verify $Gmp
    if($LASTEXITCODE){throw 'OpenCL release identity verification failed'}
    exit 0
}
$snapshot=Join-Path $BuildDir 'release-inputs.json'
if($BuildRole -eq 'release'){
    & python $payloadTool --build $BuildDir --snapshot $snapshot
    if($LASTEXITCODE){throw 'Cannot freeze OpenCL release inputs'}
}
$unix={param($path) $path.Replace('\','/')}
$configure='"{0}" -S "{1}" -B "{2}" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER="{3}" -DCMAKE_CXX_COMPILER="{3}" -DCMAKE_CUDA_COMPILER=NOTFOUND -DECM_ENABLE_CUDA=OFF -DECM_BUILD_GUI=OFF -DECM_BUILD_TOOLS=OFF -DECM_BUILD_CPU_STAGE2=OFF -DECM_WINDOWS_GMP_ROOT="{4}" -DOPENSSL_ROOT_DIR="{5}" -DOpenCL_INCLUDE_DIR="{6}/include" -DOpenCL_LIBRARY="{7}"' -f $cmake,(&$unix $repo),(&$unix $BuildDir),(&$unix $compiler),(&$unix $Gmp),(&$unix $OpenSslRoot),(&$unix $OpenClRoot),(&$unix $openClLibrary)
if($BuildRole -eq 'release'){$configure+=' -DECM_OPENCL_PORTABLE_RESOURCES=ON'}
$batch=Join-Path $BuildDir 'build.cmd'
$commands=@('@echo off',('call "{0}" -vcvars_ver={1} >nul 2>&1' -f $VcVars,$tools.Name),'if errorlevel 1 exit /b 1',$configure,'if errorlevel 1 exit /b 1',('"{0}" --build "{1}" --target ecm' -f $cmake,$BuildDir),'if errorlevel 1 exit /b 1')
[IO.File]::WriteAllText($batch,($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1 | Out-File (Join-Path $BuildDir 'build.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $BuildDir 'build.log') -Tail 50;throw 'OpenCL local build failed'}
Copy-Item (Join-Path $Gmp 'bin/gmp-10.dll') $BuildDir -Force
foreach($name in @('libcrypto-3-x64.dll','libssl-3-x64.dll')){Copy-Item (Join-Path $OpenSslRoot "bin/$name") $BuildDir -Force}
$resources=[ordered]@{}
$resourceRoot=Join-Path $BuildDir 'opencl_resources'
foreach($file in @(Get-ChildItem -LiteralPath $resourceRoot -Recurse -File)){
    $relative=$file.FullName.Substring($resourceRoot.Length+1).Replace('\','/')
    $resources[$relative]=(Get-FileHash -LiteralPath $file.FullName).Hash
}
$manifest=[ordered]@{complete=$true;role='local';backend='opencl';profile_sha256=$hash;executable_sha256=(Get-FileHash -LiteralPath (Join-Path $BuildDir 'ecm.exe')).Hash;cuda_compiler_enabled=$false;resources=$resources}
[IO.File]::WriteAllText((Join-Path $BuildDir 'build_manifest.json'),($manifest | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
if($BuildRole -eq 'release'){
    & python $payloadTool --build $BuildDir --freeze $snapshot
    if($LASTEXITCODE){throw 'OpenCL release freeze failed'}
}
Write-Host "done: $BuildDir/ecm.exe (OpenCL $BuildRole; CUDA disabled)"
