#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/stage1_checkpoint_probe_20261017',
      [switch]$Ownership)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/gmp_dependency.ps1')
$dependency=Resolve-EcmGmp -Repo $repo
$vcvars=(Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat'|Select-Object -First 1).FullName
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$exe=Join-Path $Build 'stage1_checkpoint_probe.exe'
$fixture=if($Ownership){'stage1_checkpoint_ownership_test'}else{'stage1_checkpoint_container_test'}
$script=Join-Path $Build 'build.cmd'
$sources=@("tools/test/$fixture.cpp",'src/ecm/stage1/edwards/cuda/checkpoint.cpp',
    'src/ecm/stage1/edwards/cuda/host_math.cpp',
    'src/io/stage1_checkpoint.cpp','src/io/files.cpp','src/io/save.cpp',
    'src/ecm/application/configuration/ecm_expr.cpp','src/ecm/platform/stop.cpp',
    'src/ecm/stage1/scalar.cpp','src/ecm/platform/logging.cpp',
    'src/ecm/application/configuration/runtime.cpp')
if(-not $Ownership){
    $sources+=@('src/ecm/stage1/montgomery/cuda/checkpoint.cpp',
                'tools/test/adapters/stage1_checkpoint_log.cpp')
}
$commands=@();$objects=@()
foreach($source in $sources){
    # Both curve families own checkpoint.cpp. Keep their object identities distinct.
    $object=Join-Path $Build ($source.Replace('/','_')+'.obj')
    $objects+=('"'+$object+'"')
    $commands+=("cl /nologo /c /O2 /std:c++17 /EHsc /MD /utf-8 /D NOMINMAX /I src /I include /I `"$($dependency.Root)/include`" `"$source`" /Fo:`"$object`"")
    $commands+='if errorlevel 1 exit /b 1'
}
$commands+=('cl /nologo '+($objects -join ' ')+" /Fe:`"$exe`" /link /LIBPATH:`"$($dependency.Root)/lib`" gmp.lib")
$commands+='exit /b %errorlevel%'
$line="@echo off`r`ncall `"$vcvars`" >nul 2>&1`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n"
[IO.File]::WriteAllText($script,$line,[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c "`"$script`"" *> (Join-Path $Build 'build.log')
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 25;throw 'checkpoint probe build failed'}
Copy-Item -LiteralPath (Join-Path $dependency.Root 'bin/gmp-10.dll') -Destination $Build -Force
Write-Host "Built checkpoint probe: $exe"
