param([string]$WorkerBuild='build_stage1_local',
      [Parameter(Mandatory=$true)][string]$Build,
      [switch]$WithStage2,[string]$BackendBuild='build_stage2_local')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$m=Get-Content -LiteralPath (Join-Path $WorkerBuild 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
$dest=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(-not $dest.StartsWith($repo+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Build must stay inside the repository'}
if(Test-Path -LiteralPath $dest){throw 'Use a fresh probe directory'}
New-Item -ItemType Directory -Path $dest|Out-Null
if($WithStage2){
 $backend=Get-Content -LiteralPath (Join-Path $BackendBuild 'build_manifest.json') -Raw|ConvertFrom-Json
 $objects=@($backend.link_inputs.objects.PSObject.Properties.Name)
 # Reuse only the CUDA engine, NTT and their mandatory checks; exclude app/main.
 $cut=[array]::FindIndex($objects,[Predicate[string]]{param($s) (Split-Path -Leaf $s) -eq 'transform_kernels.obj'})
 if($cut -lt 0){throw 'Unrecognized backend object inventory'}
 $includeObjects=$objects[0..$cut]
 $activeGmp=$backend.gmp_root
 $activeCompiler=$backend.host_compiler
 $command='"'+$backend.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MT /utf-8 /DECM_FRIENDLY_WITH_STAGE2 /I src /I include /I "'+
  $backend.gmp_root+'/include" /c tools/test/ecm_friendly_probe.cpp /Fo"'+$dest+'/probe.obj"'
 $link=$backend.link_inputs.command
 $firstObject=$link.IndexOf('"'+$objects[0]+'"')
 if($firstObject -lt 0){throw 'Cannot locate backend link inputs'}
 $link=$link.Substring(0,$firstObject)+('"'+$dest+'/probe.obj" ')+(($includeObjects|ForEach-Object{'"'+$_+'"'}) -join ' ')+
  ' -L "'+$backend.gmp_root+'/lib" -lgmp -o "'+$dest+'/ecm_friendly_probe.exe"'
 $lines=@('@echo off',$backend.compile_setup,'if errorlevel 1 exit /b 1',$command,'if errorlevel 1 exit /b 1',$link,'exit /b %errorlevel%','')
}else{
 $activeGmp=$m.gmp_root
 $activeCompiler=$m.host_compiler
 $command='"'+$m.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MD /utf-8 /I src /I include /I "'+
  $m.gmp_root+'/include" tools/test/ecm_friendly_probe.cpp /Fo"'+$dest+'/probe.obj" /Fe"'+$dest+
  '/ecm_friendly_probe.exe" /link /LIBPATH:"'+$m.gmp_root+'/lib" gmp.lib'
 $lines=@('@echo off',$m.compile_setup,'if errorlevel 1 exit /b 1',$command,'exit /b %errorlevel%','')
}
$batch=Join-Path $dest 'build.cmd'
[IO.File]::WriteAllText($batch,[string]::Join([Environment]::NewLine,$lines),[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Tee-Object -FilePath (Join-Path $dest 'build.log')
if($LASTEXITCODE){throw 'ECM-friendly native probe compilation failed'}
Copy-Item -LiteralPath (Join-Path $activeGmp 'bin/gmp-10.dll') -Destination $dest
$versionText=Get-Content -LiteralPath 'include/ecm/platform/program_version.h' -Raw
$programVersion=[regex]::Match($versionText,'version\[\] = "([^"]+)"').Groups[1].Value
[ordered]@{program_version=$programVersion;experiment_version='1.0.0';compiler=$activeCompiler;gmp=$activeGmp;
    engine='standalone GMP; 4M+4S doubling; a=1 9M addition, a=-1 8M addition; NAF w6';
    production_dispatch_changed=$false;stage2_backend_reused=[bool]$WithStage2;cuda_source_compilations=0}|
 ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dest 'manifest.json') -Encoding utf8
