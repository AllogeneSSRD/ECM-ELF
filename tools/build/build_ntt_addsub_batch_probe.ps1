#Requires -Version 5.1
param([Parameter(Mandatory=$true)][string]$SourceBuild,
      [Parameter(Mandatory=$true)][string]$Build,
      [Parameter(Mandatory=$true)][ValidateSet(0,1)][int]$Mask,
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$SourceBuild=(Resolve-Path -LiteralPath $SourceBuild).Path
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $Build){throw 'Use a fresh build directory'}
$manifestPath=Join-Path $SourceBuild 'manifest.json'
$upstream=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if($upstream.gl_fixed_mode -ne 3 -or $upstream.compiled_outer_u -ne 0){throw 'Require fixed PTX3/original unroll'}
foreach($entry in $upstream.sources.PSObject.Properties){
    $path=Join-Path $SourceBuild ('sources/'+$entry.Name)
    if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne $entry.Value){throw 'Upstream source changed'}
}
$upstreamBuilder=Get-Content -LiteralPath (Join-Path $SourceBuild 'sources/tools/build/build_ntt_outer_v_probe.ps1') -Raw
if($upstreamBuilder -notmatch "-DNTT_GL_ADD_SUB_MASK=$Mask -DNTT_GL_FIXED_MODE=3"){throw 'Upstream arithmetic mask differs'}
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
New-Item -ItemType Directory $Build | Out-Null
$deps=@('tools/bench/ntt_poly_probe.cu','tools/bench/ntt_coop_outer.cuh',
    'tools/bench/ntt_goldilocks_reduce.cuh','tools/bench/ntt_goldilocks_ptx.cuh',
    'tools/bench/ntt_carry_partial.cuh','tools/bench/ntt_goldilocks_addsub.cuh')
$own=@('tools/test/ntt_addsub_batch_probe.cu','tools/build/build_ntt_addsub_batch_probe.ps1')
$hashes=[ordered]@{}
foreach($dep in ($deps+$own)){
    $src=if($dep -in $own){Join-Path $repo $dep}else{Join-Path $SourceBuild ('sources/'+$dep)}
    $hashes[$dep]=(Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash.ToLower()
    $dest=Join-Path $Build ('sources/'+$dep)
    New-Item -ItemType Directory -Force (Split-Path -Parent $dest) | Out-Null
    Copy-Item -LiteralPath $src -Destination $dest
}
$exe=Join-Path $Build 'ntt_addsub_batch_probe.exe';$log=Join-Path $Build 'build.log'
$src=Join-Path $Build 'sources/tools/test/ntt_addsub_batch_probe.cu'
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -DNTT_GL_ADD_SUB_MASK=$Mask -DNTT_GL_FIXED_MODE=3 -DNTT_OUTER_UNROLL_U=0 -Xptxas -v -I third_party/gmp-zen3/dist/include -Xcompiler /wd4819 `"$src`" -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$watch=[Diagnostics.Stopwatch]::StartNew();& cmd.exe /d /c $line;$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 35;throw 'Batch arithmetic probe build failed'}
foreach($dep in ($deps+$own)){
    $src=if($dep -in $own){Join-Path $repo $dep}else{Join-Path $SourceBuild ('sources/'+$dep)}
    if((Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash.ToLower() -ne $hashes[$dep]){throw 'Source changed during build'}
    if((Get-FileHash -LiteralPath (Join-Path $Build ('sources/'+$dep)) -Algorithm SHA256).Hash.ToLower() -ne $hashes[$dep]){throw 'Snapshot changed'}
}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build
[ordered]@{exe=$exe;sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLower();
    architecture=$Arch;gl_fixed_mode=3;compiled_outer_u=0;arithmetic_mask=$Mask;build_seconds=$watch.Elapsed.TotalSeconds;
    source_build_manifest_sha256=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLower();
    sources=$hashes;toolkit=(& nvcc --version | Out-String).Trim()} |
    ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host ("built {0} ({1:N1}s)" -f $exe,$watch.Elapsed.TotalSeconds)
