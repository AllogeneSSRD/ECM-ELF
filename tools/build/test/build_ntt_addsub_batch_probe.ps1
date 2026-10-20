#Requires -Version 5.1
param([Parameter(Mandatory=$true)][string]$SourceBuild,
      [Parameter(Mandatory=$true)][string]$Build,
      [Parameter(Mandatory=$true)][ValidateSet(0,1)][int]$Mask,
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
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
# Frozen snapshots retain their original layout. Select only a manifest-verified file.
$builderNames=@('tools/build/test/build_ntt_outer_v_probe.ps1','tools/build/build_ntt_outer_v_probe.ps1')
$builderName=@($builderNames | Where-Object { $_ -in $upstream.sources.PSObject.Properties.Name })
if($builderName.Count -ne 1){throw 'Upstream builder missing or ambiguous in manifest'}
$upstreamBuilder=Get-Content -LiteralPath (Join-Path $SourceBuild ('sources/'+$builderName[0])) -Raw
if($upstreamBuilder -notmatch "-DNTT_GL_ADD_SUB_MASK=$Mask -DNTT_GL_FIXED_MODE=3"){throw 'Upstream arithmetic mask differs'}
. (Join-Path $PSScriptRoot '../internal/cuda_frozen_ntt_probe.ps1')
$result=Invoke-EcmFrozenNttProbeBuild -Repo $repo -SourceRoot (Join-Path $SourceBuild 'sources') -Build $Build -Arch $Arch `
    -Source 'tools/test/ntt_addsub_batch_probe.cu' -Name 'ntt_addsub_batch_probe' `
    -Defines @("NTT_GL_ADD_SUB_MASK=$Mask",'NTT_GL_FIXED_MODE=3','NTT_OUTER_UNROLL_U=0') -AllowedSources $upstream.sources
$result.gl_fixed_mode=3;$result.compiled_outer_u=0;$result.arithmetic_mask=$Mask
$result.source_build_manifest_sha256=(Get-FileHash $manifestPath).Hash.ToLower()
$result|ConvertTo-Json -Depth 8|Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
Write-Host "built $($result.exe)"
