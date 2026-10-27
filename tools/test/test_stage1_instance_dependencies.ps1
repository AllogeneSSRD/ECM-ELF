# Check real compiler-discovered dependencies without modifying any source.
param([Parameter(Mandatory=$true)][string]$Build,
      [Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'tools/build/internal/cuda_build_state.ps1')
$Build=Resolve-EcmSourcePath $repo $Build
$manifest=Get-Content (Join-Path $Build 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
$header='src/ecm/stage1/montgomery/cuda/prac/cgbn_stage1_prac_single_add.cuh'
$catalog=Get-Content (Join-Path $Build 'cuda_modules/stage1/modules.json') -Raw|ConvertFrom-Json
$expected=@($catalog.modules|Where-Object{$_.family -in @('prac_single','prac_constants')}|ForEach-Object{$_.source})
if(-not $expected.Count){throw 'Select a PRAC build containing the 4608 candidate families'}
$actual=@()
foreach($entry in Get-EcmTargetCompileEntries $Build ecm_cuda|Where-Object{$_.file -like '*.cu'}) {
    $state=Get-EcmCompileState $repo $entry $manifest.compile_setup -Production
    if(-not (Test-EcmCompileState $state)){throw "Actual object no longer reusable: $($entry.file)"}
    if($state.source_hashes.Contains($header)){$actual+=$entry.file}
}
if(@(Compare-Object ($expected|Sort-Object) ($actual|Sort-Object)).Count){
    throw 'Single-add dependency escaped its declared sparse candidate families'
}
$result=[ordered]@{complete=$true;header=$header;affected_sources=@($actual);cuda_compiles=0;source_mutations=0}
$result|ConvertTo-Json -Depth 4|Set-Content -LiteralPath $Output -Encoding UTF8
Write-Host "PASS: only $($actual.Count) sparse candidate units depend on single-add implementation"
