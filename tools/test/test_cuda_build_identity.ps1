# CPU-only mutation gates for real CUDA build/dependency identity helpers.
# No CUDA compilation, GPU queries, or edits to repository sources.
param([Parameter(Mandatory=$true)][string]$Output, [string]$AlternatePowerShell = '')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'tools/build/internal/stage1_release_sources.ps1')
$out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if (Test-Path -LiteralPath $out) { throw 'Use a fresh output directory' }
New-Item -ItemType Directory -Path $out | Out-Null
$tests = New-Object 'System.Collections.Generic.List[string]'
function Check([bool]$Condition, [string]$Name) {
    if (-not $Condition) { throw "FAIL: $Name" }
    $tests.Add($Name)
}
function Reject([scriptblock]$Action, [string]$Message, [string]$Name) {
    $errorText = ''
    try { & $Action | Out-Null } catch { $errorText = $_.Exception.Message }
    Check ($errorText -like "*$Message*") $Name
}
function Write-Fixture([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText($Path, $Text, [Text.UTF8Encoding]::new($false))
}
$fixture = Join-Path $out 'fixture'
foreach ($dir in @('src','headers/nested','build','tools/bench')) {
    New-Item -ItemType Directory -Path (Join-Path $fixture $dir) -Force | Out-Null
}
$source = Join-Path $fixture 'src/main.cu'
$header = Join-Path $fixture 'headers/nested/table.inc'
$inl = Join-Path $fixture 'headers/nested/loop.inl'
$def = Join-Path $fixture 'headers/nested/widths.def'
$macro = Join-Path $fixture 'headers/macro.h'
$rsp = Join-Path $fixture 'build/includes.rsp'
$obj = Join-Path $fixture 'build/main.obj'
Write-Fixture $source "#include <nested/table.inc>`n"
Write-Fixture $header "#include `"loop.inl`"`n"
Write-Fixture $inl "#include `"widths.def`"`n"
Write-Fixture $def 'WIDTH(1024)'
Write-Fixture $macro 'macro dependency'
Write-Fixture $obj 'compiled object'
Write-Fixture $rsp ('-I"' + (Join-Path $fixture 'headers') + '" -DFLAG=1')
$entry = [pscustomobject]@{ file=$source; directory=$fixture; command='nvcc --options-file build/includes.rsp -c src/main.cu -o build/main.obj'; output=$obj }
$base = Get-EcmCompileState $fixture $entry
Check ($base.source_hashes.Count -eq 5) 'transitive angle/quoted .inc/.inl/.def and response file'
Check (-not (Test-EcmCompileState $base)) 'old object without dependency record compiles once'
Save-EcmCompileState $base
Check (Test-EcmCompileState (Get-EcmCompileState $fixture $entry)) 'unchanged object reusable'
$record = "$obj.ecm-state.json"
$badRecord = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
$badRecord.source_hashes.'headers/nested/widths.def' = '0'*64
Write-Fixture $record ($badRecord | ConvertTo-Json -Depth 5)
Check (-not (Test-EcmCompileState (Get-EcmCompileState $fixture $entry))) 'corrupted dependency hash table rejected even with unchanged fingerprint'
Save-EcmCompileState $base
$mtime = (Get-Item -LiteralPath $def).LastWriteTimeUtc
Write-Fixture $def 'WIDTH(2048)'
(Get-Item -LiteralPath $def).LastWriteTimeUtc = $mtime
Check (-not (Test-EcmCompileState (Get-EcmCompileState $fixture $entry))) 'same-timestamp nested definition content change invalidates object'
Write-Fixture $def 'WIDTH(1024)'
Write-Fixture (Join-Path $fixture 'src/report.cpp') 'unrelated host report changed'
Check (Test-EcmCompileState (Get-EcmCompileState $fixture $entry)) 'unrelated host reporting does not invalidate CUDA object'
Write-Fixture $rsp ('-I"' + (Join-Path $fixture 'headers') + '" -DFLAG=2')
Check (-not (Test-EcmCompileState (Get-EcmCompileState $fixture $entry))) 'response-file compile definition invalidates object'
Write-Fixture $rsp ('-I"' + (Join-Path $fixture 'headers') + '" -DFLAG=1')
$entry.command += ' -DCHANGED=1'
Check (-not (Test-EcmCompileState (Get-EcmCompileState $fixture $entry))) 'compile command change invalidates object'
$entry.command = $entry.command.Replace(' -DCHANGED=1','')
Write-Fixture $obj 'replacement object'
Check (-not (Test-EcmCompileState (Get-EcmCompileState $fixture $entry))) 'object tampering rejected'
Write-Fixture $obj 'compiled object'
$spacedDir = Join-Path $fixture 'headers/space dir'
New-Item -ItemType Directory -Path $spacedDir | Out-Null
$spacedHeader = Join-Path $spacedDir 'hash#table.h'
Write-Fixture $spacedHeader 'escaped compiler dependency'
$escapedHeader = $spacedHeader.Replace('\','/').Replace(' ','\ ').Replace('#','\#')
Write-Fixture "$obj.ecm.d" ($obj.Replace('\','/') + ': ' + $source.Replace('\','/') + ' ' + $macro.Replace('\','/') + ' ' + $escapedHeader)
$withDepfile = Get-EcmCompileState $fixture $entry
Check ($withDepfile.source_hashes.Contains('headers/macro.h')) 'compiler depfile adds macro include'
Check ($withDepfile.source_hashes.Contains('headers/space dir/hash#table.h')) 'compiler depfile decodes spaces and hash characters'
Save-EcmCompileState $withDepfile
$depfileText = [IO.File]::ReadAllText("$obj.ecm.d")
$experiment = Join-Path $fixture 'tools/bench/experiment.cuh'
Write-Fixture $experiment '// experimental macro include'
Write-Fixture "$obj.ecm.d" ($depfileText + ' ' + $experiment.Replace('\','/'))
Reject { Get-EcmCompileState $fixture $entry -Production } 'depends on experiment' 'compiler-discovered experimental macro include rejected'
Write-Fixture "$obj.ecm.d" $depfileText
$obsolete = Join-Path $fixture 'headers/obsolete-sdk.h'
Write-Fixture $obsolete 'old SDK dependency'
Write-Fixture "$obj.ecm.d" ($depfileText + ' ' + $obsolete.Replace('\','/'))
Save-EcmCompileState (Get-EcmCompileState $fixture $entry)
Remove-Item -LiteralPath $obsolete
Check (-not (Test-EcmCompileState (Get-EcmCompileState $fixture $entry))) 'removed obsolete compiler dependency requests recompilation instead of blocking it'
Write-Fixture "$obj.ecm.d" $depfileText
Save-EcmCompileState $withDepfile
Write-Fixture $macro 'changed macro dependency'
Check (-not (Test-EcmCompileState (Get-EcmCompileState $fixture $entry))) 'macro header mutation invalidates object'
Write-Fixture $macro 'macro dependency'
Remove-Item -LiteralPath $def
Reject { Get-EcmCompileState $fixture $entry } 'Unresolved local include' 'deleted transitive dependency rejected'
Write-Fixture $def 'WIDTH(1024)'
Write-Fixture (Join-Path $fixture 'headers/new.h') 'new header'
Write-Fixture $source "#include <nested/table.inc>`n#include <new.h>`n"
Check (-not (Test-EcmCompileState (Get-EcmCompileState $fixture $entry))) 'new local dependency invalidates object'
Write-Fixture (Join-Path $fixture 'tools/bench/experiment.cuh') '// experiment'
Write-Fixture $source "#include `"../tools/bench/experiment.cuh`"`n"
Reject { Get-EcmSourceClosure $fixture @('src/main.cu') @() -Production } 'depends on experiment' 'production cannot include experiment source'
New-Item -ItemType Directory -Force (Join-Path $fixture 'tools/test') | Out-Null
Write-Fixture (Join-Path $fixture 'tools/test/fixture.h') '// test-only fixture'
Write-Fixture $source "#include `"../tools/test/fixture.h`"`n"
Reject { Get-EcmSourceClosure $fixture @('src/main.cu') @() -Production } 'depends on experiment' 'production cannot include a test adapter or oracle'
Write-Fixture $source "#include <nested/table.inc>`n"
$names = Get-EcmObjectNames @('one/runtime.cpp','two/runtime.cpp','unique.cu')
Check ($names['one/runtime.cpp'] -ne $names['two/runtime.cpp'] -and $names['unique.cu'] -eq 'unique') 'same-basename object names remain unique; existing unique names preserved'
$outside = [pscustomobject]@{file=$source; directory=$fixture; command=''; output=(Join-Path $out 'outside.obj')}
Reject { Get-EcmObjectPath $outside (Join-Path $fixture 'build') } 'outside build directory' 'object paths constrained to build directory'
$entries = @(
    [ordered]@{file=$source;directory=$fixture;command='nvcc -o CMakeFiles/ecm_cuda.dir/main.obj'},
    [ordered]@{file=$source;directory=$fixture;command='nvcc -o CMakeFiles/cuda_kernels.dir/kernel.obj'},
    [ordered]@{file=$source;directory=$fixture;command='nvcc -o CMakeFiles/unrelated.dir/kernel.obj'})
Write-Fixture (Join-Path $fixture 'build/compile_commands.json') ($entries | ConvertTo-Json)
Write-Fixture (Join-Path $fixture 'build/cuda_target_graph.json') '{"schema":1,"targets":{"ecm_cuda":["ecm_cuda","cuda_kernels"]}}'
Check (@(Get-EcmTargetCompileEntries (Join-Path $fixture 'build') ecm_cuda).Count -eq 2) 'object-module target closure includes kernels and excludes unrelated target'
$state = Get-EcmCompileState $fixture $entry
$manifest = [pscustomobject]@{identity_schema=1;compile_setup='';compile_units=[pscustomobject]@{
    'src/main.cu'=[pscustomobject]@{entry=$entry;fingerprint=$state.fingerprint;object_sha256=(Get-FileHash $obj).Hash}}}
Assert-EcmBuildUnits $fixture (Join-Path $fixture 'build') $manifest @('src/main.cu')
Check $true 'complete compile-unit manifest accepted'
Reject { Assert-EcmBuildUnits $fixture (Join-Path $fixture 'build') $manifest @('src/main.cu','src/missing.cu') } 'set changed' 'missing manifest translation unit rejected'
Write-Fixture $obj 'replaced again'
Reject { Assert-EcmBuildUnits $fixture (Join-Path $fixture 'build') $manifest @('src/main.cu') } 'object changed' 'manifest object tampering rejected'
Reject { Assert-EcmBuildUnits $fixture (Join-Path $fixture 'build') ([pscustomobject]@{}) @('src/main.cu') } 'rebuild old manifests' 'old manifest cannot claim current identity'
Write-Fixture $obj 'compiled object'
$otherSource = Join-Path $fixture 'src/other.cu'
$otherObject = Join-Path $fixture 'build/other.obj'
Write-Fixture $otherSource "#include <nested/table.inc>`n"
Write-Fixture $otherObject 'second CUDA object'
$otherEntry = [pscustomobject]@{file=$otherSource;directory=$fixture;output=$otherObject;
    command=$entry.command.Replace('src/main.cu','src/other.cu').Replace('build/main.obj','build/other.obj')}
$otherState = Get-EcmCompileState $fixture $otherEntry
$manifest.compile_units | Add-Member -NotePropertyName 'src/other.cu' -NotePropertyValue ([pscustomobject]@{
    entry=$otherEntry;fingerprint=$otherState.fingerprint;object_sha256=(Get-FileHash $otherObject).Hash})
Assert-EcmBuildUnits $fixture (Join-Path $fixture 'build') $manifest @('src/main.cu','src/other.cu')
Check $true 'multiple CUDA translation units accepted without single-object assumptions'
Write-Fixture $otherObject 'tampered second CUDA object'
Reject { Assert-EcmBuildUnits $fixture (Join-Path $fixture 'build') $manifest @('src/main.cu','src/other.cu') } 'src/other.cu' 'second CUDA object tampering rejected'
if ($AlternatePowerShell) {
    $captureScript = Join-Path $out 'capture_fingerprint.ps1'
    $entryPath = Join-Path $out 'entry.json'
    $fingerprintPath = Join-Path $out 'alternate_fingerprint.txt'
    Write-Fixture $entryPath ($entry | ConvertTo-Json -Depth 4)
    $capture = @'
param([string]$Helper,[string]$Repo,[string]$EntryPath,[string]$Output)
. $Helper
$entry=Get-Content -LiteralPath $EntryPath -Raw -Encoding UTF8|ConvertFrom-Json
$state=Get-EcmCompileState $Repo $entry
[IO.File]::WriteAllText($Output,$state.fingerprint,[Text.UTF8Encoding]::new($false))
'@
    Write-Fixture $captureScript $capture
    & $AlternatePowerShell -NoProfile -ExecutionPolicy Bypass -File $captureScript -Helper (Join-Path $repo 'tools/build/internal/cuda_build_state.ps1') -Repo $fixture -EntryPath $entryPath -Output $fingerprintPath
    if ($LASTEXITCODE -ne 0) { throw 'Alternate PowerShell fingerprint failed' }
    $current = Get-EcmCompileState $fixture $entry
    Check ([IO.File]::ReadAllText($fingerprintPath) -eq $current.fingerprint) 'same content fingerprint across PowerShell runtimes'
}
$inventory = Get-EcmCudaInventory $repo (Join-Path $out 'cuda_sources.json')
$ntt=@('src/ecm/cuda/ntt/shape.cpp','src/ecm/cuda/ntt/runtime.cpp','src/ecm/cuda/ntt/graph.cpp','src/ecm/cuda/ntt/checks.cpp','src/ecm/cuda/ntt/device/transform_kernels.cu')
Check (@($inventory.groups.STAGE2_NTT).Count -eq 5 -and
    @(Compare-Object ($inventory.groups.STAGE2_NTT|Sort-Object) ($ntt|Sort-Object)).Count -eq 0 -and
    @($inventory.groups.STAGE2_ARITHMETIC).Count -eq 1 -and
    $inventory.groups.STAGE2_ARITHMETIC[0] -eq 'src/ecm/stage2/cuda/device/arithmetic_kernels.cu') 'Stage2 arithmetic and NTT module membership'
$expectedProduction=@($inventory.groups.STAGE2_ENGINE)+@($inventory.groups.STAGE2_ARITHMETIC)+@($inventory.groups.STAGE2_NTT)
$engine=@('constants','context','host_wait','options','resources','plan','engine','batch','points','product','tree','remainder','s4','factors','checks','diagnostics','tune') | ForEach-Object { 'src/ecm/stage2/cuda/'+$_+'.cpp' }
$engine += 'src/ecm/cuda/ntt/module_dispatch.cpp'
Check (@($inventory.groups.STAGE2_ENGINE).Count -eq 18 -and
    @(Compare-Object ($inventory.groups.STAGE2_ENGINE|Sort-Object) ($engine|Sort-Object)).Count -eq 0) 'Stage2 host module membership'
$expectedProduction=@($expectedProduction|Where-Object {$_})
$production=@($inventory.groups.STAGE2_PRODUCTION)
Check ($production.Count -eq $expectedProduction.Count -and
    @(Compare-Object ($production|Sort-Object) ($expectedProduction|Sort-Object)).Count -eq 0 -and
    @($inventory.groups.STAGE2_DEVELOPMENT).Count -eq 1 -and
    $inventory.groups.STAGE2_DEVELOPMENT[0] -eq 'tools/experiments/cuda/stage2_tree/worker.cu' -and
    @($production|Where-Object {$_ -like 'tools/*'}).Count -eq 0) 'production and development source groups separate'
$set = Get-EcmStage2SourceSet $repo $inventory production @('include','src','src/ecm/cuda')
Check (-not @($set.Dependencies | Where-Object { $_ -match '^tools/(bench|experiments)/' }).Count) 'actual production Stage2 source closure isolated'
Check (@($set.Dependencies | Where-Object { $_ -like '*.def' }).Count -ge 2) 'Stage2 definition tables present in source identity'
$stage1 = Get-Stage1ReleaseSources $repo
foreach ($name in @('src/ecm/cuda/dispatch/tiers.def','src/ecm/edwards_stage1/cuda/edwards_runtime.inl','src/ecm/edwards_stage1/cuda/workflow.cpp')) {
    Check ($stage1.Contains($name)) "Stage1 release records $name"
}
$enabled = Get-Stage1ReleaseSources $repo -EnablePrac $true
Check ($enabled.Contains('src/ecm/montgomery_stage1/cuda/prac/ecm_prac_plan.cpp') -and -not $stage1.Contains('src/ecm/montgomery_stage1/cuda/prac/ecm_prac_plan.cpp')) 'PRAC source membership follows explicit feature flag'
Reject { Assert-Stage1ReleaseSources ([pscustomobject]@{}) $stage1 } 'Stage1 source changed' 'missing release source entries rejected'
$moduleSource = Join-Path $fixture 'src/host_module.cpp'
$moduleHeader = Join-Path $fixture 'src/module_private.inc'
Write-Fixture $moduleSource "#include `"module_private.inc`"`n"
Write-Fixture $moduleHeader 'host module input'
$entries += [ordered]@{file=$moduleSource;directory=$fixture;command='cl /FoCMakeFiles/host_module.dir/module.obj'}
Write-Fixture (Join-Path $fixture 'build/compile_commands.json') ($entries | ConvertTo-Json)
Write-Fixture (Join-Path $fixture 'build/cuda_target_graph.json') '{"schema":1,"targets":{"ecm_cuda":["ecm_cuda","cuda_kernels","host_module"]}}'
$linked = Get-Stage1ReleaseSources $repo -BuildDir (Join-Path $fixture 'build')
Check ($linked.Contains((Get-EcmSourceKey $repo $moduleHeader))) 'linked host-module private includes enter Stage1 release identity'
Write-Fixture $moduleHeader 'changed host module input'
Reject { Assert-Stage1ReleaseSources $linked (Get-Stage1ReleaseSources $repo -BuildDir (Join-Path $fixture 'build')) } 'Stage1 source changed' 'linked host-module dependency mutation rejected'
$msvcHeader = Join-Path $fixture 'sdk/macro include.h'
New-Item -ItemType Directory -Force (Split-Path -Parent $msvcHeader) | Out-Null
Write-Fixture $msvcHeader 'SDK/macro include content'
$msvcObject = Join-Path $fixture 'build/host.obj'
Write-Fixture $msvcObject 'MSVC host object'
$msvcEntry = [pscustomobject]@{file=$moduleSource;directory=$fixture;output=$msvcObject;
    command="cl /std:c++17 /MT /EHsc /c `"$moduleSource`" /Fo`"$msvcObject`" /sourceDependencies `"$msvcObject.ecm-msvc.json`""}
$msvcDeps = [ordered]@{Version='1.2';Data=[ordered]@{Source=$moduleSource;Includes=@($msvcHeader);ImportedModules=@();ImportedHeaderUnits=@()}}
Write-Fixture "$msvcObject.ecm-msvc.json" ($msvcDeps | ConvertTo-Json -Depth 5)
$msvcState = Get-EcmCompileState $fixture $msvcEntry
Save-EcmCompileState $msvcState
Check ($msvcState.source_hashes.Contains((Get-EcmSourceKey $fixture $msvcHeader))) 'MSVC compiler discovers macro and SDK includes with spaces'
Check (Test-EcmCompileState $msvcState) 'MSVC dependency record reused with unchanged inputs'
# New compiler dependencies may legitimately retire an old SDK/macro include.
# Recheck actual bytes, including relative keys under a different caller cwd.
$beforeRetirement=Get-EcmCompileState $fixture $msvcEntry
$msvcDeps.Data.Includes=@()
Write-Fixture "$msvcObject.ecm-msvc.json" ($msvcDeps | ConvertTo-Json -Depth 5)
$afterRetirement=Get-EcmCompileState $fixture $msvcEntry
Assert-EcmCompileInputs $beforeRetirement $afterRetirement
Check $true 'retired unchanged compiler include accepted after real content recheck'
Check (-not (Test-EcmCompileState $afterRetirement)) 'retired include still invalidates the previous object fingerprint'
Check ($beforeRetirement.source_root -eq [IO.Path]::GetFullPath($fixture)) 'retired relative dependency resolved from its source root rather than caller cwd'
Write-Fixture $msvcHeader 'changed during compilation after retirement'
Reject { Assert-EcmCompileInputs $beforeRetirement $afterRetirement } 'Source changed during build' 'retired dependency changed during compilation rejected'
Remove-Item -LiteralPath $msvcHeader
Reject { Assert-EcmCompileInputs $beforeRetirement $afterRetirement } 'missing' 'retired dependency removed during compilation rejected'
Write-Fixture $msvcHeader 'SDK/macro include content'
$missingContext=[pscustomobject]@{source_hashes=$beforeRetirement.source_hashes}
Reject { Assert-EcmCompileInputs $missingContext $afterRetirement } 'without source root' 'unverifiable retired dependency fails closed'
$msvcDeps.Data.Includes=@($msvcHeader)
Write-Fixture "$msvcObject.ecm-msvc.json" ($msvcDeps | ConvertTo-Json -Depth 5)
$sameInputs=Get-EcmCompileState $fixture $msvcEntry
Assert-EcmCompileInputs $beforeRetirement $sameInputs
Check $true 'unchanged active compiler includes remain accepted'
Write-Fixture $msvcHeader 'changed active include during compilation'
$changedInputs=Get-EcmCompileState $fixture $msvcEntry
Reject { Assert-EcmCompileInputs $beforeRetirement $changedInputs } 'Source changed during build' 'active compiler dependency mutation remains rejected'
Write-Fixture $msvcHeader 'SDK/macro include content'
$newHeader=Join-Path $fixture 'sdk/new compiler include.h'
Write-Fixture $newHeader 'newly discovered compiler dependency'
$msvcDeps.Data.Includes=@($msvcHeader,$newHeader)
Write-Fixture "$msvcObject.ecm-msvc.json" ($msvcDeps | ConvertTo-Json -Depth 5)
$expandedInputs=Get-EcmCompileState $fixture $msvcEntry
Assert-EcmCompileInputs $beforeRetirement $expandedInputs
Check $true 'newly discovered compiler include remains accepted and enters the new fingerprint'
$msvcDeps.Data.Includes=@($msvcHeader)
Write-Fixture "$msvcObject.ecm-msvc.json" ($msvcDeps | ConvertTo-Json -Depth 5)
$stamp = (Get-Item -LiteralPath $msvcHeader).LastWriteTimeUtc
Write-Fixture $msvcHeader 'changed SDK/macro include content'
(Get-Item -LiteralPath $msvcHeader).LastWriteTimeUtc = $stamp
Check (-not (Test-EcmCompileState (Get-EcmCompileState $fixture $msvcEntry))) 'MSVC discovered header content invalidates object despite preserved timestamp'
$msvcDeps.Data.Source=$source
Write-Fixture "$msvcObject.ecm-msvc.json" ($msvcDeps | ConvertTo-Json -Depth 5)
Reject { Get-EcmCompileState $fixture $msvcEntry } 'Invalid MSVC compiler dependency' 'MSVC dependency file cannot identify another source'
$msvcDeps.Data.Source=$moduleSource; $msvcDeps.Data.ImportedModules=@('unexpected-module')
Write-Fixture "$msvcObject.ecm-msvc.json" ($msvcDeps | ConvertTo-Json -Depth 5)
Reject { Get-EcmCompileState $fixture $msvcEntry } 'explicit build contract' 'unsupported MSVC module imports are rejected rather than ignored'
$msvcDeps.Data.ImportedModules=@();$msvcDeps.Version='future'
Write-Fixture "$msvcObject.ecm-msvc.json" ($msvcDeps | ConvertTo-Json -Depth 5)
Reject { Get-EcmCompileState $fixture $msvcEntry } 'Invalid MSVC compiler dependency' 'unknown MSVC dependency format fails closed'
$result = [ordered]@{complete=$true;checks=$tests.Count;tests=@($tests);gpu_runs=0}
$result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $out 'result.json') -Encoding utf8
Write-Host "PASS: $($tests.Count) CUDA build identity mutation gates (CPU only)"
