param([Parameter(Mandatory=$true)][string]$Results)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$runner=Get-Content (Join-Path $repo 'tools/bench/bench_stage2_reduce_ab.ps1') -Raw
$generic=[regex]::Match($runner,'(?s)        if \(\$Target -ne ''scaled_descent'' -and \$rows.Count -gt 0 -and \(\(\$Target -ne ''output_window''.*?throw "A/B results or coefficient count changed; inspect \$log"\s*\}')
$root=[regex]::Match($runner,'(?s)        if\(\$Target -eq ''root_fold''\) \{.*?throw "Root-fold checksum/workload/transfer delta disagrees; inspect \$log"\s*\}\s*\}')
$output=[regex]::Match($runner,'(?s)        if\(\$rows.Count -gt 0\) \{\s*foreach\(\$field in @\(''groot_root_hash''.*?throw "A/B output changed: \$field; inspect \$log"\s*\}\s*\}\s*\}')
if(-not $generic.Success -or -not $root.Success -or -not $output.Success){throw 'root-fold runner guard seam missing'}
$data=@(Import-Csv -LiteralPath $Results)
$rows=@($data[0]);$candidate=$data[1]
if($rows[0].mode -ne 'host_root' -or $candidate.mode -ne 'device_root'){throw 'expected host/device root evidence'}
foreach($r in @($rows[0],$candidate)) {
 foreach($p in @($r.PSObject.Properties)) {
  if($p.Name -notin @('oracle_signature','factors','hit_primes','groot_root_hash','leaf_hash','root_fold_digest_sum','root_fold_digest_xor') -and
     $p.Value -match '^\d+(?:\.\d+)?$') {$r.($p.Name)=[double]$p.Value}
 }
}
$guard=[scriptblock]::Create($generic.Value+"`n"+$root.Value+"`n"+$output.Value)
$log='saved root handoff candidate';$NHex='7'+('f'*1105);$mode='device_root'
$env:NTT_GROOT_TO_FOLD='1';$rootLine='digest_kind=mixsum_xor_v1';$passed=0
foreach($change in @('none','omitted_fnv','d2h_gib','h2d_gib','coeffs','raw_async','oracle_signature','leaf_hash',
    'root_fold_trees','root_fold_words','root_fold_digest_words','root_fold_digest_sum','root_fold_digest_xor',
    'root_fold_avoided_h2d_bytes','root_fold_avoided_d2h_bytes','root_fold_hash_complete',
    'fold_device_h2d_bytes','fold_device_d2h_bytes','fold_device_avoided_h2d_bytes','ntt_launches','ntt_pairs','legacy_target')) {
 $row=$candidate.PSObject.Copy();$Target='root_fold'
 if($change -eq 'legacy_target'){$Target='small_prime'}
 elseif($change -eq 'omitted_fnv'){$row.groot_root_hash='0000000000000001'}
 elseif($change -in @('oracle_signature','leaf_hash','root_fold_digest_sum','root_fold_digest_xor')){$row.$change='corrupt'}
 elseif($change -ne 'none'){$row.$change=[double]$row.$change+1}
 $rejected=$false
 try {& $guard} catch {$rejected=$true}
 if($rejected -ne ($change -notin @('none','omitted_fnv'))){throw "root runner regression: $change rejection=$rejected"}
 ++$passed;Write-Output "PASS root runner $change"
}
Write-Output "passed: $passed failed: 0"
