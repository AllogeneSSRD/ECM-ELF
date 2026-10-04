param([Parameter(Mandatory=$true)][string]$Results)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$runner=Get-Content (Join-Path $repo 'tools/bench/bench_stage2_reduce_ab.ps1') -Raw
$matrix=[regex]::Match($runner,'(?ms)^        if\(\$Target -eq ''resident_checks''\) \{.*?(?=^        if\(\$Target -eq ''root_fold'')')
$generic=[regex]::Match($runner,'(?s)        if \(\$Target -ne ''scaled_descent'' -and \$rows.Count -gt 0 -and \(\(\$Target -ne ''output_window''.*?throw "A/B results or coefficient count changed; inspect \$log"\s*\}')
$output=[regex]::Match($runner,'(?s)        if\(\$rows.Count -gt 0\) \{\s*foreach\(\$field in @\(''groot_root_hash''.*?throw "A/B output changed: \$field; inspect \$log"\s*\}\s*\}\s*\}')
$og=[regex]::Match($runner,'(?s)        if \(\$oracle.Count -ne 16.*?throw "oracle validation failed; inspect \$log"\s*\}')
$cg=[regex]::Match($runner,'(?s)        if \(-not \$carry.Success.*?throw "carry/staging validation failed; inspect \$log"\s*\}')
foreach($m in @($matrix,$generic,$output,$og,$cg)){if(-not $m.Success){throw 'resident guard seam missing'}}
$guard=[scriptblock]::Create(($matrix.Value,$generic.Value,$output.Value,$og.Value,$cg.Value -join "`n"))
$data=@(Import-Csv -LiteralPath $Results)
if($data.Count -ne 8 -or $data[0].mode -ne 'blocking' -or $data[3].mode -ne 'checks_both'){throw 'mirrored matrix evidence missing'}
$baseline=$data[0];$candidate=$data[3]
foreach($r in @($baseline,$candidate)){foreach($p in @($r.PSObject.Properties)) {
 if($p.Name -notin @('oracle_signature','factors','hit_primes','groot_root_hash','leaf_hash','root_fold_digest_sum','root_fold_digest_xor') -and
    $p.Value -match '^\d+(?:\.\d+)?$'){$r.($p.Name)=[double]$p.Value}
}}
$text=Get-Content -LiteralPath $candidate.log -Raw
$line=[regex]::Match($text,'(?m)^s4_oracle_stats:.*').Value;$baseOracle=@{}
foreach($field in @('async','selected','queued','compared','samples','pending','ring_waits','fallbacks','signature',
                    't_wait','t_copy_host','t_gmp','host_total','pack','t_num','t_mod')) {
 $baseOracle[$field]=[regex]::Match($line,"(?:^| )$field=([^\s]+)").Groups[1].Value
}
$carryLine=[regex]::Match($text,'(?m)^real_batched_carrydefer:.*').Value
$carryPattern='chunks_deferred=(\d+) finishes=(\d+) deferred_slices=(\d+) batch_enabled=(\d+) checked_chunks=(\d+) max_group=(\d+)'
$checked=[regex]::Match("$($candidate.oracle_samples)",'(\d+)')
$upload=$transfers=$rawVolume=$backVolume=$carryTime=$carryLedger=[regex]::Match('0 0 0','(\d+) (\d+) (\d+)')
$Target='resident_checks';$NHex='7'+('f'*1105);$rootLine='digest_kind=mixsum_xor_v1'
$oracleAsync='1';$oraclePack='1';$carryBatch='1';$log='saved matrix candidate';$rows=@($baseline);$passed=0
foreach($change in @('none','leaf_hash','oracle_signature','coeffs','ntt_launches','ntt_pairs','h2d_gib','d2h_gib',
 'root_fold_requested','root_fold_trees','root_fold_words','root_fold_digest_words','root_fold_digest_sum',
 'root_fold_digest_xor','root_fold_hash_complete','fold_device_h2d_bytes','fold_device_d2h_bytes',
 'oracle_async','oracle_samples','oracle_selected','oracle_compared','oracle_queued','oracle_pending','oracle_fallbacks',
 'carry_checked','carry_finishes','carry_group')) {
 $row=$candidate.PSObject.Copy();$oracle=$baseOracle.Clone();$cl=$carryLine
 if($change -like 'oracle_*' -and $change -notin @('oracle_signature')) {
  $field=$change.Substring(7);$oracle[$field]=([UInt64]$oracle[$field]+1).ToString()
 } elseif($change -eq 'carry_checked'){$cl=$cl -replace 'checked_chunks=\d+','checked_chunks=1'}
 elseif($change -eq 'carry_finishes'){$cl=$cl -replace 'finishes=\d+','finishes=0'}
 elseif($change -eq 'carry_group'){$cl=$cl -replace 'max_group=\d+','max_group=1'}
 elseif($change -in @('leaf_hash','oracle_signature','root_fold_digest_sum','root_fold_digest_xor')){$row.$change='corrupt'}
 elseif($change -ne 'none'){$row.$change=[double]$row.$change+1}
 $carry=[regex]::Match($cl,$carryPattern);$rejected=$false
 try {& $guard} catch {$rejected=$true}
 if($rejected -ne ($change -ne 'none')){throw "resident guard regression: $change rejection=$rejected"}
 ++$passed;Write-Output "PASS resident runner $change"
}
Write-Output "passed: $passed failed: 0"
