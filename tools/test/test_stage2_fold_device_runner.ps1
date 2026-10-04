param([Parameter(Mandatory=$true)][string]$Results)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$runner=Get-Content (Join-Path $repo 'tools/bench/bench_stage2_reduce_ab.ps1') -Raw
$generic=[regex]::Match($runner,'(?s)        if \(\$Target -ne ''scaled_descent'' -and \$rows.Count -gt 0 -and \(\(\$Target -ne ''output_window''.*?throw "A/B results or coefficient count changed; inspect \$log"\s*\}')
$traffic=[regex]::Match($runner,'(?s)        if\(\$Target -eq ''fold_device'' -and \$rows.Count -gt 0\) \{.*?throw "Device fold transfer delta disagrees with exact resident words; inspect \$log"\s*\}\s*\}')
$output=[regex]::Match($runner,'(?s)        if\(\$row.window_d2h_words.*?throw "Output/readback accounting disagrees; inspect \$log"\s*\}')
if(-not $generic.Success -or -not $traffic.Success -or -not $output.Success){throw 'runner guard seam missing'}
$data=@(Import-Csv -LiteralPath $Results)
$rows=@($data[0]);$candidate=$data[1]
if($rows[0].mode -ne 'host_fold' -or $candidate.mode -ne 'device_fold'){throw 'expected host/device evidence'}
# Import-Csv yields strings: the live runner uses numeric ledgers. Preserve signatures/outputs as strings.
foreach($r in @($rows[0],$candidate)) {
 foreach($p in @($r.PSObject.Properties)) {
  if($p.Name -notin @('oracle_signature','factors','hit_primes') -and $p.Value -match '^\d+(?:\.\d+)?$') {
   $r.($p.Name)=[double]$p.Value
  }
 }
}
$guard=[scriptblock]::Create($generic.Value+"`n"+$traffic.Value+"`n"+$output.Value)
$log='saved production candidate';$passed=0
foreach($change in @('none','h2d_gib','d2h_gib','coeffs','oracle_signature','raw_async',
                    'flat_calls','flat_zero_bytes','fold_device_avoided_h2d_bytes',
                    'fold_device_avoided_d2h_bytes','final_copied_words','ntt_launches','ntt_pairs','legacy_target')) {
 $row=$candidate.PSObject.Copy();$Target='fold_device'
 if($change -eq 'legacy_target'){$Target='small_prime'}
 elseif($change -eq 'oracle_signature'){$row.oracle_signature='corrupt'}
 elseif($change -ne 'none'){$row.$change=[double]$row.$change+1}
 $rejected=$false
 try {& $guard} catch {$rejected=$true}
 if($rejected -ne ($change -ne 'none')){throw "runner regression: $change unexpected rejection=$rejected"}
 ++$passed;Write-Output "PASS runner $change"
}
Write-Output "passed: $passed failed: 0"
