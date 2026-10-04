param([Parameter(Mandatory=$true)][string]$BaselineLog,[Parameter(Mandatory=$true)][string]$CandidateLog)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$runner=Get-Content (Join-Path $repo 'tools/bench/bench_stage2_reduce_ab.ps1') -Raw
$m=[regex]::Match($runner,'(?s)        if \(\$Target -ne ''scaled_descent'' -and \$rows.Count -gt 0 -and \(\(\$Target -ne ''output_window''.*?throw "A/B results or coefficient count changed; inspect \$log"\s*\}')
if(-not $m.Success){throw 'guard seam missing'}
$traffic=[regex]::Match($runner,'(?s)        if\(\$Target -eq ''device_gleaf'' -and \$rows.Count -gt 0\) \{.*?throw "Device leaf raw H2D delta disagrees with exact leaf words; inspect \$log"\s*\}\s*\}')
if(-not $traffic.Success){throw 'device leaf traffic guard missing'}
function Row($path) {
 $s=Get-Content $path -Raw
 $r=[pscustomobject]@{}
 foreach($key in @('coeffs','factors','hit_primes','hits','oracle_samples','oracle_jobs','oracle_signature','carry_deferred','carry_slices','raw_async','out_async','pack_launches','h2d_gib','d2h_gib','device_leaf_device_leaf_words')) {
  $pattern=switch($key){
   'coeffs' {'s4_multiply_stats:.*?coeffs_reduced=(\d+)'}
   'oracle_samples' {'s4_oracle_stats:.*? samples=(\d+)'}
   'oracle_jobs' {'s4_oracle_stats:.*? selected=(\d+)'}
   'oracle_signature' {'s4_oracle_stats:.*? signature=(\S+)'}
   'carry_deferred' {'real_batched_carrydefer:.*?chunks_deferred=(\d+)'}
   'carry_slices' {'real_batched_carrydefer:.*?deferred_slices=(\d+)'}
   'h2d_gib' {'real_batched_rawupload:.*?volume=([0-9.]+)'}
   'd2h_gib' {'real_batched_coeffback:.*?volume=([0-9.]+)'}
   'device_leaf_device_leaf_words' {'(?m)^device_gleaf:.*?device_leaf_words=(\d+)'}
   default {"(?:^| )$key=(\S*)"}
  }
  $match=[regex]::Match($s,$pattern)
  if(-not $match.Success){throw "missing $key"}
  $value=$match.Groups[1].Value
  if($key -notin @('factors','hit_primes','oracle_signature')){$value=[double]$value}
  $r|Add-Member -NotePropertyName $key -NotePropertyValue $value
 }
 return $r
}
$rows=@((Row $BaselineLog));$candidate=Row $CandidateLog
$guard=[scriptblock]::Create($m.Value+"`n"+$traffic.Value)
$log='saved production candidate';$passed=0
foreach($change in @('none','h2d_gib','d2h_gib','coeffs','oracle_signature','raw_async','legacy_target')) {
 $row=$candidate.PSObject.Copy();$Target='device_gleaf'
 if($change -eq 'legacy_target'){$Target='small_prime'}
 elseif($change -eq 'oracle_signature'){$row.oracle_signature='corrupt'}
 elseif($change -ne 'none'){$row.$change=[double]$row.$change+1}
 $rejected=$false
 try {& $guard} catch {$rejected=$true}
 if($rejected -ne ($change -ne 'none')){throw "runner regression: $change unexpected rejection=$rejected"}
 ++$passed;Write-Output "PASS runner $change"
}
Write-Output "passed: $passed failed: 0"
