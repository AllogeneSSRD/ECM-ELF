#Requires -Version 5.1
<#
.SYNOPSIS
    Same-binary ABBA benchmark of direct division versus Montgomery coefficient reduction.
.DESCRIPTION
    Defaults to the production shape in DEV_GPUOWL_NTT_NOTES.md section 32.  Writes a log
    for each run, provenance.json and results.csv.  The mode, GMP checks, factors/hit primes,
    coefficient count and arena overflow are checked before a speed comparison is reported.
    All environment overrides are restored.  Device 1 is used unless explicitly overridden.
#>
param(
    [string]$Exe = '',
    [string]$NHex = ('1' + ('f' * 1315)),
    [UInt64]$B1 = 1000,
    [UInt64]$B2 = 1940000000000,
    [UInt64]$D = 1231230,
    [int]$Sigma = 26,
    [int]$Device = 1,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
if (-not $Exe) { $Exe = Join-Path $repo 'build_cuda_cmake\stage2_tree_gpu.exe' }
if (-not (Test-Path -LiteralPath $Exe)) { throw "missing binary: $Exe" }
if (-not $Output) { $Output = Join-Path $repo ('build_cuda_cmake\_reduce_ab_' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$binaryHash = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash
$runArgs = @('--real', '--n-hex', $NHex, '--sigma', "$Sigma", '--b1', "$B1", '--b2', "$B2",
             '--d', "$D", '--device', "$Device")
$order = @('montgomery', 'division', 'division', 'montgomery')
$overrides = @{ NTT_NAME_MAX='1'; NTT_S4_BATCH_MB='32'; NTT_S4_ASYNC='1';
                NTT_S4_DEFER_CARRY='1'; NTT_S4_HOSTPACK='0'; NTT_S5_ON='0'; NTT_S4_OLDTAIL='1';
                NTT_S5_REDDUMP='0' }
$saved = @{}
foreach ($key in $overrides.Keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
@{ exe=$Exe; sha256=$binaryHash; args=$runArgs; order=$order; env=$overrides;
   started=(Get-Date -Format o); head=(& git rev-parse HEAD) } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Output 'provenance.json') -Encoding UTF8
$rows = @()
try {
    foreach ($key in $overrides.Keys) { [Environment]::SetEnvironmentVariable($key, $overrides[$key], 'Process') }
    for ($i = 0; $i -lt $order.Count; ++$i) {
        $mode = $order[$i]
        $env:NTT_S4_OLDTAIL = $(if ($mode -eq 'montgomery') { '1' } else { '0' })
        if ((Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash -ne $binaryHash) {
            throw 'binary changed during A/B; comparison invalid'
        }
        $log = Join-Path $Output (('{0}_{1}.log' -f ($i+1), $mode))
        Write-Host ("run {0}/4: {1}; log={2}" -f ($i+1), $mode, $log)
        $runStarted = Get-Date
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $ErrorActionPreference = 'Continue'   # capture native stderr in the evidence log
        & $Exe @runArgs > $log 2>&1
        $rc = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        $sw.Stop()
        $runEnded = Get-Date
        $text = Get-Content -LiteralPath $log -Raw
        $stage = [regex]::Match($text, 'stage2:.*?hits=(\d+) bad_factors=(\d+) factors=([^\s]*) hit_primes=([^\s]*) elapsed=([0-9.]+)')
        $reduce = [regex]::Match($text, 's4_multiply_stats:.*?coeffs_reduced=(\d+) t_reduce=([0-9.]+)')
        $arena = [regex]::Match($text, 'real_batched_breakdown:.*?arena_overflow=(\d+)')
        if ($rc -ne 0 -or -not $stage.Success -or -not $reduce.Success -or -not $arena.Success -or
            $stage.Groups[2].Value -ne '0' -or $arena.Groups[1].Value -ne '0' -or
            $text -notmatch ("s4_reduce_mode: algorithm=" + $mode) -or
            $text -cmatch 'FATAL|MISMATCH|gmp_bad=[1-9]|gmp_selftest_bad=[1-9]|gmp_check_bad=[1-9]|mismatches=[1-9]|slot_canonical_bad=[1-9]') {
            throw "A/B validation failed (exit=$rc); inspect $log"
        }
        $row = [pscustomobject]@{ run=$i+1; mode=$mode; elapsed=[double]$stage.Groups[5].Value;
            wall=[math]::Round($sw.Elapsed.TotalSeconds,3); t_reduce=[double]$reduce.Groups[2].Value;
            coeffs=[UInt64]$reduce.Groups[1].Value; hits=$stage.Groups[1].Value;
            factors=$stage.Groups[3].Value; hit_primes=$stage.Groups[4].Value; log=$log;
            started=$runStarted.ToString('o'); ended=$runEnded.ToString('o') }
        if ($rows.Count -gt 0 -and ($row.coeffs -ne $rows[0].coeffs -or $row.factors -ne $rows[0].factors -or
            $row.hit_primes -ne $rows[0].hit_primes -or $row.hits -ne $rows[0].hits)) {
            throw "A/B results or coefficient count changed; inspect $log"
        }
        $rows += $row
        $rows | Export-Csv -LiteralPath (Join-Path $Output 'results.csv') -NoTypeInformation -Encoding UTF8
        Write-Host ("  elapsed={0:F2}s wall={1:F2}s t_reduce={2:F3}s coeffs={3}" -f
                    $row.elapsed, $row.wall, $row.t_reduce, $row.coeffs)
    }
    $old = $rows | Where-Object mode -eq 'montgomery'
    $new = $rows | Where-Object mode -eq 'division'
    $oldTime = ($old | Measure-Object elapsed -Average).Average
    $newTime = ($new | Measure-Object elapsed -Average).Average
    $oldReduce = ($old | Measure-Object t_reduce -Average).Average
    $newReduce = ($new | Measure-Object t_reduce -Average).Average
    Write-Host ("ABBA mean: elapsed {0:F2} -> {1:F2}s ({2:F1}%); t_reduce {3:F3} -> {4:F3}s ({5:F1}%)" -f
                $oldTime, $newTime, (100*($newTime/$oldTime-1)), $oldReduce, $newReduce, (100*($newReduce/$oldReduce-1)))
} finally {
    foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
}
