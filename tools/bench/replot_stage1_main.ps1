param(
    [string]$BenchmarkRoot = 'data/experiments/stage1_main_matrix_06317',
    [string]$GmpEcmCgbn,
    [string]$OutputDirectory = 'data/figures/stage1_main_matrix_06317',
    [string]$VersionOutputDirectory = 'data/figures/stage1_version_throughput',
    [string]$VersionSummary = 'data/benchmarks/stage1_version_throughput/selected.csv',
    [switch]$Publish
)

$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '../..')
try {
    $cgbn = 'data/experiments/stage1_slice_baseline_0637'
    $priorCgbn = 'data/benchmarks/stage1_cgbn_curve_cost_0623'
    $prime95 = 'tools/benchmarks/prime95'
    $paired = 'data/experiments/stage1_prac_production_0640/paired'
    $images = foreach ($model in 'ed', 'mont0') {
        $curve = if ($model -eq 'ed') { 'ed' } else { 'mont' }
        $priorNtt = if ($model -eq 'ed') { 'ed_ntt/summary.json' } else { 'mont_ntt_0599.json' }
        $prefix = if ($model -eq 'ed') { 'stage1-edwards-ntt-cgbn-prime95' } else { 'stage1-ntt-best-cgbn-prime95' }
        $plotArgs = @(
            'tools/bench/plot_stage1_ntt_best.py',
            '--ntt', "$BenchmarkRoot/gpu1/$curve/summary.json",
            '--previous-ntt', "data/experiments/stage1_forward_tile_0622/$curve/summary.json",
            '--previous-ntt', "data/experiments/stage1_ntt_segments/$curve/summary.json",
            '--previous-ntt', "data/experiments/stage1_unified_4060_0600/$priorNtt",
            '--overlay-ntt', "$BenchmarkRoot/gpu0/$curve/summary.json",
            '--overlay-ntt', "data/experiments/stage1_ntt_128_0624/${curve}128/summary.json",
            '--overlay-ntt', "data/experiments/stage1_4070ti_0623/$curve/summary.json",
            '--overlay-ntt', "data/experiments/stage1_4070ti_0623/mixed_$curve/summary.json",
            '--comparison', "$priorCgbn/gpu1_${model}_summary.json",
            '--overlay-cgbn', "$priorCgbn/gpu0_${model}_summary.json",
            '--cgbn-baseline', "$cgbn/gpu1/$model.json",
            '--cgbn-baseline', "$cgbn/gpu0/$model.json",
            '--hide-version', '0.62.3', '--hide-ladder-retests',
            '--device-mhz', '1500', '--overlay-mhz', '2600',
            '--output', "$OutputDirectory/$model"
        )
        if ($model -eq 'ed') {
            $plotArgs += @('--prime95-stage1', "$prime95/worktodo_ecm_stage1_wide.txt", "$prime95/screen_ed_matrix.log")
        } else {
            $plotArgs += @('--prime95-stage1', "$prime95/worktodo_ecm_stage1_wide.txt", "$prime95/screen_mont_matrix.log",
                           '--prime95-stage1-missing', '61')
            foreach ($device in 0, 1) {
                $plotArgs += @('--cgbn-retest', "$device", "$paired/gpu$device/comparison.json")
            }
        }
        if ($GmpEcmCgbn) {
            $plotArgs += @('--gmp-ecm-cgbn', $GmpEcmCgbn)
        }
        & python @plotArgs | Out-Host
        if ($LASTEXITCODE) { throw "Main Stage1 plotting failed for $model with exit code $LASTEXITCODE" }
        "$OutputDirectory/$model/$prefix-log.png"
        "$OutputDirectory/$model/$prefix-linear.png"
    }
    & python tools/bench/plot_stage1_version_throughput.py --comparison-root $OutputDirectory `
        --ntt-matrix-root $BenchmarkRoot --output-dir $VersionOutputDirectory --summary $VersionSummary
    if ($LASTEXITCODE) { throw "Version Stage1 plotting failed with exit code $LASTEXITCODE" }
    $images += "$VersionOutputDirectory/stage1-ed-version-throughput.png", "$VersionOutputDirectory/stage1-mont0-version-throughput.png"
    $images | Get-Item | Select-Object Name, Length
    if ($Publish) {
        foreach ($image in $images) {
            Copy-Item -LiteralPath $image -Destination 'docs/performance/attachments'
        }
    }
} finally {
    Pop-Location
}
