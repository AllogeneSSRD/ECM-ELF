#Requires -Version 5.1
<#
.SYNOPSIS
    Snapshot GPU-Z sensors and summarize low-load intervals against a Stage2 A/B run.
.DESCRIPTION
    GPU Load is sampled busy time, not SM occupancy. Phase boundaries inferred from existing
    wall timers are approximate. GPU-Z does not identify the device in its CSV header.
#>
param(
    [string]$SensorLog = 'C:\Users\Elysia\Documents\GPU-Z Sensor Log.txt',
    [string]$BenchOutput = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
if (-not $Output) { $Output = Join-Path $repo ('build_cuda_cmake\_gpuz_' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$captured = Get-Date
$lines = @(Get-Content -LiteralPath $SensorLog)
$headers = @($lines[0].Split(',') | ForEach-Object { $_.Trim() })
$dateIndex = [Array]::IndexOf($headers, 'Date')
$loadIndex = [Array]::IndexOf($headers, 'GPU Load [%]')
$clockIndex = [Array]::IndexOf($headers, 'GPU Clock [MHz]')
$memoryIndex = [Array]::IndexOf($headers, 'Memory Used [MB]')
$controllerIndex = [Array]::IndexOf($headers, 'Memory Controller Load [%]')
$busIndex = [Array]::IndexOf($headers, 'Bus Interface Load [%]')
$cpuIndex = -1
for ($i=0; $i -lt $headers.Count; ++$i) {
    if ($headers[$i] -like 'CPU Temperature*') { $cpuIndex=$i; break }
}
$indices = @($dateIndex,$loadIndex,$clockIndex,$memoryIndex,$controllerIndex,$busIndex)
if (($indices | Measure-Object -Minimum).Minimum -lt 0) { throw 'GPU-Z header lacks required sensors' }
$maxIndex = ($indices | Measure-Object -Maximum).Maximum
$culture = [Globalization.CultureInfo]::InvariantCulture
$samples = @(
    foreach ($line in $lines | Select-Object -Skip 1) {
        $parts = $line.Split(',')
        if ($parts.Count -le $maxIndex) { continue }  # writer may have an incomplete last row
        $stamp = [datetime]::MinValue
        if (-not [datetime]::TryParseExact($parts[$dateIndex].Trim(), 'yyyy-MM-dd HH:mm:ss',
                 $culture, [Globalization.DateTimeStyles]::None, [ref]$stamp)) { continue }
        $values = @(); $valid = $true
        foreach ($index in @($loadIndex,$clockIndex,$memoryIndex,$controllerIndex,$busIndex)) {
            $value = 0.0
            if (-not [double]::TryParse($parts[$index].Trim(), [Globalization.NumberStyles]::Float,
                                      $culture, [ref]$value)) { $valid = $false; break }
            $values += $value
        }
        if ($valid) {
            $cpuTemp = $null
            if ($cpuIndex -ge 0 -and $parts.Count -gt $cpuIndex) {
                $value = 0.0
                if ([double]::TryParse($parts[$cpuIndex].Trim(), [Globalization.NumberStyles]::Float,
                                      $culture, [ref]$value)) { $cpuTemp = $value }
            }
            [pscustomobject]@{ time=$stamp; gpu_load_pct=$values[0]; clock_mhz=$values[1];
                memory_mb=$values[2]; memory_controller_pct=$values[3]; bus_interface_pct=$values[4];
                cpu_temperature_c=$cpuTemp }
        }
    }
)
if ($samples.Count -lt 2) { throw 'fewer than two complete GPU-Z samples' }
$samples | Export-Csv -LiteralPath (Join-Path $Output 'sensors.csv') -NoTypeInformation -Encoding UTF8
function Summarize-Samples($items) {
    $cpuItems = @($items | Where-Object { $null -ne $_.cpu_temperature_c })
    $cpuMean = $null
    if ($cpuItems.Count) { $cpuMean = [math]::Round(($cpuItems | Measure-Object cpu_temperature_c -Average).Average,3) }
    [pscustomobject]@{ samples=$items.Count;
        mean_load_pct=[math]::Round(($items | Measure-Object gpu_load_pct -Average).Average,3);
        mean_clock_mhz=[math]::Round(($items | Measure-Object clock_mhz -Average).Average,3);
        mean_memory_mb=[math]::Round(($items | Measure-Object memory_mb -Average).Average,3);
        peak_memory_mb=($items | Measure-Object memory_mb -Maximum).Maximum;
        mean_memory_controller_pct=[math]::Round(($items | Measure-Object memory_controller_pct -Average).Average,3);
        mean_bus_interface_pct=[math]::Round(($items | Measure-Object bus_interface_pct -Average).Average,3);
        mean_cpu_temperature_c=$cpuMean;
        low_le5_samples=@($items | Where-Object gpu_load_pct -LE 5).Count;
        low_lt20_samples=@($items | Where-Object gpu_load_pct -LT 20).Count }
}
$intervals = @(); $begin = $null; $last = $null; $count = 0
foreach ($sample in $samples) {
    if ($sample.gpu_load_pct -le 5) {
        if ($null -ne $begin -and ($sample.time - $last).TotalSeconds -gt 1.5) {
            $intervals += [pscustomobject]@{ start=$begin; end=$last; samples=$count }
            $begin = $null; $count = 0
        }
        if ($null -eq $begin) { $begin = $sample.time }
        $last = $sample.time; ++$count
    } elseif ($null -ne $begin) {
        $intervals += [pscustomobject]@{ start=$begin; end=$last; samples=$count }
        $begin = $null; $count = 0
    }
}
if ($null -ne $begin) { $intervals += [pscustomobject]@{ start=$begin; end=$last; samples=$count } }
$coverage = @()
if ($BenchOutput) {
    $provenance = Get-Content -LiteralPath (Join-Path $BenchOutput 'provenance.json') -Raw | ConvertFrom-Json
    $cursor = [datetime]::Parse($provenance.started).ToLocalTime()
    foreach ($run in (Import-Csv -LiteralPath (Join-Path $BenchOutput 'results.csv'))) {
        $inferred = -not $run.started
        $start = $(if ($inferred) { $cursor } else { [datetime]::Parse($run.started).ToLocalTime() })
        $end = $(if ($inferred) { $start.AddSeconds([double]$run.wall) }
                 else { [datetime]::Parse($run.ended).ToLocalTime() })
        $cursor = $end
        $text = Get-Content -LiteralPath $run.log -Raw
        $phases = [regex]::Match($text, 'real_batched_wall: pre=([0-9.]+) loop_wall=([0-9.]+) post=([0-9.]+)')
        $stageStart = $end.AddSeconds(-[double]$run.elapsed)
        $ranges = @([pscustomobject]@{phase='setup_and_baby';start=$start;end=$stageStart})
        if ($phases.Success) {
            $loopStart = $stageStart.AddSeconds([double]$phases.Groups[1].Value)
            $postStart = $loopStart.AddSeconds([double]$phases.Groups[2].Value)
            $ranges += [pscustomobject]@{phase='pre_Ftree_reciprocal';start=$stageStart;end=$loopStart}
            $ranges += [pscustomobject]@{phase='giant_Gtree_fold_loop';start=$loopStart;end=$postStart}
            $ranges += [pscustomobject]@{phase='descent_accum_naming';start=$postStart;end=$end}
        }
        foreach ($range in $ranges) {
            $items = @($samples | Where-Object { $_.time -ge $range.start -and $_.time -lt $range.end })
            if ($items.Count -eq 0) { continue }
            $coverage += [pscustomobject]@{ run=[int]$run.run;mode=$run.mode;phase=$range.phase;
                approximate_start=$range.start;approximate_end=$range.end;
                run_time_inferred=$inferred;summary=(Summarize-Samples $items) }
        }
    }
}
$summary = [pscustomobject]@{ source=$SensorLog;captured=$captured;first=$samples[0].time;
    last=$samples[-1].time;metrics=(Summarize-Samples $samples);low_load_intervals=$intervals;
    run_phase_coverage=$coverage;
    interpretation='GPU Load is sampled busy time, not SM occupancy. Phase alignment is approximate; legacy run times are inferred from cumulative wall time. The log header does not identify the device.' }
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Output 'summary.json') -Encoding UTF8
$summary.metrics | Format-List
Write-Host "Saved sensors.csv and summary.json to $Output"
