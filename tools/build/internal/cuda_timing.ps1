# Timing flags are appended only to the executed command. They do not change
# the semantic compile/link identity or cause an incremental rebuild.
function ConvertTo-EcmTimedNvccCommand([string]$Command, [string]$Build) {
    $python = (Get-Command python -ErrorAction Stop).Source
    $launcher = Join-Path $PSScriptRoot 'timed_nvcc.py'
    $prefix = '"' + $python + '" "' + $launcher + '" --root "' + $Build + '" -- '
    $pattern = '(?im)(^|&&\s*)("[^"\r\n]*[\\/]nvcc(?:\.exe)?"|(?:[^\s"]*[\\/])?nvcc(?:\.exe)?)(?=\s)'
    return [regex]::Replace($Command, $pattern, { param($match) $match.Groups[1].Value + $prefix + $match.Groups[2].Value })
}

function Enable-EcmNvccBatchTiming([string]$CommandFile, [string]$Build) {
    $content = [IO.File]::ReadAllText($CommandFile)
    [IO.File]::WriteAllText($CommandFile, (ConvertTo-EcmTimedNvccCommand $content $Build), [Text.UTF8Encoding]::new($false))
}

function New-EcmNvccTiming([string]$Build, [string]$Label, [string]$Command) {
    $directory = Join-Path $Build ('timing/' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $csv = Join-Path $directory 'nvcc.csv'
    return [pscustomobject]@{ Label = $Label; Csv = $csv; Record = (Join-Path $directory 'command.json')
        Command = $Command; ActualCommand = ($Command + ' --time "' + $csv + '"')
        Started = [DateTime]::UtcNow; Timer = [Diagnostics.Stopwatch]::StartNew() }
}

function Complete-EcmNvccTiming($Timing, [int]$ExitCode, [double]$ElapsedSeconds = -1) {
    $Timing.Timer.Stop()
    $present = (Test-Path -LiteralPath $Timing.Csv) -and (Get-Item -LiteralPath $Timing.Csv).Length -gt 0
    [ordered]@{ schema = 1; label = $Timing.Label; started_utc = $Timing.Started.ToString('o')
        command = $Timing.ActualCommand; semantic_command = $Timing.Command
        elapsed_seconds = $(if ($ElapsedSeconds -ge 0) { $ElapsedSeconds } else { $Timing.Timer.Elapsed.TotalSeconds }); exit_code = $ExitCode
        csv = $Timing.Csv; csv_present = $present; status = $(if ($ExitCode) { 'failed' } elseif ($present) { 'complete' } else { 'timing_missing' }) } |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Timing.Record -Encoding UTF8
    if ($ExitCode -eq 0 -and -not $present) { throw "NVCC succeeded without phase timing: $($Timing.Label), $($Timing.Record)" }
}
