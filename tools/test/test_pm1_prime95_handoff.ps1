# Exercise the three M400381 handoff directories produced by test_pm1_stage1_queue.py.
param(
    [Parameter(Mandatory)][string]$Prime95,
    [Parameter(Mandatory)][string]$Samples,
    [Parameter(Mandatory)][string]$Evidence
)
$ErrorActionPreference = "Stop"
$source = (Resolve-Path -LiteralPath $Samples).Path
$binary = (Resolve-Path -LiteralPath $Prime95).Path
foreach ($case in @("p95_stage2", "p95_forced", "p95_auto")) {
    $b1 = if ($case -eq "p95_forced") { 1000 } else { 500 }
    $root = Join-Path $source "$case/pm1/M400381-B1-$b1"
    $dir = (New-Item -Path (Join-Path $Evidence $case) -ItemType Directory).FullName
    # This Prime95 version's -w parser stops at whitespace.
    if ($dir -match '\s') { throw "Use a Prime95 test directory without spaces" }
    Copy-Item -LiteralPath (Join-Path $root "m0400381") -Destination $dir
    Copy-Item -LiteralPath (Join-Path $root "worktodo.add") -Destination $dir
    $settings = @("StressTester=0", "UsePrimenet=0", "WellBehavedWork=1", "AllowLowB1=1",
                  "KeepPminus1SaveFiles=1", "NumWorkers=1", "CoresPerWorker=2", "Memory=512", "ScreenLog=1")
    if ($case -ne "p95_auto") { $settings += "Pminus1BestB2=0" }
    if ($case -eq "p95_forced") { $settings += "Stage1GCD=-1" }
    $settings += @("[Windows]", "HideIcon=1", "TrayIcon=0")
    Set-Content -LiteralPath (Join-Path $dir "prime.txt") -Value $settings -Encoding ascii
    Set-Content -LiteralPath (Join-Path $dir "worktodo.txt") -Value "" -Encoding ascii
    $process = Start-Process -FilePath $binary -ArgumentList "-w$dir" -WorkingDirectory $dir -WindowStyle Hidden -PassThru
    try {
        $report = Join-Path $dir "results.json.txt"
        $deadline = [DateTime]::UtcNow.AddSeconds(120)
        while (-not (Test-Path -LiteralPath $report) -and [DateTime]::UtcNow -lt $deadline -and -not $process.HasExited) {
            Start-Sleep -Milliseconds 500
        }
        $row = (Get-Content -LiteralPath $report -Tail 1) | ConvertFrom-Json
        if ($row.status -ne "F" -or $row.b1 -ne $b1 -or $row.b2 -le $b1 -or $row.worktype -ne "P-1") {
            throw "Unexpected Prime95 Stage2 result in $report"
        }
        if (-not @($row.factors | Where-Object { ([bigint]::Parse($_) % 671839319) -eq 0 }).Count) {
            throw "Prime95 missed the known factor 671839319"
        }
        $screen = Get-Content -LiteralPath (Join-Path $dir "screen.log") -Raw
        if ($screen -notmatch 'Inversion of stage 1 result complete' -or $screen -match 'M400381 stage 1 complete') {
            throw "Prime95 did not resume at the Stage1 boundary; inspect $dir/screen.log"
        }
        "$case passed; Prime95 B1=$b1 B2=$($row.b2) factors=$($row.factors)"
    } finally {
        if (-not $process.HasExited) {
            [void]$process.CloseMainWindow()
            if (-not $process.WaitForExit(2000)) { $process.Kill() }
        }
        $process.Dispose()
    }
}
