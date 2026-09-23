# Integration test for ecm_p95feeder against a sandboxed Prime95 directory.
#
# Self-contained: stage 0 runs the real ecm.exe to produce local .tmp saves, then
# a fake Prime95 dir is driven through 7 poll cycles. Scratch dir:
# tools/test/_feeder_scratch (git-ignored).
$ErrorActionPreference = "Continue"
$root = "D:\code\MPA-OpenCl"
$t = "$root\tools\test\_feeder_scratch"
$ecm = "$root\build_vs18\Release\ecm.exe"
$feeder = "$root\build_vs18\Release\ecm_p95feeder.exe"
$local = "$t\local"
$p95 = "$t\p95"
$alt = "$t\alt"

function Show-State($label) {
    Write-Host "--- $label ---"
    Write-Host "  worktodo.add:"
    if (Test-Path "$p95\worktodo.add") {
        Get-Content "$p95\worktodo.add" | ForEach-Object { "    |$_" }
    } else { Write-Host "    (absent)" }
    Write-Host "  p95 saves: $((Get-ChildItem $p95 -Filter 'e0*' -ErrorAction SilentlyContinue | ForEach-Object { $_.Name }) -join ', ')"
    Write-Host "  local .tmp: $((Get-ChildItem $local -Filter '*.tmp' | ForEach-Object { $_.Name }) -join ', ')"
}

# ---- stage 0: produce real stage-1 saves with ecm.exe (local only) --------
if (-not (Test-Path $ecm) -or -not (Test-Path $feeder)) {
    Write-Host "ERROR: build ecm + ecm_p95feeder first (cmake --build build_vs18 --config Release)"
    exit 1
}
Remove-Item -Recurse -Force $t -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $local, $p95, $alt | Out-Null

Write-Host "=== stage 0: generating saves with ecm.exe ==="
"2^991-1" | & $ecm --edwards -sigma 105413044550089 -gpucurves 4 --tmp-dir $local 1000000 2>&1 |
    Select-String -Pattern "stage1 save" | ForEach-Object { "  $($_.Line)" }
"2^677-1" | & $ecm --edwards -sigma 6581585141005897 -gpucurves 2 --tmp-dir $local 1000000 2>&1 |
    Select-String -Pattern "stage1 save" | ForEach-Object { "  $($_.Line)" }
# one save whose B1 does NOT match the validation worktodo below (must be skipped).
# Local names carry B1 (e{n:07d}_B{B1}_c{curve}), so this one lands beside the
# B1=1000000 saves instead of overwriting them.
"2^991-1" | & $ecm --edwards -sigma 105413044550089 -gpucurves 1 --tmp-dir $alt 200000 2>&1 | Out-Null
Copy-Item "$alt\e0000991_B200000_c000001.tmp" "$local\" -Force
Write-Host "  local .tmp: $((Get-ChildItem $local -Filter '*.tmp' | ForEach-Object { $_.Name }) -join ', ')"

# ---- fake Prime95 installation -------------------------------------------
Set-Content -Path "$p95\prime.txt" -Encoding ASCII -Value @"
NumWorkers=4
MaxHighMemWorkers=2
[Internals]
Foo=1
"@
Set-Content -Path "$p95\worktodo.txt" -Encoding ASCII -Value @"
[Worker #1]

[Worker #2]

[Worker #3]
Pminus1=N/A,1,2,550169,-1,66427649,0,68,"3444032551901327"

[Worker #4]
"@

# ---- local worktodo used for the N/B1 validation --------------------------
Set-Content -Path "$local\worktodo.txt" -Encoding ASCII -Value @"
[Worker #1]
ECM=1,2,991,-1,1000000,0,100,"8218291649"
ECM=1,2,677,-1,1000000,0,100,"1943118631"
"@

Set-Content -Path "$t\feeder.ini" -Encoding ASCII -Value @"
tmp_dir = $local
p95_dir = $p95
worktodo = worktodo.txt
poll_seconds = 2
max_in_flight = 0
keep_tmp = 0
verbose = 1
"@

Write-Host "############ CYCLE 1 (dry-run) ############"
& $feeder --ini "$t\feeder.ini" --once --dry-run
Show-State "after dry-run (must be unchanged)"

Write-Host ""
Write-Host "############ CYCLE 2 (real) ############"
& $feeder --ini "$t\feeder.ini" --once
Show-State "after cycle 2"

Write-Host ""
Write-Host "############ CYCLE 3 (nothing should change: in-flight cap = MaxHighMemWorkers = 2) ############"
& $feeder --ini "$t\feeder.ini" --once
Show-State "after cycle 3"

Write-Host ""
Write-Host "############ CYCLE 4: simulate Prime95 consuming worktodo.add ############"
if (Test-Path "$p95\worktodo.add") {
    $add = Get-Content "$p95\worktodo.add"
    # p95 appends each [Worker #] section's entries into worktodo.txt, then deletes worktodo.add
    $cur = Get-Content "$p95\worktodo.txt"
    $out = New-Object System.Collections.Generic.List[string]
    $pending = @{}
    $w = 0
    foreach ($l in $add) {
        if ($l -match '^\[Worker #(\d+)\]') { $w = [int]$Matches[1]; if (-not $pending.ContainsKey($w)) { $pending[$w] = @() }; continue }
        if ($l.Trim() -ne '' -and $w -gt 0) { $pending[$w] += $l }
    }
    $w = 0
    foreach ($l in $cur) {
        if ($l -match '^\[Worker #(\d+)\]') {
            $out.Add($l); $w = [int]$Matches[1]
            if ($pending.ContainsKey($w)) { foreach ($p in $pending[$w]) { $out.Add($p) }; $pending.Remove($w) }
            continue
        }
        $out.Add($l)
    }
    Set-Content -Path "$p95\worktodo.txt" -Value $out -Encoding ASCII
    Remove-Item "$p95\worktodo.add" -Force
}
Show-State "after simulated p95 consume (worktodo.add gone, tasks now in worktodo.txt)"

Write-Host ""
Write-Host "############ CYCLE 5 (still in flight -> no new delivery) ############"
& $feeder --ini "$t\feeder.ini" --once
Show-State "after cycle 5"

Write-Host ""
Write-Host "############ CYCLE 6: simulate one task finishing (M677 line removed) ############"
$cur = Get-Content "$p95\worktodo.txt" | Where-Object { $_ -notmatch 'ECM=.*,677,' }
Set-Content -Path "$p95\worktodo.txt" -Value $cur -Encoding ASCII
& $feeder --ini "$t\feeder.ini" --once
Show-State "after cycle 6 (one slot freed -> next save delivered)"

Write-Host ""
Write-Host "############ CYCLE 7 (feed the remaining slot) ############"
$cur = Get-Content "$p95\worktodo.txt" | Where-Object { $_ -notmatch 'ECM=.*,991,' }
Set-Content -Path "$p95\worktodo.txt" -Value $cur -Encoding ASCII
& $feeder --ini "$t\feeder.ini" --once
Show-State "after cycle 7"

Write-Host ""
Write-Host "############ CYCLE 8: only the mismatched-B1 save is left -> must be skipped ############"
# Drop every in-flight handoff line and every other pending save, so the only
# candidate is e0000991_B200000_c000001.tmp, whose B1=200000 does not match the
# validating worktodo (which only lists B1=1000000 for n=991/677).
$cur = Get-Content "$p95\worktodo.txt" | Where-Object { $_ -notmatch '^ECM=' }
Set-Content -Path "$p95\worktodo.txt" -Value $cur -Encoding ASCII
Remove-Item "$p95\worktodo.add" -Force -ErrorAction SilentlyContinue
Get-ChildItem $local -Filter "*.tmp" | Where-Object { $_.Name -ne "e0000991_B200000_c000001.tmp" } |
    ForEach-Object { Remove-Item $_.FullName -Force }
Write-Host "  pending before cycle 8: $((Get-ChildItem $local -Filter '*.tmp' | ForEach-Object { $_.Name }) -join ', ')"
& $feeder --ini "$t\feeder.ini" --once
Write-Host "  remaining local .tmp: $((Get-ChildItem $local -Filter '*.tmp' | ForEach-Object { $_.Name }) -join ', ')"
Write-Host "  worktodo.add present: $(Test-Path "$p95\worktodo.add")  (expect False: rejected save must NOT be delivered)"

$mismatch_ok = (-not (Test-Path "$p95\worktodo.add")) -and (Test-Path "$local\e0000991_B200000_c000001.tmp")
Write-Host ""
Write-Host "############ RESULT ############"
Write-Host "  B1-mismatch save rejected and kept locally: $(if ($mismatch_ok) { 'PASS' } else { 'FAIL' })"
if (-not $mismatch_ok) { exit 1 }
exit 0
