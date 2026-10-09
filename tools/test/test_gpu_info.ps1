# test_gpu_info.ps1 -- `ecm_cuda.exe --gpu-info` (docs/usage/GUI.md D4).
#
# --gpu-info answers "which kernel tier would you run, and how many curves does a full GPU
# need" WITHOUT running a curve. That is what the worktodo generator (M6) will consume, so
# this test pins both the contract and the numbers:
#   A. format     : every line is key=value (tier lines carry the leading `tier` token),
#                   exit 0, no timestamps, fast
#   B. numbers    : tier ladder ascending, tpi in {4,8,16,32}, ipb == tpb/tpi,
#                   curves_min == blocks_min*ipb, curves_wave == blocks_wave*ipb ==
#                   blocks_min_of_one_wave * blocks_per_sm * ipb (the kernel's own formula),
#                   blocks_min == sm_count
#   C. selection  : --bits N reports exactly the smallest tier with bits >= N + carry_bits
#   D. errors     : -d out of range and an --bits no tier covers fail cleanly (exit 1)
#   E. no effects : reads the ini (CLI beats it) but writes nothing at all
#   F. OpenCL     : the same flag prints not_applicable and still exits 0
#
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gpu_info.ps1
#          [-Exe build_cuda_cmake\ecm_cuda.exe] [-OpenClExe build_reltest\ecm.exe]
param(
    [string]$Exe = 'build_cuda_cmake\ecm_cuda.exe',
    [string]$OpenClExe = '',
    [int]$Device = 0
)

$ErrorActionPreference = 'Continue'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not [System.IO.Path]::IsPathRooted($Exe)) { $Exe = Join-Path $repo $Exe }
if (-not (Test-Path $Exe)) { Write-Host "missing exe: $Exe" -ForegroundColor Red; exit 2 }

$work = Join-Path $repo '.bench_tmp\test_gpu_info'
if (Test-Path $work) { Remove-Item $work -Recurse -Force }
New-Item -ItemType Directory -Force -Path $work | Out-Null

$pass = 0; $fail = 0
function Check([string]$name, [bool]$ok, [string]$detail = '') {
    if ($ok) { Write-Host "  [PASS] $name"; $script:pass++ }
    else { Write-Host "  [FAIL] $name $(if ($detail) { "-- $detail" })"; $script:fail++ }
}

# Runs the driver with a pristine temp CWD and an ini that does not exist, so the CLI
# defaults are what is measured; returns @{ exit; lines; seconds }.
# (The array is called $argv on purpose: $args is PowerShell's automatic variable and
#  splatting a reassigned $args silently passes nothing.)
function Invoke-GpuInfo([string[]]$Extra) {
    $argv = @('--gpu-info') + $Extra + @('-ini', (Join-Path $work 'does_not_exist.ini'))
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $out = & $Exe @argv 2>&1 | ForEach-Object { "$_" }
    $sw.Stop()
    return @{ exit = $LASTEXITCODE; lines = $out; seconds = $sw.Elapsed.TotalSeconds }
}

function Get-Key([string[]]$Lines, [string]$Key) {
    $hit = $Lines | Where-Object { $_ -match "^$([regex]::Escape($Key))=" } | Select-Object -First 1
    if (-not $hit) { return $null }
    return ($hit -split '=', 2)[1].Trim()
}

# Parses `tier bits=.. tpb=.. ...` into an array of hashtables.
function Get-Tiers([string[]]$Lines) {
    $tiers = @()
    foreach ($l in $Lines) {
        if ($l -notmatch '^tier\s') { continue }
        $t = @{}
        foreach ($tok in ($l -split '\s+')) {
            if ($tok -eq 'tier') { continue }
            if ($tok -match '^([a-z_]+)=(.*)$') { $t[$Matches[1]] = [int64]$Matches[2] }
        }
        $tiers += , $t
    }
    return , $tiers      # the leading comma keeps an empty result an ARRAY, not $null
}

# ------------------------------------------------------------------ A. format ------------
Write-Host "=== A. format (device $Device) ==="
$r = Invoke-GpuInfo @('-d', "$Device")
$allLines = @($r.lines)
Write-Host "  (exit $($r.exit), $([math]::Round($r.seconds,2)) s, $($allLines.Count) lines)"
foreach ($l in $allLines) { Write-Host "    $l" }

if ((Get-Key $allLines 'gpu_info') -eq 'not_applicable') {
    Check "OpenCL build says not_applicable" $true
    Check "OpenCL exit code is 0"           ($r.exit -eq 0)
    Check "OpenCL names the backend"        ((Get-Key $allLines 'backend').Length -gt 0)
    Check "OpenCL gives a reason"           ((Get-Key $allLines 'reason') -match 'not applicable')
    Write-Host ""
    Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
    if ($fail -gt 0) { exit 1 }
    exit 0
}

Check "exit code 0"                       ($r.exit -eq 0)
Check "gpu_info=1"                        ((Get-Key $allLines 'gpu_info') -eq '1')
Check "backend is CUDA/CGBN"              ((Get-Key $allLines 'backend') -eq 'CUDA/CGBN')
Check "device echoes -d"                  ((Get-Key $allLines 'device') -eq "$Device")
Check "name is a GPU name"                ((Get-Key $allLines 'name').Length -gt 3)
Check "sm_count > 0"                      ([int](Get-Key $allLines 'sm_count') -gt 0)
Check "cc looks like major.minor"         ((Get-Key $allLines 'cc') -match '^\d+\.\d+$')
Check "carry_bits = 6"                    ((Get-Key $allLines 'carry_bits') -eq '6')
Check "gpu_param = 3 by default"          ((Get-Key $allLines 'gpu_param') -eq '3')
Check "picked = 0 without --bits"         ((Get-Key $allLines 'picked') -eq '0')
Check "every non-tier line is key=value"  (@($allLines | Where-Object { $_ -notmatch '^tier\s' -and $_ -notmatch '^[a-z_]+=' }).Count -eq 0)
Check "every tier token is key=value"     (@($allLines | Where-Object { $_ -match '^tier\s' } | ForEach-Object { ($_ -split '\s+') | Where-Object { $_ -ne 'tier' -and $_ -notmatch '^[a-z_]+=' } }).Count -eq 0)
Check "no timestamps on the output"       (@($allLines | Where-Object { $_ -match '^\[' }).Count -eq 0)
Check "finishes in under 15 s"            ($r.seconds -lt 15)

$tiers = Get-Tiers $allLines
$sm = [int](Get-Key $allLines 'sm_count')
Check "tier_count matches the tier lines" ([int](Get-Key $allLines 'tier_count') -eq $tiers.Count)
Check "at least one tier"                 ($tiers.Count -gt 0)

# ------------------------------------------------------------------ B. numbers -----------
Write-Host "=== B. tier numbers ==="
$ascending = $true; $prev = 0
foreach ($t in $tiers) { if ($t['bits'] -le $prev) { $ascending = $false }; $prev = $t['bits'] }
Check "bits ascend strictly"              $ascending
Check "tpb is a plausible launch size"    (@($tiers | Where-Object { @(64,128,256,512,1024) -notcontains $_['tpb'] }).Count -eq 0)
Check "tpi is one of 4/8/16/32"           (@($tiers | Where-Object { @(4,8,16,32) -notcontains $_['tpi'] }).Count -eq 0)
Check "tpi never decreases with bits"     (@(0..($tiers.Count-2) | Where-Object { $tiers[$_+1]['tpi'] -lt $tiers[$_]['tpi'] }).Count -eq 0)
Check "ipb == tpb / tpi"                  (@($tiers | Where-Object { $_['ipb'] -ne [int]($_['tpb'] / $_['tpi']) }).Count -eq 0)
Check "blocks_min == sm_count"            (@($tiers | Where-Object { $_['blocks_min'] -ne $sm }).Count -eq 0)
Check "curves_min == blocks_min * ipb"    (@($tiers | Where-Object { $_['curves_min'] -ne $_['blocks_min'] * $_['ipb'] }).Count -eq 0)
Check "blocks_per_sm >= 1"                (@($tiers | Where-Object { $_['blocks_per_sm'] -lt 1 }).Count -eq 0)
Check "blocks_wave == sm_count * blocks_per_sm" (@($tiers | Where-Object { $_['blocks_wave'] -ne $sm * $_['blocks_per_sm'] }).Count -eq 0)
Check "curves_wave == blocks_wave * ipb"  (@($tiers | Where-Object { $_['curves_wave'] -ne $_['blocks_wave'] * $_['ipb'] }).Count -eq 0)
Check "curves_wave >= curves_min"         (@($tiers | Where-Object { $_['curves_wave'] -lt $_['curves_min'] }).Count -eq 0)
# The dev build carries only the small tiers; the full build must cover a 1024-bit N.
Check "the ladder starts at 128 bits"     ($tiers[0]['bits'] -eq 128)

# ------------------------------------------------------------------ C. selection ---------
Write-Host "=== C. --bits picks the smallest fitting tier ==="
$carry = [int](Get-Key $allLines 'carry_bits')
$probes = @(256, 700, 1024, 3000, 9000)
foreach ($n in $probes) {
    $rp = Invoke-GpuInfo @('-d', "$Device", '--bits', "$n")
    $pt = Get-Tiers @($rp.lines)
    $expected = $tiers | Where-Object { $_['bits'] -ge ($n + $carry) } | Select-Object -First 1
    if (-not $expected) {
        Check "--bits $n fails when no tier fits" ($rp.exit -ne 0)
        continue
    }
    $ok = ($rp.exit -eq 0) -and ($pt.Count -eq 1) -and ($pt[0]['bits'] -eq $expected['bits']) -and
          ($pt[0]['tpi'] -eq $expected['tpi']) -and ((Get-Key @($rp.lines) 'picked') -eq '1')
    Check "--bits $n -> $($expected['bits'])-bit tier" $ok `
        "got $(if ($pt.Count) { "$($pt[0]['bits']) bits" } else { 'nothing' }), exit $($rp.exit)"
}

# A request below the smallest tier still resolves to that tier (N < 128 bits).
$rlow = Invoke-GpuInfo @('--bits', '64')
$tlow = Get-Tiers @($rlow.lines)
Check "--bits 64 -> the 128-bit tier" (($rlow.exit -eq 0) -and $tlow.Count -eq 1 -and $tlow[0]['bits'] -eq $tiers[0]['bits'])

# ------------------------------------------------------------------ D. errors ------------
Write-Host "=== D. error paths ==="
$rbad = Invoke-GpuInfo @('-d', '99')
Check "-d 99 exits 1"                     ($rbad.exit -ne 0)
Check "-d 99 explains itself"             ((@($rbad.lines) -join ' ') -match 'out of range')
$rhuge = Invoke-GpuInfo @('--bits', '100000000')
Check "--bits 100000000 exits 1"          ($rhuge.exit -ne 0)
Check "--bits 100000000 explains itself"  ((@($rhuge.lines) -join ' ') -match 'no tier|error')

# ------------------------------------------------------------------ E. no side effects ---
Write-Host "=== E. reads the ini, writes nothing ==="
$ini = Join-Path $work 'probe.ini'
@"
# probe ini for --gpu-info
gpu_param = 0
device = 0
save_sync_dir_1 = Z:\definitely\not\here
[Worker #1]
gpu_param = 0
[Worker #2]
gpu_param = 2
device = 0
"@ | Set-Content -Path $ini -Encoding ASCII

$before = @(Get-ChildItem $work -Recurse -File | Select-Object -ExpandProperty FullName)
$rini = (& $Exe --gpu-info -ini $ini 2>&1 | ForEach-Object { "$_" })
$eini = $LASTEXITCODE
$after = @(Get-ChildItem $work -Recurse -File | Select-Object -ExpandProperty FullName)
Check "ini run exits 0"                   ($eini -eq 0)
Check "gpu_param comes from the ini"      ((Get-Key $rini 'gpu_param') -eq '0')
Check "ini is named in the output"        ((Get-Key $rini 'ini') -match 'probe\.ini')
Check "no file was created"               (($after.Count -eq $before.Count))
Check "no log/tmp/save in the temp dir"   (@($after | Where-Object { $_ -match '\.(log|tmp|save|ini)$' }).Count -eq 1)  # only probe.ini

$rcli = (& $Exe --gpu-info -ini $ini --gpu-param 2 2>&1 | ForEach-Object { "$_" })
Check "--gpu-param beats the ini"         ((Get-Key $rcli 'gpu_param') -eq '2')
$rworker = (& $Exe --gpu-info -ini $ini --worker 2 2>&1 | ForEach-Object { "$_" })
Check "worker 2 reads its own section"    ((Get-Key $rworker 'gpu_param') -eq '2')
Check "worker is echoed"                  ((Get-Key $rworker 'worker') -eq '2')

# The CUDA run path prints its own choice as "CGBN<TPI, BITS> kernel, N is k bits", so the
# report is checked against the run itself and not just against its own arithmetic.
Write-Host "=== E2. the run path agrees with the report ==="
$nfile = Join-Path $work 'n.txt'
# Exactly 63 bits (2^63 - 25); a literal, because PowerShell's 2**63 is a double.
[System.IO.File]::WriteAllText($nfile, "9223372036854775783`n", ([System.Text.Encoding]::ASCII))
$runOut = (& cmd /c "cd /d `"$work`" && `"$Exe`" -gpu -d $Device --gpu-param 0 -v -gpucurves 1 1e3 0 < `"$nfile`"" 2>&1 | Out-String)
$m = [regex]::Match($runOut, 'CGBN<(\d+),\s*(\d+)>')
if ($m.Success) {
    $runTpi = [int]$m.Groups[1].Value
    $runBits = [int]$m.Groups[2].Value
    $expected = $tiers | Where-Object { $_['bits'] -ge (63 + $carry) } | Select-Object -First 1
    Check "run picks the reported tier (CGBN<$runTpi, $runBits>)" `
        ($runBits -eq $expected['bits'] -and $runTpi -eq $expected['tpi']) `
        "report says bits=$($expected['bits']) tpi=$($expected['tpi'])"
    # The same numbers must come out of --bits for that N.
    $r63 = Invoke-GpuInfo @('-d', "$Device", '--gpu-param', '0', '--bits', '63')
    $t63 = Get-Tiers @($r63.lines)
    Check "--bits 63 matches the run" ($t63.Count -eq 1 -and $t63[0]['bits'] -eq $runBits -and $t63[0]['tpi'] -eq $runTpi)
    # ... and the run's own warning quotes the same curves_min.
    $wm = [regex]::Match($runOut, 'raise -gpucurves to about (\d+) \((\d+) curves/block')
    if ($wm.Success) {
        # @() matters: Where-Object returns a bare hashtable for a single hit, and
        # indexing THAT would look up key 0 (null) instead of element 0.
        $expectedMin = @($tiers | Where-Object { $_['bits'] -eq $runBits })[0]
        $saidCurves = [int]$wm.Groups[1].Value
        $saidIpb = [int]$wm.Groups[2].Value
        Check "the kernel's suggestion == curves_min ($saidCurves)" `
            ($saidCurves -eq $expectedMin['curves_min'] -and $saidIpb -eq $expectedMin['ipb'])
    } else {
        Write-Host "  [SKIP] the run printed no under-occupancy warning (gpucurves large enough)"
    }
} else {
    Check "the GPU run printed its tier choice (CGBN<tpi, bits>)" $false `
        "output was: $($runOut.Substring(0, [Math]::Min(400, $runOut.Length)))"
}

# ------------------------------------------------------------------ F. OpenCL -----------
if ($OpenClExe) {
    if (-not [System.IO.Path]::IsPathRooted($OpenClExe)) { $OpenClExe = Join-Path $repo $OpenClExe }
    Write-Host "=== F. OpenCL build ==="
    if (Test-Path $OpenClExe) {
        $ro = (& $OpenClExe --gpu-info 2>&1 | ForEach-Object { "$_" })
        $eo = $LASTEXITCODE
        Check "OpenCL --gpu-info exits 0"        ($eo -eq 0)
        Check "OpenCL prints not_applicable"     ((Get-Key $ro 'gpu_info') -eq 'not_applicable')
        Check "OpenCL prints a reason"           ((Get-Key $ro 'reason') -match 'OpenCL')
    } else {
        Write-Host "  [SKIP] $OpenClExe not built"
    }
} else {
    Write-Host "=== F. OpenCL build === (pass -OpenClExe to cover it)"
}

Write-Host ""
# The suite runner (test_gui_all.ps1) parses this exact line.
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($fail -gt 0) { exit 1 }
exit 0
