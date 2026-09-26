# ---------------------------------------------------------------------------
# test_mont_checkpoint.ps1 -- end-to-end mid-stage-1 checkpoint test for the
# Suyama-sigma Montgomery CPU path (docs/ECM_Montgomery_STAGE1.md 搂17).
#
#   E1 fixed sigma : a run killed mid-ladder and then resumed must produce a .save
#                    file BYTE-IDENTICAL to one uninterrupted run with the same
#                    command line.  This is the whole contract of a checkpoint.
#   E2 random sigma: a killed run's checkpoints pin the sigmas of curves that were
#                    still queued; the resumed run must adopt them instead of
#                    drawing new ones (otherwise "resume" would silently change the
#                    curve set).  Checked by comparing the SIGMA values in the
#                    final .save against the SIGMA lines of the pre-resume .ckpt
#                    files.
#   E3 scalar path : the same kill/resume on --backend gmp (one curve per task).
#
# usage:  pwsh -File tools\test\test_mont_checkpoint.ps1
# ---------------------------------------------------------------------------
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$exe  = Join-Path $root 'build_vs18\Release\ecm.exe'
if (-not (Test-Path $exe)) { throw "ecm.exe not found: $exe (build the ecm target first)" }

$base = Join-Path $root 'build_vs18\test_mont_ckpt'
if (Test-Path $base) { Remove-Item -Recurse -Force $base }
New-Item -ItemType Directory -Path $base | Out-Null

# M1277, B1 = 1e6: ~0.98 s per curve on this machine, 32 curves over 4 threads =
# 4 batches of 8, so a kill after ~3 s lands in the middle of every batch.
$N     = '(2^1277-1)'
$B1    = '1e6'
$CURVES = 32
$THREADS = 4
$KILL_AFTER_S = 3

function Run-Ecm {
    param([string[]]$EcmArgs, [string]$Dir, [int]$TimeoutMs = 600000)
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    $job = Start-Job -ScriptBlock {
        param($exe, $eargs, $dir, $n)
        Set-Location $dir
        $n | & $exe @eargs 2>&1
    } -ArgumentList $exe, $EcmArgs, $Dir, $N
    $done = Wait-Job $job -Timeout ($TimeoutMs / 1000)
    if (-not $done) { Stop-Job $job; Receive-Job $job | Out-Null; Remove-Job $job -Force; return @{ text = ''; killed = $true } }
    $text = Receive-Job $job
    Remove-Job $job -Force
    return @{ text = ($text | Out-String); killed = $false }
}

function Start-EcmKilled {
    param([string[]]$EcmArgs, [string]$Dir, [int]$AfterSeconds)
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.WorkingDirectory = $Dir
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    # Windows PowerShell 5.1: ProcessStartInfo has no ArgumentList, so build the
    # command line here (no argument in this test contains a space).
    $psi.Arguments = (($EcmArgs | ForEach-Object { '"' + $_ + '"' }) -join ' ')
    $p = [System.Diagnostics.Process]::Start($psi)
    $p.StandardInput.WriteLine($N)
    $p.StandardInput.Close()
    Start-Sleep -Seconds $AfterSeconds
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()
    if (-not $p.HasExited) { $p.Kill() }
    $p.WaitForExit()
    return @{ text = ($outTask.Result + $errTask.Result); exit = $p.ExitCode }
}

$fails = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  [PASS] $what" -ForegroundColor Green }
    else     { Write-Host "  [FAIL] $what" -ForegroundColor Red; $script:fails++ }
}

function Save-Path([string]$dir, [string]$name) { Join-Path $dir $name }

# The save lines carry WHO=<user@host> and TIME=<clock>, which differ between two
# runs by construction, so the comparison has to be on the mathematical content:
# one normalized line per curve (SIGMA, B1, N, X, CHECKSUM).
function Norm-Save([string]$save) {
    if (-not (Test-Path $save)) { return @() }
    $lines = Get-Content $save | ForEach-Object {
        (($_ -replace ' WHO=[^;]*;', '') -replace ' TIME=[^;]*;', '').Trim()
    }
    return ($lines | Where-Object { $_ -ne '' } | Sort-Object)
}
function Sigma-Set([string]$save) {
    if (-not (Test-Path $save)) { return @() }
    $t = Get-Content -Raw $save
    return [regex]::Matches($t, 'SIGMA=(\d+)') | ForEach-Object { $_.Groups[1].Value }
}
function Ckpt-Sigmas([string]$dir) {
    $set = @()
    Get-ChildItem -Path $dir -Filter '*_c*.ckpt' -ErrorAction SilentlyContinue | ForEach-Object {
        $m = Select-String -Path $_.FullName -Pattern '^SIGMA=(\d+)'
        if ($m) { $set += $m.Matches[0].Groups[1].Value }
    }
    return $set
}

$common = @('--method','mont','--tmp-dir','.','-gpucurves',"$CURVES",'--stage1-threads',"$THREADS",$B1)

Write-Host "=== E1: fixed sigma, kill + resume vs uninterrupted (identical content) ==="
$dirA = Join-Path $base 'e1_ref'
$dirB = Join-Path $base 'e1_resume'
$argsA = $common + @('-sigma','123456789012345','--ckpt','0')
$argsB = $common + @('-sigma','123456789012345','--ckpt','1')
$ra = Run-Ecm -EcmArgs $argsA -Dir $dirA
Check (-not $ra.killed) 'reference run finished'
$rb = Start-EcmKilled -EcmArgs $argsB -Dir $dirB -AfterSeconds $KILL_AFTER_S
$ckBefore = Get-ChildItem -Path $dirB -Filter '*_c*.ckpt' -ErrorAction SilentlyContinue
Check ($ckBefore.Count -gt 0) "kill left checkpoints behind ($($ckBefore.Count) file(s))"
$inflight = ($ckBefore | Where-Object {
    (Select-String -Path $_.FullName -Pattern '^STATUS=INFLIGHT' -Quiet) -and
    (Select-String -Path $_.FullName -Pattern '^X0=' -Quiet) }).Count
Check ($inflight -gt 0) "checkpoints contain mid-ladder states ($inflight in flight)"
$rc = Run-Ecm -EcmArgs $argsB -Dir $dirB
Check ($rc.text -match 'resumed from checkpoint') 'the second run reports a resume'
$saveA = Save-Path $dirA 'm1277_1e6.save'
$saveB = Save-Path $dirB 'm1277_1e6.save'
Check (Test-Path $saveA) 'reference save exists'
Check (Test-Path $saveB) 'resumed save exists'
if ((Test-Path $saveA) -and (Test-Path $saveB)) {
    $na = Norm-Save $saveA
    $nb = Norm-Save $saveB
    $same = ($na.Count -eq $nb.Count) -and (-not (Compare-Object $na $nb))
    Check $same "the resumed save holds the same $($nb.Count) curve line(s) (SIGMA/B1/N/X/CHECKSUM)"
    if (-not $same) {
        $d = Compare-Object $na $nb | Select-Object -First 4
        $d | ForEach-Object { Write-Host ("      " + $_.SideIndicator + " " + $_.InputObject.Substring(0, [Math]::Min(90, $_.InputObject.Length))) }
    }
}
$left = Get-ChildItem -Path $dirB -Filter '*_c*.ckpt' -ErrorAction SilentlyContinue
Check ($left.Count -eq 0) 'checkpoints are removed once the save is written'

Write-Host ""
Write-Host "=== E2: random sigmas are adopted from the checkpoints ==="
$dirC = Join-Path $base 'e2_resume'
$argsC = $common + @('--ckpt','1')
$null = Start-EcmKilled -EcmArgs $argsC -Dir $dirC -AfterSeconds $KILL_AFTER_S
$sigBefore = Ckpt-Sigmas $dirC
Check ($sigBefore.Count -gt 0) "killed run seeded $($sigBefore.Count) curve sigma(s)"
$rd = Run-Ecm -EcmArgs $argsC -Dir $dirC
Check ($rd.text -match 'resumed from checkpoint') 'the resume run reports restored curves'
$saveC = Save-Path $dirC 'm1277_1e6.save'
$sigAfter = Sigma-Set $saveC
Check ($sigAfter.Count -eq $CURVES) "the save has $CURVES curve lines"
$kept = 0
foreach ($s in $sigBefore) { if ($sigAfter -contains $s) { $kept++ } }
Check ($kept -ge ($sigBefore.Count - 1)) "adopted sigmas survive into the save ($kept/$($sigBefore.Count))"

Write-Host ""
Write-Host "=== E3: scalar backend (--backend gmp) ==="
$dirD = Join-Path $base 'e3_scalar_ref'
$dirE = Join-Path $base 'e3_scalar_resume'
# one curve per task on the scalar backend, ~4 s/curve at B1 = 1e6 on this machine,
# with a 0.5 s autosave interval, so a kill after 1.5 s lands mid-ladder and after
# at least one automatic checkpoint
$scalarCommon = @('--method','mont','--backend','gmp','--tmp-dir','.','-gpucurves','8',
                  '--stage1-threads','2','-sigma','777000111','--ckpt','0.5','1e6')
$rd1 = Run-Ecm -EcmArgs $scalarCommon -Dir $dirD
Check (-not $rd1.killed) 'scalar reference finished'
$rd2 = Start-EcmKilled -EcmArgs $scalarCommon -Dir $dirE -AfterSeconds 2
$sc = Get-ChildItem -Path $dirE -Filter '*_c*.ckpt' -ErrorAction SilentlyContinue
Check ($sc.Count -gt 0) "scalar kill left checkpoints ($($sc.Count))"
$scMid = ($sc | Where-Object { Select-String -Path $_.FullName -Pattern '^X0=' -Quiet }).Count
Check ($scMid -gt 0) "scalar checkpoints hold mid-ladder states ($scMid)"
$rd3 = Run-Ecm -EcmArgs $scalarCommon -Dir $dirE
Check ($rd3.text -match 'resumed from checkpoint') 'the scalar resume reports restored curves'
$sA = Save-Path $dirD 'm1277_1e6.save'
$sB = Save-Path $dirE 'm1277_1e6.save'
if ((Test-Path $sA) -and (Test-Path $sB)) {
    $na = Norm-Save $sA
    $nb = Norm-Save $sB
    $same = ($na.Count -eq $nb.Count) -and (-not (Compare-Object $na $nb))
    Check $same "the scalar resumed save holds the same $($nb.Count) curve line(s)"
} else {
    Check $false 'scalar saves exist'
}

Write-Host ""
if ($fails -eq 0) { Write-Host 'ALL OK' -ForegroundColor Green } else { Write-Host "FAILURES: $fails" -ForegroundColor Red }
exit $fails


