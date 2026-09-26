# test_worktodo_pipeline.ps1 -- regression tests for tools/ecm_worktodo/ecm.py
#
# The generator is the layer in front of both consumers of the ECM split:
#   stage 1 : our driver (ecm_cuda.exe) reads the ECMSTAGE2= lines
#   stage 2 : Prime95 reads the ECM=/ECM2= lines
# so the tests below pin down (a) the wire formats against REAL fixtures that live in the
# repository, (b) the processing rules agreed in the design review (filter -> dedup ->
# rewrite -> sort), and (c) the failure modes that would silently break a handoff.
#
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_worktodo_pipeline.ps1
#        [-Python python] [-KeepWork]
param(
    [string]$Python = "python",
    [switch]$KeepWork
)

$ErrorActionPreference = 'Continue'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$tool = Join-Path $repo 'tools\ecm_worktodo\ecm.py'
$fixtures = Join-Path $repo 'tools\ecm_worktodo'
$work = Join-Path $repo '.bench_tmp\test_worktodo_pipeline'

if (-not (Test-Path $tool)) { Write-Host "missing $tool"; exit 2 }
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Force -Path $work | Out-Null

$pass = 0; $fail = 0; $skip = 0
function Check([string]$name, $ok, [string]$detail = '') {
    $ok = [bool]$ok
    if ($ok) { Write-Host "  [PASS] $name"; $script:pass++ }
    else { Write-Host "  [FAIL] $name"; if ($detail) { Write-Host "         $detail" }; $script:fail++ }
}
function Skip([string]$name, [string]$why) { Write-Host "  [SKIP] $name ($why)"; $script:skip++ }

# Run the generator; returns @{exit=..; out=..}
function Run-Py([string[]]$argv) {
    $out = & $Python $tool @argv 2>&1 | Out-String
    return @{ exit = $LASTEXITCODE; out = $out }
}
# Always a real array (call sites must still wrap with @() before indexing).
function Lines([string]$path) {
    if (-not (Test-Path $path)) { return @() }
    return @((Get-Content $path -Encoding UTF8) | Where-Object { $_ -ne '' })
}
function First([string]$path) { $a = @(Lines $path); if ($a.Count) { return [string]$a[0] } return '' }
function FileText([string]$path) { if (Test-Path $path) { (Get-Content $path -Raw) } else { '' } }

$sortedCsv = Join-Path $fixtures 'sorted.csv'                 # real assignment fixture
$sampleCsv = Join-Path $fixtures 'assignment_sample.csv'      # real fixture WITH BOM (UTF-8)
$p95Stage2 = Join-Path $fixtures 'worktodo_p95_stage2.txt'    # real product of the old tool

# The whole point of this suite is regression against the REAL samples, so a missing fixture is
# an environment error, not a skip: tools/ecm_worktodo/{sorted.csv,worktodo_p95_stage2.txt} must
# be committed (they are real assignment/product files, not scratch).
$missing = @($sortedCsv, $p95Stage2, $sampleCsv) | Where-Object { -not (Test-Path $_) }
if ($missing.Count) {
    Write-Host "error: missing fixture(s):"
    foreach ($m in $missing) { Write-Host "         $m" }
    Write-Host "       commit the real samples under tools/ecm_worktodo/ (git add) and re-run"
    exit 2
}

Write-Host "=== 1. input parsing: both prefixes, AID / N/A / no AID / FFT2= / sigma / factors ==="
$mixed = Join-Path $work 'mixed.csv'
@(
    'ECM2=331033CC5703176E20F666024A8B23B3,1,2,991,-1,1e6,0,3,"101,103"',
    'ECM=1,2,992,-1,1e6',
    'ECM=N/A,1,2,993,-1,1e6,0,1',
    'ECM=AID123,FFT2=192K,1,2,994,-1,1e6,0,1,105413044550089,"8218291649"',
    'ECM=1,2,995,-1,1e6,0,1,105413044550089',
    '# comment line',
    '',
    'ECMSTAGE2=1,2,996,-1,"x.save",0,0,3'
) | Set-Content -Path $mixed -Encoding ASCII
$r = Run-Py @('--input', $mixed, '--dry-run')
Check "reads ECM2=, ECM=, N/A, AID+FFT2=, sigma, factors (5 of 8 lines)" `
      (($r.out -match 'tasks read\s+:\s+5') -and ($r.out -match 'skipped \(comment\)\s+:\s+2') -and
       ($r.out -match 'skipped \(ECMSTAGE2\):\s+1')) $r.out

$r = Run-Py @('--input', $sampleCsv, '--dry-run')   # BOM + CRLF fixture
Check "reads a UTF-8-with-BOM, CRLF assignment file" ($r.out -match 'tasks read\s+:\s+2') $r.out

Write-Host "=== 2. regression against the REAL product of the old tool ==="
$out = Join-Path $work 'p95.txt'
$null = Run-Py @('--input', $sortedCsv, '--set-b1', '110e6', '--gpu-curves', '192',
                 '--sort-by', 'n', '--out-ecm', $out)
$mine = ((FileText $out) -replace "`r`n", "`n").TrimEnd()
$theirs = ((FileText $p95Stage2) -replace "`r`n", "`n").TrimEnd()
Check "our --out-ecm reproduces worktodo_p95_stage2.txt byte for byte" ($mine -eq $theirs) `
      ("first line: " + (First $out).Substring(0, [Math]::Min(90, (First $out).Length)))

Write-Host "=== 3. dedup identity (k,b,n,c) + conflict policy + factor union ==="
$dup = Join-Path $work 'dup.csv'
@(
    'ECM2=AAAA,1,2,991,-1,5e6,100000000,90,"103"',
    'ECM2=BBBB,1,2,991,-1,110000000,100000000,192,"101"',
    'ECM=1,2,991,-1,44e6,100000000,50',
    'ECM2=1,2,1200,-1,1e6,0,10'
) | Set-Content -Path $dup -Encoding ASCII
$stage2 = Join-Path $work 'dup_stage2.csv'
$r = Run-Py @('--input', $dup, '--sort-factors', '--out-ecmstage2', $stage2)
$lines = @(Lines $stage2)
Check "3 duplicate rows collapse to 1 (n=991 keeps the largest B1/curves and a real AID)" `
      (($lines.Count -eq 2) -and ($lines[0] -like 'ECMSTAGE2=BBBB,1,2,991,-1,*')) ($lines -join ' | ')
Check "known factors are UNIONed across the group" ([bool]($lines[0] -match '"101,103"')) $lines[0]
Check "winner's B2/curves/spelling survive" ([bool]($lines[0] -match ',100000000,0,192,')) $lines[0]

Write-Host "=== 4. factor ordering is optional ==="
$unsorted = Join-Path $work 'factororder.txt'
$null = Run-Py @('--input', $dup, '--out-ecm', $unsorted)
$line = [string](@(Lines $unsorted | Where-Object { $_ -match ',991,' }) | Select-Object -First 1)
Check "default keeps first-appearance order from the input (103 is listed first there)" `
      ([bool]($line -match '"103,101"')) $line
$null = Run-Py @('--input', $dup, '--sort-factors', '--out-ecm', $unsorted)
$line = [string](@(Lines $unsorted | Where-Object { $_ -match ',991,' }) | Select-Object -First 1)
Check "--sort-factors normalizes numerically ascending" ([bool]($line -match '"101,103"')) $line

Write-Host "=== 5. --verify-factors is optional and catches a non-divisor ==="
$bad = Join-Path $work 'badfactor.csv'
'ECM2=1,2,991,-1,1e6,0,3,"101,999999999999999999999999"' | Set-Content -Path $bad -Encoding ASCII
$r = Run-Py @('--input', $bad, '--dry-run')
Check "off by default: a non-dividing factor is not checked" ($r.exit -eq 0) "exit=$($r.exit)"
$r = Run-Py @('--input', $bad, '--verify-factors')
Check "--verify-factors rejects it with a non-zero exit" `
      (($r.exit -ne 0) -and ($r.out -match 'does not divide')) "exit=$($r.exit) out=$($r.out)"
$good = Join-Path $work 'goodfactor.csv'
'ECM2=1,2,991,-1,1e6,0,3,"8218291649"' | Set-Content -Path $good -Encoding ASCII
$r = Run-Py @('--input', $good, '--verify-factors', '--dry-run')
Check "--verify-factors accepts a real factor of 2^991-1" ($r.exit -eq 0) "exit=$($r.exit)"

Write-Host "=== 6. the <...>_<B1>.save contract for ECMSTAGE2= / command lines ==="
$r = Run-Py @('--input', $dup, '--save-pattern', 'resume_p{n}_ECM_TE_B1_{b1}.p95',
              '--out-ecmstage2', (Join-Path $work 'never.csv'))
Check "a save name that breaks _<B1>.save fails with a non-zero exit" `
      (($r.exit -ne 0) -and ($r.out -match 'does not match')) "exit=$($r.exit) out=$($r.out)"
Check "the failure message explains why (driver reads B1 from the name)" `
      ([bool]($r.out -match 'B1 from the save name')) $r.out
$r = Run-Py @('--input', $dup, '--save-pattern', 'resume_p{n}_ECM_TE_B1_{b1}.p95',
              '--out-ecmstage2', (Join-Path $work 'never.csv'), '--allow-invalid-save-name', '--dry-run')
Check "--allow-invalid-save-name is the escape hatch" ($r.exit -eq 0) "exit=$($r.exit)"

Write-Host "=== 7. output shapes: three prefixes, quoting, worker section, --ecm-prefix ==="
$s2 = Join-Path $work 'shape_stage2.csv'
$ec = Join-Path $work 'shape_ecm.txt'
$null = Run-Py @('--input', $sortedCsv, '--out-ecmstage2', $s2, '--out-ecm', $ec)
$l2 = First $s2
Check "ECMSTAGE2= field order: [aid,]k,b,n,c,quoted-save,B2,skip,curves[,quoted-factors]" `
      ([bool]($l2 -match '^ECMSTAGE2=[0-9A-F]+,1,2,5153,-1,"m5153_[^"]+\.save",[0-9]+,0,192,"')) $l2
Check "save name is quoted" ([bool]($l2 -match ',"m5153_')) $l2
Check "default prefix is ECM2=" ((First $ec) -like 'ECM2=*') (First $ec)
$null = Run-Py @('--input', $sortedCsv, '--out-ecm', $ec, '--ecm-prefix', 'ECM')
Check "--ecm-prefix ECM switches the prefix" ((First $ec) -like 'ECM=*') (First $ec)
$w = Join-Path $work 'worker.txt'
$null = Run-Py @('--input', $sortedCsv, '--out-ecm', $w, '--worker', '3')
$wl = @(Lines $w)
Check "--worker N wraps the batch in one [Worker #N] section" ($wl[0] -eq '[Worker #3]') ($wl[0..1] -join ' | ')

Write-Host "=== 8. sort determinism and the AID-less sort trap of the old pipe ==="
$noaid = Join-Path $work 'noaid.csv'
@('ECM2=1,2,5153,-1,1e6,0,3', 'ECM2=AID,1,2,991,-1,1e6,0,3', 'ECM2=1,2,1200,-1,1e6,0,3') |
    Set-Content -Path $noaid -Encoding ASCII
$o1 = Join-Path $work 'sort1.txt'; $o2 = Join-Path $work 'sort2.txt'
$null = Run-Py @('--input', $noaid, '--out-ecm', $o1, '--sort-by', 'n')
$null = Run-Py @('--input', $noaid, '--out-ecm', $o2, '--sort-by', 'n')
Check "sorting works with mixed AID / no-AID rows (n ascending)" `
      ([bool](((@(Lines $o1)) -join '|') -match ',991,.+1200,.+5153,')) ((@(Lines $o1)) -join ' | ')
Check "two runs are byte-identical (deterministic)" ((FileText $o1) -eq (FileText $o2))
$null = Run-Py @('--input', $noaid, '--out-ecm', $o1, '--sort-by', 'n', '--desc')
Check "--desc reverses" ([bool](((@(Lines $o1)) -join '|') -match ',5153,.+1200,.+991,')) ((@(Lines $o1)) -join ' | ')

Write-Host "=== 9. pipeline order: filtering happens BEFORE dedup ==="
$order = Join-Path $work 'order.csv'
@('ECM2=1,2,991,-1,79e6,0,90', 'ECM2=1,2,991,-1,110e6,0,192') | Set-Content -Path $order -Encoding ASCII
$orderOut = Join-Path $work 'order_out.txt'
$null = Run-Py @('--input', $order, '--max-curves', '100', '--out-ecm', $orderOut)
$ol = @(Lines $orderOut)
Check "with --max-curves 100 the number survives via its low-curve row (filter first)" `
      (($ol.Count -eq 1) -and ([bool]($ol[0] -match ',79e6,'))) ($ol -join ' | ')

Write-Host "=== 10. encoding / newline defaults ==="
$crlf = Join-Path $work 'crlf.csv'; $lf = Join-Path $work 'lf.csv'
$null = Run-Py @('--input', $sortedCsv, '--out-ecmstage2', $crlf)
$null = Run-Py @('--input', $sortedCsv, '--out-ecmstage2', $lf, '--newline', 'lf')
$b1 = [System.IO.File]::ReadAllBytes($crlf); $b2 = [System.IO.File]::ReadAllBytes($lf)
Check "default output is CRLF" ([bool]([System.Text.Encoding]::ASCII.GetString($b1) -match "`r`n")) 'no CRLF found'
Check "--newline lf emits LF only" (-not ([bool]([System.Text.Encoding]::ASCII.GetString($b2) -match "`r`n")))
Check "output has no BOM" (-not ($b1[0] -eq 0xEF -and $b1[1] -eq 0xBB)) ("first bytes: " + ($b1[0..2] -join ','))

Write-Host "=== 11. legacy switch aliases behave like the new ones ==="
$new = Join-Path $work 'alias_new.csv'; $old = Join-Path $work 'alias_old.csv'
$null = Run-Py @('--input', $sortedCsv, '--out-ecmstage2', $new)
$null = Run-Py @('--input', $sortedCsv, '--out-windows', $old)
Check "--out-windows == --out-ecmstage2" ((FileText $new) -eq (FileText $old))
$shOld = Join-Path $work 'old.sh'; $shNew = Join-Path $work 'new.sh'
$null = Run-Py @('--input', $sortedCsv, '--out-linux', $shOld)
$null = Run-Py @('--input', $sortedCsv, '--emit-cli', $shNew, '--emit-cli-kind', 'sh')
Check "--out-linux == --emit-cli (kind sh)" ((FileText $shOld) -eq (FileText $shNew))
Check "sh emitter keeps the historical gmp-ecm line shape" `
      ([bool]((First $shOld) -match "^echo '\(1\*2\^\d+-1\)/?.*' \| \./ecm -v -savea .* -gpu -gpuckpt \d+ -gpucurves \d+ \S+ \S+$")) (First $shOld)

Write-Host "=== 12. command-line emitters for our driver ==="
$ps1 = Join-Path $work 'cli.ps1'; $bat = Join-Path $work 'cli.bat'
$null = Run-Py @('--input', $sortedCsv, '--emit-cli', $ps1, '--emit-cli-kind', 'ps1', '--device', '1')
$null = Run-Py @('--input', $sortedCsv, '--emit-cli', $bat, '--emit-cli-kind', 'bat', '--device', '1')
Check "ps1 emitter quotes the N expression with single quotes" `
      ([bool]((First $ps1) -match "^echo '\(1\*2\^5153-1\)/.*' \| ecm_cuda\.exe -v -gpu -d 1 ")) (First $ps1)
Check "bat emitter escapes ^ as ^^ (cmd would otherwise eat it)" `
      ([bool]((First $bat) -match '2\^\^5153')) (First $bat)
Check "emitters reference the same save name as the ECMSTAGE2= output" `
      ([bool]((First $ps1) -match 'm5153_[^ ]+\.save')) (First $ps1)

Write-Host "=== 13. cross-implementation acceptance: our C++ parser must accept the output ==="
$gmpInc = Join-Path $repo 'third_party\gmp-zen3\dist\include'
$gmpLib = Join-Path $repo 'third_party\gmp-zen3\dist\lib\gmp.lib'
$vsCandidates = @(
    'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
    'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
)
$vs = $vsCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$exe = Join-Path $work 'worktodo_parse_test.exe'
if ($vs -and (Test-Path $gmpInc) -and (Test-Path $gmpLib)) {
    $ecmLines = Join-Path $work 'cpp_in.txt'
    $s2Lines = Join-Path $work 'cpp_in2.txt'
    $null = Run-Py @('--input', $sortedCsv, '--out-ecm', $ecmLines)
    $null = Run-Py @('--input', $sortedCsv, '--out-ecmstage2', $s2Lines)
    $probe = Join-Path $work 'probe.cpp'
    @(
        '#include "ecm_worktodo.h"',
        '#include <cstdio>',
        '#include <string>',
        'int main(int argc, char **argv) {',
        '  for (int a = 1; a < argc; ++a) {',
        '    FILE *f = fopen(argv[a], "r"); if (!f) { printf("NOFILE %s\n", argv[a]); return 2; }',
        '    char buf[65536]; int n = 0, bad = 0;',
        '    while (fgets(buf, sizeof buf, f)) {',
        '      std::string line(buf);',
        '      while (!line.empty() && (line.back() == 10 || line.back() == 13)) line.pop_back();',
        '      if (line.empty() || line[0] == 35) continue;',
        '      std::string err; ++n;',
        '      if (line.compare(0, 10, "ECMSTAGE2=") == 0) {',
        '        EcmStage2Task t;',
        '        if (!ecm_parse_stage2_line(line, t, err)) { printf("BAD %s\n", err.c_str()); ++bad; }',
        '      } else {',
        '        Ecm2Task t;',
        '        if (!ecm_parse_ecm2_line(line, t, err)) { printf("BAD %s\n", err.c_str()); ++bad; }',
        '      }',
        '    }',
        '    fclose(f);',
        '    printf("%s: %d lines, %d rejected\n", argv[a], n, bad);',
        '    if (bad) return 1;',
        '  }',
        '  return 0;',
        '}'
    ) | Set-Content -Path $probe -Encoding ASCII
    # /utf-8 is required: ecm_worktodo.cpp carries UTF-8 Chinese comments and the default
    # GBK code page would swallow the newline after them (same trap as docs 6.14).
    $build = "cd /d `"$work`" && call `"$vs`" >nul 2>&1 && " +
             "cl /nologo /EHsc /utf-8 /I `"$gmpInc`" /I `"$repo\src\core`" `"$probe`" " +
             "`"$repo\src\core\ecm_worktodo.cpp`" /Fe:`"$exe`" `"$gmpLib`""
    $bo = cmd /c $build 2>&1 | Out-String
    if (Test-Path $exe) {
        # gmp-10.dll must be reachable at run time (the harness is not in the build dir).
        $gmpBin = Join-Path $repo 'third_party\gmp-zen3\dist\bin'
        $oldPath = $env:PATH
        $env:PATH = "$gmpBin;$env:PATH"
        $ro = cmd /c "`"$exe`" `"$ecmLines`" `"$s2Lines`"" 2>&1 | Out-String
        $rc = $LASTEXITCODE
        $env:PATH = $oldPath
        Check "our C++ parser accepts every emitted ECM=/ECM2= and ECMSTAGE2= line" `
              (($rc -eq 0) -and ($ro -match '0 rejected')) "exit=$rc out=$ro"
    } else {
        Skip "C++ cross-parser acceptance" "MSVC could not compile the harness"
        Write-Host ("         " + ((($bo -split "`r?`n") | Where-Object { $_ -match 'error|fatal|LNK' } |
            Select-Object -First 2) -join ' / '))
    }
} else {
    Skip "C++ cross-parser acceptance" "MSVC (vcvars64.bat) or third_party/gmp-zen3 not available"
}

Write-Host ""
if (-not $KeepWork) { Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
if ($fail -eq 0) { Write-Host "ALL OK ($pass checks$(if ($skip) { ", $skip skipped" }))" }
else { Write-Host "FAILED: $fail of $($pass + $fail)" }
exit $(if ($fail -eq 0) { 0 } else { 1 })
