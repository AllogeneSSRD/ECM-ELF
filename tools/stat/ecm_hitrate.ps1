# ---------------------------------------------------------------------------
# ecm_hitrate.ps1 -- stage-1 hit-rate statistics over small primes.
#
# A prime p is embedded in a large composite  N = p * (2^521-1).  That cofactor is
# REQUIRED: with N = p the only possible stage-1 hit is gcd == N, which the driver
# discards as trivial, so the measured rate would be 0 and prove nothing.  With the
# cofactor a hit is reported as the proper factor p.
#
# Engines (all of them run stage 1 only, B2 = 0):
#   edwards : CPU Edwards/Atkin-Morain, backends simd (auto/on/off fold) and gmp
#   mont    : CPU Suyama param0 (--method mont), backends simd and gmp
#   gpu     : ecm_cuda, param 0 (Suyama) or param 3 (gmp-ecm batch) -- the two GPU
#             parametrizations can be compared directly, and param 0 can be compared
#             against the CPU "mont" engine, which runs the SAME curves.
#
# 口径（重要）：`hits` 是**逐曲线**命中数，`pct = hits / (primes * curves)` 就是"一条随机
# 曲线的 stage-1 命中率"；另外打印 `primesHit/primePct` = 至少有一条曲线命中的素数比例
# （GPU 一次跑一批曲线时，只有前者能和曲线族/模型比较）。所有引擎都只跑 stage 1（B2=0）。
#
# 与独立的 Python 模型对拍（tools/ecm_prob，逐曲线独立判 Z == 0，bit20 / B1=256）：
#   param0 (Suyama Z/12)               30.472%
#   param3 (batch 32-bit, Z/4)         21.641%
#   Edwards Z/2xZ/8 (x48 torsion)      32.665%
# 本脚本实测（64 primes x 32 curves，bit20 / B1=256）：
#   CPU mont(simd) param0 = 625/2048 = 30.518%      GPU param0 = 625/2048 = 30.518%
#       -> CPU 与 GPU 逐曲线完全一致（同一 sigma 同一曲线族，命中数一模一样）
#   GPU param3            = 458/2048 = 22.363%
#   Edwards Z/2xZ/8       = 2611/8000 = 32.6375%（1000 primes x 8 curves；simd-mont 与 gmp
#       完全一致。注意本脚本的 "simd-auto" 是**脚本内**的写法，驱动器只认
#       auto|simd|gmp：以前原样传过去 ⇒ 驱动器每次报 "Invalid --backend"，1000 个素数
#       全是 0 命中却没有任何提示；现在翻译成 --backend simd --field auto，并且对
#       "有输出但没有结果行"的运行显式报警。）
#
# 历史陷阱（2026-09-24 修）：本脚本曾写 `@($out | Select-String -Pattern ...).Count`，而
# `$out` 是 `| Out-String` 得到的**单个多行字符串** —— Select-String 把整串当一个输入对象，
# 只返回 1 条匹配，于是"每个素数的多次命中"被压成 1，速率被低估到 6.25%（真值 30.47%）。
# 凡是要数多行输出，必须先 `-split "`r?`n"` 或改用 [regex]::Matches。
#
# usage (see -? / the parameter list for everything):
#   powershell -NoProfile -File tools\stat\ecm_hitrate.ps1                      # defaults
#   ... -Engine gpu -GpuParam 0 -BitsFrom 20 -BitsTo 24 -Count 200 -B1 1000
#   ... -Engine gpu -GpuParam 0 -All -BitsFrom 18 -BitsTo 18 -B1 1e4 -Curves 64
#   ... -Engine mont,gpu -GpuParam 0 -Count 0            # 0 = every prime in the file
#   ... -Engine edwards -Backend simd-auto,simd-mont,gmp
# ---------------------------------------------------------------------------
param(
    # Which stage-1 engine(s) to measure: edwards | mont | gpu (comma separated).
    [string]$Engine = "mont",
    # GPU parametrization, only used by -Engine gpu: 0 = Suyama param0, 3 = batch.
    [int]$GpuParam = 0,
    # CPU backend selection for the edwards/mont engines (comma separated), or "" for
    # each engine's default set.
    [string]$Backend = "",
    # Prime bit size: single value (-Bits) or a range.  BitsTo = 0 means "same as From".
    [int]$Bits = 0,
    [int]$BitsFrom = 20,
    [int]$BitsTo = 0,
    # How many primes to sample per bit size (evenly spread over the file).  0 or the
    # -All switch means: use EVERY prime in that bit size's file (can be slow!).
    [int]$Count = 1000,
    [switch]$All,
    [int]$B1 = 256,
    [int]$Curves = 8,
    # CPU threads for one curve; the GPU always uses one kernel per device.
    [int]$Threads = 1,
    [int]$Device = 0,
    [string]$CudaExe = "",
    [string]$EcmExe = "",
    [string]$Output = "",
    # Silence the per-prime lines (only the summaries are printed).
    [switch]$Quiet
)
$ErrorActionPreference = "Continue"   # native tools write progress to stderr
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$scratch = Join-Path $PSScriptRoot "_run"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null

function Resolve-Exe([string]$p, [string]$def) {
    if ([string]::IsNullOrEmpty($p)) { $p = $def }
    if (-not [System.IO.Path]::IsPathRooted($p)) { $p = Join-Path $repo $p }
    return $p
}
$ecmExe  = Resolve-Exe $EcmExe  "build_vs18\Release\ecm.exe"
$cudaExe = Resolve-Exe $CudaExe "build_cuda_cmake\ecm_cuda.exe"
$env:PATH = (Join-Path $repo "third_party\gmp-zen3\dist\bin") + ";" + $env:PATH

if ($Bits -gt 0) { $BitsFrom = $Bits }
if ($BitsTo -le 0) { $BitsTo = $BitsFrom }
if ($All) { $Count = 0 }

# ---- which engines -------------------------------------------------------------
$engines = @($Engine -split "," | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
foreach ($e in $engines) {
    if ($e -notin @("edwards", "mont", "gpu")) { Write-Host "FAIL: unknown -Engine '$e' (edwards|mont|gpu)"; exit 2 }
}
if ($engines -contains "gpu" -and -not (Test-Path $cudaExe)) {
    Write-Host "FAIL: ecm_cuda not found: $cudaExe (build it, see README 'CUDA backend')"; exit 2
}
if (($engines -contains "edwards" -or $engines -contains "mont") -and -not (Test-Path $ecmExe)) {
    Write-Host "FAIL: ecm.exe not found: $ecmExe"; exit 2
}
if ($GpuParam -notin @(0, 3)) { Write-Host "FAIL: -GpuParam must be 0 or 3"; exit 2 }

# ---- job list (engine x backend) ----------------------------------------------
$defaultBackends = @{
    "edwards" = @("simd-auto", "simd-mont", "gmp")
    "mont"    = @("simd", "gmp")
    "gpu"     = @("gpu")
}
$knownBackends = @("auto", "simd", "gmp", "simd-auto", "simd-mers", "simd-mont")
$jobs = @()
foreach ($e in $engines) {
    $bes = $defaultBackends[$e]
    if (-not [string]::IsNullOrEmpty($Backend)) { $bes = @($Backend -split "," | ForEach-Object { $_.Trim() }) }
    foreach ($b in $bes) {
        if ($e -ne "gpu" -and $b -notin $knownBackends) {
            Write-Host ("FAIL: unknown backend '{0}' for -Engine {1} (use {2})" -f $b, $e, ($knownBackends -join "|"))
            exit 2
        }
        $label = if ($e -eq "gpu") { "gpu-param$GpuParam" } else { "$e/$b" }
        $jobs += [pscustomobject]@{ engine = $e; backend = $b; label = $label }
    }
}

Write-Host ("engines: {0}" -f (($jobs | ForEach-Object { $_.label }) -join ", "))
Write-Host ("sizes  : bits {0}..{1}, {2} per size, B1={3}, curves/prime={4}, N = p * (2^521-1)" -f `
             $BitsFrom, $BitsTo, $(if ($Count -eq 0) { "ALL primes" } else { "$Count sampled" }), $B1, $Curves)
Write-Host ""

$q = [System.Numerics.BigInteger]::Pow(2, 521) - 1
$nin = Join-Path $scratch "nin.txt"
$summary = @()

foreach ($bits in $BitsFrom..$BitsTo) {
    $bin = Join-Path $repo ("tools\ecm_prob\data\primes\bits{0}.bin" -f $bits)
    if (-not (Test-Path $bin)) { Write-Host ("  (skip bits={0}: no {1})" -f $bits, $bin); continue }
    $bytes = [System.IO.File]::ReadAllBytes($bin)
    $total = [int]($bytes.Length / 8)

    # build the prime list: every prime, or an evenly spread sample
    $indexes = @()
    if ($Count -eq 0 -or $Count -ge $total) {
        $indexes = 0..($total - 1)
    } else {
        $stride = [math]::Max(1, [math]::Floor($total / $Count))
        for ($i = 0; $i -lt $Count; $i++) {
            $idx = $i * $stride
            if ($idx -ge $total) { break }
            $indexes += $idx
        }
    }
    $primes = @($indexes | ForEach-Object { [System.BitConverter]::ToUInt64($bytes, $_ * 8) })
    Write-Host ("=== bits={0}: {1} prime(s) of {2} in the file ===" -f $bits, $primes.Count, $total)

    foreach ($job in $jobs) {
        $tmp = Join-Path $scratch ("tmp_" + ($job.label -replace '[\\/:]', '_') + "_b$bits")
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force -Path $tmp | Out-Null

        $hits = 0; $nCurves = 0; $badFactor = 0; $noOutput = 0; $runsHit = 0
        $failed = 0; $failSample = @()
        $example = @()
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        for ($i = 0; $i -lt $primes.Count; $i++) {
            $p = $primes[$i]
            $N = ([System.Numerics.BigInteger]$p) * $q
            $sigma = 1000003 + 7919 * $i
            Get-ChildItem $tmp -File -ErrorAction SilentlyContinue | Remove-Item -Force

            $out = ""
            # N is ALWAYS fed through an ASCII file + cmd redirection: piping a string to
            # a native program from Windows PowerShell 5.1 prepends a UTF-8 BOM, and the
            # driver's expression parser rejects that (silently producing 0 hits).
            [System.IO.File]::WriteAllText($nin, "$N`n", ([System.Text.Encoding]::ASCII))
            switch ($job.engine) {
                "gpu" {
                    $argv = "-gpu -d $Device --gpu-param $GpuParam -gpucurves $Curves " +
                            "-sigma $sigma --ckpt 0 $B1 0"
                    $out = cmd /c "cd /d `"$tmp`" && `"$cudaExe`" $argv < `"$nin`"" 2>&1 | Out-String
                }
                "mont" {
                    $argv = "--method mont --backend $($job.backend) --stage1-threads $Threads " +
                            "-gpucurves $Curves -sigma $sigma --tmp-dir `"$tmp`" --ckpt 0 $B1 0"
                    $out = cmd /c "cd /d `"$tmp`" && `"$ecmExe`" $argv < `"$nin`"" 2>&1 | Out-String
                }
                "edwards" {
                    # 后端名的翻译：simd-auto / simd-mers / simd-mont 是本脚本的写法，
                    # 驱动器只认 auto|simd|gmp + --field auto|mersenne|montgomery。
                    # （曾经直接把 "simd-auto" 当 --backend 传过去 ⇒ 驱动器每次都报
                    #  "Invalid --backend"，1000 个素数全部 0 命中却毫无提示。）
                    $bk = $job.backend; $field = "auto"
                    if ($bk -eq "simd-auto") { $bk = "simd"; $field = "auto" }
                    elseif ($bk -eq "simd-mers") { $bk = "simd"; $field = "mersenne" }
                    elseif ($bk -eq "simd-mont") { $bk = "simd"; $field = "montgomery" }
                    $argv = "--method edwards --backend $bk --stage1-threads $Threads --naf-w 12 " +
                            "--field $field -gpucurves $Curves -sigma $sigma --tmp-dir `"$tmp`" " +
                            "--ckpt 0 $B1 0"
                    $out = cmd /c "cd /d `"$tmp`" && `"$ecmExe`" $argv < `"$nin`"" 2>&1 | Out-String
                }
            }
            if ([string]::IsNullOrWhiteSpace($out)) {
                $noOutput++
            } elseif (($out -notmatch 'curves=\d+') -and ($out -notmatch 'factor\[')) {
                # 有输出但没有结果行 ⇒ 驱动器根本没跑（典型：backend/参数名非法被拒），
                # 这种情况若不显式报出来，就会被当成"命中率 0%"。
                $failed++
                if ($failSample.Count -lt 2) {
                    $failSample += (($out -split "`r?`n" | Where-Object { $_ -match '\S' } |
                                     Select-Object -First 1) -replace '^\s+', '').Trim()
                }
            }

            $nCurves += $Curves
            # Count hits from the driver's own "factor[<i>]=<value>" lines: they are
            # printed identically by every engine and carry the value we verify below.
            # (Do NOT count "factor found": the CUDA kernel's own line is gated behind
            # -v and, until 2026-09-24, rendered its factor through gmp-ecm's %Zd which
            # the project logger does not support.)
            #
            # NOTE ($out is ONE string because of `| Out-String`): piping a multi-line
            # string into Select-String matches it as a single line and returns exactly
            # ONE MatchInfo, so `@($out | Select-String ...).Count` silently counts
            # "runs with >=1 hit" instead of hits.  That undercounted every rate by the
            # number of hits per run (6.25% where the truth was 30.47%, 2026-09-24).
            # Always split into lines first, or use [regex]::Matches.
            $hitLines = @($out -split "`r?`n" | Where-Object { $_ -match 'factor\[\d+\]=' })
            $hits += $hitLines.Count
            if ($hitLines.Count -gt 0) { $runsHit++ }
            foreach ($line in $hitLines) {
                $v = [System.Numerics.BigInteger]::Parse(($line -split "=")[-1].Trim())
                $seen = @($example | Where-Object { $_ -like "p=$p *" }).Count -gt 0
                if ($v -ne [System.Numerics.BigInteger]$p) {
                    $badFactor++
                    if (-not $seen -and $example.Count -lt 3) { $example += ("p=$p got=$v") }
                } elseif (-not $seen -and $example.Count -lt 3) { $example += ("p=$p -> $v") }
            }
        }
        $sw.Stop()
        $rate = 100.0 * $hits / [math]::Max(1, $nCurves)
        $runRate = 100.0 * $runsHit / [math]::Max(1, $primes.Count)
        $secs = $sw.Elapsed.TotalSeconds
        if (-not $Quiet) {
            Write-Host ("  [{0,-14}] hits={1,6} / {2,6} curves = {3,6:N3}%   primes hit {4}/{5} = {6,6:N3}%   ({7:N1}s, {8:N4} s/curve)" -f `
                        $job.label, $hits, $nCurves, $rate, $runsHit, $primes.Count, $runRate,
                        $secs, ($secs / [math]::Max(1, $nCurves)))
            if ($badFactor -gt 0) {
                Write-Host ("                 !! {0} hit(s) reported a factor != p : {1}" -f $badFactor, ($example -join "; "))
            } elseif ($example.Count -gt 0) {
                Write-Host ("                 e.g. {0}" -f ($example -join "; "))
            }
            if ($noOutput -gt 0) { Write-Host ("                 !! {0} run(s) produced no output" -f $noOutput) }
            if ($failed -gt 0) {
                Write-Host ("                 !! {0}/{1} run(s) did not produce a result line " +
                            "(the rate below is meaningless): {2}" -f `
                            $failed, $primes.Count, ($failSample -join " | "))
            }
        }
        $summary += [pscustomobject]@{
            bits = $bits; engine = $job.engine; backend = $job.backend; param = $(if ($job.engine -eq "gpu") { $GpuParam } else { "" })
            primes = $primes.Count; curves = $nCurves; hits = $hits; pct = [math]::Round($rate, 4)
            primesHit = $runsHit; primePct = [math]::Round($runRate, 4)
            # failedRuns/badFactors 放在 seconds 之前：Format-Table -AutoSize 在控制台宽度不够时
            # 会**从右边丢列**，把这两个诊断列丢在最右边等于看不见。
            failedRuns = $failed; badFactors = $badFactor; secs = [math]::Round($secs, 2)
        }
    }
    Write-Host ""
}

Write-Host "=== summary ==="
$summary | Format-Table -AutoSize
# Format-Table -AutoSize 在控制台放不下时会静默丢右边的列，所以诊断信息再单独吼一遍
$suspect = @($summary | Where-Object { $_.failedRuns -gt 0 -or $_.badFactors -gt 0 })
if ($suspect.Count -gt 0) {
    Write-Host ""
    Write-Host "!! 有可疑结果（速率不可信）："
    foreach ($s in $suspect) {
        Write-Host ("   bits={0} engine={1} backend={2} failedRuns={3} badFactors={4}" -f `
                    $s.bits, $s.engine, $s.backend, $s.failedRuns, $s.badFactors)
    }
}
if (-not [string]::IsNullOrEmpty($Output)) {
    $summary | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $Output
    Write-Host ("CSV written: {0}" -f $Output)
}
Write-Host "reference: Edwards Z/2xZ/8, B1=256, all 38635 20-bit primes -> 32.66%"
Write-Host "note: param0 (CPU mont and GPU param0) runs the SAME curves for the same sigma,"
Write-Host "      so their hit sets must be identical; param3 is a different curve family."
Write-Host "note: the edwards engine multiplies its exponent by 48 (torsion), while mont/gpu"
Write-Host "      use lcm(1..B1) exactly, so edwards vs mont rates are not comparable at equal B1;"
Write-Host "      param0 vs param3 (both lcm(1..B1)) IS an apples-to-apples family comparison."
