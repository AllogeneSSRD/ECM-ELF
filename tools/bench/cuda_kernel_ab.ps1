# cuda_kernel_ab.ps1 -- 给 CUDA stage-1 kernel 做 A/B 计时：固定 N/B1/曲线数，重复若干次取中位数，
# 输出 curve-bits/s（= curves * s_bits / gputime）。用于 kernel 改动的前后对比与探针实验。
#
# 用法：
#   powershell -NoProfile -File tools\bench\cuda_kernel_ab.ps1 -Label "baseline" -Bits 1021
#   ... -GpuParam 3 -Curves 4096 -B1 1e5 -Device 1 -Repeats 3 -Exe build_cuda_dev\ecm_cuda.exe
#
# 注意：
#   * N 固定用 N = 2^Bits - 1（Bits 取素数指数时为素数，绝不会命中因子 ⇒ 每次都跑满全部 bit，
#     计时不受"提前找到因子"影响）
#   * -Device 默认 1：本机 GPU 0 常被其它任务占用，计时必须在空闲卡上做
#   * 每次运行都新建临时目录并清空，避免 ckpt/save 残留影响
param(
    [int]$Bits = 1021,
    [string]$B1 = "1e5",
    [int]$Curves = 4096,
    [int]$Device = 1,
    [int]$Repeats = 3,
    [int]$GpuParam = 3,
    [string]$Exe = "build_cuda_dev\ecm_cuda.exe",
    [string]$Label = "",
    # N 表达式；默认 N = 2^Bits-1。**探针实验必须传素数**（例如 -NExpr "(2^521-1)"）：
    # 用合数时，-DECM_PROBE_ADD_DENSITY=k>1 的垃圾状态可能恰好命中 N 的小因子，驱动会因
    # "退化曲线"检查提前结束整个批次，于是拿不到 gputime。
    [string]$NExpr = "",
    [switch]$Quiet
)

$ErrorActionPreference = "Continue"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not [System.IO.Path]::IsPathRooted($Exe)) { $Exe = Join-Path $repo $Exe }
$tmp = Join-Path $repo ".bench_tmp\kernel_ab"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
if ([string]::IsNullOrEmpty($NExpr)) { $NExpr = "(2^$Bits-1)" }
$nfile = Join-Path $tmp "n_$Bits.txt"
[System.IO.File]::WriteAllText($nfile, "$NExpr`n", ([System.Text.Encoding]::ASCII))

$sBits = 0
$times = @()
for ($r = 1; $r -le $Repeats; $r++) {
    Get-ChildItem $tmp -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "n_$Bits.txt" } | Remove-Item -Force -ErrorAction SilentlyContinue
    $argv = "-gpu -d $Device --gpu-param $GpuParam -sigma 3:12345678 -gpucurves $Curves --ckpt 0 $B1 0"
    $out = cmd /c "cd /d `"$tmp`" && `"$Exe`" $argv < `"$nfile`"" 2>&1 | Out-String
    $m = $out -split "`r?`n" | Select-String -Pattern 'gputime=([0-9.]+)'
    if ($m.Count -eq 0) {
        if (-not $Quiet) { Write-Host "  运行失败（没有 gputime）：" ; ($out -split "`r?`n" | Select-Object -Last 4) | ForEach-Object { Write-Host "    $_" } }
        continue
    }
    $times += [double]($m[0].Matches[0].Groups[1].Value)
    if ($sBits -eq 0) {
        # 进度条最后一行会打印 s 的总位数： "... 100.0%  144344, +2138 bits ..."
        $sb = $out -split "`r?`n" | Select-String -Pattern '100\.0%\s+(\d+),'
        if ($sb.Count -gt 0) { $sBits = [int]$sb[0].Matches[0].Groups[1].Value }
    }
}
if ($times.Count -eq 0) { Write-Host "FAILED $Label"; exit 1 }

$sorted = $times | Sort-Object
$median = if ($sorted.Count % 2 -eq 1) { $sorted[[int]($sorted.Count / 2)] } else { ($sorted[$sorted.Count / 2 - 1] + $sorted[$sorted.Count / 2]) / 2 }
$cbps = if ($sBits -gt 0) { [double]$Curves * $sBits / ($median / 1000.0) / 1e6 } else { 0 }

if (-not $Quiet) {
    Write-Host ("  {0,-22} N=M{1} B1={2} curves={3} dev={4} param={5}" -f $Label, $Bits, $B1, $Curves, $Device, $GpuParam)
    Write-Host ("    gputime ms: {0}" -f (($times | ForEach-Object { [math]::Round($_, 2) }) -join ", "))
    Write-Host ("    median {0:N2} ms  ->  {1:N2} M curve-bits/s   (s_bits={2})" -f $median, $cbps, $sBits)
}
# 便于脚本化收集：最后一行固定格式
Write-Output ("RESULT|{0}|{1:N2}|{2:N2}|{3}" -f $Label, $median, $cbps, $sBits)
