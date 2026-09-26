# ---------------------------------------------------------------------------
# test_checkpoint.ps1 — 断言式验收: SIMD 批量的 checkpoint / resume 闭环.
#
#   阶段 1: 跑一次 (随机 sigma, 不指定), 落出 .ckpt + .tmp; 把 .tmp 存为参照
#   阶段 2: 只删 .tmp (保留 .ckpt), 同样的工作再跑一次
#   断言  : (a) 出现 "resume from .ckpt" ; (b) 续跑产出的 .tmp 与参照**逐字节相同**
#
# 为什么能断言逐字节相同: 续跑走的是同一条阶梯 (digit 索引两边都是 digits[total-1-i]),
# 只是从存档的 bitnum 继续; 域运算结果都是 N 的规范余数, 所以最终 Qx/Qz 必然一致.
#
# 用法: powershell -File tools/test/test_checkpoint.ps1 [-Exe <path>] [-B1 <n>]
# ---------------------------------------------------------------------------
param(
    [string]$Exe = "",
    [int]$B1 = 120000
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)     # tools/test -> repo
if (-not $Exe) { $Exe = Join-Path $repo "build_vs18\Release\ecm.exe" }
if (-not (Test-Path $Exe)) { Write-Host "FAIL: ecm.exe not found: $Exe"; exit 2 }

$run   = Join-Path $PSScriptRoot "_ckpt_run"
$saves = Join-Path $run "saves"
$ref   = Join-Path $run "ref"
Remove-Item $run -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $saves, $ref | Out-Null

$env:PATH = "$repo\third_party\gmp-zen3\dist\bin;$env:PATH"
$N = "(2^3001-1)"

function Write-Worktodo {
    # 队列管理器处理完会把任务从 worktodo 移走, 所以每次运行前都要重写.
    Set-Content (Join-Path $run "worktodo.txt") -Value "ECM2=1,2,3001,-1,$B1,0,8`r`n" -Encoding ascii
}

# 注意: 故意**不设 sigma** —— 这正是自动续跑曾经失效的场景 (每轮随机 sigma 配不上存档).
Set-Content (Join-Path $run "ecm.ini") -Value @"
edwards = 1
edwards_threads = 2
tmp_dir = $saves
edwards_backend = simd
edwards_naf_w = 12
ckpt_seconds = 1
worktodo = $($run)\worktodo.txt
finished = $($run)\finished.txt
log_file =
"@ -Encoding ascii

Write-Host "=== stage 1: full run (random sigma) ==="
Write-Worktodo
$o1 = & $Exe -ini (Join-Path $run "ecm.ini") *>&1 | Out-String
$ck1 = (Get-ChildItem $saves -Filter *.ckpt -ErrorAction SilentlyContinue | Measure-Object).Count
$tm1 = (Get-ChildItem $saves -Filter *.tmp  -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "  .ckpt=$ck1  .tmp=$tm1"
if ($ck1 -lt 1) { Write-Host "FAIL: no .ckpt written (checkpoint interval too small for this B1?)"; exit 1 }
if ($tm1 -lt 8) { Write-Host "FAIL: expected 8 MIDSTAGE results, got $tm1"; exit 1 }
Copy-Item "$saves\*.tmp" $ref -Force

Write-Host "=== stage 2: resume (keep .ckpt, drop .tmp) ==="
Get-ChildItem "$saves\*.tmp" | Remove-Item -Force
Write-Worktodo
$o2 = & $Exe -ini (Join-Path $run "ecm.ini") *>&1 | Out-String
$resume_lines = ($o2 -split "`n" | Select-String -Pattern "resume from \.ckpt").Count
($o2 -split "`n" | Select-String -Pattern "found \.ckpt|resume from \.ckpt" | Select-Object -First 2) |
    ForEach-Object { "  " + ($_.Line -replace "`e\[[0-9;]*[A-Za-z]","").Trim() }
if ($resume_lines -lt 1) { Write-Host "FAIL: no 'resume from .ckpt' line -> auto-resume did not trigger"; exit 1 }
Write-Host "  resume lines: $resume_lines"

Write-Host "=== assert: resumed results == full-run results (byte-exact) ==="
$same = 0; $diff = 0
Get-ChildItem $ref -Filter *.tmp | Sort-Object Name | ForEach-Object {
    $o = Join-Path $saves $_.Name
    if (-not (Test-Path $o)) { Write-Host "  MISSING after resume: $($_.Name)"; $diff++ }
    elseif ((Get-FileHash $_.FullName).Hash -eq (Get-FileHash $o).Hash) { $same++ }
    else { Write-Host "  DIFF: $($_.Name)"; $diff++ }
}
Write-Host "  identical=$same differing=$diff"
if ($diff -ne 0 -or $same -ne 8) { Write-Host "RESULT: FAIL"; exit 1 }
Write-Host "RESULT: PASS"
exit 0
