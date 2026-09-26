# ---------------------------------------------------------------------------
# make_old_ladder.ps1 -- regenerate .bench_tmp\ab_old\old_ladder.cpp
#
# tools/bench/mont_ckpt_ab.cpp times the pre-checkpoint ladder against today's one
# IN THE SAME PROCESS (interleaved), which is the only way to measure a ~1% effect
# without being fooled by CPU frequency drift between runs.  That needs the old
# ladder source compiled into the same binary, and the old ladder lives in git
# history (commit 10a9de5, "Optimize SIMD mont ..."), so this script pulls it out
# and renames its symbols so both versions can be linked together.
#
#   usage:  powershell -File tools\test\make_old_ladder.ps1 [<git-ref>]
#   then:   tools\build_tool.bat tools\bench\mont_ckpt_ab.cpp src\cpu\simd_mont_curve.cpp `
#               src\cpu\simd_mont_ifma.cpp src\cpu\ecm_mont_cpu.cpp .bench_tmp\ab_old\old_ladder.cpp
# ---------------------------------------------------------------------------
param([string]$Ref = '10a9de5')
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$out = Join-Path $root '.bench_tmp\ab_old'
New-Item -ItemType Directory -Force $out | Out-Null

Push-Location $root
try {
    $src = git show "${Ref}:src/cpu/simd_mont_curve.cpp"
    if ($LASTEXITCODE -ne 0) { throw "git show ${Ref}:src/cpu/simd_mont_curve.cpp failed" }
} finally {
    Pop-Location
}
$src = ($src -join "`n")

# Namespace the entry points and the context helpers so both ladders can coexist.
foreach ($p in @('mont_soa_stage1_bits', 'mont_soa_stage1', 'mont_soa_op_counts',
                 'mont_soa_init', 'mont_soa_clear')) {
    $src = $src -replace "\b$p\b", "old_$p"
}
# keep the public struct + the new declarations from the current header
$dst = Join-Path $out 'old_ladder.cpp'
[System.IO.File]::WriteAllText($dst, $src, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "wrote $dst ($((Get-Item $dst).Length) bytes) from $Ref"
