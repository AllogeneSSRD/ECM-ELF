# Build + run the focused Edwards speed benchmark with a chosen MSVC flag set.
# Usage: powershell -File tools/bench/build_speed.ps1 -Flags "/O2" -Tag "O2" -Bits 991 -B1 1000000 -W 8 -Reps 3
param(
    [string]$Flags = "/O2",
    [string]$Tag = "O2",
    [int]$Bits = 991,
    [long]$B1 = 1000000,
    [int]$W = 8,
    [int]$Reps = 3
)

$ErrorActionPreference = "Stop"
$root = "D:\code\MPA-OpenCl"
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$gmp = "D:\code\vcpkg\installed\x64-windows"
$exe = Join-Path $root "tools\bench\_speed_$Tag.exe"

$cline = "cl /nologo $Flags /EHsc /utf-8 /I `"$gmp\include`" /I `"$root\src\cpu`" `"$root\tools\bench\ecm_edwards_speed.cpp`" `"$root\src\cpu\ecm_edwards_cpu.cpp`" /Fe:`"$exe`" /link `"$gmp\lib\gmp.lib`""

$build = @"
@echo off
call "$vcvars" >nul
$cline
"@
$tmp = Join-Path $env:TEMP "build_speed_$Tag.bat"
Set-Content -Path $tmp -Value $build -Encoding ASCII
cmd /c $tmp
if ($LASTEXITCODE -ne 0) { Write-Host "COMPILE FAILED ($Tag)"; exit 1 }

$env:PATH = "$gmp\bin;" + $env:PATH
& $exe $Bits $B1 $W $Reps
Write-Host "exit=$LASTEXITCODE tag=$Tag"
