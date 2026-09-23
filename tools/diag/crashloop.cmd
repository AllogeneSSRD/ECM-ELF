# crashloop.cmd -- run the "forced mersenne on a non-Mersenne N" case repeatedly and
# report the exit code of every attempt. Plain cmd redirection (no PowerShell pipeline
# involvement) so a crash cannot be confused with a pipeline hang.
@echo off
setlocal enabledelayedexpansion
set EXE=D:\code\MPA-OpenCl\build_vs18\Release\ecm.exe
set LOG=D:\code\MPA-OpenCl\.bench_tmp\crashloop.log
set PATH=D:\code\MPA-OpenCl\third_party\gmp-zen3\dist\bin;%PATH%
echo === crashloop start %DATE% %TIME% === > "%LOG%"
for %%T in (1 8) do (
  for /L %%I in (1,1,6) do (
    set D=D:\code\MPA-OpenCl\.bench_tmp\cl_%%T_%%I
    if exist "!D!" rmdir /s /q "!D!"
    mkdir "!D!"
    echo 1048575 > "!D!\in.txt"
    "%EXE%" --edwards --edwards-backend simd --edwards-mersenne on --edwards-threads %%T -gpucurves 8 --tmp-dir "!D!" 256 0 < "!D!\in.txt" > "!D!\out.txt" 2> "!D!\err.txt"
    echo threads=%%T attempt=%%I exit=!ERRORLEVEL! >> "%LOG%"
  )
)
echo === crashloop done %DATE% %TIME% === >> "%LOG%"
type "%LOG%"
