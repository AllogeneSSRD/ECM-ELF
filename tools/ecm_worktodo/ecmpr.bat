@echo off
@REM ============================================================================
@REM ecmpr.bat -- 生成给 Prime95 的 worktodo（ECM=/ECM2= 原生行）
@REM
@REM 用本仓库自带的 Windows 原生工具：tools\ecm_worktodo\ecm.py
@REM 换机器时只改下面这一组变量。
@REM ============================================================================
setlocal

set "PYTHON=C:\anaconda3\envs\web\python.exe"
set "PIPE=%~dp0ecm.py"
set "SRC=%~dp0sorted.csv"

set "OUT_PRmers=D:\code\GIMPS\prmers\prmers-windows_v4.18.2\worktodo.txt"
set "OUT_P95=%~dp0worktodo_add.csv"

@REM stage-1/2 参数：B1、每批曲线数；B2=0 = 让 Prime95 自动选 B2
set "B1=100000"
set "CURVES=50"

@REM 这一路用 .p95 存档名（P95 侧自己从存档里取 B1），所以显式关掉我们驱动的
@REM "<...>_<B1>.save" 校验：--allow-invalid-save-name。若把这个文件喂给 ecm_cuda.exe
@REM 的队列，驱动会判 save_name does not end with .save —— 那是给另一个消费者的文件。
set "SAVE_PATTERN=resume_p{n}_ECM_TE_B1_{b1}.p95"

if not exist "%PIPE%" (
  echo [ecmpr] missing %PIPE%
  exit /b 2
)
if not exist "%SRC%" (
  echo [ecmpr] missing %SRC%
  exit /b 2
)

@REM 原生 Prime95 行（默认前缀 ECM2=，可用 --ecm-prefix ECM 改成旧写法）
"%PYTHON%" "%PIPE%" ^
  --input "%SRC%" ^
  --set-b1 %B1% --set-b2 0 --gpu-curves %CURVES% --skip-curves 0 --sort-by n ^
  --save-pattern "%SAVE_PATTERN%" --allow-invalid-save-name ^
  --append-ecm ^
  --out-ecm "%OUT_PRmers%"

@REM 同一批工作的 ECMSTAGE2= 版本（存档 + B2 + skip + curves）
"%PYTHON%" "%PIPE%" ^
  --input "%SRC%" ^
  --set-b1 %B1% --set-b2 0 --gpu-curves %CURVES% --skip-curves 0 --sort-by n ^
  --save-pattern "%SAVE_PATTERN%" --allow-invalid-save-name ^
  --append-ecmstage2 ^
  --out-ecmstage2 "%OUT_P95%"

endlocal
