@echo off
@REM ============================================================================
@REM ecmcuda.bat -- 生成"给我们驱动跑 stage 1"的 worktodo（ECMSTAGE2= 行）
@REM
@REM 用本仓库自带的 Windows 原生工具（不需要 WSL / 外部 ECM.py）：
@REM   tools\ecm_worktodo\ecm.py
@REM 换机器时只改下面这一组变量。
@REM ============================================================================
setlocal

set "PYTHON=C:\anaconda3\envs\web\python.exe"
set "PIPE=%~dp0ecm.py"
set "SRC=%~dp0assignment.csv"
set "OUTPATH=D:\code\GIMPS\GIMPS_同步"

@REM 两份输出内容逐字节相同，只是投放位置不同：
@REM   OUT_QUEUE -> 我们驱动的队列目录（ecm_cuda.exe 读 ECMSTAGE2= 跑 stage 1）
@REM   OUT_SAVE  -> 另一路（例如 P95 侧接档）
set "OUT_QUEUE=%OUTPATH%\worktodo_add.csv"
set "OUT_SAVE=%OUTPATH%\worktodo_save.csv"

@REM stage-1 参数：B1 / 每批曲线数；B2=0 的语义是"让 Prime95 自动选 B2"，不是"不做 stage2"
set "B1=110e6"
set "CURVES=960"

if not exist "%PIPE%" (
  echo [ecmcuda] missing %PIPE%
  exit /b 2
)
if not exist "%SRC%" (
  echo [ecmcuda] missing %SRC%
  exit /b 2
)

@REM 队列文件：save 名必须满足 <...>_<B1>.save —— 驱动的 B1 是从存档名里抽出来的
"%PYTHON%" "%PIPE%" ^
  --input "%SRC%" ^
  --set-b1 %B1% --set-b2 0 --gpu-curves %CURVES% --skip-curves 0 ^
  --sort-factors ^
  --append-ecmstage2 ^
  --out-ecmstage2 "%OUT_QUEUE%"

"%PYTHON%" "%PIPE%" ^
  --input "%SRC%" ^
  --set-b1 %B1% --set-b2 0 --gpu-curves %CURVES% --skip-curves 0 ^
  --sort-factors ^
  --append-ecmstage2 ^
  --out-ecmstage2 "%OUT_SAVE%"

@REM 追加写入时先按 (k,b,n,c) 去重（冲突保留 B1 最大 -> curves 最大 -> 真 AID 优先 -> 先出现者），
@REM 所以重复运行这个脚本不会让同一个数在队列里出现两份工作。
@REM 需要逐任务命令行脚本时加：--emit-cli worktodo.ps1 --emit-cli-kind ps1 --device 1

endlocal
pause
