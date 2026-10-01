# Windows 崩溃弹窗把测试/子代理命令挂住：成因与三层防御

> 现象：探针因访问违例崩溃时，Windows 弹出"**该内存不能为 read**"对话框。对话框把崩溃的进程**挂在那里**，
> 父 `cmd.exe` 因此永不返回，于是**启动它的那一方**（测试脚本、套件项、或子代理的命令行）一直等，
> 直到自己那个长得多的超时（几十分钟）才结束。手动关掉弹窗才能继续。
>
> 本文记录**三层防御**，每层都有实测证据。三层是"每机 / 每进程 / 每次运行"，任一层单独都有效，叠加最好。

## 第一层（每机，已在本机启用并验证）：关掉 Windows Error Reporting 的界面

```
HKCU\Software\Microsoft\Windows\Windows Error Reporting
    DontShowUI             = 1     (DWORD)
    DontSendAdditionalData = 1     (DWORD)
```

* `HKCU` 即可，**不需要管理员**；只影响当前用户，而子代理与测试都以该用户运行。（`HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\DontShowUI` 需要管理员，本机未设置。）
* **实测证据**：`tools/diag/crash_test.c`（故意对空指针写入）编译后运行 —— 修改前会挂住，修改后
  `exit=1`、**0.3 秒返回**、无弹窗。用一个**必然崩溃**的小程序来验证这一层是重点：不要靠"看起来没弹窗"来判断。
* 这条是**唯一一条**对**别人的程序**（例如驱动、旧工具）也生效的防御。

```powershell
# 复现/重设（注意：某些受限沙箱会拒绝写注册表，报 "Requested registry access is not allowed"）
$k='HKCU:\Software\Microsoft\Windows\Windows Error Reporting'
New-ItemProperty -Path $k -Name 'DontShowUI' -Value 1 -PropertyType DWord -Force
```

## 第二层（每进程，写进我们自己的工具）：`SetErrorMode`

我们的探针/基准工具都是自己的源码，所以在 `main` 开头加三行，**从根上不产生这个对话框**（与注册表无关，换机器也有效）：

```c
#ifdef _WIN32
#include <windows.h>
static void no_crash_dialog(void) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
}
#else
static void no_crash_dialog(void) {}
#endif

int main(int argc, char **argv) {
    no_crash_dialog();     /* FIRST line of main, before any allocation or CUDA init */
    ...
```

* 只需这样一行；没有返回值要检查，也不会改变任何计算行为。
* 建议做法：在 `tools/bench/` 下放一个 `no_crash_dialog.h`，每个探针 `#include` 并调用 —— **但要在你没有别人正在编辑那个文件时改**（本仓库有并发子代理编辑的教训：文档 §14.8 的"镜像与实现各自漂移"）。
* **未在本机逐文件落地**（写作时有两个文件正被子代理编辑）：`stage2_tree_gpu.cu`、`ntt_poly_probe.cu`、`cufft_kron_probe.cu`、`stage2_ref.cpp`、`stage2_tree_ref.cpp` 都还没有这一行；第一层已经在保护它们。

## 第三层（每次运行，harness 侧）：硬超时 + 杀进程树

即使前两层都失效（例如别人写的程序弹窗），**运行方必须能自己脱身**。`tools/test/run_with_timeout.ps1` 就是干这个的：

```powershell
# 建议这样调用（见下方"引号陷阱"）：
& tools\test\run_with_timeout.ps1 -Exe build_cuda_cmake\stage2_tree_gpu.exe `
      -Arguments '--evaluate-batched --n 12345' -TimeoutSec 300 -Log build_cuda_cmake\_run.log
```

它做的事：把子进程的 stdout+stderr **重定向到文件**（不走管道，管道写满会阻塞）、给一个**墙钟预算**、
超时后**杀掉整棵进程树**并返回 **124**（与 coreutils `timeout` 同约定），因此"崩溃+弹窗"会表现为 `timeout` 而不是无限等待。

**实测证据**（用 `tools/diag/sleep_test.c`，一个可指定睡眠秒数的程序）：

| 场景 | 结果 |
|---|---|
| 5 s 预算跑 30 s 睡眠 | `exit=timeout`、6.3 s 返回、**残留进程 0** |
| 30 s 预算跑 2 s 睡眠 | `exit=0`、2.1 s 返回、日志含 `sleeping 2 s` / `done` |

### 三层都踩过的坑（写在这里省下一次踩）

1. **`Start-Process -ArgumentList` 会把命令行里的引号重新拼坏** ⇒ 子进程根本没启动、日志为空、立刻返回 1。改用
   `[System.Diagnostics.ProcessStartInfo]` + `UseShellExecute=$false`（`Arguments` 原样传给 `CreateProcess`）。
2. **`cmd /c "<exe>" args` 以引号开头时 `cmd` 会剥掉外层引号** ⇒ 同样立刻返回 1。**只在参数确实含空格时才加引号**，并用 cmd 的双引号形式 `""path""`。
3. **`taskkill` 在受限沙箱里被拒**（`ERROR: Access denied`）⇒ 主路径用 `taskkill /T /F`，**兜底**用 CIM 递归（`Get-CimInstance Win32_Process -Filter "ParentProcessId=$id"` 递归后 `Stop-Process -Force`）。杀不干净时**残留子进程会占着日志文件**，让下一次运行瞬间失败并显示**上一次的日志**（这个"假日志"很容易误导）。
4. **`powershell -File script.ps1 -Arr @('a','b')` 会把数组拼成一个字符串** ⇒ 参数一律用**单个字符串**传并在脚本内自己处理引号（本仓库 README 早有一条同类记录）。
5. **后台 job 里 `Start-Process -NoNewWindow` 被拒**（本仓库既有记录）⇒ 用 `-WindowStyle Hidden` 或 .NET `CreateNoWindow`。

## 给测试/子代理的推荐做法（可直接抄）

* 任何**可能崩溃**或**可能挂住**的命令，都通过 `run_with_timeout.ps1` 跑，预算按形状给（小形状 60 s、真实形状 900 s 之类），并**永远在崩溃/超时后确认没有残留进程**（残留会污染下一次运行）。
* 崩了先看**退出码**：`0xC0000005`（访问违例）/`0xC0000374`（堆损坏）在 C/C++ 工具里通常意味着**主机端越界或 use-after-free**——本仓库真实踩过两次：GMP 的 `mpz_clear` 之后复用同一个 `mpz_t`（约 5 % 的运行在首行输出前堆损坏猝死），以及数组下标回绕。
* **`ok=0` / 崩溃的运行不许报时间**（这条纪律已经在 §17.4 与各测试里执行）。
