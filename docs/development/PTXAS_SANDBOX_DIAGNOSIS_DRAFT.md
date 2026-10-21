# Stage2 local：ptxas 严格句柄检查诊断草稿

日期：2026-10-21。范围：Windows 上 CUDA 13.3.73 的 Stage2 local 单文件编译失败。本草稿保留诊断过程与证据，完成后由用户决定如何处理，不自动删除。

## 已确认结论

直接触发条件是 **ptxas 的单文件并行优化路径与 Windows 严格句柄检查同时启用**。本次沙箱中的进程启用了 `ProcessStrictHandleCheckPolicy`，Flags 为 `0x3`；沙箱外默认 Flags 为 `0x0`。

`ptxas --split-compile=8` 在沙箱内失败、沙箱外默认成功。在沙箱外仅为同一次测试启动的 ptxas 子进程启用严格句柄检查，仍能复现 `0xC0000008`；保持严格检查并将 split compile 设为 1 则成功。因此，严格检查是可独立复现的因果变量，不需要项目源码、MSVC、多个 CU 同时编译、项目输出目录或整个构建脚本才能触发。

二进制调试进一步定位到 ptxas 内部并行同步路径：向 `WaitForSingleObject` 传入对象内部的内存地址。在沙箱外默认策略下，观察到该调用返回 `0xFFFFFFFF`（`WAIT_FAILED`），ptxas 仍继续并成功退出；严格检查下同一路径引发致命的 `STATUS_INVALID_HANDLE`。这是本机该编译器二进制的观测，不代表 NVIDIA 已确认缺陷，也不外推至其他 CUDA 或 Windows 版本。

Windows 官方说明：严格句柄检查会使无效句柄操作成为致命错误；通常一旦启用便无法关闭。见 [SetProcessMitigationPolicy](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessmitigationpolicy)、[严格句柄检查标志](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-process_mitigation_strict_handle_check_policy)。`WAIT_FAILED` 的返回语义见 [WaitForSingleObject](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject)。

## 两层并行的区别

- **编译单元之间**：同时启动多个 nvcc 进程，各自编译一个 `.cu`。Stage1 的 `Jobs` 属于这一层。
- **单个编译单元内部**：nvcc/ptxas 的 `--split-compile=N` 允许编译优化使用多个线程。Stage2 的 `SplitCompile` 属于这一层，即使只编译一个 `.cu` 也会进入。

本机 ptxas 的 `--help` 说明该参数限制编译优化的并发线程数；1 表示不启用该选项，0 表示按 CPU 数量选择线程数。因此 **0 不表示关闭并行**。本次 nvcc 的详细日志显示 `--split-compile=8` 被转交给 ptxas。相关语义见 [NVIDIA NVCC 文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#split-compile)；链接目前展示的在线版本可能高于本机版本，本次同时核对了本机 13.3.73 帮助和实际调用。

## 复现与对照

权威汇总：[summary.json](../../data/experiments/ptxas_invalid_handle_20261021/summary.json)。原始命令、最小输入、对象、日志及临时调试脚本均位于 `data/experiments/ptxas_invalid_handle_20261021/`，由 Git 忽略；文档随仓库保留，原始证据只保留在本机。

### 原始编译单元

从 P0 失败现场复制 `arithmetic_kernels.compile.cmd`，只把输出路径改到独立实验目录。工具链仍为 CUDA 13.3.73、MSVC 14.51.36231、`sm_89`；保持 `-std=c++17 -O3`、fixed mode 3、outer unroll 0、add/sub mask 1、GMP/include 路径和依赖文件生成参数。

- 沙箱内，SplitCompile=8：重现 `Internal error` 和 `ptxas ... 0xC0000008`；nvcc 退出码 8。见 [日志](../../data/experiments/ptxas_invalid_handle_20261021/sandbox_original.log)。
- 沙箱内，SplitCompile=1：编译成功，nvcc 退出码 0，生成独立对象。见 [命令](../../data/experiments/ptxas_invalid_handle_20261021/arithmetic_serial.compile.cmd)、[日志](../../data/experiments/ptxas_invalid_handle_20261021/sandbox_arithmetic_serial.log)。
- 沙箱外，SplitCompile=8：同一命令成功，nvcc 退出码 0。见 [命令](../../data/experiments/ptxas_invalid_handle_20261021/arithmetic.compile.cmd)、[日志](../../data/experiments/ptxas_invalid_handle_20261021/outside_arithmetic_parallel.log)。

原始 `kernels/cuda/stage2/arithmetic_kernels.cu` 的 SHA256 为 `A6E7C2783CCC51288B8409C062B1B7849BD9BD0E9389EEA4D24ECEBC727E2931`，与 P0 冻结提交 `57351082d0d909985f0c2c47580de94359d17e0b` 的摘要一致。诊断期间关联聊天继续修改构建脚本；以上对照均重放冻结命令，没有把那些改动混入变量。

### 最小输入与重复验证

最小 CUDA 内核只执行一次整数加法，也能失败。使用 nvcc 的 keep 模式保留 PTX 后，直接运行 ptxas 即可复现，排除了 CUDA 前端与 MSVC 为必要条件。输入见 [minimal.cu](../../data/experiments/ptxas_invalid_handle_20261021/minimal.cu)、[minimal.ptx](../../data/experiments/ptxas_invalid_handle_20261021/minimal.ptx)。

以下五种组合各独立启动子进程重复 3 次，共 15 次，全部符合预期：

- 沙箱内默认策略，子进程严格检查 Flags=3，split=8：3/3 返回 `0xC0000008`。
- 沙箱内默认策略，子进程严格检查 Flags=3，split=1：3/3 成功。
- 沙箱外默认策略，子进程严格检查 Flags=0，split=8：3/3 成功。
- 沙箱外只为测试子进程启用严格检查，Flags=3，split=8：3/3 返回 `0xC0000008`。
- 沙箱外只为测试子进程启用严格检查，Flags=3，split=1：3/3 成功。

见 [沙箱内重复记录](../../data/experiments/ptxas_invalid_handle_20261021/repeat_sandbox.json)、[沙箱外重复记录](../../data/experiments/ptxas_invalid_handle_20261021/repeat_outside.json)。测试启动器在子进程挂起时查询实际策略，随后恢复并等待退出；不依赖调试器制造异常，不修改系统或用户配置。最终重复矩阵的启动器只保留默认策略或显式启用更严格检查，不关闭继承的检查。

补充单次探针中，沙箱内 split=0/2/4 同样失败；直接继承输出、捕获输出、cmd 文件重定向、标准输入接 nul 均未消除 split=8 的错误。由此不采用“修复标准输出句柄”作为此问题的处理方式。

### 二进制调用定位

沙箱内外 `ptxas.exe` 文件相同，SHA256 均为 `86B2FF48655C55F1547EDC59B503D39A62AA5C71745B7ECF6BE526C29AF0B693`。以下为相对本次二进制映像基址的偏移，不能用于其他编译器版本：

- `+0x1457D98`：主线程同步路径调用等待函数；参数为 `rbx+0xA8`。
- `+0x1457E29`：工作线程同步路径调用等待函数；参数为 `rbx+0x70`。
- 两处导入槽 `+0x1481138` 均解析为 `WaitForSingleObject`。沙箱内外解析结果一致。
- 沙箱外在两个调用后的 `+0x1457D9E` 和 `+0x1457E2F` 均读到返回值 `0xFFFFFFFF`；进程最终退出 0。
- 沙箱内捕获 `KiRaiseUserExceptionDispatcher` 的 `0xC0000008`，栈中包含 `WaitForSingleObjectEx` 和上述 ptxas 同步路径。

证据：[沙箱内调用参数与异常](../../data/experiments/ptxas_invalid_handle_20261021/debug_sandbox8_probes_fixed.jsonl)、[沙箱外调用及返回值](../../data/experiments/ptxas_invalid_handle_20261021/debug_outside8_wait_returns.jsonl)、[导入表](../../data/experiments/ptxas_invalid_handle_20261021/ptxas_imports.txt)。软件断点仅用于本次诊断启动的子进程，未修改磁盘上的编译器。该层探针辅助解释机制，因果判断以上面的无调试器重复矩阵为准。

探索日志中保留了命令换行/转义错误、调试器句柄权限不足以及早期显式子进程策略关闭探针；它们不属于最终重复矩阵，也不作为构建建议。原始证据不删除或改写。

## 构建处理建议与验证边界

需要保持当前默认编译配置（SplitCompile=8）时，沿用已成功的沙箱外正式 local 构建。关联重构草稿中的 P0 已另行记录完整 Stage2 沙箱外构建；本诊断不替代该构建与行为验收。

需要在沙箱内编译时，保留严格句柄检查，使用现有 `-SplitCompile 1` 参数。建议指定独立目录，例如：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_local.ps1 -SplitCompile 1 -Build build_cuda_local/stage2_serial
```

该命令是完整构建的建议入口。本诊断实际验证的是原始算术 TU 的串行编译以及最小 PTX 的重复矩阵，**未在本聊天中重新编译全部 Stage2 TU、链接或运行数学/性能验收**。SplitCompile 属于编译身份，改变它应产生独立构建身份；不得将不同配置的对象混用或声称生成物逐字节一致。

未修改生产源码、构建器默认并行度或主程序版本。关联聊天的主流程重构及构建器改动不属于本诊断补丁。保留严格检查的构建处理不要求关闭沙箱或降低进程保护。

## TODO

- 若后续升级 CUDA，先用相同最小 PTX/对应版本生成的 PTX及严格检查矩阵验证；通过后再决定是否恢复沙箱内并行。不能将 13.3 生成的 PTX 直接交给不支持其 PTX ISA 的旧工具链作比较。
- 若需要向 NVIDIA 报告，使用最小输入、工具链哈希、严格检查启动条件及调用偏移整理复现材料；本次未对外提交报告。
- 本次未研究 ptxas 为何产生该等待参数、其他 CUDA 版本是否受影响，以及该路径对沙箱外编译 CPU 占用的影响。需要源码/厂商确认或新增测量后才能进一步下结论。
