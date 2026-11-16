# 隐式环境变量

这些变量由程序或实验工具从进程环境读取。日常配置从 [`ecm.toml` 参考](../ECM_TOML_REFERENCE.zh-CN.md) 查找。下文标明可由环境覆盖的旧入口、当前只有环境入口的实验项、诊断开关，以及由启动器写入的内部变量。

- [Stage1、GUI 和公共运行时](#stage1)见本页。
- [Stage2 规划、点生成、队列和 OpenCL](ENVIRONMENT_STAGE2.md)包括 baby 轮筛和 giant 递推。
- [CUDA NTT 与 Stage2 多项式](ENVIRONMENT_CUDA_NTT.md)包括变换、卷积、F/G 树、余式、批处理和计数。
- [构建、独立探针和实验工具](ENVIRONMENT_TOOLS.md)说明各变量适用的入口和产物。

## 如何设置和清除

在启动程序的终端设置变量。GUI 或队列启动的 worker 通常继承启动器的环境，因此修改环境后需从该终端重新启动相应程序。PowerShell 示例在实验目录直接从一个 Stage1 save 续算，比较关闭 baby 轮筛的完整曲线耗时。

```powershell
# 在新开的 PowerShell 中运行，替换保存点、边界和 D。
$env:NTT_BABY_WHEEL = '0'
try {
    .\ecm_cuda_stage2.exe --save .\input.save --device 1 --b2 26000000000 --d 180180
} finally {
    Remove-Item Env:NTT_BABY_WHEEL -ErrorAction SilentlyContinue
}
```

在 CMD 中用 `set "NTT_BABY_WHEEL=0"` 设置，用 `set "NTT_BABY_WHEEL="` 清除。在支持 POSIX 环境赋值的 shell 中，可对单条命令使用 `NAME=value ./program ...`，用 `unset NAME` 清除当前 shell 中的变量。这里的数值和枚举以字符串传入，按各项要求使用 `0`、`1` 或有效枚举值。部分诊断以变量是否存在为准，赋值 `0` 仍会启用。恢复自动选择时应删除变量。查看当前终端已设置的 ECM 专用变量可用 `Get-ChildItem Env: | Where-Object Name -Match '^(ECM_|NTT_|ED_SOA_)'`。

实验时固定输入、算术路径和参数，记录显式环境值、程序版本和启动日志中的实际选型。若对应变量会改变数学工作量、启用提前停止或注入故障，使用独立队列、输出和 tune 文件。部分 Stage2 覆盖会暂停学习或被调优入口拒绝，限制见 [Stage2 环境变量](ENVIRONMENT_STAGE2.md) 与 [CUDA NTT 环境变量](ENVIRONMENT_CUDA_NTT.md)。

## Stage1

### CUDA CGBN

| 变量 | 默认与取值 | 生效条件和用途 |
|---|---|---|
| `ECM_GPU_STAGE1_ALGO` | `ladder`，接受 `ladder`、`resident`、`prac` | Montgomery CGBN 点乘链。省略 `stage1.cuda_montgomery_chain` 时读取。PRAC 用于 PARAM0，`resident` 是历史实验入口，取决于所用构建。 |
| `ECM_CUDA_STAGE1_SLICE_DEPTH` | `3`，接受 `2`、`3`、`8` | 省略 `stage1.cuda_slice_depth` 时读取。`2` 或 `3` 滚动提交相应数量的切片，`8` 每组最多提交八片再等待。仅在允许预提交的 CGBN 路径生效。 |
| `ECM_STAGE1_TPI` | `0` 由程序选择，另可设 `16` 或 `32` | 请求 Montgomery PARAM0 专用 CGBN 调度所用的每整数线程数。需要构建包含对应位宽/TPI 实例。 |
| `ECM_GPU_DUMP` | 未设置或 `0` 关闭，非零整数打开 | Montgomery 通用 CGBN 路径将 kernel 切片边界状态写到当前目录 `dump.csv`。导出会改变提交方式和计时。 |
| `ECM_EDWARDS_STAGE1_ALGO` | 省略按 NAF、资源和恢复状态决定。可设 `naf`、`binary` | 固定新批次的 Edwards 算法。`binary` 与显式 `--naf-w` 不可同时指定。 |
| `ECM_EDWARDS_DBL_ALGO` | `5` 为 2XY（默认），`4` 为 HWCD | Edwards NAF 倍点，省略 `stage1.edwards_doubling` 时读取。恢复沿用检查点的公式。 |
| `ECM_EDWARDS_HOST_MEMORY_MIB` | 未设置，上限按可用主机内存估算 | Edwards CGBN 主机暂存测试，设无符号 MiB 整数以进一步压低预算。 |
| `ECM_EDWARDS_GPU_MEMORY_MIB` | 未设置，上限按可用显存与 `naf_memory_mb` 估算 | Edwards CGBN 设备内存测试，设无符号 MiB 整数以进一步压低预算。适用时会缩小窗口或回退 binary，仍不足会报错。 |

主机和设备的测试上限都支持 `0`，无效数值沿用正常预算。相关实现见 [Montgomery 策略](../../src/ecm/stage1/montgomery/cuda/host_policy.cpp)、[切片提交](../../src/ecm/stage1/cuda_slices.h)、[Edwards 策略](../../src/ecm/stage1/edwards/cuda/workflow.cpp)和[设备内存准入](../../src/ecm/stage1/edwards/cuda/runtime.cpp)。

### Montgomery PRAC

| 变量 | 默认与取值 | 生效条件和用途 |
|---|---|---|
| `ECM_PRAC_SEARCH` | `7`，`1…50` | 每个素数的候选种子数。CPU IFMA PRAC 直接读取。CUDA PRAC 在省略 `stage1.cuda_prac_search` 时读取。恢复检查点时保持原搜索宽度。 |
| `ECM_PRAC_THREADS` | 自动，按硬件线程数取 `1…8`。可设 `1…32` | PRAC 计划生成线程数。调用方传入非零线程上限时，使用该上限和自动值中的较小者。命中缓存时无需重建。 |
| `ECM_PRAC_PLAN_CACHE` | 继承 `[shared]` 的指数缓存设置 | 执行 CUDA PRAC 时覆盖计划缓存目录。 |
| `ECM_PRAC_TARGET_MS` | `100`，有限小数 `10…500` | CUDA PRAC 自适应切片时长目标，调节使用目标的 80%～120% 范围。 |
| `ECM_PRAC_REG_TARGET` | `0` 默认调度。也接受 `128`、`168`、`255` | CUDA PRAC 选择已编译的寄存器策略。`255` 选择自然寄存器策略，`128` 只与 `single-compact` 组合。 |
| `ECM_PRAC_VARIANT` | `baseline`，另有 `compact`、`outline-add`、`single-add`、`single-compact` | CUDA PRAC 诊断 kernel。非基线一般要求 `168` 或 `255`，`single-compact` 另支持 `128`。需要所选构建包含该实例。 |
| `ECM_PRAC_CONSTANTS` | `none`，另有 `runtime`、`np0`、`m4423` | 对照常数特化。要求 4608/TPI16、`single-compact`、寄存器策略 `168`。`np0` 要求模数低 32 位全 1，`m4423` 要求模数恰为 `2^4423−1`。 |
| `ECM_GPU_STAGE1_SAMPLE_SECONDS` | `0` 或未设置为完整流程。窗口模式未设置时默认 `2` | CUDA PARAM0 专用路径只测主循环的一段。正值到时在切片边界停止。窗口模式接受 `(0,600]` 秒，非窗口模式接受 `[0,10^9)` 秒。 |
| `ECM_PRAC_WINDOW` | 省略为完整流程。可选 `prefix`、`middle`、`tail` | 只重复 PRAC 前、中、尾的指定计划记录。用于局部诊断。 |
| `ECM_PRAC_WINDOW_COUNT` | `16`，`1…32` | 窗口包含的记录数，受计划记录总量限制。 |
| `ECM_PRAC_WINDOW_CHUNK` | `0` 等于窗口记录数 | 单次提交的记录数，正值不得超过实际窗口记录数。 |
| `ECM_PRAC_WINDOW_WARMUP` | `2`，`0…32` | 窗口预热轮数。 |
| `ECM_PRAC_WINDOW_DUMP` | `0`，可设 `1` | 打印窗口结束时的设备点，供参考实现复核。 |

设置 `ECM_PRAC_WINDOW` 或非基线 `ECM_PRAC_VARIANT` 时须已选 PRAC。如果 TOML 中已有 `stage1.cuda_montgomery_chain`，可以在其中设 `"prac"`。窗口循环不代表完整 B1 的曲线吞吐。实现见 [CUDA PRAC 运行](../../src/ecm/stage1/montgomery/cuda/prac/runtime.cu)和[计划生成](../../src/ecm/stage1/montgomery/prac/ecm_prac_plan.cpp)。

### Edwards NTT 和 CPU

| 变量 | 默认与取值 | 生效条件和用途 |
|---|---|---|
| `ECM_EDWARDS_NTT_DICTIONARY` | 按位宽、窗口和曲线数自动选择，接受 `host` 或 `gpu` | 原生 Edwards NTT 奇数倍点字典在 CPU/GMP 或 GPU 上构建。显式 `gpu` 关闭自动策略中的主机回退。二进制和 x-only 路径不读取此项。 |
| `ECM_EDWARDS_CPU_MEMORY_LIMIT_MIB` | 未设置，按可查到的空闲主机内存估算 | CPU Edwards NAF 内存压力测试。以无符号 MiB 整数压低实际可用预算；无效值或无法查询主机内存时沿用正常行为。 |
| `ECM_EDWARDS_NAF_TIMING` | 关闭。仅字符串 `1` 开启 | CPU Edwards 标量/IFMA 的 NAF 主循环计时，有助于分开比较编码、建表与点乘。 |
| `ED_SOA_DEBUG` | 未设置时关闭。设任意值即开启 | CPU Edwards SIMD 的字典、归约、点乘和检查点诊断。即使设 `0` 也会输出。 |
| `ED_SOA_DUMP_STEP` | `0`。正整数为间隔（NAF 数字数） | 与 `ED_SOA_DEBUG` 一起用，按间隔打印 lane 0 全坐标，便于对拍。 |

建表默认条件见 [GPU NTT 选型](../../src/ecm/stage1/montgomery/cuda/ntt/workflow.cpp)。CPU 计时与调试输出见 [标量 NAF](../../src/ecm/stage1/edwards/cpu/ecm_edwards_cpu.cpp)和 [IFMA NAF](../../src/ecm/stage1/edwards/cpu/simd_edwards.cpp)。

## 公共运行时和 GUI

| 变量 | 默认与取值 | 生效条件和用途 |
|---|---|---|
| `CUDA_MODULE_LOADING` | CUDA worker 未发现变量时设为 `LAZY`。可事先设 `EAGER` 对照 | NVIDIA CUDA 模块加载策略。在启动 worker 前设置。 |
| `CUDA_VISIBLE_DEVICES` | 由 CUDA 运行时决定可见设备 | CUDA 运行时变量，可限制并重排可见 GPU。CLI 设备号使用重排后的编号。 |
| `LOGNAME`、`USERNAME` | 先取非空 `LOGNAME`，再取 `USERNAME` | 旧 OpenCL save 生成 `WHO` 字段时将用户名与主机名组合。提交 JSON 的 `user`、`computer` 仍来自 [结果身份配置](ECM_RESULTS.md)。 |
| `ECM_GUI_NVML` | 未设置时搜索 `nvml.dll`，再尝试 NVIDIA 的 NVSMI 目录 | 在 Windows GUI 测试中指定 NVML DLL 路径。指定不存在的路径可测试监控不可用的降级流程。 |
| `ECM_GUI_CJK_FONT` | 自动搜索系统字体。可设可读字体路径或 `none` | Windows 旧版 GUI 的 CJK 字体诊断。`none` 模拟无 CJK 字体，其他找不到的路径继续系统搜索。 |
| `WINDIR` | Windows 环境省略时用 `C:\Windows` | 旧版 GUI 在这个目录的 `Fonts` 子目录中搜索系统字体。 |
| `PATH` | 继承系统搜索路径 | GUI v2 群阶工具搜索 `gp.exe` 时使用其中的绝对目录。构建工具的其他用途见 [工具环境变量](ENVIRONMENT_TOOLS.md)。 |

读入口见 [CUDA 加载](../../include/ecm/platform/cuda_loading.h)、[存档](../../src/io/save.cpp)、[GUI 监控](../../src/ecm/application/gui/gpu_monitor.cpp)、[Windows 字体](../../src/ecm/application/gui/platform_win32.cpp)和 [PARI/GP 搜索](../../src/ecm/application/gui_v2/group_order_job.cpp)。
