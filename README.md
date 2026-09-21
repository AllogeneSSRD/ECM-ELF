# OpenCL-ECM

本仓库为椭圆曲线因子分解算法的 **OpenCL** 实现。支持 **Windows, Linux, macOS & Android** , 同时移植了可在 **Windows - CUDA** 环境运行的 **[GMP-ECM](https://gitlab.inria.fr/zimmerma/ecm)** (Montgomery param 3)。

程序兼容 **GPM-ECM** & **Prime95** 的 savefile 格式, 支持 checkpoint, 自定义算子 (Montgomery 乘/平方与模加/模减) 并针对 **AMD GPU (GCN、RDNA)** 进行汇编与 ISA 调优。

英文概览见 [README_en.md](README_en.md)。


![Static Badge](https://img.shields.io/badge/language-C-blue)
![GitHub License](https://img.shields.io/github/license/AllogeneSSRD/opencl-ecm)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/AllogeneSSRD/opencl-ecm)
![GitHub last commit](https://img.shields.io/github/last-commit/AllogeneSSRD/opencl-ecm)


---

## Contents 目录

| 章节 | 说明 |
|------|------|
| [Quick Start 快速开始](#quick-start-快速开始) | 最短路径：构建 → `ecm` → 微基准 |
| [命令行选项](#命令行选项) | 命令行选项 |
| [从源代码构建 (Windows)](#从源代码构建) | 桌面构建、使用与 OpenCL 能力 |
| [构建 CUDA 后端](#构建CUDA后端CGBN) | NVIDIA CGBN stage-1 构建与使用 |
| [Android](#android) | ECM运行、微基准 |
| [开发与文档](#开发与文档) | 数学原理、param、算子分析、工具、bench、AMD 汇编 |
| [其他文档索引](#其他文档索引) | 正文未单独展开的子文档列表 |

---

## Quick Start 快速开始

```powershell
# ECM stage-1
echo '(2^347-1)' | .\ecm.exe -v -d 0 -gpu -gpucurves 1 1e4 0
echo '(2^421-1)' | .\ecm_cuda.exe -v -gpu -sigma 3:268526266 -gpucurves 1 1e5 0
:: has factor 22000409

echo "(2^991-1)" | .\ecm.exe -v --go -gpu -gpucurves 384 1e5 0
echo '(2^347-1)' | .\ecm.exe -v -d 1 -gpu -sigma 3:561219477 -gpucurves 1 1e4 0
:: has factor 14143189112952632419639

# 检测支持OpenCL的设备
opencl_platform_test.bat
# 运行ECM并于已知因子验证
test_validate_factors.bat
# 显示帮助
ecm.bat
.\ecm.exe -h

# 算子基准测试
.\build_rel\Release\cpu_addsub_bench.exe -a 1,3,5,7,9,11,13,15 512 1e6 16 5 -t 8
.\build_rel\Release\opencl_ecm_addsub.exe --bits 512 10000 128 3 --fixed
.\build_rel\Release\opencl_ecm_montsqr.exe --bits 512 1000 128 1
```

列出全部可切换内核路径：`build\Debug\ecm.exe --showkernel`

---

## 命令行选项

### 运行ECM stage-1（`ecm.exe`）

```text
echo "N" | ecm.exe <-gpu> [-gpucurves <n>] [...] <B1> <B2>
```

从标准输入读取合数 **N**（十进制或表达式），执行 stage-1；`-gpu` 启用 OpenCL 批处理曲线。
尖括号 < >：表示必需提供的参数。
方括号 [ ]：表示可选参数。

```powershell
echo "(2^991-1)" | build\Debug\ecm.exe -gpu -gpucurves 384 1e6 0
echo "(2^4003-1)" | build\Debug\ecm.exe -gpu -gpucurves 384 -v --go --add asm_b32 1e6 0
build\Debug\ecm.exe --showkernel

:: Release build
echo "(2^991-1)" | build_rel\Release\ecm.exe -v --go -gpu -gpucurves 384 1e6 0
```

| 选项 | 说明 |
|------|------|
| `<B1>` `<B2>` | 必选位置参数，在命令末尾 |
| `-gpu` / `-gpucurves <n>` | GPU stage-1 与每批曲线数 |
| `-d <index>` | OpenCL 设备索引 |
| `-v` | verbose 输出详细信息 |
| `--mul` / `--sqr` / `--add` / `--sub` / `--special-mult <path>` | 覆盖各算子内核路径（id/别名/auto） |
| `--showkernel` | 从注册表枚举全部算子（id、别名、文件、支持平台） |

可选: `--go` 计算 Group Order 群阶并分解
- 安装 [Pari/GP](https://pari.math.u-bordeaux.fr/) ; 将 `gp.exe` 添加到环境变量或指定路径 `--gp <path>` 。

上游 `-param`、`-sigma` 等 ECM 参数语义见 [docs/README](docs/README) 第 6 节；

### 高级参数（已取代环境变量）

所有自定义环境变量已移除，改为命令行参数（统一收敛到 `EcmRuntimeConfig`，见
`include/opencl_ecm_runtime_config.h`）。`ecm` 主程序：

| 旧环境变量 | 新参数（`ecm`） | 说明 |
|------------|----------------|------|
| `CGBN_OPENCL_DEVICE_INDEX` | `-d <index>` | OpenCL 设备索引 |
| `ECM_KERNEL_ROOT` / `CGBN_KERNEL_ROOT` | `--kernel-root <dir>` | 覆盖 `.cl` 内核树目录 |
| `CGBN_OPENCL_CACHE_DIR` | `--kernel-cache-dir <dir>` | 二进制缓存目录 |
| `CGBN_OPENCL_CACHE_DISABLE` | `--no-kernel-cache` | 禁用二进制缓存 |
| `CGBN_OPENCL_CACHE_VERBOSE` | `--kernel-cache-verbose` | 缓存命中/未命中详情 |
| `CGBN_OPENCL_COMPILE_VERBOSE` | `--compile-verbose` | 输出编译计时 |
| `ECM_OPENCL_TPI` | `--tpi <1..32>` | 每实例线程数（默认 8） |
| `ECM_STAGE1_FORCE_NORMALIZE` | `--force-normalize <0\|1>` | 强制 normalize 路径 |
| `ECM_MP_ADD_MOD_FUSED_UNROLL` | `--addsub-fused-unroll <1\|2>` | add/sub 融合展开变体 |
| `ECM_PROFILE_OPS` / `_FILE` | `--profile-ops` / `--profile-ops-file <f>` | 算子计数 / CSV |
| `ECM_VERIFY_GPU_RESULTS` / `_STRICT` | `--verify-gpu` / `--verify-gpu-strict` | CPU 交叉校验 |
| `ECM_SYNC_EACH_BATCH` | `--sync-each-batch` | 每批同步 |
| `ECM_GPU_DUMP` / `_FILE` | `--gpu-dump` / `--gpu-dump-file <f>` | 转储 GPU 状态 |
| `ECM_LOG_TIMESTAMP=0` | `--no-log-timestamp` | 关闭日志时间戳（默认开） |
| `ECM_GP_BIN` / `PARI_GP_BIN` | `--gp <path>` | `--go` 所用 `gp` 路径 |

基准 / 诊断工具：`opencl_ecm_addsub` 用 `--no-asm`、`--asm-b64`、`--addsub-fused-unroll`、`--csv`；
`opencl_ecm_montsqr` 用 `--wg-impl`、`--wg-impl4-unroll`、`--csv`、`--kernel-root`、`-d`。

> 仅 `LOGNAME` / `USERNAME` 等系统标准变量仍按系统约定读取（非本程序自定义）。
> Android 无命令行：JNI 入口直接写入 `EcmRuntimeConfig`，缺省沿用默认值。

OpenCL 后端骨架说明：[kernels/opencl/README.md](kernels/opencl/README.md)

---

## 从源代码构建

### 依赖项

| 依赖 | 说明 | 指定路径 |
| ---- | ---- | ---- |
| CMake 3.20+ | 推荐 Visual Studio 2022 或 vcpkg | -DCMAKE_TOOLCHAIN_FILE |
| OpenCL ICD | NVIDIA / AMD / Intel 运行时 | / |
| OpenSSL | 推荐 vcpkg | -DOPENSSL_ROOT_DIR |
| GMP | 推荐 vcpkg | -DECM_WINDOWS_GMP_ROOT |

### 构建

```powershell
cd opencl-ecm
# 1. Debug build (开发调试)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug

# 2. Release build (生产部署，MSVC /O2)
cmake -S . -B build_rel -DCMAKE_BUILD_TYPE=Release
cmake --build build_rel --config Release

# 如有需要, 可以显式指定 vcpkg toolchain, OpenSSL 与 GMP 路径
cmake -S . -B build_rel -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DOPENSSL_ROOT_DIR=vcpkg/installed/x64-windows
cmake --build build_rel --config Release
```

产物在 `build/Debug/` 或 `build_rel/Release/`。主要目标：`ecm.exe`、`opencl_ecm_addsub.exe`、`opencl_ecm_montsqr.exe`、`opencl_asm_selftest.exe`、`opencl_*_isa_export.exe` 等。

> Release 构建后 CMake 自动将 `libcrypto-3-x64.dll`、`libssl-3-x64.dll`、`gmp.dll` 从 vcpkg 复制到输出目录，无需额外 PATH 设置。

### 使用：算子微基准

参数形式（两工具相同）：

```text
<exe> [--bits <bits>] <kernel_iterations> <instances> <launch_repeats>
```

```powershell
build\Debug\opencl_ecm_addsub.exe --bits 512 10000 128 3
build\Debug\opencl_ecm_montsqr.exe --bits 512 1000 128 1
```

- 追加 CSV：`--csv <file>`
- 跨厂商 512/4096 对比报告：[bench/0530_report.md](bench/0530_report.md)

### 其他：OpenCL 与运行时

| 主题 | 说明 | 详细文档 |
|------|------|----------|
| OpenCL 实现总览 | stage-1 主机/内核分工、与 CUDA 差异 | [docs/OPENCL_IMPLEMENTATION.md](docs/OPENCL_IMPLEMENTATION.md) |
| 程序二进制缓存 | FNV-1a 键、`/.opencl_cache/` | 实现见 `kernels/opencl/impl_opencl.cpp`；变量见下表 |
| 内核树与 manifest | `.cl` 注册、路径枚举 | [kernels/opencl/bench/mp_addsub/README.md](kernels/opencl/bench/mp_addsub/README.md) |
| 调试参数 | `--profile-ops`、`--verify-gpu` 等 | [docs/DEBUG_PARAMETERS_GUIDE.md](docs/DEBUG_PARAMETERS_GUIDE.md) |

---

<a id="构建CUDA后端CGBN"></a>

## 构建CUDA后端（CGBN）

`ecm_cuda` 是基于上游 CGBN 的原生 CUDA stage-1（`kernels/cuda/cgbn_stage1.cu`），与 OpenCL `ecm` **共享同一 driver / 参数解析 / 检查点 / 保存 / 日志**，仅在链接期通过选择不同后端（`include/ecm_backend.h`）切换 GPU 实现（OpenCL glue：`src/opencl_backend_glue.cpp`；CUDA glue：`src/cuda/ecm_cuda_backend.cu`）。

### 依赖

| 依赖 | 说明 |
|------|------|
| CUDA Toolkit | 含 `nvcc`（本仓库在 12.6-13.3 上验证），host 编译器需为匹配的 MSVC |
| CGBN | CUDA高精度整数库，置于 `cgbn/` `git clone https://github.com/NVlabs/CGBN.git` |
| GMP / OpenSSL | 同 OpenCL 构建 |

### 为何单独构建

Visual Studio 生成器（`build_rel`）需要 CUDA 的 MSBuild 集成文件；**仅安装 Build Tools 时通常缺失**，此时 CMake 检测不到 CUDA 编译器并**自动禁用** `ecm_cuda`（对现有 OpenCL 工程零影响）。因此改用 **NMake（或 Ninja）生成器**，并在 `vcvars64` 环境下让 `cl` 与 `nvcc` 同时可见：

```bat
# 推荐使用便捷脚本(in `build_cuda/`)

:: 在 "x64 Native Tools Command Prompt"，或先 call vcvars64.bat

# PowerShell
# 修改为本地vcvars64.bat路径
cmd /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake -DECM_CUDA_ARCHITECTURES=80 -S . -B build_cuda_cmake && cmake --build build_cuda_cmake --target ecm_cuda"

# CMD
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
  -S . -B build_cuda_cmake

cmake --build build_cuda_cmake --target ecm_cuda
```

产物：`build_cuda_cmake\ecm_cuda.exe`（GMP DLL 自动复制到同目录）。

### 便捷脚本（`build_cuda/`）

`build_cuda/` 下有一组bat脚本，均先 `call vcvars64.bat` 再执行，无需手动进 Native Tools。**脚本内硬编码了 vcvars64 / cmake / `sm_89` 等路径，换需自行修改；正式构建架构以 `ECM_CUDA_ARCHITECTURES` 为准。**

| 脚本 | 作用 |
|------|------|
| `cfg_cuda.bat` | **配置**：以 NMake 生成器配置到 `build_cuda_cmake`（等价上文 `cmake -G "NMake Makefiles" ...`） |
| `build_cuda_target.bat` | **编译**：`cmake --build build_cuda_cmake --target ecm_cuda`，重定向错误输出 `build_cuda\build_err.txt` |
| `compile_cu.bat` | **诊断**：`nvcc -c` 单独编译 `kernels/cuda/cgbn_stage1.cu`（`--ptxas-options=-v` 看寄存器占用），只编译不链接 |
| `smoke_build.bat` | **冒烟测试**：`nvcc` 直接编译 CGBN 自带 `samples/sample_01_add`，验证 `nvcc + CGBN + cl + gmp` 工具链可用 |

前两个是正式的两步构建流程；后两个仅用于排错 / 环境验证，不产出 `ecm_cuda.exe`。

### CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `-DECM_ENABLE_CUDA` | 检测到 `nvcc` 时 `ON` | 是否构建 `ecm_cuda` |
| `-DCMAKE_CUDA_COMPILER` | 从`Path` `环境变量` 读取 | 修改为 `nvcc.exe` 路径 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" |
| `-DECM_CUDA_ARCHITECTURES` | `80` | CUDA 计算能力（`89`=RTX 40 系；按 GPU 调整，如 `86`=RTX 30 系） |
| `-DECM_CUDA_FULL_BUILD` | `OFF` | `ON` 时编译 CGBN 全尺寸 kernel；默认 dev build 仅支持 **N ≤ 1024 bit**，编译更快 |
| `-DCMAKE_BUILD_TYPE` | `DEBUG` | `Release` |
| `-DCMAKE_CUDA_FLAGS` | / | 传递给 `nvcc` 的参数`="--verbose --ptxas-options=-v"` |

### 使用

命令行与 `ecm.exe` **完全一致**，`-d` 选择 CUDA 设备（`-gpu` 下枚举 NVIDIA 设备；`--mul`/`--sqr`/`--add`/`--sub`/`--special-mult` 为 OpenCL 专用，CUDA 后端忽略）。

```powershell
echo "(2^421-1)" | build_cuda_cmake\ecm_cuda.exe -v -d 0 -gpu -sigma 3:268526266 -gpucurves 32 1e4 0
:: -> factor[0]=614002928307599
```

> 默认 dev build 支持 N ≤ 1024 bit；更大位宽需 `-DECM_CUDA_FULL_BUILD=ON` 重新配置（编译时间显著增加）。

### 队列管理器模式（工作队列 + `ecm.ini`）

`ecm_cuda.exe` 除单次命令行模式外，还内置了一个**工作队列管理器**（对齐原 `work_manager.ps1` 的编排功能，无需外部脚本）：

- **触发**：**无位置参数**（即命令行不带 `<B1>`）时进入队列模式；`-ini <path>` 可指定自定义 ini。
- **配置**：读取 exe 目录下的 `ecm.ini`（`key = value`，`#` 注释）。首次运行缺失时会**自动生成一份带中英双语注释的默认模板**。
- **工作队列**：逐行读取 `worktodo.txt`（`ECMSTAGE2` 格式），每行一个任务；成功后追加到 `worktodo.finished.txt` 并从 worktodo 移除；出错的行就地改写为 `# ERROR <原行>`（`finished` 只保留成功项）。
- **任务行格式**（参考 `pipeline/ecm.py::to_stage2_line`）：
  ```
  ECMSTAGE2=[<aid>,]<k>,<b>,<n>,<c>,<save_name>,<B2>,<skip_curves>,<curves_to_run>[,"factors"]
  ```
  其中 `B1` 从 `save_name` 提取（`m{n}_{b1}.save`，如 `m8237_110e6.save` → B1=`110e6`），`N = (k*b^n+c)/(f1*f2*…)`；`B2`、`skip_curves` 解析后忽略（stage-1 专用）。
- **日志**：所有输出带时间戳**追加**写入 `screen.log`（`log_file` 配置），并同时回显 stdout。
- **同步**：启动时全量、每任务增量地把 `*.save` 同步到两个同步目录（`save_sync_dir_1/2`）。
- **进度条**：ASCII 进度条，颜色由 `progress_color` 配置；`remaining` 用最近 50 个 batch 的平均速度估算。

代码结构与开发指南见 [docs/DEV_ECM_CUDA_QUEUE_MANAGER.md](docs/DEV_ECM_CUDA_QUEUE_MANAGER.md)。

---

## Android

完整 stage-1 **`ecm` 驱动当前为 Windows 桌面目标**；Android 侧提供 **OpenCL 可用性探测**与 **ECM 同源算子微基准**（add/sub、mont mul/sqr），用于真机选型与编译缓存验证。

### 构建

1. 用 Android Studio 打开 **`Android/ECM`**（非仓库根目录）。
2. 确认 **`jniLibs/` 内无** 从手机 `adb pull` 的 `libOpenCL.so`（16 KB 页设备会因对齐崩溃）。
3. 真机 **arm64-v8a** 构建并 Run。

Gradle 会在构建前同步 OpenCL 内核到 APK assets（`syncAddsubKernels`）。总览与 16 KB 页约束：[Android/README.md](Android/README.md)。

### 使用：探测与微基准

| 步骤 | 说明 |
|------|------|
| 设备探测 | 启动 App 自动枚举平台/设备；成功标志 `RESULT: PASS (OpenCL usable)` |
| ECM add/sub | UI 四参数对应桌面 `opencl_ecm_addsub.exe` |
| ECM mont mul/sqr | 对应桌面 `opencl_ecm_montsqr.exe`（WG、tpi=4；不含 AMD asm） |

桌面命令对照与默认参数、512-bit 路径列表输出格式：[Android/ECM/README.md](Android/ECM/README.md)。

```bash
adb logcat ECM-OpenCL:I *:S
adb shell run-as com.example.ecm ls -la code_cache/opencl_cache/
```

### 其他：Android 特有行为

- **OpenCL 加载**：`uses-native-library` + 运行时 `dlopen`，不打包 vendor `.so` — [Android/README.md](Android/README.md)
- **编译缓存**：`codeCacheDir/opencl_cache/`；驱动无法导出 binary 时使用 **live program cache** — [Android/ECM/README.md](Android/ECM/README.md)「OpenCL 编译缓存」
- **与桌面差异**：无 AMD 汇编路径；首次编译大型 `mont_priv*.cl` 可能需数分钟

---

## 开发与文档

以下按主题索引子目录文档；**正文仅作入口，细节以子文档为准**。

### 数学原理与 GPU-ECM 流程

| 文档 | 简介 |
|------|------|
| [docs/ECM_GPU_FLOW.md](docs/ECM_GPU_FLOW.md) | stage-1 数学流程：Montgomery ladder、`s` 比特扫描、检查点 |
| [docs/README.gpu](docs/README.gpu) | 上游 CUDA/CGBN GPU-ECM 启用与用法 |
| [docs/README](docs/README) | 上游 ECM/P-1/P+1 基础与 `-param` 选项 |

### GPU-ECM `param` 与调试

| 文档 | 简介 |
|------|------|
| [docs/DEBUG_PARAMETERS_GUIDE.md](docs/DEBUG_PARAMETERS_GUIDE.md) | `cgbn_ecm_stage1` / batch 参数、`gpu_ecm()` 调试输出 |
| [docs/DEV_ECM_CUDA_QUEUE_MANAGER.md](docs/DEV_ECM_CUDA_QUEUE_MANAGER.md) | 队列管理器 + `ecm.ini` 代码结构与开发指南 |
| [docs/README.lib](docs/README.lib) | `ecm_params` 结构与 `ecm_factor()` 返回值 |

### 算子分析

| 文档 | 简介 |
|------|------|
| [docs/ECM_OPERATOR_ANALYSIS.md](docs/ECM_OPERATOR_ANALYSIS.md) | stage-1 算子混合比、微基准数据、优化优先级（Montgomery 为首要热点） |

### 工具（`tools/`）

| 文档 / 入口 | 简介 |
|-------------|------|
| [tools/DISASM_SETUP.md](tools/DISASM_SETUP.md) | Windows 安装 objdump / llvm-objdump，配合 ISA 导出 |
| [kernels/opencl/bench/mp_addsub/README.md](kernels/opencl/bench/mp_addsub/README.md) | add/sub 内核布局、`gen_all.py` 再生成、bench 优先级 |
| `tools/gen_*.py`、`disasm_*_isa.ps1` | Montgomery/addsub 展开与 asm 块生成；反汇编脚本 |

### 性能测试（`bench/`）

跨厂商总报告：[bench/0530_report.md](bench/0530_report.md)（512 / 4096-bit，NVIDIA / AMD / Intel iGPU）。

| 系列 | 文档 | 主题 |
|------|------|------|
| Montgomery WG | [MONT_WG_SWITCHABLE_FRAMEWORK_CN.md](bench/MONT_WG_SWITCHABLE_FRAMEWORK_CN.md) | 可切换 WG 框架 |
| | [MONT_WG_IMPL4_CROSS_VENDOR_TUNING_CN.md](bench/MONT_WG_IMPL4_CROSS_VENDOR_TUNING_CN.md) | impl4 跨厂商 unroll 调参 |
| | [MONT_WG_MINIMAL_IMPL4_PLAN_CN.md](bench/MONT_WG_MINIMAL_IMPL4_PLAN_CN.md) | 最小 impl4 方案 |
| | [MONT_ISA_4096_ANALYSIS.md](bench/MONT_ISA_4096_ANALYSIS.md) | 4096-bit Montgomery ISA |
| Add/Sub 优化 | [ADDSUB_BASELINE_CN.md](bench/ADDSUB_BASELINE_CN.md) | 4096-bit 纯核基线（AMD gfx1150） |
| | [ADDSUB_ADDMOD_SPECULATIVE_CN.md](bench/ADDSUB_ADDMOD_SPECULATIVE_CN.md) | 投机减模 |
| | [ADDSUB_ADDMOD_FULL_UNROLL_CN.md](bench/ADDSUB_ADDMOD_FULL_UNROLL_CN.md) | 全展开 |
| | [ADDSUB_ADDMOD_ASM_4096_CN.md](bench/ADDSUB_ADDMOD_ASM_4096_CN.md) | 4096-bit asm |
|  profiling / TPI | [RadeonGPUProfiler_1.md](bench/RadeonGPUProfiler_1.md) | RGP 分析记录 |
| | [TPI_1.md](bench/TPI_1.md) | TPI 相关测试 |
|  Intel iGPU | [0530_Intel.md](bench/0530_Intel.md) | 2026-05-30 Intel 核显 raw 记录 |

### AMD 汇编优化

| 文档 | 简介 |
|------|------|
| [docs/README.dev.asm](docs/README.dev.asm) | 上游 asm-redc 目录约定（历史参考） |
| [bench/ADDSUB_ADDMOD_ASM_4096_CN.md](bench/ADDSUB_ADDMOD_ASM_4096_CN.md) | add/sub-mod 4096-bit AMDGCN asm |
| [bench/MONT_ISA_4096_ANALYSIS.md](bench/MONT_ISA_4096_ANALYSIS.md) | Montgomery 4096 ISA 与 asm 路径 |
| `tools/disasm_mont_isa.ps1` | 配合 `opencl_mont_isa_export` 反汇编 |

### IM Compiler（整数乘法代码生成）

| 文档 | 简介 |
|------|------|
| [docs/IM_Compiler/分段整数乘法.md](docs/IM_Compiler/分段整数乘法.md) | 分段整数乘法思路 |
| [docs/IM_Compiler/IMCompiler论文.md](docs/IM_Compiler/IMCompiler论文.md) | 论文摘要 |
| [docs/IM_Compiler/IMCompiler：面向密码学整数乘法的高性能GPU内核自动生成框架.md](docs/IM_Compiler/IMCompiler：面向密码学整数乘法的高性能GPU内核自动生成框架.md) | 框架总述 |

### NPU（Ryzen AI）

| 文档 | 简介 |
|------|------|
| [RyzenAI/README_ADDSUB.md](RyzenAI/README_ADDSUB.md) | NPU add/sub 微基准，对标 OpenCL `opencl_ecm_addsub` |
| [RyzenAI/quicktest/README.md](RyzenAI/quicktest/README.md) | 快速验证脚本 |

### 仓库布局（简图）

```
ECM-OpenCl/
├── src/                    # host code (compiled per target)
│   ├── core/               #   shared driver: ecm_driver, params, checkpoint, save, gpu_common
│   ├── cuda/               #   CUDA backend glue (ecm_cuda_backend.cu)
│   ├── opencl_backend_glue.cpp   # OpenCL backend glue (ecm_backend_* hooks)
│   ├── opencl_ecm_stage1.cpp     # OpenCL stage-1 host
│   └── ...                 #   micro-benchmarks, cl_probe, logging, registry
├── include/                # public headers (ecm_backend.h, cgbn_stage1.h, ...)
├── kernels/opencl/         # OpenCL kernel sources
│   ├── common/             #   shared helpers, operator interface, mp primitives
│   ├── mont_mul/           #   Montgomery multiply kernels
│   ├── add_mod/            #   modular addition kernels
│   ├── sub_mod/            #   modular subtraction kernels
│   ├── bench/              #   micro-benchmark kernels (addsub, mont, asm selftest)
│   ├── impl_opencl.cpp     #   OpenCL backend runtime (context, build, binary cache)
│   └── ecm_stage1*.cl      #   stage-1 ladder entry points
├── kernels/cuda/           # CUDA/CGBN stage-1 (cgbn_stage1.cu) + port shims
├── cgbn/                   # CGBN header-only library (include/, samples/, ...)
├── docs/                   # principles, debug, upstream README copies
├── bench/                  # performance records and tuning notes
├── tools/                  # generators and disassembly
├── Android/ECM/            # Android App
├── RyzenAI/                # NPU micro-benchmarks
└── test/                   # CUDA/OpenCL correctness & bench suite (Makefile)
```

---

## 参考与感谢

本仓库参考并感谢上游 **[ZIMMERMANN Paul / ecm · GitLab](https://gitlab.inria.fr/zimmerma/ecm)**（GMP-ECM）的算法、接口与 GPU 路线设计。

上游原始说明文档保存在本仓库 [`docs/`](docs/) 目录（自上游同步，便于离线查阅）：

| 文件 | 内容 |
|------|------|
| [docs/README](docs/README) | GMP-ECM 基本用法、B1/B2、表达式语法、`-param` / `-sigma` 等 |
| [docs/README.gpu](docs/README.gpu) | 上游 CUDA/CGBN GPU 版说明 |
| [docs/README.lib](docs/README.lib) | `libecm` 库接口与 `ecm_params` |
| [docs/README.dev](docs/README.dev) | 上游 autotools 开发构建 |
| [docs/README.dev.asm](docs/README.dev.asm) | 上游架构相关汇编说明 |



## 其他文档索引

以下为仓库内 **未在上文单独展开** 的 Markdown / 说明文件（已排除 `.gitignore` 中的 `.refactor/`、`.github/`、`build/`、`docs/ecm/` 等）：

| 路径 | 说明 |
|------|------|
| [README_en.md](README_en.md) | 英文项目说明 |
| [docs/README.dev](docs/README.dev) | 上游 autotools 开发说明 |

`test/` 下 Makefile 驱动的 CUDA/OpenCL 测试源文件（无独立 `.md` 索引）用于内核正确性验证；CUDA bench 头文件见 `test/bench_cgbn_*.h`。
