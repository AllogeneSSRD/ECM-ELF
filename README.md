# ECM-ELF

[English](README.en.md) | 中文

椭圆曲线因子分解（ECM）**多后端 stage-1 引擎**：**OpenCL**（Windows / Linux / macOS / Android; param 3)，**CUDA**（CGBN；param 0 Suyama 与 param 3 batch）、**GMP** & **AVX-512 IFMA**（Edwards / Montgomery, x86 CPU 8 curves/线程），三者共用同一个 driver / 参数解析 / 检查点 / 存档逻辑，在链接期切换实现。

程序兼容 **GPM-ECM** & **Prime95** 的 savefile 格式, 支持 checkpoint, 自定义算子 (Montgomery 乘/平方与模加/模减)；另含队列管理器（worktodo / 存档同步 / 断点续跑）与 Prime95 stage-2 交接工具 `ecm_p95feeder`。

> **本仓库原名 OpenCL-ECM。** 改名原因：现在的主力后端已是 CUDA/CGBN（param 0 + param 3）与 AVX-512 批路径，旧名只描述其中一个后端。**二进制名（`ecm.exe` / `ecm_cuda.exe` / `ecm_p95feeder.exe`）、`.save` 格式、`ecm.ini` 键名一律不变**，老脚本与存档不受影响。

![Static Badge](https://img.shields.io/badge/language-C-blue)
![GitHub License](https://img.shields.io/github/license/AllogeneSSRD/ECM-ELF)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/AllogeneSSRD/ECM-ELF)
![GitHub last commit](https://img.shields.io/github/last-commit/AllogeneSSRD/ECM-ELF)


---


## 与现有主流工具相比，本仓库做了什么

面向 GIMPS 生态（Prime95 / gmp-ecm / PrMers）的 **stage-1 工程化实现**：多后端（OpenCL / CUDA-CGBN / AVX-512）、与现有 stage-2 工具**存档互通**、包含梅森数ECM辅助工具。

### 所有后端共用（工具链层）

| 能力 | 说明 |
|---|---|
| **任务队列** | `worktodo.txt` 逐行任务，支持 `ECM=` / `ECM2=`（两者等价）/ `ECMSTAGE2=` 三种行式；成功完成后追加到 `worktodo.finished.txt` 并从队列移除，失败行**就地**改写为 `# ERROR <原行>` |
| **单一 ini 配置** | `ecm.ini`：GPU / CPU / 曲线参数化 / 后端 / 线程 / 亲核性 / 存档命名模板 / 检查点间隔 / 进度条颜色…，；文件缺失时自动生成模板 |
| **中途检查点** | GPU `.ckpt`、Edwards `e{n}_c{k}.ckpt`、Montgomery `m{n}_{b1}_c{k}.ckpt`。默认 600 s 定时保存，`Ctrl+C` 打断时保存 |
| **存档与同步** | 同步 `.save` 存档到 Prime95文件夹，添加 worktodo.add |
| **实时进度** | 命令行进度条显示速率，完成比例，ETA |

### GPU · CUDA / CGBN（`ecm_cuda.exe`）

- **相对 gmp-ecm 的新增能力**
  - `gpu param 0`：**Suyama σ（Z/12）**，与 gmp-ecm `-param 0` 相同，存档支持 gmp-ecm 与 Prime95 继续做 stage 2；
  - **原生 Windows** CUDA 构建（Visual Studio / NMake 两条路，不需要 Linux / WSL / msys2 工具链），并完整接入队列管理器 / 中途检查点 / 存档同步；
  - 曲线参数化可选 `gpu_param = 0 | 3`。
- 支持 **N ≤ 16384 bit**，**推荐用于 `< 12288 bit`** 的整数。最大可到 N ≤ 65536，但是显著慢与 FFT/NTT 实现

### GPU · OpenCL（`ecm.exe`）

- 跨平台（Windows / Linux / macOS / Android）的 gmp-ecm （param 3）路径：支持自定义算子（`--mul/--sqr/--add/--sub/--special-mult`）
- 定位：**小位宽**整数。单条曲线耗时更长，但单曲线资源占用更低、同一位宽下可同时容纳的曲线数更多（实测约 **4×** CGBN），靠曲线数换总吞吐 ⇒ 建议 **≲1024 bit**，更大位宽交给 CUDA/CGBN 或 CPU。
- 归一化到"每流处理器 / CUDA 核心 × 同频"的吞吐对比（CGBN = 100%，作者实测）：

  | 位宽（算子级） | CGBN 基准（Ada Lovelace） | AMD RDNA3.5 | Qualcomm Adreno 830 |
  |---|---|---|---|
  | 256 | 100% | 460% | — |
  | 384 | 100% | 200% | — |
  | 512 | 100% | 152% | 57.1% |
  | 1024 | 100% | 93.6% | — |

  即：AMD iGPU 的**每流处理器**效率在小位宽下远高于 CGBN（到 1024 bit 转为略低）

### CPU（Edwards / Montgomery 两条曲线族 × gmp / AVX-512 两种后端）

- **Edwards（Atkin–Morain，a=1，Z/2×Z/8）**：

  - 支持生成 Prime95 风格的 stage1存档，例如 `e0001213` 
  - 配合 `ecm_p95feeder` 自动投递 ⇒ Prime95 直接执行 stage 2。

- **Montgomery（Suyama σ，Z/12）**：

  - 与 gmp-ecm `-param 0` 同曲线（`A = (v−u)³(3u+v)/(4u³v) − 2`）；
  - 输出 **gmp-ecm 风格文本存档**（`METHOD=ECM; SIGMA=<64 位>; …X=0x…`），可交付并继续进行stage2： Prime95  `ECMSTAGE2=` 队列行、gmp-ecm `-resume`。

- **两种后端**：`backend = gmp` 与 `backend = simd`（**AVX-512 IFMA + int52 radix、8 曲线/批**）。SIMD 路径具有显著性能优势
- **性能对照（归一化单线程、每曲线秒、B1=1e6）**，实测点 + 拟合点（出处：`docs/ECM_Montgomery_STAGE1.md` §14.5）：

  | N | FFT 档 | 本实现 | GMP-ECM 7.0.6 | Prime95 v31 | 本实现/GMP-ECM | 本实现/Prime95 |
  |---|---|---|---|---|---|---|
  | M127 | 128 | **0.118 s** | — | 3.86 s | — | **32.7×** |
  | M521 | 128 | **0.371 s** | — | 3.86 s | — | **10.4×** |
  | M1277 | 128 | **0.979 s** | 2.484 s | 3.86 s | 2.54× | **3.94×** |
  | M2203 | 128 | **2.285 s** | 5.804 s | 3.86 s | 2.54× | 1.69× |
  | M3001 | 256 | **4.137 s** | 9.586 s | **5.65 s**（实测）| 2.32× | 1.37× |
  | M3500 | 256 | **5.164 s** | 11.656 s | 5.65 s | 2.26× | 1.09× |
  | M4001 | 256 | **6.290 s** | 14.624 s | 5.65 s | 2.32× | 0.90× |
  | M5755（拟合）| 384 | 12.05 s | ~21 s | 8.06 s | ~1.7× | 0.67× |
  | M8527（拟合）| 512 | 24.9 s | ~43 s | 10.06 s | ~1.7× | 0.40× |

- **位宽建议**：**建议用于小于 4096 bit 的梅森数**（该区间对 GMP-ECM 快 2.26–2.55×）；超过约 6000 bit 交给 Prime95 的 GWNUM FFT 更划算。
- **梅森数**：`N = 2^k−1` 走折叠域（每模乘 madds 减半），同尺寸下比一般整数快 **~ 2×**。
- **多线程**：利用SMT通常只有10%提升 建议每个物理核心只运行一个线程

### 附带工具

**`ecm-report`** —— PrimeNet ECM 进度统计与可视化：数据源 `www.mersenne.org/report_ecm/`

![ECM progress 1-20000](tools/ecm_report/ecm_progress_1-20000_factored_overlay.png)

**`ecm-prob`** —— 基于启发式模型 + 实测标定的 ECM 概率推算工具（纯 Python，研究用途）：

覆盖 **10 种曲线参数化**，并实现了对应的ECM算法（4 种 Edwards 扭子 Z/4、Z/2×Z/4、Z/12、Z/2×Z/8；Montgomery param 0/1/2/3；p−1 / p+1），底层是 GMP-ECM `rho.c` 的忠实移植（Dickman-ρ + local-ρ + Brent-Suyama）

支持用素数集**实测标定有效除子 D_eff**、T-level 计算 反解 `{bit, B1, Curves, prob(miss factor)}`。


![emp_d_eff_vs_bit](docs/emp_d_eff_vs_bit.png)


| <img src="docs/emp_success_vs_bit.png" width="420"> | <img src="docs/success_vs_B1.png" width="420"> |
|---|---|
| 成功率 vs 位宽（实测） | 成功率 vs B1（实测 + 预测） |


**工作分配系统** —— 把 GPU 上的 stage 1 自动接到 CPU 的 stage 2：

- **`ecm_p95feeder`（本仓库，ECM）**：向 Prime95 发送 Edwars stage-1，统计队列内stage2任务（`worktodo.txt` + `worktodo.add`，支持 `[Worker #N]` ） 确保按依次运行
- **AutoWorktodo（配套项目，独立仓库，不在本仓库内）**：针对 **P-1** 流水线的同类自动化 —— GPU 运行 stage 1，Prime95 运行 stage 2 
  1. **转移**：支持 GpuOwl， PrMers 产出的 `resume_p<exp>_B1_<b1>.p95` 按命名模板 `m{head36}{tail6}`（与 Prime95 的 P-1 存档名一致）复制到目标目录，并把 stage-2 行从暂存区移到 Prime95 消费的 `worktodo.add` 文件；
  2. **自动分配**：同时为每个指数预生成改写好 `B2` 的 stage-2 行；
  3. **可视化仪表盘**（ECharts）：三类任务量、当前任务的进度/IPS/ETA、按速度外推的 works/小时·天·周·月、完成历史柱状图（可按 B1/B2 阶段与指数区间筛选）、运行环境与因子数，支持深浅主题与中英文。


---

## Contents 目录

| 章节 | 说明 |
|------|------|
| [与现有主流工具相比](#与现有主流工具相比本仓库做了什么) | 本仓库的改进点：多后端、与 Prime95/gmp-ecm 存档级互通、性能定位（含每张表的出处） |
| [Quick Start 快速开始](#quick-start-快速开始) | 最短路径：构建 → `ecm` → 微基准 |
| [命令行选项](#命令行选项) | 命令行选项 |
| [CPU stage-1 教程](#cpu-stage-1-教程edwardsatkin-morain与-suyama-montgomery--ecmini-配置) | Edwards / Suyama-Montgomery 两条 CPU 路径与 `ecm.ini` 配置、线程/批数、单线程性能对照（vs GMP-ECM / Prime95）、常见坑 |
| [从源代码构建 (Windows)](#从源代码构建) | 桌面构建、使用与 OpenCL 能力 |
| [构建 CUDA 后端](#构建CUDA后端CGBN) | NVIDIA CGBN stage-1 构建与使用 |
| [Android](#android) | ECM stage-1 分解、设备探测与微基准 |
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

从标准输入读取合数 **N**（十进制或表达式），执行 stage-1；`-gpu` 启用 GPU 批处理曲线（`ecm.exe` = OpenCL，`ecm_cuda.exe` = CUDA/CGBN，启动横幅会打印实际后端）。
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
| `-d <index>` | GPU 设备索引（`ecm.exe` 下是 OpenCL 设备，`ecm_cuda.exe` 下是 CUDA 设备） |
| `-v` | verbose 输出详细信息 |
| `--mul` / `--sqr` / `--add` / `--sub` / `--special-mult <path>` | 覆盖各算子内核路径（id/别名/auto） |
| `--showkernel` | 从注册表枚举全部算子（id、别名、文件、支持平台） |
| `--edwards` | 启用 CPU Edwards stage-1（Atkin-Morain，a=1） |
| `--edwards-backend <auto\|simd\|gmp>` | `simd` = AVX512-IFMA 8 曲线/批（缺 ISA 时直接报错），`gmp` = 标量 |
| `--edwards-mersenne <auto\|on\|off>` | 归约域：`on` 强制 Mersenne 折叠（需 `N = 2^k-1`），**`off` 强制蒙哥马利归约** |
| `--edwards-threads <n>` / `--edwards-naf-w <w>` | Edwards 工作线程（0=auto）与 NAF 窗口（默认 12） |
| `--mont` | 启用 CPU Suyama-Montgomery stage-1（gmp-ecm `-param 0` / Prime95 `sigma_type=1` 语义） |
| `--mont-backend <auto\|simd\|gmp>` / `--mont-threads <n>` | Montgomery 后端（8 曲线/批）与工作线程（0=auto，1=串行） |
| `--mont-torsion <1\|12>` | 指数 torsion：`1` = gmp-ecm `lcm(1..B1)`（结果可与 gmp-ecm 逐点对齐），`12` = Prime95 `choose12` |
| `--affinity <list>` | 绑定工作线程到逻辑 CPU（`0,1,2,3`、范围 `0-7`、混合 `0-3,8,10-11`）；混合核机器上请读教程 4.1 |

> 两条 CPU 路径（`--edwards` / `--mont`）互斥，且它们同样使用 `-gpucurves` 指定曲线数；
> 完整教程（ini 键、线程与批数、存档续跑、常见坑）见
> [CPU stage-1 教程](#cpu-stage-1-教程edwardsatkin-morain与-suyama-montgomery--ecmini-配置)。

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

## CPU stage-1 教程：Edwards（Atkin-Morain）与 Suyama-Montgomery + `ecm.ini` 配置

CPU 侧有**两条独立的 stage-1 路径**，它们共享同一个 `ecm.ini`（`ecm.exe` 与 `ecm_cuda.exe` 都用
`src/core/ecm_queue_config.cpp` 这一份配置），**命令行开关与 ini 键一一对应**。本节是配置教程；
算法与开发细节见 [docs/ECM_EDWARDS_STAGE1.md](docs/ECM_EDWARDS_STAGE1.md) 与
[docs/ECM_Montgomery_STAGE1.md](docs/ECM_Montgomery_STAGE1.md)。

### 1. 选择方法

| 方法 | 命令行 | ini | 曲线族 / 指数 | 何时用 |
|---|---|---|---|---|
| GPU（OpenCL / CUDA/CGBN） | `-gpu -gpucurves <n>` | `method = gpu` | Montgomery，suite 与 GPU 核一致；**跑哪个实现由 exe 决定**（`ecm.exe`=OpenCL、`ecm_cuda.exe`=CUDA/CGBN，启动时打印 `gpu backend : ...`） | 有 GPU 设备时 |
| **CPU Edwards** | `--edwards` | `method = edwards` | Atkin-Morain（twisted Edwards，a=1）；指数与 PrMers/Prime95 同源 | 想在 CPU 上跑、且要 Edwards 语义 |
| **CPU Suyama-Montgomery** | `--mont` | `method = mont` | Suyama-σ（Prime95 `sigma_type=1` / gmp-ecm `-param 0`） | **要与 gmp-ecm param0 结果逐点对齐**、或需要 σ 64 位 |

> 两条 CPU 路径**互斥**：命令行上 `--mont` 优先（同时给两个开关时 Edwards 被忽略）；ini 里
> `mont = 1` 会直接关掉 Edwards，避免同一任务跑两遍。

### 2. 配置 ini 键（CPU 路径）

| ini 键 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `method` | `gpu` / `edwards` / `mont` | `gpu` | **选 stage-1 引擎（三选一）**：`gpu`=GPU 批量（OpenCL 还是 CUDA/CGBN 由 exe 决定，ini 不区分，启动横幅打印真实后端）；`edwards`=CPU Edwards/Atkin-Morain；`mont`=CPU Suyama-σ 蒙哥马利（与 gmp-ecm `-param 0` / Prime95 `sigma_type=1` 同族） |
| `backend` | `auto` / `simd` / `gmp` | `auto` | **两条 CPU 路径共用**。`simd` = AVX512-IFMA 8 曲线/批；`gmp` = 标量 mpn（1 曲线/任务）。`simd` 在缺 AVX512-IFMA 的机器上会**直接报错**，`auto` 自动回退 |
| `field` | `auto` / `mersenne` / `montgomery` | `auto` | **SIMD 归约域（两条 CPU 路径共用）**：`auto` = `N = 2^k-1` 时用 Mersenne 折叠、否则蒙哥马利；`mersenne` = 强制折叠（形状不符直接报错）；`montgomery` = 强制蒙哥马利 CIOS（A/B 对照用）。实际选中的域打印在 `field layer :` 行 |
| `stage1_threads` | `0` = auto，`1` = 串行，`n` | `0` | CPU stage-1 工作线程（两条路径共用）；被"任务数"夹住（见第 4 节） |
| `naf_w` | 3..12 | `0`（=12） | 仅 `method = edwards` 生效：NAF 窗口；字典大小 `2^(w-2)` |
| `exponent` | `lcm` / `choose12` | `lcm` | 仅 `method = mont` 生效：**`lcm` = `lcm(1..B1)`，与 gmp-ecm `-param 0` 逐点对齐（本项目验收口径）**；`choose12` = `12·lcm(1..B1)`，与 Prime95 `sigma_type=1` 对齐（交给 Prime95 做 stage 2 时用，见文档 §16.7）|
| `sigma` | 0 = 随机 | `0` | 固定 σ 时从该值起递增：第 i 条曲线用 `sigma + i`（**64 位**，ini 可写十进制；建议 ≤ 2^63，见 §6 常见坑）|
| `affinity` | `""` / `1,3,5,7` / `0-7` | `""` | 工作线程 `t` 绑定到列表中的 `cpu[t % len]`；支持范围与混合列表。**本机实测不绑定最快（SMT 兄弟勿同用）**，见 4.1 |
| `tmp_dir` | 目录 | `.` | 本地 stage-1 存档目录（`.save` 与中途检查点 `.ckpt` 都放这里）|
| `ckpt_seconds` | 秒，`0` = 关闭 | `600` | **stage-1 中途检查点间隔**（命令行 `--ckpt`）。三种方法都有：GPU（OpenCL 与 CUDA 共用同一套 v4 头，头里记录参数化）存曲线缓冲 + 指数偏移，Edwards 存 `e{n}_c{k}.ckpt`，Montgomery 存 `m{n}_{b1}_c{k}.ckpt`。`0` = 不做定时保存，但 **Ctrl+C 仍会保存一次** |
| `save_name_pattern` | `m{n}_{b1}.save` | 同左 | **写出侧**存档名模板（两条 CPU 路径共用）。读取侧**不依赖模板**：只取文件名最后一个 `_` 与 `.save` 之间那段作为 B1（`m8237_110e6.save` → B1=`110e6`）|
| `worktodo` / `finished` / `log_file` | 路径 | `worktodo.txt` / `worktodo.finished.txt` / `screen.log` | 队列模式输入、成功项、带时间戳日志 |

### 3. 配方：四个可直接抄的 `ecm.ini`

**(a) GIMPS 风格 Suyama-Montgomery（推荐默认）** —— 与 gmp-ecm `-param 0` 逐点对齐：

```ini
method = mont
backend = simd
exponent = lcm
stage1_threads = 0          # auto = min(任务数, 核数)
tmp_dir = saves
save_name_pattern = m{n}_{b1}.save
```

**(b) Prime95 `choose12` 语义**（指数为 `12·lcm(1..B1)`，用于与 Prime95/PrMers 对齐）：

```ini
method = mont
exponent = choose12         # Prime95 sigma_type=1；交给 Prime95 做 stage 2 时用
```

**(c) Edwards + Mersenne 折叠（`N = 2^k-1` 时最快）**：

```ini
method = edwards
backend = simd
field = auto                # N = 2^k-1 ⇒ 折叠域，模乘 madds 减半
stage1_threads = 0
naf_w = 12
```

**(d) Edwards 强制蒙哥马利归约（"mont 模式"）** —— 三种典型用途：

```ini
method = edwards
backend = simd
field = montgomery          # 强制 Montgomery CIOS，无论 N 是否 2^k-1
```

| 用途 | 说明 |
|---|---|
| `N` 不是 `2^k-1` | 例如 `(2^k-1)/f` 这类已被部分分解的 N：折叠域不适用（`2^k ≢ 1`），必须蒙哥马利归约 |
| A/B 基准对照 | 与折叠域跑同一条曲线，量化"C2 对称平方 + 折叠"到底省了多少（`docs/ECM_EDWARDS_STAGE1.md` §13 的闸门用同一个开关）|
| 正确性交叉验证 | 一条曲线的结果应与折叠域**逐点相同**（只是归约方式不同），可用来抓归约内核的 bug |

> 代价：蒙哥马利 CIOS 下每次模乘约 `n(4n+3)` 条 madd（折叠域是 `2n²`，且平方只要 `n(n−1)+2n`），
> 位宽越大差距越明显 ⇒ 能用折叠域就别用 `montgomery`。

> **键名迁移（2026-09-24 整理）**：旧键仍可用（每次运行提示一次），对应关系为
> `edwards`/`mont` → `method`；`edwards_backend`/`mont_backend` → `backend`；
> `edwards_threads`/`mont_threads` → `stage1_threads`；`edwards_mersenne` → `field`；
> `edwards_naf_w` → `naf_w`；`mont_torsion` → `exponent`；`mont_save_pattern` → `save_name_pattern`；
> `gpuckpt_seconds` → `ckpt_seconds`。
> 命令行同理：`--backend` / `--field` / `--stage1-threads` / `--naf-w` / `--exponent` / `--method` / `--ckpt`，
> 旧的 `--edwards-*` / `--mont-*` / `-gpuckpt` 作为别名保留。

### 3.1 中途检查点：跑一半被打断，重跑同一命令即可续跑

续跑不需要额外参数：**重跑同一条命令行**即可（检查点保存每条曲线的 σ，所以曲线集合不会变），跑完后检查点自动删除，`.save` 成为唯一持久产物。启动日志会输出是否resume：

```
checkpoint      : saves/m3001_1e6_c*.ckpt  [12 done, 8 mid-ladder, 964 to run]  autosave 600 s
  resumed from checkpoint: 12 curve(s) already done, 8 mid-ladder
```

- Montgomery 路径的检查点格式是**内部格式**（明文，带校验和与 `END`，见
  [docs/ECM_Montgomery_STAGE1.md](docs/ECM_Montgomery_STAGE1.md) §17）；要交给 gmp-ecm/Prime95 的
  是跑完后写出的 `.save`，它仍然逐点互通。
- 固定 σ（`-sigma`）时被打断的曲线会**精确**接着上次的位偏移跑；实测「杀进程 + 续跑」得到的
  曲线内容与一次跑完**完全一致**（`tools/test/test_mont_checkpoint.ps1`）。
- 检查点的开销可忽略：同进程 A/B 实测 **−0.6%±1%**（开进度/检查点回调 vs git HEAD 的旧 ladder）。

### 3.2 启动固定开销：`s = lcm(1..B1)` 的构造成本

`START` 到第一条进度之间只做两件事：构造指数 `s = torsion·lcm(1..B1)`（素数筛 + 乘积树）和把它展开成
每 bit 一字节的数组。两件事都不依赖 N 与曲线数，所以是**每条任务**的固定成本：

| B1 | 构造 `s` | 展开位数组 | 说明 |
|---|---|---|---|
| 1e5 | 0.002 s | 可忽略 | |
| 1e6 | 0.016 s | 0.002 s | |
| **1e7** | **0.23 s** | 0.02 s | 队列里常见的 B1；旧实现要 **29 s**（已修，见 §18） |
| 1.1e8 | 5.3 s | 0.2 s | 其中筛法 0.4 s，其余是 GMP 的 FFT 大数乘法（GPU 路径同一份实现，所以 CUDA 版也 ~5 s） |

> 三种方法（GPU 批量、CPU Edwards、CPU Montgomery）现在共用 `src/core/ecm_stage1_exp.cpp` 里的
> 同一份乘积树实现。B1 ≥ 1e7 时的成本下界就是"把 1.6e8 bit 的乘积算出来"本身，没有捷径；
> B1 ≤ 1e6 时它完全可以忽略。细节与排查过程：`docs/ECM_Montgomery_STAGE1.md` §18。

### 4. 线程与"批数"：为什么线程数常常跑不满

SIMD 路径**一次算 8 条曲线**（8 lane SoA，整批共享同一个指数），所以并行度上界是
`ceil(曲线数/8)`，不是核数：

- `stage1_threads = 0` ⇒ 自动取 `min(任务数, 核数)`；`1` = 串行；`n` = 指定（会被任务数夹住）；
- 想要 16 个核都忙，`-gpucurves` 至少给到 `8 × 16 = 128`；
- 启动日志会如实打印，例如：
  ```
  stage1 threads  : 8 worker(s) x 8 task(s) of 8 curves
  work split      : 16 batch(es) of 8 curves -> 8 thread(s) busy (8 requested)
  ```
- `--mont-backend gmp`（标量）时 1 曲线 = 1 任务，线程数上界就是曲线数。

### 5. 存档与续跑

- **一个任务一个共享存档**：同一 `(N, B1)` 的所有曲线写进同一个 `m{n}_{b1}.save`，每行一条曲线，
  行内字段自包含（`METHOD=ECM; SIGMA=<64 位十进制>; B1=…; N=…; X=0x…; CHECKSUM=…;`）。
  `{n}` = `N = 2^k-1` 时的 Mersenne 指数 `k`，否则 `N` 的位长；`{b1}` 为紧凑写法（`1e5`、`110e6`）。
- `ecm -resume` 需要**位置参数 B1/B2**（否则报 `Invalid arguments`）：
  ```
  ecm.exe -resume saves\m3001_1e5.save 100000 50000
  ```
- 队列模式：逐行处理 `worktodo.txt`，成功追加到 `worktodo.finished.txt` 并从队列移除；
  出错的行就地改写为 `# ERROR <原行>`。

### 6. 常见坑

| 现象 | 原因 / 处理 |
|---|---|
| `ERROR: … simd requested but this CPU lacks AVX512-F/DQ/IFMA` | `*_backend = simd` 是硬要求；改 `auto`（自动回退标量）或 `gmp` |
| 曲线数 < 2 时 SIMD 没生效 | `auto` 需要至少 2 条曲线才走批路径；显式 `simd` 会强制 |
| 线程数设了却只有一个线程在跑 | 批数上界（见第 4 节）：`-gpucurves` 给得太少 |
| `field = mersenne` 直接报错 | 折叠域要求 `N = 2^k-1`；改用 `field = auto` / `montgomery` |
| ini 里写了 `sigma` 但曲线不对 | 检查 σ 是否 **≥ 2^63**：可以算，但 Prime95 的 `ECMSTAGE2` 用 `atoll()` 读 σ，会截断成另一条曲线（见文档 §16.7.1）；交给 gmp-ecm 做 stage 2 则无此限 |
| 改 ini 后行为没变 | 一定是旧键名（如 `mont_torsion`）——运行时会打印一次 `NOTE: ecm.ini uses the pre-2026-09-24 key ...`，照提示改名即可 |
| `mont = 1` 后 Edwards 设置全部无效 | 两条 CPU 路径互斥，属预期 |
| 多进程同时跑同一 `(N, B1)` | 它们会**追加到同一个 `.save`**；请用不同 `tmp_dir`，或让任务粒度覆盖全部曲线 |
| σ 想复现某条曲线 | `sigma = <值>` 后第 i 条曲线是 `sigma + i`（64 位；gmp-ecm 自动 σ 也超过 32 位）|
| `affinity = 0-7` 反而慢了 20%+ | 该机上 `0-7` 是 **4 个物理核**的 SMT 兄弟（兄弟为相邻编号）⇒ 8 worker 变成 2 路 SMT。不要绑，或改列不同物理核（如 `1,3,5,7`）|
| 绑满 24 逻辑核后变慢 35% | 绑定强制小核簇整簇开 SMT 并被压满（簇占满时 SMT −19%、小核每核效率降到 55%）；不绑定让调度器"物理核优先、SMT 殿后"，吞吐从 5.85× 升到 9.01× |
| `affinity` 里写了非法项 | 现在会打印 `Affinity: ignoring invalid entry '…'` 并忽略该项（旧版本会把 `0-7` 静默解析成 `0`，把全部线程钉到 CPU 0）|

### 7. 一个完整可跑的例子（队列模式）

```
myrun\
  ecm.exe          ← build_vs18\Release\ecm.exe
  ecm.ini          ← 上面的配方 (a)
  worktodo.txt
  saves\
```

`worktodo.txt`（`N = (k*b^n+c)/(f1*…​)`，`ECM2=` 与 `ECM=` 等价）：

```
ECM2=1,2,3001,-1,100000,0,64
```

`ecm.ini`：把配方 (a) 的 `tmp_dir` 改成 `saves`，并确保 `worktodo = worktodo.txt`。
直接运行 `ecm.exe`（**不带位置参数**即进入队列模式），预期输出：

```
===== ECM queue manager =====
method          : montgomery (Suyama sigma, AVX512-IFMA 8-lane batch, torsion=1)
stage1 exponent : s_bits=144344 (lcm(1..100000) x 1)
stage1 threads  : 8 worker(s) x 8 task(s) of 8 curves
  save            : D:\myrun\saves/m3001_1e5.save (64 curve lines, shared)
  curves=64  hits=0  wall=5.02s  (0.078 s/curve)
===== queue done, 1 task(s) processed =====
```

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
cd ECM-ELF
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

### 曲线参数化：`gpu_param = 0 | 3`（`--gpu-param`）

`gpu_param` 选 GPU stage-1 用哪一族曲线（ini 与 CLI 同名）：

| 值 | 曲线族 | 挠点 / 成功率 | 存档形态 | 可用后端 |
|---|---|---|---|---|
| **0** | **Suyama param0**（Prime95 `sigma_type=1` / gmp-ecm `-param 0`）——**与 CPU `--method mont` 是同一条曲线、同一个 σ** | Z/12；有效除子 D≈21–23（batch 族 ≈6.4–7.6，**D 的比值 ≈3×**）。⚠ 单曲线成功率比值远小于 3×：B1=256 实测 bit15–40 为 **1.30×–1.8×**（bit20：30.47% vs 21.64%） | param0 文本（**无** `PARAM=`），带**原始 N** ⇒ gmp-ecm `-param 0` 与 Prime95 都能接着做 stage 2 | 仅 CUDA/CGBN（OpenCL 内核对 0 会明确报错） |
| 3 | gmp-ecm batch 参数化（`P=(2:1)`，`d = σ/2^32`）——历史 GPU 路径 | Z/4 | 带 `PARAM=3` | CUDA 与 OpenCL |

默认 3（旧 ini 无该键时行为完全不变；`ecm.ini` 模板写 0 并注明推荐）。参数化换来的是**成功率**：
param0 每条曲线多花约 22% 时间，但每因子期望代价只有 batch 族的约 0.27×。
实测（M3001，B1=1e5，4096 曲线，RTX 4070 Ti）：param0 = 11.7 M curve-bits/s（= 整机 24 线程 CPU 的 3.8×，
单核的约 35×）；保存的 stage-1 结果交给 gmp-ecm 续 stage 2 已实测成功。细节：
[docs/ECM_Montgomery_STAGE1.md](docs/ECM_Montgomery_STAGE1.md) §19（可行性/实测基线）与 §20（实现与验收）；
回归测试：`tools/test/test_cuda_param0.ps1`。

> CUDA 全量构建默认只带 512 间隔的 TPI=16 档位（2560…8192）。256 间隔曾经加过，实测没有吞吐收益、
> 只让全量构建时间近乎翻倍，已回退。param0 的 kernel 现在与 param3 **同一张档位表**
> （TPI=4 128–512、TPI=8 768–2048、TPI=16 2560–8192、TPI=32 9216–16384），代价是全量构建的实例化
> 数量翻倍；dev 构建仍只带小档。两者能否共用实例化见
> [docs/ECM_Montgomery_STAGE1.md](docs/ECM_Montgomery_STAGE1.md) §21（结论：不能，每 bit 算术不同；
> 但若不再需要 batch 族，直接删掉 param3 才是真正省一半编译时间的方式）。
>
> **CGBN 本体还有多少可挖**：见 [docs/ECM_CGBN_OPTIMIZATION.md](docs/ECM_CGBN_OPTIMIZATION.md) ——
> ① 架构默认乘法变体（sm_70+ 的 WMAD）在 Ada 上已是最优，强切 XMAD/IMAD 慢 1.5–2.3×；
> ② `mont_sqr` 就是 `mont_mul(a,a)`，专用平方上限 12–14%；③ 删掉每 bit 8 次冗余的
> `normalize_addition` 实测 **+5.4%（param3）/ +6.7%（param0）**（已落地，18/18 验收全过）；④ **add-chain（PRAC/NAF）
> 已被证据关闭**：x-only 下窗口法不合法（差分加法需要"每个窗口都不同的差值点"），且一维差分链的加法次数下界
> 1.44/bit 高于赢所需门槛 1.26/bit，PRAC 实测算子数比梯子多 8–16%。两套计时探针（`-DECM_PROBE_ADD_DENSITY=k`、
> `-DECM_PROBE_CHAIN_W=M`）保留下来做后续 A/B；**顺带量出真正的新杠杆**：加法半边占 59% 运行时间，
> 但单位算子成本比倍点高 44%（名义算子数相同）⇒ 是调度/占用率问题，见文档 §5.1/§8。
> **吞吐相关默认值（2026-09-25 实测后已改）**：`TPB=128`、`MAX_ROTATION=1`，并对 ≤2048 bit 的
> kernel 源文件加 `--maxrregcount=56`（新开关 `-DECM_MAXRREG_SMALL`）。占用率是这条路径的主变量：
> M511/M761、8192 曲线实测——寄存器 72→56 得 **+4.7%**，TPB=512（block 数减半）反而 **−15%**，
> MAX_ROTATION 1/2/4 差 ≤0.4%（无影响）。**每批曲线数**同样决定 block 数：8192 曲线比 4096 快
> **7.6%**（32768 快 10.4% 后饱和），所以**建议每批 ≥8192**；填不满设备时程序会打印告警并给出
> 建议的 `-gpucurves` 值。注意已有 build 目录的 CMake cache 会保留旧值，需显式
> `-DECM_TPB=128 -DECM_MAX_ROTATION=1` 或删 cache 重配。
>
> **param2（batch 2，6-挠）已实现并与 gmp-ecm 逐字节对齐**（`--gpu-param 2`）：同一 σ/B1 下我们
> 的 stage-1 x 与 gmp-ecm `-param 2` 完全相同（回归测试 `tools/test/test_cuda_param2.ps1`，7/7）。
> 实测比 param0 快 **5.7%**（M511、B1=1e5；B1 越大越接近算子数给出的 ~11%，因为主机侧建曲线
> 0.127–0.227 ms/curve 会被摊薄），成功率与 Suyama 同档。**注意**：存档带 `PARAM=2`，
> **gmp-ecm 能吃、Prime95 不能吃**（`sigma_type` 只认 0/1/3）—— 所以 stage 2 要么交给 gmp-ecm，
> 要么保持 param0。细节见文档 §5.6。
> 工具：`tools/bench/cgbn_op_probe.cu`（逐算子单价 + 变体 A/B）、`tools/bench/cuda_kernel_ab.ps1`（整 kernel A/B）。


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

Android 侧已实现**完整的 ECM stage-1 分解**（OpenCL）：UI 与桌面 `ecm.exe -gpu -gpucurves B1 B2` 参数一一对应（N 表达式/预设、sigma、checkpoint、内核路径覆盖、worktodo 批量执行、`-save` 存档）；另有 **OpenCL 设备探测**与**同源算子微基准**（add/sub、mont mul/sqr）。运行原生分解需链接 GMP：`Android/ECM/README_ECM_FACTORIZATION.md`。

### 构建

1. 用 Android Studio 打开 **`Android/ECM`**（非仓库根目录）。
2. 确认 **`jniLibs/` 内无** 从手机 `adb pull` 的 `libOpenCL.so`（16 KB 页设备会因对齐崩溃）。
3. 真机 **arm64-v8a** 构建并 Run。

Gradle 会在构建前同步 OpenCL 内核到 APK assets（`syncAddsubKernels` / `syncEcmStage1Kernels`）。总览与 16 KB 页约束：[Android/README.md](Android/README.md)。

### 使用：ECM 分解、探测与微基准

| 步骤 | 说明 |
|------|------|
| ECM stage-1 分解 | UI 对应桌面 `ecm.exe -gpu -gpucurves B1 B2`；未链接 GMP 时运行会提示构建说明 |
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

`tools/` 按用途分子目录：`gen/`（内核/参数代码生成器）、`refactor/`（一次性迁移脚本）、
`bench/`（基准与 A/B 脚本）、`test/`（单测/集成测试与夹具）、`disasm/`（反汇编/ISA 检查）、
`ecm_prob/`、`ecm_report/`、`log_parser/`（各自带 README）。索引见
[tools/README.md](tools/README.md)。

| 文档 / 入口 | 简介 |
|-------------|------|
| [tools/README.md](tools/README.md) | tools 目录索引（各子目录职责、常用命令） |
| [tools/disasm/DISASM_SETUP.md](tools/disasm/DISASM_SETUP.md) | Windows 安装 objdump / llvm-objdump，配合 ISA 导出 |
| [kernels/opencl/bench/mp_addsub/README.md](kernels/opencl/bench/mp_addsub/README.md) | add/sub 内核布局、`tools/gen/gen_all.py` 再生成、bench 优先级 |
| `tools/gen/gen_*.py`、`tools/disasm/disasm_*_isa.ps1` | Montgomery/addsub 展开与 asm 块生成；反汇编脚本 |

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
| `tools/disasm/disasm_mont_isa.ps1` | 配合 `opencl_mont_isa_export` 反汇编 |

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

本仓库引用 **[ZIMMERMANN Paul / ecm · GitLab](https://gitlab.inria.fr/zimmerma/ecm)**（GMP-ECM）的算法、接口与 GPU 路线设计。

上游原始说明文档保存在本仓库 [`docs/`](docs/) 目录：

| 文件 | 内容 |
|------|------|
| [docs/README](docs/README) | GMP-ECM 基本用法、B1/B2、表达式语法、`-param` / `-sigma` 等 |
| [docs/README.gpu](docs/README.gpu) | 上游 CUDA/CGBN GPU 版说明 |
| [docs/README.lib](docs/README.lib) | `libecm` 库接口与 `ecm_params` |
| [docs/README.dev](docs/README.dev) | 上游 autotools 开发构建 |
| [docs/README.dev.asm](docs/README.dev.asm) | 上游架构相关汇编说明 |

