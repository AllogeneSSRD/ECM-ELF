# 构建和辅助工具的环境变量

在启动脚本的同一个 PowerShell 窗口中设置变量。脚本及其子进程继承该值，退出窗口或运行 `Remove-Item Env:\变量名` 可恢复默认解析。命令行参数和构建方法见 [构建与发布](BUILD.md)。

## 构建依赖

| 变量 | 读取入口与取值 | 未设置时 |
| --- | --- | --- |
| `CUDA_PATH_V<主版本>_<次版本>` | 通用 CUDA 工具链解析器读取对应版本的 Toolkit 根目录，例如 `CUDA_PATH_V13_3`、`CUDA_PATH_V12_6`、`CUDA_PATH_V11_8`。显式 CUDA 根目录参数优先。 | 在 `ProgramFiles/NVIDIA GPU Computing Toolkit/CUDA/v<版本>` 查找；通用解析器在调用者未指定版本时根据目标架构选 12.6 或 13.3，发布入口使用各自指定的版本。 |
| `VCPKG_ROOT` | 设为 vcpkg 根目录，脚本和根 CMake 会在其下拼接 `installed/x64-windows`。发布构建优先使用显式 `-Gmp`。根 CMake 优先使用 `-DECM_WINDOWS_GMP_ROOT=...`，其次尝试本机 Zen3 GMP，再读取该变量。 | 发布脚本接着尝试 `VCPKG_INSTALLATION_ROOT` 和约定安装路径。本机构建脚本默认使用仓库内 `third_party/gmp-zen3/dist`。 |
| `VCPKG_INSTALLATION_ROOT` | 发布脚本的第二个 vcpkg 根目录候选，路径规则与 `VCPKG_ROOT` 相同。 | 继续查找仓库相邻的 `vcpkg`、`D:/code/vcpkg` 和 `C:/code/vcpkg`。 |
| `CUDA_PATH` | OpenCL 构建可在该 SDK 根目录寻找 `include/CL/cl.h` 和 OpenCL 导入库。显式 `-OpenClRoot` 优先。 | 桌面 OpenCL 入口查找 OpenSSL 前缀和已安装 NVIDIA Toolkit；独立探针入口查找 vcpkg 前缀和已安装 NVIDIA Toolkit。 |
| `OPENCL_ROOT` | 独立 OpenCL Stage2 构建脚本在 `-OpenClRoot` 省略时优先检查此 SDK 根目录；本机 OpenCL Stage2 包装器也调用此脚本。 | 继续查找 `CUDA_PATH` 和其他 SDK 候选。 |
| `ProgramData` | Windows 提供的共享应用数据路径。NVIDIA GPU 选择诊断脚本从中定位 NVIDIA 安装数据。 | 使用系统提供的值。 [NVIDIA GPU 选择诊断](../../tools/diag/repair_npcf_gpu_selection.ps1) |
| `VSCMD_ARG_TGT_ARCH` | GWNUM 本机构建脚本要求其值为 `x64`，并检查 `cl.exe`、`nmake.exe`。由 Visual Studio x64 Native Tools Command Prompt 或 x64 Developer PowerShell 设置。 | 脚本报错。上游需要汇编时还应能从 `PATH` 找到 `UASM64`。 |

[CUDA 工具链解析](../../tools/build/internal/cuda_toolchain.ps1) · [GMP 依赖解析](../../tools/build/internal/gmp_dependency.ps1) · [根 CMake](../../CMakeLists.txt) · [桌面 OpenCL 构建](../../tools/build/build_opencl_local.ps1) · [OpenCL Stage2 构建](../../tools/build/test/build_opencl_stage2_ntt.ps1) · [GWNUM 构建](../../tools/build/build_gwnum_local.ps1)

例如，使用其他目录中的 CUDA 13.3 工具链运行本机构建。

```powershell
$env:CUDA_PATH_V13_3 = 'D:\SDK\CUDA\v13.3'
.\tools\build\build_stage1_local.ps1 -Arch 89
Remove-Item Env:\CUDA_PATH_V13_3
```

## 辅助工具、测试和采样

| 变量 | 用途 | 默认值与用法 |
| --- | --- | --- |
| `ECM_DB` | `tools/ecm_report` 的导入、绘图及查询脚本共用的 SQLite 路径。 | 默认是这些脚本旁的 `ecm_progress.db`。按次设置可直接传 `--db`，该参数优先。 [ECM 报告工具](../../tools/ecm_report/) |
| `PORT` | 直接启动 `tools/log_parser/app.py` 时传给 Flask 的监听端口。 | 默认 `9500`；使用十进制端口号。 [日志解析器](../../tools/log_parser/app.py) |
| `PLAYWRIGHT_MODULE` | GUI v2 Chromium 资源测试与宠物图集渲染 `require()` 的模块名或绝对路径。 | 默认 `playwright`。 [GUI v2 测试](../../tools/test/gui_v2/) · [图集渲染](../../tools/gen/render_gui_v2_pet.cjs) |
| `MPLCONFIGDIR` | OpenCL Stage2 绘图脚本传给 Matplotlib 的配置和缓存目录。 | 默认使用报告目录下的 `matplotlib`。 [CUDA 对照报告](../../tools/bench/report_opencl_cuda_stage2.py) · [OpenCL 演进报告](../../tools/bench/report_opencl_stage2_evolution.py) |
| `CUDA_PATH_V13_3` | 发布运行测试 `test_release_runtime.py` 和架构切换测试 `test_release_switching.py` 使用此目录下 `bin/cuobjdump.exe`。 | 默认查找标准 CUDA 13.3 安装目录；前一个测试还支持 `--inspector`。 [运行测试](../../tools/test/test_release_runtime.py) · [切换测试](../../tools/test/test_release_switching.py) |
| `ECM_PRAC_REG_TARGET` | CUDA PRAC 对拍脚本的 `--registers` 缺省值，也会传给被测程序。 | 默认 `0`；可用 `0`、`128`、`168`、`255`，仍需满足脚本对位宽、TPI 和公式变体的组合约束。 [PRAC 对拍](../../tools/test/test_cuda_prac.py) |
| `ECM_OPENCL_POINTS` | OpenCL 验证脚本检测到非空覆盖值时，跳过针对 NVIDIA 自动选型的 tune 表断言。 | 默认未设置；验证自动选型时清除该变量。 [OpenCL worker 验证](../../tools/test/check_opencl_stage2_worker.py) |
| `RYZEN_AI_INSTALLATION_PATH` | 随附 Ryzen AI 独立 quicktest 用此 SDK 根目录寻找 ONNX 模型和 Phoenix 装载文件。 | 必须由 SDK 环境提供。NPU 检测脚本只报告 `XLNX_VART_FIRMWARE`、`VAIP_DRIVERS_PATH`、`VART_FIRMWARE`、`XRT_PATH` 等 SDK 变量。 [Quicktest](../../tools/RyzenAI/quicktest/quicktest.py) · [NPU 环境检测](../../tools/RyzenAI/NPUtest.py) |

`PROCESSOR_IDENTIFIER` 由系统提供，CPU Stage2 基准将其写入元数据。`PATH`、`TEMP`、`ProgramFiles`、`SystemRoot`、`WINDIR`、`ComSpec` 和 `PSModulePath` 由系统或终端提供给工具查找可执行文件、库、临时目录及 PowerShell 模块。Android 的 Gradle 包装器沿用 `JAVA_HOME`、`JAVA_OPTS`、`GRADLE_OPTS` 的标准行为；仓库的 Android GMP 路径通过 `ecm.android.gmp.root` Gradle 属性传入。 [CPU Stage2 基准](../../tools/bench/bench_edwards_cpu_stage2.py) · [Android 构建配置](../../Android/ECM/app/build.gradle.kts)

### 基准与测试如何传递变量

基准脚本通常把 `ECM_*`、`NTT_*`、`CUDA_*` 写入子进程环境，经由命令行参数或实验用例设定对照组。有些脚本先剔除继承变量，再为每个样本设置实验值。以脚本的 `--help`、用例和运行目录中记录的命令为准。发布运行验证会自行设置 `CUDA_FORCE_PTX_JIT`、`CUDA_DISABLE_PTX_JIT`、`CUDA_MODULE_LOADING`、`CUDA_CACHE_PATH`、`CUDA_CACHE_MAXSIZE`，并在采集 CUPTI 时设置 `CUDA_INJECTION64_PATH` 和 `ECM_JIT_TRACE_DIR`。后者由测试观察 DLL 读取，指定已创建的追踪输出目录。 [发布运行验证](../../tools/test/test_release_runtime.py) · [CUPTI 观察 DLL](../../tools/test/cuda_jit_trace.cpp)

旧 OpenCL workgroup 基准仍写入 `ECM_DISABLE_MONT_WG`、`ECM_PROFILE_OPS` 和 `ECM_BENCH_CSV`，当前构建对这三项已无读取点。算子计数使用 worker 的 `--profile-ops`，独立算子基准的 CSV 路径使用 `--csv`。 [旧 workgroup 基准](../../tools/bench/bench_wg_scaling.ps1) · [worker CLI](../../src/io/cli.cpp) · [独立算子基准](../../tools/bench/opencl_ecm_addsub_bench.cpp)

测试夹具自行设置 `ECM_LAYOUT_FIXTURE`、`ECM_LAYOUT_ENTRY`（构建目录夹具）、`ECM_TEST_PROTOCOL_PROBE`（GUI 假 worker 探针路径）、`ECM_DIAG_GP_COUNTER`、`ECM_DIAG_HOST_COUNTER`、`ECM_DIAG_HOST_MODE`（依赖诊断夹具）以及 `ECM_TEST_GP_COUNTER`、`ECM_TEST_GP_MODE`、`ECM_TEST_BAD_CURVE_FIRST`（群阶回退夹具）。直接运行上层测试脚本即可；脚本负责建立输入和子进程环境。 `GUI_V2_NO_PAUSE` 和 `PAUSE_AT_END` 也属于包装器内部状态，对应命令行 `--no-pause` 和 `--pause`。 [构建目录测试](../../tools/test/test_stage1_layout_cache.py) · [GUI 启动测试](../../tools/test/gui_v2/test_startup_dispatch.py) · [依赖诊断夹具](../../tools/test/diagnostics_probe.cpp) · [群阶回退夹具](../../tools/test/go_fallback_fixture.cpp)

## 独立性能探针

| 变量 | 探针行为 |
| --- | --- |
| `ED_SOA_FIELD` | Edwards IFMA 独立基准的算术选型。`mont` 或 `0` 指定 Montgomery，`mers` 或 `1` 指定梅森折叠，其余走默认 `auto`。 |
| `ED_SOA_FSELFTEST` | Edwards IFMA 字段自检的次数，默认 `3`。 |
| `ED_SOA_S_OVERRIDE` | 用十进制整数替换 Edwards IFMA 基准的主循环标量。诊断时可用小标量追踪点公式。 |
| `ED_SOA_DEBUG` | 变量存在时在 `verify` 模式打印第一条 Edwards 曲线的详细对拍。清除变量恢复常规输出。 |
| `IFMA_NOTail` | 变量存在时，Montgomery IFMA 独立探针返回归约收尾前的 limb，用来分离收尾开销。清除变量恢复完整算术。 |
| `NTT_V_DENSE_BAD` | 变量存在时，独立 outer V 测试在指定 dense 掩码下破坏探针结果，供负测试验证。 [outer V 探针](../../tools/test/ntt_outer_v_probe.cu) |
| `S2_NCU_TARGET` | 部分冻结源码会编入 Nsight Compute 采样钩子。默认不启动捕获；设成 `n27_tile_forward`、`n27_outer_m7_forward`、`n27_outer_m8_forward` 或 `n11_b990_tile_forward`（也可用 `_inverse` 后缀）捕获首个匹配区间。 [Stage2 Nsight 采集脚本](../../tools/bench/profile_stage2_ntt_ncu.py) · [采集钩子](../../tools/experiments/cuda/stage2_tree/stage2_ntt_profile_range.cuh) |

[Edwards IFMA 探针](../../tools/bench/simd_edwards_bench.cpp) · [Montgomery IFMA 探针](../../tools/bench/simd_mont_notail.cpp)

### 实验 Stage2 和 NTT 引擎

`build_stage2_dev.ps1` 的实验引擎和 `tools/experiments/cuda` 的独立探针各自解析 `NTT_*`，同名变量的缺省值、边界和执行效果以编译进去的读取点为准。对照试验建议从环境覆盖最少的终端启动，使用 benchmark/test 驱动成对设置变量。例如，`test_stage2_tree_gpu.ps1` 成组设置 carry、读回、分块和故障注入，用退出码和参考结果检查路径；`bench_stage2_reduce_ab.ps1` 成组切换归约路径。大多数布尔变量按整数解析，可用 `0`/`1`；少数跟踪开关检查变量是否存在，需要清除变量才能关闭。 [开发构建](../../tools/build/build_stage2_dev.ps1) · [树引擎对拍](../../tools/test/test_stage2_tree_gpu.ps1) · [归约 A/B](../../tools/bench/bench_stage2_reduce_ab.ps1)

| 实验 | 变量和用法 |
| --- | --- |
| Goldilocks 归约 | `NTT_GL_SHORT_REDUCE=0` 请求折叠，`1` 请求短归约。短归约时 `NTT_GL_PTX_REDUCE=1` 请求 PTX。默认随构建，固定后端发现冲突时退出。`NTT_GL_SHIFT_SCALE=1` 试用移位缩放，默认 `0`。 |
| Tile 和 outer 排布 | `NTT_FUSE_T` 指 tile 的 log₂ 长度，默认 `12`。`NTT_FUSE_M` 指普通 outer 层上限，范围 `1..4`，默认 `4`。`NTT_FUSE_COOP_OUTER` 取 `0` 关闭、`1` 强制尝试、`2` 走 RTX 4060 Laptop 的实测选型表；使用强制 cooperative 模式时，`NTT_FUSE_COOP_M` 可选 `5..8`，默认 `8`。`NTT_FUSE_WARP_TAIL` 默认 `0`，`NTT_FUSE_COMPACT_SCRATCH` 默认 `1`。`NTT_OUTER_NARROW` 接受 `0..3` 的 CTA 排布掩码，默认 `0`。 |
| Carry 和显存 | `NTT_CARRY_CHECK_FUSED=1` 请求带校验的 carry 工作区；`NTT_CARRY_CHECK_ALLOC_FAIL=1` 模拟该工作区的分配拒绝。`NTT_CARRY_ROUNDS` 只允许提高按形状算出的进位归约轮数。`NTT_ARENA_WORKSPACE_POOL=0` 关闭工作区池，默认启用。`NTT_MAXCOEFF_LIMBS` 指最大系数探针的 limb 数上限，默认 `8000000`。 |
| 观测和正确性探针 | 设置 `NTT_FUSE_TRACE`、`NTT_HOST_BREAK` 或 `NTT_PROBE_DEBUG` 即可启用对应输出、主机分段计时或特殊输入。`NTT_FUSE_WARP_TEST=1` 扩展 warp 对拍；Stage2 实验入口使用 `NTT_FUSE_COOP_TEST=1` 启动 cooperative 对拍，可用 `NTT_FUSE_COOP_BAD=1` 注入错误。`NTT_S4_KEEPHOUT=1` 在有归约 hook 时仍生成并回读 slot 投影，供 A/B 比较。 |

表中缺省值取自 NTT 探针。实验 Stage2 的队列入口会在初始化时为未设置的变量补充自己的默认值，例如 `NTT_FUSE_WARP_TAIL=1`、`NTT_FUSE_COOP_OUTER=2`、`NTT_S4_BATCH_MB=64`。点生成、S4 下传和 fold 的同名选项见对应的 Stage2 环境变量说明。旧实验的 arena、S5 调试和内置 Stage1 标量等专用选项集中在开发构建小节。 [Stage2 点生成](ENVIRONMENT_STAGE2.md) · [Stage2 NTT 与多项式](ENVIRONMENT_CUDA_NTT.md) · [开发构建的附加变量](ENVIRONMENT_CUDA_NTT.md#开发构建的附加变量)

检查某个实验构建的取值边界时，可用以下命令定位直接和间接读取。 [NTT 实验引擎](../../tools/experiments/cuda/ntt_variants/engine.cuh) · [Cooperative outer](../../tools/experiments/cuda/ntt_variants/ntt_coop_outer.cuh) · [Stage2 实验队列默认值](../../tools/experiments/cuda/stage2_tree/worker.cu) · [Stage2 树引擎](../../tools/experiments/cuda/stage2_tree/engine.cuh)

```powershell
rg -n -C 2 'getenv\(' tools/experiments/cuda
rg -n -C 2 '\b(?:NTT_|S2_NCU_)[A-Z0-9_]+\b' tools/experiments/cuda
```
