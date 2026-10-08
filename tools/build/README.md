# CUDA 构建入口

从任意工作目录调用 PS1，或双击同名 BAT。根目录保留 Stage1 / Stage2 × 开发 / 本机生产 / 发布生产六组入口，全部默认并行编译。BAT 将参数原样转交 PS1；无参数运行会在结束时暂停，有参数运行直接返回构建退出码。默认参数在 PS1 中维护。

## 六个入口

- `build_stage1_local.ps1` / `.bat`：完整 Stage1，输出 `build_cuda_cmake/ecm_cuda.exe`。
- `build_stage1_dev.ps1` / `.bat`：同一 Stage1 构建逻辑，独立输出 `build_cuda_dev/ecm_cuda.exe`。默认仍编译全部位宽；用 `-Tiers 4608` 等显式缩小开发范围。
- `build_stage2_local.ps1` / `.bat`：生产 Stage2，输出 `build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe`，可读取 save / worktodo / INI。
- `build_stage2_dev.ps1` / `.bat`：实验 Stage2 队列程序，输出 `build_cuda_cmake/development_stage2/ecm_cuda_stage2.exe`，使用 `tools/bench/ecm_cuda_stage2_dev.cu`。运行时遵循开发引擎要求，使用 `--log-level debug`；参数比较可显式选择其他算术后端。
- `build_stage1_release.ps1` / `.bat`：六架构并行编译 → 按架构拆分 → 自动打包及 ZIP。默认构建目录 `build_cuda_release`，发布输出 `dist/cuda`。
- `build_stage2_release.ps1` / `.bat`：生产引擎六架构编译 → 自动打包及 ZIP，输出 `dist/cuda-stage2/cuda-stage2-sm<arch>`；`-Arch sm_89` 只发布一个架构。

本机构建默认针对 `sm_89`，适合本机两张 Ada GPU；其他设备通过 Stage1 的 `-Arch 86` 或 Stage2 的 `-Arch sm_86` 指定。此处“本机”表示单架构编译，不会自动检测 GPU，也不会自动发布。

Stage1 三类入口均默认 `Jobs=8`，并行编译独立 CUDA TU 后链接。Stage2 三类入口均默认 `SplitCompile=8`，使用 nvcc 内部编译并行；多架构 Stage2 按架构依次构建，各架构内保持八路编译优化，避免同时叠加多个大型 nvcc 进程。两阶段发布默认架构均为 `60,70,75,86,89,120`。

工具链分组：`60,70 -> CUDA 12.6`，`75,86,89,120 -> CUDA 13.3`。程序通过绝对路径调用对应 nvcc；Stage1 的 nvprune/cuobjdump 也来自同一工具链。默认安装目录由 `CUDA_PATH_V12_6` / `CUDA_PATH_V13_3` 或 Program Files 下的对应版本解析，发布入口可分别用 `-LegacyCudaRoot` / `-CudaRoot` 覆盖。

脚本从所选 CUDA 的 `host_config.h` 读取 MSVC 版本范围，选择本机最新的受支持工具集，并显式传给 vcvars/nvcc/CMake。当前 12.6 选择 MSVC 14.16，13.3 选择 14.51；不默认跳过编译器版本检查。`-VcVars` 可指定 Visual Studio 安装入口，仍会选择该安装内兼容的工具集。

## 默认参数

Stage1：

```text
BuildDir=build_cuda_cmake; Arch=89; Jobs=8; Tiers=""; ECM_CUDA_ENABLE_PRAC=OFF
Release; FULL_BUILD=ON; COMPRESS=ON; EMBED_PTX=OFF
TPB=128; MAX_ROTATION=1
MAXRREG=0; MAXRREG_SMALL=0; MAXRREG_SUYAMA=0; REG_TARGET_FORCE=0
MERS_FOLD=0; NO_PARAM2=0; PROBE_ADD_DENSITY=1; PROBE_CHAIN_W=0
```

`REG_TARGET_FORCE=0` 使用内核已有的分档寄存器策略，不统一强制上限。显式重设折叠域、全局寄存器限制和探针开关，避免已有 CMake cache 残留实验值。`-Extra "-D..."` 最后追加，供明确的实验覆盖使用。

TPB=128 保留现有通用基线；256/512 的收益依赖位宽、曲线族和批量。折叠域仅支持特定模数/曲线族，PRAC 的最佳 TPI/寄存器配置仍受已测范围限制，均不加入通用默认。最近 Stage1 结论见 [PRAC 终结哨兵报告](../../docs/ECM_STAGE1_PRAC_RULE_SENTINEL_20261007.md)。构建入口不覆盖运行时 INI 的算法、曲线批量或切片时长。

Stage1 三类入口默认不编译实验 resident/PRAC 模块，包括全部 PRAC CUDA TU 和 CPU 链规划器。显式 `-EnablePrac`（CMake：`-DECM_CUDA_ENABLE_PRAC=ON`）才加入；旧缓存中的 ON 会被入口默认 OFF 覆盖。未编入时请求 `ECM_GPU_STAGE1_ALGO=prac|resident` 会给出明确错误。实验推荐 `-EnablePrac -Tiers 4608` 缩小实例化范围。

原 `cgbn_stage1_prac_tpi32.cu` 的 8 档大型模板实例化拆成四组，每组 2 档（9216/10240、11264/12288、13312/14336、15360/16384），由轻量调度 TU 连接，可并行编译；内核算法未改。用户原记录为 1,553.6 s，此次未重新计量完整 PRAC 构建，拆分后的加速比尚未实测。发布拆分只处理当前 `compile_commands.json` 列出的 CUDA 对象，旧 PRAC 对象即使仍在目录也不进入本次清单。

Stage2：

```text
Engine=production; Arch=sm_89; GlBackend=ptx
OuterUnrollU=0; AddSubMask=1; SplitCompile=8
```

固定 PTX / AddSubMask=1 沿用已选定的生产算术组合；不默认启用 OuterUnrollU=4 等形状相关实验。`SplitCompile=8` 控制编译并行度，不改变运行时算法。开发入口默认 `Engine=development`，显式传入 `AddSubMask=1` 与生产保持相同算术基线；其余默认由本机入口维护。

## 使用

在仓库根目录：

```powershell
# 本机完整构建；Stage1 并行编译 CUDA TU，Stage2 使用 nvcc split compile。
.\tools\build\build_stage1_local.bat
.\tools\build\build_stage2_local.bat

# Stage1 单档开发：4608-bit 容器，而不是输入 N 恰好为 4608 bits。
.\tools\build\build_stage1_dev.bat -Tiers 4608 -Incremental

# 显式加入实验 PRAC；默认不编译。
.\tools\build\build_stage1_dev.bat -EnablePrac -Tiers 4608

# Stage2 开发引擎，以及显式算术 A/B。
.\tools\build\build_stage2_dev.bat
.\tools\build\build_stage2_dev.bat -AddSubMask 0 -Build build_cuda_cmake/stage2_mask0

# 只重编 Stage2 host：要求已有 manifest 和匹配的 CUDA 对象/参数。
.\tools\build\build_stage2_local.bat -Build build_cuda_cmake/production_stage2 -HostOnly

# 发布生产：编译完成后自动生成独立可用目录、ZIP 和 SHA256。
.\tools\build\build_stage1_release.bat
.\tools\build\build_stage2_release.bat

# 显式缩小 Stage1 发布架构，PTX 回退架构必须包含在列表中。
.\tools\build\build_stage1_release.bat -Archs 86,89 -PtxArch 86

# Stage2 批量发布；逐架构构建独立程序，无需对单架构产物再 nvprune。
.\tools\build\build_stage2_release.bat -Archs 60,70,75,86,89,120

# 只发布旧架构；Stage1 的 PTX 架构须在此次列表中，或设为空关闭 PTX。
.\tools\build\build_stage1_release.bat -Archs 60,70 -PtxArch 60
.\tools\build\build_stage2_release.bat -Archs 60,70
```

Stage1 的 `-Incremental` 依赖时间戳，修改编译选项后应省略该开关或用 `-FullRebuild`。`-RelinkOnly` 直接交给 CMake，可能仍因依赖变化重编 CUDA。Stage2 会核对源文件哈希；旧产物的 `SplitCompile=1` 与新默认 8 不同，首次复用旧 CUDA 对象时需传入原编译值，或正常重建。

## 子目录

- `dev/`：通用 CMake 开发构建、GUI、独立 `stage2_tree_gpu.exe` 数学/性能驱动，以及 `basic256.bat` / `fold256.bat` 显式实验。
- `test/`：NTT / 点运算 / Stage2 探针构建、CPU 树参考和树门禁脚本。运行构建入口不会自动启动这些测试。
- `internal/`：共享并行编译驱动、生成配置门禁、Stage1 拆分和两阶段打包/归档辅助。发布入口位于根目录。

```powershell
# 低层 Stage2 树实验驱动；与上面的队列开发程序用途不同。
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/dev/build_stage2_tree_gpu.ps1

# Stage2 构建并打包；Stage1 使用同目录 build_stage1_release.ps1。
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_release.ps1 -Arch sm_89
```

工具链：PowerShell 5.1+、Visual Studio x64 C++ 工具、CMake、CUDA/nvcc 和 GMP。完整默认发布需要同时安装 CUDA 12.6、13.3 及各自支持的 MSVC；缺少任一所需工具链会在编译前报错。两阶段本机/开发入口也支持 `-CudaRoot`、`-VcVars`，按单个目标架构自动选择默认 CUDA；Stage1 还支持 `-OpenSslRoot`。例如本机旧架构构建使用 `-Arch 60 -BuildDir build_cuda_legacy`，Stage2 使用 `-Arch sm_60 -Build build_cuda_cmake/stage2_legacy`。

## GMP 与 CPU 兼容性

本机/开发入口默认 `third_party/gmp-zen3/dist`。发布入口默认解析 vcpkg `installed/x64-windows`：优先 `VCPKG_ROOT` / `VCPKG_INSTALLATION_ROOT`，然后仓库同级 vcpkg 及已知本机路径；找不到时明确报错，不回退到 Zen3。Stage1、Stage2 均可通过 `-Gmp <prefix>` 显式选择其他匹配的头文件、导入库和 DLL。

当前本机 vcpkg GMP 的构建记录选择 `x86_64/k8` 汇编、MSVC `/O2`，没有强制 Zen3 ISA。仓库 Zen3 构建选择 `x86_64/zen3` 的 `mul_1`、`addmul_1`、`mul_basecase`，实际使用 `mulx/adcx` 等 BMI2/ADX 指令；没有运行时 fat 调度，不能面向所有 x64 CPU 分发。在满足实际指令集的其他 AMD/Intel CPU 上可能运行，但“使用 Zen3 优化”不等于“只能 AMD”，也不等于“所有 x64 兼容”。GMP 官方说明见 [CPU 类型与 fat 构建](https://gmplib.org/manual/Build-Options)。

vcpkg 路径本身不能证明任意自定义 triplet/port 都通用；发布时应保留使用的构建基线。当前 MSVC vcpkg port 的 `fat` feature 不支持原生 Windows；默认 x64 包也不是运行时选择 Zen3 的 fat DLL。显式 `-Gmp` 自定义依赖时，CPU 兼容范围由该依赖的真实构建决定。

发布入口使用同一 prefix 的 `include/gmp.h`、`lib/gmp.lib`、`bin/gmp-10.dll`，记录 SHA256 并在打包前核对。Stage2 的编译签名也包含这些依赖，切换 GMP 会触发重建；旧清单未记录 GMP 身份时，不能直接 `-HostOnly` 或 `-SkipBuild` 复用，需正常重建。不能只替换发布包中的 DLL 来绕过一致性检查。包内附带该依赖的 `share/gmp/copyright`，命名为 `GMP-COPYRIGHT.txt`；自定义发布 prefix 也需提供此文件。

Stage1 的 CMake 构建后 DLL 复制也按 `GMP_LIBRARY` 所在 prefix 选择；脚本再次同步该 DLL，覆盖旧产物目录里遗留的其他版本。

Stage2 编译日志统一显示构建摘要、编译开始、各 TU 的 `ok/FAIL + seconds`、CUDA 对象复用、编译汇总、链接耗时和产物路径/总耗时；完整 nvcc 输出仍保留在 `_objects/*.log`，失败时显示日志路径及末尾内容。

## 发布输出与续用

```text
Stage1:
  build_cuda_release/cuda12_6/    # 60,70; build/split/package manifests
  build_cuda_release/cuda13_3/   # 75,86,89,120; build/split/package manifests
  dist/cuda/ecm_cuda_sm<arch>.exe
  dist/cuda/cuda-stage1-sm<arch>/
  dist/cuda/cuda-stage1-sm<arch>.zip[.sha256]
Stage2 (-Arch sm_89):
  dist/cuda-stage2-sm89/
  dist/cuda-stage2-sm89.zip[.sha256]
Stage2 (default / -Archs):
  dist/cuda-stage2/cuda-stage2-sm<arch>/
  dist/cuda-stage2/cuda-stage2-sm<arch>.zip[.sha256]
```

包中包含 exe、GMP DLL、共享 INI 模板、空队列、`ECM_INI_REFERENCE.md`、`LICENSE` 和 `GMP-COPYRIGHT.txt`。不分发 `*manifest.json`、`DEV_*`、`README.md` 及其他 Markdown 文档。重新打包会清理包目录顶层遗留的这些元数据/文档，保留已有 INI 和队列；ZIP 从独立暂存目录生成，避免收录用户 save、日志、进度及运行配置。暂存、构建/拆分/打包清单及 Stage1 原始/裁剪对象留在构建目录，便于核查；ZIP 旁保留 `.sha256`。

Stage1 默认 `Archs=60,70,75,86,89,120; PtxArch=75; Jobs=8`。每组独立编译多架构程序，使用未压缩 fatbin 以支持对应工具链的 nvprune。指定的 `sm_75` 包保留 PTX；60/70 及其他包为 SASS。拆分失败立即停止，恢复原始 CUDA 对象和该组多架构 exe；成功后核对各产物实际 cubin 架构/PTX，再打包。`-SkipBuild` 必须匹配对应组清单中的源码、依赖、CUDA 路径/版本、MSVC、架构、PRAC 开关及 exe；原始对象在拆分前也核对哈希。`-SkipSplit` 仅构建，`-SkipPackage` 仅构建和拆分。

Stage2 默认 `Archs=60,70,75,86,89,120; SplitCompile=8`，保持固定 PTX / AddSubMask=1 / OuterUnrollU=0。此处固定 PTX 指模运算的内联 PTX 后端，不表示程序包含用于驱动 JIT 的 compute PTX。显式 `-Arch sm_89` 只构建该架构；`-Archs` 与显式 `-Arch` 互斥。多架构时自定义 `-Build` / `-OutDir` 用作父目录，各架构选择对应工具链。`-SkipBuild` 仍检查各架构生产清单、源码及 CUDA/MSVC 身份；旧清单需正常重建以补充这些字段。`-Stage1Exe` 可附带用户提供的已验收 Stage1 程序，不替它编译或验证 GPU 兼容性。

原 Stage1 根构建目录的单清单不再作为新分组构建的缓存使用。显式切换 CUDA 或 MSVC 时，已有 CMake 目录的工具链不匹配会明确报错，应另选目录；脚本不会自动删除旧对象或缓存。

这些入口不会运行测试，也不会把构建成功标记为所有架构运行验收；清单保留 `tests_run=false`。Stage1 包的详细说明见 [Stage1 发布说明](../../docs/ECM_CUDA_STAGE1_RELEASE.md)。

## 旧路径迁移

```text
local_build.ps1             -> build_stage1_local.ps1
build_ecm_cuda_stage2.ps1   -> build_stage2_local.ps1
build_dev.ps1              -> dev/build_dev.ps1
build_gui.ps1              -> dev/build_gui.ps1
build_stage2_tree_gpu.ps1   -> dev/build_stage2_tree_gpu.ps1
basic256.bat / fold256.bat  -> dev/
build_*_probe.ps1           -> test/
build_stage2_tree_ref.ps1   -> test/build_stage2_tree_ref.ps1
check_stage2_tree_gpu.ps1   -> test/check_stage2_tree_gpu.ps1
release_build.ps1          -> build_stage1_release.ps1
release_stage2.ps1         -> build_stage2_release.ps1
release_split.ps1          -> internal/split_stage1_release.ps1
parallel_nvcc.ps1          -> internal/parallel_nvcc.ps1
check_ecm_config.cmake     -> internal/check_ecm_config.cmake
```

仓库内当前调用和说明已同步更新；外部自动化需使用新路径。历史实验快照保持原始路径和哈希，快照工具按原清单读取。旧 Stage2 build manifest 仍记录旧脚本位置：重新使用搬迁后的构建入口生成 manifest 后再打包；不要手改清单以跳过源文件核对。
