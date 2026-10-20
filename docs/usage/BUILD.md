# 构建与发布

工具位于 `tools/build/`。从任意目录调用 PS1，或双击同名 BAT；BAT 转交参数，无参数运行结束后暂停。默认值由 PS1 维护。

Stage1 CUDA 所需 CGBN 头文件位于 `third_party/cgbn/include`；CMake 通过 `CGBN_INCLUDE_DIR` 指向该目录，需要其他安装位置时可在配置时覆盖。

## 六组入口

独立 CPU Edwards Stage2 使用新功能入口 `tools/build/test/build_cpu_stage2.ps1`，不依赖 CUDA；依赖和输出见 [CPU Stage2](EDWARDS_CPU_STAGE2.md#构建与运行)。它尚未包含在下列 CUDA 发布包中。

| 程序 | 开发 | 本机生产 | 发布生产 |
| --- | --- | --- | --- |
| Stage1 | `build_stage1_dev` | `build_stage1_local` | `build_stage1_release` |
| Stage2 | `build_stage2_dev` | `build_stage2_local` | `build_stage2_release` |

每项均有 `.ps1` 与 `.bat`。Stage1 默认 8 路独立 CUDA TU 并行；Stage2 默认 `SplitCompile=8`，多架构依次构建，每个架构内并行。构建入口不自动运行测试。

本机/开发默认 sm89。Stage1 `-Arch 86`、Stage2 `-Arch sm_86` 可选择其他架构；本机表示单架构构建，不自动检测设备。Stage1 默认输出 `build_cuda_cmake/ecm_cuda.exe`，Stage2 默认输出 `build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe`。开发分别使用独立目录。

## 默认策略

Stage1：完整位宽/曲线族、通用 Montgomery、TPB128、分档寄存器策略；resident/PRAC 默认不编译，显式 `-EnablePrac` 才加入。`-Tiers` 可缩小开发档位。折叠域、统一寄存器上限和算术探针不是通用发布默认。

Edwards binary/NAF使用32档容器、七组 `kernels/cuda/cgbn_edwards_instances_*.cu`，同样受 `-Tiers` 筛选；容器边界仍按Edwards的一位余量判断，Montgomery按六位。新功能构建入口 `tools/build/test/build_edwards_gpu_workers.ps1` 可用 `-Stage stage1 -Tiers '' -Jobs 8` 编译完整三曲线族。该worker测试入口默认仅1024；检查其他位宽前需显式选档或使用空列表编全档。

`build_edwards_gpu_stage1_probe.ps1` 默认8路并行；`-WorkerBuild <stage1构建目录>` 在源文件、命令、架构及对象摘要全部匹配时，复用该worker的Edwards对象，只编译探针和CPU参考，再链接独立验证程序。构建日志 `par_nvcc/timings.json` 区分队列实际墙钟和各TU同时运行的时间之和，后者不作为串行构建实测或加速比。

独立探针默认编全档，可用 `-Tiers 384` 构建单档；验证时执行 `edwards_gpu_stage1_probe.exe 1 <新输出目录> 384` 检查回退、上限和三种执行模式，全档执行时省略最后的384。探针编译和链接均显式使用 `/MD` 与worker一致。短标量CPU参考使用w2，边界输入仅按奇数模数和有效曲线构造筛选，避免把大素数搜索或大CPU字典作为算术验收的前置工作；末点、GCD、点不变量和恢复比较均保留。

Stage2：独立生产 CUDA 引擎，固定 Goldilocks PTX 后端、canonical 减法、`AddSubMask=1`、`OuterUnrollU=0`。这里“PTX 后端”是内联模算术，不表示一定携带驱动 JIT 的 compute PTX。开发引擎由独立 wrapper 选择。

## 工具链与架构

两阶段发布默认 `Archs=60,70,75,86,89,120`。

| 架构 | CUDA |
| --- | --- |
| sm60、sm70 | 12.6 |
| sm75、sm86、sm89、sm120 | 13.3 |

需要安装所选 CUDA、相容的 Visual Studio x64 C++ 工具集、CMake、PowerShell 5.1+ 和 GMP。脚本按 CUDA 头文件声明的 MSVC 范围选择受支持工具集，不默认绕过版本检查。通过 `-LegacyCudaRoot`、`-CudaRoot`、`-VcVars` 指定安装入口。

Stage1 分工具链构建多架构程序，再用对应 nvprune/cuobjdump 拆分；默认 sm75 包保留 compute75 PTX，其余包携带各自 SASS。Stage2 逐架构直接构建生产程序。不同 CUDA/MSVC 的缓存目录分离。构建成功不等于每个架构均已完成运行验收。

## GMP 与打包

本机/开发默认使用本地 Zen3 GMP；发布默认使用 vcpkg 通用 x64 GMP，缺失时明确报错，不自动回退到 Zen3。自定义 `-Gmp <prefix>` 必须提供匹配的头文件、导入库、DLL 和许可证。Zen3 库使用 BMI2/ADX 等指令，其 CPU 兼容范围由实际构建决定。

包内只分发程序、GMP DLL、共享 TOML 模板、空队列、`ECM_TOML_REFERENCE.md` 与许可证。不包含 `*manifest.json`、`DEV_*`、README 或其他项目文档。重新打包保留已有 TOML/队列；ZIP 从干净暂存目录生成，不收录 save、日志和运行数据。

Stage1 输出 `dist/cuda/`，Stage2 多架构输出 `dist/cuda-stage2/`；每架构有目录、ZIP 和 SHA256。构建/拆分/打包清单留在构建目录，包括源码、依赖、编译器、架构和程序身份。

## 复用构建

Stage1 的 `-Incremental` 按编译命令、响应文件、递归依赖内容与对象 SHA256 判断 CUDA 对象是否可复用；保留时间戳也不能绕过内容校验。缺少当前依赖记录的旧对象会重编一次。`-RelinkOnly` 是显式交给 CMake 处理主机编译和链接的入口，不等同于已通过 CUDA 身份校验。

Stage2 的 `-HostOnly` 只复用完整清单中身份一致的 CUDA 对象，并重建主机部分；架构、后端、工具链、GMP 头、CUDA 依赖或任一 CUDA 对象改变时拒绝。普通复用和发布 `-SkipBuild` 同时核对程序、编译单元、依赖和对象；旧 manifest 缺少身份记录时须正常重建。保留构建目录中的 `.ecm.d`、`.ecm-state.json`、源清单和 manifest；不能修改清单或替换 DLL 来绕过核对。

## CUDA 源清单与依赖身份

翻译单元唯一清单为 `tools/build/internal/cuda_sources.cmake`：分别维护 Stage1 共用源、CUDA 源、可选 PRAC、Stage2 主机源、生产引擎和开发 wrapper。CMake 直接使用，PowerShell 从该文件导出构建目录内的 `cuda_sources.json`。生产源码闭包拒绝包含 `tools/bench/`、`tools/experiments/` 和 `.refactor/`；开发 wrapper 使用单独源组。当前算法文件位置和 PRAC 显式开关保持。

CMake 同时导出 `cuda_target_graph.json`，收集目标及其链接的项目对象库/静态库；条件目标按保守全集收集，导入库由工具链与依赖清单另行记录。对象枚举和 CUDA 完全重编按该目标闭包执行。`cuda_build_state.ps1` 扫描本地递归包含，包括 `.inc/.inl/.def`，并用 NVCC 依赖文件补齐宏包含及 SDK 头。每个对象的 `.ecm-state.json` 记录命令、依赖内容指纹与对象摘要；摘要编码在 Windows PowerShell 5.1 和 PowerShell 7 间一致。

P0 独立验证入口为 `tools/build/test/build_cuda_p0.ps1`，用 `-Stage stage1/stage2/development` 选择构建；CPU 故障注入为 `tools/test/test_cuda_build_identity.ps1`，短曲线双版本对照为 `tools/test/test_cuda_p0_runtime.py`。这些入口不替代完整位宽、多架构或性能验收，证据写入 `data/experiments/`。

`tools/build/dev/` 为开发和 GUI 入口，`test/` 为探针与参考实现构建，`internal/` 为配置检查、并行编译、拆分和打包辅助。详细参数入口见 [构建工具](../../tools/build/README.md)；配置生成合同见 [配置维护](../architecture/CONFIGURATION.md)。
