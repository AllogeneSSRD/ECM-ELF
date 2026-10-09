# 构建与发布

工具位于 `tools/build/`。从任意目录调用 PS1，或双击同名 BAT；BAT 转交参数，无参数运行结束后暂停。默认值由 PS1 维护。

## 六组入口

| 程序 | 开发 | 本机生产 | 发布生产 |
| --- | --- | --- | --- |
| Stage1 | `build_stage1_dev` | `build_stage1_local` | `build_stage1_release` |
| Stage2 | `build_stage2_dev` | `build_stage2_local` | `build_stage2_release` |

每项均有 `.ps1` 与 `.bat`。Stage1 默认 8 路独立 CUDA TU 并行；Stage2 默认 `SplitCompile=8`，多架构依次构建，每个架构内并行。构建入口不自动运行测试。

本机/开发默认 sm89。Stage1 `-Arch 86`、Stage2 `-Arch sm_86` 可选择其他架构；本机表示单架构构建，不自动检测设备。Stage1 默认输出 `build_cuda_cmake/ecm_cuda.exe`，Stage2 默认输出 `build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe`。开发分别使用独立目录。

## 默认策略

Stage1：完整位宽/曲线族、通用 Montgomery、TPB128、分档寄存器策略；resident/PRAC 默认不编译，显式 `-EnablePrac` 才加入。`-Tiers` 可缩小开发档位。折叠域、统一寄存器上限和算术探针不是通用发布默认。

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

包内只分发程序、GMP DLL、共享 INI 模板、空队列、`ECM_INI_REFERENCE.md` 与许可证。不包含 `*manifest.json`、`DEV_*`、README 或其他项目文档。重新打包保留已有 INI/队列；ZIP 从干净暂存目录生成，不收录 save、日志和运行数据。

Stage1 输出 `dist/cuda/`，Stage2 多架构输出 `dist/cuda-stage2/`；每架构有目录、ZIP 和 SHA256。构建/拆分/打包清单留在构建目录，包括源码、依赖、编译器、架构和程序身份。

## 复用构建

Stage1 的 `-Incremental` 依赖时间戳，改变选项后应正常重建。Stage2 的 `-HostOnly`、发布的 `-SkipBuild` 要求源码/依赖/编译配置与冻结清单相符；不能修改清单或单独替换 DLL 来绕过核对。

`tools/build/dev/` 为开发和 GUI 入口，`test/` 为探针与参考实现构建，`internal/` 为配置检查、并行编译、拆分和打包辅助。详细参数入口见 [构建工具](../../tools/build/README.md)；配置生成合同见 [配置维护](../architecture/CONFIGURATION.md)。
