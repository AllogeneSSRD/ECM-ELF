# ECM-ELF

[English](README.en.md) | 中文

椭圆曲线因子分解工具：CUDA/CGBN 与 OpenCL GPU Stage1、GMP/可选 AVX-512 IFMA CPU Stage1、独立 CUDA 多项式 Stage2，以及 Windows GUI、队列和因子数据集工具。

## 程序

- `ecm_cuda.exe`：NVIDIA GPU Stage1，支持受构建覆盖的曲线族、批处理和检查点。
- `ecm.exe`：CPU Montgomery/Edwards 与 OpenCL 构建入口。
- `ecm_cuda_stage2.exe`：读取归一化 PARAM0 Stage1 save，执行树/NTT Stage2，支持独立队列逐曲线续跑。
- `ecm_gui.exe`：配置、Stage1 worker、队列生成、日志、GPU 状态和结果展示。
- `ecm_p95feeder`：CPU 本地存档的 Prime95 交接工具；驱动也提供完成任务交接配置。

程序的 save/checkpoint 格式及参数化支持以对应入口为准；不是所有外部保存点均可互用。Stage2 手动输入支持2…16384 bits，Auto B2 仅接受匹配且通过审计的成本 profile；当前生产组合需显式 B2。

## 开始使用

1. 选择与 GPU 架构和 CPU 依赖匹配的发布包，或使用 `tools/build/` 一键编译。
2. 编辑共用 `ecm.ini`，Stage1 使用 `worktodo.txt`，Stage2 使用独立 `stage2_worktodo.txt`。
3. Stage1 生成完成的 save；Stage2 直接读档或消费其任务队列。两程序目前不自动合并为总流程。
4. 使用程序帮助和 [INI 参考](docs/ECM_INI_REFERENCE.md) 核对参数、默认值和路径。

构建入口分 Stage1/Stage2 × 开发/本机生产/发布生产，均有 PS1/BAT。发布默认架构60、70、75、86、89、120，分别使用支持它们的 CUDA12.6/13.3，自动打包；运行资格仍需实际设备验证。

## 文档与工具

- [文档目录](docs/README.md)：使用、架构、当前性能依据和 TODO。
- [构建与发布](docs/usage/BUILD.md)、[构建参数入口](tools/build/README.md)。
- [Stage1 使用](docs/usage/STAGE1.md)、[Stage2 使用](docs/usage/STAGE2.md)、[GUI](docs/usage/GUI.md)。
- [工具目录](tools/README.md)、[数据集](tools/ecm_dataset/README.md)、[Android](Android/ECM/README_ECM_FACTORIZATION.md)。

实验、统计和生成图表位于 Git 忽略的 `data/`；外部源码位于 `.refactor/`，论文与译述保留在 `docs/paper/`。项目状态文档记录当前实现；数学与第三方分析见[固定资料](docs/reference/README.md)。

## 依赖与许可

基于 GMP-ECM 框架，CUDA 多精度算术使用 CGBN；GUI 使用 Dear ImGui。具体构建需要对应 CUDA/OpenCL、C++ 工具链和 GMP。发布默认通用 x64 GMP，本地 Zen3 库不代表所有 CPU 兼容。

项目许可见 [LICENSE](LICENSE)，分发依赖需保留各自版权与许可。
