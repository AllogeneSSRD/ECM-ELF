# CUDA 构建工具

用户流程、默认策略、工具链和分发内容以 [构建与发布](../../docs/usage/BUILD.md) 为准。

## 一键入口

| 程序 | 开发 | 本机生产 | 发布生产 |
| --- | --- | --- | --- |
| Stage1 | `build_stage1_dev.ps1/.bat` | `build_stage1_local.ps1/.bat` | `build_stage1_release.ps1/.bat` |
| Stage2 | `build_stage2_dev.ps1/.bat` | `build_stage2_local.ps1/.bat` | `build_stage2_release.ps1/.bat` |

Stage1 默认 `Jobs=8`，Stage2 默认 `SplitCompile=8`。PS1 为默认参数来源，BAT 原样转交参数。无参数 BAT 结束暂停，有参数返回退出码。入口不自动运行测试。

## 参数定位

- Stage1 本机/开发：`-Arch`、`-BuildDir`、`-Jobs`、`-Tiers`、`-EnablePrac`、`-FullRebuild`、`-Extra`。默认验证并复用 CUDA 对象，`-Incremental` 保留兼容；默认不编译 PRAC，改变构建参数后仅重编失效单元。
- Stage2 本机/开发：`-Arch`、`-Build`、`-SplitCompile`、`-GlBackend`、`-AddSubMask`、`-OuterUnrollU`、`-HostOnly`、`-Rebuild`。默认逐单元复用；HostOnly 在编译前验证 CUDA 对象/参数和源身份，与 Rebuild 互斥。
- 发布：`-Archs`，默认60,70,75,86,89,120；Stage2 的显式 `-Arch` 与 `-Archs` 互斥。Stage1 另有 `-PtxArch`。
- 工具链：`-CudaRoot`、`-LegacyCudaRoot`、`-VcVars`、`-Gmp`；Stage1 另有 `-OpenSslRoot`。60/70 用CUDA12.6，其余默认CUDA13.3。
- 缓存/拆分：`-SkipBuild` 要求冻结身份一致；Stage1 的 `-SkipSplit`/`-SkipPackage` 控制发布步骤。

完整参数以各 PS1 的 param 与帮助为准。发布只附 `ECM_TOML_REFERENCE.md` 这一份项目说明；源码/构建/拆分/打包清单保留在构建目录，ZIP 旁保留 SHA256。

## 子目录

- `dev/`：GUI、CMake辅助、独立树实验驱动及显式实验构建。
- `test/`：NTT/点运算探针、CPU树参考和数学门禁构建。
- `internal/`：CUDA 权威源清单与依赖身份、并行 nvcc、配置一致性、工具链选择、Stage1 拆分及两阶段打包。

CUDA 源组在 `internal/cuda_sources.cmake` 维护，构建目录中的 JSON 由它生成；不要在脚本中另抄源文件列表。递归包含、编译命令和对象身份由 `internal/cuda_build_state.ps1` 校验；复用规则和测试入口见 [CUDA 源清单与依赖身份](../../docs/usage/BUILD.md#cuda-源清单与依赖身份)。

Stage1 的位宽与实例分组由 `include/ecm/cuda/tiers.def` 统一维护；主机驱动和设备实例职责见 [CUDA 模块边界](../../docs/architecture/STAGE1.md#cuda-模块边界)。`test/build_cuda_p1.ps1` 提供清单配置、Stage1 和 Stage2 的分步构建入口。

Stage2 算术和 NTT 的职责见 [Stage2 模块边界](../../docs/architecture/STAGE2.md#cuda-模块边界)。`test/build_cuda_p2_arithmetic_probe.ps1` 与 `test/build_ntt_graph_probe.ps1` 可复用通过身份校验的生产对象；使用生产 NTT 的其他探针统一通过 `internal/cuda_ntt_probe.ps1` 编译显式源清单，并保留各自的默认流模式。

本机完整位宽验收入口为 `tools/test/test_cuda_p5_native.py`；另备发布静态检查 `test_cuda_p5_release.py` 和发布镜像运行 `test_cuda_p5_images.py`。本轮验收限定本机LOCAL，多架构发布暂缓；实际覆盖及前置条件见 [CUDA 模块验收](../../docs/performance/CUDA_STRUCTURE_QUALIFICATION.md)。`tools/bench/prepare_cuda_refactor_qualification.py --inputs-only` 只准备输入，生成的计划不能进入采样；`qualify_cuda_refactor.py` 要求就绪计划与通过的前置门禁，使用冻结程序进行配对性能验收。所有测试和性能入口均须显式运行，构建器不自动调用。

`dev/build_stage2_tree_gpu.ps1` 的原型驱动与根目录 Stage2 生产/开发队列程序不同。实际生产算术和当前限制见 [NTT](../../docs/architecture/NTT.md)、[Stage2使用](../../docs/usage/STAGE2.md)。

独立 CPU Edwards Stage2 使用 `test/build_cpu_stage2.ps1`；`-GwPoly` 启用 GWNUM/polymult 多线程快速多项式后端，无此开关保留 libecm 构建。依赖、运行参数和输出见 [CPU Stage2](../../docs/usage/EDWARDS_CPU_STAGE2.md)，当前不随 CUDA 发布入口自动打包。
