# 构建工具

用户流程、默认策略、工具链和分发内容以 [构建与发布](../../docs/usage/BUILD.md) 为准。

## 一键入口

| 程序 | 开发 | 本机生产 | 发布生产 |
| --- | --- | --- | --- |
| Stage1 | `build_stage1_dev.ps1/.bat` | `build_stage1_local.ps1/.bat` | `build_stage1_release.ps1/.bat` |
| Stage2 | `build_stage2_dev.ps1/.bat` | `build_stage2_local.ps1/.bat` | `build_stage2_release.ps1/.bat` |

完整发布使用 `build_release.ps1/.bat`，默认生成 cu11.8 与 cu13.3 完整包；表中的 release 入口只构建/验证相应阶段。Stage1 默认 `Jobs=8`；Stage2 local 默认 `SplitCompile=8`，release 默认 `Jobs=8, SplitCompile=1`。PS1 为默认参数来源，BAT 原样转交参数。无参数 BAT 结束暂停，有参数返回退出码。入口不自动运行测试。

## 参数定位

- Stage1 本机/开发：`-Arch`、`-BuildDir`、`-Jobs`、`-Tiers`、`-EnablePrac`、`-HostOnly`、`-FullRebuild`、`-Extra`。默认验证并复用 CUDA 对象，`-Incremental` 保留兼容；默认不编译 PRAC，改变构建参数后仅重编失效单元。
- Stage2 本机/开发：`-Arch`、`-Build`、`-SplitCompile`、`-GlBackend`、`-AddSubMask`、`-OuterUnrollU`、`-HostOnly`、`-Rebuild`。默认逐单元复用；HostOnly 在编译前验证 CUDA 对象/参数和源身份，与 Rebuild 互斥。
- 发布：`-Profile all/cu11.8/cu13.3`、`-BuildRoot`、`-OutDir`、`-Jobs`；固定 PTX50/SASS60 与 PTX75/SASS80。完整入口还支持 `-Component all/stage1/stage2`；阶段入口是相应组件的包装。
- 发布工具链：`-Cuda118Root`、`-Cuda133Root`、`-VcVars`、`-Gmp`；发布要求同一 vcpkg 通用 GMP，缺失或身份变化即失败。local/dev 保留 `-CudaRoot` 和各自依赖参数。
- 复用/打包：`-SkipBuild` 仍核对冻结身份；`-SkipPackage` 构建组件后停止。已有版本目录不覆盖，重打包指定新的 `-OutDir`。

local BAT/PowerShell 默认输出为 `build_stage1_local/ecm_cuda.exe`、`build_stage2_local/ecm_cuda_stage2.exe`，GUI 为 `build_gui_local/ecm_gui.exe`。dev/release 仍按用途/profile 隔离在 `build_cuda_dev/release`；显式本机/开发目录是最终叶目录。固定短目录中的 `profile.json` 仍拒绝跨用途、跨阶段或不兼容配置复用；需要并存时指定不同 `-BuildDir`/`-Build`，不自动覆盖。旧缓存仅经正式清单核对后原位接纳。Stage1 的 `RelinkOnly` 绕过入口已关闭，主机修改使用正常增量或严格 `HostOnly`。目录字段和发布参数的根/叶语义见 [用途与 profile 目录](../../docs/usage/BUILD.md#用途与-profile-目录)。

完整参数以各 PS1 的 param 与帮助为准。发布只附 `ECM_TOML_REFERENCE.md` 这一份项目说明；源码/构建/拆分/打包清单保留在构建目录，ZIP 旁保留 SHA256。

## 子目录

- `dev/`：GUI、CMake辅助、独立树实验驱动及显式实验构建。
- `test/`：NTT/点运算探针、CPU树参考和数学门禁构建。
- `internal/`：CUDA 权威源清单与依赖身份、并行 nvcc、配置一致性、工具链选择、Stage1 拆分及两阶段打包。

CUDA 源组在 `internal/cuda_sources.cmake` 维护，构建目录中的 JSON 由它生成；不要在脚本中另抄源文件列表。递归包含、编译命令和对象身份由 `internal/cuda_build_state.ps1` 校验；复用规则和测试入口见 [CUDA 源清单与依赖身份](../../docs/usage/BUILD.md#cuda-源清单与依赖身份)。

Stage1 的位宽与实例分组由 `src/ecm/backend/cuda/dispatch/tiers.def` 统一维护；主机驱动和设备实例职责见 [CUDA 模块边界](../../docs/architecture/STAGE1.md#cuda-模块边界)。`test/build_cuda_p1.ps1` 提供清单配置、Stage1 和 Stage2 的分步构建入口。

Stage2 算术和 NTT 的职责见 [Stage2 模块边界](../../docs/architecture/STAGE2.md#cuda-模块边界)。`test/build_cuda_p2_arithmetic_probe.ps1` 与 `test/build_ntt_graph_probe.ps1` 可复用通过身份校验的生产对象；使用生产 NTT 的其他探针统一通过 `internal/cuda_ntt_probe.ps1` 编译显式源清单，并保留各自的默认流模式。

本机完整位宽验收入口为 `tools/test/test_cuda_p5_native.py`。双 CUDA 发布使用 `tools/test/test_release_build_contracts.py`、`tools/test/test_release_runtime.py`；后者显式指定基线、CUPTI观察器、设备及证据目录。`test/build_cuda_jit_trace.ps1` 构建观察器，不随发布分发。原 `test_cuda_p5_release.py` 和 `test_cuda_p5_images.py` 保留六架构阶段的冻结检查，不能替代当前矩阵。所有测试和性能入口须显式运行，实际覆盖以对应证据为准。

`test/build_release_probes.ps1` 将独立 GMP 点/算术参考链接到经过身份核对的发布对象；`test/build_release_ptx_fixture.ps1` 从冻结对象生成 PTX 字节相同的独立测试副本。`tools/test/test_release_switching.py` 验证同进程切换与重复任务，`test_release_bundle.py` 验证完整包、搬移目录和精简 PATH。方法、限制与证据见[发布/JIT验证](../../docs/performance/RELEASE_JIT.md)。

`dev/build_stage2_tree_gpu.ps1` 的原型驱动与根目录 Stage2 生产/开发队列程序不同。实际生产算术和当前限制见 [NTT](../../docs/architecture/NTT.md)、[Stage2使用](../../docs/usage/STAGE2.md)。

独立 CPU Edwards Stage2 使用 `test/build_cpu_stage2.ps1`；`-GwPoly` 启用 GWNUM/polymult 多线程快速多项式后端，无此开关保留 libecm 构建。依赖、运行参数和输出见 [CPU Stage2](../../docs/usage/EDWARDS_CPU_STAGE2.md)，当前不随 CUDA 发布入口自动打包。

OpenCL Stage1 的本机入口为 build_opencl_local.ps1；使用隔离的 build_opencl_local/stage1/&lt;key&gt; 叶目录、普通 CXX 及随附生产资源，不启用 CUDA。支持 -BuildDir、-Gmp、-OpenSslRoot、-OpenClRoot、-VcVars；当前只提供 local，不附带 release 打包。资源与加载合同见 [OpenCL](../../docs/architecture/OPENCL.md#资源与-local-构建)。

GUI 统一使用 `build_gui.bat`，不区分 dev/local/release；它把参数和退出码交给现有 `dev/build_gui.ps1`，后者仍是默认参数来源。无参数双击构建完成后暂停；有参数调用直接返回。默认输出 `build_gui_local/`，只构建 `ecm_gui`，可传 `-Selftest` 构建后自检。测试 helper 在 `tools/test/gui/`，必须显式指定 `-Targets`，不会进入默认构建；详见 [GUI 构建与验证](../../docs/usage/GUI.md#构建与独立验证)。

GUI v2 独立入口 `build_gui_v2.bat` 默认生成 `build_gui_v2_local/ecm_gui_v2.exe`，默认使用DX11，`--renderer vulkan`保留诊断路径，不修改原GUI。`package_gui_v2.ps1 -BuildDir build_gui_v2_local -Output <新目录>`创建GUI-only候选包及校验清单，不附计算worker。要求构建输出的源目录身份清单与exe哈希相符。当前运行条件和待验门禁见 [GUI v2](../../docs/usage/GUI_V2.md)。

共享 PARI 诊断的构建入口为 `test/build_diagnostics.ps1`；正式 Stage1/Stage2 和 GUI v2 自动附带同一助手目标及完整依赖，默认 DLL → GP，GMP 仅显式选择。构建前后的源码/依赖冻结及分发要求见[共享 PARI 诊断依赖](../../docs/usage/BUILD.md#共享-pari-诊断依赖)。
