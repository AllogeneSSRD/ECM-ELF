# CUDA 构建工具

用户流程、默认策略、工具链和分发内容以 [构建与发布](../../docs/usage/BUILD.md) 为准。

## 一键入口

| 程序 | 开发 | 本机生产 | 发布生产 |
| --- | --- | --- | --- |
| Stage1 | `build_stage1_dev.ps1/.bat` | `build_stage1_local.ps1/.bat` | `build_stage1_release.ps1/.bat` |
| Stage2 | `build_stage2_dev.ps1/.bat` | `build_stage2_local.ps1/.bat` | `build_stage2_release.ps1/.bat` |

Stage1 默认 `Jobs=8`，Stage2 默认 `SplitCompile=8`。PS1 为默认参数来源，BAT 原样转交参数。无参数 BAT 结束暂停，有参数返回退出码。入口不自动运行测试。

## 参数定位

- Stage1 本机/开发：`-Arch`、`-BuildDir`、`-Jobs`、`-Tiers`、`-EnablePrac`、`-Incremental`、`-Extra`。默认不编译 PRAC；改变构建参数后正常重建。
- Stage2 本机/开发：`-Arch`、`-Build`、`-SplitCompile`、`-GlBackend`、`-AddSubMask`、`-OuterUnrollU`、`-HostOnly`。HostOnly 只在 CUDA 对象/参数和源身份匹配时复用。
- 发布：`-Archs`，默认60,70,75,86,89,120；Stage2 的显式 `-Arch` 与 `-Archs` 互斥。Stage1 另有 `-PtxArch`。
- 工具链：`-CudaRoot`、`-LegacyCudaRoot`、`-VcVars`、`-Gmp`；Stage1 另有 `-OpenSslRoot`。60/70 用CUDA12.6，其余默认CUDA13.3。
- 缓存/拆分：`-SkipBuild` 要求冻结身份一致；Stage1 的 `-SkipSplit`/`-SkipPackage` 控制发布步骤。

完整参数以各 PS1 的 param 与帮助为准。发布只附 `ECM_INI_REFERENCE.md` 这一份项目说明；源码/构建/拆分/打包清单保留在构建目录，ZIP 旁保留 SHA256。

## 子目录

- `dev/`：GUI、CMake辅助、独立树实验驱动及显式实验构建。
- `test/`：NTT/点运算探针、CPU树参考和数学门禁构建。
- `internal/`：并行 nvcc、配置一致性、工具链选择、Stage1 拆分及两阶段打包。

`dev/build_stage2_tree_gpu.ps1` 的原型驱动与根目录 Stage2 生产/开发队列程序不同。实际生产算术和当前限制见 [NTT](../../docs/architecture/NTT.md)、[Stage2使用](../../docs/usage/STAGE2.md)。
