# 构建输出目录整理草稿

当前使用规则见[构建说明](../usage/BUILD.md#用途与-profile-目录)。此草稿保留实施和验证信息，完成后由用户决定是否清理。

## 实施

- Stage2 默认 local 入口固定使用 `build_stage2_local`；有效同仓库旧 profile 整目录归档到 `build_dev/stage2_local_profiles/`。损坏、跨阶段、跨用途、显式目录及 HostOnly 的身份检查保留。
- 现有指定的 `build_stage2_local_c1cd7a763fb2ca5e` 通过 profile 和 EXE 摘要验证后提升到固定目录，旧固定目录保留于归档。
- 根目录迁移执行138项移动：其中136项测试／实验目录，另两项为旧本机归档和指定产物提升。逐项完成记录位于 `build_dev/migrations/20261101T000345Z_7e6a630f992242788051521a66a03a8b.json`。不改写缓存中的绝对路径，不删除原始测量数据。
- 63份 PowerShell 构建／测试入口与22个 Python 工具，合计94处默认路径改为 `build_dev/` 下；正式 worker 输入路径及显式用户路径保持独立。

## 验证

- `test_cuda_build_profile.ps1`：修改前复现后缀选择；修改后54项 CPU-only 检查通过，包含固定路径、归档保留字节、只读拒绝、损坏标记拒绝和显式路径限制。
- `test_build_migration.ps1`：真实临时 Git 仓库验证计划模式、实际移动、指定版本提升、旧程序保留、含空格／方括号目录、已跟踪源码保留及路径越界拒绝。
- PowerShell 5.1：54项 profile 检查、迁移夹具及19项构建目录／忽略检查通过；实际参数 AST 审查确认测试默认路径均位于正式目录或 `build_dev`。Python 工具的直接 argparse 默认路径与语法检查通过。
- 迁移后138项清单全部完成，根目录保留12个正式或源码构建目录；指定 EXE 与原 build manifest 摘要一致，`--version` 返回0.56.1。产物与迁移清单均被 Git 忽略。未运行完整 Stage2 性能矩阵。
