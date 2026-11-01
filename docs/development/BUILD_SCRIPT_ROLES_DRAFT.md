# 构建脚本用途与 Git 跟踪整理

状态：已按用户确认完成；草稿保留待用户处理。正式目录说明见[构建工具](../../tools/build/README.md#子目录)。不修改主程序版本。

2026-11-01，用户确认将开发相关脚本迁到 `tools/build/dev` 并忽略，`test` 保留诊断等依赖且不忽略。此授权针对本机实验脚本，不改变构建输出目录内部 `.gitignore` 为 `*` 的规则。

原 GUI 与 GUI v2 的正式 PowerShell 实现迁到 `internal/`，BAT、GUI 主机测试、增量边界验证和当前文档同步路径。GUI v2 仍委托受跟踪的 `test/build_gui_v2.ps1`，不改变配置生成、源身份验证、默认参数、目标和返回码。

22 个纯实验/基准入口迁到本机 `dev/`：15 个 `build_ntt_*`、cuFFT、冻结 pre-NAF 基准、Stage2 D 搜索实验、实验适配器、独立 Stage2 GPU 探针、CPU 树参考构建与树原型验收。连同原有 basic256、fold256、通用 CMake 开发助手和 GPU 树原型构建一起取消跟踪，全部保留本机文件。`tools/build/.gitignore` 持久排除 `/dev/`。实验脚本自引用的依赖摘要路径和现有实验调用说明同步；新克隆不附带这些入口。冻结草稿及原始证据不重写。

`test/` 保留数学正确性、构建身份、发布资格、checkpoint/主机合同测试及共享 PARI、GUI v2、OpenCL 构建依赖。Stage1 等待探针继续跟踪，现有主机拆分回归可直接使用。正式入口和共享构建助手不依赖忽略的 `dev/`。

验证：120 个 PowerShell 构建脚本语法及字面兄弟脚本依赖检查通过；隔离夹具运行实际 GUI v2 BAT 和迁移后的包装，验证含空格参数与退出码 17 的透传。Stage1 缓存回归 2 项、GUI v2 构建前配置同步回归 3 项通过。验证未编译 CUDA 单元。迁移前字节备份、明确路径清单及验证结果位于忽略的 `data/experiments/build_script_roles_20261101/`。
