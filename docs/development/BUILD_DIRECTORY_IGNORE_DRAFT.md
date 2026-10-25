# 构建目录内部忽略规则实施草稿

状态：已完成实现与下述验证；保留草稿，待用户决定后续处理。正式使用说明见 [构建与发布](../usage/BUILD.md#用途与-profile-目录)。此项只调整构建工具，不改变主程序版本。

## 要求与范围

创建或复用构建输出目录时，在目录内部生成一行 `*` 的 `.gitignore`，在写入构建产物前完成；标记自身也忽略。覆盖 dev、local、release、OpenCL、GUI 和测试构建入口。不向根 `.gitignore` 添加构建目录规则，不强制跟踪标记，不改变分发包载荷。

## 实现

- `tools/build/internal/build_directory.ps1` 提供 `Initialize-EcmBuildDirectory`：创建目录、补齐规则、保持正确标记时间戳，同时初始化仓库内嵌套的 `build` / `build_*` 容器。拒绝仓库根、其父目录、文件系统根及受保护的源码目录。
- 正式 CUDA/OpenCL 构建通过 profile 入口初始化；已有正确 profile 的复用路径也补齐规则。只有忽略文件的目录可进行首次构建；未知产物、profile 不匹配和 SkipBuild 缺少 profile 时继续拒绝。
- dev、独立测试、共享探针、计时及内部并行构建入口接入初始化。已有直接写入相同规则的测试入口保留其实现。冻结 Edwards 基线构建在原有新目录检查后初始化，不修改冻结源码；PTX 派生测试在 Python 创建新目录后、写入测试产物前生成标记。
- 新共享助手纳入正式构建输入清单及共享探针依赖身份。所有新目录限制均在初始化之前执行；没有删除原有身份检查或修改编译参数。
- 验证时发现已有 Stage2 身份测试仍期待17个主机模块，但当前清单已包含 `host_wait.cpp` 共18个；同步修正测试和正式文档的成员数，没有修改生产源组。

## 验证与证据

2026-10-25，Windows PowerShell 5.1。证据目录为 `data/experiments/build_ignore_policy_20261025/`，属于 Git 忽略目录。

- `tools/test/test_build_directory.ps1`：19项通过。覆盖新建、复用、缺失修复、部分规则替换、标记时间戳、既有产物保留、带空格/方括号路径、真实 Git 自身及产物忽略、嵌套容器、源码目录保护，以及 profile/SkipBuild 拒绝条件。结果见 `policy_01/result.json`；最终代码复核结果另见 `policy_final/result.json`。
- `tools/test/test_cuda_build_profile.ps1`：33项隔离/身份检查通过，结果见 `profile_01/`。
- `tools/test/test_cuda_build_identity.ps1`：54项变异检查通过，结果见 `identity_02/`；首次执行在上述过期成员断言处失败，保留 `identity_01/`，没有改写原始样本。
- `python -m unittest tools.test.test_release_prebuild tools.test.test_release_build_contracts`：13项通过，覆盖配置同步、失败阻断、发布布局及生成边界等现有合同。
- `tools/build/test/build_pet_behavior.ps1`：实际 MSVC CPU 构建与行为测试通过；删除该构建目录的忽略文件后，复用同一入口再次构建成功并补齐规则。产物与日志位于 `pet_build/`。
- 所有 `tools/build/` PowerShell 文件通过 Windows PowerShell 语法检查；最终扫描数量和结果保存于 `validation.json`。

未进行完整 CUDA 重编译、GPU 运行或完整发布打包；本次验证限定于目录策略、现有构建身份合同及轻量原生测试入口。
