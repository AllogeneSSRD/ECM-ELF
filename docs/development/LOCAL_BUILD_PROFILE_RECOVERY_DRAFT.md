# Local 默认构建目录冲突修复草稿

状态：已完成目录选择修复、自动回归、两阶段真实构建及无改动复用验证。正式规则见[用途与 profile 目录](../usage/BUILD.md#用途与-profile-目录)。保留草稿，后续处理由用户决定；构建工具改动不增加主程序版本。

## 复现与原因

2026-10-26，无参数调用 `tools/build/build_stage1_local.ps1`，连续两次在 profile 校验处出现 `Build profile/role/engine differs`。证据位于 `data/experiments/local_profile_failure_20261026/repro_01.log`、`repro_02.log`。

旧 `build_stage1_local/profile.json` 的内容 SHA256 与记录一致。与实际入口生成的请求逐字段比较，唯一差异为 `option/configure` 中新增 `ECM_CUDA_ISOLATE_TIERS=ON`；旧 CMake 缓存为 OFF。CUDA13.3.73、MSVC14.51.36231、CMake4.3.2及依赖身份没有变化。校验正确阻止了把旧 grouped 对象当成 isolated 对象；默认入口仍固定指向旧目录，使正常无参数构建被阻断。

Stage2 的同类现场差异为 `option/host_compile` 新增 `src` 包含路径。其原 profile 完整，但与当前正式编译命令不同，同样不能直接改写 marker 复用。

## 修复与边界

- `Resolve-EcmBuildDirectory` 在 local 未指定最终目录或 BuildRoot 时优先保留短目录。短目录已有同源码根、同用途、同阶段且完整的另一份 profile 时，选择确定的同层 `build_stageN_local_<配置SHA256前16位>`。
- 不重写旧 profile，不删除、移动或重新标记旧对象。重跑相同配置选择同一目录，仍经正常完整身份验证；已有备用目录不匹配时拒绝，不再另选“最新”目录。
- 显式 `-BuildDir` / `-Build` 不改选，dev/release 与显式 BuildRoot 的目录语义保持。损坏标记、跨用途或阶段及不同源码根不能触发自动改选。未知未登记目录沿用原有接纳规则。
- 输出变化字段和实际目录；显式冲突错误也提供目录及差异字段。GUI、后续运行与测试须使用本次构建实际输出路径。

## 验证

- 回归先失败：`regression_red/` 在“默认 Stage1 配置变化应选择独立目录”处失败。
- `tools/test/test_cuda_build_profile.ps1`：46项通过，包含 Stage1布局切换、Stage2配置切换、确定性重跑、原目录内容保留、显式路径、损坏标记、外来用途及备用目录冲突拒绝，结果见 `regression_final/result.json`。
- `tools/test/test_build_directory.ps1`：19项通过，见 `ignore_policy/result.json`。
- `python -m unittest tools.test.test_release_build_contracts tools.test.test_release_prebuild`：13项通过。
- 修改的 PowerShell 脚本语法检查及差异空白检查通过。
- 实际入口显式指定旧 Stage1/Stage2 目录时仍拒绝不匹配配置，分别报告 `option/configure` 和 `option/host_compile`，见 `explicit_rejection_checks.json`。
- Stage2 正式入口 `-Jobs 2` 完整构建通过，输出 `build_stage2_local_c1cd7a763fb2ca5e/ecm_cuda_stage2.exe`；44个单元经内容身份验证全部复用，编译0个单元，最终 profile/程序摘要及 `--help` 检查通过。原 Stage2 目录的 profile、manifest及exe摘要保持不变。日志及汇总为 `stage2_build_verified.log`、`stage2_validation.json`。
- Stage2 首轮诊断在接纳后中止；首次完整编译44个单元并链接后，因构建过程中修改入口帮助注释，源码变更保护拒绝最终验收。失败日志 `stage2_build.log` 保留；随后在源码稳定时重跑，得到上述通过结果，没有改写身份记录绕过核对。
- 无参数 Stage1 正式入口完整构建通过：sm89、全档、isolated、默认8路，输出 `build_stage1_local_c869d1f184d3b595/ecm_cuda.exe`，128个cubin、无PTX；`--help` 通过。再次无参数运行仍选择同一目录，128/128个CUDA对象通过身份检查后复用，编译0个单元；EXE、对象映射及manifest摘要完全相同。原 Stage1 目录的 profile、manifest及exe摘要保持不变。日志为 `stage1_default_build.log`、`stage1_unchanged_reuse.log`，汇总为 `stage1_validation.json`。

本次验证构建与命令入口，不包含 GPU 数学、性能或发布包资格测试。现场两份旧缓存及所有失败日志均保留，没有通过改写旧 marker 或身份记录获得通过结果。
