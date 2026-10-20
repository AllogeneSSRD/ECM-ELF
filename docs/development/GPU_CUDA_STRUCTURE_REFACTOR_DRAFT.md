# CUDA 结构重构草稿

## 范围与约束

按已确认方案分 P0～P5 实施。本轮只完成 P0：构建源清单、依赖与身份校验；不进入位宽清单或设备模块迁移。算法、配置、存档及程序输出语义保持，软件修订号由 0.44.0 增至 0.44.1。

基线为 `36171f4`（0.44.0），当前 TOML、队列及输出行为作为基线。478 个已跟踪 `src/`、`kernels/`、`include/` 文件的原始摘要保存在 `data/experiments/cuda_structure_p0_20261020/baseline.json`。除版本头外，不修改这些文件。

## P0 实现

- `tools/build/internal/cuda_sources.cmake` 唯一维护 Stage1 共用源、CUDA 源、可选 PRAC、Stage2 主机源及生产/开发引擎源，保持原编译和链接次序。CMake 直接使用，PowerShell 每次导出到构建目录中的 `cuda_sources.json`，不维护第二份手抄清单。
- CMake 导出真实目标依赖图；对象收集、完全重编与并行编译可覆盖链接的对象/静态模块。未变化的算法编译宏和宿主身份 sidecar 保持。
- `cuda_build_state.ps1` 扫描当前本地递归包含关系，并用 NVCC 编译依赖文件补足宏包含及 SDK 依赖。条件包含按保守全集处理；生产闭包拒绝实验源码。响应文件、命令、依赖内容、对象摘要共同决定复用，不依赖 mtime。
- Stage1 发布身份改为所选目标源闭包，包含 `.inc/.inl/.def` 和链接主机模块的私有包含文件，并冻结 CMake 编译命令与目标图；PRAC 源随显式开关加入，保留既有构建和发布能力。
- Stage2 本机/开发/发布消费相同源清单；HostOnly 校验完整 CUDA 源集、每个编译单元及对象；普通复用与 SkipBuild 同时核对 exe、源成员和对象，不能遗漏新依赖或通过替换对象绕过。
- 保持现有无重名对象的文件名；未来同名翻译单元使用区分目录的对象名。旧 manifest 缺少依赖身份时要求正常重建一次。

## 验证记录

P0 分为源清单与 CPU 故障注入、原生构建、复用/发布校验及短曲线回归四步，前一步通过后进入下一步。所有新证据写入上述实验目录；失败输出保留。

- `identity_01`：测试辅助函数名碰到 PowerShell 的 `Write` 别名，未实际写入夹具，失败保留；已改用独立函数名。
- `identity_02`：暴露 PowerShell 5.1 对 JSON 数组输出的包裹差异，目标闭包筛选失败；修正数组读取后，`identity_03` 的 28 项 CPU 故障注入门禁通过。
- 指纹格式改为 schema 2：UTF-8 长度分帧、路径按 ordinal 排序，避免 Windows PowerShell 5.1 与 PowerShell 7 的 JSON 转义/排序差异。`identity_cross_shell_v2.json` 与 `identity_10/result.json` 核对跨运行时一致性。
- `identity_10`：37 项 CPU 门禁通过，覆盖递归 `.inc/.inl/.def`、保留 mtime 的内容变化、响应文件、宏包含、带空格和 `#` 的依赖路径、被移除的旧编译器依赖、源成员遗漏、命令/对象篡改、多 CUDA 编译单元、生产包含实验源码的拒绝，以及链接主机模块的私有依赖。夹具使用独立目录，不改仓库算法源码。
- `inventory_parity.json`：Stage1 共用、CUDA、PRAC 及 Stage2 源清单与 0.44.0 的成员和次序完全相同。`target_graph/build_02/cuda_target_graph.json` 验证 CMake 对对象库、条件链接和传递静态库的保守收集。
- 原生工具链为 CUDA 13.3、MSVC 14.51.36231，架构 sm89。Stage1 本机/发布验证限制为 `Tiers=384`、PRAC 关闭；未声称完整位宽、旧 CUDA 或全部发布架构通过。`stage1_build_v2.log`、`stage2_build_v2.log`、`development_build_02.log` 分别记录 Stage1、生产 Stage2、开发 wrapper 构建通过。
- `stage1_incremental_02.log`：21 个 CUDA 对象全部复用，实际重编 0 个；`stage2_hostonly_01.log`：校验后复用 CUDA 对象，重编 4 个主机文件；`stage2_reuse_01.log`：源、依赖、对象和 exe 相同，完整复用。`phase_deps_04/result.json`：改写临时 manifest 的 phase 头摘要后，HostOnly 在编译前拒绝，原构建目录未变。
- `runtime_04/summary.json`：冻结的 0.44.0 与最终本机构建在 GPU 1（RTX 4060 Laptop）对照，Stage1 5 组、每组 4 条曲线（Montgomery PARAM0/2/3、Edwards binary/NAF）存档字段一致；Stage2 5 组（Montgomery 普通、Edwards 普通/承载/relocation/主机回退）目标环叶指纹和非计时结果字段一致。输入、完整参数与二进制 SHA256 随各用例保存。只排除版本、源构建身份、时间及相应校验字段；没有性能采样或性能结论。
- `edwards_probe_01.log`：独立 GMP 点运算参考的 3 组用例通过，每组 33 条曲线，包含续算、自适应重启；`relocation_geometry_01.log`：157752 项检查，bad=0。
- `source_freeze_01.json`：478 个已跟踪算法/接口文件中只改版本头，另外 477 个摘要与基线相同。开发 wrapper 额外补上已使用的日志声明头；`development_baseline_02.log` 用原编译参数复现缺声明错误，未改设备算法。
- `release_build_02.log`、`release_skip_split_package_01.log`：通用 x64 GMP 的 Stage1 sm89/384 档发布构建、SkipBuild、21 个 CUDA 对象拆分、对象/exe 还原和隔离 ZIP 打包通过；`release_identity_02.json` 确认拆分还原后身份仍成立，并拒绝对象摘要错误、源摘要遗漏和 exe 摘要错误。首轮手写真实构建故障注入夹具的属性索引错误，说明保留在 `release_identity_fixture_01_failure.txt`，修正后才记为通过。
- `stage2_release_package_01.log`：通用 x64 GMP 的生产 Stage2 sm89 发布构建和隔离 ZIP 打包通过。`stage2_release_skip_01.json`：SkipBuild 身份通过，已有配置与队列保持原摘要，ZIP 使用默认模板且不含内部 manifest。

## 验证边界与保留的失败证据

沙箱内的 MSVC 自检因 PDB 环境报 C1902（`stage1_build_01.log`），原生工具链构建通过。`phase_deps_01`～`03` 是辅助进程的执行策略、模块路径或运行时定位问题，`phase_deps_04` 使用已定位运行时通过；未放宽身份条件。

`runtime_01` 的 PARAM2/3 以 M127 输入运行受限 384 档，触发当前实现精确档位选择后找不到 192 档的限制；改用需要 384 容器的 256 位输入进行双版本对照。`runtime_02` 的原始叶指纹一致，结果比较器把嵌套计时当成稳定字段，修正比较边界后 `runtime_03`、`04` 通过。本轮未修改上述档位选择实现。所有失败输出继续保留。

## 后续

P0 已通过上述限定范围的分步验证。P1～P5 尚未实施。保留本草稿并提示用户处理，不自动删除。正式构建说明只记录已验证的当前行为；原始源码快照、失败夹具和日志不更改。
