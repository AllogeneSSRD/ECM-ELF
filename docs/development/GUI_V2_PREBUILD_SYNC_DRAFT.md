# GUI v2 独立构建前同步修复草稿

状态：已完成入口修复、回归、公开 BAT 完整构建与无变化复用验证。正式流程见[GUI v2 构建与启动](../usage/GUI_V2.md#构建与启动)。保留草稿，待用户决定后续处理；不修改主程序或 GUI 版本。

## 复现与根因

2026-10-31，用户从 `tools/build/` 运行 `build_gui_v2.bat`，在 CMake 的 `check_gui_v2_config.cmake` 处报 `Stale GUI v2 sidecar: config/ecm_options.json`。

直接调用同一 CMake 校验器连续两次得到相同错误；证据为 `data/experiments/gui_v2_prebuild_20261031/repro_1.log`、`repro_2.log`。业务配置生成器的 `--check` 通过，GUI 生成器的 `--check` 则发现 `generated/business_options.h` 已过期。release 总入口已有配置同步，独立 GUI v2 入口直接冻结源码并运行 CMake，没有同步步骤。

## 实现

- 在共用 `tools/build/test/build_gui_v2.ps1` 解析仓库根后，依次执行 `tools/gen/generate_ecm_config.py`、`generate_gui_v2_state.py`，报告当前步骤，检查退出码并在失败时停止。
- 同步发生在依赖解析及源码摘要冻结之前，覆盖公开 BAT、dev 包装、直接测试构建和 release 对共用入口的调用。使用绝对生成器路径，不依赖调用时工作目录。
- 将配置定义、业务配置生成记录和两个生成器加入保守输入清单；构建过程中修改配置或生成器仍会被源码身份复查拒绝。
- 保留 CMake 过期检查及最终源码/程序清单，不改变校验门限，不手工修改生成的哈希记录。生成器自身已有无变化不写入行为。
- 现场旧 GUI 缓存的 `views/pages.cpp.obj.d` 为空，`compiler_depend.make` 只有 CPP 自身，未记录业务选项头文件；单纯同步并运行 CMake 会漏掉重编译。已为 GUI/模型源文件添加业务/GUI 生成头文件的显式 `OBJECT_DEPENDS`，确保同步结果进入程序，同时保留无变化复用。原始依赖记录保存在证据目录的 `.before` 文件。

## 验证

- 新回归 `tools/test/test_gui_v2_prebuild.py` 修复前失败：实际入口跳过两个生成器，直接进入夹具依赖边界。
- `python -m unittest tools.test.test_gui_v2_prebuild tools.test.test_release_prebuild`：9项通过，覆盖顺序、任意调用目录、两个生成器失败阻断及既有 release/无变化生成合同。
- 新 PowerShell 入口语法检查通过。两个真实生成器的 `--check` 以及原 CMake 校验器在同步后通过。
- 真实配置后的 Makefile 已明确记录 `pages.cpp.obj → business_options.h`、`ui_state.cpp.obj → ui_values.h` 等生成头依赖；实际触发重新编译并链接新 EXE。
- 验证过程中，工具附带的裁剪版 PowerShell Utility 模块缺少 `Get-FileHash`，直接 Windows PowerShell 正常。仅对验证子进程前置 Windows 系统模块路径，未修改项目或系统设置。最小复现为 `nested-probe.ps1`；失败构建日志保留。
- 共享工作区在验证期间还发生了业务配置/生成记录修改，原 CMake 或源码摘要门禁按预期拒绝中途变化；相关日志保留，没有放宽检查。
- 旧缓存重新编译还发现损坏的 NMake 依赖记录报 U1076。确认没有活动 GUI 编译后，将根 CMakeCache 与三个 CMakeFiles 目录移入 `cache_before_reconfigure/`，保留原 EXE 副本；没有删除源码、配置、队列或旧证据。原 BAT 重新探测工具链，生成正常非空依赖文件并完成重建，见 `public_build_reconfigured.log`。
- 从用户相同的 `tools/build/` 工作目录运行原 BAT，完整构建通过，输出 `build_gui_v2_local/ecm_gui_v2.exe`。再次无改动运行也通过，47个对象以及生成物/输入/EXE共65个文件的内容和时间戳均保持一致，没有 CXX 编译或重新链接；见 `public_build_unchanged.log`、`reuse_before.json`。
- 最终 EXE 摘要匹配 manifest，295个保守源码清单条目逐项匹配；两个生成器的 `--check` 及原 CMake 过期校验全部通过。汇总为 `validation.json`。未执行 GUI 交互或 GPU 数学/性能验收。

当前工作区另有未提交的业务配置更新；运行入口会据此同步对应 GUI 生成物。本次脚本修复不将其他任务的配置修改一并提交。
