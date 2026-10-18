# GUI 自动滚动与 worker 版本修复草稿

版本：0.38.3。当前使用方式见 [GUI 使用说明](../usage/GUI.md)和 [Stage2 使用说明](../usage/STAGE2.md)。本草稿保留复现、定位和验证过程，由用户决定后续处理。

## 问题与复现

- 每个 worker 输出窗同时显示 GUI 版本和 worker 版本，用户要求仅保留后者。显示值取自实际 worker 横幅；例如旧二进制报告 `ECM-ELF 0.38.0` 时，界面也应显示这个值，而不是当前 GUI 的源码版本。
- Stage1 正常运行已有版本横幅；CUDA Stage2 原先仅在 `--gui-events` 下报告。用空 worktodo 执行普通 `--dry-run`，输出只有 `dry_run_complete`，没有版本。
- ImGui 日志子窗首次绘制 40 行时，布局前滚动上限是 0；下一帧上限变成 496，但当前滚动位置仍为 0，原有“位于底部”判断随即失效。新增 20 行后上限变成 836，仍停在顶部。独立复现程序保存在忽略目录 `data/experiments/gui_scroll_repro.cpp`；同一调用路径已移入 GUI 自检，修复前自检报告 `initial=0 appended=0 scroll=0 max=836`。

## 修复

日志子窗继续在用户手动向上翻阅时暂停自动跟随，但定位改为本帧最后一项的底部，而不使用布局前的旧 `ScrollMax`。Stage2 主程序在普通队列、直接读档和 dry run 等入口统一输出程序名与版本；内部只输出机器数据的辅助入口保持原样。GUI 输出窗仅显示实际 worker 横幅解析得到的版本。

代码入口：[GUI 日志窗和回归自检](../../src/gui/app.cpp)、[GUI 自检调用](../../src/gui/main_win32.cpp)、[Stage2 主程序](../../src/core/ecm_cuda_stage2_main.cpp)、[公共版本](../../src/core/ecm_program_version.h)。

## 验证

GUI 自检在修复前为 36 通过、1 失败；改用本帧末尾定位后为 37 通过、0 失败，覆盖首次输出、追加输出和手动翻阅暂停。假 worker 的进程与版本捕获自检为 54/54；日志解析 85/85，分别确认旧 Stage1 与新 Stage2 worker 横幅；中英文 GUI 条目数均为 258。

Stage2 完整 CUDA 重编译首次由 `ptxas` 报 `INVALID_HANDLE`，降低 split compile 后仍长时间无进展，已停止该尝试。正式构建脚本校验 CUDA 依赖和已有对象的哈希后，以 host-only 模式重编译 4 个主机翻译单元并成功链接。空 worktodo 的普通及 `--gui-events` dry run 均只输出一次 `ECM-ELF 0.38.3 (Stage2)` 横幅并以状态 0 退出。未运行真实曲线算术或性能测试。
