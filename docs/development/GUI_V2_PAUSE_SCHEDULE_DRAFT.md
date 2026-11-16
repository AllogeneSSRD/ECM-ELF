# GUI v2 定时暂停开发草稿

计划采用工作流 TOML 的字符串数组，每项包含应用、Worker、星期和时间。旧三组规则等同于应用组为 `*`。星期接受英文缩写和 1–7，跨午夜按开始日匹配。Worker 选择区分数量、单编号和方括号列表，列表使用现有的交接 Worker 编号解析。当前操作说明见 [GUI v2 定时暂停](../usage/GUI_V2.md#worker-定时暂停)。

调度在 Controller 更新前计算暂停集合，使用现有的安全停止协议。恢复在 Controller 清理停止状态、GUI 释放该 Worker 的队列控制租约后使用自动启动入口。任务监听继续检查其他 Worker。`string_array` 加入共用 TOML schema，使 Stage1 和 Stage2 可以跳过 GUI 调度语义并继续读取多行工作流。

回归覆盖数字/英文星期、跨午夜与端点、Worker 数量选择与重叠、分组报错、多行 TOML 读写，以及同一 App 内定时停止、任务监听和恢复。Worker 进程由真实队列锁测试夹具代替。

证据保存在 `data/experiments/gui-v2-pause-schedule/`。共用配置验证使用 `tools/test/test_ecm_toml.ps1`，产物和编译日志保存在 `build_dev/gui_v2_pause_toml_check/`。本轮覆盖 Stage1 和 Stage2 的共用配置读取入口，未重新生成计算程序发布包。

匹配语义参考 Prime95 本地上游 `undoc.txt` 的 `PauseWhileRunning`。GUI 仅在命名应用规则所处的时间窗内约每秒枚举进程，优先使用完整执行路径，匹配时忽略大小写和空白。进程快照临时失败时保留最近一次成功的枚举结果。App 回归通过，验证仅目录名命中的实际进程，以及同一分钟内随该进程退出恢复 Worker 的流程。证据位于 `data/experiments/gui-v2-pause-applications/`。
