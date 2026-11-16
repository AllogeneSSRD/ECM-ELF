# GUI v2 任务监听开发草稿

Worker 工具栏在「安全停止」右侧提供「任务监听」。启用后立即检查；此后每五分钟检查所有 Worker，约每秒采样一次各 `worktodo` 文件的修改时间和大小，共用路径只采样一次。文件变更时仅复查使用该路径的 Worker。使用说明见 [GUI v2](../usage/GUI_V2.md)。

`App::poll_task_watch` 在 Controller 更新状态和释放已结束进程的控制锁后执行。队列检查沿用 Worker 的优先级和跳过规则。启动交给自动调度入口，由现有机制检查队列占用并选择 Stage1 或 Stage2。停止请求和熔断状态会阻止监听再次拉起 Worker。全局安全停止关闭监听，单个 Worker 安全停止暂停该 Worker 的监听。监听状态仅在当前 GUI 会话和工作流中有效。

验证在 `tools/test/gui_v2/app_test.cpp` 中复用具备真实队列锁的假 Worker。测试用可控时钟触发五分钟整轮扫描，并检查空队列、变更触发、占用期间拒绝启动、停止后保持暂停和熔断行为。

初次测试在安全停止后只等到进程退出，仍有一轮待清理的停止请求。复测改为等待 Controller 清除停止状态，并逐次确认新进程取得队列锁。原始证据保存在 `data/experiments/gui-v2-task-watch-verified/`，复测证据保存在 `data/experiments/gui-v2-task-watch-retest/`。一键 DX11 测试的 `--only app` 会一并运行该回归。
