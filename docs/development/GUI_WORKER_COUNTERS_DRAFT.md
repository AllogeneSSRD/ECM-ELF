# GUI Worker 完成与命中计数修复

生产目录 `D:/code/GIMPS/ecm-win-x86_64-sm89-cuda13.3_param0` 在 2026-11-19 对 M62701 找到一个 Stage1 因子。`ecm.log` 中相邻的 `factor[148]` 和 `FACTOR FOUND` 描述同一结果，公开结果文件也只有一条对应的 F。原 Controller 把两条日志都计为命中，完成数则只有初始化及阶段切换时复制，未从工作进程累加。用真实因子消息及一个队列完成汇总回放，稳定得到 `tasks_done=0, hits=2`。

修复后通过明确数值的因子事件计数，在当前任务内去重。Stage1 增加队列提交成功后的 `stage1_task_done` 通知；Stage2 沿用 `task_done`。队列完成汇总只补足该次进程尚未通知的任务数，避免实时通知与最终汇总重复。每次计算进程启动重置该进程的汇总基数，Controller 的累计值跨重启及阶段切换保留。当前统计口径见 [GUI v2](../usage/GUI_V2.md)。

验证覆盖生产日志回放、分多次 drain 的重复因子、Stage1 实时完成与汇总、手动重启、Stage2 曲线与任务区分及混合队列切换。证据保存在 `data/experiments/gui-worker-counters/`。生产目录只读，当前已运行 GUI 的旧计数需重新载入工作流后开始统计。

生产回放修复后为 `1 / 1`；Controller 的 12 组场景、66 项检查及日志解析的 113 项检查通过。GUI v2 构建及 App 回归通过。Stage1 的任务完成消息通过 `src/main.cpp` 语法编译检查，计算程序发布包待重新构建。汇总在 `final/verified.json`，App 证据在 `gui-app/summary.json`。

现有 Controller 测试夹具曾因 `WorkerSettings` 增加字段后仍使用位置初始化，把 3000 毫秒误当成 Worker 数量。验证期间已停止隔离测试进程，改为按字段赋值并检查只创建一个 Worker。旧失败目录及证据保留。
