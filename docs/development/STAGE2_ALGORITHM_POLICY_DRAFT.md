# Stage2 按位宽选型开发草稿

正式用法见 [Stage2 使用](../usage/STAGE2.md#选择运行设备)。配置项由 `config/ecm_options.json` 生成。默认策略从 4096 bit 起让 Edwards 优先使用 GWNUM，GPU 的候选范围截止到 16,384 bit，最后允许 Edwards 走配置的 CPU 后端。区间语法和 Stage1 `gpu_algorithm` 使用同一解析器。

统一入口以第一条选中存档记录的实际 N 选型。队列后端在获取现有 worker 所有权后复核 worktodo 条目和顺序，每次处理一条，再交回统一入口选型。CPU、CUDA 和 OpenCL 共用现有发布、整曲线检查点和队列停止流程。GPU 还匹配当前输入和 worktodo 原始指数上限。找不到适用算法时原条目留在队列。

## 验证

- `tools/test/test_combined_stage2.py` 使用 4095、4096、16384 和 16385 bit 校验切换边界，覆盖全角冒号、worker 覆盖、无匹配规则时保留任务，以及 `--target` 和 `--backend` 的显式选择。
- 同一队列在 RTX 4060 Laptop CUDA 和本机 GWNUM 之间切换，含 M127 的两条 Edwards 曲线和 M4423 的一条 Edwards 曲线。另用 GUI 同型停止事件暂停第一条任务后续算，确认仅完成一条曲线即停止，`--once` 只推进一条任务。
- 原生 OpenCL worker 在 RTX 4060 Laptop 上完成已知因子 10007 的任务后，统一入口用 CPU/libecm 完成同队列的下一条任务。日志和请求参数保留于 `data/experiments/stage2_algorithm_0700_final/`。
