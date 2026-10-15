# 工具目录

复用脚本按用途维护，临时实验产物写到忽略的 `data/`。项目功能、算法与性能结论集中在 [docs](../docs/README.md)。

| 目录 | 用途与入口 |
| --- | --- |
| [build](build/README.md) | 两阶段 × 开发/本机/发布的 PS1/BAT；dev/test/internal 辅助 |
| test | CPU/设备算术、输入、队列、GUI 和规划验证；构建不自动执行 |
| bench | 计时、NTT/点算术探针、剖析、规划、标定和图表 |
| [ecm_dataset](ecm_dataset/README.md) | 梅森数/因子核心数据及每因子一个 sigma |
| ecm_worktodo | 任务生成、分配和转换 |
| [log_parser](log_parser/README_PRIME95_ECM_BENCH.md) | Prime95 ECM 日志、完整/提前结束样本和 CPU/GPU配对 |
| stat / ecm_prob | 参数化、点阶、成功率与吞吐模型 |
| ecm_report | 因子结果汇总与外部因子资料 |
| gen | 配置及其他可复用生成器 |
| disasm | 设备代码/ISA 分析 |
| diag | 可选现场诊断；不作为普通计算启动步骤 |
| gwnum_probe | 外部 gwnum/polymult 探针，不是生产 Stage2 引擎 |
| refactor | 工程辅助脚本 |
| p95feeder | 本地 save 的独立 Prime95 交接程序 |

## 当前 Stage2 工作入口

- [位宽/B2测量](bench/README_STAGE2_N_SCALING.md)：数据库只读输入、三遍余因子与完整梅森对照、统计和阶段堆积图。
- [短NTT分块A/B](bench/README_STAGE2_SHORT_NTT.md)：冻结输入的开关对照、完整耗时、长度计数、GMP/执行计划门禁及图表。
- [短NTT Graph A/B](bench/README_STAGE2_GRAPH.md)：完整曲线、缓存生命周期门禁、管理员Systems/Compute与图表。
- [CUDA/Tensor单卡并发](bench/README_NTT_TENSOR_CONCURRENCY.md)：私有数据双流、纯CUDA控制、96-bit候选及范围剖析。
- [承载与内存规划](bench/README_STAGE2_CARRIER_PLAN.md)：N/M、B/Q、请求程序、NTT/S4/giant 组件与台账。
- [当前管线](../docs/architecture/STAGE2.md)、[内存合同](../docs/architecture/MEMORY.md)、[Auto B2](../docs/architecture/AUTO_B2.md)。

benchmark 需保存输入、exe/DLL/源码身份、实际参数、设备状态和所有样本。性能计时与额外诊断分别执行；没有请求时构建入口不自动启动 GPU 或测试。
