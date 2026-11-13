# Stage2 checkpoint 与发布整合开发草稿

正式说明见 [Stage2 队列进度格式](../architecture/CHECKPOINT.md#7-stage2-队列进度格式) 和 [ECM 结果输出](../usage/ECM_RESULTS.md#stage2-队列与结果发布)。

## 实现

- 新任务以 schema 5 ckpt 为唯一曲线进度来源，复用进程输出管道传递验证结果，移除逐曲线私有提交账本和 JSONL 回执依赖。
- 先保存完成位图、NF 键或因子，再发布因子。最终 NF 先在 ckpt 中保存确切文本，再原子替换提交文件。成功后依次写 finished、移除队列行、删除 ckpt。
- 当前任务内保留数学去重；跨任务的 NF 汇总行没有 sigma。历史逐曲线去重需要旧账本，schema 5 不用历史提交总数推算已经执行过的曲线。
- schema 1–4 任务一次性重建完成曲线及待发布结果。迁移先持久化新版，再归档旧档；旧 SHA 文件名的 schema 5 只迁移名称。单独执行 save 和 CPU Edwards Stage2 沿用原发布入口。
- 默认关闭新队列的逐曲线 JSONL。显式路径仍可用于诊断，诊断写入错误作为警告。已准备好的 NF 锁定最终任务策略，续跑先完成这次提交。

## 验证记录

- 存储契约 49 项，覆盖乱序完成、前缀推进、重复键、破损摘要和旧格式读写。证据目录 `data/experiments/stage2-checkpoint-v5-storage-final/`。
- 发布一致性与旧账本兼容测试通过。覆盖重放、去重、CRC、共享发布历史及路径别名处理。证据目录 `data/experiments/stage2-checkpoint-v5-results-final/`。
- 使用 RTX 4060 Laptop GPU 在隔离队列运行 M521 各 4 条曲线。单槽使用 B1=1000、B2=10000；双槽使用 B1=1000、B2=20000、D=30。另用一条已含 1009 因子的 save 验证因子补发。分别阻断结果替换和 finished 写入，恢复只完成发布和队列推进，提交行不重复。证据目录为 `data/experiments/stage2-checkpoint-v5-gpu-final-*`。
- 显式启用诊断 JSONL 后再删除它，续跑仍可从 checkpoint 完成发布，见 `data/experiments/stage2-checkpoint-v5-gpu-final-diagnostics/`。
- 旧 schema 2–3 六个恢复用例通过，覆盖 pending、旧 B2 导入及计划缩减。证据目录 `data/experiments/stage2-checkpoint-v5-legacy-run/`。
- B1 文件名、同名冲突和旧档发现通过 51 项存储契约。CUDA 本地构建通过；迁移集成 13 项全部通过，覆盖 schema 1–4、旧 SHA 文件名的 schema 5、双槽缺口、账本/JSONL 移走后的继续发布、已封存 NF 去重、归档中断重试和旧账本缺失时保留进度。证据为 `data/experiments/stage2-ckpt-b1-storage/`、`data/experiments/stage2-ckpt-b1-migration-verified/` 和 `data/experiments/stage2-ckpt-b1-results/`。

## 兼容成本

按生产 C++ 中的相关代码段统计，排除测试、空行和独立注释行。schema 1 约 27 行，负责旧单文件的复合身份解析。schema 2 约 62 行，负责旧摘要、同位数文件扫描与 B2 候选读取。schema 3 的 pending 归一化和读取大部分与 schema 4 共用。旧回执重建和归档的主体约 200 行，保留 schema 4 时仍需要这部分。

建议在程序的恢复入口保留当前格式和上一格式，再将更早的格式放到有验证的独立迁移入口。删除旧解码器之前应先发布可直接迁移的过渡版本；否则跳过该版本更新的用户会失去恢复路径。
