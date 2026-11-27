# Stage1 分批交接

## 已确认的规则

- 默认开启。只有实际接收方为 ECM-ELF 时才分批投递。
- PART 使用对应存档区间，完成后保留 schema 6 断点。最终行覆盖整个任务，复用已完成区间，并认领空闲的缺口。
- 同组允许多个 worker。区间独占，最终行等待其余 worker 完成后汇总。
- Stage1 后续命中因子时，旧组只结案。最终行的范围截止到命中批次之前。继续分解余因子时，在原文件追加投影记录。
- 整档原文和原始记录文本不用于恢复校验。保留存档自身的 `CHECKSUM` 和 `QCHECKSUM`。原有断点由生成它的版本继续运行。

## 验证重点

- PART 在 Stage1 停止前可消费，Stage1 续跑时沿用原组和记录偏移。
- 不同 worker 的区间可以重叠运行。最终行等待已认领的区间，只计算剩余的曲线，并只发布一条汇总 `NF`。
- 收尾重试、延迟送达的 PART、已发现因子的旧组和余因子区间。
- 修改元数据后可运行，修改算术字段导致存档校验失败。

## 本地验证

Windows `0.74.0` 使用 RTX 4060 Laptop 运行 CUDA NTT Stage1，并用 CPU Stage2 执行 `test_stage2_partial_handoff.py --resume`。输入为 M4423、B1=25000、B2=26000，5 条曲线按 2/2/1 分批。模拟第二段持锁时，最终行补算最后一条后等待，所有区间完成后只汇总一条 5 曲线 NF。另一次运行在 Stage1 停止前投递并开始消费首批，重启沿用组 ID，未重新投递首批。

Edwards 余因子回归覆盖开启、关闭及按余因子位宽路由。独立样本验证 PART 自行跳过已命中的组、最终行只收尾以及 `QCHECKSUM` 改坏时拒绝输入。关闭分批使用 `handoff_partial=false`，关闭交接使用 `handoff_mode=off`，两种设置均通过验证。

完整日志位于 `data/experiments/stage2_handoff/`，其中 `e2e_combined_resume`、`interrupted_stage1_verified`、`factor_regression`、`factor_cancel` 和 `config_*` 分别保留上述样本。

## 同版调整为共享曲线断点

共享断点按实际模数和绝对记录号保存每条结果，因此 PART 可以改区间。重叠任务通过单一认领文件的记录字节锁协调。B2 是曲线结果字段，调整 B2 时沿用已完成记录。Stage1 的发布游标也保存在共享断点。余因子任务用 `project=cofactor` 在读档时投影，原始 `.save` 保持单行记录。

- `bounds_shared_cpu` 使用 M127 的 `skip=100,count=50`，先完成 20 条 B2=100，再由两个 worker 处理重叠范围。断点共 50 条独立结果，其余 30 条 B2=300，最终 `NF` 报告 B2=100。
- `e2e_shared_cpu` 用 4060 Laptop 跑 M4423 CUDA NTT Stage1，暂停并恢复 2/2/1 批次。CPU Stage2 覆盖了修改 PART、等待两条被认领曲线、复用结果及元数据编辑。
- `factor_shared_cpu` 和 `factor_shared_gpu` 验证了同一条 `FACTOR` 记录上的余因子投影。CPU 和 CUDA Stage2 均在该样本找到 10007，`.save` 保留一条原记录。
