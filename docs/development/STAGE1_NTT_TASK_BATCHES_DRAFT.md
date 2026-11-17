# Stage1 NTT 自动拆批开发草稿

- 算术后端在任务归一化后按实际模数和资格匹配。默认 CUDA 规则为 `ntt[16384:], cgbn[]`。一次任务固定曲线和标量，可在批间选不同 NTT 批量及计划。
- 任务级容器沿用原 `.ckpt` 命名，在同一原子写入中保存已完成前缀和当前批布局、真实曲线数及整数快照。原生 NTT 位／列快照仍由算术 workflow 验证。旧单批检查点可按原格式继续执行。
- 生产样本独立于合成调参。真实曲线以连续 sigma 区间交付。只为尾批选填充时扩展设备维度。临时槽借用算术输入，回读后裁掉。
- 原始验收数据放在 `data/experiments/stage1_ntt_task_batches/`。正式说明见 [使用](../usage/STAGE1.md#任务曲线数与设备批量)、[架构](../architecture/STAGE1.md#ntt-任务拆批与恢复) 和 [测试](../performance/STAGE1_MERSENNE_NTT.md#gpu1-验证)。
- 尾批补齐改为分配真实 sigma。检查点 identity 仍绑定原任务请求量，末批和完成前缀允许有限超出该数量。读取旧档时沿用其已记录的真实区间。
