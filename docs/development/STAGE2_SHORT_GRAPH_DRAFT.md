# 短NTT Graph提交融合草稿

## 范围与合同

用户授权推进短NTT Graph、管理员Nsight Compute 2026/Systems采集、固定D完整Stage2 A/B。保持默认55W/1800MHz上限，不调整设备控制，不替换生产目录的程序或队列。

先比较同一二进制的环境开关off/on。候选仅合并两次forward和融合inverse的主机提交，使用两个Graph保留当前forward/inverse计时事件；carry、S4、GMP/oracle、分块数量和算术检查不变。专用nonblocking capture stream只记录节点，执行仍在原default stream中；不改变整个工程的default-stream编译合同。

缓存仅接收arena稳定workspace与已缓存表，身份包含device、长度、batch、计划、根、数组/表地址及scale选项。在workspace释放/增长、根表驱逐及arena释放前失效；没有缓存的临时/导出路径沿用原实现。冷形状先累计调用次数，限制长度、batch×length和Graph条目数；驱动Graph对象不伪装成已计账的device数组。

## 验收

1. 独立小规模GMP/卷积、tail batch、复用和释放/重建，保持完整检查和字数门限。
2. 固定save/N/B1/B2/D/预算，短/长NTT、普通/梅森归约、不同batch及owner回退完整A/B，保留所有样本及明显回退。
3. Systems检查提交次数、GPU空隙、kernel时间和分阶段占比；Compute捕获实际热点的geometry/occupancy/吞吐/stall，不把重放时长当作性能结果。
4. 同二进制交错完整曲线至少3次/侧，因素和NTT计数/叶子指纹一致；没有完整曲线证据不改变默认。

原始日志、保存点、trace与冻结身份放`data/experiments/`，汇总和图分别放`data/benchmarks/`与`data/figures/`。草稿保留供用户在完成后决定整理方式。

## 实施与验证记录

- 实现生产运行时的双Graph缓存、稳定地址与表身份、workspace/表驱逐失效、默认关闭及阶段级统计。没有修改GPU kernel算术或全工程默认流合同。
- 独立门禁完整GMP整数乘积检查368次、输出14779392word，bad=0；故障注入恰检出一个错误并返回非零。workspace、FuseCtx生命周期和容量检查通过。
- 同一二进制SHA256 `d47873056543eeb6f396101e84684f39b2928c27ae27f73da726118345ecce27`，完成9组默认准入×4/侧、3组扩大准入×4/侧、生产P=155520×3/侧，共102条完整曲线；没有丢弃warmup。额外4423/7995-bit的4条叶子指纹诊断均一致。
- 首轮矩阵总中位时间少0.32%；扩大准入与生产P均基本持平，未达到默认部署资格。4423/256项首轮−0.42%、复测+1.35%，保留两个独立测量组，不挑选组合。
- 管理员Systems4条完整时间线、Compute6个真实kernel捕获已完成；所有曲线算术通过，Compute形状/资源与Systems一致。Systems的kernel数不变；短batch720→2880的吞吐基本不增，Compute显示大batch的DRAM压力增加。重放时钟变化明确记录，不拿插桩秒数作为正式收益。
- 工具修正记录：早期smoke collector的str/Path转换错误发生于启动前；已保留失败目录。NCU报告在沙箱内导出不能读取用户Sections路径，改为非沙箱只读导出；失败导出不覆盖成功表。所有原始trace与有效样本保留。

正式当前说明与下一步条件见[Graph性能报告](../performance/NTT_GRAPH.md)，复现见[工具说明](../../tools/bench/README_STAGE2_GRAPH.md)。本草稿按仓库规则保留，完成后由用户决定整理，不自行删除。
