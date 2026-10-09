# Stage1 性能依据

## 口径与当前策略

CUDA 默认 ladder、TPB128、按容器选择 TPI，分档寄存器；PRAC 默认不编译。已测局部最佳不能扩大到其他位宽/设备或直接替换默认。

完整曲线 wall、GPU kernel 时间、批量摊销 s/curve 和短时投影分别报告。生产 B1=10e6…260e6 的短时采样按完整/窗口工作比投影，排除冷准备、最终归一化/save；checkpoint 不算已完成曲线。

计时先预热，再按相反/交错顺序重复，保留完整矩阵、失败和原始日志。管理员 NCU 会重放，采集时间不作吞吐；Nsight Systems 的自身事件间隙不等于整卡空闲比例。

## 已测 PRAC 范围

RTX4060 Laptop、24 SM、sm89、N=2^4423−1、4608/TPI16、lcm、TPB128 的显式实验中，批量1536、cap168、目标切片50ms 为已测有效配置。TPI32/cap128 能改善某些受限批量的资源或局部吞吐，但没有超过该范围最佳吞吐。

在终结规则专用体的冻结对照中，目标50ms、每配置4次短时前缀，得到：

| B1 | 参考 PRAC s/curve | 专用体 s/curve | 时间变化 |
| --- | ---: | ---: | ---: |
| 10e6 | 5.063071 | 5.050395 | −0.250% |
| 260e6 | 132.510655 | 132.178987 | −0.250% |

这是速度投影的中位数，未计完整曲线墙钟。冻结专用二进制 SHA=`464f124878ab755fba5e698f26962c15a1425c7bc0f4de83ed85e8df773d2b88`；108 条采样矩阵完整，GPU 忙时 SM=1800MHz。后续共享 DBL 的配对测试保持 TPI16/cap168/50ms 为最佳范围，不能把此表当作不同二进制的最新完整性能。

当前可复现选项、源码和采样脚本见 [实现](../architecture/STAGE1.md)、[bench_cuda_prac.py](../../tools/bench/bench_cuda_prac.py)、[bench_stage1_tpi_matrix.py](../../tools/bench/bench_stage1_tpi_matrix.py)。证据在 `data/experiments/stage1_rule_sentinel_*`、`stage1_shared_dbl_*`；基准必须同时记录实际 exe SHA、variant、TPI、寄存器、C 和切片策略。

## 资源与瓶颈

4608/TPI16 PRAC 的一个自然分配样本为172 regs、实际176，TPB128 时寄存器最多容纳2 blocks/SM；cap168 可容纳3。强制 TPI32 样本为109/112 regs、4 blocks/SM，但同时减小每 block 曲线数、增加 padding/跨 lane 成本，端到端未因此更快。

已采集的 natural/168/TPI32 achieved occupancy 为16.34%/19.37%/27.12%，issue active约30.60%/30.93%/38.93%。这些 profiler 值说明资源权衡，不能独立证明瓶颈成因或加速比例。DRAM 吞吐很低，主要成本仍在依赖多精度算术、发射与局部存储/调度。

normalized PRAC 按 ADD=A、DBL=D 统计：ADD=4M+2S，DBL=3M+2S；若平方与乘法同价，工作权重为6A+5D。该权重用于投影，不代表固定 GPU 周期。

## 限制与后续

本机默认1800MHz/55W条件下继续测量，不混合临时提高功率的数据。不同批次、驱动、采样窗口和构建的点不能直接拼成新的加速比。

对通用默认的改变仍需更多位宽、曲线族、设备和完整 B1 验证；现阶段优化优先级及其他未完成工作见 [TODO](../TODO.md)。
