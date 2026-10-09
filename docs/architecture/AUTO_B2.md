# Auto B2 与 tune

## 目标与可用性

Auto B2 优化连续生成/处理曲线的单位时间收益，计入 Stage1 成本，即使本次读取 save。它不是有限已有 save 的每条 Stage2 时限选择器，也不是强制某个 T2/T1 比值。

当前源码已有原生 reader、搜索、CLI/INI/队列和导出工具，但当前生产算术/内存组合没有匹配的新合格成本 profile。使用显式 B2；不能把通过其他冻结二进制校准的 profile 交给当前程序。

## 原生接口

`--auto-b2 --cost-profile <file>` 对最终 B2=0 的任务生效。显式同时给非零 `--b2` 与 `--auto-b2` 报冲突；队列或 INI 已有非零 B2 时保留固定值。

`--auto-min-b2`/`--auto-max-b2` 只缩小已测范围，`--d` 限制实测 D，`--arena-mb`/`--owner-budget-mb` 选择匹配路径，`--stage1-batch` 选择实测 Stage1 摊销，`--stage1-seconds-per-curve` 可提供有限正成本，`--stage2-ratio-adjust` 对预测 T2 乘正系数 R。

每个 curve worker 按当前 free VRAM 重新选择，成功结果保留真实 B2/D、原始请求及规划时长；规划/执行失败保留未完成队列。plan-only 不推进队列。

## 收益和成本

P=φ(D)/2，I=⌊B2/D⌋+2，G=⌈I/P⌉。沿用 Prime95 相对收益近似：

a=1.96617−0.06781·log₁₀B1；K=0.11343+0.88657·(log₁₀(B2/B1)/2)ᵃ。

选择 score=K/[T1+R·(Tengine+Tcold)] 最大的已测候选。K 不是绝对成功概率，R 不是目标阶段耗时比。

Tengine 按实际树组、Newton 逆、G1 局部逆/根归约、fold、scaled 下降、giant chain/ladder、pack/copy 和固定项组合；运行 profile 每个 scope 有 17 个非负率。冷启动差额单独估计，不保证每条进程 wall 达到同样精度。

搜索在各 scope 的并集内部生成对数 B2 点，并加入 G 树、giant chunk、chain 阈值与整数平台端点。每项仍通过 P/G/B2/D/path 范围及预算检查；范围边界最优或 `range_limited=false` 都不证明连续全局最优。

## Profile 合同

`.cprof` v2 是版本化文本，不依赖 Python/JSON runtime。身份包含程序 SHA、GPU UUID/SM、CUDA runtime/driver、算术后端、outer、feature/accounting、命名策略与来源/审计 SHA。范围包含位宽、B1、D、P/G/B2、arena、resident 与实际 chain/ladder 覆盖。

reader 限制 1 MiB，拒绝旧格式、缺失 END、重复键、非法整数、非有限/负率、未覆盖路径及身份不符。Auto B2 目前限制已测精确梅森模数和≤8192 bits；余因子/梅森承载、高 B1、choose12、其他设备不自动外推。当前 canonical 等算术变化的保护会拒绝没有相应校准的组合。

组件准入不能保证进程峰，输出 `process_peak_guaranteed=false`。当前全流程显存模型边界见 [内存](MEMORY.md)。

## Tune

`--tune ntt` 测不同 log₂L 的精确域卷积。参数含 `--length-log2 <a:b>`、`--tune-repeats <n>`、`--tune-memory-mb <MiB>` 与 `--tune-file <path>`；默认重复 5，预热不计中位数。需要固定 Goldilocks 后端。

iter/s 指每秒完整 field convolution 次数，不是每秒 ECM 曲线，也不含 S4 系数归约、树准备和 GCD。profile 同时记录设备/后端、实际长度、容量、预热/样本和精确性核对。一次 NTT tune 不能单独推导最优 B2。

按位宽校准 Auto B2 还需真实 Stage1 摊销、阶段成本、非满树/根操作、分块和驻留/回退数据。生成拟合、冻结预测、独立验证与收益排名后才能导出运行 profile。

## 标定资格与当前证据

导出要求完整声明网格的身份/算术/覆盖通过，逐条时间误差≤10%，收益排名损失≤5%；失败 scope 不可删除后发布部分子集。重放点参加精度检查，但不充当独立收益排名样本。

完整 G1/G2/bridge 校准与验证共 1134 条曲线、63 个 scope，算术/身份通过；42 个 scope 达到时间门限，21 个失败，误差−60.138%…+13.718%。六组收益排名通过、最大损失0.633%，整体仍不合格，导出器拒绝生成新 profile。证据在 `data/experiments/ecm_auto_b2_bridge_20261006_{profile,audit,evidence}.json`。

两次同输入时间 a≤b 的固定单值预测最小最坏相对误差为 (b−a)/(b+a)。±10% 要求 b/a≤1.2222；存在超过这个比例的重复样本时，仅换拟合器不能满足该合同。需先处理等待波动，或明确批准新的长期均值/时间区间合同。

## 入口

- [ecm_stage2_cost_profile.h](../../src/core/ecm_stage2_cost_profile.h#L32)：reader、scope、`Work` 与 `choose`。
- [驱动资格检查](../../src/core/ecm_cuda_stage2_main.cpp#L576)、[收益公式](../../src/core/ecm_stage2_cost_profile.h#L185)。
- [measure_ecm_costs.py](../../tools/bench/measure_ecm_costs.py)、[fit_ecm_costs.py](../../tools/bench/fit_ecm_costs.py)、[validate_ecm_costs.py](../../tools/bench/validate_ecm_costs.py)、[audit_ecm_costs.py](../../tools/bench/audit_ecm_costs.py)、[export_ecm_cost_profile.py](../../tools/bench/export_ecm_cost_profile.py)。
- [当前 TODO](../TODO.md)。
