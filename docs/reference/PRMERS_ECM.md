# PrMers Gaussian ECM BSGS：算法、状态与访存

固定第三方源码分析，提取自 `STAGE2_GPU_COMPARE_PRMERS.md`。对象为本地 PrMers v99.98 的 Gaussian-Mersenne ECM 优化路径，不是 P−1 V-trace，也不声明其上游最新状态或本仓库实现资格。

## 对象与入口

- [`RunGaussianMersenneEcmFast.cpp`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmFast.cpp#L607)：选择优化 BSGS 路径。
- [`RunGaussianMersenneEcmOptimized.cpp`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L1014)：fused Montgomery 引擎与 Stage1/Stage2 交接。
- [`RELEASE_V99.98_GAUSSIAN_ECM_FUSED_BSGS.md`](../../.refactor/PrMers-main/RELEASE_V99.98_GAUSSIAN_ECM_FUSED_BSGS.md#L37)：该版本 `-bsgs` 路径和验证边界。

`RunPM1.cpp` / Pair95 运行幂或迹，不是椭圆曲线点；不能以其公式和吞吐代替 ECM 对照。

## Prime plan 与所需 baby 集

主机分段筛枚举 Stage2 候选 q，映射为 `q=kD±d`，`d=|kD−q|`，筛选 `d≤D/2`、`gcd(d,D)=1` 的条目。计划保存 k/d，baby 集为实际 d 的去重集合。

它不默认生成从 1 到 D/2 的完整表，也不一定生成全部 φ(D)/2 个相对素数 residue。只对 plan 需要的 d 生成 `[d]Q`；该副本按每个 d 调 scalar ladder，再复制到 baby X/Z 槽并准备 multiplicand。

入口：[`prime plan`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L708)、[`baby residue 与 D`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L773)、[`baby 生成`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L1347)。

## Giant 流与交叉差

先保留 base=`[D]Q` 和相邻 giant 状态。按 k 推进：

```text
prev=[(k-1)D]Q
cur =[kD]Q
next=xADD(cur,base,prev)
处理当前 k 下的各 prime entry
prev=cur; cur=next
```

每项计算 `Xg·Zb−Zg·Xb`，乘入 ACC；这是齐次 x 相遇判据，避免逐配对求逆。点的域和 prepared multiplicand 必须与引擎表示一致。

入口：[`giant 初始化`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L1361)、[`交叉差与 ACC`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L1452)、[`giant 推进`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L1485)。

## 存储与访存

`OptLayout` 把 baby X/Z、base、giant 与 next 等映射到 engine register 编号。这是引擎的逻辑寄存器/存储槽，不能直接理解为一个 CUDA 线程的硬件寄存器数。

该版本默认 D=210；注释以 24 个 baby x/坐标对对应 48 个槽描述容量，自动模式限制所需 baby 数。主机通过二分查找确定 baby 槽，device engine 执行 prepared multiply、subtract 和 accumulate。

giant 只保留相邻状态，不构建长度 B2/D 的完整坐标表；对同一 k 的多个 prime entry 复用当前 giant。它减少显式 giant 表容量和重复表读取，同时留下依赖 chain、host 提交和同步边界。逻辑槽驻留不意味着没有全局显存访问；实际物理流量须沿 engine 内核核对。

入口：[`布局`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L855)、[`槽查找`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L1003)、[`布局初始化`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L1157)。

## GCD 与恢复

该路径默认每 256 terms 同步、投影 ACC 并执行主机 GMP GCD。`1<gcd<N` 是有效因子；`gcd=N` 时回退 legacy Stage2，另有第三方自己的 checkpoint/resume。

入口：[`GCD 批量与饱和恢复`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L1400)、[`term 与检查点`](../../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp#L1441)。

## 与多项式算法的关系

直接 BSGS 的工作按实际候选项增长；baby/giant 状态与 plan 的优化减少准备和搬运，不消除逐项 cross-product/ACC。多项式方法将大批差项改写为 F/G 积树、模 F fold 和一次多点求值，增加较大工作区换取批处理。

可借鉴 residue 去重、相邻 giant chain、乘积/GCD 的恢复边界。Gaussian-Mersenne 字段和引擎表示不能直接用于普通 N；未按同一整数、曲线、界限、设备及计时边界测量，不能给出速度比。

数学联系见[多项式 Stage2](ECM_STAGE2_MATH.md)，Prime95 实现见[Poly 分析](PRIME95_STAGE2.md)。
