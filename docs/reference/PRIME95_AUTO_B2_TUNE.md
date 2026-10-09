# Prime95 Auto B2 与 PRPLL tune 源码分析

固定参考资料。来源：`STAGE2_AUTO_B2_TUNE_DESIGN.md §§2–4`。保留数学推导、第三方源码分析及其条件；源码行号针对原文研究的本地副本，第三方上游可能已有变化。本文不声明本仓库当前支持、默认值或性能。涉及实验数字时只按原文条件理解；当前实现见 [架构](../architecture/OVERVIEW.md)，当前计时口径见 [性能](../performance/STAGE2.md)。

## 2. 术语与计量单位

- `N`：待分解整数；`S=bit_length(N)`，`W=ceil(S/64)`。
- `B1`：save 中完成的第一阶段界限；`B2>B1`：本次第二阶段请求界限。
- `D`：Stage2 步长；`P=phi(D)/2`：baby 集大小。
- 当前引擎 `I=floor(B2/D)+2`：giant 点数；`G=ceil(I/P)`：G 树批数；fold 次数为 `G−1`。见 [I 的生成](../../tools/bench/stage2_tree_gpu.cu#L10998) 和 [批次数](../../tools/bench/stage2_tree_gpu.cu#L9694)。这不是 Prime95 的区间/relocation 公式。
- `L=2^k`：NTT length，避免沿用 NTT 代码局部变量 `N` 而与待分解整数混淆。
- `q_GL=2^64−2^32+1`：NTT 域模数，与待分解的 N 无关。
- `D_eff`：概率工具的群阶有效除子/挠子收益参数，**不等于 Stage2 步长 D**。
- 显存/RAM 使用 bytes、MiB=2^20 bytes；时间使用秒，吞吐量必须明确一次 iteration 的内容。


## 3. Prime95 Auto B2 的实际算法

### 3.1 默认如何进入自动选择

[默认界限与开关](../../.refactor/p95v3106b01.source/ecm.cpp#L6847)：C=0 先变为 100B1；当 C=100B1、非 QA 且 `ECMBestB2=1` 时设置 `optimal_B2`。因此这份源码中显式给出 100B1 也可能触发自动选择。

[Stage2 规划入口](../../.refactor/p95v3106b01.source/ecm.cpp#L6111) 将当前可用内存转换为 gwnum 数量 `numvals`，在 MIDSTAGE 调用 `ecm_choose_B2`。内存来自 worker 的 [avail_mem](../../.refactor/p95v3106b01.source/ecm.cpp#L7871)，并有分配失败后的缩减。这是 CPU 主内存和 gwnum 工作区模型，不能把 MB 直接换成我们的 CUDA arena MB。

### 3.2 最大化的目标

[Kruppa 相对曲线价值](../../.refactor/p95v3106b01.source/ecm.cpp#L1431)：

```text
r = B2/B1 >= 1
a(B1) = 1.96617 − 0.06781 log10(B1)
K(B1,B2) = 0.11343 + 0.88657 [log10(r)/2]^a(B1)
K(B1,100B1) = 1
efficiency = K / (stage1_cost + stage2_cost + gcd_cost)
```

最终效率在 [ecm_stage2_cost](../../.refactor/p95v3106b01.source/ecm.cpp#L5903) 计算。K 是经验相对收益，不是任意 N、任意未知因子大小的绝对成功概率。公式没有直接输入 N；N 通过 FFT 长度、gwnum 大小、GCD/逆元成本等影响分母。源码注释说明它根据 GMP-ECM 6.04 的研究修订，但未在这里给出所有拟合适用范围，首版应标记为 `kruppa_p95_v1` 经验模型。

重要区别：Auto B2 并不是最大化覆盖整数数目 `(B2−B1)/seconds`，也不是预设 `T2/T1=常数`。不同 B2 的收益增长是对数函数，成本增长则取决于算法和内存。

### 3.3 成本、内存与 N 如何进入

[Stage1 成本](../../.refactor/p95v3106b01.source/ecm.cpp#L5923) 按第一阶段 FFT 次数估计：Montgomery 路径约 `25.55 B1`、Edwards 路径约 `21.95 B1`。这些是 Prime95 算术与乘法链的测量常数，不能用作本仓库 GPU ladder 的秒数。

[实现枚举](../../.refactor/p95v3106b01.source/ecm.cpp#L5943) 带入 Stage1/Stage2 FFT 长度、线程数、N 的 bit length，比较 pairing/poly 等实现，再调用 `best_stage2_impl` 枚举 D。我们的首版只需要当前 poly 后端。

Poly 模型包括以下项：

- [内存约束](../../.refactor/p95v3106b01.source/ecm.cpp#L5786)：4 或 6 块多项式加 pool 临时量，扣除 F/R 压缩收益，加入 `polymult_mem_required`；超过 numvals 则拒绝。
- [poly 计算速率](../../.refactor/p95v3106b01.source/ecm.cpp#L5804)：每系数成本随 poly_size 的 log2 增长，再根据每线程工作区相对 L2 容量增加惩罚。
- [G 树、fold、初始余式](../../.refactor/p95v3106b01.source/ecm.cpp#L5835)：按批数及 log2(poly_size) 计费。
- [F 树保存/重建](../../.refactor/p95v3106b01.source/ecm.cpp#L5853)：多余内存可保留更多层，降低重建成本；磁盘保存还有转换费用。
- [换算和校正](../../.refactor/p95v3106b01.source/ecm.cpp#L5889)：变换与 polymult 成本换算至 Stage1 FFT 长度，加入逆元，再乘比例校正和 Stage1/Stage2 线程比。

所以 Memory 同时改变“可行算法/形状”和“重建成本”，并非单纯设定最高 B2。我们的对应项是 NTT length/batch、table 缓存、owner 驻留、GPU baby/leaf 路径及 host staging。

### 3.4 B2 搜索方法

[ecm_choose_B2](../../.refactor/p95v3106b01.source/ecm.cpp#L5997)：

1. 设置倍率上限 `MaxOptimalB2Multiplier`，默认随 numvals 增大，最多默认 10^7。
2. 起始探测倍率 50，随后按**实际覆盖端点倍率**乘 5 扩展；每个 B2 都重新选择 Stage2 实现/D。
3. 左右扩展到包围较好效率，再做类似二分的区间细化。
4. 返回实际端点对应倍率，而非原始试探倍率。

实际端点来自 [poly 批次补齐与 relocation](../../.refactor/p95v3106b01.source/ecm.cpp#L5772)，并不总等于请求 B2。**不应照抄这部分端点扩展到我们的实现**；我们应记录当前引擎真正保证的区间，再决定收益函数使用什么端点。

这种搜索依赖包围峰值的启发式。GPU 中的 NTT 2 幂跳变、partial tree、owner 回退会使成本曲线不光滑；首版宜用对数网格加局部/形状边界细化，不宣称全局最优。

### 3.5 EcmStage2RatioAdjust 的含义

[成本乘数](../../.refactor/p95v3106b01.source/ecm.cpp#L5899)：该值越大，相同 B2 的 Stage2 预测成本越高，通常会使最优 B2 减小。

[自动更新](../../.refactor/p95v3106b01.source/ecm.cpp#L9852)：

```text
rho_measured = T2_actual/T1_actual
adjust_correct = adjust_old × rho_measured/rho_predicted
adjust_correct = clamp(adjust_correct, 0.333, 3.0)
adjust_new = 0.9 adjust_old + 0.1 adjust_correct
```

仅在两阶段计时都非零且 `EcmStage2AutoAdjust` 启用时更新。它是预测误差的滚动校正，不能用 NTT event 秒数除以 Stage1 完整墙钟替代。


## 4. PRPLL tune 的实际行为

本地 PRPLL 位于 `.refactor/gpuowl`，README 从 [第16行](../../.refactor/gpuowl/README.md#L16) 标为 PRPLL。源码帮助使用 `-tune`；本文沿用用户提出的 `--tune` 作为本项目拟议接口。

### 4.1 测量的是完整 PRP 迭代

[timeConfig](../../.refactor/gpuowl/src/tune.cpp#L111) 创建相应 Gpu 配置并调用 [Gpu::timePRP](../../.refactor/gpuowl/src/Gpu.cpp#L2900)：

- quick=1..10 对应 20000..400 个迭代；默认 quick=7 是1200次。
- 先运行20次 warmup，等待队列完成，再计时剩余迭代。
- 工作主体是 PRP square，包含块间 check 用的 modMul；并非只测一个 FFT kernel。
- 队列结束后执行结果检查，失败返回 infinity，配置不会进入最佳结果。
- 返回 `microseconds/PRP iteration`，即 `iter/s = 10^6/cost`。

其中 GPU 构造、编译等发生于 timePRP 的计时外；它侧重热运行吞吐。我们当前生产逐曲线启子进程，需另外计入冷启动成本。

### 4.2 配置搜索和拐点文件

[Tune::tune](../../.refactor/gpuowl/src/tune.cpp#L361) 区分设备、FFT/NTT 类型、inplace、指数区间和 quick；同一函数还搜索若干执行参数。随后按 [shape 最大指数排序](../../.refactor/gpuowl/src/tune.cpp#L1265)，测量支持的 variant/carry 组合。

[TuneEntry::update](../../.refactor/gpuowl/src/TuneEntry.cpp#L10) 保留 cost 与 maxExp 的非支配前沿：成本更高、支持范围又不更大的配置被去掉。改进时 [写 tune.txt](../../.refactor/gpuowl/src/tune.cpp#L1408)，文件保存 cost、配置 spec 和 maxExp。

运行时 [FFTConfig::bestFit](../../.refactor/gpuowl/src/FFTConfig.cpp#L342) 优先遵守显式 FFT spec，否则遍历 tune 的成本顺序，选择支持当前指数、bits-per-word 下限和设备能力的最快配置；没有匹配则使用内建回退。

这更接近“按支持范围选择最快配置的前沿”，不只是预先计算相邻 length 的交点。我们的查询变量还有 batch、S、驻留状态及内存，可行前沿的维度更多。

### 4.3 可借鉴与不能直接移植的部分

可借鉴：显式 tune、预热、测量实际工作单元、失败候选不参与排名、增量保存、显式配置优先、无匹配时清楚回退。

需重新实现：Goldilocks 的整数 exactness/packing 上界、CUDA 形状/调度参数、不同操作类型成本、联合显存预算和来源指纹。

PRPLL 的 [maxBpw/ROE 调优](../../.refactor/gpuowl/src/tune.cpp#L129) 是浮点精度/roundoff 边界。我们的可用 bpw 应由整数上界证明，再做算术核验；不能把“偶尔没有错”当作放宽上界的依据。
