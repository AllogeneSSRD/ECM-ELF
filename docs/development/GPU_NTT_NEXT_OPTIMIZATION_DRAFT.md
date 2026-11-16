# GPU NTT 下一轮优化

以 CUDA Stage1 GPU NTT 的长批次吞吐为主指标，同时记录单曲线延迟。第一轮用 0.63.11 作基线，主测原生 Edwards 2XY NAF。Montgomery PARAM0 x-only 用来区分通用 NTT 收益和 Edwards 公式收益。先在同一 Goldilocks 计划下减少数据移动和等待，再探索能降一档变换长度的算术后端。第一轮数据见[剖析与候选筛选](#第一轮-gpu1-剖析与候选筛选)。

## 当前证据与优先级

| 工作负载 | 已观察到的现象 | 下一步要判定什么 |
| --- | --- | --- |
| 短变换、多曲线 | 0.62.4 在 4070 Ti 把 2–32 点的逐点乘加与逆变换放进 warp 子组，128 曲线的整段耗时下降 10.34%–19.89%。 | 128–1024 点是否还能用较小 CTA 或一块多变换提高每波有效工作量。先测既有快路径作对照。 |
| 中宽变换、多曲线 | GPU1 的 0.61.5 剖析中，2048、2560、10240、16384 点在 30 曲线时，tile 占已记录 kernel 时间的 45.9%–54.1%，carry 为 25.0%–32.9%，outer 为 0%–14.6%。 | 在当前实现上，把 tile 拆成前向、融合逐点逆向两项，再区分算术依赖、同步、寄存器溢出和访存。 |
| 宽变换、单曲线 | 历史轨迹在 2560 点出现 43.8% carry，在 16384 点出现 30.3% outer。同一个 16384 点布局改动曾让加权单曲线缩短约 1.7%，但 30 曲线变慢约 8.8%。 | 单曲线是否缺可发射 warp，批量是否受额外 outer 全数组读写限制。计划必须按方向、长度和批量分别选。 |
| 原生 Edwards 加权归约 | 0.63.10 的共享去权缓存消除了部分重复域乘。选中的三档宽位形状在 4060 Laptop 的 30 曲线变化小于 0.5%；4070 Ti 同样三档的 30 曲线耗时减少约 3.9%–7.1%。强制启用在另两个形状出现退化。 | 共享内存驻留和访存代价是否抵消了重算节省。按变换形状和卡采集，保留原归约对照。 |

最明确的方向是 **tile／outer 的执行排布和中间数据生命周期**。此前整个编译单元限 32 个寄存器引发 spill，限 48 个也没有稳定收益。把逐点乘加的输入指针表改为只读 grid 参数后，每线程栈从 96 B 降到 0；GPU1 的 2560 点、30 曲线吞吐提高约 19%。应先识别热点里的长生存期和局部内存访问，再调整寄存器约束。现有 Stage1 时间线可以定位耗时，无法单凭这些占比判定共享内存 bank conflict、L2 命中率或 DRAM 已饱和。 [短逆变换](../performance/STAGE1_MERSENNE_NTT.md#短逆变换) · [tile 剖析](../performance/STAGE1_MERSENNE_NTT.md#tile-剖析与逆变换缩放) · [宽档布局](../performance/STAGE1_MERSENNE_NTT.md#16384-点布局搜索) · [去权缓存](../performance/STAGE1_MERSENNE_NTT.md#原生-edwards-共享去权缓存) · [Nsight Compute 指标](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

## 1. 重做基线，再优化 tile 和 outer

1. **分别定位前向、逆向和跨 tile 的成本。** 用 Nsight Systems 统计 pack、radix-5/PFA、前向 tile、outer、逐点乘加与首段逆向、去权与 limb 汇集、进位与跨段写回，另记 Graph 节点间空档。Nsight Compute 只采代表性热点，记录实际寄存器、动态共享内存、local load/store、活跃及可发射 warp、阻塞原因、共享内存额外事务、L2/DRAM 字节和整数流水线。计数器重放会影响缓存和时间线，吞吐用独立无插桩 A/B。 [前向及逆向内核](../../kernels/cuda/stage1_ntt_dataflow.cuh) · [共享 tile](../../src/ecm/cuda/ntt/device/tile_pass.cuh) · [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
2. **按计数器选择结构改动。** 若 128–1024 点显示 CTA 有大量闲置线程，比较较小 CTA 和每块多 tile，保持每 warp 连续访问。若逆向 tile 显示 spill 或指令依赖，先缩短同时存活的频谱及根，分别编译前向与逆向候选，最后才试 kernel 专属的寄存器上限。若共享内存事务超额，再比较索引置换与 padding。若根表或输入挤出缓存，再试对齐成对读写、按 stage 重排根表或现场生成常用旋转因子。每次只改一项。 [共享外层调度](../../src/ecm/cuda/ntt/device/outer_coop.cuh) · [Goldilocks tile 原语](../../src/ecm/cuda/arithmetic/goldilocks/goldilocks_tile.cuh)
3. **按算式和批量分别测计划。** 现有 NTT 校准调用 x-only ladder，原生 Edwards 的并行乘法、非零加点列和字典驻留会改变最佳计划。先用相同的真实 NAF 列序列比较两者选出的计划，再决定是否把 Edwards 结果纳入共享 tune，按实际算式、长度、卷积和批量区分来源。前向和逆向分别搜索 tile 层数、CTA 宽度和 outer 分解。一个全数组 outer pass 对 $B$ 条长度 $L$ 的 64-bit 变换需约 $16LB$ 字节的逻辑读写，实际 DRAM 流量取决于缓存。入围结果看两种批量下完整变换和整批 Stage1 的时间。 [当前校准入口](../../src/ecm/stage1/montgomery/cuda/ntt/tune.cpp#L164) · [Stage2 outer 的硬件对照](../performance/NTT_OUTER.md#管理员剖析)

首轮可只取 M8191、M106531 和 M262139，每个指数各测 1／30 曲线。M8191 看短 tile 的线程利用率，M106531 看 5120 点 PFA 与去权缓存，M262139 看 16384 点逆向及 outer。有初步归因后，再按下文矩阵扩测。

## 2. 让 2XY 和 carry 少搬中间数据

- **继续减少宽档频域组合的读写。** 256／512 点已使用逆向 warp 读入时的直接计算。5120／16384 点的组合 kernel 仍可试共享平方结果。一个候选是先逆变换 $J=2(C-Z^2)$，归约 $F$ 时读取并加上已逆变换的 $G$。相对逐 CTA 重算四个坐标的 8 次域乘，直接逆变换 $E,J,G,H$ 共用 6 次；与独立组合的 4 次相比仍多 2 次，但可能省一次 kernel 提交。先对照新增系数读取与同步成本。 [频域组合](../architecture/NTT.md#edwards-2xy-第一层频域融合) · [双卡对照](../performance/STAGE1_MERSENNE_NTT.md#原生-edwards-2xy-第一层频域融合)
- **测试 carry 与下一次前向首层融合。** 宽位已有分段进位及 pack 摘要解析。先选只被下一层变换消费的槽，尝试避免规范值全局写回再读入。若共享内存驻留或同步抵消收益，则保留旧路径。完成点、检查点及仍被整数运算消费的坐标使用规范 limb。 [分段进位与 pack](../architecture/NTT.md#stage1-分段进位与-pack-融合) · [原生 Edwards 去权缓存](../architecture/NTT.md#原生-edwards-的共享去权缓存)
- **保留有明确消费者的频谱。** 固定点字典和 x-only 的固定操作数已经复用频谱。可进一步尝试复用相同的规范输入，或在归约完成后顺带生成下一次前向的首段。去权、进位和模数折叠会改变 digit，未经归约的乘积频谱继续串乘需要另证系数界。分别计纯倍点列、非零列和按真实 NAF 序列运行的整批时间。 [Stage1 点公式的数据流](../performance/STAGE1_MERSENNE_NTT.md#stage1-点公式的数据流)

这里的 $C$ 是全满 digit 向量的加权平方。若拆开 $J$ 与 $G$，二者逐槽均在 $[0,2U]$ 内，归约前相加仍满足 $4U<q$。该候选尚待实测，优先选 16384 点多曲线和 5120 点 128 曲线。

## 3. 先算能缩多少长度，再换算术后端

当前 weighted 使用 $q=2^{64}-2^{32}+1$，以 $2L(2^{\lceil p/L\rceil}-1)^2<q$ 作为无符号乘积的保守准入界，已支持二次幂和 radix-5 PFA。先用实际 digit 宽度收紧这一界。令 $e_i=\lceil pi/L\rceil$、$d_i=2^{e_{i+1}-e_i}-1$。对输出 $k$，枚举 $0\le i<L$，每项以 $j=(k-i)\bmod L$ 及 $\delta=e_i+e_j-e_k-p[i>k]$ 计算。无符号系数的逐槽上界为

$$
U=\max_{0\le k<L}\sum_{i=0}^{L-1}2^\delta d_i d_j.
$$

离线整数核算给出 M65521 在 2560 点的 $U/q\approx0.742$，旧保守界约为 $1.25q$，因此值得试 4096 → 2560。M4423 在 160 点的 $U/q\approx0.801$，但当前生产的混合长度下限为 2560，短形状需另外评估。M262139 在 10240 点仍约为 $2.997q$。这三个数只描述规范无符号操作数的系数容量。较短计划可能需要关闭 x-only 松弛差值，也可能无法使用 2XY 频域组合。应单列比较「长度更短」和「现有长度、融合更多」的完整耗时。生产准入可按长／短 digit 及 $\delta\in\{0,1\}$ 计数，避免逐对枚举。 [当前加权界](../performance/STAGE1_MERSENNE_NTT.md#加权循环卷积) · [长度准入](../../include/ecm/stage1/ntt_length.h)

若尝试新的 PFA 分解，$L\mid(q-1)$ 只保证有单位根。加权构造还需要 $\rho^L=2^p$。Goldilocks 的 $q-1$ 只含一个因子 3，而 2 的阶含因子 3。对通常不被 3 整除的 $p$，$3\times2^k$ 无法直接沿用当前加权构造；线性卷积可单独评估。 [加权卷积与 radix-5 PFA](../architecture/NTT.md#stage1-radix-5-pfa)

| 候选 | 值得尝试的条件 | 主要成本与验证 |
| --- | --- | --- |
| **3161，双域精确 NTT** | 第一个精度扩展原型。用 GPUowl 的两个二次扩域，或更适合本地 CUDA 算术的两个 NTT 素数，优先筛能减半长度的宽位形状。 | 两路变换、CRT、宽进位和工作集可能抵消长度收益。若系数有符号且绝对值至多 $U$，CRT 恢复需 $2U<Q_1Q_2$。GPUowl 的 GF($M_{31}^2$) 没有五次单位根，不能共用现有 radix-5。 |
| **3261，FP32 + GF61** | FP32 通道估计整数系数，GF61 给精确余数。用于双域未胜出的短形状，前提是该形状能提高有效 bpw。 | 若用近似值在 $r+kQ$ 中选同余类，误差须严格小于 $Q/2$，包括转换及重构误差。需要按形状、输入界和运算类型证明，或构造能验证恢复结果的精确回退。 |
| **323161，FP32 + GF31 + GF61** | 当以上两种仍无法降到所需长度时比较。 | 多一个算术通道与更多缓冲。须计 FFT、两路 NTT、CRT、carry、显存峰值及回退成本。 |
| **32-bit 素域 NTT** | 独立比较相同形状下更窄的 butterfly 和频谱。可先做单域，再对比两个 31-bit NTT 素数配合 CRT。 | 单域通常需要更窄 digit，可能增加长度。两域的 CRT 容量至多约 62 bit，适用于降低算术与单路访存成本。能否保留长度和加权根要逐形状验证。 |
| **短形状纯 FP32** | 用作低 bpw 对照，测同模数的完整模平方和两输入模乘。 | 小数残差或随机对拍无法排除恢复出了错误整数。进入生产需要误差证明，或保证结果正确的检测加精确回退。 |

例如 M262139 从 16384 点缩至 8192 点，最多需要 32 位 digit，保守无符号界约为 $2^{78}$。31+61 位双域有足够的 CRT 容量，代价是宽系数重构和进位。先测该档及一个相邻台阶的完整模乘，证明显存和时间都有可接受的余量后，才连到 Stage1 点公式。 [GPUowl 类型、容量与重构](../reference/GPUOWL_NTT.md) · [GPUowl 类型源码](https://github.com/gwoltman/gpuowl/blob/481d11ae/src/FFTConfig.h#L26-L57)

GPUowl 在消费卡上的宽位经验主要来自约 80M–150M bit。现有 3161／3261／323161 的最短原生形状从 393,216 位起，超过当前 Stage1 上限 262,144 位。GPU1 的最短固定形状复测是在低 bpw 运行，纯 FP32 较快。同指数模平方对照在 M225287 上为 ECM-ELF 35.21 µs、PRPLL FP32 88.1 µs，后者使用较大的形状，计时含少量 PRP 校验。新的短形状需按相同指数、设备、批量和进位边界重测。仓库已有的精确整数 Tensor 映射跑完整卷积落后于生产 CUDA，优先级低于上述候选。 [GPUowl 最短形状实测](../reference/GPUOWL_NTT.md#最短原生形状复测) · [同卡完整模平方](../performance/STAGE1_MERSENNE_NTT.md#同卡完整模平方与-prpll) · [Tensor NTT 完整卷积](../performance/NTT_TENSOR.md#完整卷积随长度变化)

## Stage1 统一测试口径

| 层级 | 计时与验收 |
| --- | --- |
| 共用算子 | 正向、逆向和完整模平方／模乘分列。模乘覆盖 pack、两个不同输入的前向、逐点乘、逆向、去权和 carry。模平方复用一个前向。报告单路 µs／次和多路操作／秒。 |
| 固定 Stage1 计划 | GPU1 RTX 4060 Laptop 首筛，GPU0 RTX 4070 Ti 复验。短档选约 4K／8K／16K bit；中档覆盖 2560／5120 点及启用边界；宽档覆盖 8192／10240／16384 点。每格先测 1／8／30 曲线，满足显存准入且吞吐仍有提升时补 64／128。固定指数、B1、sigma、Edwards w8、卷积、长度、tile、pass 和 Graph 窗口，单项消融只改一个条件。探针预热 100 列或步后重置，基线与候选交替至少三批，报中位数和离散度。 |
| 生产选型 | 使用同一实际任务，允许各版本自行选批量与形状，另用较长 B1 复验。记录曲线／主循环秒、曲线／整批墙钟秒、µs／列／曲线、字典与初始化、显存峰值和活跃频率／功耗。收益须超过重复噪声。若单曲线或另一张卡退化，按验证过的设备和批量启用。 |

若 $C_b$ 为实际曲线数、$J$ 为每条实际执行的列数或 ladder 步数、$T$ 为整批 GPU 主循环秒数，则 $10^6T/(C_bJ)$ 为 µs／列或步／曲线。公式不同的方案比较相同 B1 的曲线吞吐。PRP 的 µs／iter 用于完整模平方对照。正确性包括独立 GMP、最大 digit、全传播进位、混合长度台阶、已知因子和非单位分母、Graph／直接提交、检查点恢复，以及 Stage2 读取 Stage1 save。 [Stage1 生产采集器](../../tools/bench/bench_stage1_ntt_production.py) · [原生 Edwards 字典采集器](../../tools/bench/bench_stage1_ntt_dictionary.py)

## Stage2 复测哪些收益

Goldilocks 域乘、tile 内层、根表访问和 outer 调度可走同一个共享实现。NAF、Mersenne carry、PFA 打包及公式融合依赖 Stage1 的输入语义。Stage2 用 Kronecker 槽构建线性卷积，随后回收系数并做 S4。Stage2 的根加权路径把可省的首层前向融入打包，逆向仍重建完整输出。换域还会改变槽的 digit 数、carry、显存准入及可选 D/P。 [共用调度](../../src/ecm/cuda/ntt/runtime.cpp) · [Stage2 打包长度](../../src/ecm/cuda/ntt/shape.cpp) · [Stage2 加权实测](../performance/NTT_OUTER.md#根加权半长卷积)

1. Stage1 通过的共享改动先测 Stage2 的 2048 点短卷积，在实际 720–2880 批量中覆盖 L2 内外。长变换从实际 $2^{21}$–$2^{27}$ 的长度与方向分布选 GPU 时间累计最高的两三个档，固定原生产计划，逐词检查完整卷积，再分别计 pack、变换、carry、S4。
2. 选普通和梅森承载各一条固定 Stage1 save，锁定实际 B2、D/P、布局、预算及队列配置，每侧至少三次交替复测。核对完整曲线、长度计数、G 根摘要、下降叶值及 GMP 抽查，另跑已知因子。共享算子有收益后，再单独验证是否值得重新选形状。
3. 用 $\sum_i f_i\delta_i$ 估算整曲线的相对省时，$f_i$ 是该长度／方向／批量的旧耗时占比，$\delta_i$ 是实测省时比例。并行队列另测关键路径和整窗吞吐。历史 $2^{25}$ 探针的 7.57% 收益在生产协作计划上消失；该档只占相关曲线约 4.8%，即便全部兑现也约为 0.36 个百分点。 [短变换真实批量](../performance/NTT_GRAPH.md#batch-与实际吞吐) · [长变换按长度分项](../performance/NTT_OUTER.md#nsight-分项与半长逆向计划) · [Stage2 分项工具](../../tools/bench/analyze_stage2_ntt_phases.py)

## 第一轮 GPU1 剖析与候选筛选

RTX 4060 Laptop 8 GiB、驱动 610.88，构建 CUDA 13.3.73／MSVC 14.51.36231／sm89。使用完整梅森模数、原生 Edwards 2XY、weighted、w8、Graph 上限 16、起始 sigma 26。B1=512 对应 747 列。字典、Graph 构建、末点回读和发布在主循环计时之外。0.63.11 基线每格六次，按三对 A/A 交替。利用率至少 50% 的采样均为 1500 MHz，基线板功耗 9.21–41.19 W。GPU0 继续原工作。

Nsight Systems 2026.1.3 每格另采一条，使用 CUDA Graph 节点与事件时间戳。用开始／结束 CUDA event 划定同一 stream 的主循环。事件区间与程序报告最大相差 0.049 ms。下表各列是 kernel 累计时间，单位 ms。carry 包含模加减、去权、limb 汇集和跨段写回。Pack 包含独立 PFA，融合打包计入前向 tile。

| 指数 | 曲线数 | NTT／tile 层 | 前向 tile | 逆向 tile | Outer | Carry | Pack／PFA | 事件区间 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8191 | 1 | 512／9 | 7.55 | 7.19 | 0 | 14.30 | 0 | 46.03 |
| 8191 | 30 | 512／9 | 13.93 | 13.74 | 0 | 21.79 | 0 | 64.38 |
| 106531 | 1 | 5120／10 | 8.32 | 9.15 | 0 | 22.15 | 11.97 | 85.27 |
| 106531 | 30 | 5120／10 | 94.00 | 107.26 | 0 | 170.74 | 72.19 | 456.68 |
| 262139 | 1 | 16384／12 | 22.43 | 28.14 | 19.29 | 29.35 | 6.26 | 134.96 |
| 262139 | 30 | 16384／12 | 300.29 | 362.84 | 168.26 | 413.07 | 99.87 | 1359.20 |

16384 点的前逆 tile 合计占 30 曲线 kernel 时间的 49.3%，outer 占 12.5%。5120 点单曲线的 carry 家族占 kernel 时间的 42.9%。宽档逆向 tile 是最耗时的单个 kernel 族，六格都使用每线程 40 个寄存器，轨迹显示 local 分配为零。10／12 层逆向 tile 每块动态共享内存分别为 8／32 KiB。Nsight Compute 尝试返回 `ERR_NVGPUCTRPERM`，因此本轮无法判定实际 bank 冲突、缓存命中或指令 stall 的贡献。

初版候选让 warp 开头的 9／10／12 层路径分别以 radix-2 或 radix-4 直接写全局。B1=512 和 4096 各做六格、每侧三次交替，配对保存点逐条一致。16384 点两档 B1 的单曲线主循环减少 2.00%／1.64%，30 曲线减少 0.97%／0.96%。5120 点 30 曲线减少 0.87%／0.81%，单曲线差值在 ±1% 内。512 点的 30 曲线在两档分别快 1.11% 和慢 1.31%，A/A 的两组中位数也相差 2.78%。去掉 9 层直写后，10 层单曲线在短档对测中曾慢 2.54%。最终用独立的末级 radix-4 直写 kernel，仅在 12 层和至少 24 条活跃曲线的 10 层 tile 时分派。 [0.63.12 最终对测](../performance/STAGE1_MERSENNE_NTT.md#逆向-tile-末级直写)

原始记录位于 `data/experiments/stage1_ntt_first_round_06311/`，含 `baseline-repeats/`、`systems/`、`direct-ab/`、`direct-holdout/`、`systems-direct/` 及本地汇总脚本。阶段分项用 Nsight 的单轨迹，性能差值用独立无插桩采样。

## 第二轮 短加权形状

0.63.13 源码按大小 digit 和去权因子对系数做常数时间的整数准入。M65521／2560 的系数界约为 $0.742432q$，足以容纳单个无符号乘积。该形状的 $2U>q$，松弛差值不能套用同一证明。原生 Edwards 2XY 的 weighted 路径和独立平方探针使用收紧界。生产路径由 `ntt.tight_weighted_bound` 控制，默认开启。卷积和长度都为自动时，原生 Edwards 先检查只有收紧界才能准入的短计划，其他情况继续使用 x-only 的保守调优计划。异常重放需要时重新推导长度。 [闭式系数界](../architecture/NTT.md#stage1-加权卷积的无符号系数界)

独立的整数枚举与闭式计算比较了 3277 个小形状、M65521／2560 的全部系数，另对 20,000 个机器字累加输入验证了溢出检查。GPU1 GMP 探针覆盖全宽 digit、窄顶 limb、直接和 Graph 提交、分段与非分段归约，以及 2560／5120／10240 的相邻接受与拒绝例子。例如 2560 点接受 $p=66048$，拒绝 $p=66049$，又接受 $p=66050$。因此程序以实际指数验证，不能在调优表中用一个连续的指数上限表示扩展范围。x-only 探针按原有界拒绝 M65521／2560。

整数审计、GMP 边界和初轮数据保存在 `data/experiments/stage1_ntt_weighted_bound_06312/`。两卡扩测的脚本、执行日志、端点 GMP 对拍和自动分派记录位于 `data/experiments/stage1_ntt_weighted_matrix_06313/`。 [双卡加权自动选长矩阵](../performance/STAGE1_MERSENNE_NTT.md#双卡加权自动选长矩阵) · [GPUowl Z 和 ROE](../reference/GPUOWL_NTT.md#运行日志中的-z-和-roe)

## 第三轮 Edwards 进位与打包

0.63.14 让原生 2XY NAF 使用已有的跨段进位与 pack 融合。按下一步消费者选择需要独立写回的中间值。在 M106531／5120 点上，747 列少 913 次写回。30 曲线的 pack 成本上升，但小于写回节省。两卡长短 B1 对照、波动复测、Sanitizer、检查点和 Stage2 交接均留在 `data/experiments/stage1_ntt_edwards_deferred_carry_06314/`。 [分段进位](../architecture/NTT.md#写回与下一批打包) · [双卡结果](../performance/STAGE1_MERSENNE_NTT.md#原生-edwards-延迟进位)

## 第四轮 Edwards 2XY 第一层频域组合

0.63.15 使用全满 digit 卷积的频谱做模零偏置。在第一批逆向前逐点写出 $E,F,G,H$，继续用非负归约。两卡在原长度下对照 0.63.14，覆盖 256–16384 点及 1、8、30、128 曲线。另验证 $4U/q$ 在 1 两侧和 M65521 的同长度回退。实测覆盖直接提交、Graph、断点续算和 Stage2 交接。证据保存在 `data/experiments/stage1_ntt_edwards_2xy_frequency_06315/`。 [正系数偏置](../architecture/NTT.md#edwards-2xy-第一层频域融合) · [双卡结果](../performance/STAGE1_MERSENNE_NTT.md#原生-edwards-2xy-第一层频域融合)

## 第五轮 短逆变换读入与主机求逆

以 0.63.17 的 Stage1 路径为基础，比较先写四路频谱和由逆向 CTA 自行计算四路。原型支持强制在宽档运行，用两卡样本区分一次提交的节省和重复域乘的代价。256／512 点的测试收益在两卡同向，默认覆盖同系数界的单 tile 计划。5120／16384 点的多曲线结果在 −0.94% 至 0.87% 之间。主机构造分开测量，合并同一曲线的四项分母可减少参数映射阶段的逆元计算。原始结果保存在 `data/experiments/stage1_next_optimization/`。[短逆向实测](../performance/STAGE1_MERSENNE_NTT.md#逆变换读入时组合倍点频谱) · [构造实测](../performance/STAGE1_MERSENNE_NTT.md#参数映射的批量求逆)
