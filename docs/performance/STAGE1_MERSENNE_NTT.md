# 少曲线全宽 NTT Stage1

独立 CUDA 实验入口可对完整梅森模数 `M=2^p−1` 运行 Suyama Montgomery PARAM0 Stage1。所有点坐标在 GPU 驻留，一条曲线也会将同层的独立乘法一起提交。GPU1 RTX 4060 Laptop 上，8191／16001 位的单曲线设备阶段短于本机 CGBN 对照；把 CGBN 批量提高到 192 条后，CGBN 的吞吐仍更高。新路径通过了 131071 位的整条曲线验算，超过所用生产 CGBN 对照的位宽范围。

当前是独立实验程序，不是生产 Stage1 的可选后端。入口只接受完整 `2^p−1`、PARAM0、`lcm(1…B1)` 和 1–8 条曲线，不接入生产任务队列、save 发布或断点恢复。可输出真因子或归一化 Montgomery x；全模数 GCD 单列。测试过 262144 位的人工算术状态，整条曲线性能矩阵最高为 131071 位。

## 实现与正确性

对 p 位操作数选择最大整数 digit 宽度 b，令 `c=ceil(p/b)`，要求 `c·(2^b−1)^2 < 2^64−2^32+1`。变换长度 `L=nextpow2(2c+1)`，所以 Goldilocks 单模数足以精确恢复每个卷积系数，不会把线性卷积误做循环卷积。生产 NTT 模块处理 forward、点乘／逆缩放、inverse 和进位；其输入打包是完整整数，独立于 Stage2 的多项式槽宽和 D/P。

进位后的乘积按实际第 p 位拆成低半和高半。用 `2^p≡1 (mod M)` 折叠成一份残数，末 limb 即使不足 32 位也以实际位宽传播。每个 limb 将输入进位映射为 kill、propagate 或 generate；warp 票选加块内前缀求出回卷进位，全 propagate 的和 M 归一为 0。减法复用同一运算：`a−b ≡ a+(M−b)`。整个模乘没有 CGBN REDC，也没有中间值经 PCIe 传给 GMP。

标量按二进制 ladder 扫描，起始寄存 `[1]P` 和 `[2]P`。每位计算一组 xDBL 和 xADD，共四次平方和六次乘法，按依赖拆成三层：第一层同时算平方及交叉项并共享输入频谱，第二层算两平方、倍点的 X 及 `a24·E`，第三层完成两点的 Z。每位实际变换为 12 次 forward、10 次 inverse，分别组织为三层批处理；`a24` 和差点仿射 x 的频谱只生成一次。Graph 默认测 16 位窗口，余数用独立短图；标量索引在设备上更新，无逐位回读。直接提交用相同算法。

实现见 [ladder](../../src/ecm/stage1/montgomery/cuda/ntt/ladder.cu#L159)，其中 [回卷与规范化](../../src/ecm/stage1/montgomery/cuda/ntt/ladder.cu#L96) 与 [独立 GMP 整曲线对照](../../tools/test/stage1_mersenne_ntt.cpp#L27) 分别负责设备归约和验算。显式驻留数组规模为 `15·8·B·L + 23·4·B·ceil(p/32)` 字节，另有标量、缓存表、Graph、CUDA 上下文等；这不是进程峰值。

- 32／33／64／65／8192／8193／262144 位人工状态：覆盖对齐与非对齐边界、全宽进位、减法为零，直接与 Graph 的两组 X/Z 完全一致，并与 GMP 一致。
- 8193 位人工状态的直接与 Graph 路径通过 Compute Sanitizer memcheck 和 synccheck，均为 0 errors，日志见 `sanitizer_local/`。
- `M65`、B1=40：sigma=11 在 Stage1 得到真因子 `253921=31×8191`；sigma=12 构造即得 `31`。同批构造命中槽剔除后保留其余曲线的 sigma 身份。
- 32749 位，起始 sigma=4294967311，4 条曲线通过 GMP 核对。所有 NTT 计时样本先核对跨样本输出，再核对 GMP 的两个 X/Z 点。8191／16001 位的 1／4 曲线在 CGBN 独立保存点上再做一次归一化 x 核对。

## 长度、pass 和 GPUOWL 形状

本实验先从 31 到 16 找满足精确性不等式的最大整数 b，再取最小满足 `L≥2c+1` 的二次幂。两段 c-digit 乘积最多有 `2c−1` 个非零卷积系数；`2c+1` 是实现预留的零尾，供进位与提取。以 8191 位为例，b=27、c=304、`L=1024`。长度只随模数位宽和精确性条件变化；B1、曲线数、Graph 窗口和 tile 都不改变 L。Stage2 多项式 packing 的槽宽模型不用于这里。

设 `k=log2 L`，Goldilocks 域的正向根取 `ω=7^((Q−1)/L) mod Q`；逆根为 `ω⁻¹`。每个频域位置存一个 64 位域元素。每次正向用 DIF，逆向用 DIT，沿同一内部频域排列做点乘，因此不单独做一次全局位倒序。`tile=t` 是最后 t 层的融合尺寸，单块处理 `2^t` 个元素，占 `8·2^t` 字节动态 shared memory；现有实验中 tile block 为 512 线程。此前的 `k−t` 层拆成 outer pass，实验限制每个 outer 最多融合 4 层，radix 最大 16；根表按形状缓存。tile 内用 shared radix-4/2 与 warp shuffle 尾部，逆向 tile 同时完成点乘与 `L⁻¹` 缩放。正向 `outer→tile`，逆向 `tile→outer`；当 `t=k` 时，每个方向只需一个 tile pass。见 [计划](../../src/ecm/cuda/ntt/runtime.cpp#L345)、[变换调度](../../src/ecm/cuda/ntt/runtime.cpp#L602)及 [tile](../../src/ecm/cuda/ntt/device/tile.cuh#L157)。

GPUOWL/PRPLL 本地副本 `cb560c9` 使用的是另一套大整数表示。`W:M:H` 中的 W/M/H 是变换维度，`NWORDS=2WMH` 是整数 word 数，一对 word 进入一个复数或扩域元素，`ND=WMH` 才是这层的数据元素数。实际平均 `bpw=p/NWORDS`，由相邻的大小字实现，iBDWT 权重让卷积在梅森模数上直接循环。`fftbpw.h` 存的是**容量上限**而非实际每字位数：按 FFT/NTT 类型、形状和 W/M/H 变体查 `maxBpw()`，还受 carry 宽度限制；`bestFits()` 用 `maxExp≈NWORDS·maxBpw` 与 `minBpw` 筛选形状，再优先选择 `tune.txt` 里成本最低的可用项。此副本 `256:1:256` 已是 131072 个整数 word、65536 对，表里的 `19.204` 等也不是 `log2` 尺寸或我们可移植的 digit 参数。

| 阶段 | ECM-ELF 本实验 | GPUOWL/PRPLL 本地副本 |
|---|---|---|
| 前端 | 固定整数 b-bit digit，打包、补零；先做外层 DIF | `fftP` 加大小字权重并做 width 正向 |
| 中端 | 可选 outer pass 融合高位层；shared/warp tile 完成最内层；inverse tile 融合频域乘积与缩放 | `fftMiddleIn` 做中间方向、twiddle 和转置；`tailSquare` / `tailMul` 做 height 正向、配对运算、height 逆向；`fftMiddleOut` 逆中段与转置。本地部分奇数 MIDDLE 可用 PFA |
| 后端 | 逆 outer，精确进位，按第 p 位高低折叠并规范化，输出整数坐标 | `fftW` 及 carry 路径逆 width、逆权重、按类型做舍入或 CRT、回卷进位。连续平方可用 `carryFused` 输出下一轮已完成 width 正向的状态 |
| 数域 | 单 Goldilocks 素域，按整系数上界保证精确恢复 | 可选 FP64、GF61、GF31/GF61 与浮点混合等；按类型走扩域根、CRT 和/或浮点舍入 |

两者都融合多层 butterfly、缓存 twiddle、尽量减少显存往返，也可以减少提交开销。本实验的 Graph 减少提交，跨点运算依赖层仍回到规范整数；GPUOWL 的 `LEAD_WIDTH`/`LEAD_MIDDLE` 是实际数据状态，能跳过已经计算的变换阶段。不要把两边同名 `tail`、`bpw` 或长度数字当成同一单位。细节与本地源文件索引见 [GPUOWL NTT](../reference/GPUOWL_NTT.md)。

## GPU1 实测

RTX 4060 Laptop、24 SM、8 GiB、驱动 610.88，NTT 实验源码版本 0.58.11、CUDA 13.3.73／sm89；CGBN 使用本机保留的 0.57.11 worker。GPU0 未参与。没有修改设备频率或功耗设置；主矩阵以 200 ms 间隔记录整卡遥测，利用率≥20% 的 98 个样本为 1500 MHz、2.34–14.71 W、61–68 °C，功耗上限查询 N/A。功耗是设备采样值，不能当作单次曲线功耗。

下表使用完整 `2^p−1`，B1=256、LCM、sigma 从 11 开始；标量去掉最高 1 后有 362 位。NTT 先以单曲线在 B1=41 筛 tile，再对同位宽的 1／4 曲线使用同一 tile。每格 3 个独立进程，NTT 每进程 3 个 event 样本，CGBN 每进程 1 个；不丢弃首样本。表中单位 ms/批，先取每个进程中位数，再取三个进程中位数。

| p | NTT 长度 / digit 位宽 | NTT 1 条 | NTT 4 条 | CGBN 1 条 | CGBN 4 条 | CGBN 192 条 |
|---:|---:|---:|---:|---:|---:|---:|
| 8191 | 1024 / 27 | 27.046 | 29.134 | 106.850 | 101.737 | 145.319 |
| 16001 | 2048 / 27 | 34.360 | 45.864 | 268.389 | 271.823 | 392.180 |
| 32749 | 4096 / 26 | 48.387 | 76.149 | — | — | — |
| 65521 | 8192 / 26 | 80.860 | 127.491 | — | — | — |
| 131071 | 16384 / 25 | 148.197 | 272.128 | — | — | — |

![整曲线设备阶段延迟与吞吐](attachments/stage1-mersenne-ntt-4060.png)

NTT event 包含驻留点乘、打包、变换、进位、折叠及 Graph 窗口之间的空隙；排除初始曲线、CUDA 初始化、频谱／Graph 准备、结果回读和 GMP。CGBN 的 `gputime` 是原 worker 的全局 event，包含设备准备和切片间隔，但不含最终 save。因此两列适合比较各自设备阶段，不能当作纯 kernel 或完整应用墙钟的等边界倍率。原始数据另留 NTT 准备、回读、CPU 验算及两条路径的进程墙钟；GMP 是正确性参考，未作为 CPU 性能基线。

四曲线使 NTT 相对一曲线的摊销吞吐提高约 2.18–3.71 倍，代价是批次延迟增加。8191／16001 位的 CGBN 192 曲线分别约 **0.757／2.043 ms/曲线**，仍低于 NTT 四曲线的 **7.283／11.466 ms/曲线**。直接提交的 8191／32749 位单曲线中位数为 112.380／159.892 ms；相同形状的 Graph 降至 27.046／48.387 ms。32749 位、B1=5000、7210 个 ladder 步的单曲线 event 中位数为 **0.906 s**（一进程、三样本）。以上没有测出 16K 以上的 CGBN 整曲线交叉点。

为减少准备成本在短任务中的权重，另用 B1=5000、sigma=11 复测单曲线。8191 位 NTT／CGBN 为 **0.533／1.737 s**，16001 位为 **0.665／5.139 s**。取样和计时边界与主矩阵相同，各条件的归一化点仍一致。B1=256 观察到的单曲线延迟优势在这组更长标量上仍成立。

## 并行曲线数

在 GPU1 上对每个 `L` 独立扫描 **1…8** 条曲线。先用 B1=13 从 tile 6／8／10／12（超过 k 时截断、去重）筛每个批量，再用 B1=41、57 个 ladder 步、窗口 16 做复测。复测每格 2 个进程、每进程 3 个 event 样本，两轮反向遍历曲线数；中位数取法和设备边界同上。176 个筛选进程与 96 个复测进程都与 GMP 核对。设备仍为 4060 Laptop GPU1、版本 0.58.11；200 ms 遥测中利用率≥20% 的 207 个样本均为 1500 MHz，整卡功耗 1.76–11.59 W、60–67 °C。原始数据见 `batches_final/summary.json` 与 `batch_gpu1_telemetry.csv`。

设 `T(B)` 为整批耗时：追求尽快返回，最小化 `T(B)`；追求每秒完成的曲线数，最大化 `B/T(B)`。这六档在已支持的 1…8 条内均是 **1 条最低批次延迟、8 条最高吞吐**。8 是实现的当前上限；该扫描不说明 9 条以上会退化。

| L | 模数位宽 | b | 1／4／8 条批次 ms | 8 条吞吐 / 单条 | 达 8 条吞吐 90% 的最小批量 | 延迟不超单条 1.25 倍的最大批量 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4003 | 28 | 3.730／3.873／4.257 | 7.01× | 8 | 8 |
| 1024 | 8191 | 27 | 4.622／4.960／6.174 | 5.99× | 8 | 6 |
| 2048 | 16001 | 27 | 6.420／8.285／10.824 | 4.75× | 8 | 3 |
| 4096 | 32749 | 26 | 10.593／17.018／23.292 | 3.64× | 7 | 2 |
| 8192 | 65521 | 26 | 24.726／38.858／54.472 | 3.63× | 7 | 1 |
| 16384 | 131071 | 25 | 60.960／109.868／165.832 | 2.94× | 6 | 1 |

![不同 NTT 长度下的批次延迟与吞吐](attachments/stage1-ntt-batch-sweep-4060.png)

曲线数增加的是同一 stream 中每个算术阶段的 `grid.y`，没有为每条曲线建 stream。每位的三个 forward 批量是 `4B / 5B / 3B`，inverse 是 `4B / 4B / 2B`。一次 tile pass 的块数等于该次变换数乘 `L/2^t`。例如 `L=512,t=9`，第一层单曲线只有 4 块，8 条可给 32 块，让 24 SM 有更多独立工作；`L=16384,t=12` 时这一层单曲线已经有 16 块。较大的 L 还提高每条曲线的显存流量和归约工作，扩批的边际收益更小。较小 tile 增加块数但可能多一次 outer pass；因此 tile 也需按 `(L,B)` 筛选。比如 `L=2048` 的实测 1 条选 t=8，8 条选 t=11。

对最长的 `L=16384` 另以相同模数、B1=256、362 个 ladder 步复测（各 1 进程、3 样本，均核对 GMP）。1／6／7／8 条的批次为 148.333／346.801／398.682／448.982 ms，吞吐分别为 6.74／17.30／17.56／17.82 条/秒。6 条已达 8 条吞吐的 **97.1%**，批次延迟少 **22.8%**，适合作为该长度的折中；仍要最高吞吐时选 8 条。该复测的 18 个高利用率（≥20%）遥测点为 1500 MHz、3.24–21.69 W、62–66 °C，证据见 `batches_long_l16384/`。

复现批量扫描可用 `tools/bench/bench_stage1_mersenne_ntt.py --mode batches --rounds 2`，同时指定 `--probe` 与新 `--output` 路径。上表为短标量下的选型数据；最长档的一曲线平均每步由 B1=41 的约 1.07 ms 变为 B1=256 的约 0.410 ms。本轮没有隔离 Graph 尾部和中间操作数（包括进位链）的各自成本，不把短样本每步耗时直接外推到任意 B1。

## 参考与下一步

- Prime95 31.06b01 的 `ecm.cpp`（约 1982–2156 行）利用延迟 FFT、变换保留和融合乘加减少点公式的重复 FFT；可借鉴调度与数据流，不能直接套用它的浮点时序。当前 NTT 实现仍会把每个产品归一化后再进入下一层；进一步可把频谱索引合进 inverse 的首 pass，省掉 gather 的全局写回。
- GPUOWL 本地 `.refactor/gpuowl/src/fftbpw.h` 和 `src/cl/carryutil.cl` 用有效位宽、舍入误差门槛和进位来约束 iBDWT。加权循环变换有机会去掉线性卷积的补零和折叠，但任意 p 的大小字权重、乘法和复合点式都须单独验证；不能用一次 LL 平方的成绩推算整条 ECM 曲线。当前路径采用整数精确性上界，未测 FP32／FP64 FFT。
- 完整梅森模数才可直接按第 p 位回卷。对已剔除因子的余因子，可研究把模余因子构造的值提升到完整梅森承载环，再按实际余因子验证/GCD；通用整数则需要非二次复杂度归约。两者均未接入。生产接入还需资格验证曲线身份、save 与可恢复窗口；之后再评估 PRAC 或 Edwards 的整曲线成本。

## 复现

```powershell
pwsh -NoProfile -File tools/build/test/build_stage1_mersenne_ntt.ps1
build_dev/stage1_mersenne_ntt/stage1_mersenne_ntt.exe 1 32749 5000 11 1 3 8 16 print-x
python tools/bench/bench_stage1_mersenne_ntt.py --probe build_dev/stage1_mersenne_ntt/stage1_mersenne_ntt.exe --output data/experiments/stage1_mersenne_ntt_check
```

参数依次为设备、指数、B1、首 sigma、曲线数、样本数、tile、Graph 窗口；窗口 0 为直接提交，`print-x` 可省略。采集器 `--mode bench --cgbn <worker.exe>` 可复测矩阵。证据目录 `data/experiments/stage1_mersenne_ntt_20261109/`，主矩阵为 `matrix_final/summary.json`，长标量对照为 `long_compare/summary.json`；正确性见 `qualify_final/`，设备遥测见 `gpu1_telemetry_final.csv`。[开发草稿](../development/STAGE1_MERSENNE_NTT_DRAFT.md)保留本轮实现与验证记录。
