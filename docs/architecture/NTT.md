# Stage2 精确 NTT 与系数归约

## 数学合同

域素数 q=2⁶⁴−2³²+1，支持长度 L=2ᵏ、k≤32 的单位根。项目以精确整数 NTT 实现系数多项式乘法，不依赖浮点误差控制。Goldilocks 模数 q 与待分解整数 N、算术承载 M 相互独立。

将 S-bit 系数放入足够宽的 Kronecker 槽。最大 operand 含 A 个系数时，slot_bits=2S+max(1,⌈log₂A⌉)；实际 shape query 决定 digit 宽 b、slot_words、补零与 L。不能用手工近似替代执行端的 `choose_cfg`。

每个整数卷积系数须满足 Lterms·(2ᵇ−1)²<q，随后按 b-bit digit 精确进位。槽位上界和 carry 检查防止循环卷积别名、丢失高位及系数串槽。强制 digit 宽同样必须通过精确性检查。

## 一次多项式乘法

1. 按形状将 A/B 系数 pack 到 digit 工作区，补零到 L。
2. A、B 正向 NTT；使用 tile 与 outer 多 pass，按同一索引/根布局。
3. pointwise 乘法、逆归一化与 inverse 结合，结果写回 A 工作区。
4. carry 将域系数重建为普通整数 digit。
5. 按槽提取乘积系数，S4 归约到 [0,M)，返回后续树/fold/下降。

工作量约为三次 O(L log₂L) 变换加 O(L) pointwise/carry、pack 和系数归约。每 butterfly 的访存/指令取决于融合层数、shared 布局和根缓存；该渐近式不能直接换算 GPU 周期数。

独立生产程序默认使用pack/zero融合：`stage2_pack_zero=true`，可用false或`--pack-zero 0`关闭；共享/Worker TOML、CLI和环境覆盖见[配置参考](../ECM_TOML_REFERENCE.md#stage2_pack_zero)。普通上传、resident gather及reverse-B按最终digit地址一次覆盖完整工作区，融合有效digit写入与所有补零；无需额外设备数据数组，不改变数学NTT或carry/归约。驱动把解析值传入曲线及tune子进程，底层探针仍使用显式`NTT_S4_PACK_ZERO=0|1`。索引合同、两卡450条正式曲线、复测和默认资格统一见[准备与同步](../performance/STAGE2_PREPARATION.md#packzero-融合路径)，共享内核位于[s4_input_pack.cuh](../../src/cuda/stage2/s4_input_pack.cuh)。

## 当前算术与调度

生产使用固定 PTX Goldilocks 归约和 canonical 减法。2⁶⁴≡2³²−1 mod q 将 128-bit 乘积折成少量显式进位链；规范范围保证专用减法成立。后端由编译身份确定；可移植tune摘要默认允许忽略backend条件复用，跨后端时间不能据此声明为本机实测。旧.cprof保持独立的严格后端合同。

tile 在 shared 中融合若干层，outer 按 radix/pass 分解大变换，缓存前向/逆向 pass 与 radix 根表。生产使用自动 coop 选择；显式 radix 比较需要开发引擎。outer V 收窄、展开、逆 scale 移位及 carry/check 融合仍有可选实验实现，不能把局部结果当作默认收益。

自动表另提供默认关闭的 k21–23 M6 实验开关 `NTT_FUSE_MID_COOP=1`，限定4060 Laptop及当前tile/warp/compact条件。它与任意 radix 覆盖不同，显存模型与执行共用同一计划描述；条件、验证和完整曲线结果见[outer性能](../performance/NTT_OUTER.md)。

`NTT_FUSE_JOINT=1` 默认关闭，在相同受支持条件下包含中段M6，并选k23 tile11/M6、k25 tile12/M7；其他长度不变。当前1428算子样本与80条固定D曲线仅显示约0.19%的额外矩阵收益，未启用默认。[准备与同步报告](../performance/STAGE2_PREPARATION.md)说明默认关闭的 `NTT_HOST_TRACE=1` 诊断范围，以及后续pack/补零融合的证据。

NTT性能表的字段、计时边界、tile级数、外层radix与处理趟数，以及摘要如何影响D/P选型，统一见[NTT性能摘要表](AUTO_B2.md#ntt-性能摘要表)。当前NTT tune校准成本，不自动搜索或部署内核配置。

Tensor Core 的整数拆分路线有独立实验实现，未成为生产默认。仅满足小字长 FHE 模数的库不能直接替换 Goldilocks64/多精度 carry 管线；适配参考见 [算法资料](../reference/ECM_ALGORITHMS.md)。

当前独立实验提供Tensor64及稀疏有符号根的Tensor40，两者在生产outer/根表上比较完整变换和卷积，另提供Graph提交诊断。它们未接入运行或tune选型；已测长度、batch、算术资格与负收益见[Tensor NTT性能](../performance/NTT_TENSOR.md)，精确分解见[数学资料](../reference/TENSOR_NTT_MATH.md#有符号稀疏根与-dft8-分解)。

实验另有96-bit重构及64-register约束入口N96r64，完整GMP和同卡CUDA/Tensor双流对照已通过。两任务使用私有A/B与只读根表，Systems确认内核重叠，Compute按整个应用范围采集；当前仍慢于同工作量的纯CUDA控制。该探针不代表生产已支持多曲线并发，详见[单卡并发](../performance/NTT_TENSOR.md#单卡cuda--tensor并发与96-bit候选)。

## S4 系数归约

通用 M 使用已归一化除数与 2-by-1 reciprocal 长除法，直接求乘积系数余数。主乘减量级由双重消去降为约 W² 的主项，加商估计、校正和位移；真实数量由槽长度决定，不是所有输入恰好 W²。

M=2ˢ−1 时使用 2ˢ≡1 mod M 的折叠归约；点 Montgomery 乘法还需处理 R⁻¹ 对应的旋转/表示转换。承载模式保证 M 能被目标 N 整除，所有因子语义仍在 N 上处理。

S4 形状常量按 slot_bits/slot_words/b 缓存，不能只用 slot_bits 作为身份。首次形状强制验证 96 窗口；普通批次的 canonical counter 和异步 oracle 路径必须排空，错误不可被日志等级屏蔽。

## 缓冲与分块

NTT pool 保留最大工作区容量，shape 小缓冲/表按键缓存。非导出 digit 的正常 pool 路径，inverse 后 B 无读者，carry 输出 Q 可借用 B；物理 A/B 为两份，Q 不拥有第三份。非适用路线保持独立 Q 或按次分配。

默认 S4 请求按 batch 预算执行整数减半分块，并保留至少一个 slice；因此 soft batch cap 不保证单请求一定能放入预算。grid.y 超过设备上限时继续物理拆分，模型与执行须使用相同 inner batch。分块改变内存与 launch，不改变数学请求集合。

短 NTT 的实验性分块选型见 [Auto B2 / tune](AUTO_B2.md#短-ntt-批处理选型实验选项)。它使用同长度 batch 表和完整尾批估算，并同步驱动请求/NTT/S4内存规划，默认关闭。[固定 D 冻结实验](../STAGE2_SHORT_NTT_BATCH_DRAFT.md)未显示稳定收益；较小batch的独立NTT吞吐优势不能覆盖整条S4流水线的成本。

短NTT还提供默认关闭的Graph提交缓存，环境变量 `NTT_SHORT_GRAPH=1` 显式启用。每项两个Graph合并A/B forward与融合inverse，重放仍在原默认流；仅借用稳定workspace与已缓存根表，在其释放/增长/驱逐前使缓存失效。它不改变pack、carry、S4、分块或数学NTT数，驱动Graph内部内存另于NTT payload账本。固定D小/大P、普通/承载、不同batch及owner回退的完整验证和限制，统一见[Graph性能](../performance/NTT_GRAPH.md)。当前完整曲线未显示稳定收益，不自动部署。

### 实际运行的长度与运算量

生产 Stage2 通过 `stage2_ntt_counts=1` 开启按长度计数，默认 0，关闭时不采集或输出长度表；CLI `--ntt-counts 0|1` 覆盖选定 worker 的 TOML。Edwards/Montgomery 使用相同入口。实验引擎不支持此选项，开启时报错。

开启后每条曲线结束输出 `ntt_count_table` 表，按长度升序列出 length、log2、convolutions、forward、inverse、total。每个 batch 成员独立计数，每次卷积含两个正向与一个逆向，因此 forward=2·convolutions、inverse=convolutions、total=3·convolutions。Graph 重放和物理分块不改变数学变换数；它不是 CUDA kernel 启动数。表显式开启后在 quiet 控制台也输出，正常日志中保留一份。

兼容行 `ntt_length_counts: log2_length:num_of_ntt` 按 k 升序列出 **L=2ᵏ 的独立数学变换数**；debug 另输出按长度的卷积数。消费旧直方图的分析工具仍可读取此行，新运行需要显式开启上述选项。

同一张表的 `ntt_work` 汇总卷积数、数学变换数、S4分块数及下列逻辑工作量。对各长度的卷积数 cₖ：

- transforms=3·Σcₖ。
- elements=3·Σ(cₖ·2ᵏ)：全部正向/逆向处理的元素总数。
- logical_radix2_butterflies=(3/2)·Σ(cₖ·2ᵏ·k)：等价 radix2 蝶形数量；融合 radix 实际指令数不等于此数。

计数包含经过多项式层的 F/G 树、inverse、fold和下降请求；不包含底层算术自检内部独立发起的变换，也不含 Stage1。每个生产 worker 进程处理一条曲线，多曲线 save 逐条输出各自的表，每条都包含本曲线初始化，不是整份 save 的累计表。缓存工作区容量、batch×length 与单条长度分别记录，不能互相替代。

计数器使用主机固定 33 档数组，在已完成的多项式调用处更新，不增加设备计数、显存分配、CUDA event 或同步。关闭时在进入长度分档前返回。计数采集成本属于原生 Stage2 时间；结束表的格式化/输出发生在主计算计时结束后，另用进程墙钟检查该部分。

0.32.0 在 GPU1 RTX4060 Laptop 上固定 B1=50000、B2=27,015,978,990、D=180180、普通余因子模式，383/10232-bit 输入的开关各测 5 次，交替顺序、首轮保留。383-bit 原生 Stage2 中位秒数关闭/开启为 0.525023/0.532992（+1.52%）；10232-bit 为 47.844802/47.763551（−0.17%）。配对变化分别为 −1.97%…+2.87%、−0.42%…+0.11%，未显示稳定损失，不能据此保证所有规模零开销。包含结束输出的进程墙钟中位数分别为 1.349713/1.246619 秒、48.772801/48.782988 秒，也保留全部原始样本。

该开关对照未固定或采集 GPU 功率/时钟，CPU 基准在后台运行，不与其他固定频率实验混合。程序 SHA256=`908b51f96416679e58242b20a47e69456724d4e0e0719f4390485ae2aadc06a3`；构建、输入、13 项功能门禁、原始样本及汇总位于 `data/experiments/stage2_ntt_counts_20261017/`。`perf_01` 的小输入 10 次均完整；同批大输入首次因缺少目标元数据而在结果发布时报错，未计入耗时样本，原证据保留；补齐原指数/已知因子后的 `perf_02` 完成大输入 10 次。汇总为 `analysis_01/summary.json`。开关检查入口是 [test_stage2_ntt_counts.py](../../tools/test/test_stage2_ntt_counts.py)，性能复现入口是 [bench_stage2_ntt_counts.py](../../tools/bench/bench_stage2_ntt_counts.py)；大余因子需用 `--study` 携带已经验证的目标元数据。

## 代码入口与依据

- [ntt_runtime.cuh](../../src/cuda/stage2/ntt_runtime.cuh#L2773)：`choose_cfg`；[shape query](../../src/cuda/stage2/ntt_runtime.cuh#L3158)。
- [ntt_goldilocks_ptx.cuh](../../src/cuda/stage2/ntt_goldilocks_ptx.cuh)、[ntt_goldilocks_reduce.cuh](../../src/cuda/stage2/ntt_goldilocks_reduce.cuh)、[ntt_goldilocks_sub.cuh](../../src/cuda/stage2/ntt_goldilocks_sub.cuh)：域算术。
- [ntt_coop_outer.cuh](../../src/cuda/stage2/ntt_coop_outer.cuh)、[ntt_carry_partial.cuh](../../src/cuda/stage2/ntt_carry_partial.cuh)：outer 与 carry。
- [ecm_cuda_stage2.cu](../../src/cuda/ecm_cuda_stage2.cu#L2011)：S4 长除法；[梅森归约](../../src/cuda/ecm_cuda_stage2.cu#L2079)。
- [内存模型](MEMORY.md)、[性能结论](../performance/STAGE2.md)、[tune 合同](AUTO_B2.md)。
- 固定资料：[GPUOWL NTT](../reference/GPUOWL_NTT.md)、[Goldilocks 证明](../reference/GOLDILOCKS_ARITHMETIC.md)、[开源库分析](../reference/GPU_NTT_LIBRARIES.md)。
