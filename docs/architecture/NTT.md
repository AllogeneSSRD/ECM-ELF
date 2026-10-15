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

## 当前算术与调度

生产使用固定 PTX Goldilocks 归约和 canonical 减法。2⁶⁴≡2³²−1 mod q 将 128-bit 乘积折成少量显式进位链；规范范围保证专用减法成立。后端由编译身份确定；可移植tune摘要默认允许忽略backend条件复用，跨后端时间不能据此声明为本机实测。旧.cprof保持独立的严格后端合同。

tile 在 shared 中融合若干层，outer 按 radix/pass 分解大变换，缓存前向/逆向 pass 与 radix 根表。生产使用自动 coop 选择；显式 radix 比较需要开发引擎。outer V 收窄、展开、逆 scale 移位及 carry/check 融合仍有可选实验实现，不能把局部结果当作默认收益。

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

短 NTT 的实验性分块选型见 [Auto B2 / tune](AUTO_B2.md#短-ntt-批处理选型实验选项)。它使用同长度 batch 表和完整尾批估算，并同步驱动请求/NTT/S4内存规划，默认关闭。[完整曲线测量](../performance/STAGE2.md#短-ntt-分块选择)未显示稳定收益；较小batch的独立NTT吞吐优势不能覆盖整条S4流水线的成本。

短NTT还提供默认关闭的Graph提交缓存，环境变量 `NTT_SHORT_GRAPH=1` 显式启用。每项两个Graph合并A/B forward与融合inverse，重放仍在原默认流；仅借用稳定workspace与已缓存根表，在其释放/增长/驱逐前使缓存失效。它不改变pack、carry、S4、分块或数学NTT数，驱动Graph内部内存另于NTT payload账本。固定D小/大P、普通/承载、不同batch及owner回退的完整验证和限制，统一见[Graph性能](../performance/NTT_GRAPH.md)。当前完整曲线未显示稳定收益，不自动部署。

### 实际运行的长度与运算量

正常曲线结束输出 `ntt_length_counts: log2_length:num_of_ntt`，按 k 升序列出 **L=2ᵏ 的独立数学变换数**。每次卷积含两个正向与一个逆向，所以 num_of_ntt=3·卷积数；不是 CUDA kernel 启动数。debug 另输出按长度的卷积数。

阶段日志 `ntt_work` 汇总引擎累计卷积数、数学变换数、S4分块数及下列逻辑工作量。对各长度的卷积数 cₖ：

- transforms=3·Σcₖ。
- elements=3·Σ(cₖ·2ᵏ)：全部正向/逆向处理的元素总数。
- logical_radix2_butterflies=(3/2)·Σ(cₖ·2ᵏ·k)：等价 radix2 蝶形数量；融合 radix 实际指令数不等于此数。

计数包含经过多项式层的 F/G 树、inverse、fold和下降请求；不包含底层算术自检内部独立发起的变换，也不含 Stage1。普通独立 save worker 每进程一条曲线，包含该曲线的初始化；多曲线内部调试入口使用累计口径，共享初始化只累计一次。缓存工作区容量、batch×length 与单条长度分别记录，不能互相替代。

## 代码入口与依据

- [ntt_runtime.cuh](../../src/cuda/stage2/ntt_runtime.cuh#L2773)：`choose_cfg`；[shape query](../../src/cuda/stage2/ntt_runtime.cuh#L3158)。
- [ntt_goldilocks_ptx.cuh](../../src/cuda/stage2/ntt_goldilocks_ptx.cuh)、[ntt_goldilocks_reduce.cuh](../../src/cuda/stage2/ntt_goldilocks_reduce.cuh)、[ntt_goldilocks_sub.cuh](../../src/cuda/stage2/ntt_goldilocks_sub.cuh)：域算术。
- [ntt_coop_outer.cuh](../../src/cuda/stage2/ntt_coop_outer.cuh)、[ntt_carry_partial.cuh](../../src/cuda/stage2/ntt_carry_partial.cuh)：outer 与 carry。
- [ecm_cuda_stage2.cu](../../src/cuda/ecm_cuda_stage2.cu#L2011)：S4 长除法；[梅森归约](../../src/cuda/ecm_cuda_stage2.cu#L2079)。
- [内存模型](MEMORY.md)、[性能结论](../performance/STAGE2.md)、[tune 合同](AUTO_B2.md)。
- 固定资料：[GPUOWL NTT](../reference/GPUOWL_NTT.md)、[Goldilocks 证明](../reference/GOLDILOCKS_ARITHMETIC.md)、[开源库分析](../reference/GPU_NTT_LIBRARIES.md)。
