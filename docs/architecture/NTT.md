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

生产使用固定 PTX Goldilocks 归约和 canonical 减法。2⁶⁴≡2³²−1 mod q 将 128-bit 乘积折成少量显式进位链；规范范围保证专用减法成立。后端绑定编译身份，不能在同一成本 profile 下任意换后端。

tile 在 shared 中融合若干层，outer 按 radix/pass 分解大变换，缓存前向/逆向 pass 与 radix 根表。生产使用自动 coop 选择；显式 radix 比较需要开发引擎。outer V 收窄、展开、逆 scale 移位及 carry/check 融合仍有可选实验实现，不能把局部结果当作默认收益。

Tensor Core 的整数拆分路线有独立实验实现，未成为生产默认。仅满足小字长 FHE 模数的库不能直接替换 Goldilocks64/多精度 carry 管线；适配参考见 [算法资料](../reference/ECM_ALGORITHMS.md)。

## S4 系数归约

通用 M 使用已归一化除数与 2-by-1 reciprocal 长除法，直接求乘积系数余数。主乘减量级由双重消去降为约 W² 的主项，加商估计、校正和位移；真实数量由槽长度决定，不是所有输入恰好 W²。

M=2ˢ−1 时使用 2ˢ≡1 mod M 的折叠归约；点 Montgomery 乘法还需处理 R⁻¹ 对应的旋转/表示转换。承载模式保证 M 能被目标 N 整除，所有因子语义仍在 N 上处理。

S4 形状常量按 slot_bits/slot_words/b 缓存，不能只用 slot_bits 作为身份。首次形状强制验证 96 窗口；普通批次的 canonical counter 和异步 oracle 路径必须排空，错误不可被日志等级屏蔽。

## 缓冲与分块

NTT pool 保留最大工作区容量，shape 小缓冲/表按键缓存。非导出 digit 的正常 pool 路径，inverse 后 B 无读者，carry 输出 Q 可借用 B；物理 A/B 为两份，Q 不拥有第三份。非适用路线保持独立 Q 或按次分配。

S4 请求按 batch 预算执行整数减半分块，并保留至少一个 slice；因此 soft batch cap 不保证单请求一定能放入预算。grid.y 超过设备上限时继续物理拆分，模型与执行须使用相同 inner batch。分块改变内存与 launch，不改变数学请求集合。

## 代码入口与依据

- [ntt_runtime.cuh](../../src/cuda/stage2/ntt_runtime.cuh#L2773)：`choose_cfg`；[shape query](../../src/cuda/stage2/ntt_runtime.cuh#L3158)。
- [ntt_goldilocks_ptx.cuh](../../src/cuda/stage2/ntt_goldilocks_ptx.cuh)、[ntt_goldilocks_reduce.cuh](../../src/cuda/stage2/ntt_goldilocks_reduce.cuh)、[ntt_goldilocks_sub.cuh](../../src/cuda/stage2/ntt_goldilocks_sub.cuh)：域算术。
- [ntt_coop_outer.cuh](../../src/cuda/stage2/ntt_coop_outer.cuh)、[ntt_carry_partial.cuh](../../src/cuda/stage2/ntt_carry_partial.cuh)：outer 与 carry。
- [ecm_cuda_stage2.cu](../../src/cuda/ecm_cuda_stage2.cu#L2011)：S4 长除法；[梅森归约](../../src/cuda/ecm_cuda_stage2.cu#L2079)。
- [内存模型](MEMORY.md)、[性能结论](../performance/STAGE2.md)、[tune 合同](AUTO_B2.md)。
- 固定资料：[GPUOWL NTT](../reference/GPUOWL_NTT.md)、[Goldilocks 证明](../reference/GOLDILOCKS_ARITHMETIC.md)、[开源库分析](../reference/GPU_NTT_LIBRARIES.md)。
