# Tensor NTT：byte MMA、132-bit 重构与资源公式

固定参考资料。来源：`STAGE2_TENSOR_GOLDILOCKS_EXPERIMENT.md §§2、3、5`。保留数学推导、第三方源码分析及其条件；源码行号针对原文研究的本地副本，第三方上游可能已有变化。本文不声明本仓库当前支持、默认值或性能。涉及实验数字时只按原文条件理解；当前实现见 [架构](../architecture/OVERVIEW.md)，当前计时口径见 [性能](../performance/STAGE2.md)。

## 2. 参考实现与适配边界

固定阅读 Terminus-IMRC/tensor-core-ntt 提交 `6f407daa8a4cef96331511ae86b922b520d7aa33`，Apache-2.0。其 `include/polyarith/cuda/ntt.cuh:71` 将 64bit 值拆成 byte，以矩阵 MMA 累加；`include/polyarith/modular.cuh:326` 的归约器构造函数拒绝 ≥63bit 模数。Goldilocks `q=2⁶⁴−2³²+1` 是 64bit，因此需要独立证明累加与归约范围。[固定矩阵源码](https://github.com/Terminus-IMRC/tensor-core-ntt/blob/6f407daa8a4cef96331511ae86b922b520d7aa33/include/polyarith/cuda/ntt.cuh#L71)、[模数限制](https://github.com/Terminus-IMRC/tensor-core-ntt/blob/6f407daa8a4cef96331511ae86b922b520d7aa33/include/polyarith/modular.cuh#L326)。本仓库实现未导入该库源码。

使用 PTX `mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32`。lane 的矩阵 fragment 依照 NVIDIA 定义组织，整 warp 执行相同 MMA；寄存器结果随后由 CUDA 指令重构并归约。[PTX 整数 fragment 合同](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-integer-type)。该映射适用 sm80+；原文的实例为 sm89，不外推其他设备性能。


## 3. 精确计算：64 个 byte MMA 与 132bit 和

对 `A[16×16] · B[16×8]`，一个输出为 `S=Σ(k=0..15) Aik·Bkj`。输入 canonical，均在 `[0,q)`：

```text
S ≤ 16(q−1)² < 2^132
A = Σ(a_u · 2^(8u)), B = Σ(b_v · 2^(8v)), u,v=0..7
s_d = Σ(u+v=d) Σ(k=0..15) a_ik,u · b_kj,v, d=0..14
s_d ≤ 16·8·255² = 8,323,200
```

每对 byte 调用一次 MMA，共 `8×8=64` 次 warp MMA；按 15 个对角依次累加、传播 base256 carry。carry 最大不超过32767，`s_d+carry` 仍低于 signed32 上界。无需同时保留15个对角的全部 accumulator。

第14个对角处理后，剩余 carry 的 low8 写入 high64 的最高 byte；`top=carry>>8` 为完整和的 bits128..131，范围0..15。保持 `S=lo+2⁶⁴·hi+2¹²⁸·top`，利用：

```text
2^64  ≡ 2^32−1 mod q
2^128 ≡ −2^32   mod q
result = gl_sub_dev(gl_reduce(lo,hi), top<<32)
```

`gl_reduce` 复用当前任意128bit Goldilocks fold，结果 canonical；减数 `top<<32<q`。丢弃 top 会得到错误余数，roundtrip 单独成功不足以证明正确。GMP 直接保存完整和，并独立比较最终余数及 `floor(S/2¹²⁸)`。

代码：[MMA:14](../../tools/bench/ntt_tensor_goldilocks.cuh#L14)、[对角/carry:28](../../tools/bench/ntt_tensor_goldilocks.cuh#L28)、[132bit 归约:45](../../tools/bench/ntt_tensor_goldilocks.cuh#L45)、[GMP dot:115](../../tools/test/ntt_tensor_goldilocks_probe.cu#L115)。


## 5. 计算量、容量和传输公式

### 5.1 运算计量

一个16×8输出块有128个word、2048个完整64bit乘积项。64次 MMA 合计执行 `64·16·8·16=131072` 个 byte MAC，即 **1024 byte MAC/输出word**。整次 tile pass 为 `L/2` 次 warp MMA、`1024L` byte MAC、`L` 次132bit重构/归约。这是源码操作数，未测硬件 cycle 或实际 issued instruction 数。

生产 warp tile 的 stage0 已去单位根乘法，stage1 已将 ±2⁴⁸ 变为 shift/fold。因此 forward 通用 `gl_mul` 调用数为 `(t−2)L/2`，带B/scale的 inverse 为 `(t−2)L/2+2L`。Tensor 替换 stage0..3 后分别为 `(t−4)L/2` 与 `(t−4)L/2+2L`，另增加上述 MMA/重构。t12 三变换卷积的 tile 部分：生产 **17L** 次通用模乘；候选 **14L** 次通用模乘 + **1.5L** 次 warp MMA + **3072L** byte MAC + **3L** 次累加和归约。运算类型不同，不能把它们的计数直接相加评价速度。

若一次多项式乘法有 b 个 slice，将上述计数乘 b。完整曲线按实际乘法集合 `C(B2,D,S)` 求和，`P=φ(D)/2`、当前 `I=floor(B2/D)+2`、`G=ceil(I/P)`；`S=bitlen(N)`。各乘法 `m=max(na,nb)`、`slot_bits=2S+max(1,ceil(log2 m))`，选择满足 `m·sw·(2^bpw−1)²<q` 的 bpw，`sw=ceil(slot_bits/bpw)`，`L=nextpow2(2m·sw+1)`。例如总 byte MAC 为 `3072·Σ(c∈C)b_cL_c`。B2 通过 I/G 和树/折叠调用数影响工作，不能用 π(B2) 代替。

### 5.2 Global/shared 与显存

主数组 forward 读写 `16L B`，inverse 读A、读B、写A为 `24L B`。两个forward+inverse的 tile payload 为 **56L B/slice**，两实现相同，根表读取另计。这是逻辑payload，cache命中与实际DRAM流量没有计数器证据。t12 shared数组为 **8T=32768 B/CTA**；两实现同容量，当前布局保留同样的高层 shared 交换。

Probe 的 `TileBuffers` 持有 A/B 两个数组、双向 tile 表、两个 packed root 表，以及各4个tile的input/B/expected池：

```text
单任务设备requested payload = 16L + 16(T−1) + 4096 + 96T + 8
                            = 16L + 112T + 4088 bytes
双任务设备requested payload = 2·(16L + 112T + 4088)
```

t12单任务 L=2²⁴/2²⁵/2²⁶实测peak为268898296/537333752/1074204664 B，即256.441/512.441/1024.441 MiB；双任务2²⁴/2²⁵为537796592/1074667504 B，即512.883/1024.883 MiB。结束 `live_bytes=0`。计量涵盖本工具的cudaMalloc请求，未包含context/module/driver，未测NVML总峰。

tile Tensor forward/inverse寄存器 **74/76**，LOCAL0；生产CUDA两个方向 **40**，LOCAL0。资源API在32KiB shared下允许TC CTA128/256各3块/SM、CTA512只有1块/SM；生产CUDA CTA512为3块/SM。TC CTA256即使可驻留3块，也只有768线程，CUDA CTA512对应1536线程。上述是容量上限，实际occupancy/cycle/带宽仍未测。前阶段 NCU 2026.2.1 返回 `ERR_NVGPUCTRPERM`，本阶段未重复请求相同权限条件。

### 5.3 Host 数据与边界传输

性能probe使用GPU将4个GMP参考tile周期复制到全部L输出，host长期向量payload **112T+4080 B**，t12为462832 B；GMP初始化临时对象及vector allocator开销另计。初始化H2D也为该payload，和L无关；每次输入fill向设备A/B写16L B。计时内H2D/D2H均为0；计时外逐字校验A/B所有L输出，比较器扫描至少16L B，D2H仅8 B错误计数。该构造适合隔离算子吞吐，真实Stage2的数据生成/传输成本见流水线报告。

自然序矩阵性能probe的device payload为 `2048·batches+37112 B`，包含两个主数组、矩阵/根表和16组参考pattern；batches65536约128.035MiB。完整门禁使用更小的fixture并回传全部结果及top，不能把其传输口径混入性能probe。

代码：[计账:7](../../tools/test/ntt_tensor_goldilocks_probe.cu#L7)、[TileBuffers:258](../../tools/test/ntt_tensor_goldilocks_probe.cu#L258)、[资源查询:447](../../tools/test/ntt_tensor_goldilocks_probe.cu#L447)。
