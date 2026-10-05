# 开源 GPU NTT 与 Tensor Core NTT 实现调研

## 1. 调研目标

目标是寻找：

- 开源的 GPU NTT（Number Theoretic Transform）实现；
- 优先关注能够高效利用 NVIDIA Tensor Core 的方案；
- 区分：
  - 传统 CUDA Core 高性能 NTT；
  - 真正针对 Tensor Core / WMMA / MMA 重构的 NTT；
  - 仅用于实验或研究验证、尚不适合直接用于生产的实现。

总体结论：

> **真正“开源 + GPU NTT + Tensor Core”的项目目前仍然不多。**
>
> Tensor Core 对 NTT 并非天然适配，因为 NTT 的核心是有限域上的整数模乘和模约减，而 Tensor Core 更擅长规则的矩阵乘加。因此，高效 Tensor Core NTT 的关键通常不是“直接用 Tensor Core 做一次模乘”，而是将多层 butterfly 融合并重写为较大的矩阵乘法，从而摊薄数据拆分、重组和模约减开销。

---

## 2. 推荐项目总览

| 项目 | Tensor Core | 成熟度 | 主要用途 |
|---|---:|---:|---|
| `Terminus-IMRC/tensor-core-ntt` | ✅ 核心目标 | ★★★★☆ | 最值得优先研究的 Tensor Core NTT |
| TensorFHE | ✅ Tensor Core NTT | ★★★★☆（论文级） | FHE、大 batch、高吞吐 NTT |
| `Artemarius/cuda-zkp-ntt` | ✅ WMMA 实验 | ★★★☆☆ | 研究 Tensor Core NTT 的有效/无效映射 |
| `Alisah-Ozcan/GPU-NTT` | ❌ 传统 CUDA Core | ★★★★★ | 强力 CUDA NTT baseline |
| Ingonyama ICICLE | ❌ 主要为 CUDA Core | ★★★★★ | 工程级 ZK / NTT 实现 |
| Supranational sppark | ❌ 主要为 CUDA Core | ★★★★★ | ZK / 有限域高性能 NTT baseline |
| `Amazingqaq/Tensnor-core-NTT` | ✅ WMMA / Ozaki | ★★☆☆☆ | Kyber、小模数 Tensor Core 实验 |

---

# 3. Tensor Core NTT 项目

## 3.1 Terminus-IMRC/tensor-core-ntt

GitHub：

https://github.com/Terminus-IMRC/tensor-core-ntt

这是目前与“高性能 Tensor Core NTT”目标最直接匹配的开源项目之一。

项目说明明确指出：

> NTT implementation using Tensor Cores on NVIDIA GPU

其对应论文为：

**Improved Implementation of Number Theoretic Transform on NVIDIA GPU with Tensor Cores**

作者：

- Y. Sugizaki
- D. Takahashi

会议：

- SupercomputingAsia 2026
- HPCAsia 2026

项目的核心改进包括：

1. 分析并移除此前 Tensor Core NTT 中的冗余操作；
2. 改进有限域 modular reduction；
3. 在多数测试配置下优于此前 Tensor Core NTT 方法；
4. 使用 Apache-2.0 License。

### 评价

这是目前最值得优先阅读的实现。

如果目标是自行设计：

```text
NTT
  ↓
Butterfly fusion
  ↓
Matrix formulation
  ↓
Tensor Core MMA
```

那么这个仓库应当作为第一参考对象。

---

# 4. TensorFHE

论文：

**TensorFHE: Achieving Practical Computation on Encrypted Data Using GPGPU**

arXiv：

https://arxiv.org/abs/2212.14191

TensorFHE 的重要思想是：

> 利用 Tensor Core Unit（TCU）加速 NTT，并通过 operation-level batching 提高 GPU 数据并行度和 Tensor Core 利用率。

论文重点关注的并非单个 NTT 的最低 latency，而是单位时间内完成尽可能多的 FHE / NTT 运算。

在 NVIDIA A100 上，论文报告：

- NTT：约 913 KOPS；
- 相比其对比的 GPU FHE 实现，NTT 性能约提升 2.61×。

## 4.1 对实际设计的启示

Tensor Core NTT 很可能更适合：

```text
大量独立 NTT
        ↓
      batching
        ↓
更大的矩阵运算
        ↓
 Tensor Core 高利用率
```

例如：

```text
NTT_0
NTT_1
NTT_2
...
NTT_1023
```

相比：

```text
只计算一个 NTT
```

前者更容易充分利用 Tensor Core。

因此，Tensor Core NTT 的优势通常更偏向：

> **Throughput**

而不一定是：

> **Single-NTT latency**

---

# 5. Artemarius/cuda-zkp-ntt

GitHub：

https://github.com/Artemarius/cuda-zkp-ntt

该项目非常适合研究：

> 哪些 Tensor Core NTT 思路有效，哪些实际上会更慢。

---

## 5.1 直接 INT8 拆分方案

项目测试过类似以下方案：

```text
31-bit finite-field integer
            ↓
        INT8 slicing
            ↓
          WMMA
            ↓
       reconstruct
            ↓
          mod p
```

以 BabyBear 31-bit field 为例，项目 README 给出的实验结果显示：

> INT8 Tensor Core 版本反而比普通 CUDA Core 慢约 2～12 倍。

主要原因是：

```text
整数拆分
+
Tensor Core 输入组织
+
结果重组
+
模约减
```

产生的额外开销，超过了 Tensor Core 矩阵乘法本身节省的计算成本。

### 关键结论

不推荐直接将：

```text
a × b mod p
```

映射成：

```text
INT8 decomposition
        ↓
Tensor Core
        ↓
reconstruction
```

因为单个有限域模乘的粒度通常太小。

---

# 6. 更合理的方法：Butterfly Fusion → DFT → GEMM

`cuda-zkp-ntt` 后续尝试了更合理的方法：

```text
4 stages radix-2 butterfly
             ↓
           DFT-16
             ↓
     16 × 16 matrix multiply
             ↓
       WMMA Tensor Core
```

其本质是将多层 butterfly 融合。

对于 radix-2 NTT：

```text
stage 0
stage 1
stage 2
stage 3
```

融合成：

```text
DFT-16
```

然后构造矩阵：

\[
Z_{ij} = \omega_{16}^{ij}
\]

通过 Tensor Core 执行矩阵乘法。

---

## 6.1 为什么这种方法更合理？

普通 butterfly 的计算粒度较小：

\[
u = a + \omega b
\]

\[
v = a - \omega b
\]

如果逐个 butterfly 使用 Tensor Core：

```text
butterfly
   ↓
Tensor Core
```

Tensor Core 启动和数据组织成本过高。

而如果融合 4 层：

```text
4 radix-2 stages
      ↓
   DFT-16
      ↓
Matrix Multiply
```

就可以形成更适合 Tensor Core 的高密度计算。

因此比较有前途的路线是：

\[
\boxed{
\text{Butterfly Fusion}
\rightarrow
\text{Small DFT}
\rightarrow
\text{GEMM}
\rightarrow
\text{Tensor Core}
}
\]

---

# 7. cuda-zkp-ntt 当前限制

虽然该项目已经实现 DFT-16 Tensor Core stage，但需要注意：

> 当前完整 hierarchical Tensor-Core NTT 仍存在 decomposition / twiddle correctness 问题。

也就是说：

- 单独的 Tensor Core DFT stage 很有研究价值；
- 完整的大尺寸 NTT 尚不能直接视为成熟正确实现。

因此它更加适合：

- 算法研究；
- Tensor Core 映射实验；
- 性能模型分析；

而不是直接作为生产库。

---

# 8. GPU-NTT：重要 CUDA Baseline

GitHub：

https://github.com/Alisah-Ozcan/GPU-NTT

这是传统 CUDA NTT 中非常值得作为 baseline 的项目。

实现包括：

- Merge NTT；
- 4-Step NTT；
- 32-bit finite field；
- 64-bit finite field；
- Barrett reduction。

项目目前会根据数据类型自动支持：

```text
32-bit arithmetic
64-bit arithmetic
```

并使用 Barrett reduction。

对应论文：

**High-Performance Number Theoretic Transform on GPU Through radix2-CT and 4-Step Algorithms**

发表于：

- IEEE Access
- 2025

---

## 8.1 为什么它很重要？

研究 Tensor Core NTT 时，必须避免只比较：

```text
Tensor Core NTT
vs.
naive CUDA NTT
```

更合理的是：

```text
Tensor Core NTT
vs.
高度优化的 CUDA NTT
```

GPU-NTT 正适合作为这样的 baseline。

否则很容易得到：

```text
Tensor Core 比 naive CUDA 快 3×
```

但实际上：

```text
优化 CUDA Core NTT
```

可能已经比 Tensor Core 版本更快。

---

# 9. ICICLE

项目：

https://github.com/ingonyama-zk/icicle

ICICLE 是 GPU cryptography / ZK 生态中较成熟的工程库。

NTT 支持：

```text
Radix-2 NTT
Mixed-Radix NTT
```

并支持多个 backend，例如：

```text
CPU
CUDA
Metal
```

同时具有比较完整的工程 API。

---

## 9.1 数据传输优化

ICICLE 的 NTT best-practice 文档特别强调：

> 对实际 GPU NTT 而言，CPU ↔ GPU 数据传输可能占据非常大的总运行时间。

推荐使用多个 CUDA stream 并行：

```text
Stream 1:
Device → Host
previous NTT result

Stream 2:
Host → Device
next NTT input

Stream 3:
current NTT computation
```

形成：

```text
D2H ────────────────
      H2D ────────────────
            NTT ────────────────
```

从而隐藏 PCIe transfer latency。

这说明比较 Tensor Core / CUDA Core 时需要区分：

```text
Kernel time
```

和：

```text
End-to-end time
```

---

# 10. Supranational sppark

GitHub：

https://github.com/supranational/sppark

`sppark` 是 Supranational 的高性能 ZK primitives 库。

其中包含：

```text
ntt/
msm/
ff/
hash/
```

并明确提供：

> NTT CUDA kernels

支持多种有限域和椭圆曲线环境，例如：

- BLS12-381；
- BLS12-377；
- Pasta curves。

它主要适合作为：

```text
ZK / SNARK / STARK
```

场景下的大有限域 NTT baseline。

---

# 11. Amazingqaq/Tensnor-core-NTT

GitHub：

https://github.com/Amazingqaq/Tensnor-core-NTT

这是一个较小型的 CUDA NTT benchmark 项目。

包含：

```text
ntt_benchmark.cu
barrett_ntt_benchmark.cu
plantard_ntt_benchmark.cu

tensor_ntt_benchmark.cu
ozaki_tensor_ntt.cu
```

主要面向 Kyber 参数：

\[
N = 256
\]

\[
q = 3329
\]

---

## 11.1 实现路线

项目对比了：

```text
普通 %
    ↓
Barrett reduction
    ↓
Plantard reduction
    ↓
FP16 Tensor Core
    ↓
Ozaki decomposition + Tensor Core
```

其中：

### `tensor_ntt_benchmark.cu`

使用：

```text
WMMA
16 × 16 × 16
FP16 input
FP32 accumulator
```

### `ozaki_tensor_ntt.cu`

使用：

```text
integer
   ↓
high / low splitting
   ↓
Tensor Core GEMM
   ↓
result reconstruction
   ↓
modular reduction
```

这个仓库比较适合：

- 学习 Tensor Core NTT 原型；
- 比较不同 modular reduction；
- 研究小模数 NTT；
- 快速做 benchmark。

但其定位更偏实验性，而不是成熟工业库。

---

# 12. Tensor Core NTT 的核心困难

Tensor Core 设计目标主要是：

\[
D = A \times B + C
\]

通常适合：

- FP16；
- BF16；
- TF32；
- INT8；
- 新架构上的其它低精度格式。

而 NTT 需要的是：

\[
a \cdot b \bmod p
\]

因此存在明显的数据类型鸿沟。

---

## 12.1 直接映射的问题

如果：

\[
p \approx 2^{31}
\]

则一个 field element 可能必须拆成：

```text
a =
a0
+ a1·2^8
+ a2·2^16
+ a3·2^24
```

于是：

\[
a b
\]

变成多个 INT8 partial products。

理论上 Tensor Core 吞吐量很高，但实际增加：

- slicing；
- packing；
- shared-memory traffic；
- reconstruction；
- carry；
- modular reduction。

因此总时间可能反而增加。

---

# 13. 更推荐的 Tensor Core NTT 结构

推荐结构：

```text
                    N-point NTT
                         │
                  Cooley-Tukey
                         │
              ┌──────────┴──────────┐
              │                     │
          Local NTT             Local NTT
              │                     │
          DFT-16                 DFT-16
              │                     │
        Tensor Core           Tensor Core
              │                     │
              └────── twiddle ──────┘
                         │
                  next hierarchy
```

核心思想不是：

```text
modular multiplication
→ Tensor Core
```

而是：

```text
multiple butterflies
       ↓
      fusion
       ↓
small matrix transform
       ↓
Tensor Core
```

---

# 14. 推荐研究路线

如果准备自行开发高性能 GPU NTT，推荐按照以下顺序研究。

## 第一阶段：传统 GPU NTT baseline

优先阅读：

### GPU-NTT

https://github.com/Alisah-Ozcan/GPU-NTT

理解：

- memory layout；
- shared memory；
- radix；
- 4-Step NTT；
- Barrett reduction；
- kernel fusion。

然后使用：

### ICICLE

https://github.com/ingonyama-zk/icicle

或者：

### sppark

https://github.com/supranational/sppark

建立工程级 benchmark。

---

## 第二阶段：Tensor Core 映射

重点阅读：

### tensor-core-ntt

https://github.com/Terminus-IMRC/tensor-core-ntt

理解：

```text
NTT
↓
matrix formulation
↓
Tensor Core
```

---

## 第三阶段：研究失败方案

阅读：

### cuda-zkp-ntt

https://github.com/Artemarius/cuda-zkp-ntt

重点关注：

```text
INT8 decomposition
```

为什么会失败。

理解：

> Tensor Core 峰值算力高，不等于有限域 NTT 一定快。

---

## 第四阶段：研究 Butterfly Fusion

重点研究：

```text
radix-2
radix-4
radix-8
radix-16
```

与：

```text
WMMA tile size
```

之间的关系。

例如：

```text
4 radix-2 stages
       ↓
    DFT-16
       ↓
16×16 Tensor Core GEMM
```

---

# 15. Hopper / Blackwell 上的进一步方向

如果目标 GPU 是：

- H100；
- H200；
- B100；
- B200；

则建议不要只研究 CUDA WMMA API。

可以进一步研究：

```text
WMMA
 ↓
mma.sync PTX
 ↓
WGMMA
```

其中 Hopper 的 WGMMA 可以提供更大的 warp-group matrix multiply 粒度。

理论上，这可能更适合：

```text
large fused NTT tile
```

例如：

```text
NTT tile
  ↓
shared memory
  ↓
WGMMA
  ↓
modular correction
```

---

# 16. 推荐优先阅读的三个仓库

综合来看，最值得优先阅读的是：

## 1. Terminus-IMRC/tensor-core-ntt

https://github.com/Terminus-IMRC/tensor-core-ntt

用途：

> 学习真正针对 Tensor Core 设计的 NTT。

---

## 2. Alisah-Ozcan/GPU-NTT

https://github.com/Alisah-Ozcan/GPU-NTT

用途：

> 建立足够强的 CUDA Core baseline。

---

## 3. Artemarius/cuda-zkp-ntt

https://github.com/Artemarius/cuda-zkp-ntt

用途：

> 理解 INT8 decomposition 为什么可能失败，以及 DFT-16 GEMM 为什么更加合理。

---

# 17. 最终建议

如果目标是开发新的 Tensor Core NTT，建议优先考虑：

\[
\boxed{
\text{Radix fusion}
+
\text{small DFT}
+
\text{GEMM formulation}
+
\text{Tensor Core}
}
\]

而不是：

\[
\boxed{
\text{individual modular multiplication}
\rightarrow
\text{Tensor Core}
}
\]

比较合理的整体架构可能为：

```text
Input
  │
  ▼
Global-memory layout transformation
  │
  ▼
Shared-memory tile
  │
  ▼
Fused radix stages
  │
  ▼
DFT-16 / DFT-32
  │
  ▼
Tensor Core MMA / WGMMA
  │
  ▼
Modular correction
  │
  ▼
Twiddle multiplication
  │
  ▼
Next hierarchy
  │
  ▼
Output
```

对于大量独立 NTT，还应结合：

```text
Batch NTT
+
CUDA Streams
+
H2D / D2H overlap
+
Tensor Core batching
```

追求的目标应更多是：

\[
\text{NTTs per second}
\]

而不只是：

\[
\text{latency of one NTT}
\]

---

# 18. 参考资料

1. Terminus-IMRC, `tensor-core-ntt`  
   https://github.com/Terminus-IMRC/tensor-core-ntt

2. Fan et al., **TensorFHE: Achieving Practical Computation on Encrypted Data Using GPGPU**  
   https://arxiv.org/abs/2212.14191

3. Artemarius, `cuda-zkp-ntt`  
   https://github.com/Artemarius/cuda-zkp-ntt

4. Alisah-Ozcan, `GPU-NTT`  
   https://github.com/Alisah-Ozcan/GPU-NTT

5. Ingonyama, `ICICLE`  
   https://github.com/ingonyama-zk/icicle

6. Supranational, `sppark`  
   https://github.com/supranational/sppark

7. Amazingqaq, `Tensnor-core-NTT`  
   https://github.com/Amazingqaq/Tensnor-core-NTT

# 19. 本仓库 Goldilocks 64bit 实验结果（2026-10-05）

在固定Terminus提交6f407daa基础上研究byte-MMA映射，独立实现完整132bit累加与q=2^64−2^32+1归约；参考库≥63bit模数限制仍需注意。没有导入外部源码。sm89/GMP门禁35/0通过，覆盖矩阵输出/top、真实tile频谱/逆向/stride/B只读和故障拒绝。

单独DFT16自然序接口约1–3%改善；真实t12 tile CTA256在独立k24..26复测前向慢约18.3%、逆向慢约5.6%、roundtrip慢约10.9%。两个独立CUDA+Tensor任务双流比CUDA+CUDA双流慢约9.1%，Systems只约47μs kernel overlap。byte拆分、64个MMA、carry/归约、寄存器和排列成本须进入比较；生产保留优化CUDA tile。

这些是单算子测量，Tensor峰值与CUDA峰值不能相加推导Stage2收益。当前方案保留为实验，后续优先CUDA低层单位根特化；若继续Tensor，先减少固定根矩阵的byte-MMA数量。完整数学、资源/传输公式、固定SHA、源码行号、样本与并发边界见[Tensor实验报告](D:/code/MPA-OpenCl/docs/STAGE2_TENSOR_GOLDILOCKS_EXPERIMENT.md)。

# 20. 接续 CUDA 算术：Goldilocks 短归约（2026-10-05）

低层根移位特化引出的任意128位短归约，已扩展到全部device Goldilocks算术。完整纯NTT同binary快34.06%–36.63%，对旧probe独立参考快31.88%–34.72%；真实Stage2八条固定D/Q交叉测量69.778377→59.482459s（14.76%），对旧Stage2两条独立参考均值67.173660s快11.45%。检查和内存/传输合同保持，未使用MMA。新后端完整Stage2188/0、最终D保护36/0和shared/warp14/0通过，旧D系数在short1下回退，尚未提升生产默认。

Systems NTT池28.55→19.10s，point仍约19.2s，无本进程事件间隙约9.5s保持。下一优先级为新D拟合/holdout、CPU准备/提交和剩余CUDA算术；再次比较Tensor时须以新CUDA后端为对照。来源、数学证明、原文件line、负候选、分派开销和全部证据见[短归约报告](D:/code/MPA-OpenCl/docs/STAGE2_GOLDILOCKS_SHORT_REDUCTION.md)。
