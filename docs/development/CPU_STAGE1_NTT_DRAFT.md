# CPU Stage1 加权 NTT 探针草稿

诊断入口为 [cpu_stage1_weighted_ntt.cpp](../../tools/bench/cpu_stage1_weighted_ntt.cpp)。首个原型支持完整 M6011 的 320 点与 M12511 的 640 点，使用同一个小于 \(2^{50}\) 的素数域。点位按 \(e_j=\lceil pj/L\rceil\) 划分。对每个循环卷积系数逐项使用真实 digit 上界和 1／2 权重求和，确保单域能无歧义恢复。初始化核对原根和加权根。

PFA 把 320／640 点变换分成五路 64／128 点和五点变换。内核用 IFMA52 实现常数模乘和频域逐点 Montgomery 乘法。逆长度、逆权重及退出 Montgomery 域合并成常数乘法。乘积系数依次向更高位进位，越过 p 位的进位回到首 digit。GMP 对拍包含零、单位元、接近模数的输入和随机输入，依赖链的末点另与 GWNUM 核对。

```powershell
tools/build/test/build_cpu_stage1_ntt_probe.ps1
build_dev/cpu_stage1_weighted_ntt_probe/cpu_stage1_weighted_ntt.exe
```

当前探针要求 AVX-512F/DQ/IFMA/VL 和本地 `gwnum64.lib`，Windows 下固定逻辑 CPU 1。见 [算术耗时与边界](../performance/STAGE1.md#cpu-加权-ntt-算术探针)。

后续验证先分解 PFA 打包、蝶形、逐点乘法和进位耗时。将频谱保留与融合乘加接入点公式前，逐式证明扩大的系数范围。M12511 的 640 点单域最大系数已超过域模数的一半，任意双乘积加减需要调整形状、增大系数余量或改用其他算术路径。完整点步和单核多曲线吞吐作为后续验证。

## AVX-512 优化计划

第一目标是找到 CPU Stage1 NTT 相对现有 IFMA/Karatsuba 的吞吐交叉点。单曲线时同时记录延迟。Prime95/GWNUM 用于比较差距，保留其原生算术表示与融合能力。历史两个探针的完整模运算用时是 GWNUM 的 5.9–7.9 倍，因此只优化某一个 butterfly 不足以判定端到端胜负。

### 1. 建立同口径剖析

现有探针只接受 M6011/320 点和 M12511/640 点，已经用 52 位 IFMA 实现定常数域乘和逐点 Montgomery 域乘，也已合并逆权重、逆长度和退出 Montgomery 域。优先测量已有路径的 `pack/置换/乘权 → forward → pointwise → inverse → 去权/逆置换 → carry/limb 写回`。记录两次前向、一次前向（平方）、固定一侧频谱、完整模运算和整条依赖链的耗时。分项用整批累积或单独的探针采样，关闭剖析再测总耗时，避免对微秒级单次调用计时造成干扰。

模算术采用相同完整梅森模数、相同输入和单核亲和性。记录单条依赖链的 µs/op，以及 1/2/4/8 条独立链的总 ops/s。IFMA limb 后端要同时报告单曲线占槽和满八路的实际吞吐。GWNUM 比较保留原生数据布局，并单测固定操作数频谱和融合公式。固定 HX 370 的核心类型与逻辑核心，记录有效频率或明确频率证据的缺口。首轮保留 M6011、M12511，扩展到 4K、8K、16K、32K、64K 附近。先用现有路径的数据做回归，待根条件扩展后再覆盖更宽档位。最终用 s/curve 与 curves/s，明确计时是否含准备、GCD、save 和缓存。

### 2. 单域 IFMA52 作为第一条内核路线

- 将定常数蝶形改为有范围合同的 Harvey lazy 路径。当前素数 \(q=1125899893734401<2^{50}\) 允许保持小于 \(4q<2^{52}\) 的中间输入。按 forward DIF 和 inverse DIT 分别证明入口、出口范围。逐点乘法目前要求自己的输入范围，先在其边界规范化，再决定是否放宽。IFMA 乘数只读取低 52 位，需覆盖范围端点及带符号差值。[Intel HEXL 论文](https://arxiv.org/abs/2103.16400)
- 把二次幂子变换的 `half=1/2/4` 合为驻留寄存器的 8 点 codelet，优先省去反复 load/store、置换和 `W=1` 的通用乘法，再试 radix-4/8 层融合。保留 DIF/DIT 共用的频谱顺序。PFA 的五路根表、置换、权重全部按计划预计算，验证是否值得将最靠近打包或去权的一层合并到同一个循环。
- 对照 YMM 256 位（AVX512VL）和 ZMM 512 位，先用同核微基准测 IFMA 吞吐和跨 lane shuffle，再用实际变换确认。测试单曲线 `lane=相邻系数`，以及多曲线 `lane=不同曲线的同一系数`。后者共用根表并消除变换内部跨 lane 蝶形交换，但要计入转置和工作集。按 \(8L\) 字节/频谱/曲线估算，\(L=640\) 时八曲线单个频谱槽已占 40 KiB，点步同时持有多个槽时要试 2/4 曲线、混合分组和按 tile 处理。
- 打包先对照预计算源 limb 与变位量、合并加权和首层变换。去权将末层写回顺序与进位顺序一起评估。现有进位是标量回卷，先量化；只有该项成为热点时，再试分块 carry、短向量前缀或与下一轮 pack 融合。GPU 的 block 级分段门限不直接沿用到 CPU。

### 3. 选长度和域，明确精确恢复范围

按同一 \(p\) 枚举 `2^k` 和 `5×2^k`，用现有加权卷积规则核对 digit 位置、原根、加权根和最大系数，取实测最好的合法形状。每个长度需有本原 \(L\) 次单位根，还要在域中解 \(\rho^L=2^p\)。当前 \(q-1\) 的 2-adic 指数为 10，不能覆盖任意更长的二次幂。新增素数要同时满足根条件和 IFMA 的 lazy 界限。用无权线性卷积作精确后备和计时控制。现有逐项系数检查为 \(O(L^2)\)，扩展宽位前应改为经证明的快速上界或计划缓存，单列冷构造耗时。

令 \(C\) 为当前 unsigned digit 方案单个乘积的最大卷积系数。单乘积要求 \(C<q\)，两个乘积的和按 \(2C<q\) 做保守判定；两个乘积的差若范围为 \([-C,C]\)，也需 \(2C<q\) 才能唯一恢复。当前 M6011/320 点的 \(C/q\approx0.122\)，M12511/640 点约 0.685。后者需要更大长度、证明更紧的逐式范围或改用 CRT。平衡 digit 可作为独立候选，需重新证明有符号范围和进位合同。双 30/31 位域可用作第二条路线，测试两域各自的根条件、32 位打包、CRT 和归约成本。将合成域容量按乘积计算，实际吞吐计入两套变换和重建，先用端到端数据决定是否继续。

### 4. 把变换次数降到点公式需要的数量

先实现同一坐标在一次点步中被多次消费时只做一次 forward，曲线常数的频谱按曲线生命周期保留。随后加 `A·B±C·D` 的频域逐点融合，在容量允许时省掉一次 inverse/归约。先测 Montgomery PARAM0 的 xDBL+xADD，便于与当前 ladder 对拍，再测 Edwards 的倍点与 NAF 混合加点。借鉴 [GWNUM 的操作数状态和复合算术](../reference/GWNUM_POLYMULT.md) 与 [GPU Stage1 的依赖分层](../performance/STAGE1_MERSENNE_NTT.md)。逐式维护频谱所代表的 digit 范围，乘积依赖后续乘法时先归约到有界表示，除非有独立的范围证明。完整点步要记录实际 forward/inverse 次数和 scratch 规模。

验证顺序为 NTT 单蝶形与往返、GMP 卷积系数、模乘/平方/融合式、点步、完整 Stage1。覆盖最坏 digit、零和全一代表、跨 limb 与回卷进位、计划切换、原位别名、固定 sigma 因子命中和恢复。默认路径的变更以相邻位宽、两种以上批量的整曲线吞吐稳定领先为依据。若单域在模算术和点步上都无法接近当前 IFMA/Karatsuba，先保留为实验路径，再独立评估宽位双域或 FP64 加权 FFT。Stage2 的长卷积和多项式槽另行计时。

## 第一轮复现

[内核](../../tools/bench/cpu_stage1_ntt_kernels.h)独立组合 YMM/ZMM、规范归约/lazy、逐层/8 点短层。正向 DIF 在行内保持 \([0,2q)\)，逆向 DIT 保持 \([0,4q)\)。两方向的定常数乘数输入都小于 \(4q<2^{52}\)，Shoup 乘法保留 \([0,2q)\)。进出二次幂行时保持五点层和逐点域乘所需的规范值。8 点短层合并 `half=1/2/4`，同时省掉 `half=1` 的单位旋转因子乘法。ZMM 逐层规范归约保留原版行内核。

两种 52 位向量宽度均经反汇编核对实际 `VPMADD52` 编码。独立短测把 8–1024 点行变换与原实现对拍 3200 次。正式矩阵还对每种 NTT 实现、每个位宽和运算测 32 个边界与随机输入，三种后端的依赖链末尾分别与 GMP 对拍。相位与端到端计时由同一程序分开采集。

```powershell
$build = 'build_dev/cpu_stage1_ntt_round1'
$data = 'data/experiments/cpu_stage1_ntt_round1'
New-Item -ItemType Directory -Force $data | Out-Null
tools/build/test/build_cpu_stage1_ntt_probe.ps1 -Build $build
& "$build/cpu_stage1_weighted_ntt.exe" --profile-only > "$data/baseline-phases.csv"
& "$build/cpu_stage1_weighted_ntt.exe" > "$data/matrix-final.csv"
$exe = (Resolve-Path "$build/cpu_stage1_weighted_ntt.exe").Path
$run = Start-Process -FilePath $exe -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$data/matrix-priority.csv"
$run.PriorityClass = 'AboveNormal'
$run.WaitForExit()
python tools/bench/plot_cpu_stage1_ntt.py "$data/matrix-priority.csv" --phases "$data/baseline-phases.csv"
```

`time` 行记录位宽、运算、批量、样本号、后端、长度（IFMA 为 limb 数）、Karatsuba 深度和 µs/op。`phase` 行记录同一条件的六个分项。`gwfft` 行记录 GWNUM 的每操作变换计数，用于核对普通模乘、平方和固定频谱模式的计时口径。曲线级集成与检查点验证在下一轮点步实验之后开展。
