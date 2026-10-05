# 当前 CUDA ECM Stage2 实现：逐步骤说明

日期：2026-10-03；量化及优化补充：2026-10-04。历史测量基线：`0cfa589`；代码行号已更新到当前尺寸策略 NTT 引擎，device leaf 性能证据见§31.5–§31.6，独立生产驱动见§32，GPU 驻留 fold 历史基线见§33，根交接见§34，组合调度见§35，当前瓶颈与候选排序见§36，开源 NTT 适配见§37，xADD6与D见§38，cooperative v2见§39，尺寸策略与新D标定见§40。§26–§31 保留各自测量基线。

本文按一条曲线的实际执行顺序说明算法、输入输出、CPU/GPU 分工和数据生命周期。代码链接均指向当前原文件的一处入口，行号为本基线的一基行号；后续修改源码时行号可能变化。

**量化阅读入口：** §23 给出计算量及可校准的周期公式，§24 区分 RAM/显存峰值、累计数据生成和传输量，§25 保留上一轮段积实验的阶段分摊，§26 记录 fold 实验。**§27 按 fold-flat 历史基线逐步骤列账**，细化 seed、积树及参数外推；**§28 用 device seed 实测修正操作数、资源和阶段占比**；**§29 给出 Mersenne 归约实测**；**§30 给出本轮小素数复用的同二进制 A/B、资源和 GPU 时间线**。**§31 汇总最新基线每一步的计算/周期公式、容量、生成/传输量和时长占比，并量化当前 device leaf 候选。**公式推导、日志实测和规划预估分别标明。

## 1. 范围与入口

本文主线是 [stage2_tree_gpu.cu](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10694) 的 `--real` → `run_real()` → `run_batched()`。多项式乘法直接包含并调用 [ntt_poly_probe.cu](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:87)，不是另写一份 NTT。

需要区分三个实现：

- **CUDA 多项式树实验引擎**：本文对象。F/G 积树、模 F fold、余式树下降、分块 GCD，构建为 `stage2_tree_gpu.exe`。
- **CPU GMP 树参考**：[stage2_tree_ref.cpp](D:/code/MPA-OpenCl/tools/bench/stage2_tree_ref.cpp:1)，提供小形状独立结果和 F dump。
- **CUDA pairing/BSGS 路径**：[cgbn_stage2.cu](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage2.cu:1)，使用逐候选点关系和模乘，执行步骤与本文树版不同。

共用树引擎仍位于 `tools/bench/`；新增独立生产入口 `ecm_cuda_stage2.exe`，支持 param0 Stage1 文本 save、`ECMSTAGE2=` 队列与 `ecm.ini`，参见 [编译及使用说明](ECM_CUDA_STAGE2.md)。原 Stage1 驱动仍有 [Prime95 交付调用](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:3148)。实验 CLI 保留 `--real` 和 `--check-F`，生产入口通过 `run_real` 的可选 save Q 参数跳过 Stage1。

`--real` 会自己重算 Stage1 Q；`--check-F` 从 CPU dump 读取 Q、baby/F 参考信息并验证。[run_check_F](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11668) 是独立检查入口，不能用它通过来替代 `--real` 输入生成正确性检查。

### 1.1 本文采用的优化组合

最新生产组合使用xADD6、warp tile及按实际NTT长度的outer模式2，D系数分别匹配原NTT/新尺寸策略；支持scope、回退、量化与验收见§40和[P3报告](D:/code/MPA-OpenCl/docs/STAGE2_NTT_SHAPE_D_CALIBRATION.md)。下列分支默认和§26–§35组合保留实验引擎及各轮测量的口径，生产wrapper的默认值单独记录。

本轮 `NTT_S4_ORACLE_ASYNC=1`、`NTT_S4_CARRY_BATCH=1` 采用§35经重新测量的组合；§33/34性能仍保留原blocking/per-chunk对照，不把旧等待分项当作当前纯CPU成本。

本轮 `NTT_GROOT_TO_FOLD=1` 在 device fold owner 启用时直接交接根，实验缺省0、生产缺省1，详见§34。

“当前代码默认值”与“最近成功 A/B 使用的组合”不同。逐步骤主线以下列测量组合为例，回退分支另行说明：

- S4 设备系数归约开启，`NTT_S4_OLDTAIL=0`：直接长除法归约。
- `NTT_S4_MERSENNE=1` 可选择精确 `N=2^S−1` 的专用fold归约；默认0、通用N回退长除法，详见§29。
- 本轮 `NTT_SMALL_PRIME_REUSE=1` 复用 baby 归一化的 unit/GCD 结果；默认0，缺失点、无缓存或输入不匹配仍走原ladder/GCD。§23–§29的小素数工作量为未复用对照值，候选折减与适用范围见§30。
- `NTT_GROOT_DEVICE=1`：G 树中间层驻留设备；§33对照读回根，§34启用交接后不再读回根。
- `NTT_SCALED_DESCENT=1`：scaled 下降；`NTT_S5_ON=0`。
- `NTT_S4_OUTPUT_WINDOW=1`、`NTT_S4_CHUNK_OUTPUT=1`：只返回需要的系数窗口，并复用 chunk 输出空间。
- `NTT_S4_PACK_DIRECT=1`、`NTT_S4_FLAT_DIRECT=1`：直接打包到 NTT 工作区，尽量借用已有 flat 输入。
- `NTT_GROOT_LEAF_STAGING=1`、`NTT_GROOT_COMPACT_RAW=1`：复用 pinned staging，raw A/B 分别定容。
- `NTT_GFINV_BATCH=1`：64 段窗口批量求逆；`NTT_S4_CARRY_BATCH=0`。
- `NTT_FOLD_FLAT=1`：本轮A/B两边均用连续host fold；`NTT_GFINV_SEG_EXACT=1`为精确段积默认，`NTT_GIANT_SEED_DEVICE=0/1`是§28的唯一性能对照变量。
- 最近生产 A/B 的 oracle async=0，sample=96、check every=8；异步 oracle 是另外的可选路径。

其中 G-root device、scaled descent、output window、GFINV batch 都是 **opt-in，代码默认关闭**；pack direct、flat direct、chunk output、G-root leaf staging/compact raw 默认开启，final whole readback 默认关闭，carry batch 默认关闭。
device seed及fold-flat仍默认关闭；精确段积默认开启，旧尺度仅由显式诊断开关恢复。
出处：[S4 与算法开关](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1542)、[G-root 内存开关](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1604)、[carry batch](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1627)、[GFINV batch](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:407)。

## 2. 总流程与符号

```mermaid
flowchart TD
    A[CPU: N、sigma、B1/B2、D 与 baby 索引] --> B[CPU/GPU: 曲线参数与 Stage1 Q]
    B --> C[GPU: baby 点 jQ]
    C --> D[CPU: 批量归一化、线性叶]
    D --> E[CPU 编排 / GPU 乘法: 完整 F 树]
    E --> F[GPU 乘法 / CPU 编排: rev F 的 Newton 逆]
    F --> G[GPU: 一批 giant 点 iDQ、16 点段积]
    G --> H[CPU: 段积求逆、混合射影和仿射叶]
    H --> I[GPU: G 积树上升，读回根]
    I --> J[CPU/GPU: H = G H mod F]
    J --> K{还有 giant 批次?}
    K -->|有| G
    K -->|无| L[CPU: 用 Gamma 逆元消除射影缩放]
    L --> M[CPU 编排 / GPU 乘法: 沿 F 树下降]
    M --> N[GPU: 每 64 个叶值求乘积]
    N --> O[CPU: 块 GCD、叶 GCD、可选素数命名]
    O --> P[CPU: 合并退化因子、排空检查、输出]
```

符号约定：

- `N`：待分解的奇数模数；`S=bits(N)`，`W=ceil(S/64)`。
- `Q`：Stage1 后的 Montgomery 曲线点。本文用 `Y` 表示多项式变量，避免与点坐标 `X` 混淆。
- `D`：baby/giant 步长。`J={j:1≤j≤D/2, gcd(j,D)=1}`，`P=|J|=φ(D)/2`。
- `x_j=x([j]Q)`：baby 仿射横坐标；`u_i=x([iD]Q)`：giant 仿射横坐标。
- `I=floor(B2/D)+2`：当前引擎的 giant 数量，索引为 `i=1..I`。
- `F(Y)=∏_(j∈J)(Y−x_j)`；`G_b(Y)`：第 b 批 giant 对应的积多项式。
- `H`：逐批累积的 giant 多项式，通常保持为模 F 的余式。
- `n_ntt`：变换长度；NTT 模数记作 `p_ntt`，它与 ECM 模数 N 不同。

**为什么可用多项式寻找因子？** 对 N 的某个素因子 q，正常点关系中 `x([iD]Q)=x([j]Q) mod q` 对应 `[iD±j]Q=O mod q`。因此积多项式在 baby 点的值含有这些差值因子，最终通过 GCD 提取 N 的因子。无穷远点、不可逆 Z 和退化 chain 需按代码的专门分支处理。

giant/baby 组合是候选超集，不是“每个线性因子对应一个已经筛过的 Stage2 素数”；素数区间与素性过滤发生在小素数分支及命名分支。
出处：[giant 范围](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9582)、[命名方向说明](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10417)。

## 3. 数据表示：贯穿全部步骤的约定

### 3.1 多项式系数

系数是 **普通域中的 `[0,N)` 整数**，一个系数占 W 个 little-endian 64-bit limb。flat 布局为：

```text
poly[k*W+t] = 第 k 个系数的第 t 个 limb
系数顺序：常数项、一次项、二次项……
```

`CPoly` 是“一系数一个 vector”的主机表示；积树节点通常用 flat vector。`cp_from_flat/cp_to_flat` 在两种容器间复制。它们仍都存普通模 N 系数，不是 NTT 频谱。
出处：[布局约定](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:20)、[转换函数](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4573)。

### 3.2 点坐标与三个算术域

- 点 kernel 采用 Montgomery 模乘，参数包含 `R=2^(64W) mod N`、`ninv=−N⁻¹ mod 2^64`、Montgomery image 的 `a24/Q`。
- 多项式接口输入输出是普通模 N 系数。
- NTT 内部是 Goldilocks 素域 `p_ntt=2^64−2^32+1`。

`ladder_points` 默认输出普通域坐标；device seed 可直接借用其 Montgomery image，chain 输出也保留 image 射影坐标。**同一对坐标共同乘 R 仍代表同一仿射点**，但其射影叶的标量也包含 R。当前默认段积已恢复为实际返回 Z 的普通乘积；旧尺度、修正公式及独立验证见第 13、17、28 节。
出处：[LadderCtx](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4261)、[chain seeds 转域](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9038)、[实际 Z 段积合同](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9774)、[NTT 模数](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:89)。

## 4. 第 0 步：解析任务、规划 D、枚举 baby 集

**位置/分工：CPU；CUDA 查询设备与显存。**

1. 读取 N，要求 N 为奇数，计算 S/W，选择 `--device`。
2. 查询空闲显存，默认预留 768 MiB，用剩余容量形成 arena 预算；`NTT_ARENA_RESERVE_MB / NTT_ARENA_CAP_KB` 可覆盖。
3. 使用指定 `--d`，或 `--choose-d` 根据拟合的 loop/tree/giant/glue 成本及形状容量选择可容纳的 D。
4. 枚举 `j≤D/2 且 gcd(j,D)=1`，严格检查数量等于 φ(D)/2。
5. 计算 giant 数 I、G 批次数 `ceil(I/P)`、fold 次数 `ceil(I/P)−1`。

D 越大，giant 数往往越少，但 F 树、逆多项式和下降随 P 增大；当前选择策略并非“取能放下的最大 D”。源码仍有一些早期注释，实际应以决策逻辑为准。

**输出：** baby 索引、P、I、批次数、形状及显存预算。baby 索引留在主机，点生成时上传。
出处：[解析和显存预算](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10694)、[D 决策](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10927)、[baby 集](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10951)。

## 5. 第 1 步：曲线、Stage1 Q 与上下文准备

这一节是 Stage2 的输入准备；Stage1 运算本身不计入 `stage2_full_wall.total`。

**CPU：** Suyama sigma 参数化计算 `a24` 和起点 `(X:Z)`，准备 Montgomery 常数；参数化退化时当前 `--real` 报错退出。生成各素数最高幂组成的 lcm 指数因子列表。

**GPU：** `ladder_product` 在一个顺序 ladder chain 中计算 Q。各指数因子之间有数据依赖，不按这些因子并行。

```text
NTT_STAGE1_EXTRA=1  → Q = [lcm(1..B1)]P0
NTT_STAGE1_EXTRA=12 → Q = [12*lcm(1..B1)]P0
```

生产 ini 的相应设置是 `method=gpu`、`gpu_param=0`、`exponent=choose12`。探针不读取该 ini，必须显式设 `NTT_STAGE1_EXTRA=12`；对齐 Prime95 还需相同 N/sigma/B1，最好核对完整 Q。

Q 准备完毕后，建立 LadderCtx、NTT arena、S4 输入/输出区、归约常数及检查上下文。此后的 S4 等 mandatory 自测计入 Stage2 init；Q 之前的点 Montgomery 自测不在这个 init 边界内。

**传输：** 起点、指数列表和常数 H2D，最终 Q D2H；CPU 准备 Stage2 的 Q Montgomery image。上下文和 arena 存活到本次 `run_real` 结束。
出处：[Montgomery 参数](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10983)、[Stage1 EXTRA](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11020)、[Stage2 init 起点](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11070)、[ini 文档](D:/code/MPA-OpenCl/docs/DEV_ECM_INI.md:65)。

## 6. 第 2 步：GPU 生成 baby 点 `[j]Q`

**输入：** Q、a24、N 和全部 baby 索引。GPU 用 x-only Montgomery ladder，按点并行，返回 `bx/bz` 两个 flat 坐标数组。普通 ladder 与 chain 的线程模型不同，不能把它视为一个 CGBN 生产 Stage1 kernel。

**输出：** 每个 baby 点的普通域 `(X_j,Z_j)`，回到 CPU。

baby 点数组大小各为 `P*W*8` bytes。后续构造完线性叶和 F 树后，局部 bx/bz 被释放；F 树保留。
出处：[baby 调用](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11185)、[ladder kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:839)、[ladder_points](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4268)。

## 7. 第 3 步：CPU 批量归一化 baby，并构造 monic 叶

按 **256 点一个 segment**，由 CPU/GMP 构造 Z 的 prefix 积：

```text
pv[0] = 1
pv[j+1] = pv[j]*Z_j mod N
iprod = inverse(pv[segment_length]) mod N

从末尾到开头：
    Z_j_inverse = iprod*pv[j] mod N
    x_j = X_j*Z_j_inverse mod N
    iprod = iprod*Z_j mod N
```

通常一个 segment 只做一次 `mpz_invert`，其余是模乘。若某个 Z=0 或组合积不可逆，整个 segment 走逐点 affine helper。该 helper 的实际规则为：Z=0 返回 x=0 且标记成功；非零 Z 不可逆则返回 X、标记失败，调用处另计算 `gcd(Z,N)` 收集退化信息。

最终叶系数为 `[-x_j,1]`，即 `Y−x_j`，因此 F 及其所有真实节点 monic，后续反转多项式的常数项可逆。

**重要修复：** 旧代码漏乘 `pv[j]`，把 `X_j/(Z_0…Z_j)` 当成 x_j。本基线已修复；此前 `--real` 性能记录不能证明正确 ECM 的性能。新真实入口门禁核对独立全部 baby/F，冻结因子已恢复。

**资源：** 固定 256 点 GMP 临时窗口、P 个 2W-word 叶 vector；不在 GPU 求 baby 的逆元。
出处：[256 点窗口及回退](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11209)、[正确逆扫描](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11260)、[affine helper](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:353)、[独立门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_real_baby.py:1)。

## 8. 第 4 步：上升构建完整 F 积树

把叶数补到 `Fpad=next_power_of_two(P)`，使用二叉 heap 编号：根为 1，孩子为 2v/2v+1，叶从 Fpad 开始。

- 真实叶为 `Y−x_j`，padding 叶为常数 1，**不是多加一个零根**。
- 每层按孩子多项式长度分组；同形状的多个 pair 一起调用 batched 乘法。
- 常数 1 子树使用 passthrough，无需 NTT 乘法。
- F 树保留全部需要的节点和次数信息，供最后下降读取；不能套用 G 树的“只保留根”策略。

**分工/传输：** CPU 建 heap、组装每层 flat 输入；GPU 做多项式乘法及 mod N 归约；当前 F 树节点结果仍读回主机。F 全树尚未按 G-root 的方式全程驻留 GPU。

**输出：** `Ft/Fdeg/Fpad`，根 `Ft[1]` 是完整 F。
出处：[build_tree_flat](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3937)、[真实 F 构建](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11302)。

## 9. 公共算术步骤：一次 GPU 多项式乘法如何执行

F 树、G 树、Newton 逆、fold、下降共用这一链路。单次 batch 内每个 slice 是独立多项式乘法，变换长度按最大输入长度规划。

### 9.1 主机分组、形状与 chunk

`poly_mul_batch_modN` 接受 ma/mb、nbatch 和输出窗口 first/count。检查长度、stride、输入/输出 alias 与驻留 offset；按 batch 预算拆成多个 chunk。

普通路径上传 flat 模 N 原始系数；驻留 G 树路径使用已经在 raw frontier 的系数，额外上传 offset metadata，不再上传整层系数。
出处：[乘法入口合同](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3392)、[flat 输入借用/补零](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7297)。

### 9.2 Kronecker 打包

把多项式系数看成彼此留有足够空隙的大整数槽，槽位宽至少：

```text
slot_bits = 2*S + max(1, ceil(log2(max_input_coeff_count)))
slot_words = ceil(slot_bits/bpw)
slot_stride = slot_words*bpw
```

第 k 个系数从 `k*slot_stride` bit 开始，每个 NTT digit 有 bpw 个有效 bit。不能把系数 k 写到第 k 个 digit；一个系数占很多 digits。

`pack_direct=1` 时 GPU pack kernel 直接写 NTT 引擎的 A/B scratch，省去单独 packed input 的临时显存与后续 D2D。resident 路径按 metadata gather；普通路径从上传后的 raw 输入打包。NTT 仍需要 A/B 工作区，优化没有让输入频谱消失。
出处：[NTT 形状规划](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3045)、[普通 pack](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3243)、[直接 scratch pack](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3356)、[resident gather](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3321)。

### 9.3 精确性预算

卷积在 `p_ntt=2^64−2^32+1` 上执行。单个 digit 卷积的累加项数 L_terms 满足：

```text
L_terms = max_input_coeff_count*slot_words
L_terms*(2^bpw−1)^2 < p_ntt
```

结合零填充和槽位容量，该不等式保证 NTT 结果能恢复为普通整数卷积，而不是只得到模 p_ntt 的模糊结果。它是整数精确性约束，不是浮点误差估计；每种形状规划/使用时检查，不能直接套用其他位宽的 bpw。
出处：[精确界计算](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2459)、[调用前检查](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3154)。

### 9.4 Forward、逐点乘法和 inverse

GPU fused forward 对 A/B 做 DIF：自然顺序输入、bit-reversed 频谱输出；inverse 使用 DIT，消费相同频谱顺序并恢复自然顺序。逐点相乘及 inverse 长度缩放融合进 inverse tile pass，避免单独扫数组；不需要独立 bit-reversal kernel。

多个 stage 在 tile 内融合，临时寄存器/共享内存和全局存储属于底层 NTT 实现。arena 复用 FuseCtx、工作区和表；当前仍是普通完整卷积，不是 Prime95 的 NO_UNFFT 或共享父谱算法。
出处：[实际 NTT passes](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3336)、[融合 inverse](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3388)、[NttArena](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1744)。

### 9.5 Carry 恢复整数 digits

inverse 得到 digit 卷积系数后，GPU 把超出 bpw 的部分向高 digit 传递。常见形状用 carry cone 把有限依赖链展开到寄存器，并在 kernel 内完成 binary propagation；超出 cone 支持范围时执行旧 height-reduction 路径再收尾。

必须保留 carry convergence/residual verdict。defer carry 及 carry batch 只调整检查完成时机，不允许跳过中间 chunk 的检查；当前测量组合 carry batch=0。
出处：[carry 执行](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3401)、[chunk verdict 生命周期](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3601)。

### 9.6 系数 mod N 归约

GPU 每线程处理一个输出系数：

1. 检查对应源槽的高位界，即使该系数不在请求输出窗口内也检查。
2. 对请求窗口，将 bpw digits 重组为 little-endian 64-bit limbs。
3. 当前默认直接长除法：把 N 左移规范化，借助高两字/一字除法及预计算 reciprocal 估计商，逐字乘减，过估时加回修复，最后去除规范化 shift，得到普通 `C mod N`。
4. `NTT_S4_OLDTAIL=1` 或 forensic dump 保留旧 REDC 消元再 Montgomery restoration 路径，供同二进制对照/诊断。

因此当前主线已经接入长除法；“还需要把原语接进 reduce”的历史计划已完成。它同时替代旧消元和域恢复两部分，不是只换最后一行乘法。
出处：[归约 kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2096)、[直接长除法分支](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2149)、[长除法 helper](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1939)、[常数预计算](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2414)。

### 9.7 输出窗口、检查和结果交付

- output window 开启时，仅归约/返回 `[first,first+count)` 的系数；**仍执行完整 NTT 和 carry**，不是截断 NTT。
- 普通调用用 chunk 输出区，D2H 到交替 pinned staging；event 保证前一 chunk 被消费后才能覆写，调用结束排空待处理结果。
- resident G 树调用把结果 scatter 回下一个 raw frontier，常规模式不把中间节点读回 CPU。
- GMP oracle 根据抽样/全检查策略核对独立重构的系数；可选异步 worker 使用快照和 fence，必须排空后才报告成功。抽样并非全部系数逐个做 GMP。
- 关闭 pinning/异步的回退保留 blocking D2H；final whole readback 开关用于恢复原完整布局对照。

出处：[归约窗口检查](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2113)、[scatter/D2H/event](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3755)、[oracle 检查](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3150)、[oracle drain](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3105)。

## 10. 第 5 步：小素数补充检查与曲线 workspace

进入 `run_batched()` 后建立 S3Workspace，上传 N、Q、a24、Montgomery one；点数组和累积数组按需增长并复用。

先枚举 `(B1,B2]` 内且 `p≤D/2` 的素数。这部分不能完全依赖 `i≥1` 的 giant/baby 覆盖，包含 D 的相关小素数。未启用复用时，GPU批量计算 `[p]Q`，CPU对其Z做GCD；启用 `NTT_SMALL_PRIME_REUSE=1` 且输入键匹配时，直接使用实际baby索引j=p的归一化证明和非单位元GCD。p|D等缺失点继续GPU ladder/CPU GCD。

baby批量逆元成功证明整批Z为unit；失败批保留逐点GCD，特别记录Z=0对应GCD=N，因为原仿射helper将它转为叶值0并返回成功。缓存不保存坐标，仅保存索引、输入键和非单位元GCD；原叶值、因子去重及prime hint处理保持一致。`NTT_SMALL_PRIME_CHECK=1` 重算整组小素数点并逐项核对复用证明，用于门禁，不用于性能对照。

得到非平凡因子时通过 `s3_record` 记录因子及该 p；当前流程仍继续后面的树运算。
出处：[workspace](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8788)、[小素数分支](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9606)。

## 11. 第 6 步：预计算反转 F 的 Newton 逆

当 G 批次不止一个时，准备：

```text
rev_P(F)[k] = F[P−k]
finv = 1/rev_P(F) mod Y^(P+1)
Newton: g_new = g*(2−rev_P(F)*g) mod Y^next_length
```

F monic，因此 rev(F) 的常数项是 1；不需要假设模 N 是素域。每轮精度最多翻倍，两次多项式乘法经 GPU；2−ag 等系数构造由 CPU 完成。inverse 长度不能随零尾项 trim，否则 Newton 精度增长可能失效。

**生命周期：** finv 在全部 fold 期间保留，scaled 下降根转换可复用；不是每一个 giant 批次重算。仅一批 G 时可能不预先建立，后续下降按需处理。
出处：[一次性 finv](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9686)、[Newton 实现](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4679)。

## 12. 第 7 步：按 point chunk 生成 giant 点

G 树每批最多 P 点；外层 point chunk 可含整倍数的 G 批次。当前点预算以 256 MiB 两坐标 payload 为目标，并向上取整到整批，至少一批；**不是严格保证 256 MiB 峰值上限**，也不含 seeds、段积及其他区。

**大 chunk：chain。** CPU 枚举每段的前两个倍数和 DQ，GPU ladder 计算 seeds。默认仍读回普通域seed，在CPU转成Montgomery image再上传；`NTT_GIANT_SEED_DEVICE=1` 让ladder直接输出Montgomery image并留在S3Workspace的设备X/Z缓冲，chain以stride=2读取两个连续seed和末尾DQ，不经CPU转域。每个 GPU 线程顺序生成一个 chain block，使用差分加法生成后续 `[iD]Q`；默认每线程 64 点。

**小 chunk/对照：ladder。** 默认点数小于 32768 时逐点 ladder；`NTT_GIANT_LADDER=1` 或调整 chain_min 可强制。chain 固定 seed/传输成本使它在很小形状上未必划算。

chain 遇到因子模上的退化点后可能影响本 block 后续坐标；64 点块限制传播范围，但不是“自动重算所有退化点为独立 ladder”。可选 chain check 运行两种方法，按仿射点比较。

**输出/访存：** giant X/Z 当前都 D2H 到主机 `gx/gz`，并非从 chain 输出直接构造 device 叶。device seed候选去掉六个seed设备临时数组及其上传，直接借用workspace；ox/oz、段积设备buffer仍按chunk malloc/free。S3Workspace 存在不代表这些临时区已经全部池化。
出处：[点预算](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9712)、[选择阈值](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9746)、[chain 组织](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9027)、[顺序 chain kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1011)、[D2H 与释放](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9131)。

## 13. 第 8 步：16 点段积、可逆性分类与批量求逆

每个 point chunk 按 **16 点 segment** 建立 Z 积网格：chain 在 GPU 算，ladder 回退在 CPU 算。当前默认 `NTT_GFINV_SEG_EXACT=1`，两条路径都返回各自**实际坐标字的普通模N乘积**。

设一个段实际有 m 点，返回坐标为 `Ztilde_j`：

```text
Gamma_seg = product(实际返回 Ztilde_j) mod N
旧 chain GPU 初始累积 p = Gamma_seg*R^(-(m−1)) mod N
当前默认 m>1 时：Mont(p,R^m) = Gamma_seg
m=1：原始 Z 字已经等于 Gamma_seg，直接返回
```

旧式来自以第一个Z为初值、再执行m−1次Mont乘法。当前每个非单点段额外一次Mont乘法恢复普通乘积；R^m表按modulus/workspace缓存，末段使用实际m，16点段W70的表为9520bytes。`NTT_GFINV_SEG_EXACT=0` 保留旧尺度作为诊断对照；ladder host段积原本就是普通乘积，不受此开关影响。

本轮已动态复现旧尺度，并对64/129/4423/5261位输入、完整段/尾段/单点建立独立GMP逐段比较。`NTT_GFINV_SEG_CHECK=1`还比较真实chain的每个段与实际返回gz的普通乘积；独立Python验证单位元形状的monic叶指纹。结果及局限见§28，不能将旧路径互比当作这次尺度证明。

默认对每个查询段积调用 `mpz_invert`。`NTT_GFINV_BATCH=1` 时：

1. 对现有段积网格取最多 64 段的对齐窗口，不改变 16 点分段。
2. 生成 prefix，求一次组合积逆元，逆扫描恢复各段逆元。
3. 同窗口重复查询从缓存复制结果。
4. 组合积不可逆时逐段 invert，准确区分组内好段和坏段；不能把整组全部判坏。

缓存在 point chunk 的 G 批次循环外，跨 G 批次片段共享，只有一个窗口；离开 point chunk 清除 GMP 对象。这里的“64 段求逆窗口”和“16 点 segment”“64 点 chain”“64 叶 GCD 块”是四个不同概念。
出处：[段积 kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1087)、[Montgomery 模乘定义](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:678)、[host 普通段积](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:555)、[chain 段积读回](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9136)、[GfinvBatch](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:416)、[缓存生命周期](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9823)。

## 14. 第 9 步：CPU 构造 mixed giant 叶，累计 Gamma 逆元

**好段：** 段积可逆，说明段内每个 Z 都可逆。直接构造 `[-X_i,Z_i]`，即：

```text
Z_i*Y−X_i = Z_i*(Y−u_i)
```

不逐点求仿射逆元：取负和复制 limb 即可。把实际gseg的逆元只累计一次到名为Ginv的变量；默认精确段积让chain和ladder都与各自普通叶尺度一致。诊断旧模式仍带第13节的R幂。

**坏段：** 对当前片段尝试原有 affine 批量路径，必要时逐点处理；非零不可逆 Z 的 GCD 通过 `s3_record(..., count_hit=false)` 保留。这些叶采用 `[-affine_helper_result,1]`，不计入 Gamma。

`first_touch` 防止跨 G 批次的同一 segment 被多计；段网格索引是 point chunk 的局部索引，不是全部 giant 的全局索引。末段使用实际 seglen。

**输出：** 主机 `bleaf`（每点 2W words）和累积 Ginv。即使 G 树中间层驻留 GPU，当前这一步仍在 CPU。
出处：[射影叶合同](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9846)、[first_touch/Gamma](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9891)、[好段写叶](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9922)、[坏段回退](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9956)。

## 15. 第 10 步：构造本批 G，保留根

G 树只服务 fold，后面不沿它下降，因此允许释放孩子。

**resident 路径：**

1. CPU 根据 leaf degree 构造树次数和 offset metadata。
2. 借用既有 pinned output staging 将 leaf H2D 到 raw A；event 保护复用，不能覆盖尚未完成的 D2H/H2D。
3. raw A 容纳最大 leaf frontier，raw B 按首 parent frontier 定容。每层仅保留当前/下一 frontier。
4. CPU 按孩子长度分组，上传 offsets；GPU gather → 公共 NTT/归约 → scatter。
5. 常数 1 子树 passthrough 用 D2D；空 padding 不生成伪根。
6. 最终仅 D2H 根 `(actual_batch_points+1)*W` words；返回的 heap 只含可用根 payload。中间节点不读回，临时 metadata 释放。

**回退：** `build_groot_select` 检查 S4、root-only、direct pack、关闭 final whole readback/host pack 等条件；不满足时用主机编排的 `build_tree_flat`，并统计 fallback。该主机路径仍可 root-only，在消费后释放子节点。

G-root 使用的 raw pair 被独占借用，不能同时被普通 raw upload 覆写。leaf staging 只复用已有 pinned 容量，不表示整个 bleaf 已从主机消失；不可用时额外 flatten 到 pageable vector 再上传。
出处：[选择条件](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4517)、[resident 树](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4354)、[pinned 叶 staging](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4383)、[按层 gather/scatter](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4458)、[仅根读回](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4510)。

## 16. 第 11 步：fold，逐批 `H←G·H mod F`

第一批直接 `H=G_0`；后续批次执行：

```text
T = G_b*H
若 deg(T)<P：H=T
否则：
    k = deg(T)−P+1
    qrev = low_k(rev_deg(T)(T) * low_k(finv))
    q = reverse(qrev)
    qb_low = low_P(q*F)
    H = low_P(T)−qb_low mod N
```

常见完整形状有三次多项式乘法：G*H、逆序商、q*F。finv 跨所有批次复用；请求低系数窗口减少归约/读回，但底层仍完整卷积。

**分工/访存：** G 根、Fpoly、H 在 CPU；`cp_mul` flatten/补零并经 GPU 运算，结果回 CPU；逆序、q 构造、`cp_coeff_sub` 和 trim 由 CPU/GMP 处理。此阶段没有完整 H/F/G 驻留的 device fold，仍有反复 H2D/D2H 与临时 CPoly/vector 构造。

首次 H 可能仍为 P 次；只有一批 G 时没有 fold，下降入口需自行处理 H mod F。
出处：[初始化 H 与 fold](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10065)、[cp_mul 主机准备](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4647)、[系数减法](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4608)。

## 17. 第 12 步：CPU 消除射影缩放

所有好段的射影叶使 giant 积带上 `Gamma=∏(实际返回 Z_i)`。fold 在模 F 环中保持这个常数因子，循环完成后 CPU 逐系数做：

```text
H[k] = H[k]*Ginv mod N
```

没有额外 GPU kernel；每系数 words↔GMP、乘法、取模、导出。代码检查 `projective leaves == Gamma 覆盖 points`，并检查 Ginv 可逆。这两个断言证明覆盖数量和 unit 属性，**不能证明 Ginv 数值恰好等于实际 Gamma 的逆元**。

默认精确段积下，按first_touch累计一次得到 `Ginv=Gamma^−1 mod N`。独立段积GMP比较验证实际gz的乘积，单位元端到端Python叶指纹另行验证unscale结果。两个覆盖/unit断言本身仍不构成数值证明。
诊断旧模式下，chain好段长度为m_s时，仍为 `Ginv=Gamma^−1·R^(Σ_s(m_s−1))`；本轮在65点单位元形状复现了最终叶多出的这个R幂。可逆常数不改变GCD，但精确系数不同；旧生产指纹不能继续用作当前默认精确unscale的固定叶参考。坏段仍按原回退代表处理，本轮没有消除chain退化传播或g=N饱和块问题。
出处：[unscale 与覆盖断言](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10135)。

## 18. 第 13 步：只沿 F 树下降一次，得到 `H(x_j)`

最终叶按 baby 索引排列，而不是 giant 索引。主循环已经把所有 giant 批次折叠进 H，所以无需逐批做一次完整下降。

### 18.1 当前优化组合：ordinary-coefficient scaled descent

对 monic 节点多项式 M、次数 d，状态可解释为：

```text
r = H mod M
S_M = rev_(d−1)(r) / rev_d(M) mod Y^d
```

根状态由逆序 H 与 finv 相乘取低 P 项得到；如果 H 次数还达到 P，先做根余式。对目标孩子次数 a、兄弟次数 b：

```text
child_state = coefficients [b, b+a) of
              parent_state * rev_b(sibling_polynomial)
```

同一层按 `(a,b)` 分组，一组 batch 乘法。兄弟次数为 0 时直接拷贝父状态，目标次数为 0 时跳过；每层次数守恒和长度都检查。

到 monic 线性叶 `Y−x_j` 时 d=1，状态只剩一个系数，就是 `H(x_j)`。无需为每个孩子重算 Newton inverse/完整 divmod。

**内存/分工：** F 树和 cur/next 状态仍在主机；CPU pack 组输入、反转兄弟、组装下一层，GPU 乘法并返回窗口。只保留两个状态 frontier，但保留 F 全树。当前尚未共享同一父状态的 NTT 频谱。
出处：[scaled state 独立解释](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7774)、[根转换](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7814)、[child 窗口递推](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7863)、[叶输出](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7901)。

### 18.2 默认及可选分支

- S4 开启、scaled 关闭：`descent_batched`，逐层按形状批量余式除法；线性叶有单独 Horner 路径。
- S4 关闭：`descent_slow`；单次多项式乘法仍可使用共享 NTT，精确系数的模 N 归约由主机执行，不能简单叫“整个算法纯 CPU”。
- `NTT_S5_ON=1`：`descent_batched_dev`，另一套设备下降/叶驻留实验；与 scaled 不可同时启用，**不属于本次 96.848 秒结果**。
- descent check 可跑 slow 参考逐叶比较；检查开关增加额外工作，clean 性能测量应与其区分。

出处：[下降分支选择](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10256)、[batched 下降](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4825)、[线性叶 Horner](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7180)、[S5 设备下降](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:6697)。

## 19. 第 14 步：GPU 分块乘积，CPU GCD 和可选命名

scaled/default S4 下降的全部叶在 CPU `values`。先 flatten 为 `P*W` words，再 H2D 到 ws.dvals。GPU 每 **64 个叶值** 求分块乘积，D2H 返回每块一个 W-word 值；它不是把全体叶只归约成一个最终累加器。

该 kernel 对普通叶值连续使用 Montgomery 模乘。m 个叶的实际输出是 `product(values)*R^(-(m−1)) mod N`，没有再恢复普通乘积尺度。这里输出只用于 GCD，R 是 unit，所以 GCD 与普通块乘积相同；不要把返回的 limb 数值当成普通乘积做逐字参考比较。

CPU 对每个块乘积做 `gcd(block_product,N)`；当得到 `1<g<N` 时进入该块，检查单叶 GCD：

- 不命名或超出命名预算：记录单叶的非平凡 GCD，因子保留。
- 允许命名：对该 baby 索引 j 扫 `i=1..I`，生成 `iD−j` 和 `iD+j`，只保留 `(B1,B2]` 的素数；GPU ladder 计算这些候选倍数，CPU 对 Z 做 GCD确认并记录 `hit_primes`。
- 没有确认的素数名也保留叶 GCD，记为 unnamed。

`NTT_NAME_MAX` 与 block budget限制诊断成本；`hits/hit_primes` 是归因计数，不能直接当作完整因子集合大小。`s3_record` 去重因子、排除 1/N，核对是否整除 N。

**当前代码的具体边界：** 块 GCD 为 N 时，不进入上述单叶检查分支；当前 batched 尾部也没有“最后再求一次全部块乘积 GCD”的代码。不能把早期流程文档的“每块+末尾 GCD”当作现状。饱和块递归定位是否需要补充，需另建独立用例；本报告仅记录静态行为，没有做新复现。

出处：[block product kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8758)、[当前叶上传](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10381)、[块 GCD 条件](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10460)、[候选扫描/确认](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10521)、[unnamed 保留](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10558)、[record 合同](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9252)。

## 20. 第 15 步：合并、排空检查、计时和释放

`run_batched` 返回后先排空 GMP oracle，合并 baby 转换收集的退化因子字符串，再输出 factors/hit_primes、精确 full/main、NTT/归约/拷贝/内存统计。当前 CLI 输出实验结果，没有在这里完成生产队列回写或 Prime95 交付替换。

计时边界：

- `shape`：baby 索引枚举；其值已加入 init，不要再加到 total。
- `init`：Q 准备完成后的上下文/归约准备、自测、baby/F 树，加 shape。
- `main` / `stage2.elapsed`：`run_batched`、GCD/命名、最终 oracle drain 等。
- `total=init+main`：Stage2 完整时间，不含 Stage1。
- 进程 wall：还有程序启动、Stage1 等开销，不能和 total 混用。
- `--curves K` 复用同一 Q/F/init 重复运行，本入口不是 K 条不同 sigma 曲线的正式调度；clean 要求单次且没有额外诊断 fixture/real dump/额外 S2 对照。

所有权结束时释放 S3Workspace、segment GMP cache、S4/arena、点与树的主机容器。arena 能跨调用复用，但缓存高水位直到 owner 生命周期结束才归还；局部 vector 的 capacity 与整个进程峰值不是一个指标。
出处：[init 计时](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11343)、[返回后的 drain/merge](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11380)、[full 输出合同](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11399)、[workspace 析构](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8801)。

## 21. 当前数据流和性能状态

### 21.1 哪些部分还会往返主机

1. baby ladder：坐标回 CPU → GMP 归一化 → F 叶。
2. F 树：各层系数结果回 CPU，完整树留在主机。
3. giant chain：seeds GPU→CPU→GPU；全部 giant X/Z、段积回 CPU。
4. giant 叶：CPU 构造 → 叶 H2D；G 树中间层驻留，根仍 D2H。
5. fold：主机 CPoly 构造/逆序/减法，三次 GPU 乘法的输入输出仍往返。
6. scaled 下降：每层主机 pack/组装 frontier，GPU 窗口乘法后读回。
7. 末尾：主机叶再次上传做 block product，再返回块乘积求 GCD。

所以 G-root residency、direct pack 与窗口优化已经减少了特定传输和临时区，但当前 Stage2 还没有形成从点输出一直驻留到因子累积的完整 device pipeline。

### 21.2 段积求逆阶段的已有测量

最近 GPU1/RTX4060 Laptop 的同二进制 ABBA 使用 M4423、sigma26、B1=1000、Stage1 extra12、actual B2=2011326186870、D1231230、P115200。I=1633592，15 批 G、14 次 fold。

- 段积 individual 对照完整均值 **99.0950825 秒**；batch 候选 **96.8483615 秒**，少 2.246721 秒/2.267%。
- main **84.4200835→82.224508 秒**；init **14.6749995→14.623853 秒**。
- 段逆元阶段 **3.8715→1.532 秒**；102100 次逐段 invert 降为 1598 次组合 invert。
- 候选主要阶段：giant **14.5115 秒**、G trees **27.9555 秒**、fold **18.4775 秒**、descent **8.854 秒**、finv **2.158 秒**。具体项以原 summary 为准；阶段间存在其他 host 工作与 drain，不能把这些数当成互不重叠的完整成本账。
- `t_reduce=9.808 秒` 是上述多项式阶段内部的归约 kernel 时间，不能再加到总时间。
- 模 N 归约系数 **40218760**，H2D/D2H 主账 rounded **11.37/5.46 GiB**，两边相同；该账不包括全部 metadata 等传输。
- 新 GMP limb 缓存峰值 **143944 bytes**；整卡 NVML峰值均 **5027 MiB**；观察到的进程 private 峰值均值 **7938→8041.5 MB**，未证明 RAM 减少。
- 约 1Hz full GPU busy 均值约 **77.94%**，不是 SM occupancy；当前仍存在主机准备和等待空档。
- 候选仍比既有 Prime95 CPU 单核 full90.460秒慢约 **7.062%**；CPU本轮未重跑，D/degree/检查不同，不能当成严格同实现比较。

原始证据：[ABBA summary](D:/code/MPA-OpenCl/build_cuda_cmake/_gfinv_ab_20261003/summary.json)、[GPU1 采样](D:/code/MPA-OpenCl/build_cuda_cmake/_gfinv_ab_20261003/gpu1_summary.json)、[开发日志 §50](D:/code/MPA-OpenCl/docs/DEV_GPUOWL_NTT_NOTES.md:3760)。build 证据目录被 Git 忽略，其他 checkout 未必含这些本机结果；算法说明与源文件链接不依赖该目录。

### 21.3 正确性覆盖和未完成部分

最新同一二进制已有真实 baby/F 独立门禁12/0、段积 batch门禁16/0、原完整188/0、scaled41/0、G-root87/0；冻结已知因子 **59649589127497217**、hit prime **114713** 与独立完整叶指纹恢复。

当前尚未完成的工作包括生产 save/队列接入、完整 device leaf/fold、父状态共享 NTT、饱和 GCD 块的独立行为验证、坏段密集场景性能及整体 RAM 高水位收敛。chain 段积/Gamma 的独立尺度验证已在§28补充；门禁通过不等于其余集成和边界都已实现。
出处：[真实入口门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_real_baby.py:1)、[段积门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_gfinv_batch.py:1)、[scaled 门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_scaled_gpu.py:1)、[G-root 门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_groot_device.py:1)。

## 22. 阅读及运行定位

- 总入口：[run_real](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10694)。
- 主算法：[run_batched](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9575)。
- 公共 GPU 多项式乘法：[poly_mul_batch_modN](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3392) → [ntt_poly_mul_batch_dev](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3778)。
- 构建：[build_stage2_tree_gpu.ps1](D:/code/MPA-OpenCl/tools/build/build_stage2_tree_gpu.ps1:1)，默认 sm89、单 TU、链接 GMP；`-Rebuild` 强制重新编译。
- 同二进制 A/B：[bench_stage2_reduce_ab.ps1](D:/code/MPA-OpenCl/tools/bench/bench_stage2_reduce_ab.ps1:1)，最新求逆优化目标 `-Target gfinv_batch`。脚本默认 N/Stage1Extra 与最近 M4423/extra12 测量不同，复现时需显式传参及预期 Q。
- 演进记录：[DEV_GPUOWL_NTT_NOTES.md](D:/code/MPA-OpenCl/docs/DEV_GPUOWL_NTT_NOTES.md:3760)。本文说明当前步骤；旧阶段测量和旧注释应结合修正记录阅读。

§1–25 初稿仅整理源码和已有证据；本轮连续缓冲 fold 的实现及新测量见 §26。

## 23. 可量化成本模型：约定、计算量与周期

### 23.1 参数与统计口径

以下公式针对当前 `run_real → run_batched` 路径的一条曲线，假定没有提前发现因子而退出。Stage1 的计算不计入 Stage2。N 表示待分解整数；NTT 长度另记为 n，避免把两者混淆。

| 符号 | 含义及本例值 |
| --- | --- |
| S、W | `S=bit_length(N)`，`W=ceil(S/64)`；M4423 为 S=4423、W=70，每个定宽模 N 系数为 8W=560 bytes |
| B1、B2、D | Stage1 上界、实际 Stage2 上界、轮转间隔；本例 1000、2011326186870、1231230 |
| P、K、h | `P=φ(D)/2` 个 baby；`K=2^ceil(log2 P)` 为树补齐叶数，`h=log2 K`；本例 115200、131072、17 |
| I、G | 当前程序生成 `I=floor(B2/D)+2` 个 giant；`G=ceil(I/P)` 个 G 树；本例 1633592、15 |
| n_b | 第 b 个 G 树的实际叶数，`n_b=min(P,I-bP)`，b 从 0 开始；末批 20792 |
| C、J | giant chunk 点数及 chunk 数。预算 A=256 MiB 时，`C=P·ceil(max(P,floor(A/(16W)))/P)`，`J=ceil(I/C)`；本例 C=345600、J=5 |
| L、s、g、q | chain 每线程最多 L=64 点；段积 s=16 点；CPU 段积批量求逆 g=64 段；GCD 块 q=64 叶 |
| c_j、E、U | chunk j 的点数；`E=Σ_j ceil(c_j/s)` 个段；`U=Σ_j ceil(ceil(c_j/s)/g)` 个求逆组；本例 E=102100、U=1598 |
| M、r、n、m | 一组多项式乘法中，输入较长一侧的**系数数** M、每系数 NTT digit 数 r、NTT 长度 n，以及本次 GPU chunk 的独立乘法数 m；不是多项式 degree |

`I` 包含覆盖边界的额外 giant，并非 `π(B2)-π(B1)`；当前不会为每个待测素数逐一生成点。D 自动选择还取决于可用显存、成本搜索和内存预算，所以性能不能只写成 B2 的函数。GiB/MiB 均为二进制单位；后文 bytes 均可直接复算。

出处：[baby/D/I 的确定](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10927)、[chunk 预算与取整](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9719)、[实际实验参数与统计](D:/code/MPA-OpenCl/build_cuda_cmake/_gfinv_ab_20261003/summary.json)。

### 23.2 点生成、归一化与段积

**GPU 点运算。** 当前 `xDBL` 是 5 次 Montgomery 乘法，`xADD` 是 8 次，包括以乘法实现的平方；这里不是常见的 6 次乘法差分加法版本。对于正整数 k，令 `ℓ(k)=floor(log2 k)+1`：

```text
ladder 内核内部 Montgomery 乘法数 = 5 + 13(ℓ(k)-1) = 13ℓ(k)-8
普通 ladder 点输出再将 X、Z 转成普通域：+2
baby 点生成总数 = Σ_{j∈baby_j} [13ℓ(j)-6]
                   ≤ P [13ℓ(floor(D/2))-6]
```

k=0 走单位元特殊分支，不套用上式。对本例 D，baby 上界为每点 254 次，共 **≤29260800 次 Montgomery 乘法**。它是按最大标量位长的上界，实际小标量更少。

chain 的每个短链先准备两个 seed，随后每新增一个点执行一次 `xADD`。设各短链长度为 a_t，则 chain 主循环为 `8Σ_t max(a_t-2,0)` 次 Montgomery 乘法；seed ladder 另按其标量位长累加，还包括每 chunk 的 DQ 准备与域转换。因此 chain 主循环为 O(IW²)，逐 giant ladder 则为 O(I·log(B2)·W²)。不能把 seed 或短链尾巴忽略后称为精确总数。

当前单线程 Montgomery 乘法采用 limb 乘法和 REDC，主要包含约 `2W²` 次 64 位乘加，另有进位、循环、模加减等操作。于是点生成主项可写为 `2W² × Montgomery 乘法总数`。每个“乘加”内部还拆为低/高位乘法和带进位加法，**它不是单条 GPU 指令，也不是一个时钟周期**。

**CPU baby 归一化。** 所有 Z 可逆时，256 点一组的实现执行 `4P` 次 GMP 模乘和 `ceil(P/256)` 次 GMP 求逆；本例为 **460800 次模乘 +450 次求逆**。记一次 S 位 GMP 模乘成本为 `M_CPU(S)`，求逆为 `Inv_CPU(S)`，则主项为：

```text
T_baby_affine ≈ 4P·M_CPU(S) + ceil(P/256)·Inv_CPU(S) + 导入/导出/构造叶开销
```

**段积及其求逆。** 旧GPU段积从第一点Z开始，执行 `I-E` 次Montgomery乘法，本例1531492次；当前默认精确模式另加 `E_>1` 次，E_>1为实际长度大于1的段数。本例没有单点段，合计 **1633592 次**。CPU成功组执行 `3E` 次模乘和U次求逆，替代旧路径E次求逆；本例 **306300 次模乘 +1598 次求逆**。本例求逆调用数减少98.435%，不是整个Stage2加速98.435%。组失败时退回逐段求逆，可能额外执行该组每段一次求逆；含非单位元时不能沿用全成功成本。全局Gfinv累积和H缩放另有O(E+P)次CPU模乘。

段积精确尺度的当前公式见§13/§17，本轮独立动态证据见§28；§25–§27保留此前测量，不将新增补偿成本混入旧二进制统计。

出处：[xDBL](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:725)、[xADD](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:768)、[ladder](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:809)、[chain 输出与段积](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9117)、[段积 kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1087)、[baby batch inverse](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11209)、[Gfinv batch inverse](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:440)。

### 23.3 一次 GPU 多项式乘法的成本

先用每组的实际 ma、mb、输出窗口长度 w 建立模型，不能一律假定完整 2P 次卷积结果都归约。

```text
M = max(ma,mb)
b = 2S + max(1,ceil(log2 M))          # slot_bits，位
r = ceil(b/bpw)                      # slot_words，Goldilocks digit 数
n ≥ 2Mr，n 为 planner 选择的 2 的幂
k = log2 n
```

bpw 由精确性界和 planner 决定，满足 digit 卷积不溢出 Goldilocks 素数 `p=2^64-2^32+1` 的条件。r 与 n 不是只由 W 线性推出；跨 planner 阈值时会跳变。用未融合 radix-2 网络计数，一组 m 个独立乘法的“两次正变换+一次逆变换”为 **`(3/2)m n k` 个 butterfly**，另有约 `2mn` 个 pointwise/逆变换缩放模乘。当前融合、radix-4 和特殊 twiddle 会改变实际指令数，前者是统一比较算法工作量的参考计数。

输出 w 个系数时，对 `mw` 个系数执行 S4 归约。令 `d=ceil(r·bpw/64)`，实际归一化后的非零 limb 数为 d'，`d'≤d+1`；长除法每个系数最多尝试 `Q=max(0,d'-W+1)` 个商位，每商位做 W 个 64 位乘法减去整段除数，另含 2-by-1 商估计及至多两次 add-back 修复。因此：

```text
长除法主要 64 位乘法数 ≤ mw·W·max(0,d-W+2)
常见 d≈2W+O(log(M)/64)：主项约 mw·W²
输出装配/搬运与归一化：O(mw·(d+W)) limb 操作
```

窗口省掉的尾系数可以省归约和输出；当前仍要执行既定 n 的完整 NTT，不能按窗口比例宣称 NTT butterfly 同比例减少。线程局部数组的静态载荷为 `8[(2NW+4)+NW]=24NW+32 bytes/线程`，其中 NW 是模板容量而非实际 W；寄存器/本地内存分配及 spill 取决于编译器，不能直接乘线程数当成 `cudaMalloc` 显存峰值。

**周期表示。** 若需要硬件周期，使用待校准的 `c_butterfly`、`c_point`、`c_qdigit(W)`：

```text
C_poly_work ≈ m[(3/2)n k·c_butterfly + 2n·c_point
                + w·Q·c_qdigit(W)] + pack/carry/内存指令成本
C_point_chain ≈ 链上 Montgomery 次数 · C_Mont(W)
```

这里的 C 是工作量或单依赖链周期模型；并行 kernel 墙钟需要再考虑并发、占用、带宽和启动开销。尚无当前二进制对应的周期级测量，不能给各 primitive 编造固定周期。CPU GMP 同样用 M_CPU/Inv_CPU 保留可校准系数；用墙钟秒数乘 GPU 频率会混入 CPU 工作与空闲，不能代替算法周期数。

出处：[shape/slot/NTT 长度](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3045)、[乘法 chunk 预算](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3506)、[长除法](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1939)、[S4 kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2096)。

### 23.4 将乘法模型应用到整条 Stage2

对每个实际乘法对 i 的 `(ma_i,mb_i,w_i)` 求出 n_i、k_i，整条 NTT 参考工作量为 `Σ_i (3/2)n_i k_i` butterfly；归约主项为 `Σ_i w_i W Q_i` 次 64 位乘法。这比一个统一“大 NTT 次数”更能解释底层小乘法与顶层大乘法的差异。

| 步骤 | 数量/主项，忽略特殊提前退出 |
| --- | --- |
| D/baby 索引准备 | 遍历约 D/2 个 j，检查 gcd(j,D)，Euclid 操作数上界 O(D log D)；D 成本搜索另计 |
| 小素数补充分支 | `K_s=max(0,π(min(B2,floor(D/2)))−π(B1))`；GPU `Σ_p(13ℓ(p)−6)` 次Mont，CPU K_s次GCD及约D/2次整数素性判断；本例50108点/11544588次Mont，归于pre而非giant |
| F 树 | 实际两子树都有正 degree 的乘法 P-1 次，本例 **115199**；degree=0 的补齐分支复制，不做等价大乘法 |
| rev(F) 的 Newton 逆 | 目标 k=P+1，最多 `2ceil(log2(P+1))` 次截断乘法，本例 **34**；每轮长度翻倍，另有 O(P) 系数加减/复制 |
| G 树 | `Σ_b(n_b-1)=I-G` 次实际乘法，本例 **1633577**；当前中间层驻留，不代表这些乘法被消除 |
| fold | G-1 次，每次通常 3 次多项式乘法，本例 **42**；另外 O((G-1)PW) 主机逆序/截取/减法/复制 |
| scaled 下降 | root transform 1 对加每个实分叉两对，`2P-1` 个乘法对，本例 **230399**；一轮可批量包含很多对，不能把 pairs 当作 kernel launches |
| block product/GCD | `P-ceil(P/q)` 次 Montgomery 乘法；本例 **113400**；CPU 检查 `ceil(P/q)=1800` 个块 GCD，命中块才检查叶 |
| 因子命名 | 每命中叶最多检查 2I 个 `iD±j` 候选；还受命名预算限制，本例没有命中，命名时长为 0 |

对于 P 为 2 的幂的平衡树，F 第 l 层有 `P/2^l` 个 degree `2^l` 的节点；构建该层的输入系数数为 `2^(l-1)+1`。于是可以逐层累加上一节成本。非 2 的幂时应按真实子树 degree 计算，补齐分支不会贡献同样的 NTT。

固定 D、S 和 planner 选项，I≈B2/D，G≈B2/(DP)：giant、G 树、fold 的累计量大体随 B2 线性增长，F/init、finv、下降和无命中的末尾 GCD 基本不随 B2 增长。改变 D 会同时改变 P、I、NTT shape 和峰值空间，应重新计算，而非只按 B2 比例外推。

出处：[F 树](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3937)、[Newton](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4679)、[主循环](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9575)、[scaled 下降](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7814)、[末尾 GCD/命名](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10432)。

## 24. 主机内存、显存、数据生成与传输

### 24.1 长期对象与阶段临时区

以下首先列**系数载荷**，不含 vector 容器、allocator 碎片、GMP capacity、CUDA context、缓存及 oracle。不同阶段的峰值不能直接相加。

| 对象 | 逻辑载荷或分配模型 | 本例 |
| --- | --- | --- |
| baby 索引 | 8P bytes | 0.879 MiB |
| baby X/Z、两系数叶 | 每一份分别 16PW bytes；构造期间可能同时存在多份 | 每份 123.047 MiB |
| 小素数X/Z与S3 seed workspace | 小素数阶段CPU/GPU各16WK_s；S3设备坐标容量为 `16W·max(K_s,各chunk seed数)`，后续借用 | 本例各56120960bytes /53.521MiB，标量400864bytes另计；这份设备容量存活到tail结束，不再另加一份借用seed缓冲 |
| CPU 完整 F 树 | `8W[(h+1)P+2K]` bytes；各层 degree 总和 P，全部 2K 个槽含一个常数系数 | **1247.422 MiB** |
| F 树 metadata | degree 数组 `2K·sizeof(size_t)`；容器 `2K·sizeof(vector)`，另有每节点分配开销 | 64 位 size_t 数组 2 MiB；容器大小取决于 C++ ABI |
| H、finv、单个 G 根 | 分别约 8PW、8(P+1)W、8(n_b+1)W bytes | 每份约 61.52 MiB |
| giant chunk 的 CPU X/Z | 16c_jW bytes；所有 chunks 累计 16IW，不同时全留 | 最大 369.141 MiB，累计 1744.865 MiB |
| giant GPU X/Z 与段积 | `16c_jW +8Wceil(c_j/s)`，另加 seeds/上下文 | 最大约 380.676 MiB，阶段结束释放 |
| giant CPU 叶 | 16n_bW bytes，加容器；还与当前 chunk X/Z 等共存 | 最大 123.047 MiB |
| G 树 device 两层 frontier/raw | 叶到第一层峰值 `8W[2n_b+n_b+ceil(n_b/2)]`，约 28n_bW | **225792000 bytes /215.332 MiB**，实际计数相符 |
| G 树 metadata | 最大三组 K/2 个 64 位字，`12K` bytes；主机 staging 另计 | device 最大 1.5 MiB |
| scaled 两层 state | `16PW` bytes 的逻辑系数载荷 | **123.047 MiB**，容器与 pack 另计 |
| scaled pack/product | 依各组 ma/mb/窗口、vector capacity 决定，不能仅按 state 载荷估算 | 实测统计峰值 **395461376 bytes /377.141 MiB** |
| GMP 段逆元 cache | 2g+1 个 prefix/inverse 加少量 GMP 临时数，limb 数与 capacity 随 S 增长，O(gW) | 实测 limb 分配峰值 **143944 bytes /140.570 KiB**，不含 GMP 对象头 |
| block product 输出 | `8Wceil(P/q)` bytes | **1008000 bytes /0.961 MiB** |

CPU F 树公式是 `size()` 载荷；实际 `capacity()` 可以更大。fold 的 T 通常最多 2P+1 个系数，另有 reverse、quotient、qF、输入 pack 和结果副本；应按同时存活的各 vector capacity 求和。不会把所有临时名义大小相加后称为已测 RAM 峰值。

**NTT 显存模型。** 一次 m 组、长度 n 的 A/B/Q scratch 基本项为 `24mn bytes`。另有 retained output/residual、twiddle tables、S4 原始输入和归约输出：

```text
V_live ≈ 24mn + T_twiddle(n,配置) + V_retained_small
         + 16mMW + 8mwW + V_resident_frontier + 其他上下文
```

同一对象被多个 wrapper 借用时只计一次。非 direct pack 还可能增加 `16mn` 的 packed A/B 临时区；direct pack 省掉该区及对应 D2D。当前 workspace pool 复用大 A/B/Q；不同 shape 的 small buffers/table 仍会保留，峰值不等于单次公式的最小载荷。每组 chunk 选择检查 `(3nm + (2M-1)m)·8 ≤ batch_budget`，至少保留一组；**64 MiB batch budget 不是全程序显存上限**。

修复后 ABBA 候选：workspace owned 峰值 **3519583952 bytes /3356.537 MiB**；计入其 full 口径为 **3656064320 bytes /3486.695 MiB**。整卡 NVML 峰值 **5027 MiB**，还含其他分配/context 等，不能再把 workspace 加到 NVML。主机 observed private 峰值均值 **8041.5 MB（沿原记录单位）**，并非以上系数载荷之和或系统物理 RAM 占用。raw pinned **129025120 bytes**，window 双 staging pinned **258049120 bytes**；G 叶借用 existing pinned staging 最大 **129024000 bytes**，新增独立 host staging 计数为 0，不应重复算一份新增内存。

出处：[F 树容量跟踪](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3937)、[giant 点分配](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9117)、[arena 所有权与缓存](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1744)、[原始上传/pack 分配](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3645)、[本例内存高水位](D:/code/MPA-OpenCl/build_cuda_cmake/_gfinv_ab_20261003/summary.json)。

### 24.2 点/树边界的累计数据生成与 PCIe 传输

| 边界 | 累计字节公式 | 本例 |
| --- | --- | --- |
| baby 标量 H2D | 8P | 0.879 MiB |
| baby X/Z D2H | 16PW；CPU 再生成等大叶数据 | 123.047 MiB |
| giant seed 往返 | 每 chunk 约 `2ceil(c_j/L)+1` 个 seed 点；每点 X/Z 为 16W；ladder 标量另加 8 bytes/seed，转换及 chain 上传按实际路径计 | 约 O(IW/L)，相比全部 giant 坐标小一个 L 量级 |
| 小素数点 | 标量H2D `8K_s`，普通X/Z D2H `16WK_s`；host GCD只使用Z | 本例400864 /56120960bytes；S3workspace坐标容量保留后复用于seed，勿再加一份相同seed缓冲 |
| giant X/Z D2H | 16IW | **1829623040 bytes /1.7040 GiB** |
| giant 段积 D2H | 8WE | **57176000 bytes /54.527 MiB** |
| CPU giant 叶生成及叶 H2D | 分别 16IW；是不同方向/位置的数据事件 | 每项 **1.7040 GiB** |
| device G 树 metadata H2D | 24(I-G)，每实际乘法三字 metadata | **39205848 bytes /37.390 MiB** |
| G 根 D2H | `8WΣ_b(n_b+1)=8W(I+G)` | **914819920 bytes /0.8520 GiB** |
| block 叶 H2D | 8PW；当前 host leaf 路径 | 61.523 MiB |
| block product D2H | 8Wceil(P/q) | 0.961 MiB |

以上不包括 F 树、Newton、fold、下降的通用多项式上传/读回，也不包括检查样本、carry 状态、twiddle 构建和常量。G intermediate resident 的统计为 **2052522710 words**，对应约 **15.293 GiB** 的所统计内部 word 事件；这是 GPU 内部处理/写入账，既不是额外显存峰值，也不是 PCIe 传输量，更不是所有 DRAM 读写的完整总账。

### 24.3 通用乘法、窗口与 CPU pack 的累计量

对 host-input 的一组 m 对乘法，当前上传两边都按较长输入 M 补齐，因此原始 **H2D=16mMW bytes**，而非 `8mW(ma+mb)`。CPU staging memcpy 通常还搬相同字节一次。返回 host 的窗口为 **D2H=8mwW bytes**；device-output 的 G 树中间层这一项为 0。令集合 A 为所有 host-input 调用，集合 H 为所有 host-output 调用：

```text
raw_H2D = Σ_{i∈A} 16m_i M_i W
coeff_D2H = Σ_{i∈H} 8m_i w_i W
device 归约输出载荷 = Σ_all 8m_i w_i W
```

本例 output-window ledger 记录 source **50419417**、reduced/returned **40218760**、skipped **10200657** 个系数，省去 **20.23%** 的原完整输出归约；按 W=70，省去约 **5.3201 GiB 的输出载荷生成**。这不是全部省去的 PCIe 字节：有些原本就留在 device。全部归约输出载荷为 **20.9757 GiB**；实际 host-output window 计数 **762790490 words=5.6832 GiB**，其余包括 device resident 输出。主运行主账 rounded H2D/D2H 为 **11.37/5.46 GiB**，与 window 总账的阶段范围不同，不应强行相等或相加；独立根、点、metadata 等边界也不能无条件并入主账。

scaled 的 state 统计为 `(h+1)PW=145152000 words`，即各层累计逻辑 state 载荷 **1.0815 GiB**，并非同时占用；zero-degree/pass-through 不一定触发乘法传输。当前 flat prepare 累计 memcpy **1095269280 bytes /1.0200 GiB**、memset **2061516800 bytes /1.9199 GiB**；相对 control 分别避免 copy 与 zero **2448588800 bytes /2.2804 GiB**。这是 CPU 内存写入账，不能当作减少的 PCIe 量，也不能按节约 bytes 直接推导墙钟加速。

NTT 内部 DRAM 可用 pass 模型比较：一次全数组 read+write 为 `16mn bytes`。三次变换共 `16mn(F_A+F_B+F_I)`，另加 pointwise 读取 B、pack、carry 和归约；F 表示实际融合 pass 数，不是 k 个未融合 stage。未融合 radix-2 时 F=k；当前融合显著减少 global passes。twiddle、局部 spill、cache 命中和每系数 digit 重读必须另计，因此这里只给算法流量模型，不宣称为 profiler 测得的 DRAM 总量。

出处：[固定 M 的双边上传](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3645)、[window 计数/host-output 边界](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3888)、[详细运行统计](D:/code/MPA-OpenCl/build_cuda_cmake/_gfinv_ab_20261003/summary.json)。

## 25. 阶段时长占比与外推边界

采用 §21 的**已修复真实 baby 归一化**的 batch 候选 ABBA 均值，总 Stage2 **96.8483615 秒=100%**。下表作为近似阶段分摊：计时器间仍有未归属 host 工作与 drain，残差不是一个单独 kernel。各百分数直接除以 total，四舍五入后可能相差 0.01%。

| 步骤/计时项 | 秒 | Stage2 占比 | 随参数增长的主要关系 |
| --- | ---: | ---: | --- |
| init：shape、context、自测、baby/F 树 | 14.6239 | 15.10% | 点生成约 P log D·W²，F 树逐层 NTT；不随固定 D 下的 B2 增长 |
| giant 点/chain、seed、坐标与段积生成读回 | 14.5115 | 14.98% | 约 I·W² 加 seed 与 16IW 读回 |
| giant CPU 叶处理：in/invert/out 合计 | 2.2025 | 2.27% | 3E 模乘 + U 求逆 + O(IW) 叶构造 |
| G 树 | 27.9555 | 28.87% | G 棵至多 P 叶的树，累计 I-G 对乘法 |
| rev(F) finv | 2.1580 | 2.23% | O(log P) 轮长度倍增乘法，当前单曲线一次 |
| fold | 18.4775 | 19.08% | G-1 轮、通常三次乘法/轮，加 O(GPW) 主机操作 |
| Gfinv 对 H 缩放 | 0.5200 | 0.54% | 约 P 次模乘及导入/导出 |
| scaled 下降 | 8.8540 | 9.14% | 2P-1 对窗口乘法与逐层 pack，不随固定 D 下的 B2 增长 |
| block product/GCD/累积 | 0.1490 | 0.15% | 约 P 次 Montgomery 乘法与 P/q 次 GCD；命中时增加叶检查 |
| 未归属残差 | 7.3965 | 7.64% | 包括 small-prime 处理、其他准备/同步/drain 等；不能全部归为数据传输 |

init 的单次原日志另报 baby ladder **8.958 秒**、baby affine **2.060 秒**；不能把这个单次分解当成 ABBA 均值的精确拆分。`t_reduce=9.808 秒` 约占 total **10.13%**，包含在 G 树/fold/finv/下降等项中；它是内部 kernel 成本，**不能作为上表额外一行求和**。段逆元 invert **1.532 秒**、cache prepare **1.436888 秒**也为嵌套计时，包含于 giant 叶处理。

当前主账 H2D+D2H 约 16.83 GiB。若假设有效单向 PCIe 带宽 B_PCIe=8 或 12 GiB/s 且全部串行，则仅此账的数据搬运时间分别为约 **2.10/1.40 秒**；这只是带宽模型，未覆盖全部传输、双向竞争、小拷贝延迟及 CPU staging，也没有计入重叠。约 22% GPU 非 busy 样本时间不能直接解释成 22% PCIe 时间。大量小乘法/同步、串行 point seed、GMP 与主机对象准备同样会形成空档。

固定 S、D、planner 和 batch 配置，可作局部估算：

```text
T(B2') ≈ T_fixed + (B2'/B2)·T_giant_related + T_other(B2')
T_fixed 包括 baby/F、finv、下降、无命中的末尾 GCD
T_giant_related 包括 giant、G 叶、G 树、fold，以及相关 host 工作
T_other 是未归属残差，当前 B2 下约 7.397 秒，外推关系尚未校准
```

按表内可识别项目，T_giant_related≈63.147 秒，T_fixed≈26.305 秒；7.397 秒残差尚不能可靠归入两类。这个拟合仅用于相邻参数规划：跨 NTT 长度阈值、显存预算、D 选择、非单位元密度、CPU/GPU 时钟或 oracle 配置变化时必须重新测量。

优化上限也可定量约束：若只把当前归约 kernel 加快一倍且完全暴露在关键路径上，理想 total 约减至 **91.944 秒（耗时下降 5.06%）**；彻底消掉该项的理想上限为 **87.040 秒（耗时下降 10.13%）**。真实收益可能更低。进一步降低 28.87% 的 G 树与 19.08% 的 fold、减少 host/device 边界及串行准备，才有机会继续扩大整体收益；不能用旧的未修复 `--real` 数据证明当前 ECM 加速。

证据：[ABBA 均值与所有计数](D:/code/MPA-OpenCl/build_cuda_cmake/_gfinv_ab_20261003/summary.json)、[单次候选日志](D:/code/MPA-OpenCl/build_cuda_cmake/_gfinv_ab_20261003/2_segment_batch.log)、[运行计时输出合同](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11399)。本节复算已有数据；本轮连续 fold 的新测量见 §26。

## 26. 连续缓冲 host fold：本轮实现与验证

### 26.1 开关与数据生命周期

新增 `NTT_FOLD_FLAT=1`，默认 0。只有 S4 存在时启用，否则退回原 CPoly fold；`real_batched_foldflat.enabled` 表示实际路径。仍使用 `T=G·H`、`qrev=rev(T)·finv mod x^k`、`qF` 三次现有 GPU 乘法，保持 NTT shape、输出窗口、归约及 GMP 检查策略。

1. H 在全部 G 批次间以连续定宽系数保存，首个 G 根直接 move；finv 转换一次，Ft 根直接复用，跳过旧 fold 的额外 Fpoly。
2. 调用已有 `flat_mul_batch`，符合 stride 的输入借用，仅短输入补齐；qrev 原地反转作为 q，构造 qF 前释放 ra/rbi。
3. 整轮余式减法复用两个 GMP 临时数和一个 W-word 缓冲，保留 `mpz_sub + mpz_mod` 对合数 N 的原语义。
4. 仅余式 trim，T 和 finv 保留声明长度；下降前恢复 H、完整 finv 的 CPoly，保持 inverse cache 复用。

这是 CPU 布局和准备优化；G 根、三次乘法仍经过主机边界，没有减少 NTT 或 PCIe 工作量。
出处：[开关](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:385)、[flat fold](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7376)、[S4 关闭回退](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9590)、[逐批 fold](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10066)、[恢复 H/finv](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10129)。

### 26.2 可复算的主机准备及传输预算

发生除法的一轮，令 ng、nh 为 G、H 的系数数，`nt=ng+nh-1`、`k=nt-P`。旧路径 G、T、ra、rbi、qrev、q、qF 七类中间对象创建约 `ng+nt+4k+P` 个系数 vector。按本例 full 批次 H 保持 P 个系数、首轮 H 为 P+1 的尺寸模型，14 轮合计约 **12335971 次**，对应约 **6.4337 GiB** 的 W-word 载荷初始化，另有填入系数的复制。这是结构估算，未用 allocator profiler 实测；额外 trim 会改变数量。

候选用少量连续大缓冲替代逐系数对象。余式减法的 GMP init/clear 对象数从 `2P(G-1)=3225600` 降为 `2(G-1)=28`，仍做相同 **1612800 次系数模减法**。主要临时系数 capacity 峰值在首轮第二次乘法处为：

```text
8W[(P+1)+(2P+1)+3(P+1)] = 8W(6P+5)
本例：387074800 bytes = 369.143 MiB
```

该统计覆盖 H、T、ra/rbi、qrev 或 qF 等同时存活缓冲；不包括固定 Ft/finv、当前 G、flat input 补齐区、pinned staging 或其他进程对象，不能当整机 RAM 峰值。

按同一尺寸模型，fold 三次乘法的累计原始 H2D 为 **4.9484 GiB**、窗口 D2H 为 **3.2661 GiB**，公式为 `Σ_round 16W·M_round`、`Σ_round 8W·w_round`。这是后续 device fold 的边界预算，尚未被本轮减少。

### 26.3 正确性与测量方式

最终 sm89 二进制 SHA256：**A9A94DB278EC2507AC89EB77B64A1B188CE8917E084E7A58D65DC41A42732490**。

- 新增 [test_stage2_fold_flat.py](D:/code/MPA-OpenCl/tools/test/test_stage2_fold_flat.py:1)，最终 **18/0**。三个 N 的 fixture 每次 30 个 fold，用独立 GMP schoolbook 乘法和 monic 经典长除法比较完整余式；覆盖零、近 N、常数/短输入、非 2 的幂、重复 fold 和较短末批。
- 实际 Stage2 配对比较完整 G 根、叶指纹、carry trace、projective/Gamma 计数及因子/素数集合。冻结向量必须恢复 **59649589127497217 /114713** 和独立完整叶指纹 **7706779146789021619**；覆盖 N15/35 坏段、choose12、M4423/M5261、S4 关闭回退和故障注入。
- 原完整门禁在 `NTT_FOLD_FLAT=1` 下 **188/0**，使用同一最终二进制。这批历史检查未单独证明chain精确尺度；后续§28补充尺度验证，g=N饱和块仍未单独验证。

证据：[新增门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_flat_final_gate_20261003/summary.json)、[完整门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_flat_final_full_gate_20261003.log)、[独立 GMP fixture](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7676)。新增 [A/B fold_flat target](D:/code/MPA-OpenCl/tools/bench/bench_stage2_reduce_ab.ps1:54) 采用 vector/flat/flat/vector，两边均开启段积 batch、device G-root、scaled/window/chunk-output，只切换 fold 布局。§21/§25 保留段积阶段数据，本轮结果在本节单列。

### 26.4 最终二进制生产 ABBA

GPU1/RTX4060 Laptop，M4423/sigma26/extra12，B1=1000、actual B2=2011326186870、D1231230、P115200、I1633592。batch64MiB/arena6300MiB，carry batch=0、GMP sample96/every8、oracle async=0；没有 profile 或可选 fixture。四轮完整时间 **103.265444 /96.226066 /96.962568 /99.447771 秒**，均 exit0、full clean1、overflow0。

| 指标 | vector 对照均值 | flat 候选均值 | 解释 |
| --- | ---: | ---: | --- |
| 完整 Stage2 | 101.3566075 s | **96.594317 s** | 少4.7622905 s，耗时下降 **4.6985%** |
| main | 85.309476 s | 80.753009 s | 下降5.3411% |
| init | 16.0471315 s | 15.841308 s | 该段算法未改，差异含运行波动 |
| fold | 20.1565 s | **14.271 s** | 少5.8855 s，下降 **29.1990%** |
| 进程 wall | 110.733 s | 106.025 s | 包含启动和 Stage1，区别于完整 Stage2 |
| t_reduce | 9.7945 s | 9.8085 s | 未证明 GPU 归约变快 |
| observed private peak | 7972 MB | **7759 MB** | 观察均值少213 MB，沿原诊断单位；不是物理 RAM |
| NVML 整卡显存峰值 | 5036 MiB | 5036 MiB | 未减少 GPU 峰值 |
| NTT full workspace payload 峰值 | 3656064320 bytes | 相同 | 所有权/容量模型未改 |
| 主账 H2D/D2H | 11.37/5.46 GiB | 相同 | rounded，非全部传输账 |

两边 G 根 hash **105d6128bbf522db**、完整叶 hash **9100612758855566221**、Q hash **33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f** 一致；Q 匹配实际 Prime95 保存点。403次乘法调用/1979251对、40218760归约系数、66139 GMP samples/2400 selftests/4 full checks一致且bad0；carry checked/finishes8241，未省检查。

候选每次 14 folds/42 multiplies/1612800系数模减法；prepare **0.536074 s**、multiply **13.104712 s**、subtract **0.321191 s**、bridge **0.129317 s**。这些项不能全与 fold 相加：bridge 部分在 fold 外，multiply 是含准备/检查/等待的 host 墙钟，统计未覆盖所有容器析构。flat input ledger 的 calls从23→65，是新增42次 fold 进入该统计；旧 CPoly 的展平复制原本未记入该 ledger，不能拿新增 copy_bytes 当成全流程拷贝增加。

约1Hz full GPU busy **75.885→77.947%**，main **77.262→79.403%**；full低≤5%样本27/200→17/190。不是SM occupancy或精确空闲时间。最终候选阶段占比：G树 **29.50%**、fold **14.77%**、giant **15.11%**、descent **9.75%**；内部归约 **10.15%** 已包含于多项式阶段，不额外求和。

等价串行预算 `3600/T_full` 为约 **35.518→37.269 条/小时（+4.93%）**，只是在同成本假设下从单曲线时间换算；没有测试不同 sigma 的实际队列吞吐，也没有并行不同曲线。相对既有、未重跑且 D/degree/检查不同的 Prime95 单核90.460秒，候选仍慢 **6.134317秒/6.781%**，长期目标尚未证明达到。控制两轮有约3.82秒差异，收益应按配对均值和样本范围理解，不能拿跨轮历史绝对秒数直接推断额外加速。

过程记录：初版同二进制测量为105.9637415→97.351111秒（−8.128%），仅作附加证据；最终报告采用上表。更早一次漏设新 target 顺序而进入旧归约对照，已停止并标无效，没有用于性能结论。最终构建、源码、runner、分析脚本及测量 exe 均保存快照；本机 build 目录被 Git 忽略。

证据：[最终 summary](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_flat_ab_final_20261003/summary.json)、[results CSV](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_flat_ab_final_20261003/results.csv)、[GPU1采样](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_flat_ab_final_20261003/gpu1_summary.json)、[源码与exe快照](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_flat_ab_final_20261003/measured/stage2_tree_gpu.cu:7075)。

### 26.5 下一步边界

优先验证 device leaf/fold 和固定 F/finv 变换复用，分别减少主机边界/准备和重复 forward NTT；同时可用独立归约微基准研究协作 limb 运算缩短依赖链。每项继续保留独立完整系数/冻结因子门禁与同二进制对照。

并行不同 Stage2 曲线仍可提高吞吐，但 GPU1 总显存8188MiB，当前单进程整卡峰值5036MiB，不能直接同时复制两套相同配置。应先研究共用 GPU scratch、交错执行 GPU 工作并重叠另一曲线 CPU 准备，以及将当前全局可变状态迁入 curve/context owner；真实多曲线按不同 sigma/save 的完整结果和 curves/hour 验证。当前重复同一 Q/F 的 `--curves K` 不构成该吞吐证据。新开关保留默认0，小形状没有证明收益；启用方式为 `NTT_FOLD_FLAT=1`，或运行 `-Target fold_flat` 对照。

## 27. 量化补充：fold-flat 基线的逐步骤成本账

本节对应 `bc4a68f` 的 **fold-flat 候选组合**，参数、二进制与测量均采用§26.4。日期为2026-10-04，仅复算已保存日志与源码公式，没有新增GPU跑分；这里保留2026-10-03的历史规划基线，下一轮候选见§28。

本节的M_seed/M_seg数值按bc4a68f原路径计算；后续§28加入精确段积补偿和device seed后，相应Mont次数及seed传输预算需按新开关修正，不能把这里的旧操作数当作候选实数。其余多项式树/窗口/fold载荷模型继续适用。

### 27.1 通用计算量与周期记号

沿用 §23 的 `S=bits(N)`、`W=ceil(S/64)`、`P=φ(D)/2`、`I=floor(B2/D)+2`、`G=ceil(I/P)`。定义一次多项式乘法的形状 `a=(ma,mb,w)`，其中 ma/mb 是输入系数数，w 是实际返回窗口长度；其成本写成：

```text
M_a = max(ma,mb)
b_a = 2S + max(1,ceil(log2 M_a))
r_a = ceil(b_a/bpw_a)
n_a = planner 选择的 2 的幂，n_a ≥ 2M_a r_a
Q_a = max(0,d'_a-W+1)                   # 长除法实际商位尝试数

Butterfly(a) = (3/2)n_a log2(n_a)       # 未融合 radix-2 等价网络计数
Mul64_reduce(a) ≈ w W Q_a              # 不含商估计/add-back
C_poly(a) ≈ Butterfly(a)c_bf + 2n_a c_point
            + w Q_a c_qdigit(W) + C_pack/carry(a)
C_Mont(W) ≈ 2W² c_MAC + C_carry/loop(W)
```

对阶段 α 的实际乘法对集合 `A_α`，`C_α=Σ_(a∈A_α) C_poly(a)`；同形状一批 m 对就是 m 倍。由于 A 由 P、I、G 和树各节点 degree 决定，这也是包含 N、B2、D 的可计算公式；不能只按乘法对数乘某个固定“大 NTT 周期数”。`c_bf/c_point/c_MAC/c_qdigit` 是需在指定 GPU、编译结果和 shape 上校准的周期系数，当前未测；`2W²` 只计 Montgomery 主项的 64 位乘加，不能当作同样数量的机器周期。

硬件墙钟规划可写为 `T_kernel ≳ max(C_work/(f_GPU·p_eff), V_DRAM/B_DRAM)`，再考虑依赖链下界、launch 和同步。这里 `p_eff` 是对上述工作单位校准的有效并发量，不是 CUDA core 数。长除法的连续商位、chain 的连续点以及每线程 limb carry 都有串行依赖，简单除以 GPU 核心数会严重低估时间。

源码：[shape planner](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3045)、[Montgomery 模乘](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:678)、[长除法](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1939)、[归约窗口](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2096)。

### 27.2 每个步骤：计算、驻留载荷、累计传输与基线占比

下表的 `V_NTT` 指 §24.1 的**共享工作区模型**，不是给每一行额外分配一份。RAM 列是代表性逻辑载荷，容量、GMP、容器和 pinned staging 另见 §24；字节均按 W 个 64 位 limb。全部 Z 为单位元、无提前退出，采用 chain/device G-root/scaled/fold-flat 路径。

| 执行步骤 | 计算量/周期模型 | 主机 RAM；阶段显存载荷 | 累计数据生成与 PCIe 边界 | 最新完整 Stage2 时长分摊 |
| --- | --- | --- | --- | --- |
| 配置、D 规划、baby 索引 | 约 D/2 次 gcd；规划另计 | baby 索引 8P；常量 O(W) | 索引生成 8P，baby 标量 H2D 8P | 与 mandatory selftests/context 同属 init，未单独拆时 |
| baby GPU ladder | `Σ_j(13ℓ(j)-6)·C_Mont(W)`，j∈baby 集 | CPU/GPU X/Z 各 16PW | 普通 X/Z D2H 16PW | 8.9795 s /9.30%，嵌套于 init |
| baby CPU 归一化、叶生成 | `4P·M_CPU(S)+ceil(P/256)·Inv_CPU(S)`，另有模负/导入导出 | 叶 16PW，可能与 X/Z 共存；GMP O(256W) | 生成 16PW 叶；随后进入 F 树输入上传 | 2.9740 s /3.08%，嵌套于 init |
| F 完整积树与其余 init | P-1 对，`Σ_(A_F)C_poly(a)`；补齐节点复制 | F 树 `8W[(h+1)P+2K]`；device `V_NTT` | 每实乘节点 H2D `16W·max(dl+1,dr+1)`、D2H `8W(dl+dr+1)` | init 剩余约3.8878 s /4.02%，含建树、规划、自测等，未单独测 F |
| 小素数补充 | K_s定义见§23.4，GPU `Σ_p(13ℓ(p)−6)·C_Mont(W)`、CPU K_s次GCD及素性筛选 | CPU/GPU X/Z各16WK_s；设备workspace后续复用 | 标量H2D8K_s、坐标D2H16WK_s | 已含残差，旧基线未独立拆时；本轮pre混合分摊见§29.5 |
| rev(F) Newton 逆 finv | 最多 `2ceil(log2(P+1))` 对；每轮长度翻倍 | 固定 finv `8(P+1)W`，临时 O(PW)；`V_NTT` | 按每轮实际 ma/mb/w 求和，不能拿 P 乘轮数当精确流量 | 2.5690 s /2.66% |
| giant seed/chain/段积 | §27.3 的 seed、chain、段积总次数乘 `C_Mont(W)` | chunk X/Z 各侧 16c_jW，段积 `8Wceil(c_j/s)`，seed/ws 另计 | seed 往返，全部 giant X/Z D2H 16IW，段积 D2H 8WE | 14.5910 s /15.11% |
| giant CPU 段逆元、混合叶 | `3E·M_CPU+U·Inv_CPU`；Ginv 累积 E 次模乘；叶 O(IW) | CPU 叶最大16PW，段 cache O(gW) | 生成叶16IW，叶 H2D16IW | 3.0340 s /3.14%，in/invert/out 合计 |
| device G 积树 | I-G 对，`Σ_(A_G)C_poly(a)` | device frontier/raw 最大约28PW；CPU 留当前根 `8(n_b+1)W` | metadata H2D24(I-G)，根 D2H8W(I+G)；中间结果留 GPU | 28.4970 s /29.50% |
| H fold 模 F | G-1 轮，通常3对/轮；`Σ_(A_fold)C_poly(a)`，另有(G-1)P次模减法 | flat 临时峰值模型`8W(6P+5)`；固定F/finv及`V_NTT`另计 | T/qrev/qF 每轮分别上传/读回，精确尺寸模型见§26.2/§27.4 | 14.2710 s /14.77% |
| Ginv 对 H 消除缩放 | 约P次 GMP 模乘/模约减、O(PW)导入导出 | H 8PW，GMP O(W)；无新大型device对象 | CPU 重写8PW系数，随后交给下降；这里无独立GPU上传 | 0.7390 s /0.77% |
| scaled 余式树下降 | 2P-1 对，`Σ_(A_desc)C_poly(a)` | 两层state逻辑16PW；pack峰值依shape；`V_NTT` | 每真分叉孩子degree a、兄弟degree b：输入ma=a+b/mb=b+1，输出窗口w=a；按§24.3求和 | 9.4175 s /9.75% |
| GPU块积、CPU GCD/累积 | `P-ceil(P/q)` 次Mont乘；`ceil(P/q)`次块GCD；命中再查叶 | 叶8PW；CPU/GPU块输出各`8Wceil(P/q)` | 叶H2D8PW，块D2H`8Wceil(P/q)` | 0.1740 s /0.18% |
| 因子命名、合并、oracle drain等 | 命名每命中叶最多2I候选，受预算限制；检查数量按开关 | 命名/GMP/检查队列另计 | 样本、carry与检查流量不能由上述系数流量完整代替 | 命名0；其余未归属残差7.4605 s /7.72% |

前三个 init 子计时和 init 剩余合计 **15.841308 s /16.40%**，它们不能再与 init 总项重复相加；全表非重复总量为 **96.594317 s /100%**。baby 子计时由最新两次候选原日志取均值，而非引用上一轮的单次分解；“init 剩余”是减法推得的混合成本，不是 F 树独立测量。内部 `t_reduce=9.8085 s /10.15%` 和 oracle wait=6.2915 s 是嵌套/局部计时，也不额外求和。

本节基线资源高水位：主机 observed private 峰值均值 **7759 MB**（原记录单位）；NVML 整卡显存峰值 **5036 MiB**；NTT full workspace 载荷峰值 **3656064320 bytes /3486.695 MiB**。这三种口径不能相加。固定 F 树逻辑载荷 **1308016640 bytes /1247.422 MiB**，连续 fold 临时峰值 **387074800 bytes /369.143 MiB**；二者之外还存在坐标、叶、缓存、容器及 pinned staging，详见 §24。

计时来源：[最终 ABBA summary](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_flat_ab_final_20261003/summary.json)、[候选2原日志](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_flat_ab_final_20261003/2_flat_fold.log)、[候选3原日志](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_flat_ab_final_20261003/3_flat_fold.log)。步骤源码：[baby归一化](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11209)、[积树](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3937)、[Newton](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4679)、[G叶处理](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9896)、[fold](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7376)、[H缩放](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10142)、[下降](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7814)、[GCD/命名](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10432)。

### 27.3 点生成的精确静态计数：包含 seed 成本

设 chunk j 从 giant 索引 `o_j+1` 开始，有 c_j 个点，短链长度 L。令 `t_j=ceil(c_j/L)`；链 t 的实际长度 `a_jt=min(L,c_j-tL)`，从 t=0 起。GPU ladder 的 seed 标量列表实际为：

```text
Kseed_j = { (o_j+1+tL)D, min(o_j+2+tL,o_j+c_j)D : 0≤t<t_j } ∪ {D}
# 这是保留重复项的列表，最后单列 DQ；一点评尾链也会计算两个相同 seed。
Zseed = Σ_j(2t_j+1)
M_baby = Σ_(j∈baby集)[13ℓ(j)-6]
M_seed = Σ_(k∈所有Kseed_j)[13ℓ(k)-6]
M_chain = 8Σ_(j,t) max(a_jt-2,0)
M_seg = I-E
M_giant = M_seed + M_chain + M_seg
```

本例 chunks 为 **345600×4 +251192**，短链总数 **25525**，seed点 **51055**，与 `real_giant_chain` 日志一致。逐标量位长复算得到：

- baby：**26709966** 次 Montgomery 乘法；§23.2 的29260800次仍是上界，不是实数。
- seed ladder：**26178817** 次；chain递推：**12660336** 次；段积：**1531492** 次。giant 合计 **40370645** 次，seed占 **64.8462% 的 giant Montgomery 次数**，不是64.8462%的giant墙钟。
- 对 W=70，baby/giant limb乘加主项分别约 **261757666800 /395632321000** 次；GCD块积另有113400次Mont乘。自测、检查及退化回退未纳入这组算法计数。
- seed在CPU转回Montgomery image需 **2Zseed=102110次 GMP 模乘/模约减**，当前每个seed的X、Z分别处理。

seed 边界还可以精确计算：标量H2D为 `8Zseed=408440 bytes`，seed普通X/Z的D2H和转域后seed X/Z的H2D **分别**为 `16WZseed=57181600 bytes /54.533 MiB`。这与 giant全坐标D2H的 **1829623040 bytes /1.7040 GiB** 是不同事件；每chunk seed设备缓冲为 `16W(2t_j+1)`，seed ladder工作区还有独立坐标缓冲，不能只算chain上传数组。

以上只适用于本例五个chunk都走chain。小chunk走ladder时，该chunk应替换为 `Σ_i[13ℓ(iD)-6]`，并将段积成本移到CPU普通模乘；打开chain check时同时执行两条路径并增加affine比较。本节按旧基线统计；精确Gamma的后续修正和独立验证见§13/§17/§28。

源码：[ladder位循环/普通域输出](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:809)、[chain每点xADD](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1021)、[seed枚举/转域/拷贝](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9038)、[段积](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1087)。

### 27.4 积树、下降、fold 与数据生成的复算方法

对任意 p 叶树，先补到 `K_p=2^ceil(log2 p)` 个槽，degree数组初始为p个1、其余0，然后两两向上相加。令 `T_p` 是两个孩子degree均大于0的父节点集合，节点孩子degree为dl/dr：

```text
树的真乘法对数 = |T_p| = p-1
输入原始H2D（host树） = 16W Σ_(v∈T_p) max(dl_v+1,dr_v+1)
乘法产生的系数载荷 = 8W Σ_(v∈T_p) (dl_v+dr_v+1)
```

device G树仅在首次叶输入上传，后续输入借用frontier，因此不能将上式的host树H2D套到它。补齐/单边节点的复制另计。本例F树真乘法生成 **1159781840 bytes /1.0801 GiB** 的系数结果，等量结果读回；输入H2D尺寸模型 **1250958240 bytes /1.1650 GiB**。它与保留全部F树的1.2182 GiB逻辑载荷不同：后者还含叶、constant padding及pass-through节点。

对全部15个G树求和，真乘法结果生成 **16420181680 bytes /15.2925 GiB**，与日志 `resident_words=2052522710` 一致。这是累计生成量，device raw/frontier峰值仍只有 **225792000 bytes /215.332 MiB**；根读回另为 **914819920 bytes /0.8520 GiB**，不是15.2925GiB全部经过PCIe。

scaled下降每个真分叉产生两个乘法对：孩子degree a>0、兄弟degree b>0，形状为 `(a+b,b+1,a)`，窗口从b开始；另加root形状 `(P,P,P)`。因此将这些形状代入§27.1即可计算全部NTT/归约工作量，将 `16W·max(ma,mb)` 和 `8Ww` 求和即可计算host-input/host-output边界；pass-through复制另计。每层全部有效state的degree总和为P，所以累计state逻辑载荷为 `8W(h+1)P=1161216000 bytes /1.0815 GiB`，同时存活两层仅16PW，不是保留h+1层。

fold一轮实际G/H系数数为ng/nh，令 `nt=ng+nh-1`、`k=nt-P`。需要除法时三次形状为 `(ng,nh,nt)`、`(k,k,k)`、`(k,P+1,P)`；因此：

```text
H2D_fold = 16W Σ_round [max(ng,nh) + k + max(k,P+1)]
D2H_fold = 8W Σ_round [nt + k + P]
CPU模减法次数 = (G-1)P
```

本例首轮nh=P+1，其后nh=P，末轮ng=20793，在未trim额外零系数的模型下，H2D/D2H分别为 **5313302400 /3506936160 bytes（4.9484 /3.2661 GiB）**，共8.2145GiB；最新flat优化没有减少这项。finv/F已在CPU固定保存，也仍在相关乘法中重复上传，不能把“固定对象”理解成“只上传一次”。

代码现有 `MulCost` 还提供一个不依赖planner bpw的统一**operand-bit工作量代理**：每真乘法 `2M[2S+ceil(log2 M)]`。当前日志各阶段为：F **19779489992**、G **280135309920**、fold **84092415352**、下降 **71317857280**、finv **8729751060** bit，总计 **464054823604** bit；对应总代理工作量占比4.26%/60.37%/18.12%/15.37%/1.88%。它不是PCIe字节、实际NTT digit数、硬件指令数或周期数；其占比与墙钟占比不同，反映shape效率和主机开销不同。

最新flat-input统计的memcpy/memset累计为 **2666844880 /3738842800 bytes（2.4837 /3.4821 GiB）**。这是已进入该统计的prepare子集，旧CPoly fold未计同一口径，不能拿它与旧ledger直接做全流程拷贝差。当前主账H2D/D2H仍约 **11.37 /5.46 GiB**，仅覆盖该账定义的调用范围；上述F/seed/叶/根/窗口等存在包含关系，不能逐行相加冒充全程序PCIe总量。

源码：[树补齐和节点degree](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3937)、[G树resident结果统计](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3760)、[G根读回](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4512)、[scaled形状和窗口](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7868)、[flat fold](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7376)、[MulCost代理公式](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1181)、[固定M原始上传](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3645)。

### 27.5 参数变化与优化收益预算

固定N、D、planner及batch参数，令 `λ=B2'/B2`。由最新非重复分摊构造局部模型：

```text
T_fixed ≈ 28.740808 s     # init、finv、H缩放、下降、无命中块GCD
T_giant_related ≈ 60.393 s # giant、G叶、G树、fold
T_stage2(B2') ≈ T_fixed + λ T_giant_related + T_residual(B2')
T_residual(B2) ≈ 7.460509 s
```

例如B2加倍，若残差固定，模型为 **156.9873 s**；若残差也同比增长，则为 **164.4478 s**。这是两种规划假设，不是实测或严格上下界；实际G/chunk数量取整、尾批shape、clock和退化密度会改变结果。固定D下长期树/finv内存通常基本不随B2增加，giant chunk受预算约束；累计生成/传输和G树数量则随I≈B2/D增长。增大S会令W及原始系数载荷近线性增长，点算术/归约主项近W²增长，而NTT长度还随slot/planner阈值跳变。

**访存改进预算。** 将单位元giant的设备坐标直接生成device叶，理论上可以避免 `16IW` 坐标D2H和 `16IW` 叶H2D，本例合计 **3.4079 GiB**，同时减少CPU叶生成及对象准备。若有效带宽取8GiB/s、全部暴露在关键路径，仅此大块数据项约 **0.4260 s**；小拷贝/同步、pinned staging和CPU准备的收益需另测。device fold边界共8.2145GiB，在相同假设下约 **1.0268 s**；不能直接把14.271秒fold全部算作可省PCIe时间，也不能无条件将两个预算相加。

**归约改进预算。** 用当前10.15%的内部归约计时作Amdahl模型，若其速度翻倍且节约全部落在关键路径，完整时间约 **91.6901 s**，耗时下降5.08%；彻底消除此项的理想模型约 **86.7858 s**，下降10.15%。同样，64.85%的seed算术份额说明需评估seed生成效率，但改变L还会改变串行链长度、并发及退化传播，不能据次数下降预测同幅墙钟改善。

**多曲线容量预算。** 共享工作区设计应使用 `V_parallel(k)=V_shared+k·V_curve+V_queue(k)`，而不是k倍NTT池，也不能直接以8188/5036推导实际可并行曲线数。当前仅知道一进程整卡峰值5036MiB，尚未把共享与每曲线驻留高水位拆开；真实多曲线吞吐仍需实际sigma/save队列测量。由本例换算的 **37.269曲线/小时** 只是单曲线串行预算。

## 28. 当前候选修正：精确段积与 device seed（2026-10-04）

本节替代§27中 seed/段积的旧成本数值，其余树、下降和 fold 的形状与载荷公式继续适用。测试均采用隔离构建和 GPU1 RTX 4060 Laptop（8188 MiB）；GPU0 的用户 Stage1 作业未改动。没有使用 Nsight 测得 primitive 周期，§23/§27的周期系数仍待校准。

### 28.1 段积尺度与新增成本

对一段 m 个实际返回的 Z 字，令 `Γ=∏Z_i mod N`、`R=2^(64W) mod N`。原段积只有 m−1 次 Montgomery 乘法，所得为 `ΓR^{−(m−1)}`；它可用于 GCD，但不能直接作为普通射影叶的精确 Γ。当前默认在 m>1 时追加 `Mont(p,R^m)=Γ`，单点段直接返回 Z。表按 modulus/workspace 缓存，末段按实际 m 取表项。

```text
E_>1 = 实际长度大于1的段数
M_seg_exact = I-E+E_>1
V_segfix = 8W(s+1) bytes
```

本例 E=E_>1=102100，段积新增 **102100 次 Mont**，合计 **1633592 次**；s=16/W=70 的表仅 **9520 bytes**，首次生成、上传后复用。普通 ladder 的 host 段积原本就是普通乘积。`NTT_GFINV_SEG_EXACT=1` 默认开启；0仅作旧尺度诊断。旧差异为可逆 R 幂，**不因此判定旧 ECM 因子结果无效**，但其完整系数指纹不同；这也不是之前 Stage1/baby 归一化错误。

源码：[开关](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:390)、[段积补偿](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1087)、[表所有权](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8831)、[实际段检查](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9142)。

### 28.2 device seed：实际省掉什么

`NTT_GIANT_SEED_DEVICE=1` 让 seed ladder 直接输出 image 坐标，chain 借用同一 workspace 的交错 seed 对，stride=2；末尾 DQ也从同一缓冲取。省掉普通域输出、seed坐标D2H、CPU乘R转域、六个独立seed数组的分配/上传。标量枚举和 H2D 仍在 CPU；完整 giant X/Z 仍读回，叶仍在 CPU 构造，尚未实现 device leaf。

对§27.3的 `Zseed=Σ_j(2ceil(c_j/L)+1)`：

```text
避免的 seed D2H = 避免的 seed H2D = 16W·Zseed bytes
避免的 GPU Mont输出转换 = 避免的 CPU 模乘 = 2Zseed
避免的独立设备分配/释放对数 = 6J_chain
M_giant_host_exact = M_giant_legacy + E_>1
M_giant_device_exact = M_giant_legacy + E_>1 − 2Zseed
```

本例 Zseed=51055、J_chain=5：两个方向**各57181600 bytes /54.533 MiB**，合计 **109.065 MiB**；CPU/GPU分别省 **102110 次模乘/Mont**，省 **30次设备分配/释放对**。host seed精确模式giant总Mont数 **40472745**，device seed精确模式 **40370635**；后者只比旧尺度基线40370645少10次，因为额外段积补偿抵消了大部分GPU转换次数。不能据此预测点生成全面提速。原有 `16IW` 坐标读回和 `16IW` 叶上传合计3.4079GiB，仍未消除。

seed workspace 容量复用并不会自动降低全程序峰值：其生命周期可能不与主导峰值重叠，借用的 dx/dz 也仍保留容量。旧六数组的逻辑最大载荷 `16W(2ceil(C/L)+1)=12097120 bytes`，约11.537MiB；不是几十或几百MiB的峰值显存收益。设备输出 ox/oz 和段积 dsp 仍逐 chunk 分配释放。

源码：[image输出](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:843)、[交错读取](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1018)、[seed生成/校验](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9054)、[借用与释放](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9107)、[统计](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11470)。

### 28.3 正确性与可复现范围

隔离 exe 的 SHA256为 **D185D317F9797700529655FA51F2738E5DBBC9F259C332BF56DF59752614A557**。新增 [test_stage2_device_seed.py](D:/code/MPA-OpenCl/tools/test/test_stage2_device_seed.py:1) 在同一二进制 **31通过/0失败**，原完整门禁 **188通过/0失败**。

- 64/129/4423/5261位段积 fixture：每宽91个case、1022个完整段比较，独立 GMP 普通乘积验证；覆盖空输入、单点、不同段长、零/近N和尾段。旧尺度在各宽368或369段上与普通积不同，精确模式全部一致。
- 实际 chain 全段与实际返回gz的普通乘积比较；device seed 的每个坐标字与原普通输出再乘R的结果比较。真实配对检查根/叶/carry/退化计数和因子。
- 独立 Python 仿射 monic 乘积覆盖单位元64/127位、冻结向量、M4423/extra12、M5261；比较完整叶的定宽指纹。冻结因子59649589127497217/114713及指纹 **7706779146789021619** 恢复一致。不是导出所有最终系数后逐字比较的生产门禁。
- N15/35坏段、forced ladder回退、单点tail通过；毒化段积被拒绝，旧尺度开启实际段检查也被拒绝。g=N饱和块的独立行为仍未验证。

原 PowerShell 5 门禁最早因预期stderr遇到Stop而提前退出；重试186/2的两个失败来自Out-String默认折行。保留既有断言，修正预期stderr捕获及4096列宽后得到188/0，没有删除检查。源码：[捕获helper](D:/code/MPA-OpenCl/tools/test/test_stage2_tree_gpu.ps1:78)、[check输出宽度](D:/code/MPA-OpenCl/tools/build/check_stage2_tree_gpu.ps1:79)。

证据：[31/0](D:/code/MPA-OpenCl/build_cuda_cmake/_seed_device_final_gate_20261004/summary.json)、[188/0完整日志](D:/code/MPA-OpenCl/build_cuda_cmake/_seed_device_full_gate_final_20261004.log)、[构建日志](D:/code/MPA-OpenCl/build_cuda_cmake/_seed_device_build_20261004.log)。

### 28.4 同二进制 ABBA：时间与阶段占比

[runner](D:/code/MPA-OpenCl/tools/bench/bench_stage2_reduce_ab.ps1:21) 新增 `-Target seed_device`，明确顺序host/device/device/host并检查实际开关、工作量和指纹。两边fold-flat/Gfinv batch/device G-root/scaled/window/chunk均开启，精确段积=1；carry batch=0，oracle async=0，sample96/every8，无额外fixture。参数：M4423/sigma26/B1=1000/extra12，actualB2=2011326186870，D1231230/P115200/I1633592/G15，batch64MiB/arena6300MiB，chain L64/min32768。

四轮完整Stage2为 **97.716019 /96.343413 /95.955048 /96.611685 s**；均exit0、full clean1、overflow0。均值host→device如下：

| 口径 | host seed | device seed | 解释 |
| --- | --- | --- | --- |
| 完整Stage2 | 97.163852 s | **96.1492305 s** | 少1.0146215 s /1.0442% |
| main Stage2 | 81.042852 s | 80.396682 s | 少0.646170 s /0.7973% |
| init | 16.1209995 s | 15.7525485 s | 未改init，差异提示运行波动 |
| giant | 14.6195 s | **14.462 s** | 少0.1575 s /1.0773%，最直接相关阶段 |
| G树 /fold /下降 | 28.490 /14.5495 /9.6645 s | 28.493 /14.485 /9.4365 s | 算法和工作量未改，差异不可全归于seed |
| 内部t_reduce | 9.8285 s | 9.8415 s | 没有归约加速证据 |

控制首末相差1.104334s，且没有置信区间；**目前证明字节、转换和分配次数减少，计时只支持小幅收益趋势**，不把完整1.01秒都解释为seed关键路径收益。不能与§26旧尺度二进制的绝对秒数直接做性能对照。

当前候选完整时间的非重复分摊：

| 阶段 | 均值秒 | 完整Stage2占比 |
| --- | --- | --- |
| init | 15.7525485 | 16.38% |
| giant seed/chain/段积 | 14.462 | 15.04% |
| G叶 in/inverse/out | 2.916 | 3.03% |
| G树 | 28.493 | 29.63% |
| rev(F)逆finv | 2.421 | 2.52% |
| fold | 14.485 | 15.07% |
| H消除Γ | 0.6705 | 0.70% |
| 余式树下降 | 9.4365 | 9.81% |
| 块积/GCD/累积 | 0.1755 | 0.18% |
| 未归属残差 | 7.337182 | 7.63% |

总计96.1492305s；内部归约 **9.8415s /10.24%** 已包含在多项式阶段，不再求和。残差含主机准备、检查/等待、账目未单列等，不能直接称为PCIe时间。oracle host/wait约6.890/6.277s也不是可独立相加的阶段。主导方向仍为G树、giant、fold及归约依赖链。

四轮完整Q hash **33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f** 匹配实际Prime95保存点，G根hash **105d6128bbf522db** 不变。精确unscale叶hash为 **10619321735931855904**，两模式一致；与旧尺度9100612758855566221不同属预期。403calls/1979251pairs/40218760reduced、66139GMP samples/2400selftests/4full checks和8241carry checked/finishes一致且bad0。

### 28.5 RAM、VRAM和GPU空闲采样

| 指标 | host seed | device seed | 口径 |
| --- | --- | --- | --- |
| observed private峰值均值 | 7762 MB | 7758 MB | 仅少4MB，不支持显著RAM下降 |
| NVML整卡显存峰值 | 5048 MiB | 5032 MiB | 1Hz采样，16MiB差异不全归于seed |
| NTT workspace owned/full载荷峰值 | 3356.537 /3486.695 MiB | 相同 | 容量账，不含所有CUDA对象 |
| workspace malloc/grow/hit | 24 /8 /8890 | 相同 | 不覆盖seed独立cudaMalloc，故看不到省掉的30次 |
| full GPU busy均值 | 77.45% | 80.80% | 约1Hz NVML；不是SM occupancy |
| full低busy≤5%样本 | 18/193 | 16/191 | 不能直接当精确空闲时长比例 |
| main GPU busy均值 | 79.26% | 82.19% | 近似按日志阶段边界对齐 |

传输主账仍为rounded H2D11.37/D2H5.46GiB，其口径不覆盖所有seed边界；新增seed独立账记录实际避免的109.065MiB。多项式window D2H仍5.683GiB，raw pinned约123.05MiB、window pinned约246.10MiB；flat prepare累计copy/zero仍2.4837/3.4821GiB。省下的seed流量没有改变这些大项。

等成本串行预算 `3600/T_full≈37.442曲线/小时`；不是不同sigma并行实测。相对旧且未重跑、D/degree/检查不同的Prime95单核90.460s仍慢5.6892s，尚未证明达到长期目标。

证据：[summary](D:/code/MPA-OpenCl/build_cuda_cmake/_seed_device_ab_20261004/summary.json)、[CSV](D:/code/MPA-OpenCl/build_cuda_cmake/_seed_device_ab_20261004/results.csv)、[GPU1采样汇总](D:/code/MPA-OpenCl/build_cuda_cmake/_seed_device_ab_20261004/gpu1_summary.json)、[原采样](D:/code/MPA-OpenCl/build_cuda_cmake/_seed_device_ab_20261004/gpu1_sensors.csv)、[源码hash复核](D:/code/MPA-OpenCl/build_cuda_cmake/_seed_device_ab_20261004/source_verified.json)。源码/exe/runner/分析脚本快照在同目录measured；这些本机证据被Git忽略，tracked runner/gate提供复现入口。

### 28.6 下一阶段可量化预算（尚未实现）

1. **Mersenne专用系数归约。** 对 `N=2^S−1`，反复 `C←(C mod 2^S)+floor(C/2^S)`，最终将N映射为0，可把每系数约W²个乘法减法换为O(W)加法/移位。本例C的位长约2S+ceil(log2 M)，只需少数fold；通用N仍走长除法。必须覆盖partial limb、carry、小S/边界并测GMP一致性。按当前归约账，速度翻倍的理想完整时间91.2285s，彻底消除此项86.3077s；这不是候选实测预测。
2. **GPU组Γ与device leaf。** 全好段CPU现有约 `(3E+E)·M_CPU+U·Inv_CPU`；只需组Γ时可降到约 `(E+U)·M_CPU+U·Inv_CPU`，若组积也在GPU，则CPU主项约U次模乘和U次求逆。64段一组本例U1598，组Γ单方向仅8WU=894880bytes，而E个段积为57176000bytes。坏组保留逐段/逐点回退。后续直接device叶可省3.4079GiB边界，持有giant坐标可能增加约369.14MiB同时驻留预算，需测实际高水位。
3. **固定谱复用与并行队列。** 顶层NTT n=2^27时，每个8n字节谱为1GiB；F/finv各缓存一份约2GiB，与坐标驻留叠加会明显挤压8188MiB设备容量。先评估一份缓存、显式所有权/释放和共享scratch，再测不同曲线curves/hour。多进程各复制约5GiB工作集不适合这张卡。

以上是本阶段由shape/计时/生命周期导出的规划量；其中Mersenne归约的后续实现见§29，组Γ/device leaf/固定谱仍待实现。device seed默认仍0，可用 `NTT_GIANT_SEED_DEVICE=1` 或 `-Target seed_device` 复现。

## 29. Mersenne 专用设备系数归约（2026-10-04）

### 29.1 算法、适用性与所有权

`NTT_S4_MERSENNE=1` 仅在精确检查 `N>1` 且 `popcount(N+1)=1` 后启用。位数保存在当前modulus的 `S4Reduce::mersenne_bits`，经每个launch显式传入；没有新增可变全局modulus常量。对普通未归约卷积系数C反复做：

```text
lo = C mod 2^S
hi = floor(C/2^S)
C ← lo + hi             # 2^S ≡ 1 (mod N)
直到 C < 2^S；若 C=N，则输出0
```

这是普通余数，不引入Montgomery缩放。对q=floor(S/64)、b=S mod64，high limb由 `t[i+q]>>b | t[i+q+1]<<(64−b)` 拼成，b=0单独处理以避免移位64；先读后写保证q=0的小S也能原地运行。进位可能增加一字，实际live长度始终更新；没有将常见“2或3次fold”写死为所有输入的上限。

采用独立 `s4_reduce_kernel<NW,true>` 实例。槽装配、全部source槽的位界检查和输出窗口合同保留；旧Montgomery/forensic调试路径仍用原实例，S4关闭时继续host归约。默认开关0；旧对照target明确清0，避免两边误用fast路径。

源码：[普通Mersenne余数](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2000)、[专用kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2095)、[launch选择](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2250)、[精确modulus检测](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2421)。

### 29.2 可量化计算与资源模型

设一次归约有mw个输出系数，第v个系数第f次fold的输入live limb数为d_vf，则每fold处理 `L_vf=max(W,d_vf−floor(S/64))` 个limb。新成本为：

```text
C_reduce_mersenne_work ≈ Σ_v Σ_f L_vf·c_add_shift_carry + C_assemble/store
商位乘法数 = 0
额外cudaMalloc / pinned载荷 / PCIe传输 = 0
```

对当前大S、`C<2^(2S+ceil(log2 M))` 且S远大于log2M的shape，最多少数fold，主项O(mwW)；小S或超宽fixture按实际fold次数计算。通用长除法主项仍约mwW²次64位乘法减法。M4423本例W70、40218760个归约系数：旧W²代理约 **197071924000** 次乘法，新若按最多3个约W+1长度fold规划，约 **8566595880** 个limb迭代；每迭代还含读取、移位、carry等，二者不是同单位的机器周期，不能直接作23倍速度预测。装配和NTT完全未减少。

同一最终exe由 `cuobjdump --dump-resource-usage` 读取的NW128实例：generic/Mersenne均 **40寄存器/线程**，STACK **6192→3104 bytes/线程**，SHARED0。对应NW64为3120→1568bytes；专用实例让不可达的除法/REDC临时区从编译结果消失。STACK是静态栈帧口径，不能当作独立cudaMalloc载荷与整卡显存相加，也没有因此证明运行时occupancy或spill流量变化。

证据：[最终二进制资源表](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_resources_20261004.txt)。周期系数仍待校准；Nsight Compute 2026.2.1尝试在GPU1取样时返回ERR_NVGPUCTRPERM，未取得性能计数器或有效报告，未修改驱动权限/时钟设置。该工具运行不能作为周期、occupancy或性能对照证据。

### 29.3 独立正确性与回退门禁

隔离[exe](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_20261004/stage2_tree_gpu.exe) SHA256 **AB915DD0F3E43DD78F5A673310BB9237D49250ADED0E836B35B6557381FC7554**；编译273.9s、链接3.2s，exit0。

新增[test_stage2_mersenne.py](D:/code/MPA-OpenCl/tools/test/test_stage2_mersenne.py:1) 最终 **36/0**，原完整套件显式Mersenne/device seed/fold-flat/exact均1 **188/0**，使用同一exe。

- 独立primitive fixture共720cases/20688个GPU输出字：S=2、3、31、63、64、65、127、128、129、255、256、4423、5261、8191、8192；0、N±1、N²±1、2^(2S+20)附近、全一字及超宽随机输入。CPU同一helper和GPU各自与GMP普通余数比对，bad0；host累计12136次fold仅反映这些超宽fixture，不代表生产fold数。
- 实际快速S4的139个窗口/stride/alias样例与独立GMP卷积一致，窗口外故意违例和空窗口都保持原检测合同；flat输入完整比较也通过。
- 实际63/64/65/127/128/521/4423/5261/8191/8192位Mersenne与division配对，完整根/叶/carry指纹、因子/退化计数一致。127/521/4423/5261位及两种非Mersenne形状另与独立Python仿射monic叶指纹一致；生产全系数仍按原GMP抽样和完整指纹覆盖。
- 非Mersenne近似值和冻结N=2^128+1确实选择division；OLDTAIL=1确实选择Montgomery；S4关闭并关闭依赖它的scaled下降后，两模式host回退叶一致。故意毒化GPUprimitive输出被拒绝。

补回退门禁时两次脚本重跑分别因非法的S4-off+scaled组合、错误的leaf日志前缀失败；修正测试组合和前缀后36/0，CUDA源码和二进制没有改变。不是算法不一致，也未删除回退断言。

证据：[36/0](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_final_gate3_20261004/summary.json)、[188/0](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_full_gate_20261004.log)、[primitive与窗口检查](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_final_gate3_20261004/primitive_all_widths.log)、[NCU权限失败日志](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_profile_division_20261004.log)。小形状同二进制ABBA预检也通过；该毫秒级结果不用于生产加速结论。

### 29.4 生产同二进制 ABBA 与阶段占比

新增 `-Target mersenne`，顺序division/mersenne/mersenne/division，只改变NTT_S4_MERSENNE；明确检查实际eligible/enabled和algorithm，禁止非Mersenne输入冒充快速对照。两边device seed/exact/fold-flat/Gfinv batch/device G-root/scaled/window/chunk均1，OLDTAIL/carry batch/oracle async均0，sample96/every8。参数与§28相同：GPU1 M4423/sigma26/B1=1000/extra12、actualB2=2011326186870、D1231230/P115200/I1633592/G15、batch64MiB/arena6300MiB。

四轮完整Stage2为 **95.780642 /87.739980 /87.638803 /95.585735 s**，全部exit0/full clean1/overflow0。控制首末差0.194907s、候选两次差0.101177s；没有置信区间，结论采用本轮同二进制均值。

| 项目 | division | Mersenne | 变化 |
| --- | --- | --- | --- |
| 完整Stage2 | 95.6831885 s | **87.6893915 s** | **−7.993797 s /−8.3544%** |
| main | 79.902341 s | 72.384038 s | −7.518303 s |
| init | 15.7808475 s | 15.3053535 s | −0.475494 s，F树归约也采用新内核 |
| 内部t_reduce | 9.847 s | **1.9855 s** | **约4.959倍 /−79.8365%** |
| G树 | 28.2995 s | 22.513 s | −5.7865 s |
| fold | 14.3225 s | 13.589 s | −0.7335 s |
| 下降 | 9.3185 s | 8.3235 s | −0.9950 s |
| giant | 14.457 s | 14.4585 s | 点算术未改，没有giant加速证据 |
| 进程wall | 105.0755 s | 97.0415 s | 含启动和Stage1，区别于完整Stage2 |

本轮候选非重复阶段占比：init17.45%、giant16.49%、G叶3.30%、G树25.67%、finv2.78%、fold15.50%、H消Γ0.80%、下降9.49%、块GCD0.20%、残差8.32%。对应秒数15.3053535/14.4585/2.894/22.513/2.4355/13.589/0.701/8.3235/0.172/7.297538；总计87.6893915s。内部归约只剩 **2.26%**，嵌套于多项式阶段，不重复求和。原样本中的残差还包含未单列的小素数分支等工作，不能全解释为CPU准备或传输。

四轮Q/G根/叶hash分别为 **33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f /105d6128bbf522db /10619321735931855904**，完全一致。403calls/1979251pairs/40218760reduced、66139GMP samples/2400selftests/4full checks、8241carry checked/finishes一致且bad0，未减少精度、位界检查或oracle。seed避免字节/转换、段积补偿次数亦不变。

主机observed private峰值均值 **7766.5→7767 MB**，没有RAM下降证据；NVML两边整卡峰 **5032MiB**，NTT workspace owned/full **3356.537/3486.695MiB**、24malloc/8grow/8890hit均不变。主账H2D/D2H11.37/5.46GiB、window D2H5.683GiB不变。约1Hz full busy79.51→80.34%、main80.95→82.27%，full低≤5%样本21/189→18/174；不是SM occupancy或精确idle时长。

串行预算 `3600/T_full` **37.624→41.054曲线/小时（约+9.12%）**，没有测试不同sigma并行队列。87.689s比旧Prime95单核90.460s记录低2.7706s/3.06%，但CPU未重跑，D/degree/检查范围不同，**不能据此证明公平超越Prime95或完成长期目标**。

证据：[summary](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_ab_20261004/summary.json)、[CSV](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_ab_20261004/results.csv)、[GPU采样](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_ab_20261004/gpu1_summary.json)、[源码/exe复核](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_ab_20261004/source_verified.json)。这些本机证据被Git忽略；tracked runner/gate可复现。

### 29.5 Nsight Systems：传输带宽与主机边界的区别

另用Nsight Systems 2026.1.3在同一exe/生产参数、GPU1上采集CUDA时间线，关闭需要管理员权限的CPU采样/context-switch，不改驱动设置。有效 `.nsys-rep`/SQLite生成并exit0；默认NVTX报告无数据，GPU kernel文本统计受工具UTF-8错误影响，以下直接查询原始SQLite numeric事件与kernel名称。没有用这轮profiler时间替代ABBA。

原始表记录 **114528个kernel**，均deviceId1；8923次S4归约都是`<128,(bool)1>`，累计kernel时长1.9528s。全部kernel累计73.7887s含Stage1的单线程chain8.6637s。Stage1 chain之后首个GPU kernel至最后GPU事件的近似Stage2窗口为 **87.6897s**，kernel/copy/memset区间去重并集 **68.4649s /78.08%**；该窗口有 **19.2248s未记录到本进程GPU工作**。它省略首个kernel前的CPU-only准备，不是精确Stage2计时、整卡空闲或SM occupancy。

全采集scope的H2D/D2H分别 **13570897568 /9366472536 bytes**，GPU copy-engine事件累计 **1.0666 /0.7859s**；D2D另23173360bytes/0.000290s，GPU memset约427.587GB/1.4882s。全scope包括Stage1/mandatory setup，不能与主账11.37/5.46GiB逐项相减当漏账；传输和memset的事件也可能与kernel重叠。

最有用的区别：**11292个同步cudaMemcpy的host API区间共33.9657s**；按correlationId关联的GPU copy事件仅 **0.2905s**，载荷3340194472bytes。其余API区间含等待先前GPU工作、driver调度、staging等，不能称为33.97s纯PCIe搬运，也不能将差值全部称为GPU空闲。cudaDeviceSynchronize的36.0066s同样含Stage1/点kernel等待，不是纯同步开销。CUDA时间线支持优先减少host边界和准备，而不支持把全部问题归于大块传输带宽。

NTT相关tile/outer_fwd/outer_inv累计约15.892/9.554/4.816s；ladder kernel23.878s还覆盖baby、small-prime和seed，chain kernel3.027s，阶段归属需按调用/范围另拆。下一轮可先加明确阶段范围或从correlation记录定位CPU空隙，不把这些全scope累计数直接放入§29.4的非重复阶段占比。

**小素数成本与复用机会。** 当前[small-prime分支](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9606)生成B1<p≤min(B2,D/2)全部素数点，本例K_s=50108，静态计数11544588次Mont、GPU→CPU坐标56120960bytes、scalar上传400864bytes。生产候选 `pre−finv` 两轮均值约 **6.578s /7.50%**；这是小素数加ws/前置杂项的混合计时，从7.297538s残差中细分，不额外相加。先前的baby+giant静态计数没有包含这组点，不能把它当整条Stage2点运算总数。

对于p≤D/2且gcd(p,D)=1，baby集合已含j=p，按同一Q/ladder生成的Z可复用；成功的baby批量逆元已证明该组全部Z为unit，失败组的逐点GCD也已计算。当前程序却重新ladder并GCD。下一轮可缓存覆盖范围与必要的非单位元/prime hint，保留p|D的缺失点和参数/输入不匹配回退；必须独立验证因子、hit-prime及饱和边界，不直接删除分支。B1≥D/2时K_s=0，此优化没有收益。

证据：[Systems报告](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_nsys_20261004.nsys-rep)、[SQLite](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_nsys_20261004.sqlite)、[查询汇总/口径](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_nsys_summary_20261004.json)、[采集与工具诊断](D:/code/MPA-OpenCl/build_cuda_cmake/_mersenne_nsys_20261004.log)。

### 29.6 下轮方向与容量边界

先验证小素数分支复用baby结果，避免本例约1154万次重复Mont；收益受B1/D关系限制。随后实现GPU组Γ并衔接device leaf：全好组的CPU工作从3E次批量模乘和E次Ginv累计缩到U组逆元/累计，坏组保留逐段回退；组Γ单方向仅894880bytes。device leaf可减少3.4079GiB边界与CPU叶生成，持有坐标可能多约369MiB同时驻留，必须用显式owner/预算和实际峰值验证。当前GPU事件空隙约22%提示还有CPU准备/队列重叠机会，不能直接复制两份5GiB进程工作集。

固定F频域缓存和点Montgomery专用Mersenne归约也值得评估：前者顶层每谱1GiB，先测单谱缓存；后者必须保持 `Mont(a,b)=abR^−1 modN` 的域合同，普通S4 fold不能直接替换点Mont乘法。归约t_reduce已降到2.26%，继续只优化它的理想上限有限，应更新D成本模型并向点生成、重复NTT及host边界推进。生产save/队列接入、不同sigma吞吐和公平CPU对照仍未完成。


## 30. 小素数分支复用 baby 证明（2026-10-04）

### 30.1 算法、缓存合同和边界

上一轮小素数分支在baby已经生成同一个j=p的情况下，仍重新ladder并GCD。本轮 `NTT_SMALL_PRIME_REUSE=1` 复用baby归一化结果：成功批量求逆证明整批Z为unit；失败批保留非单位元GCD。普通单位元不用逐点存GCD或坐标。缓存由当前 `run_real()` 持有，存D/B1/B2、真实有序baby索引以及LadderCtx的N/Q/a24/Montgomery one、宽度和ninv；逐项精确匹配后才复用，没有跨曲线可变全局缓存。

特别处理Z=0：原 `affine_x_gmp_checked()` 返回叶值0和成功，不能因此将GCD证明记为1；本轮显式保存GCD=N，原叶值和因子逻辑不变。仍按原小素数升序调用 `s3_record`，保留因子去重、每个prime hint和hits；GCD=N仍不报作因子。p|D等实际baby集合缺失点保留ladder/GCD。缓存不匹配、未完成或缺失时全量回退；`--check-F`目前没有该缓存，走同一回退。B1≥min(B2,D/2)时没有小素数，也没有这项收益。

`NTT_SMALL_PRIME_REUSE`默认0；新ABBA target为 `small_prime`，两边Mersenne归约/device seed/exact/fold-flat/Gfinv batch/device G-root/scaled/window/chunk均1，仅切换小素数复用。`NTT_SMALL_PRIME_CHECK=1` 重算全组小素数并逐项比较实际GPU Z的GMP GCD，故意毒化会拒绝；检查、stale/poison标志会令full timing clean=0，生产测量均关闭。开启检查时 `avoided_*` 表示复用路径的逻辑节省，不能当作包含检查工作的净流量节省。

源码：[缓存/精确输入键](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9531)、[小素数复用及回退](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9606)、[Z=0及非单位元证明](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11238)、[缓存完成及传递](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11338)、[同二进制runner](D:/code/MPA-OpenCl/tools/bench/bench_stage2_reduce_ab.ps1:55)。

### 30.2 计算、数据和容量公式

令 `S=bitlen(N)`、`W=ceil(S/64)`、集合T为B1<p≤min(B2,floor(D/2))的素数，T_c为输入匹配且真实baby集合覆盖的子集，K_s=|T|、K_c=|T_c|、K_m=K_s−K_c。关闭检查、无其他退化提前退出时：

```text
原小素数 GPU Montgomery 次数 = Σ_{p∈T} (13ℓ(p)−6)
新小素数 GPU Montgomery 次数 = Σ_{p∈T\T_c} (13ℓ(p)−6)
避免 CPU GCD 次数 = K_c；新增缓存查找/MPZ set及原素数枚举仍执行
避免依赖 MAC 主项代理 ≈ 2W² Σ_{p∈T_c}(13ℓ(p)−6)
GPU依赖链周期代理节省 ≈ Σ_{p∈T_c}(13ℓ(p)−6)·C_Mont,GPU(W)
CPU周期代理节省 ≈ K_c·C_GCD,CPU(S) − C_cache_lookup/record,CPU
两种硬件时钟不同，不能直接相加当墙钟；系数尚未测得
避免小素数 X/Z 生成载荷 = 16WK_c bytes
避免 scalar H2D = 8K_c；避免 X/Z D2H = 16WK_c bytes
```

本例T_c=T、K_s=K_c=50108、K_m=0：避免 **11544588次Mont +50108次CPU GCD**，MAC代理 **113136962400**；避免普通坐标载荷和D2H **56120960bytes /53.521MiB**，H2D **400864bytes /0.382MiB**。没有减少F/G/NTT的系数数或精度。

缓存的capacity载荷约 `8P+40W + capacity(nonunit)·sizeof(pair<size_t,string>) +Σ string capacity`；通常全unit时，本例 **924400bytes /0.882MiB**，不含容器头/allocator。最坏退化记录O(P)，其GCD字符串长度随S增长，不能把全unit容量作为所有曲线的硬上限。

设各giant chunk的seed数为s_j，`s_max=max_j s_j`，S3点缓冲持久容量（scalar+X/Z）从 `(8+16W)max(K_s,s_max)` 变为 `(8+16W)max(K_m,s_max)`。本例s_max=10801，50108→10801点，逻辑持续device分配减少 **44338296bytes /42.284MiB**。未保存小素数坐标，CPU临时X/Z也避免53.521MiB；全程RAM峰值不一定同比下降，因为树/GMP/pinned等对象峰值时刻不同。未新增GPU kernel。

### 30.3 门禁与生产 ABBA

隔离[exe](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_20261004/stage2_tree_gpu.exe) SHA256 **98EB15DA858086183F849B18C4E537B89090BC3EF6E14E33643EA2B87DD5031F**；最终编译272.1s/link2.7s exit0。新[test_stage2_small_prime.py](D:/code/MPA-OpenCl/tools/test/test_stage2_small_prime.py:1) **31/0**，原完整套件显式reuse/Mersenne/device seed/exact/fold-flat均1 **188/0**，同一exe；小形状runner ABBA预检也通过。

门禁逐项检查复用/缺失点计数、节省公式、全量GPU→GMP GCD证明、根/最终叶/carry/projective/Gamma及因子/prime hint。覆盖不同sigma、extra12、M4423/M5261、B2截断、无小素数、输入键不匹配、S4-off、两次同Q调用、poison；独立Python ladder核对小素数命中因子及GCD=N。N=10403/B1=2在p=5的因子103分别走缓存与p|D回退；N=103覆盖两种饱和路径；冻结N=2^128+1/D300300缓存114713对应59649589127497217。

首次专用门禁实际抓到Z=0证明遗漏，修正CUDA并重编译后饱和检查通过。第二次门禁因测试假设根/carry统计在 `--curves` 内重置而失败：既有统计为累积值；改为跨配对运行按对应轮次比较，最终31/0，第二次失败后CUDA/exe未变。不能据此声称 `--curves` 是不同sigma流水。

生产GPU1、M4423/sigma26/B1=1000/extra12、actualB2=2011326186870、D1231230/P115200/I1633592/G15，batch64MiB/arena6300MiB、sample96/every8、carry batch/oracle async均0，与§29的形状相同。顺序small_ladder/baby_reuse/baby_reuse/small_ladder，四轮full **87.860651 /81.201935 /80.923097 /87.920495s**，全部exit0/full clean1/overflow0。

| 项目 | 未复用 | baby复用 | 变化 |
| --- | --- | --- | --- |
| 完整Stage2 | 87.890573s | **81.062516s** | **−6.828057s /−7.7688%** |
| main | 72.5745945s | 65.8498435s | −6.724751s |
| init | 15.3159775s | 15.2126725s | −0.103305s，不能归为baby计算减少 |
| 小素数直接计时 | 6.5491625s | **0.2775545s** | −6.271608s，约23.60倍 |
| 内部t_reduce | 1.985s | 1.975s | 归约算法未变 |
| giant/G树 | 14.4405/22.5175s | 14.4385/22.5025s | 基本不变 |
| finv | 2.457s | 2.1835s | −0.2735s，算法/工作量未改 |
| 进程wall | 97.3035s | 90.433s | 含Stage1及启动 |
| observed host private峰均 | 7762.5 MB | **7717.5 MB** | −45 MB，沿日志单位 |
| 逐轮整卡NVML采样峰 | 5020/5020MiB | **4988/4976MiB** | 候选均4982，较对照少38MiB |

控制首末差0.059844s、候选两次差0.278838s；只有两次/模式，没有置信区间。小素数直接计时下降解释约91.85%的full减少；剩余约0.556449s含其他阶段变化和测量波动，不宣称为缓存新增算法收益。GPU整卡采样峰与S3逻辑42.284MiB节省不要求逐字节相等。

Q/Groot/leaf指纹仍为 **33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f /105d6128bbf522db /10619321735931855904**，四轮一致。403calls/1979251pairs/40218760reduced、66139GMP samples/2400selftests/4full checks、8241carry checked/finishes及projective/seed/Gamma计数一致bad0。workspace owned/full3356.537/3486.695MiB、24malloc/8grow/8890hit、主多项式账H2D/D2H11.37/5.46GiB、window D2H5.683GiB均不变；小素数流量属于另外的点生成边界。

候选非重复阶段：init **15.2126725s /18.77%**、giant **14.4385 /17.81%**、G叶 **2.9115 /3.59%**、G树 **22.5025 /27.76%**、finv **2.1835 /2.69%**、fold **13.5605 /16.73%**、H消Γ **0.691 /0.85%**、下降 **8.342 /10.29%**、块GCD **0.1725 /0.21%**、残差 **1.0478435 /1.29%**。小素数 **0.2775545 /0.34%** 已在残差中；t_reduce **1.975 /2.44%** 已在多项式阶段中，不重复相加。

1Hz full busy80.34→82.84%、main82.42→84.93%；full低≤5%样本18/175→11/160。full平均SMclock1761.17→1757.53MHz、温度65.38→66.21℃；这些为NVML近似窗口，不是SM occupancy或精确空闲时长。串行预算 `3600/T_full` **40.960→44.410曲线/小时（+8.423%）**，不包含Stage1/启动且未验证不同sigma流水。旧Prime95 CPU90.460s未重跑、D/degree/内存及检查范围不同，公平对照和长期目标仍未完成。

证据：[31/0](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_gate3_20261004/summary.json)、[188/0](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_full_gate_20261004.log)、[ABBA summary](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_ab_20261004/summary.json)、[CSV](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_ab_20261004/results.csv)、[阶段复算](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_ab_20261004/quantitative.json)、[NVML](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_ab_20261004/gpu1_summary.json)、[源码/exe hash及快照](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_ab_20261004/source_verified.json)。本机实验文件被Git忽略，tracked runner/gate可复现。

### 30.4 Systems 核对与下一阶段

同一最终exe/生产参数/GPU1另采Nsight Systems，exit0，SQLite记录 **114521个kernel**，仅GPU1；相对§29的114528少7次，ladder启动31→24。其余NTT主要kernel次数及8923次Mersenne S4归约不变；S4累计1.9387s。全scope H2D **13570496704bytes**、D2H **9310351576bytes**，与前一轮同形状trace分别相差 **400864 /56120960bytes**，恰好吻合小素数省略边界。不是减少主多项式账11.37/5.46GiB。

Stage1之后首个GPU kernel至末事件的近似Stage2窗口 **81.6345s**，kernel/copy/memset并集 **64.1096s /78.53%**，**17.5249s没有本进程GPU事件**。CPU-only初始准备被省略；不能把这个窗口或独立trace中的不同时间直接替代生产ABBA、整卡idle或occupancy。全scope ladder累计19.8104s（仍含baby/seed等）、Stage1 chain8.6637s单列排除。

11289个同步cudaMemcpy host API共 **33.6180s**，匹配的GPUcopy仅 **0.2867s /3283672648bytes**，差额仍包含GPU等待、driver/staging等。主机边界依旧值得优化，不能把host API总时间当成纯PCIe时间。既有文本kernel报告仍有UTF-8问题，使用原始SQLite numeric事件；未取得NCU硬件计数器。

下一阶段推进 **GPU组Γ→device leaf**，全好组的segment前缀/逆元传播与叶构造适合GPU，坏组保留可验证回退；减少CPU模乘和约3.4079GiB坐标回读/叶上传，新增同时驻留坐标约369MiB须纳入owner/预算。当前G树27.76%、fold16.73%、giant17.81%和baby/init18.77%比小素数/归约更值得继续优化。再评估固定F单谱缓存（约1GiB）、更新D成本模型及不同sigma共享scratch流水；生产save/队列接入、公平CPU重跑仍待完成。

证据：[Systems报告](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_nsys_20261004.nsys-rep)、[SQLite](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_nsys_20261004.sqlite)、[查询汇总](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_nsys_summary_20261004.json)、[采集与工具诊断](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_nsys_20261004.log)。


## 31. 最新逐步骤量化账与 device leaf 候选预算（2026-10-04）

### 31.1 范围、参数和周期口径

本节回应逐步骤量化需求。**实测时长来自已提交 `0cfa589` 的 small-prime 复用生产 ABBA 两次候选均值**，不采用§27的96.59秒旧基线。本轮候选实现 GPU组Γ/device leaf，默认关闭，专用门禁39通过/0失败。§31.1–§31.4保留补充文档时的基线/静态预算；随后完成的同二进制生产A/B及trace见§31.5–§31.6。测试中的逐字读回检查不纳入正常生产流量公式。

统一记号：`N`为待分解整数，`S=bitlen(N)`，`W=ceil(S/64)`；一系数载荷`8W bytes`。`D`为baby/giant步长，`P=φ(D)/2`（实际baby单位元集合大小），`I=floor(B2/D)+2`，`G=ceil(I/P)`。chunk j有`c_j`点，`E=Σ_j ceil(c_j/16)`段、`U=Σ_j ceil(ceil(c_j/16)/64)`组，`h=ceil(log2 P)`、`K=2^h`。B1影响Stage1输入Q和小素数集合；该I公式是当前实现的覆盖边界，不能换成素数计数π(B2)。D或chunk预算改变后必须重算取整。

测量实例：`N=2^4423−1`、σ26、B1=1000、Stage1 extra12；actual B2=2011326186870，D=1231230，W=70，P=115200，I=1633592，G=15；chunk为345600×4+251192，E=102100，U=1598。GPU1 RTX4060 Laptop，batch64MiB/arena6300MiB。Stage1不计入以下81.062516秒。

周期使用可校准系数而非虚构固定值：`C_Mont(W)≈2W²c_MAC+C_carry(W)`；CPU S位模乘/逆元/GCD分别记`C_mul(S)`、`C_inv(S)`、`C_gcd(S)`。`ℓ(k)=floor(log2 k)+1`。一次乘法形状a=(ma,mb,w)的模型为：

```text
M_a=max(ma,mb); r_a=ceil([2S+max(1,ceil(log2 M_a))]/bpw_a)
n_a=planner选择的2次幂，n_a≥2M_a r_a
C_poly(a)≈(3/2)n_a log2(n_a)c_bf+2n_a c_point
          +w C_reduce(S,M_a)+C_pack/carry(a)
C_reduce_generic≈W Q_a c_mulsub+C_estimate/addback，Q_a=max(0,d'_a−W+1)
C_reduce_Mersenne≈κ(S,M_a)W c_fold+C_normalize
C_stage=Σ实际shape C_poly(a)+该阶段CPU/点算术成本
```

κ为实际Mersenne折叠/修复轮数，不能假定所有输入一轮完成。Mersenne归约按条件启用；一般N仍走约W²主项长除法。周期模型是工作量/依赖链模型，不是GPU墙钟；有效并发、带宽、launch和同步另计。固定N/D时I≈B2/D，giant/G树/fold累计量近线性随B2增长；S增长同时影响W、NTT长度与精确性planner阈值。相关原语和公式见§23、§27、§29。

### 31.2 最新基线每一步：计算量及时间

| 步骤 | 可复算计算量或周期主项 | 实例时长 / 完整Stage2占比 |
| --- | --- | --- |
| 参数、D/baby枚举、context、自测 | baby枚举约D/2次整数gcd；D搜索另计（本次379419候选） | 包含在init剩余3.462673s /4.2716%，与F树混合，不能独立归因 |
| baby GPU ladder | `Σ_j(13ℓ(j)−6) C_Mont(W)`，j为实际baby索引；本例26709966次Mont | 8.967s /11.0618%，init子项 |
| baby CPU归一化/叶 | `4P C_mul+ceil(P/256) C_inv+O(PW)`导入导出；460800模乘、450逆元 | 2.783s /3.4331%，init子项 |
| F完整积树 | P−1=115199对；各节点形状的`Σ C_poly` | 含在init剩余；**init合计15.212673s /18.7666%** |
| 小素数补充 | 缓存覆盖Kc个点后GPU仅`Σ_(p∈T\Tc)(13ℓ(p)−6) C_Mont`；本例省11544588Mont与50108次CPU GCD，仍枚举/记录结果 | 0.277555s /0.3424%，已含下方残差，不重复加 |
| rev(F) Newton逆 | 至多`2ceil(log2(P+1))`对，本例34对；长度逐轮翻倍 | 2.1835s /2.6936% |
| giant seed/chain/精确段积 | seed 26076707次Mont（device seed省102110），chain12660336，精确段积1633592；合计40370635次Mont，约395632223000个limb MAC主项 | 14.4385s /17.8116% |
| CPU段分类/求逆/混合G叶 | 全好组`3E C_mul+U C_inv`，另有E次Γ累积模乘、O(IW)叶生成；坏组回退另计 | 2.9115s /3.5917% |
| G根积树 | I−G=1633577对，各shape `Σ C_poly`，中间层驻留GPU | 22.5025s /27.7594% |
| fold `H←GH mod F` | G−1=14轮，通常42对；另有(G−1)P=1612800次系数模减 | 13.5605s /16.7284% |
| Γ消除射影缩放 | 约P次CPU模乘/约减及O(PW)重写 | 0.691s /0.8524% |
| scaled余式树下降 | 2P−1=230399对（root一次、实分叉各两个），输出窗口限制归约量 | 8.342s /10.2908% |
| 块乘积、GCD/命名 | q=64：P−ceil(P/q)=113400次Mont、1800块GCD；命中后逐叶GCD，命名最多每命中叶2I候选 | 0.1725s /0.2128%；本次命名0 |
| 其余准备、检查排空、统计等 | 按实际检查开关和命中计算，无法仅由B2/N定值 | 残差1.047843s /1.2926%，包含上述小素数时间 |

init三个子项只计一次。以上非重复项合计81.062516s/100%；内部S4归约1.975s/2.4364%嵌套在多项式阶段，不能再相加。init剩余由减法得到，不能称为F树独立测量。baby两次日志分别ladder8.965/8.969、affine2.804/2.762秒。完整阶段[量化JSON](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_ab_20261004/quantitative.json)、[原日志2](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_ab_20261004/2_baby_reuse.log)、[原日志3](D:/code/MPA-OpenCl/build_cuda_cmake/_small_prime_ab_20261004/3_baby_reuse.log)。

### 31.3 每一步容量、累计生成量和传输量

以下为逻辑载荷而非allocator/容器容量。`V_NTT`是共享池，每阶段借用同一份；一次m对、长度n的A/B/Q基本项24mn bytes，另有raw/frontier、输出、twiddle、carry（§24）。CPU/GPU同时存在的对象分列，累计量不能与峰值相加。shape a的普通host边界定义`H(a)=16W max(ma,mb)`、`R(a)=8Ww`；实际trim、padding、检查和ledger范围可能不同。

| 步骤 | RAM逻辑活跃载荷；显存逻辑活跃载荷 | 累计生成与正常PCIe边界 |
| --- | --- | --- |
| 索引、context | CPU索引8P；CPU/GPU常量O(W) | baby标量H2D8P=921600bytes，常量另计 |
| baby点与monic叶 | CPU X/Z16PW+叶16PW；GPU坐标16PW及标量8P（S3容量复用） | 坐标D2H16PW=129024000bytes/123.047MiB；叶生成同量，F输入上传计入F树 |
| F全树 | CPU `8W[(h+1)P+2K]`=1308016640bytes/1247.422MiB；GPU V_NTT | 实乘输出生成/读回1159781840bytes；host输入模型1250958240bytes；树补齐复制另计 |
| 小素数复用 | CPU证明cache924400bytes/0.882MiB；S3后续容量按max(缺失点,seed最大点)保留 | 本例省标量H2D400864bytes、坐标D2H56120960bytes；GPU容量逻辑减42.284MiB |
| Newton finv | CPU固定8(P+1)W=64512560bytes、临时O(PW)；GPU V_NTT | 每轮按H(a)/R(a)相加；固定finv后续仍会上传 |
| giant链与段积 | 每chunk CPU/GPU坐标各16c_jW；段积各8Wceil(c_j/16)；seed/S3另计 | 全坐标生成/D2H16IW=1829623040bytes/1.7040GiB；段积生成/D2H8WE=57176000bytes/54.527MiB；device seed避免seed X/Z两向各57181600bytes，仍标量H2D408440bytes |
| 基线混合G叶 | CPU最大16PW=123.047MiB，段cache/GMP另计；GPU与raw共用 | 叶生成/H2D16IW=1829623040bytes/1.7040GiB |
| device G树 | CPU当前根8W(n_b+1)，GPU相邻frontier/raw峰225792000bytes/215.332MiB，另V_NTT | 真乘法累计生成16420181680bytes/15.2925GiB；metadata H2D24(I−G)=39205848bytes；根D2H8W(I+G)=914819920bytes/0.8520GiB |
| fold | CPU flat峰`8W(6P+5)`=387074800bytes/369.143MiB，固定F/finv另计；GPU V_NTT | §27.4形状模型H2D5313302400、D2H3506936160bytes，共8.2145GiB；CPU模减重写约8W(G−1)P |
| Γ消缩放 | CPU H8PW=64512000bytes/61.523MiB；GMP O(W) | 重写8PW；独立缩放阶段无GPU传输 |
| scaled下降 | CPU两层state16PW=123.047MiB，pack/GMP另计；GPU V_NTT | state累计8W(h+1)P=1161216000bytes；各真分叉shape(a+b,b+1,a)，root(P,P,P)，求和H/R即边界，不是保存所有state |
| 最终块积/GCD | CPU最终叶8PW及块8Wceil(P/q)；GPU上传叶与块 | 叶H2D64512000bytes；块D2H1008000bytes；命中回退另计 |

G树累计生成15.29GiB留在GPU，仅根回读约0.852GiB；数据生成量并不等于PCIe流量。主多项式ledger H2D/D2H约11.37/5.46GiB；全scope Systems H2D13570496704、D2H9310351576bytes，包含不同入口/检查/Stage1范围。以上各项有包含关系，应按边界归账，不能相加成第三个“总流量”。

最新基线实测资源：observed private峰均7717.5MB（原日志单位，含容器/GMP/staging）；1Hz NVML整卡峰4988/4976MiB，均4982MiB；NTT owned/full工作区3356.537/3486.695MiB。不是RAM与VRAM相加，也不能以整卡峰减allocator账精确计算剩余容量。多曲线应使用`V_shared+k V_curve+V_queue(k)`并考虑生命周期，不能直接复制两套当前workspace。

### 31.4 当前 GPU组Γ/device leaf：可计算增减量

开关`NTT_DEVICE_GLEAF=1`、坐标保留预算`NTT_DEVICE_GLEAF_MAX_MB=512`。好组GPU将64个精确段积再合成Γ，CPU每组一次求逆及一次Γ累积；设备直接写[-X,Z]到已有G树rawA。失败组只回读该组段积/X/Z，原分类及仿射叶回退补丁保留。单位元段即便处于坏组仍可用射影device叶。因此Γ求逆失败不等于整chunk回退。预算/后端合同不满足才整chunk走原路径。

令坏组内点数`I_bad`、段数`E_bad`，实际仿射补丁点数`A≤I_bad`。全部公式对active chunks求和；关闭CHECK/poison：

```text
新增GPU组Γ Mont次数 = Σ_j[(E_j−U_j)+U_{j,长度>1}] ≈ E
全好组避免CPU段逆元传播 ≈ 3E次模乘；求逆仍U次
CPU Γ累积模乘 E→U（节省E−U），新增组fix表构建每chunk 65次模乘/约减
device leaf构造 = O(IW) limb取负/拷贝，无逐点逆元
坐标D2H：16WI →16W I_bad（节省16W(I−I_bad)）
段/组D2H：8WE →8WU+8W E_bad（净节省8W(E−U−E_bad)）
叶H2D：16WI →16WA（节省16W(I−A)）
新增group-fix H2D = #chunks·8W·65
保留坐标显存 =16W max(c_j)，直到该chunk最后一棵G树完成
新增组结果/补偿表显存峰 =8W max_j(U_j+65)
CPU坏组坐标 =16W max_j(I_bad,j)，补丁叶16WA（按batch生存）
```

全部好组实例：新增约102100次GPU Mont，省306300次CPU传播模乘与100502次CPU Γ累积模乘；组逆元仍1598次。组Γ D2H894880bytes/0.853MiB；省坐标D2H和叶H2D各1829623040bytes，省段积D2H57176000bytes，但新增组读回后净段/组节省56281120bytes。加上5×36400bytes表上传，净PCIe节省**3715345200bytes/3.4602GiB**。这是静态净边界预算，不含检查或坏组。

额外延长坐标驻留峰387072000bytes/**369.141MiB**，组表/结果峰225680bytes/**0.215MiB**；原段积GPU缓冲继续保留。**全好组仍分配每chunk主机gseg零数组**，本例最大12096000bytes/11.536MiB，尚未消失；54.527MiB是整曲线段积累计生成量。原16c_maxW=387072000bytes的全坐标读回对象可避免，原batch叶对象16PW也避免；实际RAM峰需生产采样。坐标显存本来在点生成存在，新代价主要是与G树/NTT阶段峰值重叠，不能把369MiB直接当作已实测整卡峰增量。512MiB仅限制坐标，不限制NTT、段积、S3或整卡总显存。

基线G叶直接计时仅2.9115s/3.5917%；仅消除此项、其余不变的理想预算为78.151016s（非预测上界，传输/同步部分还可能在别的阶段）。按有效PCIe 8GiB/s、全部节省暴露于关键路径，纯字节预算约0.4325s；不能和整项2.9115s无条件相加。新增GPUΓ kernel、坐标存活、坏组密度会改变结果。当前候选**39/0门禁**覆盖链/ladder、部分组、单叶尾批、非单位元/饱和、宽limb、blocking、预算和后端回退及毒化拒绝；这些门禁证明所测边界正确；生产时长/峰值证据另见§31.5。

源码：[device叶kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1125)、[resident G树填充入口](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4354)、[坐标owner及组Γ](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8960)、[预算/后端选择](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9781)、[设备叶与稀疏补丁](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10013)、[独立流量统计](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10604)、[门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_device_gleaf.py:1)、[39项结果](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_gate2_20261004/summary.json)。bad patch H2D单独记patch_words，尚未并入原raw上传ledger，比较退化实例需把两项合并。


### 31.5 生产同二进制 ABBA、资源与验证

隔离exe SHA256 **D4FD5C831B5E45BF485C419B2F6DA3E26F24C3D6EE9B9A3CF6E3DDE627527A5D**；compile276.3s/link3.0s，exit0。CUDA源码SHA256 **3626A87FED56C12E58F3B73B752622FB4D82D35595D5FFB65051ECDE7BB1C138**；生产前后源码及exe与provenance核对一致，快照保存在measured目录。没有改动GPU0用户Stage1工作。

先完成device leaf专用39/0、原完整回归188/0（关闭新开关验证旧后端/调度兼容），开启新开关的 `--check-F --evaluate --evaluate-batched` CPU/GMP冻结向量验收也全部通过：baby/F逐项、chain逐点、因子、hits和hit_primes与CPU一致。39门禁逐字核对设备叶、GMP组Γ，覆盖链/ladder、宽limb、坏组/饱和、部分组和尾批、预算/后端回退及毒化拒绝。生产配置关闭这些额外检查，保留原mandatory selftests与NTT/GMP样本。

首次生产ABBA在第二轮被runner中旧“H2D必须相等”的断言拒绝；当时根/最终叶/系数/样本签名均一致，主账H2D11.37→9.66GiB正是预期节省，小形状因两位小数取整未触发。对保存日志重放同一断言复现后，只调整device_gleaf的H2D合同，按**8×device_leaf_words**核对raw ledger差额（两位小数差额容差0.011GiB），D2H/数值/检查/异步调度约束保持。稀疏补丁上传另记，不在raw ledger。新增[runner回放门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_device_gleaf_runner.ps1:1) **7/0**：正常候选可通过，H2D/D2H、coeffs、oracle signature、raw async异常和旧target的不匹配仍拒绝。此失败不是CUDA数值修复；CUDA/exe未重编译。第一次两轮证据保留为诊断，不混入最终均值。

最终配置与§31.1相同，target `device_gleaf`，顺序host/device/device/host；Mersenne/device seed/exact/Gfinv batch/fold-flat/G-root device/scaled/window/chunk均1，carry batch/oracle async均0，sample96/every8，仅切换新开关。四轮full秒数 **81.758986 /75.688208 /76.753408 /80.681035**，均exit0、clean1、overflow0。

| 项目 | host叶 | device叶 | 解释 |
| --- | --- | --- | --- |
| 完整Stage2 | 81.220011s | 76.220808s | Δ=-4.999202s |
| main精确计时 | 65.822471s | 61.521532s | Δ=-4.300939s |
| init | 15.397540s | 14.699276s | Δ=-0.698263s |
| giant | 14.448500s | 14.198500s | Δ=-0.250000s |
| G叶完整phase | 2.946500s | 0.284000s | Δ=-2.662500s |
| G树 | 22.473500s | 21.911500s | Δ=-0.562000s |
| fold | 13.551500s | 13.315000s | Δ=-0.236500s |
| 下降 | 8.327000s | 8.209500s | Δ=-0.117500s |
| finv | 2.266000s | 2.191000s | Δ=-0.075000s |
| 内部归约（嵌套） | 1.977000s | 1.972500s | Δ=-0.004500s |
| observed private峰均 | 7718.0 MB | 7495.5 MB | −222.5MB，沿日志单位 |
| 1Hz整卡NVML采样峰 | 4976/4976MiB | 4988/4988MiB | +12MiB，不能替代allocator/实际瞬时峰 |
| 主raw多项式H2D/D2H账 | 11.37/5.46GiB | 9.66/5.46GiB | 坐标/组传输不在该主账 |

完整均值 **81.220011→76.220808s（耗时−6.1551%）**；main减少4.300939s。两次候选均比两次对照快，但各模式内部范围约1.07s，仅两次/模式，没有置信区间。init算法未改，−0.698264s不能归于device叶；不能将完整5.00s全部当作CPU叶/Gamma或PCIe收益。单曲线串行换算 **44.324→47.231曲线/小时**，不是不同sigma并发吞吐实测。

注意计时范围修正：本轮G叶phase用 `real_batched_wall.gleaves`，包含组Γ准备、分类及循环开销。`real_batched_gleaves.in/invert/out`仅是局部子计时，不能继续拿三者之和代表迁移后完整phase；§31.2旧基线2.9115s为旧子项和，当时完整phase均值2.917s。device fill计入G树阶段；它的host launch计时不是kernel墙钟。最新非重复分摊如下：

| 阶段 | 秒 | 完整Stage2占比 |
| --- | --- | --- |
| stage2_init | 14.699276 | 19.2851% |
| giant | 14.198500 | 18.6281% |
| gtrees | 21.911500 | 28.7474% |
| fold | 13.315000 | 17.4690% |
| descent | 8.209500 | 10.7707% |
| inverse | 2.191000 | 2.8745% |
| gscale | 0.581000 | 0.7623% |
| accum | 0.173500 | 0.2276% |
| gleaves | 0.284000 | 0.3726% |
| residual | 0.657532 | 0.8627% |

生产active chunks5、好组1598、好段102100、device树15；bad groups/patch/check均0。device叶228702880words，组D2H894880bytes，避免坐标D2H/叶H2D各1829623040bytes，避免旧段D2H57176000bytes；坐标/组payload峰387072000/225680bytes，全部与§31.4公式一致。全好组旧GfinvCache.requests/groups变为0，新组逆元记在device_gleaf.groups，不能解读成没有Γ求逆。NTT full workspace仍3486.695MiB、24malloc/8grow/8890hits；逻辑坐标新增生命周期与整卡采样+12MiB口径不同，不能声称坐标只占12MiB。

全部四轮G根 **105d6128bbf522db**、最终叶 **10619321735931855904**、Q SHA **33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f** 一致。403底层批调用、1979251对乘法、40218760归约系数、66139GMP样本、2400S4自测、4full checks、8241carry finishes/checked保持；oracle jobs1126/签名b9cbd2041266767a保持。

NVML full busy约79.525→84.881%，main80.992→87.213%；giant/G树/fold窗口≤5%低负载样本4/106→0/98。采样busy不是SM occupancy，阶段边界从墙钟推算；不能称空闲完全消失。1Hz峰和既有RAM快照可能漏掉瞬时高水位。

证据：[最终ABBA](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_ab_final_20261004/summary.json)、[量化复算](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_ab_final_20261004/quantitative.json)、[NVML](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_ab_final_20261004/gpu1_summary.json)、[源码核对/快照](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_ab_final_20261004/source_verified.json)、[188门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_full_gate_20261004.log)、[CPU/GMP验收](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_acceptance_20261004.log)、[runner门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_runner_gate_final_20261004.log)。GPU入口与39项门禁均已tracked，测量文件位于ignored build目录。重跑：同exe运行 `tools/bench/bench_stage2_reduce_ab.ps1 -Target device_gleaf` 并使用§31.1全部参数；检查gate的exe/output/device参数分别指定隔离exe、新空目录、1。

### 31.6 Systems：传输节省、GPU新增开销与下一步

同exe/GPU1/生产参数另采Systems2026.1.3，exit0；只trace CUDA/NVTX，CPU采样/context-switch关闭。SQLite显示 **114541个kernel，全部device1**，相对§30 trace增加20次：5组Γ、15device叶。借用原segprod kernel的两类grid可区分：组Γ gridX=6×4+4×1/block64，累计 **0.191290s**；旧16点段积grid338×4+246×1，累计0.282050s；device叶累计 **0.058235s**。新增GPU时间约 **0.249525s**。8923次Mersenne S4归约累计1.939887s，NTT工作量未消除。

全scope H2D **11741055664bytes**、D2H **7424447416bytes**、D2D23173360bytes。相对§30同形状trace分别减少 **1829441040 /1885904160bytes**，D2D不变；恰好等于坐标/段/叶省略减新增组结果和5张补偿表，净 **3715345200bytes/3.460185GiB**。这是实际trace字节验证，不把全scope流量等同Stage2主多项式账，也不将3.46GiB直接换成5秒纯PCIe收益。

Stage1后首kernel至末GPU事件的近似窗口 **76.927277s**，kernel/copy/memset并集 **64.086358s/83.3077%**，无本进程GPU事件 **12.840919s/16.6923%**。§30对应窗口81.6345s/64.1096s/17.5249s；两份独立trace表明非事件时间约少4.684s，但不是精确Stage2 range、整卡idle或ABBA计时，初始CPU-only准备被省略，profile也有开销。

11284个同步cudaMemcpy host API累计 **33.854123s**，匹配GPUcopy仅 **0.124606s/1397950488bytes**。即使传输字节减少，host API仍包括GPU等待、staging/driver/scheduling；不能把33.85秒都当作可省传输或CPU空闲。文本kernel summary仍遇到UTF-8问题，采用原SQLite numeric事件。NCU硬件计数器权限在前轮已被拒绝，本轮未取得周期/occupancy/DRAM实测值，因此§31周期系数继续待校准。

下一阶段优先研究 **device fold与固定F/finv设备驻留**：当前fold17.47%、G树28.75%，host fold边界模型仍8.2145GiB且反复上传固定输入；先建立设备多项式owner、共享scratch租用和坏路径回退合同，再比较原算法结果。新GPU叶kernel只有0.058s，组Γ0.191s不应先占据主要优化预算。初始化19.29%、giant18.63%、下降10.77%也仍可优化；不同sigma流水必须拆开每曲线与共享内存，验证真实curves/hour，不能把重复同Q的--curves当作证明。生产save/队列接入、公平Prime95 CPU重跑仍未完成；现有90.460s CPU参考配置/检查范围不同，**本轮不声称长期目标已达成**。

证据：[Systems](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_nsys_20261004.nsys-rep)、[SQLite](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_nsys_20261004.sqlite)、[numeric查询](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_nsys_summary_20261004.json)、[采集日志](D:/code/MPA-OpenCl/build_cuda_cmake/_gleaf_nsys_20261004.log)。

## 32. 独立生产编译、Stage1 save 与可选队列字段（2026-10-04）

新增 `ecm_cuda_stage2.exe`，通过 [独立脚本](D:/code/MPA-OpenCl/tools/build/build_ecm_cuda_stage2.ps1:1) 编译，共用本报告的 CUDA 树引擎；不需要编译 Stage1/CGBN。完整配置、限制、失败语义与证据见 [生产入口说明](ECM_CUDA_STAGE2.md)。

- [save 解析](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:106)：param0、普通仿射 X，支持 N 表达式、64位 sigma 和可选校验和。校验和存在时验证 `(B1*sigma*N*X) mod 4294967291`；不能识别历史归一化 bug。
- [入口](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:37) 调用 [run_real](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10694) 的可选 saved Q；[11021](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11021) 装入 `(X:1)`，跳过指数生成和 Stage1 ladder，不再次执行 choose12。其后的算法、资源和检查仍遵循§31。
- [可选队列字段](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:301) 支持 `ECMSTAGE2=[AID,]k,b,n,c,filename[,B2-or-zero][,skip_curves][,num_curves][,"known-factors"]`；B2 缺省/0使用 CLI/ini，skip缺省0，num缺省/0为剩余全部。文件名/因子用引号。用户的 B2=26000000000/skip960/num10 精确选记录961–970；占位因子 xxx 被识别后因无效数值拒绝。
- [配置/调度](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:443) 复用 ini 和 worker 章节。每条曲线为独立子进程，顺序执行，隔离全局计数/缓存；成功写 JSONL，整任务完成才写 finished 并移除 worktodo。修正 [worktodo 移除合同](D:/code/MPA-OpenCl/src/core/ecm_worktodo.cpp:557) 以拒绝错删已改变的首任务。一个队列文件同时一个消费进程。

独立 exe SHA256=`c63249542418de3cc6af2bb93b443ec39eca053a29b95d3df125eefcf31b4860`，26项验收0失败。固定 N=2^128+1 的真实 Stage1 save 输出 CPU 参考因子59649589127497217；不同sigma26/27各自Q匹配CPU；队列跳过/worker隔离/输入失败/CUDA失败均已核对。

M4423同§31生产形状，从真实choose12 Q恢复：init14.808017/main60.308395/full75.116411s（单次集成检查，不构成A/B）；Q SHA与**最终叶哈希10619321735931855904**均与§31一致，403批/1979251对/40218760coeffs、2400自检/66139GMP/4full checks保持且bad=0。保存Q归一化成X:1后，投影G根的原始scale可改变，本次root_hash=acd84d2595022e3e；Gamma校正后的叶值才是跨入口逐字比较依据，见 [校正](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10136)。

本入口无Stage2中途checkpoint、索引持久恢复、跨曲线共享流水线或PrimeNet；重跑未移除的任务可能重复已完成记录，源save不改写。这里没有更改§31性能模型，也没有完成公平Prime95 CPU性能比较。证据：[验证汇总](D:/code/MPA-OpenCl/build_cuda_cmake/_production_stage2/acceptance/summary.json)、[large-N日志](D:/code/MPA-OpenCl/build_cuda_cmake/_production_stage2/acceptance/m4423_engine.log)。

## 33. GPU 驻留 fold：算法、访存与验证（2026-10-04）

本轮在 `79c649e` 上修改，与 §31 的 device leaf 默认组合相比只切换 `NTT_FOLD_DEVICE`。实验 exe 缺省关闭，独立生产 wrapper 缺省开启；超过预算或不满足后端条件时沿用主机 flat fold。它复用原 S4/NTT、carry 和 GMP 采样，不改变 poly 模 F 的数学操作数或精度。

### 33.1 执行步骤及代码

1. **准备逆多项式**：仍计算 `finv = rev(F)^(-1) mod X^(P+1)`，由原 Newton 路径完成。把它转为连续 `finvflat`，供 fold 和后续 scaled 下降复用。[准备及 owner 初始化](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9696)。
2. **申请驻留 owner**：两块独立显存 A/B，不借用会被下一次 NTT 覆盖的 arena/raw staging。A 存 F、finv、H、G、逆序输入；B 存 T、qrev、qF。[owner 合同](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3281)、[分配与回退](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7537)。
3. **装入固定多项式**：F 和 finv 各上传一次，模数 N 上传一次；第一棵 G 根作为初始 H 上传。[upload/seed](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7579)、[批次接入](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10066)。当前 G 根仍先由树读回 CPU，随后上传到 fold owner；直接 GPU→GPU 交接尚未实现。
4. **形成 T**：对余下每棵 G，上传根系数，计算 `T=G·H`。canonical 系数由 GPU gather 转入原 NTT digit buffers，S4 归约结果 scatter 回 B；保留 carry 和 GMP 检查。[驻留乘法](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7625)、[输入/边界校验](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3412)。
5. **决定是否除法**：设 `nT=nG+nH−1`。若 `nT≤P`，T 已是余式，D2D 复制回 H，保留旧路径的声明长度；否则令 `k=nT−P`。
6. **求逆序商**：GPU 抽取并逆序 T 的最高 k 个系数，计算 `qrev=(rev(T)_k·finv_k) mod X^k`。[逆序 kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7496)。
7. **求 qF 低项**：逆序 qrev 得到 q，计算 `(q·F) mod X^P`。启用 output window 时只归约所需低 P 项；未启用时保留完整归约，owner 仅接收所需输出。
8. **求新 H**：GPU 逐系数计算 `H=(T_low−qF_low) mod N`，借位时加 N；输入输出都是 ordinary canonical 整数，不加入 Montgomery 因子。[模减法](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7503)。GPU 用 atomicMax 求最高非零系数位置，每次除法只读回 8 字节长度，保留原 trim 语义，零多项式长度为 1。
9. **继续 fold**：H 留在 A，下一批只上传新 G，重复 4–8。短 G、不满 P 的尾批和零值走相同合同。[step](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7647)。
10. **接回下降**：全部 G 完成后只读回最终 H，释放 fold owner，转成原 CPoly；原 Γ⁻¹ 校正、scaled 下降、accum/GCD/naming 与因子验证继续执行。[finish 与桥接](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10128)。F 树和 host finvflat 仍保留，当前没有实现全程 device descent。

### 33.2 计算量与容量公式

沿用 §23 约定：N 的位数为 n，`W=ceil(n/64)`；`P=φ(D)/2`，本引擎 `I=floor(B2/D)+2`，G=`ceil(I/P)`，fold 次数 L=G−1。第 j 次除法令 `k_j=nG_j+nH_j−1−P`；乘法实现的统一 padded operand stride 为 `m=max(na,nb)`。下面公式适用于非空、每次均需除法的生产形状；短/零路径按实际计数扣除。

- 算术仍为每次 **3 次**底层批乘法：`G·H`、逆序高项乘 finv、q 乘 F。对 L 次 fold：`C_fold=Σ[C_mul(nG_j,nH_j,nT_j)+C_mul(k_j,k_j,k_j)+C_mul(k_j,P+1,P)]+C_reverse+C_sub`。`C_mul` 采用 §23 的精确 NTT/归约模型；没有少做 NTT 或依赖 MAC。
- 新逆序处理 `2WΣk_j` words；减法处理 `LPW` limbs，每 limb 常数次减法/借位，借位系数额外常数次加法/进位，再扫描 W limbs 判零。以周期计可写 `C_reverse≈2WΣk_j·c_copy`、`C_sub≈LPW·(c_sub+c_scan)+W·R_borrow·c_add+C_atomic`。这些 c 需按设备实测校准，不能直接当作整卡墙钟；数据并行、访存和同步另计。
- 两块 owner 容量：A=`5(P+1)W` words；B=`(4P+2)W` words；额外模数 W words、metadata 24 bytes、长度 8 bytes。**显存公式 `V_fold=8W(9P+8)+32` bytes**。M4423/P115200/W70 为 **580612512 bytes / 553.715 MiB**。owner 在 finv 后申请、Γ/下降前释放；它不在 arena cap 账目内。
- CPU 不再为每个 fold 构造 T/qrev/qF 的返回向量及逆序/减法暂存，`fold_flat.peak_bytes=0` 仅表示这项主机 scratch 已省去。F 树、finvflat、G 根、最终 H、下降和 GMP 工作内存仍在，不能把这个 0 理解为总 RAM 为 0 或总 RAM 峰值必然下降。

### 33.3 数据生成与传输公式

生产计时关闭 `NTT_FOLD_DEVICE_CHECK/TEST`，output window=1。记 `m1_j=max(nG_j,nH_j)`、`m2_j=k_j`、`m3_j=max(k_j,P+1)`。

- 主机路径被省去的输入上传 `U_old=16WΣ(m1_j+m2_j+m3_j)` bytes；是两个 padded operand 的 raw H2D，而不是仅有效系数。被省去的结果读回 `D_old=8WΣ(nT_j+k_j+P)` bytes；window=0 则每次输出长度替换为 `2m−1`。
- 新 owner 实际 H2D：`U_new=8W[2(P+1)+Σ_allG nG+1]+24·M` bytes，M 为驻留乘法数；依次为 F/finv、全部 G 根（第一根直接 seed H）、模数和 metadata。没有把同一个 G 的 seed 再上传一次。
- 新 owner 实际 D2H：`D_new=8W·nH_final+8·L_div` bytes，只有最终 H 和除法长度。每次检查模式还会额外读 H，单独列 `checked_words`，不纳入生产性能。
- 此轮 M4423：L14/M42，`U_old=5313302400`、`D_old=3506936160`、`U_new=1043846608`、`D_new=64512112` bytes；净减少 **7711879840 bytes / 7.182 GiB**。全局 kernel/GMP/carry 小数据传输仍在。这些是 fold 边界精确计数，不能直接解释为省去等量 PCIe 时间。
- `real_batched_rawupload/coeffback` 只统计原乘法 ledger，新 owner 上传/读回由 `real_batched_folddevice` 独立统计；所以原 ledger 9.66→4.71、5.46→2.19 GiB 的差额是 U_old/D_old，而净收益还需减去 U_new/D_new。全部 S4 output 账目增加独立 owner 返回字数，原 G-root resident_words 不混入 fold。

### 33.4 门禁、生产测量和限制

本轮新门禁 [test_stage2_fold_device.py](D:/code/MPA-OpenCl/tools/test/test_stage2_fold_device.py:1) 共 21 个顺序 GPU1 进程：65/127/129 位，30 个 GMP schoolbook/monic long-division fixture，零/near-N/短/重复 fold，冻结因子，两种窗口，budget=0、注入 allocation failure、copied-pack、carry-trace 回退，故意污染 H 必须被 GMP check 拒绝。注入分配失败不等于真实显存耗尽测试。现有 suite **188/0**；CPU F/giant/端到端冻结因子对照通过。生产测量关闭可选 fixture/check，mandatory selftests/GMP sampling 保留。

A/B runner 新增 `-Target fold_device`，固定已验证 device leaf 等基线开关，执行同 binary host/device/device/host。保持 Q、原投影 Groot、最终叶值/因子、乘法/归约/carry/GMP workload；42 个 fold 移入 owner 使 raw_async/out_async 各少 42，按精确差额校验。14 项 runner 污染检查确认伪造传输量、乘法/系数数目、抽样签名或 owner 账目会被拒绝；原 device leaf runner 的 7 项回归也通过。首次 A/B 在旧 async-count 等值条件处停止，该对仅用于诊断；最终计时采用修复 runner 后的完整 ABBA，日志重放不是额外性能测量。

M4423/sigma26/B1=1000/B2=2011326186870/D1231230/P115200/I1633592/G15，GPU1 RTX4060 Laptop 8 GiB，extra12，batch64/arena6300 MiB。同 exe SHA256 `cb72a2d0d2a57a665d1e8bd894fe5f410778f23b963d2bebeb7b9faa53d5e44f`，Q SHA256 `33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`。最终 ABBA 完整顺序 total=76.613115/74.221700/74.199262/76.613500 s，全部 exit0/clean1/overflow0。

- **完整 Stage2：76.6133075→74.210481 s，−3.1363%**；main61.223926→58.7900895 s，init15.3893815→15.4203915 s。init未优化。串行曲线吞吐由46.99→48.51 curves/h；不是不同 sigma 并发吞吐。更早一组完整 ABBA 得76.397928→74.090262 s，但前两次与CPU编译重叠，两种模式各占一次；最终结论用上述无编译重叠组。
- **fold：13.2055→10.6725 s，−19.183%**；其 multiply 子计时12.198126→10.5159955、prepare .4658375→0、subtract .298807→.052518 s。owner setup均.0260495、最终H读回.0052265 s另计在main/full内；初始seed上传也计入完整执行，不能仅按fold子计时推算total。reverse日志.0005575 s是主机enqueue计时，GPU耗时用下述Systems结果。
- 新路径的非重复阶段占比：init约20.8%、giant19.1%、G树29.4%、fold14.4%、下降10.9%、finv2.85%，余下为G叶、Γ校正、GCD/命名/准备等。t_reduce1.971→1.974 s保持，约占新total2.66%，嵌套在各多项式阶段，不另加。
- 403批调用/1979251对乘法/40218760归约系数/2400自测/66139 GMP samples/4 full checks/8241 carry checked不变，bad0；Groot `105d6128bbf522db`、最终115200叶/8064000 words哈希 `10619321735931855904` 四次一致。
- sampled NVML GPU1峰4988→5544 MiB，+556 MiB，接近owner公式；arena完整缓存payload峰3486.695 MiB保持、24 malloc/8 grow/8890 hit不变。host observed private峰7490/7536→7599/7558 MB，均值7513→7578.5 MB（+65.5 MB）；未观察到总RAM下降。host padding控制容量123.0 MiB保持，输出pinned容量246.1→123.0 MiB；不要把单项减少代替进程峰值。
- 1Hz NVML推断full Stage2 mean GPU busy84.40→86.38%，≤5%样本10/151→7/146；main89.27→89.84%。这是busy采样，不是SM occupancy；阶段边界由wall计时倒推，为近似。读取了当前GPU-Z日志作实时参考，但它不记录CUDA device ID，设备归属以NVML GPU1为准。GPU0外部Stage1未改。

另采Systems2026.1.3，114625 kernels全部GPU1：新增28次reverse共.013165195 s、14次subtract共.044685393 s，另有42次resident结果scatter；NTT/gather算法未少做。相较§31.6相同输入trace，H2D11741055664→7471599872、D2H7424447416→3982023368 bytes，净减少4269455792/3442424048 bytes，逐字吻合上述U/D公式，D2D23173360 bytes保持。实际copy GPU时长合计约1.489→.954 s；同步cudaMemcpy对应host时长33.854→34.059 s并未下降，包含等待/staging/scheduling，不能用其表示纯PCIe耗时。

Stage1之后近似GPU事件窗口由76.927277→72.975641 s，事件并集64.086358→63.501751 s，无本进程GPU事件12.840919→9.473890 s（16.69%→12.98%）；这不是精确Stage2范围，也不是整卡idle。profiling另跑、存在工具开销，性能结论以ABBA为准。

本地证据：[最终ABBA](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_device_ab_clean_20261004/results.csv)、[量化](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_device_ab_clean_20261004/quantitative.json)、[NVML](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_device_ab_clean_20261004/gpu1_summary.json)、[新门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_device_gate_20261004/summary.json)、[Systems汇总](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_device_nsys_summary_20261004.json)。这些ignored测量文件不随源码提交；编译及测量源文件哈希在candidate_source.json/provenance.json中。


生产 wrapper 的 `NTT_FOLD_DEVICE=1` 和 `NTT_FOLD_DEVICE_MAX_MB=640` 可显式覆盖；0 关闭。分配前检查预算、NTT 形状、当前 free VRAM，以及最大 workspace 剩余增长+1 GiB reserve；两块大缓冲 OOM 时释放已分配部分并回退。该检查不构成全程序硬显存上限，后续坐标/树/上下文还要使用显存。`g_s4_final_readback`、carry trace、hostpack、非 direct-pack 等后端使用原路径；若根本未启用 flat/finv，owner 不启动。

本阶段规划的 G-tree 根直接交接已在§34实现，消除 G 根 CPU 往返；随后减少 GMP/carry 检查导致的细粒度同步，并试不同 sigma 的共享 scratch 流水。新的 owner 显存按每个并发曲线另计；本轮是单曲线串行速度，未证明多曲线吞吐或公平 Prime95 CPU 加速比。

## 34. G-root 直接交接给 GPU fold（2026-10-04）

本节接续 §33 的 GPU fold，历史对照为 `1741392`。实验开关 `NTT_GROOT_TO_FOLD=1` 消除 G 根的 CPU 系数向量与 D2H→H2D 往返。仍使用原树构建、NTT、S4 归约和三乘法模 F fold；不减少素数覆盖、乘法对数或检查样本。

### 34.1 执行步骤与生命周期

1. owner 在 F/finv 完成后按 §33 预算创建；固定 F、finv、模数上传一次。
2. 每棵 eligible G 树最后一层完成后调用 root sink。借用的 raw A/B 源必须是实际 raw 缓冲首地址，`count*W` 在其容量内，且 `0<count<=P+1`。
3. 第一根 D2D 拷到 owner 的 H 槽；其余拷到 G 槽。拷贝与输入摘要在默认 stream 排序，然后树构建器才返回。owner 持有独立显存，不保留 raw 指针；后续 S4 raw 复用不破坏 H/G。没有让两个曲线同时占用共享 raw。
4. 后续根调用 `step_loaded`，直接执行原 T=GH、rev(T)·finv、rev(qrev)·F 和模 N 减法；不再调用 host `upload(G)`。
5. 最终 H 仍读回 CPU，接 Γ⁻¹、scaled 下降和 GCD；此轮并未消除 F-tree、下降或 Gamma 的主机数据。
6. owner 未启动、非 root-only 或 G-device 后端回退时，继续物化 host 根并走旧 seed/step。singleton/短尾根亦覆盖；要求 CHECK 的大 P 仍明确拒绝。

代码入口：[root sink构建器](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4354)、[sink调用](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4508)、[摘要kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7460)、[FoldDeviceState](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7521)、[交接检查与D2D](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7592)、[run_batched调度](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10005)、[日志](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11412)。

### 34.2 数值门禁与日志合同

CPU 的串行 FNV 需要读所有 G 根，会抵消读回优化。`real_batched_groot` 新增 `root_hash_complete`；仅为 1 时 `root_hash` 才代表全部根。直接路径为 0，旧 FNV 可能是初值或部分根摘要，不能作为完整结果比较。

两种模式都对 **实际 fold owner 输入** 计算 `mixsum_xor_v1`：全局 word 序号跨根连续，分别做位置混合后的 64 位 sum 与 xor（模 2⁶⁴）。不同位置参与混合，因此会检测常见顺序变化；它是诊断摘要，非密码学承诺。只读回最终 16 字节。`real_batched_rootfold` 记录 requested/trees/words、双向避免字节数、检查字数、digest_words/sum/xor 和 handoff enqueue 时长。最终 Γ 校正叶值 FNV、因子、Q、NTT/carry/GMP 工作量仍保持旧门禁。

`NTT_GROOT_TO_FOLD_CHECK=1`（P<=64）从 raw 源与独立 owner 逐字读回比较；故意污染第一 word 必须失败。`NTT_FOLD_DEVICE_CHECK` 再独立对照 GMP 多项式长除法。性能运行关闭可选 CHECK/TEST_BAD，保留 mandatory 自检/GMP sampling。

门禁：新路径 35/0，含 65/127/129/8192 位、singleton、短尾、冻结因子、window0、blocking、host leaf、各回退与污染拒绝；摘要 kernel 从原源码原样提取单独编译，32/0、5250248 words、跨 256-thread/1024-block 与分段边界以及大序号，GPU 对照串行 host 摘要。原 fold 21/0、原 suite 188/0、CPU F/giant/因子对照通过。新runner污染门禁22/0（两组CSV均通过），原fold runner14/0、leaf runner7/0。

A/B 仅对 root_fold 免除 **完整 G-root FNV 等值**，替换为两个输入摘要和精确传输差额；最终叶 FNV、403 批/1979251 对/40218760 系数、抽样签名、raw/out async 等仍要求一致。启动脚本曾遗漏 root_fold 的 seed-enabled 预期而停止，修正后重新完整 ABBA；该失败运行不计性能。

### 34.3 计算、容量和传输公式

记 W=ceil(bits(N)/64)，I 为全部 giant points，G 为 G 多项式数，每根系数数 `nG_j=degree_j+1`，故总根 words `R=WΣnG_j=W(I+G)`。

- 原根 CPU FNV 的 R 次串行依赖乘加和根向量物化被省；新增设备摘要 O(R) word loads/整数混合，最多 1024 个 256-thread blocks/根，归约后每 block 两个 64 位 atomics。该检查成本两种 A/B 模式共有。可校准周期模型为 `C_digest≈R*c_mix/GPU并发效率+Σblock归约及atomic成本`；不能用 FLOP/s 推算整数摘要时间。
- 省 H2D=`8R`，省 D2H=`8R`；新增 D2D=`8R`，并新增最终 16 B 摘要读回。与本轮 host control 比，净主机边界减少 `16R`；与无摘要的 §33 旧 exe 比需扣除 16 B。
- owner 总容量由 `8W(9P+8)+32` 增为 **`8W(9P+8)+48` bytes**，仅多一个 16 B 摘要缓冲；无新大型显存区。root payload 原 host 峰约 `8W(P+1)`，直接路径为 0，树节点 header/metadata 与其他 CPU 数据仍在。
- 新 owner H2D 为 `8W[2(P+1)+1]+24M`；若有回退根还加 `8WΣfallback nG_j`。owner D2H 为 `8W*nH_final+8L_div+16`（CHECK 另计）。不改变旧 fold 避免上传/读回统计。
- M4423：W70/P115200/I1633592/G15，R=114352490，`8R=914819920 B`（0.852 GiB），净省 **1829639840 B / 1.704 GiB**。owner 580612528 B；H2D1043846608→129026688 B，D2H64512128 B 两种模式相同。G 根 host 系数 capacity64512560→0 B（包括 allocator capacity 尾量）；不等于整个进程 RAM 减少 61.5 MiB。

### 34.4 测量与下一轮

同 exe SHA256 `b70818c167acab6f7ec57b22386ed88ae78bb6d7f5cad6c589fb215683eb7198`，GPU1 RTX4060 Laptop 8 GiB；M4423/sigma26/B1=1000/B2=2011326186870/D1231230，extra12/batch64/arena6300 MiB。所有运行均 clean1/overflow0/exit0，无 CPU 编译重叠；GPU0 外部生产任务未改。

- 第一组 ABBA total75.571258/74.744045/75.301211/75.165291 s，均值75.3682745→75.022628 s（−0.459%），main59.6452195→59.7453275 s，未改善。
- 第二组 ABBA total75.040112/74.026664/74.171704/74.570975 s，均值74.8055435→74.099184 s（−0.944%），main59.429326→58.8242125 s（−1.018%）。
- **八次合并：total75.086909000→74.560906000 s（−0.7005%）**；main59.53727275→59.28477025 s（−0.4241%）；init15.5496365→15.276136 s。初始化没有被本改动优化，不能把全部 total 差额归因于交接。主阶段波动约 .2–.6 s，只有四个样本/模式，未证明统计显著的大幅加速。串行观测吞吐约47.94→48.28 curves/h，非多曲线并发吞吐。
- pooled G树21.93575→21.80175 s，fold10.75125→10.70125 s，下降8.3785→8.51425 s；giant14.2095→14.20875 s、finv2.337→2.3205 s。候选非重复分摊约init20.5%、giant19.1%、G树29.2%、fold14.35%、下降11.42%、finv3.11%；其余为G叶、Γ、GCD/命名/准备。归约约1.97 s，嵌套在上述多项式阶段。
- 八次实际输入 digest_words114352490 /sum`033a77303713f3a6` /xor`d16047fb14d39b21`、仿射 Q SHA256`33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`、最终115200叶/8064000 words FNV`10619321735931855904` 一致。403批/1979251对/40218760系数、2400 mandatory自检、66139 GMP samples/4 full checks、8241 carry checked、抽样签名`b9cbd2041266767a`不变，错误0。
- 两组 NVML GPU1 观测峰都5544 MiB；owner只增加16 B，NTT完整缓存payload3486.695 MiB、24 malloc/8 grow/8890 hit、host padding123 MiB与输出pinned123 MiB保持。进程 observed private峰均7602.0→7562.5（日志标MB，实际按MiB取整），−39.5 MiB；数据源为PagefileUsage/PeakPagefileUsage，表示私有提交量及其OS历史峰，不是物理RAM驻留峰。最后记录之后仍可能增长。
- 第一组1Hz NVML full mean busy82.00→85.02%，main85.52→88.44%，≤5% full样本11/150→11/148；第二组见原始汇总。busy不是SM occupancy，阶段边界由计时反推，仅作观察。当前GPU-Z日志也读取，但没有CUDA设备ID，GPU归属以NVML GPU1为准。

另采 Systems2026.1.3：114640 kernels全部GPU1，15次摘要合计.004582046 s。H2D **6556779952 B**，D2H **3067203464 B**，D2D **937993280 B**；与§33 trace相比，分别−914819920、−914819904（包含新增16 B摘要）、+914819920 B，逐字吻合公式。全部GPU copy合计.785863052 s，§33为约.954 s；同步cudaMemcpy host33.385136711 s仍包含大量等待，实际对应GPU copy仅.066474165 s，不是PCIe本身耗时。

本次Stage1后近似事件窗口75.638082906 s，并集63.492047669 s，无本进程GPU事件12.146035237 s（16.06%）。没有观察到对§33单次trace间隙的下降；不同采样运行/主机准备波动使它不能用于宣称交接减少了GPU空闲。该范围也不是准确Stage2计时或整卡idle。A/B的主要确定收益是数据量及主机payload减少，延迟收益较小。

证据：[第一组ABBA](D:/code/MPA-OpenCl/build_cuda_cmake/_root_fold_ab_clean_20261004/results.csv)、[重复ABBA](D:/code/MPA-OpenCl/build_cuda_cmake/_root_fold_ab_repeat_20261004/results.csv)、[八次汇总](D:/code/MPA-OpenCl/build_cuda_cmake/_root_fold_20261004/pooled.json)、[GPU时间线汇总](D:/code/MPA-OpenCl/build_cuda_cmake/_root_fold_nsys_summary_20261004.json)、[35项门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_root_fold_gate_20261004/summary.json)、[摘要门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_root_fold_20261004/digest/summary.json)。ignored测量证据及源码/分析快照保留本地，不随源码提交。


生产 wrapper 默认 `NTT_GROOT_TO_FOLD=1`，显式0恢复原host根交接；实验 exe 仍缺省0。生产重编译 sm89/CUDA13.3成功（CUDA265.2 s），exe2636800 bytes，SHA256 `08df4c392b24c040ce9892c2591a3de0a6f5a8fc7b5bf5eef980c2932300ec8e`。save/ini/worktodo基本21项加CUDA失败队列保留、saved-X因子和M4423实际恢复，共 **24/0**。用户 B2/skip/num 后缀仍精确选择961–970，字面xxx拒绝，队列不变。

生产 M4423 save：init15.231398/main58.984101/total74.215499 s，15 roots/114352490 words全部直接交接，14 folds/42 muls；最终叶FNV`10619321735931855904`同基线，GMP/carry/NTT计数和bad0保持。这是部署验收单次计时，非新的A/B。save的Q=(X:1)与实验Stage1的等价投影Q使原始根输入摘要分别为sum`3cf2f49cf1972d5d`、xor`3fafa10f6f7f6f62`，与实验摘要不同；Γ⁻¹校正后叶值一致。跨入口不能把原始根摘要当作仿射结果摘要。

[生产验收](D:/code/MPA-OpenCl/build_cuda_cmake/_root_fold_20261004/production_accept/summary.json)、[M4423引擎日志](D:/code/MPA-OpenCl/build_cuda_cmake/_root_fold_20261004/production_accept/m4423_engine.log)。

下一轮优先评估 GMP/carry 小读回和同步排队，其后处理下降中的主机准备。不同 sigma 流水需要每曲线独立 Q/Γ/H/模数状态、共享大 NTT scratch 的明确调度和 RAM/VRAM 总预算；当前全局上下文不能直接多线程并发调用。当前 M4423 设备峰约 5.4 GiB，两套完整 arena 的直接复制不适合 8 GiB GPU。此轮仅证明单曲线交接，公平 Prime95 CPU 重跑与实际跨曲线吞吐仍待测。

## 35. 驻留管线的异步 oracle + carry 合并（2026-10-04）

本轮基于 `fcab474`，**未修改 CUDA 树或 NTT 算子**。复用已有两个调度开关，在§34完整驻留组合上测试四种组合：00阻塞oracle/逐中间chunk读carry，10异步oracle，01合并carry，11同时开启。早期§33开发日志的异步结果未证明收益，§49旧carry性能还早于归一化修复；不能直接作为当前默认值依据。这里重新测量真实 ECM 输入。

### 35.1 执行与检查顺序

- `NTT_S4_ORACLE_ASYNC=1`：选样方法、位置、样本数与GMP谓词不变。在对应GPU digit/reduced缓冲被复用前，将两组样本排队D2H到独立pinned环槽并记录事件。后续hook非阻塞查询最老槽，ready后在CPU做GMP比较。环满或最终drain才允许等待；槽重用前必须比较完成，成功返回/销毁前要求selected=compared、pending=0。没有把仍会重写的GPU指针交给异步CPU。
- `NTT_S4_CARRY_BATCH=1`：同一(NTT长度、slice数)的连续interior chunks保留各slice的atomic residual/count和max-bit贡献，合并一次读回。尾块、非deferred块、shape/m变化前以及函数退出先finish，不得先清零或换arena再查。第一与尾chunk原诊断仍保留；实际减少的是读取次数。
- 合并模式不会跳过GMP或carry检查。任何非零 residual、样本误差或最终未验证快照仍失败；两类故意污染均通过门禁。pinned分配失败使用原阻塞捕获，改变速度不改变谓词。

原文件入口：异步开关与ring [stage2_tree_gpu.cu:1834](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1834)，快照消费 [3082](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3082)，最终drain [3105](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3105)，捕获 [3150](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3150)，carry边界 [3609](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3609)，底层finish [ntt_poly_probe.cu:3981](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3981)。

实验CLI仍两个开关默认0；生产wrapper将两者默认1，设置 `NTT_S4_ORACLE_ASYNC=0`、`NTT_S4_CARRY_BATCH=0` 可分别回退；ring缺省4，允许1–8。

### 35.2 计算、内存、传输公式

检查计算量保持：oracle为同一 `C_sample` 个精确整数重建/模N比较，carry为同一批残留诊断kernel与atomic贡献。`T_full` 不能由删除的同步API等待时长直接推算。

记环槽数K、每槽最大digit/reduced word容量d_k/r_k；host pinned oracle容量 `M_oracle=8Σ(d_k+r_k)` B，额外GPU大型缓冲0。近似上界 `8K*c_max*(slot_words_max+W)`，c_max由sample_limit/小批完整检查确定；默认96与K4的本次实测1649664 B（1.573 MiB）。大样本/不同形状应按实际cap计账。raw H2D staging仍以借用事件保护；carry合并模式交替两个operand-pair槽，额外host容量 `16*m_slot1*P_slot1*W`（实际capacity求和而非仅有效字数）。本次129025120→202425440 B，增加73400320 B /70 MiB。它不是新增VRAM，pinned属于RAM/私有提交并影响物理内存压力。

每carry诊断slice为两个uint64。逐interior读回量 `D0=16Σ_chunks m_c`；合并读回量 `D1=16Σ_groups m_g`，同组m固定，保留第一/尾chunk其他读回。M4423 interior chunks8241→group252，覆盖slice数1893402不变，读取slice数1893402→46193，因此D0=30294432、D1=739088 B，净少29555344 B（28.19 MiB）；次数减少7989（96.94%）。oracle仍读相同样本，两次D2H/job由同步API改为异步API，不减少其数据量。

主要operand/系数数据量与§34相同，owner容量580612528 B、host H2D/D2H摘要4.71/1.34 GiB保持；新组合的全trace D2H少的是carry诊断。所有114640 kernel数量及NTT多项式逻辑工作量保持。此轮不是FFT复用优化。

### 35.3 八次矩阵测量与正确性

GPU1 RTX4060 Laptop8GiB，M4423/sigma26/B1=1000/B2=2011326186870/D1231230/P115200/I1633592/G15，extra12/batch64/arena6300MiB，使用§34同exe SHA256 `b70818c167acab6f7ec57b22386ed88ae78bb6d7f5cad6c589fb215683eb7198`。顺序00/10/01/11/11/01/10/00，无编译或其他自启GPU任务重叠。两个样本/组合，控制分别73.148435/72.995641 s，11分别71.741286/71.628758 s；其他组合见CSV。

- 00 full73.072038/main58.6242905/init14.4477475 s。
- 10 full72.3770585/main57.9390495 s：full−0.951%，main−1.169%。
- 01 full72.455481/main58.084886 s：full−0.844%，main−0.920%。
- **11 full71.685022/main57.379686/init14.3053355 s：full−1.898%，main−2.123%，完整曲线少1.387016 s**。同开关也影响F-tree初始化，init改善约.1424 s；不能把main外计时忽略。观测串行49.27→50.22 curves/h，未验证不同sigma并发吞吐，也没有公平Prime95新加速比。
- 11 G树21.5495→20.9175 s、下降8.675→8.1155 s，fold10.7215→10.751 s、giant14.206→14.2335 s；收益主要在G树及下降调度。full候选近似非重复占比init20.0%、giant19.9%、G树29.2%、fold15.0%、下降11.3%，其余finv/Γ/GCD等。归约1.974→1.976 s基本不变，属于嵌套计时。
- oracle host5.728858→1.4520925 s、显式wait5.2785905→0、GMP.296132→.290741 s；pinned增长的t_alloc约1.1156 s，包含该调用可能等待前序GPU，不能当作纯内存分配计算成本。carry total readback50.59946→40.00836 s，其中group17.260625→3.8552035 s；等待有转移，不能把分项节省相加。
- NVML四组合峰全5544MiB，完整arena payload3486.695MiB、24malloc/8grow/8890hit保持。日志OS私有提交峰00均7557.5、11均7590.5 MiB（原标签MB），增加33 MiB，不是物理RAM驻留峰；应同时考虑70 MiB raw pinned增量和oracle环。full1Hz busy87.43→88.30%、main90.03→91.68%，阶段倒推且不是SM occupancy；low≤5%full样本7/144→7/142。GPU-Z日志已读，设备归属采用GPU1 NVML，GPU0外部生产任务未改。

所有8次403批/1979251对/40218760系数/2400mandatory自检/66139GMP/1126job/8241carry覆盖保持，bad0、selected=compared、pending0、fallback0。异步模式queued1126；抽样签名`b9cbd2041266767a`、Q SHA`33cc6c63...809f092f`、根输入sum`033a77303713f3a6`/xor`d16047fb14d39b21`、最终115200叶/8064000word/FNV`10619321735931855904`一致。

新 [组合门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_resident_checks.py:1) **26/0**：65/127/129/8192位四组合，强制chunk2使合并实际发生，ring1/8、window0、blocking output、host fold预算回退，以及oracle最终快照/首个interior carry污染拒绝。CPU F逐系数、全部baby/giant及冻结因子59649589127497217/hit114713对照通过，root/owner独立逐字与GMP检查开启。新runner污染 **27/0**，原root/fold/leaf runner22/14/7均本轮重新通过。既有同引擎188/0证据保留，未声称本轮重跑188项；本轮没有CUDA算子源码变化。

`-Target resident_checks` 固定最新驻留开关，记录8次实际mode_controls与完整矩阵；跨模式保持Q/结果、NTT/GMP/carry覆盖及所有payload账目。新carry统计门禁增加 `checked_chunks <= finishes*max_group`，防止覆盖数与分组上限自相矛盾。

### 35.4 Systems交叉核对

同binary/输入分别采00和11，114640 kernels全GPU1，两次串行完成。H2D6556779952、D2D937993280 B保持；D2H3067203464→3037648120 B，次数14462→6473，差29555344 B/7989次逐字等于carry公式。同步Memcpy调用11330→1089，差10241=7989+2*1126，恰好为carry合并和oracle两组样本改Async。

同步Memcpy host33.429796852→22.556052346 s，而对应GPU copy.067059330→.037021497 s；全部GPU copy总计约.821800912→.818009074 s基本不变。主要收益来自调度/准备及避免小诊断反复排空，不能解释成10.87 s的PCIe加速。

Stage1后近似窗口74.865202016→73.796982751 s、GPU事件并集63.571711602→63.601756454 s，无本进程GPU事件11.293490414→10.195226297 s（15.09%→13.82%，少1.0983 s）。支持间隙减少，但只是两个插桩运行的近似范围，不是准确Stage2计时/整卡idle，墙钟结论采用未插桩8次矩阵。

本地证据：[矩阵CSV](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_ab_20261004/results.csv)、[量化](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_ab_20261004/quantitative.json)、[NVML](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_ab_20261004/gpu1_summary.json)、[26项门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_gate_20261004/summary.json)、[CPU对照日志](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_20261004/accept_all.log)、[两份trace核对](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_20261004/profile_comparison.json)。ignored测量及源码快照保留本地，不随Git提交。

生产wrapper已将两开关默认设为1，sm89/CUDA13.3重编译成功（CUDA245.1 s）。新 `ecm_cuda_stage2.exe` 为2636800 bytes，SHA256 `ecee5b977ca52a38c71e284d55f3179e34674d9da24b85d2709ef8fde376b90f`；编译manifest中引擎SHA仍为 `e89b554d462b2e594ecdb39c394cff76c6fc01186a8856a273993d5ef5ed4f19`、NTT SHA仍为 `66cb4ffe1e5dcd8f765ba9059a29a317980223ba106613b3451644ddee2edd51`。生产入口基本21项加CUDA失败保留队列、saved-X已有因子及实际M4423恢复，共 **24/0**。

生产M4423 save恢复init15.007673/main57.193577/total72.201250 s；oracle selected/queued/compared1126、samples66139、pending0/fallback0，carry checked8241/finishes252/max_group222，根15/114352490 words、fold14/42 muls，最终叶FNV `10619321735931855904`一致。save归一化Q的原始根摘要sum`3cf2f49cf1972d5d`/xor`3fafa10f6f7f6f62`与§34生产入口一致；不要求它等于实验射影Q的原始根摘要。本次单次计时只作为生产接入验收。

[生产验收](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_20261004/production_accept/summary.json)、[编译manifest](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/build_manifest.json)、[M4423日志](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_20261004/production_accept/m4423_engine.log)。

### 35.5 下一算法候选：固定 F 频谱复用（尚未实现）

Prime95 [ecm.cpp:9282](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9282) 对F做polymult_preprocess，reciprocal同理见[9295](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9295)。默认ECMPolyCompress=1为预转置/压缩等预处理；隐藏−1/−2额外设置POLYMULT_PRE_FFT，增加内存。不能把所有默认预处理都称作完整频谱缓存。

当前GPU [ntt_run_passes:3336](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3336) 在[3379](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3379) 对B前向变换，inverse只写A，B可在首次变换后复制到独立cache。先针对fold的第三乘法q·F做固定F缓存：每条曲线独立、精确shape/input版本键、预算及free-VRAM/future-growth预留检查、default stream先拷再允许scratch复用、拒绝/分配失败回退原路径、退出前释放。必须保留GMP/carry采样，验证缓存复用的逐字频谱与污染拒绝；不能以只匹配最终因子替代门禁。

M4423纯shape公式：slot_bits=2S+ceil(log2(P+1))=8863，最大精确bpw19、slot_words467、N_NTT=nextpow2(2(P+1)*467+1)=134217728。因此一个F频谱需8N_NTT=1073741824 B /1 GiB；14个fold第三乘法的首次构建后可省13次F前向NTT。周期收益模型 `(G−2)*C_fwd(N_NTT)−C_D2D(8N_NTT)−C_alloc`，不是实测承诺。当前5.4GiB设备峰不能当作硬上限，缓存需要实际headroom检测。

finv前缀在首次k=P+1与以后k=P可能不同，不能仅凭相同FFT长度直接复用同一频谱；F先行更易满足正确性和1GiB预算。不同sigma还需不同F/cache/Q/Γ，频谱不能跨曲线共用。共享scratch并发仍要限制累计host F-tree/pinned/owner/cache资源；公平CPU重跑、真实跨曲线吞吐和FFT缓存收益仍未证明，长期优化目标未完成。

## 36. 当前瓶颈与候选优化优先级（2026-10-04）

本节在a178184之后分析已有数据，没有修改算法、重编译或启动新的GPU测量。适用负载为§35的M4423/extra12/B1=1000/B2=2011326186870/D1231230/P115200/G15，GPU1、完整驻留组合、oracle_async/carry_batch均1。通用N、不同B2/D及多曲线需要分别测量。

### 36.1 瓶颈判定：NTT、点算术与主机间隙并存

八次矩阵候选均值total71.685022 s，init14.3053355/main57.379686 s。墙钟阶段约为G树20.9175、giant14.2335、fold10.751、下降8.1155 s；这些阶段各自混合GPU执行、主机准备和等待，不能称为纯CPU/GPU成本。

重新读取§35候选Nsight原始SQLite，排除Stage1的ladder-chain kernel后，得到近似Stage2窗口的kernel累计 **61.295378169 s**：

- NTT变换：tile15.887600971、outer-forward9.514640460、outer-inverse4.798495325，合计 **30.200736756 s**，约占该窗口kernel总和49.27%。
- 点生成：ladder19.790310972、giant-chain3.026996376，合计 **22.817307348 s**，约37.23%。ladder包含baby和giant seeds，不能全记到giant阶段；日志例子baby ladder8.962 s。
- carry-cone2.653462729、residual1.081497368，合计 **3.734960097 s**；S4模N归约 **1.943791090 s**。
- gather/pack/scatter合计 **1.892796196 s**；其他kernel约.705787 s。另有memset18493次、427586615000 B、**1.488726245 s**；memset不在上述kernel总和内。

全部trace拷贝约 **.818009 s**（含Stage1少量拷贝），同步Memcpy host22.556 s的大部分是前序GPU工作等待。把这22.556 s全部当PCIe带宽成本，会严重误判。当前Mersenne路径归约仅约1.94 s，早期一般归约54 s的排序已失效；其他模数仍可能重新成为归约瓶颈。

近似Stage1后GPU事件窗口73.796983 s，并集63.601756 s，约 **10.195226 s /13.82%** 未记录本进程GPU事件。它混合CPU计算、分配、提交、依赖边界和工具开销，不能全部称为可删除的CPU工作。若该窗口GPU工作完全不变且理想消除全部空档，才得到最多13.82%的窗口时间减少；这个条件模型不是准确Stage2加速承诺。

仅有Systems时间线不能判定NTT内部究竟由DRAM、64位整数乘法依赖、shared bank conflict、barrier还是寄存器/驻留限制主导。既有NCU2026.2.1尝试报ERR_NVGPUCTRPERM，未得到有效计数器，本节没有声称已有roofline/occupancy证据。

[原始trace](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_nsys_20261004.sqlite)、[本次只读复算](D:/code/MPA-OpenCl/build_cuda_cmake/_optimization_candidates_20261004/analysis.json)、[矩阵均值](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_ab_20261004/quantitative.json)。kernel总和、事件并集与未插桩wall分属不同口径，百分比不能互相直接相加。

### 36.2 优先候选A：减少点生成的依赖模乘

当前 [s2g_xadd:769](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:769) 先算XpXq、ZpZq、XpZq、ZpXq，再两次平方、两次乘差点坐标，共 **8次Montgomery乘法/xADD**；[xdbl:726](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:726) 为5次。[ladder:809](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:809) 每个后续bit一次xADD+一次xDBL，主要为13次模乘；[Mont乘法:682](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:682) 的schoolbook和REDC各有W²次MAC。

可构造A=Xp+Zp、B=Xp−Zp、C=Xq+Zq、D=Xq−Zq，计算u=D*A、v=C*B，然后X'=Zd*(u+v)²、Z'=Xd*(u−v)²。该形式为 **6次Mont乘法**，xADD算术调用少25%，ladder主循环13→11、少15.38%；以2W²计每次Mont主MAC，xADD约16W²→12W²。它增加O(W)加减，对W70通常值得考察，但不保证kernel同比加速。

标准新公式的两坐标共同多一个4的射影尺度。若在平方前对u±v做精确modular-half（N奇数，奇值先加N再右移，正确保留最高carry），可恢复原公式的坐标尺度，仍不用额外大模乘。无论选择保留尺度还是依赖Γ校正，都必须覆盖alias、canonical limb、非单位元/退化链、普通域与Mont image，以及最终Γ校正叶逐项一致。不能只比较最终因子。Prime95 [ell_add_xz_scr:2494](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2494) 使用融合mulmulsub及FFT复用，不能直接把其FFT数当CUDA的8次SOS/REDC成本。

第二步候选是专用Montgomery平方（利用对称项），再考虑CIOS/分块Comba或warp协作limb。平方的schoolbook主乘积理论由W²项变为W(W+1)/2，REDC仍有W²，需处理双交叉项与carry。当前[一线程一点:839](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:839) 有多个大local数组，warp协作可能减少单线程依赖/溢出，也可能因shuffle和减少并行点数变慢。应先做6-Mont xADD这个更小的改变，再决定是否重写乘法后端。

可优化kernel池为22.82 s；若这个池整体快10%，理想少约2.28 s（占71.69 s的3.18%）。这只用于排序，不是实际收益预测。新公式几乎不增加持久显存，优先级高于为约1%收益占用1GiB频谱。

### 36.3 优先候选B：重新校准D和chain/batch参数

[D模型:10813](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10813) 仍用旧RP51840/RIM3400110、RLOOP265.756、RTREE50.537、RGIANT50.655、RGLUE30.339、RSETUP25的系数。tree约974.9 us/leaf、giant14.90 us/step；新代码已做Mersenne归约、设备叶、fold/root驻留及调度优化，旧模型不再代表当前阶段比例。

建议用当前路径拟合 `T(D)=T_baby(P,W)+T_F(P,W)+T_finv(P,W)+(G−1)*T_fold(P,W)+T_Gtree(I,P,W)+T_giant(I,W,L)+T_descent(P,W)+T_host`，I≈B2/D、P=φ(D)/2、G=ceil(I/P)，L为chain block。NTT长度的2幂跳变、batch拆分、短尾和缓存可用性必须进入模型；仅重缩一个旧系数不够。对内存可行的少数D邻居再实测，比较同一真实Q和素数范围的完整曲线/curves-hour，不比较不同NTT工作量的旧分项。

[chain block:9744](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9744) 默认64；每个chunk约2ceil(n/L)+1个ladder seeds。增大L降低seed成本，却增加串行链、减少链线程数并扩大退化传播窗口。不能只按seed数选128/256。batch64也需按RAM/VRAM headroom和小shape拆块开销评估；不能只扩大arena。D与F缓存存在竞争：更大D可能让NTT_N翻倍，同时把1GiB cache变为2GiB而失去可行性。

### 36.4 优先候选C：NTT kernel、打包与carry的实际工作

- **Tile/outer按shape调优。** [tile:1219](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1219) 当前512线程、缺省t12（32KiB shared）、radix4；[outer:939](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:939) 256线程、M上限4，[planner:1452](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1452) 固定环境参数。按NTT_N/nbatch区分t、block、radix与尾pass组合，考察warp shuffle完成tile低层、减少shared往返/barrier、提高独立整数指令并行度。旧提高M到5失败不能证明每个新shape都最优，但禁止无测量全局改大。NTT池30.20 s若整体快10%/20%，理想省3.02/6.04 s，约full4.21%/8.43%，是高收益池。
- **Pack融合清零。** [s4_pack_final_input:3356](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3356) 每次先清A/B，再写有效digits；pack/gather/scatter1.89 s，memset约1.49 s。可以构造覆盖全部NTT位置、有效位直接生成/其余显式写0的packer，减少双写流量；需保留slot空隙、尾部、短operand、device gather和scratch复用合同。总memset1.49 s是该方向的宽松上界，融合未必快，不能直接把全部pack与memset时间算作可省。
- **Carry融合诊断。** [carry_cone:865](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:865) 约2.65 s，[residual:555](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:555) 约1.08 s。研究在carry生成同一最终digit时同时归约residual/max-bit，省一次全数组读取和8898次launch。必须覆盖全N，包括输出窗口外的digits，保留原atomic累计语义及长all-mask lookback；不能以输出槽检查取代全digit诊断。此方向先于大规模重写进位算法。
- **低层专用乘法。** monic baby叶的degree1×degree1仅需一次大模乘和加法，可为F树首层实现独立kernel；G projective叶非monic，通常需4次大模乘，不能沿用同一收益。先比较shape临界点，再考虑少数低degree schoolbook，避免为小多项式做大Kronecker NTT。更多高层middle/low product/TFT可减少只需要窗口的卷积工作，但NTT、carry和modN协议都要改变，成本较高，放后续。

### 36.5 可用候选D：固定F频谱缓存，收益需降到合理预期

§35.5的1GiB/13次省变换公式保持。此次从原trace按当前t12、N_NTT=2²⁷、单slice，抽取46次full-length前向变换（4个outer+1个forward tile），kernel平均 **.067871282 s/次**；这些是同shape各种operand，尚无F专用tag。

由此 `13*C_fwd≈.882326660 s`，约full **1.23%**，还未扣首次1GiB D2D、分配、频谱检查及挤出其他缓存的损失。无需承诺5–10%的整曲线收益；应增加F/finv/G/H的NVTX标记确认。skip F pack/清零还有少量附带收益，未单独测量。源码变换位于[3303/3304](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3378)，fold乘法位于[7654](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7654)。

当前峰5544MiB加1024MiB约6568MiB，只是容量尺度，不是未来峰保证。独立owner、shape/input版本键、live-VRAM和arena未来增长预留、allocation fallback仍必需。finv前缀首次P+1与之后P不同，需分别建立对应谱或证明重用等价；默认不增加第二个1GiB缓存。F缓存可作为较局部、低算法风险的下一提交，但不应排除收益可能更大的点公式和D调优。

### 36.6 候选E：CPU准备/提交优化和共享scratch流水

1. **扁平F-tree + 设备下降frontier。** [build_tree_flat:3937](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3937) 保存host heap，[descent_scaled:7814](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7814) 每层创建vector/map、复制父state和反转sibling、上传、读回、拆分next。先用连续host层数组、预计算shape/offset、复用pinned容量，再将下降state保存在device双frontier，仅上传对应F层、最后读回P个叶。总下降约8.12 s包含真实NTT，不能全部当作CPU拷贝收益；应独立记录host pack、H2D、D2H和GPU math。F-tree全部搬GPU是后续容量决策，当前全层逻辑系数约8W[P(log2(Fpad)+1)+2Fpad]≈1.218GiB，另有metadata/capacity，不应把7.6GiB进程私有峰全当F-tree。
2. **Baby归一化GPU分工。** [host prefix/inverse:11221](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11221) 当前每256点一逆元，单次日志affine约2.027 s。可让GPU做prefix/suffix与乘逆元，CPU只逆每组总积并处理非单位元；保存small-prime的unit证明与坏组回退。2 s是现有CPU路径规模，GPU新增多精度工作及传输可能抵消收益；优先连续内存/组复用，再决定迁移。
3. **Event/pinned池 + 小shape CUDA Graph。** trace有114640 launches（host1.421 s）、62287 event create/62291 destroy（合计约.198 s）、68809 record（.617 s）。先复用事件与预分配有限pinned槽，避免中途grow/free引起排空；再对固定shape的pack→NTT→carry→reduce/scatter构造graph，检查和ring接续放在明确边界。NVIDIA文档说明Graph可降低CPU提交成本；这里尚无端到端收益。[CUDA Graphs官方说明](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。不能把所有API等待相加成可省时间，也不能靠禁用mandatory/GMP/carry减少launch。
4. **删除已由stream保证的中间排空，并预准备下一块。** [giant chain:9130](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9130) chain后和segprod后分别DeviceSynchronize；第一处可研究靠同stream依赖排队segprod，同时CPU先准备下一G树metadata。需要CPU观察Γ/gcd的最后边界仍须等待；trace56次DeviceSynchronize/31.92 s多在等待点算术，不能宣称删除调用就省31.92 s。
5. **不同真实曲线共用scratch流水。** 目标是curves/hour：一条GPU执行时准备另一条CPU数据。共享NTT arena/GPU scratch、每曲线独立Q/a24/N/Γ/F/H/cache/诊断状态；当前可变全局先迁入context。8GiB卡无法直接并发两套5544MiB进程。host约7.6GiB私有提交峰也不能粗暴复制；限制活跃曲线数、host树/pinned、device owner/cache，测量总private及物理RAM/VRAM。固定同Q的--curves K不算真实不同曲线吞吐证据。

### 36.7 推荐推进顺序与完成条件

先补足已有trace的shape/阶段/operand标记和CPU准备计账，以免误读等待；首批并列为 **6-Mont xADD、小范围D/chain参数重新校准、固定F缓存**。其中F缓存实现更局部，但收益尺度约1%；xADD和D有更大潜力。第二批是tile/outer按shape优化、清零/pack融合、carry诊断融合；第三批为连续F层/设备下降，最后共享scratch多曲线和middle-product算法。

每个实现必须同二进制开关、相同真实Q/B1/B2、完整wall含初始化/最终验证、串行交叉测试，并记录检查覆盖、传输与RAM/VRAM预算。D或点射影尺度变化时，原始根/样本数可能依算法变化，需用数学覆盖与独立逐项结果替代错误的旧指纹合同；保留可逆性/坏组/GCD=N、尾块与故意污染拒绝。当前没有有效NCU计数器，不把某个NTT内核预判为纯带宽或纯算力瓶颈，也没有用新分析更新性能结论。

## 37. CUDA Core 开源 NTT 适配分析与实施顺序（2026-10-04）

按用户的新要求，本轮优先推进NTT，暂不使用Tensor Core。§36的全局候选仍保留。本节先核对 `gpu_ntt_tensor_core_survey.md` 所列三个CUDA Core项目，引用固定Git提交，避免把其他素域或硬件的论文加速比当成本机收益。参考源码保存在ignored目录 `.refactor/ntt_sources_20261004`，版本清单为其中的 `sources.json`；没有构建或引入这些库。

### 37.1 适配合同

本节访存公式中的N专指NTT长度；ECM模数记为N_ecm。所列字节数均为单slice，nbatch条slice须再乘nbatch；不包含twiddle、事务粒度、spill或缓存重复取数。

本仓库使用Goldilocks p=2^64−2^32+1、primitive root7、普通域canonical uint64。forward DIF自然序→bit-reversed，inverse DIT bit-reversed→自然序，pointwise与1/N在inverse tile中融合；因此无需单独bit reversal。Kronecker精确性仍要求 `L*(2^bpw−1)^2<p`，不能仅将模数换成GPU-NTT支持的较小素数。批处理slice stride、只读B、覆盖整个N的carry诊断及oracle scratch生命周期同样必须保留。

### 37.2 参考实现逐项核对

**sppark：第一参考。** 固定提交 `9e5c7951d4ff4992f78af26f48d3c9230b8c4136`，Apache-2.0。

- [ff/gl64_t.cuh:47](https://github.com/supranational/sppark/blob/9e5c7951d4ff4992f78af26f48d3c9230b8c4136/ff/gl64_t.cuh#L47) 明确支持相同Goldilocks模数；to/from在389–394是canonical/partially-reduced选择，不应因函数名误称Montgomery转换。不能直接把其允许冗余表示的运算拷进本仓库canonical合同。
- [GS narrow:19](https://github.com/supranational/sppark/blob/9e5c7951d4ff4992f78af26f48d3c9230b8c4136/ntt/kernels/gs_mixed_radix_narrow.cu#L19) 使用每线程两个值及z_count批次的寄存器数组；高层跨warp依赖用shared交换，低层 [89](https://github.com/supranational/sppark/blob/9e5c7951d4ff4992f78af26f48d3c9230b8c4136/ntt/kernels/gs_mixed_radix_narrow.cu#L89) 通过shuffle交换。CT narrow对应反向过程；coalesced load/store和warp转置负责中间pass布局。
- [ntt.cuh:104](https://github.com/supranational/sppark/blob/9e5c7951d4ff4992f78af26f48d3c9230b8c4136/ntt/ntt.cuh#L104) 按logN分1/2/3/4个pass：logN27采用9+9+9，共3个；本仓库t12/M4为4个outer+1个tile，共5个。这是减少global pass的候选，需适配不同索引、twiddle、寄存器数量、batch及合并pointwise，尚无本机比较。

**GPU-NTT：参考merge与4-Step访存组织。** 固定提交 `d03c5eaeadaa780d153496afcb3b6a9b79a13a63`，Apache-2.0。

- [modular_arith.cuh:64](https://github.com/Alisah-Ozcan/GPU-NTT/blob/d03c5eaeadaa780d153496afcb3b6a9b79a13a63/src/include/gpuntt/common/modular_arith.cuh#L64) 与176起的GPU版明确限制Data32模数≤30bit、Data64≤62bit；其plain addition及Barrett前提不适用于64bit Goldilocks。移植的是布局与融合方法，算术继续用现有gl_add/sub/mul。
- [merge ntt.cu:12](https://github.com/Alisah-Ozcan/GPU-NTT/blob/d03c5eaeadaa780d153496afcb3b6a9b79a13a63/src/lib/ntt_merge/ntt.cu#L12) 在CTA内shared完成多层蝶形，避免每层global读写；本仓库已有类似tile融合，直接照搬不构成新的收益。
- Merge host dispatch [2088](https://github.com/Alisah-Ozcan/GPU-NTT/blob/d03c5eaeadaa780d153496afcb3b6a9b79a13a63/src/lib/ntt_merge/ntt.cu#L2088) 检查0<logN<29，并以25区分standard/large kernel；其尺寸覆盖2^27，主要适配障碍仍是模数算术。换用≤62bit素数需要重新推导bpw/精确性、根表、长度和buffer计划，属于新的backend工作。
- [4-Step Transpose_Batch:37](https://github.com/Alisah-Ozcan/GPU-NTT/blob/d03c5eaeadaa780d153496afcb3b6a9b79a13a63/src/lib/ntt_4step/ntt_4step.cu#L37) 是独立16×16转置kernel；行/列NTT改善长跨距访问，但完整转置每次多16N字节流量，单独输出buffer多8N字节。N=2^27时分别为2GiB流量和1GiB容量尺度，需计算转置次数、是否可融合及同时活跃buffer，不能只看NTT pass数。
- 当前4-Step host dispatch在 [2509–2532](https://github.com/Alisah-Ozcan/GPU-NTT/blob/d03c5eaeadaa780d153496afcb3b6a9b79a13a63/src/lib/ntt_4step/ntt_4step.cu#L2509) 以case24结束，之后报ring size unsupported；其他forward/inverse及RNS分支也以24结束。因此本仓库N2^27还需要扩展其分解/调度，不能把现有库当作该长度的即用backend。

**ICICLE：优先参考接口和资源管理。** 固定提交 `625532a624e5aaa6e9d31a1c92587f1fcc30dc76`。当前公开树有通用CPU NTT与cuda_pqc小模数代码，普通CUDA backend通过外部安装；本次没有找到可直接研究的公开通用Goldilocks CUDA NTT。 [README:318](https://github.com/ingonyama-zk/icicle/blob/625532a624e5aaa6e9d31a1c92587f1fcc30dc76/README.md#L318) 明确：frontend MIT，未在该树公开的backend另有special license。因此调研中“MIT开源工程级GPU实现”需要按具体backend/版本限定。ordering、batch、device input/output、stream与async配置值得参考，但不是本轮内核移植依据。

### 37.3 NTT推进优先级与收益模型

1. **P1：warp内完成tile低层。** 保留shared radix4高层，低6层通过每lane两个寄存器值与shuffle完成（t5只做5层），减少shared往返和CTA barrier，分离编译baseline/candidate kernel，以 `NTT_FUSE_WARP_TAIL=0/1` 做同二进制对照。t<5退回原路径；不增加持久显存、global pass或传输。实现不复制外部源码，只参考warp通信方法。本阶段必须先过独立GMP DFT、forward顺序、inverse及batch/边界门禁，再测整曲线；shuffle/分支/寄存器压力可能抵消收益。
2. **P2：宽mixed-radix outer pass。** 参考sppark跨warp交换与coalesced重排，把本仓库每线程2^M寄存器爆炸改为寄存器+warp/shared协作；目标5→3/4 pass。每个64bit全数组读写pass为16N字节，单次卷积3个变换理论少 `3*16N*Δpasses`，N2^27且Δpasses2为12GiB；实际还含twiddle及pointwise差异。旧单纯M5回退不能用作新协作设计的结果。
3. **P3：按shape自动调参。** 对N/batch区分tile t、block、outer组合、缓存与短尾，限制shared/register/arena容量。先用少数代表shape校准，生产采用确定性表；不能只按最少pass或最大tile选优。接续P1/P2测量，避免一次混入多变量。
4. **P4：4-Step/转置融合原型。** 仅在宽pass受限或大N收益可覆盖转置成本时推进，保留8GiB headroom。其次twiddle分解/on-the-fly及有限域指令调优；目前table已按shape缓存，不优先以额外模乘换掉所有查表。

当前tile池15.8876 s，全部NTT池30.2007 s，完整曲线约71.685 s。P1若tile快10%/20%，理想整曲线省1.59/3.18 s（2.22%/4.43%）；P2若整体NTT快10%/20%，理想省3.02/6.04 s（4.21%/8.43%）。这是用于排序的条件模型，不是开源库或新代码实测。算术主项仍为每变换N*log2(N)/2蝶形；GPU周期数取决于Goldilocks指令、依赖与驻留，缺有效NCU计数器时不提供虚构周期常数。

### 37.4 第一项实施与验收记录

先完成以上分析，再修改P1。实际结果、是否启用生产默认及后续调整在此追加；其余候选按依赖顺序推进。

源码入口：[warp pair:1197](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1197)、[tile dispatch:1219](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1219)、[forward:1610](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1610)、[GMP fixture:4188](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:4188)、[独立门禁脚本](D:/code/MPA-OpenCl/tools/test/test_stage2_ntt_warp.py:1)。默认实验开关为0，t≥6以每lane两个寄存器完成64点子变换，t5用32点版本，t<5回退。DIF每层shuffle交换pair索引位与lo/hi寄存器位，输出变成相邻pair；DIT反转这些置换。外部bit-reversed顺序、根表和乘法域保持一致。inverse permutation写shared前用warp同步，后续高层前用CTA同步；arena在每次命中时为当前kernel维护足够的dynamic shared容量上限（最终收敛见§37.5）。

t12阶段CTA barrier forward6→3、inverse6→4（初次载入及pointwise的barrier不计）。按源码读写次数，forward少48N字节shared payload，inverse少32N；两次forward+一次inverse共少128N。global pass和传输payload不变，持久显存新增0，tile仍32KiB shared。最初one-word-per-lane原型先过8/8与188/188门禁；复核发现半warp模乘，改为full-butterfly-per-lane后再过8/8与188/188。原型没有用于收益对照。

warp-only阶段的独立门禁每个开关覆盖216组合/6854400结果字：t4/5/6/7/8/11/12、M1..4、compact/wide scratch、uncached/cached/evicted、nbatch3，各slice的pointwise乘数为1/2/3；GMP分别验证全部前向频谱和逆变换，并检查生命周期无泄漏。另一次完整Stage2门禁188/0，包含冻结因子、普通模数、不同shape、packing/alias/尾块及故障拒绝。该阶段证据在 `build_cuda_cmake/_ntt_warp_20261004/pair_gate/summary.json` 与 `pair_full_gate.log`；最终常量根版本的重新验证见§37.6。

性能runner新增 `-Target ntt_warp`，固定当前完整驻留组合及oracle/carry=1，ABBA只切换warp flag。初版把NTT_FUSE_TRACE设为PowerShell $null，实际生成空值；旧C++trace开关按存在性启用，导致逐kernel同步。该轮前两次及中断的第三次均作废，存于 `_ntt_warp_ab_20261004`；已改为显式Remove-Item并在启动前断言不存在、结束后拒绝任何`[trace]`日志。有效测量使用独立目录 `_ntt_warp_ab_clean_20261004`，旧§35测量不受这次新增runner配置错误影响。

warp-only有效ABBA：full73.316171→73.063244 s（−.345%），main57.753206→57.498258 s（−.441%），仅两次/模式且B样本跨度.789 s，不能认定稳定收益。相同新binary的两次Systems profile排除Stage1后，tile15.821303→15.576781 s（−1.546%），NTT30.115274→29.855848 s；全部后Stage1kernel61.174145→60.887044 s。profile分项与未插桩wall分开，不相减拼成新的CPU时间。两侧26694个tile均为对应bool模板，检查/工作量/根叶摘要一致，GPU1峰5544MiB未增加。

cuobjdump静态资源：warp-only的forward/inverse均REG40、STACK/LOCAL0；baseline forward40/inverse36，dynamic shared仍按t12申请32KiB。它没有寄存器spill证据；静态资源不等于实测occupancy，也不能据此判定所有barrier已被隐藏。warp-only保留实验参考，未启用生产默认。[有效量化](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_ab_clean_20261004/quantitative.json)、[同binary profile](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/profile_comparison.json)。

### 37.5 P1继续：单位根与四次单位根的通用模乘消除

此处stage0/1指NTT蝶形的低两层，与ECM Stage1无关。

sppark的 [GS末层:114](https://github.com/supranational/sppark/blob/9e5c7951d4ff4992f78af26f48d3c9230b8c4136/ntt/kernels/gs_mixed_radix_narrow.cu#L114) 直接加减，不乘单位根。本轮先补这个低成本改动，再推进P2的宽pass。warp stage0的根只有1，直接返回；stage1的根为1或±2^48，后者用128bit乘积的lo=`v<<48`、hi=`v>>16`送入现有gl_reduce，再按根的正负取canonical符号。其他stage和未识别根仍用通用gl_mul，不改变素域、根顺序或精确性界。

因为令x=2^32，则Goldilocks中x²−x+1=0，所以x³=−1、(2^48)²=−1；这证明±2^48是四次单位根。t≥6、单slice tile通用模乘调用由两forward加inverse/pointwise的 `N*(3t/2+2)` 降到 `N*(3t/2−1)`，t12为20N→17N（少15%）；stage1非单位根仍做移位约减，蝶形加减数与pointwise/1/N不变。这是源码调用数，不是GPU周期或端到端收益。

未加编译约束的常量根版本cuobjdump显示inverse REG46（forward40），仍无LOCAL/STACK；尚未用于性能结论。随后给warp模板加512线程/3 CTA目标的launch bounds，baseline模板用1 CTA目标，核对最终资源及实际容量API结果再测量。launch bounds是编译寄存器约束提示，不保证实际驻留，参见 [NVIDIA CUDA语言扩展说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#launch-bounds)。动态shared仍32KiB，其他架构/tile参数须重新评估。

同时把CUDA函数属性设置限定到当前选中的两个kernel，并在每次arena命中按该arena的最大shared容量重新pin；baseline恢复原有API调用数。新增同一arena中模式0/1/0/1和大/小/大tile12/5/12的GMP检查，用来覆盖kernel选择刷新、缓存命中和函数全局属性的边界。它增加4次forward/inverse调用及98304比较字，独立于原216组合的计数。

### 37.6 P1最终结果：warp、常量根和容量约束

最终源码入口：[常量根约减:1160](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1160)、[warp pair:1197](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1197)、[tile:1219](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1219)、[属性选择:1306](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1306)、[arena容量:2154](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2154)。实验默认0，生产 [wrapper:6](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6) 默认1；显式 `NTT_FUSE_WARP_TAIL=0` 保留原算法路径。生产默认决策依据是本节sm89测量，其他架构和工作负载尚未验证收益。

最终cuobjdump资源：baseline forward/inverse REG40/36，candidate40/40，四个kernel的STACK/LOCAL均0。512线程、32KiB动态shared时，CUDA occupancy容量API返回candidate每SM最多3个CTA；这是资源允许值，不是实测驻留。未增加算法持久buffer，arena/twiddle表布局及global pass数保持。相比warp-only binary，baseline inverse SASS相同；forward均552行指令但寄存器分配/调度文本有变化，所以不声称旧binary与最终baseline机器码完全相同。墙钟比较均使用最终同一binary。

**正确性重新验证：** [最终独立门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/final_gate/summary.json) **12项通过/0失败**，分别在0/1下检查216组合/6854400字的GMP前向DFT与逆变换、tile边界、三slice、缓存模式切换、shared容量、无泄漏及LOCAL0；每模式另有4次/98304字的缓存切换比较。[完整Stage2门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/final_full_gate.log) 本轮重跑 **188/0**。冻结因子59649589127497217、普通模数、packing/alias/尾块及故意故障拒绝通过。

**性能：** GPU1 RTX4060 Laptop8GiB，M4423/sigma26/extra12，B1=1000、B2=2011326186870、D1231230、P115200、I1633592、G15，batch64MiB/arena6300MiB。固定完整驻留组合与oracle/carry=1，串行ABBA再BAAB，共8次、每模式4次，只有warp flag改变，无trace。最终实验exe SHA256为 `a7b712cb11aeb2c781fcfed2a4daca6f654d2c67111cc763b08df24c56ff8f37`。

- 完整Stage2墙钟含初始化与最终检查：**73.1216075→72.52802925 s，少.59357825 s / .81177%**；main57.4541215→56.922694 s，少.5314275 s / .92496%；init15.66748575→15.6053355 s。
- baseline四次full为73.044634/73.074201/73.145732/73.221863 s；candidate为72.376652/72.329133/72.364111/73.042221 s。ABBA均值改善约.97%，BAAB约.66%；最后candidate接近baseline。样本数少、存在波动，没有置信区间，不外推为所有曲线稳定加速。
- G树20.94775→20.5005、fold10.61725→10.46775、下降7.87025→7.76475 s；归约1.97575→1.975 s基本不变。阶段计时嵌套，不能将所有分项节省累加。
- 两次独立Systems采样：tile **15.853348237→14.968390207 s（−5.58%）**，全部NTT **30.120484455→29.247851644 s（−2.90%）**；排除Stage1 chain后的kernel累计61.197432119→60.276680704 s。每侧26694个tile为相应模板，全部114640个kernel均GPU1。这些插桩kernel和不是8次未插桩wall，未获得有效NCU计数器。

所有8次保持403批NTT、1979251对乘法、40218760归约系数、2400 mandatory自检、66139 GMP样本/1126 oracle jobs/4 full checks；carry8241块/252 finishes/1893402 slices/max_group222。bad0、pending0、fallback0、overflow0、clean1，oracle签名 `b9cbd2041266767a`，根mixsum `033a77303713f3a6`/xor `d16047fb14d39b21`，最终115200叶/8064000字/FNV `10619321735931855904` 一致。Q SHA256 `33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`。

访存/容量合同保持：host H2D/D2H摘要4.71/1.34GiB；Systems精确H2D6556779952、D2H3037648120、D2D937993280 B及次数保持。18493次memset/427586615000 B保持。NVML GPU1峰均5544MiB、arena payload3486.695MiB、private提交峰约7.6GiB；private不是物理RAM驻留。P1仅优化tile内部shared/通信和有限域工作，没有消除主机准备或PCIe边界。

证据：[8次量化与完整样本](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/final_quantitative.json)、[ABBA](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_const_final_ab_20261004/results.csv)、[BAAB](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_const_final_baab_20261004/results.csv)、[同binary profile](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/final_profile_comparison.json)、[baseline SASS核对](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/baseline_sass_comparison.json)。源码NTT SHA256 `f7c98f0bed2b00661f480f40b447098a0947654b53a5f31b794ebe5bf3d7acd7`，Stage2树引擎仍 `e89b554d462b2e594ecdb39c394cff76c6fc01186a8856a273993d5ef5ed4f19`。

**生产接入：** sm89/CUDA13.3编译成功，CUDA252.8 s；工作区exe2691072 bytes，SHA256 `905826a8988e2046831e4ecd57406c46a6d8b0db2ac22551b2d6be03d18b588c`。[生产验收](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/production_accept/summary.json) **26/0**：基础21项、CUDA失败队列保留、saved-X因子、M4423实际存档恢复、新默认1及显式0回退。用户示例仍选择961–970，字面xxx拒绝并保留队列。实际save恢复init15.776226/main57.036366/total72.812592 s，最终叶同上，oracle/carry全部验证；这是接入验收单次计时。save Q=(X:1)的原始根摘要与实验投影Q不同，仍与旧生产摘要一致。编译只更新工作区生产输出，GPU0外部生产运行未改。

### 37.7 接续P2：减少全数组pass的实施边界

下一项按§37.3顺序做协作式outer，先保留P1与原outer作为控制；尚未实现P2，5→3/4只是目标。

1. 先明确跨lane/warp的输入、每层twiddle指数及输出置换，参考sppark narrow/wide的合并访问。以每lane少数值、shared跨warp交换处理6–9层，不能直接将现有每线程2^M数组的M从4调大。
2. 单独证明自然序→bit-reversed与反向合同；覆盖中间短pass、t/outer边界、nbatch/stride、B只读和inverse pointwise/1/N融合。先小shape的独立GMP频谱逐字比较，再现有Stage2/generic N/故障门禁。
3. planner按实际寄存器、shared与合并访问安排pass；原t12/M4在k27是4 outer+1 tile，候选可先试2 outer+1 tile或3 outer+1 tile。不新增全长8N scratch是优先约束，任何新增表/buffer须进入arena预算与eviction生命周期。
4. 分别测outer kernel池、完整NTT、初始化和整曲线；仍采用同binary串行交叉开关、相同Q与检查覆盖。两pass节省12GiB/单slice卷积是payload公式，不保证kernel或wall同比改善。通过后再推进P3确定性shape表，最后考虑P4转置。


## 38. xADD6 与驻留流水线自动选 D 重标定（2026-10-04）

本轮按点运算→D→NTT的顺序推进。六模乘xADD恢复原projective scale，same-binary串行ABBA+BAAB共8次：完整Stage2均值72.830970→68.270377 s（−6.26%），giant14.21325→11.78925 s（−17.05%）。P/Q/Γ与403批NTT、40218760归约系数、66139 GMP样本、最终叶摘要保持；GPU1峰5535MiB和传输量不变。原语门禁67/0及完整Stage2门禁188/0通过。

D按真实NTT长度、partial tree/Newton工作及驻留owner预算重拟合。大界选D1381380/P126720，两次64.743013/62.405344 s；小界独立holdout选D330330，13.966437/13.997466 s。绝对秒数误差约5–7%，限于已测RTX4060 Laptop/M4423/B1=1000/当前resident/check配置，其他scope回原模型。20项规划检查通过，CPU选D约.13 s，单列于d_scan_wall。D重选不是固定工作量的xADD A/B，叶数及采样随D改变。

详细推导、MAC/容量/传输公式、经验系数、来源行号、数据及生产验收见 [专门报告](D:/code/MPA-OpenCl/docs/STAGE2_XADD_D_OPTIMIZATION.md)。生产`NTT_XADD6`/`NTT_D_MODEL`缺省1，显式0可以分别回退；新D模型仍受scope限制。

## 39. 协作式 NTT outer 实验进度（2026-10-04）

已实现CTA shared radix、连续coarse offset访问的5–8层outer，原DIF/DIT顺序与pointwise/scale合同保持。N2²⁷/t12可由4outer+tile降到2outer+tile，无新增全长scratch。新实验开关NTT_FUSE_COOP_OUTER缺省0，NTT_FUSE_COOP_M缺省8；t<5自动沿用原planner。D经验模型在coop开启时回退，避免用旧NTT权重选择新流水线。

独立GMP前向频谱/逆向96组、27131904字，以及4次缓存0/1/0/1切换3145728字通过；8项门禁通过，故意损坏被拒绝。v1的M5..8共8个template LOCAL0、REG45..48，M7/M8实际容量API只允许2CTA/SM，M5/M6为5CTA/SM。旧路径216组/6854400字在warp0/1各复测一次、14次生命周期/leaked0均通过。宽pass的纯NTT和Stage2结果记录继续见专门报告；尚未作为生产默认。


v1纯NTT M8慢15.86%，未提升默认；v2复用twiddle/coarse power并修复shared根发布时序，N2²⁷/M8纯卷积.198026→.184210 s（6.98%）。最终同生产exe/save四条ABBA full69.173718→67.363070 s（2.62%），检查和根/叶摘要一致；每模式2样本，未建立置信度。v2 REG45..53、LOCAL/STACK0，独立NTT门禁8/0及原warp0/1两套容量/生命周期通过。最终生产F85EAD72…F7F1、4038144 bytes，save/ini/worktodo30/0加cooperative实际恢复通过；xADD6与D模型缺省1、cooperative缺省0。完整公式/样本/源码行号见专门报告§6。接续P3按N选择M、重新标定D，再评估Tensor Core及多曲线吞吐。


## 40. 按 NTT 尺寸选 outer 与 D 重新标定（2026-10-04）

按NTT长度选择outer：RTX4060 Laptop/sm89/t12/warp/compact/M4 scope，k24用M6、k25..27用M8，其他shape回原planner。模式2不在线搜索；模式0回退，模式1保留强制实验。八次串行ABBA+BAAB固定D/Q与检查：full68.813051→67.098667 s（−2.49%），main55.397621→53.599529（−3.25%），init未改善；NTT arena完整payload峰减少314552032 B（约300MiB），主A/B/Q仍3GiB。没有测本轮NVML总峰，S4传输量保持。此前P2的2.62%也以原M4为对照，不能与2.49%相加。

新D系数以六条多D曲线+四条fixed-D锚点拟合，旧NTT/新策略各自匹配；最终独立B2=1e11 holdout330330每条快于510510，但秒数预测偏低8.23–10.27%。新默认大界D1381380、小界D330330；经验scope与G/owner/arena过滤保持。共用header+实际backend CPU32/0，独立NTT10/0含88项策略检查，旧warp0/1复测通过。

Systems单次池NTT29.388810→27.812577 s、point约19.24 s保持，近似Stage2无本进程事件9.77→9.57 s，占14.31→14.41%；不是整卡idle。全范围传输字节/次数保持，同步copy host等待远大于相关GPU copy。Compute尝试实际ERR_NVGPUCTRPERM，未获得有效occupancy/带宽/cycle；没有改变权限。

生产3CF38065…19F0E、4042752 bytes、CUDA593.7 s，默认outer2，入口32/0；原F85快照保留。具体样本、公式、系数、剖析范围、实际save时间、source/line和回退见[尺寸策略/D报告](D:/code/MPA-OpenCl/docs/STAGE2_NTT_SHAPE_D_CALIBRATION.md)。下一阶段独立整数Tensor Core Goldilocks探针，之后再评估整NTT接入；多曲线吞吐需独立状态和共享workspace预算，尚未实施。

## 41. 精确整数 Tensor Core tile 与双流吞吐实验（2026-10-05）

实现独立u8整数MMA/132bit Goldilocks归约，并将低4层DFT16接入真实t12 tile布局；production header/默认/入口保持P3。GMP门禁35/0，1920 cases/1536000矩阵word加614400top检查，以及2560 cases/11228160 tile与保护word；故意损坏被拒绝，LOCAL0/live0。初版shared radix4 caller遗漏CTA屏障导致CTA512/t10,t12随机错误，补每个helper调用后的barrier后通过，调试日志已移除。回归fixture保留真实调用边界。

最终probe SHA2DD2C230…8B52A，sm89/CUDA13.3，2797056 bytes，构建34.24s；同binary serial ABBA+BAAB，CTA128/256/512、forward/inverse/roundtrip。锁定最佳TC256后独立复测k24并留出k25/26：前向慢18.23–18.30%，逆向慢5.55–5.60%，roundtrip慢10.90–10.93%。自然序DFT16的约1–3%收益未延续到真实tile，未接入生产。

双任务CUDA+TC并发比CUDA+CUDA并发慢9.13–9.17%；Systems同exe目标128kernel、除warm每模式12pair，C+TC平均overlap47.342μs、pair span6.025034ms（.786%），主要顺序执行。该trace不证明同SM同时issue。32KiB shared下CUDA40reg/CTA512可3块；TC74/76reg，CTA256可3块、CTA512仅1块，这是容量API上限。无有效NCU硬件计数器。

令L为NTTword数、T=2^t：单probe逻辑显存16L+112T+4088 B，双任务乘2；k24/t12为256.441MiB与512.883MiB，context/driver另计，未测NVML总峰。每tile pass L/2次warp MMA、1024L byte MAC、L次132bit归约；主global forward16L/inverse24L B不变。host参考/初始化H2D仅112T+4080 B；性能probe由GPU周期填充，event内host传输0，每次在计时外比较全部L输出。真实曲线传输口径见此前报告。

生产仍868c449/3CF38065…19F0E，14依赖复核通过。下一优先级为CUDA stage2/3单位根移位特化，再定位P3约9.57s的host提交/准备空隙；多曲线需要独立状态与共享NTT workspace lease和RAM/VRAM预算。当前两个独立tile任务不能折算curves/h。详细公式、原文件/line、样本、证据和复现命令见[Tensor实验报告](D:/code/MPA-OpenCl/docs/STAGE2_TENSOR_GOLDILOCKS_EXPERIMENT.md)。长期Prime95对照目标继续。

## 42. Goldilocks 短归约与完整 Stage2 收益（2026-10-05）

低层根按值分支/按指数移位分别慢55.56%/3.88% roundtrip；加入精确短归约后局部快约2.2%。将短归约用于所有device Goldilocks算术，任意128位的carry/borrow证明和GMP检查通过；完整纯NTT k24..27同binary快34.06%–36.63%，对保留的旧probe独立参考快31.88%–34.72%。新mode初版造成forward local8B，volatile模式加载单变量修正为local0；真实Stage2 trace也确认warp tile40reg/local0。

固定M4423/Q/sigma26/extra12/B1=1000/B2=2011326186870/D1231230/P115200，八条串行ABBA+BAAB：完整**69.778377→59.482459 s（快14.76%）**，main55.356895→45.725436（快17.40%）。旧Stage2 binary的两条独立参考均值67.173660 s，新后端相对它快11.45%；该参考不交错，不能把控制差額全归因于分派。两顺序组均改善，检查/根/叶摘要、工作量保持，未建立置信区间。188/0新后端完整门禁通过；旧后端完整门禁因900s协调时限止于182通过/无失败，保留记录；两侧36/13项独立NTT门禁完整通过。

显存arena payload峰3341512288 B、workspace3GiB保持；host private峰均7314→7318MiB保持。Systems NTT池28.545614→19.095016 s（−33.11%），point约19.2s保持；本进程无GPU事件间隙9.554751→9.513572 s保持，占比14.25%→16.44%。全部113739个kernel在GPU1，copy字节/次数保持；同步copy host等待21.18→16.10s，相关GPU copy仍约.034s。没有新NCU硬件计数，间隙不是整卡idle。

新开关NTT_GL_SHORT_REDUCE缺省0，explicit1试新后端，低层根特化仍独立。旧D经验模型在short1下回退，日志gl_short可核对；最终guard重编译48A183D1…2C846，实际规划36/0、最终shared/warp GMP/resource14/0。性能binary为17BF4813…E58A2，两者不可混用SHA。生产exe仍3CF38065…19F0E，未提升新默认；新D拟合/holdout后再发布。下一优先级为新D标定、CPU准备/提交间隙及剩余点/NTT算术。数学、源文件line、全部样本、容量/传输公式与复现见[短归约报告](D:/code/MPA-OpenCl/docs/STAGE2_GOLDILOCKS_SHORT_REDUCTION.md)。

## 43. 短归约匹配 D 模型与生产发布（2026-10-05）

短归约后重新冻结k16..27卷积权重，六条多D曲线加四条同binary锚点拟合profile3；独立B2=1e11留出验证330330每条快于510510，绝对秒数仍低估5.99–11.76%，未回调拟合。大界D1381380、小界D330330，owner512MiB时大界D1141140。真实NTT backend/C++模型与整数features **48/0**，实验planner **22/0**、归约scope **36/0**、shared/warp NTT **14/0**；三个自动D真实曲线均GMP bad0/pending0/clean1。模型不增加GPU数组/传输。

生产33/0、最终实际S4后端选择器30/0；后者为早停决策门禁，不冒充完成曲线。额外修正OLDTAIL/S4_OFF scope，并将同一S4启用快照提前；错误指针初版28/2失败候选保留且未发布。最终E0139328…BE331，4127232 bytes，CUDA571.1 s；15原始依赖及快照核验。生产save固定D1381380四条ABBA：64.647779、56.586517、57.082008、65.380452 s，均值**65.014116→56.834263 s（快12.58%）**，每模式2条、无CI，叶/oracle/因子一致。生产short默认1，显式0回旧归约与resident_shape；实验仍默认0。旧3CF产物已保留。

D138标定均值full56.643870s：init26.59%，baby14.08%、CPUaffine6.10%、F树剩余6.41%；main73.41%，G树23.96%、giant18.28%、下降13.46%、fold10.69%。owner609.086MiB，NTT arena完整payload3341481200B，主workspace3GiB；host输出窗口实际回读2838813040B，不等于所有曲线D2H，没有新NVML全进程峰。具体计算量、内存与传输公式及各阶段误差见[短归约 D 标定与生产报告](D:/code/MPA-OpenCl/docs/STAGE2_SHORT_REDUCTION_D_CALIBRATION.md)。

下一轮优先GPU baby批量规范化。原Systems最大2.784s间隙在X/Z回读后、下次copy提交前，与CPUaffine准备关联；trace无CPU栈，仍为源码推断。giant叶是[-X,Z]，baby F树要求[-X/Z,1]，须增加设备prefix/逆元传播并保留坏Z的GCD/因子语义；先仅回读P个常数再接设备叶frontend。之后复测gl_mod规范化/低层根；多曲线仍需独立状态与workspace lease及RAM/VRAM预算，公平Prime95新对照待做。

## 44. GPU baby 乘积树、逆元传播与生命周期（2026-10-05）

将baby前缀积/逆元传播改为GPU分组乘积树，每256点CPU求一次根逆元；普通逆元与Mont图像配合直接生成[-X/Z,1]，保留不可逆Z/零Z的GCD、原X回退和small-prime cache。独立kernel42组/379552字、host9/0、Python/CPU baby/F12/0及实际D scope1/0通过，故意损坏与分配回退被门禁覆盖。完整实验A5ADDCD9…E7F8，CUDA581.4/link3.2秒，8依赖前后SHA及快照核对。

固定M4423/Q/B1=1000/B2=2011326186870/D1381380/P126720、short1/outer2，八次串行ABBA+BAAB：**full 57.302417→53.294626秒（快6.99%）**，affine **3.576000→0.276000秒（快92.28%）**；main均值基本保持，两顺序组均改善，每模式4条、无CI。叶/oracle/Q/因子结果和mandatory检查覆盖一致。正常组全部可逆，D2H减少70686000B，H2D增加277695B。

Systems配对同binary/D：本进程无GPU事件间隙8.674279→6.068857秒，16.23%→11.97%；旧baby回读后2.353秒大空隙缩至.236秒。它不是整卡idle或observed occupancy。新增16kernel共.158秒，NTT约17秒、point约18.5秒保持。Device分配逐地址账本峰减少136.321MiB，旧8静态ladder buffers余额对应此差额；不是NVML总峰。新临时payload271.409MiB在F树前释放。NW128三个kernel REG40/47/40、LOCAL0，但STACK4112/5136/6160B，不能据LOCAL0声称无线程私有存储。

NTT_BABY_DEVICE默认0，实验显式1启用；现有D模型在请求新路径时回legacy，性能先固定D。生产E013保持；下一轮匹配新D、独立holdout和save入口后再发布。首次Nsight因应用stdout未进入CLI日志被核验拒绝，保留记录，改独立app日志后在新目录完整采集并逐项验证。具体原文件/line、计算量/数据量/预算公式、样本和证据见[GPU baby 归一化报告](D:/code/MPA-OpenCl/docs/STAGE2_GPU_BABY_NORMALIZATION.md)。后续处理剩余准备间隙及point/NTT MAC，多个曲线需workspace lease与RAM/VRAM生命周期预算，公平Prime95对照待做。


## 45. GPU baby 新成本模型与生产默认（2026-10-05）

§44路径已完成D重标定与原生save/ini/worktodo接入，生产默认NTT_BABY_DEVICE=1，显式0回CPU baby及对应旧模型。profile4 resident_baby_v1仅用于已测short1/尺寸策略/M4423/B1=1000/设备/检查scope，并硬过滤baby临时payload与live_free−64MiB；原scope诊断、预算不足和分配失败注入不套新系数。

六条多D加四GPU锚点，独立B2=1e11 holdout每条330330快于510510；大界仍1381380、小界330330、owner512仍1141140。仅限制baby128MiB则候选600600，只验证了planner。native模型与整数features64/0、输入8/0、生产入口33/0、旧scope30/0、新scope80/0、原生planner9/0；owner512未拟合D真实56.421647秒、同参考叶值。

发布A1910CB4…56EC6A，17原始依赖哈希与快照核验。完整存档同exe固定D ABBA **53.569084→50.520226秒，快5.69%**（2样本/模式，无CI）；新路径affine2.096→.2405秒。default/rollback和叶/oracle/因子合同保持。E013已备份，GPU0外部生产保持。源码、B2/N/W/P/I/G公式、完整时长和生命周期说明见[本轮报告](D:/code/MPA-OpenCl/docs/STAGE2_GPU_BABY_D_CALIBRATION.md)。该模型未增加设备数组/传输，本轮未重采整进程VRAM/host-private峰，不能将owner512当作总VRAM限制。后续继续point/NTT、CPU准备和多曲线lease；公平Prime95对照仍待完成。


## 46. 逆 NTT 归一化的移位候选（2026-10-05）

新增默认关闭的 NTT_GL_SHIFT_SCALE，精确 k/scale 判断后每 inverse slice 删除 N 次通用 Goldilocks 模乘；没有新增数组、完整 pass 或生产数据传输。纯完整卷积 k24..27 八次交叉快约0.30–0.42%；同 save/D 原生整曲线八次 51.770018→51.965438s，尚未证明稳定改善。原语/GMP频谱/逆向/cache/lifecycle/故障门禁通过，实际scope192/0、frozen因子2/0；生产A191保持，新请求拒绝旧D经验模型。数学、全部样本、容量/传输合同及最新source/line见[逆归一化报告](D:/code/MPA-OpenCl/docs/STAGE2_NTT_INVERSE_SCALE_OPTIMIZATION.md)。


## 47. Goldilocks 显式进位链（2026-10-05）

新增NTT_GL_PTX_REDUCE，short1下将任意128位归约换为显式32bit CC链，默认0。依赖链快12.55%，真实完整卷积相对冻结旧探针快2.66–6.76%；内部A/B基线有新分派开销，数据分开报告。完整native八条内部 50.922277→49.382666s，另A191真实交叉 51.214982→50.778877s；检查覆盖与结果一致。scope192/0、native warp/frozen因子2/0，18源码冻结。数组/传输payload保持，生产A191未更新；PTX请求保护旧D模型，待重新标定和编译特化。[完整报告与当前source/line](D:/code/MPA-OpenCl/docs/STAGE2_GOLDILOCKS_PTX_REDUCTION.md)。


## 48. 固定NTT后端（2026-10-05）

GlBackend可选编译将Goldilocks归约固定，移除每次mode load/判断，保留runtime A/B；环境冲突拒绝，PTX实际后端保护D scope。固定short/PTX门禁16/0、runtime8/0，所选kernel寄存器/占用率不变。k24..27相对旧runtime/PTX1纯卷积快2.26/5.33/4.67/8.10%；8条完整Stage2相对生产A191 51.365461→49.575260s，快3.4852%，检查覆盖相同。native selector48/0及普通因子/warp/冲突通过。算法数据payload保持；生产A191未发布替换，下一步新后端D拟合与holdout。[完整说明与source/line](D:/code/MPA-OpenCl/docs/STAGE2_NTT_FIXED_BACKEND.md)。


## 49. 固定PTX D重标定与生产发布（2026-10-05）

固定PTX/GPU baby新profile5 resident_fixed_ptx_v1：冻结12个NTT权重、6条多D和4条固定D锚点，旧profile数值保持。独立小界留出390390相对510510快约6.4%；390390与330330近似持平，不宣称.16%微小收益。大界仍1381380，小界390390，owner512预算1141140，baby128预算600600；硬预算和路径scope先于预测。

输入22/0、C++模型80/0、实际planner9/0、selector90/0、native完整入口30/0、发布路径22/0分别验收。最终同Q/D默认检查8条串行交叉A191→DCF 49.9000155→48.33592125s，减少3.1345%；叶/oracle/因子与检查覆盖一致。生产目录现为DCF70B11…289D4，18原始依赖冻结；A191及17源码已备份。重建发布需-GlBackend ptx，builder默认runtime保留。公式/容量/传输/阶段占比、样本和source/line见[本轮报告](D:/code/MPA-OpenCl/docs/STAGE2_FIXED_PTX_D_CALIBRATION.md)。后续先测point MAC/CPU准备，多曲线需workspace lease；公平Prime95目标仍待完成。


## 50. Mersenne点模乘：共用SOS乘积、线性折叠/旋转（2026-10-05）

新实验NTT_POINT_MERSENNE默认0，精确N=2^s−1才启用，结果仍abR⁻¹。主导MAC2W²→W²，不增加点/NTT持久数组；设备状态4B。原语109/0、局部8次快46.07%；actual native18/0、入口21/0。完整内部8条46.997380→39.800719s（−15.31%），另生产DCF对照4条−19.49%，Q/叶/oracle/检查覆盖保持。新fit未完成，请求该路径拒绝旧D模型；生产仍DCF。编译强制inline主动终止、整函数outline小批次负结果均保留，最终只outline O(W)归约。profile点池17.960→8.574s，CPU准备/多曲线lease/公平Prime95仍需推进。[数学、全部样本、容量与source/line](D:/code/MPA-OpenCl/docs/STAGE2_POINT_MERSENNE_MONTGOMERY.md)。


## 51. 点折叠scratch复用与D发布（2026-10-05）

原out由caller共享给线性归约，真实NW128七种点内核stack各少1024B。profile6按六条多D+四锚点独立重拟，冻结NTT权重来源不变；预算首选D未变。模型96/0、输入27/0、planner9/0、selector96/0、点native18/0、入口30/0。DCF/893完整八条同Q/D/检查48.8331495→39.0421205s，少20.05%；发布默认fixedPTX point1，0回旧Mont/profile5，generic fallback。生产893F6E90…01F69D/19原始依赖，旧版备份，发布路径21/0。[公式/占比/容量/全样本/source-line](D:/code/MPA-OpenCl/docs/STAGE2_POINT_FOLD_D_CALIBRATION.md)。

## 52. NTT单位根固定PTX留出（2026-10-05）

独立方法3在实际PTX3下36/0门禁通过；k24..26每长度forward/inverse/roundtrip八次交叉，roundtrip慢约7.5%，资源与baseline相同。候选不接生产；旧四fold上的收益不适用于新后端。下一项outer根乘积/同步成本，CPU准备与多曲线workspace lease继续保留。[独立负结果与source-line](D:/code/MPA-OpenCl/docs/STAGE2_NTT_SMALL_ROOTS_FIXED_PTX.md)。

## 53. NTT outer ILP 编译候选（2026-10-05）

合作outer的u循环展开4组，完整卷积k24..27独立重复少0.70–0.78%；默认与unroll1八实例GPU指令相同，后者启动波动不记收益。u4 M8 forward REG48→78，M6可驻留capacity5→4；shared/LOCAL0与MAC/pass/payload保持。源码默认NTT_OUTER_UNROLL_U0，native builder可选-OuterUnrollU4，候选停用旧D经验profile，必须用显式D做公平对照。

native48BD…0360EF/19源，原生点/实际scope18/0，基本入口21/0。八条full38.594806→38.453166s名义少0.367%，两顺序组方向不同，未证明整曲线稳定收益，生产893保持。接入宏0/4各13/0且默认0/候选4/native4各八SASS实例匹配对应旧探针；叶/oracle/因子/强制检查保持。[完整报告、公式、source-line与复现](D:/code/MPA-OpenCl/docs/STAGE2_NTT_OUTER_ILP.md)。下一项关注更大规模的跨层融合/根同步收益及CPU准备空隙，多曲线需要共享workspace lease。


## 54. NTT进位与必需诊断融合（2026-10-05）

源码可选NTT_CARRY_CHECK_FUSED=1，默认0。既有carry cone的最终digit直接生成warp/block摘要，独占8B/CTA再按slice finish，保持长mask精确传播和此前错误累积。用L=每slice NTT长度、m=batch、T=ceil(L/256)，新增显存8TmB，诊断逻辑访问8Lm→16TmB；NTT模乘/pass、输出排列、两launch和verdict readback保持。只在Lm≥2²⁰使用，arena预算不足/分配失败fallback；真实最大4MiB已计账。

原语180cases/657000words、12组R/bpw×18shape×两模式bad0；实际runner16cases/122850words，transform13/0，native两模式各18/0及故障/真实分配回退。R5 REG35→39/R6 39→40/shared64B，finish REG24/shared96B，真实STACK0/LOCAL0。完整八条预热后ABBA+BAAB均值38.2264815→37.9273095s（少0.783%），main少0.940%，Q/叶/oracle/因子/全部必需覆盖保持。Systems carry+check少16.9%，PCIe主体量保持；自身事件空隙仍约13.5%，不代表整卡闲置。

native8A78…7955DA/20源码，生产893与默认0保持。请求新检查模式拒绝旧D profile，固定D实验；下一步先冻结新NTT权重并多D/独立holdout再决定提升。CPU准备和多曲线共享workspace lease仍需推进。[详细算法、公式、全部样本与代码原文件line](D:/code/MPA-OpenCl/docs/STAGE2_NTT_CARRY_CHECK_FUSION.md)。
