# ECM GPU Stage 2：本仓库树形实验与 Prime95 Poly 方法对照

**调查日期：2026-10-02；实验追记：2026-10-03**
**范围：ECM Stage 2；主线是本仓库 CUDA 多项式树实验与 Prime95 Poly 方法。** PrMers Gaussian ECM BSGS 仅作补充对照；PrMers 的 P−1 V-trace 不属于椭圆曲线 Stage 2。
**方法：**初始调查静态阅读本仓库、Prime95 与 PrMers 源码，以及 docs/DEV_GPUOWL_NTT_NOTES.md；后续实测与代码变更按 §28 逐轮追记，计时边界和验证以对应开发日志为准。Prime95 ECM 源文件实际位于 .refactor/p95v3106b01.source/ecm.cpp（不是 ecm/ 子目录），多项式乘法实现位于同版本 gwnum/polymult.c/.h。

## 本次补充摘要（ECM Poly Stage 2 主线）

- Prime95 Poly 的主算法是 F baby-root 积树、按 giant block 构造 G、递推 H ← G·H mod F，再沿 F 树做 Bernstein scaled remainder descent；Prime95 原码流程与当前 CUDA 树版逐段对应，详见 §27.1–27.3。
- 当前 GPU 路线的主要实现差别是算术/数据通路：Prime95 用 Gwnum 浮点 FFT 与 roundoff guard；本仓库使用精确系数 NTT、设备打包与模 N 归约。树形算法相似不代表底层访存成本相同。
- 初始调查的生产记录：D=1,231,230 / P=115,200，elapsed 273.55 s；成本驱动 D 搜索与 arena 跨形状驱逐已带来 −31.2%。后续打包、存储寿命、重复读回与输出窗口的实测见 §28；不同阶段/二进制的数据不能直接相减归因。
- S5 分组下降曾实现 1.9× 小形状加速但结果不正确，代码已回退。只有逐层余数和叶值对拍通过，并在生产形状 A/B 证明更快，才应重新启用。
- PrMers 的直接 BSGS 对照保留在旧章节作为旁支；它不能直接代表 Poly/NTT Stage 2 的访存或性能。

## 1. 结论摘要

1. **PrMers 与本仓库的 CUDA pairing/BSGS 原型在数学上属于同一类算法。**两者都选偶数 `D`，对候选素数 `q` 写成 `q=kD±d`，检验 `[kD]Q` 与 `[d]Q` 的 x 坐标是否相同，并累乘交叉乘积 `X_g Z_b − Z_g X_b`。这是 projective 坐标下避免求逆的 BSGS 判据。[PrMers `RunGaussianMersenneEcmOptimized.cpp:1452-1469`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp)；[本仓库 `cgbn_stage2_kernel.h:10-25`](../kernels/cuda/cgbn_stage2_kernel.h)
2. **主要差异不在公式，而在 GPU 映射与数据搬运。**PrMers 对一个曲线在 GPU 引擎寄存器中保存必要的 baby 点和 giant 状态，逐 giant 扫描关联 prime 项并周期性同步做 GCD；本仓库原型把全量 baby/giant 点作为 per-curve 32-bit limb 数组放显存，prime 索引也放显存，再由 CGBN 实例以 segment 分工执行扫描。前者降低显存表流量，后者支持多曲线/分段并行但带来大表容量和非连续索引读取。
3. **本仓库另有一条更有扩展性的多项式树 GPU 实验路线**：将 Stage 2 写成 baby roots 的乘积多项式、giant 块多项式及余式树下降，借助 NTT 多项式乘法。它与 PrMers 的逐素数 cross-product BSGS 不是简单的访存优化版，而是批处理算法重构；源码仍在 `tools/bench/`，开发计划明确标作研究/benchmark 引擎，不能据此称为已接入默认 ECM 生产流程。[`stage2_tree_gpu.cu:2-47`](../tools/bench/stage2_tree_gpu.cu)；[`DEV_STAGE2_FLOW.md` §0–§2](DEV_STAGE2_FLOW.md)
4. **性能记录显示本仓库的直接 pairing GPU 路线不适合大 B2。**开发文档基于现有 modmul 吞吐估算，认为配对路线约需 2.7e9 次/秒才能达到 6 s/曲线，而记录吞吐约为 4060 上 35.7 Mops/s、4070 Ti 约 9e7 ops/s；文档据此将其定位为正确性沙箱/小 B2 兜底。树版则在生产形状的记录中显著快于早期 ladder/pairing 方向，但仍受多项式树构建、host 协调、下降与同步影响。[`DEV_STAGE2_GPU_PLAN.md` §2.1–2.2、§18、§41](DEV_STAGE2_GPU_PLAN.md)

## 2. 比较对象与边界

### 本仓库

- **直接 pairing/BSGS CUDA 实现**：`kernels/cuda/cgbn_stage2.cu`（主机侧准备、分配和 launch）与 `kernels/cuda/cgbn_stage2_kernel.h`（设备 x-only 算术、表生成、配对累乘）。文件头注明这是 `stage2_ref.cpp --algorithm pairing` 的 GPU 对照实现；接口要求奇数模数且 tier 上限按实例化列表决定。[`cgbn_stage2_kernel.h:1-25, 46-71`](../kernels/cuda/cgbn_stage2_kernel.h)；[`cgbn_stage2.cu:219-250`](../kernels/cuda/cgbn_stage2.cu)
- **CPU pairing oracle**：`tools/bench/stage2_ref.cpp`，用于算法正确性与命中集合对拍，不是生产 GPU 实现。[`cgbn_stage2_kernel.h:1-15`](../kernels/cuda/cgbn_stage2_kernel.h)
- **乘积树/余式树 GPU 实验**：`tools/bench/stage2_tree_gpu.cu`；CPU 参考为 `tools/bench/stage2_tree_ref.cpp`。这是不同的 Stage 2 算法工程路线，必须与直接 pairing 分列讨论。[`stage2_tree_gpu.cu:2-47`](../tools/bench/stage2_tree_gpu.cu)；[`DEV_STAGE2_GPU_PLAN.md` §2.2、§18](DEV_STAGE2_GPU_PLAN.md)

### PrMers

- 对应对象是 **Gaussian-Mersenne ECM 的 v99.98 fused Montgomery Stage 1 + opt-in BSGS Stage 2**：`src/modes/RunGaussianMersenneEcmOptimized.cpp`。入口选择见 `RunGaussianMersenneEcmFast.cpp:607-617`，发布说明明确 `-bsgs` 才走优化路径，并且切生产前需通过正确性及 Radeon VII A/B 性能门槛。[PrMers `RunGaussianMersenneEcmFast.cpp:607-617`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmFast.cpp)；[PrMers `RELEASE_V99.98_GAUSSIAN_ECM_FUSED_BSGS.md:37-87`](../.refactor/PrMers-main/RELEASE_V99.98_GAUSSIAN_ECM_FUSED_BSGS.md)
- PrMers 的 `RunPM1.cpp` V-trace / Pair95 是 **P−1 Stage 2**，运行对象是幂 `H^q`，不是 ECM 椭圆曲线点，不纳入公式/访存对照。CLI 自身将其命名为 `P-1 Stage 2 scalar trace BSGS`。[PrMers `src/io/CliParser.cpp:90-104`](../.refactor/PrMers-main/src/io/CliParser.cpp)；[PrMers `README.md:493-537`](../.refactor/PrMers-main/README.md)

## 3. 算法逐步对照

| 步骤 | 本仓库 pairing/BSGS | PrMers Gaussian ECM BSGS |
|---|---|---|
| Stage 1 点输入 | 可直接传存档仿射 `x`，设 `QZ=1`；可选没有 `x` 时在 GPU 上根据起点重做 `[s]P0`。stage 2 的 cross-product 使用齐次量，不需把输入 `x` 先求逆或统一 Montgomery 域。[kernel `:130-157, 270-320`](../kernels/cuda/cgbn_stage2_kernel.h) | Stage1/Stage2 fused Montgomery engine 中得到当前曲线的 `Q`；Stage2 入口为每条曲线运行 BSGS。发布说明把该路径描述为融合 Montgomery Stage1 + differential BSGS。[PrMers optimized `:1014-1165`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp)；[release `:37-52`](../.refactor/PrMers-main/RELEASE_V99.98_GAUSSIAN_ECM_FUSED_BSGS.md) |
| D 与 prime 计划 | 主机 segmented sieve 枚举 `(B1,B2]`；`p|D` 跳过；`p<=D/2` 单独作为 small case，其余映射到 `i,j`。计划表 `p_i/p_j/p_small` 传入设备。[`cgbn_stage2.cu:134-190, 254-265`](../kernels/cuda/cgbn_stage2.cu) | 主机 segmented sieve 得到 Stage2 primes，并将每个 `q` 映射为 `k, d=|kD-q|`，只保留 `d<=D/2` 且 `gcd(d,D)=1` 的条目；baby residue 集合是实际条目 `d` 的去重集合。[PrMers optimized `:708-761, 773-852`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp) |
| Baby points | 生成 `j=1..D/2` **连续整张表**，首两点由 Q 和 xDBL 得到，后续点用 xADD chain。[kernel `:285-312`](../kernels/cuda/cgbn_stage2_kernel.h) | 仅为 prime plan 需要的 unique `d` 生成 baby 点；当前代码逐个 `d` 调 scalar ladder，再复制到 `layout.baby_x/z` 并预设 multiplicand。[PrMers optimized `:823-851, 1347-1359`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp) |
| Giant points | 先用 ladder 算 `[D]Q`，再连续 xDBL / xADD 生成 `iD` 点，全部写入 global-memory 表。[kernel `:312-340`](../kernels/cuda/cgbn_stage2_kernel.h) | 保存 `[D]Q` 为 `base` 和相邻 giant 状态；循环递推 `[(k+1)D]Q=[kD]Q+[D]Q`，用前一 giant 作为 differential point。无需构造或保留从 1 到 `B2/D` 的整张 giant 表。[PrMers optimized `:1361-1388, 1485-1504`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp) |
| prime hit 判据 | 普通配对项累乘 `X_i Z_j-X_j Z_i`；small prime 累乘 `[p]Q` 的 `Z`。候选项可能是可命中 prime 的超集，所以与 ref 的 hit 口径需一致。[kernel `:10-25, 338-376`](../kernels/cuda/cgbn_stage2_kernel.h) | 每个 `q=kD±d` 对应同一个当前 giant；算 `X_g Z_b−Z_g X_b` 后乘入 `ACC`。发布说明给出相同 cross-product 和模数 GCD 判据。[PrMers optimized `:1452-1469`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp)；[release `:43-52`](../.refactor/PrMers-main/RELEASE_V99.98_GAUSSIAN_ECM_FUSED_BSGS.md) |
| GCD/恢复 | GPU 每个 `(curve,segment)` 各算局部乘积；结果拷回 host，将各 segment 相乘后每曲线做 GMP gcd；stage1 `Z` 另行检查。[`cgbn_stage2.cu:492-547`](../kernels/cuda/cgbn_stage2.cu) | 默认按 256 个 term 批量 `sync → project ACC → gcd`，遇到 `gcd==N` 则回退 legacy Stage2；提供独立 Stage2 checkpoint。[PrMers optimized `:1400-1439, 1441-1483`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp)；[release `:73-76`](../.refactor/PrMers-main/RELEASE_V99.98_GAUSSIAN_ECM_FUSED_BSGS.md) |

## 4. GPU 数据布局、访存和并行

### 本仓库 pairing/BSGS：表驱动、跨实例并行

- **大整数布局**：每个 number 是 `BITS/32` 个小端 `uint32_t` limb；每曲线 baby/giant 表按 `(curve, point, X/Z, limb)` 索引，`baby` 大小随 `curves × (D/2+1) × 2 × limbs` 线性增长，`giant` 随 `curves × (B2/D+3) × 2 × limbs` 增长。[`cgbn_stage2_kernel.h:65-97`](../kernels/cuda/cgbn_stage2_kernel.h)；[`cgbn_stage2.cu:325-344`](../kernels/cuda/cgbn_stage2.cu)
- **Global memory 流量**：table kernel 每生成 chain 项都将 X/Z 写出；pair kernel 对每个 prime 根据 `i,j` 读 giant 和 baby 两点；每条配对至少涉及两次 cross product 的模乘、一次差、一次 accumulator 模乘。随机 prime 顺序会让 baby/giant 访问不如连续扫描友好。prime 索引按 `k += segs` 切分，减少单条累乘依赖，但重复访问曲线表。[`cgbn_stage2_kernel.h:285-376`](../kernels/cuda/cgbn_stage2_kernel.h)
- **并行粒度**：第一 kernel 一个 CGBN instance 对应一个 curve；第二 kernel 一个 instance 对应一个 `(curve,segment)`。`segs` 增大可拆短串行 accumulator 链，但增加 accumulator 输出、host 规约量和每曲线拼接工作。[`cgbn_stage2_kernel.h:97-100, 285-376`](../kernels/cuda/cgbn_stage2_kernel.h)；[`cgbn_stage2.cu:239-242, 467-475, 492-510`](../kernels/cuda/cgbn_stage2.cu)
- **算术/寄存器**：CGBN TPI=4 或 8，block TPB=128，shared-memory limit=0；每实例由协作线程分摊大整数 limb。每个 `xdbl` 约 2S+2M，`xadd` 约 4M+2S；固定 BITS tier 也关系到 Montgomery `R=2^BITS`，host 和 device tier 必须一致。[`cgbn_stage2_kernel.h:31-43, 108-127, 198-209`](../kernels/cuda/cgbn_stage2_kernel.h)
- **显存风险**：host 有显式 `cudaMemGetInfo` 预算检查，不能装下则失败并提示减少 curves、D 或 B2；代码不会自动把 baby/giant 表分页或批处理。[`cgbn_stage2.cu:325-344`](../kernels/cuda/cgbn_stage2.cu)

### PrMers：寄存器驻留 baby + 常数状态 giant 流

- **baby 数据集缩减**：计划不是默认保存所有 `1..D/2`，而是从实际 prime 候选中抽出去重 `d`；当前生产默认 D=210，注释明确 24 个 baby x 坐标需要 48 个寄存器，自动模式限制 baby 数，防止寄存器 footprint 过大。[PrMers optimized `:791-852`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp)
- **device-resident register layout**：`OptLayout` 将每个 baby 的 X/Z 映射到 engine register 编号，后续 giant/base/next 状态也留在寄存器布局中；打印总寄存器数。[PrMers optimized `:855-883, 1157-1165`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp)
- **低表流量**：每条 prime entry 在 host 上已含 `k,d`，CPU 通过二分查找找到 baby register；device engine 对 register 做 prepared multiply、subtract、accumulate。热循环没有本仓库那种对 global-memory baby/giant 表的 X/Z 随机 load。[PrMers optimized `:1003-1008, 1452-1469`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp)
- **giant 递推换存储**：只保留当前/下一个 giant 和 `[D]Q` base，连续推进并在同一个 giant `k` 内处理多条 prime entry，降低显存占用与表带宽；代价是 giant chain 有依赖，不能把不同 giant 都作为独立并行项。[PrMers optimized `:1361-1388, 1485-1504`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp)
- **Prime 聚合与 host-device 同步**：在累计达到 GCD batch 时 sync、读回 accumulator 再做 host GMP GCD；每 256 term 批量控制同步频率，另有 checkpoint/resume。同步/投影成本是这个寄存器流算法的明确边界。[PrMers optimized `:1400-1438`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp)

### 树形 GPU 路线：多项式并行换代数结构

本仓库树版先构造 `F(X)=∏(X-x_j)` 的平衡积树，再针对 giant 点块构造 G 树，做模 `F` 的折叠/累积，最终走余式树下降得到叶值并乘积/GCD。源码按树层对相同 shape 的多项式乘法分批，下降也对同一除法形状分组。[`stage2_tree_gpu.cu:2377-2465`](../tools/bench/stage2_tree_gpu.cu)；[`stage2_tree_gpu.cu:2900-3045`](../tools/bench/stage2_tree_gpu.cu)

多项式系数以扁平的 `uint64_t` 系数组织，采用与 NTT 输入一致的 coefficient-major/limb-contiguous 布局；NTT 的缓冲、多趟读写和 host/device 打包会产生完全不同于 PrMers 的带宽开销。该路线的目标是把 prime-by-prime 数十亿次重复判定，改成少量大规模批量多项式运算；但实现复杂度、临时缓冲、启动/同步与下降工作也更高。[`stage2_tree_gpu.cu:19-47`](../tools/bench/stage2_tree_gpu.cu)；[`DEV_STAGE2_GPU_PLAN.md` §5.2–5.3、§18]

开发记录的实测提示不能只看 kernel 吞吐：生产形状下曾发现 host 端 giant 仿射转换超过 NTT；后续批量求逆/投影叶子优化改变了分布。当前文档还记录默认路径性能和 S5 设备下降优化经历，说明树版仍在快速演进、对设备形状和验证门禁敏感。[`DEV_STAGE2_GPU_PLAN.md` §41、§54–§55、§58]

## 5. 算法成本与规模影响

- 直接 pairing 的 Stage 2 候选 prime 数随 `(B2−B1)/log(B2)` 增长；每 prime 都要一次映射/查表和 accumulator 更新，所以即便表足够小，prime 数本身仍是主导项。开发计划对大参数给出的运算量与实测 modmul rate 对比，结论是仅增大 D 或优化表布局不能消除每 prime 的工作量。[`DEV_STAGE2_GPU_PLAN.md` §2.1]
- PrMers 的 `D` 决定 baby footprint 与 giant 次数之间的折中；实际只保留被 prime 用到的 residue。较大 D 减少 giant 次数但 baby registers 增多，可能降低 occupancy 或超过寄存器预算。默认小 D 与自动上限体现的是寄存器容量约束，而不是追求最大 D。[PrMers optimized `:791-852, 1157-1165`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp)
- 椭圆曲线点的 giant 递推是串行链。PrMers 用少量寄存器和较低访存换取链式依赖；本仓库表版将 giant 构造并行化并可对多曲线/segment 并行，但付出全表显存与写读成本。哪条更快取决于曲线数、B2、D、模数位宽、设备寄存器/显存及 host-sync，而不能从源代码结构单独断定。
- 树版避免逐 prime 的单独交叉项，转而跑大型多项式乘法和下降，算法总工作量受 baby polynomial degree、giant block count、NTT 形状影响。计划中生产形状的主耗时曾集中在 G 树和 fold，后来又发现 host 侧 giant affine conversion 占很大比例，因此任何数字都要注明对应代码版本、命令与设备。[`DEV_STAGE2_GPU_PLAN.md` §18、§41、§55]

## 6. 结果、限制与可借鉴点

### 对 PrMers 实现的评价

- **优点**：必要 baby 表寄存器驻留、只留必要 residues、prime 数据在主机预筛、giant recurrence 状态很小、GCD 可 checkpoint；适合单曲线流式跑，不受 `O(B2/D)` giant 表容量限制。
- **限制**：prime-by-prime 工作量仍在；giant recurrence 有依赖；baby 数增大即吞寄存器；host 控制 prime entry 与周期性 sync/GCD 会产生调度成本。源码中的 `D<=10000`、`B2<=2^32−1` 等是该 Gaussian ECM optimized path 的实现边界，不代表 ECM 数学本身的限制。[PrMers optimized `:708-735, 791-794`](../.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp)
- **状态边界**：v99.98 发布说明称该 BSGS 为 opt-in，只有通过 same-curve benchmark 且更快才建议切生产；不能将发布说明里的 golden correctness 误读成生产吞吐已胜出。[PrMers release `:78-87`](../.refactor/PrMers-main/RELEASE_V99.98_GAUSSIAN_ECM_FUSED_BSGS.md)

### 对本仓库实现的评价

- **直接 CUDA pairing** 更适合做严谨 GPU correctness oracle、批曲线并发、小范围 Stage 2 或性能基线。若继续打磨它，优先验证分块 prime stream、baby table cache/coalescing、限制 giant 表规模、按 giant 分组 prime entries、和 accumulator 分层规约；但这些不能改变其近似逐 prime 算法复杂度。
- **PrMers 值得借鉴的局部工程点**：从真实 prime 计划抽 unique baby residues；把一个 giant 下的 prime entries 连续分组；利用 device resident registers 保存常量小表；按照资源预算选择 D/baby count；批量 GCD 与断点恢复。
- **若目标是大 B2 吞吐**，当前仓库 docs 的主研究方向是乘积树/余式树而非继续扩大 CGBN pairing。树版要继续降低 host orchestration、批量 affine/point 投影成本、NTT launch/copy 和 descent 的 shape-specific 开销，并以 GPU 与 CPU oracle 的逐叶/因子集合一致性守住正确性。开发文档记录了树版研究阶段的 benchmark 优势，但这个实验程序仍不能等同于已集成的用户生产路径。[`DEV_STAGE2_FLOW.md` §2；`DEV_STAGE2_GPU_PLAN.md` §18、§41、§54–§58]

## 7. 关键原文件索引

| 代码/文档 | 位置 | 用途 |
|---|---|---|
| 本仓库直接 CUDA pairing | `kernels/cuda/cgbn_stage2_kernel.h:1-127, 285-376` | x-only 算法、算术、数据结构与设备 kernel |
| 本仓库直接 CUDA host orchestration | `kernels/cuda/cgbn_stage2.cu:134-190, 219-547` | prime sieve、host 内存/显存准备、launch、GCD |
| 本仓库 pairing oracle | `tools/bench/stage2_ref.cpp` | pairing CPU 参考 |
| 本仓库 GPU tree | `tools/bench/stage2_tree_gpu.cu:2377-2465, 2900-3045` | F tree、分批树乘和 remainder descent |
| 本仓库 Stage2 总体开发计划 | `docs/DEV_STAGE2_GPU_PLAN.md:1-` | 研究背景、资源/benchmark/当前优化状态（章节引用见正文） |
| 本仓库 Stage2 流程手册 | `docs/DEV_STAGE2_FLOW.md:1-` | CPU/GPU 分工与路线定位 |
| 本仓库可行性记录 | `docs/DEV_STAGE2_SELFHOST_FEASIBILITY.md:1-` | Prime95 对照、pairing 与树版成本判断 |
| PrMers ECM BSGS | `.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp:680-706, 708-883, 1347-1504` | prime plan、register layout、Montgomery xADD、baby/giant、hit 累乘、GCD |
| PrMers optimized-path dispatch | `.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmFast.cpp:607-617` | 说明 `-bsgs` 进入优化实现 |
| PrMers 发布与验收条件 | `.refactor/PrMers-main/RELEASE_V99.98_GAUSSIAN_ECM_FUSED_BSGS.md:37-87` | 判据、默认 D、checkpoint、上线前 A/B 条件 |
| PrMers P−1 Stage2 区分 | `.refactor/PrMers-main/src/io/CliParser.cpp:90-104` | 避免把 P−1 V-trace 与 ECM BSGS 混为一谈 |


> **阅读顺序：**下面的 §27 是本报告主线，追踪 Prime95 ECM Poly Stage 2 的数学流程、源码及其与当前 CUDA 实验的对应关系；原有 §1–6 保留为 PrMers 直接 BSGS 的补充比较。主要性能结论取自 [DEV_GPUOWL_NTT_NOTES.md](DEV_GPUOWL_NTT_NOTES.md) 最新记录。

## 27. ECM Stage 2 核心：Prime95 Poly 方法与当前 CUDA 实验

### 27.1 Poly Stage 2 在算什么

令 Stage 1 输出点为 Q，选取偶数 D。对 D 的相对素数 residue 生成 baby 点 x_j=x([j]Q)，构造 F(X)=∏(X−x_j)。对每个 giant block，生成点 Y_i=[m_iD]Q，构造 G_b(X)=∏(X−x(Y_i))。然后将每块 giant 信息折叠进 H(X)←G_b(X)H(X) mod F(X)。沿 F 的余式树下降可得到每个 baby 根上的 H(x_j)；它等价于该 baby 点与已处理 giant 点差项的批量乘积（符号/可逆尺度因子按实现约定处理）。最终把叶值合并并对模数 N 求 GCD，以发现因子。

这把 BSGS 的逐项 X_g Z_b−Z_g X_b 命中判定改写成“多项式积—模 F 折叠—多点求值”。D 仍决定 baby 集大小与 giant block 数的折中；算法没有消除所有点运算，而是把大量 prime pair 的交叉项变成平衡树上的批量多项式运算。

### 27.2 Prime95 源码路径（可直接按行复查）

1. **D、P、B2 与内存成本选择。**numrels 是小于 D/2 的相对素数个数；Poly 实现以它作为多项式长度。每个 giant block 的 section 数向 poly_size 的倍数补齐，保持批形状规则；成本模型估算 F/R/G/H、Ftree 与 polymult scratch 的内存，并加入超出 L2 后的惩罚。代码还搜索更合适的 B2：Poly 下经济区间可以远大于传统 pairing。见 [ecm.cpp:429–436, 731–739, 5756–5810, 6011–6055](../.refactor/p95v3106b01.source/ecm.cpp)。
2. **F 产品树。**从 nQx 的相对素数点建立线性 monic 因子，先用专用 helper 合并小因子，再逐层两两 polymult。FFT 计划按相近形状保存/重用；每层后批量 unFFT/FFT 系数。见 [ecm.cpp:9095–9212](../.refactor/p95v3106b01.source/ecm.cpp)。
3. **F 的倒数多项式与预处理。**用 Newton 倍增迭代求 reciprocal 1/F，然后对 F/R 预转置、压缩，以减少内存占用。见 [ecm.cpp:9232–9295](../.refactor/p95v3106b01.source/ecm.cpp)。
4. **G 产品树。**每个 outer loop 通过 mQ_next_array 生成一块巨点，构造 G 的平衡乘积树。polyG/polyH 共用连续分配，polyGaux 只保留一部分临时结果，注释明确这是用少量辅助空间省掉整块复制。见 [ecm.cpp:9320–9327, 9354–9435](../.refactor/p95v3106b01.source/ecm.cpp)。
5. **折叠 H←GH mod F。**首个 G block 初始化 H；之后用三段运算：先乘 G·H，乘 1/F 取高位得到商，再用 FMA 减去商乘 F 并只留低位余数。见 [ecm.cpp:9458–9474](../.refactor/p95v3106b01.source/ecm.cpp)。
6. **Bernstein scaled remainder descent。**先把 H 乘 1/F 形成 scaled remainder，再逐层把父余数对左右子多项式取余，最终落到线性因子/叶值。为了控制峰值内存，Prime95 释放 F/R、切片处理 H，并按可用内存从内存、磁盘保存或重建 Ftree 行。每个父 H 同时对左右两个子树求余时，polymult_several 让两个 child 运算共享共同操作数的变换。见 [ecm.cpp:9538–9578, 9627–9737](../.refactor/p95v3106b01.source/ecm.cpp)。
7. **合并叶值与 GCD。**helper 线程生成部分积，合并后进入 ECM GCD；在此之前释放大块多项式工作集。见 [ecm.cpp:9760–9880](../.refactor/p95v3106b01.source/ecm.cpp)。

gwnum/polymult 是 Prime95 的浮点 FFT/实数大整数表示路径，调用方检查 roundoff 和 safety margin；它不是本仓库的精确整数 NTT。Poly 选项包括 monic 输入、只取乘积高/低系数、FMA、保留计划及复用计划；polymult_several 允许一份输入乘多个相关多项式。见 [polymult.h:128–157](../.refactor/p95v3106b01.source/gwnum/polymult.h)、[polymult.c:4903–5050](../.refactor/p95v3106b01.source/gwnum/polymult.c)。FFT/Karatsuba 阈值处还留有 “Fix me” 注释，见 [polymult.c:531–539](../.refactor/p95v3106b01.source/gwnum/polymult.c)。

### 27.3 Prime95 与当前 GPU 分支：算法、数据流和访存差异

| 维度 | Prime95 ECM Poly | 本仓库 CUDA 实验 |
|---|---|---|
| 多项式乘法 | Gwnum 浮点 FFT，配 safety margin/roundoff 检查 | uint64_t 系数、coefficient-major 扁平缓冲；打包为 NTT digits，做精确卷积与模 N 系数归约。S4 的统一批乘入口见 [stage2_tree_gpu.cu:2133–2144](../tools/bench/stage2_tree_gpu.cu)，H2D、设备打包和 NTT 调用见 [stage2_tree_gpu.cu:2252–2284](../tools/bench/stage2_tree_gpu.cu)。 |
| F/G 树调度 | F/G 按层构造；F 树计划可重用 | F 树及每层同 shape 节点在 host 聚合后调用 poly_mul_batch_modN；数据在 host 扁平数组与 GPU scratch 间搬运。见 [stage2_tree_gpu.cu:2389–2485](../tools/bench/stage2_tree_gpu.cu)。 |
| 巨点与 G block | mQ_next_array 提供一段 giant 点，再构造 G | chunk/chain 产生巨点并构造 G 树；对可逆 Z 的段用 projective 叶子，整段只求一次逆并把尺度补偿进 H，遇到不可逆段回退到 affine 路径。见 [stage2_tree_gpu.cu:6620–6635, 6720–6850](../tools/bench/stage2_tree_gpu.cu)。 |
| H fold | 三次专用 multiply/FMA，使用高/低半积 | CPU 侧按精确系数实现三乘法折叠：完整 G·H、对 reciprocal 取商、减 qF 得余数。见 [stage2_tree_gpu.cu:6870–6895](../tools/bench/stage2_tree_gpu.cu)。 |
| Descent | 分层 scaled remainder，必要时切片并重建/读入 Ftree；两个 child remainder 可共享输入变换 | 默认是主机驱动、同层按除法形状分组的批量下降；节点余式递推可见 [stage2_tree_gpu.cu:2912–3055](../tools/bench/stage2_tree_gpu.cu)，Newton 倒数与两次乘法的批量 divmod 见 [stage2_tree_gpu.cu:5551–5605](../tools/bench/stage2_tree_gpu.cu)。S5 设备下降由 NTT_S5_ON 显式开启，默认关闭，见 [stage2_tree_gpu.cu:6975–7045](../tools/bench/stage2_tree_gpu.cu)。 |
| 工作集策略 | 成本模型纳入保存 Ftree 到磁盘、压缩 F/R、workspace 及 L2 影响 | NTT arena 复用、驱逐其它形状缓存；单次批乘按显存 budget 切 chunk。Ftree 目前需供 descent 访问；全局合并 G batch 会增加峰值缓冲。见 [stage2_tree_gpu.cu:2202–2233](../tools/bench/stage2_tree_gpu.cu)。 |

### 27.4 当前实验状态：已完成项与当前热点

开发日志记录的生产形状为 D=1,231,230、P=115,200，14 个 G blocks、13 次 fold，Stage 2 elapsed=273.55 s；其中 gtrees 101.142 s、fold 35.285 s、descent 76.458 s，NTT 合计 159.067 s（58.1%）。旧 D=570,570 记录为 397.83 s，因子集合相同；变化是 −31.2%。arena 驱逐与成本模型 D 搜索已落地，不应当再列成“待实现优化”。见 [DEV_GPUOWL_NTT_NOTES.md §19–20](DEV_GPUOWL_NTT_NOTES.md)。

日志 §26 在小 B2（5e6）下测的是 P=115,200 的 **F 树**：17 层各约 0.26–0.75 s；一层 57,600 次乘法约 0.75 s，而顶层仅 1 次乘法也约 0.71 s，作者据此估算约 80% 是与节点数弱相关的每层成本。把相近层数外推到 G 树得到约 84 s，与实测 101 s 同量级，但这仍是外推，不是大 B2 G 树逐层实测。日志提出下一步在 B2≈1e11 下打印 G 树自身阶梯，并拆 poly_mul_batch_modN 的 host pack、H2D、forward/inverse、exact reduce/check、D2H 与同步时间。见 [DEV_GPUOWL_NTT_NOTES.md §26](DEV_GPUOWL_NTT_NOTES.md)；树层计时点见 [stage2_tree_gpu.cu:2417–2485](../tools/bench/stage2_tree_gpu.cu)。

### 27.5 结合 Prime95 Poly 设计的优化候选（按优先级）

1. **先定位每层固定成本，再扫批次预算。**这是当前最大且有数据支撑的机会：在大 B2 下分别记录 F/G 每层节点数、shape group 数、batch chunk 数、host 打包、H2D/D2H、forward/inverse、精确归约/check 与同步。现有 NTT_S4_BATCH_MB 可改变单次 GPU 批量预算（默认值见 [stage2_tree_gpu.cu:1239, 2219–2249](../tools/bench/stage2_tree_gpu.cu)）。固定 D/N/参数扫预算，确认 chunk round-trip 是否造成地板；放大预算可能减少启动和复制，但会抬高显存峰值，必须同时记录 arena_mb/overflow 与 exactness 检查。
   **（当天已执行：见 [DEV_GPUOWL_NTT_NOTES.md §27](DEV_GPUOWL_NTT_NOTES.md)。结论是分块数确实是杠杆，但**不是**传输量——三次跑的 H2D/D2H 体积逐字节相同，只有分块数变；省下的是每个分块约 2 ms 的固定开销。默认预算已 32 → 64 MB，96 MB 起实测 OOM，上限由显存而非代码给出。）**
2. **把 Prime95 的“一个父项、两个余式子项”共享工作映射到 GPU。**Prime95 的 polymult_several 对同一父余数乘两个 sibling divisor，共用一份输入的 FFT。当前 host descent 虽会按 (deg dividend, deg divisor) 组批，但 divmod_batch 内部仍有 reciprocal、quotient multiply、q·B multiply 等多阶段安排。可探索把同一层 sibling 的反转/倒数数据、输入变换和 workspace 做成多输出批任务，减少重复 pack、NTT、归约与 launch；先对比所有叶值再谈吞吐。
3. **重新设计 S5 分组除法，不能复用已回退实现。**开发日志 §21–23 显示逐节点 S5 在 D=30030 上约 2.542 ms/除法，慢于 host batched 路径约 330 µs/除法；按同层形状分组曾得到 1.9× speedup，却出现结构化叶值错误并完全回退。下一版应按层对比 quotient/remainder 和每个叶值，重点检查每 slice 的 B 指针、hook 输出步长、frontier/code 连续性；只有在 P=24/240/2880 及生产 P 上逐叶一致、且 A/B 更快后再启用。生产形状是否受益不能从小形状推断。见 [DEV_GPUOWL_NTT_NOTES.md §21–23](DEV_GPUOWL_NTT_NOTES.md)。
4. **下降层内 CPU 并行是近期低风险候选，但要先确认瓶颈归属。**同一层节点独立，日志按 76.5 s 估出约 35 s 的理论节省；实际代码共享 PolyLayer、batch buffers 与 GPU stream，需要先把线程私有 scratch、批次形成和 device 同步路径分清。用逐层 hash/leaf 对拍防止只比总 GCD 掩盖错值。预计收益是日志估算而非本次验证结果。见 [DEV_GPUOWL_NTT_NOTES.md §24](DEV_GPUOWL_NTT_NOTES.md)。
5. **给 baby ladder/归一化单独计时，再考虑批量逆元或 projective 表示。**日志从进程 wall 与已知计时器差额估计 baby ladder 加 affine 约 30 s，但当前没有独立计时；日志估计每段一次逆可能节省其中约 7 s，置信度低于已计时的 tree/descent 热点。巨点分段 projective 补偿已有实现，不等于 baby 侧可以直接照搬：要证明缩放因子对 F、reciprocal、fold 与最终 GCD 全程一致。
6. **D/B2 与跨 G-block 融合留到固定成本下降后重新搜索。**D 成本模型 + arena eviction 已让生产形状从 397.83 s 降到 273.55 s；继续增大 D 能减 G blocks，但增大 P、Ftree、reciprocal 和 descent 工作。跨 block 融合有显存代价；应在 per-level floor 消除后重新做 D optimum 搜索，而非预设“更大 D/全合并更快”。

### 27.6 与 PrMers 对比的边界与 benchmark 口径

PrMers 的 BSGS 路径是逐 prime cross-product 扫描，与 Prime95/本仓库 Poly 路径是不同算法映射。它适合说明小工作集驻留寄存器、巨点在线推进、减少表流量等思路，不能直接推断 Poly Stage 2 的复杂度或 CUDA NTT 性能。Prime95 日志和当前 GPU 数据对应不同 CPU/GPU、不同模数/边界/构建；本报告只作代码路径与瓶颈形态比较，不把秒数当同参数 A/B 结论。

## 28. 当前性能差距：日志计时边界与算法工作量复核（2026-10-03）

本节以当前起点 `8b5b57f` 和本机日志/源码重新核对。§27 中性能记录和源码行号是 10-02 的历史快照；
当前预算、归约和 direct pack 已经过多轮修改，最新证据见 DEV_GPUOWL_NTT_NOTES.md §37–38。
本轮目标仍是 CUDA ECM Poly Stage2，同时减少 RAM/VRAM；本次并未证明超过 Prime95 单核。

### 28.1 首先统一比较口径

- 日志为 [p95v3104/screen.log](D:/code/GIMPS/p95v3104/screen.log:1)，源码为本仓库的 v31.06b01。
  两个版本不能当作同一次构建；以下源码用于核对机制，最终同参数比较需记录精确 binary、参数和线程数。
- [screen.log:11](D:/code/GIMPS/p95v3104/screen.log:11) 将 worker1 绑定到 logical CPU1，
  [screen.log:86](D:/code/GIMPS/p95v3104/screen.log:86) 明确为 polymult helper 绑定 logical CPU3。
  **单个 worker 不等于单线程。**不能把这份 M3613 秒数宣传为单核成绩，也不能仅凭 logical CPU 编号判断物理核拓扑。
  [ecm.cpp:6839](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:6839) 的
  stage2_threads=stage1_threads+Stage2ExtraThreads，并在 [ecm.cpp:7852](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:7852) 交给 polymult。
- `Stage 2 init complete` 在 [ecm.cpp:9335](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9335) 结束并清掉 init timer/count；
  [ecm.cpp:9351](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9351) 重新开始主体 timer；
  [ecm.cpp:9844](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9844) 的 `Total time` 是主体到 GCD 前，
  不是包含 init 的全 Stage2，也不是仅 giant 主循环。最终 GCD 另计。
- M8317 worker3 curve456 的完整窗口为 [screen.log:8195](D:/code/GIMPS/p95v3104/screen.log:8195)：
  B1=110000000，actual B2=496623617490，D=510510，degree=46080，2038 MB、Ftree cache budget=7，FFT=512。
  init **3.214 s**，主体 **71.386 s**，相加 **74.600 s**（不含另计 GCD）；日志没有证明这条曲线只有一个执行线程。
  PolyG 约 2.1 s/块、PolyH 约 1.35 s/更新，末尾有 scaled H 与 F up/down。
- 本仓库 §37 的对照为 **5261-bit Mersenne 模数 M5261（GPU 走通用奇数归约）**、B1=1000、B2=1.94e12、D=1231230、P=115200、GPU1。
  模数、曲线参数、边界、degree、giant block 数、硬件和线程配置都不同；208 s 与 74.6 s 不能直接相除后叫加速比。
  Prime95 的 transforms 是 Gwnum 变换计数，也不能直接和 GPU wrapper/NTT launch count 相比。
- GPU 的 CLI elapsed 从 [stage2_tree_gpu.cu:9170](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9170) 的 run_batched 前开始；baby/Ftree 已在此前建立，Ftree 时间另列 f_tree_incl。进程 wall 还包括 stage1、设置和自检。因而当前 elapsed 也不能直接作为完整 Stage2 wall；公平基线需要单独划定 stage1/Stage2 分界。

### 28.2 性能优势分层，而非只看 FFT 核心

**A. 特殊模数算术。**日志的 M3613/M8317 是 Mersenne 对象，而当前 GPU 生产 fixture 是 M5261，但设备算术仍走任意奇数的通用归约。
[gwnum.h:96](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/gwnum/gwnum.h:96) 描述 K*B^N+C 的专用 setup；
[gwnum.h:105](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/gwnum/gwnum.h:105) 对 generic modular setup 给出更高成本说明。
该注释不能当作本机 ECM 的实测 3× 比例，但足以说明 Mersenne 路径与本仓库通用 long division 不公平等价。
本仓库精确 Kronecker NTT 的槽宽包含乘积位宽和累加界，随后逐系数做多 limb 模 N 归约；
这是当前精确通用路线的算术成本，不能靠删 host 副本完全消除。

**B. 选择所需输出、融合算术与复用存储。**[ecm.cpp:9465](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9465)
复用 polyGH 的上下半；随后 quotient 只返回高半，remainder 通过 MULLO+FNMADD 融合差法得到。
[polymult.h:128](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/gwnum/polymult.h:128) 还包含隐含 monic、MULHI/LO/MID 和 FMA 接口。
当前 GPU fold 在 [stage2_tree_gpu.cu:8074](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8074) 附近仍是三次完整 cp_mul，
生成完整 qrev/qb 再截断/做 host 差法。只剪 D2H 不等于截断 convolution；但 GPU 先只归约/回传所需窗口可降低具体工作量，
之后再实现高/低/中间乘积，才能进一步减少 transform 范围。

**C. scaled descent 是实际算法差距。**[ecm.cpp:9542](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9542)
是 H×R 的高半，不是常数 Γ 去缩放。之后每个父项对两个 sibling 构造 circular+MULHI 的 scaled 子项，
[ecm.cpp:9706](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9706) 描述两份输出，
[ecm.cpp:9729](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9729) 用 polymult_several 共用父输入变换。
[polymult.c:4904](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/gwnum/polymult.c:4904) 明确输入只读和 FFT 一次。
当前 GPU 默认下降调用 divmod_batch；每组先反转 divisor 并跑 inv_series_batch Newton，再 qrev、q·B、host remainder。
见 [divmod_batch](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:6739)、
[inv_series_batch](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:6640)。
因此当前实现与 Prime95 并非只是同一下降算法的设备位置不同；直接套一个 transposed 子式到普通 remainder 上不成立。
必须定义完整 scaled 状态、索引/反转/monic 约定、初始 scaling 和最终叶值，逐层对拍，不能凭相同 GCD 证明正确。

**D. 表示与工作集生命周期。**Prime95 NO_UNFFT/NEXTFFT 保留 Gwnum 系数的变换状态，SAVE/USE_PLAN 保留同形状计划。
这与多项式的 PRE_FFT 不是同一个层次。尤其 [ecm.cpp:9283](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9283)
默认 ECMPolyCompress=1 只做预处理/预转置；PRE_FFT 仅隐藏选项 -1/-2，PRE_COMPRESS 另由 2/-2 开启。
不能把默认性能归因于已经缓存了 F/R 多项式 FFT，或看到 `Poly compress` 就断言启用了 PRE_COMPRESS。
Prime95 在下降前释放 F/R，按 slice 读入/重建 Ftree，并预先回收工作内存；缓存预算见
[ecm.cpp:5857](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5857)；缓存单位是 poly_size 个 gwnums 的容量，
不是 GPU batch chunk 数，也不能解释为只存两棵节点多项式。
GPU 目前保留 host word vectors、CPoly、arena 和多种 pools，并反复 materialize/pad/上传输入。

### 28.3 本轮行动与后续验收

先实施 §37 候选的 flat input 借用：已满足 P stride 且不与 out 别名时直接传 const view，
只为短输入和别名构造副本；有同二进制开关、完整 GMP alias/stride 门禁与 byte/time ledger。
此改动减少 host 准备，不宣称减少 NTT、设备传输或通用 modular reduction 工作量。

下一阶段优先选择可验证的 output-window 接口和 scaled descent 原型，而非继续仅增大 chunk：

1. 固定反转、scale、monic 和叶值契约，用独立 GMP 算法核对每层 scaled 输出及最终 H(x_j)。
2. 将 ordinary-divmod 与 scaled 路径并存，以相同 F/H/N 比较工作量、全部叶值、因子和时间。
3. 做 output window 后再设备驻留：统一 arena/pool/spectrum 生命周期和峰值预算；一个大 spectrum 已占 1 GiB。
4. 若 host prepare 改善却 GPU 空闲不降，先用 Nsight Systems 分辨 CPU 准备、CUDA API 等待和 stream gap；
   Nsight Compute 用于随后确认的热点 kernel。剖析运行与无 profiler 的性能 A/B 分开。
5. 建立 Prime95 真正单执行线程、同模数/边界/输入点或同曲线参数的基线，同时报告 full Stage2 wall、stage1 分界、
   RAM/VRAM 峰值、曲线失败/因子口径和重复性。达成这个基线并超越前，长期目标保持未完成。

### 28.4 2026-10-03 后续：根节点消费者与整树消费者

Nsight 短窗口和本轮代码细节见 [DEV_GPUOWL_NTT_NOTES.md §39](D:/code/MPA-OpenCl/docs/DEV_GPUOWL_NTT_NOTES.md:2126)。
观察到的长间隙有许多没有 CUDA API 覆盖，但采集完整性有警告、没有 CPU stack sampling，不能证明具体准备函数的因果。
当前先明确对象寿命：[run_batched](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8048)
的 G 块只消费根，而 F 树及 [run_stage2_tail](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:6948)
的 G 树需要完整下降节点。因此 root-only 只用于前者，每层消费完释放 children，构造后释放重复叶输入。
这借鉴的是 Prime95 对工作集寿命的管理，算法仍是普通 product/remainder tree，未实现 scaled descent、
poly spectrum 驻留或 MULHI/MULLO。独立 GMP fixture 还覆盖常数项为 1 的非恒等多项式，
修复此前 tree/descent 将 `X+1` 误判为常数 1 的边界；完整同二进制门禁和生产结果在 §39 记录。

最终门禁 **110 passed / 0 failed**。本形状同二进制 ABBA 的 CLI elapsed 均值
**220.41 → 198.22 s（−10.07%）**，观测进程私有提交峰值均值 **10734 → 9315 MiB（−13.22%）**；
arena 仍 **6215.6 MiB**。full 两轮波动 12.04 s，root 两轮 0.46 s，因此不把该比例外推到其他形状。
主循环约 1 Hz NVML mean load **53.48 → 62.29%**，低负载样本占比 **25.61 → 16.67%**，下降阶段仍有大量间隙。
这个进展没有减少通用 mod-N 的理论算术、NTT 系数数或回传量，也没有达成与 Prime95 单执行线程的公平速度比较。

### 28.5 2026-10-03：NTT scratch 容量复用与剩余算法差距

后续已接入 [跨 shape 的 A/B/Q 工作区](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1729)，
默认 `NTT_ARENA_WORKSPACE_POOL=1`，`=0` 同二进制 keyed 对照。
仅共享 default-stream 临时 scratch，dRes 保持按 shape 隔离，borrowed digits 导出保持 keyed lifetime；
别名输入在缓存查询前保存，失败回滚和 actual deferred-carry 状态均有门禁。
完整过程、代码行号和证据见 [开发日志 §40](D:/code/MPA-OpenCl/docs/DEV_GPUOWL_NTT_NOTES.md:2279)。

完整门禁 **116 passed / 0 failed**。GPU1/M5261 生产 ABBA：
CLI elapsed 均值 **205.61 → 191.01 s（−7.10%）**，进程 wall **240.762 → 225.590 s（−6.30%）**；
A/B/Q 分配 **1416 → 21 次**、缓存驱逐 **36 → 0 次**，两次候选均重复缓存账本。
NVML 整卡采样显存峰值 **5989 → 5759 MiB**；A/B/Q ledger 峰值 **3654.6 → 3072.0 MiB**。
三类 tracked payload 峰值 **3656.7 → 3339.0 MiB** 不含 FuseCtx 自身的 scratch/tile tables，不能当作总 VRAM。
主循环低负载样本占比 **18.08 → 18.22%**，频繁空闲仍在；本轮未减少 H2D/D2H、完整 NTT 或归约数量。

这推进了工作集生命周期管理，仍没有实现 Prime95 的 MULHI/MULLO、融合 remainder 或 scaled descent。
下一轮先明确 fuse 预算拒绝分支的临时对象 owner 和 release，补齐全部表/scratch 账本及重复 fallback 门禁；
随后在完整 NTT/carry 上实现必要输出窗口，降低归约、回传与输出缓冲，再做 scaled descent 数学原型。
Prime95 单执行线程、同模数/边界/曲线或输入点的完整 Stage2 基线仍缺，不能据此宣称已超越 Prime95。

### 28.6 2026-10-03：完整 FuseCtx 账本与实际 scratch 容量

已修复预算拒绝后临时 FuseCtx 的释放：
[FuseCallGuard](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1345)
按实际 owner 覆盖 host/device batch、single host 和提前返回，缓存借用仍由 arena 释放。
[fuse_init](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1266)
遍历 forward/inverse 实际 pass 取最大 scratch 容量，默认 `NTT_FUSE_COMPACT_SCRATCH=1`，`=0` 保留宽容量对照。
M=1 仍给足 N/2；纯 tile 不分配没有消费者的 coarse/radix scratch。
[base ledger](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1197)
和 arena full 字段补齐此前遗漏的 mandatory tile tables/scratch；full 仍不包含外部 Stage2 pools 与驱动开销。

完整门禁 **122 passed / 0 failed**，包括独立 GMP DFT 的 **72 cases / 1585152 words** 和宽/紧容量寿命守恒。
同二进制 M5261 生产 ABBA 的 CLI elapsed 均值 **190.600 → 191.910 s（+0.69%）**，没有证明加速。
完整 arena 缓存 payload 峰值 **4508828064 → 3629192080 bytes（−19.51%）**；
NVML 显存 **5759 → 4903 MiB**，观测进程私有提交峰值均值 **9213 → 8360 MiB**。
主循环低负载样本比例 **17.99 → 19.01%**，H2D/D2H 仍 **38.81/37.27 GiB**，空闲与传输量未解决。
证据和范围见 [开发日志 §41](D:/code/MPA-OpenCl/docs/DEV_GPUOWL_NTT_NOTES.md:2456)。

这一步缩小工作区并修复所有权；Prime95 MULHI/MULLO、scaled descent 和父输入变换复用的算法差距仍在。
下一步先实现保持完整 NTT/carry 的输出窗口，减少必要 mod-N 归约与回传，再验证 scaled 状态的下降原型。
隔离 Prime95 单线程配置已准备，但启动验证未进入 ECM；实际二进制版本 31.7.1.0 与参考源码 31.6b1 不同。
目前仍没有可比较的 CPU Stage2 时间，完整 Stage2 计时边界和匹配曲线的公平基线继续作为必要验收。

### 28.7 2026-10-03：chunk 输出之后的重复 D2H

输出窗口审计发现 [poly_mul_batch_modN](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2877)
在完整 chunk 回传/drain 后，再整批回传 C.d_out 并重新写 out。
默认 `NTT_S4_FINAL_READBACK=0` 跳过这次复制，1 恢复对照；
[实际分支](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3297)
单独统计额外回传，最终全字指纹覆盖两条路径返回给消费者的结果。
这减少相同数据的重复搬运，普通 remainder 算法、NTT/carry/归约数量保持一致。

完整门禁 **135 passed / 0 failed**。Nsight 小型夹具的 D2H 恰少 **36 次 / 9520 bytes**，
与软件 ledger 完全吻合，H2D/kernel counts 与最终输出不变。
M5261 生产 ABBA 均值 **192.450 → 182.045 s（−5.41%）**，进程 wall **227.1965 → 216.480 s（−4.72%）**；
candidate 两轮差 11.21 s，control 差 0.68 s，收益限定于本形状/环境。
移除 **38.579 GiB** 额外 D2H（所有 S4 调用含 F-tree），主 batched chunk D2H 仍 **37.27 GiB**。
局部临时 host payload 峰值 **218.85 MiB →0**，但 NVML 仍 **4903 MiB**，
进程私有提交峰值均值 **8359.5 → 8406.5 MiB**，整体内存峰值改善未获证明。
主循环低负载样本 **17.72 → 19.74%**，频繁空闲仍在。
计时口径、传感器、代码与原始日志见 [开发日志 §42](D:/code/MPA-OpenCl/docs/DEV_GPUOWL_NTT_NOTES.md:2577)。

隔离 Prime95 改为无参数启动后成功完成 M4423/sigma26 的 Montgomery/poly smoke；
requested B2=5e6 实际扩展为 5187000，短任务未验证 OS 线程并发，未与 GPU 匹配边界。
启动障碍已解决，公平单执行线程 full Stage2 基线仍未建立。下一步将 chunk device 输出寿命与
first/count 窗口一起定义，再减少必要归约/回传；scaled descent 和 spectrum 驻留仍是更大的算法候选。

### 28.8 2026-10-03：输出窗口已接入，总速度收益未获证明

[NttReduceHook](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3364)
支持 first/count，完整输入/NTT/carry 保持，输出 slice stride 使用 count。
[设备归约](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1685)
先检查全部源槽 canonical 上界，再跳过窗口外 mod-N MAC；
[独立 GMP snapshot](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2721)
从原始 first+compact_k 读 digits，按紧凑 stride 读输出。
Newton inverse、反转商和 q·divisor 的消费者改为请求必要前缀，支持短乘积补零和空输出。
任意窗口（含非零 first）的接口为后续 scaled descent 中间区间输出作准备。
目前尚未实现 Prime95 MULHI/MULLO 的 NTT 工作量缩减或共享父 spectrum。

完整门禁 **160/0**，短/83-limb 独立 GMP 窗口、树/下降验证通过。
Nsight 小夹具减少 **120 次 /26240 bytes D2H**，与 ledger 一致，H2D/kernel count及最终输出相同。
M5261生产同二进制 ABBA 主计时均值 **173.555→173.905 s（+0.20%）**，
进程wall **208.092→208.099 s**，未证明总时间改善；归约 **18.6475→16.6910 s（−10.49%）**，
全部S4输出少 **9.106872 GiB**，局部设备输出容量少 **72.949 MiB**。
整体NVML峰值仍4913 MiB，GPU频繁空闲未解决；`NTT_S4_OUTPUT_WINDOW=1` 保持可选，默认0。
同二进制full对照也已使用紧凑主机前缀，不能与上一轮历史数据直接相减归因。
详情、覆盖边界、原始日志和下一轮数学契约见
[开发日志 §43](D:/code/MPA-OpenCl/docs/DEV_GPUOWL_NTT_NOTES.md:2710)。

Prime95单逻辑CPU资格运行已完成：31.7.1.0、M4423、Montgomery sigma26、B1=1000，
requested B2=1.94e12，实际扩展至 **2011326186870**；D=1531530、degree138240。
init11.277 s、main107.943 s，报告完整Stage2 **119.220 s**，无因子。
229个样本均为进程affinity mask8388608（仅逻辑CPU23），主要计算由一个worker承担。
包含原有Prime95后台竞争，仅一轮资格记录；GPU尚未匹配 N/实际B2与完整Stage2边界，
**不能将本轮M5261时间与119.220 s当作公平速度比，也未达到长期目标。**

### 28.9 2026-10-03：chunk复用与匹配Prime95 Q后的实际差距

本小节记录提交83470ad时的结果；其生产Stage1待修问题已由28.10后续阶段解决。

[poly_mul_batch_modN](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3006)
改为按outer chunk请求输出设备容量，默认 `NTT_S4_CHUNK_OUTPUT=1`，0保留整批对照。
各chunk复用C.d_out起点；oracle快照/D2H在同一默认stream中先于下一次覆盖排队。
final_readback=1自动回到整批布局。完整门禁 **188/0**；
Nsight小夹具全部输出/传输/kernel相同，输出576→264 bytes，CUDA malloc97→95。

M4423生产同二进制ABBA：输出容量 **184.570→123.047 MiB（−61.523 MiB）**，
完整Stage2均值 **135.759394→135.855210 s（+0.07%）**，未证明加速。
整体NVML峰值仍4945 MiB，下降阶段4731→4669 MiB；H2D/D2H仍33.60/32.27 GiB。
这一步是局部容量优化，数据往返和乘法数量尚未减少。

复核既有约定：[Prime95 Montgomery Stage1](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:7463)
包含额外2²·3，指数为12·lcm(1…B1)，此前仅对齐sigma/B1并未对齐Q。
实际31.7.1.0 exe导出完整Q后，独立整数ladder和GPU `NTT_STAGE1_EXTRA=12` 均逐字匹配，
M4423/sigma26/B1=1000的规范hex SHA256为
`33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`。
默认extra1保留原有冻结测试，匹配比较显式使用extra12。
生产 `exponent=choose12` 已在ini/CLI和CPU Montgomery实现；当前GPU批量builder仍固定torsion1。
试接GPU已有选项时暴露大位宽短指数Q错误，已在原exe的M4423/B1=4/lcm复现，根因待定位。
试改已撤回；探针的Q对齐与188/0门禁不能代表这个生产CGBN问题已解决，详见开发日志§44.7。

新CPU两轮全进程affinity仅逻辑CPU23，1 worker、Stage2ExtraThreads=0，
Montgomery/poly、AVX-512 FFT256、Memory2048 MiB，actualB2=2011326186870。
完整Stage2 **90.336 /90.584 s**，均值 **90.460 s**；
GPU使用相同M4423/Q/B1/实际B2，完整均值 **135.855210 s**，耗时比 **1.501826**。
GPU仍多50.18%，尚未超过CPU单逻辑核；不同D/degree/内存预算及检查机制必须保留在解释中。
这轮不与带原有Prime95竞争的119.220 s历史资格时间直接作优化归因。

下一候选为Prime95的scaled descent：
[root MULHI](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9540)和
[sibling polymult_several](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9729)
避免本仓库逐节点Newton inverse/qrev/q·B，随后共享父spectrum并保持device系数驻留。
新增 [普通系数整数原型](D:/code/MPA-OpenCl/tools/test/test_stage2_scaled_contract.py:1)
已验证480cases /10176states /4176真实叶值，无错误；它尚不是CUDA/GMP实现或生产性能证据。
详细实现行号、计时边界、证据和推进顺序见
[开发日志 §44](D:/code/MPA-OpenCl/docs/DEV_GPUOWL_NTT_NOTES.md:2870)。

### 28.10 生产Stage1基线修复（2026-10-03，延续28.9）

28.9记录的GPU choose12接线与Q问题现已修复：原CGBN WMAD乘法只修正radix进位，
可返回≥N；删除归一化破坏了梯子的加/减范围前提，原生产exe的24个短指数回归有11个Q错误。
当前 [规范乘法包装](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:351)
恢复 `[0,N)`，param2乘2也归一化；
[GPU指数调用](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:2828) 使用现有exponent选择torsion1/12。
完整生产回归66cases/94Q/0失败，覆盖param0/2/3、TPI4/8/16/32、ini/worker和checkpoint；
CUDA checkpoint升为5，旧v4重新计算。生产M4423/sigma26/B1=1000/choose12完整Q已与实际Prime95导出相等。
代码原文件行号、轨迹、二进制hash和证据见开发日志§45。
这次未修改poly Stage2或获得新性能数据，完整135.855210s对CPU90.460s的差距仍待scaled descent等后续优化解决。

### 28.11 Prime95 scaled descent 的 CUDA 首次接入（2026-10-03）

[descent_scaled](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:6587)
实现根scaled状态及每孩子一次sibling反转多项式乘法，通过 `NTT_SCALED_DESCENT=1` 启用，默认0保留对照。
参考Prime95原文件的 [root MULHI](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9540)、
[sibling polymult_several](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9729)。
根复用fold已有finv，非平凡孩子取乘积中间窗口，去掉原逐节点Newton inverse、商和乘回。
普通系数合同、非二次幂/零度padding/短H/根degree=P边界及全部原文件行号见开发日志§46。

当前仍是host frontier + GPU完整Kronecker线性NTT，没有Prime95的共享父FFT、CIRCULAR或NO_UNFFT。
NTT素数p不同于ECM模数N，层间仍需carry和modN归约；device驻留应保留归约后的规范word，
同层两个孩子可共享父forward spectrum。独立GMP逐节点、Horner逐叶门禁 **41/0**，
2550夹具/54060节点状态/22185叶；原有回归 **188/0**。最终注释整理重编译也重复41/0。

M4423/相同完整Q/B1/实际B2的同二进制ABBA，两边OUTPUT_WINDOW=1、CHUNK_OUTPUT=1：

- 完整Stage2 **125.054669→107.862620s（−13.75%）**；下降 **26.2565→8.6445s（−67.08%）**。
- main H2D **33.60→26.16GiB**，D2H **24.49→19.90GiB**；全部115200叶指纹一致，无因子。
- 整卡NVML峰值均4933MiB，private commit观察峰值均值7932.5→7934.0MiB，整体内存收益尚未证实。
- 与既有匹配Prime95单逻辑CPU完整均值90.460s相比，仍多 **19.24%**；GPU使用D1231230/P115200，
  CPU使用D1531530/degree138240、AVX-512 FFT256/Memory2048MiB，检查机制不同。

候选Gtrees39.967s/fold17.5445s/giant14.550s几乎未变；下降已只占main约9.2%。
下一轮优先G-root逐级device驻留，根返回host一次供fold，随后共享scaled父FFT并减少host padding。
单独消除全部下降也不足以超过当前CPU基线。
源码、计时边界、二进制指纹、逐轮结果、NVML原始证据和下一步门禁要求见
[开发日志§46](D:/code/MPA-OpenCl/docs/DEV_GPUOWL_NTT_NOTES.md:3178)。
