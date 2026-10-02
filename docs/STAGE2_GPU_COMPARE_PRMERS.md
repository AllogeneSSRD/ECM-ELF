# ECM GPU Stage 2：本仓库树形实验与 Prime95 Poly 方法对照

**调查日期：2026-10-02**  
**范围：ECM Stage 2；主线是本仓库 CUDA 多项式树实验与 Prime95 Poly 方法。** PrMers Gaussian ECM BSGS 仅作补充对照；PrMers 的 P−1 V-trace 不属于椭圆曲线 Stage 2。
**方法：**静态阅读本仓库、Prime95 与 PrMers 源码，以及 docs/DEV_GPUOWL_NTT_NOTES.md。性能数字只引用开发日志已记录的数据；本次没有重新运行 benchmark。Prime95 ECM 源文件实际位于 .refactor/p95v3106b01.source/ecm.cpp（不是 ecm/ 子目录），多项式乘法实现位于同版本 gwnum/polymult.c/.h。

## 本次补充摘要（ECM Poly Stage 2 主线）

- Prime95 Poly 的主算法是 F baby-root 积树、按 giant block 构造 G、递推 H ← G·H mod F，再沿 F 树做 Bernstein scaled remainder descent；Prime95 原码流程与当前 CUDA 树版逐段对应，详见 §27.1–27.3。
- 当前 GPU 路线的主要实现差别是算术/数据通路：Prime95 用 Gwnum 浮点 FFT 与 roundoff guard；本仓库使用精确系数 NTT、设备打包与模 N 归约。树形算法相似不代表底层访存成本相同。
- 最新生产记录：D=1,231,230 / P=115,200，elapsed 273.55 s；成本驱动 D 搜索与 arena 跨形状驱逐已带来 −31.2%。此后优先检查 gtree 每层固定成本；F 树逐层数据已显示 0.26–0.75 s 的层成本地板，但高 B2 的 G 树仍需单独测量。
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
