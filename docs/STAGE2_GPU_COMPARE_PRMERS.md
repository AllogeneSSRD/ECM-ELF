# GPU Stage 2 实现对比：本仓库与 PrMers

**调查日期：2026-10-02**  
**范围：ECM Stage 2（Montgomery x-only）**；本报告不把 PrMers 的 P−1 V-trace Stage 2 当作同一实现。  
**方法：**静态阅读本仓库源码、PrMers 子树源码及 `docs/DEV_STAGE2_*` 开发记录。这里的性能数字引用仓库已有记录，不是本次重新跑出的 benchmark。

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
