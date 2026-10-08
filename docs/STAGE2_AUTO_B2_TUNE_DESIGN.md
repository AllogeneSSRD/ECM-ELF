# GPU ECM Stage2：Auto B2 与 tune 的适用性及实现设计

日期：2026-10-05。依据当前仓库 `17f023a`、本地 Prime95 31.06 源码及 `.refactor/gpuowl` 中的 PRPLL 源码。本文是源码调查与设计，**没有实现新 CLI、修改生产算法或新增性能实验**。引用均为本次调查时的原文件和一基行号；本地参考树可能包含修改，不将其等同于最新上游版本。

用户已确认默认目标：**Prime95 式总流程收益，即连续生成并处理新曲线时的单位时间收益；读取 Stage1 save 时仍计入估算的 Stage1 成本。** 本文据此完成首版方案。有限存档的时间预算模式作为后续扩展。

后续状态：用户已授权开始实现，P0基础与P1最小版见 [第一轮实现说明](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_TUNE_IMPLEMENTATION.md)。本文“当前不可执行/未实现”等描述记录调查时的状态；已接入的命令以实现说明及CLI文档为准，Auto B2本身仍未接入。

## 1. 结论与适用范围

两项功能都适用且可实现，推荐顺序为 **统一规划接口与内存计账 → tune → Auto B2**。

- Auto B2 的基础已经部分存在：当前程序会选择 D，已有多阶段成本模型、真实 NTT shape 查询和不同位数的显存实验。缺少的是通用的标定数据库、Stage1 成本输入、收益函数及联合搜索。
- tune 的基础更完整：已有纯卷积计时、按 D 采集完整 Stage2、拟合、离线排名及冻结来源检查。应整合这些能力，避免另写一套 NTT 或算术实现。
- 最大短板是**模型覆盖范围**。当前生产 D 模型严格限定 M4423、B1=1000、RTX4060 Laptop 和特定路径；2203/4423/8191 bits 的预算实验使用统一 legacy 策略控制 D，并非三个位数的最优 D 标定。
- Auto B2 必须同时考虑驻留与非驻留路径。owner 超预算只使该路径不可用，不意味着整个候选 D 不可用。
- 本仓库 Stage2 是 Goldilocks NTT 加整数 packing/归约。PRPLL 的浮点 FFT 最大 bits-per-word、roundoff 调优不能直接移植；可复用的是测量、配置筛选和持久化方法。

首版范围建议保持当前生产输入合同：有效 Suyama PARAM=0 Stage1 save、当前支持的模数/位宽及必需检查。针对 Mersenne 快路径标定的数据不能自动用于通用奇模数。

必要性方面，持续生产且N/B1/设备/预算会变化时，Auto B2能省去反复人工试界限；单一固定任务仍可显式B2。tune既用于选配置，也用于让D/B2预测有可靠秒数依据。它本身不保证内核更快，收益可能来自选择更合适的D或计算更多但更有价值的Stage2范围。

调优投入可用 `n_break_even = ceil(T_tune / (T_fixed_plan − T_tuned_plan))` 估计回本曲线数，前提是相同B2/工作合同且确有稳定节约。Auto B2会改变收益和工作量，应比较score，不能只比较秒数。频繁改内核时优先局部增量tune；长生产批次前再完成完整profile。

## 2. 术语与计量单位

- `N`：待分解整数；`S=bit_length(N)`，`W=ceil(S/64)`。
- `B1`：save 中完成的第一阶段界限；`B2>B1`：本次第二阶段请求界限。
- `D`：Stage2 步长；`P=phi(D)/2`：baby 集大小。
- 当前引擎 `I=floor(B2/D)+2`：giant 点数；`G=ceil(I/P)`：G 树批数；fold 次数为 `G−1`。见 [I 的生成](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10998) 和 [批次数](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9694)。这不是 Prime95 的区间/relocation 公式。
- `L=2^k`：NTT length，避免沿用 NTT 代码局部变量 `N` 而与待分解整数混淆。
- `q_GL=2^64−2^32+1`：NTT 域模数，与待分解的 N 无关。
- `D_eff`：概率工具的群阶有效除子/挠子收益参数，**不等于 Stage2 步长 D**。
- 显存/RAM 使用 bytes、MiB=2^20 bytes；时间使用秒，吞吐量必须明确一次 iteration 的内容。

## 3. Prime95 Auto B2 的实际算法

### 3.1 默认如何进入自动选择

[默认界限与开关](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:6847)：C=0 先变为 100B1；当 C=100B1、非 QA 且 `ECMBestB2=1` 时设置 `optimal_B2`。因此这份源码中显式给出 100B1 也可能触发自动选择。

[Stage2 规划入口](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:6111) 将当前可用内存转换为 gwnum 数量 `numvals`，在 MIDSTAGE 调用 `ecm_choose_B2`。内存来自 worker 的 [avail_mem](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:7871)，并有分配失败后的缩减。这是 CPU 主内存和 gwnum 工作区模型，不能把 MB 直接换成我们的 CUDA arena MB。

### 3.2 最大化的目标

[Kruppa 相对曲线价值](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:1431)：

```text
r = B2/B1 >= 1
a(B1) = 1.96617 − 0.06781 log10(B1)
K(B1,B2) = 0.11343 + 0.88657 [log10(r)/2]^a(B1)
K(B1,100B1) = 1
efficiency = K / (stage1_cost + stage2_cost + gcd_cost)
```

最终效率在 [ecm_stage2_cost](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5903) 计算。K 是经验相对收益，不是任意 N、任意未知因子大小的绝对成功概率。公式没有直接输入 N；N 通过 FFT 长度、gwnum 大小、GCD/逆元成本等影响分母。源码注释说明它根据 GMP-ECM 6.04 的研究修订，但未在这里给出所有拟合适用范围，首版应标记为 `kruppa_p95_v1` 经验模型。

重要区别：Auto B2 并不是最大化覆盖整数数目 `(B2−B1)/seconds`，也不是预设 `T2/T1=常数`。不同 B2 的收益增长是对数函数，成本增长则取决于算法和内存。

### 3.3 成本、内存与 N 如何进入

[Stage1 成本](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5923) 按第一阶段 FFT 次数估计：Montgomery 路径约 `25.55 B1`、Edwards 路径约 `21.95 B1`。这些是 Prime95 算术与乘法链的测量常数，不能用作本仓库 GPU ladder 的秒数。

[实现枚举](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5943) 带入 Stage1/Stage2 FFT 长度、线程数、N 的 bit length，比较 pairing/poly 等实现，再调用 `best_stage2_impl` 枚举 D。我们的首版只需要当前 poly 后端。

Poly 模型包括以下项：

- [内存约束](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5786)：4 或 6 块多项式加 pool 临时量，扣除 F/R 压缩收益，加入 `polymult_mem_required`；超过 numvals 则拒绝。
- [poly 计算速率](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5804)：每系数成本随 poly_size 的 log2 增长，再根据每线程工作区相对 L2 容量增加惩罚。
- [G 树、fold、初始余式](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5835)：按批数及 log2(poly_size) 计费。
- [F 树保存/重建](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5853)：多余内存可保留更多层，降低重建成本；磁盘保存还有转换费用。
- [换算和校正](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5889)：变换与 polymult 成本换算至 Stage1 FFT 长度，加入逆元，再乘比例校正和 Stage1/Stage2 线程比。

所以 Memory 同时改变“可行算法/形状”和“重建成本”，并非单纯设定最高 B2。我们的对应项是 NTT length/batch、table 缓存、owner 驻留、GPU baby/leaf 路径及 host staging。

### 3.4 B2 搜索方法

[ecm_choose_B2](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5997)：

1. 设置倍率上限 `MaxOptimalB2Multiplier`，默认随 numvals 增大，最多默认 10^7。
2. 起始探测倍率 50，随后按**实际覆盖端点倍率**乘 5 扩展；每个 B2 都重新选择 Stage2 实现/D。
3. 左右扩展到包围较好效率，再做类似二分的区间细化。
4. 返回实际端点对应倍率，而非原始试探倍率。

实际端点来自 [poly 批次补齐与 relocation](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5772)，并不总等于请求 B2。**不应照抄这部分端点扩展到我们的实现**；我们应记录当前引擎真正保证的区间，再决定收益函数使用什么端点。

这种搜索依赖包围峰值的启发式。GPU 中的 NTT 2 幂跳变、partial tree、owner 回退会使成本曲线不光滑；首版宜用对数网格加局部/形状边界细化，不宣称全局最优。

### 3.5 EcmStage2RatioAdjust 的含义

[成本乘数](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:5899)：该值越大，相同 B2 的 Stage2 预测成本越高，通常会使最优 B2 减小。

[自动更新](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9852)：

```text
rho_measured = T2_actual/T1_actual
adjust_correct = adjust_old × rho_measured/rho_predicted
adjust_correct = clamp(adjust_correct, 0.333, 3.0)
adjust_new = 0.9 adjust_old + 0.1 adjust_correct
```

仅在两阶段计时都非零且 `EcmStage2AutoAdjust` 启用时更新。它是预测误差的滚动校正，不能用 NTT event 秒数除以 Stage1 完整墙钟替代。

## 4. PRPLL tune 的实际行为

本地 PRPLL 位于 `.refactor/gpuowl`，README 从 [第16行](D:/code/MPA-OpenCl/.refactor/gpuowl/README.md:16) 标为 PRPLL。源码帮助使用 `-tune`；本文沿用用户提出的 `--tune` 作为本项目拟议接口。

### 4.1 测量的是完整 PRP 迭代

[timeConfig](D:/code/MPA-OpenCl/.refactor/gpuowl/src/tune.cpp:111) 创建相应 Gpu 配置并调用 [Gpu::timePRP](D:/code/MPA-OpenCl/.refactor/gpuowl/src/Gpu.cpp:2900)：

- quick=1..10 对应 20000..400 个迭代；默认 quick=7 是1200次。
- 先运行20次 warmup，等待队列完成，再计时剩余迭代。
- 工作主体是 PRP square，包含块间 check 用的 modMul；并非只测一个 FFT kernel。
- 队列结束后执行结果检查，失败返回 infinity，配置不会进入最佳结果。
- 返回 `microseconds/PRP iteration`，即 `iter/s = 10^6/cost`。

其中 GPU 构造、编译等发生于 timePRP 的计时外；它侧重热运行吞吐。我们当前生产逐曲线启子进程，需另外计入冷启动成本。

### 4.2 配置搜索和拐点文件

[Tune::tune](D:/code/MPA-OpenCl/.refactor/gpuowl/src/tune.cpp:361) 区分设备、FFT/NTT 类型、inplace、指数区间和 quick；同一函数还搜索若干执行参数。随后按 [shape 最大指数排序](D:/code/MPA-OpenCl/.refactor/gpuowl/src/tune.cpp:1265)，测量支持的 variant/carry 组合。

[TuneEntry::update](D:/code/MPA-OpenCl/.refactor/gpuowl/src/TuneEntry.cpp:10) 保留 cost 与 maxExp 的非支配前沿：成本更高、支持范围又不更大的配置被去掉。改进时 [写 tune.txt](D:/code/MPA-OpenCl/.refactor/gpuowl/src/tune.cpp:1408)，文件保存 cost、配置 spec 和 maxExp。

运行时 [FFTConfig::bestFit](D:/code/MPA-OpenCl/.refactor/gpuowl/src/FFTConfig.cpp:342) 优先遵守显式 FFT spec，否则遍历 tune 的成本顺序，选择支持当前指数、bits-per-word 下限和设备能力的最快配置；没有匹配则使用内建回退。

这更接近“按支持范围选择最快配置的前沿”，不只是预先计算相邻 length 的交点。我们的查询变量还有 batch、S、驻留状态及内存，可行前沿的维度更多。

### 4.3 可借鉴与不能直接移植的部分

可借鉴：显式 tune、预热、测量实际工作单元、失败候选不参与排名、增量保存、显式配置优先、无匹配时清楚回退。

需重新实现：Goldilocks 的整数 exactness/packing 上界、CUDA 形状/调度参数、不同操作类型成本、联合显存预算和来源指纹。

PRPLL 的 [maxBpw/ROE 调优](D:/code/MPA-OpenCl/.refactor/gpuowl/src/tune.cpp:129) 是浮点精度/roundoff 边界。我们的可用 bpw 应由整数上界证明，再做算术核验；不能把“偶尔没有错”当作放宽上界的依据。

## 5. 本仓库已经具备什么、缺少什么

### 5.1 生产接口与 B2 合同

[CLI 参数](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:198) 目前没有 auto/tune 模式；[INI](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:270) 只读固定 `stage2_b2` 等。

[当前优先级](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:489) 是非零 CLI B2 > 非零 worktodo B2 > INI B2。worktodo 的 B2=0 回到配置；若最终仍为0，会在 [输入规划](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:506) 因 B2≤B1 失败。**今天 B2=0 尚不表示自动选择。**

驱动 [每条曲线启动子进程](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:361)，随后 [调用当前引擎](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:415)。这影响规划位置、显存快照、冷启动开销和结果记录；离线规划后不应让引擎在另一套规则下再次改 D。

### 5.2 shape 与 D 成本模型

[ntt_shape_query](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3052) 是真实 shape 查询；[Python 几何复现](D:/code/MPA-OpenCl/tools/bench/calibrate_stage2_d.py:99) 可作离线规划辅助：

```text
m = 输入最大系数数（fold 为 P+1，tree 按实际节点大小）
slot_bits = 2S + max(1, ceil(log2(m)))
选最大的 b<=62，使 m ceil(slot_bits/b) (2^b−1)^2 < q_GL
slot_words = ceil(slot_bits/b)
L = 2^(bit_length(2m slot_words))
```

最后一式取严格大于 `2m slot_words` 的2幂。生产应继续以真实 C++ 查询和支持条件为准。

[DPhaseModel](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:79) 使用真实 L、经验 shape 权重、tree/Newton 工作量，再 [估算各阶段](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:119)。[采集](D:/code/MPA-OpenCl/tools/bench/calibrate_stage2_d.py:159)、[拟合](D:/code/MPA-OpenCl/tools/bench/fit_stage2_d.py:1)、[离线规划](D:/code/MPA-OpenCl/tools/bench/plan_stage2_d.py:36) 已经构成 tune 的原型。

但 [运行时 scope](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10781) 限定 S=4423、N 的 popcount=4423、B1=1000、B2范围、设备名、检查频率和后端。shape 权重改变的是经验特征，**不等于一份各长度的绝对 seconds/iteration 表**。

[离线工具的 --bits](D:/code/MPA-OpenCl/tools/bench/plan_stage2_d.py:39) 可以改变几何，但不会自动把仅 M4423 的 rates 变成其他位宽的有效标定。未来必须显式拒绝不匹配 scope 或标记为低可信估计。

现有 integrated D 搜索还有 [有限候选检查停止条件](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10912)，作为启发式排名使用；不能把它输出的 D 称为所有内存可行形状中的严格全局最优。

### 5.3 NTT 测量原型

[实际卷积 fixture](D:/code/MPA-OpenCl/tools/test/ntt_coop_outer_probe.cu:24) 已支持 k=16..27，使用实际 NTT 代码，计时 `forward(A)+forward(B)+inverse/product`，一轮预热加三次 CUDA event 测量；验证在 events 外。

[bench_ntt_fixed_backend.py](D:/code/MPA-OpenCl/tools/bench/bench_ntt_fixed_backend.py:1) 已封装来源冻结、固定后端确认、交叉顺序和原始记录。fixture 输入稀疏，但计算覆盖整个 L，核验所有 L 个输出。这能做域卷积基线，尚未测 packing、carry、模 N 归约和生产传输。

`ntt_poly_probe bench` 是大整数乘法口径，见 [run_bench](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:4949)，也不能直接充当 ECM 多项式操作吞吐量。

## 6. tune 应提供的三层数据

### 6.1 第一层：不同 NTT length 的吞吐量

建议对外保留 `iter/s`，同时输出明确的 `unit`：

```text
unit=field_convolution
一次 = 2 forward + pointwise product/scale + 1 inverse，按实际融合实现
conv_iter_per_s = R / measured_seconds
若每次调用有 batch=b：field_products_per_s = b R / measured_seconds
```

不能将上述结果除以3命名为单次 transform/s：forward 和 inverse 的代价/融合内容不同。若需要 transform/s，应分别直接测 forward 和 inverse。

记录 L、batch、实际 passes、tile/outer/warp、固定归约后端、寄存器/local/shared 资源，以及 payload/cold setup。纯域运算基本不依赖 ECM 的 S，可按相同后端/L/batch 复用；packing/carry/模 N 归约需要第二层。

可附带科学比较量：radix-2 理论蝶形数 `L log2(L)/2` 每次变换；三变换卷积约 `3L log2(L)/2`，它是等价工作量而非 GPU 指令数或周期数。若各 full-array pass 均读写一次，主体流量约 `16L batch (2 passes_fwd + passes_inv)` bytes，加上实际融合乘积、表读取等。不要把这个估算宣称为实测带宽。

第一轮范围以既有 k16..27 为起点，再按生产调用直方图补小长度和 batch。不能因为未测 k<16 就把低层成本当0；更大长度先查询内存/支持条件，可标记 skipped。首版不扫描所有参数的笛卡尔积。

length 的可行下限由 packing/exactness 决定，吞吐量表不能批准更短的不合法 L。如果要尝试“增加 padding 换取更快配置”，需先让真实 planner 支持更大 L 并确认布局/提取合同，再比较完整操作时间；首版只在当前合法 shape 上选择已支持的调度。

### 6.2 第二层：不同位宽的真实多项式操作

以 `(modulus_family,S,W,mA,mB,output_kind,batch,path)` 为键，测实际 API：输入已驻留、host 输入、compact low/high 输出等分别记录。操作至少包含 multiply、tree level、Newton inverse/fold 所用组合及 scaled descent。

每条记录保留 GPU events 和完整调用墙钟：[NttMulStats](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2947) 已有 fwd/inv/slot、host pack/scan/H2D、check、plan/opcopy 和 total_call 等基础。纯 event 表用于解释 kernel 变化；Auto B2 使用完整路径的时间。

不同 S 即使使用相同 L，仍可能有不同 W、slot_words、提取/归约量和点运算时间。不能只用 `L log L` 从 M4423 外推全部位宽。矩形/截断输出、固定输入已变换、batch 也不能合并成同一个 iter。

### 6.3 第三层：Auto B2 所需的完整阶段参数

每个位宽或经验证的位宽区间需要：

- Stage1 每曲线摊销秒数/模型，含 backend、曲线类型、exponent、批大小。
- baby/affine、giant seed/xADD、G tree、fold、inverse、descent、accum/GCD、检查及 host glue 的速率和固定开销。
- 驻留/非驻留两套路径；GPU baby 或 leaf 回退也要区分。数据有限时先使用覆盖范围较窄的 profile。
- 冷启动、计划/表构建、save 读取及检查策略。逐曲线子进程的成本不能用跨曲线热缓存模型替代。
- 不同预算下的可行 shape、实际 owner payload、同一阶段活跃内存，以及实际接口传输量。

离线可预计算：D/phi/P 列表、各 S/P 的 exact shape、tree 节点/partial batch 工作量、内存公式、收益函数。与硬件有关的绝对秒数必须 tune；实时 free VRAM/并发任务状态必须在运行时查询，不能写死在静态表里。

首先用已有2203/4423/8191 bits 扩展，但预算实验仅可作为外部锚点。B1=1000小于D，包含小素数桥接/命名负担；最终目标 B1≈2.6亿时应有独立留出，以检验模型对 B1 的处理。

## 7. 联合内存规划是必要前置条件

建议区分 `vram_total`、`ntt_big`、`ntt_cache`、`fold_owner`、baby 临时量及 host/pinned RAM，既允许单项上限，也受总可用显存约束：

```text
M_total_plan = max_phase(sum of allocations live in that phase)
M_total_plan <= min(user_total_budget, free_VRAM_at_plan − reserve)
M_big = 24 L batch bytes
M_owner = 8W(9P+8)+48 bytes
```

**各模块自己的峰值不能直接相加。** owner 跨阶段存活；baby、G frontier、输出和 NTT scratch 有不同生命周期。保留缓存的容量也需纳入阶段清单；host pinned 不属于 VRAM。

现在 [arena 预算](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10725) 还会受真实剩余显存约束；但 [缓存拒绝路径](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2244) 可以使用 per-call 表/分配，因此缓存预算不是进程显存硬上限。之前的 big budget 是外部 shape 过滤加实际 peak 核验，也不是 allocator 硬上限。

已有一项计账不一致需先处理或明确绕开：FuseCtx 用 [旧保守 need 式](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2268) 记缓存预算，而 [payload 峰值](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1930) 按实际 table/base 统计，[淘汰](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2002) 又按另一口径减账。预算实验出现 ledger约6149MiB、真实 NTT full peak约3187MiB。未修正前不能以 ledger 宣称某设备需要这么多物理显存，也不能仅凭 full_peak 宣称整条曲线一定能分配成功。

详细量化见 [B2 与两类显存预算实验](D:/code/MPA-OpenCl/docs/STAGE2_B2_MEMORY_BUDGET_SCALING.md:57)。其中 M4423 固定 D=1531530 的640/1024MiB owner对照：回退119.80s、驻留110.95s，回退多约7.98%；这说明路径要单独计费，不能将回退候选全部排除。

## 8. Stage1 成本输入：首版不能省略

现有 save 的 [TIME 字段](D:/code/MPA-OpenCl/src/core/ecm_save.cpp:189) 是写入日期，不是 Stage1 elapsed seconds；独立 Stage2 的 Record 也 [没有 Stage1 时长](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:101)。不能从 TIME 相减恢复成本。

建议优先级：用户提供每曲线秒数 → 同一 Stage1 backend/位宽/批配置的实际统计 → tune 的 Stage1 模型。均无匹配时只输出估计/缺失原因，不把不同算法的 Prime95 FFT 常数冒充有效标定。

GPU Stage1 多曲线并行时，应使用 `T1_batch / successful_usable_curves` 的吞吐摊销成本，含规范化和 save 必要费用；不能拿整个 batch latency 当作每曲线成本。共用指数生成费用按实际复用曲线数摊销。不同设备流水运行的最大吞吐目标需要资源队列模型，首版按用户选择的单一加总成本政策实现，并输出这个假设。

可建模为：

```text
ell = bit_length(torsion × lcm(1..B1))
T1_per_curve ≈ C1_setup_amortized + (ell−1) c_ladder_bit(S,backend,batch)
               + C1_normalize_and_save
```

实际操作计数以对应后端为准。指数生成在 [ecm_build_lcm_exponent](D:/code/MPA-OpenCl/src/core/ecm_stage1_exp.cpp:221)，CUDA 接口接收实际 [s_num_bits](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1196)。`lcm|choose12` 分别使用 torsion=1/12；Stage2 恢复时不再乘12。CPU PRAC、Edwards NAF 的模型应另建。现有 [ecm_cost.py](D:/code/MPA-OpenCl/tools/ecm_prob/ecm_cost.py:1) 提供操作计数思路，尚不是当前 GPU 的秒数标定。

为兼容旧 save，可先读独立性能 profile/用户参数；以后 Stage1 写可选 sidecar，保存 elapsed、有效曲线数、exponent、backend 与设备来源，避免破坏原有 save 校验和合同。

## 9. Auto B2 的建议算法

### 9.1 目标函数和两层搜索

```text
T2*(B2) = min_{feasible D, path, NTT config} T2_full(N,B1,B2,D,path,config)
score(B2) = K(B1,B2) / [T1_per_curve + adjust(profile) T2*(B2)]
choose argmax(score) subject to B2 range/time/memory constraints
```

T2_full 含 init/main 与必要 GCD/检查，已有项不重复加；还应补该生产入口的计划/冷启动费用。当前 [stage2_full_wall](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11424) 不包含 [D 扫描](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10977) 和驱动的全部子进程成本，要分别记录。

计划步骤：

1. 读取 N/B1/输入合同、匹配 tune profile，查询 device/free VRAM；读独立单项预算及总预算。
2. 枚举当前支持的47-smooth D；缓存 `(S,P,operation,batch,config)` 的 shape/静态工作量。用户指定 D 时固定它，仅搜索 B2。
3. 对每个 B2，比较驻留与回退的可行路径，计算完整成本；生产必须能执行所选 path。profile未覆盖的路径不给“已标定”分数。
4. B2 从有限的倍率区间做对数采样，扩大到收益下降或配置上限，随后细化较好候选附近及 G/partial tree/NTT shape 的变化点，做整数相邻值比较。
5. 在上界仍最佳时报告 `range_limited`；无可行候选报告原因；预测收益差落在模型误差内时优先较低内存/较短运行的候选。
6. 在曲线实际分配前复核 free VRAM。资源变化导致 path 改变时重新规划或明确记录回退预测，不能沿用原来驻留时间承诺。

搜索上限是政策参数，不是“内存算出的最高 B2”。固定可行 D 可以通过增加 G 处理更大 B2，时间随之增长。所有界限/倍率乘法需检查整数溢出及引擎 signed index 范围。

固定 D 时可以用 `T2≈A+C B2` 或 `A+C_G G` 作解释/局部近似；可调整 D 时近似 `aD+bB2/D` 才导出平方根趋势。真实搜索必须用离散模型。预算实验已提供反例：owner128回退仍alpha约0.55；small_big驻留且D固定后alpha达0.975。

### 9.2 建议接口与兼容性

以下是**拟议接口，当前不可执行**：

```text
ecm_cuda_stage2.exe --tune ntt --device 1 --length-log2 16:27 --tune-file FILE
ecm_cuda_stage2.exe --tune stage2 --device 1 --bits 2203,4423,8191 --tune-file FILE
ecm_cuda_stage2.exe --save FILE --auto-b2 --stage1-seconds-per-curve SECONDS
ecm_cuda_stage2.exe --save FILE --auto-b2 --plan-only
```

INI 可拟议 `stage2_auto_b2`、`stage2_tune_file`、`stage1_seconds_per_curve`、`stage2_ratio_adjust`、`stage2_vram_mb`、`stage2_ntt_big_mb`、`stage2_fold_mb` 及 Auto B2 的界限。float 秒数/倍率应有有限正数校验，不能复用只解析 u64 的全部配置流程。

兼容原则：**显式非零 B2 固定**，包括恰好100B1；非零 CLI > worktodo > INI 的现有顺序保留。零/缺省仅在开启 auto 时触发自动规划；auto 与 CLI 非零 B2 同时出现应报清楚冲突。显式 D 仍固定。未开启 auto 时保留现有行为。

`--plan-only` 输出 B2/D/P/I/G、各路径/形状、T1来源、各阶段秒数、score、内存/传输预测、profile和可信范围。运行的 JSONL 追加实际B2/D/path及预测误差；原 worktodo 文本写入 finished时保留原文，并靠结果/计划记录追踪自动值。

规划最好在实际 curve worker 获取设备状态后执行，通过同一引擎规划 API返回并直接使用计划；parent 可显示静态草案。避免 parent 与 child 分别扫描 D，也避免 Python/C++ 两份公式成为不同真相。首版先用独立工具验证数据流，再接入生产 CLI。

同一直接输入save含不同N/B1时，按匹配的记录组分别规划；队列目前要求同一save具有相同B1，继续遵守。每条结果都记录其实际计划。不能用第一条记录的B1为所有记录计算收益。

### 9.3 自校正

以匹配 profile 的完整 `T2_actual/T2_predicted` 更新乘数，可采用0.1的新观测权重。秒数模型已有 T1 输入时，不必强行计算 Stage1/Stage2比；缺少真实Stage1时间则不更新Stage1校正。

应按 device、后端、位宽/shape范围、驻留状态、检查策略隔离。一次 fallback、失败曲线、被其他负载干扰或范围外输入不能更新一个全局 ini 乘数。优先校正各阶段；只有稳定小偏差才使用总乘数。

### 9.4 概率模型扩展

首版使用已选定的 Prime95 K，相对收益单位易解释且不要求未知因子大小。后续可支持目标 factor bits/先验分布：最大化 `p_success/(T1+T2)`，而非把 S 当因子位数。

仓库已有 [predict_fraction](D:/code/MPA-OpenCl/tools/ecm_prob/model.py:118) 和 rho 的 Stage1+Stage2概率模型。这里 `bit` 是因子位数，`D_eff` 是群阶有效除子，需要区分参数命名并验证适用范围；不是当前性能 Auto B2 已经具备的功能。

## 10. tune 文件、测量与复用规则

建议输出可读 JSON，schema version 固定，包含以下分区：

- identity：GPU UUID/name/SM、CUDA runtime/driver、CPU/host环境、binary SHA、原始编译依赖 SHA、build flags。
- geometry：有效 N family/S范围、D/P、L/bpw/slot_words、batch、操作类型。
- configuration：实际启用的归约/outer/warp/point、驻留与回退、arena策略及检查频率。
- samples：warmup/cold分开、重复次数、events和wall原始样本、median/离散度、实际payload/传输、检查结果与日志路径。
- models：Stage1成本、分阶段rates、预测误差、留出结果、校正值；每个字段标记 measured/derived/estimated。

不同后端/编译源码不得混用绝对时间。纯 field NTT 在同一device/后端/L/batch/config下可以跨 S 复用；位宽相关操作需对应profile。相邻位宽插值只在相同kernel/shape合同且有留出验证时启用；W跳变、NTT跳变及非Mersenne切换处需新采样。

采用有限时长预热及自适应重复，使样本超出 event 分辨率；执行参数扫描和长曲线串行，原始样本增量写入。GMP参考/构建放在计时外。量测时默认保留生产必需检查；诊断trace的同步计时与生产吞吐分开。

已有预算实验的NVML利用率可解释空闲段，但不是 PCIe 计时；`ntt_seconds` 也包含wrapper/host。带检查的端到端wall与events差值只能作为待归因开销，不能直接全部命名为传输成本。

## 11. 建议实施顺序与验收目标

### P0：规划合同与预算基础

抽出共享的 shape/工作量/路径/预算查询，保留原算术调用。统一 arena allocation/charge/eviction口径，建立按阶段的活跃容量模型；区分缓存预算和总显存上限。提供 plan-only，记录实际路径。此步是可信 Auto B2 的前置条件。

### P1：NTT tune 最小版

封装当前固定PTX卷积fixture，输出每L的 `conv_iter/s`、batch、事件秒数、冷启动及payload。先测当前支持和生产常用形状，保存指纹和原始结果。配置搜索仅限已有正确性覆盖的候选；实验outer/carry方案不因局部iter/s胜出就自动成为生产默认。

### P2：按位宽与路径标定完整 Stage2

接入真实多项式API和现有phase采集/拟合。首先覆盖2203/4423/8191 bits、不同D/G、owner驻留/回退及实际目标B1。用整条曲线留出检验排序和时长，单独补Stage1实际摊销成本或显式用户秒数。

建议验收指标：留出完整时长相对误差目标≤10%；在实际候选集内选中的方案耗时距实测最快≤5%；显存可行性不误报可分配。这些是**拟议验收目标，不是已达成结果**。差距小于测量噪声时保留多个候选。

### P3：Auto B2 离线规划与生产接入

先输出联合计划/排名，再接 CLI/INI/worktodo；显式B2兼容、混合save B1处理、无profile、溢出、资源变化和G=1路径都应有明确合同。G=1使用不同直接余式路径，目前resident fit排除它，不能用G−1=0简单套旧模型。

接入时复用生产build入口和清单：[build_stage2_tree_gpu.ps1](D:/code/MPA-OpenCl/tools/build/dev/build_stage2_tree_gpu.ps1:47)、[独立生产builder](D:/code/MPA-OpenCl/tools/build/build_stage2_local.ps1:1)。将新增规划/profile依赖纳入构建来源，而不把某台设备的tune结果硬编码为所有设备默认。

### P4：反馈与扩展

基于干净完成的曲线自校正；增加更多位宽、GPU和泛型模数profile；按需要提供有限save时间预算和已知因子目标模型。若以后Stage1/Stage2跨设备流水，应显式建吞吐/资源排队目标，重新审视简单T1+T2。

## 12. 本轮交付与尚未完成的工作

本轮完成源码调用链、公式/单位、现有实现边界与可执行的分阶段设计，并确认默认优化目标。未启动编译、GPU benchmark或测试，未修改现有B2语义、生产二进制及算法默认值。

下一项最具体的工作是 **P0共享plan接口和显存计账整理，然后P1的NTT tune最小版**。现有数据足以确定方向，还不足以给通用Auto B2写一个可信的默认倍率表。
