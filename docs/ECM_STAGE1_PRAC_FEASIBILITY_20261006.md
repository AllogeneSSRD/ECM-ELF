# CUDA ECM Stage1：PRAC 与离线 Lucas 链的可行性和性能分析

日期：2026-10-06。接续 [Stage1/CGBN 性能审查](D:/code/MPA-OpenCl/docs/ECM_STAGE1_CGBN_PERFORMANCE_REVIEW_20261006.md:1)。主要对象为 Suyama param0，沿用 `exponent=lcm|choose12` 和当前 normalized CGBN 算术。

后续已实现 CUDA 原型、完整 Q 门禁及生产 B1 短时采样，见 [实施与性能记录](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_IMPLEMENTATION_20261006.md:1)。本文保留实施前的离线分析结论。

本轮实现了离线链规划、成本统计、活跃点分析与实验计划导出，并执行了相关整数计算。下列比例是 **算子成本分析，不是 GPU 实测加速比**。本轮没有执行 GPU 基准、有限曲线完整 Q 对拍或修改生产 Stage1 内核。

## 1. 结论

**PRAC 值得进入 GPU 原型阶段，优先采用 Prime95 风格的三点状态实现。**

- 默认十组种子、`PracSearch=7`，完整 B1=10³/10⁴/10⁵/10⁶ 的 param0 名义算术节省约 **13.61% / 12.29% / 11.68% / 11.20%**。
- B1=2.6亿附近的一个素数区间包含 5,159 个素数，区间乘积的名义节省约 **10.01%**。这不是对整个 2.6亿 Stage1 的完整统计。
- 本轮生成的 PRAC 计划最多需要 **三个持久 XZ 点**；点算术临时量另计。当前 ladder 是两个点，PRAC 的寄存器和调度开销有可能抵消约 10% 的算术优势。
- `.refactor/ecm` 中的离线 Lucas 链可以作为小增益候选，但应约束点状态预算。现有代码表只覆盖 11～9973，无法证明生产 B1 的全范围收益。
- GPU DADD 应使用 **`4M+2S` 的 projective-difference 公式**。直接照搬 Prime95 的 FFT 点加表达式会需要更多完整模乘，本轮成本模型中反而输给 param0 ladder。

建议下一实现阶段：**三点 PRAC + 编译期明确的点算术 + Montgomery 域驻留 + 素数边界分片**，并与同样驻留 Montgomery 域的 ladder 做 A/B，区分链算法收益与重复转换消除的收益。

## 2. 两套参考源码的关系

### 2.1 Prime95：运行时搜索的 PRAC

本地 Prime95 参考文件是 `.refactor/p95v3106b01.source/ecm.cpp`：

| 位置 | 内容 | 移植时应保留的行为 |
|---|---|---|
| [ecm.cpp:2640](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2640) | DBL=10、ADD=12 的 FFT 成本 | 转为 GPU 实际算子权重，不直接套 FFT 时间 |
| [ecm.cpp:2645](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2645) | `lucas_cost` | 初始/最终运算及各归约规则必须全部计数 |
| [ecm.cpp:2711](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2711) | `lucas_mul` | A/B/C 的交换、差值点和 scratch 语义 |
| [ecm.cpp:2733](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2733) | 第一条规则消除部分 C=A 拷贝 | 可优化复制，但不能改变点关系 |
| [ecm.cpp:2777](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2777) | 简化规则，阈值 2.96 | 使用实际启用的规则，不计入注释掉的原始规则 |
| [ecm.cpp:2847](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2847) | `lucas_cost_several` | 邻近 d 搜索、相同成本时保持首次候选 |
| [ecm.cpp:2857](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2857) | `ell_mul`，十组比例种子 | 初值使用 ceil；默认每组搜七个邻近 d |
| [ecm.cpp:6904](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:6904) | `PracSearch` | 默认 7，范围 1～50 |
| [ecm.cpp:7448](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:7448) | 素数重复次数 | 对每个 p 执行 floor(log_p B1) 次 [p] |
| [ecm.cpp:7461](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:7461) | choose12 | 额外两个 2、一个 3 |

当前 CGBN 的 S≈M 时，优化后的 GPU DBL/DADD 成本为 5/6，恰为 Prime95 搜索权重 10/12 的一半。因而在这个简化成本模型内，直接采用其种子和邻域搜索是合理起点。加减、拷贝、spill、特殊首步及未来更便宜的平方都会影响实际最优计划。

本轮规划器针对 Stage1 的 **素数乘法重复**。它没有声称覆盖 Prime95 `lucas_mul` 对任意复合 64 位乘数的递归外层；实际 Stage1 在 [ecm.cpp:7469](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:7469) 调用的也是当前素数。

### 2.2 `.refactor/ecm`：带离线优化 Lucas 链的 GMP-ECM 分支

用户指定的 `.refactor/ecm` 包含另外一套实现：

- [ecm.c:549](D:/code/MPA-OpenCl/.refactor/ecm/ecm.c:549)：`generate_Lucas_chain`，解码已有 uint64 链码。
- [ecm.c:1027](D:/code/MPA-OpenCl/.refactor/ecm/ecm.c:1027)：最大延伸，按已知差值点扩展链。
- [ecm.c:1086](D:/code/MPA-OpenCl/.refactor/ecm/ecm.c:1086)：16 槽 X/Z 环。
- [ecm.c:1121](D:/code/MPA-OpenCl/.refactor/ecm/ecm.c:1121)：加载 `Lchain_codes.dat`。
- [ecm.c:1187](D:/code/MPA-OpenCl/.refactor/ecm/ecm.c:1187)：从 p=11 开始逐素数读取一个 uint64；EOF 后回到 PRAC。
- [ecm.c:1214](D:/code/MPA-OpenCl/.refactor/ecm/ecm.c:1214)：执行链中的 DBL/DADD。
- [LCG_macros.h:356](D:/code/MPA-OpenCl/.refactor/ecm/LucasChainGenerator/src/LCG_macros.h:356)：在相同搜索长度下优先保存倍点数更多的候选。
- [LucasChainGenerator/README](D:/code/MPA-OpenCl/.refactor/ecm/LucasChainGenerator/README:1)：生成器用法和历史生成时间。

生成器优化最小/近最小链长度，并在相关长度内偏好多倍点；这不自动等于 GPU 上全局最短时间。需要进一步计入活跃点和 field operation 的实际成本。

本地链码文件为 9,800 字节、1,225 个 uint64 记录，按从第一个素数 11 起的顺序解码，最后一个是 9973。首尾及每条使用的差值关系已在本轮离线计算中核对。文件没有包含 B1、版本、字节序或校验和的 header；新的生产缓存应补上这些元数据。

## 3. 旧成本脚本的修正

旧 `tools/stat/prac_cost.py` 存在数个影响结论的问题，本轮已替换为新的规划器调用：

1. **缺少完整候选搜索。** 只使用黄金比例附近一个 floor 初值，无法代表 Prime95 的十种子、ceil 和邻域搜索。
2. **重复加权素数幂。** 先枚举 `(p,1),(p,2),…,(p,e)`，又分别乘以指数 e，得到 `1+…+e` 次乘法，而正确重复数是 e。例如 B1=1000 的 p=2 应重复 9 次，该错误会加权为 45 次。
3. **B/C 状态交换不正确。** `ell_add(...,&C)` 后交换 B/C，正确的新 B 是和点，新 C 是旧 B。旧的标量跟踪没有保留这一关系；不能把逐素数不一致解释为“符号检查先天不可靠”。
4. **使用旧 param3 的成本定价 param0。** 新默认模型是 param0=`6M+4S`、DBL=`3M+2S`、projective DADD=`4M+2S`。
5. **用聚合误差修正逐条错误计数。** 新规划在每个实际操作处检查差值关系，并检查最终乘数；不再使用一个总体校正系数掩盖错误链。

另有一个历史文档数字需要撤回：`bitlength(lcm(1..100000))` 是 **144344**，不是 144352。本轮用平衡整数乘积得到 144344，再由 Python `math.lcm(1,…,100000)` 独立整数计算确认。`B1=1000/10000/1000000` 分别为 1438/14447/1442099。

## 4. 成本模型与完整统计

### 4.1 模型定义

令 `s=t*lcm(1..B1)`，`b=bitlength(s)`，D/A 为整个计划的 DBL/DADD 次数，α 为 S/M。全部素数幂的重复次数，以及每条 PRAC 的首个 DBL、最后 DADD 都包括在 D/A 中。

\[
W_{\rm ladder}=(b-1)(6+4\alpha),\qquad
W_{\rm PRAC}=D(3+2\alpha)+A(4+2\alpha).
\]

当前 α=1 时为 `10(b−1)` 与 `5D+6A`。这里比较主标量计算的 field operation；公共曲线构造、ladder 在 CPU 准备初始 2P 的那一个 DBL、域转换、加减及控制流单列。短 B1 时这些固定成本更需要独立测量。

### 4.2 完整 B1 统计，t=1、PracSearch=7

| B1 | 精确 b | DBL D | DADD A | `5D+6A` | PRAC / ladder | 名义算术减少 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1,438 | 474 | 1,674 | 12,414 | 0.863883 | 13.61% |
| 10,000 | 14,447 | 3,643 | 18,081 | 126,701 | 0.877066 | 12.29% |
| 100,000 | 144,344 | 30,713 | 186,882 | 1,274,857 | 0.883214 | 11.68% |
| 1,000,000 | 1,442,099 | 277,057 | 1,903,501 | 12,806,291 | 0.888032 | 11.20% |

数据源：[本地完整统计 JSON](D:/code/MPA-OpenCl/docs/data/prac_cost_20261006_final_full.json:1)。原始 JSON 属于已排除的本地实验产物；本报告保留可复现的结果摘要。

在 B1=10⁶，密度为 `D/(b−1)=0.19212`、`A/(b−1)=1.31995`；总代数点操作约 1.51207/bit。可见较多的 projective DADD 仍能以减少总操作数补偿其高于固定仿射差值 ADD 的成本。

### 4.3 生产 B1 附近的区间研究

对 B1=260,000,000，仅统计 `[259900000,260000000]` 内的素数：

- 素数数目：5,159，均只重复一次。
- 区间素数乘积的位数：144,213；**它不是整个 Stage1 的标量位数**。
- D=26,825，A=193,936。
- PRAC / 对应该区间乘积的 ladder：0.899884，即名义减少约 10.01%。

数据：[区间 JSON](D:/code/MPA-OpenCl/docs/data/prac_cost_20261006_final_window.json:1)。此结果说明较大素数上仍存在名义算术空间，但未遍历全部 2.6亿以内素数，不能将 10.01% 写成全范围的精确收益。

### 4.4 搜索规模与平方敏感性

- B1=10000，十种子但 `PracSearch=1`：PRAC / ladder=0.884120；默认 search=7 为 0.877066。增加邻域搜索在这个规模下额外减少约 0.71% 的 ladder 算术成本。
- B1=100000，α=0.75、t=12：PRAC / ladder≈0.897589，名义减少约 10.24%。便宜平方同时利好 ladder，故不能把专用平方与 PRAC 的各自百分比直接相加。
- B1=1000，t=12：b=1442，D=477、A=1675；额外 12 计入三个 DBL、一个 DADD。PRAC / ladder≈0.862942。

相关本地数据：[search=1](D:/code/MPA-OpenCl/docs/data/prac_cost_20261006_search1.json:1)、[平方敏感性](D:/code/MPA-OpenCl/docs/data/prac_cost_20261006_sqr075.json:1)、[choose12](D:/code/MPA-OpenCl/docs/data/prac_cost_20261006_final_choose12.json:1)。α=0.75 是情景分析，不是已经实现或测出的 GPU 平方成本。

## 5. GPU 点算术应怎样实现

### 5.1 保留当前 DBL，使用较少模乘的 DADD

DBL 使用当前 param0 的完整 a24 公式，成本 `3M+2S`。相关基线见 [cgbn_stage1_kernel.h:816](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:816)。

对一般差值点 `(Xd,Zd)`，DADD 可用：

\[
u=(X_1+Z_1)(X_2-Z_2),\quad v=(X_1-Z_1)(X_2+Z_2),
\]
\[
X_3=Z_d(u+v)^2,\qquad Z_3=X_d(u-v)^2.
\]

四次 M、两次 S，另加六次点加减。每次模乘/平方继续使用 normalized wrapper；不能删除范围检查或条件减 N。

### 5.2 为什么不能直接移植 Prime95 的 FFT 点加表达式

[Prime95 ell_add_xz_scr:2494](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2494) 先形成 `x1z2−z1x2` 和 `x1x2−z1z2`，两个差式共四次乘法，平方后再乘差值坐标，按普通 field operation 计为 **6M+2S**。

FFT 的合并算子和变换复用使它在 Prime95 中合理；普通 CGBN 没有相同定价。若 CGBN 直接使用该表达式，在 B1=10⁶ 时成本约 `(5D+8A)/ladder=1.1520`，反而多约 15.2% 的名义工作。

较少模乘的 u/v 公式与上述表达式相差公共 projective 缩放，不改变非退化点的仿射 x；有限曲线和因子情形仍应在 GPU 原型中验证。

### 5.3 alias 与状态管理

PRAC 常令输出覆盖某个输入或差值点。若输出与差值点共用存储，在先写 `X3` 后计算 `Z3` 时可能覆盖仍需使用的 `Xd`。

新规划器的 slot 分配依赖 **alias-safe 点操作**：所有旧源坐标读取完毕后再提交两个结果。持久点槽数不包含用于完成这一动作的 field temporaries。

同批曲线共享 p、d、重复次数及链规则，控制序列不依赖 sigma；PRAC 不必引入各曲线不同链造成的 SIMT 发散。数值相关的规范化和异常处理仍存在。

## 6. 寄存器、活跃点与胜出门槛

### 6.1 三点 PRAC 的成本

本轮全部已分析 PRAC 计划经活跃区间分配，最多三个持久 XZ 点。与两个点的 ladder 相比多一对坐标，但不再需要完整 `xdiff` 常量。

- 当前 param0 ladder 持久大整数：四坐标、a24、xdiff、N，共七个。
- 三点 PRAC：六坐标、a24、N，共八个。

仅按持久 limb 值，净增加一个 bn，即每协作线程约 L 个 uint32 值：2203 位默认档位 L=5，4423 和 8191 位为 L=9。**这不是完整 kernel 寄存器数预测**；临时变量活跃区间、CGBN carry/shuffle、计划解码、编译器复用都会改变实际结果。

现行寄存器限制见 [kernel:301](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:301)。不能把当前 128 上限直接套到 PRAC 后假定收益保持；必须同时检查自然寄存器、spill、实际 block 驻留和批量波形。

### 6.2 算术优势可以容忍多少额外延迟

设 r 为上表算术比，s 为 PRAC 中每个模乘等价单位相对 ladder 的平均慢化比，f 为原总时间中可按该模型缩放的部分，q 为其余工作量比，h 为额外开销占原总时间的比例：

\[
T_{\rm PRAC}/T_{\rm ladder}\approx f r s+(1-f)q+h.
\]

只计 field operations 时，B1=10⁶ 的 r=0.888032，胜出要求 s<1/r≈1.1261；生产素数区间的门槛约 1.1113。约 11～13% 的每单位成本恶化就足以抹去算术优势。

示例而非测量：f=0.9、q=1、h=0 时，B1=10⁶ 的 s=1/1.10/1.15 对应总时间比约 **0.8992 / 0.9792 / 1.0191**。每单位慢 10% 时只剩约 2.1% 总时间改善；慢 15% 时已经退化。

点加减也会增加：若分开计算 DBL/DADD，约为 `4D+6A` 次；fused ladder 每位约八次。B1=10⁶ 的该项约 8.688/bit，高于八次。因此上述模乘等价模型已经偏乐观，应争取在相邻 ADD+DBL 中复用中间和差，并用 SASS 检查是否实际减少工作。

## 7. 离线 Lucas 链是否更值得优先

### 7.1 现有码表的算术收益很小

B1=10000 全范围都有已有码：

- 纯 PRAC：126701 个模乘等价单位。
- 逐素数挑选成本更低的 Lucas，允许任意点槽：126398。
- 相比 PRAC，仅额外减少约 **0.239% 的 PRAC 算术量**，或约 0.210% 的原 ladder 算术量。

因此不能为这个增量先实现复杂的 16 点 GPU 环。

### 7.2 点状态预算的实际影响

1,225 条链码中，18 条解码计划经本轮活跃区间分配需要超过三个槽，最大五个；其余不超过三个。这里依赖可复用死输入的 alias-safe 点操作，不包含点算术 scratch。

超过三槽且算术确实胜过 PRAC 的记录只有 p=7681：Lucas 成本 115、PRAC 116，需要四槽。限制为三个点后，B1=10000 的混合成本变为 **126399**，仅少获得一个单位的收益。

工具默认 `--point-budget 3`，超出预算的 Lucas 候选回到 PRAC；`--point-budget 0` 可作不限槽位的离线比较。已有 C 代码的 16 槽是执行存储策略，不能当作每条链必需的最小活跃点数。

数据：[三槽约束 JSON](D:/code/MPA-OpenCl/docs/data/prac_cost_20261006_3slots.json:1)。本轮统计只覆盖现有码表；更大素数的结果不能由此推断。

### 7.3 生成成本及缓存

生成器 README 的历史 i9-13900K 记录：B1=10⁶ 约 18.6 s，B1=260e6、8 线程约 580 min。它们不是本轮测量，也不能代表本机或 GPU；但说明生产范围的最短链搜索应离线缓存，不能放进每次 Stage1 启动。

码表有效载荷是 `8*(π(B1)−4)` 字节，不含元数据。未来缓存至少包含编码版本、B1 覆盖范围、素数序列约定、字节序、校验和、搜索/成本策略。GPU 适用性还需记录活跃点上限，避免“最短链”默认为“最快链”。

## 8. 计划数据量与传输

当前 ladder 指数载荷约 b/8 字节。对生产 b≈375.1 Mbit，约 44.7 MiB。

PRAC 可选择：

1. **每素数 seed 描述。** 对 p、d 均能放入 uint32 的 B1，可用两个 uint32 加重复次数/flags 的定长布局，典型对齐为 12 字节/素数，即约 `12π(B1)` 字节。若另有隐式素数序列，可以单独保存 d，但素数生成/传输也必须记账。
2. **规则流。** 压缩 PRAC 规则可以避免 GPU 每条曲线重复运行整数状态归约，但需要首步、交换、倍点、最后输出及 prime/repetition 边界编码。
3. **完整槽位操作流。** 适合小 B1 的实验。约 1.5b 个操作时，单字节已约 537 MiB；明确源/目标槽位的 16 位操作流约 1.07 GiB，仍不含边界记录。不可无预算地展开生产计划。

建议 v1 GPU 原型使用已选择 seed 和明确的 A/B/C 寄存器角色；同一 p/d 控制在 warp 内一致。计划按素数块准备并复用，必要时流水上传；避免新增数 GiB 长期驻留缓冲。

目前 Python 离线完整分析 B1=10⁶ 约几十秒，包含搜索、SSA、生命周期和精确整数标量；这不是生产 C++ 规划器的耗时。需要缓存、流式生成或原生实现后再讨论总流程准备成本。

## 9. 本轮新增工具和用法

### 9.1 实现内容

- [ecm_prac_plan.py:196](D:/code/MPA-OpenCl/tools/stat/ecm_prac_plan.py:196)：十种子及邻域搜索，采用当前 GPU 成本权重。
- [ecm_prac_plan.py:218](D:/code/MPA-OpenCl/tools/stat/ecm_prac_plan.py:218)：PRAC 的 SSA 点操作计划。
- [ecm_prac_plan.py:263](D:/code/MPA-OpenCl/tools/stat/ecm_prac_plan.py:263)：Lucas 链码解码。
- [ecm_prac_plan.py:40](D:/code/MPA-OpenCl/tools/stat/ecm_prac_plan.py:40)：死输入复用及持久点槽位分析。
- [ecm_prac_plan.py:402](D:/code/MPA-OpenCl/tools/stat/ecm_prac_plan.py:402)：精确标量位数。
- [prac_cost.py:37](D:/code/MPA-OpenCl/tools/stat/prac_cost.py:37)：完整/区间成本统计、点预算筛选、JSON 输出。

生成时检查每个 DADD 的差值关系，以及最终符号乘数是否等于目标素数。这覆盖本轮离线计划的数学关系，不能替代合数模数上的异常路径或 CGBN 代码验证。

### 9.2 完整离线成本比较

从仓库根目录执行；以下只使用 CPU：

```powershell
python tools/stat/prac_cost.py 1000 10000 100000 1000000 --cgbn `
  --lucas-codes .refactor/ecm/Lchain_codes.dat --point-budget 0 `
  --output docs/data/prac_cost_full.json
```

输出包含精确 b、D/A、名义成本、点槽统计、输入文件及脚本 SHA256。默认 α=1；`--cgbn` 强制使用现行 CGBN 的平方成本模型。`--sqr 0.75` 可做情景分析；`--torsion 12` 使用 choose12 标量。

### 9.3 生产范围的局部密度

```powershell
python tools/stat/prac_cost.py 260000000 --cgbn `
  --prime-window 259900000 260000000 `
  --output docs/data/prac_cost_window.json
```

结果明确标记 `scope=prime-window-only`。区间分析不允许导出冒充完整 Stage1 的计划。

### 9.4 导出小规模实验计划

```powershell
python tools/stat/prac_cost.py 1000 --cgbn --torsion 12 `
  --lucas-codes .refactor/ecm/Lchain_codes.dat --point-budget 3 `
  --emit-plan docs/data/prac_plan_b1000.json
```

计划记录每素数重复次数、来源、输入/输出槽位及 DBL/DADD 操作。默认最大序列化预算 64 MiB，可用 `--max-plan-mib` 更改；统计模式不会展开整个 Stage1 操作表。该 JSON 是实验格式，当前生产 exe 不读取它。

## 10. 下一阶段 GPU 实施和性能验收

### 10.1 原型边界

建议先实现 opt-in param0 PRAC；使用三个明确命名的 XZ 点、完整 a24、共享 N、normalized CGBN 算子，不直接声明动态索引的 `point[16]` 数组。点寄存器别名及交换应检查实际编译结果，避免自动落入 local memory。

先按 **完成的素数乘法边界** 分片。中断时保存当前目标 P、prime index、该素数重复进度、B1/t、sigma、模数/档位和计划身份。若以后允许在一条 PRAC 内恢复，还必须保存 A/B/C、d/e 和规则位置。

现有 ladder v5 的 bit offset 不能解释 PRAC 状态，需新版本和算法标记。Stage1 最终 save 保持相同曲线、sigma、N、B1、仿射 x 的语义；更改算法后也需重新校准 Auto B2 使用的 Stage1 时间及 profile 身份。

### 10.2 三组对照

1. 现行正确 ladder：当前分片转换和主机行为。
2. 同域驻留 ladder：单独量化重复入/出域消除的收益。
3. 同域驻留 PRAC：与第 2 组比较链算法本身。

至少覆盖 n=2203/4423/8191，欠填充/充分填充批量、小 B1/较大 B1、lcm/choose12、通用合数/梅森数/余因子。记录纯 kernel、全部墙钟、准备、传输、寄存器和 spill；不能用旧 `gputime` 当纯 kernel。

### 10.3 正确性条件和停止条件

GPU 原型需有限曲线完整 Q、64 位 sigma、规范范围、非单位分母/最终 Z、因子命中及 checkpoint 对照。符号链合法仍可能在合数模数上遇到退化点；错误返回码、save checksum 与少数因子命中不能代替这些检查。

若当前主力宽度上每模乘等价单位的实际退化达到约 11～13%，或 spill 使完整 kernel 没有超出噪声的收益，应优先改善点操作活跃区间和批量配置；不应以继续扩大链搜索补偿 kernel 实现的问题。

首轮离线 Lucas 只允许不超过三个活跃点，保留 PRAC 回退。更大状态的链需要在明确节省更多算术成本时单独试验。

## 11. 证据范围

本轮操作是源码审查和离线整数/链计算；原始数据均保留于被 Git 排除的 `docs/data/`。生产 Stage1 的性能变化仍需实际 GPU A/B，当前不能宣称 PRAC 已使生产提速 10%。

链码 SHA256：`06abdc1232fc8cdeacf562b39e2dd85bb49759d12100edf756fe453c39f5fb15`。

本轮最终原始结果文件的 SHA256：

- 完整 B1 统计：`35dd4fe2ff648ae2519c4760e901be843bfbe41c013f17e7726280c3af99b237`。
- 生产素数区间：`2c9199b17e4ac667d2978461154dd8231bfe60d3ac81483eb46402d474ff39fd`。
- choose12：`ce22a4e7b37d5cc630bf9592dda01382b07cfa4592811d65492c116f33f0a002`。

源码出处及当前 CGBN 成本、档位、内存基线见 [Stage1/CGBN 审查](D:/code/MPA-OpenCl/docs/ECM_STAGE1_CGBN_PERFORMANCE_REVIEW_20261006.md:1)；本轮脚本和引用源码摘要记录在各统计 JSON 中。
