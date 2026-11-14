# Montgomery ECM：Suyama、差分算术与 Lucas/PRAC

固定参考资料。来源：`ECM_Montgomery_STAGE1.md §§2–4、7`, `ECM_STAGE1_PRAC_FEASIBILITY_20261006.md §2`。保留数学推导、第三方源码分析及其条件；源码行号针对原文研究的本地副本，第三方上游可能已有变化。本文不声明本仓库当前支持、默认值或性能。涉及实验数字时只按原文条件理解；当前实现见 [架构](../architecture/OVERVIEW.md)，当前计时口径见 [性能](../performance/STAGE2.md)。

## 1. 曲线与差分点

Montgomery 曲线 `B y²=x³+A x²+x`，非奇异时 `B≠0`、`A≠±2`。射影 x 坐标为 `(X:Z)`，`x=X/Z`；P 与 −P 共享 x。一般加法不能只由 x(P)、x(Q) 唯一决定，差分加法额外输入 x(P−Q)。设 `a24=(A+2)/4`。

```text
xDBL(X,Z):
  U=(X+Z)^2; V=(X-Z)^2; E=U-V
  X2=U*V; Z2=E*(V+a24*E)

xADD(P,Q,D=P-Q):
  U=(XP+ZP)*(XQ-ZQ); V=(XP-ZP)*(XQ+ZQ)
  X3=ZD*(U+V)^2; Z3=XD*(U-V)^2
```

通用差分点成本 `4M+2S`；归一化差分点 `ZD=1` 时为 `3M+2S`。倍点为 `2M+2S+1·a24`，曲线常数乘法若与普通乘法同价则为 `3M+2S`。归一化 ladder 合计 `6M+4S`，通用差分形式合计 `7M+4S`。S 与 M 的相对成本由具体域算术决定。

## 2. Stage1 与 ladder 不变量

`s=lcm(1,…,B1)=∏ℓ≤B1 ℓ^⌊logℓ B1⌋`。在模素数 p 的非奇异曲线上，`[s]P=O ⇔ ord(P)|s`；只说群阶的素因子≤B1并不足够，还需素数幂指数。额外乘 12/48 会改变可消去的点阶，必须明确指数约定。

```text
(R0,R1)=(O,P)
for bit in scalar s, from MSB to LSB:
  if bit==0: (R0,R1)=(2*R0,R0+R1)
  if bit==1: (R0,R1)=(R0+R1,2*R1)
return R0
```

右侧使用更新前状态，恒有 `R1−R0=P`。可用条件交换实现同形计算。恢复中间运算需要两相邻点及固定差点的有效表示，不能仅保存 R0。

## 3. Suyama PARAM0

`u=σ²−5`、`v=4σ`、`P=(u³:v³)`；`a24=(v−u)³(3u+v)/(16u³v)`，`A=4a24−2`。所有除法表示模逆，分母不可逆时先对 N 求 GCD。差分公式若采用仿射输入，必须用 `u³/v³`，不能把射影 X=u³ 直接当作 x。

## 4. Lucas / PRAC 的复用与成本

链的整数调度由标量和成本权重决定，不依赖 N 或 sigma；同一链可供不同曲线复用。点表和差分点依赖具体曲线，不能跨曲线共享。计划构建、筛法、缓存 IO 的耗时需实测，不能对大 B1 一概宣称构造成本可忽略。

## 2. 两套参考源码的关系

### 2.1 Prime95：运行时搜索的 PRAC

本地 Prime95 参考文件是 `.refactor/p95v3106b01.source/ecm.cpp`：

| 位置 | 内容 | 移植时应保留的行为 |
|---|---|---|
| [ecm.cpp:2640](../../.refactor/p95v3106b01.source/ecm.cpp#L2640) | DBL=10、ADD=12 的 FFT 成本 | 转为 GPU 实际算子权重，不直接套 FFT 时间 |
| [ecm.cpp:2645](../../.refactor/p95v3106b01.source/ecm.cpp#L2645) | `lucas_cost` | 初始/最终运算及各归约规则必须全部计数 |
| [ecm.cpp:2711](../../.refactor/p95v3106b01.source/ecm.cpp#L2711) | `lucas_mul` | A/B/C 的交换、差值点和 scratch 语义 |
| [ecm.cpp:2733](../../.refactor/p95v3106b01.source/ecm.cpp#L2733) | 第一条规则消除部分 C=A 拷贝 | 可优化复制，但不能改变点关系 |
| [ecm.cpp:2777](../../.refactor/p95v3106b01.source/ecm.cpp#L2777) | 简化规则，阈值 2.96 | 使用实际启用的规则，不计入注释掉的原始规则 |
| [ecm.cpp:2847](../../.refactor/p95v3106b01.source/ecm.cpp#L2847) | `lucas_cost_several` | 邻近 d 搜索、相同成本时保持首次候选 |
| [ecm.cpp:2857](../../.refactor/p95v3106b01.source/ecm.cpp#L2857) | `ell_mul`，十组比例种子 | 初值使用 ceil；默认每组搜七个邻近 d |
| [ecm.cpp:6904](../../.refactor/p95v3106b01.source/ecm.cpp#L6904) | `PracSearch` | 默认 7，范围 1～50 |
| [ecm.cpp:7448](../../.refactor/p95v3106b01.source/ecm.cpp#L7448) | 素数重复次数 | 对每个 p 执行 floor(log_p B1) 次 [p] |
| [ecm.cpp:7461](../../.refactor/p95v3106b01.source/ecm.cpp#L7461) | choose12 | 额外两个 2、一个 3 |

在 S≈M 且 DBL/DADD 权重为 5/6 的成本模型下，恰为 Prime95 搜索权重 10/12 的一半。因而在这个简化成本模型内，直接采用其种子和邻域搜索是合理起点。加减、拷贝、spill、特殊首步及未来更便宜的平方都会影响实际最优计划。

这里的 Stage1 对照是 **素数乘法重复**，不覆盖 Prime95 `lucas_mul` 对任意复合 64 位乘数的递归外层；实际 Stage1 在 [ecm.cpp:7469](../../.refactor/p95v3106b01.source/ecm.cpp#L7469) 调用的也是当前素数。

### 2.2 `.refactor/ecm`：带离线优化 Lucas 链的 GMP-ECM 分支

用户指定的 `.refactor/ecm` 包含另外一套实现：

- [ecm.c:549](../../.refactor/ecm/ecm.c#L549)：`generate_Lucas_chain`，解码已有 uint64 链码。
- [ecm.c:1027](../../.refactor/ecm/ecm.c#L1027)：最大延伸，按已知差值点扩展链。
- [ecm.c:1086](../../.refactor/ecm/ecm.c#L1086)：16 槽 X/Z 环。
- [ecm.c:1121](../../.refactor/ecm/ecm.c#L1121)：加载 `Lchain_codes.dat`。
- [ecm.c:1187](../../.refactor/ecm/ecm.c#L1187)：从 p=11 开始逐素数读取一个 uint64；EOF 后回到 PRAC。
- [ecm.c:1214](../../.refactor/ecm/ecm.c#L1214)：执行链中的 DBL/DADD。
- [LCG_macros.h:356](../../.refactor/ecm/LucasChainGenerator/src/LCG_macros.h#L356)：在相同搜索长度下优先保存倍点数更多的候选。
- [LucasChainGenerator/README](../../.refactor/ecm/LucasChainGenerator/README#L1)：生成器用法和历史生成时间。

生成器优化最小/近最小链长度，并在相关长度内偏好多倍点；这不自动等于 GPU 上全局最短时间。需要进一步计入活跃点和 field operation 的实际成本。

#### 大 B₁ 的离线生成时间

本地 GMP-ECM 参考版本为 `c84da4480154cf0bd300aae83201aaf55a2cc38f`。上游 README 的时间覆盖生成整个 `Lchain_codes.dat`，包括为所有目标素数寻找并编码链。代表机器为 Intel Core i9-13900K，约 5 GHz。README 未给出重复次数、编译器、功耗或分阶段计时。本表引用上游记录。

| $B_1$ | 线程 | 生成时间 |
| ---: | ---: | ---: |
| 1,000,000 | 7 | 18.6 秒 |
| 11,000,000 | 7 | 7 分 32 秒 |
| 43,000,000 | 7 | 46 分 55 秒 |
| 110,000,000 | 7 | 3 小时 18 分 11 秒 |
| 260,000,000 | 8 | 9 小时 40 分 24 秒 |
| 850,000,000 | 7 | 45 小时 25 分 32 秒 |

从 $B_1=10^6$ 到 $2.6\times10^8$，素数个数约增至 181 倍，上游生成时间约增至 1,872 倍，两个样本分别用 7 和 8 线程。仅按素数个数线性外推会明显低估生成时间。这两个样本无法单独量化深度、分段读写和线程调度各自的成本。 [上游 README](../../.refactor/ecm/LucasChainGenerator/README#L30)

- **搜索按链长展开。** 外层逐个增加允许的链长 $L$。该轮新目标最大值为 $\min(B_1,F_{L+2})$，上一轮仍未找到链的素数进入后续轮。递归枚举可用的差分加法和倍点，靠 Fibonacci 最大延伸、整除条件与当前最小目标剪枝。$B_1$ 越大，目标素数越多，达到高位目标所需的链也越深。即使共享前缀并剪枝，留到后面的分支仍会随着搜索深度增加。 [外层链长循环](../../.refactor/ecm/LucasChainGenerator/src/LucasChainGen.c#L4203) · [递归与剪枝](../../.refactor/ecm/LucasChainGenerator/src/LucasChainGen.c#L2858)
- **同一长度还比较候选链。** 找到可以到达某个素数的链后，该轮仍统计其他候选，优先保留倍点多的链，再偏好更短的编码。链长超过 26 时才启用该处的多线程分配，将预设子树交给 worker，随后按目标素数合并。目标表和待写链码有固定容量，长链轮可能按子区间重复运行枚举并读写中间表。上游未给出各项耗时占比。 [线程分配](../../.refactor/ecm/LucasChainGenerator/src/LucasChainGen.c#L4548) · [结果合并](../../.refactor/ecm/LucasChainGenerator/src/LucasChainGen.c#L3872) · [分段](../../.refactor/ecm/LucasChainGenerator/src/LucasChainGen.c#L4263)
- **生成后顺序消费链码。** Stage1 可逐素数读取 64 位链码，利用目标素数补全省略的最大延伸步，再执行相应的点运算。表中的分钟和小时是一次性生成时间。 [链码编码](../../.refactor/ecm/LucasChainGenerator/src/LucasChainGen.c#L2885) · [Stage1 链码读取](../../.refactor/ecm/ecm.c#L1187)

#### 可复用的链码文件

- 参考工作树中的 `Lchain_codes.dat` 是由 Git 忽略的本机生成物。旁边的 `LCG.log` 标注 $B_1=10{,}000$。文件为 9,800 B、1,225 个小端 uint64，从素数 11 到 9,973。新格式生成器本轮生成的 300 万档文件，其前 9,800 B 与这份本机文件逐字节相同。 [本地链码](../../.refactor/ecm/Lchain_codes.dat) · [生成日志](../../.refactor/ecm/LCG.log)
- Git 历史提交 `f4af741fb129af96f6decb771a4cabc747fd6705` 确实跟踪过 `Lchain_codes.dat`，共 2,361,056 B、295,132 个小端 uint64，逐条解码验证后确认覆盖到素数 4,182,163。提交 `12b86e7` 删除该旧文件，`6b3058a` 引入新编码。旧、新文件均从素数 11 开始。旧版首段占 4 bit，当前首段占 3 bit，处理历史文件时须使用对应提交的解码逻辑。可用 `git -C .refactor/ecm show f4af741:Lchain_codes.dat` 读取原始字节。 [当前分发说明](../../.refactor/ecm/INSTALL-ecm#L109)
- 这两种上游格式都是无头的每素数链码。归档时记录编码版本、最大覆盖素数和摘要。ECM-ELF v2 缓存的记录是本项目 PRAC 赢家种子，两个格式的执行表示不同。


参考：[EFD Montgomery XZ](https://hyperelliptic.org/EFD/g1p/auto-montgom-xz.html)、GMP-ECM `parametrizations.c`、Prime95 `ecm.cpp` 的 Lucas 链入口。
