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

本地链码文件为 9,800 字节、1,225 个 uint64 记录，按从第一个素数 11 起的顺序解码，最后一个是 9973。首尾及每条使用的差值关系按原文的本地链码副本核对。文件没有包含 B1、版本、字节序或校验和的 header；新的生产缓存应补上这些元数据。


参考：[EFD Montgomery XZ](https://hyperelliptic.org/EFD/g1p/auto-montgom-xz.html)、GMP-ECM `parametrizations.c`、Prime95 `ecm.cpp` 的 Lucas 链入口。
