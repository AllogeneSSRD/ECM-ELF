# GPU ECM Stage2：梅森承载余因子与 D/P、显存联合规划

日期：2026-10-09。源码基线：`3bc9d7f`（Stage2 Benchmark）。
第1–9节是研究快照；代码链接行号已同步至第10节实现的工作树。

**阶段进展**：已实现默认关闭的`--carrier-exponent`。7995-bit目标、B2=2.6e12、
固定D的同二进制交错计时从165.18降至113.17 s，减少31.49%；完整叶子摘要与
算术检查通过。D/P联合规划仍处于设计阶段，下一项是审核并验证NTT的B/Q复用。

## 1. 结论与实施顺序

**建议先实现“目标余因子 N、算术承载模数 M”分离，再统一普通 Stage2 与 Auto B2 的内存规划。**

1. **梅森承载在数学上可行，而且有 GMP-ECM 的直接实现依据。** 令 `N | M`、
   `M=2^p−1`，设备加减乘和多项式计算在 M 中进行；分母求逆、非单位判定、
   因子 GCD 和存档身份仍针对 N。不能只把当前 `L.N` 改成 M。
2. **首个实验应选 `N=M8011/80111`，固定当前 D。** 7995→8011 bits 使 limb 数
   125→126，但现有三档 B2 对应的 fold NTT 长度均不增加。最大档当前
   `D=810810,P=77760` 的 owner 只从519.11增加到523.26 MiB。
3. **不能对所有余因子无条件启用。** 现有9个位宽×3档 B2 的固定 D 几何分析中，
   7/27格会使 fold NTT 长度增加，其中383→1009 bits 的中档增加到4倍。
   6797→7001 bits 虽只增加204 bits，最大档也会从 `2^27` 跳到 `2^28`。
4. **当前 D 规划确有改进空间，但增大预算本身不能解决最大位宽的问题。**
   普通固定 B2 入口使用旧代价排序；Auto B2 有独立的 profile 与驻留筛选，
   但二者尚未共享完整的同时存活分配模型，owner 布局口径也不同。
5. **8 GiB 设备上不能直接照搬 Prime95 的更大 P。** 对8011-bit承载，
   `P=92160` 已使 NTT A/B/Q 从3072跳到6144 MiB。按当前生命周期，四项
   必需分配的下界为7428.62 MiB，超过历史可用7106 MiB；这还没有加表、
   S4输出等。扩大 fold 或 arena 配额不能使这组现有布局可行。

第1–9节保留首轮研究快照：源码研究、公式推导、离线规划工具与图表；其中的
候选收益不代表实测加速。后续承载实现、校验与实测单独记录在第10节。

## 2. 研究基线与问题定位

### 2.1 已有 CPU/GPU 记录

完整计时、配对规则、功耗条件与样本波动见
[CPU/GPU 对比报告](D:/code/MPA-OpenCl/docs/PRIME95_GPU_STAGE2_COMPARISON_20261009.md)。
本报告继承其中“实际整数 N 相同”的配对，避免把完整梅森数和余因子混为同一任务。

对 `N=M8011/80111`、实际7995 bits、请求 `B2=2.6e12`：

- Prime95 CPU：109.987 s；GPU：182.948 s。
- GPU giant生成34.798 s、G树90.224 s、fold29.001 s，合计84.2%。
- GPU baby生成/归一化约12.106 s；整个初始化16.021 s。
- 嵌套 S4 设备归约事件46.295 s，除以总墙钟为25.3%。它已包含在上述父阶段内，
  不能再次堆叠。
- CPU参考日志选 `D=1531530,P=138240`，12个PolyG；GPU选
  `D=810810,P=77760`，42个G多项式、41次fold。

GPU主数据来自55 W条件。功耗修复后的三遍补测仅覆盖7995 bits、中档 B2，
不能将其10.2%的改善套用于本报告最大 B2 数据。CPU/GPU的B1、sigma、实际B2
边界和并行背景任务也不完全相同，因此这些记录用于定位，而非严格同曲线A/B。

**优化目标必须同时覆盖：点模乘、系数归约、多项式调用数量和 NTT 长度台阶。**
即使在串行预算近似中完全扣掉46.295 s归约事件，也仍约136.65 s；这不是运行时间
预测，只说明单独优化归约尚不足以解释全部差距。

### 2.2 本轮使用的资料

- 引用讨论“查找 ECM stage2 算法来源”及本地
  [文献综述](<D:/code/MPA-OpenCl/.refactor/Stage2/ECM Stage 2 文献综述与关键章节译述.md>)。
- 现行生产源码 `src/cuda/ecm_cuda_stage2.cu` 与共享规划头文件。
- `.refactor/p95v3106b01.source/ecm.cpp`；这是31.6b01参考源码，实测CPU程序为
  31.4b05，不能把参考源码结论当作对旧二进制每条指令的确认。
- `.refactor/gmp-ecm/mpmod.c` 的特殊形式表示和原始模数处理。
- `.refactor/Stage2` 中Montgomery、Zimmermann、Brent等原论文；另核对近年的
  原地多项式运算论文，见第8节。

## 3. 梅森模数怎样承载余因子计算

### 3.1 两个模数必须有明确职责

定义：

```text
N = 实际待分解余因子
M = 2^p - 1，且 M % N == 0
S_N = bit_length(N)                 S_M = p
W_N = ceil(S_N / 64)                W_M = ceil(S_M / 64)
R_M = 2^(64 W_M) mod M              // 点算术 Montgomery 基数的剩余
π(a) = a mod N                      // 从承载环投影到目标环
```

因为 `N | M`，自然映射 `Z/MZ → Z/NZ` 保持加减乘：

```text
π(a + b mod M) = π(a) + π(b) mod N
π(a · b mod M) = π(a) · π(b) mod N
```

因此 xADD/xDBL、乘积树、fold、scaled descent 的加乘运算都可在 M 中承载。
该论证**不要求 `gcd(N,M/N)=1`**，也不要求 M 或 N 为素数。

求逆需要另一条规则。若 `gcd(z,N)=1`，先计算 `u=z^−1 mod N`，再把整数
`0≤u<N` 提升到 M 的表示中。此时 `π(zu)=1`，足以保证后续投影正确，
**不要求 `zu=1 mod M`**。

一个会实际影响代码分支的例子：

```text
M = 2047 = 23 × 89，N = 89，z = 23
z^-1 mod N = 31
23 × 31 = 713 = 1 mod 89，713 != 1 mod 2047
gcd(z,M) = 23，但 gcd(z,N) = 1
```

若代码仍在 M 中判断分母是否可逆，就会把已经剥离的23重新当成异常。
本方案允许承载点在被剥离因子对应的分量上退化，只要求其在目标 N 上的投影
满足原算法的不变量。

### 3.2 多项式除法为何仍正确

当前 baby 多项式为 `F(X)=∏(X−x_j)`，是首一多项式。首一多项式除法在交换环
上成立；反转后的 F 常数项为1，其形式幂级数逆可由 Newton 迭代构造。
这里不需要把 `Z/MZ` 当作域，也不需要对任意非单位求逆。

巨点当前使用射影叶子 `Z_i X−X_i`，相应G树并不必须在 M 中首一。
其尺度积 Γ 的逆在 N 中取得并提升后，最终校正投影仍正确。不能要求校正后的
G/H在 M 中与通用 N 路径逐字一致。

代码依据：

- [射影G叶与尺度关系](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7414)。
- [多项式逆常数项处理](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4686)、
  [另一逆实现入口](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6079)。
- [Γ逆的最终处理](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7771)。

### 3.3 Prime95 与 GMP-ECM 已怎样处理

Prime95构造目标N并除去已知因子，但 `gwsetup(k,b,n,c)` 仍使用原特殊形式：

- [目标N构造](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:1162)。
- [初始算术设置](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:7208)、
  [Stage2重新选择FFT](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:7990)。
- [求逆使用实际N](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2992)、
  [逆转换回gwnum](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:3009)。

GMP-ECM提供更直接的工程参考：

- [BASE2表示初始化](D:/code/MPA-OpenCl/.refactor/gmp-ecm/mpmod.c:839) 保留
  `orig_modulus=N`，随后构造 `2^p±1` 并检查其能被N整除。
- [特殊形式折叠](D:/code/MPA-OpenCl/.refactor/gmp-ecm/mpmod.c:112) 没有要求
  每次乘法后再归约到原始N。
- [mpres_invert](D:/code/MPA-OpenCl/.refactor/gmp-ecm/mpmod.c:1996)、
  [mpres_gcd](D:/code/MPA-OpenCl/.refactor/gmp-ecm/mpmod.c:2038)、
  [mpres_equal](D:/code/MPA-OpenCl/.refactor/gmp-ecm/mpmod.c:2051)
  都针对原始模数处理逆、GCD或相等。

因此这是已有算法工程方法向本项目移植，主要难点是当前单模数接口与运行状态，
而不是提出新的 ECM Stage2 算法。

### 3.4 当前 GPU 已有什么，缺什么

已有快速归约：

- [S4梅森折叠](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2044)：
  利用 `2^p≡1`，用低位加高位替代整数长除法。
- [S4分派](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2227)、
  [只接受输入本身为exact Mersenne的判定](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2389)。
- [点Montgomery梅森归约](D:/code/MPA-OpenCl/src/cuda/stage2/stage2_point_mersenne.cuh:6)：
  折叠后通过循环位移处理 `R_M^−1`。
- [点乘分派](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:695)。

缺口：

- [PolyLayer](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1308) 只有一个 `N`，
  同时代表运算模数和待分解整数。
- [run_real初始化](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8251) 直接用输入N
  决定S/W和所有后续算术。
- [运行/规划API](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2.h:14)
  未传入原梅森指数或承载信息。
- [Stage1记录结构](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:93)
  保存求值后的N和X，但没有保留可直接用于承载证明的结构化原始表达式。
  队列本身有指数信息，见
  [任务指数](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:375)。

## 4. 收益与代价：先看具体输入

### 4.1 计算量变化

当前通用点乘包含学校式乘积和Montgomery消去，两个主要双重循环的量级约为
`2 W_N²` 个64-bit乘加；梅森路径保留乘积，但消去变为线性折叠和位移，约为
`W_M²+O(W_M)`。依据
[乘积与通用消去循环](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:685)。

对于125→126 limbs，仅比较二次项：

```text
2 × 125² / 126² = 1.9684
```

这只是主要乘加数量之比，**不是点内核或完整Stage2的加速比**。指令调度、
进位链、寄存器、访存和发射开销仍在。

S4通用系数尾部现已是归一化整数长除法，不应继续按旧版两次Montgomery消去
估算。若待除数有约 `2W_N` 个limbs，长除法的商位×除数循环约为
`(W_N+1)W_N`，另加修正；梅森折叠主要为 `O(W_M)`。
见[当前长除法](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1998)。

多项式乘法的NTT和pack/unpack仍然执行；承载位宽增加还可能扩大NTT。
因此较合理的阶段模型为：

```text
T_stage2 = T_points(backend,W) + Σ T_NTT(L,batch,policy)
         + Σ T_pack/reduce(backend,S,coeff_count)
         + T_transfer + T_CPU_prepare + T_GCD + T_other
```

不能用一个 `W²` 公式代替整个 Stage2。

### 4.2 7995→8011 bits 是最合适的首个目标

在当前最大档 `P=77760`：

```text
                     余因子通用路径        梅森承载候选
S                    7995                 8011
W                    125                  126
fold NTT L           2^27                 2^27
packing bpw          19                   19
每系数slot digits    843                  845
fold owner / MiB     519.1107             523.2636
```

多项式调用结构、G数量和最大NTT长度可保持一致，有利于归因。
位宽与owner的增长分别约0.2%和0.8%；NTT长度不增长并不表示pack、归约或表缓存
开销完全相同，仍需测量。

### 4.3 哪些输入不宜直接启用

基于完整GPU分析中27组余因子的**原D固定不变**，新增工具重算了承载的fold长度：

- 318→503 bits：前两档B2长度增加2倍，最高档不增加。
- 383→1009 bits：三档分别增加2、4、2倍；算术位宽本身也明显增加。
- 4667→5003 bits：最低档增加2倍。
- 6797→7001 bits：最高档增加2倍。
- 1939→2003、2726→3001、3600→4001、5872→6011、7995→8011 bits：
  这批记录对应的三个D上均不增加。

这些是**几何变化**，不是速度预测。最终应枚举 `(backend,D)`，允许通用N路径
赢过梅森承载，也允许承载路径改选较小P以避开台阶。

原实验直接输入完整M8011的692.507 s不能用于预测承载性能：当时分解目标也
变成M，已知因子80111重新进入非单位检测，G叶准备/回退耗时544.441 s。
本方案继续分解N，需要阻止这种目标语义变化。

## 5. 当前 D/P、内存规划的四个问题

### 5.1 普通固定 B2 路径的校准模型没有启用

[cache_rates_valid=false](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8323)
以及后面的[calibrated=false](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8354)
使冻结阶段速率在当前策略下失效。这是代码明确保留的保护，不能简单去掉条件
就恢复为“已校准”。

实际使用的
[legacy_56_1模型](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8375)
按旧参考点估算 `I log P`、P、I和批次数，不反映当前位宽的归约差异及离散NTT跳变。

其owner/baby驻留筛选放在 `if(calibrated)` 分支，见
[模型分支](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8402)；
因此普通路径可能选到运行时退回非驻留fold的候选。运行时仍有owner预算检查，
见[FoldDeviceState::init](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5248)。

### 5.2 Auto B2 仍绑定旧策略，未接通当前生产布局

不能把上一个问题概括成“所有planner都忽略fold预算”。Auto B2在
[候选过滤](D:/code/MPA-OpenCl/src/core/ecm_stage2_cost_profile.h:162)
明确检查arena和resident owner。

这里调用 `geometry(p,bits,query,geom)` 没有传owner reuse值；
[默认参数](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:60)
为0，而生产
[kFoldOwnerReuse](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5152) 为3。

```text
reuse=0: owner = 8W(9P+8)+48
reuse=3: owner = 8W(7P+7)+48

S=7995, P=77760:
Auto B2默认布局计算 667.43 MiB > 640 MiB
生产reuse=3实际布局 519.11 MiB < 640 MiB
```

**补充追踪：** [select_auto配置约束](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:606)
明确要求旧 `NTT_FOLD_OWNER_REUSE=0`，并同时限制旧seed、gscale、root/frontier策略；
当前生产入口要求reuse=3。因此这是旧成本profile尚未接通新生产策略，不能描述成
“当前Auto B2静默使用错误owner”。旧profile作用域内，reuse=0公式有其匹配依据。
而且目前select_auto只接受exact Mersenne输入，7995-bit余因子也会先被拒绝；上面
的数字是同几何的新旧布局比较，并非一次已发生的Auto B2任务退化。

旧离线工具也保留 `9P+8`：
[plan_stage2_d.py](D:/code/MPA-OpenCl/tools/bench/plan_stage2_d.py:46)、
[calibrate_stage2_d.py](D:/code/MPA-OpenCl/tools/bench/calibrate_stage2_d.py:140)。
接通时应让运行、Auto B2、校准、离线查询共享同一个layout参数，扩展backend作用域，
更新profile版本并重新标定。不能仅改公式或删除这些配置保护。

### 5.3 arena估计不是共享池的实际同时占用

[共享geometry](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:62)
近似累加fold形状和两个tree形状的 `3L+out_slots`。
但当前NTT默认使用
[一个共享A/B/Q工作池](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:1873)，
按最大 `L×nbatch` 增长，且
[增长前释放旧池](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:1917)。

例如7995 bits、P77760：旧arena估计6146.37 MiB；主工作池3072 MiB，已有
运行统计NTT full峰约3186 MiB。该差异说明旧筛选偏保守，**不说明多出来的预算
都可以给更大P**：owner、G树与坐标等分配在arena之外，同时存活。

新模型要分别给出workspace、缓存表、small、fuse base和外部owner，按阶段求峰值。
不要将当前几个日志模块的各自峰值直接求和。

### 5.4 树形状和巨点chunk也必须来自真实执行结构

共享geometry只查询名义 `P/2+1` 系数的tree top。
当前产品树按2的幂补齐，真实顶层子树可能有
`h=2^floor(log2(P−1))` 个叶子。
P77760时，名义38881系数的NTT为 `2^26`，真实65537系数子树可为 `2^27`。
在该例中fold最大形状仍更大或相等，但树计时、缓存和batch分配不能只看名义值。
见[G树构造](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4357)。

巨点坐标目标预算硬编码256 MiB，随后**向上**取整到P的整数倍：
[chunk计算](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7263)。
P77760、W126的chunk含155520点，X/Z共299.00 MiB；256 MiB在这里不是硬上限。
新规划应显式枚举chunk，或者改成以整棵G树为单位向下取整的有界策略，并评估
增多的seed/chain初始化成本。

## 6. 可计算的显存下界与 NTT 台阶

![承载位宽、D/P与显存台阶](D:/code/MPA-OpenCl/docs/figures/stage2_carrier_plan_20261009.png)

图上方叠加的是**已核对生命周期、在重复G树/fold阶段同时存活的四项分配**。
它是完整进程显存的下界；下方右图是固定D时的NTT长度比，不是速度比。

### 6.1 exact packing 与大工作池

设m为当前查询的最大操作数系数数，当前Goldilocks质数 `q=2^64−2^32+1`：

```text
slot_bits = 2S + max(1, ceil(log2 m))
b = 满足 m·ceil(slot_bits/b)·(2^b−1)^2 < q 的最大合法digit位宽
slot_digits = ceil(slot_bits/b)
L = 最小的2次幂，满足 L >= 2m·slot_digits + 1

fold查询: m = P+1
单次最大fold大工作池: V_big = 3·8·L = 24L bytes
```

这是当前实现的保守exact卷积打包规则，见
[choose_cfg](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2690)、
[digit上界](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2746)、
[shape query](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:3068)。
批量小树调用应使用 `24·max_j(L_j batch_j)`，并另计输出和缓存；这里的24L
专用于单个最大fold形状的下界。

### 6.2 owner、raw G 与坐标

```text
W = ceil(S/64)
I = floor(B2/D)+2                         // 当前GPU覆盖规则
G = ceil(I/P)                            // 重复G树时
V_owner = 8W(7P+7)+48                     // 当前reuse=3
V_rawG = 8W(3P+ceil(P/2))                 // compact raw A/B
k = max(P, floor(256 MiB/(16W)))
C = min(I, P·ceil(k/P))                   // 当前向上取整的坐标chunk
V_coords = 16WC

V_concurrent_lower = 24L + V_owner + V_rawG + V_coords
```

该式适用于本次 `I>P` 的重复G树分析。raw G两个容量来自
[rawA/rawB申请](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4372)；
它们由S4状态保留。owner在
[巨点循环之前](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7257) 申请，
坐标由
[ResidentGiant](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6633) 保有并跨多个G树使用。

尚未计入：NTT tables/fuse base/small、S4设备输出、seed、segment products、
索引、oracle相关缓冲等。因此**下界超预算可排除当前布局；下界通过不能证明可行。**
对非驻留、不同chunk、非compact路径或单G树，需要单独的生命周期模型。

### 6.3 8011-bit、B2=2.6e12 的候选

离线计算的关键结果，单位MiB：

```text
D        P       G    NTT大池  owner    raw G   坐标     同时存活下界
510510    46080  111    3072   310.08   155.04  265.78   3802.91
690690    63360   60    3072   426.36   213.18  365.45   4076.99
810810    77760   42    3072   523.26   261.63  299.00   4155.90
1021020   92160   28    6144   620.16   310.08  354.38   7428.62
1141140  103680   22    6144   697.68   348.84  398.67   7589.19
1381380  126720   15    6144   852.72   426.36  487.27   7910.34
1531530  138240   13    6144   930.24   465.12  265.78   7805.14
```

这里GPU覆盖公式得到D1531530对应13个G；Prime95日志的12个PolyG来自它自己的
起始位置、覆盖取整和切片规则，二者不能直接等同，也不能用42/12预测收益。

对8011-bit承载，fold长度限制为 `2^27` 时最大P是79417；当前77760已很靠近
该台阶。7995-bit通用输入对应上界79606。两者都容不下92160。

历史日志在CUDA上下文建立后报告free7106 MiB；扣reserve768后可规划6338 MiB，
arena配置6300 MiB。P92160的下界比6338多1090.62 MiB，还未计其他缓冲。
因此先做以下两类工作：

1. 在 `L=2^27` 平台内寻找更优D/P、chunk和缓存策略。
2. 若要跨到 `L=2^28`，研究降低大池或变换的内存复杂度，再评估更大P。

例如该长度一个64-bit变换数组是2048 MiB。若未来能通过合法的生命周期复用减少
一个完整数组，量级足以跨越上述缺口；**当前尚未证明A/B/Q可以直接二合一**。
别名、异步oracle和借出的digit指针必须逐一审计。

## 7. 建议的实现设计

### 7.1 第一阶段：明确算术域与目标域

建议加入共享 `Stage2ModulusContext`，职责至少包括：

```text
target_N, carrier_M
target_bits, carrier_bits, target_words, carrier_words
carrier_kind = generic | mersenne
carrier_exponent
Montgomery radix/domain constants for carrier_M
```

普通模式令M=N。各层借用同一上下文，避免在两个类中各自维护会失配的N副本。

需要逐项审计：

- **M域**：device模数、n_inv64、R/R²/R逆表示、S4归约参数、点归约、poly
  运算、NTT打包位宽、整系数GMP oracle、设备结果是否在 `[0,M)`。
- **N域**：Suyama构造中的分母、baby/giant仿射逆、segment单位判断、Γ逆、
  数学零/相等判断、GCD、因子输出、队列expected N、Stage1存档校验与X范围。
- **转换**：N中求出的逆是普通整数；提升后必须用M域的正确表示编码，不能
  继续使用原W_N对应的Montgomery常数。
- **验证**：内部M算术对照仍在M中；通用N与承载M的端到端比对则投影到N。
  射影点用N上的交叉乘积或归一化比较，原始M字数组digest不要求相等。
- **零判断**：M中为0一定投影为0，反向不成立。凡零值影响控制流的地方，
  必须判断其在N中的语义；仅作运算优化且不改变投影结果的判零可另行证明。

最主要的修改落点：

- [模数与bit/word初始化](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8251)。
- [Montgomery常数](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8578)、
  [曲线构造](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8600)。
- [saved X检查](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8612)、
  [存档checksum](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:136)。
- [baby批逆](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8808)、
  [giant设备分组逆](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6662)。
- [giant segment逆](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7485)、
  [备用segment/仿射路径](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7524)。
- [最终block GCD](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8011)、
  [叶子因子检查](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8097)。

形如 `2^-k`、Montgomery解码、F的常数1求逆属于算术表示，不能把所有
`mpz_invert(...,L.N)` 机械替换成target_N。

### 7.2 承载信息的来源与启用条件

队列的 `(k,b,n,c)=(1,2,p,−1)` 可以提供p，但仍需对实际存档N检查 `N | 2^p−1`。
单独读取save时，应保留解析后的原始表达式来源，或提供显式的承载指数参数。
不得只依据N的bit_length猜测p。

第一版只支持已有内核覆盖范围内的梅森承载，`p≤16384`，检查载入N为合法奇数、
saved X在N范围内、M能被N整除、M域所有shape受支持。不能满足时保留通用模式。

建议第一轮作为显式实验选项，固定D、同save、同算法开关和相同目标N；在正确性
与计时证据完成前，不将所有余因子的默认行为切换到承载。

### 7.3 第二阶段：所有入口共用 MemoryPlan

建立一个共享planner，普通固定B2、Auto B2、tune和离线查询调用相同逻辑。
输入包含：

```text
目标/承载上下文、B1/B2、候选D
实际CUDA free/total、reserve、arena/fold配置
owner reuse、workspace pool、S4 batch/output policy
giant chunk、G树布局、scaled frontier与缓存策略
```

输出包含每阶段存活对象、最大 `L×batch`、mandatory payload、可驱逐缓存、
预计峰值、驻留与fallback路径、置信范围和不能选用的原因。

生命周期至少覆盖：baby生成、F树、inverse、G树/fold循环、fold到frontier交接、
下降和GCD。尤其
[fold/frontier交接](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7698)
可能继续借用owner，不能在进入descent时一律假设owner已释放。

候选筛选先满足强制工作集，再决定可保留哪些表和缓存。最终仍以运行时申请和
实际free为准；GPU可能被其他进程占用，planner结果不是cudaMalloc成功承诺。
现有owner申请的
[额外1 GiB headroom](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5253)
也要纳入一致的策略，避免planner说驻留、运行时又按另一套规则回退。

### 7.4 第三阶段：以真实形状和后端成本联合选 D

候选不应只包含D：

```text
candidate = (generic_N | mersenne_M, D/P, giant_chunk,
             S4_chunk, fold_resident, table_cache_policy)

固定B2: minimize predicted Stage2 time
Auto B2: maximize estimated ECM benefit(B1,B2) / (T_stage1 + T_stage2)
```

Auto B2保留已确认的“总流程收益”目标，Stage1时间按实际target_N和Stage1后端估算；
Stage2按carrier_M及其后端估算，不能为了启用承载就把两阶段位宽键同时改成p。

tune需记录：实际NTT `(L,batch,policy)` 吞吐、每类系数归约、点链/ladder、
搬运和CPU准备成本，以及驻留/回退两个策略的阶段时间。
已有
[Auto B2 Work树统计](D:/code/MPA-OpenCl/src/core/ecm_stage2_cost_profile.h:90)
可复用，但应和真实树调用结构保持一致。

profile键增加target/carrier位宽、backend、owner layout版本和内存policy。
当前
[profile校验](D:/code/MPA-OpenCl/src/core/ecm_stage2_cost_profile.h:51)
已有binary/device/hash与accounting版本约束；应扩展它并重新标定，不能复用旧速率
后仅修改版本常量。当前profile范围到8192 bits，亦不同于内核16384-bit输入上限。

不建议永久禁止非驻留策略：若将来其实测整体时间更好，应允许参与比较；当前必须
先避免在未建模的情况下意外退化。

### 7.5 最小实验次序与验收指标

以下是后续实施后的验证计划，本轮未启动这些运行。

1. **域分离正确性**：覆盖上文2047/89式“目标可逆、承载不可逆”，以及
   `gcd(N,M/N)>1` 的小合成例；检查所有分支对目标N的意义。
2. **固定D归因**：同一个有效 `M8011/80111` Stage1 save，固定当前D/P，
   同二进制切generic/carrier，保留算术检查。先小B2确认，再用最大档读阶段计时。
3. **既有因子命中**：确认两路径对N找到相同有效因子；不能只比较NF计时或hash。
4. **联合选形**：分别固定backend扫描NTT台阶附近的D/P，再比较planner的选择；
   记录预测/实际显存、fallback、G数量、实际L、阶段和完整墙钟。
5. **跨位宽回归**：特别检查7001、1009这两类承载扩大NTT的输入，允许选择generic。

性能指标以完整Stage2墙钟和各父阶段为主；`t_reduce`、NTT事件只作归因。
不将点乘MAC比、G数量比或NTT长度比直接写成完整加速比。

## 8. 论文中哪些方法值得进一步采用

### 8.1 当前框架已经采用的基础

Montgomery1992论文的POLYEVAL、重复G批次及固定F/RECIPF复用，与本仓库主线对应；
本地PDF物理第55–59页，尤其Fig.4.3.1。它适合指导减少固定操作数的重复变换。
见[原论文](<D:/code/MPA-OpenCl/.refactor/Stage2/An FFT Extension of ECM of Factorizatimontgomery.pdf>)。

《20 Years of ECM》物理第8–10页讨论多项式Stage2、Kronecker打包和分批的计算/内存
权衡。其简化模型的最优批数不含本项目NTT台阶、传输和驻留回退成本，不能直接
当作GPU的最优G数。[作者论文](https://members.loria.fr/PZimmermann/papers/ecm-submitted.pdf)

Brent–Kruppa–Zimmermann的FFT extension论文对reciprocal多项式除法和POLYEVAL/
POLYGCD给出背景；本项目已经走POLYEVAL与scaled下降路线。本轮优先优化表示和
规划，暂不以POLYGCD替换整套算法。
[作者论文](https://maths-people.anu.edu.au/~brent/pd/rpb264.pdf)

Bernstein2004的scaled remainder tree在本仓库已采用。原文物理第7页的父变换和
F变换复用值得核查；但不同层的Kronecker打包位宽与padding变化，使原论文的
变换节省比例不能原样用于当前NTT。
[本地论文](D:/code/MPA-OpenCl/.refactor/Stage2/scaledmod-20040820.pdf)

### 8.2 固定 F、finv 的频域缓存：可行但要服从内存规划

各次fold使用固定F/finv，缓存其打包后的forward NTT可能减少重复工作。
不过 `L=2^27` 时，每个64-bit频域数组为1024 MiB，两个就是2048 MiB；
8 GiB设备上可能因此失去owner驻留或更优D。

优先核实当前融合策略到底执行了多少次固定操作数forward变换，再按“节省时间 /
新增同时存活字节”决定是否缓存。若采用更小的middle/short product形状，其缓存
长度也应重新查询；不能默认所有操作都需要完整fold的L。

### 8.3 2020、2024、2026：原地 remainder 与 accumulation

Giorgi–Grenet–Roche2020研究原地除法、多点求值和插值。适合作为减少树frontier和
临时多项式存储的候选，但额外重算与依赖会影响GPU并行性。
[论文](https://arxiv.org/abs/2002.10304)

Dumas–Grenet2024直接研究不保存完整商的多项式余数、over-place余数和累加模乘。
这与fold临时数组有较强对应。其“常数额外空间”模型允许改写并恢复输入，且基础
乘法可以另有工作空间；因此不等价于本项目的6144 MiB NTT池消失。全文以域上的
代数RAM表述，迁移到复合模数时须检查标量逆的单位条件。
[作者全文，§1.1与后续算法](https://membres-ljk.imag.fr/Bruno.Grenet/publis/DuGre24rem.pdf)

《Fast in-place accumulation》扩展上述工作，期刊卷期为JSC134（2026），预印本
日期2025-10-22，包含累加乘法、short product和模余数等。已核对出版商摘要与章节
预览，尚未完成全文的CUDA执行映射。它提供减少临时商/乘积存储的方向，不是可直接
替换当前NTT的GPU库。
[出版商条目](https://www.sciencedirect.com/science/article/pii/S0747717125001051)

建议优先级：

1. 利用现有梅森归约内核，完成target/carrier分离。
2. 统一layout与生命周期规划，校准D/backend联合代价。
3. 审核三大NTT数组的用途；研究blocked/truncated/short/middle product降低
   实际变换长度和scratch，处理P跨台阶问题。
4. 再评估原地fold/下降与固定操作数频域缓存；每项同时记录算术、访存、显存和重算。

第3、4项的算法可结合，但其工程风险与性能常数明显高于第1项，不建议同时改动，
否则难以判断收益来源。

## 9. 本轮工具与复现

新增：

- [analyze_stage2_carrier_plan.py](D:/code/MPA-OpenCl/tools/bench/analyze_stage2_carrier_plan.py)：
  用整数公式计算目标/承载的NTT形状、owner、坐标chunk、并存下界及NTT临界P。
- [plot_stage2_carrier_plan.py](D:/code/MPA-OpenCl/tools/bench/plot_stage2_carrier_plan.py)：
  输出PNG/SVG；若有此前完整位宽扫描分析，则追加固定D的承载长度比图。
- [工具说明](D:/code/MPA-OpenCl/tools/bench/README_STAGE2_CARRIER_PLAN.md)。

仓库根目录运行：

```powershell
python tools/bench/analyze_stage2_carrier_plan.py --output data/stage2_carrier_research_20261009
python tools/bench/plot_stage2_carrier_plan.py --input data/stage2_carrier_research_20261009/analysis.json --output-prefix docs/figures/stage2_carrier_plan_20261009
```

结果：14个目标/承载几何候选，27个已有测量形状的承载重算。
`analysis.json` 标记 `measured=false`，记录被分析源文件与既有分析文件的SHA256。
它不启动ECM、不申请CUDA显存，也不替代生产C++ shape query。

沿用现有排除规则，数据留在根目录`data/`，图留在`docs/figures/`，均不提交；
源码工具和本报告可提交。文中的数字与公式保留在报告内，图可由上述命令重建。

## 10. 第一阶段实现与实验：目标 N、梅森承载 M 分离

### 10.1 已实现的接口与职责

新增实验选项 `--carrier-exponent <p>`，0为默认通用路径。非零时要求保存点目标
`N | (2^p−1)`，p不大于16384。读取保存点、校验checksum、队列整数身份和结果
中的N均继续使用原始余因子；载入的归一化X不变。

- [ModulusContext](D:/code/MPA-OpenCl/src/core/ecm_stage2_modulus.h:12)
  保存目标与承载，验证奇数、位宽、整除关系。为控制改动范围，既有引擎的
  `L.N`成员名保留为**算术承载模数**；数学目标必须显式写`L.target()`。
- [公开API](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2.h:14)
  为run/plan增加可省略的承载参数；[命令行解析](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:258)
  与子进程传参同步支持。非零p进入队列进度身份，避免混用不同算术计划。
- [run_real初始化](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8246)
  以承载M决定S/W、Montgomery常数、NTT打包、S4归约与显存形状。曲线a24在
  [目标N中构造](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8600)，再编码到M域。
- baby的分组根求逆、host回退与零判断针对N；[GPU baby校验](D:/code/MPA-OpenCl/src/cuda/stage2/stage2_baby_host.cuh:115)
  把设备叶子投影到N后比较，不要求其原始M代表元等于N代表元。
- [巨点base单位判定](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6711)、
  [驻留Gamma分组求逆](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6662)、
  segment逆、非单位回退和最终block/leaf/candidate GCD均针对N。
  Γ乘积和多项式运算继续在M中计算；Γ逆只需在N中正确。
- [最终GCD](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8011)
  不会重新报告已剥离的M/N因子。NTT、S4、Montgomery内核的独立算术oracle
  仍比较**承载域M**中的精确算术，没有改为仅比较N来放宽内核检查。
- [目标叶子摘要](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7900)
  由`NTT_TARGET_LEAF_HASH=1`开启，按N的limb数输出完整投影序列的64-bit摘要
  （xor/multiply，沿用起始值1469598103934665603、乘数1099511628211）。
  它用于跨后端对照，并不替代独立算术检查或CPU oracle。

当前保留实验边界：默认不启用；使用显式B2；不接入旧Auto B2成本profile；
开发引擎明确拒绝非零承载参数。尚未自动识别最优承载，也没有把旧D模型标记为
已校准。NTT内核调度、F/G树与fold算法均沿用原实现。

### 10.2 已通过的正确性对照

以下均使用同一生产引擎二进制、GPU1，强制算术检查没有关闭。

1. **M8011/80111，N=7995 bits**：B1=20、sigma=26、B2=26000000000、
   D=180180、P=17280。两个后端的17280个目标叶子、2160000个目标limb摘要相同：
   `1676070873149334833`，全部非零。GPU chain与独立ladder覆盖138240个巨点，
   两条路径均零不匹配；baby、seed、segment检查通过。该轮开启额外诊断，
   **93.7412/72.4654 s仅为校验用时，不属于性能样本**。
2. **M37/223，N=30 bits**：24个叶子的完整摘要与独立CPU monic计算一致，
   为`17212892664144190202`。参考ladder证实baby scalar=29、giant scalar=6090
   和12180的分母满足`gcd(Z,M)=223`、`gcd(Z,N)=1`。承载路径不因这些已剥离
   因子的非单位而回退或报因子。
3. **M67/193707721，N=40 bits**：算术limb从1增为2；24个叶子的摘要与CPU
   oracle一致，为`15702641285353165899`。
4. **M29/233**：独立ladder发现目标分母非单位，两个后端均报告1103、2089。
5. **M253/23**：`gcd(N,M/N)=23`，展示不要求两个商因子互素的情况；独立
   分母GCD证据与两个后端均得到47、96209。非单位回退的X可能依赖projective
   代表元，不把这类叶子当作一般monic求值oracle。
6. **M16384，目标N=2^8192+1（8193 bits）**：算术采用256个64-bit limb，
   同时覆盖最大承载位宽及`p%64=0`边界。24个叶子、3096个目标limb均与独立
   CPU monic参考一致，摘要为`12805860702850186770`，全部非零。参考计算发现
   88个仅在已剥离因子中非单位的分母；目标域中均可逆。

命令行边界检查使用同一7995-bit保存点：p=8011接受；p=8009（不整除）、
p=7990（不足目标位宽）、p=16385（超上限）均以退出码2拒绝，未启动曲线或改写队列。

另在正式计时的完整形状`B2=2.6e12、D=810810、P=77760`上单独运行两个后端，
比较77760个叶子投影到N后的9720000个limb；全部非零，摘要均为
`15186198829546221849`。每条路径还通过2304个GMP自检、51956项运行检查，
零错误。该轮使用`check --projection-only`，省去额外逐点ladder诊断，仍保留
强制算术检查；165.061235/113.225171 s仅作为校验用时，未混入§10.4的n=4统计。
日志的`clean`标记目前不识别新增叶子摘要开关，采集器以显式环境区分诊断与计时，
不依赖该标记将校验曲线认定为正式样本。

小规模输入来自
[prepare_stage2_carrier_inputs.py](D:/code/MPA-OpenCl/tools/bench/prepare_stage2_carrier_inputs.py:36)，
使用既有`tools/stat/suyama_mont_ref.py`计算Stage1、baby和giant；保存点checksum
按当前格式重新计算。参考源码SHA256：
`69df92e7d56e90c03ce5712bc08f902e96fe2a878925e98c9a1b944bf936132e`。
上述小规模检查是功能证据，不用于推断大位宽性能。

实验记录位于被排除的`data/stage2_carrier_ab_20261009/`。早期采集器曾只读取
可读日志或只读取调试日志，导致缺少检查字段；原始成功曲线和失败采集记录均保留。
最终采集器同时读取两路日志，校验矩阵经过身份核对后恢复原始成功调用，未将其
重新归类为正式计时样本。

### 10.3 D/P联合规划的下一项具体内存机会

继续追踪NTT大池的真实用途后，识别到一个比泛称“减少scratch”更具体的候选：
**在生产的非导出digit路径中，inverse结束后复用B作为carry输出Q**。

源码的先后关系是：

1. [A、B各自forward](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:3475)。
2. [inverse(A,B)](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:3489)
   将pointwise乘积、scale与inverse融合，结果原地写A。
3. [carry(A,Q)](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:3510)
   读取A、写Q，后续S4读取Q；此后的主路径没有B读者。

因此B与Q的主路径生命周期不重叠，且同一default stream提供先后顺序。
这只是**源码层面的候选复用证明**，本阶段没有修改分配或通过运行时别名门禁。
首次实施建议只覆盖`workspace_pool && allow_pool`，继续保留导出digit的旧布局；
[digits_out控制pool许可](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:3934)。
还需同步处理`words/3`容量口径、独立所有权与释放、输入别名快照、失败回退、
deferred carry和异步oracle生命周期，不能只把`dQ=dB`后继续原释放逻辑。

若这一候选通过验证，单fold大池可由`24L`变为`16L`：

```text
L=2^27：3072 -> 2048 MiB，节省1024 MiB
L=2^28：6144 -> 4096 MiB，节省2048 MiB

8011 bits，B2=2.6e12，假设仅改变大池：
D=1021020 P= 92160 G=28：并存下界7428.62 -> 5380.62 MiB，owner=620.16 MiB
D=1381380 P=126720 G=15：并存下界7910.34 -> 5862.34 MiB，owner=852.72 MiB
D=1531530 P=138240 G=13：并存下界7805.14 -> 5757.14 MiB，owner=930.24 MiB
```

这些仍是遗漏tables、S4输出等的下界，不是可行性承诺或性能预测。尤其需要联合
调高owner预算、检查实际剩余显存，并计入更长fold NTT的成本：G数量变少不代表
fold总工作一定减少。

当前巨点坐标chunk向上取P的倍数也会造成不连续：P=126720时坐标约487.27 MiB，
P=138240时反而约265.78 MiB。联合规划应枚举满足下限的chunk大小，评估减少坐标
容量带来的seed/launch开销，而不是把256 MiB当作现有实现的严格上限。

还须把GPU baby的独立预算纳入同一计划。当前
[d_baby_payload_bytes](D:/code/MPA-OpenCl/src/cuda/stage2/stage2_d_model.cuh:74)
与运行时使用3P坐标/输出、8层product tree、索引与常量的同一公式；W=126、
P=138240时约532.10 MiB，超过默认512 MiB，仍会退回host归一化。提高owner
预算并不改变这项限制。P=126720约487.76 MiB，处于该baby预算内。

建议下一阶段顺序：先以独立开关验证B/Q复用和真实内存ledger，再把新布局加入
MemoryPlan；之后测量D/P/owner/chunk组合，更新普通D和Auto B2的共同成本模型。
第一阶段实测不包含这一候选，避免把两项优化的收益混在一起。

### 10.4 同二进制高B2实测：减少31.49%的完整Stage2时间

**结论范围**：本节是同一个目标、同一保存点、固定D的承载后端A/B，证明该形状
下的收益。它不是所有位宽的平均加速，也不包含D/P联合规划、B/Q复用或Stage1。

实验配置：

```text
设备：GPU1，NVIDIA GeForce RTX 4060 Laptop GPU，sm_89
目标：N=(2^8011−1)/80111，7995 bits
保存点：B1=20，sigma=26，归一化Z=1
B2=2600000000000，D=810810，P=77760
giant_points=3206671，G树=42，fold=41
arena=6300 MiB，resident fold owner=640 MiB，S4 batch=256 MiB
A：--carrier-exponent 0；B：--carrier-exponent 8011
各预热1次，然后ABBA+BAAB；各路径n=4
额外叶子/逐点诊断关闭，生产强制算术检查保留
```

小B1只用于快速生成有效保存点；这里计时从读取归一化点开始，不含Stage1。
保存点本身对本次测试有效，但换成生产B1、其它sigma或有因子命中的曲线仍需独立
测量。固定D用于隔离表示与归约收益，没有重新搜索D。

正式样本按实际顺序列出，单位s；预热165.374966/113.042928不计入统计：

```text
ABBA：A 165.423438；B 113.167897；B 113.187561；A 165.137519
BAAB：B 113.183276；A 165.023390；A 165.135484；B 113.126752

A mean=165.179958，sample SD=0.170856，range=[165.023390,165.423438]
B mean=113.166372，sample SD=0.027730，range=[113.126752,113.187561]
时间减少=31.4890%，速度比=1.45962×，每曲线节省52.013586 s
两个四次交错组分别减少31.5238%、31.4542%
```

未剔除慢样本；两组均无因子命中、`bad_factors=0`，强制Montgomery、S4除法和
GMP检查均零不匹配，异步oracle队列排空。以报告的完整`stage2_full_wall.total`
为统计口径，不以CUDA事件时间代替墙钟。

父阶段均值与占完整时间的比例：

```text
阶段                     通用N：s（%）       承载M：s（%）
初始化                    14.174（8.58）        9.262（8.18）
巨点生成                  34.445（20.85）      20.876（18.45）
G树                       79.537（48.15）      55.852（49.35）
fold                      24.959（15.11）      17.016（15.04）
下降                       6.109（3.70）        4.997（4.42）
多项式逆                   2.299（1.39）        2.201（1.94）
其它（含GCD及未归类开销）   3.657（2.21）        2.963（2.62）

嵌套设备事件S4 t_reduce：41.562250 -> 8.015000 s，减少80.72%
```

最后一行已包含在上述父阶段中，**不能再相加**。归约的设备事件减少33.55 s，
完整墙钟减少52.01 s；梅森点算术等也同时获益，不能把全部收益归于S4。G树仍占
49.35%，继续优化NTT及G批次数具有价值。

![相同目标的完整Stage2时间与阶段占比](D:/code/MPA-OpenCl/docs/figures/stage2_carrier_20261009.png)

两组`ntt_workspace_stats.full_peak_bytes`均为3340874032，即3186.106 MiB；
大池均为3072 MiB。owner峰值由519.111增至523.264 MiB，增加4.153 MiB，仍在
640 MiB预算内。本形状没有NTT长度跳变。这些是模块容量峰值，不能相加作为
进程显存峰值。

全程仅每2 s只读采样GPU1，没有修改功耗、时钟或设备状态；GPU0原有任务继续
运行。利用率不低于90%的采样中，通用/承载的平均核心频率分别为2386.64/
2381.93 MHz、平均功耗76.23/75.82 W。采样较稀疏且受阶段混合影响，只作为
运行条件记录，不能据此证明所有瞬时功耗限制相同。两组交错结果一致，未见本轮
收益随运行顺序消失。

历史GPU182.948 s样本来自55 W时期；历史CPU约109.987 s使用不同曲线、B1和
覆盖条件。因此不能用113.17与这些数直接计算“对CPU加速比”，也不能把历史到
当前的全部改善归于本补丁。

### 10.5 构建与证据身份

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/carrier_stage2 -Arch sm_89 -SplitCompile 8
python tools/bench/bench_stage2_carrier.py --exe build_cuda_cmake/carrier_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --b2 2600000000000 --d 810810 --mode timing --telemetry --output data/stage2_carrier_ab_20261009/m8011_high_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/stage2_carrier_ab_20261009/m8011_high_timing/measurements.json --output docs/benchmarks/stage2_carrier_20261009_analysis.json --figure-prefix docs/figures/stage2_carrier_20261009
```

本轮CUDA13.3、MSVC14.51.36231、GMP zen3，PTX=3、add/sub mask=1、outer=0；
构建耗时103.7 s。只生成仓库内独立二进制，未替换外部生产目录。

```text
binary SHA256:
c6a9e7b6bb5ca7a172cec8ee4f8fbff6b38a1e4b6eccc65376e2a06ca087195b
build_manifest SHA256:
30434d5c97d8cb74a8838046895508521922ded30a90793ea1bf229ce944712f
frozen_sources_manifest SHA256:
37871cb77d647a9bde68dde0e55717d0cbed598b02a822c976160e0ff86c6765
Stage1 save SHA256:
971b10df29925b9a74858e50d3cd7b3550f83e3c918368f97d5921bb4bf6ba71
formal measurements.json SHA256:
1c30b9b1ad31ce4d7680004b34cce72e1fe30f5e616af14fdf3ba8bf2e7ed352
```

[bench_stage2_carrier.py](D:/code/MPA-OpenCl/tools/bench/bench_stage2_carrier.py)
保存每次命令、环境、两路原始日志、结果和源码闭包身份；
[analyze_stage2_carrier_bench.py](D:/code/MPA-OpenCl/tools/bench/analyze_stage2_carrier_bench.py)
只接受完整正式矩阵，校验原始日志摘要，输出均值、标准差、范围和阶段图。
原始证据和图沿用仓库排除规则；报告内保留数字，脚本可重建图形。

下一阶段继续按§10.3实施内存布局与联合规划。承载后端暂保持显式启用：此前
27个形状中有7个在承载后NTT长度增大，不能将本次31.49%外推后直接默认开启。
