# GPU ECM Stage2：梅森承载余因子与 D/P、显存联合规划

日期：2026-10-09。源码基线：`3bc9d7f`（Stage2 Benchmark）。

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

本轮交付是源码研究、公式推导、离线规划工具与图表。**没有修改生产 CUDA 行为，
也没有新增 GPU 计时结果；文中的候选收益不代表实测加速。**

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

- [射影G叶与尺度关系](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7413)。
- [多项式逆常数项处理](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4685)、
  [另一逆实现入口](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6078)。
- [Γ逆的最终处理](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7770)。

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
- [点乘分派](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:692)。

缺口：

- [PolyLayer](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1305) 只有一个 `N`，
  同时代表运算模数和待分解整数。
- [run_real初始化](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8231) 直接用输入N
  决定S/W和所有后续算术。
- [运行/规划API](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2.h:12)
  未传入原梅森指数或承载信息。
- [Stage1记录结构](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:92)
  保存求值后的N和X，但没有保留可直接用于承载证明的结构化原始表达式。
  队列本身有指数信息，见
  [任务指数](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:368)。

## 4. 收益与代价：先看具体输入

### 4.1 计算量变化

当前通用点乘包含学校式乘积和Montgomery消去，两个主要双重循环的量级约为
`2 W_N²` 个64-bit乘加；梅森路径保留乘积，但消去变为线性折叠和位移，约为
`W_M²+O(W_M)`。依据
[乘积与通用消去循环](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:682)。

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

[cache_rates_valid=false](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8301)
以及后面的[calibrated=false](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8332)
使冻结阶段速率在当前策略下失效。这是代码明确保留的保护，不能简单去掉条件
就恢复为“已校准”。

实际使用的
[legacy_56_1模型](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8353)
按旧参考点估算 `I log P`、P、I和批次数，不反映当前位宽的归约差异及离散NTT跳变。

其owner/baby驻留筛选放在 `if(calibrated)` 分支，见
[模型分支](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8380)；
因此普通路径可能选到运行时退回非驻留fold的候选。运行时仍有owner预算检查，
见[FoldDeviceState::init](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5247)。

### 5.2 Auto B2 仍绑定旧策略，未接通当前生产布局

不能把上一个问题概括成“所有planner都忽略fold预算”。Auto B2在
[候选过滤](D:/code/MPA-OpenCl/src/core/ecm_stage2_cost_profile.h:162)
明确检查arena和resident owner。

这里调用 `geometry(p,bits,query,geom)` 没有传owner reuse值；
[默认参数](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:60)
为0，而生产
[kFoldOwnerReuse](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5151) 为3。

```text
reuse=0: owner = 8W(9P+8)+48
reuse=3: owner = 8W(7P+7)+48

S=7995, P=77760:
Auto B2默认布局计算 667.43 MiB > 640 MiB
生产reuse=3实际布局 519.11 MiB < 640 MiB
```

**补充追踪：** [select_auto配置约束](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:597)
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
见[G树构造](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4356)。

巨点坐标目标预算硬编码256 MiB，随后**向上**取整到P的整数倍：
[chunk计算](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7262)。
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
[rawA/rawB申请](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4371)；
它们由S4状态保留。owner在
[巨点循环之前](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7256) 申请，
坐标由
[ResidentGiant](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6632) 保有并跨多个G树使用。

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

- [模数与bit/word初始化](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8231)。
- [Montgomery常数](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8555)、
  [曲线构造](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8577)。
- [saved X检查](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8589)、
  [存档checksum](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:135)。
- [baby批逆](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8784)、
  [giant设备分组逆](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6661)。
- [giant segment逆](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7484)、
  [备用segment/仿射路径](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7523)。
- [最终block GCD](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7991)、
  [叶子因子检查](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8077)。

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
[fold/frontier交接](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7697)
可能继续借用owner，不能在进入descent时一律假设owner已释放。

候选筛选先满足强制工作集，再决定可保留哪些表和缓存。最终仍以运行时申请和
实际free为准；GPU可能被其他进程占用，planner结果不是cudaMalloc成功承诺。
现有owner申请的
[额外1 GiB headroom](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5252)
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
