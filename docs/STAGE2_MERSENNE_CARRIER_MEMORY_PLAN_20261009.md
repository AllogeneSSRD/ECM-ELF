# GPU ECM Stage2：梅森承载余因子与 D/P、显存联合规划

日期：2026-10-09。源码基线：`3bc9d7f`（Stage2 Benchmark）。
第1–9节是研究快照；代码链接行号已同步至第23节实现的工作树。

**阶段进展**：已实现默认关闭的`--carrier-exponent`。7995-bit目标、B2=2.6e12、
固定D的同二进制交错计时从165.18降至113.17 s，减少31.49%；完整叶子摘要与
算术检查通过。已实现默认关闭的NTT B/Q复用：大形状实际少1024 MiB，固定D的
正式计时基本持平。扩大D的同二进制正式计时113.09→96.84 s，减少14.37%；
第11–13节记录B/Q、阶段释放、共享数量策略与驻留/回退检查。
第15节完成物理工作区分块实验：子调用减少45.06%，完整时间反而慢0.635%，
因此该策略保持默认关闭。
第16节已完成输出释放与驻留对照：95.95→85.03 s，减少11.38%；保持原分块策略，
尚未推广发布默认或接入完整联合规划。
完整生命周期MemoryPlan和新Auto B2成本仍待完成。
第17节修正padded树顶尺寸，新增精确树请求/保留容量模型及实际阶段观测；
旧共享大池重复计费尚未作为完整准入模型替换。
第18节补齐单树NTT的表/base合同，修复子集观测重复计费，并验证驻留前回收冷缓存。
第19节覆盖完整设备分配生命周期，得到同时存活owned payload峰及逐分配来源，
发现giant坐标chunk向上取整使较小D的完整占用反而更高。
第20节实现并验证整批坐标预算：D138完整owned峰减少259.01 MiB，同binary完整
时间85.18→91.44 s，增加7.35%；默认保持原策略，完整MemoryPlan仍待完成。
第21节建立覆盖五阶段的预测式乘法请求program；12次算术运行与40组规划配置
核对通过，两个高位宽D的无淘汰NTT保留峰与实测相等。尚未替代完整owned
payload生命周期模型或D/Auto B2准入；这一轮没有新增正式性能收益结论。

第22节接入有序NTT分配器模拟：共享池、keyed digits、表/base、cap淘汰与拒绝前缀；
CPU分配器核对及无曲线规划查询通过。非NTT生命周期和真实free准入仍待接入。
第23节接入giant组件生命周期，覆盖保留seed工作区、chain/ladder尾块、segment/group
与最终累积缓冲，并修正S3扩容统计。已核对历史同时存活分配台账；尚未与NTT、
S4、owner边界合成完整MemoryPlan，不新增运行加速结论。
用户已恢复4060lp默认1800 MHz/55 W，后续计时以此为新基线；历史约79 W、
2385 MHz的计时保留原条件，不能混合作为新基线。

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

- [射影G叶与尺度关系](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7511)。
- [多项式逆常数项处理](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4751)、
  [另一逆实现入口](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6153)。
- [Γ逆的最终处理](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7881)。

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

- [S4梅森折叠](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2076)：
  利用 `2^p≡1`，用低位加高位替代整数长除法。
- [S4分派](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2259)、
  [只接受输入本身为exact Mersenne的判定](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2421)。
- [点Montgomery梅森归约](D:/code/MPA-OpenCl/src/cuda/stage2/stage2_point_mersenne.cuh:6)：
  折叠后通过循环位移处理 `R_M^−1`。
- [点乘分派](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:700)。

缺口：

- [PolyLayer](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1340) 只有一个 `N`，
  同时代表运算模数和待分解整数。
- [run_real初始化](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8363) 直接用输入N
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
[乘积与通用消去循环](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:690)。

对于125→126 limbs，仅比较二次项：

```text
2 × 125² / 126² = 1.9684
```

这只是主要乘加数量之比，**不是点内核或完整Stage2的加速比**。指令调度、
进位链、寄存器、访存和发射开销仍在。

S4通用系数尾部现已是归一化整数长除法，不应继续按旧版两次Montgomery消去
估算。若待除数有约 `2W_N` 个limbs，长除法的商位×除数循环约为
`(W_N+1)W_N`，另加修正；梅森折叠主要为 `O(W_M)`。
见[当前长除法](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2030)。

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

[cache_rates_valid=false](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8435)
以及后面的[calibrated=false](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8466)
使冻结阶段速率在当前策略下失效。这是代码明确保留的保护，不能简单去掉条件
就恢复为“已校准”。

实际使用的
[legacy_56_1模型](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8487)
按旧参考点估算 `I log P`、P、I和批次数，不反映当前位宽的归约差异及离散NTT跳变。

其owner/baby驻留筛选放在 `if(calibrated)` 分支，见
[模型分支](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8514)；
因此普通路径可能选到运行时退回非驻留fold的候选。运行时仍有owner预算检查，
见[FoldDeviceState::init](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5313)。

### 5.2 Auto B2 仍绑定旧策略，未接通当前生产布局

不能把上一个问题概括成“所有planner都忽略fold预算”。Auto B2在
[候选过滤](D:/code/MPA-OpenCl/src/core/ecm_stage2_cost_profile.h:162)
明确检查arena和resident owner。

这里调用 `geometry(p,bits,query,geom)` 没有传owner reuse值；
[默认参数](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:180)
为0，而生产
[kFoldOwnerReuse](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5217) 为3。

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

[共享geometry](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:183)
近似累加fold形状和两个tree形状的 `3L+out_slots`。
但当前NTT默认使用
[一个共享A/B/Q工作池](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:1891)，
按最大 `L×nbatch` 增长，且
[增长前释放旧池](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:1934)。

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
见[G树构造](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4422)。

巨点坐标目标预算硬编码256 MiB，随后**向上**取整到P的整数倍：
[chunk计算](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7353)。
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
[choose_cfg](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2773)、
[digit上界](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2829)、
[shape query](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:3151)。
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
[rawA/rawB申请](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4437)；
它们由S4状态保留。owner在
[巨点循环之前](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7346) 申请，
坐标由
[ResidentGiant](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6709) 保有并跨多个G树使用。

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

- [模数与bit/word初始化](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8363)。
- [Montgomery常数](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8781)、
  [曲线构造](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8803)。
- [saved X检查](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8815)、
  [存档checksum](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:136)。
- [baby批逆](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:9097)、
  [giant设备分组逆](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6738)。
- [giant segment逆](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7582)、
  [备用segment/仿射路径](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7621)。
- [最终block GCD](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8122)、
  [叶子因子检查](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8208)。

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
[fold/frontier交接](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7799)
可能继续借用owner，不能在进入descent时一律假设owner已释放。

候选筛选先满足强制工作集，再决定可保留哪些表和缓存。最终仍以运行时申请和
实际free为准；GPU可能被其他进程占用，planner结果不是cudaMalloc成功承诺。
现有owner申请的
[额外1 GiB headroom](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5318)
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
- [run_real初始化](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8357)
  以承载M决定S/W、Montgomery常数、NTT打包、S4归约与显存形状。曲线a24在
  [目标N中构造](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8803)，再编码到M域。
- baby的分组根求逆、host回退与零判断针对N；[GPU baby校验](D:/code/MPA-OpenCl/src/cuda/stage2/stage2_baby_host.cuh:115)
  把设备叶子投影到N后比较，不要求其原始M代表元等于N代表元。
- [巨点base单位判定](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6787)、
  [驻留Gamma分组求逆](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6738)、
  segment逆、非单位回退和最终block/leaf/candidate GCD均针对N。
  Γ乘积和多项式运算继续在M中计算；Γ逆只需在N中正确。
- [最终GCD](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8122)
  不会重新报告已剥离的M/N因子。NTT、S4、Montgomery内核的独立算术oracle
  仍比较**承载域M**中的精确算术，没有改为仅比较N来放宽内核检查。
- [目标叶子摘要](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8010)
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

1. [A、B各自forward](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:3558)。
2. [inverse(A,B)](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:3572)
   将pointwise乘积、scale与inverse融合，结果原地写A。
3. [carry(A,Q)](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:3593)
   读取A、写Q，后续S4读取Q；此后的主路径没有B读者。

因此B与Q的主路径生命周期不重叠，且同一default stream提供先后顺序。
这只是**源码层面的候选复用证明**，本阶段没有修改分配或通过运行时别名门禁。
首次实施建议只覆盖`workspace_pool && allow_pool`，继续保留导出digit的旧布局；
[digits_out控制pool许可](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:4017)。
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

## 11. 第二阶段：NTT B/Q复用及扩大D的验证

### 11.1 两个物理缓冲承载A/B/Q

实现实验环境开关`NTT_WORKSPACE_REUSE_BQ=1`，默认0。仅在共享pool且调用不
导出digit指针时，使`Q=B`；A保持独立。在inverse写完A之后，carry才开始向B写
规范化digit。使用同一默认流保证inverse的全部B读者先完成，未增加同步或复制。

对单次长度L、批数b、容量恰好足够的调用：

```text
三缓冲 big payload = 3 × L × b × 8 bytes
两缓冲 big payload = 2 × L × b × 8 bytes
省去Q payload      =     L × b × 8 bytes
```

池复用更大的旧容量时，实际收费按保留的capacity计算，而不是按当前L×b缩小。
此改动减少分配与占用，没有删减forward/inverse/carry步骤，不能将33.3%的大池
节省解释为33.3%的NTT工作量或运行时间减少。

关键实现：

- [BigEntry](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:1876)
  增加每个物理分配的`capacity`和`q_alias_b`；`words`只表示实际拥有的总字数。
  查容量与输入跨度不再依赖`words/3`。
- [池分配](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2145)
  根据开关分配2或3个缓冲；确认A/B成功后才设置Q借用B。失败回滚只释放真正
  申请过的指针，不会把别名当作第三份所有权。
- [drop_workspace](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:1934)
  释放Q时检查所有权，避免double free；增长前释放旧池，保留原来的串流等待语义。
- [input_span](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:1947)
  识别A/B/Q的完整物理范围。设备输入属于arena时，原有调用层先保存两个输入，
  再查找或增长目的缓冲；Q=B不会绕过交叉输入快照。
- `digits_out!=nullptr`继续使用三个独立的keyed缓冲；关闭pool和每调用分配回退
  同样保留三缓冲。该阶段没有改变这些接口的借用生命周期。
- dOut、每个形状自己的dRes、S4归约输出和oracle的pinned host快照均未合并。
  [oracle异步D2H](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:3199)
  在默认流中先捕获digit与归约结果，下一次forward才能覆写B；CPU延迟核查读取
  的是已捕获的host快照，不再持有Q设备指针。
- [布局统计](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2010)
  单列requested、实际别名、capacity、物理缓冲数、复用调用数及省去Q的容量峰值。
  保持既有`full_peak_bytes`为真实模块拥有的分配容量，未把虚拟Q重复计入。
  原`aliases`字段仍统计输入快照，B/Q复用单独记录在新layout字段中。

默认仍关闭，旧Auto B2 profile明确要求该开关为0。本节第一版的普通D几何和起始
内存摘要仍采用三缓冲估算，第12节再统一策略；目前只在显式D/B2实验中启用，
完整联合MemoryPlan尚未接入。第10节的“尚未修改分配”是第一阶段快照。

### 11.2 独立检查与接口边界

扩展现有
[ntt_workspace_check](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:4224)，
同时覆盖两/三缓冲以及pool开/关，128项断言、260个输出word与GMP参考对比，
`bad=0`。具体包括：

1. `L×b`相等而形状不同的容量复用；单独的dRes保持原值。
2. 每项大分配的失败注入、增长回滚、cap拒绝、重复release与收费归零。
3. 交叉A/B输入、旧Q作为下一次输入，以及Q=B情况下先保存两个输入再写目的区。
4. 导出digit时Q与B独立；keyed缓存被驱逐时输入仍正确。
5. 低cap触发每调用回退，carry不会错误地声明为延迟检查。
6. 连续两次延迟carry不重置dRes；其单调max字段的sentinel=123在最终readback中
   仍为123，同时输出与GMP一致。

继承第10节五个独立参考输入，按同一算术后端只切换B/Q布局：M37、M67、M29、
M253、M16384全部通过。unit案例完整目标叶子摘要与CPU一致，非单位案例的独立
因子证据保持一致。另对`N=2^8192+1`关闭承载、使用8193-bit通用归约，两个布局
的完整叶子摘要及强制检查也一致。检查用时不作为性能样本。

### 11.3 复现与实验范围

本阶段独立构建目录为`build_cuda_cmake/workspace_bq_stage2`；CUDA13.3、sm_89、
PTX=3、add/sub=1、outer=0，构建102.6 s。第10节二进制与原始证据保持原样。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/workspace_bq_stage2 -Arch sm_89 -SplitCompile 8
python tools/bench/bench_stage2_carrier.py --comparison workspace-bq --exe build_cuda_cmake/workspace_bq_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --b2 2600000000000 --d 810810 --mode check --projection-only --telemetry --output data/stage2_bq_20261009/m8011_high_check
python tools/bench/bench_stage2_carrier.py --comparison workspace-bq --exe build_cuda_cmake/workspace_bq_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --b2 260000000000 --d 810810 --mode timing --telemetry --output data/stage2_bq_20261009/m8011_mid_timing
```

正式时序固定为各一次预热加ABBA+BAAB，固定目标、carrier、保存点、D、B2和预算，
只改变`NTT_WORKSPACE_REUSE_BQ=0/1`。采集器要求两种布局的launch、poly_muls、
coeffs_reduced与强制GMP检查覆盖相同。扩大D的单路径实验使用
`check --projection-only --single-arm two_buffer`，明确标记为探索校验，不能混入
上述正式A/B统计。

### 11.4 完整大形状的实际显存节省

对第10节相同保存点、承载8011、B2=2.6e12、D=810810、P=77760，在同一个本阶段
二进制中只切换布局。两个完整检查调用均通过，目标域全部77760叶子、9720000个
limb的摘要仍为`15186198829546221849`；强制算术检查覆盖相同且零错误。

```text
                              三缓冲             两缓冲
容量：每个物理分配             134217728 words    134217728 words
big实际占用/峰值               3072 MiB           2048 MiB
完整NTT模块容量峰值            3186.106 MiB       2162.106 MiB
驻留fold owner                 523.264 MiB        523.264 MiB
两秒采样设备已用显存的最大值    5086 MiB           4062 MiB
pool增长                       5                  5
pool cudaMalloc次数            15                 10
legacy cudaMalloc次数          0                  0
arena overflow                 0                  0
```

省去1073741824 bytes（1024 MiB）的Q是实际CUDA分配减少，不再只是第10节的预测。
table、small、fuse_base和owner容量不变；没有借助关闭强制检查、转移到legacy或
降低P来实现节省。两秒采样观察到的设备用量也相差1024 MiB，但它包含整个设备
上的其它占用，采样可能漏掉瞬时峰值，不能代替分配ledger。

校验调用完整时间为113.365658/113.048922 s。每种布局仅一条曲线，并开启了额外
投影摘要，因此**只用于校验与内存证据，不作正式加速结论**。同长度的NTT及
归约工作未减少；下一步的时间机会来自在新内存条件下改选D/P。

```text
binary SHA256:
01b34ce5acf391bdcadf6dfc629815a8d4072c73c4ef14b481d0792d332a27e9
build_manifest SHA256:
de72be2aa51bd1c228b3e8f5ec66ef9e752b9798f22ba63ed312c55939226734
frozen_sources_manifest SHA256:
23efa573b14e90fe808492bc15ffa018ab1ffd7f4305709cec6aa3a0049f98c3
```

### 11.5 固定D的正式计时：基本持平

GPU1、同一7995-bit目标/Stage1保存点/承载8011，固定D=810810、P=77760，
B2=2.6e11（5棵G树）。与§11.4相同长度的大NTT，arena=6300 MiB、owner=640 MiB、
S4 batch=256 MiB。额外摘要和fixture全部关闭，`clean=1`；生产强制检查保留。

```text
预热（不纳入统计）：三缓冲28.584389 s；两缓冲28.611062 s

ABBA：三 28.534395；两 28.567975；两 28.486124；三 28.533648
BAAB：两 28.486329；三 28.560645；三 28.626190；两 28.623582

三缓冲 n=4：mean=28.5637195 s，sample SD=0.0434980 s
两缓冲 n=4：mean=28.5410025 s，sample SD=0.0672005 s
均值差0.022717 s（0.07953%）
两个四次组分别减少0.02443%、0.13451%
```

均值差小于组内标准差，本轮不足以证明稳定提速。两组乘法次数、归约系数、launch
和强制GMP检查覆盖相同，无因子命中或算术错误。两秒采样的高负载平均时钟分别为
2383.60/2376.28 MHz，仍有阶段与时钟波动；不将0.08%推广为算法收益。

完整NTT模块峰值由3185.970降至2161.970 MiB，采样设备已用显存最大值由5084降至
4060 MiB。主循环形状未改变，收益仍是省去1024 MiB实际分配，为选择其它D留下空间。

![固定D的正式计时和父阶段占比](D:/code/MPA-OpenCl/docs/figures/stage2_bq_20261009.png)

![模块容量与两秒采样设备用量](D:/code/MPA-OpenCl/docs/figures/stage2_bq_20261009_memory.png)

原始矩阵位于`data/stage2_bq_20261009/m8011_mid_timing/`，均值/范围/标准差与GPU
采样汇总为`docs/benchmarks/stage2_bq_20261009_analysis.json`；图片和数据继续按
已有规则排除。`analyze_stage2_carrier_bench.py`可重建上述两图。

### 11.6 扩大D的第一轮探索：旧headroom阻断驻留

在§11.4的同一目标、保存点、B2、两缓冲后端上分别执行一条完整检查曲线。
arena=6300 MiB、S4 batch=256 MiB、baby=512 MiB；D1021020的fold预算640 MiB，
另两档1024 MiB。各模块数值是自身容量峰值，不构成同一时刻的分配清单。

```text
D          P        G树数   full(s)     NTT full峰(MiB)  owner实际占用
810810     77760    42      113.048922  2162.106          523.264 MiB
1021020    92160    28      121.511653  4611.961          0 / headroom
1381380    126720   15       96.911804  4612.618          0 / headroom
1531530    138240   13      112.413539  4613.356          0 / headroom
```

后三档的fold NTT长度从2^27变为2^28；即使采用两缓冲，big也由2048增至4096 MiB。
它们均未超owner本身的预算，却被旧headroom检查按三缓冲预留增长量而回退，
因此不能把耗时变化单纯归于G树数量或NTT长度。D1531530还跨过131072个叶子的
补齐树边界，baby payload约532.10 MiB超过512 MiB而回退，初始化增至24.97 s。

三个候选的目标域完整叶子摘要依次为：

```text
D1021020: leaves= 92160 words=11520000 hash= 9850534875102471826
D1381380: leaves=126720 words=15840000 hash=14409977791789275297
D1531530: leaves=138240 words=17280000 hash= 1525678423161840651
```

每档强制算术检查零错误，目标因子输出均为空。不同D的baby集合不同，不能
相互比较上述摘要；只用于同D后续实现的回归。全部是n=1探索校验、`clean=0`，
不作为正式加速结论。原始目录为`data/stage2_bq_20261009/m8011_d<D>_check/`。

## 12. 共享NTT布局策略与驻留headroom修正

### 12.1 分配、几何与驻留检查使用同一缓冲数量

引入[NttWorkspacePolicy](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:1815)，
只读取pool和B/Q开关，无CUDA分配。`big_buffer_count(allow_pool)`仅在pool允许且
B/Q启用时返回2，其余返回3。NttArena继承该策略；原生形状查询和普通D几何也
读取同一策略，输出`workspace_buffers`，不再在实验启用时仍显示三缓冲容量。

- [共享Geometry](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:176)接受2或3个物理
  缓冲，默认3兼容旧调用。其arena估计仍是原有保守求和，不等于进程峰值。
- [real_shape_words/real_run_geometry](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8319)
  与分配器共用数量；[计划JSON](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:9607)
  明示布局，便于计划和实际日志核对。
- [fold headroom](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5318)及
  [frontier headroom](D:/code/MPA-OpenCl/src/cuda/stage2/scaled_frontier.cuh:27)
  均使用真实物理数量。两处独立保留的三缓冲公式都需要修正。

```text
target_big = physical_buffers × largest_fold_length × 8
growth     = max(0, target_big − currently_owned_pool_big)
fold允许   = owner_bytes + growth + 1 GiB <= available_at_fold_init
frontier允许= metadata_bytes + growth + 1 GiB <= available_at_frontier_init
```

frontier阶段owner已分配，不能再次从available中扣除整个owner。两处调试日志
记录available、实际缓冲数、target/current/growth与预留量，不改变普通日志粒度。
current_big表示共享pool容量，不包含另行保留的keyed导出缓存。
本次没有缩小固定1 GiB预留，也没有宣称此预留是完整未来分配ledger。

### 12.2 保留的中间失败与验证范围

第一版只修正fold检查时，D1021020实际保留650288064 B owner，但frontier仍按
三缓冲收费而回退。曲线正常完成，完整Stage2=111.469030 s；
`--require-resident`随后明确拒绝该矩阵，未把它当作驻留通过。原始日志保留在
`data/stage2_bq_20261009/headroom_d1021020_check/`。

该调用fold检查的available=1761607680 B、当前big=4294967296 B、growth=0，
owner后尚余1111319616 B，略高于1 GiB。frontier旧式却额外预留2147483648 B增长。
这证明错误的物理数量确实会阻断驻留；是否可长期安全驻留仍须以完整运行验证。

仅修正两处headroom的二进制独立保存在`build_cuda_cmake/workspace_bq_headroom_v2_stage2/`，构建
105.6 s，工具链与§11一致。原生plan门禁检查D810810/D1021020、pool开/关与B/Q
开/关的8组组合：分配数量、fold big、旧arena估计及fits谓词全部一致，执行曲线0。
工具为[test_stage2_workspace_plan.py](D:/code/MPA-OpenCl/tools/test/test_stage2_workspace_plan.py)，
原始JSON/环境/源码身份位于`data/stage2_bq_20261009/headroom_v2_plan/`。

修正两处数量后，D1021020完整叶子摘要保持一致，full=111.658056 s；fold驻留、
frontier仍回退。新增日志显示frontier前available=864026624 B、metadata=2211840 B、
growth=0，确实低于固定1 GiB余量。这次回退不能再归因于三缓冲公式。
D1381380在fold前仅1658847232 B可用，扣894143424 B owner也不足1 GiB。
两个现象进一步指向阶段结束后仍保留的临时分配。

### 12.3 阶段释放已失效的S4 raw输入

新增实验环境开关`NTT_PHASE_TRIM_RAW=1`，默认0，与B/Q独立：

1. [inverse→fold](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7335)：Newton输入已用完，
   F/finv已保存在host向量；在申请fold owner前释放raw A/B。下一棵G树根据自己的
   raw形状重新申请，旧Newton容量不再永久保留。
2. [fold→descent](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7818)：仅当设备owner和
   Gamma校正可供scaled root使用时释放G树raw A/B；最终H、F和finv均由独立owner
   持有。驻留下降的gather读取owner，不依赖这些raw输入。

[raw_release](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1305)只释放两份raw分配，
并同时清空指针与capacity；保留d_out、NTT大池、表、S4归约状态和oracle快照。
cudaFree完成此前默认流的读者后才使指针失效，不提前复用仍在读取的内存。
若frontier因预算或分配原因仍回退，原有host路径可通过raw_reserve重新申请。
两处`stage2_phase_trim`调试日志单列实际释放字节和耗时，阶段计时包含释放成本。

此开关不改变算术或覆盖；[Auto B2 scope](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:612)
仍拒绝未经标定的新配置。最后构建目录为`build_cuda_cmake/workspace_bq_phase_trim_stage2/`，
CUDA13.3/sm89，构建104.3 s。前述所有中间二进制/失败和原始记录保留。

### 12.4 最终构建的门禁与扩大D结果

最终原生plan 8组再次通过；M37/M67/M29/M253/M16384各两种布局，共10条完整
参考曲线通过，workspace fixture每次128项、260 words、0错误。另通用8193-bit
两条通过。开启阶段释放后，M29仍检出1103/2089，M253仍检出47/96209；单位案例
完整monic叶子摘要均与独立CPU参考一致。

对M37两缓冲路径分别注入既有`NTT_SCALED_FRONTIER_ALLOC_FAIL=1`与
`NTT_FOLD_DEVICE_ALLOC_FAIL=1`，两条曲线正常完成、完整目标叶子摘要与独立参考
保持一致。前者在释放G树raw后进入host下降并重新申请raw，验证了实际回退路径，
不只验证成功驻留。控制记录在`data/stage2_bq_20261009/phase_trim_fallback_controls/`。

对7995-bit目标、carrier8011、B1=20、sigma26、B2=2.6e12，统一arena=6300、
fold=1024、baby=640、S4 batch=256 MiB，B/Q与阶段释放均开启。以下每档仍是
一条开启完整投影摘要的探索检查，`clean=0`，不用于正式收益统计：

```text
D          P        G    full(s)     init(s)  giant    G树      fold     descent   驻留fold/frontier
1021020     92160   28   109.257004   10.934  13.406   43.336   29.653    6.623    是/是
1141140    103680   22    98.783721   12.173  11.699   39.057   23.484    7.290    是/是
1381380    126720   15    95.957427   14.780   8.770   32.136   21.067   12.093    否/否
1531530    138240   13   104.195318   16.769  12.294   33.263   18.699   14.046    否/否
```

D1021020/D1381380/D1531530的完整目标叶子摘要与§11.6同D旧构建逐项相同，
因子输出一致，强制检查零错误。D1141140的完整目标摘要为
`leaves=103680 words=12960000 hash=4761271289637678243`，原生驻留门禁通过。
不同D的摘要不可相互等同。所有原始调用在`phase_trim_d<D>_check/`，目录前缀为
`data/stage2_bq_20261009/`。

D1021020两次分别释放185796576/325140480 B；frontier前available从仅修正公式
时的864026624增至1191182336 B，保持1 GiB余量而通过。其下降避免3096576000 B
父状态H2D、1548288000 B中间状态D2H；兄弟F上传、最终叶子读回和强制检查保留。
下降不新增持久多项式缓冲，只复用owner并申请2211840 B metadata。

D1141140分别释放209020896/365783040 B；owner=731573184 B，frontier metadata=
2488320 B。初始化检查通过，但frontier检查超过1 GiB余量的差额仅5900288 B，
约5.63 MiB。单次通过不能作为其它负载/设备上的可靠默认准入保证。

D1381380释放255469536 B旧Newton输入后，fold前available=1914699776 B；扣除
894143424 B owner仍不足1 GiB，继续回退。它虽增加传输，却因G树由42降至15而
取得本轮最短探索full；联合规划必须比较驻留和回退的总成本，不能先排除回退。

D1531530提高baby预算后避免了§11.6的baby回退，init由24.97降至16.77 s。这
同时变更了阶段释放和baby预算，不能将差额全部归于释放。其P超过131072补齐
边界；当前256 MiB巨点策略又从2P/chunk变为P/chunk，13个坐标chunk比D1381380
的8个更多，giant反而更慢。这是D/P、NTT长度、坐标chunk与独立预算需要联合
规划的实测例子。

![两缓冲下的候选并存显存下界与NTT拐点](D:/code/MPA-OpenCl/docs/figures/stage2_bq_plan_20261009.png)

图是整数布局公式分析，不是实测同时峰值；big、owner、G raw和巨点坐标是在
G/fold阶段同时存活的下界，仍遗漏表、S4输出、seed等。通过预算线不证明可申请。
使用`--workspace-buffers 2 --fold-mib 1024 --baby-mib 640`可重建。

```text
最终二进制 SHA256:
83b86b37dab892de9547ed61b42fa89d554dc320abfee29b7d0db5ada9f0a1b5
build_manifest SHA256:
b27b85e87d4791ebcf87cafe4aef6c6ee164b9246e6a6865e2f8ce0bda4da463
frozen_sources_manifest SHA256:
0adf250c58fe775e729f91a8f160bc5858b08d83b07430d3b9ab2c31d57806a6
```

## 13. 同二进制D对照

### 13.1 范围与方法

GPU1 RTX4060 Laptop，最终§12二进制；目标N=M8011/80111、7995 bits、Stage1
B1=20、sigma26，固定carrier8011、B2=2.6e12。两臂均开启B/Q复用和阶段raw释放，
arena=6300、owner=1024、baby=640、S4 batch=256 MiB，chain block64/min32768。
只切换D=810810与1381380，未调整功耗、时钟或外部生产程序。

选择1381380是因为§12.4四档探索中其完整时间最短，而不是因为它的P最大或全部
驻留。探索结果不参与正式统计。各臂先预热一次，再ABBA+BAAB，每臂n=4；额外
叶子摘要、逐点/故障fixture关闭，强制GMP检查保留，正式调用要求`clean=1`。

最终原D的额外大形状检查也完成：77760个目标叶子、9720000个limb摘要与§11.4
相同，fold/root/frontier全部驻留。D1381380完整目标摘要已与§11.6旧构建一致。
不同D改变工作量，采集器只要求同一D内的乘法、归约、launch与强制检查覆盖
保持一致，不要求两个D有相同的叶子集合或运算次数。

计时口径是既有`stage2_full_wall.total`：init包含baby/F树及其强制检查；main
包含逆多项式、G树/fold/下降/累积与最终检查。进程启动、读save和单独的候选D
扫描不包含在该字段中，原始driver/curve_done日志保留这些更外围的时间。
未将有额外投影的检查时间或嵌套`t_reduce`加进正式墙钟。

### 13.2 正式结果

```text
预热（不计）：D810810 113.220060 s；D1381380 96.495869 s

ABBA：A 112.910395；B 96.377058；B 97.307139；A 113.157640
BAAB：B  96.658806；A 113.030344；A 113.272511；B  97.023610

A D810810 ：n=4 mean=113.09272250 s；sample SD=0.15670841 s
B D1381380：n=4 mean= 96.84165325 s；sample SD=0.40786726 s
均值减少16.25106925 s / 14.36968612%
ABBA减少14.32481952%；BAAB减少14.41450617%
```

两个顺序组同向，差额远大于本批组内波动。这是本目标/位宽/边界与预算下的D
选择收益，不是NTT单核加速，也未证明该D在所有sigma、B1或硬件上全局最优。
四档候选中“全驻留”并未赢得本次最快时间，不能把驻留作为成本排序的先决条件。

父阶段均值及其占完整Stage2墙钟比例：

```text
阶段                 D810810 s / %        D1381380 s / %
init                 9.22975 /  8.16      14.86809 / 15.35
giant               20.87500 / 18.46       8.76450 /  9.05
G trees             55.88075 / 49.41      32.13725 / 33.19
fold                16.99750 / 15.03      21.23825 / 21.93
descent              5.00500 /  4.43      12.42325 / 12.83
polynomial inverse   2.15150 /  1.90       3.05625 /  3.16
accumulation         0.22050 /  0.19       0.31625 /  0.33
```

其它准备、bridge、收尾时间由完整墙钟差额统计，图中为Other。核心取舍是giant
与G树合计少35.854 s，init/fold/descent/逆多项式合计增加18.202 s；不能只看
G树节省。完整总差额还包含上述Other变化。

```text
工作/容量                    D810810             D1381380
G树 / fold次数               42 / 41             15 / 14
S4 launch                    1174                474
poly_muls                    3440064             2262396
coeffs_reduced               72302186            45937081
嵌套S4 t_reduce均值(s)       8.016               4.874
NTT big峰(MiB)               2048                4096
完整NTT峰(MiB)               2162.106            4612.618
fold owner实际占用(MiB)      523.264             0 / headroom
frontier                     驻留                root_unavailable回退
2秒采样设备已用峰(MiB)      4062                6442
```

每个固定D内，所有预热/正式调用的上述工作计数及强制检查覆盖相同；跨D计数改变
是算法形状改变的结果。强制selftest/抽样检查零错误，所有样本因子输出为空。
`t_reduce`包含在各父阶段中，不能再次堆叠；不同模块容量/峰值不能直接相加，
2秒采样可能漏过瞬时峰值且包含整台设备其它占用。

高负载采样平均SM时钟分别2385.08/2390.22 MHz，平均功耗74.83/71.33 W，平均
温度75.03/74.68°C；未改变功率或频率设置。样本有动态波动，全部慢样本保留。
本次与§10.4的31.49%是不同阶段的对照，不能直接相加；历史CPU109.987 s又使用
不同曲线、B1和运行条件，不把本次96.84 s解释为已建立严格CPU/GPU加速比。

![D对照完整时间与百分比堆积图](D:/code/MPA-OpenCl/docs/figures/stage2_bq_d_20261009.png)

![D对照模块峰值与设备用量采样](D:/code/MPA-OpenCl/docs/figures/stage2_bq_d_20261009_memory.png)

正式`measurements.json` SHA256：
`24219bccacdda52d0e9f7a7a8a53663322d4df5dd3a9b65d3eae51a6cd3cae11`。
构建身份为§12.4；Stage1 save身份为§10.5。分析JSON/CSV和图保存在
`docs/benchmarks/stage2_bq_d_20261009_analysis.*`及`docs/figures/stage2_bq_d_20261009*`，
沿用排除规则，全部数值与重建脚本写入仓库。

### 13.3 复现

```powershell
python tools/bench/bench_stage2_carrier.py --comparison plan --exe build_cuda_cmake/workspace_bq_phase_trim_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --b2 2600000000000 --d 810810 --candidate-d 1381380 --fold-mb 1024 --baby-mb 640 --mode timing --trim-phase-raw --telemetry --output data/stage2_bq_20261009/plan_d810810_d1381380_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/stage2_bq_20261009/plan_d810810_d1381380_timing/measurements.json --output docs/benchmarks/stage2_bq_d_20261009_analysis.json --figure-prefix docs/figures/stage2_bq_d_20261009
```

`--require-resident`用于需要全部驻留的正确性门禁；此对照允许并记录合法回退。
数据保存命令、环境、预算、输入/二进制/源码闭包和原始日志摘要，分析器只接收
完整矩阵，预热剔除、慢样本保留。字段说明见
[工具README](D:/code/MPA-OpenCl/tools/bench/README_STAGE2_CARRIER_PLAN.md)。

阶段释放回退门禁已整理为可复用工具，并在最终二进制重新通过：

```powershell
python tools/test/test_stage2_phase_trim.py --exe build_cuda_cmake/workspace_bq_phase_trim_stage2/ecm_cuda_stage2.exe --reference-check data/stage2_bq_20261009/phase_trim_m37_check/measurements.json --output data/stage2_bq_20261009/phase_trim_fallback_gate
```

它要求已完成、源码身份匹配且含独立单位案例CPU oracle的reference check，强制
两种分配失败，对比全部目标叶子，保存原始命令、环境、日志及检查结果。

## 14. 尚待完成的联合规划

本阶段完成可单独控制的布局、阶段释放、共享数量策略和实际D候选验证，尚未
把完整MemoryPlan接入普通D选择或Auto B2。后续按以下顺序实施：

1. **用生命周期ledger替代旧arena求和。** D1021020的原生两缓冲arena估计仍为
   8194.8125 MiB，因而`arena_estimate_fits=false`；实际完整NTT峰4611.9609 MiB，
   无overflow。这证明仅同步2/3个缓冲数量仍不够：共享大池应按最大存活容量
   收费，不能将fold与两棵tree的big相加。表缓存、小输出、fuse base、S4输出、
   raw、坐标、seed及owner仍需按阶段计入；导出/每调用回退保留三缓冲合同。
2. **统一计划与实际分配合同。** planner/allocator共享shape、chunk、owner布局、
   保留capacity与释放点；分配层报告同时存活和峰值，各模块自身峰值只作诊断。
   free/reserve、各模块cap与驻留预算分别定义；不要将arena_cap当作总进程VRAM上限。
3. **比较驻留/回退策略的完整成本。** 对每个合法D及算术后端，分别预测初始化、
   giant、G树、fold、下降和累积；只在同时存活分配通过预算时排序。驻留减少
   传输，较大P可能减少G树，也可能跨NTT/补齐边界；不能将其中一项单独作为目标。
4. **将giant chunk和S4 chunk纳入规划。** 现有S4分块仍按三缓冲保守预算选择，
   本阶段保持原launch策略；后续才考虑两缓冲下扩大batch。巨点chunk的向上取整
   和device_gleaf独立上限也需建模，避免P略增却使坐标chunk次数增加。
5. **测量后更新D/Auto B2成本scope。** 当前证据只有一个高位宽目标/承载及固定
   B1/sigma；不能据此发布所有N、B1、B2的新默认，更不能沿用旧cprof。至少补
   production B1、其它位宽、generic/Mersenne、驻留/回退和不同budget的交错测量，
   再统一普通D规划与Auto B2，目标继续采用已确认的总流程收益K/(T1+T2)。

条件成本模型的基本结构可写为：

```text
I ≈ B2/D + 2； P=phi(D)/2； G=ceil(I/P)
T2(D,strategy) ≈ T_init(P) + T_inverse(P)
                + T_giant(I,chunk) + G*T_Gtree(P)
                + (G−1)*T_fold(P,strategy) + T_descent(P,strategy) + T_accum(P)
admit(D,strategy) = max_phase(actual_owned_payload + reserved_margin) <= budget
```

每项要依赖实际NTT长度和整数分块台阶，而不是用连续P或单个bits拟合。上式是
下一阶段模型结构，未声称已经标定、得到全局最优D，或提供完整进程峰值保证。
梅森承载与B/Q/阶段释放仍显式启用，发布默认不在本阶段更改。

## 15. 物理工作区驱动的 S4 分块

### 15.1 目的与预算合同

在§14第4项中先推进S4 chunk：`ca6dc50`已支持两缓冲pool，但S4仍按三缓冲的
请求大小选择chunk。新实验开关`NTT_S4_WORKSPACE_BUDGET=1`（默认0）让分块器
读取同一arena的物理缓冲数量。默认路径保持原halving序列与原预算判定。
它是完整MemoryPlan的一部分，尚不替代D选择或Auto B2。

设NTT长度为`n`，输出digit槽为`o=2m−1`，chunk slice数为`c`，物理缓冲数为`b`：

```text
old_request(c) = 8*c*(3*n + o)
new_request(c) = 8*c*(b*n + o + 2) ; b∈{2,3}
c∈{nbatch, floor(nbatch/2), floor(nbatch/4), ... ,1}
chunk = first c with new_request(c) <= batch_mb*2^20
if no c fits: chunk=1 ; record single_over_budget_calls
chunk = min(chunk, configured chunk_max) ; if chunk_max != 0
```

两个额外字是每slice carry诊断的dRes。pool关闭或无arena时`b=3`；正常两缓冲
pool为`b=2`。pool申请失败仍可进入已有三缓冲每调用回退，不将名义请求量误报为
回退实际分配量。`batch_mb`保持NTT单次请求的软预算；完整fold本来就可能是单个
slice而超过它，仍需要arena预算及设备headroom的另行约束。

共享整数函数位于
[chunk_request_bytes / chunk_slices](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:28)，
实际选择位于
[poly_mul_batch_modN](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:3533)，
不改变NTT精确性、S4模归约、carry分组、oracle和非单位回退算法。
Auto B2在
[成本scope守卫](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:618)
拒绝新分块策略，避免套用旧计时profile。

### 15.2 观测口径

[S4Ctx计数](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1274)记录：

```text
calls / changed_calls              parent S4 calls / chunk changed from legacy
chunks / legacy_chunks             actual outer NTT subcalls / legacy-policy subcalls
request_peak_bytes                 nominal physical-layout request maximum
single_over_budget_calls           parent calls whose mandatory one slice exceeds budget
owned_subset_observed_peak_bytes   maximum observed live arena + retained S4 raw/pack/output
process_peak_complete=0            no total-process peak guarantee
```

同时存活子集在
[每个成功NTT子调用后](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:3776)
读取当前arena payload与fuse base，加上S4实际保留的raw A/B、临时pack及归约输出。
它不是不同模块峰值的求和；它也不覆盖reduction常量/oracle、坐标、树metadata、
fold/frontier owner或已经释放的每调用临时工作区。名称中的observed/subset与
`process_peak_complete=0`必须保留，不能据此宣称完整MemoryPlan已完成。

`s4_multiply_stats.launches`与`real_batched_breakdown.ntt_launches`均为父批次调用数，
不是GPU kernel launch数。本次采集器对每形状`s4_reduce_stats.launches`求和，另存
真实`reduce_hook_calls`。内部grid-y切分也可能令hook数大于外层chunk数。

### 15.3 正确性门禁

构建`build_cuda_cmake/workspace_chunk_stage2/`：CUDA13.3、sm89、PTX3、add/sub=1、
outer unroll=0、nvcc split compile=8，100.9 s。可执行文件SHA256：
`6ff6b71a162fe62bb2e3273cfe3cc3e0b28ccf520b855518c46fb0c0f31bf5dd`。

已通过：

1. 580例原生整数分块检查，覆盖非二次幂batch、两/三缓冲、预算阈值、单块超预算、
   非法缓冲数与整数溢出；原有workspace生命周期128例检查继续通过。
2. M37/M67/M29/M253/M16384，各两臂，沿用独立CPU目标叶子或已知非单位因子oracle。
   M16384实际目标为8193-bit的`2^8192+1`；1 MiB预算使新策略实际改变11个父调用，
   子调用86→59，仍匹配独立CPU目标叶子。
3. 同一8193-bit目标的generic算术两臂对照通过，完整投影叶子相等，强制GMP检查无误。
4. 原生plan-only八种pool/reuse/D组合仍通过，不执行曲线。
5. [控制/失败门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_chunk_budget.py:27)：
   强制chunk_max=1，pool关闭、三缓冲、arena拒绝三种合法回退全部匹配M37完整CPU叶子；
   延迟carry内层诊断污染在`P=2, nbatch=12, accumulated=10`报错退出，不发布曲线结果。

门禁开发中的两个采集器问题已保留原始记录：第一版环境cap被CLI arena覆盖，未真正
触发拒绝；第二版请求生产不允许的`NTT_S4_CHUNK_OUTPUT=0`，被既有配置守卫拒绝。
修正驱动后`control_gate_v3`四项全部通过，没有删除前两次记录或将其计作成功。

高位宽输入沿用§10.5：N=M8011/80111、target bits=7995、B1=20、sigma=26、
B2=2.6e12、D=1381380。arena6300/fold1024/baby640/batch256 MiB，B/Q与阶段释放
均开启。新旧分块都得到126720个目标叶子、15840000个目标字，摘要
`14409977791789275297`，与§12/13同D记录一致；全部强制算术检查通过。
该完整高规模检查采用叶子摘要对照，不宣称拥有全高规模独立CPU逐叶oracle。

检查数据保存在`data/stage2_chunk_20261009/`，不提交raw/binary；重建步骤见工具README。

### 15.4 完整墙钟交错计时

固定同二进制、同D、同carrier、同输入和预算，仅切换分块策略。两臂各一次预热，
随后ABBA+BAAB，各n=4；保留所有样本、强制检查与GPU只读遥测。额外目标投影关闭，
计时采用`stage2_full_wall.total`，口径同§13。

正式矩阵完成；候选**未通过性能门槛，保持默认关闭**。完整样本如下，秒：

```text
warmup（排除） old 95.907850 ; new 96.428248
ABBA old 96.421493 ; new 96.767818 ; new 96.806528 ; old 95.928709
BAAB new 96.607304 ; old 96.004539 ; old 95.873053 ; new 96.484720

                  old                 new
n                 4                   4
mean              96.05694850         96.66659250
sample SD          0.24893264          0.14879659
range             95.873..96.421      96.485..96.807
```

完整时间增加0.609644 s（**慢0.634669%**）；ABBA/BAAB分别慢0.636414/0.632920%。
方向在两种顺序一致，但只代表当前N/D/设备/预算；不推广为所有分块或硬件的结论。
之前单次额外检查96.770709/99.511943 s不计入正式样本。全部十次调用`clean=1`，
无额外目标投影/逐点诊断，未筛选慢样本或改变设备功率/频率。

父阶段均值及其完整墙钟占比：

```text
phase              old seconds / %     new seconds / %
init               14.88168 / 15.49     14.84733 / 15.36
giant               8.77150 /  9.13      8.76625 /  9.07
G trees            32.10975 / 33.43     32.70200 / 33.83
fold               21.12975 / 22.00     21.08775 / 21.81
descent            11.94900 / 12.44     12.07825 / 12.49
polynomial inverse  2.98400 /  3.11      2.96850 /  3.07
accumulation        0.30275 /  0.32      0.31375 /  0.32
```

G trees增加0.59225 s、descent增加0.12925 s，是主要可见回归；其它阶段部分抵消。
G tree与descent的归约事件属于父阶段，不再次相加。

```text
work/capacity                        old                new
parent S4 calls                      474                474
changed parent calls                 0                  203
NTT subcalls / actual reduce hooks    5260 / 5260        2890 / 2890
poly_muls                            2262396            2262396
coeffs_reduced                       45937081           45937081
GMP selftest cases                    2400               2400
GMP sampled coefficient checks        42283              24828
full coefficient checks               4                  3
mean nested t_reduce (s)               4.873              5.35850
mean t_reduce_host (s)                12.709              2.257
event-ring waits                      268                30
mandatory single-slice over budget    176                176
NTT big peak (MiB)                    4096               4096
full NTT peak (MiB)                   4612.618           4614.284
observed NTT+S4 subset peak (MiB)      5446.804           5448.470
2-second sampled GPU usage peak       6442               6444
fold owner / frontier                 disabled           disabled
```

子调用减少45.06%，主机归约计时减少10.452 s、ring waits减少238次，完整时间仍未
下降；这些主机等待并非可直接从墙钟中扣除的独立串行成本。归约事件增加0.4855 s。
按源码中事件的位置，它测量in-stream归约区间，而非整个父阶段或完整NTT。逐shape
分析显示例如P=2049的hook555→238次，而归约区间增加0.084 s；P=2的hook578→290次，
归约增加0.0635 s。这支持继续检查每shape吞吐及调度，不支持单按launch数排序。
具体缓存/occupancy原因尚未通过profile验证，不能仅凭这里的时间归因。

GMP抽样规则仍为每shape首调用及每8次hook触发；减少hook自然减少抽样总量。两臂
内部工作/检查计数稳定，跨臂数学工作与selftest覆盖相同；完整高规模目标叶子另已
对照通过。表中如实列出检查覆盖差异，没有关闭强制检查来制造加速。

高负载采样平均SM时钟2387.65/2384.01 MHz，功耗70.16/71.91 W，温度73.21/73.24°C。
模块峰值、观测子集峰与设备用量互相包含，不相加；2秒采样可能遗漏瞬时峰值。
本轮既未减少主要大池，也未启用D138的owner/frontier，不能声称完成驻留优化。

![物理工作区分块的时间与阶段占比](D:/code/MPA-OpenCl/docs/figures/stage2_chunk_20261009.png)

![分块策略的容量及同时存活子集观测](D:/code/MPA-OpenCl/docs/figures/stage2_chunk_20261009_memory.png)

勘误（第18节）：旧版`owned_subset_observed_peak_bytes`重复计入fuse base，图中
该子集系列偏大，不能用于精确容量或显存收益判断。其它NTT模块峰、GPU采样和
完整计时不受影响；新观测版本及真实组件核对见第18节。

原始正式矩阵SHA256：
`0bded32f947a85e10da04038b0d8028a480ef585a4b8ec22c509c509d59dd40b`。
build manifest SHA256：
`276ebc35f3add9bcb181c04a53192e154e94e2144912c46d87783487f2e29c08`；
frozen source manifest SHA256：
`6e1cf8e8d18f5c12526b6161af9bd97e895498a304fba38bee5f598b99f54a3b`。
采集器保存所有命令、环境、预算、输入及二进制/源码闭包身份；分析器验证完整矩阵
与原始日志摘要。阶段/每shape分析JSON/CSV及图像按既有排除规则保存在
`docs/benchmarks/stage2_chunk_20261009_analysis.*`和`docs/figures/stage2_chunk_20261009*`。

### 15.5 复现与下一步

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/workspace_chunk_stage2 -Arch sm_89 -SplitCompile 8
python tools/bench/bench_stage2_carrier.py --comparison chunk --exe build_cuda_cmake/workspace_chunk_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode timing --telemetry --output data/stage2_chunk_20261009/m8011_d138_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/stage2_chunk_20261009/m8011_d138_timing/measurements.json --output docs/benchmarks/stage2_chunk_20261009_analysis.json --figure-prefix docs/figures/stage2_chunk_20261009
python tools/test/test_stage2_chunk_budget.py --exe build_cuda_cmake/workspace_chunk_stage2/ecm_cuda_stage2.exe --reference-check data/stage2_chunk_20261009/m37_check/measurements.json --output data/chunk_control_replay
```

完整联合规划仍需§14中的phase ledger与实际shape/chunk序列；尤其树top应依赖补齐树
的实际最大子节点degree，不能只将`P/2+1`当成所有tree的最大operand。还需比较
驻留/回退、giant坐标chunk、保留pool容量与cache eviction，并使用实际NTT台阶
及重新标定的完整阶段成本选择D/Auto B2。新分块开关暂不进入发布默认。

下一项具体实验是**阶段S4归约输出容量释放**。D1381380在inverse→fold边界已释放
raw，但日志仍为free1914699776 bytes，扣除owner894143424后只余973.278 MiB，
距固定1 GiB future reserve差50.722 MiB。此曲线S4输出模块记录的全程峰为243.634 MiB。
这是待验证的候选，不代表边界上恰好保留同样容量，更不能把全程峰直接扣进边界
ledger。应在释放点读取实际`d_out_cap`，确认host finv和异步oracle已获得数据、
没有借用指针，再同步释放、重置容量并在后续hook按需申请；比较实际owner/frontier
启用状态、未来重申请峰值与完整墙钟。保持1 GiB余量，不通过降低安全余量制造驻留。

## 16. 阶段归约输出容量释放

### 16.1 生命周期与实现合同

推进§15.5的候选：新增默认关闭的`NTT_PHASE_TRIM_OUTPUT=1`。它与raw释放独立，
在以下两个边界回收S4 `d_out`：

1. `inverse_to_fold`：Newton已将finv返回到主机CPoly，并转换到独立`finvflat`；
   F树根也在主机，随后fold owner从这些主机向量初始化。
2. `fold_to_descent`：仅在已存在live fold owner、Gamma设备校正成功（或校正系数为1）且根degree合适
   时执行；H/F/inverse由独立owner持有，S4临时输出不承载下一阶段状态。

[S4Ctx::output_release](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1315)
执行同步`cudaFree`，随后将指针和容量清零；后续hook按实际新请求重新申请。
[Newton之后的释放点](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7335)与
[下降之前的释放点](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7818)
继续遵守原有raw/owner lifetime边界。

异步GMP oracle在
[输出主机快照](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:3201)
排队D2H后持有自己的pinned host副本；延迟CPU比较不保存S4输出设备指针。
同步free等待这些default-stream读者完成，不关闭oracle或抹去carry诊断。
NTT arena、digits、reduction常量、fold/frontier owner均不由此函数释放。
释放时日志直接记录`8*d_out_cap`，不用全程输出峰值替代边界实际容量。

保持fold/frontier的1 GiB future reserve不变；不能保证新策略使所有D驻留，
也不能把arena cap当成总进程显存预算。Auto B2
[scope守卫](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:613)
拒绝未经新成本标定的输出释放策略。

### 16.2 构建与正确性

构建`build_cuda_cmake/phase_output_stage2/`，105.4 s；CUDA13.3/sm89/PTX3、
add/sub=1、outer=0、split compile=8，与§15工具链一致。身份：

```text
binary SHA256   d6dc7cbed55271d65fff841f2bbd57dc5de122b985f0fa7ce35c528bb8e77f2e
build SHA256    7b50cb5473ecd1a0bc64c19ad38d230098c73cbb0099fbb86dd56de512aa2882
source SHA256   4141de41ed187c5fe10d9fbb783fce1fca817f3c5df349e1a3f85e16716e3fa9
```

已通过M37/M67/M29/M253/M16384，各两臂的独立CPU叶子或非单位因子参考；generic
8193-bit目标两臂的完整投影叶子对照。M37加`--require-resident`与workspace fixture，
覆盖128例工作区合同及580例整数分块检查；原生八种plan-only组合仍通过。

沿用同构建M37、独立CPU完整目标叶子参考，强制fold/frontier分配失败两项通过，
验证释放后合法回退与重申请；pool关闭、三缓冲、arena拒绝三项也通过。
内层carry污染仍报错退出，不发布结果。门禁工具兼容旧workspace/chunk参考和新的
`phase-output`参考，分别消费其候选臂。

高规模对照使用§15同一Stage1 save：7995-bit N、M8011、B1=20、sigma26，
B2=2.6e12、D1381380；arena6300/fold1024/baby640/batch256 MiB。
两臂都使用B/Q复用与raw阶段释放，**不启用§15的物理分块策略**，仅切换输出释放。
完整目标叶子和驻留状态检查通过：两臂126720个目标叶子、15840000个目标字、
摘要`14409977791789275297`，与§12/13/15同D记录一致。仍不把它表述为全高规模
独立CPU逐叶oracle；独立参考覆盖的是上述小规模案例。

```text
boundary                  raw released bytes     output released bytes
inverse_to_fold            255469536              127734768 = 121.817 MiB
fold_to_descent            447068160              255468528 = 243.634 MiB
```

121.817 MiB是Newton结束时的实际容量，确实小于输出模块全程峰243.634 MiB。
本次native headroom记录（bytes）：

```text
                           baseline              reclaimed output
fold available             1914699776            2042626048
fold owner request          894143424             894143424
future reserve             1073741824            1073741824
fold enabled               0 / headroom          1 / none
root/frontier enabled      0 / root_unavailable  1 / none
frontier available         n/a                   1128267776
frontier metadata          n/a                      3041280
frontier margin above 1GiB n/a                     51484672 = 49.100 MiB
```

保留了既有1 GiB headroom合同；frontier在该合同之上的余量仅49.1 MiB，外部显存占用
改变时仍可能合法回退，不能把本设备此时的驻留结果推广为所有显存状态。
单次额外检查完整时间97.151372→85.433390 s，仅作为检查记录，不计入正式性能样本。

### 16.3 完整时间与显存

先完成额外目标投影检查，再执行正式ABBA+BAAB，各臂一次预热、四个正式样本。
保留生产强制检查、所有慢样本与只读遥测。计时为`stage2_full_wall.total`；
额外检查、进程启动与D scan不进入正式阶段时间。释放/free及重申请开销包含在内。

正式矩阵完成，各臂n=4；全部十次（含预热）`clean=1`，所有样本保留。

```text
warmup（排除） old 96.625851 ; new 84.798333
ABBA old 95.674066 ; new 84.979645 ; new 85.170326 ; old 96.067527
BAAB new 85.090533 ; old 96.351004 ; old 95.696362 ; new 84.881466

                   retained output       reclaimed output
mean seconds       95.94723975           85.03049250
sample SD           0.32406694            0.12642868
range seconds      95.674..96.351        84.881..85.170
```

完整时间减少10.91674725 s（**11.377865%**）；ABBA/BAAB分别减少11.260792/11.494751%。
两个顺序一致；本结论只对应当前N、carrier、D、设备和预算，不直接推广到其它位宽、
production B1或所有显存状态，也不与§10/13的百分比相加。

父阶段均值及占完整墙钟百分比：

```text
phase              retained s / %      reclaimed s / %
init               14.78397 / 15.41     14.79842 / 17.40
giant               8.76775 /  9.14      8.75850 / 10.30
G trees            32.09950 / 33.46     31.79450 / 37.39
fold               21.11000 / 22.00     15.96900 / 18.78
descent            12.05525 / 12.56      8.71225 / 10.25
polynomial inverse  2.96550 /  3.09      3.00850 /  3.54
accumulation        0.30350 /  0.32      0.31025 /  0.36
other               3.86177 /  4.02      1.67908 /  1.97
```

fold节省5.141 s、descent节省3.343 s，准备/bridge/收尾等Other节省2.18269 s；
G trees节省0.305 s，其它阶段有小幅波动。释放和重申请成本包含在完整时间内；
两个释放边界（raw+output）合计平均0.009532 s，基线raw单边界0.002453 s。

这次两臂实际数学工作与强制检查覆盖也完全相同：parent S4 calls=474、
poly_muls=2262396、coeffs_reduced=45937081、GMP selftest=2400、GMP sampled coefficients=42283、
full_checks=4，零错误。没有通过减少抽样制造加速。各臂驻留状态在全部样本中稳定：
基线fold/root/frontier均0，候选均1。

嵌套归约事件4.87325→4.87300 s，基本不变；主机归约计时12.695→17.63525 s，
ring waits268→366，完整时间却明显下降。这些等待包含流水线回压，不能简单当成
串行成本从墙钟中扣除或再次加到父阶段。本轮收益来自生命周期释放后启用驻留路径，
未修改归约数学或NTT内核。

候选fold模块报告逻辑省去H2D=10692148320 bytes、D2H=7115548608 bytes；
frontier报告逻辑省去parent H2D=4334690304 bytes、state D2H=2167345152 bytes，
同时仍上传F siblings=2422810656 bytes。这些字段各有其参考路径/统计合同，
不冒充完整进程的PCIe总量，更不直接将其和缓存峰值相加。

```text
capacity / sampled usage (MiB)      retained           reclaimed
NTT big peak                       4096               4096
full NTT peak                      4612.618           4612.618
resident fold owner                   0                852.722
2-second sampled GPU usage peak     6442               7296
```

**释放暂存让常驻owner成立，总设备用量反而上升约854 MiB。** 本轮是显存生命周期
与传输成本的取舍，不能表述为总显存下降。模块峰值与设备用量互相包含，不相加；
2秒采样可能遗漏瞬时峰值。frontier admission中49.1 MiB是满足未来1 GiB余量合同后
的额外余量，不是所有后续阶段始终保留1 GiB空闲的保证。

高负载只读采样平均SM时钟2388.58/2389.11 MHz，功耗72.45/77.96 W，温度74.70/75.55°C。
没有更改功率/频率/设备设置，GPU0的其它任务未干预。较高GPU利用及较少CPU准备
可能改变功耗，这里不以功耗变化单独解释因果。历史CPU结果仍不是本次同曲线/同B1
严格对照，不据此建立新CPU/GPU加速比。

![输出释放与驻留的阶段耗时占比](D:/code/MPA-OpenCl/docs/figures/stage2_output_trim_20261009.png)

![常驻owner与设备用量采样](D:/code/MPA-OpenCl/docs/figures/stage2_output_trim_20261009_memory.png)

正式矩阵SHA256：
`0b55f67a3236fa7a7c5ee37bcf8f611d25c4854ba620cb018da2923a098e043b`。
构建身份见§16.2，输入身份见§10.5；所有原始命令、环境、预算、二进制/源码闭包、
可读/调试日志及结果均保留。分析文件为
`docs/benchmarks/stage2_output_trim_20261009_analysis.json/.csv`，图为同名figure prefix。
两张PNG已人工检查布局；分析器仅接收完整正式矩阵并验证原始日志摘要。

### 16.4 复现

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/phase_output_stage2 -Arch sm_89 -SplitCompile 8
python tools/bench/bench_stage2_carrier.py --comparison phase-output --exe build_cuda_cmake/phase_output_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode check --projection-only --output data/stage2_output_trim_20261009/m8011_d138_check
python tools/bench/bench_stage2_carrier.py --comparison phase-output --exe build_cuda_cmake/phase_output_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode timing --telemetry --output data/stage2_output_trim_20261009/m8011_d138_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/stage2_output_trim_20261009/m8011_d138_timing/measurements.json --output docs/benchmarks/stage2_output_trim_20261009_analysis.json --figure-prefix docs/figures/stage2_output_trim_20261009
```

原始证据位于`data/stage2_output_trim_20261009/`，仍不提交data、图像或二进制。
所有实验开关保持默认关闭；完整D/P/生命周期MemoryPlan与Auto B2新成本仍按§14推进。

## 17. 精确树形状、共享容量模型与阶段观测

### 17.1 本轮改动与合同

接续`9c60b75`，先统一联合规划的shape/chunk合同。发现旧`geometry()`和内存日志
把树顶operand写成`P/2+1`；真实树先补齐到二次幂，最大非空子树可能更大。
例如D1531530、P138240，真实最大子树degree131072，因此查询131073个系数。
旧69121系数查询给出NTT `2^27`，修正后为`2^28`。fold查询也统一为打印的P+1。

源码：[树顶及group枚举](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:82)、
[树容量模型](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:112)、
[几何准入](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:183)、
[计划调用](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8655)、
[阶段记录](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:3555)、
[阶段输出](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:9524)、
[计划JSON](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:9610)。
普通D选择与Auto B2共用geometry，因而都获得正确的树尺寸；未改算术内核、缓存
分配器、驻留阈值、代价系数或实验开关默认。原生JSON新增`geometry_version=3`；
`accounting_version=2`仍表示实际arena分配记账版本，两者不是同一字段。

整数模型（P>1）：

```text
h_top = 2^floor(log2(P−1)); tree operand = h_top+1
每层 h=1,2,4,... < P：q=floor(P/(2h)); r=P mod (2h)
完整group：(ma,mb,nb)=(h+1,h+1,q)，q>0时产生
部分group：(ma,mb,nb)=(r−h+1,h+1,1)，r>h时产生
0<r<=h：直接复制单个子树，不产生NTT
```

部分group按(min,max)排序先于完整group，与设备树实际map顺序一致。完整树乘法
pair总数仍为P−1；group数与pair数不同，短尾group/短尾chunk单独计入。P=1不
产生乘法。该枚举只需O(log P)，不分配P个节点；门禁用独立dense树检查它。

模型使用真实backend的`ntt_shape_query()`与已有`chunk_slices()`：原分块按固定
三缓冲预算，物理分块按实际b=2/3并计入carry。单slice超预算仍执行单slice。
它不会把第15节性能失败的扩大分块策略默认启用。

```text
NTT单请求 big = 8*b*N(max(ma,mb))*slices
开启pool的大池峰 = max_request(big)，不是fold+2*tree的和
digits/verdict保留 = Σ_(N,slices) 8*slices*(max(out_slots)+2)
树S4输出峰 = max_group(8*ceil(bits/64)*chunk*(ma+mb−1))
关闭pool的keyed big保留 = Σ_(N,slices) 24*N*slices
```

`tree_workspace`是**单棵树、无缓存淘汰**的组件模型；它不包含表/fuse base、之前
阶段遗留缓存、raw/坐标/seed/owner或失败时每调用分配。pool关闭时
`shared_big_peak_bytes`只表示最大单请求，使用`keyed_big_retained_bytes`查看按key
保留量。`supported=false`明确排除host pack、最终readback等非支持控制路径。
不得把这些组件峰值相加成进程峰，也不得据此承诺驻留。

实际S4调用新增按ftree/gtrees/fold/descent/inverse记录的`s4_phase_memory`：groups、
pairs、chunks、请求大池/输出峰、实际存活大池峰、同时存活NTT+S4子集峰。子集
采样在完成的NTT边界读取现有分配capacity，含全部arena/base和S4 raw/pack/output，
不含临时每调用、reduce/oracle、点或owner，仍有`process_peak_complete=0`。
每阶段仅一行debug输出，不增加可读日志刷屏。

**旧保守准入尚未替换。** `arena_estimate_bytes`明确标为
`arena_estimate_kind=legacy_additive`，仍保留fold+2*tree的求和，只修正树尺寸。
新tree组件模型不是完整MemoryPlan，不能借它放宽总显存准入。第14节的缓存
淘汰/失败回退、各阶段完整并存分配及驻留/回退成本仍待接入；旧Auto B2 profile
也不支持新B/Q与释放开关，继续拒绝未标定策略。

### 17.2 计划与真实大形状核对

GPU1、N=M8011/80111、carrier8011、B1=20/sigma26/B2=2.6e12；全部预算保持
arena6300、owner1024、baby640、batch256 MiB，两缓冲pool、原分块。原生计划
的单棵F树结果如下，MiB=2^20 B；group/NTT子调用是精确整数，不是kernel数。

```text
D         P       tree operand  tree NTT  groups  NTT chunks  pool big MiB  digits MiB
810810     77760    65537        2^27      23        137       2048          2.968
1021020    92160    65537        2^27      19        222       2048          2.564
1381380   126720    65537        2^27      23        267       2048          2.774
1531530   138240   131073        2^28      20        248       4096          4.966
524288    131072    65537        2^27      17        239       2048          2.719
```

D1531530的真实F树，两臂均精确吻合：20groups、138239pairs、248chunks、
4294967296 B请求/实际大池峰、139346928 B S4输出请求峰；同时存活NTT+S4子集
观测峰5411984800 B。全程NTT模块峰4837454864 B，且无eviction或legacy malloc。
勘误（第18节）：前述旧子集量含一次多余的fuse base；同形状新构建实测为
5239813808 B。全程NTT模块峰没有此错误，不更改旧算术及计时结果。
前者已包含NTT容量，不再加后者；它们均不是完整进程峰。

释放输出的候选仍不驻留：Newton边界实际释放raw278693856 B、输出139346928 B，
但owner准入读到available2040528896 B；owner975428544 B加既有1073741824 B
future reserve仍差8641472 B / **8.24115 MiB**。fold/root/frontier分别以
headroom/root_unavailable回退。没有降低余量换取通过。

两个带完整目标投影的检查调用full分别104.847310/104.086892 s，`clean=0`，
不是正式性能样本；不据n=1/arm的0.76 s差额声称加速。当前D153探索不支持用
更大P代替第16节正式验证的D138方案。两臂138240个目标叶子、17280000个limb，
nonzero138240、hash1525678423161840651完全相同；强制检查零错误、无因子。
这是同构建/同目标的完整投影对照，不是高位宽独立CPU全叶oracle。

### 17.3 验证、身份及复现

- 新dense padded-tree整数门禁：697 cases/0 bad，含P=65535/65536/65537、
  131071/131072/131073、两/三缓冲与两种chunk策略、短尾、retention及溢出。
  原128项workspace及580项chunk检查继续通过。
- 原生计划：5个D×4种pool/reuse×2种chunk策略=40组，另18个native二次幂
  anchor查询；Python独立dense树验证group、变换长度、chunk、大池/digits/输出。
  D153两条真实F树记录与原生计划核对通过。M37另8组计划和两条真实记录通过。
- M37/M67独立CPU单位参考，M29/M253非单位及已知因子，满16384-bit承载，
  generic8193-bit两臂均通过。M37保留独立CPU全部叶子oracle。
- fold/frontier强制分配失败与重申请、pool关闭、三缓冲、arena拒绝、延迟carry
  污染门禁通过；污染在发布结果前失败。CPU成本solver（合成packing）与2项
  exact-tree/G=1回归通过；没有用它们替代GPU算术门禁。

首轮计划工具忘记显式传`--batch-mb`，实际查询了默认64 MiB；其自校验通过但
不用于256 MiB运行核对。第二轮40组256 MiB通过，收尾工具从环境读取batch时
发现实际CLI优先、环境没有该键；修正为读取保存的CLI参数后完整重新运行通过。
未修改或覆盖失败矩阵，未重跑高曲线以丢弃慢样本。

```text
最终binary SHA256:
98b9698bf81cfebcabf9e20e04a3031434cdf4d6cfd083465536bf4c96c1349c
build_manifest SHA256:
915d85912d9601aa8958ab1a385a8d08f2c3ba5e1a608ee46d98ffa625a9f86f
frozen_sources_manifest SHA256:
a039473cadd519787fc871b6b895b1725eea6a47cfe4f8e4dd6c4bc467829f23
大形状check矩阵 SHA256:
6d6a3fa1e422070d5e5e734bcb143d1aa01708ba1229f5d0792fb8666dfb822f
```

最终构建105.0 s，CUDA13.3/sm89/PTX3/addsub1/outer0/split8；输入身份沿用§10.5。
开发中一次编译遇到Windows的`small`宏，改名后修复；另一次因编译期间源码改变
被构建身份门禁拒绝，未作为最终二进制。最终完整重编译成功，冻结源码已核对。
GPU0其它任务未触碰，未改变功率/时钟或设备设置。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/tree_memory_plan_stage2 -Arch sm_89 -SplitCompile 8
python tools/bench/bench_stage2_carrier.py --comparison phase-output --exe build_cuda_cmake/tree_memory_plan_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --b2 2600000000000 --d 1531530 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode check --projection-only --telemetry --output data/stage2_tree_memory_20261009/m8011_d153_check
python tools/test/test_stage2_workspace_plan.py --exe build_cuda_cmake/tree_memory_plan_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --runtime-check data/stage2_tree_memory_20261009/m8011_d153_check/measurements.json --output data/stage2_tree_memory_20261009/native_plan_runtime256_current
```

原始证据及全部回退/小曲线门禁位于`data/stage2_tree_memory_20261009/`，不提交。
同方向后续继续第14节：把树请求与Newton/fold/下降完整序列汇入同一生命周期
ledger，加入表/base/淘汰及阶段外缓冲，再共用完整MemoryPlan替换旧准入；随后
重新测量各策略完整成本并接入D/Auto B2。发布默认保持第16节前的实验开关状态。

## 18. 单树完整NTT保留合同与驻留前冷缓存回收

### 18.1 观测错误及完整组件合同

接续`f3c03df`。扩展生命周期记账时发现旧NTT+S4子集观测把arena的`bytes`与
fuse base相加；但`ntt_arena_fuse()`申请base和table时，已将两者都计入`bytes`。
这只影响`owned_subset_*`与第17节新观测值，不影响`ntt_workspace_stats.full_*`、
缓存准入、驻留free查询、完整计时或数学结果。不能拿旧偏大子集与新字段作
显存节省对照。此次删除重复加项，新增`subset_accounting_version=2`。

分配层新增独立`payload()`重算：共享及keyed大池、digits/verdict/carry scratch、
cached pass table与实际base指针，四项按当前同时存活capacity求和。输出
`ntt_arena_accounting.calculated_bytes/mismatch`，门禁要求与实际`bytes`一致。
源代码：[payload快照](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:1970)。
全进程未纳入此快照，per-call临时分配不属于arena，仍明确排除。

第17节单树请求模型现在按NTT长度去重，直接调用分配器`fuse_describe()`及
`fuse_planned_*_words()`，继承T/M、compact scratch和GPU shape策略，而非复制
另一份容易偏移的pass公式。`tree_workspace.cache_shapes`列出每个N的table/base，
无eviction/导出且无额外carry scratch时：

```text
NTT_tree_retained = pooled max big（或keyed big合计）
                  + shape-local digits/verdict retained
                  + Σ_unique_N(pass table bytes + fuse base bytes)
```

该合同包含单树完整NTT模块保留容量，尚未包含之前阶段遗留缓存、raw/点/owner
或每调用回退。配置不在scope时`supported=false`；它不是完整进程MemoryPlan。
源码：[cache计划](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:152)、
[调用同一描述器](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8660)。

D1531530、carrier8011、batch256、两缓冲pool/原分块的F树精确计划与两条真实
运行一致，NTT无eviction或每调用分配：

```text
big                     4294967296 B = 4096 MiB
digits + verdict           5206864 B
cached pass tables       363878560 B
fuse base                172170992 B
完整F树NTT保留          4836223712 B = 4612.18234 MiB
F树NTT+S4同时存活子集   5239813808 B = 4997.07585 MiB（版本2）
```

完整NTT数与子集重叠，不相加。旧子集5411984800 B恰多了172170992 B的base。
不同阶段digits会增长，因此不能把F树保留量当作全程NTT峰4837454864 B。

### 18.2 回收策略与准入实测

新增`NTT_OWNER_TRIM_FUSE=1`，默认0。在fold/frontier实际free不足时，回收
目标NTT长度之外容量最大的完整fuse上下文，包含table/base；同步释放、实际
free重查，达到`申请量+workspace growth+原1 GiB余量`即停止。没有足够缓存
则保留原headroom回退；不释放大池、digits/verdict或owner，不降低余量。
完整上下文删除后原分配器在后续首次遇到shape时重建并重新缓存，避免每次调用
都临时构造表。只在旧阶段读者已完成的两个边界调用。

源码：[冷上下文回收](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:1983)、
[free重查循环](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2369)、
[fold准入](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5326)、
[frontier准入](D:/code/MPA-OpenCl/src/cuda/stage2/scaled_frontier.cuh:36)、
[Auto B2 scope](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:614)。
`stage2_cache_trim`与累计`ntt_phase_cache_stats`只在debug中记录。

沿用第17节目标/保存点/预算与D1531530，两臂均BQ1/raw trim1/output trim1，
只改变冷缓存0/1。额外完整投影的检查运行：

```text
inverse_to_fold:
  required = 2049170368 B；before = 2040528896 B（差8.24115 MiB）
  释放1个上下文25305088 B / 24.1328125 MiB
  after = 2065694720 B；fits=1；释放计时0.000645 s
fold_to_descent:
  required = 1077059584 B；before = 1048576000 B
  释放2个上下文50609152 B / 48.2646484 MiB
  after = 1098907648 B；fits=1；释放计时0.001056 s
```

实际free变化与逻辑释放量不同，包含驱动分配粒度，不能用逻辑量替代free查询。
候选fold/root/frontier全部驻留；控制全部回退。两条检查full104.123632/
92.838199 s、`clean=0`，只作探索，不计入正式样本。
完整138240目标叶子/17280000 limbs、nonzero138240、hash1525678423161840651相同，
并与第17节旧构建相同。强制检查零错误、无因子；高位宽未另做独立全叶CPU oracle。

### 18.3 门禁、身份与作用域

- workspace fixture144项/0 bad（原128+16）：额外覆盖完整payload重算、cold
  eviction、目标shape保留、dRes污染字隔离、形状重建的独立卷积与释放后零占用。
  原697项dense树及580项chunk整数门禁继续通过。
- 40组高位宽原生plan+18个anchor，另8组M37 plan，真实大小F树的完整NTT保留
  与预测一致；纯组件计划中的无eviction假设不替代整个进程准入。
- M37/M67独立CPU参考、M29/M253非单位/已知因子、满16384-bit承载和generic8193
  两臂通过；M37作为2项owner/frontier回退及4项pool/三缓冲/arena/carry门禁参考。
- 新策略只显式启用，旧Auto B2成本scope拒绝该开关。完整生命周期MemoryPlan、
  普通D准入替换、各位宽/production B1校准及发布默认仍待第14节后续。

```text
最终binary SHA256:
3c42e03da67d29e6625bc8942fe4b5b34d796298cb20c485a50e2baff8e28090
build_manifest SHA256:
b028e08de3c9fe973f2214b0172dbf8f7e17bb054e159b7ec2dcca85dfc5443a
frozen_sources_manifest SHA256:
27995115a8f9ab686f3d3f1c952d8103a2f1136f818a382808bfac4cdbcac9bc
大形状检查矩阵 SHA256:
92166137a942522a32f14ffd54c329141c0348b57d6c64eb311c4c190b61df16
```

最终构建103.6 s，CUDA13.3/sm89/PTX3/addsub1/outer0/split8；输入身份沿用§10.5。
原始证据位于`data/stage2_owner_cache_20261009/`，沿用排除规则。

### 18.4 正式交错计时：完整Stage2减少10.96%

4060 Laptop GPU1，目标7995 bits、M8011承载、B1=20、sigma26、B2=2.6e12，
D1531530/P138240/G树13。相同binary、arena6300/fold1024/baby640/batch256 MiB，
BQ/raw/output释放均启用，physical chunk关闭；仅`NTT_OWNER_TRIM_FUSE`为0/1。
先各1次预热，随后ABBA+BAAB，每臂4次；全10条`clean=1`，预热排除。

```text
控制完整秒数：103.910813, 104.040927, 104.478412, 104.453403
候选完整秒数： 92.804627,  92.792784,  92.770072,  92.825417
均值±样本SD：104.220889±0.288048 -> 92.798225±0.023113 s
减少：11.422664 s / 10.96005%；两交错组：10.74977% / 11.16935%
```

父阶段均值及占完整时间比例（控制→候选）：

```text
初始化    16.894 -> 16.831 s   16.2% -> 18.1%
giant     12.297 -> 12.295 s   11.8% -> 13.2%
G trees   33.312 -> 33.031 s   32.0% -> 35.6%
fold      18.235 -> 13.513 s   17.5% -> 14.6%
descent   14.086 -> 10.122 s   13.5% -> 10.9%
inverse    4.534 ->  4.536 s    4.3% ->  4.9%
其余       4.864 ->  1.949 s    4.7% ->  2.1%
```

其余为完整墙钟扣除上述互不重叠父阶段的残差，包含GCD、阶段交接与收尾等，
不将其全部归因于某个传输操作。主要可定位收益为fold −4.72175 s、descent
−3.96425 s。设备归约事件4.93000→4.92975 s基本不变；嵌套主机归约等待
10.407→15.9815 s、ring waits180→272，不能加到父阶段或独立解释墙钟收益。

四条候选均回收3个上下文/75914240 B，并全程fold/root/frontier驻留；四条控制
均未回收、三处回退。完整算术工作和强制检查覆盖两臂相同：379 launches、
2112427 poly muls、44238490归约系数、2496 selftest、33086 GMP检查系数、
3次full check，全部0错误。额外全叶投影另见§18.2，不混入正式样本。

两臂NTT大池峰均4096 MiB、完整NTT峰均4613.35646 MiB；候选owner930.24115 MiB。
2秒采样GPU用量峰6278→7210 MiB。回收是为更有价值的常驻数据腾出阶段余量，
不是降低全程显存峰；模块峰和采样用量重叠，不能相加。后续shape重建已包含
完整计时。采样不能证明真实瞬时进程峰或捕获全部短暂分配。

仅对GPU1读遥测，未改变功率/时钟或GPU0其它任务。控制204样本/150 busy，候选
184/153 busy（utilization≥90%）；busy平均时钟2391.7/2387.9 MHz、功率67.83/
71.42 W、温度72.48/73.41°C。外部占用仍可能触发合法回退。

![完整时间和阶段百分比](figures/stage2_owner_cache_20261009.png)

![重叠显存统计，不能求和](figures/stage2_owner_cache_20261009_memory.png)

正式矩阵SHA256：`553a53a6616b377036889a4952e318bdccf1edafb591febd46e8dace890ba9e0`。
核对8个算术/计时矩阵24条运行、40个冻结源码、全部原始log/debug/result哈希、
4组门禁身份及本报告122个行号边界；5个Python工具AST通过。图表和原始证据不提交。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/owner_cache_stage2 -Arch sm_89 -SplitCompile 8
python tools/bench/bench_stage2_carrier.py --comparison owner-cache --exe build_cuda_cmake/owner_cache_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --b2 2600000000000 --d 1531530 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode timing --telemetry --output data/stage2_owner_cache_20261009/m8011_d153_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/stage2_owner_cache_20261009/m8011_d153_timing/measurements.json --output docs/benchmarks/stage2_owner_cache_20261009_analysis.json --figure-prefix docs/figures/stage2_owner_cache_20261009
```

本结果只证明固定D153下缓存生命周期策略有效，未证明该D全局最优；不能与旧
D138的85.03 s跨binary直接形成新D选择结论。B1=20用于快速生成有效保存点，
尚未验证production B1=10e6～260e6的同等收益。实验默认0保持，旧Auto B2 scope
继续拒绝新策略。下一步将Newton、整/残G树、fold和下降的shape序列、缓存重建、
raw/点/seed/归约/owner同时存活量汇入完整MemoryPlan，再验证候选D成本，替换
legacy_additive准入并重新校准Auto B2；单树完整NTT合同不能提前代替全流程预算。

## 19. 完整设备分配生命周期台账与giant预算拐点

### 19.1 合同、实现与阶段边界

接续`e5155c7`。此前模块peak和NTT+S4子集遗漏了同时存活的giant坐标、seed、
归约常量/诊断、临时metadata及分配器短暂请求；2秒采样也不能还原其生命周期。
新增默认关闭的`NTT_MEMORY_LEDGER=1`，在production CUDA翻译单元及包含的NTT/
point/tune源文件中，统一包装`cudaMalloc/cudaFree`。原CUDA调用和错误码保留，
只登记成功申请、只在成功释放后删除；每次申请保存文件/行号，按唯一指针记账。

```text
owned_payload(t) = Σ bytes(pointer) ; pointer已成功申请且未成功释放
peak_owned       = max_t owned_payload(t)
interval_peak    = 两次checkpoint之间的max_t owned_payload(t)
```

实际增长与释放事件更新峰，包含短暂分配和容量重建期间的重叠；峰值同时保存
当时的allocation-site组成，其合计严格等于峰。这与各模块独立峰值求和不同。
台账只记录本源码闭包拥有的CUDA申请payload，**不是驱动占用/完整物理VRAM**：
context、module加载、驱动分配粒度、其它进程与pinned host不在合同内。实际free
查询和既有余量继续作为驻留准入条件，不能由此削减reserve。

源码：[统一拦截入口](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:60)、
[台账与分配登记](D:/code/MPA-OpenCl/src/cuda/stage2/device_memory_ledger.cuh:21)、
[快照](D:/code/MPA-OpenCl/src/cuda/stage2/device_memory_ledger.cuh:79)、
[会话生命周期](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8361)。
checkpoint覆盖baby前、F树前/后、inverse后、fold准入后、giant循环后、frontier准入
后、下降后及全部曲线局部owner销毁后的final；frontier未尝试时没有该checkpoint。
`stage2_memory_ledger`摘要与`stage2_memory_site`组成只走debug通道，常规控制台不变。
台账默认关闭，不把插桩运行当作新的正式性能样本；旧Auto B2 scope拒绝该开关。

### 19.2 持久缓存与第一次失败

首版检查把final所有未释放数据判为泄漏，M37结束剩616 B/8次申请，因此采集器
拒绝矩阵。追溯全部8个site，均来自`ladder_points()`现有静态CUDA缓存：5个常量，
以及js/X/Z坐标，其设计就是跨调用复用，并非曲线局部泄漏。没有修改其释放行为，
改为显式`PersistentScope`标记，并将持久缓存纳入每个时刻的真实payload。
源码：[静态ladder缓存](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4334)。

final允许存在显式标记的persistent数据，但必须满足：

```text
final.live_bytes       = final.persistent_bytes
final.live_allocations = final.persistent_allocations
allocations + baseline_allocations - frees = final.live_allocations
unknown_frees = 0
Σ live sites           = live_bytes
Σ interval_peak sites  = interval_peak_bytes
Σ global_peak sites    = peak_bytes
```

M37等额外ladder检查会保留缓存；两条真实高位宽检查均最终0 B/0个申请。会话也能
记已有persistent基线，但当前生产驱动每条曲线由独立child执行，本批native检查
baseline均0；未以本批检查宣称同进程跨device复用已验证。第一版原始失败与旧
binary保留在`data/stage2_allocation_ledger_20261009/m37_check/`，未计为成功。

### 19.3 实际峰组成与D/P、point chunk耦合

沿用§18的目标/输入/预算。新binary的D138/D153各两条完整目标叶子检查；均开启
BQ/raw/output释放，physical chunk关闭；两臂只改冷cache回收。不是正式计时矩阵。

```text
D1381380/P126720：两臂fold/root/frontier驻留，cache eviction=0
  peak_owned = 6977513600 B = 6654.27551 MiB（两臂相同）
D1531530/P138240：控制回退，候选驻留
  控制 peak_owned = 5902320192 B = 5628.89117 MiB
  候选 peak_owned = 6877748736 B = 6559.13232 MiB
  差值            =  975428544 B =  930.24115 MiB（恰为fold owner总payload）
```

D153控制544次申请/544次释放；候选588/588；unknown free=0，最终0 B。
峰发生在giant/G树循环内，包含并存的raw、归约输出、owner、NTT及坐标/seed。
候选在frontier准入后live5506.763 MiB，下降后4754.637 MiB；其下降区间峰
5688.042 MiB，不能把这几个不同时间的数字相加。

D138和D153驻留候选同为4096 MiB共享大池，但D138的完整owned峰反而高95.143 MiB。
逐分配site给出原因：giant坐标分别487.265625/265.78125 MiB，fold owner分别
852.721619/930.241150 MiB，raw/output与seed还同时变化。
坐标来自[ox/oz申请](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6875)。
本节测量使用的legacy point chunk规则为整批向上取整；当前可切换实现见第20节：

```text
C = 256*2^20 ; w = LadderCtx.nw ; per_point = 16*w
k = max(P, floor(C/per_point))
Q_old = P*ceil(k/P)

w=126，floor(C/(16*w))=133152
D138/P126720：Q_old=253440=2P，坐标487.265625 MiB
D153/P138240：Q_old=138240= P，坐标265.781250 MiB
```

原注释称point chunk受256 MiB预算限制，实际向上整批会超出预算；最低1P本身
超过预算时也必须允许，属于显式最低工作集。它还会改变seed和segment缓冲、
chunk数与重复准备成本，因此D/P与giant chunk预算必须联合规划。

下一项实验候选为`Q_floor=P*max(1,floor(C/(16*w*P)))`：D138可从2P降至P，
仅坐标减少243.6328125 MiB；D153保持1P。需要完整叶子/非单位/seed/回退门禁和
同binary交错完整计时，因chunk数增加，**目前不能声称时间会减少或修改默认**。
由此也不能简单把“更大D”或“更多驻留”当作最优解。

![D153完整owned生命周期](figures/stage2_memory_ledger_d153_20261009.png)

![D138完整owned生命周期](figures/stage2_memory_ledger_d138_20261009.png)

### 19.4 门禁、复现与未完成项

内置20项ledger fixture覆盖增长/释放、地址复用、峰组成、重复指针、未知free、
null free、溢出及persistent标记，0 bad。Python10项parser正/负检查覆盖持久
基线守恒、峰和live组成、泄漏/unknown free及重复final拒绝。8组native两臂
矩阵（M37/67/29/253/16384、generic8193及D138/D153高位宽）通过，完整目标叶子
保持之前各D摘要；5条成功回退均闭合，carry污染仍拒绝发布、不输出正常final。
另外当前采集器两臂M37再次通过，台账默认关闭的M37两臂也通过、没有ledger输出。
原workspace144、dense树697、chunk580门禁保持0 bad。

本批没有修改算术/分块/驻留默认。D153首条检查前段与小规模正确性检查重叠，
该次耗时不作为性能证据；台账记录自身分配不计其它进程，驻留状态按真实free查询。
D138两臂串行测量，owned峰完全相同。所有binary/source身份与raw日志保留；41个
编译源文件已冻结，图与原始证据沿用不提交规则。

```text
binary SHA256: 0ebdf18e268f6eb4f6969644f93de873669ecd51f768e416513bdbcb9888d476
build_manifest SHA256: 7504fcfb759946a3f154eeaa596e5402e4e5698607d73829ad41a5633e7c8622
frozen_sources SHA256: de5f88d46a63b97f673162f235343b870452d5f00dcd9df0eb55a5dc879bc942
D138检查矩阵 SHA256: e2791e3253e9c0c6784ec357c7c92230085f9ea9aaa7cfd05d99176c06f12b63
D153检查矩阵 SHA256: acbcaed76406374e0c77866fa10f91549544e5424f52bff18e56f9bdd80dc68a
```

最终构建106.3 s、CUDA13.3/sm89/PTX3/addsub1/outer0/split8。
复现工具：[采集](D:/code/MPA-OpenCl/tools/bench/bench_stage2_carrier.py)、
[守恒门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_memory_ledger.py)、
[生命周期图](D:/code/MPA-OpenCl/tools/bench/analyze_stage2_memory_ledger.py)。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/memory_ledger_v2_stage2 -Arch sm_89 -SplitCompile 8
python tools/bench/bench_stage2_carrier.py --comparison owner-cache --exe build_cuda_cmake/memory_ledger_v2_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --memory-ledger --mode check --projection-only --output data/memory_ledger_d138_check
python tools/bench/analyze_stage2_memory_ledger.py --input data/memory_ledger_d138_check/measurements.json --output docs/benchmarks/memory_ledger_d138.json --figure docs/figures/memory_ledger_d138
```

台账补齐了全源码设备payload的实测合同，**尚未完成预测式MemoryPlan**。仍需将
Newton、整/残G树、fold、下降的shape/容量/淘汰序列和这些非NTT分配公式共用到
普通D与Auto B2候选规划；现有`legacy_additive`准入保持。下一轮先验证giant预算
向下整批的实际收益，随后用台账逐阶段核对预测，再校准各位宽/production B1
下的完整时间与驻留/回退成本；不将本轮插桩检查的时间混入旧正式profile。

## 20. Giant点坐标预算：整批取整与完整峰验证

### 20.1 可切换策略与预算合同

接续`4bb4340`。新增`NTT_GIANT_CHUNK_FLOOR`（默认0）和
`NTT_GIANT_POINT_BUDGET_KB`（默认262144 KiB=256 MiB），固定D/P，按完整G树批次
选择坐标chunk。所有算术内核、chain64/阈值32768、G树批次和驻留准入保持原合同。
[纯整数计划函数](D:/code/MPA-OpenCl/src/cuda/stage2/giant_chunk_plan.cuh:14)、
[执行入口](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7352)。

```text
C = 1024 * NTT_GIANT_POINT_BUDGET_KB
w = LadderCtx.nw ; P = phi(D)/2 ; I = giant_points
k = floor(C/(16*w))
Q_legacy = P*max(1, floor(k/P) + [k%P != 0])
Q_floor  = P*max(1, floor(k/P))
K        = ceil(I/Q)
coordinates_max_bytes = 16*w*min(I,Q)
minimum_over_budget   = [16*w*P > C]
```

`Q_floor`保证坐标预算，当且仅当最低1P本身可装下；否则仍保留完整1P，明确标记
`minimum_over_budget=1`。这是X/Z payload预算，**不是整个进程显存预算**，不得
从中扣掉seed/segment/NTT/fold等其它申请。纯整数函数检查0尺寸、0预算以及每步
乘法溢出；不复用原`k+P-1`的潜在溢出表达式。

原生debug记录`giant_chunk_plan`的预算/P/nw/Q/最低超限/预计chunk数，
`giant_chunk_done`记录实际chunk数和chain/ladder分支。常规控制台不新增日志。
[Auto B2 scope](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:617)拒绝未标定
的floor和非默认点预算；尚未加入INI/发布默认或替换`legacy_additive`全流程准入。

### 20.2 完整同时存活峰：实际减少259.01 MiB

输入为M8011/80111，target7995 bits，carrier8011 bits，w=126，B1=20，sigma=26，
B2=2.6e12。固定arena6300/fold1024/baby640/batch256 MiB，两臂BQ/raw/output
释放及冷cache准入均开启、physical chunk关闭；只切换point floor。顺序执行，
检查时开启完整目标叶子摘要和owned台账，不将这些耗时当正式性能数据。

```text
D1381380 / P126720 / I1882177 / G15
                 legacy             floor
Q                253440=2P          126720=P
chunks           8                  15
coord_peak       510935040 B         255467520 B
owned_peak       6977513600 B        6705922688 B
owned_peak MiB   6654.275513         6395.266235
fold/root/frontier 均驻留；cache eviction=0；最终0 B，unknown free=0
完整目标叶子：126720 leaves / 15840000 words / hash14409977791789275297
```

完整峰减少271590912 B=**259.009277 MiB（3.89%）**，由同时存活的7个分配site
组成差值严格复算，不叠加各模块自己的peak：

```text
X/Z                 2 * 127733760 = 255467520 B  (243.6328125 MiB)
segment products                     7983360 B  (  7.6135254 MiB)
seed X/Z            2 *   3991680 =   7983360 B  (  7.6135254 MiB)
seed scalar indices                    31680 B  (  0.0302124 MiB)
group products                        124992 B  (  0.1192017 MiB)
sum                                271590912 B
```

[坐标与segment申请](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6875)、
[seed缓冲](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6586)、
[group准备](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6715)。最大完整chunk n时，
原始容量公式（另计已有workspace的历史最大保留量）：

```text
B = ceil(n/64) ; ns = ceil(n/16) ; ng = ceil(ns/64)
seed X/Z bytes = 16*w*(2*B+1)
seed js bytes  = 8*(2*B+1)
segment bytes  = 8*w*ns
group bytes    = 8*w*ng
fixed group table bytes = 8*w*(64+1)
```

两臂有效points、15个G树、归约launch474、poly_muls2262396、coeffs45937081、
GMP样本42283及full checks4均相同。segment117637、group1842也相同；seed
points只从58828增加至58835（每个新增chunk带一个D seed）。传输的group
D2H仍为1856736 B，避免坐标D2H和leaf H2D各3794468832 B；不是通过减少数学
工作得到内存收益。group `t_prepare`在检查样本中0.923→1.520 s，重复chunk
启动/准备成本必须实测，而不能仅凭seed点总数估算全部耗时。

D1531530/P138240的最低1P坐标为265.78125 MiB，本来就超过256 MiB；两臂Q=P、
13个chunk、owned峰6559.132324 MiB、3次冷cache淘汰均相同。完整目标叶子
138240/17280000 words/hash1525678423161840651与上一版一致。
本结果不证明D138对所有预算最优；D/P和point chunk的预算阶跃须一起纳入规划。

![D138坐标预算与完整owned峰](figures/stage2_giant_chunk_memory_20261009.png)

### 20.3 正确性、回退与失败记录

原生298项计划fixture覆盖P=1/48/126720/138240、w=1/126/258、预算恰低于/等于/
高于完整批次、两种取整与0/溢出，0 bad。五种承载输入M37/67/29/253/16384，
1 KiB预算下覆盖多个ladder chunk与最低工作集超限；generic8193也通过。
unit曲线逐个对照独立CPU完整叶子；非单位曲线对照已知分母因子，不把不同的
projective fallback X向量当同一monic向量。

为覆盖生产固定阈值，新增可指定exponents/giant-count的CPU参考生成器。M37
使用32790个giant点、B2=6885480、512 KiB坐标预算：legacy Q32784，一次chain+
一次ladder；floor Q32760，两次ladder。两臂完整叶子hash11456702685322392679
与独立CPU参考一致，chain/seed/segment额外检查通过。目标分母均可逆，承载M域
有1131个已移除因子导致的非单位点，继续验证逆元应针对target的合同。

首轮尝试强制chain_min=0被production配置门禁拒绝，未启动曲线；没有放宽门禁，
改用上述真实规模。首版fixture错误地把legacy公式写成直接ceil(C/(16*w*P))，
而旧代码先floor(C/(16*w))再整批，产生15个测试失败；修正独立期望的取整层次，
重新编译。分块算法本体未因该fixture修正改变；两组失败原始证据均保留且不计为成功。

9个两臂完整检查=18条运行、298项预算fixture、20项allocator fixture、10项Python
守恒检查通过；42个编译源冻结。frontier/fold失败、pool关闭、三缓冲及arena拒绝
5条正常回退，独立叶子与内存守恒闭合；carry污染仍拒绝发布，不产生正常final。
原workspace144/dense tree697/physical chunk580门禁保持0 bad。

```text
binary SHA256: 49cc4a260c90669aad5d4fb589fdd84a8d38d62a28aa08aab25935963e8a5312
build_manifest SHA256: 3faf68d2cf87ec63b5e0f479b0974e1f7acdd1c4a84d33965ecbe2efdaccc1e0
frozen_sources SHA256: 8f1eb81a41463b4b5d64d7b86d2783356a8316a17af31ca37ffbaa83f0d9fd38
```

最终构建106.6 s，CUDA13.3/sm89/PTX3/addsub1/outer0/split8。
[采集与公式检查](D:/code/MPA-OpenCl/tools/bench/bench_stage2_carrier.py)、
[预算/路径审计](D:/code/MPA-OpenCl/tools/test/test_stage2_giant_chunk.py)、
[CPU参考生成器](D:/code/MPA-OpenCl/tools/bench/prepare_stage2_carrier_inputs.py)。
原始数据在`data/stage2_giant_chunk_20261009/`，图表与数据按既有规则不提交。

### 20.4 同binary正式计时与采用决策

同一production binary，同输入、D138和预算，关闭owned台账、额外leaf/seed/point
检查，必要算术检查保留。预热两条排除，ABBA+BAAB共8条正式样本，每臂n=4。
所有曲线fold/root/frontier驻留，cold eviction=0；两臂NTT work、GMP覆盖完全相同。

```text
complete Stage2 wall seconds                 legacy            floor
samples                                     85.067590         91.448147
                                            85.325821         91.331198
                                            85.186981         91.522362
                                            85.139660         91.467231
mean                                        85.180013         91.442235
sample SD                                    0.108897          0.080435
wall increase                                  +6.262222 s / +7.351750%
ABBA / BAAB increase                           +7.2690% / +7.4345%

mean parent phases seconds                  legacy            floor
giant                                        8.75825          14.47325
G trees                                     31.82350          31.80700
fold                                        15.97250          16.00875
descent                                      8.70550           8.70550
inverse                                      3.01675           3.02575
```

额外6.262 s中，giant增加5.715 s（91.26%）；G树/fold/下降基本不变。完整阶段
分账同时显示CPU/GPU准备杂项增加，不能把归约event再叠加到parent计时。数学工作
与核心NTT规模保持：共享大池4096 MiB、NTT完整模块4612.62 MiB、驻留owner
852.721619 MiB；这些模块量互相重叠，不与owned峰相加。

本轮只读遥测的忙时平均clock为2388.67/2384.62 MHz，温度77.01/75.17 °C，功率
79.08/71.33 W。没有修改功耗/频率设置；不假设旧测试的55 W条件。2秒采样的device
usage峰7296/7036 MiB，和owned台账的259.01 MiB差值一致到采样精度，但仍含运行时
开销且可能漏掉短峰。不能把采样曲线当精确分配生命周期。

[seed launch](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1035)与
[chain launch](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1114)的TPB均为64，
每个thread处理64个连续点。整块Q=126720/253440时，chain thread1980/3960、grid
31/62 blocks；更小grid与更频繁启动是候选解释，尚无event/profiler证据把5.715 s
细分到seed、chain、segment、分配或同步，不能宣称已证明occupancy原因。
只增加7个D seed不能解释全部时间；纯按点总数的成本模型不足以预测本拐点。

**采用决策：默认保持legacy，floor作为低显存实验策略保留。** 当前有充足显存的
D138任务采用legacy更快；内存受限时floor减少完整峰259.01 MiB，付出约7.35%时间。
未在本轮验证它能在更低arena/owner预算下避免回退，不能提前宣称该情况下会加速。

![完整时间及百分比阶段](figures/stage2_giant_chunk_20261009.png)

![模块容量及采样device usage](figures/stage2_giant_chunk_20261009_memory.png)

正式矩阵SHA256：`af3f6634fa0ecf8b5211a04bd1125d70a5596d9a2f952bf98d082b5815a2ef98`。
预算审计最终覆盖10个矩阵28条运行，原生fallback与ledger门禁另见§20.3。
所有原始log/debug/result、source/collector/Stage1 save身份保留；不存在运行中换
binary或更改采集器。计时未使用独立CPU长计算并行占用；GPU0的既有任务未操作。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/giant_chunk_v2_stage2 -Arch sm_89 -SplitCompile 8
python tools/bench/bench_stage2_carrier.py --comparison giant-chunk --exe build_cuda_cmake/giant_chunk_v2_stage2/ecm_cuda_stage2.exe --save data/stage2_n_scaling_20261008/study_v2/inputs/m8011_cofactor/stage1.save --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode timing --require-resident --telemetry --output data/giant_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/giant_timing/measurements.json --output docs/benchmarks/giant_timing.json --figure-prefix docs/figures/giant_timing
```

下一步把Q/K、seed/segment/group真实容量和跨调用保留量接入全生命周期MemoryPlan，
连同Newton/F树/完整及残G树/fold/下降的NTT shape及缓存淘汰序列一起核对owned台账。
普通D和Auto B2候选应在满足完整峰与reserve时最小化完整运行时间，而非总是取最大D
或最小chunk；还需各位宽、production B1=10e6～260e6及驻留/回退成本校准。本轮
B1=20用于快速有效存档，未据此声明生产B1或其他显卡的同等收益。完整预测式规划
仍未完成，下一轮继续；本阶段没有推广新发布默认。

## 21. 预测式全流程乘法请求 program：MemoryPlan 的输入合同

接续`033d71f`。本轮把单F树估算扩展为曲线执行前生成的五阶段请求program，
用于后续分配器/lifetime模拟。它不是采集完日志再重放，也不是完整MemoryPlan。
算术路径及发布默认保持，普通D/Auto B2仍用原准入。没有新增正式加速结论。

### 21.1 有序请求、残批和重复压缩

令`P=phi(D)/2`、`I=floor(B2/D)+2`、`G=ceil(I/P)`、`q=floor(I/P)`、`r=I mod P`。
每个请求为`(phase,ma,mb,nb,first,count)`；系数位宽S取算术承载位宽，
`w=ceil(S/64)`。有序生成器见
[ecm_stage2_requests.h:65](D:/code/MPA-OpenCl/src/core/ecm_stage2_requests.h:65)。

1. F树：逐层按`(min operand,max operand)`的map顺序合并非空兄弟；补齐到二次幂
   的空子树执行copy，不计NTT。请求输出`count=ma+mb−1`。
2. inverse：`g=1`起，`n=min(2g,P+1)`，每步为`(n,g,1,0,n)`及
   `(g,n,1,0,n)`。实际实现每步显式resize，故系数末尾为零也不缩短下一步长度；
   见[cp_inv_series](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4742)。
3. 第一个完整G树：只seed H，H长度`P+1`。第二个G树后首轮fold依次为
   `G(P+1)×H(P+1) -> 2P+1`、`reverse(P+1)×inverse(P+1) -> P+1`、
   `q(P+1)×F(P+1) -> P`。
4. 后续完整G树：每棵G树后fold，H长度按合同保持P；三次乘法为
   `(P+1,P)->2P`、`(P,P)->P`、`(P,P+1)->P`。
5. 最后残G树有r个点，根长度`r+1`。若已有至少两棵完整G树，三次fold为
   `(r+1,P)->P+r`、`(r,r)->r`、`(r,P+1)->P`；若只有一棵完整G树，
   H仍为P+1，第二个乘法长度为r+1。不能把残G树当完整树计费。
6. scaled root：P×P，仅保留P个输出。随后从root往下，按`(child degree a,
   sibling degree b)`的map顺序，乘`ma=a+b,mb=b+1,nb=组内节点数`，取
   `[first=b,count=a]`。无sibling时copy；这是不同于product tree的请求组织。

程序按`initial F/inverse; first G; second G/fold; repeated middle G/fold;
partial G/fold; root/descent`存储，最多6个block。中间重复次数可很大，NTT保留
需求只需处理每个block的不同请求一次；顺序签名以模2^64线性变换的二分幂合成，
复杂度随重复次数的对数增长，见
[RequestSignature](D:/code/MPA-OpenCl/src/core/ecm_stage2_requests.h:32)。

**合同边界**：要求resident batched、cached inverse、scaled root/frontier、fold
余式长度保持P、没有额外诊断乘法。实际H长度由设备最高非零系数确定，是数据
依赖量；截短/零余式可改变后续请求。I<=P的单G批次需要local inverse，完整G根
还需root division，当前明确`valid=false`，没有套用G>=2的模型。非单位、退化、
设备/arena拒绝和其它回退仍走原算术路径，不能使用本合同保证其请求/内存。

### 21.2 全请求 NTT 保留容量，仍非完整显存峰

[request_plan](D:/code/MPA-OpenCl/src/core/ecm_stage2_requests.h:130)直接调用实际
`ntt_shape_query(max(ma,mb),S)`，fuse表/base直接用当前device分配器descriptor。
每组nb沿原halving序列选择chunk c；末尾不足c的slice也保留独立key。

对于**无淘汰、无额外scratch且满足上述请求合同**的情况，令`o=2max(ma,mb)−1`：

\[
B_{big}^{pool}=\max_{requests}8bNc,\qquad b\in\{2,3\};
\]

\[
B_{digits}=\sum_{(N,s)\in keys}8s\left(\max_{requests\ at\ (N,s)}o+2\right);
\]

\[
B_{tables/base}=\sum_{N\in requested\ lengths}
  \left(B_{table}(N)+B_{base}(N)\right),
\quad B_{NTT}=B_{big}+B_{digits}+B_{tables/base}.
\]

pool关闭时，`B_big=sum_keys 24Ns`，不是共享最大值。S4的请求输出峰另外为
`max_requests 8w count c`。这些是NTT保留与请求组件，不能直接加上其它模块各自
峰值形成进程峰。尤其保留容量、当次请求容量、临时重建峰是三个不同概念。

生产入口的`--plan-only`新增`request_program.version=1`、compressed blocks、
各阶段group/pair/chunk及大池/输出请求峰、NTT保留组成、顺序签名；
见[plan JSON](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:9637)。
始终标记`process_peak_complete=false,admission_model=false`。

### 21.3 原生核对结果与规划启示

同一冻结v2 binary、GPU1 4060 Laptop，使用有效B1=20存档、sigma26，
高位宽目标`M8011/80111`，承载M8011，B2=2.6e12，BQ1；arena/fold/baby/batch
6300/1024/640/256 MiB，raw/output释放与cold cache策略启用。两臂仅point
chunk取整不同，本轮所有运行都是诊断check，不用于正式计时结论。

- D1381380：P126720、I1882177、G15。188条compressed请求表示474个真实S4
  group；各phase `(group,pair,chunk)`为：F树`(23,126719,267)`，
  G树`(344,1882162,3996)`，fold`(42,42,42)`，descent`(31,253439,921)`，
  inverse`(34,34,34)`。合计2262396个多项式乘积、5260个NTT子调用。
- D1531530：P138240、I1697650、G13。172条compressed请求表示379个真实S4
  group；F树`(20,138239,248)`，G树`(263,1697637,3054)`，fold`(36,36,36)`，
  descent`(24,276479,857)`，inverse`(36,36,36)`；合计2112427个乘积、4231个
  NTT子调用。group/chunk不是CUDA kernel launch总数。
- D138的NTT预测与实测均`4836680360 B = 4612.617836 MiB`；其中大池
  `4294967296 B`、digits`5663512 B`、表`363878560 B`、base`172170992 B`。
- D153均`4837454864 B = 4613.356461 MiB`；大池、表、base相同，digits
  `6438016 B`。D153有3次冷context回收，释放75914240 B；全局NTT峰相等
  并不验证淘汰的时序或与owner同时存活的容量。

**更直接的生命周期证据**：D138的F树大池为2048 MiB，inverse增长到4096 MiB。
后续G树虽然每次最多请求2048 MiB，实际仍持有4096 MiB共享大池。模型不能只按
G树请求大小计算giant阶段；同样不能把每个shape的大池重复求和。D153的padded
树顶已跨下一档，F/G树请求本身就是4096 MiB。

两臂五phase的group/pair/chunk、大池/输出请求峰、每phase顺序与跨phase顺序
全部吻合；D138两种point Q产生相同NTT请求序列。这进一步确认§20较小point Q
的耗时回退没有减少NTT工作，尚不能据此推断具体SM occupancy或stall原因。

### 21.4 门禁、身份与待完成工作

`--request-audit`只启用默认关闭的`NTT_REQUEST_AUDIT`；原生日志仅写debug，
见[S4请求记录](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:3559)。
rolling signature不是密码学身份；仍保留SHA、独立完整目标叶子及GMP/carry门禁。
旧Auto B2 profile拒绝插桩，见
[main.cpp:616](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:616)。

- CPU-only C++独立dense padded树/下降oracle：**70010检查，0 bad**；覆盖P1–257，
  65536边界、126720/138240/262145、高重复与溢出，两/三缓冲及两分块策略。
  fixture曾错误要求P48/I=ULL_MAX的分phase计数溢出；这些计数其实可表示，
  修正为P1的真正fold group溢出用例。G1缺少local inverse的首版计划已在v2
  明确拒绝；上述失败不作为成功数据。
- 40组native plan：5个D×4个pool/reuse组合×2种分块，单树原门禁与新独立
  全流程dense topology均通过。另4个native边界计划：G1部分/完整明确不支持，
  G2首个部分fold支持，B2=9e18的大重复仍<=6个block且执行曲线0。
- M37/M67/M16384 carrier、generic8193、8011-bit D138/D153，共6矩阵12次check：
  五phase计数/顺序及保留容量核对通过；小unit用独立CPU完整叶子，通用/高位宽
  与上一冻结binary完整目标叶子相同。所有必要GMP/carry检查通过。
- 12条native owned台账、5条正常回退、10条parser门禁通过；fold/frontier失败、
  pool off、三缓冲、arena拒绝正常返回独立CPU目标结果，carry污染被拒绝。

工具为[request gate](D:/code/MPA-OpenCl/tools/test/test_stage2_request_program.py)、
[CPU fixture](D:/code/MPA-OpenCl/tools/test/stage2_request_program_fixture.cpp)。
复现命令见[README](D:/code/MPA-OpenCl/tools/bench/README_STAGE2_CARRIER_PLAN.md)。
原始证据目录`data/stage2_request_program_20261009/`继续忽略、不提交。
冻结production v2，CUDA13.3/sm89/PTX3/addsub1/outer0/split8/GMPzen3/MSVC14.51：
编译83.6 s、完整build86.4 s，43个编译源冻结。

```text
binary SHA256   4f42b75e213adfcc509cedee3210002cb3ca5ae6c5d9a3bc13464c82f3863ab4
build SHA256    9eaff7d2f895474314fd2551516b5fd39ba8f7d1240fe505f2046423a3e30df1
snapshot SHA256 1059f1f998adda0f667b9f93bb10dd881a17a6f4b23db51c244fc8d276c8e261
request gate    a2c4dc99275cc0e5b1562f29179278a8efe6720b7bcb25573f4f19a2dd8e1f26
```

**下一步是分配器状态模拟和非NTT生命周期，不是继续把模块峰相加**：按program
请求顺序模拟共享池容量、shape-local digits、table/base、cap拒绝/淘汰及驻留
边界cold trim；同时按实际申请/释放顺序放入S4 raw/output/reducer、自检临时、
baby/giant坐标、seed/segment/group、G metadata、fold/frontier owner等。用§19
完整owned台账逐checkpoint核对，并单独以真实free扣除driver/runtime及reserve。
补齐数据依赖的短余式、G1和回退合同后，再让普通D与Auto B2共用MemoryPlan。
随后在各位宽及production B1下标定NTT/点chunk启动与驻留成本；本轮没有把旧
profile用于新布局排序，完整规划仍在推进。

## 22. 有序 NTT arena 分配器模拟：预算淘汰与成功前缀

接续`061cf17`。五阶段请求program现已驱动纯CPU的分配器状态模型，见
[NttMemoryState](D:/code/MPA-OpenCl/src/core/ecm_stage2_ntt_memory.h:35)和
[ntt_memory_plan](D:/code/MPA-OpenCl/src/core/ecm_stage2_ntt_memory.h:204)。
这是D/P、batch、arena预算联合规划的NTT组成部分；普通D与Auto B2仍使用原
准入。此阶段没有修改算术内核、发布默认，也没有新增GPU算术或正式计时结论。

### 22.1 按分配顺序计算，而非只求所有shape最大值

生产分配器的三个关键顺序见
[fuse lookup](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2324)、
[buffer lookup](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2130)和
[cap eviction](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2070)：

1. **先解析fuse context，再申请大缓冲。** 新context必须先满足当前cap，
   这一处不主动淘汰其他context。context拒绝会走per-call路径，不能先假设稍后
   的工作区增长一定会释放足够旧缓存。
2. **共享大池在增长前释放旧池。** 两缓冲只在pool+B/Q reuse同时启用时有效；
   pool关闭时各`(N,slices)`仍持有独立三缓冲。共享池命中保留历史最大容量，
   不随当前请求变小而缩小。
3. **大缓冲超cap才淘汰其他shape。** 保留当前N的外层表，以及当前
   `(N,slices)`的keyed缓冲；其他shape的外层表、big/digit/verdict缓冲被释放。
   carry scratch也被释放，但不计入原`tbl_words_freed`统计。
4. **外层表淘汰不释放context base。** 后续命中该context时，原实现不重新
   缓存外层表；通过已有scratch临时生成twiddle。不能把下一次命中等同于
   完整表重建，或把所有context base也当作已经释放。
5. **digits增长只检查增量，不再触发淘汰。** 先检查增量cap，再释放旧dOut、
   申请新dOut；dRes保留。模型在fuse/big/digits三个预算拒绝点分别停止，
   保存当时已成功分配的owned NTT payload，不假装已模拟per-call回退临时峰。

令`t`表示一次成功分配或释放后的状态，`C(t)`为共享池单缓冲capacity words，
`K(t)`为keyed big集合，`O_k(t)`为keyed digits的out_cap，`s_k`为slices，
`b=2`或`3`。NTT同时存活量为：

\[
B(t)=8bC(t)+\sum_{k\in K(t)}24N_ks_k;
\]

\[
S(t)=8\sum_{k\in digits(t)}(O_k(t)+2)s_k+C_{carry}(t);
\]

\[
A(t)=B(t)+S(t)+\sum_{f\in cached\ tables(t)}T_f+
\sum_{f\in live\ contexts(t)}Base_f,\qquad Peak_{NTT}=\max_t A(t).
\]

共享池模式通常没有keyed big；上式也明确区分pool关闭的三缓冲。这里的
`Base_f/T_f`直接使用实际device的fuse descriptor，而非按NTT长度猜测比例。
算术位宽、shape与请求拓扑继承第21节的条件合同。

**carry内部分片**：大缓冲以整个NTT batch分配，但pass runner每批最多65535
slices。对每个内部批次`m`，只有`Nm>=2^20`才请求
`8m ceil(N/256)` bytes carry scratch；保留最大成功容量。cap不够时只改用
原检查核，不拒绝该NTT请求。初版模拟按整个batch计费，已在最终冻结前修正并
覆盖65535、65536、131071边界。生产plan的既有supported条件仍拒绝额外carry
诊断路径，未把这项组件检查当作完整诊断运行的生命周期保证。

### 22.2 重复块、冷context接口与报告语义

重复G树/fold不按B2线性展开。每个相同chunk、每个重复block执行到其
**完整allocation state**不再改变后，按已观察的计数增量乘上剩余次数；
有序context列表、table是否仍缓存、keyed容量、共享池与carry容量全部参与
相等判断。不是只比较一个total bytes。乘加计数溢出明确失败。buffer lookup还与原分配器一致，按三缓冲的保守size_t
边界检查输入，即使启用B/Q复用也不放宽该检查；整数边界错误不当作正常cap拒绝。

[cold_trim](D:/code/MPA-OpenCl/src/core/ecm_stage2_ntt_memory.h:166)可以按实际
`drop_cold_fuse`的规则释放最大非hot完整context，同大小时保留插入顺序；
再次请求时会重建base与表。此接口只接受调用者提供的**逻辑payload余量**。
当前生产plan尚未在fold/frontier边界调用它，因为真实同时存活的非NTT缓冲
还没有全部进入预测式模型。不能假设逻辑释放bytes等于驱动返回的free增量。

`--plan-only`新增`ntt_memory.version=1`，见
[接入](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8682)和
[JSON输出](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:9677)。

- `valid`：成功完成这一条件模型的计算；**预算拒绝前缀也可以valid**。
- `finished`：模型走完全部请求，未遇到上述cap拒绝。不是完整Stage2能驻留。
- `peak_bytes/final_payload/checkpoints`：按分配顺序计算的NTT组件峰、末态和
  每个压缩block末态。遇到拒绝时peak只覆盖成功前缀。
- `stopped_at`：首次拒绝的block、重复序号、request序号、phase、ma/mb、N、
  slots和slices；正常完成或不支持的G1输出null。
- `counters`：NTT子调用、fuse命中/创建、大池增长、cap淘汰次数及释放bytes。
- `cold_trim_modeled=false,fallback_modeled=false,process_peak_complete=false,
  admission_model=false`：明确阻止把这一层NTT预测当成完整任务准入。

编译脚本把新头文件加入源闭包、增量编译和HostOnly身份校验。
模型只在显式请求plan时计算，正常曲线入口不增加模拟开销。

### 22.3 原生规划查询结果与实际优化启示

GPU1、target7995/carrier8011 bits，有效B1=20存档，B2=2.6e12、batch256 MiB。
只初始化CUDA读取规划参数，**执行曲线0**，不是新的显存分配或性能实测。
5个D×4个pool/reuse组合×2个chunk策略共40组：16组走完模型，24组明确报告
成功前缀和首次cap拒绝。当前arena cap6300 MiB：

- D1381380/P126720的三缓冲共享池在inverse请求
  `(ma=126721,mb=65536,N=2^28,slices=1)`首次拒绝大缓冲。
  此时历史前缀峰3585.998039 MiB不是完成整个任务需要的容量；其大缓冲
  单项要求6144 MiB，加上已保留context base便超过6300 MiB。
- D1531530/P138240三缓冲在F树顶请求
  `(ma=7169,mb=131073,N=2^28,slices=1)`已经发生同类拒绝。
  两个D的最终最大NTT长度相同，**首次跨档阶段不同**，全生命周期准入必须
  保留阶段顺序，不能只看最终最大长度。
- 两缓冲、原chunk策略、cap6300 MiB：D138走完5260子调用，预测峰
  `4836680360 B=4612.617836 MiB`；D153走完4231，
  `4837454864 B=4613.356461 MiB`。两者与第21节的无淘汰容量相等。
  这仅复核NTT层；旧D153实测有3次边界cold trim，尚未在新plan中预测。
- 同一D138、同batch256 MiB与chunk策略，cap降到4600 MiB，模型仍走完5260
  子调用：1次cap淘汰释放80467208 B，末态表从363878560降至286327808 B，
  base仍172170992 B；峰`4759122416 B=4538.652817 MiB`。
  NTT子调用数相同，但被淘汰的外层表改为临时生成twiddle，不能假设耗时相同。
  此处可以看到未来“主动留出owner预算、允许回收表缓存”的选择空间，尚未
  实测该预算下完整驻留、性能或物理VRAM峰。

另6组边界query验证G1部分/完整均不支持、G2首个部分fold支持、B2=9e18仍只
显式模拟6个block并跳过1785714285714282个稳态重复、arena1 MiB首次在F树
`N4096/slices495`拒绝；统一batch256 MiB的cap4600案例见上。首轮边界query
曾使用CLI默认batch32 MiB，原始记录保留，最终冻结版本的统一预算结果使用`edges_final`，不混比。

### 22.4 验证、构建身份与后续范围

CPU门禁为
[test_stage2_ntt_memory.py](D:/code/MPA-OpenCl/tools/test/test_stage2_ntt_memory.py:21)。
它从**当前生产源码直接提取**`NttWorkspacePolicy/NttArena`、buffer lookup、
fuse lookup及cold trim代码，在CPU opaque allocation ledger中编译运行；
没有CUDA库/driver/context调用，也不按测试重新实现一套arena淘汰算法。

- 最终**681389项检查、0 bad、GPU调用0**：两/三缓冲、pool关闭、digits增长、
  cap拒绝、carry拒绝、cap淘汰保留base、冷context完整释放/重建，以及完整
  请求program逐次执行对压缩执行。
- 原独立dense topology CPU回归仍为**70010项、0 bad**。
- fake fuse descriptor使用可核算的合成容量；原生40+6组query另外使用实际
  device descriptor。前者证明分配规则一致，不证明NTT算术或驱动OOM行为。
- 初版fixture的overflow案例实际上仍可表示；改为每块3个子调用的真正
  溢出。手算base合计曾误写4096而非3072 B，修正期望后再次完整执行；失败
  记录保留在cpu_v1/v2，未计作通过或性能收益。
- 普通沙箱首次nvcc编译发生host compiler ACCESS_VIOLATION，随后沙箱外
  编译成功；这是编译工具失败，不作为GPU硬件稳定性的证据。

冻结production v3：CUDA13.3/sm89/PTX3/addsub1/outer0/split8/GMPzen3/MSVC14.51，
44个编译源，compile83.1 s、完整build86.0 s。原始日志、生成CPU fixture、
规划query与身份在`data/stage2_ntt_memory_20261009/`，继续忽略、不提交。

```text
binary SHA256   8b06bd35bdb8fa02ed9fcd9cbc106de8a042308e3c7d824356d9c2b3b0e49441
build SHA256    5142f8f434db5aa6c40de972cfbb2ede6cfeca67754b11d9ec2b6d817cbbc449
snapshot SHA256 e369d21a9d202c4fd3285fa62fa268fc7bf1d5c87d1cc8ae4f6dde0db594c4d6
```

复现命令见[工具说明](D:/code/MPA-OpenCl/tools/bench/README_STAGE2_CARRIER_PLAN.md)。
用户已决定4060lp保持默认1800 MHz/55 W；本阶段无计时，不调整电源/频率，
旧约79 W/2385 MHz结果不作为新条件下的校准值。

**下一步继续同方向**：把非NTT申请/释放序列与本状态机合并，按fold/frontier
实际边界调用cold trim，逐checkpoint对照第19节owned台账，再加真实free与
reserve准入。还需补齐短余式、G1、回退路径及额外自检合同，才能替换普通D
和Auto B2；不能因为NTT层`finished=true`就放行完整Stage2任务。成本校准须在
新55 W基线上测量表淘汰、batch/Q与驻留的完整时间，不能只用NTT子调用数排序。

## 23. Giant组件生命周期：保留容量、尾块路由与S3统计修正

本阶段沿用第19–22节联合规划方向，继续处理非NTT分配。新增
[纯整数组件模型](D:/code/MPA-OpenCl/src/core/ecm_stage2_giant_memory.h:42)，
由[生产plan-only入口](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8695)读取实际策略，
输出`giant_memory.version=1`。正常曲线不执行此模拟；没有修改算术内核、D排序、
Auto B2或发布默认。完整MemoryPlan仍未完成。

### 23.1 容量与同时存活关系

约定`w=ceil(S_bits/64)`、`I=floor(B2/D)+2`、`Q`为既有giant chunk规划结果。
对某个实际chunk，`n=min(Q,剩余I)`，chain每线程点数为`c`（正常64），
segment大小`s=16`，inversion group大小`g=64`。所有公式为owned payload字节，
不含CUDA上下文、驱动、分配器粒度、页锁定主存或其他Stage2组件。

- chain seeds：`k=2*ceil(n/c)+1`；ladder工作区点数：`k=n`。
- 保留点容量：`C_pt=max(C_initial,此前所有chunk的k)`；它不会在短尾块缩小。
- 点工作区：`8*C_pt*(1+2*w)`，含indices及X/Z。
- S3固定常量：`40*w`；giant base一旦构建，保留`16*w`。
- exact segment修正表一旦构建，保留`8*(s+1)*w`。
- chain输出X/Z：独立临时owner，`16*n*w`。ladder的X/Z借用上述工作区，不能再加一次。
- segment输出：`8*ceil(n/s)*w`。chain和驻留ladder在设备上生成；非驻留ladder在CPU生成。
- 驻留group输出及修正表：`8*(ceil(ceil(n/s)/g)+g+1)*w`，保留到该chunk的所有G树和fold完成。
- 旧host seed路径额外六个设备seed缓冲：`8*(4*ceil(n/c)+2)*w`；它们与chain输出和segment重叠，
  但在group申请前释放。生产入口禁止该算法切换，组件规则仅通过CPU门禁覆盖。

令`W_s3`为固定常量、保留点容量、base、segment修正表的和。则chain生成期为
`W_s3+16*n*w+segment+legacy_seed`，驻留树处理期为
`W_s3+16*n*w+segment+group`。驻留ladder树处理期只为`W_s3+segment+group`。
非驻留chain的坐标和segment在G树前释放，G树期间本组件只剩`W_s3`。

最终累积调用`need_vals(P)`，当前仍同时保留`dvals/dprod`各`8*P*w`，因此
`accumulation_bytes=W_s3+16*P*w`。这是明确的同组件阶段状态，不是全进程峰。
相关原代码：[S3工作区](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6491)、
[驻留group](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6691)、
[chain](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6758)。

满chunk重复时分配容量不变，所以只模拟一个满chunk和一个尾chunk，`repeat`记录次数，
即使`B2=9e18`也只有至多两个组件状态。它是条件组件程序：若早期NTT失败，实际运行
可能不会到达giant阶段；本结果不能越过NTT拒绝前缀作为准入依据。

`C_initial`来自小素数处理。正常baby证明cache可复用且匹配时，只有`p|D`、
`B1<p<=min(B2,D/2)`的素数需要补ladder；D138/B1=20为1个，D153为0个。
因此不能固定假设初始容量为1。JSON明确`small_prime_cache_assumed=true`这一条件；
cache失配、故意污染和诊断额外工作区不属于完整运行保证。

### 23.2 查询结果与已知台账

M8011承载、B1=20、D138=`1381380`、P=`126720`、B2=`2.6e12`、w=126：

- 原chunk向上取整：Q=253440，seed保留容量7921；giant结束S3为16056296 B
  （15.312477 MiB），giant组件峰543273560 B（518.106041 MiB）。
- floor策略：Q=126720，seed容量3961；giant结束S3为8041256 B（7.668739 MiB），
  组件峰271682648 B（259.096764 MiB），减少259.009277 MiB。
- 构造`I=253440+32000`，尾块切换到ladder后，seed容量从7921增至32000，
  giant结束仍保留64792192 B（61.790649 MiB）。单看最大chain seed数量会低估该状态。
- 强制ladder时，点容量253440一直保留到最终累积，组件峰变为768452256 B
  （732.853180 MiB），且峰发生在累积期；不能仅用giant阶段坐标峰比较算法。
- `NTT_DEVICE_GLEAF_MAX_MB=1`导致非驻留路径：G树期间只剩S3工作区，但chain生成期
  仍要申请大坐标owner，组件峰仍为517.805153 MiB。关闭驻留不会消除坐标生成峰。

D153/P=138240/Q=138240的giant组件峰296372456 B；同一组件模型也匹配8193-bit
通用模数、I=65/Q=24的驻留ladder路径，峰142608 B。
这些值均与旧二进制冻结源码对应的分配site核对：6个运行、24个live边界、6个
`after_giant_loop`同时存活峰成分全部相等。使用旧源码确定site行号，不用当前行号
解释旧日志。边界为before_inverse、after_giant_loop、after_frontier_admission、after_descent。

**NTT峰与giant峰仍不能相加作为进程峰。** 第20节实测floor策略减显存但变慢的结论
保持；本轮没有新的运行计时，不能从上述容量变化宣称加速或更新发布默认。

### 23.3 修正与门禁

`S3Workspace::need_pts/need_vals`原先释放旧缓冲后只向`bytes`加新尺寸，没有扣旧容量，
导致多次增长后统计虚高。本轮在成功替换后扣除旧点/值容量；申请尺寸、释放顺序、
保留策略和算术均未改变。
[点扩容](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6578)、
[值扩容](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6593)。

验证证据保存在忽略目录`data/stage2_giant_memory_20261009/`：

- `cpu_final_v4`：86068检查0 bad、GPU调用0；直接提取当前生产S3源码，CUDA/GMP/launch
  替换为CPU分配台账/空操作，检查扩容、借用、临时owner和压缩重复。它不验证算术。
  冻结的修复前源码在同一fixture触发`S3 bytes counter includes released capacities`，
  作为预期失败保存；不是把旧失败记成通过。
- 同目录核对6份历史owned台账，日志、旧binary和冻结源文件SHA均验证。
- `ntt_regression_v1`：原NTT门禁681389检查0 bad，dense请求回归70010检查0 bad。
- `plan_gate_v1`：40组NTT/giant联合字段的原生规划查询通过，曲线0。
- `native_final`：7个原生计划响应（其中1个诊断路径明确不支持）及3个既有生产算法
  保护拒绝通过；包括floor、chain→ladder尾块、强制ladder、驻留预算回退、超大B2。
  旧seed、关闭small-prime reuse、短chain实验切换由生产保护拒绝，不算成功规划。

早期失败文件保留：native_v1/v2的算法保护是测试预期遗漏；cpu_v2的历史matrix筛选
过严；cpu_final漏收调用者中的ladder segment site。修正验证器后完整重跑，未掩盖失败。

production v1构建45份源闭包，CUDA compile69.5 s、总build85.6 s，binary SHA：
`dbea561a9631b4e321c5a35323fb9460b9153c64196e9eb2ffda4b3db08d983e`。
build SHA：`bf13ceaa4e21c7f5925ba5059b615210dc72daa980b7a505198a32a5502a791d`；
snapshot SHA：`55e833b3182f0c71ee67372a47fdb4abb102aa7b3d5db18f1a512b5531312afa`。
本轮无GPU算术、完整计时或物理OOM验证，4060lp保持用户默认1800 MHz/55 W。

### 23.4 下一步

1. 把S4 raw/output、reducer shape/selftest暂存及G树metadata接入有序请求，再与本组件
   和NTT状态机合成同时存活状态；补fold/frontier申请、释放、cold trim和实际free准入。
2. 独立验证`dprod`按块数分配：
   [唯一生产消费者](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6978)只写/读`ceil(P/64)`个乘积，
   当前却分配P个。理论可省`8*w*(P-ceil(P/64))`，D138为119.913025 MiB，
   D153为130.814209 MiB；这是候选容量收益，尚未实现或实测时间收益。
3. 完整模型通过G1、短余式、诊断/回退合同后，再统一普通D与Auto B2候选准入。
   新55 W基线需重新测全流程成本，包含Q、驻留、表重建和NTT长度拐点。
