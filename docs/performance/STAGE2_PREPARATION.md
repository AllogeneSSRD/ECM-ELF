# Stage2 准备、同步与补零开销

## 输入与证据范围

准备/同步剖析及首组固定D实验使用GPU1 RTX4060 Laptop、24SM、8GiB，UUID `8a67b1f8ef1c3177a822813a7ac2224d`，驱动610.88、CUDA13.3.73、MSVC14.51.36231，55W/1800MHz上限；该组GPU0有外部任务，设置上限不代表实际恒频恒功耗。两卡SM1500MHz的扩大验证另列下文，不与该组样本混合。测试工具不修改显卡控制。

主要剖析输入为M4423、B1=20、sigma26有效Stage1 save，B2=2.6×10¹⁰、D60060、P5760，batch256、arena6300、fold640 MiB；Graph、短batch选型和短校准关闭，中段M6开启。Nsight Systems2026.1.3以管理员运行，保留全部CUDA API和CPU NVTX范围，采样/CPU上下文切换采集关闭。所有剖析运行均核对因子、rootfold、NTT工作量、S4分块与未插桩输入一致，不进入性能统计。

正式联合计划矩阵、重复数、全部位宽和B2见[outer性能](NTT_OUTER.md#tileouter-联合组织)。性能二进制为 `3a6da7e504b80f3da45cb9cf9c1879444e209b5f14e39d7ba9c5a191fd427a2e`；扩充CPU标记的剖析二进制为 `6e42ab02480240a9512e40c5ad72b4a9f6f16a5319a2fb38984076dbb1112f90`。两版完整SASS文本SHA256均为 `8973bd25d87e5ff566e7bb80e6a5f7e415abcb085373d70c926a64dfc33a2831`，GPU指令相同；后者另外通过完整曲线工作量/输出核对。它们的主机标记范围不同，不混合计时。

## 计时边界

`stage2_full_wall.total=init+main`。init已包含单独打印的shape时间，不能再加shape；这里shape是baby索引枚举，D/显存候选规划在计时之外。

- init：索引枚举，加Montgomery启动自检之后的上下文/S4初始化、baby生成与F树；包含S4 pack/division和形状自检。
- main：inverse、giant/G树/fold、下降、累乘、GCD、命名与最终oracle排空。
- Montgomery启动自检、保存点验证、进程启动、D/显存规划及结果发布位于引擎total之外。

本页GPU窗口从本进程**第一个到最后一个GPU事件**，包含kernel、copy和memset并集。它不覆盖启动前和最后GPU事件后的全部进程活动，不能称为完整进程墙钟或全卡空闲率。

原二进制同一矩阵off侧各4样本中位数为：

| 输入 | 引擎total | 进程墙钟 | 差值 |
|---|---:|---:|---:|
| M521、短B2 | 0.149092 s | 0.616537 s | 0.467445 s |
| M2203、短B2 | 0.596838 | 1.082648 | 0.485810 |
| M4423、B2=2.6×10¹⁰ | 6.299864 | 6.869408 | 0.569545 |
| M2203、P155520 | 19.540058 | 20.218640 | 0.678582 |

差值包含读取、规划、CUDA初始化、计时外检查和收尾，不是纯CPU准备时间。短曲线会明显暴露这些固定成本；长曲线应分别评价引擎和进程指标。

## GPU空隙的互斥分解

扩充标记的一条诊断曲线GPU窗口6.349079秒，本进程没有kernel/copy/memset事件的区间合计0.598152秒，即9.42%。算法将这些区间与同一worker的CUDA runtime调用做区间交集，按不重叠片段计账；不同API或NVTX范围的嵌套时长不重复相加。

| 空隙所在位置 | 秒 |
|---|---:|
| 已记录runtime API之外 | 0.256011 |
| cudaMemcpy调用内 | 0.174105 |
| kernel提交调用内 | 0.044563 |
| cudaDeviceSynchronize内 | 0.024775 |
| cudaMemsetAsync内 | 0.021675 |
| cudaMemcpyAsync内 | 0.019838 |
| cudaFree内 | 0.018067 |
| cudaMalloc内 | 0.011546 |
| cudaHostAlloc / cudaFreeHost内 | 0.008958 |
| cudaFuncSetAttribute内 | 0.003155 |

未列小项保留在JSON中；以上表格不是全部API持续时间。空隙落在某调用内不证明该API是唯一原因，可能包含驱动调度和剖析影响。

runtime API外的0.256011秒，按最内层CPU范围归属为：

| 范围 | 秒 |
|---|---:|
| 尚未标注 | 0.079498 |
| S4请求准备 | 0.045451 |
| S4常量初始化 | 0.031888 |
| 主机树驱动 | 0.030125 |
| Montgomery自检 | 0.019810 |
| NTT形状/缓存规划 | 0.016544 |
| NTT请求准备 | 0.008868 |
| S4形状自检 | 0.006561 |
| GMP pack自检 | 0.004281 |
| baby准备 | 0.003307 |
| G树level准备 | 0.003155 |
| GMP快照校验 | 0.002554 |

其余小项在原始JSON。范围覆盖函数的CPU执行和等待，不能直接等同于该函数的可消除成本。Montgomery启动自检处于引擎total之外；各项优化价值还要按上一节计时边界评价。

![GPU空隙及CPU范围归属](../../data/figures/stage2_prep_gaps_20261015.png)

## 传输与等待

相同剖析中的真实GPU copy事件为：

| 方向 | 事件数 | bytes | GPU累计时间 |
|---|---:|---:|---:|
| H2D | 1549 | 155,880,627 | 13.703 ms |
| D2H | 2146 | 125,198,016 | 12.165 ms |
| D2D | 309 | 324,326,240 | 1.809 ms |

参考CPU范围追踪中的1843次同步cudaMemcpy API累计约4.56秒，大部分与GPU工作重叠。默认流上读回结果依赖前面的计算；API持续时间不能当作PCIe传输时间，也不能与kernel持续时间重复相加。保留同步前须先识别其结果消费者、缓冲复用和算术断言依赖。

当前样本不支持“传输带宽是主要瓶颈”的结论。常规及全API追踪中，属性设置在空隙中约3–4ms，pinned申请/释放约9–23ms；单独缓存它们无法解释数秒API等待。

## pack与补零的候选空间

GPU memset共4391次，写入84,325,430,832 bytes（78.53GiB），累计0.298869秒。按发起调用的correlationId关联CPU范围，NTT.request内4315次贡献84,325,430,216 bytes和0.298839秒；该范围包含A/B工作区补零及诊断计数器，不能把所有次数都归为大数组清零。

同一曲线pack kernel累计0.256848秒。默认direct/gather输入路径先对A/B整个NTT数组补零，再写占用的digits。对一个m-slice请求，长度L、系数位宽S、digit宽b、两输入系数数ma/mb，该路径写入量近似为：

- 全数组补零：16·m·L bytes。
- 有效digit覆写：8·m·(ma+mb)·⌈S/b⌉ bytes；contiguous路径按padding后的P打包。

完整输出覆盖的pack/zero融合把这两次写入组织成一次，并减少两次独立memset提交。槽内高位、空系数、NTT尾部仍必须写零；0.299秒不能全部视为可消除收益。融合路径的当前合同和已测范围如下。

### pack/zero 融合路径

独立生产程序默认启用pack/zero融合，配置键为`stage2_pack_zero=true`；设为false或使用`--pack-zero 0`恢复原packer。优先级为CLI、显式TOML、`NTT_S4_PACK_ZERO`环境变量、内置默认值；TOML支持共享设置和Worker覆盖，环境/CLI接受严格0/1。父进程把解析值传给曲线及tune子进程，tune条件记录`env_s4_pack_zero`。完整规则见[TOML参考](../ECM_TOML_REFERENCE.md#stage2_pack_zero)。旧严格.cprof要求关闭融合；未显式覆盖且指定该配置时，驱动保持false，不放宽成本合同。

融合不改变D、NTT长度、分块、carry、设备归约或GMP抽查。显式B2的普通save输入和resident gather共用相同digit表示，reverse-B有独立编译实例。不符合整数索引范围和slot容量的输入回退原packer。底层探针/库入口由显式环境开关控制，生产默认值由驱动读取统一配置后设置。

每个线程负责一个最终digit地址，连续写入A或B；有效digit从原始系数提取，其他地址写零。每对输入通常由两个融合kernel替换两个memset及两个pack kernel。单个launch的slice上限65535，更多slice分段提交。slot内的除法用32-bit reciprocal、multiply-high及一次余数修正，保留sw=1分支；输入最大NTT地址必须可由uint32表示。

两输入最终写入量为16·m·L bytes，少掉的是原先有效digit的重复覆写，近似8·m·(ma+mb)·⌈S/b⌉ bytes；不是少写16·m·L bytes。contiguous路径以padding后的P代替ma/mb。无需额外设备数组或host staging，原owner、metadata、arena借用和回退生命周期保持不变。

`s4_pack_zero_stats`按进程累计：contiguous/gather计操作数批次调用次数（一次可含m个slice），reverse为gather子集；launches为实际融合kernel数，words为最终覆盖的digit数，avoided_memsets为替换掉的独立清零提交数。`replaced_memset_bytes=8·words`只是被替换清零操作的规模，不能当作避免的全部显存写入量，更不是PCIe流量。

独立探针与生产共用[s4_input_pack.cuh](../../src/cuda/stage2/s4_input_pack.cuh)中的原始及融合内核。663项GMP验证完整38,030,604 words，覆盖1–16384 bits、bpw1/7/19/31/62、普通/gather/reverse、空/单/多系数、非零metadata偏移、slot尾、NTT尾、脏目标、两侧guard及65535-slice边界；专用故障仅产生bad1并返回3。另有1,048,576组uint32商余数边界检查，包含sw1–65536、随机值及UINT32_MAX，bad0。

在本页相同GPU/工具链设置下，21个孤立准备形状每侧6样本、每样本8次、不丢弃预热，共252样本；设备事件包括默认memset+pack或完整融合kernel，排除源数据上传和结果检查。已测318/2203/4423/7995/12011 bits，P3/32/1024/5761/77761，batch1/4/64/512，三种输入路径耗时减少24.87%–73.63%。这是单操作数准备时间，不是NTT或完整Stage2收益。sm89探针的ptxas结果为新路径20 registers/thread、原路径38–40，均无spill；并不由寄存器数直接推断加速。

原始证据为`data/experiments/stage2_pack_zero_20261015/`，复现入口见[pack/zero工具](../../tools/bench/README_STAGE2_PACK_ZERO.md)。旧严格cprof成本配置要求该开关为0，不能直接把旧配置当成融合路径的已测成本。

### 固定D完整曲线结果

同一生产二进制SHA256=`b4ca90b0542c2081cc992060574d41c3c60e2fc99b68bb6e1b784d5591cba9d0`，探针SHA256=`d728876ed737e0bb4395e28da4280b33d30c2533fca30e1cb5b5afc7df947688`。GPU/驱动/CUDA/GMP条件同本页，保持55W/1800MHz上限，不调整设备设置。10个save、B2、固定D和预算沿用[outer矩阵](NTT_OUTER.md#固定-d-完整曲线)，B1=20、sigma26；Graph、MID_COOP、JOINT、短batch选型及短校准关闭，只改变PACK_ZERO=0/1。每侧4样本、交错顺序、没有丢弃预热，总计80条正式曲线。

各格引擎init+main中位数（off→on）为：

- M521、batch64：0.138931→0.136033秒，少2.09%。
- M2203、batch64：0.597498→0.573956秒，少3.94%。
- M4423、batch64：1.398738→1.338799秒，少4.29%；batch256：1.377218→1.349973秒，少1.98%。
- 7995-bit普通余因子、batch256：5.424563→5.361173秒，少1.17%；同余因子承载8011：3.777614→3.727039秒，少1.34%。
- M4423、host fold：1.547628→1.521478秒，少1.69%。
- M4423、B2=2.6×10¹⁰：6.726268→6.517520秒，少3.10%；7995-bit普通余因子、相同B2：16.065545→15.785060秒，少1.75%。
- M2203、P155520、B2=2.6×10¹²：19.680968→19.193714秒，少2.48%。

各格中位数之和56.734972→55.504744秒，少2.17%；不是某一生产任务时长。计时边界见本页前文；进程墙钟另保留在CSV中。GMP、自检、因子/rootfold、NTT长度分布、S4分块与检查数量均一致。各格NTT模块full_peak均相同，包括最大P项约3190.44 MiB；不是全进程显存峰，融合路径也没有新增设备临时数组。

![准备阶段与完整曲线耗时减少](../../data/figures/stage2_pack_zero_20261015.png)

图上半为单操作数准备时间，下半为完整曲线；两者不能相加。下半误差线为两侧MAD经比值传播后的离散尺度，不是置信区间。原始样本在`ab/`，孤立准备在`probe/`，汇总为`data/benchmarks/stage2_pack_zero_20261015/`；独立smoke、补充输入、诊断及剖析不加入这80条样本。

M4423/batch64的独立6次/侧复测为1.381022→1.337825秒（3.13%），两侧MAD为0.003554/0.006237秒。它支持该短曲线存在收益，但不把首轮4.29%视为固定增益；单对最终smoke曾仅少约0.16%。`recheck_short/`、smoke及正式矩阵全部保留，复测不覆盖正式样本。

正式矩阵GPU利用率≥80%的740个采样点中，SM1515–1800 MHz、中位1800；功率15.60–54.73 W、中位48.45。频率实际变化，不能称为恒频/恒功耗测量。原始GPU采样保留，不用空闲时的异常功率读数解释硬件限制。

另18条补充曲线、每侧3样本：M2203、B1=20、sigma26、B2=2.6×10⁸、D30030、batch16、arena1/fold0 MiB为0.756954→0.743536秒（1.77%）；M12011真实save的11694-bit余因子，B1=44000000、sigma606255614715327、B2=2.6×10⁸、D30030、batch64，普通为12.889000→12.825073秒（0.50%），承载12011为7.629668→7.556426秒（0.96%），后两项arena6300/fold640 MiB。此组与主矩阵分开保存为`extended/`。

高位宽收益受其他阶段限制：上述普通余因子的一个on样本中，giant为8.360/12.844秒，约65.1%，而G树0.616秒、fold0.134秒、下降0.535秒；这些阶段计时不再与完整total相加。该输入应优先研究点生成，不应由孤立packer的约50%收益推断完整曲线也改善约50%。

另8条普通/承载/大P完整叶子诊断一致。B/Q复用加原生flat输入和workspace fixture通过：flat为16项/16,800 words，workspace为144检查/260 words，覆盖两/三缓冲、别名、失败回滚及拥有/借用回退；新增packer仍通过完整曲线GMP和叶子指纹。carry计数器故障被检出并使生产队列返回曲线失败，不能因融合而擦除诊断结果。诊断和故障均不进入性能统计。

### 两卡SM1500MHz扩大验证

设备为上述4060LP和GPU0 RTX4070 Ti、12282MiB，UUID `d9139abeea58ffc8b97d2066e52c9702`；驱动/工具链相同，GMP DLL SHA256为`969f0e96ac587dedfecf611ca3442fc7211f9218f1a553214c83548dcdbcfd6e`。用户锁定两卡SM1500MHz、暂停GPU0外部计算，逐卡测试；工具未改时钟、功耗。正式高负载采样中SM均为1500MHz。显存未独立锁定：4060LP采样7001–8001MHz、中位8001，4070 Ti为810–10251MHz、中位10251；采样含状态过渡，不能据此声称每个kernel的显存频率恒定。

性能对照均用上述b4ca90b0二进制，仅切换PACK_ZERO，同一save、D/自动D、预算、检查和NTT工作量；交错次序，不丢弃预热。B1=20，sigma26，M2003使用27；Stage1不计时。新默认版SHA256为`da6bf9bcc893151ae7de5a9dcf2cc406df1eb7f2ce66026e7e2dbcc6f5f85e4a`，复用完全相同的CUDA对象SHA256=`87c49973a28b5d0f759fb31226da8d0b4691308abd34ff872a17a95e284edc64`，变化在TOML/CLI与子进程配置。构建器逐项核对CUDA依赖及对象哈希，另通过14项配置检查、普通/承载配对tune和4条新二进制曲线冒烟。

| 正式矩阵 | 格数；每侧样本 | 引擎中位数之和 off→on，秒 | 减少 | 进程墙钟中位数之和减少 |
|---|---:|---:|---:|---:|
| 4060LP，318–7995 bits，三档B2 | 27；3 | 115.485890→112.601388 | 2.50% | 1.89% |
| 4060LP，10232/12285/14328/16272 bits，普通/承载 | 16；2 | 199.930811→197.946799 | 0.99% | 0.73% |
| 4060LP，功能/预算/自动D组合 | 17；2 | 66.047362→64.189795 | 2.81% | 2.48% |
| 4070 Ti，位宽/B2及适用组合 | 26；3 | 198.273808→191.375170 | 3.48% | 1.90% |

共86格、450条正式曲线。主矩阵三档B2为2.6×10⁸、2.6×10⁹、2.6×10¹⁰，固定D分别30030/60060/120120；高位宽使用B2=2.6×10⁷/10⁸、D30030，承载指数10267/12511/14401/16319。组合包含Graph、中段M6、joint tile/outer、B/Q复用、关闭workspace池、arena1/fold0、16/64/256MiB分块、自动D，以及M2203、B2=2.6×10¹²、D1711710/P155520。常规预算为arena6300/fold640MiB。M6/joint表仅在已测4060LP适用，GPU0使用其设备支持的路径。

四矩阵的总计各自是格中位数之和，不是某条任务或跨设备总时长。NTT分布、分块、自检、GMP抽查、S4 launch与因子/rootfold均一致；各格NTT模块报告的full_peak相同，新packer不新增设备数据数组。该字段不是进程峰值，也不能据pool关闭时的模块字段反推全部临时分配。数学变换覆盖k7–27；两卡另通过完整叶子诊断共16条曲线。

![两卡位宽与B2的收益](../../data/figures/stage2_pack_zero_expanded_1500_20261015_heatmap.png)

图格为100·(1−Ton/Toff)，正值更快，空白为未测组合；纵轴是实际余因子N位数，carrier面板仍以目标N标轴，实际算术位宽为承载指数。图只显示规则位宽/B2格，自动D、预算及组合见完整[条图](../../data/figures/stage2_pack_zero_expanded_1500_20261015.png)和汇总CSV。GPU0较大的正值也含分配等待抖动，不将18%之类上界称为稳定算法收益。

五个负收益格均各侧追加6次，不替换原样本；GPU0复测关闭本工具的周期采样，沿用用户锁频，保留未连续采样的条件差别：

| 输入 | 正式减少率 | 独立复测减少率 |
|---|---:|---:|
| GPU1，1939-bit，B2=2.6×10⁸ | −0.80% | +1.36% |
| GPU0，7995-bit，B2=2.6×10⁸ | −0.27% | +2.44% |
| GPU0，12285-bit，B2=2.6×10⁷ | −0.98% | +1.89% |
| GPU0，3600-bit，B2=2.6×10⁹，自动D | −12.76% | +17.25% |
| GPU0，7995-bit/承载8011，B2=2.6×10⁹，自动D | −3.94% | +3.06% |

复测共96条，还包含正向异常项：GPU1的383-bit中档B2由+8.15%复测为+3.89%，GPU0的2726-bit短B2由+18.21%复测为+2.15%。GPU0自动D的17.25%同样带较大离散，其off/on MAD约0.254/0.074秒；这些数字用于核对回退，不作稳定加速承诺。关闭本工具采样后仍有等待，不能将波动归因于该采样或SM频率一个因素。复测数据与图保存在独立汇总目录。

两卡各自通过663项/38,030,604 words独立GMP、1,048,576商余数及bad1/退出3的故障检查；SM锁频条件下各21准备形状、3样本/侧，减少范围分别为4060LP 25.98%–72.65%、4070 Ti 29.32%–84.44%。GPU0另外9个2726/7995-bit精确形状、6样本/侧减少46.50%–72.19%，完整数组与guard一致。这些是准备操作收益，不是完整Stage2收益。

当前默认资格依据为四矩阵总计均下降、五项回退复测通过、完整算术/叶子诊断及准备探针通过。原始证据位于`data/experiments/stage2_pack_zero_expand_20261015/`；正式汇总为`data/benchmarks/stage2_pack_zero_expand_20261015_matrix/`，复测为同前缀`_rechecks/`。GPU0的2595MHz筛查单独保留在`gpu0_matrix/`，不加入450条正式统计。其他发布架构和驱动尚无本轮设备验收。

### GPU0分配等待的管理员定位

Systems2026.1.3以管理员各采一条M3001、2726-bit余因子、B2=2.6×10⁸、D30030、batch64曲线，GPU0锁SM1500MHz，输出/工作量与参考一致，剖析不加入正式计时：

- memset381→141次，bytes为1,685,109,664→277,408，少240次A/B清零，与packer统计一致；诊断清零保留。
- pack kernel累计9.433→2.205ms，连同全部memset为11.726→2.386ms，准备GPU时间少79.65%；H2D/D2H/设备内复制的次数和bytes相同。
- 本进程GPU事件并集0.545392→0.540868秒；窗口内无自身GPU事件的时间0.293301→0.823348秒。该对引擎却为0.820918→1.348964秒，不能据剖析墙钟评价算法回退。
- 互斥空隙归属中，cudaHostAlloc为0.112128→0.398494秒，cudaMalloc为0.004696→0.261417秒，增加约0.543秒；总额外空隙约0.530秒，其他分区略降。两种API全程调用次数分别保持35和283，不是融合新增分配。最大单空隙约397.784ms，约396.346ms落在host.pinned_reserve的cudaHostAlloc内。

这些事件定位了该对样本的等待位置；底层驱动/系统延迟成因仍未证实。`cuLibraryGetKernel`落在空隙内的时间很小，不以CUDA lazy loading解释这些大等待。优先研究pinned缓冲增长/生命周期和设备准备数组复用，保留失败回退与检查；GPU0完整曲线的离散不能用SM锁频或kernel耗时单独解释。依据为`gpu0_nsys_off/`、`gpu0_nsys_on/`及`data/benchmarks/stage2_pack_zero_expand_20261015_profiles/gaps_off.json`、`gaps_on.json`。

### GPU1的1800MHz上限管理员剖析

Systems2026.1.3以管理员各采一条相同M4423、B2=2.6×10¹⁰、D60060、batch256曲线，开启全API与CPU范围，算术、工作量和输入身份通过采集器核对。使用上述同一b4ca90b0二进制，所有剖析秒数排除正式A/B：

- GPU memset：4391→1527次；84,325,430,832→7,206,448 bytes；0.299407→0.001507秒。少掉2864次清零提交，与融合统计的2864个操作数批次完全一致，余下诊断等清零保持。
- 原pack kernel累计0.255337秒，新pack/zero累计0.305582秒。新kernel还承担补零，不能单看这一项认为变慢；两边连同全部memset比较为0.554745→0.307089秒，少44.64%。
- 本进程GPU事件窗口6.823283→6.611632秒，空隙0.631112→0.605956秒，即9.25%→9.16%。API外空隙0.266244→0.259290秒，CPU准备问题仍存在，不能宣称已经消除。
- H2D/D2H事件数量和bytes相同，累计分别约13.6/12.1–12.3 ms；收益没有来自减少PCIe数据量。

Compute2026.2.1以管理员采集新路径的一个热点gather实例，核对Systems的mangled kernel及第61次调用、grid32768×1、CTA256、20 registers/thread、动态shared0。16次replay，clock/cache control均为none，没有修改GPU控制；指标为SM约1799 MHz、单次kernel308.288 μs、DRAM吞吐占峰值96.04%、SM吞吐18.69%、活跃warp占峰值74.67%、local spilling0。原始导出中DRAM字节总吞吐约214.93 GB/s。这是单实例replay诊断，不是生产全程保证；SM吞吐较低不等于GPU空闲，当前完整写出已接近该实例的DRAM上限。

融合侧outer/tile累计3.517414秒，占全部GPU kernel累计5.976605秒的约58.9%；该分母是kernel事件时间，不是GPU窗口或引擎墙钟。M4423长B2下应继续研究NTT遍历与tile/outer组织；11694-bit短B2样本则优先研究giant点生成。不能据一个输入指定全位宽的唯一瓶颈。

证据：`nsys_off/`、`nsys_on/`、`ncu_pack/`，互斥空隙为汇总目录`gaps_off.json`、`gaps_on.json`，组件为`profile_comparison.json`。这组上限频率的剖析不与SM1500MHz矩阵混合；当前默认与扩大范围见上文，其他发布架构仍待设备验收。

## 当前实现与下一步

`NTT_HOST_TRACE=1` 默认关闭，启用CUDA附带NVTX3的CPU范围标记；不改变算术检查数量或日志。没有NVTX3头文件的构建在请求该功能时明确拒绝。正式矩阵不启用此开关。

优先级：

1. 扩大其他发布架构/驱动及tune自动选型全矩阵；两卡SM89的当前默认资格不代表全部硬件已验收。继续核对预算逼近边界及真实分配失败。
2. NTT继续测试tile11/CTA256与outer联合组织，以及中段M6与pack融合的组合；已有联合扫描的tile均为CTA512，没有CTA256收益证据。
3. GPU0优先减少pinned/device反复分配，GPU1继续研究高位宽点生成及生产高B2准备/同步；不同设备的空隙比例差别很大，约9%不是通用上限。
4. 评估同N连续曲线的初始化/计划复用；需要完整的模数/表示身份、状态重置和验证合同，不减少现有必需检查。

## 复现与代码依据

- [工具入口](../../tools/bench/README_NTT_OUTER_PLANS.md)：联合扫描、完整曲线、管理员范围追踪与空隙分解。
- 原始证据：`data/experiments/stage2_tile_sync_20261015/`，其中`nsys_all/`为常规全API追踪，`nsys_nvtx_off/`、`nsys_nvtx_on/`、`nsys_nvtx_full/`为CPU范围追踪。
- 汇总：`data/benchmarks/stage2_tile_sync_20261015/gaps_nvtx_full.json`；剖析和正式性能使用独立的构建身份。
- [ntt_host_trace.cuh](../../src/cuda/stage2/ntt_host_trace.cuh)：默认关闭的范围标记；[ntt_runtime.cuh](../../src/cuda/stage2/ntt_runtime.cuh)中的`ntt_shape_plan`、`ntt_run_passes`、`ntt_poly_mul_batch_dev`为NTT范围入口。
- [s4.cpp](../../src/cuda/stage2/s4.cpp)：`s4_reduce_init`；[checks.cpp](../../src/cuda/stage2/checks.cpp)：`s4_compare_snapshot`；[product.cpp](../../src/cuda/stage2/product.cpp)：`poly_mul_batch_modN`、`s4_gather_final_input`、`s4_pack_final_input`；[tree.cpp](../../src/cuda/stage2/tree.cpp)：`build_tree_flat`；[points.cpp](../../src/cuda/stage2/points.cpp)：`giant_chunk_chain`。
- [pack.cuh](../../kernels/cuda/stage2/pack.cuh)：商余数修正及完整写入；[product.cpp](../../src/cuda/stage2/product.cpp)：`stage2_pack::launch` 分段提交，gather和contiguous接入保留形状校验及不支持时的默认路径。
- [配置解析:418](../../src/core/ecm_cuda_stage2_main.cpp#L418)、[子进程传递:569](../../src/core/ecm_cuda_stage2_main.cpp#L569)；默认值/双语说明源为[ecm_options.json:1109](../../config/ecm_options.json#L1109)，经生成器同步TOML及参考。GPU0分配范围入口为[s4.cpp](../../src/cuda/stage2/s4.cpp) 的 `pin_words`。
- 同文件`stage2_init_begin`、`stage2_init_seconds`及`run_batched`返回处定义引擎计时；[ecm_stage2_phase_times.h](../../src/core/ecm_stage2_phase_times.h)定义互斥阶段，不能将旧嵌套timer重新相加。
