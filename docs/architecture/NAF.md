# Edwards Stage1 的窗口 NAF

NAF 是 **Non-Adjacent Form**（非相邻形式）。窗口 NAF 把标量的二进制表示重编码为带符号的奇数数字，使大部分列只做倍点。CPU GMP、CPU IFMA 和 CUDA CGBN 均可在 Edwards 点坐标上运行该算法。Edwards CUDA NTT 当前使用映射后的 Montgomery x-only 阶梯，它以相差固定基点的一对点推进。通用 NAF 字典不能提供差分加法需要的动态差点。原生 Edwards NAF 的 NTT 方案已有整数参考探针，尚无生产设备实现。见 [NTT 的原生 Edwards NAF 参考数据流](../performance/STAGE1_MERSENNE_NTT.md#原生-edwards-naf-的参考数据流)。

## 从标量到点乘

ECM-ELF 对 Edwards 构造同一个整数标量 $s=48\operatorname{lcm}(1,\ldots,B_1)$。实际进 Stage1 的模数 $N$、曲线参数 $d$ 和 sigma 决定基点及点算术，标量和编码可在不同曲线间共用。若 $L=\operatorname{bitlen}(s)$，则 $L\approx B_1/\ln 2$。精确位数由实际 LCM 决定。见 [共用标量入口](../../src/ecm/stage1/task.cpp#L20)、[CPU Edwards 构造标量](../../src/ecm/stage1/edwards/cpu/workflow.cpp#L918) 和 [CUDA Edwards 构造标量](../../src/ecm/stage1/edwards/cuda/workflow.cpp#L207)。

给定宽度 $w$，非零数字 $d_i$ 为满足 $|d_i|<2^{w-1}$ 的奇数，标量值为 $s=\sum_i d_i2^i$。标准窗口编码从低位处理。当前剩余值为奇数时，取模 $2^w$ 的有符号代表，减掉该数字，再除以 2。这个代表的选择使接下来的 $w-1$ 列为零。实际执行从高列向低列扫描，初始点为恒等点，每列先做 $R\gets[2]R$，若 $d_i\ne0$ 再做 $R\gets R+[d_i]P$。例如 $31=2^5-1$，w3 的从高到低数字为 `1,0,0,0,0,-1`，依次得到 $P,2P,4P,8P,16P,31P$。字典只存正奇数倍点，负点在使用时按 Edwards 坐标规则取反。见 [点和坐标公式](../reference/ECM_EDWARDS_MATH.md#3-naf-标量乘法)、[CUDA 编码](../../src/ecm/stage1/scalar.cpp#L394) 和 [CPU 编码](../../src/ecm/stage1/edwards/cpu/ecm_edwards_cpu.cpp#L214)。

最高位进位或高位合并可能改变实际列数 $L_w$。CUDA 使用居中的低位窗口编码。CPU 按源自 Prime95 的逐位构造处理前缀和进位。CPU 稠密与打包表示保留相同的 CPU 数字顺序。记录过的列号始终要和生成它的后端、算法及窗口配套；跨实现验证比较归一化末点和真因子。见 [NAF 单元测试](../../tools/test/edwards_naf_test.cpp) 和 [CPU 与 CUDA 对拍](../../tools/test/test_edwards_naf_memory.py)。

## 点字典与执行成本

宽度 $w$ 需要 $m=2^{w-2}$ 项 $P,3P,\ldots,(2^{w-1}-1)P$。先倍点生成 $2P$，再反复加 $2P$ 生成奇数倍。表索引为 $(|d_i|-1)/2$。各后端的点表和运算边界如下。

| 后端 | 字典布局 | 基本倍点 | 非零列额外成本 |
| --- | --- | ---: | ---: |
| CPU GMP 定长／IFMA | 仿射 $(x,y,dxy)$ | $4S+3M$ | 约 $8M$ |
| CUDA CGBN HWCD | 投影 $(X,Y,Z,dT)$ | $4S+3M$ | $10M$ |
| CUDA CGBN 2XY | 同一投影字典 | $3S+4M$ | $10M$ |

$S$ 和 $M$ 分别代表一次模平方和模乘，表中未计模加减、坐标搬运及设备调度。HWCD 通过平方求 $2XY$。2XY 直接相乘，省掉一次平方及部分加减，多做一次模乘。CPU 和 GPU 在非零列为加点补算倍点的扩展坐标 $T$。CPU 仿射加点通常跳过输出 $T$，收尾或诊断需要完整点时另计。CGBN 每个非零列含补算 $T$ 的一次模乘和投影加点的九次模乘。切片末尾为零列时，CGBN 另补一次 $T$，供写回和快照保持 $XY=ZT$。见 [CPU GMP 点算术](../../src/ecm/stage1/edwards/cpu/ecm_edwards_cpu.cpp#L323)、[IFMA 主循环](../../src/ecm/stage1/edwards/cpu/simd_edwards.cpp#L908)、[CGBN 倍点](../../src/ecm/stage1/edwards/cuda/naf.cuh#L38) 和 [缓存加点](../../src/ecm/stage1/edwards/cuda/point.cuh#L70)。

两条 CUDA 路径只在主循环中求 $E$ 时分支。HWCD 的 $(X+Y)^2-X^2-Y^2$ 在已有 $X^2$、$Y^2$ 的基础上执行一次模加、一次模平方和两次模减。2XY 的 $2(XY)$ 执行一次模乘和一次模加。默认 CGBN 模平方和模乘调用同一个 Montgomery 乘法核心，只有平方的两个输入相同。2XY 在源码层面省去两次模减，实际收益还受编译后的关键路径和寄存器分配影响。建字典时两条路径都用 HWCD。见 [CGBN Montgomery 算术](../../third_party/cgbn/include/cgbn/impl_cuda.cu#L1027) 和 [两卡配对基准](../performance/STAGE1.md#cuda-edwards-倍点-hwcd-与-2xy)。

`[stage1] edwards_doubling` 配置新 CUDA CGBN Edwards NAF 批次。0 为 HWCD，日志显示 `doubling=4`。1 为 2XY（默认），日志显示 `doubling=5`。TOML 省略此项时，旧环境变量 `ECM_EDWARDS_DBL_ALGO=4|5` 仍可选用原内部编号。TOML 设置只用于新批次，断点续跑读取已存编号。显式设置旧环境变量时仍会检查它与存档是否匹配。CPU、CUDA NTT 和 Edwards binary 路径不使用此 TOML 键。见 [配置参考](../ECM_TOML_REFERENCE.zh-CN.md#edwards_doubling) 和 [运行时选择](../../src/ecm/stage1/edwards/cuda/workflow.cpp#L214)。

CPU 建表做批量模逆以得到仿射项，GMP 每条曲线、IFMA 每个 lane 各处理自身分母，非单位分母需检查因子或走已有回退路径。CGBN 的字典留在投影坐标，预存 $dT$ 使每次加点省掉一乘。对 $m>1$，CGBN 建表中的点算术为一次完整倍点、一次 $2P$ 的 $dT$ 预乘、$m-1$ 次缓存加点及 $m$ 次写入字典前的预乘，合计 $4S+(10m-4)M$，另计 Montgomery 域转换。$w=2$ 只需基点一项，准备核跳过 $2P$ 的构造。见 [CUDA 字典核](../../src/ecm/stage1/edwards/cuda/naf.cuh#L6) 和 [CPU IFMA 建表](../../src/ecm/stage1/edwards/cpu/simd_edwards.cpp#L767)。

设 $A_w$ 为非零列数。对长标量，$A_w\approx L/(w+1)$。CGBN HWCD 循环约为 $4L_wS+(3L_w+10A_w)M$，加上上述片末修正。CPU 对应的乘法项约为 $3L_w+8A_w$。宽度每加一位，预计少做 $L/((w+1)(w+2))$ 次加点，字典多出 $2^{w-2}$ 项。配置里的 $w$ 控制编码宽度和表容量，不做 PRAC 那样的候选链搜索。见 [PRAC 搜索机制](PRAC.md#搜索强度与重现性)。

## Prime95 对照

本地比较采用 Prime95 v31.06 build 1 的 `ecm.cpp`。其 Edwards Stage1 可拆成若干由 `ecm_calc_exp` 形成的指数块，每块先估字典大小，最高位的第一个奇数倍从字典直接读出作为累加点。编码时把非零位标记写回指数缓冲区，另分配约 $\lceil L/8\rceil$ 字节保存索引和符号。字典项数 $m$ 可为任意有效整数。ECM-ELF 对完整 $s$ 编码，从恒等点逐列扫描，以 $w$ 决定 $m$。两套实现的工作游标和字典配置需分别解释。见 [Prime95 编码](../../.refactor/p95v3106b01.source/ecm.cpp#L4803) 和 [指数分块](../../.refactor/p95v3106b01.source/ecm.cpp#L7546)。

Prime95 估算随字典项数变化的 GWNUM 变换数

$$
F(m)=a\,m+\frac{bL}{\log_2 m+3}.
$$

普通 Edwards 用 $(a,b)=(27,14)$，twisted Edwards 用 $(29,16)$。程序先寻找模型下的较优项数，再按 `DictionaryMemory` 默认 256 MiB 限制到 $\lfloor\mathrm{bytes}/(3\,\mathrm{gwnum\_size})\rfloor$。`DictionarySize` 可覆盖所选项数，源码最后夹到至少 7 项。已进入该指数块的存档沿用原字典项数。ECM-ELF 的 `naf_memory_mb` 用十进制 MB，覆盖该后端的计算缓冲预算。两者的设置单位和计量范围不同。见 [成本函数](../../.refactor/p95v3106b01.source/ecm.cpp#L4887)、[内存配置](../../.refactor/p95v3106b01.source/ecm.cpp#L7225) 和 [字典选择](../../.refactor/p95v3106b01.source/ecm.cpp#L7558)。

## 内存、复用与配置

标量的 GMP 对象、NAF 数字编码、奇数倍点字典是三个不同的分配。指数缓存可复用标量。NAF 编码按实际整数 $B_1$ 和窗口选择，CPU worker 保留最近一个不超过 256 MiB 的只读编码。若 `int16_t` 编码预计超过 64 MiB，优先采用打包表示。同一任务的所有 IFMA 批次、GMP 曲线共享一份编码。CGBN 在主机编码后把两份位图各上传一次，批内全部曲线及切片共同读取。不同 sigma 各有点表。固定模数、曲线、sigma 和窗口时，点表与 $B_1$ 无关，但改变 $B_1$ 会改变整个 LCM 数值，不能取旧 NAF 编码的前缀代用。生产目前没有 NAF 磁盘缓存。见 [CPU 编码及缓存](../../src/ecm/stage1/edwards/cpu/ecm_edwards_cpu.cpp#L194) 和 [CUDA 设备分配与上传](../../src/ecm/stage1/edwards/cuda/edwards_runtime.inl#L119)。

以下均为缓冲区逻辑字节数。设 $C$ 为曲线数，$W$ 为 CUDA 容器的字节数，$n_{52}=\lceil\operatorname{bitlen}(N)/52\rceil$。

| 对象 | 字节数 |
| --- | ---: |
| CPU 稠密编码，每任务一份 | $2(L+1)$ |
| CPU 打包编码，每任务一份 | $8(\lfloor L/32\rfloor+2)$ |
| CUDA 打包编码，主机和设备各一份 | $8(\lfloor(L+32)/32\rfloor+1)$ |
| CUDA 主机编码期间的标量导出缓冲 | $4\lceil L/32\rceil$ |
| CUDA 字典，每批 | $4CWm$ |
| IFMA 持久字典，每个八路 worker 批次 | $3\cdot8\cdot8\cdot n_{52}m$ |
| GMP 定长投影和仿射两表，同时持有 | $7\cdot(10240/8)m$ |

实际 GPU 预算还要计入点状态和快照缓冲。CGBN NAF 的基础设备缓冲为 $CW(14+4m)+8(\lfloor(L+32)/32\rfloor+1)+32$ B，最后的 32 B 为 CGBN 报告；CUDA 上下文、并行切片元数据和分配器消耗由空闲显存与预留覆盖。CPU 预检另为进程内其他批次状态、线程、模逆和域算术工作区留出预算。见 [设备预估](../../src/ecm/stage1/edwards/cuda/memory.h#L10) 与 [CPU 工作区预估](../../src/ecm/stage1/edwards/cpu/workflow.cpp#L790)。

`edwards_naf_window_bits = 0` 在有 NAF 的 CPU 路径从 w12 开始，在 CUDA CGBN 从 w6 开始。生产显式窗口为 CPU 3–12、CUDA CGBN 2–12。GMP 定长上限为 10240 位，显式标量宽位路径使用 GMP `mpz` 二进制点乘。`naf_memory_mb` 最小正值为 1 MB，0 表示使用空闲预算。CPU 对主内存保留 `max(20%, 64 MiB)`，CUDA 对设备空闲显存保留 `max(20%, 16 MiB)`，再与配置上限取较小值。CPU 尝试在同一窗口下使用可承受的表示和 worker 数，至少一个 worker 都放不下才缩窗。CUDA 固定曲线数，从请求的窗口向下选能装下的最大窗口，自动算法在必要时可切换到 binary。CPU 标量构造和 CUDA 主机准备另查可用内存。配置与作用域见 [TOML 参考](../ECM_TOML_REFERENCE.zh-CN.md#naf_memory_mb)。

## 检查点与 Stage2 交接

检查点记录的是点状态和当前编码的游标。CPU 保存已处理的 NAF 列数和实际字典项数。CUDA 保存实际窗口、进度、倍点算法、sigma 和普通整数点状态。显式请求因内存缩窗时，CUDA V4 还保存原请求。续跑按记录中的实际窗口重建所需编码和字典。当前预算若容纳不了该窗口，任务和检查点留待重试。快照在安全边界导出。见 [检查点格式与恢复](CHECKPOINT.md#42-gpu-edwards-v3v4) 和 [CUDA Edwards 运行时](STAGE1.md#gpu-edwards)。

最终 Edwards 坐标 $(X:Y:Z)$ 映射到 Montgomery $(Z+Y):(Z-Y)$。$\gcd(Z-Y,N)$ 为 1 时写归一化 Stage2 点，得到真因子时单独记录。GCD 等于 $N$ 的恒等结果可完成曲线但无可交给 Stage2 的点。窗口及编码表示属于 Stage1 执行参数，完成的 save 仍按曲线、sigma、模数、$B_1$ 和点或因子识别。见 [Edwards save](SAVE.md#5-edwards-save-version1)。

## 常用 B₁ 的实际列数与编码时间

在 Ryzen AI 9 HX 370 的逻辑 CPU 1 上直接调用 0.63.4 源码中的 CUDA 主机编码器，MSVC 14.51 `/O2`。每个 $B_1$ 和窗口交错三次，取中位数。计时从已构造或从指数缓存加载的标量开始，含导出整数、分配和打包，排除指数构造或读取、GPU 上传及点乘。CPU 功耗和实际频率未独立采样。表格各组依次为 w6／w8／w10／w12。

| $B_1$ | 实际 NAF 列数 | 非零列 | 编码 ms | 每份位图 MiB |
| ---: | --- | --- | --- | ---: |
| 11,000,000 | 15,869,677／15,869,678／15,869,672／15,869,674 | 2,266,827／1,763,182／1,442,617／1,220,715 | 31.3／25.1／20.9／18.0 | 3.78 |
| 44,000,000 | 63,477,242／63,477,242／63,477,239／63,477,241 | 9,067,032／7,052,119／5,770,384／4,882,496 | 125.4／98.9／82.9／71.5 | 15.13 |
| 260,000,000 | 375,102,582／375,102,581／375,102,577／375,102,581 | 53,587,269／41,679,644／34,101,823／28,855,470 | 735.3／580.1／521.8／445.2 | 89.43 |

$B_1=260$ 百万时，w6 到 w12 每曲线少 24,731,799 次加点，估计少约 2.47 亿次模乘。若平方和模乘权重相同，HWCD 的主循环从约 31.62 亿降到 29.14 亿次模运算，降幅约 7.8%。原始计时、精确字节数和复现探针见 `data/experiments/naf-architecture-0634/counts.csv` 与同目录的 `probe.cpp`。CPU 稠密／打包对照、点字典计时和模拟磁盘缓存见 [常用 B1 的编码开销](../performance/STAGE1.md#常用-b1-的编码开销)。

## 按整批吞吐选择窗口

同一设备、同一输入和批量下，增宽窗口的收益为主循环减少的时间，减去编码和建表增加的时间。记整批准备时间为 $P_w$、主循环为 $T_w$，忽略相同的收尾时间，则 w12 优于 w10 的条件为

$$
T_{10}(1-T_{12}/T_{10})>P_{12}-P_{10}.
$$

两边同时除以曲线数就是摊销秒／曲线。生产中数百条曲线的实际算术可能持续数小时。此前 8191 位、4060 Laptop 384 条曲线的 w10→w12 主循环降幅约 1.7%。若同一批的循环为 10,000 秒，对应约 170 秒节省的量级。该数字由固定负载短测外推，完整生产任务未复测。w12 的字典约 1.50 GiB，比 w10 多约 1.125 GiB。4070 Ti 的 960 曲线下 w12 字典约 3.75 GiB。内存能保住批量时先试 w12，批量受限时比较 w10／w8 的曲线吞吐。CPU 也可按同一不等式估计，IFMA 的仿射字典准备在宽位下更贵。各设备测量条件、短任务的最佳窗口、单曲线验证及证据见 [Stage1 NAF 窗口短测](../performance/STAGE1.md#edwards-naf-窗口短测) 和 [长任务测量](../performance/STAGE1.md#长任务的窗口选择)。

性能日志中的 `columns`、`additions` 对应完整编码，续跑时并不等于本次执行的列数和加点数。`dictionary_gpu_ms`、`loop_gpu_ms` 是本次 CUDA event 计时，`api_seconds` 已包含设备分配、建表、主循环和回读等工作。比较前先统一边界。需要复现实测矩阵时，使用 [NAF 采集脚本](../../tools/bench/bench_stage1_naf_windows.py)；分配失败、降低预算、缓存命中和降级后续跑的验证可用 [NAF 内存回归](../../tools/test/test_edwards_naf_memory.py)。
