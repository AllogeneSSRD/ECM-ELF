# Stage1 算法与数据

## 曲线与标量

Suyama PARAM0 使用 sigma 构造 Montgomery 曲线和初始 X:Z 点。设 u=σ²−5、v=4σ，起点为 u³:v³；曲线参数通过模逆构造。不可逆分母需在目标 N 上求 GCD，不能略过。

光滑标量 s=∏(素数 ℓ≤B1) ℓ^⌊logℓ B1⌋；`lcm=12` 使用 12s。主机分段筛生成素数，分块累乘，用带身份/校验的缓存复用指数；指数缓存与曲线 checkpoint 分离。

CPU Edwards 路径使用自身曲线构造及扩展坐标公式；CPU Montgomery、CUDA 和 OpenCL 的点表示/算术域不同。只有投影或归一化后的数学点可跨实现对比，不能比较未经解码的字数组。

## 方法选择与输入校验

Stage1 单次运行和队列模式共用五个独立选择项：`curve`（Montgomery / Edwards）、`target`（CPU / GPU）、`impl`（自动 / IFMA SIMD / GMP 标量）、`reduction`（自动 / Montgomery / Mersenne fold）及 `lcm`（素数幂最小公倍积 / 其 12 倍）。默认是 Montgomery 曲线、GPU 执行；Edwards GPU 需要编入该模块的 CUDA 可执行文件，OpenCL 会拒绝该组合。`Auto` 在内部 CPU 实现及 SIMD 归约枚举中均为 0；调用现有 IFMA 接口时显式转换，避免枚举序号与实现常量耦合。INI 和 CLI 均拒绝无效选择值，参数默认值及范围由 [配置定义](../../config/ecm_options.json) 维护。

INI 使用 `curve`、`target`、`impl`、`reduction`、`lcm`；CLI 使用同名的双横线选项。`curve=mont|ed` 与 `target=cpu|gpu` 是不同轴；`impl=auto|simd|scalar` 只用于 CPU，`reduction=auto|montgomery|mersenne-fold` 控制 CPU SIMD 约减，`lcm=1|12` 控制 Montgomery 标量的倍数。Edwards 路径按曲线约定固定使用 48·lcm(1…B1)，因此拒绝 `lcm=12`；GPU 执行不接受非 auto 的 CPU 专用选项，显式归约必须同时选择 `impl=simd`，避免 auto 落到标量后忽略该设置。

GPU 参数化的语法值为 0、2、3，均属于 Montgomery 曲线；可执行范围取决于链接的后端和构建配置。CPU Montgomery 只实现 PARAM0，因此 CPU 单次运行只接受 `-sigma 0:s` 这种带参数化前缀的种子；Edwards 不接受参数化前缀，`--gpu-param` 仅适用于 GPU。启动曲线计算前，驱动查询当前 GPU 后端是否支持该参数化；OpenCL 仅实现 PARAM3，CUDA 的 PARAM2 与 PARAM3 还受是否编入相应内核及折叠构建配置约束。CUDA PARAM0 在曲线构造、检查点及 save 中保留完整 64 位 sigma，批次不得在 2⁶⁴ 处回绕。PARAM2/3 的固定 sigma 及曲线批次须满足 sigma+曲线数≤2³²；不再截取低 32 位继续计算。参数化与 sigma 的冲突、后端不支持项在初始化 GPU 前报错。队列任务的 sigma 不符合要求时停止并保留原 worktodo。

`ECMSTAGE2` 作为 Stage1 输入时，文件名前缀决定该任务的曲线构造，末尾字段决定 B1；覆盖 INI 后再校验后端能力和 sigma 范围。每条任务复制基础选项，防止混合队列之间传播覆盖。Edwards 前缀同时恢复其固定标量约定；执行目标保持 INI 设置。名称、旧 save 历史复制及交接合同见[队列输入](../usage/STAGE1.md#程序与输入)。

这一步统一的是选择项解析与能力检查。CPU 曲线实现、GPU 内核、save 发布及各后端 checkpoint 仍沿用各自的路径。入口见 [ecm_stage1_selection.h](../../src/core/ecm_stage1_selection.h)、[ecm_driver.cpp](../../src/core/ecm_driver.cpp) 和 [ecm_backend.h](../../include/ecm_backend.h)。

## GPU 点乘

Montgomery 默认 ladder 按 s 的比特进行差分加法和倍点，状态由相邻点及固定差点组成。GPU 以切片推进，避免一个超长 kernel 占用整段 B1。每曲线的末点先解码、检查，求 Z 与 N 的 GCD；可逆时归一化 X=X/Z mod N，再输出 save。

### GPU Edwards

[CUDA NAF 批量入口](../../src/cuda/ecm_cuda_edwards_naf.cu)以 CGBN Montgomery 域的 `(X,Y,Z,T)` 执行 `48·lcm(1…B1)`。`naf_w=0` 在 GPU 选择窗口 6，显式允许 2..12；每曲线在 GPU 预计算 2^(w−2) 个奇数倍投影点，不做字典模逆。主机以两份紧凑位图保存非零列和有符号索引，设备按每片最多 256 个 NAF 列推进。设备累加点在算术域跨片保留，只回传四个规范普通坐标供回调及检查点使用。连续倍点省略暂时不用的 T，在加法前及片边界恢复完整 `XY=ZT`。入口与位图定义见 [ecm_edwards_naf.h](../../src/core/ecm_edwards_naf.h)，设备字典与切片见 [cgbn_edwards_naf.cuh](../../kernels/cuda/cgbn_edwards_naf.cuh)。

容器选择要求 N 至少留一位余量，选择本次构建中最小的可用容器。完整网格为128/192/256/384/512（TPI4）、768..2048间隔256（TPI8）、2560..8192间隔512（TPI16）、9216..17408间隔1024（TPI32），共32档，TPB128。Montgomery仍预留六位，不能把两个模型的边界条件互换。`ECM_TIERS`同时筛选Edwards实例；缺档时使用更大可用档，超过本次构建可用上限时报错。日志的 `GPU Edwards geometry` 给出实际容器、TPI及网格，查询入口为 `ecm_cuda_edwards_tier`。

二进制与NAF的实例共享[位宽清单](../../src/cuda/ecm_cuda_edwards_tiers.inc)，核函数和七组实例化文件位于 `kernels/cuda/cgbn_edwards_instances*`，按TPI及位宽范围拆分并行编译；`src/cuda/`保留接口、输入校验、分派和主机驱动。模板算术与切片逻辑见[共用实例实现](../../kernels/cuda/cgbn_edwards_instances.cuh)。字典显存为曲线数×容器字节数×4坐标×2^(w−2)，分配失败会报错并保留未完成队列，不静默改变显式窗口。

主循环默认倍点为 HWCD；`ECM_EDWARDS_DBL_ALGO=4|5` 选择 HWCD（内部编号 4）或 2XY 多项式（编号 5），字典构造使用 HWCD。后者采用 Prime95 算法 5 的代数形式，CGBN 不模拟其 FFT 复合指令；内部编号 4 不代表 Prime95 算法 4。参考版本与运算计数见 [数学说明](../reference/ECM_EDWARDS_MATH.md#21-prime95-倍点算法编号与-fft-条件)。`ECM_EDWARDS_STAGE1_ALGO=binary|naf` 可显式选择 [原二进制路径](../../src/cuda/ecm_cuda_edwards.cu)或 NAF，用于对照；binary 不接受窗口或 NAF 倍点设置。未指定时使用 NAF，恢复时遵从已保存的模式。

[曲线构造](../../src/core/ecm_edwards_curve.h)先在辅助 Weierstrass 曲线上用 Jacobian 坐标计算 sigma 倍点，再构造 Atkin–Morain 参数。模逆失败在实际目标 N 上区分真因子与退化曲线，同时检查 d(1−d) 的单位性和基点方程。末点检查 `XY=ZT` 与 Edwards 方程，用 `(Z+Y):(Z−Y)` 转到 Montgomery，检查分母 GCD 后归一化；真因子独立标记。有效恒等点且 GCD=N 时没有可用 Stage2 点，该曲线完成但不写点记录。

[worker 接入](../../src/core/ecm_edwards_gpu_driver.inl)支持 CLI 和队列、周期及安全停止检查点。当前 GPU Edwards 使用 V3 二进制载荷，保存完整 N、算法、进度、各 sigma 和普通整数状态；指数只保存位数与指纹。恢复重建基点、核对点不变量并采用已保存的模式，显式不匹配选择报错。载荷不绑定 TPI，恢复按 N 重新选择容器。NAF 列数与 binary 位数不能互换。完整字段、兼容读取和恢复资格统一见 [Checkpoint](CHECKPOINT.md#42-gpu-edwards-v3)。

CPU/GPU Edwards 共用完成点文本，可由 CUDA Stage2 和 [CPU Stage2](../usage/EDWARDS_CPU_STAGE2.md) 消费；sigma 不能按 Suyama PARAM0 解释。字段、状态、校验及外部程序能力统一见 [Save](SAVE.md#5-edwards-save-version1)。

Stage1 的 Edwards 日志和因子行携带 `curve_model=ed`，省略 Montgomery PARAM。Stage2 的曲线开始与结果事件传递相同模型身份；GUI 结果文件按该身份保存，旧版未带模型的 Stage2 日志沿用 Suyama PARAM0 解释。具体字段与管道规则见 [GUI 日志与结果](../usage/GUI.md#日志结果与监控)。完整大界限、随机 sigma 恢复和小素数回退的正确性证据见 [验收记录](../development/GPU_EDWARDS_STAGE1_STAGE2_DRAFT.md#2026-10-16大界限随机恢复与-gui-曲线身份验收)。

`arithmetic api_seconds` 覆盖标量编码、设备初始化/分配、字典、切片、回传及进度回调，排除之前的指数/曲线构造和之后的 GCD/归一化/save。`dictionary_gpu_ms` 与 `loop_gpu_ms` 是 CUDA event 区间，可能包含首次模块加载或主机发射间隙，不能当作 profiler 的纯 kernel 时间，也不能再与进程 wall 相加。

GPU Edwards 在 NAF 列或二进制位的片边界更新共用进度输出，GUI 管道可读取百分比。当前32档构建的三曲线族同exe数据见[性能依据](../performance/STAGE1.md#gpu-edwards-细分实例三曲线族对照)，32档CPU末点、旧检查点跨容器恢复及Stage2交接见[细分实例验收](../development/GPU_EDWARDS_FINE_TIERS_DRAFT.md#2026-10-16完整正确性门禁)。冻结七档的binary/NAF、窗口/倍点对照及大界限证据分别见[性能参考](../performance/STAGE1.md#gpu-edwards-初始基准与-naf-对照)和[窗口优化草稿](../development/GPU_EDWARDS_STAGE1_NAF_DRAFT.md)，初始流水线见[开发草稿](../development/GPU_EDWARDS_STAGE1_STAGE2_DRAFT.md)；已测范围不等于全部发布架构或生产 B1/B2 的性能保证。

### Montgomery resident 与 PRAC

resident 路径将点状态保持在 Montgomery 域，减少切片间往返转换。PRAC 将素数幂标量分解为规则链，以更多活跃点和控制流换取较少点运算；主机离线计划可缓存，并以分片保持检查点和进度语义。

PRAC 的 `single-compact` 专用体将 xADD 统一到一个调用点，倍点使用紧凑临时状态；终结规则和非终结规则共同维护同一组点角色。此路径仅在已编译的位宽/TPI/寄存器组合可选，缺失实例须报错。它不是发布默认算法。

## 位宽与资源

Montgomery CUDA 容器需覆盖 N bits+6 个进位预留位。默认 2560…8192-bit 容器使用 TPI16，9216…16384 使用 TPI32；更小容器由 TPI4/8 分派。N=4423 bits 使用 4608/TPI16；N=8191 bits 因预留位需要 9216/TPI32，不能按输入是否超过 8192 判断。

默认 TPB128。每 block 曲线数=TPB/TPI，grid=⌈曲线批量/(TPB/TPI)⌉。寄存器上限不改变 TPB；实际 blocks/SM 还受寄存器分配粒度、shared memory 和线程限制影响，由 CUDA occupancy API 查询。

PRAC 可显式比较替代 TPI、寄存器 cap 和切片目标时长。强制 TPI32 会增加某些容器的 CGBN 内部 padding 与跨 lane 通信，不能仅凭每线程 limb 少就推断更快。

## 计算量与存储

设 C 为曲线批量，K 为容器位数，ℓ=K/32，b=bit_length(s)。ladder 计算量为 C·(b−1) 次组合点步骤；通用基数乘法的主项随 ℓ² 增长。PRAC 应分别统计实际 ADD/DBL 数及规则开销，不能用单一“每素数一次”近似。

Suyama CUDA 主状态每曲线包含 N、a24、差点和两相邻点，共 7ℓ 个 32-bit 字，即 28ℓC bytes。其他曲线族的布局由自身入口定义。指数和计划由批量共享，存档/检查点及主机对象另计。一次上传/读回量应按实际切片路线统计，不能把逻辑字节当作全部 PCIe 流量。

checkpoint 外壳、各后端载荷大小、备份和整批重写开销见 [Checkpoint 存储](CHECKPOINT.md#8-存储写入成本和边界)；完成点文本及追加合并成本见 [Save 成本](SAVE.md#63-成本和体积)。指数重建、缓存、NAF 计划和设备字典另计，不能以磁盘文件大小代替进程内存或显存。

## 正确性与持久化

保存点记录规范化普通 X；checkpoint 则记录足以恢复下一切片的数学状态和布局身份。恢复须检查 N/B1/参数化/指数模式和实际 TPI，不能让 checkpoint 的状态域与当前内核失配。

Stage1 各后端共用二进制批次外壳和原子轮换写入，载荷保留各自数学表示及版本；不同层的校验算法和覆盖范围不同。格式、身份、锁、恢复、旧档兼容及完成清理顺序统一见 [Checkpoint](CHECKPOINT.md)，save 的发布和去重见 [Save](SAVE.md#6-写入去重与发布)。操作说明见 [Stage1 使用](../usage/STAGE1.md#检查点与交接)。

跨实现验证应覆盖生成点、分片边界、最终投影/仿射点、有效因子及文件语义。选择不同寄存器预算不改变数学状态；改变 TPI 的恢复资格按当前格式检查。checksum 不替代 CPU/GMP 独立点验证。

## 代码入口

- [ecm_stage1_exp.cpp](../../src/core/ecm_stage1_exp.cpp)：指数生成和缓存。
- [ecm_stage1_checkpoint_runtime.cpp](../../src/core/ecm_stage1_checkpoint_runtime.cpp)、[ecm_stage1_binary.h](../../src/core/ecm_stage1_binary.h)：二进制外壳、长度边界、CRC32C、锁及轮换恢复。
- [ecm_stage1_legacy_gpu.h](../../src/core/ecm_stage1_legacy_gpu.h)：旧 CUDA ladder V5 导入及逐 lane 模数校验。
- [ecm_edwards_gpu_checkpoint.h](../../src/core/ecm_edwards_gpu_checkpoint.h)：GPU Edwards 二进制/旧文本载荷及状态验证。
- [cgbn_stage1.cu](../../kernels/cuda/cgbn_stage1.cu)：曲线布局、归一化、调度及保存点。
- [cgbn_stage1_kernel.h](../../kernels/cuda/cgbn_stage1_kernel.h#L1163)：TPI 容器分档和 ladder。
- [cgbn_stage1_prac_host.cuh](../../kernels/cuda/cgbn_stage1_prac_host.cuh)：计划、策略、切片及恢复。
- [cgbn_stage1_prac_single_add.cuh](../../kernels/cuda/cgbn_stage1_prac_single_add.cuh)：专用链规则。
- [cgbn_edwards_point.cuh](../../kernels/cuda/cgbn_edwards_point.cuh)：GPU Edwards 扩展坐标算术，由批量 worker 与独立探针共用。
- [ecm_mont_cpu.cpp](../../src/cpu/ecm_mont_cpu.cpp)、[ecm_edwards_cpu.cpp](../../src/cpu/ecm_edwards_cpu.cpp)：CPU 实现。

使用和已测最佳范围见 [Stage1 使用](../usage/STAGE1.md)、[性能](../performance/STAGE1.md)。

完整数学见 [Montgomery / Suyama / PRAC](../reference/ECM_MONTGOMERY_MATH.md)、[Edwards](../reference/ECM_EDWARDS_MATH.md) 和[参数化](../reference/ECM_PARAMETERIZATIONS.md)。
