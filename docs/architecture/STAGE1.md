# Stage1 算法与数据

## 曲线与标量

Suyama PARAM0 使用 sigma 构造 Montgomery 曲线和初始 X:Z 点。设 u=σ²−5、v=4σ，起点为 u³:v³；曲线参数通过模逆构造。不可逆分母需在目标 N 上求 GCD，不能略过。

光滑标量 s=∏(素数 ℓ≤B1) ℓ^⌊logℓ B1⌋；`lcm=12` 使用 12s。主机分段筛生成素数，分块累乘，用带身份/校验的缓存复用指数；指数缓存与曲线 checkpoint 分离。

CPU Edwards 路径使用自身曲线构造及扩展坐标公式；CPU Montgomery、CUDA 和 OpenCL 的点表示/算术域不同。只有投影或归一化后的数学点可跨实现对比，不能比较未经解码的字数组。

## 方法选择与输入校验

Stage1 单次运行和队列模式共用五个独立选择项：`curve`（Montgomery / Edwards）、`target`（CPU / GPU）、`impl`（自动 / IFMA SIMD / GMP 标量）、`reduction`（自动 / Montgomery / Mersenne fold）及 `lcm`（素数幂最小公倍积 / 其 12 倍）。默认是 Montgomery 曲线、GPU 执行；GPU Edwards 尚无实现，组合校验会拒绝它。`Auto` 在内部 CPU 实现及 SIMD 归约枚举中均为 0；调用现有 IFMA 接口时显式转换，避免枚举序号与实现常量耦合。INI 和 CLI 均拒绝无效选择值，参数默认值及范围由 [配置定义](../../config/ecm_options.json) 维护。

INI 使用 `curve`、`target`、`impl`、`reduction`、`lcm`；CLI 使用同名的双横线选项。`curve=mont|ed` 与 `target=cpu|gpu` 是不同轴；`impl=auto|simd|scalar` 只用于 CPU，`reduction=auto|montgomery|mersenne-fold` 控制 CPU SIMD 约减，`lcm=1|12` 控制 Montgomery 标量的倍数。Edwards 路径按曲线约定固定使用 48·lcm(1…B1)，因此拒绝 `lcm=12`；GPU 执行不接受非 auto 的 CPU 专用选项，显式归约必须同时选择 `impl=simd`，避免 auto 落到标量后忽略该设置。

GPU 参数化的语法值为 0、2、3，均属于 Montgomery 曲线；可执行范围取决于链接的后端和构建配置。CPU Montgomery 只实现 PARAM0，因此 CPU 单次运行只接受 `-sigma 0:s` 这种带参数化前缀的种子；Edwards 不接受参数化前缀，`--gpu-param` 仅适用于 GPU。启动曲线计算前，驱动查询当前 GPU 后端是否支持该参数化；OpenCL 仅实现 PARAM3，CUDA 的 PARAM2 与 PARAM3 还受是否编入相应内核及折叠构建配置约束。CUDA PARAM0 在曲线构造、检查点及 save 中保留完整 64 位 sigma，批次不得在 2⁶⁴ 处回绕。PARAM2/3 的固定 sigma 及曲线批次须满足 sigma+曲线数≤2³²；不再截取低 32 位继续计算。参数化与 sigma 的冲突、后端不支持项在初始化 GPU 前报错。队列任务的 sigma 不符合要求时停止并保留原 worktodo。

这一步统一的是选择项解析与能力检查。CPU 曲线实现、GPU 内核、save 发布及各后端 checkpoint 仍沿用各自的路径。入口见 [ecm_stage1_selection.h](../../src/core/ecm_stage1_selection.h)、[ecm_driver.cpp](../../src/core/ecm_driver.cpp) 和 [ecm_backend.h](../../include/ecm_backend.h)。

## GPU 点乘

默认 ladder 按 s 的比特进行差分加法和倍点，状态由相邻点及固定差点组成。GPU 以切片推进，避免一个超长 kernel 占用整段 B1。每曲线的末点先解码、检查，求 Z 与 N 的 GCD；可逆时归一化 X=X/Z mod N，再输出 save。

resident 路径将点状态保持在 Montgomery 域，减少切片间往返转换。PRAC 将素数幂标量分解为规则链，以更多活跃点和控制流换取较少点运算；主机离线计划可缓存，并以分片保持检查点和进度语义。

PRAC 的 `single-compact` 专用体将 xADD 统一到一个调用点，倍点使用紧凑临时状态；终结规则和非终结规则共同维护同一组点角色。此路径仅在已编译的位宽/TPI/寄存器组合可选，缺失实例须报错。它不是发布默认算法。

## 位宽与资源

CUDA 容器需覆盖 N bits+6 个进位预留位。默认 2560…8192-bit 容器使用 TPI16，9216…16384 使用 TPI32；更小容器由 TPI4/8 分派。N=4423 bits 使用 4608/TPI16；N=8191 bits 因预留位需要 9216/TPI32，不能按输入是否超过 8192 判断。

默认 TPB128。每 block 曲线数=TPB/TPI，grid=⌈曲线批量/(TPB/TPI)⌉。寄存器上限不改变 TPB；实际 blocks/SM 还受寄存器分配粒度、shared memory 和线程限制影响，由 CUDA occupancy API 查询。

PRAC 可显式比较替代 TPI、寄存器 cap 和切片目标时长。强制 TPI32 会增加某些容器的 CGBN 内部 padding 与跨 lane 通信，不能仅凭每线程 limb 少就推断更快。

## 计算量与存储

设 C 为曲线批量，K 为容器位数，ℓ=K/32，b=bit_length(s)。ladder 计算量为 C·(b−1) 次组合点步骤；通用基数乘法的主项随 ℓ² 增长。PRAC 应分别统计实际 ADD/DBL 数及规则开销，不能用单一“每素数一次”近似。

Suyama CUDA 主状态每曲线包含 N、a24、差点和两相邻点，共 7ℓ 个 32-bit 字，即 28ℓC bytes。其他曲线族的布局由自身入口定义。指数和计划由批量共享，存档/检查点及主机对象另计。一次上传/读回量应按实际切片路线统计，不能把逻辑字节当作全部 PCIe 流量。

## 正确性与持久化

保存点记录规范化普通 X；checkpoint 则记录足以恢复下一切片的数学状态和布局身份。恢复须检查 N/B1/参数化/指数模式和实际 TPI，不能让 checkpoint 的状态域与当前内核失配。

跨实现验证应覆盖生成点、分片边界、最终投影/仿射点、有效因子及文件语义。选择不同寄存器预算不改变数学状态；改变 TPI 的恢复资格按当前格式检查。checksum 不替代 CPU/GMP 独立点验证。

## 代码入口

- [ecm_stage1_exp.cpp](../../src/core/ecm_stage1_exp.cpp)：指数生成和缓存。
- [cgbn_stage1.cu](../../kernels/cuda/cgbn_stage1.cu)：曲线布局、归一化、调度及保存点。
- [cgbn_stage1_kernel.h](../../kernels/cuda/cgbn_stage1_kernel.h#L1163)：TPI 容器分档和 ladder。
- [cgbn_stage1_prac_host.cuh](../../kernels/cuda/cgbn_stage1_prac_host.cuh)：计划、策略、切片及恢复。
- [cgbn_stage1_prac_single_add.cuh](../../kernels/cuda/cgbn_stage1_prac_single_add.cuh)：专用链规则。
- [ecm_mont_cpu.cpp](../../src/cpu/ecm_mont_cpu.cpp)、[ecm_edwards_cpu.cpp](../../src/cpu/ecm_edwards_cpu.cpp)：CPU 实现。

使用和已测最佳范围见 [Stage1 使用](../usage/STAGE1.md)、[性能](../performance/STAGE1.md)。

完整数学见 [Montgomery / Suyama / PRAC](../reference/ECM_MONTGOMERY_MATH.md)、[Edwards](../reference/ECM_EDWARDS_MATH.md) 和[参数化](../reference/ECM_PARAMETERIZATIONS.md)。
