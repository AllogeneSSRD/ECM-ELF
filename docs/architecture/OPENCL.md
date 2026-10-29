# OpenCL Stage1 与 Stage2

OpenCL 实现 Stage1 曲线批处理，以及独立 `ecm_opencl_stage2.exe` 的精确多项式 Stage2。Stage2 共用现有生产 save/队列/进度/结果流程，在编译和链接时选择原生 OpenCL 引擎；默认 CUDA 程序与算术路径保持独立。两者的设备编号、内存口径、支持范围与基准分别核对。

## 原生 Stage2 NTT 与算子层

`src/ecm/opencl/ntt/` 提供原生 OpenCL 精确域卷积，`src/ecm/stage2/opencl/` 在其上实现整数打包、进位、普通／梅森 S4、驻留树、Newton 逆多项式、fold 和 scaled 下降，以及点算子。域层保留 Goldilocks 素数 q=2⁶⁴−2³²+1、7 生成的单位根、自然序输入的 DIF 正变换和消费位反序频谱的 DIT 逆变换。逆归一化及卷积逐点乘法融合在 inverse tile 的输入处；没有全长排列缓冲。设备算子库不依赖GMP；生产引擎主机侧使用GMP批量归一化、尾部块积/GCD和独立检查，探针另用GMP构造夹具。

设备基线为 OpenCL 1.2 Full Profile、64 位地址和小端缓冲，不要求 FP64、subgroup 或 64 位原子扩展。能力查询的 supported 标志只表示满足这些静态条件，不代表已经通过该设备的数学验收。设备按平台枚举，尚未按物理设备去重或提供稳定 UUID 路由。

- [runtime.h:16](../../src/ecm/opencl/ntt/runtime.h#L16)：设备能力、显式 payload 预算与资源根。
- [Runtime / Buffer / Plan / Completion](../../src/ecm/opencl/ntt/runtime.h#L46)：缓冲、视图范围、工作组、batch、stride 和 offset；共享状态保留资源至设备真正完成。所有计划使用同一有序队列，由单个主机线程提交；上传和下载为阻塞操作，变换提交无逐层主机等待。
- [Goldilocks 原语](../../src/ecm/opencl/resources/ntt/goldilocks.clh)：规范范围的加减乘及任意 128 位输入归约，提供 mul_hi 和显式 32 位部分积两种高半乘法。
- [变换内核](../../src/ecm/opencl/resources/ntt/transform.cl)：全局 radix2 passes 与 local tile，第二 NDRange 维表示 batch；offset/stride 是显式的 64 位 word 索引。根表使用低／高指数分解，分别缓存正逆方向，不为每个 batch 复制根表。

API 接受 log2_length=1..29、tile_bits=1..12 的合法计划，仍须通过预算、单对象容量、设备与具体 kernel 的工作组/local-memory 检查；这不是已测范围。当前在 RTX4060 Laptop（驱动610.88）和 AMD gfx1150（驱动3640.0）验证了 k∈{1,2,3,4,5,8,10,12,16,20}、batch=1/3，两种乘法和工作组/tile 组合。每设备84,810项GMP算术对照、44组计划、8,982,512个round-trip word、同量卷积结果word、496个直接DFT结果及229项边界/资源检查通过。Intel与更长NTT尚未验证。

构建入口为 [build_opencl_stage2_ntt.ps1](../../tools/build/test/build_opencl_stage2_ntt.ps1)，生成 `build_opencl_stage2_ntt/opencl_ntt_probe.exe`。`--list`只查询设备；`--verify --device N --max-log2 20 --output FILE`运行正确性检查，输出路径必须未存在。构建为独立CXX/OpenCL项目，不编译CUDA；OpenCL头与导入库可通过 `-OpenClRoot` 提供。验证探针使用GMP，NTT库本身不依赖GMP或OpenSSL。

当前没有应用层持久程序缓存；初始化、编译和验证时间不能用作稳态吞吐。生产接入使用独立基本D模型，不消费CUDA tune/Auto B2数据，不自动切换CUDA后端。数学算子自检与冻结证据见 [基础实施草稿](../web/opencl-stage2-port-plan-20261027.html#ntt-foundation)；完整入口见下文“生产 Stage2 入口”。

### 整数多项式乘法与 S4

[polynomial.h](../../src/ecm/stage2/opencl/polynomial.h) 的 `PolynomialPlan` 接收同一模数下的批量多项式，输出普通表示的模系数，不使用 Montgomery 表示。输入系数可以达到 S 位上限，不必已经小于模数。视图的 offset 和 stride 均以 64 位 limb 计；只读 A/B 可以共用 owner，输出必须与两者分离。两个 pack 分别写入不同 digit 缓冲，同源平方不复制系数输入。模数使用最短小端 limb 数组，支持 2≤S≤16384；梅森模式严格要求模数本身等于 2ˢ−1，尚不包含真实 save 的自动承载选择。

形状计算保留 CUDA `ntt_shape_query` 的合同。设 P 为任一输入的最大系数数，b 为 digit 位数，s 为槽的 digit 数：

- 槽有效位数 K=2S+max(1,⌈log₂P⌉)，s=⌈K/b⌉，实际槽间隔为 sb bits。
- L=Ps，精确性要求 L(2ᵇ−1)²&lt;q；采用最大的合格 b，再选满足 n≥2Ps+1 的最小二次幂 n，最大 n=2²⁹。
- OpenCL 形状实现使用等价的整数除法比较，不引入浮点裕量；GMP 探针对照现有 CUDA 公式。CUDA 形状代码仍独立维护，两者尚未提取到共同规划模块。

设备流水线为输入位宽检查／pack → 两次 forward 与融合 inverse → carry → S4。carry 首先进行 R=⌈64/b⌉ 轮局部 cone，得到 0≤x≤2ᵇ；随后以“产生、传播、终止”三种状态做工作组前缀扫描，再由每个 batch 一个工作项扫描工作组摘要。最后写规范 digit，并检查零扩展尾部和输出有效槽之后的非零数据。B 的频谱缓冲在 NTT 完成后借作 carry 临时缓冲，不新增第三份全长数组。不使用 subgroup 或 64 位原子。

普通 S4 采用规范化除数、reciprocal 2-by-1 商估计、长除法乘减和借位校正；梅森 S4 按 2ˢ≡1 折叠，并把模数自身规范为零。除数和 reciprocal 在主机计划建立时计算，不依赖 GMP；每个输出检查槽高位、归约状态及结果范围。各工作项单独写 verdict，主机排空检查后才返回成功。

当前 `multiply` 是同步检查边界：所有设备步骤在同一有序队列提交，步骤间没有主机逐层等待；末尾下载 verdict 和 carry 摘要并等待完成。数值结果保留在设备。任何异常都使该次输出无效，不能用于后续树节点或发布结果。队列错误路径也等待在途命令；尚未注入设备丢失等全部异常情形。

入口依据为 [形状与计划](../../src/ecm/stage2/opencl/polynomial.cpp#L38)、[提交与检查](../../src/ecm/stage2/opencl/polynomial.cpp#L108)、[pack／carry／S4 内核](../../src/ecm/opencl/resources/stage2/polynomial.cl)。原生句柄通过私有 [native.h](../../src/ecm/opencl/ntt/native.h) 在设备算子之间共享，任务层不接触 OpenCL 对象。

一次请求所需的临时 payload 为 8B[2n+3(n/g)+(a+c)+(a+c−1)] bytes，其中 B 是 batch，g 是 carry 工作组，a、c 是两个输入的系数数；包含两份 NTT 数组、三个工作组摘要区与逐系数 verdict。另有两份模数数组、NTT 根表和调用者输入／输出。该公式只统计请求的缓冲需求，不包含驱动、program 或 context。`PolynomialWorkspace` 可由同一 Runtime、单个主机提交线程下的多个计划共用；各缓冲保留容量高水位，因此实际拥有量以 Runtime 账本为准。直接构造的乘法计划默认使用请求级缓冲，`Arithmetic` 默认复用共享工作区。明确的 payload 预算不足可回收空闲 scratch 后按原请求重试一次；不驱逐 F/H 或在途结果，尚未根据预算自动缩减 D 或 batch。

构建入口同时生成 `opencl_s4_probe.exe`；`--verify --device N --output FILE` 运行 GMP 闭环检查。2026-10-27 在 RTX4060 Laptop 和 AMD gfx1150 上，2～16384 位网格的普通／梅森路径、134 次乘法、792,836 个主要输出 word、174 个 S4 前完整整数系数和原始 S4 夹具均通过；另含 2²⁰ 长度闭环和原有 NTT 回归。该范围是离散网格，不代表区间内所有组合均已验收，也没有性能资格结论。

详细输入、二进制身份和设备结果见 [S4 实施记录](../web/opencl-stage2-port-plan-20261027.html#s4-closure)。本算子模块不负责worktodo/save解析，生产引擎在其外层编排；原地输出、延迟 verdict 排空与 Intel 实机仍待验证。

### 驻留树、fold 与 scaled 下降

设备代码分层，均不依赖 GMP：域卷积在 `opencl/ntt/`；[Arithmetic](../../src/ecm/stage2/opencl/arithmetic.h) 管理规范系数视图、计划缓存、共享工作区和小型系数操作；[tree.h](../../src/ecm/stage2/opencl/tree.h) 管理乘积树与 `Resultant` 状态；Points 管理点算术及实验设备GCD。构建生成 `ecm_opencl_ntt`、`ecm_opencl_stage2_ops` 两个库；生产引擎和独立探针在库外引入GMP。

`Arithmetic` 的上传边界要求系数处于 [0,N)。低层 S4 乘法的 S 位输入规则仍保留。多项式窗口保存 buffer、offset、stride、系数数与 batch；不以主机裸指针描述设备对象。规则、互不重叠且来自同一 owner 的切片直接表示为只读 strided batch；不规则切片才在设备内 gather。末端 strided 叶子先由一个设备 kernel 打包，再一次性回读，避免逐叶阻塞传输。

F/G 树均按实际子树长度分组，同一组中的独立节点一起乘法；不跨依赖层分组。补齐到二次幂的叶子是常数 1，实际次数按非空叶子守恒，不能将补齐叶子当作零根。F 树保留下降所需节点；G 树仅在树结构中保留根。在途 copy 和系数 kernel 的输入／输出另由 lease 保留至乘法检查、阻塞传输或显式 finish 确认真正完成，账本不会提前释放其容量。

设 F(X)=∏(X−bᵢ)，Gₖ(X)=∏(X−gⱼ)。`Resultant` 缓存 reverse(F) 的 P+1 项逆级数，采用 Newton 更新 g←g(2−ag) mod Xᵏ；逆级数常数项必须为 1，所以复合模数下不需要求任意系数的逆。每批执行 H←HGₖ mod F，以最高系数反转、逆级数乘法和 qF 乘减求余；H 保持 P 个系数，尾零不要求传回主机修剪。

下降保持 CUDA 的 scaled remainder 公式：根状态为 lowₚ(reverseₚ₋₁(H)·reverseₚ(F)⁻¹)。次数 a 的子节点、次数 b 的兄弟节点，从 parent·reverseᵦ(F_sibling) 中取第 b 至 b+a−1 项。每层按 (a,b) 分组；不为每个子节点重复求逆或长除多项式。最终叶子给出 H(bᵢ)，等于全部 Gₖ(bᵢ) 的乘积。

实验入口为 `opencl_stage2_probe.exe --verify --device N --output FILE`；`--bench` 使用固定输入比较逐节点、分组批处理、请求级／共享工作区与复制／借用切片。每种设置显式执行一次准备流程，再测三次，所有结果继续对照 GMP；这与生产 tune 的 warmup 配置无关。计时从点已就绪、Arithmetic 已建立后开始，包含 F/逆多项式、G/fold、下降、检查及叶子回读；不含 CPU 点生成、CPU GCD、初始输入传输或队列任务逻辑。首次流程另记计划创建与可能的 JIT，不混入稳态中位数。

当前能力和实验依据见 [树与完整小曲线实施记录](../web/opencl-stage2-port-plan-20261027.html#tree-closure)。生产入口具有基本owned-payload准入，尚无运行中降低D/batch后重启的预算回退；giant chain、更大规模与Intel设备验收仍待完成。小型算子流程的改进不构成生产性能保证。

### 点生成、归一化与 GCD

[Points](../../src/ecm/stage2/opencl/points.h) 是独立于多项式树的设备算子；其 [实现](../../src/ecm/stage2/opencl/points.cpp) 和 [内核](../../src/ecm/opencl/resources/stage2/points.cl) 不依赖 GMP。接受最短小端 limb 表示的奇数目标模数 N≥3，容量上限为 16384 bits；每个 Runtime 由一个主机线程提交。容量不等于全部组合已验收：AMD 当前在构造时拒绝超过 512 位的点算术，包括梅森模数；Intel 未实机验收。该资格门禁只作用于 Points，不限制单独验证的 OpenCL NTT/S4。各方法等待本次 kernel 完成，归一化和 GCD 另排空逐段／逐项 verdict。结果保留设备所有权，跨方法不必下载点或叶子。

点生成对每个无符号 64 位标量独立执行 Montgomery ladder。xDBL 使用 5 次模乘，xADD 使用 CUDA 的 6 次模乘公式及半和／半差，使输出投影缩放也与 CUDA XADD6 一致。SOS 乘法采用完整乘积及 Montgomery 消去；私有容量固定、循环采用运行时 limb 数，限制首次 JIT 的展开。该点模块目前没有专用梅森快速归约，也没有 giant seed/chain。普通表示的输入 X/Z/a24 在设备转换；输出是规范普通表示的 X/Z，k=0 返回无穷远点 (1:0)。同一 Runtime 下的点坐标／叶子标量视图支持 offset 和 stride，不能把不同模数的普通剩余类直接当作同一模数的输入。

归一化按段计算 Z 的前缀积，每段一次二进制扩展 GCD 求逆，再反向求各 Z 的逆。段积不可逆时逐点重算，以保留不同 Z 上的互补因子。结果同时返回 affine X 和 gcd(Z,N)：单位点的 GCD 为 1；非单位点的 X 占位为 0，调用者必须先处理对应 GCD，不能直接把占位点用于有效曲线。迭代边界或设备错误使整个调用失败；N 本身不是严格因子，也不能作为命中发布。

`products` 对规范的目标剩余类分块相乘，输出仍是普通表示。`gcd` 接受不超过该 limb 宽度的非负整数，包括大于 N 的承载剩余类；零输入返回 N。最终严格因子判定、去重、整除检查与因子分解属于任务层。CUDA 当前对应实现仍是 GPU 块积配合 CPU/GMP GCD，依据为 [factors.cpp](../../src/ecm/stage2/cuda/factors.cpp) 和 [batch.cpp](../../src/ecm/stage2/cuda/batch.cpp)；设备 GCD 是 OpenCL 实验路径，并未更改 CUDA 默认行为。

梅森承载时，先在真实目标 N 上生成、归一化，再将规范代表元提升至 M，且需证明 N 整除 M。不能要求被剥离因子上的 Z 也可逆。生产引擎按M的limb宽度显式扩宽根，在M中计算，最终向N投影后做块积/GCD；独立算子夹具另覆盖同宽零拷贝提升。自动比较普通/承载路线尚未接入。

构建入口增加 `opencl_points_probe.exe`，`--verify` 检查设备点至树／fold／下降／GCD 的小曲线闭环；`--bench` 比较单线程 GMP 与 OpenCL，附加 `--cuda-dll` 可调用独立的现有 CUDA 内核对照。CUDA DLL 由 [build_cuda_points_reference.ps1](../../tools/build/test/build_cuda_points_reference.ps1) 构建；OpenCL 库和默认探针构建不要求 nvcc。计时、操作数、内存／传输公式及资格限制见 [点算子实验记录](../web/opencl-stage2-port-plan-20261027.html#points-closure)。设备GCD不压缩命中结果；生产尾部采用CPU/GMP，不调用该实验设备GCD。

## 生产 Stage2 入口

`ecm_opencl_stage2.exe` 支持归一化PARAM0与Edwards文本save、ECMSTAGE2队列、共用TOML及Worker覆盖、数量夹限、曲线级进度、队列重排续跑、停止策略、结果校验与发布。入口和原CUDA程序共用host sources，使用 `ECM_STAGE2_OPENCL` 编译标志；[backend.h](../../src/ecm/stage2/backend.h) 区分结果身份。既有引擎ABI暂保留CUDA前缀，OpenCL实现不链接CUDA运行库。

[engine.cpp](../../src/ecm/stage2/opencl/engine.cpp) 执行以下闭环：

1. 验证target N及可选承载M；读取OpenCL设备与owned-payload预算，固定D或从共享D表按基本模型选型。
2. 创建原生有序队列，执行设备多项式乘法启动检查。
3. 生成baby点，CPU每256个分母一次批量求逆，上传规范根并构造F树与逆多项式。
4. 分批生成giant点，按相同边界归一化，构造G树并fold。设备ladder每次提交的点数按位宽限制，独立于多项式batch；尚无giant chain。
5. 执行scaled下降并读回叶子；每条曲线选三个baby位置，以全giant范围的独立GMP乘积核对叶子，GPU点另按批取样核对。
6. CPU每64叶子归并并做GCD；饱和块继续检查单叶。因子须严格整除target N，设备排空及检查全部成功后才回调生产发布流程。

AMD点模块超过512 bits的资格限制保持不变；生产引擎显式使用CPU点生成，而F/G/NTT/S4/下降仍在OpenCL设备上。不可逆分母逐点定位真因子；仅出现模N饱和、无法形成仿射根且未得到真因子时失败，不用伪造零根发布NF。

OpenCL使用独立的保守owned-buffer模型与runtime硬预算，不能把设备global memory当作free。arena=0时默认预算为global的一半与1GiB中的较小值；显式arena为payload上限，F树、结果、临时区和根表共同计账。基本准入模型还保留32MiB固定裕量，可能拒绝本可容纳的极小预算。失败不会自动放宽预算，也不推进完成收据。

首版要求显式B2或任务/config中的B2；支持显式 `--carrier-exponent` 并验证整除。CUDA tune/Auto B2、自动承载比较、relocation、Graph和低CPU调度flag不作为OpenCL优化策略消费。使用方法见 [Stage2](../usage/STAGE2.md#原生-opencl-入口)，完整A/B见 [OpenCL Stage2性能](../performance/OPENCL_STAGE2.md)。

## 算子与选择

算子族包含 Montgomery 乘法/平方、模加/减和 special_mult。描述符维护 id、别名、内核文件、函数名、位宽/TPI/容器限制、厂商掩码与 auto 优先级。

固定位宽算子需要精确匹配容器；generic 算子按声明范围使用。手动选择和 auto 都经相同的设备/位宽资格检查，不能让别名绕过约束。`mp_` 前缀区分公共多精度助手与专用算子接口。

## 内核组装

主机根据路径计划确定 limbs、TPI、工作组和算子。组装顺序为配置宏、公共头、算子文件、统一接口和主 kernel；相同算子源只加载一次。内核缓存身份包含选择、源和版本信息，避免配置变更后复用错误程序。

普通、local 和 cooperative 路线有不同私有/LDS/协作状态。某个微基准快不代表完整 ladder 快；厂商汇编和 limb24 路线只有在对应声明与验证范围内使用。

## 算术与资源

通用 CIOS 乘法主项 O(limbs²)，每一步包括乘累加、Montgomery 消去及条件校正。寄存器/private spill、LDS 容量、同步和实际 wave 占用共同影响时间。算子正确性应检查边界/进位/借位、规范范围与 GMP 对照，再进行完整 Stage1 验证。

OpenCL 编译器对整数乘法和内联汇编的生成方式可能随驱动/设备变化，ISA 导出工具用于核对实际指令。环境中的时钟/功率条件不是跨设备性能保证。

## 代码入口

- [opencl_ecm_path_registry.h](../../src/ecm/opencl/arithmetic/path_registry.h)、[实现](../../src/ecm/opencl/arithmetic/path_registry.cpp)：路径元信息与选择。
- [opencl_ecm_runtime_config.cpp](../../src/ecm/application/configuration/runtime.cpp)：设备和运行配置。
- [opencl_ecm_stage1.cpp](../../src/ecm/stage1/montgomery/opencl/engine.cpp)、[impl_opencl.cpp](../../src/ecm/opencl/runtime/runtime.cpp)：主机/内核接线。
- [生产资源](../../src/ecm/opencl/resources/)：公共原语、算子与 Stage1 kernel；纯工具资源仍在 kernels/opencl/bench。
- [opencl_ecm_selftest.cpp](../../src/ecm/opencl/diagnostics/selftest.cpp)、[opencl_mont_isa_export.cpp](../../tools/disasm/opencl_mont_isa_export.cpp)：验证和 ISA 入口。

## 资源与 local 构建

桌面入口为 [build_opencl_local.ps1](../../tools/build/build_opencl_local.ps1)，默认输出到 build_opencl_local/stage1/&lt;profile-key&gt;。公开 include 只承载共享主机接口；OpenCL 的 context/program/cache、设备枚举和曲线实现均为 src/ecm/opencl 内的私有实现。

local 构建以普通 CXX 编译，关闭 CUDA、GUI、工具及独立 CPU Stage2 目标；profile 核对工具链、依赖与用途，拒绝不兼容或未标记的已有目录。CMake 将生产资源同步到叶目录 opencl_resources，修改资源不依赖 CXX 对象重编；build_manifest.json 记录程序及资源摘要。该入口不生成 release 包。

资源加载保留显式 --kernel-root、当前目录相对查找及编译默认根的既有优先关系。默认根指向构建叶目录随附资源；不含默认宏的实现使用 src/ecm/opencl/resources 回退。生产自检源在 diagnostics/mont.cl；旧逻辑名称 bench/mont.cl 先按旧方式查找，缺失时再加载 diagnostics/mont.cl，因此已有自定义根仍可使用原布局。实际算术、源拼装次序和设备/驱动/完整源/编译选项的缓存身份保持。

完整发布入口使用独立release用途构建并分发 `ecm.exe`、生产资源及OpenSSL运行库，GMP固定使用与GUI/CUDA worker相同的vcpkg通用版本。该构建启用 `ECM_OPENCL_PORTABLE_RESOURCES`，默认资源根由EXE所在目录派生，显式 `--kernel-root` 仍优先；local默认仍保留上述构建目录资源根。分发与身份验证以[构建与发布](../usage/BUILD.md#完整发布与按需加载)为准。OpenCL内核由设备驱动统一JIT，两套CUDA包不改变其编译策略。

Android CMake 消费同一主机源码，Gradle 从新生产根和原纯 bench 根组合资源，保留 assets/kernels/opencl 逻辑根。本轮只核对其路径合同，未进行 Android 实体构建或设备验收。
