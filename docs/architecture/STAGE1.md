# Stage1 算法与数据

## 曲线与标量

Suyama PARAM0 使用 sigma 构造 Montgomery 曲线和初始 X:Z 点。设 u=σ²−5、v=4σ，起点为 u³:v³；曲线参数通过模逆构造。不可逆分母需在目标 N 上求 GCD，不能略过。

光滑标量 s=∏(素数 ℓ≤B1) ℓ^⌊logℓ B1⌋；`lcm=12` 使用 12s。主机分段筛生成素数，分块累乘，用带身份/校验的缓存复用指数；指数缓存与曲线 checkpoint 分离。

CPU Edwards 路径使用自身曲线构造及扩展坐标公式；CPU Montgomery、CUDA 和 OpenCL 的点表示/算术域不同。只有投影或归一化后的数学点可跨实现对比，不能比较未经解码的字数组。

## 方法选择与输入校验

Stage1 单次运行和队列模式共用五个独立选择项：`curve`（Montgomery / Edwards）、`target`（CPU / GPU）、`impl`（自动 / IFMA SIMD / GMP 标量）、`reduction`（自动 / Montgomery / Mersenne fold）及 `lcm`（素数幂最小公倍积 / 其 12 倍）。默认是 Montgomery 曲线、GPU 执行；Edwards GPU 需要编入该模块的 CUDA 可执行文件，OpenCL 会拒绝该组合。`Auto` 在内部 CPU 实现及 SIMD 归约枚举中均为 0；调用现有 IFMA 接口时显式转换，避免枚举序号与实现常量耦合。TOML 和 CLI 均拒绝无效选择值，参数默认值及范围由 [配置定义](../../config/ecm_options.json) 维护。

TOML 使用 `curve`、`target`、`impl`、`reduction`、`lcm`；CLI 使用同名的双横线选项。`curve=mont|ed` 与 `target=cpu|gpu` 是不同轴；`impl=auto|simd|scalar` 只用于 CPU，`reduction=auto|montgomery|mersenne-fold` 控制 CPU SIMD 约减，`lcm=1|12` 控制 Montgomery 标量的倍数。Edwards 路径按曲线约定固定使用 48·lcm(1…B1)，因此拒绝 `lcm=12`；GPU 执行不接受非 auto 的 CPU 专用选项，显式归约必须同时选择 `impl=simd`，避免 auto 落到标量后忽略该设置。

GPU 参数化的语法值为 0、2、3，均属于 Montgomery 曲线；可执行范围取决于链接的后端和构建配置。CPU Montgomery 只实现 PARAM0，因此 CPU 单次运行只接受 `-sigma 0:s` 这种带参数化前缀的种子；Edwards 不接受参数化前缀，`--gpu-param` 仅适用于 GPU。启动曲线计算前，驱动查询当前 GPU 后端是否支持该参数化；OpenCL 仅实现 PARAM3，CUDA 的 PARAM2 与 PARAM3 还受是否编入相应内核及折叠构建配置约束。CUDA PARAM0 在曲线构造、检查点及 save 中保留完整 64 位 sigma，批次不得在 2⁶⁴ 处回绕。PARAM2/3 的固定 sigma 及曲线批次须满足 sigma+曲线数≤2³²；不再截取低 32 位继续计算。参数化与 sigma 的冲突、后端不支持项在初始化 GPU 前报错。队列任务的 sigma 不符合要求时停止并保留原 worktodo。

`ECMSTAGE1` 作为 save 型 Stage1 输入时，文件名前缀决定该任务的曲线构造，末尾字段决定 B1；覆盖 TOML 后再校验后端能力和 sigma 范围。`ECMSTAGE2` 只由独立 Stage2 程序领取。每条 Stage1 任务复制基础选项，防止混合队列之间传播覆盖。Edwards 前缀同时恢复其固定标量约定；执行目标保持 TOML 设置。名称、旧 save 历史复制及交接合同见[队列输入](../usage/STAGE1.md#程序与输入)。

这一步统一的是选择项解析与能力检查。CPU 曲线实现、GPU 内核、save 发布及各后端 checkpoint 仍沿用各自的路径。入口见 [selection.h](../../include/ecm/core/stage1/selection.h)、[ecm_driver.cpp](../../src/ecm/application/cli/ecm_driver.cpp) 和 [ecm_backend.h](../../include/ecm/service/execution/legacy_gpu.h)。

## 单任务编排与资源拥有权

直接输入和队列调用同一个 [run_stage1_task](../../src/ecm/service/stage1/task.cpp#L138)。它依次校验任务元数据与输出、打开批次、接纳旧 save 历史、执行后端、核对逐曲线身份、发布 submission/internal、seal 并结束批次。即时因子仍在原有安全位置发布；队列完成和跨阶段交接协议在任务之外。

[公开请求与结果](../../include/ecm/service/stage1/task.h)只使用普通 C++ 和 GMP。Stage1Request 在同步调用期间借用模数、命名和已解析选项，选项分为执行偏好、身份和持久化策略三组。TaskResult 独立拥有逐曲线结果，禁止复制并允许移动，通过只读访问器取数；不保留请求引用，不公开原始分配或进度字段。状态区分完成、停止和失败，并记录准备、执行、持久化、发布或设备执行失败。CUDA 明确失败优先于同时收到的停止；CPU/OpenCL 的旧标记仍通过兼容映射接纳。设备执行失败使队列停止并保留原任务，不继续使用可能失效的设备 context。

私有 Stage1TaskSession 只管理批次和锁，没有第二套 run。begin 前建立清理责任，部分开始失败也释放锁；成功结束才清除 checkpoint，失败或停止保留恢复材料。结果中的 factors、array_found、sigmas 分别清理，sigmas 不依赖 factors 是否存在。计时和总曲线数属于私有执行副本，CPU 回调不再强制修改只读选项。目前仍是单进程单活跃批次，尚未支持同进程并行任务。

任务、CPU/GPU host workflow 与唯一执行分派已成为独立普通 CXX。应用 [ecm_driver.cpp](../../src/ecm/application/cli/ecm_driver.cpp)保留单次输入、依赖组装、模式选择与展示/退出；它只组装一次实际 execute_stage1、构建身份和默认 NAF 窗口读取的三回调 [私有依赖](../../src/ecm/service/stage1/execution_contract.h)。私有计算合同由 ExecutionOptions/ExecutionResult 表达，拥有权与状态实现落 execution_contract.cpp；TaskResult::State 仅在 task.cpp 定义。计算回调包含本后端所需准备和执行，不重复建立 Runner、Session.run 或 factory/select。结果 writer 已收口为公共声明与一个普通 CPP，文件锁/编码/发布细节不再进入 task。

任务与真实持久化模块可通过普通 CXX 独立编译验证，替换计算入口的 14 个合同用例覆盖拥有权、失败/停止优先级和锁释放；生产程序另有实际后端恢复和发布回归。构建方式见 [构建说明](../usage/BUILD.md)。批次清理见 [ecm_stage1_checkpoint_runtime.cpp](../../src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp#L317)，格式与恢复接纳见 [Checkpoint](CHECKPOINT.md)。

## 应用输入与队列编排

driver 组装一次计算、身份和默认 NAF 三回调，直接模式调用 task，队列模式进入应用 [queue_source.cpp](../../src/ecm/application/cli/queue_source.cpp)。source 读取一次配置快照，解析两种任务格式，规范化参数/路径，并按实际曲线初始化 Prime95 交接；它通过 NextQueueTask 提供已规范化输入，不执行计算或完成提交。[stage1_input.cpp](../../src/ecm/application/cli/stage1_input.cpp)集中共同输入检查和命名。

[queue_workflow.cpp](../../src/ecm/service/work/queue_workflow.cpp)是唯一任务循环与完成协议。其 [公开合同](../../include/ecm/service/work/queue_workflow.h)中 QueueItem 拥有 mpz 模数、字符串和本次选项；转交配置指针借用应用状态至同步返回，不能保存 Stage1Request 的借用引用。QueuePaths 仅传队列/finished/save 同步策略，没有整个应用 Settings 或额外 Runner。

单任务发布封存仍在 task。QueueWorkflow 检查准备/持久化/发布/设备失败 → 展示因子 → 检查停止/执行失败；成功后 Stage2 pending → finished → 按原票据删除/替换源行 → 同步 save → Stage2 flush → Prime95 delivery。文件模块只做具体读写。准备/持久化/发布/设备失败或停止保留原行，其他执行失败标记原行；解析错误标记失败的既有返回 0 策略也保留。save skip 仍在执行前预读；Prime95 idle recovery 区分初始空队列和已读取无效任务。不存在逐任务重新读取 TOML。

## 选择词汇与配置依赖

[selection.h](../../include/ecm/core/stage1/selection.h)公开曲线/目标/CPU 实现/归约/标量构造类型，以及这些词汇的解析、合法组合和名称声明；11 个实际函数集中在 [selection.cpp](../../src/ecm/core/stage1/selection.cpp)。旧应用私有选择头已删除，不保留转发。字段、枚举值、ASCII 解析、sigma 参数冲突规则与名称保持。

task 的默认构造从 [生成默认值](../../src/ecm/platform/generated/configuration_defaults.h)获取同一 JSON 定义，不包含应用的 TOML 解析/完整绑定。准备模块也只依赖公共选择合同。应用完整生成绑定引用这份默认值，未复制默认值来源或新增配置服务。真实无 SDK 任务编译记录覆盖共享 selection 实现，且不包含 ecm_toml.h/ecm_config_generated.h。

## 标量和进度的实现隔离

[scalar.h](../../include/ecm/core/stage1/scalar.h)统一声明 LCM 标量与主机 NAF 位图；构造、位读取和 digit 方法全部在现有 [ecm_stage1_exp.cpp](../../src/ecm/core/stage1/ecm_stage1_exp.cpp)，没有新增标量转发层或第二个实现 TU。NAF 的字段、布局、窗口范围和编码保持，设备只消费已有位图/数据合同。

[stage1_progress.h](../../include/ecm/platform/stage1_progress.h)只声明共享 GPU screen 格式及 SpeedWindow 方法；实现位于 [stage1_progress.cpp](../../src/ecm/platform/stage1_progress.cpp)。50 样本窗口的布局、统计公式与 stdout 协议保持，不新增 Pimpl 分配。CUDA 包含声明；修改格式或速度统计函数体只需重编对应 CXX。Montgomery 对该头的多余显式 OBJECT_DEPENDS 已移除，真实 include 依赖仍正常校验。

共享选择、主机准备、CPU/GPU host workflow、Montgomery 构造/codec 和结果发布已提取；队列已提取；Montgomery 环境策略、私有执行合同和应用 PARI 已收口。实际设备同步、回读、资源管理继续依赖 CUDA。独立主机构建的真实依赖记录同时覆盖 task、标量与进度实现，不包含 CUDA SDK 头。

## CPU 执行与共享主机准备

CPU 两条完整执行路径位于 [legacy_executor.cpp](../../src/ecm/backend/cpu/stage1/legacy_executor.cpp)，通过同目录私有头的 run_edwards_stage1、run_mont_stage1 接收 N、B1/B2、曲线数、save 策略、表达式和本次执行选项，填充已有 ExecutionResult。线程/ISA 选择、点状态、恢复、分批推进、逐曲线 GCD、factor/DONE 与存档时序由该 Implementation 集中持有。它用普通基线 CXX 编译，不把 AVX512 编译设置施加到调度文件；原 SIMD 实现继续独立编译。应用仍解析 affinity 文本，执行模块只应用解析后的 CPU 集合。

[preparation.cpp](../../src/ecm/service/stage1/preparation.cpp)集中 CPU/GPU 共用的标量缓存入口、随机 sigma、保存路径、固定 sigma 范围检查及主机时钟/终端检测；私有 preparation.h 只声明实际共用函数，不增加 wrapper 类。随机状态仍是同一个进程级状态。CPU 模块与准备实现已用真实 MSVC 依赖记录确认不包含 CUDA SDK 头；GPU host workflow、分派与队列已提取到各自普通 CXX。

各完整工作流负责自己的 factor、DONE 与 save 顺序，任务层负责最终账本和批次关闭。实际设备同步和回读继续依赖 CUDA。

## 唯一执行分派与 GPU 主机流程

task 的 execute 依赖绑定 [execution.cpp](../../src/ecm/service/stage1/execution.cpp)的 execute_stage1。该文件只做后端与曲线分派。CUDA 由构建显式定义 `ECM_HAS_CUDA_STAGE1`，调用 [私有 execution.h](../../src/ecm/backend/cuda/stage1/execution.h) 的 `ecm_cuda_stage1::run_mont_stage1 / run_edwards_stage1`。两个完整入口参数粒度一致，分别位于曲线目录的 workflow.cpp；它们编排准备、恢复/构造、设备执行、末点及批次 save，不拥有任务账本或队列。

[Montgomery workflow](../../src/ecm/backend/cuda/stage1/montgomery/workflow.cpp)在准备后直接调用 CUDA 曲线运行时，不经过旧长参数 backend hook；factory 继续提供设备准备、能力查询和兼容入口。[Edwards workflow](../../src/ecm/backend/cuda/stage1/edwards/workflow.cpp)持有其状态、初始化、恢复、进度回调、末点检查、GCD 和 save；设备 binary/NAF 接口及检查点 codec 的后续分责不影响入口粒度。低层探针的链接清单不包含完整 workflow。

CUDA 关闭时调用 [OpenCL Montgomery workflow](../../src/ecm/backend/opencl/stage1/montgomery/workflow.cpp)，保留其既有参数、后端 hook 和发布行为。两条 CPU 完整流程继续位于 CPU 实现；共享分派不包含任何曲线工作流正文。共用的随机 sigma 和 batch d 声明在 `service/stage1/gpu_common.h`，与 OpenCL 入口 ABI 分离。

CUDA 工作流可以明确报告执行失败或主动停止；明确失败优先于同时收到的停止信号，不再被任务层改写为准备失败。CPU/OpenCL 的旧返回码和准备/发布标记仍通过兼容映射接纳。任务层按执行/发布调用边界归类逃逸异常，不根据结果指针是否已分配猜测阶段。Edwards 低层接口保留兼容返回码并可携带明确设备故障来源。周期存档策略和载荷模块的整理仍见实施草稿中的未完成项。

这些主机 CPP 的实际独立编译依赖没有 CUDA SDK 头。CUDA runtime、同步、设备资源与回读实现保留现有 CUDA 依赖；Montgomery 构造、结果及 checkpoint/screen/CSV 实现也已提取到普通 CXX；设备协作继续由 adapter.cu 持有。

## CUDA 故障与构造期结果

[Montgomery runtime](../../src/ecm/backend/cuda/stage1/montgomery/runtime.h)返回完成、主动停止、采样截止或失败；[运行结果合同](../../src/ecm/backend/cuda/stage1/runtime_contract.h)区分实际失败来源。旧 C 入口只负责兼容适配。CUDA/CGBN 检查通过异常返回运行边界，设备缓冲、事件、报告、诊断文件和主机数组由 [资源拥有者](../../src/ecm/backend/cuda/stage1/montgomery/resources.cuh)释放。调用方不假定设备错误后的 context 可以继续使用。Edwards 的已有缓冲、报告及事件拥有者禁止复制，设备错误同样回到任务层。

PARAM0/PARAM2 构造遇到模逆失败时计算 GCD：真因子立即通过既有 submission 通道持久化，并记录该曲线的 factor 结果；GCD=N 作为退化构造拒绝。设备批次不压紧索引：已找到因子的通道使用无害恒等状态，末点处理读取已保存的真因子，不能将该通道当作可用于 Stage2 的点。恢复时重新推导构造期因子元数据；旧载荷格式不变。有效曲线的公式与 sigma 身份不受影响；PARAM2 低层也拒绝 sigma<2，而不改用另一 sigma。

构造布局、主机数组及结果分配检查尺寸；结果数组在计算前取得完整所有权，后续失败仍由结果对象清理。临时参数和 GMP 标量也由工作流局部对象持有。共享 storage 和 task 仍拥有最终清理权限；普通成功路径的底层清理请求不会提前删除尚未封存的批次。

## Montgomery 主机数学与存档输出

[host_math.cpp](../../src/ecm/backend/cuda/stage1/montgomery/host_math.cpp)集中 param0/2/3 初始状态、param2 Jacobian 辅助计算、标量打包、np0、普通整数解包与最终结果处理。六个实际跨设备协作入口经私有 host_math.h 声明；内部转换和算术 helper 留在 CPP 内部。输出仍调用同一 factory 中的 outputf/test_verbose，verbose 状态没有复制。FULL_BUILD 条件下的诊断分支和断言语义保持。

[host_io.cpp](../../src/ecm/backend/cuda/stage1/montgomery/host_io.cpp)集中 ladder V5 与 resident/PRAC checkpoint 编码、读取/校验、保存路径、TTY/screen 和 dump CSV。私有 host_io.h 只提供实际声明与原载荷数据布局。字节格式、版本、字段次序及 size_t/time_t 类型保持；resident 恢复仍在日志前回写 sigma。progress 的累计状态留运行层，格式及节流读取显式接收该状态，未导出新的可变全局。

adapter.cu 保留真实 tier/dispatch、CGBN report、设备分配、上传/回读、事件/同步、计算循环与资源释放；global event 与标量上传继续先于 checkpoint 恢复和 CPU 构造。两份主机 CPP 无 CUDA SDK 依赖。改变数学或存档/输出函数体的实测增量为 2 CXX、0 CU；改变跨层声明或设备协作实现仍会使真实依赖失效。

## GPU 点乘

### CUDA 模块边界

CUDA 的发射描述、设备类型与位宽清单均为 `src/ecm/backend/cuda/` 内的私有实现；其中 `dispatch/tiers.def` 唯一维护位宽。公开任务与选择合同位于 `include/ecm/service/` 和 `include/ecm/core/`。`src/ecm/backend/cuda/factory.cpp` 负责后端能力查询；`src/ecm/backend/cuda/stage1/` 负责 GMP 编解码、分配释放、错误检查、切片、回调、计时和恢复。Montgomery 驱动为 `montgomery/adapter.cu`；Edwards 的参数校验在 `edwards.cpp` / `edwards_naf.cpp`，两种执行模式的主机循环由 `edwards_runtime.cu` 单独编译。

设备算术与薄发射器位于各曲线目录，独立实例放在对应 `instances/`。Montgomery 的 PARAM0/2/3 各有七组实例；Edwards binary/NAF 共用七组实例。设备实例闭包不包含项目主机驱动，也不接收 GMP 对象、进度回调或 checkpoint 对象。第三方 CGBN 在 NVCC 主机解析阶段需要 GMP 类型，由 `src/ecm/backend/cuda/arithmetic/cgbn/types.cuh` 的主机条件包含适配，设备运算接口仍使用原始缓冲和标量。

依赖方向为：任务/旧执行适配 → CUDA 私有 API → CUDA 主机实现 → 私有发射接口 → 设备实例/算术；主机与设备都可读取 `tiers.def`。实际编译源由 `tools/build/internal/cuda_sources.cmake` 维护。PRAC 设备源码、主机计划与运行时统一位于 `src/ecm/backend/cuda/stage1/montgomery/prac/`，仅在显式启用时链接 `ecm_cuda_prac` 对象目标。Stage2 的主机、算术、NTT 及实验边界见 [Stage2 模块职责](STAGE2.md#cuda-模块边界)。主机等待探针通过 `src/ecm/backend/cuda/stage1/edwards/host_wait.h` 与唯一测试适配调用真实 Edwards 运行时，探针入口不再包含私有主机实现。生产闭包禁止反向包含工具与实验源码。

Montgomery 默认 ladder 按 s 的比特进行差分加法和倍点，状态由相邻点及固定差点组成。GPU 以切片推进，避免一个超长 kernel 占用整段 B1。每曲线的末点先解码、检查，求 Z 与 N 的 GCD；可逆时归一化 X=X/Z mod N，再输出 save。

### GPU Edwards

[CUDA NAF 批量入口](../../src/ecm/backend/cuda/stage1/edwards/edwards_naf.cpp)以 CGBN Montgomery 域的 `(X,Y,Z,T)` 执行 `48·lcm(1…B1)`。`naf_w=0` 在 GPU 选择窗口 6，显式允许 2..12；每曲线在 GPU 预计算 2^(w−2) 个奇数倍投影点，不做字典模逆。主机以两份紧凑位图保存非零列和有符号索引，生产调用使用自适应切片，按实测执行时间调整每片 NAF 列数。设备累加点在算术域跨片保留，只回传四个规范普通坐标供回调及检查点使用。连续倍点省略暂时不用的 T，在加法前及片边界恢复完整 `XY=ZT`。入口与位图定义见 [标量与 NAF 合同](../../include/ecm/core/stage1/scalar.h)，设备字典与切片见 [cgbn_edwards_naf.cuh](../../src/ecm/backend/cuda/stage1/edwards/naf.cuh)。

容器选择要求 N 至少留一位余量，选择本次构建中最小的可用容器。完整网格为128/192/256/384/512（TPI4）、768..2048间隔256（TPI8）、2560..8192间隔512（TPI16）、9216..17408间隔1024（TPI32），共32档，TPB128。Montgomery仍预留六位，不能把两个模型的边界条件互换。`ECM_TIERS`同时筛选Edwards实例；缺档时使用更大可用档，超过本次构建可用上限时报错。日志的 `GPU Edwards geometry` 给出实际容器、TPI及网格，查询入口为 `ecm_cuda_edwards_tier`。

二进制与NAF的实例共享[位宽清单](../../src/ecm/backend/cuda/dispatch/tiers.def)，核函数和七组实例化文件位于 `src/ecm/backend/cuda/stage1/edwards/instances/`，按TPI及位宽范围拆分并行编译；`src/ecm/backend/cuda/stage1/`保留输入校验、分派和主机驱动，相关 API 为曲线目录内的私有头。模板算术与切片逻辑见[共用实例实现](../../src/ecm/backend/cuda/stage1/edwards/instances/instances.cuh)。字典显存为曲线数×容器字节数×4坐标×2^(w−2)，分配失败会报错并保留未完成队列，不静默改变显式窗口。

主循环默认倍点为 HWCD；`ECM_EDWARDS_DBL_ALGO=4|5` 选择 HWCD（内部编号 4）或 2XY 多项式（编号 5），字典构造使用 HWCD。后者采用 Prime95 算法 5 的代数形式，CGBN 不模拟其 FFT 复合指令；内部编号 4 不代表 Prime95 算法 4。参考版本与运算计数见 [数学说明](../reference/ECM_EDWARDS_MATH.md#21-prime95-倍点算法编号与-fft-条件)。`ECM_EDWARDS_STAGE1_ALGO=binary|naf` 可显式选择 [原二进制路径](../../src/ecm/backend/cuda/stage1/edwards/edwards.cpp)或 NAF，用于对照；binary 不接受窗口或 NAF 倍点设置。未指定时使用 NAF，恢复时遵从已保存的模式。

[曲线构造](../../src/ecm/core/curve/ecm_edwards_curve.h)先在辅助 Weierstrass 曲线上用 Jacobian 坐标计算 sigma 倍点，再构造 Atkin–Morain 参数。模逆失败在实际目标 N 上区分真因子与退化曲线，同时检查 d(1−d) 的单位性和基点方程。末点检查 `XY=ZT` 与 Edwards 方程，用 `(Z+Y):(Z−Y)` 转到 Montgomery，检查分母 GCD 后归一化；真因子独立标记。有效恒等点且 GCD=N 时没有可用 Stage2 点，该曲线完成但不写点记录。

[worker 接入](../../src/ecm/backend/cuda/stage1/edwards/workflow.cpp)支持 CLI 和队列、周期及安全停止检查点。当前 GPU Edwards 使用 V3 二进制载荷，保存完整 N、算法、进度、各 sigma 和普通整数状态；指数只保存位数与指纹。恢复重建基点、核对点不变量并采用已保存的模式，显式不匹配选择报错。载荷不绑定 TPI，恢复按 N 重新选择容器。NAF 列数与 binary 位数不能互换。完整字段、兼容读取和恢复资格统一见 [Checkpoint](CHECKPOINT.md#42-gpu-edwards-v3)。

CPU/GPU Edwards 共用完成点文本，可由 CUDA Stage2 和 [CPU Stage2](../usage/EDWARDS_CPU_STAGE2.md) 消费；sigma 不能按 Suyama PARAM0 解释。字段、状态、校验及外部程序能力统一见 [Save](SAVE.md#5-edwards-save-version1)。

Stage1 的 Edwards 日志和因子行携带 `curve_model=ed`，省略 Montgomery PARAM。Stage2 的曲线开始与结果事件传递相同模型身份；GUI 结果文件按该身份保存，旧版未带模型的 Stage2 日志沿用 Suyama PARAM0 解释。具体字段与管道规则见 [GUI 日志与结果](../usage/GUI.md#日志结果与监控)。完整大界限、随机 sigma 恢复和小素数回退的正确性证据见 [验收记录](../development/GPU_EDWARDS_STAGE1_STAGE2_DRAFT.md#2026-10-16大界限随机恢复与-gui-曲线身份验收)。

`GPU Edwards initialization` 为本次 Edwards 批次从指数准备开始、到设备可进入切片循环为止的墙钟时间，减去单独报告的字典构建墙钟；包括曲线构造/恢复、标量编码、设备分配和上传。`GPU Edwards dictionary` 为设备字典准备的墙钟，同时报告对应的 CUDA event 毫秒；binary 不构建字典，该项为零。两行均在进度条前输出。已完成的 checkpoint 不重建 NAF 设备字典，字典计时也为零。

`arithmetic api_seconds` 覆盖标量编码、设备初始化/分配、字典、切片、回传及进度回调，排除之前的指数/曲线构造和之后的 GCD/归一化/save。`dictionary_gpu_ms` 与 `loop_gpu_ms` 是 CUDA event 区间，可能包含首次模块加载或主机发射间隙，不能当作 profiler 的纯 kernel 时间。这些计时存在嵌套，不能与进程 wall 重复相加。

GPU Edwards 的 live 计时从设备准备回调开始；已恢复的累计位置只作为基准。每个完成切片产生推进量/毫秒样本，以最近最多 50 个样本的速度算术平均估算：每曲线耗时 = 总工作量 ÷ 平均速度 ÷ 曲线数；剩余时间 = 未完成工作量 ÷ 平均速度。速度样本包含设备切片与规范状态回传，排除主机进度回调中的输出及检查点写入；`elapsed` 则覆盖准备完成后的完整循环墙钟。NAF 使用列，binary 使用标量位。Montgomery 与 Edwards 共用 [GPU 文本格式](../../include/ecm/platform/stage1_progress.h)，Edwards 的计时和发布入口为 [Progress / prepared / progress](../../src/ecm/backend/cuda/stage1/edwards/workflow.cpp)，设备准备边界位于[binary/NAF 主机循环](../../src/ecm/backend/cuda/stage1/edwards/edwards_runtime.inl)。

GPU Edwards 在 NAF 列或二进制位的片边界更新共用进度输出，GUI 管道可读取百分比。当前32档构建的三曲线族同exe数据见[性能依据](../performance/STAGE1.md#gpu-edwards-细分实例三曲线族对照)，32档CPU末点、旧检查点跨容器恢复及Stage2交接见[细分实例验收](../development/GPU_EDWARDS_FINE_TIERS_DRAFT.md#2026-10-16完整正确性门禁)。冻结七档的binary/NAF、窗口/倍点对照及大界限证据分别见[性能参考](../performance/STAGE1.md#gpu-edwards-初始基准与-naf-对照)和[窗口优化草稿](../development/GPU_EDWARDS_STAGE1_NAF_DRAFT.md)，初始流水线见[开发草稿](../development/GPU_EDWARDS_STAGE1_STAGE2_DRAFT.md)；已测范围不等于全部发布架构或生产 B1/B2 的性能保证。

**切片与主机等待。** Edwards binary/NAF 入口明确设置阻塞设备调度，保留其他设备标志。NAF 先记录结束事件，通过设备同步等待，再检查事件、CGBN 报告并回读规范坐标；结束事件位置不包含主机唤醒后的工作。binary 原有设备同步也采用阻塞调度。当前每个 Stage1 调用只使用默认流，因此设备同步不会串行化另一个独立计算流；将来并行提交多流时须重新选择等待范围。

生产调用的切片参数为 0，由共用 `EcmEdwardsSlice` 从 32 列/位起步，按上一片时间调整到约 100 ms；范围为 16…4096，单次增长/缩小不超过 4 倍，末尾剩余量单独截断。NAF 使用 CUDA event 时间，binary 使用提交至设备同步返回的墙钟时间。正数 API 参数仍指定固定切片，便于独立对照。目标时间不是硬性截止时间，设备负载变化、边界限制、回读和检查点写入仍会影响实际响应。

每片仍先完成原有点运算、错误检查及坐标回读，再调用进度/停止回调；切片大小不保存进检查点，恢复后重新采样。标量、窗口、列位置和普通域累加点的存档合同保持一致。Windows 11 / WDDM、驱动 610.88 的 GPU1 实测中，仅阻塞事件标志未降低占用；原等待诊断与当前吞吐、恢复验证见 [主机等待草稿](../development/GPU_STAGE1_HOST_WAIT_DRAFT.md)。

### Montgomery resident 与 PRAC

resident 路径将点状态保持在 Montgomery 域，减少切片间往返转换。PRAC 将素数幂标量分解为规则链，以更多活跃点和控制流换取较少点运算；主机离线计划可缓存，并以分片保持检查点和进度语义。

PRAC 的 `single-compact` 专用体将 xADD 统一到一个调用点，倍点使用紧凑临时状态；终结规则和非终结规则共同维护同一组点角色。此路径仅在已编译的位宽/TPI/寄存器组合可选，缺失实例须报错。它不是发布默认算法。

## 位宽与资源

[tiers.def](../../src/ecm/backend/cuda/dispatch/tiers.def) 唯一维护位宽、TPI、实例组、Montgomery 支持/dev 标记、旧宏别名及两族 TPB。七组为：128…512、768…2048、2560…4096、4608…6144、6656…8192、9216…12288、13312…17408；17408 仅支持 Edwards。旧 `small` / `medium` 仍对应 512 / 1024，`ECM_TIERS_RESTRICTED` 与 `ECM_NO_PARAM2` 保持原筛选方式。类型兼容别名不代表已注册实例；能力查询以真实工厂返回值为准。

Edwards 选择实际编入的最小严格大于 N bits 的容器；Montgomery 要求容器至少为 N bits+6。受限构建中 PARAM2/3 仍先选择完整网格的首个适配档，该档未编入时会失败；PARAM0 可跳过缺失实例。`--gpu-info` 查询的是实际注册集合。统一清单不改变这个现存差异。Stage2 使用同文件中的 NW=4/8/16/32/64/128/256 容量表，按实际 64-bit 字数选择，不套用 Stage1 的进位余量。

Montgomery CUDA 容器需覆盖 N bits+6 个进位预留位。默认 2560…8192-bit 容器使用 TPI16，9216…16384 使用 TPI32；更小容器由 TPI4/8 分派。N=4423 bits 使用 4608/TPI16；N=8191 bits 因预留位需要 9216/TPI32，不能按输入是否超过 8192 判断。

默认 TPB128；Edwards 固定 128，Montgomery 保留构建宏 `ECM_TPB`。每 block 曲线数=TPB/TPI，grid=⌈曲线批量/(TPB/TPI)⌉。寄存器上限不改变 TPB；实际 blocks/SM 还受寄存器分配粒度、shared memory 和线程限制影响，由 CUDA occupancy API 查询。

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

- [ecm_stage1_exp.cpp](../../src/ecm/core/stage1/ecm_stage1_exp.cpp)：指数生成和缓存。
- [ecm_stage1_checkpoint_runtime.cpp](../../src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp)、[ecm_stage1_binary.h](../../src/ecm/storage/checkpoint/ecm_stage1_binary.h)：二进制外壳、长度边界、CRC32C、锁及轮换恢复。
- [ecm_stage1_legacy_gpu.h](../../src/ecm/storage/checkpoint/ecm_stage1_legacy_gpu.h)：旧 CUDA ladder V5 导入及逐 lane 模数校验。
- [ecm_edwards_gpu_checkpoint.h](../../src/ecm/backend/cuda/stage1/edwards/ecm_edwards_gpu_checkpoint.h)：GPU Edwards 二进制/旧文本载荷及状态验证。
- [adapter.cu](../../src/ecm/backend/cuda/stage1/montgomery/adapter.cu)：设备调度、同步及回读；[host_math.cpp](../../src/ecm/backend/cuda/stage1/montgomery/host_math.cpp)负责曲线布局与归一化，[host_io.cpp](../../src/ecm/backend/cuda/stage1/montgomery/host_io.cpp)负责保存点与输出。
- [arithmetic.cuh](../../src/ecm/backend/cuda/stage1/montgomery/arithmetic.cuh)、[instances.cuh](../../src/ecm/backend/cuda/stage1/montgomery/instances/instances.cuh)、[registry.cu](../../src/ecm/backend/cuda/stage1/montgomery/registry.cu)：设备 ladder、按组实例及薄发射器。
- [edwards_runtime.inl](../../src/ecm/backend/cuda/stage1/edwards/edwards_runtime.inl)、[instances.cuh](../../src/ecm/backend/cuda/stage1/edwards/instances/instances.cuh)：Edwards 主机循环与设备实例。
- [prac_runtime.inl](../../src/ecm/backend/cuda/stage1/montgomery/prac/prac_runtime.inl)：计划、策略、切片及恢复。
- [cgbn_stage1_prac_single_add.cuh](../../src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_single_add.cuh)：专用链规则。
- [cgbn_edwards_point.cuh](../../src/ecm/backend/cuda/stage1/edwards/point.cuh)：GPU Edwards 扩展坐标算术，由批量 worker 与独立探针共用。
- [ecm_mont_cpu.cpp](../../src/ecm/backend/cpu/stage1/montgomery/ecm_mont_cpu.cpp)、[ecm_edwards_cpu.cpp](../../src/ecm/backend/cpu/stage1/edwards/ecm_edwards_cpu.cpp)：CPU 实现。

使用和已测最佳范围见 [Stage1 使用](../usage/STAGE1.md)、[性能](../performance/STAGE1.md)。

完整数学见 [Montgomery / Suyama / PRAC](../reference/ECM_MONTGOMERY_MATH.md)、[Edwards](../reference/ECM_EDWARDS_MATH.md) 和[参数化](../reference/ECM_PARAMETERIZATIONS.md)。

## Montgomery 环境与诊断策略

[host_policy.cpp](../../src/ecm/backend/cuda/stage1/montgomery/host_policy.cpp)在普通 CXX 实现环境选项、PRAC 策略、计划缓存选择、切片目标、采样和窗口解析；读取仍在原执行位置。私有数据声明不包含 SDK，域编号在 [domain.h](../../src/ecm/backend/cuda/stage1/montgomery/prac/domain.h)集中维护，与实际设备分派共享。修改解析正文不重新编译 CUDA 协作；修改实际域值/数据合同仍使设备依赖失效。

诊断窗口的 GPU export、同步与回读在 CUDA，CSV 编码在 host_io.cpp。窗口部分乘积始终失败返回，不提交 NF/save 或内部完成结果。resident/PRAC 的资源、实际 dispatch、事件和定时循环继续属于 CUDA 模块，不要求脱离 CUDA。

## 应用群阶诊断

[group_order.cpp](../../src/ecm/application/cli/group_order.cpp)是 CLI 私有完整诊断，负责 gp 路径、脚本/进程、输出解析和命中报告。driver 在原时点检查 gp 可用性，在单任务结果已发布后逐命中调用；复合因子继续跳过。该实现仍以群阶和旧 32 位诊断 sigma 生成 go_min_bounds，未计算实际点阶/固定 torsion 的严格最小界限；CPU Edwards 的适用性也没有新增资格。数学条件与限制见 [重构分析](../development/MAIN_FLOW_REFACTOR_ANALYSIS_DRAFT.md#群阶诊断与最小界限公式)。
