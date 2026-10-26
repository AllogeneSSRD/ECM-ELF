# Stage1 算法与数据

## 曲线与标量

Suyama PARAM0 使用 sigma 构造 Montgomery 曲线和初始 X:Z 点。设 u=σ²−5、v=4σ，起点为 u³:v³；曲线参数通过模逆构造。不可逆分母需在目标 N 上求 GCD，不能略过。

光滑标量 s=∏(素数 ℓ≤B1) ℓ^⌊logℓ B1⌋；`lcm=12` 使用 12s。主机分段筛生成素数，分块累乘，用带身份/校验的缓存复用指数；指数缓存与曲线 checkpoint 分离。

指数缓存读入前核对声明位数、字数与已打开文件的实际长度，拒绝损坏长度后再分配载荷。并发写入使用各进程/序号独立的临时文件，再替换最终缓存；缓存校验失败可重建，缓存写失败不影响已经生成的有效指数。

CPU Edwards 路径使用自身曲线构造及扩展坐标公式；CPU Montgomery、CUDA 和 OpenCL 的点表示/算术域不同。只有投影或归一化后的数学点可跨实现对比，不能比较未经解码的字数组。

## 方法选择与输入校验

Stage1 单次运行和队列模式共用五个独立选择项：`curve`（Montgomery / Edwards）、`target`（CPU / GPU）、`impl`（自动 / IFMA SIMD / GMP 标量）、`reduction`（自动 / Montgomery / Mersenne fold）及 `lcm`（素数幂最小公倍积 / 其 12 倍）。默认是 Montgomery 曲线、GPU 执行；Edwards GPU 需要编入该模块的 CUDA 可执行文件，OpenCL 会拒绝该组合。`Auto` 在内部 CPU 实现及 SIMD 归约枚举中均为 0；调用现有 IFMA 接口时显式转换，避免枚举序号与实现常量耦合。TOML 和 CLI 均拒绝无效选择值，参数默认值及范围由 [配置定义](../../config/ecm_options.json) 维护。

TOML 使用 `curve`、`target`、`impl`、`reduction`、`lcm`；CLI 使用同名的双横线选项。`curve=mont|ed` 与 `target=cpu|gpu` 是不同轴；`impl=auto|simd|scalar` 只用于 CPU，`reduction=auto|montgomery|mersenne-fold` 控制 CPU SIMD 约减，`lcm=1|12` 控制 Montgomery 标量的倍数。Edwards 路径按曲线约定固定使用 48·lcm(1…B1)，因此拒绝 `lcm=12`；GPU 执行不接受非 auto 的 CPU 专用选项，显式归约必须同时选择 `impl=simd`，避免 auto 落到标量后忽略该设置。

GPU 参数化的语法值为 0、2、3，均属于 Montgomery 曲线；可执行范围取决于链接的后端和构建配置。CPU Montgomery 只实现 PARAM0，因此 CPU 单次运行只接受 `-sigma 0:s` 这种带参数化前缀的种子；Edwards 不接受参数化前缀，`--gpu-param` 仅适用于 GPU。启动曲线计算前，驱动查询当前 GPU 后端是否支持该参数化；OpenCL 仅实现 PARAM3，CUDA 的 PARAM2 与 PARAM3 还受是否编入相应内核及折叠构建配置约束。CUDA PARAM0 在曲线构造、检查点及 save 中保留完整 64 位 sigma，批次不得在 2⁶⁴ 处回绕。PARAM2/3 的固定 sigma 及曲线批次须满足 sigma+曲线数≤2³²；不再截取低 32 位继续计算。参数化与 sigma 的冲突、后端不支持项在初始化 GPU 前报错。选中的队列任务 sigma 不符合要求时输出来源、值和范围，标为错误行并继续；队列 I/O 或标错失败时停止。

`ECMSTAGE1` 作为 save 型 Stage1 输入时，文件名前缀决定该任务的曲线构造，末尾字段决定 B1；覆盖 TOML 后再校验后端能力和 sigma 范围。`ECMSTAGE2` 只由独立 Stage2 程序领取。每条 Stage1 任务复制基础选项，防止混合队列之间传播覆盖。Edwards 前缀同时恢复其固定标量约定；执行目标保持 TOML 设置。名称、旧 save 历史复制及交接合同见[队列输入](../usage/STAGE1.md#程序与输入)。

这一步统一的是选择项解析与能力检查。CPU 曲线实现、GPU 内核及各后端 checkpoint 保持各自职责；后端返回拥有的逐曲线结果；task 调用保存模块编码、发布并取得明确区间。入口见 [selection.h](../../include/ecm/stage1/selection.h)、[main.cpp](../../src/main.cpp) 和 [ecm_backend.h](../../include/ecm/stage1/gpu_backend.h)。

## 单任务编排与资源拥有权

直接输入和队列调用同一个 [run_stage1_task](../../src/ecm/stage1/task.cpp)。它依次校验任务元数据与输出归属、打开批次、接纳旧 save 历史、直接调用选定的后端入口、核对逐曲线身份，编码并发布最终 save，再将整批结果交给 publish_stage1_batch。发布器在内存中按 uint64 sigma 排序和聚合，最终内部结果与提交文件均成功后才结束批次。因子在后端安全位置验证，最终随整批发布；队列完成和跨阶段交接协议在任务之外。去重、双文件恢复及 Stage2 账本兼容见[结果输出](../usage/ECM_RESULTS.md#去重批次与恢复)。

[公开请求与结果](../../include/ecm/stage1/task.h)只使用普通 C++ 和 GMP。Stage1Request 在同步调用期间借用模数、命名和已解析选项，选项分为执行偏好、身份和持久化策略三组。TaskResult 独立拥有逐曲线结果，禁止复制并允许移动，通过只读访问器取数；不保留请求引用，不公开原始分配或进度字段。状态区分完成、停止和失败，并记录准备、执行、持久化、发布或设备执行失败。CUDA 明确失败优先于同时收到的停止；CPU/OpenCL 的旧标记仍通过兼容映射接纳。设备执行失败使队列停止并保留原任务，不继续使用可能失效的设备 context。

task 在栈上直接拥有 `ecm_stage1_checkpoint::Context`。Context 实际保存 records、sigma、恢复候选、备份策略与批次独占锁，通过第三个参数借给同步后端；回调只借用本次 Context，没有进程级或 thread_local 的活跃批次。begin 前建立清理责任，部分开始失败也释放锁；成功结束才清除 checkpoint，失败或停止保留恢复材料。计时和总曲线数属于执行副本。目前生产流程仍是单进程单活跃任务。

应用 [main.cpp](../../src/main.cpp)负责模式、进程启动、worker资源与队列循环。归一化当前任务后，通过 [select_stage1_entry](../../src/ecm/stage1/execution.cpp)选择一个普通函数入口；选择不执行CUDA，不改变任务。task随后直接调用选定入口，不经过另一层执行转发。构建身份、默认NAF和CUDA延迟验证能力仍由应用组装。

[私有计算合同](../../src/ecm/stage1/execution_contract.h)以 ExecutionOptions 保存执行副本，以 ComputationResult 返回完成/停止/失败及来源、拥有的逐曲线结果。每条结果区分 Factor、Stage2Point、CompletedWithoutPoint、Degenerate、Failed。CPU 与 CUDA workflow 直接构造拥有结果；Montgomery 设备数组仅在该 workflow 内适配并移动 GMP 值，LegacyExecutionResult 留给 OpenCL 兼容边界。task 不从旧返回码、flags 或数组猜测执行状态。TaskResult::State 另保存最终发布状态、明确的 save Range 和队列尚未提交的 checkpoint 责任；计算完成与文件发布完成是两个事实。

任务与真实持久化模块可通过普通 CXX 独立编译验证。25 个任务模式覆盖拥有权、失败／停止优先级、锁释放、资源别名和历史快照；生产程序另有实际后端恢复和发布回归。构建方式见 [构建说明](../usage/BUILD.md)。批次清理见 [stage1_checkpoint.cpp](../../src/io/stage1_checkpoint.cpp)，格式与恢复接纳见 [Checkpoint](CHECKPOINT.md)。

保存命名和历史导入集中在 [save.cpp](../../src/io/save.cpp)。Stage1 通过 `save_naming` 将已归一化选项投影为窄的 `ecm_save::Naming` 数据；IO 只消费曲线前缀、目标身份、模数与 B1，不依赖 Stage1Options。默认和自定义模板的前缀及字节格式保持一致。

接纳旧历史时，IO 读取一次来源字节，在这份快照上检查 METHOD／PARAM，再在目标发布锁内重查目标是否为空并持久写入同一份字节；保留原历史及已经存在的目标。结果输出共用 [write_atomic](../../src/io/files.cpp)：先写临时文件并 flush，再执行有界替换；结果临时后缀仍为 `.new`。双结果的恢复与不确定发布处理见[结果输出](../usage/ECM_RESULTS.md)。

## 应用输入与队列编排

main 按当前任务组装计算入口、身份和默认 NAF 依赖，单次与队列模式共用 task。三个来源的解析、默认与覆盖、拥有的完整输入分别集中在 [io](../../src/io/)；唯一队列循环和完成顺序在 main。解析不改动运行时单例，归一化不写队列/日志/同步文件；完整流程和目录见 [架构与术语](OVERVIEW.md)。

task接收后端的拥有结果，调用纯编码器生成save，在发布后核对本批 N、B1、curve/param、sigma 与 checksum，再发布最终结果。save发布器显式返回Range，没有thread_local Capture。队列随后强制重读 save，核对发布时确认的连续 METHOD 区间。成功后先持久保存 P95/本地 Stage2 待交接责任，再保存完成记录、发布 finished 并按原任务票据原子推进队列，清理完成 checkpoint，最后执行备份与交接重试。计算、准备、设备或发布失败/停止按来源处理；选中的非法输入标错并继续，标错/I/O/锁失败终止。

完成记录保留队列提交前后版本证据，恢复不再执行已经完成的数学工作；若外部编辑使提交状态不可判定，保留结果、checkpoint 和完成记录并停止，不猜测下一条重复任务。未来任务的正常新增、删除、重排在运行期间仍可进行。P95 的文件先就绪、任务后发布及交接消费限制见 [使用说明](../usage/STAGE1.md#prime95-持久交接)。

## 选择词汇与配置依赖

[selection.h](../../include/ecm/stage1/selection.h)公开曲线/目标/CPU 实现/归约/标量构造类型，以及这些词汇的解析、合法组合和名称声明；11 个实际函数集中在 [selection.cpp](../../src/ecm/stage1/selection.cpp)。旧应用私有选择头已删除，不保留转发。字段、枚举值、ASCII 解析、sigma 参数冲突规则与名称保持。

task 的默认构造从 [生成默认值](../../src/ecm/platform/generated/configuration_defaults.h)获取同一 JSON 定义，不包含应用的 TOML 解析/完整绑定。准备模块也只依赖公共选择合同。应用完整生成绑定引用这份默认值，未复制默认值来源或新增配置服务。真实无 SDK 任务编译记录覆盖共享 selection 实现，且不包含 ecm_toml.h/ecm_config_generated.h。

## 标量和进度的实现隔离

[scalar.h](../../include/ecm/stage1/scalar.h)统一声明 LCM 标量与主机 NAF 位图；构造、位读取和 digit 方法全部在现有 [ecm_stage1_exp.cpp](../../src/ecm/stage1/scalar.cpp)，没有新增标量转发层或第二个实现 TU。NAF 的字段、布局、窗口范围和编码保持，设备只消费已有位图/数据合同。

[stage1_progress.h](../../include/ecm/stage1/progress.h)只声明共享 GPU screen 格式及 SpeedWindow 方法；实现位于 [stage1_progress.cpp](../../src/ecm/stage1/progress.cpp)。50 样本窗口的布局、统计公式与 stdout 协议保持，不新增 Pimpl 分配。CUDA 包含声明；修改格式或速度统计函数体只需重编对应 CXX。Montgomery 对该头的多余显式 OBJECT_DEPENDS 已移除，真实 include 依赖仍正常校验。

共享选择、主机准备、CPU/GPU host workflow、Montgomery 构造/codec 和结果发布已提取；队列读写已集中于主机读写 Module；Montgomery 环境策略、私有执行合同和应用 PARI 已收口。实际设备同步、回读、资源管理继续依赖 CUDA。独立主机构建的真实依赖记录同时覆盖 task、标量与进度实现，不包含 CUDA SDK 头。

## CPU 执行与共享主机准备

CPU 的 [Edwards workflow](../../src/ecm/edwards_stage1/cpu/workflow.cpp) 和 [Montgomery workflow](../../src/ecm/montgomery_stage1/cpu/workflow.cpp) 接收统一 Stage1Request、执行副本及借用的 Context，直接返回 ComputationResult。新计算不先编码 save 文本再解析；Edwards 只在恢复边界解码并校验兼容的 DONE_V1，空点记录转为 CompletedWithoutPoint。Montgomery 在释放局部数组前移动末点。共享 ISA 探测、CPU 绑定和进度显示位于 execution_support；线程寿命、GMP 临时量及进度条由作用域管理，工作线程异常传回 workflow。普通基线 CXX 不继承 AVX512 编译设置。最终 save 和结果由 task 发布。

[preparation.cpp](../../src/ecm/stage1/preparation.cpp)集中 CPU/GPU 共用的标量准备入口、随机 sigma、保存路径、固定 sigma 范围检查及主机时钟/终端检测；私有 preparation.h 只声明实际共用函数，不增加 wrapper 类。归一化将指数缓存目录固定到任务的 persistence 选项；空目录明确禁用缓存，不在执行时重新套默认值。CLI 或配置的 off/none/0/- 保留禁用语义，PRAC 默认计划目录使用同一显式目录。随机状态仍是同一个进程级状态。CPU 模块与准备实现已用真实 MSVC 依赖记录确认不包含 CUDA SDK 头；GPU host workflow、分派与队列已提取到各自普通 CXX。

各工作流负责计算与安全点DONE/checkpoint；任务层负责最终save、结果发布和批次关闭。实际设备同步和回读继续依赖 CUDA。

## 唯一执行分派与 GPU 主机流程

[execution.cpp](../../src/ecm/stage1/execution.cpp)按实际任务的target与曲线选择入口，Montgomery各param均选择同一个run_mont_stage1。CUDA由构建显式定义ECM_HAS_CUDA_STAGE1；[execution.h](../../src/ecm/stage1/cuda_execution.h)声明两个同签名入口。两条workflow线性编排准备、恢复候选验证、布局接纳或新曲线构造、设备执行、结果收集；不拥有最终save/results发布或队列。

[Montgomery workflow](../../src/ecm/montgomery_stage1/cuda/workflow.cpp)直接调用CUDA曲线运行时；factory提供设备准备、查询和兼容入口。param在归一化后固定，新sigma只在没有接纳恢复种子时生成。ladder的[构造描述](../../src/ecm/montgomery_stage1/cuda/construction.h)集中param0/2的7-word布局、param3的5-word布局及末点偏移，checkpoint与runtime共用；resident/PRAC仍使用自己的布局合同。不同param不建立完整独立workflow，也不在同一批次混用。

[Edwards workflow](../../src/ecm/edwards_stage1/cuda/workflow.cpp)管理Batch、恢复、构造、进度和末点分类；同目录checkpoint.cpp拥有专用codec。两种曲线save_checkpoint均以bool表示持久提交成功；decode解释载荷并返回RestoreResult，候选接纳和布局检查仍留在实际拥有这些约束的位置。中途checkpoint参与安全点，最终发布失败仍可恢复已完成计算。

CUDA关闭时选择[OpenCL Montgomery workflow](../../src/ecm/montgomery_stage1/opencl/workflow.cpp)。其设备hook和算术保持，最终结果同样交给task。CPU/OpenCL旧状态映射集中在兼容转换；明确失败优先于同时收到的停止信号。任务层按计算/发布阶段归类逃逸异常。

这些主机 CPP 的实际独立编译依赖没有 CUDA SDK 头。CUDA runtime、同步、设备资源与回读实现保留现有 CUDA 依赖；Montgomery 构造、结果及 checkpoint/screen/CSV 实现也已提取到普通 CXX；Edwards 运行入口的校验/选择/错误适配也使用普通 CXX。设备协作继续由 runtime.cu 持有。

## CUDA 故障与构造期结果

[Montgomery runtime](../../src/ecm/montgomery_stage1/cuda/runtime.h)返回完成、主动停止、采样截止或失败；[运行结果合同](../../src/ecm/stage1/cuda_runtime.h)区分实际失败来源。旧 C 入口只负责兼容适配。CUDA/CGBN 检查通过异常返回运行边界，设备缓冲、事件、报告、诊断文件和主机数组由 [资源拥有者](../../src/ecm/montgomery_stage1/cuda/resources.cuh)释放。调用方不假定设备错误后的 context 可以继续使用。Edwards 的已有缓冲、报告及事件拥有者禁止复制，设备错误同样回到任务层。

PARAM0/PARAM2 构造遇到模逆失败时计算 GCD：真因子校验后记录为该曲线的factor结果，随整批最终结果发布；GCD=N 作为退化构造拒绝。设备批次不压紧索引：已找到因子的通道使用无害恒等状态，末点处理读取已保存的真因子，不能将该通道当作可用于 Stage2 的点。恢复时重新推导构造期因子元数据；旧载荷格式不变。有效曲线的公式与 sigma 身份不受影响；PARAM2 低层也拒绝 sigma<2，而不改用另一 sigma。

构造布局、主机数组及结果分配检查尺寸；两曲线的结果数组在计算前取得完整所有权，后续失败仍由结果对象清理。临时参数和 GMP 标量也由工作流局部对象持有。共享 storage 和 task 仍拥有最终清理权限；普通成功路径的底层清理请求不会提前删除尚未封存的批次。

## 运行选择与实际布局

任务在建立批次身份前，将 GPU 算法选择保存到私有 ExecutionOptions；CUDA workflow 使用同一值。Montgomery 的 host_policy 在每次运行开始持有本次环境选项和默认计划缓存路径，保留“未设置”与“显式空值”的区别。其后的算法、PRAC 变体、寄存器、TPI、常量、窗口、采样和输出选择只读取该快照，不再重复查询进程环境。它是私有后端输入，不是新的配置系统或 factory。

请求偏好在现有标量/档位阶段校验；存档先核对算法和布局约束，实际可用档位再确定不可变 MontResolvedKernel，直接供设备分配和发射使用。数学身份仍为曲线、参数化和 sigma；bits/TPI/域/PRAC 计划是执行或恢复约束。EdwardsOptions 与恢复后的批次模式决定 binary/NAF、窗口和倍点算法，统一 runtime 一次选择实际档位；完成普通坐标档直接走主机校验路径。参数非法或当前构建缺档明确拒绝，不修改数学身份以继续运行。

## Montgomery 主机数学与存档输出

[host_math.cpp](../../src/ecm/montgomery_stage1/cuda/host_math.cpp)集中 param0/2/3 初始状态、param2 Jacobian 辅助计算、标量打包、np0、普通整数解包与最终结果处理。跨设备协作所需入口经私有 host_math.h 声明；内部转换和算术 helper 留在 CPP 内部。输出仍调用同一 factory 中的 outputf/test_verbose，verbose 状态没有复制。FULL_BUILD 条件下的诊断分支和断言语义保持。

[checkpoint.cpp](../../src/ecm/montgomery_stage1/cuda/checkpoint.cpp)持有 ladder V5 与 resident/PRAC 的编码、读取、校验及保存路径；[checkpoint.h](../../src/ecm/montgomery_stage1/cuda/checkpoint.h)声明接口与载荷布局。[output.cpp](../../src/ecm/montgomery_stage1/cuda/output.cpp)持有 TTY/screen、节流显示判断与 CSV，output.h 只声明输出接口。两种曲线的生产恢复使用 Missing、Resumed、AlreadyComputed、Incompatible、Corrupt、IoFailure；离线 decode 不读取文件或发布结果。外层候选通过本后端校验后才提交种子和载荷，详见 [Checkpoint](CHECKPOINT.md)。

Edwards [batch.h](../../src/ecm/edwards_stage1/cuda/batch.h)只描述主机状态；[host_math.cpp](../../src/ecm/edwards_stage1/cuda/host_math.cpp)负责构造；[checkpoint.cpp](../../src/ecm/edwards_stage1/cuda/checkpoint.cpp)负责 V3 和旧文本读取及状态资格。GMP 数组拥有权和点不变量属于 [核心曲线数学](../../src/ecm/curve/edwards.h)，批次模型不再引入 save/文件实现。workflow 直接调用实际 save 发布函数。

CUDA workflow 将 sigma 选择与曲线参数化分开：先恢复并校验当前/兼容存档，采用载荷中的 sigma；没有算术载荷时查询已持久化种子；只有缺失种子的新任务才调用随机生成器。随后使用确定的 sigma 构造曲线。恢复校验所需的基点重建仍使用存档 sigma，不生成新种子。Montgomery 保持连续批次，Edwards 保持逐曲线种子；结果、日志和 PARAM2/3 save 均使用实际采用的 sigma。

两种 CUDA workflow 均在计算前拥有结果数组。Edwards 末点验证和归一化写入独立结果，不改写用于恢复的原始批次状态；构造期真因子校验后保存在拥有结果中，最终随整批发布。完成状态保存成功后才发布最终 save，任务封存结果后才清除恢复材料。周期保存失败作为持久化失败返回，优先于同时收到的停止；最后一片已完成时继续完成当前批次的结果发布。

磁盘版本、字段顺序和 size_t/time_t 布局保持。runtime.cu 保留 CGBN report、设备分配、上传/回读、事件/同步、计算循环和资源释放；档位注册与实际分派归 registry.cu，resident/PRAC 由独立 prac/runtime.cu 实现。global event 与标量上传继续先于 checkpoint 恢复和 CPU 构造。上述主机 CPP 均以普通 CXX 编译且不包含 CUDA SDK；接口或实际设备协作改动仍会使其真实依赖失效。增量编译最终门禁与当前实施进度见 [实施草稿](../development/CUDA_STAGE1_UNIFICATION_DRAFT.md)。

## GPU 点乘

### CUDA 模块边界

Montgomery resident/PRAC 的设备协作位于独立的 `montgomery/prac/runtime.cu`，依赖通过私有 runtime.h、registry.cuh 和各主机模块接口显式声明；不依靠父翻译单元先定义函数。catalogue/query 与 tier dispatch 留在 registry.cu，资源拥有者留在 resources.cuh。

CUDA 的发射描述、设备类型与位宽清单均为 `src/ecm/cuda/` 内的私有实现；其中 `dispatch/tiers.def` 唯一维护位宽。公开任务与选择合同位于 `include/ecm/service/` 和 `include/ecm/core/`。`src/ecm/cuda/factory.cpp` 负责后端能力查询；`src/ecm/cuda/stage1/` 负责 GMP 编解码、分配释放、错误检查、切片、回调、计时和恢复。Montgomery 驱动为 `montgomery/runtime.cu`；Edwards 的参数、规范状态、档位选择及错误适配统一位于 `runtime.cpp`，生产 workflow 直接调用 execute_edwards_batch；binary/NAF 旧入口仅为独立算术探针兼容。设备准备位于 `edwards_runtime.cu`，两种执行模式的主机循环保持独立。

设备算术与薄发射器位于各曲线目录，独立实例放在对应 `instances/`。Montgomery 的 PARAM0/2/3 各有七组实例；Edwards binary/NAF 共用七组实例。设备实例闭包不包含项目主机驱动，也不接收 GMP 对象、进度回调或 checkpoint 对象。第三方 CGBN 在 NVCC 主机解析阶段需要 GMP 类型，由 `src/ecm/cuda/arithmetic/cgbn/types.cuh` 的主机条件包含适配，设备运算接口仍使用原始缓冲和标量。

local/dev/release默认启用 `ECM_CUDA_ISOLATE_TIERS`，由 `tools/build/internal/cuda_modules.cmake` 从同一 `tiers.def` 派生32个Edwards容器模块及31×3个Montgomery参数化/容器模块；local/dev可用 `-KernelLayout grouped` 保留七组实例备选。生成文件只选择曲线/param/容器，内核及设备辅助函数的实现仍在同一编译单元可见，不拆散算术调用链。全档枚举读取纯元数据，任务选定宽度后才调用对应工厂并查询occupancy，避免信息展示触及所有设备函数。Edwards binary/NAF共享该容器模块；sigma仍是运行时参数，不生成sigma专用实例。旧NVCC不支持每核 `__maxnreg__` 时，Montgomery隔离单元按当前位宽政策设置56/128/255的编译器寄存器预算。实际按需JIT范围及驱动限制见[构建与发布](../usage/BUILD.md#完整发布与按需加载)。

依赖方向为：任务/旧执行适配 → CUDA 私有 API → CUDA 主机实现 → 私有发射接口 → 设备实例/算术；主机与设备都可读取 `tiers.def`。实际编译源由 `tools/build/internal/cuda_sources.cmake` 维护。PRAC 设备源码、主机计划与运行时统一位于 `src/ecm/montgomery_stage1/cuda/prac/`，仅在显式启用时链接 `ecm_cuda_prac` 对象目标。Stage2 的主机、算术、NTT 及实验边界见 [Stage2 模块职责](STAGE2.md#cuda-模块边界)。主机等待探针通过 `src/ecm/edwards_stage1/cuda/host_wait.h` 与唯一测试适配调用真实 Edwards 运行时，探针入口不再包含私有主机实现。生产闭包禁止反向包含工具与实验源码。

Montgomery 默认 ladder 按 s 的比特进行差分加法和倍点，状态由相邻点及固定差点组成。GPU 以切片推进，避免一个超长 kernel 占用整段 B1。每曲线的末点先解码、检查，求 Z 与 N 的 GCD；可逆时归一化 X=X/Z mod N，再输出 save。

### GPU Edwards

[CUDA NAF 批量入口](../../src/ecm/edwards_stage1/cuda/runtime.cpp)以 CGBN Montgomery 域的 `(X,Y,Z,T)` 执行 `48·lcm(1…B1)`。`naf_w=0` 在 GPU 选择窗口 6，显式允许 2..12；每曲线在 GPU 预计算 2^(w−2) 个奇数倍投影点，不做字典模逆。主机以两份紧凑位图保存非零列和有符号索引，生产调用使用自适应切片，按实测执行时间调整每片 NAF 列数。设备累加点在算术域跨片保留，只回传四个规范普通坐标供回调及检查点使用。连续倍点省略暂时不用的 T，在加法前及片边界恢复完整 `XY=ZT`。入口与位图定义见 [标量与 NAF 合同](../../include/ecm/stage1/scalar.h)，设备字典与切片见 [cgbn_edwards_naf.cuh](../../src/ecm/edwards_stage1/cuda/naf.cuh)。

容器选择要求 N 至少留一位余量，选择本次构建中最小的可用容器。完整网格为128/192/256/384/512（TPI4）、768..2048间隔256（TPI8）、2560..8192间隔512（TPI16）、9216..17408间隔1024（TPI32），共32档，TPB128。Montgomery仍预留六位，不能把两个模型的边界条件互换。`ECM_TIERS`同时筛选Edwards实例；缺档时使用更大可用档，超过本次构建可用上限时报错。日志的 `GPU Edwards geometry` 给出实际容器、TPI及网格，查询入口为 `ecm_cuda_edwards_tier`。

二进制与NAF的实例共享[位宽清单](../../src/ecm/cuda/dispatch/tiers.def)，核函数和七组实例化文件位于 `src/ecm/edwards_stage1/cuda/instances/`，按TPI及位宽范围拆分并行编译；`src/ecm/cuda/stage1/`保留输入校验、分派和主机驱动，相关 API 为曲线目录内的私有头。模板算术与切片逻辑见[共用实例实现](../../src/ecm/edwards_stage1/cuda/instances/instances.cuh)。字典显存为曲线数×容器字节数×4坐标×2^(w−2)，分配失败会报错并保留未完成队列，不静默改变显式窗口。

主循环默认倍点为 HWCD；`ECM_EDWARDS_DBL_ALGO=4|5` 选择 HWCD（内部编号 4）或 2XY 多项式（编号 5），字典构造使用 HWCD。后者采用 Prime95 算法 5 的代数形式，CGBN 不模拟其 FFT 复合指令；内部编号 4 不代表 Prime95 算法 4。参考版本与运算计数见 [数学说明](../reference/ECM_EDWARDS_MATH.md#21-prime95-倍点算法编号与-fft-条件)。`ECM_EDWARDS_STAGE1_ALGO=binary|naf` 可显式选择 [原二进制路径](../../src/ecm/edwards_stage1/cuda/runtime.cpp)或 NAF，用于对照；binary 不接受窗口或 NAF 倍点设置。未指定时使用 NAF，恢复时遵从已保存的模式。

[曲线构造](../../src/ecm/curve/edwards.h)先在辅助 Weierstrass 曲线上用 Jacobian 坐标计算 sigma 倍点，再构造 Atkin–Morain 参数。模逆失败在实际目标 N 上区分真因子与退化曲线，同时检查 d(1−d) 的单位性和基点方程。末点检查 `XY=ZT` 与 Edwards 方程，用 `(Z+Y):(Z−Y)` 转到 Montgomery，检查分母 GCD 后归一化；真因子独立标记。有效恒等点且 GCD=N 时没有可用 Stage2 点，该曲线完成但不写点记录。

[worker 接入](../../src/ecm/edwards_stage1/cuda/workflow.cpp)支持 CLI 和队列、周期及安全停止检查点。当前 GPU Edwards 使用 V3 二进制载荷，保存完整 N、算法、进度、各 sigma 和普通整数状态；指数只保存位数与指纹。恢复重建基点、核对点不变量并采用已保存的模式，显式不匹配选择报错。载荷不绑定 TPI，未完成状态恢复按 N 重新选择容器。已计算完成的普通坐标状态经完整主机校验后直接补发布，不进行设备准备、档位查找、上传或字典分配。NAF 列数与 binary 位数不能互换。完整字段、兼容读取和恢复资格统一见 [Checkpoint](CHECKPOINT.md#42-gpu-edwards-v3)。

CPU/GPU Edwards 共用完成点文本，可由 CUDA Stage2 和 [CPU Stage2](../usage/EDWARDS_CPU_STAGE2.md) 消费；sigma 不能按 Suyama PARAM0 解释。字段、状态、校验及外部程序能力统一见 [Save](SAVE.md#5-edwards-save-version1)。

Stage1 的 Edwards 日志和因子行携带 `curve_model=ed`，省略 Montgomery PARAM。Stage2 的曲线开始与结果事件传递相同模型身份；GUI 结果文件按该身份保存，旧版未带模型的 Stage2 日志沿用 Suyama PARAM0 解释。具体字段与管道规则见 [GUI 日志与结果](../usage/GUI.md#日志结果与监控)。完整大界限、随机 sigma 恢复和小素数回退的正确性证据见 [验收记录](../development/GPU_EDWARDS_STAGE1_STAGE2_DRAFT.md#2026-10-16大界限随机恢复与-gui-曲线身份验收)。

`GPU Edwards initialization` 为本次 Edwards 批次从指数准备开始、到设备可进入切片循环为止的墙钟时间，减去单独报告的字典构建墙钟；包括曲线构造/恢复、标量编码、设备分配和上传。`GPU Edwards dictionary` 为设备字典准备的墙钟，同时报告对应的 CUDA event 毫秒；binary 不构建字典，该项为零。两行均在进度条前输出。已完成的 checkpoint 不重建 NAF 设备字典，字典计时也为零。

`arithmetic api_seconds` 覆盖标量编码、设备初始化/分配、字典、切片、回传及进度回调，排除之前的指数/曲线构造和之后的 GCD/归一化/save。`dictionary_gpu_ms` 与 `loop_gpu_ms` 是 CUDA event 区间，可能包含首次模块加载或主机发射间隙，不能当作 profiler 的纯 kernel 时间。这些计时存在嵌套，不能与进程 wall 重复相加。

GPU Edwards 的 live 计时从设备准备回调开始；已恢复的累计位置只作为基准。三种 Montgomery 运行路径及 Edwards 均复用已有 SpeedWindow，各自保持原计时区间、进度单位和切片策略。每个完成切片产生推进量/毫秒样本，以最近最多 50 个样本的速度算术平均估算：每曲线耗时 = 总工作量 ÷ 平均速度 ÷ 曲线数；剩余时间 = 未完成工作量 ÷ 平均速度。速度样本包含设备切片与规范状态回传，排除主机进度回调中的输出及检查点写入；`elapsed` 则覆盖准备完成后的完整循环墙钟。NAF 使用列，binary 使用标量位。Montgomery 与 Edwards 共用 [GPU 文本格式](../../include/ecm/stage1/progress.h)，Edwards 的计时和发布入口为 [Progress / prepared / progress](../../src/ecm/edwards_stage1/cuda/workflow.cpp)，设备准备边界位于[binary/NAF 主机循环](../../src/ecm/edwards_stage1/cuda/edwards_runtime.inl)。

GPU Edwards 在 NAF 列或二进制位的片边界更新共用进度输出，GUI 管道可读取百分比。当前32档构建的三曲线族同exe数据见[性能依据](../performance/STAGE1.md#gpu-edwards-细分实例三曲线族对照)，32档CPU末点、旧检查点跨容器恢复及Stage2交接见[细分实例验收](../development/GPU_EDWARDS_FINE_TIERS_DRAFT.md#2026-10-16完整正确性门禁)。冻结七档的binary/NAF、窗口/倍点对照及大界限证据分别见[性能参考](../performance/STAGE1.md#gpu-edwards-初始基准与-naf-对照)和[窗口优化草稿](../development/GPU_EDWARDS_STAGE1_NAF_DRAFT.md)，初始流水线见[开发草稿](../development/GPU_EDWARDS_STAGE1_STAGE2_DRAFT.md)；已测范围不等于全部发布架构或生产 B1/B2 的性能保证。

**切片与主机等待。** Edwards binary/NAF 入口明确设置阻塞设备调度，保留其他设备标志。NAF 先记录结束事件，通过设备同步等待，再检查事件、CGBN 报告并回读规范坐标；结束事件位置不包含主机唤醒后的工作。binary 原有设备同步也采用阻塞调度。当前每个 Stage1 调用只使用默认流，因此设备同步不会串行化另一个独立计算流；将来并行提交多流时须重新选择等待范围。

生产调用的切片参数为 0，由共用 `EcmEdwardsSlice` 从 32 列/位起步，按上一片时间调整到约 100 ms；范围为 16…4096，单次增长/缩小不超过 4 倍，末尾剩余量单独截断。NAF 使用 CUDA event 时间，binary 使用提交至设备同步返回的墙钟时间。正数 API 参数仍指定固定切片，便于独立对照。目标时间不是硬性截止时间，设备负载变化、边界限制、回读和检查点写入仍会影响实际响应。

每片仍先完成原有点运算、错误检查及坐标回读，再调用进度/停止回调；切片大小不保存进检查点，恢复后重新采样。标量、窗口、列位置和普通域累加点的存档合同保持一致。Windows 11 / WDDM、驱动 610.88 的 GPU1 实测中，仅阻塞事件标志未降低占用；原等待诊断与当前吞吐、恢复验证见 [主机等待草稿](../development/GPU_STAGE1_HOST_WAIT_DRAFT.md)。

### Montgomery resident 与 PRAC

resident 路径将点状态保持在 Montgomery 域，减少切片间往返转换。PRAC 将素数幂标量分解为规则链，以更多活跃点和控制流换取较少点运算；主机离线计划可缓存，并以分片保持检查点和进度语义。

PRAC 的 `single-compact` 专用体将 xADD 统一到一个调用点，倍点使用紧凑临时状态；终结规则和非终结规则共同维护同一组点角色。此路径仅在已编译的位宽/TPI/寄存器组合可选，缺失实例须报错。它不是发布默认算法。

## 位宽与资源

[tiers.def](../../src/ecm/cuda/dispatch/tiers.def) 唯一维护位宽、TPI、实例组、Montgomery 支持/dev 标记、旧宏别名及两族 TPB。七组为：128…512、768…2048、2560…4096、4608…6144、6656…8192、9216…12288、13312…17408；17408 仅支持 Edwards。旧 `small` / `medium` 仍对应 512 / 1024，`ECM_TIERS_RESTRICTED` 与 `ECM_NO_PARAM2` 保持原筛选方式。类型兼容别名不代表已注册实例；能力查询以真实工厂返回值为准。

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

- [ecm_stage1_exp.cpp](../../src/ecm/stage1/scalar.cpp)：指数生成和缓存。
- [stage1_checkpoint.cpp](../../src/io/stage1_checkpoint.cpp)、[ecm_stage1_binary.h](../../src/io/stage1_binary.h)：二进制外壳、长度边界、CRC32C、锁及轮换恢复。
- [ecm_stage1_legacy_gpu.h](../../src/ecm/montgomery_stage1/cuda/checkpoint.cpp)：旧 CUDA ladder V5 导入及逐 lane 模数校验。
- [checkpoint.cpp](../../src/ecm/edwards_stage1/cuda/checkpoint.cpp)：GPU Edwards 二进制/旧文本载荷及状态验证。
- [runtime.cu](../../src/ecm/montgomery_stage1/cuda/runtime.cu)：设备调度、同步及回读；[host_math.cpp](../../src/ecm/montgomery_stage1/cuda/host_math.cpp)负责曲线布局与归一化，[checkpoint.cpp](../../src/ecm/montgomery_stage1/cuda/checkpoint.cpp)负责保存点，output.cpp 负责输出。
- [arithmetic.cuh](../../src/ecm/montgomery_stage1/cuda/arithmetic.cuh)、[instances.cuh](../../src/ecm/montgomery_stage1/cuda/instances/instances.cuh)、[registry.cu](../../src/ecm/montgomery_stage1/cuda/registry.cu)：设备 ladder、按组实例及薄发射器。
- [edwards_runtime.inl](../../src/ecm/edwards_stage1/cuda/edwards_runtime.inl)、[instances.cuh](../../src/ecm/edwards_stage1/cuda/instances/instances.cuh)：Edwards 主机循环与设备实例。
- [PRAC runtime.cu](../../src/ecm/montgomery_stage1/cuda/prac/runtime.cu)：计划、策略、切片及恢复。
- [cgbn_stage1_prac_single_add.cuh](../../src/ecm/montgomery_stage1/cuda/prac/cgbn_stage1_prac_single_add.cuh)：专用链规则。
- [cgbn_edwards_point.cuh](../../src/ecm/edwards_stage1/cuda/point.cuh)：GPU Edwards 扩展坐标算术，由批量 worker 与独立探针共用。
- [ecm_mont_cpu.cpp](../../src/ecm/montgomery_stage1/cpu/ecm_mont_cpu.cpp)、[ecm_edwards_cpu.cpp](../../src/ecm/edwards_stage1/cpu/ecm_edwards_cpu.cpp)：CPU 实现。

使用和已测最佳范围见 [Stage1 使用](../usage/STAGE1.md)、[性能](../performance/STAGE1.md)。

完整数学见 [Montgomery / Suyama / PRAC](../reference/ECM_MONTGOMERY_MATH.md)、[Edwards](../reference/ECM_EDWARDS_MATH.md) 和[参数化](ECM_PARAMETERIZATIONS.md)。

## Montgomery 环境与诊断策略

[host_policy.cpp](../../src/ecm/montgomery_stage1/cuda/host_policy.cpp)在普通 CXX 实现环境选项、PRAC 策略、计划缓存选择、切片目标、采样和窗口解析；读取仍在原执行位置。私有数据声明不包含 SDK，域编号在 [domain.h](../../src/ecm/montgomery_stage1/cuda/prac/domain.h)集中维护，与实际设备分派共享。修改解析正文不重新编译 CUDA 协作；修改实际域值/数据合同仍使设备依赖失效。

诊断窗口的 GPU export、同步与回读在 CUDA，CSV 编码在 output.cpp。窗口部分乘积始终失败返回，不提交 NF/save 或内部完成结果。resident/PRAC 的资源、实际 dispatch、事件和定时循环继续属于 CUDA 模块，不要求脱离 CUDA。

## 应用群阶诊断

[group_order.cpp](../../src/ecm/application/cli/group_order.cpp)是 CLI 私有完整诊断，负责 gp 路径、脚本/进程、输出解析和命中报告。driver 在原时点检查 gp 可用性，在单任务结果已发布后逐命中调用；复合因子继续跳过。该实现仍以群阶和旧 32 位诊断 sigma 生成 go_min_bounds，未计算实际点阶/固定 torsion 的严格最小界限；CPU Edwards 的适用性也没有新增资格。数学条件与限制见 [重构分析](../development/MAIN_FLOW_REFACTOR_ANALYSIS_DRAFT.md#群阶诊断与最小界限公式)。
