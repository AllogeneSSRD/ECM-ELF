# Stage2 tune 与联合选型开发草稿

本文件是开发中的临时草稿。目标完成后提示用户处理；不作为已实现业务规则。

## 完整目标

- 自动 D 在正确的树形与同时存活显存预算下评估较大 D，使用 tune 实测成本排序。
- 对有效 Stage1 save 的目标 N，同时评估普通模数与满足 N∣2ᵖ−1 的承载候选，显式选项优先，无有效匹配数据时保留回退。
- tune 支持等级1—10，逐级增加覆盖、细度与重复次数；除 NTT 外使用梅森素数或已验证无因子的 ECM 输入。性能数据采用可扩展、可读的命名字段，不包含路径或二进制摘要。
- 将实测性能、算术资格、内存准入分别处理；不以吞吐测量放宽算术证明，不以组件峰值相加替代同时存活容量。

## 已完成的格式与 NTT 等级基础

原生驱动默认写 TOML，支持显式 JSONL 兼容输出；两个格式均不保存二进制/manifest摘要。原子发布与运行中身份变动检查保留。NTT等级预设和显式参数覆盖已接入驱动，未更改 CUDA算术和原有 mandatory checks。

等级ℓ：log₂L=16…[20+min(ℓ−1,7)]，重复2ℓ²+1次，另预热1次。该规则当前只覆盖NTT，不代表完整 ECM 调优等级已经完成。

代码入口：`src/core/ecm_stage2_tune_format.h` 的 `effort`/`table`，`src/core/ecm_cuda_stage2_main.cpp` 的 tune 分支；构建脚本纳入新头文件的依赖摘要。

### 验证证据

- `tools/test/test_stage2_tune_format.py` 编译原生序列化 fixture，用 Python TOML reader 验证数值、数组、字符串与分节；测试非法 callback 拒绝、等级1—10与越界等级。证据：`data/experiments/tune_format_20261009/native/`。
- 驱动经 MSVC直接编译、NVCC链接，复用冻结的 CUDA对象；原脚本经 NVCC编译驱动两次出现编译子进程 `0xC0000005`，不能据此声明该完整构建流程通过。原生fixture和直接MSVC驱动编译均通过。辅助命令与对象保留在 `data/experiments/tune_format_20261009/`，旧冻结二进制未修改。
- 当前设备枚举重新为4070 Ti=0、4060 Laptop=1。短正确性验证先在设备0执行，随后在空闲设备1执行 log₂L=16…17、每个3次；TOML解析通过，两个长度 `bad=0`。这只是格式/正确性验证，不用于性能标定或选型排名。证据：`runtime_4060.toml` 与 `runtime_4060.log`。

## 后续实现与验证

### NTT/S4 联合组件进展

同一执行器已交织NTT和S4事件，保留selftest→output/raw→NTT→canonical的顺序，按树租约和阶段边界释放，并要求两边完整状态共同稳定才压缩。组件已接入plan-only，输出联合峰和峰时分项；不作为完整曲线准入、不改变默认D。

合成描述CPU矩阵：1152 cases、455270 assertions；压缩/未压缩联合峰一致，各组件最终容量与原模型一致。原组件回归：NTT681389 checks；S4 299592 checks、2136 events、192 tree leases，bad=0。证据均在 `data/experiments/stage2_workspace_memory_20261009/`。

构建诊断的最小源码对照表明：直接cmd日志文件重定向触发NVCC编译器探测崩溃，管道捕获通过；此前把问题仅归于命令传递的假设不成立。构建脚本改为独立命令文件和管道收集，新增C++/CUDA compile-only回归通过。受限环境完整CUDA编译仍遇到ptxas INVALID_HANDLE，相同参数以常规权限完整构建通过，约105秒；不改GPU设置。固定旧实验exe保持不变。

真实描述的8组native plan通过，覆盖不同D、trim、BQ、keyed、拒绝前缀及G1不支持；新测试入口为 `tools/test/test_stage2_joint_workspace_plan.py`，原workspace-plan测试保留。5872 bits、B1=20、sigma26、B2=2.6e12、D1141140/P103680，arena6300/fold640/batch256：旧静态估计9219.58 MiB，NTT/S4联合组件3588.26 MiB。未包括fold/giant，不能直接作为准入。

一条M503余因子318-bit、D180180、B2=2.6e10曲线通过，自检2016、GMP检查3032、bad=0，hits=0。新exe SHA256=`b1efdba90c099f3d9cf4412ed701369885fd721be882889b5d8fde11caa226cd`，日志/结果在 `data/experiments/stage2_workspace_memory_20261009/`。未宣称性能收益或自动D已改变。

### Fold/frontier 联动进展

`OwnerMemoryState`按生产成功路径逐项记录fold两块大缓冲与map/length/modulus/digest、frontier metadata；inverse与descent边界接入同一工作区执行器。预算前缀拒绝与动态headroom/物理分配拒绝分开，后两者尚未模拟。`resident_workspace_memory`加入owner峰时分项，原NTT/S4查询保留。

提取生产分配/释放语句，140 cases、1960 allocation/free events、6580 checks通过；metadata extent故意改变8 bytes被拒绝。联合压缩1152 cases、946250 checks通过。10个native plan及普通/承载503/fold-budget0三条完整曲线通过，规划布局与运行布局字段一致。原始证据在 `data/experiments/stage2_owner_memory_20261009/`。新exe SHA256=`1d11589c7b1eb673a25f6850f21a7c1e0a2b67f31848e4c169245155f701fe2a`，完整生产构建约94.5秒。

5872-bit、D1141140/P103680、B1=20、sigma26、B2=2.6e12、arena6300/fold640/batch256的NTT/S4/owner联合规划峰4098.979 MiB。它不是实测进程峰，未含giant；自动D仍未按该值放行。下一步将点chunk的生成/保留/销毁和S3容量变化接入共同时间线，并覆盖更早的初始化瞬时量，之后才用于准入。

### Giant 联合时间线进展

`GiantMemoryState`把S3五常量、小素数初始容量、seed/base/segfix、坐标/segment/group和叶值/积按正常成功路径加入统一执行器。S3在混合初始块的首个inverse请求前构造；point chunk生成在fold准入之后、首棵G树之前。chunk跨多棵G树/fold时保留其transient，最后一棵消费完成才释放；压缩仅跳过四组件状态都相同的完整chunk周期，尾部路线变化仍执行。

新增`curve_workspace_memory`查询；旧两种查询保持原scope。模型仍不是完整准入，自动D与承载默认未改变。5872-bit、D1141140、B2=2.6e12、arena6300/fold640/batch256四组件规划峰4406.441 MiB；floor4251.702，BQ3382.441。这些规划值不构成性能证据。

CPU joint矩阵3456 giant cases和1152原cases，共3392047 assertions；超大B2只执行10块。生产S3提取fixture2051 cases、39573逐事件对照通过。14个native plan和三条完整曲线通过；GMP坏计数与坏因子均0，S3四阶段保留容量及closed owned ledger核对通过。fallback下降先申请叶值，在runtime verifier中按实际路径单独核对，不把joint refusal前缀当作回退时间线。

构建96.4秒，exe SHA256=`a90cb1ca1045f1fdc3dd5ae906d949483e091bf68668fd6ae8471615b2ac3beb`。当前4070 Ti设备0忙，验证使用空闲4060 Laptop设备1，无功耗/频率变更。证据目录`data/experiments/stage2_giant_timeline_20261009/`。native查询collector首次引用了不存在的JSON键，修正为现有`I`字段后14例通过，失败输出保留；fallback retained-capacity初次假设叶值在下降后分配与native不符，按当前`!frontier.enabled`路径修正检查，原证据保留。

### 初始化与free快照需求

共享`baby_memory_layout`替代caller和D模型的重复容量计算，显式检查溢出。基础mont selftest在S4之前完成；S4模数之后才生成baby，临时量在F树之前释放。`curve_workspace_memory`升为version2、五个峰时分项；baby预算0返回有效拒绝前缀，增加初始化分配的诊断返回不可用。条件模型未成为完整准入，不改变生产自动D或承载默认。

联动执行器记录baby预留64 MiB、fold/frontier各自未来NTT增长+1 GiB所需的启动free。`required_free_bytes`同时包含联合峰+reserve；`initial_free_snapshot_fits`只是正常成功路径的静态估计。继续保留实际cudaMemGetInfo、cudaMalloc失败及回退，不要求先把全部失败路径精确建模才推进正常路径的tune筛选。

CPU50 cases/1500 native events/4556 checks、错误tree extent+8 mutation拒绝通过；联合初始化3456 cases，总6097043 checks，headroom公式由逐事件边界独立核对。16个native plan和最终binary三条短曲线通过。首个fixture漏了string include导致编译失败，修正后通过；失败输出保留。最终exe SHA256=`4a734f4cab13ae3b4f76d5b65bc3921cd94c826ec427cc30c14323bc018bf466`，完整构建96.6秒，所有编译源hash与当前实现相同。

另一个冻结初始化binary SHA256=`747753d0e09e0b09f181ca04b5a61cf6e42cb8968c50bb7d9f1913cd2102cbe3`完成5872-bit、B1=20、sigma26、B2=2.6e12、D1141140/P103680、arena6300/fold640/batch256曲线；固定1800 MHz，设备1，无设置变更，单条wall96.761609秒。预测/owned账本峰都为4,620,488,216 bytes，mandatory2208、GMP43253、bad=0、hits0。短普通/承载曲线也核对峰一致，fallback不做完整joint峰断言。证据`data/experiments/stage2_initial_timeline_20261009/`，包含每次build对应的frozen sources、查询、原始日志和分配账本。

下一步直接接入完整ECM tune、版本化reader及正常可驻留候选选择；继续保留现有无数据/不适用的兼容路径，不将单条相同峰验证当作所有输入的物理准入证明。

- [ ] 在正常路径联合模型的条件范围内接入候选筛选，保留实时free检查；扩大验证，并补齐动态headroom与非驻留回退模型。
- [ ] 增加有版本的实测数据reader，明确设备、后端、算术路径及测量输入适用范围；路径与二进制信息放实验审计，运行tune数据只保留必要性能与资格字段。
- [ ] 引入无因子完整ECM基准、普通/承载配对测量与D候选网格，将等级映射到完整测量计划。
- [ ] 将选型连接到显式B2主路径和Auto B2；当前静态准入及手动承载均未由本轮改动替代。
- [ ] 先复测5872-bit较大D、7995-bit承载，再覆盖更多位宽/预算与非单位场景；按同设备条件验证候选收益。
- [ ] 实施完整构建、回归与性能验证，更新权威说明。

### 完整Stage2 tune与实测主路径

完整Stage2等级、命名TOML及有界原生reader已接入。默认13个已知素数，独立GMP准备B1=20/sigma26；save可配对测普通/承载。每形状独立进程预热及重复，只有clean、无因子、mandatory/GMP检查成功且驻留的样本发布；失败保留旧配置与raw。等级同时扩展位宽、D/B2、重复和G树数量上限，0可显式解除耗时上限。格式3策略每键一行，格式2测量环境串可读取；性能文件没有路径、binary或manifest摘要。

默认目录资格检查发现8191不属于梅森素数指数，已替换。13项Lucas–Lehmer和独立Stage1参考通过；该发现前的失败CPU输出保留，所有已运行性能曲线使用521或已核验余因子。CPU reader拒绝21种坏profile，支持UTF8 BOM与旧格式2，验证设备/预算/策略不匹配。最终production CUDA完整构建约105.6秒，命名配置/资格修正经新主机对象构建，CUDA对象按依赖校验复用。

`stage2_tune_profile`/`--tune-profile`已在生产worker与plan-only应用：精确匹配target bits、B1/B2、已测D和算术类型，承载逐候选验证N∣2ᵖ−1；按median+2MAD排序并查询正常驻留联合free需求。D和算术显式覆盖保留，包括carrier=0固定普通模式。无数据/不匹配/不适用保留legacy，破损profile报错；debug日志未标定时回退。

最终exe SHA256=`5eecbb20334cdd492e8c9a71e07e8d662e437ea7eee220cc2380fb65cbb38353`。13素数26条完整GPU catalogue、普通/承载与生产选择13条短曲线、显式覆盖、INI、失败发布保留、预算跳过通过，另16个native联合plan回归通过。5872-bit两D各预热+2正式（6条）完整检查通过，140.972→97.406 s，中位数少30.904%。同CUDA对象的最终驱动以D=0自动选1141140并完整运行97.217 s，worker99.457 s，坏计数0。固定1800MHz、默认55W上限，GPU1、不修改设置；首个小D正式样本有短时host编译并行，不能把本验证替代独立交错标定。

证据`data/experiments/ecm_tune_20261010/`的`native_certified/result.json`、`runtime_named/result.json`、`joint_plans/results.json`、`wide_5872.toml`、`production_wide/result.json`及两binary的frozen source closure。正式状态已同步到AUTO_B2、MEMORY、STAGE2和性能说明。

剩余工作：更多生产B1/位宽/预算下的成本拟合与独立排名验证，正收益承载的生产自动选择完整曲线，NTT测量与阶段成本的预计算组合，Auto B2的新配置/Stage1成本接入，非驻留/G1回退模型。当前精确scope选择不宣称这些范围已完成。

### 主路径验收与承载验证

`698e207`已提交完整ECM tune和实测D/承载主路径。提交前核对最终冻结exe和53个编译依赖与当前源码一致，生成配置检查及diff检查通过。5872-bit生产自动较大D曲线已经完成；不是仅凭plan-only声明已接入。

当前验收项：

- 自动D使用tune实测排名，并允许联合内存规划通过的较大D：已有5872-bit完整生产曲线及小位宽覆盖。
- 普通/承载候选共同排名，显式覆盖优先、整除证明独立于性能适用范围：小位宽配对与覆盖检查通过；7995-bit配对及完整生产自动选择通过，自动承载8011完整时间135.540秒。
- tune等级1…10增加覆盖、D/B2网格、重复数和允许批次，支持NTT与完整Stage2：原生等级测试和完整GPU素数目录通过。
- 命名TOML、无路径/二进制字段、有界reader、原子发布及失败保留：原生非法profile与实际失败发布测试通过。
- 预计算使用已知梅森素数或经完整算术检查的无因子save曲线：13项Lucas–Lehmer、独立Stage1点与GPU目录通过；save测量不能推断其他sigma也无因子。

本次在GPU1固定1800MHz/默认55W上限下测M8011/80111（7995 bits）、B1=20、sigma26、B2=2.6e12、D810810、arena6300/fold640/batch256，普通与8011承载各预热1次、正式2次。普通200.515秒、承载135.955秒中位数，少32.197%；正式测量无本轮编译并行，预热期间有主机编译。全部6条mandatory2304/GMP51956、bad0、无因子、fold/frontier驻留。原始证据目录`data/experiments/ecm_tune_35836_14824203/`，驱动日志/性能配置/传感器记录在`data/experiments/ecm_tune_carrier_20261010/`。

### 分批预计算合并

原生`--tune-merge`接入同一reader和命名格式3 writer，不启动GPU工作；锁定输入文件读取，所有输入设备/策略/正式重复次数/预热/算法/单位一致才合并。同一scope最后输入覆盖，保留对应完整样本，不拼接计时。保留旧格式2兼容、原子发布和源文件不变约束。INI指向输出时也允许刷新输出，不将运行性能配置当作不可更新输入。

新主机构建复用相同且核对的CUDA对象，22.7秒；构建与native fixture编译均在承载预热结束前完成，正式承载样本开始后不再编译。CPU原生5项合并/5项拒绝及5次reader roundtrip通过；生产入口离线合并与6项拒绝通过，使用无效device9999仍合并成功，失败保留既有文件。证据`merge_native/result.json`、`merge_runtime_duplicate/result.json`。待配对性能文件发布后，合并5872/7995-bit两个范围并在生产入口分别核对选型。

最终主机构建21.4秒，sha=`26c144825c72e7f7907ba4109590624e265ee8e9f0c07feb88b43eebf6ca0c7d`，53项源依赖与冻结快照一致。最终CPU原生合并与roundtrip通过，生产合并得到4个scope、7项非法操作拒绝，包含禁止NTT覆盖INI引用的ECM配置。生产自动承载完整曲线135.540秒/worker137.712秒，显式关闭及同位宽不合法整除反例通过；日志/结果在`merge_native_final/`、`merge_runtime_final/`、`production_auto/`。5872-bit同配置的最终驱动自动D1141140完整曲线97.431秒/worker99.708秒，mandatory2208/GMP43253、bad0、fold/frontier驻留，证据`production_d_auto/result.json`。两种优化已在同一最终驱动、同一合并性能文件中完整运行；未测范围拟合、生产B1/预算扩展、NTT阶段成本组合和Auto B2新数据合同仍未完成。

### B2区间预测与高等级网格细化

新`linear_giant_points_v1`按同target/arithmetic bits、carrier、类型、B1和D组拟合T=α+βI。显式profile模型标记、3…128锚点、I跨度至少2倍、非负有限系数与最大留一误差≤8%为固定资格；不允许位宽/B1/D或B2区间外外推。精确点优先，预测排名增加观测留一误差与2倍最大MAD，内存针对请求B2重新计算。无标记旧文件保持精确选择；合并保留标记但资格仍逐组检查。

首次3点M521/D60060调优的留一检查失败，生产入口保持原选型；未放宽8%门限。随后按已规划的更细5点网格，每点预热1次、正式3次，最大留一误差2.986%。两组调优共40条完整曲线，原始回执与GMP日志bad0、无因子、驻留。数据和失败拟合证据均保留在`data/experiments/ecm_tune_prediction_20261010/`。三级与五级细化分别用2份/4份对数区间，高等级B2点数最高17；D目录加入810810、1021020，高等级覆盖14项，最高默认3094形状，不超过reader4096样本限制。

最终生产exe SHA256=`182f2950505e864ab31727c8c475a327bda4dce94b79a3e16ab3c1f5d7f86db7`，主机构建21.5秒，54项依赖冻结并核对，CUDA对象未改。GPU1、用户固定1800MHz/默认55W上限，本轮不修改设置、不扰动忙碌GPU0；短测试没有连续传感器记录。三个未拟合B2的完整曲线预测/实测为0.313360/0.328982、0.559302/0.559750、0.780019/0.769658秒（每点n=1），误差4.748%、0.080%、1.346%，bad0。先前最终构建的同三点及首次构建验证也保留，不用筛选结果改善结论。

CPU原生13素数证书与独立Stage1点、10等级嵌套D/B2网格、21坏profile/3身份拒绝，预测1合法/13不适用/1坏模型，7合并/5拒绝/7roundtrip通过；生产合并7scope及7非法调用拒绝，精确优先、旧标记、坏拟合、外推及显式覆盖验证通过。最终结果为`native_final/`、`prediction_native_final/`、`merge_native_final/`、`merge_runtime_publish/`、`production_publish/`，生成配置6文件及diff检查同步。

剩余完整目标不缩小：扩大生产B1、位宽和预算的候选成本/排名验证，建立NTT预计算与阶段成本组合，接入Auto B2及真实Stage1摊销；当前固定D的小位宽B2拟合不替代这些工作。需要继续测高位宽、普通/承载竞争和D台阶，不能由本轮误差宣称通用预测合格。

### 高位宽区间及候选竞争验证计划

当前启动5872-bit M6011余因子、B1=20/sigma26、D60060和120120、普通及承载6011、B2=10400000000、14707821049、20800000000、29415642097、41600000000，每形状预热1次、正式3次，共80条曲线。使用冻结生产二进制`182f2950505e864ab31727c8c475a327bda4dce94b79a3e16ab3c1f5d7f86db7`、GPU1、arena6300/fold640/batch256；不编译、不调功耗/频率、不扰动GPU0，NVML每5秒采样。证据在`data/experiments/ecm_tune_wide_prediction_20261010/`。完成前不声明profile合格或候选收益。

新增可复用独立排序验证工具，Python统计线性回归与native算法分别实现。预定未测B2为18000000000和32000000000，要求至少4个可用候选；每候选预热1次、正式2次，并另执行未指定D/承载的生产曲线。预测误差门限8%、候选排名损失门限5%，不随本批结果调整。CPU参考的解析线性、实测优先、缺少模型、区间外、非驻留、重复I、离群和负系数拒绝检查通过，结果在`reference/result.json`；完整GPU验收仍待运行。

### 后续Auto B2与阶段数据接入边界

下一阶段拟将完整ECM tune的合格成本与当前Auto B2收益目标相接，继续按K/[T1+R·T2]比较。T1必须是真实批次摊销的成本或明确提供的正`stage1_seconds_per_curve`，不能用生成B1=20基准点的准备时间替代生产Stage1。当前`.cprof`的旧二进制/算术资格仍保留；新接口不能把不合格文件当作已重新标定，也不能为使Auto B2可用而删除其门限。

拟区分实测B2点与合格区间估计，搜索范围限于合法scope并逐候选检查当前联合显存；显式D和承载锁定、每条worker重新规划、原始请求、B2上下限和range-limited状态需要贯通CLI/INI/队列及plan-only。收益和T1不能只在父进程选择后以固定B2跳过worker复核。缺少T1或完整合格候选时必须说明无法选择，不默认T1=0、不外推生产B1、位宽或预算。

NTT测量后续可作为阶段成本的独立特征/核对数据，须显式区分field convolution与S4、树准备、点乘、传输和冷启动；不能把NTT iter/s直接换成ECM曲线速度。阶段中位数有包含关系，需要定义互斥成本口径及独立留出验证后才接入自动收益排名。上述为待实现方案，正式AUTO_B2仍只描述现有功能。

### 高位宽验证结果

80条调优全部完成，20样本、无跳过，四个D/算术组全部通过固定8%资格，最大留一误差2.120%…3.557%。两个预定holdout各四候选、每候选预热1次+正式2次，并各有1条自动生产曲线，共26条；八个候选最大预测误差1.701%，两次排名损失均0，自动曲线5.450927/8.080824秒，成本预测5.485900/8.057804秒。D120120/承载6011在本候选集内最快且确实由未指定D/承载的worker执行，原始请求均保持0。全部106条曲线bad0、无因子、驻留。

数据在`data/experiments/ecm_tune_wide_prediction_20261010/`及`data/experiments/ecm_tune_17256_17927093/`。最终audit独立核对20组统计、80条调优回执及26条holdout的GMP日志；源save/性能配置/二进制未改变。测量期间没有本轮编译；GPU0持续繁忙，GPU1完成后空闲。NVML记录188完整点，利用率≥80%的159点中频率1560…1800 MHz、median1800，功率24.32…55.03 W、median49.08；不是严格锁频，1行中断尾记录保留。不得把该样本作为理想恒定频率验证或默认D全局最优证明。

独立验证工具首版在全部驻留本批上完成实测；之后核对发现reference的实测点分支应拒绝非驻留点，分组也应先过滤非驻留锚点，按native现状修正。冻结测量验证器保留在`holdouts/frozen_verifier.py`，新增拒绝检查与当前reference对全部8个原生候选的重放通过，证据`reference_replay/result.json`。未改native算法、profile、门限或原始样本来使结论通过。正式性能与工具说明已同步。

本阶段验证了高位宽固定scope内的区间成本及D/承载竞争；Auto B2新数据接口、真实Stage1摊销、生产B1/预算扩展和NTT/互斥阶段成本组合仍为下一阶段，不将当前B1=20证据替代这些工作。

### 完整tune Auto B2联合选择接入

新增独立CPU候选模块，将完整ECM实测/合格区间成本按K/[T1+R·Tguard]排名；搜索保留锚点、65位置对数网格和附近整数I平台边界。CLI/INI/worker/plan-only/队列贯通B2/D/承载、显式锁定、范围限制、原始请求及`auto_plan`；每条worker重新检查目标整除与当前正常驻留联合显存，不在联合选择后再运行第二次D/承载选择。完整tune优先于 `.cprof`，要求有限正的已摊销T1；旧二进制/算术标定门限保留，没有用新路径使不合格旧profile通过。

第一轮合成检查发现整数I平台内成本不变而B2收益继续增长，最优可能是首平台末端而非字面范围端点；修正`range_limited`为首/末I平台，不改变8%预测资格。合成Auto检查8接受/12拒绝，独立20001位置搜索score比0.9999953759；13素数证书/独立点、10等级、21坏profile、预测/合并回归通过。

初次显存拒绝fixture修改arena预算但未同步有效环境的arena_cap_kb，正确触发策略拒绝；同步fixture后，全范围不满足显存的检查反复CPU规划，在自身300秒超时退出。原PID经管理员只读查询确认终止才重跑，未因观测缺失重复启动。显存拒绝改为单一精确scope，保持数学/显存门限；全部候选不满足显存时的重复规划成本列入TODO。另一次4条曲线全部完成后，收集器把Stage1的finished键误当Stage2，读取错误路径失败；改为stage2_finished，保留此前日志与结果。CPU回归首次误用7scope生产profile作单scope合成模板，被reader拒绝；用测试定义的valid模板复跑通过，未修改reader或资格。

最终冻结生产exe SHA256=`03ab129dc1b81faf928e806dffc8e2cf7ea287abc1173181c1cdc11a9f6d89e6`，HostOnly23.1秒，复用同一CUDA对象，55依赖与当前/冻结源一致。GPU1、用户1800MHz/默认55W上限、arena6300/fold640/batch256、不改变设置、不扰动GPU0；短运行无连续传感器。最终11组选择、11组拒绝、4条完整曲线通过，含T1/batch不重复摊销与两种算术锁定；私有两条队列完成后重启不重复。5872-bit余因子T1=30决策输入选B2=20800000000、D120120、承载6011，估计5.8960122/实测5.848048秒；M521未测5.2e9估计0.3133604/实测0.307504秒。直接每输入n=1，最大误差1.905%，不是独立连续收益最优证明。额外显式B2三点回归最大误差2.048%；全部保留的15条曲线原始审计mandatory/GMP坏计数0、无因子。

证据`data/experiments/ecm_tune_auto_20261010/`，最终runtime_final、explicit_b2_runtime_regression、native_final、auto_native_final、prediction_template_regression、merge_template_regression、merge_runtime_regression、final_raw_audit和source_audit。失败/早期输出均保留；成功验证器另冻结。统一配置6生成物、AUTO_B2、使用说明、性能及TODO同步。

未完成目标仍保留：真实生产Stage1摊销预计算，更多生产B1/位宽/预算范围和独立收益排名，互斥阶段/NTT特征组合、冷启动/驱动成本及非驻留/G1回退。当前完整tune T2仅为引擎total，提供的T1=3/30是接口决策测试输入；不能把基准点准备作为生产Stage1成本，也不能宣布总流程收益模型已全面合格。临时草稿开发完成后应提示用户处理，不自行删除。

### 完整Stage1成本预计算与原生T1查询

新增`tools/bench/tune_stage1_cost.py`及`src/core/ecm_stage1_tune_profile.h`，使用完整已完成批次进程墙钟/C发布独立格式1 TOML。等级1…10扩大13项已知梅森素数、B1=20…260e6及batch1/8/64/256目录，重复2ℓ+1次；CLI可限制范围。固定ladder、PARAM0、auto TPI、exp-cache off、checkpoint off，CPU独立验证每个sigma末点及checksum，部分计时投影不得发布。reader拒绝未完成、非有限、坏统计、重复scope和不支持策略；文件只保存性能与适用条件，不含路径或二进制。原始实验另外保存身份、源和命令。

Auto B2在正的用户T1未提供时，按GPU UUID/SM/runtime/driver、实际目标位宽/类型、B1、batch及指数模式查询实测中位数。共享INI的`exponent`默认提供成本条件，CLI可覆盖；不改变save中的点。CLI/INI/worker/plan-only/队列身份和输入文件防覆盖贯通。已有正T1优先，未使用的Stage1文件不读取；旧cost路径不借独立Stage1文件绕过资格。离线`--check-stage1-tune-profile`不启动GPU。

部署Stage1二进制SHA=`5ff1f58a3a072fb37b7ef6e35d3ac2de5488304d6acc5d4f32b6555cb2c96e6e`，不是本批新Stage1构建；保存原二进制身份，不能从末点验证反推它与当前源码一致。M521/B1=20的lcm/choose12、batch1/8各1预热+3正式，以及M4423/B1=1000/lcm/batch8同轮次，总20批次104曲线全部独立末点核验。M4423的T1中位数0.0536744875秒/curve，GPU部分0.024823625秒/curve，container4608/TPI16。短B1启动/准备占比大，不外推生产10e6…260e6。

初版collector按PRAC格式读ladder geometry，完成一次M521暖机后被错误拒绝；修正为实际`CGBN<8, 768>`与curve数量行，原失败保存。随后在M4423前加入Stage1 runtime/device检查，前两个成功批次的测量源由删除该新增块重建，SHA严格匹配原raw tool hash；不会把当前测量源伪装成旧源。独立审计又复核旧记录runtime/device一致，原始JSON不改。发布超时试验在0.1ms子进程timeout时拒绝写入，已有profile字节不变，未完成时间不得发布。普通权限配置生成一度被Windows拒绝写入INI示例，授权范围内提升权限后6生成物同步完成。

M4423有效B1=1000 save完成两D、五B2各预热1+正式3，共40条完整Stage2曲线。D60060的线性I组留一误差3.5296%，合格；D120120短尾出现非单调且不合格，原生保持精确样本可用，未放宽8%。与真实Stage1 T1联合选择B2=2.6e9、D60060、普通模数：初轮1条实测1.430161秒/估计1.443278秒，误差0.917%；最终驱动再1条1.415360秒，误差1.972%。选择位于范围下界，只是所测范围内联合执行验证，不是独立总流程收益最佳证明。

初轮冻结Stage2 exe SHA=`440950b99fddc7e3d315fbf6fa4745b6b85affd25e23a75acb7a4bd3629a5807`，56依赖核对。最终帮助/诊断与配置文字补齐经HostOnly23.1秒构建，SHA=`231cd6b22725caf1f17d111d0f89120df2f82e78e24cfb1e0b02c0a7457ee224`，复用经核对的CUDA对象。CPU读写3接受/26拒绝/10等级，ECM13证书/10等级/21坏profile、预测、Auto和7合并/5拒绝/7roundtrip回归通过。最终Stage1接口4选择/15调用/3完整Stage2曲线，涵盖shared INI choose12、CLI覆盖、用户T1优先、缺失scope、设备不符、写入冲突及私有队列续跑。

最终驱动旧手工T1的首次单曲线回归出现M521预测0.313360秒/实测0.355秒（11.729%超限），算术正确但性能不计通过。按diagnose流程提出短曲线冷启动/调度、驱动初始化变化、旧profile动态偏差和外部竞争四假设，固定save/B2/D/算术/预算、旧新驱动各预热1再交错正式3次。两版median0.314233/0.318855秒，+1.471%；六个正式样本全部在原8%以内，不能确定初次失败具体原因。随后完整相同回归11选择/11拒绝/4完整曲线通过，最大误差3.643%；保留初次失败、对照和NVML样本，不放宽门限，不把重跑当作初次成功。

证据`data/experiments/ecm_stage1_tune_20261010/`：`completed_stage_audit.json`、`final_stage_audit.json`、`source_audit_final.json`、`runtime_final/`、`provided_runtime_after_probe/`、`coupled_4423_final/`、`final_drift_probe/`及Stage1原始目录；40条调优回执在`data/experiments/ecm_tune_36448_21825781/`。最终raw审计65条完整Stage2曲线mandatory/GMP bad0、无因子，包含失败性能样本，不能将其混作65条性能验收通过。GPU1同一4060 Laptop、用户1800MHz/默认55W上限，本批不修改设置、不扰动忙碌GPU0，正式短测量没有连续遥测；对照探针单独记录NVML。

已同步AUTO_B2、Stage1/Stage2使用、性能及TODO；统一配置和diff检查通过。剩余完整目标：生产B1/余因子/实际batch成本与预算扩展、独立收益排名、互斥阶段/NTT特征组合、Stage2冷启动/驱动成本及非驻留/G1回退。新脚本可请求生产B1，但尚未完成生产B1实测，CPU纯Python参考会很耗时；下一阶段先改善高B1独立参考与受控标定，再扩大预算和尾部成本范围。本草稿继续保留。
