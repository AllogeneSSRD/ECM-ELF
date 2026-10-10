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

### 高B1独立GMP参考与生产标定启动

`05d2de0`已提交完整Stage1成本查询。本阶段新增独立`stage1_gmp_reference.cpp`及纯CPU构建入口，用分段素数筛法、均衡乘积构造LCM和普通GMP ladder生成末点。它不引用生产CUDA/ECM头文件；当前仅支持13个已知梅森素数，B1=2…260e6、sigma>=6、单范围<=4096曲线，支持lcm/choose12。原生参考通过末点输入/输出协议接入现有成本collector，缺点/缺尾/乱序/越界/错误scope均拒绝，参考失败不能发布。原生进程timeout默认7200秒且排除T1；扩大batch时只补缺失sigma。性能文件格式及运行reader没有变化，二进制/冻结源/GMP身份保留在raw。

纯CPU构建3.2秒，GMP参考exe SHA=`a7cb7de0e32d0a21c82becbd731bcf1f3e60e5f9ce4894ef022858a09c8d292b`，源码与GMP依赖冻结在`build_cuda_cmake/stage1_gmp_reference_20261010/`。15个LCM标量（含跨1MiB筛块）、76个独立Python末点及26种非法输入/缺尾等拒绝通过，覆盖13个素数与两指数模式。新collector用M521/B1=20、batch1/8、各1暖机+1正式完成4批18曲线，末点/原生reader通过；这是接口回归，正式n=1不作可靠速度标定。旧成本reader3接受/26拒绝/10等级仍通过。首次应用collector修改的补丁因上下文不匹配而未生效，核对文件后重应用；没有覆盖原始实验或数学门限。

启动M521/B1=10e6、lcm、sigma从26开始、batch1/8、每scope1暖机+3正式，使用部署Stage1 SHA=`5ff1f58a3a072fb37b7ef6e35d3ac2de5488304d6acc5d4f32b6555cb2c96e6e`、冻结Stage2 SHA=`231cd6b22725caf1f17d111d0f89120df2f82e78e24cfb1e0b02c0a7457ee224`、GPU1。原生第一点33.4581秒，第一完整GPU暖机210.318895秒并与独立末点一致；本轮仍在运行，尚未发布生产B1配置或正式中位数。传感器每1秒读取，未改用户1800MHz/默认55W设置，不扰动GPU0的既有任务。管理员只读进程查询确认指定子进程运行后继续等待，未因没有即时输出重复启动。

证据目录`data/experiments/ecm_stage1_production_tune_20261010/`：`reference_tests/`、`small_native/`、`small_native.toml`、`reader_regression/`、`production_command.json`、`production_driver.log`、`production_telemetry.csv`及仍在运行的`m521_b10m/`。运行handle保存在当前任务，生产job未结束前不同时启动GPU1 Stage2计时、不修改collector或参考二进制。

同时准备`validate_stage2_tune_auto_profit.py`：读取真实T1并以完整曲线交错复验profile锚点及自动选中B2/D/承载，独立计算K/[T1+R·T2]，预设时间8%、有限候选收益排名损失5%门限；每点暖机1次+正式3次，保留完整原始数据及另列的进程墙钟。CPU收益公式/倍率/七种非法成本检查与语法检查通过，完整GPU流程尚未验收，暂不写入已通过业务规则。该有限候选比较不声称连续范围或所有B2的全局最优；后续必须用同B1 Stage2 tune成本和实际生产save验收后再提交性能结论。

### 冻结计划的NTT工作量特征

生产Stage1 job保持原handle运行；本阶段先完成CPU侧特征工具，不编译、不并行启动GPU1工作，不修改当前collector/参考二进制。M521/B1=10e6/lcm/batch1的三个正式进程墙钟分别为210.325151、210.324112、210.287599秒/curve，末点已核对；batch8的独立GMP sigma27…33参考已完成，当前batch8暖机仍在运行。整体profile尚未发布，不能将该部分数据宣布为完整标定通过。

新增`analyze_stage2_tune_workload.py`从原生request_program及S4形状提取按phase/N/slices分箱的逻辑多项式pairs、物理calls、输出系数、N和N·log₂N特征。中间G批次按repeat压缩累计，分块公式独立实现并核对原生各阶段groups/pairs/chunks。完整正式回执配对保留，init+main核对total；其他计时仍保留legacy_overlapping合同。NTT单slice数据只提供串行参照和缺失长度，不作为实际生产批量耗时或生产成本profile；策略/批量/独立留出未校准时固定ranking_qualified=false，不将NTT再加入已含NTT的ECM阶段计时。

CPU门禁通过手算满/尾块、chunk上限、万亿次repeat压缩、30个错误输入拒绝、CLI成功roundtrip及缺少scope/错误回执/设备不符的3个拒绝。已有M4423/B1=1000与5872-bit M6011余因子/B1=20的30份plan经独立密集请求重放，通过且与原生phase总数一致：前者989701 pairs/2863 calls/757 bins，后者6360924 pairs/18640 calls/1481 bins。最终工具从两组全部90份正式回执生成对应10/20scope特征TOML，没有筛掉样本。这里是CPU计划重放和计时协议验证，不是新增GPU速度证明。

证据`data/experiments/ecm_tune_workload_20261010/`：`cpu_protocol/result.json`、`m4423_final/`、`wide5872_final/`；性能TOML不含路径/二进制哈希，原始输入和工具身份另存evidence.json。用于对照的NTT文件只测N=65536/131072，匹配设备UUID/SM/runtime/driver/fixed模式，却只覆盖5872组78306/6360924个逻辑pairs，仍缺12种NTT长度；batch1与生产slice、多阶段准备/carry/S4差异尚未标定。已同步AUTO_B2、工具入口和TODO；下一步补齐策略与批量预计算、互斥阶段计时、冷启动边界及独立完整曲线留出，不由上述特征宣布组合模型已实现。

### 生产B1完成与组内拟合复用

Stage1原handle完整结束，8批36曲线通过普通GMP末点与checksum复核；M521/B1=10e6/lcm/batch1中位数210.324112 s/curve、batch8为44.357074。已发布格式1性能文件，sha=`6fe7d349450742807aac0dddee55302560d53e33855f8dd227fe0d9f9cbbadb9`，没有用部分计时投影。1秒NVML中2240个GPU1利用率≥80%的样本SM均1800MHz、功率中位数12.13 W；只说明本次小batch条件，不能推广满卡吞吐。

将B2资格检查拆为`prepare_b2_model`和请求求值。Auto B2每组锚点拟合一次，复用有限非负系数、留一误差/MAD；精确点、不合格组、端点/外推、范围限制与实际free/联合显存仍按原合同。单请求wrapper保留区间外快速拒绝，不用“缓存”跨范围或持久化准入状态。加入CPU fixture完整候选输出/纯构造计时，使用同一caller及旧冻结头文件为基线。56组、17442个候选逐字段/顺序完全一致；三组有预测范围CPU构造降至旧值12.33%…17.74%，三精确点无拟合收益。CPU成本不包括GPU查询、联合内存规划或曲线，不作为完整提速宣称。13素数证书/点、10等级、预测、Auto、合并、Stage1 reader回归通过。

顺序跟进脚本先持有既有Stage1父进程句柄；成功退出且profile/36末点复核后才编译与测Stage2，未因marker文件或观测缺失重复启动。首次收集器使用旧PowerShell且按UTF8解码其本地编码输出，打印到GBK stdout时失败；只读进程查询确认编译已终止后才继续。最小复现确认GBK打印U+FFFD失败、UTF8通过，以及旧PowerShell缺Get-FileHash而当前pwsh可用。换正确pwsh/UTF8并用新目录继续；旧日志/状态保留，不修改算法、资格门限或原始样本。

最终HostOnly22.5秒，生产exe SHA=`d1d1946e927799c21bec4f16951069179bfb0e88ec2bc13fdb63919dc67e0b30`，56依赖/current/冻结快照一致，CUDA对象未改。同B1生产save完成两D×五B2各暖机1+正式3的40曲线，10scope，无跳过。D60060线性I最大留一4.794%合格，D120120不合格且仅保留精确点。真实batch8 T1独立收益工具复测10锚点×4=40曲线，自动选26e9/D120120/普通模数，实测0.533192/估计0.534010秒，10候选最大误差7.636%，收益损失0，范围上界平台。score仍是K/[T1+R·引擎T2]；Stage2进程中位数1.081551秒另列，未假装包含驱动成本或全局范围最优。

runtime工具允许未提供choose12成本文件时明确跳过该模式接受检查，保持其缺失scope拒绝；生产B1通过2计划/13调用/3完整曲线及INI私有队列重启不重复。已有B1=20的choose12文件另实测4计划/15调用/3曲线，报告choose12_verified=true，未用缺失数据作choose12标定。首次调用误写成本文件名在启动GPU前失败，保留目录；改用实际stage1_lcm/stage1_choose12文件后在新目录验收。最终86条Stage2 raw审计160992 mandatory/385444 GMP checks，bad0。

证据`data/experiments/ecm_tune_production_auto_20261010/`：`execution/cache_comparison/`、失败`execution/production_build.log`、`encoding_probe/`、成功`execution_resume/`、`production_runtime/`、`choose12_runtime_final/`及`final_audit/result.json`。原始40曲线tune在`data/experiments/ecm_tune_23416_25986359/`，源码/二进制冻结在`build_cuda_cmake/ecm_model_cache_final_20261010/`。生成配置6文件与diff核对通过；AUTO_B2、性能及TODO同步。剩余完整目标不缩小：更大D/B2与生产位宽/预算、choose12/余因子/实际大batch成本，独立更广收益排名，NTT批量及互斥阶段组合、驱动冷启动和非驻留/G1路径。

### 生产B1大D范围与独立收益复验

确认上一handle对应真实协调器9864/收益子进程36252仍在运行后继续，未重复启动；结束后管理员进程查询确认父子进程退出才启动后续编译。M521/B1=10e6/lcm/sigma26，使用同一有效生产save、batch8真实T1与冻结exe d1d1946e…，GPU1/用户1800MHz/默认55W上限，batch256/arena6300/fold640。三D690690/1141140/1381380、五B2从260e9至2600e9，各暖机1+正式3，60曲线，15scope无跳过；三组留一最大0.459/3.509/2.337%，全部合格。与先前小D范围合并保留25scope，两个B2段之间没有外推。

独立收益25锚点各暖机1+交错正式3，100曲线；Auto从1293决策候选选择2.6e12/D1381380/普通模数，P126720/I1882177/G15。实测4.519360秒/估计4.577495，误差1.286%；全候选最大误差4.505%，收益排名损失0。D690690在同B2实测6.518491，选中D减少30.669%引擎时间；该结论不是承载收益或通用D默认。范围上界平台仍受限，Stage2进程5.128206秒另列，score未包含驱动/冷启动。

最终raw审计160曲线355200 mandatory/818108 GMP检查bad0、无因子、驻留。NVML利用率≥80%的245点SM1545…1800/median1800，power18.97…55.11/median51.78 W；实际频率非严格恒定。证据`data/experiments/ecm_tune_production_large_d_20261010/execution/`与`final_audit/result.json`，原始60调优曲线`ecm_tune_36356_26996765/`，合并profile SHA2aa2fadf…；56源/current/冻结及二进制身份核对一致。正式性能和TODO同步；生产位宽/余因子T1/预算、NTT与互斥阶段组合、冷启动和非驻留/G1仍待完成。本草稿保留。

### 余因子Stage1成本入口与完整批次验收

新增独立参考`--n HEX_N`和collector `--target-n N...`、`--sigma-first s`，默认13素数目录不变，字面目标与exponents互斥。校验实际N为odd>3/≤16384bits、sigma连续范围、同位宽/类型scope不重复；末点复用键改为actual N/B1/sigma。参考协议验证实际N，拒绝相同位宽不同模数；成功TOML只存性能scope，不写目标数字、路径或二进制。普通目标Stage1成本不受Stage2承载位宽影响。参考非单位/缺尾/末点错误/因子均不能发布，lcm/choose12语义保持。

大D测试终止后纯CPU构建4.6秒，GMP参考SHAff7a834a…，15 LCM/76素数点/20合数点/40拒绝全部通过。CPU目标协议8接受/32拒绝、ECM reader13证书/10等级/21坏profile、Stage1 reader4接受/28拒绝/10等级通过。文档补丁首次因完整段落上下文不匹配而未应用，核对后重应用，未改代码或原始证据以消除错误。

GPU1测M6011的5872-bit实际余因子/B1=20/sigma26…33/lcm及choose12/batch1、8，各1暖机+3正式，共16批72曲线，container6144/TPI16；末点与独立GMP和纯Python一致。lcm C1/C8完整process/C中位数0.256436/0.032049；choose12为0.262860/0.030819。短B1启动/准备占主导，不外推生产高B1，不把choose12小差异解释成优化。sigma40、60-bit合数C8和原默认M521入口各暖机1+正式1，另18曲线，n=1只是回归。raw审计20批90曲线与25独立Python点通过，profile身份/二进制/工具保持。

证据`data/experiments/ecm_stage1_generic_tune_20261010/`：`target_protocol/`、`native_reference/`、`ecm_reader_regression/`、`stage1_reader_regression/`、`execution/`与`stage1_audit/result.json`；参考冻结在`build_cuda_cmake/stage1_gmp_generic_20261010/`。同scope已有20个D/B2/普通与6011承载锚点，正在使用真实lcm C8 T1=0.032049执行每候选暖机1+正式3的独立收益复验；本段不预先宣称排名或时间门限通过。更高生产B1/位宽/预算、NTT互斥阶段组合、冷启动与非驻留/G1目标继续保留。

### 余因子真实T1的独立收益与队列验收

原session80884完整成功结束后才启动接口测量。20个候选各1暖机+交错正式3的80曲线全部完成；以真实5872-bit目标/B1=20/lcm C8 T1=0.032049，Auto从1284决策候选选B2=10.4e9/D120120/承载6011，P11520/I86582/G8。实测4.091650/估计4.088892秒，误差0.067%；全20候选最大误差0.538%，有限集合收益损失0。低边界平台受限，不外推低B2或生产高B1。普通与承载、不同D同B2竞争实际复验；不能把单NTT或手填T1当作本轮完整收益证明。

Stage2进程中位数4.753833秒单独记录，score仍使用engine total。raw审计80曲线138240 mandatory/694444 GMP检查bad0、无因子、驻留，56源/current/冻结及工具/profile/二进制身份核对。NVML loaded668点SM1515…1800/median1800，power18.92…55.08/median50.485 W，实际频率非严格恒定。证据`execution/profit/result.json`、逐曲线日志/回执与`profit_audit/result.json`。

扩展T1 runtime工具，私有队列的梅森来源用显式`--queue-exponent`指定，验证实际N整除原数并写入已知因子乘积；不从文件名推断原数或承载。CPU测试首次误把17列为2^8−1的不合法因子，实际上255/17=15，属于测试错误；把拒绝输入修正为19，正确实现不改、旧失败保留。`target_protocol_queue_final`通过2队列身份接受/5拒绝及原8/32目标协议。余因子INI/choose12、CLI/正T1优先、缺失scope、输出保护、私有队列重启通过4选择/15调用/3曲线；原M521/B1=10e6默认入口另2选择/13调用/3曲线。最终接口6曲线11232 mandatory/24153 GMP检查bad0，证据`runtime/`、`runtime_m521_regression/`和`runtime_audit/result.json`。正式文档与TODO改写当前已测范围，原始data不提交。

下一阶段仍保留原目标：生产高位宽/高B1/实际大batch与预算，针对大D的owner/arena联合预算独立标定，NTT完整策略与生产slice吞吐、互斥阶段组合，冷启动/驱动成本和非驻留/G1。当前30.669%大D收益和0.538%余因子有限排名不替代这些未完成项目，不宣布整个目标完成。

### 48 MiB fold预算的驻留选择验收

既有协调器session30831成功exit0，原GPU1测量已停止后才修改计时代码。M521/B1=10e6/lcm/sigma26、真实batch8 T1=44.357074，冻结Stage2 exe d1d1946e…，GPU1用户1800MHz/默认55W上限。batch256/arena6300/fold48，三D与五B2形成15计划；owner31933992/52255272/63867432 bytes，仅D690690驻留。五scope暖机1+正式3，共20曲线；十个skip无执行receipt。原fold640 profile策略拒绝、新profile强制未测D1381380缺scope拒绝，未跨预算复用成本。

独立五锚点各1+3再测20曲线，Auto选2.6e12/D690690/carrier0，322决策候选。实测6.504293/估计6.520463秒，五候选最大误差2.467%、收益损失0，原门限不变；进程7.154319秒另列，上界受限。最终40曲线88320 mandatory/317384 GMP检查bad0，无因子、驻留。loaded103点SM1725…1800/median1800，power19.41…53.19/median51.8W；并非严格恒频。raw、输入身份、56源闭包及profile0e546c12…在phase改动前完成审计，证据`data/experiments/ecm_tune_owner48_20261010/`。正式性能说明同步，本轮不代替非驻留或完整联合预算矩阵。

### 互斥阶段与worker配对成本

新增`ecm_stage2_phase_times.h`，用绝对边界划分shape/setup/baby/ftree/main_setup/inverse_setup/giant_loop/descent/accum/finalize。保留既有engine边界，前四项逐次和=init、后六项=main、总和=total；规划、保存点及外层进程仍在边界外。native tune只发布完整边界的无因子曲线，旧嵌套timer保留。性能TOML按sample保存合同和配对数组，统计中位数不相加；reader检查字段、数组长度、统计及守恒，兼容旧样本与混合合并。父child_run整体墙钟另存worker_samples和逐次worker−engine，不借此宣称纯冷启动或改动排名。

CPU51边界案例、4接受/47坏配置拒绝、非可加中位数及混合合并通过。首次缺字段测试仍从seed保留init_seconds，属于fixture生成错误；修正删除逻辑，失败目录保留。工作量工具扩展交叉核对raw阶段/发布数组与worker合同，50协议拒绝/5 CLI拒绝、2轮回及旧fold48 evidence重放通过。等级10新增数组可超过原16 MiB，reader与独立选择/收益工具扩至64 MiB；3094 scope×21次，23263161 bytes通过，超限拒绝。Stage1 reader仍限16 MiB。HostOnly加入新phase头依赖；故意修改记录hash在编译前拒绝。预测/Auto/合并/Stage1 reader、13素数证书、10等级和生成配置6文件全部回归通过。

受限完整CUDA编译41.1秒在ptxas报INVALID_HANDLE；源码和参数不变，正常权限新目录完整构建107.9秒成功，保留原失败log。该观察支持启动环境因素，不宣称已确定编译器内部根因。完整测试exe83c0efb2…，57源闭包冻结；GPU1/batch256/arena6300/fold640、用户1800MHz/默认55W未改，GPU0既有任务保持。M521/B1=10e6两D五B2共40曲线；5872-bit余因子/B1=20两D五B2普通/6011承载80曲线；M521/D1381380/B2=2.6e12四曲线。各暖机1+正式3，31 scope/124曲线、222720 mandatory/900856 GMP检查bad0、无因子、驻留；每次互斥阶段和worker残差一致。独立工作量重放31计划、2343 bins。loaded681 NVML点SM1515…1800/median1800、power20.11…55.3/median49.9W，实际频率非严格恒定。

残差正式样本组中位数M521小D0.377312、余因子0.439519、大D0.425614秒，只描述本组实测，未变成通用startup常数。测量终止后才修正64 MiB和配置注释，新目录HostOnly22.9秒，exe288e6df6…，CUDA对象未变，57项current/冻结源匹配。最终余因子4计划15调用3曲线，M521 2计划13调用3曲线，11232 mandatory/24153 GMP bad0，INI/choose12/CLI优先/private queue及重启不重复通过。首次runtime错误文件名在GPU启动前失败，保留目录，核对实际lcm.toml/choose12.toml后新目录完成。

证据`data/experiments/ecm_phase_costs_20261010/`及三原始tune目录`ecm_tune_32876_29728078/`、`ecm_tune_3744_29762359/`、`ecm_tune_34824_30531265/`，`final_audit/result.json`分清测量二进制和仅主机修改后的二进制，不冒充旧二进制来自当前修改后的源。阶段合同及工具边界同步AUTO_B2和性能文档。

### 独立留出失败与giant短尾定位

最终版本/新余因子profile计划独立12e9、33e9两个B2，每候选暖机1+正式3，8%时间/5%排名门限不变。12e9四候选16曲线全部执行后时间gate失败：D60060普通预测6.968043/实际9.629335、误差27.637%；承载预测5.462318/实际7.780269、误差29.793%。D120120两算术误差1.276/2.125%。未执行自动曲线或第二B2，不报告排名通过；所有失败scope保留在`independent_holdouts/`。

同二进制通过tune入口再测12e9/D60060普通/承载各1+3共8曲线，实际9.647932/7.757780，排除异常仅由生产调用策略造成；阶段giant_loop8.316307/6.722435。再做1条debug诊断，chunk容量184320点，I199802，尾15482<chain_min32768，1 chain+1 ladder chunk。ResidentGiant::prepare计时3.045434秒，loop_wall8.427；ladder GPU发射异步，在随后prepare的D2H等待，常规giant/G树/fold和漏掉了这部分。阈值和分块说明了线性I模型可能在内部B2出现未被锚点覆盖的台阶。尝试仅进程env chain_min0被已有production保护拒绝，没有移除门禁或更改默认。这条debug时间不计入正式性能样本。

新证据改变下一步：先把giant chunk/短尾chain-ladder结构纳入资格与阶段组合成本，并做新的独立留出，不能靠LOO合格宣称内部区间全面可靠；同时保持原完整目标的NTT批量策略/缺失长度、更多生产B1/位宽/预算、真实Stage1大batch、cold/driver及非驻留/G1验证。阶段记录已经可用，组合排名模型尚未完成。本草稿保留，不宣布整个目标完成。

### Giant分块工作量与成本标定

新增纯整数`ecm_stage2_giant_work.h`：按实际满块/尾块与chain阈值划分路线，利用标量iD的2幂边界计算ladder迭代量，不逐点枚举。新tune从当前原生giant规划取容量/阈值/策略，保存路线点数、块数及迭代量；reader重新计算并拒绝错误字段。`giant_route_cost_v2`按非负α+βI+γL+δQ拟合，保留留一8%门限；mixed至少7点、两类分别至少3点，未测ladder/正L越界/纯ladder继续精确点路径。旧linear模型识别兼容但只使用精确点，以免继续低估已确认的短尾分支。Auto组拟合与单请求路线资格分开，未测中点不会取消其他chain网格。

CPU13素数证书与末点、10等级、21坏profile/3策略拒绝通过；预测2接受/17不合格/4坏profile，Auto8接受/12拒绝，合并7接受/5拒绝，阶段51边界/47坏profile及3094×21容量检查通过。182个实际/随机分块逐点bit长度参考与原生一致，2非法输入拒绝、NumPy与原生mixed拟合一致。首次“L越界”测试使用较小标量的31000点，实际迭代总量仍低于较大标量的30000点上界，属于测试假设错误；修正到3C+32000，保留首次失败数据，不改模型门限。

新HostOnly构建23.1秒，沿用原已验证CUDA对象；source closure冻结后在GPU1普通/承载6011上测5872-bit/B1=20的短尾与既有范围，再验证12e9/33e9，GPU默认设置不变、不并行其他GPU1或编译任务。测量进行中，尚未写入生产精度结论。新增分块模型不替代NTT/互斥阶段组合、冷启动、更多生产B1/批量/预算及非驻留原目标。

短尾标定和独立验证完成：测量exe3401ad09…，58依赖冻结；D60060九B2×两算术18范围72曲线，D120120五B2×两算术10范围40曲线，暖机1+正式3。两组raw为ecm_tune_36140_32205484、ecm_tune_32064_32959531。协调器将两文件错误拼成单个逗号文件名，merge在GPU调用前exit2；原脚本/日志保留，新finish协调器重复--tune-merge正确合并，不重写测量。独立12e9/33e9四候选共34曲线通过，最大误差2.820%、排名损失0，均自动D120120/承载6011；原8%/5%不变。

新增当前原生giant规划与tune策略核对：容量、chain_min和force_ladder分别匹配，固定B2无匹配保留原路径，Auto B2无匹配报错；策略不符计数与显存拒绝分开。原测量结束后才修改main.cpp/tune_ecm.h，新目录HostOnly22.6秒，execcdd1617…、同CUDA对象、最终58依赖冻结。8次plan-only有效2接受/合成策略6拒绝，两个B2各1完整入口曲线、INI/choose12/CLI/私有队列另3曲线；测量版本同接口也3曲线，合计154条完整日志、264000 mandatory/1218380 GMP bad0，无因子。最终CPU再次通过素数/等级/reader/预测/Auto/合并、182分块+4策略和NTT工作量协议，配置6生成物及diff检查一致。

一次最终辅助脚本将curve_0.log同时作为引擎日志和console文件，运行后误覆盖该单条原日志，导致算术计数核对失败；原脚本、receipt和失败现场保留，无法补造已丢失的日志，不计入154。新目录final_runtime_retry使用不同console后缀，重新执行并完成验收；两条入口单次不纳入独立三样本时间统计。audit/result.json明确该缺失证据和原协调器merge失败。审计核对原58冻结源、最终58当前/冻结源，以及两个主机文件差异和相同CUDA对象；profile007f1e5e…，loaded1259点SM1485…1800/median1800、power9.64…55.25/median49.65W，未声称严格恒频。

短尾模型的本范围验收完成，原完整目标继续保留：按实际形状自动补充默认tune短尾点、更多生产B1/位宽/大batch/预算，NTT运行策略与生产slice/阶段组合，cold/driver及非驻留/G1。当前有限154曲线不作为全目标完成；本草稿保留供后续工作。
### 默认tune网格的自适应短尾采样

本阶段将手工挑选尾部B2改为原生规划驱动的补样。新增`ecm_stage2_tune_grid.h`，先裁剪B2>B1、I>P和G树上限，按当前原生plan的C/min/force生成ladder尾部与缺少的chain锚点。等级1/2默认关闭，3…10请求3尾点；`--tune-tail-samples 0..16`覆盖，显式B2列表默认关闭，显式正数才补样。最大B2的free不足不代替逐形状准入；追加网格在执行曲线前检查4096上限。TOML增加采样模型/数量及来源，reader检查来源实际路线，合并兼容旧数据。build将新头纳入主机源依赖，CUDA依赖未改变。

CPU反馈定位并修正两个采样边界：精确满块端点的下一尾应是当前块后一个点；被基础点占满的裁剪块不能使单尾采样遗漏另一块的可用尾部。独立枚举1407组输入通过，覆盖边界、重复I、G2/批数、关闭/全ladder及溢出；同5872/6011两容量下生成3尾点、来源与整数工作量一致。另一个CPU测试初次错误地把所有接近uint64上限的输入都预期为拒绝；实际短区间和少量尾点可以安全计数，修正为核对边界和精确工作量，不修改正确实现。早期失败目录保留。

同时发现等级3常见的3基础chain+3补尾只有6项，达不到当前混合模型总样本≥7。补chain目标据已有尾样本和请求数计算，最多4项；不降低拟合样本数或8%门限。最终默认等级1…10计划组合为12/20/207/288/638/807/1749/2068/3064/3578（CPU目录、B1=20、256MiB坐标分块、min32768，未扣静态显存/G树skip）。4096scope×21次、配对阶段/worker/分块字段的文件31931848 bytes，通过64MiB reader；新采样metadata、路线来源和新旧合并检查通过。13素数LL证书/点、预测/Auto守恒与拒绝、6生成配置检查保持通过。

新目录HostOnly26.1秒，`build_cuda_cmake/ecm_tail_grid_final_20261010/`，59项源闭包冻结，CUDA对象与上一生产版本相同。输入5872-bit/B1=20/sigma26 save重新用独立Python末点核对并验证整除M6011。GPU1保持用户1800MHz/默认55W，不修改硬件控制，GPU0外部任务不干预。默认等级3（无显式B2）/D60060和120120、普通与6011承载30scope，每scope暖1/正式3完成；原始profile正常发布，采样/计划/回执核对通过。120曲线208128 mandatory/651916 GMP检查bad0，无因子。随后12e9仅3候选可预测，12曲线完成后四候选门禁失败；原协调器session54845已exit1，23e9、自动排名和余下步骤未执行。原目录与失败结果保留。

独立NumPy诊断四组最大LOO为8.505775%/7.543259%/1.265506%/1.367384%。普通D60060拒绝来自最低B2=2.6e9、I43292，实际3.0215548/LOO2.7645481秒，MAD0.0168557。低端留一模型缺少局部成本支持，仍须区别低G/NTT台阶与采样密度；不删除样本、不改8%门限，不将仅3候选的误差1.084/1.199/1.598%称为四候选通过。现有手工10.4…41.6e9合格范围不证明新2.6…26e9整个区间。

D120120的裁剪还产生21648/21649/30719尾量，缺少小尾工作量。先将真实失败形状加入CPU回归，旧fixture明确失败并保留`clipped_regression_red/`；再优先调整q以命中目标τ，q下限ceil(max(0,lo−τ)/C)、上限floor((hi−τ)/C)，无法命中时分散可用尾区间。新CPU回归得到2047/16383/30719，原1407案例及10等级容量仍通过。此修复不解决普通D60060最低B2成本资格，也不宣称最新网格已经独立时间验收。

确认旧测量/编译无活进程，并在改源前核对测量exe、59项冻结源和CUDA对象，保存`measurement_source_audit.json`。新目录`ecm_tail_spread_20261010` HostOnly25.9秒，exe04e7ac3b…，CUDA对象不变。当前版本16次原生计划在fold1MiB预算下全部skip，得到正确两算术尾量，不发布profile、未执行曲线；另8次策略plan核对2接受/6拒绝，M521/B1=10e6显式列表与单点两范围各1暖/1正式，4曲线通过。两套版本和计时用途明确分开。

汇总136曲线236544 mandatory/716608 GMP检查bad0、无因子；其中当前4条为入口检查，不并入三重复统计。主测量/失败留出877 loaded NVML点SM1515…1800/median1800、power18.16…55.21/median47.46W，非严格恒频。当前源码/冻结59依赖匹配，6生成配置、native预测/Auto/合并和4096×21容量均回归。`audit/result.json`明确qualification_passed=false；当前采样实现的完整曲线复测、普通低G成本、四候选12e9/23e9验收仍未完成，原生产B1/位宽/预算、NTT/阶段组合、冷启动和非驻留/G1目标继续保留。

### 低端chain补样与中等等级网格加密

核对`ecm_chain_cost_20261010/offline.json`：增加chain_chunks第五特征没有改善稀疏数据的留一资格，D60060普通8.506%变为18.006%，承载7.543%变为16.880%。保留四特征模型及全部锚点，不按训练残差较小启用新模型。阶段计时仍支持研究分块成本，但不能据此认定它是唯一根因。

确认GPU1无测试/编译进程后，使用已修正采样的冻结exe04e7ac3b…、59项源闭包、相同GMP DLL/CUDA对象，在5872-bit余因子/B1=20/sigma26/GPU1默认55W和1800MHz约束下复测。30scope各暖1/正式3，120曲线通过采样/计划/回执与算术；四组LOO8.698%/8.766%/5.739%/5.118%，D60060两组仍不合格。原3个基础点和所有尾/chain补点全部保留。

追加D60060两算术B2=4e9/6e9/10e9/16e9/20e9各暖1/正式3，10scope/40曲线。原生merge汇总40scope、无替换；D60060各12锚点，LOO降至6.915%/6.773%。12e9和23e9四候选独立验证各暖1/正式3，再各自动1曲线，34曲线通过原8%/5%门限。最大候选误差3.304%、自动误差2.708%、排名损失0，两档都D120120/承载6011。新增链式点有助于本范围资格，但未证明原偏差只来自采样密度，也未证明整个区间或其他输入最优。

协调器session70787终止exit0后才改源。194曲线332352 mandatory/1186539 GMP检查bad0、无因子；1342 loaded NVML点SM1485…1800/median1800、power15.31…55.30/median47.89W，不称严格恒频。raw目录ecm_tune_35724_38105156、ecm_tune_37040_39126875；profile04dfa200…，`ecm_chain_cost_20261010/audit/result.json`明确测量与新默认网格边界。实际加载模块另存checkpoint；原失败样本与冻结源均保留。

新增CPU诊断工具`analyze_stage2_tune_sampling.py`，比较四/探索五特征的留一及逐点残差，查询使用独立NumPy参考；不修改性能配置或生产资格。测试将16次查询与原生比较，12接受/4不合格、4坏输入拒绝。新默认等级3、4每区间4份对数细分；先用旧fixture确认5点回归失败，再改源并回归13素数证书/点、10等级、1407枚举、预测/Auto/merge、4096×21容量及6生成配置。当前默认目录12/20/227/319/638/807/1749/2068/3064/3578，不放宽容量或误差门限。

新目录`ecm_dense_base_20261010` HostOnly25.5秒，exe7b91968d…、59依赖、相同CUDA对象。当前新exe复用合并profile的8次原生分块策略plan-only检查2接受/6合成拒绝通过，不执行曲线。新增默认网格尚未执行完整曲线；本194曲线属于修正尾采样+显式chain补样，不能冒充新默认网格验收。下一步运行当前默认网格及独立四候选留出，然后继续生产B1/位宽/预算、NTT/slice阶段组合、实际Stage1批次/算法、cold/driver及非驻留/G1的原目标。本草稿继续保留。

### 四份默认基础网格的直接验收

提交0ed883c后冻结7b91968d…和59项源，session69921运行新默认等级3/无显式B2，两D/普通及承载6011、5872-bit/B1=20/sigma26/GPU1默认55W及1800MHz约束、batch256/arena6300/fold640。原生计划20基础+12尾+0 chain补点=32scope，各暖1/正式3；128曲线221184 mandatory/720088 GMP检查bad0，未合并手工补样数据。四组LOO6.445%/7.471%/5.077%/4.464%。12e9/23e9四候选独立再34曲线通过，最大候选误差3.099%、自动2.621%、排名损失0，两档均D120120/承载6011。策略plan-only8次2接受/6合成拒绝。

session69921终止exit0，完整162曲线280128 mandatory/947235 GMP bad0，无因子。1155 loaded NVML点SM1515…1800/median1800、power15.10…55.17/median47.48W，不称严格恒频。终止后核对exe/DLL/save/工具、59项当前/冻结源；profile9fd4d860…、raw `ecm_tune_24644_40353015`，`ecm_chain_cost_20261010/audit_dense/result.json`确认范围。此前失败原证据仍保留。此阶段有当前默认网格的直接时间/排名证据，不代表更多位宽/B1/预算或全目标完成。

NTT覆盖探针对同例第一plan枚举2048…8388608长度、1…2880 slices、60609个逻辑乘法对；现有65536/131072单slice数据只匹配5对，且未声明完整运行策略。该计数是schema覆盖，不是耗时覆盖或成本资格；不能据此将single-slice串行参照相加到ECM总时间。下一步补NTT长度/批量标定和策略协议，再组合互斥阶段成本，并继续原生产B1/位宽/预算/实际Stage1 batch、cold/driver及非驻留/G1目标。本草稿保留。
## NTT 批量入口与实测资格

已实现NTT格式2和`--tune-slices`，等级1…10从长度8开始扩大批量与重复轮次。每个slice有独立常数项，GMP参考核对全部输出；CUDA事件明确只计两正向加乘积逆变换。CPU读取器按长度/slices精确匹配，缺失形状显式保留；格式1只作单slice串行参照，格式2另核对归约mask及完整环境策略。策略合格不等于完整曲线预测合格，生产排名仍使用完整ECM成本。

完整生产CUDA构建通过。本轮两次受限环境编译报ptxas INVALID_HANDLE；相同参数在普通本机权限环境编译成功，失败日志保留，不能据此宣称已证明编译器根因。GPU1批量/边界/预算/发布门禁通过，普通和承载两路径4条完整曲线7296 mandatory/13780 GMP核对，坏计数0。

冻结计划的19种slices、log₂长度11…23另做21次重复标定，167个实测形状、80个预算跳过，36,079,493,120字核对，坏计数0；第一份计划全覆盖。映射32scope仍只有2个scope完整覆盖、374个缺失bins，下一步扩大真实分块标定并建立NTT/互斥阶段组合与独立留出，不用pairs覆盖率代替时间覆盖率。证据和计时边界见[NTT批量标定覆盖](performance/STAGE2.md#ntt-批量标定覆盖)。
## NTT 稀疏补测与配对阶段组合

工作量工具支持策略一致的多份NTT文件，重复实测形状拒绝；每请求阶段保留覆盖和完整/不完整参照。新补测工具从冻结的已测ECM scopes提取实际形状，按长度分组，不遍历无用笛卡尔积。新增146个实际组合后，基础32scope/194形状/2422 bins完整映射；另给原生留出计划补测形状时，不导入曲线成本。

实验phase_ntt_loop_v1使用同次Tengine−Tgiant_loop固定项及NTT参照/L/Q非负拟合。四组8锚点的最大留一误差6.827/7.838/4.960/4.143%，保持8%门限。回顾12e9/23e9候选最大误差2.845%；只用于比较，没有当作新独立验收。

模型与所有预测在新曲线运行前冻结。12.5e9/23.7e9各四候选、一次预热三次正式，共32曲线通过：最大误差1.542%、排名损失0，两档均选D120120/承载6011；55296 mandatory与224700 GMP核对，坏计数0。原模型在同批数据最大误差1.708%、排名相同，不宣称额外生产速度收益。原始证据、模块身份、功耗/频率和单位见[组合验证](performance/STAGE2.md#ntt-与配对阶段组合验证)。

后续将实验模型接入原生预测并核对参考一致性和收益排名；保留现行主路径与scope/数学/显存门禁，扩大生产B1、位宽、实际Stage1成本、冷启动及非驻留/G1。此阶段没有改写生产预测规则。

## 原生组合计算层

新增ecm_stage2_tune_components.h，沿用压缩请求拓扑与真实packing查询，按精确长度/slices重建五个请求阶段的NTT参照；缺项没有完整成本资格。原生配对固定项、四特征非负拟合及LOO保持与Python公式一致，模型仍受7..128锚点、3 chain/3 ladder、跨度、预算、驻留和8%门限限制。

CPU fixture对32锚点及8份独立查询的3018 bins、244项数值通过，四组资格/系数/LOO及保护成本匹配；10项底层门禁、8项组/reader拒绝检查和缺项/端点/策略变化通过。首次测试误按模型输出顺序配对，修复为D/承载匹配；失败证据保留为native_components_v1，最终证据native_components_v3。未改算法或门限，未运行额外GPU工作。

本阶段只提供原生计算层，生产reader、D/承载排名和Auto B2尚未连接。下一步将精确NTT测量与完整ECM策略一起导出到可读性能配置，由原生调用者核对设备、环境、实际N整除关系及实时联合显存，再独立验证主路径收益。保留原范围：生产B1/位宽/预算、实际Stage1批量/算法、cold/driver、非驻留/G1；本草稿不自动删除。

## 格式4与原生组合主路径接入

本阶段新增原生NTT格式2 reader与ECM格式4内嵌数据，`--tune-ntt-profile`可重复最多64文件，在测量结束或纯合并入口导入。拒绝网格不完整、未核验、错误GMP参考/吞吐单位/输出字数、统计不一致、设备/mask/命名环境不符及重复形状；memory skip不作为实测。性能TOML只保留原始数组/策略，不包含路径、二进制或外部模型系数。兼容ECM格式2/3，无NTT时保持原选择。

ComponentEstimator使用真实引擎packing查询，按组缓存配对固定项、非负拟合与LOO；固定B2和Auto B2调用同一计算层。精确点优先，未知NTT形状/不合格组或查询继续尝试route_v2，再不合格只保留精确点。实际giant/NTT策略、N整除承载、联合显存与free继续在最终候选上重查，不缓存准入结果。帮助/统一配置/正式主题同步，新增头纳入HostOnly源闭包。

HostOnly最终exe eea672d1…，沿用原已核验CUDA对象，61源一致。CPU native_components_v8核对32锚点、8查询、3018 bins、252数值、10底层门禁、8组/reader拒绝及16格式/NTT拒绝，337形状导入、往返和生产Estimator/Python一致。workload_format4_regression保留格式2/3旧API兼容，32计划、6386314逻辑pairs、18754 physical calls、2422 bins重放通过。

component_merge_runtime_v2核对32scope+337形状纯CLI合并，旧7拒绝及新6检查通过，无CUDA设备要求；首次辅助测试在合法格式4覆盖后仍比较格式3旧字节，是测试期待错误，修正为比较最近合法输出，原失败目录保留。Auto runtime v1用true填写0/1枚举，在GPU启动前拒绝；v2在曲线成功后错误期待quiet控制台转发auto JSON，改为读取耐久receipt.auto_plan，失败现场保留。v4完成6计划+1完整曲线，T1=3仅入口输入，显式锁定、精确优先和删除一个实际训练形状后的route回退均通过。native_components_v7误将NTT TOML作为query-plan参数，读计划时失败，不重用该目录；v8正确参数完成。live_ntt_import另对M521小形状暖机1/正式1，实测后导入167形状并发布格式4成功。

component_main_holdouts对未测12.5e9/23.7e9分别执行两D×普通/承载6011四候选，暖1/正式3，再各执行未锁定自动曲线，共34；最大候选误差2.2305%，两个自动curve误差1.0970%/0.1175%，均选D120120/6011且有限排名损失0。58944 mandatory/233864 GMP bad0，无因子；用户1800MHz/默认55W设置不改、不并行GPU1或编译，GPU0外部负载不动。305 loaded NVML点SM1560…1800/median1800，power12.73…55.12/median47.14W，非严格恒频。完整边界为engine total，不含Stage1/进程/规划/发布；固定候选顺序n=3，不冒充全局或交错置信验收。原8%/5%门限不改。

component_main_audit核对102冻结测量输入和当前61二进制源。GPU测量后唯一变化的验收依赖是工作量辅助工具的旧调用兼容修正；冻结副本保留，当前工具通过最新CPU数值/完整计划重放，不将当前摘要冒充旧测量身份。正式文档改写当前主路径状态与范围，不保留“尚未接入”的失效结论；本临时草稿仍保留。

剩余完整目标继续推进：组合模型在真实Stage1成本下的独立Auto总流程收益、更多生产B1/位宽/实际batch与算法、预算、cold/driver及非驻留/G1。34固定B2曲线和T1=3入口不能替代这些验证，本轮不宣布全部完成。

## 格式4与真实Stage1成本的收益复验

收益工具接入独立组件参考，复用固定B2工具的纯Python训练/查询接口；原生训练系数不作为参考。格式4需要冻结训练计划，核对batch/buffers/physical/chunk策略，精确锚点保持优先；增加holdout B2时对全部D/算术组展开，同时保留全部原锚点。开始计时前冻结actual EXE/GMP/build源闭包及参考/性能/save/计划；收益工具按profile重建命名环境，增加只读NVML和退出清理。没有改变8%时间/5%收益损失或engine成本合同。

CPU profit_reference_cpu核对4组、8个未测查询与已冻结原生预测一致、32精确锚点优先、3坏训练目录拒绝、旧格式回退及NTT缺项不合格。GPU1计时前无编译/ECM进程，0%利用率；GPU0外部任务99%不动。实际加载模块已保存，EXE eea672d1…和GMP 969f0e96…。

使用余因子lcm/batch8实测T1=0.0320491875、B1=20/sigma26、32个锚点及12.5e9/23.7e9×两D/两算术共8留出候选，共40候选、各暖1/正式3，计划160完整曲线。未锁定Auto初选B2=2.6e9/D60060/承载6011，来自1316个原生决策候选；选择在下界，不声明连续或全局最优。证据component_auto_profit/，尚在运行；完成后核对全部候选、实际收益、算术及冻结身份，不能以计划或部分曲线宣告通过。

### 完成与独立审计

该进程已终止exit0，40候选/160曲线完整完成；最大时间误差1.71511%、有限收益排名损失0。正式选中点引擎数组2.278519/2.307135/2.324756 s，中位数2.307135，实测收益5.3064816；仍为B2=2.6e9/D60060/承载6011。276480 mandatory、944788 GMP bad0，无因子。104冻结输入、实际模块及61源闭包核对通过；独立审计与8种私有坏证据拒绝通过，原始目录不变、额外GPU曲线0。NVML1390点/1232 loaded点，SM1545…1800/median1800 MHz、power12.03…55.18/median47.135 W，非严格恒频。

另发现选中点完整进程中位数5.7157982 s，worker Auto规划中位数3.065344 s；该点每次完整搜索，其他39点固定B2/D/承载，不能混合进程收益排名。既有合同仍为引擎T2，独立端到端开销继续定位。已查明非父/worker重复搜索；组合Estimator反复构造请求、查询相同packing形状是下一待测假设，未据此宣称已优化。正式状态、性能范围和TODO同步；本草稿按仓库规则保留。
