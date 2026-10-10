# Auto B2 与 tune

## 目标与可用性

Auto B2 优化连续生成/处理曲线的单位时间收益，计入 Stage1 成本，即使本次读取 save。它不是有限已有 save 的每条 Stage2 时限选择器，也不是强制某个 T2/T1 比值。

当前有两种Stage2成本来源：完整ECM tune TOML，以及独立标定的旧组件成本 `.cprof`。完整ECM tune可联合选择B2、D和合法梅森承载，要求提供正的Stage1每曲线成本，或加载匹配的完整Stage1实测；没有数据的范围不自动外推。旧组件成本路径保留其二进制/算术与标定资格检查，当前生产组合没有新的合格 `.cprof`。

## 原生接口

`--auto-b2 --tune-profile <file.toml> --stage1-seconds-per-curve <秒>` 对最终B2=0的任务生效；INI对应`stage2_auto_b2=1`、`stage2_tune_profile`和`stage1_seconds_per_curve`。指定完整tune时优先使用它，不借用 `.cprof` 的Stage1成本，也不因tune不适用静默改用旧模型。旧接口为`--auto-b2 --cost-profile <file.cprof>`。

显式同时给非零 `--b2` 与 `--auto-b2` 报冲突；队列或INI已有非零B2时保留固定值，并按完整tune选择D/算术。

`--auto-min-b2`/`--auto-max-b2`只缩小已测范围，非零`--d`锁定D，显式`--carrier-exponent`（包括0）锁定算术。`--arena-mb`/`--owner-budget-mb`选择匹配预算，`--stage2-ratio-adjust`给Stage2成本乘正系数R。

有限正的`stage1_seconds_per_curve`优先，该值应是目标设备、位宽、B1和批次下已经摊销的每曲线Stage1成本；`stage1_batch`不会再将它除一次。未提供时可用`--stage1-tune-profile FILE.toml`或INI的`stage1_tune_profile`查询实测中位数。两者都没有则拒绝选择。T1不是本次save读取耗时，也不是Stage2 tune生成B1=20点的准备耗时。旧组件模型仍使用其自身已测摊销范围，不借用独立Stage1文件。

每个curve worker按当前free VRAM重新选择，成功结果保留真实B2/D/承载、原始请求及规划时长；规划/执行失败保留未完成队列。完整tune Auto B2结果用`auto_plan`记录联合选择，`T1_source`区分显式成本与Stage1实测，不再执行第二次D/算术选择以改变已排名的组合。未完成队列的身份包含有效Stage1实测文件摘要和指数模式，改变成本后不能静默复用原进度。plan-only输出联合选择与实际形状，不推进队列。

## 收益和成本

P=φ(D)/2，I=⌊B2/D⌋+2，G=⌈I/P⌉。沿用 Prime95 相对收益近似：

a=1.96617−0.06781·log₁₀B1；K=0.11343+0.88657·(log₁₀(B2/B1)/2)ᵃ。

完整ECM tune路径选择score=K/[T1+R·Tguard]最大的候选。精确实测点Tguard=median+2·MAD；合格区间预测Tguard=估计时间+最大留一绝对误差+2·最大MAD。输出`engine_seconds`是未加不确定量的成本，`T2=R·engine_seconds`；排名使用`guarded_engine_seconds`。K不是绝对成功概率，R不是目标阶段耗时比。

完整tune的成本口径为`stage2_full_wall.total`：包含引擎初始化、自检及Stage2主体，不含Stage1、进程启动、调优/驱动规划与结果发布。当前没有给这条路径另加冷启动或驱动开销，因此排名是该成本口径下的近似收益，不能声明已准确预测整个进程的单位时间收益。

旧组件成本路径选择score=K/[T1+R·(Tengine+Tcold)]最大的候选。

旧组件模型的Tengine按实际树组、Newton逆、G1局部逆/根归约、fold、scaled下降、giant chain/ladder、pack/copy和固定项组合；每个scope有17个非负率。冷启动差额单独估计，不保证每条进程wall达到同样精度。

完整tune搜索保留各组已测B2点；通过区间预测资格的组再加入65个对数网格位置及附近整数I平台的边界。范围限制与每组实测区间取交集，不跨越未覆盖的范围。候选按收益排序，逐项查询请求B2/D/承载下的正常驻留联合显存；选择通过有效、完整及当前free快照检查的首项。缺少T1、策略不符、无合格成本或所有候选不满足显存条件时明确失败，不以B2=0运行曲线。

`range_limited`表示选中B2处在裁剪范围首/末整数I平台，并非连续搜索最优性的证明。边界标记为false也不证明全局最优。当前全部候选不满足显存时可能重复执行较昂贵的CPU生命周期规划，需要进一步减少重复规划。

旧组件搜索在各scope并集内部生成对数B2点，并加入G树、giant chunk、chain阈值与整数平台端点；仍受其P/G/B2/D/path范围及预算检查。

## 旧组件 Profile 合同

`.cprof` v2 是版本化文本，不依赖 Python/JSON runtime。身份包含程序 SHA、GPU UUID/SM、CUDA runtime/driver、算术后端、outer、feature/accounting、命名策略与来源/审计 SHA。范围包含位宽、B1、D、P/G/B2、arena、resident 与实际 chain/ladder 覆盖。

reader限制1 MiB，拒绝旧格式、缺失END、重复键、非法整数、非有限/负率、未覆盖路径及身份不符。旧组件Auto B2限制已测精确梅森模数和≤8192 bits；余因子/梅森承载、高B1、choose12、其他设备不自动外推。当前canonical等算术变化的保护会拒绝没有相应校准的组合。

组件准入不能保证进程峰，输出 `process_peak_guaranteed=false`。当前全流程显存模型边界见 [内存](MEMORY.md)。

## 完整Stage1成本预计算

[tune_stage1_cost.py](../../tools/bench/tune_stage1_cost.py)运行完整CUDA Stage1批次，保存可复用的每曲线T1。必填参数为`--stage1 <exe>`、`--stage2 <exe>`、`--device <id>`、`--output <新证据目录>`和`--profile <输出.toml>`；原始输出置于`data/experiments/`。Stage2可执行文件只提供设备查询和原生文件检查，不参与Stage1计时。`--check-stage1-tune-profile <file>`离线校验格式、完成状态和全部样本，不查询GPU。

当前测量路径固定CUDA/CGBN ladder、Suyama PARAM0、自动TPI、关闭指数缓存和checkpoint；`--exponent [lcm|choose12]`显式选择标量。输入来自与完整Stage2 tune相同的13个已知梅森素数目录。CPU独立生成每个sigma的末点，测量后逐条验证实际N、B1、PARAM、sigma、X、Z与checksum；不能用checkpoint、部分运行速度投影、因子曲线或缺失末点发布成功文件。

高B1可加`--reference <stage1_gmp_reference.exe>`使用独立纯GMP末点参考，省去Python的大标量与点运算；不指定则保留Python参考。参考程序由`tools/build/test/build_stage1_gmp_reference.ps1 -Build <新目录> -VcVars <vcvars64.bat>`构建，纯CPU、不链接CUDA、不包含生产ECM算术代码。它用分段筛法收集最大素数幂、均衡乘积构造LCM，再用普通GMP模运算执行Montgomery ladder；choose12使用12倍标量。支持相同13个已知素数、B1=2…260e6、单范围1…4096条曲线。参考耗时完全排除T1。

每次原生参考返回完整输入范围、按sigma顺序排列的坐标和成功尾记录。收集器拒绝缺尾、缺点、重复/乱序、范围不符或越界坐标；参考返回非单位或超时则保留证据并拒绝发布。`--reference-timeout <秒>`默认7200，单独限制每次原生参考进程；Python参考没有这一子进程限制。已计算的末点在本轮内复用，扩大batch时只计算缺少的sigma。原生参考的二进制、可用的构建manifest/冻结源与GMP DLL身份只进入原始审计，不进入性能TOML。

`--tune-level <1..10>`默认1：位宽目录取前min(13,ℓ+3)项；B1目录依次为20、1000、10000、100000、1000000、10000000、26000000、100000000、260000000，取前min(9,ℓ)项；批次目录1、8、64、256，取前min(4,1+⌊(ℓ−1)/3⌋)项；每组合一次独立进程预热，正式重复2ℓ+1次。`--exponents <p...>`、`--b1 <B1...>`、`--batch <C...>`、`--repeats <n>`覆盖网格。最高默认468个范围、每范围21次正式重复，CPU独立点验证也可能很耗时。`--timeout <秒>`默认3600，只限制每个Stage1子进程，不承诺整轮时间上限。

T1样本=完整Stage1进程墙钟/C，包含进程启动、指数/曲线准备、GPU运算、归一化与save写入；不含CPU独立验证及预热。GPU秒数另存，不作为默认T1。短B1的成本可能主要由启动/准备构成，因此不把短基准外推生产B1。Stage1和Stage2两个成本口径并不包含完全相同的进程开销；完整总流程还需单独核对Stage2驱动成本。

格式1使用`[profile]`、`[device]`、`[policy]`、`[stage1.sample_<n>]`和`[summary]`，每范围保存重复时间、中位数、MAD、批次及指数模式，性能文件不含路径或二进制身份。发布前由原生reader核对所有范围和统计，再原子替换；失败保留原文件和证据。原生reader限16 MiB/4096范围，拒绝未完成、重复范围、未核验、非有限/负成本和不支持的策略。

运行查找精确匹配GPU UUID/SM、CUDA runtime/driver、目标位宽/类型、B1、`stage1_batch`及指数模式，不外推。指数模式默认读取共用INI的`exponent`，可用`--stage1-exponent [lcm|choose12]`覆盖该成本条件；它不重新计算save的Stage1点。当前脚本只发布已知完整梅森数成本，不据此匹配余因子或其他Stage1算法。文件不配置或启动生产Stage1，也不能验证将来所用Stage1构建具有同样速度；改变内核、频率、功耗、批量或缓存策略后应重新测量或提供实际成本。

## NTT tune

`--tune ntt` 测不同 log₂L 的精确域卷积。参数含 `--length-log2 <a:b>`、`--tune-repeats <n>`、`--tune-memory-mb <MiB>` 与 `--tune-file <path>`；默认重复 5，预热不计中位数。需要固定 Goldilocks 后端。

默认输出 `stage2_tune.toml`：`[profile]` 保存格式、计时单位及测量范围，`[device]` 保存设备/后端适用条件，`[ntt.length_<L>]` 保存该长度的吞吐、样本、容量与核验结果，`[summary]` 保存完成状态。字段逐行排列，数组用于重复样本和 radix；不记录路径、二进制或构建摘要。显式 `.jsonl` 输出仍可用，其 profile 行同样不再含二进制/构建摘要。运行期间仍核对程序和已有构建 manifest 未发生变化，完整成功后才原子替换目标文件，失败留下 partial。

`--tune-level <1..10>` 当前用于 NTT 测量：等级 ℓ 的默认长度范围为 log₂L=16…[20+min(ℓ−1,7)]，重复次数为 2ℓ²+1；每个长度另有一次预热。显式 `--length-log2`、`--tune-repeats` 始终覆盖等级预设，与参数顺序无关。未指定等级时保持 16…27、重复5次。内存预算不随等级提高；越界长度记录 `skipped_memory`，不参与性能选择。`effort_level=0` 表示未使用等级预设，实际范围和重复次数以同一 profile 的字段为准。

NTT TOML 不参与自动 D/承载或 Auto B2 的运行选择；不能将 field-convolution 吞吐替代完整 Stage2 成本。

iter/s 指每秒完整 field convolution 次数，不是每秒 ECM 曲线，也不含 S4 系数归约、树准备和 GCD。profile 同时记录设备/后端、实际长度、容量、预热/样本和精确性核对。一次 NTT tune 不能单独推导最优 B2。

按位宽校准 Auto B2 还需真实 Stage1 摊销、阶段成本、非满树/根操作、分块和驻留/回退数据。生成拟合、冻结预测、独立验证与收益排名后才能导出运行 profile。

## 完整 Stage2 tune

`--tune ecm` 运行生产 Stage2，默认输出 `stage2_ecm_tune.toml`。每个输入/B2/D/算术组合先独立进程预热一次，再按指定轮数运行独立进程。保存 `stage2_full_wall.total` 的样本、中位数及 MAD，中位数统计不含预热、Stage1准备、进程启动、调优规划及结果发布；阶段字段是各自中位数，存在包含关系，不能相加替代总时长。算术核验完成后才产生测量回执。

默认基准使用已知梅森素数，sigma=26、B1=20，以独立GMP ladder生成归一化Stage1点。该准备不是生产Stage1计时或Auto B2的Stage1标定。`--tune-save FILE` 使用文件第一条有效记录；`--tune-carrier-exponent p` 对该记录同时测普通模数与承载，先验证N∣2ᵖ−1。save模式只声明所测曲线无因子，不宣称目标为素数或其他曲线不会产生因子。发现因子、算术坏计数或附加诊断使计时不干净时，停止调优并保留证据，不发布成功配置。

完整Stage2默认等级为1。等级ℓ：基准位宽数min(13,ℓ+3)，D数为ℓ+2（ℓ≥7再加2），重复2ℓ+1次，单曲线最多64+16(ℓ−1)棵G树。各等级保留前一等级的输入并扩大、细化网格；高等级可能需要很长时间，不是固定秒数预算。

- 位宽依次加入：521、2203、4423、9689、1279、3217、11213、607、2281、4253、127、107、9941。
- D使用递增目录：30030、60060、120120、180180、210210、360360、570570、690690、810810、1021020、1141140、1381380、1711710、2282280。等级1…10分别取前3、4、5、6、7、8、11、12、13、14项。
- B2区间边界：2.6×10⁹、2.6×10¹⁰、2.6×10¹¹、2.6×10¹²、8×10¹²。等级ℓ覆盖前⌊(ℓ−1)/2⌋个区间；等级3…4每区间作2份对数等距细分，等级5…10作4份，内部点按最近整数取整。等级1…10的B2点数分别为1、1、3、3、9、9、13、13、17、17。等级10默认最多3094个输入/B2/D组合，实际仍受显存和G树数量限制。
- `--tune-exponents p,...`、`--tune-d D,...`、`--tune-b2 B2,...`、`--tune-repeats n`覆盖相应预设；save与exponents互斥。`--tune-max-batches n`覆盖G树数量上限，0解除该耗时限制；它防止高B2搭配很小D产生极长基准，不是算术正确性门限。显存采用有效batch/arena/fold配置；`--tune-memory-mb`只用于NTT。G1、不支持的诊断/非驻留组合和静态free快照不满足的形状记录跳过；没有成功形状则不发布。

TOML格式3按`[profile]`、`[device]`、`[policy]`、`[policy.environment]`、`[ecm.sample_<n>]`、`[summary]`组织；策略每键一行，重复时间用数组。配置只包含性能、校验与适用条件，不含路径、程序或构建摘要。reader兼容格式2的环境串并在内存中规范为命名字段。原始plan、子进程日志和测量回执保留在`data/experiments/ecm_tune_<id>/`，与可编辑性能配置分开。发布前核对设备/策略、完整状态、样本统计及运行期间程序未变化，随后原子替换；失败不覆盖已有配置。

### 汇集预计算结果

不同位宽、B1/B2或D范围可分批调优，再使用`--tune ecm`、重复的`--tune-merge FILE.toml`及`--tune-file combined.toml`汇集到一份运行配置。合并只读取文件，不运行曲线、不查询CUDA设备。所有输入必须通过原生reader，且设备、内存/后端策略、算法版本、计时单位、正式重复次数及预热次数一致；等级取最大值。不同重复次数不能直接合并，调优时可用`--tune-repeats`统一采样条件。

不同测量范围保留各自完整样本；相同范围按输入顺序使用最后一份，避免拼接未知热状态下的计时。合并输出格式3，兼容格式2输入，保留命名环境策略；输入文件不修改，输出必须与输入路径不同。完整校验后原子发布，失败保留旧输出。INI可指向待刷新的输出配置，配置更新成功后后续曲线使用新内容。设备频率、功耗和背景负载仍须由操作者保持可比，不能由文件合并证明。

## 实测 D 与承载选择

显式B2任务可配置`--tune-profile FILE.toml`或INI的`stage2_tune_profile`，CLI路径优先。每条生产curve worker和plan-only都使用原生reader。它检查UUID/SM、CUDA runtime/driver、固定后端、outer、add/sub、显存预算及NTT策略；性能配置不依赖二进制摘要。基准不启用独立debug日志，运行启用debug日志时给出未标定原因并保留现有选型；NTT_MEMORY_AUDIT非零时拒绝调优。

候选必须精确匹配目标位宽、算术位宽/类型、B1及已测D；B2使用精确实测点或下述合格区间预测。承载候选重新验证目标N实际整除2ᵖ−1。同位宽匹配是测量成本的适用条件，不是数学正确性的证明。非零显式D固定D；显式`--carrier-exponent`（包括0）固定算术模式。未指定算术时，匹配profile中的普通/合法承载候选参与同一排名。

实测排名采用median+2·MAD，预测排名采用估计时间+最大留一绝对误差+2·最大MAD。先检查排名较优候选在请求B2下的当前正常驻留联合内存模型。模型须有效、完整，且`required_free_bytes`不超过实时free快照；以联合峰、baby/fold/frontier预留需求计算，不能叠加模块各自峰。通过后把选定D和承载交给生产引擎，因此较大D不会再被legacy additive估计提前排除。引擎继续执行实际free检查、分配错误处理和回退，静态判断不保证物理驻留。无设备/策略匹配、没有合格成本或没有可用候选时记录原因并保留现有路径；损坏或未完成profile报错。

显式B2结果的`tune_plan`保留候选数、选定D/承载、实测或预测的成本与不确定量、所需free与实际free；原始D/承载请求单独保存。启用Auto B2时改用`auto_plan`记录联合选择。这些路径不外推位宽/B1/D，不读取NTT吞吐选择ECM参数；完整tune与旧组件成本适用各自合同。

GPU频率、功耗和背景负载属于测量条件，应在相同设置下调优和使用；改变设置后重新测量。当前profile的设备/策略匹配不验证这些动态条件。`stage2_plan`中的legacy耗时估计与`tune_selection`中的成本分别保留，tune排名使用后者。

### NTT工作量预计算

CPU工具`analyze_stage2_tune_workload.py`从完整tune的冻结计划与正式回执生成可读的工作量TOML，按阶段/NTT长度/物理分块统计逻辑乘法和输出规模，并保留配对计时。可与同设备的单slice NTT实测对照，但当前输出不是成本profile，也不参与生产排名：NTT运行策略、批量吞吐、互斥阶段成本及独立留出精度尚未完成资格检查。不能把field convolution iter/s直接换成完整曲线速度，也不能再将其加入已包含NTT的阶段计时。用法和单位见[调优工作量与阶段特征](../../tools/bench/README_STAGE2_CARRIER_PLAN.md#调优工作量与阶段特征)。

### B2区间成本预测

完整ECM tune发布`prediction_model="linear_giant_points_v1"`。旧profile未声明该字段时只使用精确实测点；未知模型名报错。合并保留已声明的模型，但每个输入位宽/算术/B1/D组仍独立检查资格。同组已有请求B2的实测点时，采用该实测点。

固定D的近似模型为T=α+βI，I=⌊B2/D⌋+2。只允许在同组3…128个驻留实测点之间预测，I严格递增且最大I至少为最小I的2倍；整数I必须能由double精确表示。最小二乘得到的α、β须有限且非负，每个点的留一预测相对误差均≤8%，全部检查通过后才拟合所有点。请求B2必须严格位于已测区间内部，端点只接受其精确实测结果。

输出使用`estimated_seconds`，并给出实测B2范围、点数、最大留一相对/绝对误差和最大MAD；不会将估计标为实测中位数。这些误差只描述已有样本，不是新请求的统计置信界或精度保证。尾树、分块与NTT形状台阶仍可能改变成本，当前没有位宽、B1、D或区间外外推；资格不满足则保留原选型。最终显存检查始终针对请求B2，不能复用锚点的内存准入结果。独立完整曲线验证见[性能说明](../performance/STAGE2.md#b2区间预测验证)。

### 独立候选排序验证

[validate_stage2_tune_selection.py](../../tools/bench/validate_stage2_tune_selection.py)读取性能profile和一份有效Stage1 save，使用Python独立线性回归核对原生候选成本。参数为`--exe <file>`、`--profile <file>`、`--save <file>`、`--device <id>`、`--holdout-b2 <B2...>`、`--output <新目录>`；`--repeats <n>`默认2且至少2，`--min-candidates <n>`默认4且至少2，`--timeout <s>`默认1800。输出放在`data/experiments/`。该验证工具只接受save中以字面整数记录的N，不替代生产表达式解析器。

每个holdout必须未出现在适用组的调优数据中。逐候选检查整除资格、预测资格和当前联合显存，再各运行一次预热及正式重复；实测中位数用于独立比较。每个可用候选的预测相对误差要求≤8%，自动选中候选的中位数相对实测最快候选损失要求≤5%；最后运行一条不指定D/承载的完整曲线，核对生产入口实际执行与原生选择一致。这里的最快只指本次合格候选集，不是所有D/算术的全局最优。无足够候选或任何门限/算术/驻留检查失败均保留证据并报告不合格，不删失败组、不改profile、不放宽门限。日志、回执和逐步汇总在新输出目录；性能文件本身仍不记录这些路径或二进制身份。

## 旧组件标定资格与当前证据

导出要求完整声明网格的身份/算术/覆盖通过，逐条时间误差≤10%，收益排名损失≤5%；失败 scope 不可删除后发布部分子集。重放点参加精度检查，但不充当独立收益排名样本。

完整 G1/G2/bridge 校准与验证共 1134 条曲线、63 个 scope，算术/身份通过；42 个 scope 达到时间门限，21 个失败，误差−60.138%…+13.718%。六组收益排名通过、最大损失0.633%，整体仍不合格，导出器拒绝生成新 profile。证据在 `data/experiments/ecm_auto_b2_bridge_20261006_{profile,audit,evidence}.json`。

两次同输入时间 a≤b 的固定单值预测最小最坏相对误差为 (b−a)/(b+a)。±10% 要求 b/a≤1.2222；存在超过这个比例的重复样本时，仅换拟合器不能满足该合同。需先处理等待波动，或明确批准新的长期均值/时间区间合同。

## 入口

- [ecm_stage2_cost_profile.h](../../src/core/ecm_stage2_cost_profile.h#L32)：reader、scope、`Work` 与 `choose`。
- [ecm_stage2_tune_ecm.h](../../src/core/ecm_stage2_tune_ecm.h)：完整Stage2等级、素数点准备、统计及TOML reader。
- [ecm_stage2_tune_prediction.h](../../src/core/ecm_stage2_tune_prediction.h)：B2分组、非负成本拟合及留一资格检查。
- [ecm_stage2_tune_auto.h](../../src/core/ecm_stage2_tune_auto.h)：完整tune的Auto B2候选与收益排名。
- [ecm_stage1_tune_profile.h](../../src/core/ecm_stage1_tune_profile.h)：完整Stage1成本reader及精确范围查询。
- [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp)：`run_ecm_tune`、`select_tuned`、`select_auto_tuned`和`curve_worker`。
- [驱动资格检查](../../src/core/ecm_cuda_stage2_main.cpp#L617)、[收益公式](../../src/core/ecm_stage2_cost_profile.h#L185)。
- [measure_ecm_costs.py](../../tools/bench/measure_ecm_costs.py)、[fit_ecm_costs.py](../../tools/bench/fit_ecm_costs.py)、[validate_ecm_costs.py](../../tools/bench/validate_ecm_costs.py)、[audit_ecm_costs.py](../../tools/bench/audit_ecm_costs.py)、[export_ecm_cost_profile.py](../../tools/bench/export_ecm_cost_profile.py)。
- [当前 TODO](../TODO.md)。
