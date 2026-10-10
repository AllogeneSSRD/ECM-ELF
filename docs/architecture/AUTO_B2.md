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

当前测量路径固定CUDA/CGBN ladder、Suyama PARAM0、自动TPI、关闭指数缓存和checkpoint；`--exponent [lcm|choose12]`显式选择标量。默认输入来自与完整Stage2 tune相同的13个已知梅森素数目录；`--target-n <N...>`改测指定余因子或其他模数，接受十进制/0x十六进制字面整数，不计算表达式，也不推断承载指数。N须为大于3的奇数且最多16384 bits，与`--exponents`互斥。同轮输入不能重复位宽/模数类型成本scope，以免用不同目标混合统计。

`--sigma-first <s>`默认26，要求整个批次的连续sigma处于6…9007199254740991；可为已验证输入选择适当的曲线范围。CPU独立生成每个sigma的末点，测量后逐条验证实际N、B1、PARAM、sigma、X、Z与checksum。指定合数只声明本次测量曲线完成且没有因子，不证明N为素数或未来曲线无因子；参考出现非单位、GPU找到因子、末点不符或输出缺失均拒绝发布。不能用checkpoint或部分运行速度投影发布成功文件。

高B1可加`--reference <stage1_gmp_reference.exe>`使用独立纯GMP末点参考，省去Python的大标量与点运算；不指定则保留Python参考。参考程序由`tools/build/test/build_stage1_gmp_reference.ps1 -Build <新目录> -VcVars <vcvars64.bat>`构建，纯CPU、不链接CUDA、不包含生产ECM算术代码。它用分段筛法收集最大素数幂、均衡乘积构造LCM，再用普通GMP模运算执行Montgomery ladder；choose12使用12倍标量。支持相同13个已知素数，或通过`--n <HEX_N>`指定实际模数（纯十六进制数字、不带0x）；B1=2…260e6、单范围1…4096条曲线。参考耗时完全排除T1。

每次原生参考返回完整输入范围、按sigma顺序排列的坐标和成功尾记录。收集器核对实际N，不能以位宽相同代替模数一致；拒绝缺尾、缺点、重复/乱序、范围不符或越界坐标；参考返回非单位或超时则保留证据并拒绝发布。`--reference-timeout <秒>`默认7200，单独限制每次原生参考进程；Python参考没有这一子进程限制。已计算的末点在本轮内按实际N/B1/sigma复用，扩大batch时只计算缺少的sigma。原生参考的二进制、可用的构建manifest/冻结源与GMP DLL身份只进入原始审计，不进入性能TOML。

`--tune-level <1..10>`默认1：位宽目录取前min(13,ℓ+3)项；B1目录依次为20、1000、10000、100000、1000000、10000000、26000000、100000000、260000000，取前min(9,ℓ)项；批次目录1、8、64、256，取前min(4,1+⌊(ℓ−1)/3⌋)项；每组合一次独立进程预热，正式重复2ℓ+1次。`--exponents <p...>`、`--b1 <B1...>`、`--batch <C...>`、`--repeats <n>`覆盖网格。最高默认468个范围、每范围21次正式重复，CPU独立点验证也可能很耗时。`--timeout <秒>`默认3600，只限制每个Stage1子进程，不承诺整轮时间上限。

T1样本=完整Stage1进程墙钟/C，包含进程启动、指数/曲线准备、GPU运算、归一化与save写入；不含CPU独立验证及预热。GPU秒数另存，不作为默认T1。短B1的成本可能主要由启动/准备构成，因此不把短基准外推生产B1。Stage1和Stage2两个成本口径并不包含完全相同的进程开销；完整总流程还需单独核对Stage2驱动成本。

格式1使用`[profile]`、`[device]`、`[policy]`、`[stage1.sample_<n>]`和`[summary]`，每范围保存重复时间、中位数、MAD、批次及指数模式，性能文件不含路径或二进制身份。发布前由原生reader核对所有范围和统计，再原子替换；失败保留原文件和证据。原生reader限16 MiB/4096范围，拒绝未完成、重复范围、未核验、非有限/负成本和不支持的策略。

运行查找精确匹配GPU UUID/SM、CUDA runtime/driver、目标位宽/类型、B1、`stage1_batch`及指数模式，不外推。指数模式默认读取共用INI的`exponent`，可用`--stage1-exponent [lcm|choose12]`覆盖该成本条件；它不重新计算save的Stage1点。余因子成本按普通目标N实测，不能把承载位宽或完整梅森数的Stage1成本当作余因子成本。性能文件记录适用位宽/类型，不含目标数、路径或二进制身份；原始证据保留实际目标和全部核验依据。文件不配置或启动生产Stage1，也不能验证将来所用Stage1构建具有同样速度；改变内核、频率、功耗、批量或缓存策略后应重新测量或提供实际成本。

## NTT tune

`--tune ntt` 测不同长度和批量的精确域卷积。参数含 `--length-log2 <a:b>`（3≤a≤b≤27）、`--tune-slices <s,...>`（1…65535，最多64个不同值）、`--tune-repeats <n>`、`--tune-memory-mb <MiB>` 与 `--tune-file <path>`；默认重复5次，每个形状另有一次预热，预热不计中位数。需要固定 Goldilocks 后端。

默认输出格式2的 `stage2_tune.toml`：`[profile]` 保存计时单位、长度范围、slices和重复次数，`[device]` 保存设备/后端/加减归约条件，`[policy.environment]` 保存运行策略，`[ntt.length_<L>.slices_<s>]` 保存对应形状的吞吐、样本、容量与核验结果，`[summary]` 保存完成状态。字段逐行排列，数组用于重复样本和radix；不记录路径、二进制或构建摘要。显式 `.jsonl` 输出仍可用。运行期间核对程序和已有构建manifest未发生变化，成功且至少测到一个形状后才原子替换目标文件；失败保留原文件与partial。

`--tune-level <1..10>` 的NTT预设为 log₂L=3…[20+min(ℓ−1,7)]、重复2ℓ²+1次，slices取序列1、4、16、64、256、1024、4096、16384、65535的前min(ℓ,9)项。等级10继续增加重复次数。显式 `--length-log2`、`--tune-slices`、`--tune-repeats` 覆盖对应预设，与参数顺序无关。未指定等级时保持16…27、slices=1、重复5次。内存预算不随等级提高；每个形状独立检查预算和实时free余量，超限记录 `skipped_memory`。`effort_level=0` 表示未使用等级预设，实际范围以profile字段为准。

NTT TOML 不参与自动 D/承载或 Auto B2 的运行选择；不能将 field-convolution 吞吐替代完整 Stage2 成本。

CUDA事件计时包含两次正向NTT和一次带乘积/缩放的逆NTT，不含输入生成、分配、传输和验证。若批量s的中位数为t秒，`conv_iter_per_s=s/t`，`batch_iter_per_s=1/t`；两者均不是ECM curves/s。每个slice使用不同常数项，独立GMP计算参考卷积；预热和每次正式测量都检查全部L·s个输出，包括应为零的位置。一次NTT tune不含S4归约、树准备、点运算和GCD，不能单独推导最优B2。

入口：[NTT测量](../../src/cuda/ecm_stage2_tune.cuh)、[等级预设与序列化](../../src/core/ecm_stage2_tune_format.h)、[命令与原子发布](../../src/core/ecm_cuda_stage2_main.cpp)。

按位宽校准 Auto B2 还需真实 Stage1 摊销、阶段成本、非满树/根操作、分块和驻留/回退数据。生成拟合、冻结预测、独立验证与收益排名后才能导出运行 profile。

## 完整 Stage2 tune

`--tune ecm` 运行生产 Stage2，默认输出 `stage2_ecm_tune.toml`。每个输入/B2/D/算术组合先独立进程预热一次，再按指定轮数运行独立进程。保存 `stage2_full_wall.total` 的样本、中位数及 MAD，主成本统计不含预热、Stage1准备、进程启动、调优规划及结果发布。算术核验完成后才产生测量回执。

每个新样本用`phase_accounting="exclusive_engine_v1"`声明十个互斥阶段，保留`phase_<name>_samples`配对数组及`phase_<name>_seconds`中位数：shape为已计入引擎的baby索引枚举；setup为设备上下文、归约器及必需自检；baby为生成/归一化点；ftree为F树构建与检查；main_setup为主流程准备；inverse_setup为多项式逆及循环准备；giant_loop为giant/G树/fold完整循环；descent为fold归一化及F树下降；accum为叶乘积、GCD及尾部处理；finalize为局部释放、最终算术检查及因子合并。前四项逐次和等于init，后六项逐次和等于main，全部逐次和等于total。独立阶段中位数仍不能相加代替总时长中位数。

旧`giant_seconds`、`gtrees_seconds`、`fold_seconds`、`descent_seconds`等计时字段继续保留，其边界可能嵌套，不能与上述十阶段相加。reader接受没有新计时合同的旧样本；声明新合同后必须包含全部配对数组和一致统计，不完整数据报错。合并可以保留不同样本各自的计时合同，不把旧样本冒充为互斥计时。

`worker_accounting="spawn_wait_exit_v1"`单独记录父进程从调用曲线执行器到返回的墙钟，含命令/管道/日志准备、子进程启动、运行、输出收集与退出。`worker_samples`与engine样本同次对应，`worker_overhead_samples`逐次等于worker−engine；分别保留中位数及worker MAD。残差还包含引擎边界之外的保存点读取、CUDA初始化/清理和回执写入，不是纯驱动或纯启动耗时，也不包括外层父程序启动和tune规划。它目前只供分析，未加入生产成本排名；不能同时把worker和engine相加。

默认基准使用已知梅森素数，sigma=26、B1=20，以独立GMP ladder生成归一化Stage1点。该准备不是生产Stage1计时或Auto B2的Stage1标定。`--tune-save FILE` 使用文件第一条有效记录；`--tune-carrier-exponent p` 对该记录同时测普通模数与承载，先验证N∣2ᵖ−1。save模式只声明所测曲线无因子，不宣称目标为素数或其他曲线不会产生因子。发现因子、算术坏计数或附加诊断使计时不干净时，停止调优并保留证据，不发布成功配置。

完整Stage2默认等级为1。等级ℓ：基准位宽数min(13,ℓ+3)，D数为ℓ+2（ℓ≥7再加2），重复2ℓ+1次，单曲线最多64+16(ℓ−1)棵G树。各等级保留前一等级的基础输入并扩大、细化网格；自适应补样按当前区间重新计算，不保证补样点逐级包含。高等级可能需要很长时间，不是固定秒数预算。

- 位宽依次加入：521、2203、4423、9689、1279、3217、11213、607、2281、4253、127、107、9941。
- D使用递增目录：30030、60060、120120、180180、210210、360360、570570、690690、810810、1021020、1141140、1381380、1711710、2282280。等级1…10分别取前3、4、5、6、7、8、11、12、13、14项。
- B2区间边界：2.6×10⁹、2.6×10¹⁰、2.6×10¹¹、2.6×10¹²、8×10¹²。等级ℓ覆盖前⌊(ℓ−1)/2⌋个区间；等级3…10每区间作4份对数等距细分，内部点按最近整数取整。等级1…10的基础B2点数分别为1、1、5、5、9、9、13、13、17、17。等级3、4的基础网格包含约4.624e9和14.621e9，以增加低端及中段支持；CPU网格检查不代替完整曲线精度验收。等级10基础组合3094项；当前默认256 MiB坐标分块、chain阈值32768和B1=20下，CPU目录枚举含补样共3578项。实际执行仍受显存和G树数量限制。
- `--tune-exponents p,...`、`--tune-d D,...`、`--tune-b2 B2,...`、`--tune-repeats n`覆盖相应预设；save与exponents互斥。`--tune-max-batches n`覆盖G树数量上限，0解除该耗时限制；它防止高B2搭配很小D产生极长基准，不是算术正确性门限。显存采用有效batch/arena/fold配置；`--tune-memory-mb`只用于NTT。G1、不支持的诊断/非驻留组合和静态free快照不满足的形状记录跳过；没有成功形状则不发布。

### 自适应giant短尾采样

等级1…2只有单个基础B2，默认不补样；等级3…10对每个输入/D/算术组请求3个ladder短尾点。`--tune-tail-samples <0..16>`覆盖该数量，0关闭。显式`--tune-b2`默认关闭补样，保留用户的精确列表；同时指定正的`--tune-tail-samples`才在该列表的最小/最大B2之间补样。

采样先将原B2区间与B2>B1、I>P及G树上限相交，向原生规划器查询该组的giant分块容量C、chain阈值m和强制ladder策略。I=⌊B2/D⌋+2；在C≥m且未强制ladder时，0<I mod C<m的余量进入ladder，其余正常块走chain。短尾点分布在可用首、中、末分块及约m/16…15m/16的尾量附近；优先调整分块以容纳目标尾量，无法命中时在可用尾区间分散取点。区间裁剪、整数去重和已测点可能改变最终位置。必要时另补chain锚点，使混合组有至少3个chain锚点并争取满足总样本≥7；最多补4个chain点。端点已占用时寻找其他可用尾部，不扫描整个巨大B2区间。

每个新增点与基础点使用同一显存准入、预热、正式重复和算术核验。探测最大B2时free不足不会直接取消整个组的补样，较小形状仍逐点检查。没有可用区间、单点、全ladder或没有短尾时输出原因；生成网格不等于成本拟合合格，更不等于独立时间预测已通过。纯ladder、样本不足或留一不合格的组继续只使用精确实测点。

profile声明`sampling_model="giant_tail_grid_v1"`和请求的`tail_samples`；样本用`sampling_source=[base|ladder_tail|chain_anchor]`说明来源。reader核对新增来源与实际分块工作量，兼容未声明采样模型的旧文件；合并保留各样本来源，采样数量取输入最大值，不把它解释成合并后各组的实际尾点数。控制台`ecm_tune_grid_ready`列出基础、短尾、chain及全部计划数量，最多4096项，超限在执行曲线前拒绝。原生探测计划独立保存在原始证据目录，不写进性能profile。

当前范围/路线和默认目录容量已通过CPU检查；完整曲线采样已核对30个范围的计划与回执。四候选区间验收仍未完成：冻结测量中普通D60060低B2点留一误差8.506%，模型正确拒绝预测；当前分块调整后的完整曲线复测、低G成本及独立四候选覆盖仍见[TODO](../TODO.md)。当前原生规划可生成不同尾量，不据此声明时间模型已经合格。

TOML格式3按`[profile]`、`[device]`、`[policy]`、`[policy.environment]`、`[ecm.sample_<n>]`、`[summary]`组织；策略每键一行，重复时间用数组。配置只包含性能、校验与适用条件，不含路径、程序或构建摘要。reader兼容格式2的环境串并在内存中规范为命名字段，限制64 MiB/4096个范围，可容纳等级10的完整默认网格及逐次阶段记录。原始plan、子进程日志和测量回执保留在`data/experiments/ecm_tune_<id>/`，与可编辑性能配置分开。发布前核对设备/策略、完整状态、样本统计及运行期间程序未变化，随后原子替换；失败不覆盖已有配置。

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

CPU工具`analyze_stage2_tune_workload.py`从完整tune的冻结计划与正式回执生成工作量TOML，按阶段/NTT长度/物理分块统计逻辑乘法和输出规模，保留配对计时。新样本核对原始互斥阶段回执与发布数组；旧样本保留嵌套计时，不补造新阶段。NTT格式1只提供已测单slice的串行参照，无法确认完整策略；格式2按(L,s)精确匹配，检查完整声明网格、样本统计、核验字数和吞吐单位，单独核对设备/归约/环境策略。缺失形状显式计数，不插值。匹配批量的参照为physical calls×该批量中位数；不是完整阶段成本，也不能与含NTT的阶段计时相加。目前输出不参与生产排名，仍需阶段组合模型及完整曲线独立留出验证。用法见[调优工作量与阶段特征](../../tools/bench/README_STAGE2_CARRIER_PLAN.md#调优工作量与阶段特征)。

### B2区间成本预测

完整ECM tune发布`prediction_model="giant_route_cost_v2"`。未声明预测模型或声明旧`linear_giant_points_v1`的profile仍可读取，只使用精确实测点；未知模型名报错。合并保留已声明的模型，但每个输入位宽/算术/B1/D组仍独立检查资格；缺少分块工作量的样本不能据此获得预测资格。同组已有请求B2的实测点时，采用该实测点。

每个新样本以`giant_work_model="chunk_routes_v1"`记录点分块容量C、chain阈值、强制ladder策略，以及chain/ladder点数、分块数和ladder迭代量L。I=⌊B2/D⌋+2，按C划分满块与尾块；未强制ladder且块点数≥阈值时使用chain，否则使用ladder。L为全部ladder点标量iD的⌊log₂(iD)⌋之和，利用2的幂边界精确计数，不逐点枚举。reader重新计算这些字段，拒绝不一致的工作量或缺失合同。

固定D的近似模型为T=α+βI+γL+δQ，Q为ladder分块数，系数有限且非负。同组需3…128个驻留实测点，I严格递增且最大I至少为最小I的2倍；整数特征须能由double精确表示，分块策略完全相同。只有chain的组采用α+βI；混合组至少7个点，且无ladder与含ladder的样本分别至少3个。纯ladder组当前只使用精确点。非负最小二乘及每个点的留一预测相对误差≤8%检查通过后，才拟合所有点。请求B2严格位于实测区间内部，端点只接受精确实测结果；含ladder的请求还须在实测正L的最小/最大值之间，不预测未测ladder分支。

Auto B2每组锚点只执行一次资格检查和拟合，在本次选择中复用系数、留一误差和MAD；各B2独立检查范围和分块特征。组内某个中点含未测ladder，不会使其他有资格的chain请求失去网格搜索。它不持久化模型、不复用其他位宽/B1/D的系数，也不缓存显存准入结果。未声明模型或不合格的组继续只使用精确点，误差门限、成本保护项及候选排序不变。实现入口为`prepare_b2_model`、`giant_work`与`auto_candidates`。

输出使用`estimated_seconds`，并给出实测B2范围、点数、最大留一相对/绝对误差和最大MAD；不会将估计标为实测中位数。这些误差只描述已有样本，不是新请求的统计置信界或精度保证。尾树、分块与NTT形状台阶仍可能改变成本，当前没有位宽、B1、D或区间外外推；资格不满足则保留原选型。最终显存检查始终针对请求B2，不能复用锚点的内存准入结果。独立完整曲线验证见[性能说明](../performance/STAGE2.md#b2区间预测验证)。

新样本在进入排名结果前，还核对请求的原生giant规划：分块容量、chain阈值及强制ladder策略必须与实测一致。固定B2选型不匹配时保留原路径；Auto B2没有匹配候选则明确失败。它与实时联合显存准入是独立检查，精确点也不能跳过新样本的分块策略检查。

已知限制：分块特征不能消除G树批次、NTT形状、GPU占用率及驱动成本的所有台阶，留一检查仍不是独立区间精度证明。5872-bit/B1=20、两D及普通/承载6011的12e9/33e9独立四候选检查通过，最大时间误差2.820%、排名损失0；范围及证据见[giant短尾预测验证](../performance/STAGE2.md#giant短尾预测验证)。这不覆盖其他位宽、生产B1、预算、纯ladder或所有内部B2。旧I线性模型的失败证据保留，不能据CPU合成数据或两个留出点宣称通用精度。

### 独立候选排序验证

[validate_stage2_tune_selection.py](../../tools/bench/validate_stage2_tune_selection.py)读取性能profile和一份有效Stage1 save，使用独立NumPy最小二乘与整数分块参考核对原生候选成本。参数为`--exe <file>`、`--profile <file>`、`--save <file>`、`--device <id>`、`--holdout-b2 <B2...>`、`--output <新目录>`；`--repeats <n>`默认2且至少2，`--min-candidates <n>`默认4且至少2，`--timeout <s>`默认1800。输出放在`data/experiments/`。该验证工具只接受save中以字面整数记录的N，不替代生产表达式解析器。

每个holdout必须未出现在适用组的调优数据中。逐候选检查整除资格、预测资格和当前联合显存，再各运行一次预热及正式重复；实测中位数用于独立比较。每个可用候选的预测相对误差要求≤8%，自动选中候选的中位数相对实测最快候选损失要求≤5%；最后运行一条不指定D/承载的完整曲线，核对生产入口实际执行与原生选择一致。这里的最快只指本次合格候选集，不是所有D/算术的全局最优。无足够候选或任何门限/算术/驻留检查失败均保留证据并报告不合格，不删失败组、不改profile、不放宽门限。日志、回执和逐步汇总在新输出目录；性能文件本身仍不记录这些路径或二进制身份。

总流程收益的有限候选验证由`validate_stage2_tune_auto_profit.py`执行：指定`--exe`、`--profile`、`--stage1-profile`、`--save`、`--device`、`--stage1-batch`及新`--output`目录，可选`--stage1-exponent [lcm|choose12]`、`--ratio <R>`和`--repeats <n>`（默认3，至少2）。读取匹配的完整T1实测，按全部适用B2/D/承载锚点及自动选中点构成有限集合；每点预热1次，再交错正式重复。自动选中点通过不强制B2/D/承载的生产入口执行，其他点显式固定候选。工具独立计算K/[T1+R·T2]，保持时间误差≤8%、收益排名损失≤5%；任何失败均保留证据，不发布通过结论。进程墙钟另列，不暗中加入引擎T2或改变排名合同。当前生产B1验证范围见[性能数据](../performance/STAGE2.md#生产b1标定与有限候选收益验证)。

`test_stage1_tune_runtime.py`核对实测T1的CLI/INI和私有队列。默认队列使用M521；其他梅森余因子使用`--queue-exponent <p>`明确原数，先验证save中的实际N整除2ᵖ−1，再把已知因子乘积写入私有队列。它不修改用户生产队列、不推断承载指数。该工具只接受字面整数N，支持可选`--choose12-profile`；省略时明确不验收该模式接受路径，保留缺失scope拒绝检查。

## 旧组件标定资格与当前证据

导出要求完整声明网格的身份/算术/覆盖通过，逐条时间误差≤10%，收益排名损失≤5%；失败 scope 不可删除后发布部分子集。重放点参加精度检查，但不充当独立收益排名样本。

完整 G1/G2/bridge 校准与验证共 1134 条曲线、63 个 scope，算术/身份通过；42 个 scope 达到时间门限，21 个失败，误差−60.138%…+13.718%。六组收益排名通过、最大损失0.633%，整体仍不合格，导出器拒绝生成新 profile。证据在 `data/experiments/ecm_auto_b2_bridge_20261006_{profile,audit,evidence}.json`。

两次同输入时间 a≤b 的固定单值预测最小最坏相对误差为 (b−a)/(b+a)。±10% 要求 b/a≤1.2222；存在超过这个比例的重复样本时，仅换拟合器不能满足该合同。需先处理等待波动，或明确批准新的长期均值/时间区间合同。

## 入口

- [ecm_stage2_cost_profile.h](../../src/core/ecm_stage2_cost_profile.h#L32)：reader、scope、`Work` 与 `choose`。
- [ecm_stage2_tune_ecm.h](../../src/core/ecm_stage2_tune_ecm.h)：完整Stage2等级、素数点准备、统计及TOML reader。
- [ecm_stage2_phase_times.h](../../src/core/ecm_stage2_phase_times.h)：引擎互斥阶段边界及时间守恒检查。
- [ecm_stage2_tune_prediction.h](../../src/core/ecm_stage2_tune_prediction.h)：B2分组、非负成本拟合及留一资格检查。
- [ecm_stage2_giant_work.h](../../src/core/ecm_stage2_giant_work.h)：giant满块/短尾路线与ladder迭代量精确计数。
- [ecm_stage2_tune_grid.h](../../src/core/ecm_stage2_tune_grid.h)：B2区间/G树上限和自适应短尾、chain采样。
- [ecm_stage2_tune_auto.h](../../src/core/ecm_stage2_tune_auto.h)：完整tune的Auto B2候选与收益排名。
- [ecm_stage1_tune_profile.h](../../src/core/ecm_stage1_tune_profile.h)：完整Stage1成本reader及精确范围查询。
- [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp)：`run_ecm_tune`、`select_tuned`、`select_auto_tuned`和`curve_worker`。
- [驱动资格检查](../../src/core/ecm_cuda_stage2_main.cpp#L617)、[收益公式](../../src/core/ecm_stage2_cost_profile.h#L185)。
- [measure_ecm_costs.py](../../tools/bench/measure_ecm_costs.py)、[fit_ecm_costs.py](../../tools/bench/fit_ecm_costs.py)、[validate_ecm_costs.py](../../tools/bench/validate_ecm_costs.py)、[audit_ecm_costs.py](../../tools/bench/audit_ecm_costs.py)、[export_ecm_cost_profile.py](../../tools/bench/export_ecm_cost_profile.py)。
- [当前 TODO](../TODO.md)。
