# Auto B2 与 tune

## 目标与可用性

Auto B2 优化连续生成/处理曲线的单位时间收益，计入 Stage1 成本，即使本次读取 save。它不是有限已有 save 的每条 Stage2 时限选择器，也不是强制某个 T2/T1 比值。

当前源码已有原生 reader、搜索、CLI/INI/队列和导出工具，但当前生产算术/内存组合没有匹配的新合格成本 profile。使用显式 B2；不能把通过其他冻结二进制校准的 profile 交给当前程序。

## 原生接口

`--auto-b2 --cost-profile <file>` 对最终 B2=0 的任务生效。显式同时给非零 `--b2` 与 `--auto-b2` 报冲突；队列或 INI 已有非零 B2 时保留固定值。

`--auto-min-b2`/`--auto-max-b2` 只缩小已测范围，`--d` 限制实测 D，`--arena-mb`/`--owner-budget-mb` 选择匹配路径，`--stage1-batch` 选择实测 Stage1 摊销，`--stage1-seconds-per-curve` 可提供有限正成本，`--stage2-ratio-adjust` 对预测 T2 乘正系数 R。

每个 curve worker 按当前 free VRAM 重新选择，成功结果保留真实 B2/D、原始请求及规划时长；规划/执行失败保留未完成队列。plan-only 不推进队列。

## 收益和成本

P=φ(D)/2，I=⌊B2/D⌋+2，G=⌈I/P⌉。沿用 Prime95 相对收益近似：

a=1.96617−0.06781·log₁₀B1；K=0.11343+0.88657·(log₁₀(B2/B1)/2)ᵃ。

选择 score=K/[T1+R·(Tengine+Tcold)] 最大的已测候选。K 不是绝对成功概率，R 不是目标阶段耗时比。

Tengine 按实际树组、Newton 逆、G1 局部逆/根归约、fold、scaled 下降、giant chain/ladder、pack/copy 和固定项组合；运行 profile 每个 scope 有 17 个非负率。冷启动差额单独估计，不保证每条进程 wall 达到同样精度。

搜索在各 scope 的并集内部生成对数 B2 点，并加入 G 树、giant chunk、chain 阈值与整数平台端点。每项仍通过 P/G/B2/D/path 范围及预算检查；范围边界最优或 `range_limited=false` 都不证明连续全局最优。

## Profile 合同

`.cprof` v2 是版本化文本，不依赖 Python/JSON runtime。身份包含程序 SHA、GPU UUID/SM、CUDA runtime/driver、算术后端、outer、feature/accounting、命名策略与来源/审计 SHA。范围包含位宽、B1、D、P/G/B2、arena、resident 与实际 chain/ladder 覆盖。

reader 限制 1 MiB，拒绝旧格式、缺失 END、重复键、非法整数、非有限/负率、未覆盖路径及身份不符。Auto B2 目前限制已测精确梅森模数和≤8192 bits；余因子/梅森承载、高 B1、choose12、其他设备不自动外推。当前 canonical 等算术变化的保护会拒绝没有相应校准的组合。

组件准入不能保证进程峰，输出 `process_peak_guaranteed=false`。当前全流程显存模型边界见 [内存](MEMORY.md)。

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

结果的`tune_plan`保留候选数、选定D/承载、实测或预测的成本与不确定量、所需free与实际free；原始D/承载请求单独保存。此reader不外推位宽/B1/D、不读取NTT吞吐选择ECM参数、不选择B2。Auto B2成本模型仍受前述独立标定合同约束。

GPU频率、功耗和背景负载属于测量条件，应在相同设置下调优和使用；改变设置后重新测量。当前profile的设备/策略匹配不验证这些动态条件。`stage2_plan`中的legacy耗时估计与`tune_selection`中的成本分别保留，tune排名使用后者。

### B2区间成本预测

完整ECM tune发布`prediction_model="linear_giant_points_v1"`。旧profile未声明该字段时只使用精确实测点；未知模型名报错。合并保留已声明的模型，但每个输入位宽/算术/B1/D组仍独立检查资格。同组已有请求B2的实测点时，采用该实测点。

固定D的近似模型为T=α+βI，I=⌊B2/D⌋+2。只允许在同组3…128个驻留实测点之间预测，I严格递增且最大I至少为最小I的2倍；整数I必须能由double精确表示。最小二乘得到的α、β须有限且非负，每个点的留一预测相对误差均≤8%，全部检查通过后才拟合所有点。请求B2必须严格位于已测区间内部，端点只接受其精确实测结果。

输出使用`estimated_seconds`，并给出实测B2范围、点数、最大留一相对/绝对误差和最大MAD；不会将估计标为实测中位数。这些误差只描述已有样本，不是新请求的统计置信界或精度保证。尾树、分块与NTT形状台阶仍可能改变成本，当前没有位宽、B1、D或区间外外推；资格不满足则保留原选型。最终显存检查始终针对请求B2，不能复用锚点的内存准入结果。独立完整曲线验证见[性能说明](../performance/STAGE2.md#b2区间预测验证)。

### 独立候选排序验证

[validate_stage2_tune_selection.py](../../tools/bench/validate_stage2_tune_selection.py)读取性能profile和一份有效Stage1 save，使用Python独立线性回归核对原生候选成本。参数为`--exe <file>`、`--profile <file>`、`--save <file>`、`--device <id>`、`--holdout-b2 <B2...>`、`--output <新目录>`；`--repeats <n>`默认2且至少2，`--min-candidates <n>`默认4且至少2，`--timeout <s>`默认1800。输出放在`data/experiments/`。该验证工具只接受save中以字面整数记录的N，不替代生产表达式解析器。

每个holdout必须未出现在适用组的调优数据中。逐候选检查整除资格、预测资格和当前联合显存，再各运行一次预热及正式重复；实测中位数用于独立比较。每个可用候选的预测相对误差要求≤8%，自动选中候选的中位数相对实测最快候选损失要求≤5%；最后运行一条不指定D/承载的完整曲线，核对生产入口实际执行与原生选择一致。这里的最快只指本次合格候选集，不是所有D/算术的全局最优。无足够候选或任何门限/算术/驻留检查失败均保留证据并报告不合格，不删失败组、不改profile、不放宽门限。日志、回执和逐步汇总在新输出目录；性能文件本身仍不记录这些路径或二进制身份。

## 标定资格与当前证据

导出要求完整声明网格的身份/算术/覆盖通过，逐条时间误差≤10%，收益排名损失≤5%；失败 scope 不可删除后发布部分子集。重放点参加精度检查，但不充当独立收益排名样本。

完整 G1/G2/bridge 校准与验证共 1134 条曲线、63 个 scope，算术/身份通过；42 个 scope 达到时间门限，21 个失败，误差−60.138%…+13.718%。六组收益排名通过、最大损失0.633%，整体仍不合格，导出器拒绝生成新 profile。证据在 `data/experiments/ecm_auto_b2_bridge_20261006_{profile,audit,evidence}.json`。

两次同输入时间 a≤b 的固定单值预测最小最坏相对误差为 (b−a)/(b+a)。±10% 要求 b/a≤1.2222；存在超过这个比例的重复样本时，仅换拟合器不能满足该合同。需先处理等待波动，或明确批准新的长期均值/时间区间合同。

## 入口

- [ecm_stage2_cost_profile.h](../../src/core/ecm_stage2_cost_profile.h#L32)：reader、scope、`Work` 与 `choose`。
- [ecm_stage2_tune_ecm.h](../../src/core/ecm_stage2_tune_ecm.h)：完整Stage2等级、素数点准备、统计及TOML reader。
- [ecm_stage2_tune_prediction.h](../../src/core/ecm_stage2_tune_prediction.h)：B2分组、非负成本拟合及留一资格检查。
- [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp)：`run_ecm_tune`、`select_tuned`和`curve_worker`。
- [驱动资格检查](../../src/core/ecm_cuda_stage2_main.cpp#L617)、[收益公式](../../src/core/ecm_stage2_cost_profile.h#L185)。
- [measure_ecm_costs.py](../../tools/bench/measure_ecm_costs.py)、[fit_ecm_costs.py](../../tools/bench/fit_ecm_costs.py)、[validate_ecm_costs.py](../../tools/bench/validate_ecm_costs.py)、[audit_ecm_costs.py](../../tools/bench/audit_ecm_costs.py)、[export_ecm_cost_profile.py](../../tools/bench/export_ecm_cost_profile.py)。
- [当前 TODO](../TODO.md)。
