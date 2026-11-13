# Auto B2 与 tune

## 目标与生产入口

Auto B2 选择连续生成、处理新曲线的单位时间收益，计入 Stage1 成本，即使当前读取已完成的 save。固定 B2 时，性能文件选择 D 和合法梅森承载；最终 B2=0 且启用 Auto B2 时联合选择 B2、D、承载。

默认性能文件为 `ecm.toml` 所在目录的共享 `tune.toml`。若该目录只有旧 `stage2_ecm_tune.toml`，Stage2 可先读取它，Stage1 首次写入 `tune.toml` 时会带入其中的数据。可由 `stage2_tune_profile` 或 `--tune-profile FILE.toml` 覆盖。`--auto-b2` 对应 `stage2_auto_b2=1`。显式同时提供非零 `--b2` 与 `--auto-b2` 报冲突。任务或 TOML 已有非零 B2 时使用固定值。

每个`[workers.N.stage2]`可分别设置`stage2_tune_profile`，覆盖全局`[stage2]`路径；CLI仍优先。相对路径以ecm.toml目录为基准，读取、采样写回和更新锁均使用同一个解析后路径。显式路径不自动增加worker后缀，双GPU示例见[Stage2使用](../usage/STAGE2.md#cuda-入口)。

性能数据可跨实际位宽、B1 复用，允许插值、外推。复用从不要求 B1 相等；实际 B1 仍用于区间、成功概率、Stage1 成本和真实运算。模型估算与实测在输出中分别标识。

`stage2_tune_ignore` 默认 `gpu,driver,cuda,backend,environment`，可选项另含 `memory`；空值要求这些条件匹配。默认 memory 按 fold/frontier 执行路径匹配，不要求预算数值相等。忽略条件不会取消 save、算术、整除关系和当前显存检查，也不表示其他设备的秒数已经在本机验证。

非零 `--d` 锁定 D。`--carrier-exponent 0` 锁定普通运算，非零 p 锁定承载。`stage2_auto_carrier=1` 默认开启，CLI `--auto-carrier 0|1` 可覆盖：未锁定算术时，worktodo `ECMSTAGE2=1,2,p,-1,...` 先按已知因子重建实际 N，核对 save，再验证 **N 整除 2ᵖ−1**，将普通运算与梅森承载共同纳入选型。指数须满足 2≤p≤16384；非标准梅森表达式和单独读取 save 不猜测指数。

`stage2_auto_carrier=0` 关闭自动加入承载候选，不影响显式 `--carrier-exponent p`。显式 0 始终锁定普通运算。父进程将最终开关传给曲线子进程；该开关不改变目标 N/B2，因此改变它不重置队列完成数。配置定义及双语说明见 [TOML 参考](../ECM_TOML_REFERENCE.md#stage2_auto_carrier)；入口为 `src/ecm/stage2/cuda/main.cpp::arguments/main`，候选与校准资格为 `src/ecm/stage2/planning/planning.cpp::portable_carriers/ensure_portable_calibration`。

当前代码依据：`config/ecm_options.json` 定义键及默认值；`stage2/cuda/main.cpp::arguments` 解析参数，`stage2/task.cpp::run_tasks/child_run` 识别任务和投影子进程请求；`include/ecm/io/stage2_checkpoint.h::Task` 定义曲线进度身份；`stage2/planning/planning.cpp::portable_carriers/ensure_portable_calibration` 验证整除关系及短校准覆盖。规划在 `stage2/planning`，任务/子进程请求在 `stage2`，曲线进度合同在 `include/ecm/io`。

加入候选不等于启用承载。默认 `stage2_prediction_policy=model`：优先使用精确形状实测，其他输入分别预测普通和梅森算术的位宽区间成本，经过双侧误差保护及当前显存准入后选择。`paired`恢复精确目标位宽/B2/执行形状的配对实测门禁。承载只是扩大运算环，Stage1点、因子和GCD的目标仍为save中的N。

### 自动承载未被选中时

先区分候选识别与性能资格。规划中的 `carrier_exponent=p`、`lifted=1`、`target_divides_carrier=1` 表示已识别并验证承载；最终 `carrier_exponent=0` 表示选型保留普通路径，不能据此判断TOML开关未读取。D210/B242020短探针不作为跨位宽完整曲线锚点；区间校正只使用P≥512的干净完整实测。

`paired`策略及精确形状实测还核对实际batch、物理分块、缓冲数量、giant路线及fold/frontier状态，B1不参与匹配。`model`策略从当前请求重算工作量；预算数值和梅森指数不作为采样身份门槛。区间未覆盖或预测收益不足时保留默认。

未替换时保留具体原因。`paired`中的`default_planner_protected_missing_paired_baseline`表示缺少配对基线；`model`中的`default_protected_by_prediction_uncertainty`表示候选未通过预测保护。`alternatives`分别展示普通/梅森的算术位宽、D、秒数、来源和余量。这不构成全局最优证明；队列配对的冻结验证见[开发草稿](../development/STAGE2_TUNE_TASK_PAIR_DRAFT.md)。

## G树布局身份与自动选择

ECM矩阵的`groot_balanced`为`0|1`：0为叶子左侧连续布局，1为分块空位布局。它进入完整Shape、聚合身份、精确观察、请求成本、显存准入及跨曲线缓存；同一N/B2/D的两种布局不会合并均值。NTT有限域吞吐表仍按变换形状共用，不复制一套布局表。

当前写出format 13，区分布局、点块平衡、链长、实际点后端和S4身份，可读取format 8…13。缺少布局字段的旧行按其执行规则还原：算术字数W>128且至少4个完整G批次时为1，其余为0。缺少点策略的旧行固定解释为不平衡、64链，不能用新自动规则重新解释旧成绩；无实际后端证据的旧非零请求保留为未知。只读不重写原文件，再次发布才写出format 13及明确身份字段；B1仍不成为复用门槛。

CUDA自动布局保留上述宽位资格。其他位宽须同时满足：至少4个完整G批次、P不是2的幂、按原生精确packing查询得到的G树根最大NTT长度下降、整段G树的Σ批次数×NTT长度×log₂长度下降至少10%，且输出系数工作量增幅不超过10%。最后两项包括完整批和尾批，只是工作量筛选；fold/F树及当前显存准入仍按实际形状检查。未跨NTT拐点时保持连续布局。数学及限定输入性能见[G树布局](../performance/STAGE2_GTREE_LAYOUT.md)。

正式ECM tune在自动布局与旧默认不同的D上分别测量并记录两种布局，已足够的正式测量可复用；D搜索比较已完成配对中较快的成绩。运行时先保留各算术家族的默认及历史最快布局，再通过正常任务曲线补足新布局观察。显式D限制D，不阻止同D比较布局。`stage2_groot_balanced="auto"`为默认；true/false固定CUDA布局。CLI `--groot-layout auto|true|false`覆盖当前worker配置，保留0/1别名；auto不是第三种计算布局，最终仍记录0/1。专家环境变量`NTT_GROOT_BALANCED=0|1`与明确强制值冲突时报错，其固定D和隔离要求见[使用说明](../usage/STAGE2.md#固定参数g树实验)。OpenCL当前使用自己的连续G树实现，接受auto但拒绝强制CUDA布局。

## 点块与有限链长身份

矩阵使用 `giant_tail_balance=0|1`、`giant_chain_block=32|64`。新观察的tail字段表示最终点块计划是否实际借块：允许平衡但没有借块时为0，沿用原成本模型与Shape身份，能够复用已有tail=0成绩；实际借块为1，继续与原分块隔离。旧tail=1观察保留原身份，不重写原始成绩。字段进入观察聚合、精确查找、区间预测、行排序和主机规划缓存，即使忽略environment也不合并不同执行策略。跨曲线紧凑路由hint只用于实际tail=0、链长64的形状，并在恢复hint的点块容量后重新检查是否会借块；实际借块及32链不复用缺少这些身份的hint。NTT吞吐表继续共享。

统计条件不再记录请求用的 `env_giant_tail_balance`；读取旧条件时，严格环境匹配也跳过这一冗余许可字段，实际tail与链长继续由Shape逐项核对。其他环境维度的严格匹配不放宽。非负模型的拟合缓存同样按实际tail和链长分开，不能把原路径已合格的拟合结果直接用于尚无测量的点策略。原生计划缓存仍绑定完整环境及请求许可，因为最终准入可以改变粗略点容量。

令R=stage2_tune_min_repeats，未指定时为2。`stage2_giant_chain` 默认64；显式32固定链长。auto先保持64参考，用正常任务积累R次观察，再为每个算术家族当前最快的64链项增加一个32链对照，保持该项的D、G树布局、点块和驻留路径。未完成的观察优先补足，不把全部D候选乘上链长；剩余任务不足2R条时不启动新链长对照。达到R次后直接采用已准入项的最快均值，较慢32链不能替代64参考。关闭运行时探索后，仅使用达到门槛的已有观察；没有数据时保留64。

新点策略的工作量包含seed-pair的5+11⌊log₂i⌋次模乘、每次差分xADD的6次模乘、独立ladder的5+11⌊log₂(iD)⌋次模乘；不把模乘数当作GPU周期或胜者判据。默认不平衡且64链时保留原有D成本先验，避免未经全矩阵验证就改变默认排名；精确计数可用于独立解释实际路线。真实并行效率、seed增加和段积等待由完整实测决定。大B2的seed位步数按压缩点块与二进制边界计数，不枚举每个seed。

B2凑整可与运行时D、baby／giant后端及有限链长搜索共同使用。请求B2是最低覆盖合同，每个候选D先估算自己的实际凑整B2并计入模型；仅对有限候选进行原生显存／点块准入。实际B2可能改变点块容量，至多4步检查凑整与准入是否稳定，不稳定、溢出、超过AutoB2上限或显存不足时拒绝候选。只有最终稳定且已准入的形状可执行和采样，不能先选未凑整的便宜模型，再将另一覆盖量当作其成绩。

比较目标是完成至少覆盖请求B2的曲线所需时间，保留原先的重复数、搜索与额外时间预算；达到门槛的已准入项按含点启动检查的均值选优。性能行仍按**实际B2**分开；已有行只有在其实际B2等于当前请求按该D／容量凑整得到的结果时才可精确复用。同一实际覆盖平台内的多个请求可共享成绩，不同实际B2不能合并。`tune_selection`同时打印requested_B2与被选实际B2，runtime_candidates逐项打印实际B2。

进程内完整计划缓存绑定完整Shape、凑整开关及实时显存快照。紧凑跨曲线路由hint及其显存失败位图没有逐候选实际B2，凑整时不复用这些hint；关闭凑整仍保留原快速缓存。实际内存几何不同时重新建立参考，不仅凭相同D和B2借用旧点块身份。配置和覆盖合同见[完整区间点块策略](STAGE2.md#完整区间点块策略)。启动成本探测仍固定边界，显式ECM tune仍须关闭凑整。

功能回归使用RTX4060 Laptop／GPU1顺序执行。8440-bit输入的8条曲线在请求B2=26000000、D6006、实际B2=51879828下，依次完成两次baby标量参考、两次coop32及两次32链对照，并继续复用；关闭学习后性能文件保持不变。M127的两个worker各10条曲线分别读写独立相对路径，第二个worker不改写第一个文件；D180180／实际B2=28021233240和D219450／实际B2=28440281100均得到真实观察，全部覆盖请求27015978990，结果NF、GMP零错。此为联动和文件路由验证，不是双GPU性能比较或全位宽矩阵。

纯CPU重放冻结的生产条件，确认请求B2=550000000000可匹配D746130／实际B2=567296069340并恢复已有6次观察；跨GPU只规划对照不把另一设备的记录冒充本卡均值。关闭凑整的新旧完整曲线对照具有相同NTT分布、归约数量和G根摘要。证据根`data/experiments/stage2_rounded_runtime_20261101/`，复现入口为`tools/test/test_stage2_rounded_runtime_gpu.py`及`test_stage2_runtime_tune.py --fixture bounds`；代码入口为`planning/runtime_bounds.h`和`planning.cpp::portable_runtime_candidate_admit/portable_runtime_select`。

## 协作点后端选型

`stage2_baby_backend`默认auto，可强制scalar／coop32；`stage2_giant_backend`默认auto，可强制scalar／coop16／coop32。CLI `--baby-backend`／`--giant-backend`优先于当前worker TOML，显式TOML优先于环境；省略新键时兼容显式`NTT_BABY_COOP`／`NTT_GIANT_COOP`。普通曲线向子进程传递策略，子进程读档后才决定具体后端；校准曲线传递已经选定的后端。OpenCL的两项均仅接受auto／scalar，实际使用标量。

资格按**实际算术位宽**判定：8193…16384 bits，承载时取p。Baby设备预算回退或位宽不支持时实际为标量；giant没有合格chain时也记录标量。矩阵的`baby_coop`为0／32，`giant_coop`为0／16／32，0表示实际标量。这些字段参与聚合、精确查找、模型、缓存和默认保护；即使忽略environment，也不跨后端合并。Baby32／giant32使用160／192／224／256词分档，giant16固定256词；不跨容量边界套用实测。NTT吞吐表继续共享。

Baby auto先积累R次标量观察，R为`stage2_tune_min_repeats`。实际P≥512、尚有至少2R条队列工作时，对每个算术家族当前最快的已重复标量项，只增加一个coop32对照，固定D、布局、链长与驻留策略。先完成baby对照，giant也为auto时继续比较giant，随后恢复D与32／64链长搜索；不将全部D候选乘上后端数。达到R次后选已准入项的最快平均耗时，方差只作诊断，不否决较快均值。关闭运行时学习时只读复用已合格观察；缺少数据仍使用标量。

Giant auto以当前最快且达到R次的标量giant项为参考，先试coop32，再试coop16，每次只新增一个后端。每项与对应标量参考均需达到R次，已合格项按含点检查的均值选优；尚未完成的另一项不否决现有胜者。两项同时auto时可形成标量／标量→baby32／标量→baby32／giant32或16的有效对照路径，不要求每个D遍历全部2×3组合，也不能凭单独强制协作记录启用auto。队列剩余不足2R时不开启新的后端对照；已有观察及正在完成的曲线保留。

Giant配对复用要求当前设备／环境条件、目标与算术位宽、算术家族、实际B2、baby后端、链长、布局与驻留路径匹配。并行区域按⌊log₂⌈min(I,C)/c⌉⌋划分，I为giant点数、C为点块容量、c为链长；必须存在真实chain。位宽、链长、并行区域或显存路径跨界需要新对照。普通与承载分别学习。自动D搜索继续保留P≥512；显式锁定更小D时允许运行观察及合格giant对照，baby小P仍保持标量。

B2凑整保持已选择的后端请求，直到实际B2和原生容量完成稳定准入；不能因未凑整区间暂时没有chain而丢掉选择。采样与回执记录最终实际后端，准入或执行回退不能冒充协作成绩。

显式ECM tune同样支持两项auto：先在标量giant下完成baby对照，再固定选中的baby比较两种giant协作项。同一合格区域内，相邻D只测配对胜者，不重复展开后端网格。Baby区域按⌊log₂min(P,K)⌋划分，K为规范化后的ladder分段上限，默认8192；giant区域另按独立链数划分。不同D之间的两个快慢记录不构成后端配对。满足重复数的已有校准不再重测，默认无warmup。

`mean_seconds`仍是引擎init＋main，排除启动点算术自检。新增`point_check_seconds`记录必需点检查耗时；后端比较采用两者之和，避免遗漏额外TPI检查。外层读档、选型、进程及发布另计，不混入该字段；完整曲线墙钟用于性能验收。旧记录缺少此字段时按0保守保留参考。旧条件仅有非零环境开关、缺少实际后端证据时，新字段填255（未知）；保留原数据，但不用于选型或拟合，不能据请求开关猜测实际执行路线。

无协作实测时，基础模型仅对baby部分使用保守容量先验，给出至少50%的相对误差许可；不把ladder核倍率施加到整条曲线，也不据此绕过auto的配对要求。标量默认D先验保持不变。其他GPU、大B2和不同点批量仍需实测。

代码入口：`stage2/point_backend.h`定义选项及容量／批量区域；`cuda/host_policy.cpp`解析优先级；`task.cpp::child_run/curve_worker`处理父子进程；`planning/runtime_tune.h::runtime_baby_trials/runtime_giant_trials/qualified_point_backends`及`qualified_baby_backend/qualified_giant_backend`限制配对范围，`planning.cpp::portable_measure/portable_runtime_select/round_giant_bounds`负责准入与执行；`portable_model.h`、`width_prediction.h`隔离后端统计和模型。区间补测与完整曲线边界见[位宽资格](../performance/STAGE2_BIT_SCALING.md#126-区间资格与自动选型)，当前联合后端的完整矩阵见[Stage2性能](../performance/STAGE2.md)。

### Giant seed自动选型

`stage2_seed_backend`默认auto，可强制scalar／coop32；CLI `--seed-backend`覆盖worker TOML，省略键时兼容显式`NTT_GIANT_SEED_COOP=0|32`。CUDA实际算术8193…16384 bits可使用按160／192／224／256词分档的TPI32；范围外、没有真实chain或未使用paired seed时为标量。该策略同时控制GPU构建[D]Q和相邻seed，保留非单位回退；baby和giant链的后端仍独立配置。

Auto以R=`stage2_tune_min_repeats`为重复门槛。先完成baby／giant的已有对照，再对每个算术家族当前最快且完成R次的标量seed项，只新增一个coop32项，固定D、布局、链长、baby／giant算术和驻留路径。双方达到R次后按引擎加点启动检查的平均时间选优，随后恢复D和有限链长搜索；不将整个D表乘上后端数。新对照仍受队列剩余量和原有预算约束，已有观察和未完成对照保留。

资格复用要求当前硬件条件、目标／算术位宽、算术家族、实际B2、链长、并行链数区间、布局及驻留路径一致；D和B1可在同一区间复用。已完成的64→32有限链长完整对照，可与既有后端对照组成已验证的选型路径；关闭学习后仍保留该实际胜者，不要求重跑完整2×3×2网格。未配对的强制协作记录不构成auto依据，缺数据或只读且无资格时使用标量。

实际`seed_coop`进入每行matrix、Shape、模型和缓存；即使忽略environment也不跨实际后端合并。旧标量资料按0保留，旧非零环境请求却无实际后端证据时保留为255（未知）、不参与选型；加载后条件表不再以seed请求分裂。NTT表继续共享。正式ECM tune也先配对，再在相邻D只测已合格胜者；重复运行跳过已有足够观察。

验证范围为GPU1 RTX4060 Laptop、目标1500MHz／55W。28条有效曲线覆盖R=2的`0,0,32,32`学习顺序、R=1承载、只读复用、未配对保护、三个点后端及32／64链长联合auto；真实Worker #2验证覆盖／CLI优先／环境回退，正式tune在D30030建立两项对照、D27300只增加胜者一项，再次执行不追加测量。此为功能资格，不是完整位宽性能矩阵。证据在`data/experiments/stage2_s4_20261102/auto_native_final`、`auto_config_final`、`auto_formal_final`；算法和固定D成绩见[协作seed性能](../performance/STAGE2_BIT_SCALING.md#13-可选协作paired-seed与dq)。

### S4归约策略的成本身份

`stage2_reduce_backend=auto|scalar|coop32`默认auto。auto在固定点后端及几何下，先重复标量参考，再逐项重复两遍乘减和合并乘减；采用完整引擎加启动算术检查成本最低的合格结果。先完成已开始的重复，再探索链长或D；剩余曲线门槛只限制新比较。强制结果若没有标量配对，不构成auto资格。

实际 `reduce_coop`、`reduce_threads`、`reduce_kernel` 写入 matrix、Shape、预测区域和缓存身份，即使忽略 environment 也分开匹配。标量及实际回退为 `0/0/0`，协作为 `32/T/1`（两遍）或 `32/T/2`（合并）。默认 T=128，诊断 CLI 允许 64／96／128。NTT 表忽略 S4 请求设置。读取器兼容格式 8–14，旧记录的非零 S4 请求缺实际执行证据时标记为 255（未知），不参与选型。

S4配对复用限定实际硬件条件、算术位宽、NTT区间、实际B2、布局、链长、驻留和点后端；不要求B1相同。相邻D在同一区间可复用已合格胜者，未覆盖的NTT拐点重新建立标量参考。正式tune也遵循相同规则。只读时不补采样、不修改文件。

选择coop32时，只有普通实际算术8193…16384 bits启用协作S4，较小位宽及梅森归约仍回退原内核。宽度模型另以`ceil((W+1)/32)`划分协作归约区间，不把W=160／192／224／256处新增高词组的成本平滑穿过边界；工作代理仍按平方乘积计算。没有新策略的实测锚点时只提供保守基础估计，不把标量旧表当作新策略实测。配置、数学及验收范围见[S4报告](../performance/STAGE2_S4_REDUCTION.md)。

## 算术位宽区间预测

普通路径查`generic[bit_length(N)]`，承载路径查`mersenne[p]`。索引按实际算术位宽建立；不要求采样的目标N位宽、具体N或carrier_exponent相同。目标位宽另用于投影、叶乘积和GCD成本；完整梅森数可校准余因子承载。整除验证仍不可省略。

### 无用户调优的基础模型

内置20个普通/梅森冻结参考点：普通318…11694 bits，梅森503…12011 bits，区间之外允许工作量外推至受支持的16384-bit上限。资料来自既有4060 Laptop全位宽矩阵和M12011配对。2400MHz资料的GPU主导阶段粗略乘4/3换算至1800MHz，累乘及固定项保留；不是1800MHz重新测量，也不是其他GPU的实测速度。

算术位宽s、目标位宽t对应w=⌈s/64⌉、wₜ=⌈t/64⌉，P=φ(D)/2、I=⌊B2/D⌋+2。每个候选用当前packing和请求程序计算：

- Qₕ：阶段h的NTT批成本之和。有组件数据时使用组件估算，无资料时用batch·length·log₂(length)相对工作量。
- Rₕ=Σpairs·count·ρ(w)：通用归约ρ(w)=w²，梅森归约ρ(w)=w。
- Xₕ：主机路径的逻辑传输字节；Cₕ：按batch、尾批及物理分块计算的调用数。
- baby代理=P·w²·log₂(D)，giant代理=w²·(6·chain_points+12·ladder_steps)，累乘代理=P·wₜ²。代理不等于精确GPU周期数。

多项式阶段按Q/R/X/C参考比值的非负混合缩放：通用权重`(0.65,0.27,0.04,0.04)`，梅森权重`(0.85,0.05,0.06,0.04)`。互斥阶段分别缩放后加固定项；giant/G树/fold嵌套计时只作为giant-loop的权重，不重复相加。额外主机回退流量按6GiB/s估计补成本。

上下参考位宽各自重算目标几何，再在log₂位宽中对log时间插值；区间外使用邻近参考及工作量外推。NTT长度、树填充和驻留状态重新计算，不能通过位宽插值抹平跳档。

入口为`src/ecm/stage2/planning/width_prediction.h::WidthPredictor`，冻结资料在`ecm_stage2_width_reference.h`；由`tools/bench/generate_stage2_width_reference.py`核对原日志SHA后再生成。生成物保留来源二进制和证据摘要，不需要部署原始实验目录。

### 调优校正与保护

TOML保留测量的target_bits/arithmetic_bits/carrier_exponent和最终摘要。读取后按算术家族、NTT/树级数、fold/frontier、giant路线、物理缓冲及短NTT策略形成索引，再按arithmetic_bits取上下锚点。

校正系数为`实测总秒数/基础模型秒数`；上下位宽的系数作log插值，再乘当前候选基础预测。只有单侧资料时外推。没有该执行区间的资料时保留基础估算，但不跨未覆盖区间声明收益。精确形状直接使用实测均值/标准差。

误差余量包括位宽距离、几何距离、锚点分歧和硬件变化。基础从25%起，区间资料从15%起，外推增加，上限80%。这是**工程余量，不是统计置信区间或独立校准误差**；旧误差证据不改写。

先在每个家族内选择D，再比较普通/承载。预测候选上界`T·(1+e)+2s`须小于`0.98·max(0,T_base·(1−e_base)−2s_base)`，且执行区间已覆盖；精确配对仍使用2%+2s改善条件。承载更快不自动允许任意改变承载D，D收益不足时保留该家族默认D。原生当前free及预算准入保留，实际路线改变后重新预测。

`stage2_prediction_policy=[model|paired]`默认model，CLI为`--prediction-policy`。缺失tune文件且`--short-calibration 0`时也能查询基础预测；默认首次启动短校准仍保留。相邻算术位宽≤25%的已校准家族可用于启动覆盖判断，切换相邻梅森指数不再必然触发短校准。未测矩阵不保证每格不退化。

当前6项有文件/无文件的原生只规划查询执行曲线0：冻结文件仅补齐M12011当前B2配对，M12037已可外推并选择p12037/D240240；普通预计42.900秒、承载23.654秒。缺用户tune时基础预计42.843/23.709秒；M3877保留普通路径。这不是新GPU计时或独立精度验收。证据为`data/experiments/stage2_width_prediction_20261013/acceptance/`，开发边界见[草稿](../development/STAGE2_WIDTH_PREDICTION_DRAFT.md)。

## 收益与搜索

P=φ(D)/2，I=⌊B2/D⌋+2，G=⌈I/P⌉。搜索目标为：

**收益 = Pr(实际 B1、B2，目标素因子位数)/(T1 + R·T2_guard)**。

- T1 是已摊销的 Stage1 秒/曲线，不是 save 读取或 tune 点生成时间。
- R 对应正的 `stage2_ratio_adjust`。
- T2_guard = 预测引擎时间·(1+相对误差余量)+2·s；目标边界为引擎init+main，Stage1、候选shape规划及父子进程启动另计。
- 默认目标因子位数从 B1 推荐式反推并受实际 N 约束；`stage2_target_factor_bits`可覆盖。因子规模不能直接用余因子总位宽代替。
- 原生 Dickman/半光滑概率与 `tools/ecm_prob` 对拍；当前 poly 不计入未实现的 Brent–Suyama 收益。

默认 B1<B2≤2.6×10¹²；`stage2_auto_min_b2`、`stage2_auto_max_b2`及对应 CLI 可修改。B2候选包含区间内49个对数间距点及已记录B2；自动D候选统一来自共享硬编码表，并受下文的实测/拟合资格约束。搜索不受tune网格边界限制，但有限候选搜索不证明连续范围全局最优。

先按收益或固定 B2 成本排序，再逐项要求原生计划 `valid=true`、`finished=true`、`initial_free_snapshot_fits=true`。规划后 fold/frontier 路径变化时不借用原路径排名。实际执行仍检查 free/headroom及处理物理分配失败。无可用成本时固定 B2 返回原因并保留原生选型；Auto B2 明确失败，队列保持未完成。

### Stage1 成本

优先级：正的 `stage1_seconds_per_curve` → `stage1_cost_csv` → `stage1_tune_profile`。秒/曲线不再除以 `stage1_batch`。

CSV 必填 `b1,mhz,tpi,curves,seconds_per_curve`，另提供 `container_bits` 或 `target_bits`。可选 `tpb`（默认128）、`exponent`（空值表示未指定，否则 `lcm|choose12`）。空耗时表示缺失测量，同 TPI 不接受混杂提交批量/TPB。参考表为 [stage1_cost_4060_2000mhz.csv](../../config/stage1_cost_4060_2000mhz.csv)，不自动启用。

用户表 N 明确是容器位宽。实际目标按 **target_bits+6** 选择 PARAM0 容器及 TPI；8192-bit容器覆盖实际位数≤8186，实际8192 bits进入9216-bit/TPI32。表的条件是4060 Laptop、2000 MHz、B1=110×10⁶，TPI8提交768曲线、TPI16提交384曲线，统一提交2 blocks/SM；不等于全部容器同时驻留2 blocks/SM。算法、指数模式及独立核验未给出，来源是用户报告。

初始近似 T1∝B1·container_bits²/f。归一化系数在同 TPI 内插值；范围外邻近锚点外推，无同 TPI 数据时允许跨档外推并标注。`stage1_cost_mhz=0`采用锚点参考频率，正值使用指定目标频率，不读取瞬时时钟。跨 TPI、高 B1 与不同功耗尚未独立校准。

完整 Stage1 TOML采用邻近容器锚点，线性缩放 B1、平方缩放容器；保留 batch、模数类型、指数模式匹配，不要求 B1 相等。预计算工具见 [Stage1成本工具](../../tools/bench/README_STAGE2_CARRIER_PLAN.md)。

## 完整 Stage2 tune

`--tune ecm --tune-level 1..10 --tune-file FILE.toml`默认优先使用当前worker的有效worktodo/save任务；`--tune-worktodo FILE`可指定队列，`--tune-exponents bits,...`明确选择通用位宽网格，`--tune-save FILE`明确选择一个保存点。没有可用默认队列或队列无Stage2任务时，使用通用网格；显式指定的队列缺失、或存在Stage2任务却没有有效save时明确失败。对共享表作模型排序，按NTT执行区间选择少量起点，在区间内比较邻项，预算内允许跨区间重启。缺失NTT组件时补测少数NTT。

默认 `stage2_tune_budget_seconds=1800`秒；初次设备查询在预算外，点准备、正式测量及显式预热在预算内。`stage2_tune_warmup=0`为默认，CLI `--tune-warmup 0|1`覆盖；0直接计入第一轮正式测量，1在每个新测形状前增加一轮经过核验、但不计入均值/标准差的预热。完整ECM、NTT及启动短校准采用同一开关，已有合格测量仍可复用。软预算到达后不发起新测量，正在执行的测量结束并保留有效结果。冷启动样本不会自动剔除；需要预热口径时显式开启。

### 真实任务的普通／承载配对

任务输入按已知因子精确重建N，读取skip后的首个有效save记录，核对N、checksum及可用X，将这一条原始记录冻结到实验目录；不推进worktodo，不写生产完成回执。B2采用显式`--tune-b2`列表，否则采用任务非零B2，再采用TOML的`stage2_b2`；均为0时使用等级网格，只保留B2>保存B1的档位。重复任务合并，同实际位宽但不同模数或承载指数的输入仍分别保留。

`stage2_auto_carrier=1`允许标准梅森任务在验证N整除2ᵖ−1后调优普通/承载两条路径；0只调优普通路径，显式`--tune-carrier-exponent p`配合`--tune-save`仍可指定承载。独立save可从当前worker中重建N完全相同的有效worktodo任务取得p，不从文件名或余因子位宽猜测指数。完整Mₚ的普通路径已经使用梅森归约，自动调优不重复运行相同模数的承载；显式承载仍保留。

预算先用于各实际目标/B2的普通默认D与承载默认D完整参考，再用于两条路径各自的NTT区间和D邻域。不同路径的默认D可以不同；目标是比较完整Stage2时间。同D算法对照可用显式D列表。显式列表不暗中添加默认D；省略普通默认基线时没有精确配对替代资格，预测策略仍采用区间与误差保护。预算不足保留真实正式重复数，不把未完成配对记为已验证收益。输入和配对排序入口为`src/ecm/stage2/planning/planning.cpp::portable_tune_tasks/portable_task_carrier/run_portable_tune`。

### TOML 中的算术路径

格式12每个实际目标位宽只有一个 `[width.<bits>]`，内容为列对齐的 `matrix`。不同条件、D、B1/B2、普通/承载及giant执行参数保留为不同的行，不因可选字段不同再分表。表外仅有格式/计时单位、模型误差设置和确定的设备、环境条件。矩阵中可选值缺失用空字符串表示。

- `condition` 指向 `[condition.<id>]`，保存设备、驱动/工具链、后端及实际环境设置。程序版本不参与条件匹配、统计分组、采样上限或选型资格；软件升级不自动使可兼容缓存失效。
- `seed_coop`是实际seed／[D]Q算术，取0／32／255（未知）。它与`baby_coop`、`giant_coop`共同参与ECM统计、模型和缓存身份，不受environment忽略设置影响。旧条件的`env_giant_seed_coop`只用于兼容加载缺字段的记录：0或缺失可确定为标量，非零请求保留未知；不能推断成已执行协作。NTT吞吐表与seed后端独立。
- `program_version`、`updated_at` 位于ECM和NTT的matrix行内，仅记录来源。实测行保存最近一次有可靠时间且已纳入统计的观察/测量批次；模型行保存生成该估算的版本和时间。时间采用UTC `YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ`。旧数据缺少可靠采样时间时留空，不能用读取、导入或格式化时间代替。去重忽略的观察不刷新来源；双方时间均未知时保留已有版本。合并行不展开逐次版本历史，原始来源保留在JSONL或冻结证据。
- 位宽由表名确定；`carrier_exponent=0` 表示按save的N运算，非零p表示2ᵖ−1承载。算术位宽自动由表名和承载指数恢复。`modulus_kind` 区分generic/mersenne；完整梅森N允许carrier_exponent=0。
- `d` 为D，P=φ(D)/2自动恢复；fold/frontier列自动恢复驻留路径名称。giant分块、chain门槛和ladder开关影响执行形状，保留在matrix中。
- `source`、`benchmark_kind` 区分模型、正式调优和运行采集。完全相同参数的独立测量合并为一行：`repeats` 是次数，`mean_seconds` 是算术平均耗时，`stddev_seconds` 是样本标准差。正式ECM及NTT也按真实次数合并，较慢观察同样计入。
- `verified=1` 仅在运行检查通过后写入。逐次selftest/checked/bad/hits计数、预热次数、init/main重复计时、预测值/比值及模型过程保留在内部JSONL，不重复保存到性能矩阵。用于组件缩放的互斥阶段及giant/tree/fold嵌套组件保存均值，不相加冒充总耗时。
- ECM 矩阵在 `d` 后依次写入 `mean_seconds`、`stddev_seconds`、`repeats` 和 `required_free_bytes`。显存字段以字节记录单曲线原生工作区模型要求，同形状保留已知最大值。旧行在无可用显存数据时保持空值。`preferred_order` 保存已成功采用路线的顺序。
- 合并同形状的已有汇总时，优先保留样本数较多的记录，样本数相同时采用时间较新的记录，避免重复累加可能重叠的汇总。离线导入原始 JSONL 时按队列回执去重，运行时预读仅纳入晚于同形状磁盘汇总的最新观测。

完整ECM的均值使用 `stage2_full_wall.total=init+main`，包括初始化、必需自检、GCD及算术收尾，不含Stage1、选型、进程启动、因子分解和TOML写回。每次执行前的模型预测及误差仍记录在JSONL；缓存仅保留用于成本/误差保护的摘要。

NTT表只测Goldilocks有限域卷积，不含大整数carry和模N归约，不按梅森承载重复一份。实现入口为 `ecm_stage2_tune_portable.h::compact_sample/hydrate/Profile::update/serialize`。

列优先级只在 [`src/ecm/stage2/planning/tune_columns.def`](../../src/ecm/stage2/planning/tune_columns.def) 定义：ECM使用 `ECM_STAGE2_TUNE_COLUMN`，NTT使用 `ECM_STAGE2_TUNE_NTT_COLUMN`，移动对应字段行即可调整该表顺序；未列出的可选字段按字母顺序排在后面。生产程序在编译时展开该定义，Python导入器通过 `tools/bench/stage2_tune_matrix.py::matrix_column_order` 读取同一文件，保留现有列宽对齐。调整顺序需要重编译生产程序；读取仍按列名解释，不依赖文件中的列位置。

行排序只在 [`src/ecm/stage2/planning/tune_rows.def`](../../src/ecm/stage2/planning/tune_rows.def) 定义，ECM优先级为 `condition → b2 → d → carrier_exponent → b1 → source`。数值键按数值、source按字符串升序，所有键相同则保留原有先后。NTT使用 `length → batch → tile`，tile缺失按0处理。生产程序的 `matrix_row_less`、Python导入器及NTT格式化工具的 `matrix_row_key` 消费同一定义；调整对应字段行即可改变优先级，生产程序须重编译。列位置、耗时、版本和更新时间不参与行排序。

### 正常运行累计采样与候选试跑

`stage2_tune_runtime=true` 默认开启；CLI 为 `--tune-runtime 0|1`。`stage2_tune_runtime_samples=10` 指定每组独立 NF 样本上限，范围1–1000，CLI 为 `--tune-runtime-samples N`。候选先取得R次观察，胜出路线随继续执行累计至上限。达到上限后停止该组采集，已有证据保留；调低上限不删除已有样本。关闭采集仍可读取已累计数据选型，但不再安排新试跑或写回。

同组要求相同设备、驱动/工具链/后端与环境、目标/算术位宽、承载指数、实际B2/D、packing与物理分块、giant路线及fold/frontier执行形状。B1、具体N和sigma不作为Stage2成本复用门槛；仍须分别验证当前目标、save和承载整除关系。不同程序版本的可兼容统计共同参与复用和累计；环境的可忽略维度仍受显式tune-ignore及模型误差保护约束。

固定B2、`model`策略下，用待处理曲线建立完整Stage2对照，不额外重算同一生产曲线：先试合法梅森承载的默认D，再建立普通默认D基线。两侧基线取得R次合格NF后，分别按当前算术位宽、NTT/树/驻留和giant路线生成有界D搜索。每条曲线开始时选择一次D，曲线中途不改变几何。因子命中或失败不计入成本观察。

共享搜索器以实测均值最快的重复观察点为中心。最快点位于已测P范围边缘时，优先沿改善方向延伸；步幅参照该侧实测最快的对照，避免被较慢的近邻回填点缩小。D也处于已测范围边缘时，延伸须同时保持D方向，因为D与P并非单调对应。前进允许跨NTT/树/驻留区间；出现另一侧实测边界后，转入双向细化。其他情况先按模型排名选择约25%、12%、6%、3%尺度的代表，不固定先试小P，也不逐项遍历密集相邻D。预算内仍可跳到其他执行区间或同区间远处，不假设耗时单峰。

每个算术家族先允许4个非默认候选；有路线积累到min(8,采样上限)次观察后，开放后续细化，总共最多8个。D与布局共同标识候选，名额和120秒额外引擎成本预算跨布局共享，不为新布局再开放一份额度。常规重启最多覆盖3个预测执行区间；实测改善边缘可额外进入1个区间，避免已测较慢区间封死前进方向。额外成本按非默认候选相对本家族最低合格均值的正差乘样本数估算，用模型上沿估算尚需观察的代价；明显超过当前参考均值1.5倍的预测项不进入本轮试跑。这些是工程调度门限，不是最优性证明或严格墙钟上限。已知队列剩余不足2R条时不扩展新的D搜索。预算不足或模型不具竞争力的单次观察保留，但未达到R次不得按运行均值采用。

普通与承载各自至少有2次严格匹配的运行观察时，取各自最快合格均值μ及其样本标准差σ。若`μ承载+2σ承载 ≤ 0.8×(μ普通−2σ普通)`，普通家族新候选额度收紧到2个，每次布局搜索的原生准入上限由16降到4；普通历史最快项、保护基线及已开始的重复仍保留。已有普通候选超过额度时不删除，只暂停新增。该式是保守工程筛选，不是统计置信区间。每次选型重算差距，不足20%时恢复常规探索；最终采用仍只按实测均值最小，不因探索配额拒绝更快的普通路线。

成本模型接受原生打包查询支持的NTT长度，再通过当前显存规划准入；不把NTT单独测量网格的`2^27`上限当作生产形状上限。未测长变换使用明确标注的外推成本，不视为实测，也不保证能驻留或更快。

运行时D探测另有`same_width_phase_projection_v1`先验：在相同目标/算术位宽、实际后端、链长、有效尾块策略、布局、驻留及NTT/树/巨步路线区间内，优先使用几何最接近的完整阶段观测。P和巨步点数分别限定为观测值的½…2倍；设备、驱动、CUDA、后端、环境条件严格匹配，不能用`tune_ignore`放宽此类阶段投影。条件不满足时，保留原区间模型。它仅用于正常运行的有界试跑排序和预算，关闭运行采集后的模型选型、冷启动默认D及Auto B2收益模型不启用该投影。

阶段投影要求十项互斥阶段均值非负且总和等于引擎均值；不从嵌套的giant/G树/fold计时反推出总阶段。固定准备与收尾保持固定，Baby和累乘分别按自己的工作量缩放；多项式阶段的工作比率为`0.55×NTT + 0.40×归约 + 0.03×传输 + 0.02×调用数`，各项先除以锚点对应工作量。主循环用嵌套giant/G树/fold占比加权这些工作比率，乘回完整互斥主循环阶段。这些权重是显式先验，不能视为通过拟合辨认出的设备吞吐率。实现复用[portable_model.h](../../src/ecm/stage2/planning/portable_model.h)的`Estimator::scaled`，而非维护第二套阶段计时边界。

预测误差余量为`15% + 4%×g`，其中`g=|log₂(P/P₀)|+¼|log₂(I/I₀)|`，I为巨步点数。它不是统计置信区间或独立验证误差。该先验保持`exact=false`、`validated=false`，不增加repeats或生成测量行；达到R次的精确运行观察和正式测量仍优先。稳定采用依旧经过完整曲线、自检、NF和当前内存准入。实现入口为[width_prediction.h](../../src/ecm/stage2/planning/width_prediction.h)的`phase_anchor/calculate`；定向资格见[大档B2性能](../performance/STAGE2_LARGE_B2.md)。

候选先做廉价排名，只有少量待试项进入当前原生显存/执行计划准入；实际分块和驻留路线改变时重新核对成本与观察身份。显式D锁、关闭运行采集及`paired`策略不展开新D搜索。统计充分且达到候选上限时不再每曲线重排全表；搜索不保存逐曲线调优历史，TOML继续按参数合并均值/标准差/次数。日志`tune_selection.runtime_search`记录各家族与布局的额度、已探索数量、额外成本估算、提议D及原因，`runtime_allocation`给出承载保守优势及收紧状态；不新增逐候选控制台刷屏。

显式D锁限制候选D；显式`--carrier-exponent 0|p`锁定算术。关闭`stage2_auto_carrier`时不自动加入承载。独立save不从文件名或位数猜指数；合法worktodo提供并验证p。`paired`策略仍使用正式配对调优规则。Auto B2先选择覆盖上界，再允许在同一实际B2下进行上述正常运行比较。

### 实测统计与路径保留

有限队列预算另有离线回放工具。两种策略从同一已测D开始，按同族历史试跑顺序最多尝试四个备选，第一次备选均允许。实验策略对其余候选比较两项估计，所需R次的额外代价为 `R × max(0,预测上沿−当前均值)`，后续曲线的可能收益为 `(剩余条数−R) × max(0,当前均值−预测下沿)`。当额外代价较大时停止。上、下沿取测前预测的1.25和0.75倍，这里是实验参数。两种策略均保留2R剩余门槛。回放采用前R条引擎时间选胜者，试跑按实际父进程曲线墙钟计价，其余曲线按该D的记录均值估算。它独立研究停止时机，真实队列还包含跨族选型、显存准入、120秒预算以及后续观测引起的重新排名。[回放工具](../../tools/bench/replay_stage2_queue_budget.py)

0.39.3 历史矩阵的47个可比家族均有至少三个D取得两条合格观测。剩余2或5条时两种策略给出相同选择。剩余20条时，14个家族的探测序列缩短，合计估算少228.67s（0.865%），两个家族错过更快的后续D。7995-bit 承载的估算墙钟由258.14增至277.15s，318-bit 普通由19.53增至23.32s。0.57.17 的两格大B2承载回放使用R=1，剩余5条时12285-bit估算少12.89s，16272-bit选择相同。两格的剩余2、20条结果相同。当前保留生产停止规则，继续验证短队列和错过后续胜者的风险。来源为 `data/experiments/stage2_runtime_v0_39_3_20261019/adaptive_01/plan.json`、`stage2_phase_model_20261103/`，汇总为 `data/benchmarks/stage2_s4_capacity_20261103/{queue_replay,recent_replay}.json`。

同参数的n次观察只保存一行和三个充分统计：次数n、均值μ、样本标准差s。新增m次独立观察时，按加权均值及组间均值差合并方差：

μ = (nμ₁ + mμ₂)/(n+m)

M₂ = (n−1)s₁² + (m−1)s₂² + (μ₂−μ₁)²·nm/(n+m)，s = √[M₂/(n+m−1)]。

单次观察的s记0，达到R次才建立 `runtime_shape_mean_v2` 实测估计。R=1允许快速筛查，但s=0仅表示无法估计离散程度，不能视为稳定性证明；算术自检、NF、完整时钟和实际执行路径检查均不减少。R不能超过采样上限。提高R不删除已有观察，后续任务补足后才恢复实测采用。ECM正式校准重复数至少为R，显式--tune-repeats可要求更多；独立NTT重复次数仍由其采样参数控制。通过当前设备、环境、完整形状及显存准入的合格重复观察，直接选择均值μ最小者；不再要求2%收益或用标准差否决更快均值。s保留用于诊断及未测候选的成本预算。未达到当前门槛或跨条件的模型输入继续受预测保护。已有正式采样若不足R次，也不再作为精确实测项，但可以保留为带误差的模型先验。

所有合格运行候选一起按均值排名，不因候选排列、原默认D、既有路线的样本数更多或更新时间更新而隐藏最快项。均值严格相同时优先最近成功采用的路线，再以完整形状打破同值。关闭运行采集仍按同一规则读取已有数据；开启采集时，预算内的未测候选试跑可临时优先于稳定路线。

存储身份包含B1，避免把不同完整参数写成同一行；成本复用不以B1为门槛，会合并匹配形状的统计；采集上限按各完整参数行的repeats独立计算。具体N和sigma不作为成本身份，但每条曲线仍独立验证。

承载首个试跑不等于必须采用。日志 `runtime_candidates` 输出样本次数、平均时间及标准差；稳定采用原因是`runtime_measured_fastest`。`runtime_search`另列中心D/P、已测区间数、更大D/P的模型代表，以及模型、预算和显存拒绝次数。`last_memory_rejection`记录最后一次显存拒绝的D、生命周期所需free和现场free；不能用终止原因`neighborhood_complete`单独推断全部更大D都已实测。正式调优和运行观察分开累计，不混为一个低波动总体。

运行上限按观察次数统计，不同参数行保留。正式调优批次按真实次数累积。需要汇总多个文件中的独立观察时，先一起导入原始 JSONL，再合并生成的摘要。

### 准备结果的复用

运行队列只在当前父进程内传递准备提示；重启后从空状态开始，不创建缓存文件，也不改变只记录曲线进度的checkpoint。

- **模型暂停**：每个算术家族最多一项。只有本次没有提议D、没有进入原生准入且搜索期间显存快照稳定时才能复用；完整性能文件、任务N/B1/B2、条件、实际候选及统计、完整规划环境和实时free/total组成键。新观察、执行路径或显存变化即重新排名。不能因一轮达到16次准入上限而暂停后续探索。
- **成功几何计划**：最多32项。只记D、算术家族及fold/frontier、giant分块、chain阈值等资源路线。任务N/B1/B2、程序版本、设备身份、预算、完整环境和相关选项必须相同，每次复用前重新查询设备free/total并要求精确相等。保存前后内存快照不稳定、原生规划失败、提示损坏或条件改变时回到原生规划。跨曲线不保存点、F/G树或任何算术状态；sigma不影响无曲线运算的几何规划，实际save验证和曲线执行仍分别进行。
- **worker内计划**：以完整请求Shape、sigma及规划条件为键，复用同曲线选择与观察准备的重复成功准入；同样检查实时free/total，失败结果不缓存。
- **显存失败提示**：仅接受原生计划有效、完成且明确因`required_free_bytes > free_bytes`而拒绝的候选；准入前、计划报告及准入后的free必须一致，total也必须一致。不缓存不支持、未完成或其他错误。提示以共享D表索引的位图保存，最多两个算术家族，不保存完整失败计划；free/total、规划作用域或父进程改变即失效。命中的失败项不消耗16次原生准入额度，搜索继续向后推进，仍受候选数、区域及额外引擎时间预算约束。

单次选型的估计器还复用NTT长度/batch成本、完整Shape的运行统计和位宽预测，保存原始双精度结果，不量化排名。严格条件匹配的运行样本在估计器构造时筛选；profile、条件或NTT策略改变后新建估计器。此类数学缓存没有设备分配和算术状态。

两条路径的默认D仍实时规划；实际引擎初始化、分配失败处理、承载整除验证及数学检查继续执行。提示附在现有`tune_selection`事件中，父进程仅在内存保留，带长度检查和SHA256校验；校验用于发现损坏，不作为权限机制。完整几何计划不写入tune矩阵。内部A/B开关`--runtime-prepare-cache 0|1`默认1，可一次关闭上述复用；它不改变测量身份或采用门限。

`tune_selection.runtime_prepare`给出默认规划、候选准备、搜索及总时间，另列原生调用数和缓存命中数。`runtime_search.native_admission_calls`是含成功计划缓存命中的逻辑准入请求数，不含位图跳过；`memory_rejection_cache_hits`单列位图命中，`runtime_prepare.native_plan_calls`才是真正调用原生规划器的次数。`ntt_cost_evaluations`与`ntt_cost_cache_hits`计估计器内NTT成本求值及复用，`prediction_evaluations`与`prediction_cache_hits`计位宽预测求值及复用。

`native_plan_seconds`嵌套在准备阶段内，不能重复相加；`selection_seconds`还包括外层配置准备，`observation_prepare_seconds`单列观察准备。这些均在`stage2_full_wall`外；性能需同时比较引擎与队列墙钟。读取、save解析/校验后即打印规划阶段，自动选型和观察准备归入规划；相同阶段的连续标记合并，不重复切断计时。

缓存实现为[runtime_cache.h](../../src/ecm/stage2/planning/runtime_cache.h)、[portable_model.h](../../src/ecm/stage2/planning/portable_model.h)和[width_prediction.h](../../src/ecm/stage2/planning/width_prediction.h)，接入点为[planning.cpp](../../src/ecm/stage2/planning/planning.cpp)的`portable_runtime_admit/portable_runtime_search`和[task.cpp](../../src/ecm/stage2/task.cpp)的`child_run/curve_worker`。验证范围见[准备开销](../performance/STAGE2_PREPARATION.md)与[Stage2性能](../performance/STAGE2.md#有界d搜索的验证范围)。

### 验证与持久化

只采集P≥512且具有完整多批请求拓扑的曲线；G=1及D210启动探针不作为完整区间锚点。结果必须无因子、clean=1、自检和抽查量为正、bad=0、互斥阶段计时完整，且实际D/P/giant数量及fold/frontier状态与原生准入计划相符。规划、失败、中断、save中提前发现因子、调优子进程、因子命中和改变规划路径的回退不加入NF样本。

计时保持`stage2_full_wall.total=init+main`，不包含Stage1、进程启动、选型和发布；每个观察保存采集前预测及完整数学检查。曲线结果先持久保存，再取得 tune 独占锁、重新读取文件、检查上限并原子写回。锁占用或写回错误只提示，不改变已完成曲线或队列计数。

生产 JSONL 的 `tune_observation` 可由[导入工具](../../tools/bench/import_stage2_benchmark_tune.py)的 `--runtime-results FILE --runtime-samples 10` 恢复，再经原生 `--tune-merge` 汇集。重复导入同一观察不增加计数，导入保留上限内的独立较慢样本。原始曲线、日志和失败证据放在忽略目录，TOML 保存合并统计、显存要求及必要组件均值。

实现入口为[planning.cpp](../../src/ecm/stage2/planning/planning.cpp)的`portable_runtime_select/prepare_runtime_tune/complete_runtime_tune/publish_runtime_tune`，统计在[portable_model.h](../../src/ecm/stage2/planning/portable_model.h)的`runtime_statistics`，切换规则在[runtime_tune.h](../../src/ecm/stage2/planning/runtime_tune.h)的`runtime_choice`。CPU门禁、GPU生产流程及离线恢复脚本为`tools/test/test_stage2_runtime_tune.py`、`test_stage2_runtime_selection_gpu.py`、`test_stage2_runtime_task.py`和`test_stage2_runtime_import.py`；各脚本要求新的`--output data/experiments/<目录>`。

固定配置的独立基准扫描应设置`stage2_tune_runtime=false`并冻结性能文件。此时仍可使用该文件已有的合格实测组；若需要完全固定D/算术，应同时显式锁定它们。检验正常运行时选型的矩阵则保留该开关，单独记录学习和候选试跑；稳定性能须核对无新试跑、文件未变化，并对相同D/P/算术重复测量。若直接报告性能文件的最佳合格均值，则须注明实际样本数和波动，明确这些观察也参与了选型，不冒充独立的选型后验证。两种口径不混合，复现方法见[运行时矩阵](../../tools/bench/README_STAGE2_VERSION.md#正常运行时选型矩阵)。

### 当前验证范围

完整GPU流程基线为0.37.2：在GPU1 RTX4060 Laptop上验证15次队列NF，使用M1009的383位余因子save，B1=50000、B2=10⁸、D=30030、batch=64 MiB、arena=6300 MiB、owner=640 MiB。12次独立执行观察合并为普通10次和承载2次共两行；另核对显式D/算术锁、关闭写入仍读取统计、Montgomery Auto B2计划和因子命中不入样本。采用相同验证保存点重复运行，属于功能门禁，不是新的独立ECM覆盖或性能基准；计时边界为引擎init+main。

CPU检查覆盖独立两遍方差参考、加权合并、较慢观察、10次上限、跨版本复用与设备隔离、快照选用、显存要求保留、不同B1记录、NTT参数行与布尔列、对齐矩阵稳定读写、20条不同参数、10级网格、非负拟合、105组邻域及坏文件拒绝。共享列序检查核对原生/Python表头一致、旧条件迁移、最近来源按时间选择、未知时间留空及重复导入不刷新时间。正式GPU门禁通过22项：驻留/host fold调优、512/4423位输入、跨B1复用、短校准、Auto B2、NTT、合并、输入保护、旧格式重建、D邻域、实际执行及承载规划。4条NTT摘要通过排序、数值保持和格式化幂等检查。

真实0.37.1 format 8文件在0.37.2只读规划时不重写或重新校准；续算一次NF后，原普通10次统计累计为11次，均值/方差按独立参考核对，未运行的承载行仍保留0.37.1和未知时间。恢复工具把两版24次JSONL观察合并为同一条件、普通20次和承载4次两行；重复导入/合并不增加计数，较慢观察保留，坏检查回执拒绝。

基线证据位于 `data/experiments/stage2_tune_provenance_20261018/`：`cpu_final01/`、`portable_final01/`、`provenance_final01/`、`reuse_final01/`、`gpu_final02/`及其审计、`portable_gpu_final02/`、`import_legacy01/`、`import_mixed_final01/`、`sort_final02/`；身份汇总见`final_acceptance.json`，前置运行目录保留。15条队列NF的GMP抽查18836，错误0。该基线构建为 `build_cuda_cmake/local_stage2_0_37_2/ecm_cuda_stage2.exe`，sm89、CUDA13.3.73、MSVC14.51.36231、PTX3/addsub1、GMP Zen3；CUDA对象与0.37.1一致。冻结85项编译输入，EXE SHA256为`5714f405d4bde1853666e94c5add94bdf31cbb01e6792e0ba3e3f7f8d81a9243`。

0.37.3的共享行排序在CPU验证9行ECM逐键优先级、同键稳定性、原生/Python输出及统计/来源一致；4条NTT行的数值顺序与格式化幂等通过。24次既有观察的离线导入、去重和统计检查通过；不重新测量GPU曲线。证据在 `data/experiments/stage2_tune_row_order_20261018/` 的 `portable01/`、`runtime01/`、`provenance01/`、`ntt01/`、`import01/` 和 `final_acceptance.json`。该行排序验证构建为 `build_cuda_cmake/local_stage2_0_37_3/ecm_cuda_stage2.exe`，工具链和CUDA对象与基线相同，86项编译输入经哈希核验；EXE SHA256为`eaa4822db03d9b74f6dbc577a079f5e1afc44f41f1042cc45352dc7be78bb81a`。

0.37.4的NTT展示字段在GPU1 RTX4060 Laptop核对L=2¹⁰至2¹⁴、batch=1/4、256 MiB预算，每参数独立测3次和4次，10行共70次正式观察。实际radix/趟数与底层回执一致，合并均值/方差与原始时间独立核对，重复导入不增计数，吞吐率重算且行序不变。旧0.37.2的4条NTT记录补齐展示信息时统计和来源保留，4类坏派生字段拒绝；CPU现有门禁及24次生产观察离线恢复通过。证据在 `data/experiments/stage2_ntt_metadata_20261018/` 的 `gpu01/`、`legacy_ntt01/`、`portable_final01/`、`runtime_final01/`、`provenance_final01/`、`import_final01/` 及 `final_acceptance.json`。正式local为 `build_cuda_cmake/local_stage2_0_37_4/ecm_cuda_stage2.exe`，工具链和CUDA对象与0.37.3相同，86项编译输入经核验；EXE SHA256为`4458a337e554afb663ddadbe5c0d7174e8685b05dadee779caea8e14709b0ddb`。此处仅验证元数据与持久化，不作为新的整曲线性能测量。

频率、功耗未按性能协议冻结，因此不据此给出加速比例。真实并发写锁、强制中断、其他设备和生产规模边界仍见TODO。

### 等级与通用网格

通用网格每级额外包含512 bits。位宽 a:b表示b、2b、…、ab；B2 a:r从上界 U=2.6×10¹²按 r 倍递减取 a 项，再升序排列。非零 `--auto-max-b2`可改变 U。真实队列使用实际位宽和上述B2优先级，等级仍控制正式重复数及区间数量。

显式位宽网格默认`stage2_tune_arithmetic=both`，CLI `--tune-arithmetic [both|generic|mersenne]`：普通使用该位数的可能素数；梅森使用算术指数p等于网格位宽的2ᵖ−1。梅森目标先去除来自真因子指数的公共因子及≤10000的小因子，再生成B1=20的有效点；实际目标位宽可低于p，分别记录。无效点或产生因子的运行保留证据并跳过，不计为完整计时样本。真实worktodo/save不经过此预处理，仍由`stage2_auto_carrier`决定配对。网格的两类资料校正各自算术家族，不要求互相具有同一个N。

- level1：位宽3:4096，最多2个区间，B2 3:10，正式重复2次。
- level2：位宽4:3072，最多2个区间，B2 3:10，正式重复2次。
- level3：位宽5:2048，最多3个区间，B2 4:10，正式重复2次。
- level4：位宽5:2048，最多3个区间，B2 4:10，正式重复2次。
- level5：位宽5:2048，最多3个区间，B2 6:5，正式重复3次。
- level6、7：位宽8:1536，最多4个区间，B2 6:5，正式重复3次。
- level8：位宽10:1024，最多5个区间，B2 8:5，正式重复3次。
- level9：位宽15:1024，最多6个区间，B2 8:5，正式重复5次。
- level10：位宽15:1024，最多6个区间，B2 8:5，正式重复5次。

### 共享 D 候选表

唯一维护位置是 [ecm_stage2_d_catalog.h:14](../../src/ecm/stage2/planning/d_catalog.h#L14)，当前含1094个严格递增、去重、偶数D，范围22～199879680。P=φ(D)/2统一派生并缓存，不手工维护第二份D/P值。例如D1021020=2×510510，P92160；B2是独立输入。

表覆盖Prime95 v31.06有效poly表、2幂扩展、代表位宽的fold NTT/树填充边界、既有默认平台保护项及已记录几何。来源和覆盖条件见 [D候选表草稿](../development/STAGE2_D_CATALOG_DRAFT.md)。候选允许被考虑，不代表已经更快或当前显存一定适配。

CUDA默认成本模型、portable运行/Auto B2、各级tune、旧`EcmEffort`接口及Python规划工具读取同一表。等级控制区间数量和采样精度，不缩小D覆盖域。性能文件中的表外D可保留为成本锚点，不自动扩大候选表；显式非零`--d`及`--tune-d`仍允许研究表外几何并接受完整准入。显式tune列表不暗中加入默认D。

参考全枚举包含379419个非零P的≤200000000、47-smooth D，其中282395个满足偶数D≥6条件；它包含奇数D，只用于审查来源，生产不在线枚举。全文和再生成方式见 [工具说明](../../tools/bench/README_STAGE2_D_CATALOG.md)。

### 来源、筛选与规模

D/φ/P的数学意义、非单调性和完整计算量推导集中在 [D参数数学](../reference/ECM_STAGE2_D_PARAMETERS.md)。本节只记录当前生成域、共享表和资格；Prime95源码计数另见 [固定参考](../reference/PRIME95_STAGE2.md#31-有效表与-numrels)。

**参考域。** `stage2_d_catalog.py:40`逐个选择15个≤47素数的非负指数，枚举乘积≤200000000且φ(D)/2>0的值。共有379419项；282395项满足D≥6且为偶数。精确条件为≤，包含D=200000000/P=40000000；若严格要求<200000000，条数为379418。47和200000000是工程生成域的截断，不是ECM正确性条件；现有资料没有证明47比43或53更优。当前生产不在线枚举这379419项。

**生成域与压缩。** 共享表从离线扩展池筛选，不是379419项的纯子集。扩展池由保守默认项、已核验profile、Prime95有效88项及2幂扩展、代表S上的fold NTT和树填充边界组成。冻结扩展池有2080项；当前1094项是其子集，1090项为47-smooth，另4项为1591590、1771770、3183180、3543540，含53或59。

当前筛选规则由 [压缩工具](../../tools/bench/compress_stage2_d_catalog.py)产生：

- **保护集合：**98项，包括全部14个default_plateau、88个Prime95有效项，以及已核验profile和完整矩阵中的D；各来源有交叉。
- **几何包络：**同P先选最大D，按P升序保留刷新最大D的227项。因此任意P上限内仍能找到原池中最大的D。这是覆盖启发式，不是完整曲线性能支配证明。
- **NTT边界：**52个代表算术位宽S，检查操作数P及P+1在2^14～2^27内的packing边界；每个边界各保留一项下方、上方P代表。代表S包含128、256、实际余因子/承载例子及1024…16384每512 bits的点。
- **树边界：**P=2^k、k=3…23的两侧各保留一个P代表。
- **owner边界：**默认reuse=3的fold精确payload为8·ceil(S/64)·(7P+7)+48 bytes；frontier再增加24P bytes。对48/128/256/640/1024 MiB分别保留两侧P代表。公式依据为 [FoldOwnerLayout](../../src/ecm/stage2/geometry.h#L110)、[frontier元数据](../../src/ecm/stage2/planning/owner_memory.h#L57)。

去重后1094项、1083个不同P；按D升序仍有525处相邻P下降。筛选减少47.40%的候选数，不意味着GPU计算减少相同比例。来源、全部移除项和逐边界代表保存在`data/experiments/stage2_d_compression_20261011/proposal/`；扩展池来源的冻结记录在`data/experiments/stage2_d_catalog_20261011_final/analysis.json`。性能验收与证据见 [Stage2性能](../performance/STAGE2.md)。

**覆盖范围。** 大arena的变换payload在NTT长度变化处跳变；上述NTT和树边界保留相应代表。owner公式单独反映驻留边界，不能与NTT峰简单相加。完整请求、当前free、生命周期和回退路径仍由原生规划准入；有限S和预算集合不能保证任意预算、全部尾树和giant分支的最优覆盖。CPU成本枚举/缓存读取1094项；GPU实测仍受区间与预算限制，不同时为所有候选分配显存。

**两种标记。** 参考文本最后一列`portable_even_D_ge_6`：1表示D≥6且为偶数，0表示不符合此基本候选条件；不表示素数、显存通过或已经更快。共享头文件中的布尔字段是`default_plateau`：true表示允许参与保守平台覆盖策略，false只表示没有此资格。头文件的false不表示奇数，1094项全部为偶数。

仅移植Prime95的88个有效poly候选可以得到合法先验，但其全部D恰含一个2因子，缺少本表的额外2幂几何；Prime95使用CPU FFT、缺失素数移位和B2_start成本，其间隔不能保证匹配本执行器的NTT与显存边界。例如S=5003、fold≤2²⁷的P上界为127099，Prime95上界内最大P为120960/D1291290；本表包含P126720/D1381380。在B2=2.6e12时，两者G分别17和15，后者初始化也更大，不能仅用块数宣布更快。

更贴近NTT边界也不必然改善性能。例如S=7995、fold≤2²⁷的P上界为79606，Prime95的P77760/D810810与本表的P79488/D769860都可容纳；在同一B2=2.6e12下，后者giant点更多、G=43而前者G=42。S改为8011时，769860的fold变为2²⁸，810810仍为2²⁷。这里只比较整数packing形状，不是联合显存或运行时间证明。

缩到几百项可能适用于限定设备/输入范围，但需要保持NTT/树/驻留区间代表、既有默认和合格赢家，并通过完整矩阵及明显回退复测；同NTT长度、同P或候选数量相近不足以证明可以删除。进一步缩到几百项仍待扩大预算和完整矩阵验证，不能把“200项足够”写成已确认规则。

### NTT 区间和预算内重启

每个“位宽×B2×算术路径”先测或复用默认基准；显式D列表未包含默认值时省略此步。廉价筛选排除成本域外、G=1及超过批次数限制的项，再用模型给候选排序。原生生命周期准入只对即将作为起点或接受测量的候选执行；单个起点显存拒绝不否定整个区间。

区间键包含fold、inverse、F树及下降的最大NTT长度、填充树高、fold/frontier驻留状态和giant生成分支。它是粗粒度分组，不表示区间内全部请求或联合显存相同。区间内按P升序、同P按D降序组织邻项。

按各区间最小模型成本排序，最多选上述等级允许的区间数；其他区间须满足预测排名成本≤最佳区间的1.5倍才进入本次重启集合。这个门槛只调度采样，不改变生产候选的算术/显存资格。已有配对实测赢家优先成为本区间起点，否则选择通过原生准入的模型起点。

自动候选表在每个区间使用与正常运行共用的比例跨步/细化搜索器，最多探测max(4,2×等级区间额度)个非起点D；区间之间仍按上述额度重启，不逐项走过更慢的过渡D。显式`--tune-d`研究列表保留中心与紧邻项的原有遍历。正式调优及`paired`策略中，候选A替代B仍要求 **T_A+2·s_A < 0.98·T_B−2·s_B**；正常运行合格重复观察使用上文的最小均值规则。模型不足时使用归一化完整阶段工作量作排序先验，不视为秒数实测或生产获胜证据。

完成正式重复要求的同形状测量可复用；预算结束、已有时间预计超过剩余预算或重复数不足时标为`deferred`，不当作无效项关闭方向。`rejected`表示数学/当前原生准入拒绝；`complete=1`仅表示本次选择的所有区间邻域已完成且没有延期，不表示全表或全局最优。完成的正式样本及真实重复数保留，下一次tune重新模型排序并复用它们。正式新测时间用于本次邻域比较，即使文件按均值累计规则仍保留了更快的旧部分样本；不借本次重复数提升旧样本资格。没有实测的高B2形状不根据P=24的远距离外推时间直接跳过。

纯packing形状按操作数/算术位宽缓存；工作量按完整Shape缓存，拟合及资格按模型域缓存。缓存只在相应条件和不变profile内复用，更新测量后重新创建估算器；现场free准入不由这些数学缓存代替。本轮已完成的配对参考按目标对象及完整Shape复用，防止磁盘上的更快部分摘要造成重复测量；当前邻域使用本轮正式结果，不提升旧部分摘要的重复数。实现见 [区间键](../../src/ecm/stage2/planning/portable_model.h#L74)、[正式调优调用](../../src/ecm/stage2/planning/planning.cpp)及[共享搜索器](../../src/ecm/stage2/planning/d_search.h#L27)。

`--tune-exponents bits,...`在通用家族表示**实际位数**，在梅森网格表示算术指数p；`--tune-d`、`--tune-b2`、`--tune-repeats`覆盖对应网格。通用家族构造精确位宽的GMP probable prime，sigma26、B1=20，不称为已证明素数。`--tune-save FILE`使用验证后的真实点并替代位宽列表；自动承载要求匹配真实任务，显式`--tune-carrier-exponent p`仍检查整除。旧 `--tune-tail-samples`只接受0。

完整测量默认直接正式采样，要求mandatory自检、GMP检查、clean通过、bad=0、hits=0。显式开启的预热也接受相同检查，但不计入正式时间。预算中途结束保留已完成正式重复及真实次数；仅完成预热时不生成摘要。再次tune复用已有候选并继续未完成邻域。TOML保留访问候选的最终测量/估算，不展开全表估算。

控制台保留 `ecm_tune_window`、`ecm_tune_progress`、`ecm_tune_window_done` 和完成/失败原因。子进程阶段、NTT统计及内部自检不转发到控制台，完整诊断仍保存在每次测量的证据日志；正常Stage2的控制台与日志规则不受此限制。

### 默认选型与实测保护

默认模型在最大NTT长度低于2²⁷时使用共享表的原生D成本排名；达到当前portable成本域2²⁷边界后，只比较表内标记`default_plateau`、仍驻留、最大NTT长度不增加的更大D。这14个标记只保留既有保守平台保护的资格范围；新增候选不自动获得该覆盖资格。形状查询能够返回更长NTT，这一策略边界不代表所有设备或底层NTT的绝对长度上限（`src/ecm/stage2/planning/portable_model.h:49`）。通用归约还要求默认巨步数至少16P，以摊销较贵的初始化；梅森归约不采用此附加门槛。归一化成本包含G树、fold、giant及初始化，避免在同一大NTT平台内过早减小P；显存准入仍使用完整生命周期。该几何模型不是绝对秒数预测；性能依据和范围见[Stage2性能](../performance/STAGE2.md)。

paired策略的固定B2替代需要默认D与候选D具有同实际位宽、同B2、同算术/执行路径的完整实测，并满足均值/标准差改善条件；model策略采用算术位宽区间预测与双侧保护。B1不作为匹配门槛。paired缺少基准时保留默认D；model的估算不能覆盖未支持的执行区间或省略误差保护。未锁定承载时，承载候选还须与普通默认路径比较。实际输入、当前显存和整除关系仍须核对。

固定B2先按实测摘要过滤没有显著优势的D，再对剩余候选作完整原生规划及严格执行路径匹配。合格项直接按实测均值+2×标准差排名，输出模型`paired_measured_d_v1`；这一步不需要跨D拟合。预筛不是显存准入，最终候选仍要通过当前free和完整生命周期检查。显式锁D及Auto B2保留各自的成本预测流程。

Auto B2使用概率与Stage1成本联合排名。model策略允许已覆盖执行区间中的D使用位宽预测；未覆盖区间仅考虑该B2默认D。paired策略仍要求非默认D有合格跨D拟合或对应实测。估算均保留余量，不宣称通过生产收益验证。

### NTT 与配对阶段组合

特征分别计数 NTT形状/调用、系数归约、传输、baby/giant点和积累。多G驻留请求按原生 packing 分块；giant区分chain、ladder、短尾。普通归约用limb²、梅森归约用limb初始工作量。NTT缺项以邻近长度/批量的N·log₂N比例估计，不假设NTT本身平方增长。

完整曲线的互斥阶段提供秒数尺度。giant、G树、fold嵌套计时仅用于权重，不与包含它们的主循环重复相加。同算术/位宽/D/内存形状的已测B2区间可插值；区间外走组件估算。

paired策略及保留的采样摘要估算器可用跨D非负拟合分别估计固定开销、NTT工作量、归约工作量和点运算工作量。需要同实际位宽/算术/分块路径至少六个完整曲线摘要、三个D、两个B2，逐项留一验证通过`stage2_tune_error_limit`（默认8%）；P<512的探测不参与拟合。目标P仅允许处于已测P范围的0.5～2倍内。

该估算器数据不足时使用稀疏组件估算：多项式阶段使用NTT0.55、归约0.40、传输0.03、调用0.02的初始权重，固定主循环准备不再随giant数量放大。其初始误差余量为25%，随形状距离增加；该估算不是硬件定律，不能作为paired策略的固定B2替代证据；默认model策略使用上文独立的算术位宽模型。同位宽/D/路径的已测B2区间仍可插值。

`validation_max_relative_error=|预测−实测均值|/实测均值`来自加入该样本之前的预测；无独立证据时不写零。拟合的留一误差与该字段区分，均不以训练残差冒充独立误差。`independently_validated`不等于整个网格已经通过门限，也不代替当前输入的算术检查。

## 启动短校准

默认 `stage2_short_calibration=1`，独立预算10秒。没有合适文件、必要NTT组件缺失，或当前worktodo范围完全不在可复用已测位宽覆盖内时触发；部分覆盖时不因首项宽度外推反复校准。直接读save以该目标作为单项范围。

启动成本探测固定使用已准入的D/B2，校准子进程不继承生产`stage2_b2_round_chain`凑整许可。保存B1≥20时，D=210、probe_B2=42020、giant点数为202；正常任务及`--plan-only`之后仍按其生产配置凑整B2。这两个边界分别处理，不把扩大后的探测记成原边界的成绩。显式`--tune ecm`仍拒绝开启凑整，需用`--b2-round-chain 0`关闭。

校准回执必须匹配预期D、giant点数、执行路径和算术检查。失败时保留证据并报告预期D/B2/点数、实际D/点数、clean、bad、自检数及证据路径；不得放宽校验接纳改变了边界的成绩。复现入口为[凑整隔离回归工具](../../tools/test/test_stage2_calibration_rounding_gpu.py)，实现入口为`src/ecm/stage2/planning/planning.cpp::portable_measure`。

使用已验证save中的N、sigma及仿射点，做D=210、probe_B1=min(保存B1,20)、probe_B2=probe_B1+42000的完整引擎成本探测及少数NTT测量；所需承载缺摘要时补测。探测仅复用曲线上的有效点，不声称重新执行了probe_B1的Stage1；原B1与探测B1另存证据，用户save及生产B1/B2不变。这样生产B1增大不会扩大启动探测的巨步范围。短校准发现因子且全部算术检查通过时，保留证据、跳过该性能样本并继续正常任务；不将因子命中写入无因子性能表。算术检查失败仍中止。结果供后续tune利用，不消费队列，不代表生产B2精度验收；缺少完整引擎锚点时，固定B2可回退原生规划，Auto B2仍需可用成本。plan-only也可能触发校准并更新性能文件。`--short-calibration 0`可关闭；关闭后model策略缺文件使用内置基础成本，paired固定B2保留原生规划；Auto B2仍要求可用成本及正的T1。

[复合数启动回归工具](../../tools/test/test_stage2_startup_composite_gpu.py)验证M2003/sigma27/B1=20保存点的因子4007探测不进入性能表、正常Stage2仍完成；另对经独立GMP核对的真实M521/B1=10e6保存点验证probe_B2保持42020。[当前功能证据](../../data/experiments/stage2_n_scaling_20261011_2400/calibration_final_gate/regression_report.json)与[全位宽性能扫描](../performance/STAGE2.md)分别保留，功能回归计时不混入性能矩阵。

## NTT tune

`--tune ntt --length-log2 a:b --tune-slices s,... --tune-repeats n --tune-file FILE.toml`测长度和批量，也可用等级预设。共享软预算和增量摘要规则。每秒域卷积数=batch/mean，每秒批调用数=1/mean，不是ECM曲线吞吐，不含大整数carry、归约或完整Stage2。

Stage2 卷积基准逐项对照 GMP 参考。内存不足可跳过，验证失败报错并保留证据。包含 ECM 或队列矩阵的共享 TOML 摘要采用格式 16，仅包含 NTT 的摘要采用格式 14。Stage2 仍可读取格式 8 至 15 的摘要。显式 JSONL 输出保留逐次核验数据。

### NTT 性能摘要表

共享 `tune.toml` 只写一张 `[ntt]` 矩阵。前五列依次为 `length`、`log2_length`、`iter_per_s`、`batch`、`ntt_type`。每行通过 `condition` 关联设备和环境，保存实际执行参数、程序版本及 UTC 更新时间。行仍按 length、batch、tile 稳定升序排列。

- `stage1_linear` 和 `stage1_weighted` 的吞吐单位是每秒独立曲线的 ladder 步数。表保存 digit 位宽、tile、外层 radix 上限、正逆 pass、Graph 窗口、warp 尾层、计时次数及显存。加权混合长度的 `log2_length` 向下取整，完整长度由 `length` 标识，正逆 pass 用 `r5` 标记 radix-5。兼容输入按形状复用这些行。早期小于 2560 点的混合测量行仍可读，Stage1 选型只匹配当前支持的长度。
- `stage2_convolution` 的吞吐单位是每秒独立卷积数。一次卷积包含两个正向 NTT、有限域点乘和逆缩放、一个逆向 NTT。GPU event 排除输入生成、准备和核验。批耗时由 `batch / iter_per_s` 推得。矩阵保留实测次数、样本标准差和实际变换参数。
- Stage2 新测观察按次数合并批耗时及标准差。导入同形状摘要时保留更新时间较新的聚合结果，时间相同时优先样本数较多的记录。旧 `[ntt.N]` 表在读入时归并到单个 `[ntt]`，旧统计字段转换为当前列。逐次核验、准备和预热耗时仍在原始 JSONL。

独立基准仍记录容量证据：external数组为16Lb+40b+8 bytes，另有arena实际载荷及准入规划载荷。这些不是生产Stage2进程峰值，不持久写入性能矩阵；生产始终核对当前原生计划及设备可用显存。

### tile、radix 与处理趟数

长度L=2ᵏ的NTT有k级radix-2蝶形。执行器把其中t级放入共享内存tile，其余k−t级划成若干外层融合趟，每趟覆盖mᵢ级：

**Σmᵢ=k−t；passes_forward=外层趟数+𝟙(t>0)。**

每趟对相应切片的完整数组执行一次读入和写回；趟内多级蝶形复用寄存器或共享内存。规划器先最小化外层趟数，再按末段融合宽度打破平局；在当前允许的级数范围内，外层趟数为⌈(k−t)/max_radix_bits⌉。

**`tile=t`：共享内存融合的级数。** 一个tile含2ᵗ个元素，正向NTT先做外层趟，再做tile；逆向顺序相反。当前生产固定请求t=12，长度较小时截为min(12,k)。例如t=12表示4096元素，tile主数据共享内存为4096×8=32KiB；t=10表示1024元素，不是10个元素。增大t可减少外层趟数，也会增加每block共享内存、同步和资源压力，不能仅凭t更大判断更快。

**`max_radix_bits`：每个外层融合趟允许的最大级数。** 值4表示上限radix16，值6为radix64，值8为radix256。实际各趟mᵢ可小于上限；该字段不描述tile内部蝶形，也不是每个线程的数量。k≤t时没有外层趟，即使记录上限4也没有执行radix16外层内核。

**`passes_forward`：一个正向NTT的完整数组处理趟数。** 不等于蝶形级数，也不等于整次卷积的CUDA launch数。逆向变换使用相同趟数，因此两个正向加一个逆向共有3·passes_forward个变换趟；有tile时点乘与归一化融合在逆向tile趟中。表构建、输入生成和检查的launch另计；生产carry和模N归约也不在这个字段里。

- L=2¹⁰、t=10：没有外层级，passes_forward=1。
- L=2²⁰、t=12、上限4：剩余8级，外层4+4级，passes_forward=3；一次基准卷积共9个变换趟。
- L=2²⁴、t=12、上限4：剩余12级，外层4+4+4级，passes_forward=4。
- 同一L/t采用上限6时：外层6+6级，passes_forward=3。

每趟仅计主数组读写约为16Lb bytes；点乘还读第二个输入，旋转因子表访问另计。减少趟数能减少主数组传输，但不改变k级蝶形的总数学工作量，也不保证相同比例的时间收益。

**`coop_outer`：外层是否启用block内协作融合。** 普通外层路径让线程持有一组radix元素并在寄存器中完成蝶形；协作路径由256线程block在共享内存中共同处理radix轴，支持5…8级融合，减少单线程随2ᵐ膨胀的寄存器压力。它使用普通CUDA kernel launch，不是跨block cooperative launch，也不要求全GPU共同驻留。协作需要共享内存和block屏障，并有旋转因子复用；实际mᵢ≥5的趟调用协作内核，较小趟仍用普通路径。

当前生产采用确定的设备/形状规则：只有RTX4060 Laptop、sm89、t=12、warp_tail与compact_scratch启用及普通上限4等条件成立时，k=24启用协作上限6，k=25…27启用协作上限8；其他形状保持普通上限4。该规则在源码执行器中确定，不由TOML行重新配置。

- `warp_tail`：tile的部分低层蝶形采用warp shuffle，实际使用要求t≥5；它不改变NTT长度。
- `compact_scratch`：辅助表/临时空间采用按各趟需要计算的紧凑容量，不改变数学结果。

### 摘要如何影响选型

模型以`(length,batch)`查询`mean_seconds`；满足条件复用策略的同形状记录取最低平均时间。缺少精确形状时，从邻近测量点估算：

**t(L,b)≈t(L₀,b₀)·(b/b₀)·[Llog₂L/(L₀log₂L₀)]。**

短长度 L≤2¹⁹ 且有同长度测量时，优先在 batch 轴上分段线性插值 t(L,b)，保留小批次的启动成本；小于首个 batch 使用首个实测时间，大于末个 batch 按末点每卷积时间外推。没有同长度短表或 L>2¹⁹ 时才使用上述邻近长度公式。表外均为估算，不能声明实测吞吐饱和点。只有NTT表、没有完整ECM实测锚点时，仍不足以给出完整Stage2秒数，需要ECM tune或启动短校准。

每个D决定P=φ(D)/2及树结构。模型从实际packing形状和分块规则得到各阶段的NTT长度、batch和调用次数，累计调用次数×批调用成本，再结合归约、传输、点生成及完整引擎阶段锚点预测总时间。算术位宽模型的阶段权重是先验混合，并非阶段实测占比；保留的稀疏估算器0.55也不表示NTT实测占总时间55%。精确ECM点或同形状B2区间插值可直接使用完整曲线数据，不必由NTT重新推算总时间。

结果影响固定B2的D/P与合法承载选择、Auto B2联合搜索，以及ECM tune的中心D和相邻窗口。D/P改变后，所需NTT形状和调用分布随之改变。实际NTT长度仍由操作数/算术位宽、packing及精确性约束决定。默认 batch 由任务分块预算决定；实验性 stage2_short_ntt_batch_tune 可在预算允许的分块内按短 NTT 表选择 batch。

当前`tune ntt`不搜索tile、radix或协作开关；表中这些字段记录测量时的执行形状。成本模型没有把它们作为内核候选重新配置，容量和诊断字段也不直接参与时间排名。自动搜索并部署NTT内核配置仍列为待实现事项。

### 短 NTT 批处理选型（实验选项）

`stage2_short_ntt_batch_tune=false` 为默认；命令行 `--short-ntt-batch-tune 1` 可覆盖。输入是 `stage2_tune_profile` 内满足 `stage2_tune_ignore` 的已核验 NTT 摘要。仅 L≤2¹⁹ 且同长度至少有两个实测 batch 才参与选择；缺少资料、较长 NTT 和单 slice 请求保留原分块。

设独立请求数 R、候选 batch b、单批估计时间 t(L,b)，比较：

**T_NTT(R,b)=⌊R/b⌋·t(L,b)+t(L,R mod b)，t(L,0)=0。**

候选包含原预算分块以及不大于该分块的实测 batch。预计卷积时间须比原分块降低超过10%才改变；不会将不同树层或互相依赖的请求合并。运行、请求签名、NTT/S4生命周期规划和组件模型使用相同选择函数。纯 NTT 表不含 packing、carry、系数归约、上传/读回及额外分块同步，所以这不是完整 Stage2 性能保证。

短NTT策略不改变任务曲线进度身份；调优条件中记录策略身份。策略数据 hash 改变后，不沿用原策略的精确测量或同形状B2插值，相关预测保持估算口径。

NTT 校准和完整 ECM 校准保持独立：完整 ECM/启动短校准使用原分块，存入实际条件，不伪装成启用新策略的测量。运行日志给出策略摘要和参与数据的 hash。[固定 D 冻结实验](../STAGE2_SHORT_NTT_BATCH_DRAFT.md)中，当前策略未显示稳定收益；7995-bit通用余因子两档B2均慢约2.1%，因此保持默认关闭。后续选型需要计入每分块的归约、准备、检查和同步成本，不能仅用独立NTT的Conv/s决定分块。

NTT 参数表发布时按共享行规则 `(length,batch,tile)` 数值稳定升序排列，详见[矩阵行列定义](#toml-中的算术路径)。现有生成表可用 `tools/bench/format_stage2_tune.py <profile> --in-place` 重排；此工具仅移动行，保留测量值、其他节和列结构。

### NTT 摘要代码依据

- [tune.cpp](../../src/ecm/stage2/cuda/tune.cpp)：`stage2_tune::run`，基准载荷、两个正向/一个逆向、GPU事件核验及字段输出。
- [ecm_stage2_tune_portable.h](../../src/ecm/stage2/planning/tune_portable.h)：`validate_ntt_summary`、`strip_ntt_auxiliary_timing`、`Profile::update_ntt/load/serialize`，摘要检查、时间字段剔除及参数表发布。
- [runtime.cpp](../../src/ecm/cuda/ntt/runtime.cpp)：`fuse_plan_stages`、`fuse_describe`、正向/逆向执行与 `ntt_pack_operand`；[shape.cpp](../../src/ecm/cuda/ntt/shape.cpp)：形状选择及精确性约束。
- [outer_coop.cuh](../../src/ecm/cuda/ntt/device/outer_coop.cuh)：block 协作外层内核；[NTT 模块入口](NTT.md#代码入口与依据)。
- [ecm_stage2_portable_model.h:130](../../src/ecm/stage2/planning/portable_model.h#L130)：NTT成本查询/外推；[工作量:28](../../src/ecm/stage2/planning/portable_model.h#L28)；[非负拟合:194](../../src/ecm/stage2/planning/portable_model.h#L194)；[阶段估算:236](../../src/ecm/stage2/planning/portable_model.h#L236)。

## 汇集预计算结果

当前格式按位宽组织 `[width.<bits>]`，所有 NTT 条件共用 `[ntt]`。每个表只有一个 matrix，首行列名与数值保持列宽对齐。同参数的次数、均值和样本标准差合并保存；不同参数不受逐表记录数淘汰限制。可选单元缺失用空字符串，不因字段集合差异拆表。模型节点之间的值仍为估算，不冒充实测。

来源、条件、均值/标准差、验证标记及必要独立误差保留，读回恢复确定性派生量并执行结构与统计校验。重复列、列数不符、嵌套数组、错误位宽/条件、坏核验及重复候选均拒绝。

读取格式 8 至 16，正常写回或显式合并时按内容选择格式 16（ECM 或队列）或 14（仅 NTT）。只读使用保留原文件。旧格式 8 的条件级 `program_version` 在读取时迁入行内，只有版本差异的条件按同一条件处理。缺少布局的旧行按当时的策略还原，缺实际非零后端证据时保留未知身份。格式 8 以前的中位数表需重新生成，显式合并会报错，短校准成功发布后才原子替换。固定 B2 允许使用无缓存模型，Auto B2 要求可用成本条件。旧 `.cprof` 及独立 Stage1 TOML 有各自的格式。

同条件同完整参数的独立测量按次数合并均值和标准差，相同/子集快照不重复计数，不同参数保留多行。运行和正式调优分别累计；估算不覆盖实测。写入使用独占锁，临时文件重新校验后原子替换。

合并：`--tune ecm --tune-merge A.toml --tune-merge B.toml --tune-ntt-profile NTT.toml --tune-file combined.toml`。不运行曲线或查询GPU。只有NTT的文件尚无完整引擎锚点，需要短校准或ECM tune。

显式 tune 的原始日志、计划、回执、二进制 SHA256 与可用构建清单位于工作目录的 `data/experiments/`。正常运行使用现有结果 JSONL 保存采集回执，矩阵行保留来源版本和 UTC 更新时间。队列进度身份独立于性能文件、D 或承载选型，增量 tune 保留已完成编号。Stage1 成本输入身份另行保护。

## 验证范围与限制

当前格式、统计及GPU流程检查见[当前验证范围](#当前验证范围)。非负拟合仅在合成工作量上验证模型恢复与留出资格，不证明GPU预测误差达到8%。概率核心377组参考对拍保持不变，最大绝对误差3.44×10⁻¹⁵，证据 `data/experiments/stage2_portable_tune_20261010/probability_01/` 仅用于概率组件。

稀疏跨位宽/跨B2预测已出现显著误差，早期冻结功能样本最高约343%和211%，证据在`data/experiments/stage2_portable_tune_20261010/gpu_01/`。局部D搜索可能错过不能经相邻显著改善到达的真实最优项；功能检查不能作为8%收敛或全表最优D证明。非负拟合已有资格检查，仍待孤立组件校准及更多生产B2/高位宽独立排名验证。

G=1与baby预算回退缺完整联合生命周期，成本可估算但原生准入不放行。host fold/frontier采用受限arena的保守上界；部分借用设备NTT的cap失败回退尚未完成执行验证。见 [内存](MEMORY.md)及 [TODO](../TODO.md)。旧`.cprof`接口保持独立资格合同，不套用上述模型。

共享候选表和NTT区间重启已接入当前源码。冻结历史数据不作为当前候选域或搜索的性能验收；新实现还需相同输入的完整矩阵及配对计时，见[TODO](../TODO.md)。

## 代码入口

- [ecm_stage2_d_catalog.h:14](../../src/ecm/stage2/planning/d_catalog.h#L14)：唯一D字面量表；`entries`、`degree`提供派生P，编译期检查范围、偶数、唯一和顺序。
- [plan.cpp:196](../../src/ecm/stage2/cuda/plan.cpp#L196)：原生默认成本候选。
- [planning/stage2.cpp](../../src/ecm/stage2/planning/planning.cpp)：默认几何模型；[运行时搜索](../../src/ecm/stage2/planning/planning.cpp)、[运行时采用](../../src/ecm/stage2/planning/planning.cpp)处理候选和重复观察；[portable_select](../../src/ecm/stage2/planning/planning.cpp)负责实测预筛、成本排名及准入；[run_portable_tune](../../src/ecm/stage2/planning/planning.cpp)负责正式搜索、测量与发布。[运行采集](../../src/ecm/stage2/planning/planning.cpp)负责采集与可选缓存写回。
- [ecm_stage2_portable_model.h](../../src/ecm/stage2/planning/portable_model.h#L194)：`Estimator::fitted`负责非负拟合及留一资格；同文件提供工作量、观察匹配、插值与误差余量。
- [ecm_stage2_d_search.h:27](../../src/ecm/stage2/planning/d_search.h#L27)：`DSearch`负责改善方向延伸、比例细化和区间重启；[ecm_stage2_runtime_tune.h:34](../../src/ecm/stage2/planning/runtime_tune.h#L34)负责额外成本预算，[runtime_choice:56](../../src/ecm/stage2/planning/runtime_tune.h#L56)执行试跑/采用分离及合格运行均值排名。
- [ecm_stage2_tune_portable.h:55](../../src/ecm/stage2/planning/tune_portable.h#L55)：`LocalSearch`用于显式D研究列表；[Profile:442](../../src/ecm/stage2/planning/tune_portable.h#L442)负责矩阵解析、排序与均值累计。
- [ecm_stage1_cost_csv.h](../../src/ecm/stage2/planning/stage1_cost.h)、[ecm_stage2_probability.h](../../include/ecm/stage2/probability.h)：容器模型及概率。
- [ecm_cuda_stage2_main.cpp](../../src/ecm/stage2/cuda/main.cpp)：配置、验证与队列。
- [配置参考](../ECM_TOML_REFERENCE.md)、[使用](../usage/STAGE2.md)、[工具](../../tools/bench/README_STAGE2_CARRIER_PLAN.md)。
