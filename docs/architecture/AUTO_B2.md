# Auto B2 与 tune

## 目标与生产入口

Auto B2 选择连续生成、处理新曲线的单位时间收益，计入 Stage1 成本，即使当前读取已完成的 save。固定 B2 时，性能文件选择 D 和合法梅森承载；最终 B2=0 且启用 Auto B2 时联合选择 B2、D、承载。

默认性能文件为 INI 所在目录的 `stage2_ecm_tune.toml`，可由 `stage2_tune_profile` 或 `--tune-profile FILE.toml`覆盖。`--auto-b2` 对应 `stage2_auto_b2=1`。显式同时提供非零 `--b2` 与 `--auto-b2` 报冲突；任务或 INI 已有非零 B2 时保持固定值。

性能数据可跨实际位宽、B1 复用，允许插值、外推。复用从不要求 B1 相等；实际 B1 仍用于区间、成功概率、Stage1 成本和真实运算。模型估算与实测在输出中分别标识。

`stage2_tune_ignore` 默认 `gpu,driver,cuda,backend,environment`，可选项另含 `memory`；空值要求这些条件匹配。默认 memory 按 fold/frontier 执行路径匹配，不要求预算数值相等。忽略条件不会取消 save、算术、整除关系和当前显存检查，也不表示其他设备的秒数已经在本机验证。

非零 `--d` 锁定 D。`--carrier-exponent 0` 锁定普通运算，非零 p 锁定承载。`stage2_auto_carrier=1` 默认开启，CLI `--auto-carrier 0|1` 可覆盖：未锁定算术时，worktodo `ECMSTAGE2=1,2,p,-1,...` 先按已知因子重建实际 N，核对 save，再验证 **N 整除 2ᵖ−1**，将普通运算与梅森承载共同纳入选型。指数须满足 2≤p≤16384；非标准梅森表达式和单独读取 save 不猜测指数。

`stage2_auto_carrier=0` 关闭自动加入承载候选，不影响显式 `--carrier-exponent p`。显式 0 始终锁定普通运算。父进程将最终开关传给曲线子进程；该开关不改变目标 N/B2，因此改变它不重置队列完成数。配置定义及双语说明见 [INI 参考](../ECM_INI_REFERENCE.md#stage2_auto_carrier)；入口为 `src/core/ecm_cuda_stage2_main.cpp::arguments/run`，候选与校准资格为 `src/core/ecm_stage2_portable_driver.inl::portable_carriers/ensure_portable_calibration`。

当前代码依据：`config/ecm_options.json` 定义键及默认值；`ecm_cuda_stage2_main.cpp::arguments/run/child_run` 分别解析参数、识别任务指数及传递子进程设置；`ecm_stage2_queue_state.h::Task` 定义曲线进度身份；`ecm_stage2_portable_driver.inl::portable_carriers/ensure_portable_calibration` 验证整除关系及短校准覆盖。源码均在 `src/core/`。

加入候选不等于启用承载。默认 `stage2_prediction_policy=model`：优先使用精确形状实测，其他输入分别预测普通和梅森算术的位宽区间成本，经过双侧误差保护及当前显存准入后选择。`paired`恢复精确目标位宽/B2/执行形状的配对实测门禁。承载只是扩大运算环，Stage1点、因子和GCD的目标仍为save中的N。

### 自动承载未被选中时

先区分候选识别与性能资格。规划中的 `carrier_exponent=p`、`lifted=1`、`target_divides_carrier=1` 表示已识别并验证承载；最终 `carrier_exponent=0` 表示选型保留普通路径，不能据此判断INI开关未读取。D210/B242020短探针不作为跨位宽完整曲线锚点；区间校正只使用P≥512的干净完整实测。

`paired`策略及精确形状实测还核对实际batch、物理分块、缓冲数量、giant路线及fold/frontier状态，B1不参与匹配。`model`策略从当前请求重算工作量；预算数值和梅森指数不作为采样身份门槛。区间未覆盖或预测收益不足时保留默认。

未替换时保留具体原因。`paired`中的`default_planner_protected_missing_paired_baseline`表示缺少配对基线；`model`中的`default_protected_by_prediction_uncertainty`表示候选未通过预测保护。`alternatives`分别展示普通/梅森的算术位宽、D、秒数、来源和余量。这不构成全局最优证明；M12011真实计时见[性能依据](../performance/STAGE2.md#m12011-真实-save-配对与默认无预热)。

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

入口为`src/core/ecm_stage2_width_prediction.h::WidthPredictor`，冻结资料在`ecm_stage2_width_reference.h`；由`tools/bench/generate_stage2_width_reference.py`核对原日志SHA后再生成。生成物保留来源二进制和证据摘要，不需要部署原始实验目录。

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

任务输入按已知因子精确重建N，读取skip后的首个有效save记录，核对N、checksum及可用X，将这一条原始记录冻结到实验目录；不推进worktodo，不写生产完成回执。B2采用显式`--tune-b2`列表，否则采用任务非零B2，再采用INI的`stage2_b2`；均为0时使用等级网格，只保留B2>保存B1的档位。重复任务合并，同实际位宽但不同模数或承载指数的输入仍分别保留。

`stage2_auto_carrier=1`允许标准梅森任务在验证N整除2ᵖ−1后调优普通/承载两条路径；0只调优普通路径，显式`--tune-carrier-exponent p`配合`--tune-save`仍可指定承载。独立save可从当前worker中重建N完全相同的有效worktodo任务取得p，不从文件名或余因子位宽猜测指数。完整Mₚ的普通路径已经使用梅森归约，自动调优不重复运行相同模数的承载；显式承载仍保留。

预算先用于各实际目标/B2的普通默认D与承载默认D完整参考，再用于两条路径各自的NTT区间和D邻域。不同路径的默认D可以不同；目标是比较完整Stage2时间。同D算法对照可用显式D列表。显式列表不暗中添加默认D；省略普通默认基线时没有精确配对替代资格，预测策略仍采用区间与误差保护。预算不足保留真实正式重复数，不把未完成配对记为已验证收益。输入和配对排序入口为`src/core/ecm_stage2_portable_driver.inl::portable_tune_tasks/portable_task_carrier/run_portable_tune`。

### TOML 中的算术路径

格式8每个实际目标位宽只有一个 `[width.<bits>]`，内容为列对齐的 `matrix`。不同条件、D、B1/B2、普通/承载及giant执行参数保留为不同的行，不因可选字段不同再分表。表外仅有格式/计时单位、模型误差设置和确定的设备、环境条件。矩阵中可选值缺失用空字符串表示。

- `condition` 指向 `[condition.<id>]`，保存设备、驱动/工具链、后端及实际环境设置。程序版本不参与条件匹配、统计分组、采样上限或选型资格；软件升级不自动使可兼容缓存失效。
- `program_version`、`updated_at` 位于ECM和NTT的matrix行内，仅记录来源。实测行保存最近一次有可靠时间且已纳入统计的观察/测量批次；模型行保存生成该估算的版本和时间。时间采用UTC `YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ`。旧数据缺少可靠采样时间时留空，不能用读取、导入或格式化时间代替。去重忽略的观察不刷新来源；双方时间均未知时保留已有版本。合并行不展开逐次版本历史，原始来源保留在JSONL或冻结证据。
- 位宽由表名确定；`carrier_exponent=0` 表示按save的N运算，非零p表示2ᵖ−1承载。算术位宽自动由表名和承载指数恢复。`modulus_kind` 区分generic/mersenne；完整梅森N允许carrier_exponent=0。
- `d` 为D，P=φ(D)/2自动恢复；fold/frontier列自动恢复驻留路径名称。giant分块、chain门槛和ladder开关影响执行形状，保留在matrix中。
- `source`、`benchmark_kind` 区分模型、正式调优和运行采集。完全相同参数的独立测量合并为一行：`repeats` 是次数，`mean_seconds` 是算术平均耗时，`stddev_seconds` 是样本标准差。正式ECM及NTT也按真实次数合并，较慢观察同样计入。
- `verified=1` 仅在运行检查通过后写入。逐次selftest/checked/bad/hits计数、预热次数、init/main重复计时、预测值/比值及模型过程保留在内部JSONL，不重复保存到性能矩阵。用于组件缩放的互斥阶段及giant/tree/fold嵌套组件保存均值，不相加冒充总耗时。
- `measurement_ids` 是有界运行观察或正式批次的去重身份集合，以长度前缀编码为一个字符串单元；`preferred_order` 保存已成功采用路线的顺序。它们属于matrix记录，不作为表外公共字段。

完整ECM的均值使用 `stage2_full_wall.total=init+main`，包括初始化、必需自检、GCD及算术收尾，不含Stage1、选型、进程启动、因子分解和TOML写回。每次执行前的模型预测及误差仍记录在JSONL；缓存仅保留用于成本/误差保护的摘要。

NTT表只测Goldilocks有限域卷积，不含大整数carry和模N归约，不按梅森承载重复一份。实现入口为 `ecm_stage2_tune_portable.h::compact_sample/hydrate/Profile::update/serialize`。

列优先级只在 [`src/core/ecm_stage2_tune_columns.def`](../../src/core/ecm_stage2_tune_columns.def) 定义：移动其中的字段行即可调整顺序；未列出的可选字段按字母顺序排在后面。生产程序在编译时展开该定义，Python导入器通过 `tools/bench/stage2_tune_matrix.py::matrix_column_order` 读取同一文件，保留现有列宽对齐。调整顺序需要重编译生产程序；读取仍按列名解释，不依赖文件中的列位置。

### 正常运行累计采样与候选试跑

`stage2_tune_runtime=true` 默认开启；CLI 为 `--tune-runtime 0|1`。`stage2_tune_runtime_samples=10` 指定每组独立 NF 样本上限，范围2–1000，CLI 为 `--tune-runtime-samples N`。候选先取得两次观察，胜出路线随继续执行累计至上限。达到上限后停止该组采集，已有证据保留；调低上限不删除已有样本。关闭采集仍可读取已累计数据选型，但不再安排新试跑或写回。

同组要求相同设备、驱动/工具链/后端与环境、目标/算术位宽、承载指数、实际B2/D、packing与物理分块、giant路线及fold/frontier执行形状。B1、具体N和sigma不作为Stage2成本复用门槛；仍须分别验证当前目标、save和承载整除关系。不同程序版本的可兼容统计共同参与复用和累计；环境的可忽略维度仍受显式tune-ignore及模型误差保护约束。

固定B2、`model`策略下，用待处理曲线建立完整Stage2对照，不额外重算同一生产曲线：先试合法梅森承载的默认D，再建立普通默认D基线，随后补测已记录的较快D和每个算术家族的一个模型起点。每条待比较路线至少需要两个合格NF；因子命中或失败不计数。模型起点不因一次预测偏低而不断扩展，已测路线继续参与比较。候选逐条通过当前原生显存/执行计划准入；承载优先级不预设任何百分比收益。

显式D锁限制候选D；显式`--carrier-exponent 0|p`锁定算术。关闭`stage2_auto_carrier`时不自动加入承载。独立save不从文件名或位数猜指数；合法worktodo提供并验证p。`paired`策略仍使用正式配对调优规则。Auto B2先选择覆盖上界，再允许在同一实际B2下进行上述正常运行比较。

### 实测统计与路径保留

同参数的n次观察只保存一行和三个充分统计：次数n、均值μ、样本标准差s。新增m次独立观察时，按加权均值及组间均值差合并方差：

μ = (nμ₁ + mμ₂)/(n+m)

M₂ = (n−1)s₁² + (m−1)s₂² + (μ₂−μ₁)²·nm/(n+m)，s = √[M₂/(n+m−1)]。

单次观察的s记0，至少两次才建立 `runtime_shape_mean_v2` 实测估计。切换条件为 `μ_A+2s_A < 0.98μ_B−2s_B`；这是工程切换门槛，不是置信区间。当前设备、环境和形状的合格运行组不叠加跨条件模型余量；未取得重复观察的输入继续使用预测保护。

存储身份包含B1，避免把不同完整参数写成同一行；成本复用不以B1为门槛，会合并匹配形状的统计；采集上限按各完整参数行的repeats独立计算。具体N和sigma不作为成本身份，但每条曲线仍独立验证。

波动范围重叠时保留已成功采用的路线；首轮无保留对象时从普通基线比较。承载首个试跑不等于必须采用。日志 `runtime_candidates` 输出样本次数、平均时间及标准差。正式调优和运行观察分开累计，不混为一个低波动总体。

运行上限按观察次数而非行数统计；不同参数行不按排名或数量自动淘汰。正式调优批次按真实次数累积。测量身份保证重复导入不增计数：相同/子集快照忽略，完整超集替换；部分重叠且无法恢复交集耗时的统计快照拒绝合并，须从原始JSONL去重恢复，不能猜测交集或重复计算。

### 验证与持久化

只采集P≥512且具有完整多批请求拓扑的曲线；G=1及D210启动探针不作为完整区间锚点。结果必须无因子、clean=1、自检和抽查量为正、bad=0、互斥阶段计时完整，且实际D/P/giant数量及fold/frontier状态与原生准入计划相符。规划、失败、中断、save中提前发现因子、调优子进程、因子命中和改变规划路径的回退不加入NF样本。

计时保持`stage2_full_wall.total=init+main`，不包含Stage1、进程启动、选型和发布；每个观察保存采集前预测及完整数学检查。曲线结果先持久保存，再取得tune独占锁、重新读取文件、去重、检查上限并原子写回。锁占用或写回错误只提示，不改变已完成曲线或队列计数。

生产JSONL的`tune_observation`可由[导入工具](../../tools/bench/import_stage2_benchmark_tune.py)的`--runtime-results FILE --runtime-samples 10`恢复，再经原生`--tune-merge`汇集。重复导入同一观察不增加计数；导入保留上限内的独立较慢样本，不只保留最快一条。原始曲线、日志和失败证据放在忽略目录；TOML仅保存合并统计、必要组件均值及去重身份。

实现入口为`ecm_stage2_portable_driver.inl::portable_runtime_select/prepare_runtime_tune/complete_runtime_tune/publish_runtime_tune`，统计在`ecm_stage2_portable_model.h::runtime_statistics`，切换规则在`ecm_stage2_runtime_tune.h::runtime_choice`，均位于`src/core/`。CPU门禁、GPU生产流程及离线恢复脚本为`tools/test/test_stage2_runtime_tune.py`、`test_stage2_runtime_selection_gpu.py`、`test_stage2_runtime_task.py`和`test_stage2_runtime_import.py`；各脚本要求新的`--output data/experiments/<目录>`。

基准扫描须设置`stage2_tune_runtime=false`并冻结性能文件。此时仍可使用该文件已有的合格实测组；若需要完全固定D/算术，应同时显式锁定它们。

### 当前验证范围

0.37.2在GPU1 RTX4060 Laptop上验证15次队列NF：使用M1009的383位余因子save，B1=50000、B2=10⁸、D=30030、batch=64 MiB、arena=6300 MiB、owner=640 MiB。12次独立执行观察合并为普通10次和承载2次共两行；另核对显式D/算术锁、关闭写入仍读取统计、Montgomery Auto B2计划和因子命中不入样本。采用相同验证保存点重复运行，属于功能门禁，不是新的独立ECM覆盖或性能基准；计时边界为引擎init+main。

CPU检查覆盖独立两遍方差参考、加权合并、较慢观察、10次上限、跨版本复用与设备隔离、完整/子集快照去重、部分重叠拒绝、不同B1记录、NTT参数行与布尔列、对齐矩阵稳定读写、20条不同参数、10级网格、非负拟合、105组邻域及坏文件拒绝。共享列序检查核对原生/Python表头一致、旧条件迁移、最近来源按时间选择、未知时间留空及重复导入不刷新时间。正式GPU门禁通过22项：驻留/host fold调优、512/4423位输入、跨B1复用、短校准、Auto B2、NTT、合并、输入保护、旧格式重建、D邻域、实际执行及承载规划。4条NTT摘要通过排序、数值保持和格式化幂等检查。

真实0.37.1 format 8文件在0.37.2只读规划时不重写或重新校准；续算一次NF后，原普通10次统计累计为11次，均值/方差按独立参考核对，未运行的承载行仍保留0.37.1和未知时间。恢复工具把两版24次JSONL观察合并为同一条件、普通20次和承载4次两行；重复导入/合并不增加计数，较慢观察保留，坏检查回执拒绝。

证据位于 `data/experiments/stage2_tune_provenance_20261018/`：`cpu_final01/`、`portable_final01/`、`provenance_final01/`、`reuse_final01/`、`gpu_final02/`及其审计、`portable_gpu_final02/`、`import_legacy01/`、`import_mixed_final01/`、`sort_final02/`；最终身份汇总见`final_acceptance.json`，前置运行目录保留。15条队列NF的GMP抽查18836，错误0。正式local构建为 `build_cuda_cmake/local_stage2_0_37_2/ecm_cuda_stage2.exe`，sm89、CUDA13.3.73、MSVC14.51.36231、PTX3/addsub1、GMP Zen3；CUDA对象与0.37.1一致，变更位于主机侧。冻结85项编译输入，EXE SHA256为`5714f405d4bde1853666e94c5add94bdf31cbb01e6792e0ba3e3f7f8d81a9243`。

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

唯一维护位置是 [ecm_stage2_d_catalog.h:14](../../src/core/ecm_stage2_d_catalog.h#L14)，当前含1094个严格递增、去重、偶数D，范围22～199879680。P=φ(D)/2统一派生并缓存，不手工维护第二份D/P值。例如D1021020=2×510510，P92160；B2是独立输入。

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
- **owner边界：**默认reuse=3的fold精确payload为8·ceil(S/64)·(7P+7)+48 bytes；frontier再增加24P bytes。对48/128/256/640/1024 MiB分别保留两侧P代表。公式依据为 [FoldOwnerLayout](../../src/core/ecm_stage2_geometry.h#L102)、[frontier元数据](../../src/core/ecm_stage2_owner_memory.h#L65)。

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

先测中心及前后邻项。候选A替代B要求 **T_A+2·s_A < 0.98·T_B−2·s_B**。有显著收益时移向更快邻项，只补测新暴露的邻项；否则停止。另一区间可直接重新选中心，无须经由逐项更慢的过渡D。模型不足时使用归一化完整阶段工作量作排序先验，不视为秒数实测或生产获胜证据。

完成正式重复要求的同形状测量可复用；预算结束、已有时间预计超过剩余预算或重复数不足时标为`deferred`，不当作无效项关闭方向。`rejected`表示数学/当前原生准入拒绝；`complete=1`仅表示本次选择的所有区间邻域已完成且没有延期，不表示全表或全局最优。完成的正式样本及真实重复数保留，下一次tune重新模型排序并复用它们。正式新测时间用于本次邻域比较，即使文件按均值累计规则仍保留了更快的旧部分样本；不借本次重复数提升旧样本资格。没有实测的高B2形状不根据P=24的远距离外推时间直接跳过。

纯packing形状按操作数/算术位宽缓存；工作量按完整Shape缓存，拟合及资格按模型域缓存。缓存只在相应条件和不变profile内复用，更新测量后重新创建估算器；现场free准入不由这些数学缓存代替。本轮已完成的配对参考按目标对象及完整Shape复用，防止磁盘上的更快部分摘要造成重复测量；当前邻域使用本轮正式结果，不提升旧部分摘要的重复数。实现见 [区间键](../../src/core/ecm_stage2_portable_model.h#L74)、[搜索](../../src/core/ecm_stage2_portable_driver.inl#L650)。

`--tune-exponents bits,...`在通用家族表示**实际位数**，在梅森网格表示算术指数p；`--tune-d`、`--tune-b2`、`--tune-repeats`覆盖对应网格。通用家族构造精确位宽的GMP probable prime，sigma26、B1=20，不称为已证明素数。`--tune-save FILE`使用验证后的真实点并替代位宽列表；自动承载要求匹配真实任务，显式`--tune-carrier-exponent p`仍检查整除。旧 `--tune-tail-samples`只接受0。

完整测量默认直接正式采样，要求mandatory自检、GMP检查、clean通过、bad=0、hits=0。显式开启的预热也接受相同检查，但不计入正式时间。预算中途结束保留已完成正式重复及真实次数；仅完成预热时不生成摘要。再次tune复用已有候选并继续未完成邻域。TOML保留访问候选的最终测量/估算，不展开全表估算。

控制台保留 `ecm_tune_window`、`ecm_tune_progress`、`ecm_tune_window_done` 和完成/失败原因。子进程阶段、NTT统计及内部自检不转发到控制台，完整诊断仍保存在每次测量的证据日志；正常Stage2的控制台与日志规则不受此限制。

### 默认选型与实测保护

默认模型在最大NTT长度低于2²⁷时使用共享表的原生D成本排名；达到当前portable成本域2²⁷边界后，只比较表内标记`default_plateau`、仍驻留、最大NTT长度不增加的更大D。这14个标记只保留既有保守平台保护的资格范围；新增候选不自动获得该覆盖资格。形状查询能够返回更长NTT，这一策略边界不代表所有设备或底层NTT的绝对长度上限（`src/core/ecm_stage2_portable_model.h:49`）。通用归约还要求默认巨步数至少16P，以摊销较贵的初始化；梅森归约不采用此附加门槛。归一化成本包含G树、fold、giant及初始化，避免在同一大NTT平台内过早减小P；显存准入仍使用完整生命周期。该几何模型不是绝对秒数预测；性能依据和范围见[Stage2性能](../performance/STAGE2.md)。

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

使用已验证save中的N、sigma及仿射点，做D=210、probe_B1=min(保存B1,20)、probe_B2=probe_B1+42000的完整引擎成本探测及少数NTT测量；所需承载缺摘要时补测。探测仅复用曲线上的有效点，不声称重新执行了probe_B1的Stage1；原B1与探测B1另存证据，用户save及生产B1/B2不变。这样生产B1增大不会扩大启动探测的巨步范围。短校准发现因子且全部算术检查通过时，保留证据、跳过该性能样本并继续正常任务；不将因子命中写入无因子性能表。算术检查失败仍中止。结果供后续tune利用，不消费队列，不代表生产B2精度验收；缺少完整引擎锚点时，固定B2可回退原生规划，Auto B2仍需可用成本。plan-only也可能触发校准并更新性能文件。`--short-calibration 0`可关闭；关闭后model策略缺文件使用内置基础成本，paired固定B2保留原生规划；Auto B2仍要求可用成本及正的T1。

[复合数启动回归工具](../../tools/test/test_stage2_startup_composite_gpu.py)验证M2003/sigma27/B1=20保存点的因子4007探测不进入性能表、正常Stage2仍完成；另对经独立GMP核对的真实M521/B1=10e6保存点验证probe_B2保持42020。[当前功能证据](../../data/experiments/stage2_n_scaling_20261011_2400/calibration_final_gate/regression_report.json)与[全位宽性能扫描](../performance/STAGE2.md)分别保留，功能回归计时不混入性能矩阵。

## NTT tune

`--tune ntt --length-log2 a:b --tune-slices s,... --tune-repeats n --tune-file FILE.toml`测长度和批量，也可用等级预设。共享软预算和增量摘要规则。每秒域卷积数=batch/mean，每秒批调用数=1/mean，不是ECM曲线吞吐，不含大整数carry、归约或完整Stage2。

GMP参考验证每个输出项；摘要保留平均耗时、样本标准差、次数及已验证标记；逐次检查量留在原始JSONL。内存不足可跳过，验证失败报错并保留证据。TOML用格式8参数表，显式JSONL调试输出保留逐次接口。

### NTT 性能摘要表

`[ntt.<condition>]`关联对应设备/环境条件，内容只有一个列对齐matrix。每行保存一组长度、批量、tile/radix及执行开关的统计，以及来源版本和UTC更新时间；完全相同参数重复测量合并次数和均值/标准差，不同参数保留多行。NTT长度对数、吞吐率、检查字数等确定性派生量不重复存储。

一次基准批调用包含两个正向NTT、有限域点乘与缩放、一个逆向NTT，运算域为Goldilocks素数域。GPU事件只包围这段计算，输入生成、准备、核验在计时外；不包含生产多项式的packing、carry、模N归约、传输和整条ECM曲线。

**性能矩阵字段：**

- `length=L`：每个NTT的有限域元素数，L为2的幂。
- `batch=b`：一次批调用中的独立卷积数。
- `repeats`、`mean_seconds`、`stddev_seconds`：正式观察次数、批调用平均GPU耗时、样本标准差。显式预热不计入次数。
- `tile`、`max_radix_bits`、`coop_outer`、`warp_tail`、`compact_scratch`：实际执行形状。
- `verified=1`、`measurement_ids`：通过算术验证的标记及独立批次身份。

log₂L、batch/mean、1/mean、L·b检查字数由程序推导，不重复保存。NTT使用GMP的规定输入参考：两个各含3个非零系数的多项式，各切片常数项不同，核对全部输出及零尾部；不声明随机输入覆盖。逐次bad计数、最小/最大值、预热及准备/核验耗时留在底层JSONL。

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

NTT 校准和完整 ECM 校准保持独立：完整 ECM/启动短校准使用原分块，存入实际条件，不伪装成启用新策略的测量。运行日志给出策略摘要和参与数据的 hash。[固定 D 完整曲线验证](../performance/STAGE2.md#短-ntt-分块选择)中，当前策略未显示稳定收益；7995-bit通用余因子两档B2均慢约2.1%，因此保持默认关闭。后续选型需要计入每分块的归约、准备、检查和同步成本，不能仅用独立NTT的Conv/s决定分块。

NTT 参数表发布时按 `(log2_length,length,batch,tile)` 数值升序排列，列顺序与行排序分别维护。现有生成表可用 `tools/bench/format_stage2_tune.py <profile> --in-place` 重排；此工具仅移动行，保留测量值、其他节和列结构。

### NTT 摘要代码依据

- [ecm_stage2_tune.cuh:91](../../src/cuda/ecm_stage2_tune.cuh#L91)：基准载荷；[计时与核验:119](../../src/cuda/ecm_stage2_tune.cuh#L119)：两个正向、一个逆向与GPU事件；[字段输出:141](../../src/cuda/ecm_stage2_tune.cuh#L141)。
- [ecm_stage2_tune_portable.h](../../src/core/ecm_stage2_tune_portable.h)：`validate_ntt_summary`、`strip_ntt_auxiliary_timing`、`Profile::update_ntt/load/serialize`，摘要检查、时间字段剔除及参数表发布。
- [ntt_runtime.cuh:1418](../../src/cuda/stage2/ntt_runtime.cuh#L1418)：外层级数规划；[形状选择:1488](../../src/cuda/stage2/ntt_runtime.cuh#L1488)；[正向与逆向:1678](../../src/cuda/stage2/ntt_runtime.cuh#L1678)；[packing:3224](../../src/cuda/stage2/ntt_runtime.cuh#L3224)。
- [ntt_coop_outer.cuh:15](../../src/cuda/stage2/ntt_coop_outer.cuh#L15)：block协作外层内核。
- [ecm_stage2_portable_model.h:131](../../src/core/ecm_stage2_portable_model.h#L131)：NTT成本查询/外推；[工作量:28](../../src/core/ecm_stage2_portable_model.h#L28)；[非负拟合:171](../../src/core/ecm_stage2_portable_model.h#L194)；[阶段估算:239](../../src/core/ecm_stage2_portable_model.h#L239)。

## 汇集预计算结果

格式8按位宽组织 `[width.<bits>]`，NTT按条件组织 `[ntt.<condition>]`；每个表只有一个matrix，首行列名与数值保持列宽对齐。同参数的次数、均值和样本标准差合并保存；不同参数不受逐表记录数淘汰限制。可选单元缺失用空字符串，不因字段集合差异拆表。模型节点之间的值仍为估算，不冒充实测。

来源、条件、均值/标准差、验证标记及必要独立误差保留，读回恢复确定性派生量并执行结构与统计校验。重复列、列数不符、嵌套数组、错误位宽/条件、坏核验及重复候选均拒绝。

程序升级不触发格式重建。旧format 8的条件级 `program_version` 在读取时迁入行内；仅版本不同的相同条件合并，独立测量按真实次数合并均值和方差，重复快照仍去重。只读使用不改写原文件；后续正常写入输出当前布局。format 8以前的中位数等不兼容结构仍需重新生成，不能把中位数当作均值迁移：显式合并报错，短校准成功发布后才原子替换；失败或关闭校准不覆盖旧文件。固定B2允许使用无缓存模型，Auto B2仍要求可用成本条件。旧`.cprof`及独立Stage1 TOML不受此Stage2格式规则影响。

同条件同完整参数的独立测量按次数合并均值和标准差，相同/子集快照不重复计数，不同参数保留多行。运行和正式调优分别累计；估算不覆盖实测。写入使用独占锁，临时文件重新校验后原子替换。

合并：`--tune ecm --tune-merge A.toml --tune-merge B.toml --tune-ntt-profile NTT.toml --tune-file combined.toml`。不运行曲线或查询GPU。只有NTT的文件尚无完整引擎锚点，需要短校准或ECM tune。

显式tune的原始日志、计划、回执、二进制SHA256与可用构建清单位于工作目录的`data/experiments/`；正常运行使用现有结果JSONL保存采集回执，matrix行保留来源版本和UTC更新时间；二进制SHA只留在诊断证据。TOML保留统计、必要组件均值及去重身份，不展开运行/导入历史。队列进度身份不包含性能文件、D或承载选型，增量tune不重置已完成编号；Stage1成本输入身份另行保护。

## 验证范围与限制

当前格式、统计及GPU流程检查见[当前验证范围](#当前验证范围)。非负拟合仅在合成工作量上验证模型恢复与留出资格，不证明GPU预测误差达到8%。概率核心377组参考对拍保持不变，最大绝对误差3.44×10⁻¹⁵，证据 `data/experiments/stage2_portable_tune_20261010/probability_01/` 仅用于概率组件。

稀疏跨位宽/跨B2预测已出现显著误差，早期冻结功能样本最高约343%和211%，证据在`data/experiments/stage2_portable_tune_20261010/gpu_01/`。局部D搜索可能错过不能经相邻显著改善到达的真实最优项；功能检查不能作为8%收敛或全表最优D证明。非负拟合已有资格检查，仍待孤立组件校准及更多生产B2/高位宽独立排名验证。

G=1与baby预算回退缺完整联合生命周期，成本可估算但原生准入不放行。host fold/frontier采用受限arena的保守上界；部分借用设备NTT的cap失败回退尚未完成执行验证。见 [内存](MEMORY.md)及 [TODO](../TODO.md)。旧`.cprof`接口保持独立资格合同，不套用上述模型。

共享候选表和NTT区间重启已接入当前源码。冻结历史数据不作为当前候选域或搜索的性能验收；新实现还需相同输入的完整矩阵及配对计时，见[TODO](../TODO.md)。

## 代码入口

- [ecm_stage2_d_catalog.h:14](../../src/core/ecm_stage2_d_catalog.h#L14)：唯一D字面量表；`entries`、`degree`提供派生P，编译期检查范围、偶数、唯一和顺序。
- [ecm_cuda_stage2.cu:8621](../../src/cuda/ecm_cuda_stage2.cu#L8621)：原生默认成本候选。
- [ecm_stage2_portable_driver.inl:378](../../src/core/ecm_stage2_portable_driver.inl#L378)：默认几何模型；`portable_select`负责实测预筛、成本排名及准入，`run_portable_tune`负责局部搜索、测量与发布。[运行采集:133](../../src/core/ecm_stage2_portable_driver.inl#L133)负责采集与可选缓存写回。
- [ecm_stage2_portable_model.h](../../src/core/ecm_stage2_portable_model.h#L194)：`Estimator::fitted`负责非负拟合及留一资格；同文件提供工作量、观察匹配、插值与误差余量。
- [ecm_stage2_tune_portable.h](../../src/core/ecm_stage2_tune_portable.h#L22)：`LocalSearch`负责邻域及噪声停止；`Profile`负责矩阵解析、排序与均值累计。
- [ecm_stage1_cost_csv.h](../../src/core/ecm_stage1_cost_csv.h)、[ecm_stage2_probability.h](../../src/core/ecm_stage2_probability.h)：容器模型及概率。
- [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp)：配置、验证与队列。
- [配置参考](../ECM_INI_REFERENCE.md)、[使用](../usage/STAGE2.md)、[工具](../../tools/bench/README_STAGE2_CARRIER_PLAN.md)。
