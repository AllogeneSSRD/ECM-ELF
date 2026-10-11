# Auto B2 与 tune

## 目标与生产入口

Auto B2 选择连续生成、处理新曲线的单位时间收益，计入 Stage1 成本，即使当前读取已完成的 save。固定 B2 时，性能文件选择 D 和合法梅森承载；最终 B2=0 且启用 Auto B2 时联合选择 B2、D、承载。

默认性能文件为 INI 所在目录的 `stage2_ecm_tune.toml`，可由 `stage2_tune_profile` 或 `--tune-profile FILE.toml`覆盖。`--auto-b2` 对应 `stage2_auto_b2=1`。显式同时提供非零 `--b2` 与 `--auto-b2` 报冲突；任务或 INI 已有非零 B2 时保持固定值。

性能数据可跨实际位宽、B1 复用，允许插值、外推。复用从不要求 B1 相等；实际 B1 仍用于区间、成功概率、Stage1 成本和真实运算。模型估算与实测在输出中分别标识。

`stage2_tune_ignore` 默认 `gpu,driver,cuda,backend,environment`，可选项另含 `memory`；空值要求这些条件匹配。默认 memory 按 fold/frontier 执行路径匹配，不要求预算数值相等。忽略条件不会取消 save、算术、整除关系和当前显存检查，也不表示其他设备的秒数已经在本机验证。

非零 `--d` 锁定 D。`--carrier-exponent 0` 锁定普通运算，非零 p 锁定承载。未锁定时，worktodo `ECMSTAGE2=1,2,p,-1,...`先按已知因子重建实际 N，核对 save，再验证 **N 整除 2ᵖ−1**，比较普通与承载。只读 save 而没有指数来源时不猜测承载指数。

## 收益与搜索

P=φ(D)/2，I=⌊B2/D⌋+2，G=⌈I/P⌉。搜索目标为：

**收益 = Pr(实际 B1、B2，目标素因子位数)/(T1 + R·T2_guard)**。

- T1 是已摊销的 Stage1 秒/曲线，不是 save 读取或 tune 点生成时间。
- R 对应正的 `stage2_ratio_adjust`。
- T2_guard = 预测引擎时间·(1+相对误差余量)+2·MAD；包含 shape、init、main，排除 Stage1 和父进程启动。
- 默认目标因子位数从 B1 推荐式反推并受实际 N 约束；`stage2_target_factor_bits`可覆盖。因子规模不能直接用余因子总位宽代替。
- 原生 Dickman/半光滑概率与 `tools/ecm_prob` 对拍；当前 poly 不计入未实现的 Brent–Suyama 收益。

默认 B1<B2≤2.6×10¹²；`stage2_auto_min_b2`、`stage2_auto_max_b2`及对应 CLI 可修改。候选包含区间内49个对数间距点及已记录B2；D包含14项表、实际默认值及本位宽已有实测项，并受下文的实测/拟合资格约束。搜索不受tune网格边界限制，但有限候选搜索不证明连续范围全局最优。

先按收益或固定 B2 成本排序，再逐项要求原生计划 `valid=true`、`finished=true`、`initial_free_snapshot_fits=true`。规划后 fold/frontier 路径变化时不借用原路径排名。实际执行仍检查 free/headroom及处理物理分配失败。无可用成本时固定 B2 返回原因并保留原生选型；Auto B2 明确失败，队列保持未完成。

### Stage1 成本

优先级：正的 `stage1_seconds_per_curve` → `stage1_cost_csv` → `stage1_tune_profile`。秒/曲线不再除以 `stage1_batch`。

CSV 必填 `b1,mhz,tpi,curves,seconds_per_curve`，另提供 `container_bits` 或 `target_bits`。可选 `tpb`（默认128）、`exponent`（空值表示未指定，否则 `lcm|choose12`）。空耗时表示缺失测量，同 TPI 不接受混杂提交批量/TPB。参考表为 [stage1_cost_4060_2000mhz.csv](../../config/stage1_cost_4060_2000mhz.csv)，不自动启用。

用户表 N 明确是容器位宽。实际目标按 **target_bits+6** 选择 PARAM0 容器及 TPI；8192-bit容器覆盖实际位数≤8186，实际8192 bits进入9216-bit/TPI32。表的条件是4060 Laptop、2000 MHz、B1=110×10⁶，TPI8提交768曲线、TPI16提交384曲线，统一提交2 blocks/SM；不等于全部容器同时驻留2 blocks/SM。算法、指数模式及独立核验未给出，来源是用户报告。

初始近似 T1∝B1·container_bits²/f。归一化系数在同 TPI 内插值；范围外邻近锚点外推，无同 TPI 数据时允许跨档外推并标注。`stage1_cost_mhz=0`采用锚点参考频率，正值使用指定目标频率，不读取瞬时时钟。跨 TPI、高 B1 与不同功耗尚未独立校准。

完整 Stage1 TOML采用邻近容器锚点，线性缩放 B1、平方缩放容器；保留 batch、模数类型、指数模式匹配，不要求 B1 相等。预计算工具见 [Stage1成本工具](../../tools/bench/README_STAGE2_CARRIER_PLAN.md)。

## 完整 Stage2 tune

`--tune ecm --tune-level 1..10 --tune-file FILE.toml`保留位宽和B2档位，从默认模型推荐D开始，比较邻档，仅沿实测更快的方向继续。没有轻量完整引擎种子扫描；缺失NTT组件时补测少数NTT。默认 `stage2_tune_budget_seconds=1800`秒；初次设备查询在预算外，点准备、预热、正式测量在预算内。软预算到达后不发起新测量，正在执行的测量结束并保留有效结果。

每级额外包含512 bits。位宽 a:b表示b、2b、…、ab；B2 a:r从上界 U=2.6×10¹²按 r 倍递减取 a 项，再升序排列。非零 `--auto-max-b2`可改变 U。

- level1：位宽3:4096，D4项，B2 3:10，正式重复2次。
- level2：位宽4:3072，D6项，B2 3:10，正式重复2次。
- level3：位宽5:2048，D8项，B2 4:10，正式重复2次。
- level4：位宽5:2048，D10项，B2 4:10，正式重复2次。
- level5：位宽5:2048，D10项，B2 6:5，正式重复3次。
- level6、7：位宽8:1536，D12项，B2 6:5，正式重复3次。
- level8：位宽10:1024，D12项，B2 8:5，正式重复3次。
- level9：位宽15:1024，D12项，B2 8:5，正式重复5次。
- level10：位宽15:1024，D14项，B2 8:5，正式重复5次。

D范围在 `30030,60060,120120,180180,210210,360360,570570,690690,810810,1021020,1141140,1381380,1711710,2282280` 中均匀选取并含端点。P随D确定，不是独立参数。默认网格额外加入实际默认D；显式`--tune-d`保持用户给出的搜索范围，起点取距离默认D最近的一项。

每个“位宽×B2×算术路径”使用离散下降：先测起点及相邻前后项。候选A替代B要求 **T_A+2·MAD_A < 0.98·T_B−2·MAD_B**。有显著收益时移向更快邻项，只补测新暴露的邻项；否则停止。端点、显存拒绝、G=1或超过批次数限制会关闭相应方向。有限邻域搜索不保证全局最优，也不假设NTT拐点附近的目标函数平滑。

达到正式重复要求的已有同形状测量可复用；有成对默认基准且已有显著实测赢家时，从该赢家继续邻域搜索，否则从默认模型开始。预热和失败结果不成为性能摘要。预算到期后停止发起新测量，保留已完成结果。没有实测的高B2形状不再根据P=24远距离外推时间直接跳过。默认保护和现场显存准入均有效；显式窄D表可以排除默认D，得到的局部最优项不自动成为正式选型替代。

`--tune-exponents bits,...`在此入口表示**实际位数**；`--tune-d`、`--tune-b2`、`--tune-repeats`覆盖对应网格。默认构造精确位宽的 GMP probable prime，sigma26、B1=20，不称为已证明素数。`--tune-save FILE`使用验证后的真实点并替代位宽列表；比较承载须同时给 `--tune-carrier-exponent p`并检查整除。旧 `--tune-tail-samples`只接受0。

完整测量先预热，再正式采样，要求mandatory自检、GMP检查、clean通过、bad=0、hits=0。预算中途结束保留已完成正式重复及真实次数；仅完成预热时不生成摘要。再次tune复用已有候选并继续未完成邻域。TOML保留访问候选的最终测量/估算，不展开全表估算。

控制台保留 `ecm_tune_window`、`ecm_tune_progress`、`ecm_tune_window_done` 和完成/失败原因。子进程阶段、NTT统计及内部自检不转发到控制台，完整诊断仍保存在每次测量的证据日志；正常Stage2的控制台与日志规则不受此限制。

### 默认选型与实测保护

默认模型在最大NTT长度低于2²⁷时保留原生D模型；达到后端2²⁷上限后，只比较仍驻留、最大NTT长度不增加的更大D。通用归约还要求默认巨步数至少16P，以摊销较贵的初始化；梅森归约不采用此附加门槛。归一化成本包含G树、fold、giant及初始化，避免在同一大NTT平台内过早减小P；显存准入仍使用完整生命周期。该几何模型不是绝对秒数预测；性能依据和范围见[Stage2性能](../performance/STAGE2.md)。

固定B2的tune替代需要默认D与候选D具有同实际位宽、同B2、同算术/执行路径的完整曲线实测，并满足上述中位数/MAD改善条件。B1不作为匹配门槛。缺少配对基准时保留默认D；稀疏估算不会覆盖默认结果。未锁定承载时，承载候选也要与普通默认路径比较。实际输入、当前显存和整除关系仍须核对。

固定B2先按实测摘要过滤没有显著优势的D，再对剩余候选作完整原生规划及严格执行路径匹配。合格项直接按实测中位数+2×MAD排名，输出模型`paired_measured_d_v1`；这一步不需要跨D拟合。预筛不是显存准入，最终候选仍要通过当前free和完整生命周期检查。显式锁D及Auto B2保留各自的成本预测流程。

Auto B2仍使用概率与Stage1成本联合排名。无合格跨D拟合或对应实测时，每个B2仅使用默认D，不用远距离外推去改变D；B2秒数外推仍保留不确定性，不宣称已通过生产收益验证。

### NTT 与配对阶段组合

特征分别计数 NTT形状/调用、系数归约、传输、baby/giant点和积累。多G驻留请求按原生 packing 分块；giant区分chain、ladder、短尾。普通归约用limb²、梅森归约用limb初始工作量。NTT缺项以邻近长度/批量的N·log₂N比例估计，不假设NTT本身平方增长。

完整曲线的互斥阶段提供秒数尺度。giant、G树、fold嵌套计时仅用于权重，不与包含它们的主循环重复相加。同算术/位宽/D/内存形状的已测B2区间可插值；区间外走组件估算。

跨D成本模型可用非负拟合分别估计固定开销、NTT工作量、归约工作量和点运算工作量。需要同实际位宽/算术/分块路径至少六个完整曲线摘要、三个D、两个B2，逐项留一验证通过`stage2_tune_error_limit`（默认8%）；P<512的探测不参与拟合。目标P仅允许处于已测P范围的0.5～2倍内。

数据不足或验证不通过时保留稀疏组件估算：多项式阶段使用NTT0.55、归约0.40、传输0.03、调用0.02的初始权重，固定主循环准备不再随giant数量放大。其初始误差余量为25%，随形状距离增加；该估算不是硬件定律，不能作为固定B2的D替代证据。同位宽/D/路径的已测B2区间仍可插值。

`validation_max_relative_error=|预测−实测中位数|/实测中位数`来自加入该样本之前的预测；无独立证据时不写零。拟合的留一误差与该字段区分，均不以训练残差冒充独立误差。`independently_validated`不等于整个网格已经通过门限，也不代替当前输入的算术检查。

## 启动短校准

默认 `stage2_short_calibration=1`，独立预算10秒。没有合适文件、必要NTT组件缺失，或当前worktodo范围完全不在可复用已测位宽覆盖内时触发；部分覆盖时不因首项宽度外推反复校准。直接读save以该目标作为单项范围。

使用已验证save中的N、sigma及仿射点，做D=210、probe_B1=min(保存B1,20)、probe_B2=probe_B1+42000的完整引擎成本探测及少数NTT测量；所需承载缺摘要时补测。探测仅复用曲线上的有效点，不声称重新执行了probe_B1的Stage1；原B1与探测B1另存证据，用户save及生产B1/B2不变。这样生产B1增大不会扩大启动探测的巨步范围。短校准发现因子且全部算术检查通过时，保留证据、跳过该性能样本并继续正常任务；不将因子命中写入无因子性能表。算术检查失败仍中止。结果供后续tune利用，不消费队列，不代表生产B2精度验收；缺少完整引擎锚点时，固定B2可回退原生规划，Auto B2仍需可用成本。plan-only也可能触发校准并更新性能文件。`--short-calibration 0`可关闭；关闭后固定B2缺文件可走原生规划，Auto B2仍要求可用成本和T1。

[复合数启动回归工具](../../tools/test/test_stage2_startup_composite_gpu.py)验证M2003/sigma27/B1=20保存点的因子4007探测不进入性能表、正常Stage2仍完成；另对经独立GMP核对的真实M521/B1=10e6保存点验证probe_B2保持42020。[当前功能证据](../../data/experiments/stage2_n_scaling_20261011_2400/calibration_final_gate/regression_report.json)与[全位宽性能扫描](../performance/STAGE2.md)分别保留，功能回归计时不混入性能矩阵。

## NTT tune

`--tune ntt --length-log2 a:b --tune-slices s,... --tune-repeats n --tune-file FILE.toml`测长度和批量，也可用等级预设。共享软预算和增量摘要规则。每秒域卷积数=batch/median，每秒批调用数=1/median，不是ECM曲线吞吐，不含大整数carry、归约或完整Stage2。

GMP参考验证每个输出项；摘要保留中位数、MAD、重复数及检查量。内存不足可跳过，验证失败报错并保留证据。TOML用格式7参数表，显式JSONL调试输出保留逐次接口。

### NTT 性能摘要表

`ntt.<condition>.table.<n>`按测量条件组织最终摘要。`[ntt.0.table.0]`表示关联`[condition.0]`的第一个表组，0不是GPU设备号、NTT长度或调优等级。显卡由条件中的UUID标识。表头公共字段适用于所有行，`matrix`首行给出列名，后续行按列保存每个“长度×批量”的最终结果；同一字段可能在公共部分，也可能成为列。

一次基准批调用包含两个正向NTT、有限域点乘与缩放、一个逆向NTT，运算域为Goldilocks素数域。GPU事件只包围这段计算，输入生成、准备、核验在计时外；不包含生产多项式的packing、carry、模N归约、传输和整条ECM曲线。

**规模与时间字段：**

- `length=L`：每个NTT的有限域元素数，L为2的幂；不是ECM模数位数。
- `log2_length=k`：k=log₂L。
- `batch=b`：一次批调用中的独立卷积数，各切片使用相同长度和执行形状。
- `median_seconds=t`：正式批调用GPU时间的中位数，单位秒。
- `mad_seconds`：正式时间相对中位数的绝对偏差中位数，单位秒。
- `min_seconds`、`max_seconds`：正式时间的最小、最大值。
- `repeats`：正式重复次数，另执行一次预热。
- `batch_iter_per_s=1/t`：每秒批调用数。
- `conv_iter_per_s=b/t`：每秒独立有限域卷积数，不是ECM curves/s。

性能TOML不保存`setup_seconds`、`warmup_seconds`、`verification_seconds`。这三项不参与成本选型；格式7文件读入及新摘要写入时剔除，正常重新发布后清理已有字段。底层诊断JSONL保留测量证据，原始实验文件不改写。

**来源与正确性字段：**

- `condition`：关联的条件编号；GPU、驱动、CUDA、后端和环境复用策略见本文前述规则。
- `source="measured"`、`status="measured"`：结果来自完成并通过核验的实测。
- `unit="field_convolution"`：计时单位为有限域卷积。
- `bad=0`：结果不匹配数量为零。
- `verified_words_per_sample=L·b`：每次核验的输出元素数，加载时核对其等式。
- `reference_kind="gmp_3x3_distinct_constant_v1"`：GMP生成两个各含3个非零系数的多项式卷积参考，各batch切片常数项不同；核对所有输出，包括零尾部和切片寻址。这是规定输入的核验，不代表完整随机输入覆盖。

**容量字段，均为bytes：**

- `external_arrays_bytes=16Lb+40b+8`：两个64-bit输入数组、每切片5个参考元素及一个错误计数器。
- `arena_payload_bytes`：本次基准的NTT arena实际记录载荷。
- `payload_bytes`：准入用的规划载荷，等于external数组、规划的fuse基础缓冲及缓存表之和。arena字段不覆盖所有fuse基础分配，不能用external+arena机械替代该公式。

这些是独立NTT基准的容量，不是生产Stage2的进程显存峰值；CUDA上下文、驱动及生产其他组件另计。生产仍使用当前原生规划和设备可用显存，性能表的容量记录不覆盖其检查。

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

模型以`(length,batch)`查询`median_seconds`；满足条件复用策略的同形状记录取最低中位数。缺少精确形状时，从邻近测量点估算：

**t(L,b)≈t(L₀,b₀)·(b/b₀)·[Llog₂L/(L₀log₂L₀)]。**

短长度 L≤2¹⁹ 且有同长度测量时，优先在 batch 轴上分段线性插值 t(L,b)，保留小批次的启动成本；小于首个 batch 使用首个实测时间，大于末个 batch 按末点每卷积时间外推。没有同长度短表或 L>2¹⁹ 时才使用上述邻近长度公式。表外均为估算，不能声明实测吞吐饱和点。只有NTT表、没有完整ECM实测锚点时，仍不足以给出完整Stage2秒数，需要ECM tune或启动短校准。

每个D决定P=φ(D)/2及树结构。模型从实际packing形状和分块规则得到各阶段的NTT长度、batch和调用次数，累计调用次数×批调用成本，再结合归约、传输、点生成及完整引擎阶段锚点预测总时间。当前多项式外推中NTT权重0.55是初始模型权重，不代表NTT实测占总时间55%。精确ECM点或同形状B2区间插值可直接使用完整曲线数据，不必由NTT重新推算总时间。

结果影响固定B2的D/P与合法承载选择、Auto B2联合搜索，以及ECM tune的中心D和相邻窗口。D/P改变后，所需NTT形状和调用分布随之改变。实际NTT长度仍由操作数/算术位宽、packing及精确性约束决定。默认 batch 由任务分块预算决定；实验性 stage2_short_ntt_batch_tune 可在预算允许的分块内按短 NTT 表选择 batch。

当前`tune ntt`不搜索tile、radix或协作开关；表中这些字段记录测量时的执行形状。成本模型没有把它们作为内核候选重新配置，容量和诊断字段也不直接参与时间排名。自动搜索并部署NTT内核配置仍列为待实现事项。

### 短 NTT 批处理选型（实验选项）

`stage2_short_ntt_batch_tune=false` 为默认；命令行 `--short-ntt-batch-tune 1` 可覆盖。输入是 `stage2_tune_profile` 内满足 `stage2_tune_ignore` 的已核验 NTT 摘要。仅 L≤2¹⁹ 且同长度至少有两个实测 batch 才参与选择；缺少资料、较长 NTT 和单 slice 请求保留原分块。

设独立请求数 R、候选 batch b、单批估计时间 t(L,b)，比较：

**T_NTT(R,b)=⌊R/b⌋·t(L,b)+t(L,R mod b)，t(L,0)=0。**

候选包含原预算分块以及不大于该分块的实测 batch。预计卷积时间须比原分块降低超过10%才改变；不会将不同树层或互相依赖的请求合并。运行、请求签名、NTT/S4生命周期规划和组件模型使用相同选择函数。纯 NTT 表不含 packing、carry、系数归约、上传/读回及额外分块同步，所以这不是完整 Stage2 性能保证。

关闭实验开关时保留原队列续跑身份和默认条件字段；开启时单独记录策略身份。策略数据 hash 改变后，不沿用原策略的精确测量或同形状B2插值，相关预测保持估算口径。

NTT 校准和完整 ECM 校准保持独立：完整 ECM/启动短校准使用原分块，存入实际条件，不伪装成启用新策略的测量。运行日志给出策略摘要和参与数据的 hash。[固定 D 完整曲线验证](../performance/STAGE2.md#短-ntt-分块选择)中，当前策略未显示稳定收益；7995-bit通用余因子两档B2均慢约2.1%，因此保持默认关闭。后续选型需要计入每分块的归约、准备、检查和同步成本，不能仅用独立NTT的Conv/s决定分块。

NTT 参数表发布时按 `(log2_length,length,batch,tile)` 数值升序排列，列顺序与行排序分别维护。现有生成表可用 `tools/bench/format_stage2_tune.py <profile> --in-place` 重排；此工具仅移动行，保留测量值、其他节和列结构。

### NTT 摘要代码依据

- [ecm_stage2_tune.cuh:91](../../src/cuda/ecm_stage2_tune.cuh#L91)：基准载荷；[计时与核验:119](../../src/cuda/ecm_stage2_tune.cuh#L119)：两个正向、一个逆向与GPU事件；[字段输出:141](../../src/cuda/ecm_stage2_tune.cuh#L141)。
- [ecm_stage2_tune_portable.h](../../src/core/ecm_stage2_tune_portable.h)：`validate_ntt_summary`、`strip_ntt_auxiliary_timing`、`Profile::update_ntt/load/serialize`，摘要检查、时间字段剔除及参数表发布。
- [ntt_runtime.cuh:1418](../../src/cuda/stage2/ntt_runtime.cuh#L1418)：外层级数规划；[形状选择:1488](../../src/cuda/stage2/ntt_runtime.cuh#L1488)；[正向与逆向:1678](../../src/cuda/stage2/ntt_runtime.cuh#L1678)；[packing:3224](../../src/cuda/stage2/ntt_runtime.cuh#L3224)。
- [ntt_coop_outer.cuh:15](../../src/cuda/stage2/ntt_coop_outer.cuh#L15)：block协作外层内核。
- [ecm_stage2_portable_model.h:84](../../src/core/ecm_stage2_portable_model.h#L84)：NTT成本查询/外推；[工作量:21](../../src/core/ecm_stage2_portable_model.h#L21)；[非负拟合:117](../../src/core/ecm_stage2_portable_model.h#L117)；[阶段估算:150](../../src/core/ecm_stage2_portable_model.h#L150)。

## 汇集预计算结果

格式7按位宽组织`width.<bits>.table.<n>`，NTT按条件组织`ntt.<condition>.table.<n>`；公共字段只写一次，`matrix`首行声明列名，后续行保存每组参数的最终摘要；表头和数值按相同列宽对齐。位宽区间在注释中注明；每行只代表校准节点上的一组参数，节点间值仍是模型估算，不冒充整个区间实测。表的列和行是结果结构，不是逐次采样数组；不含运行、导入历史或`[sample.*]`。

来源、条件、离散度、核验量和独立误差仍保存，读回恢复原命名字段并执行完整校验。重复列、列数不符、嵌套数组、错误位宽/条件、坏核验及重复候选均拒绝。

开发版不兼容旧TOML。显式合并旧文件报错；tune输出或默认短校准遇到旧版本时丢弃其内存中的性能数据，重新校准，成功发布后原子替换文件。校准失败不覆盖旧目标；关闭短校准后读取旧格式仍报错。旧`.cprof`及独立Stage1 TOML不受此Stage2格式规则影响。

同条件同候选仅在正式中位数**严格更低**时替换；相等或更慢保持原记录，不同条件分别保存。估算不覆盖实测；模型仅在证据变化时更新。复用忽略项不把不同条件的秒数混入同一原始最快记录。写入用独占锁，临时文件重新校验后原子替换。

合并：`--tune ecm --tune-merge A.toml --tune-merge B.toml --tune-ntt-profile NTT.toml --tune-file combined.toml`。不运行曲线或查询GPU。只有NTT的文件尚无完整引擎锚点，需要短校准或ECM tune。

原始日志、计划、回执、二进制SHA256与可用构建清单位于工作目录的`data/experiments/`，不写入性能TOML。队列身份用性能文件路径，增量tune不重置已完成编号；Stage1成本输入身份另行保护。

## 验证范围与限制

以下功能检查对应sm89/CUDA13.3.73/GMP Zen3冻结构建，二进制SHA256 `50171511de1dbc9b998e7a352173edbac4bc70cb181bb6c81da442e542d4e659`，4060 Laptop设备0。程序未修改时钟/功耗；这些短检查不是速度提升报告：

- CPU核对10级网格、105组相邻窗口、离散下降/噪声停止、最快更新、5类异常profile、6类坏参数表、CSV推断和旧格式拒绝。20行同组合成摘要展开8773字节，合并公共字段及对齐后2211字节，二次读回值一致；不能外推所有生产文件的压缩率。
- 非负拟合在合成完整工作量数据上恢复固定项与变动项；故意破坏一个留出点后拒绝资格。同形状检查确认B1变化不阻止精确复用，buffers或giant分块变化不能标为精确点。这是数学/策略检查，不证明实际GPU预测误差达到8%。
- GPU22项流程通过：512/4423 bits、D210、B2=60000、正式重复2次的驻留/fold预算0测量，跨B1=20→1000、短校准及关闭、NTT摘要/合并、旧文件拒绝与重新生成、一条4423-bit完整曲线、部分覆盖队列及M503余因子318 bits承载推断/锁定；Auto B2覆盖锁定与未锁定D。无文件、关闭短校准、未锁定D时保留默认规划，不要求返回记录含缓存字段。512-bit、B2=260000的显式五档D表验证连续邻域搜索。父日志级别为batches时，tune仍仅转发进度，内部阶段日志保留在证据文件。
- 48条冻结NTT记录验证四字段数值排序、测量值不变及Python重排幂等。
- 固定B2的27格核对实测预筛前后D/P/排名、原生请求程序及giant路径一致；仅规划查询合计111.093→27.424秒。该查询不执行曲线，完整生产时间另见[性能](../performance/STAGE2.md)。
- 无性能文件的27格默认D/P/I另与完整默认扫描核对一致；最终摘要导入工具通过14类损坏证据拒绝，保留全部冻结输入。
- 证据：`data/experiments/stage2_tune_redesign_20261011/{core_fast,gpu_final_04,sort_production_final,planning_final02,default_plans_final02,import_gate_final}/`。首轮GPU检查的错误CLI参数已修正，失败文件在`gpu_final/`保留；没有放宽实现或算术门限。进程墙钟独立记录，不与引擎阶段相加。
- 未变化的概率核心有377组参考对拍，最大绝对误差3.44×10⁻¹⁵，证据为`data/experiments/stage2_portable_tune_20261010/probability_01/`，仅作为概率组件核对。

稀疏跨位宽/跨B2预测已出现显著误差，早期冻结功能样本最高约343%和211%，证据在`data/experiments/stage2_portable_tune_20261010/gpu_01/`。局部D搜索可能错过不能经相邻显著改善到达的真实最优项；功能检查不能作为8%收敛或全表最优D证明。非负拟合已有资格检查，仍待孤立组件校准及更多生产B2/高位宽独立排名验证。

G=1与baby预算回退缺完整联合生命周期，成本可估算但原生准入不放行。host fold/frontier采用受限arena的保守上界；部分借用设备NTT的cap失败回退尚未完成执行验证。见 [内存](MEMORY.md)及 [TODO](../TODO.md)。旧`.cprof`接口保持独立资格合同，不套用上述模型。

## 代码入口

- [ecm_stage2_portable_driver.inl](../../src/core/ecm_stage2_portable_driver.inl#L138)：默认几何模型；`portable_select`负责实测预筛、成本排名及准入，`run_portable_tune`负责局部搜索、测量与发布。
- [ecm_stage2_portable_model.h](../../src/core/ecm_stage2_portable_model.h#L117)：`Estimator::fitted`负责非负拟合及留一资格；同文件提供工作量、观察匹配、插值与误差余量。
- [ecm_stage2_tune_portable.h](../../src/core/ecm_stage2_tune_portable.h#L22)：`LocalSearch`负责邻域及噪声停止；`Profile`负责矩阵解析、排序与最快更新。
- [ecm_stage1_cost_csv.h](../../src/core/ecm_stage1_cost_csv.h)、[ecm_stage2_probability.h](../../src/core/ecm_stage2_probability.h)：容器模型及概率。
- [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp)：配置、验证与队列。
- [配置参考](../ECM_INI_REFERENCE.md)、[使用](../usage/STAGE2.md)、[工具](../../tools/bench/README_STAGE2_CARRIER_PLAN.md)。
