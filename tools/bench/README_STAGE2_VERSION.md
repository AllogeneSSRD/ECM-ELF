# 数字版本 Stage2 全范围复测

`bench_stage2_version.py` 复用已验收的 Edwards / Prime95 精确输入矩阵，测 GPU1 的普通余因子与梅森承载。默认 13 个输入、两档实际 B2、两条 GPU 路径、各三次，共 156 次。CPU 和 Montgomery 不重跑。当前结果见 [Stage2 性能](../../docs/performance/STAGE2.md)，输入与历史参考见 [参考页](../../docs/performance/EDWARDS_STAGE2.md)。

## 构建与测量

使用正式 local 构建入口，测量前完成编译；不要同时运行本任务的 GPU 诊断：

以下示例记录已完成的 0.38.4 研究。重建该版本须使用对应版本源码；用当前源码开展新研究时，应同时更新目录与 `--version`，以程序实际报告的版本为准。

```powershell
tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/local_stage2_0_38_4 `
  -Arch sm_89 -Engine production -SplitCompile 8
python tools/bench/bench_stage2_version.py `
  --exe build_cuda_cmake/local_stage2_0_38_4/ecm_cuda_stage2.exe `
  --reference data/experiments/edwards_stage2_reference_20261017/matrix_01 `
  --version 0.38.4 --output data/experiments/stage2_v0_38_4_20261018/full_matrix
```

新输出目录必须不存在；参考目录须含已完成准备的 `plan.json`、13 份已独立验证的 save、26 个 Edwards Prime95 条件。工具复制 EXE/GMP、输入及采集器，核对精确余因子整除、save 状态和校验。程序在每次结果中报告的数字版本必须与 `--version` 一致。新增研究元数据和图注使用版本号，历史未记录数字版本的程序明确标注未知，不补造版本。

设备固定为 GPU1，操作者须确认它仍为 RTX 4060 Laptop。工具记录实际设备名称，不改变时钟、功耗、优先级或 GPU0 的工作。预算为 arena6300/fold owner640/batch256 MiB；D 自动选择，短标定/runtime tune/自动承载/relocation/NTT 计数关闭。每轮遍历全部精确条件，偶数轮交换普通/承载顺序；每次为独立进程，没有丢弃的预热或首轮。所有调用保留正常算术检查。

每次调用前后查询选中 GPU UUID 的计算进程：只允许本输出目录中的程序；其他 GPU 的任务不影响该检查。外部任务或无法识别的进程会中止采集，查询快照和已有数据保留。`--allow-shared-device` 仅用于明确接受共享设备的独立研究，记录在协议中。前后快照不证明调用期间每一瞬间都没有其他进程。

主计时为 `stage2_full_wall.total=init+main`，包括引擎内必需检查和 GCD，排除 Stage1、进程启动、外层选 D 及计时前预检。进程墙钟另存，250 ms 轮询包含最多一周期的退出观察延迟。NVML 观测整张选中显卡，不代表进程峰值或纯内核功耗。

## 续跑与审计

```powershell
python tools/bench/bench_stage2_version.py --resume --version 0.38.4 `
  --output data/experiments/stage2_v0_38_4_20261018/full_matrix
python tools/test/check_stage2_version.py `
  --study data/experiments/stage2_v0_38_4_20261018/full_matrix `
  --output data/experiments/stage2_v0_38_4_20261018/full_matrix/audit.json
```

计划在每条接受后原子保存。`--max-runs N` 可在完成 N 条新调用后正常停下，样本保留且矩阵仍未完成；续跑跳过已接受 ID。采集器改变、已有失败、存在未入账调用日志时拒绝续跑，须先调查并保留证据，不能覆盖原文件后补成成功样本。改变版本、输入或协议应使用新目录。

正式审计要求 156 个唯一 ID、52 格各三次，重新核对输入/save、结果版本和身份、实际 B2/D/P、φ(D)/2、完整区间点数与 G 批数、NF 状态、原始计时守恒、强制 GMP 检查、分项记录和调用设置。提前出因子的运行保留，但不满足完整 NF 时间矩阵。该审计不声称提供每个叶值的独立 oracle；正常 GMP 抽样与独立末点验证各自有明确范围。

获准只跑一次的筛查可在采集时指定`--repeats 1`，审计须显式指定`--expected-repeats 1`；仍要求完整52格、全部身份/数学检查和已声明的前后进程快照。审计默认仍为三次，不把一次结果当作三次中位数或波动估计。不同样本数的D搜索版本对照见[有界搜索验证](README_STAGE2_D_CATALOG.md#有界搜索实现验证)。

## 汇总、绘图与公开副本

```powershell
python tools/bench/plot_stage2_version.py `
  --study data/experiments/stage2_v0_38_4_20261018/full_matrix `
  --output data/benchmarks/stage2_v0_38_4_20261018/final `
  --figures data/figures/stage2_v0_38_4_20261018/final `
  --publish docs/performance/attachments
```

需要 Matplotlib 和 NumPy。新统计/图目录必须不存在；绘图先严格审计当前矩阵，逐格核对旧 Edwards 参考的身份，再生成 CSV/JSON 与 PNG/SVG。绝对耗时和 NTT 模块显存图以相同数据同时生成 `-log`、`-linear` 两版（含 PNG/SVG），线性纵轴从0开始；百分比图保持线性坐标。当前 GPU 使用中位数及 min/max，版本比较阴影组合两侧观测范围，不是置信区间。init/main 按逐调用比例取均值；NTT 模块峰单列，不加组件峰、不当作进程峰。

同一工具重绘已验收历史参考：Prime95 31.4b05 全部两曲线族、32 条 libecm 样本、CPU GW 0.33.0 的 36 次实测，以及 Montgomery 55 W 及两种频率设置下的三份 9 位宽×3 B2×3 次矩阵。不将缺测 CPU 单元插值，不按频率、B2 或线程数归一化。历史参考路径集中在绘图脚本顶部，原始研究不改写。`--references-only` 可在等待 GPU 时单独重绘这些参考，不能生成当前版本完整性能结论。

`--publish` 只复制选定 PNG 到指定 attachments 目录；原始结果、账户字段、运行校验值、EXE、遥测和全部数据只在忽略的 data 目录。公开提交只包含工具、文档和审查过的图。

## 大档B2的固定几何S4对照

`bench_stage2_large_matrix.py`复用自动点后端矩阵的13份精确Edwards save、有限链长和进程白名单，从Edwards/Prime95参考中按完整N及B1取得名义2.6e11档的实际B2。默认每格一次；普通与承载共26格。显式请求baby/giant/seed coop32及S4 coop32，记录低位／承载的实际回退；关闭warmup、学习、短校准、凑整、relocation及尾块平衡。这是归约实验的固定后端矩阵，不等于生产auto选型成绩。

首次`--exe <path> --output <new-directory>`冻结二进制、GMP、目标文件哈希和构建输入归档，按当前基础模型选D/布局一次。随后用`--baseline <completed-directory> --exe <candidate> --kernels 2,3 --output <new-directory>`冻结相同D/P/layout/chain进行交替顺序A/B；逐格核对NTT分布、G根摘要、点块策略及S4/GMP工作数。显著回退（同时>3%和50ms）保留首轮并将双方补到三次。`--wide-repeats 3`可将四个受影响的高位普通格全部补到三次；不能在另一计时仍运行时启动续跑。

`--resume --output <directory>`校验冻结采集器和EXE；改变源文件、输入或已有失败时使用新目录。`--max-runs`用于完整曲线边界停止。原始日志、遥测和调用命令保留在研究目录；独立GMP/逐叶诊断及NCU重放不加入正常计时。核验实验kernel3可使用`bench_stage2_s4_probe.py --backends 33 34`；完整诊断入口为`bench_stage2_seed_coop.py --component reduce --reduce-kernel 3 --checks`，其scalar/coop32对照与kernel2/3比较是不同实验，不混算。

`analyze_stage2_large_matrix.py --baseline <directory> --paired <directory> --output data/benchmarks/<name> --figures data/figures/<name>`重新核对save、回执、日志哈希、GMP覆盖和计时闭合，输出CSV/JSON、线性/对数耗时图及互斥阶段百分比堆积图。每侧n=3取全部样本中位数，并使用该实际调用的阶段分项保持闭合；阴影为样本min/max，不是置信区间。S4设备事件嵌套于树/fold/下降，不加入堆积总和。Prime95线为相同N/B1/实际B2的历史单次参考，不按频率或线程外推。

可选`--geometry <study> --tail-study <study>`重读定向D/P与尾块研究，核对正常回执和完整点块计数，另生成geometry.csv及带样本范围的独立图，不替换固定26格成绩。`bench_stage2_seed_coop.py --tail-balance`在其两侧使用相同的既有尾块平衡配置；诊断调用保留`--checks`，不能混入普通计时。

## 位宽复杂度的离线重读

`analyze_stage2_bit_complexity.py`只读已有冻结结果，不运行GPU。主输入为`--study data/experiments/stage2_point_matrix_20261031/min1_full03`，固定D控制为`--fixed-study data/experiments/opencl_full_matrix_20261030/quick_13`；显式指定新的`--output`和`--figures`目录，默认不修改docs。可选`--html docs/web/stage2-bit-complexity-20261101.html`生成自包含SVG阅读页；`--public-figures docs/performance/attachments`复制三张公开PNG。

分析器核对主矩阵52次回执及程序／profile／save哈希、非单位／错误状态、实际NTT分布；固定D控制核对156条CUDA的日志和回执哈希及每格3次完整性。主统计取off侧每格n=1，on侧仅作同工作量敏感性核对；固定D旧构建单独拟合，不合并到当前矩阵。输出保留CSV、JSON、分析器快照、图和样本边界，经验幂指数不是算法复杂度证明。解释见[位宽性能](../../docs/performance/STAGE2_BIT_SCALING.md)。

## 点专用平方验证

构建入口`tools/build/test/build_stage2_point_square_probe.ps1`，输出目录使用内部`.gitignore`。`bench_stage2_point_square.py --exe <probe> --output <新目录> --suite correctness|performance --device 1`直接测试生产头中的原语；`--bits`指定逗号分隔位宽，`--recurrence`控制依赖平方次数，性能候选默认是生产包装mode2。正确性覆盖独立／原地输出及三类模数，并要求故障注入被拒绝；全部事件样本保留，元数据查询在事件计时之外。

完整曲线入口`bench_stage2_square_curves.py --summary <位宽分析summary.json> --exe <候选> --stock <旧版> --output <新目录>`复用冻结save、实际B2和D／布局／chain；默认只取10232／12285／16272，可用`--bits`扩宽。`--diagnostic`改用B2=2600000、D2310及完整叶检查，不纳入正常速度。各调用保存二进制／DLL、命令、配置、save、日志、结果、GPU门禁和遥测；只允许通过`--allow-gpu-pid`明确授权并绑定路径的界面上下文。

独立复测可省略`--stock`并指定`--reverse-first`，先on后off，与首轮顺序相反；原始观察保留，不替换首轮。默认值／显式0／显式1／非法值及真实队列子进程由`tools/test/test_stage2_point_square_native.py --exe <生产程序> --save <sigma26、B1=20的独立M127保存点> --output <新目录> --device 1`检查。这是完整小算例的行为测试，不充当性能样本；使用相同进程门禁和显式PID白名单。

离线入口`analyze_stage2_square.py --study <完成的正常研究> --output <新统计目录> --figures <新图目录>`，可用`--extra`合并独立复测，按每格中位数汇总，不把学习或诊断混入；`--public-figures`复制公开PNG。样本数、计时边界和限制见[位宽性能报告](../../docs/performance/STAGE2_BIT_SCALING.md)。所有实验默认写`data/`，不修改设备设置或真实生产队列。

## 协作点算术与giant链验证

`tools/build/test/build_stage2_point_coop_probe.ps1 -Tpi 4|8|16 -MaxWords 64|128|256`构建一种TPI的字段探针；小TPI可限制容量以控制模板编译量。`build_stage2_point_coop_chain_probe.ps1 -Tpi 16|32 -Words 160|192|224|256`构建引用生产协作头的真实链探针，默认16／256。每个二进制只包含一个协作组合及标量控制，不把全部模板组合放入一份探针。这两个入口默认SplitCompile8，在受限环境外运行；受限环境内显式`-SplitCompile 1`，避免已知并行编译INVALID_HANDLE。输出目录先创建内部`.gitignore`，不改根忽略规则。

`bench_stage2_point_coop.py --exe <probe> --output <新目录> --device 1 --suite correctness|performance`冻结EXE／GMP／manifest、保存采集器快照，逐次检查GPU进程与0.2s遥测。`--probe field`默认测试乘法／平方；`--tpis 0,16 --bits 4096,8192,16381 --counts 1,2048,8192 --rounds 2 --recurrence 8`为字段性能示例，0为本二进制标量控制。count1为0边界，不把它标为随机单次延迟。正确性配合`--kinds 0,1,2`覆盖三类模数、别名、单步／递推和强制故障拒绝。

链测试添加`--probe chain --tpis 0,16 --bits 16319 --counts 2048 --lengths 32 --tails 0 --seed-strides 2 --rounds 2`，0尾长表示完整链。正确性可用`--correct-count 37 --tails 0,1,2 --seed-strides 1,2`；37及以下链全部GMP核对，大批量只核对首末链，并要求两种后端完整输出SHA256一致。它不是每次中间状态都回读的检查。`--exploratory`显式标记探索性采集，离线分析拒绝将其列入合格性能。

`bench_stage2_square_curves.py`通过`--experiment giant-coop`复用固定D／布局／链长的正常曲线协议，将POINT_SQUARE固定1、GIANT_COOP设0／16，旧版使用0。`--diagnostic`在此模式默认B2=26000000、D2310/P240，使giant点数达到chain准入，不沿用平方模式的较小诊断B2。正常／诊断均核对模式资格、实际协作launch、自检及数学工作量。独立复测使用`--reverse-first`并保留原样本；离线仍使用`analyze_stage2_square.py`，由研究元数据识别模式，拒绝混合实验，输出`stage2-giant-coop-curves.png`。原平方模式是默认，保留旧命令行为。

TPI32及分档对照使用`--coop-tpi 32 --coop-middle`，每格测旧版默认、新版默认、固定16k的TPI16、新版TPI32。最高位独立逆序复测可省略stock并使用`--coop-control 16 --coop-tpi 32 --reverse-first`；离线汇总把该off归入middle，不混入新版标量默认。`--stem`为新图命名以保留不同资格研究。引擎init+main只包含S4形状检查；点模乘／平方／协作自检在它之前。工具另保存并汇总`curve_done.wall`，不把采集器门禁／退出轮询算入曲线墙钟。

原生开关检查使用`tools/test/test_stage2_point_square_native.py --check giant-coop`，配合同一M127／sigma26／B1=20的有效保存点，验证默认、小位宽回退、显式0／16／32、非法值和队列继承；不充当性能数据。

`analyze_stage2_chain_plans.py --study <完整链性能研究> --study <另一研究> --output <新统计目录> --figures <新图目录>`比较多个冻结二进制。以固定16k／TPI16为每个输入的控制，核对全部输出SHA、完整事件样本、程序／DLL身份、调用唯一性后，按实际位宽／模数族／TPI／容量统计。不同容器未覆盖的位宽不补测或插值；没有对应TPI16控制则拒绝汇总。该工具是链探针分析，不含seed、归一化和树成本。

可重复添加`--ncu-study <已完成捕获>`，复用宽CSV解析器校验报告／表格哈希，并要求相同链数／链长／模数族且输出与未剖析样本相符；生成单独的NCU图。`analyze_stage2_square.py`同时输出引擎图及`-curve-wall`图，旧数据从保留console提取原始curve_done.wall；不会把图标签修正变成重测或改写旧原始结果。

管理员采集入口`profile_stage2_point_coop.py --reference <已完成的GPU1链call目录> --output <新目录> --run-admin`，可带显式授权的`--allow-gpu-pid`。仅支持本协议GPU1，自动冻结探针及工具身份，以隐藏管理员进程运行NCU2026.2.1；clock/cache控制均none。按kernel名称选第二次提交，检查剖析后的GMP／全输出SHA与参考一致。CSV与报告保留哈希；诊断时间永不混入正常成绩。没有`--run-admin`时只生成具体脚本，可稍后用`--collect-only`汇总已完成采集。

`analyze_stage2_point_coop.py --study <字段性能研究> --extra <其他TPI研究> --output <新统计目录> --figures <新图目录>`只读已完成证据。可配对添加`--ncu-scalar <目录> --ncu-coop <目录>`，核对相同工作／输出及报告哈希后生成NCU图。正式结果、二进制身份、样本边界和默认策略见[位宽报告第11节](../../docs/performance/STAGE2_BIT_SCALING.md#11-可选协作giant链)。

### 协作baby ladder验证

`tools/build/test/build_stage2_point_coop_ladder_probe.ps1 -Words 160|192|224|256`构建单档TPI32及标量控制，直接实例化生产`point_coop_ladder.cuh`，默认SplitCompile8，在受限环境外构建。`bench_stage2_point_coop.py --probe ladder --suite correctness --bits <列表> --tpis 0,32 --kinds 0,1,2`覆盖0／1／64-bit标量边界、三类模数、Montgomery／普通输出；GMP用独立齐次公式验证全部输出，并注入一次主机读回损坏。小批量正确性时间不作为吞吐量成绩。

性能使用`--suite performance --probe ladder --counts 8192 --scalar-bits 17 --operations 0 --tpis 0,32 --rounds 2`；这里operations的0／1表示内部／普通输出，scalar-bits仅用于性能输入。保留每进程全部事件，默认5次，可用`--event-samples 1|3|5`缩短ladder边界筛查，两进程反转顺序；大批量每次GMP抽查17个固定分布点，全部输出SHA跨后端比较，不把抽查称为全量复算。`analyze_stage2_ladders.py --study <目录> [--study <另一档>] --output <新汇总目录> --figures <新图目录>`核对二进制、完整输出SHA、GMP覆盖数与样本边界；全量GMP资格由独立correctness研究提供。

完整曲线使用`bench_stage2_square_curves.py --experiment baby-coop`：固定`NTT_POINT_SQUARE=1`，仅比较baby0／32；`--giant-fixed 0|32`默认32，设0可独立测baby，`--stock`加入冻结旧版标量控制。`--families ordinary,carrier`可限制已有矩阵家族，输入仍须有正常NF参考。此处off为强制baby标量控制，不代表所有运行默认；分析器禁止混合不同giant模式。`--diagnostic`另跑完整叶校验；正常计时继续核对G根摘要、NTT分布及S4工作量。使用`analyze_stage2_square.py --stem <新名称>`生成引擎和完整曲线墙钟两图。计时前完成编译和CPU重验证，若与测量重叠则保留原记录、另建串行研究验收。

`test_stage2_point_backends_gpu.py --scope configuration|formal|giant|unpaired|fallback --exe <程序> --save <保存点> --output <新目录> --allow-gpu-pid <已授权界面PID>`验证实际worker协议。configuration覆盖优先级、非法值、预算回退、六条有效曲线的自动学习和只读复用；formal检查当前D配对、邻D只用胜者、增量不重测；giant检查三种后端及短块回退；unpaired检查只有强制协作成绩时auto仍保留标量。这四组用宽位NF保存点，fallback用M127等低位点核对三种请求均实际回退标量。程序、GMP、输入、进程门禁和遥测逐项冻结。CPU身份与模型验证使用`test_stage2_runtime_tune.py --fixture point-backends`，TOML枚举／worker覆盖见`test_ecm_toml.ps1`；原有低位队列回退继续由`test_stage2_point_square_native.py --check baby-coop`验证。

## 正常运行时选型矩阵

### Baby／giant自动后端的小B2矩阵

`bench_stage2_backend_matrix.py --study <完整的点策略研究> --exe <本机EXE> --reference-layout <已审计的布局/Prime95汇总> --output <新目录> --allow-gpu-pid <已授权界面PID>`复用13份精确Edwards余因子及普通／承载26格，固定来源中已选D、布局、链长和实际B2。默认标注0.57.9，也可用`--version`指定；实际回执版本须与标签一致。仅小档，实际B2约270.16亿，M6011约275.67亿。GPU1串行，batch256、arena6300、fold640 MiB，无warmup，不修改显卡设置或真实队列。

每格使用独立空性能文件，最少观察数2；提供16条重复保存点的计划，在所需后端对照完成后请求原生停止，保留已开始的额外曲线。重复同一点只测成本，不增加独立ECM覆盖。冻结性能文件后，以相反交错顺序各测一次当前标量／标量和auto胜者；学习时间不加入成绩。自动项同时超过当前标量或历史参考5%及20ms时，两侧各追加2次，保留首轮并取中位数。每个输入的最终普通／承载路径由冻结训练成绩决定，不从独立复测中挑最小值。

`--max-cells 1`可验证首格后用`--resume --output <目录>`继续；采集器、程序、输入和性能文件冻结，脚本变化不能继续写入同一研究。逐调用保留进程门禁、NVML、命令、回执、完整日志及SHA256。`analyze_stage2_backend_matrix.py --study <完整研究> --output <新统计目录> --figures <新图目录> --publish docs/performance/attachments`重新解析全部证据，核对NF、GMP、G根摘要、NTT分布、归约量、固定调度及只读性能文件，生成耗时曲线、完整曲线墙钟、阶段秒数／百分比堆积图和CSV。绝对耗时分别输出线性／对数纵轴，N位宽横轴间隔1000；阶段柱图按输入等距排列。阴影范围为原样本min/max，不是置信区间。

原生功能回归见`test_stage2_point_backends_gpu.py --scope giant-auto|giant-small|giant-formal`：固定／凑整B2上的三后端学习、只读复用、未配对保护、显式小P及邻D／增量ECM tune。完整后端队列需留足剩余曲线；最少观察数2时使用10条计划，覆盖四组主要比较及后续复用。该回归使用独立宽位NF输入，全部为功能证据，不加入上述性能矩阵。

### 小档 B2 串行与双槽对照

`bench_stage2_parallel_matrix.py` 读取小档 13 点冻结摘要的 `best` 行和原始 save。每个点默认强制双槽，运行两批、每批两条。保留相同的 N、B1/B2、承载与点核配置。显存不足时试同 D 的另一布局或较小 D，逐项保存准入日志。每条曲线通过引擎自检、下降摘要、NTT 计数和阶段互斥检查后，窗口才进入图表。

```powershell
python tools/bench/bench_stage2_parallel_matrix.py --exe build_stage2_local/ecm_cuda_stage2.exe --device 1 `
  --reference data/benchmarks/stage2_giant_auto_20261101/matrix01/summary.json `
  --inputs data/experiments/stage2_giant_auto_20261101/matrix01/inputs `
  --out data/experiments/stage2_parallel_0620/small_b2_pairs --curves 2 --repeats 2
```

`--curves 9 --repeats 1` 可复查长窗口。短批使用完整队列墙钟除以曲线数，长窗口同时提供完整与去头尾的中段秒数。自动准入使用长窗口的数据。

串行补测使用合并后的双槽 `summary.json`。可先运行下方分析器，省略 `--serial-summary` 和 `--publish`，将 `--output`、`--figures` 分别指向 `data/benchmarks/stage2_parallel_0620/small_b2_pairs` 和 `data/figures/stage2_parallel_0620/small_b2_pairs`。

候选包含旧矩阵的普通与承载优选、当前规划器为两种算术返回的 D/P 和布局，以及实际双槽配置。先对每种唯一配置测一条，距首轮最快值 3% 内有多个候选时各补一条，按引擎耗时选定串行路线。选定后独立跑两批、每批两条。短批使用空 tune 与 `--parallel 2 --tune-runtime 0`，程序在一槽运行并输出原生队列墙钟。采集器核对实际并行槽数及算术结果。

```powershell
python tools/bench/bench_stage2_serial_matrix.py --exe build_stage2_local/ecm_cuda_stage2.exe --device 1 `
  --reference data/benchmarks/stage2_giant_auto_20261101/matrix01/summary.json `
  --pair-summary data/benchmarks/stage2_parallel_0620/small_b2_pairs/summary.json `
  --inputs data/experiments/stage2_giant_auto_20261101/matrix01/inputs `
  --out data/experiments/stage2_parallel_0620/small_b2_serial
```

```powershell
python tools/bench/analyze_stage2_backend_matrix.py `
  --reference-summary data/benchmarks/stage2_giant_auto_20261101/matrix01/summary.json `
  --parallel-summary data/benchmarks/stage2_parallel_0620/small_b2_pairs/summary.json `
  --serial-summary data/experiments/stage2_parallel_0620/small_b2_serial/summary.json `
  --output data/benchmarks/stage2_parallel_0620/small_b2_comparison `
  --figures data/figures/stage2_parallel_0620/small_b2_comparison --publish docs/performance/attachments
```

分段采集可多次传入相应的摘要参数。对照图要求两种模式覆盖同一 13 个位宽，GPU、版本和样本规模一致。线性和对数绝对耗时图使用相同数据，百分比图和 `serial_vs_parallel.csv` 以单槽每曲线耗时为分母。

### 启动短校准与生产凑整隔离

`tools/test/test_stage2_calibration_rounding_gpu.py --stage2 <EXE> --save <B1=20的NF成本探测点> --output <新目录> --allow-gpu-pid <已授权界面PID>`用空性能文件验证五种配置。`--expect-bug`记录修复前短校准被凑整的失败及两个通过对照；默认要求短校准保持D210／B2=42020／202点，正常规划仍保留生产凑整，显式ECM tune仍拒绝凑整但接受CLI关闭。`--run-curve`将short_on继续执行为一条真实Stage2，核对NF、实际B2及GMP零错误。该工具不消费生产队列，不修改源保存点；原始输入与程序、逐调用进程门禁及遥测均独立冻结。此为功能回归，不计入性能矩阵。

### B2凑整与有限运行时搜索

`tools/test/test_stage2_rounded_runtime_gpu.py --stage2 <EXE> --save <NF保存点> --case wide|workers|default --output <新目录> --allow-gpu-pid <已授权界面PID>`验证实际曲线及性能文件。wide用8440-bit等协作范围输入，固定D6006／请求B2=26000000，8条曲线完成baby与32链比较，另跑一条关闭学习的只读复用。workers用M127输入，两worker在不同cwd各执行10条自动D曲线，性能路径来自同一ecm.toml下的两个worker覆盖，检查全局文件不被创建、第二个worker不改第一个文件，以及不同D按各自实际B2采样。两worker顺序使用同一空闲测试GPU，不作为双GPU性能证据。

default另需`--baseline <冻结旧EXE>`，关闭凑整和采样，比较完整曲线的GMP、NTT分布、归约量与G根摘要；这是工作量／算术控制，不代表全矩阵性能验收。`test_stage2_runtime_tune.py --fixture bounds`覆盖边界、幂等、AutoB2上限及默认身份；生成的fixture可用`--profile <冻结生产tune> <请求B2>`在原记录条件下核对实际覆盖平台和已有观察数，不调用GPU。

### 布局与旧tune的有界验证

`bench_stage2_layout_matrix.py`复用已完成的默认矩阵及旧性能文件，当前验证范围限定为较小的名义B2=26000000000档：13份精确余因子、普通/承载默认26格，各n=1筛查，明显回退单项补测。默认矩阵若按用户要求中途缩减范围，须保留明确的`authorized_scope_stop.json`及边界计划；新研究只提取已完整验收的小B2格，不改写原始记录。

每个位宽先对旧性能文件中最快合格几何的**当前自动布局**重测2条真实曲线，再解锁D、算术和布局，用正常worktodo继续学习。合计至多8次完整调用；原生停止事件保留至多1条已经开始的额外调用。继承48条计划队列，避免剩余曲线不足4条而禁用搜索。旧文件的条件、时间和来源不重标记；本机资格仍由生产选择器检查。达到3次新非probe调用也可提前结束，不声称搜索穷尽。

全部学习完成后冻结性能文件，用生产只规划入口核对已准入重复均值的最快选择，然后每格独立续算3条，关闭写回。主成绩只使用这39条新复测，学习调用和旧时间不加入成绩。原始命令、NF回执、GMP检查、实际布局、阶段与逐样本NVML全部保留。

入口为`bench_stage2_layout_matrix.py --defaults <已完成小B2的默认研究> --seed-profile <旧tune> --output <新目录>`，`--max-queues 1`可检查首格，随后`--resume --output <目录>`继续。`--allow-gpu-pid <PID>`只用于用户明确允许的界面上下文，同时绑定PID与完整程序路径，仍拒绝其他外部进程；不等同于允许任意共享设备。

绘图沿用`plot_stage2_version.py`，添加`--layout-study <研究目录>`；`--study`指向该研究的`defaults/`，其他输出/图片/公开附件参数不变。先重读回执核对三波测量、D/P、布局、NF、自检、进程协议及冻结身份，再生成本轮独立均值、旧学习均值对照和阶段占比堆积图。Baby从init拆出，剩余init标为F树/初始化余项；main的求逆、giant点、G树、fold、下降和累乘分别计入，残差闭合引擎时钟，不叠加嵌套NTT/S4计时。

`rescreen_stage2_matrix.py --study <研究目录> --reference <旧默认研究>`筛出同时超过旧中位数5%及20ms的默认格，再各跑2遍，保留原首轮，生成`rescreens/`。加`--legacy-controls`则只重跑当前自动布局与旧策略不同的格，每格1遍、固定原D及旧布局，生成`legacy_controls/`。两者使用相同冻结二进制和精确输入，不启用学习；布局对照强制检查非空根摘要并比较。输出目录必须不存在，已有证据不覆盖。少量单遍对照不替代n=3因果矩阵。

当前详细百分比图额外拆出`real_batched_wall.gleaves`（Giant叶准备／同步）；旧日志缺此项，因此另绘共同分项图，旧叶准备保留在main余项。主机范围计时可能把异步设备工作记在下一处阻塞边界，不能直接解释为孤立GPU核函数时间或纯CPU求逆时间。默认成绩取中位数；分项秒数及百分比采用各次均值，图注区分口径。

以下只重读已完成的0.54.7小档证据并重绘，不启动GPU测试。新输出目录须不存在；正式报告引用相邻attachments公开副本。

```powershell
python tools/bench/plot_stage2_version.py `
  --study data/experiments/stage2_layout_matrix_20261031/low_warm2/defaults `
  --layout-study data/experiments/stage2_layout_matrix_20261031/low_warm2 `
  --output data/benchmarks/stage2_layout_matrix_20261031/low_final `
  --figures data/figures/stage2_layout_matrix_20261031/low_final `
  --publish docs/performance/attachments
```

输出包括`summary.json`、`cells.csv`、`report_fragment.md`及绘图器快照，四组PNG/SVG分别为绝对耗时、当前细分阶段、跨版本共同阶段、曲线墙钟分项。工具拒绝不完整波次、混合B2、重复调用、身份改变、算术检查缺失或不同布局根摘要不一致；不会把范围缩减后的半份双档矩阵标为全范围通过。

### 从空白性能文件学习

当前矩阵使用 `build_cuda_cmake/stage2_d_search_0_39_3/` 的 ECM-ELF 0.39.3，保留 0.38.4 完整默认矩阵作参考。`bench_stage2_runtime_matrix.py` 冻结指定 EXE/GMP、13 份已独立验证的原始 Stage1 点。每个实际 N/B2 条件提供一条 **上限 48 曲线的连续 worktodo 队列**，共 26 条队列；当当前最快合格参数组完成第 3 个入账观察时，提前结束该条件。相同保存点重放仅测量耗时和选型，不增加独立 ECM 覆盖。

```powershell
python tools/bench/bench_stage2_runtime_matrix.py `
  --baseline data/experiments/stage2_v0_38_4_20261018/full_matrix `
  --exe build_cuda_cmake/stage2_d_search_0_39_3/ecm_cuda_stage2.exe `
  --version 0.39.3 `
  --import-study data/experiments/stage2_runtime_v0_39_3_20261019/matrix_01 `
  --output data/experiments/stage2_runtime_v0_39_3_20261019/adaptive_01
```

D/P 和普通/承载全由正常运行时选型决定；不执行单独 tune，不锁定 D 或算术。每组采样上限保持默认 10，启用准备缓存。队列提供并验证原指数、已知因子和精确 N；48 条计划用于保留正常的队列剩余量判断和父子进程准备复用，外部采集器按第 3 次观察提前停止。先运行最小位宽低 B2，随后按位宽和 B2 降序，优先核对最大位宽高 B2。

停止条件要求完整形状相同、当前调用不是探测对、更新后的均值仍不高于其他已准入重复候选；探测过程中的两个观察不会触发停止。采集器用继承的原生停止事件请求退出，保留已经开始的至多一条额外曲线、未完成 worktodo 和检查点，不修改原始结果。此规则是用户指定的采集启发式，不表示候选已穷尽：程序可能在更多重复之后开放更大的搜索配额。

本轮保留原规则的三条完整队列，以及用户要求立即停止的 14328-bit 高 B2 队列中 42 次完整 NF，共 186 次调用；后续条件采用提前停止。`--import-study` 只接受记录了用户批准停止的原始研究，并核对同一冻结程序、输入和数学结果。中断时未完成的一次调用只保留日志，不计入 NF 或成绩；该队列未落盘的遥测和精确进程墙钟明确标为不可用，完整曲线计时与算术检查仍逐条验证。原目录与日志不改写。全新研究省略 `--import-study`，从空白性能文件开始。不同条件实际执行次数和被选行样本数分别报告。

逐调用保留完整引擎计时、`curve_done.wall`、实际 D/P/承载、选择和搜索理由、GMP 检查；逐队列保留整卡遥测、进程墙钟和性能文件快照。调用前后查询 GPU1 外部进程，发现竞争即保留证据并拒绝该轮；边界快照不能证明每一瞬间绝无竞争。GPU0 用户工作不作更改。预算保持 arena6300/fold640/batch256 MiB，短标定、relocation、NTT 计数关闭。

成绩采用最终 tune 的最佳合格实测决策：全部队列完成后通过生产 worktodo 的只规划入口读取，关闭新采样、不锁定参数。程序须返回 `runtime_measured_fastest`、至少两次合格观察，均值为已准入重复候选中的最小者。记录实际样本数、均值、样本标准差及原始 min/max；这些观察也参与选型，**不冒充独立的选型后 n=3 验证**，也不证明全部 D 的全局最优。默认普通/承载仍使用原 n=3 中位数，图注区分统计口径。

```powershell
python tools/bench/bench_stage2_runtime_matrix.py --resume `
  --output data/experiments/stage2_runtime_v0_39_3_20261019/adaptive_01
python tools/test/check_stage2_runtime_matrix.py `
  --study data/experiments/stage2_runtime_v0_39_3_20261019/adaptive_01 `
  --output data/experiments/stage2_runtime_v0_39_3_20261019/adaptive_01/audit.json
python tools/bench/plot_stage2_version.py `
  --study data/experiments/stage2_v0_38_4_20261018/full_matrix `
  --runtime-study data/experiments/stage2_runtime_v0_39_3_20261019/adaptive_01 `
  --output data/benchmarks/stage2_runtime_v0_39_3_20261019/final `
  --figures data/figures/stage2_runtime_v0_39_3_20261019/final `
  --publish docs/performance/attachments
```

审计要求完整 26 个条件、每条实际执行的唯一完整 NF，核对提前停止的触发观察、原生退出确认及队列剩余状态，同时核对点重放、输入、版本、原始计时、数学检查及覆盖；用 JSONL 的完整精度观察独立重算各组均值/方差，六位小数文本计时另按舍入精度对照。保留队列另与导入前的原始计划逐项核对。最终只规划输出须与被选性能行和原始观察对应。

`stage2_ecm_tune.toml` 为活动文件，最终完全相同的 `stage2_ecm_tune_frozen.toml` 可通过 `stage2_tune_profile` / `--tune-profile` 复用。它们留在忽略的 data 目录，不公开设备身份或运行去重标识。学习队列的引擎、队列和进程时间分别报告，不与最优成绩重复相加；计时边界见 [准备结果的复用](../../docs/architecture/AUTO_B2.md#准备结果的复用)。

## 协作paired seed验证

`stage2_point_coop_seed_probe.cu`直接包含生产scalar／cooperative算术头；独立GMP多项式oracle核对Montgomery坐标及相邻点布局。通过`tools/build/test/build_stage2_point_coop_seed_probe.ps1 -Words 160|192|224|256`分别构建；构建必须完成后再计时。

`bench_stage2_seed_coop_probe.py`接受`--exe`、`--bits`（空格分隔）、`--suite correctness|performance`、`--output`与可选`--allow-gpu-pid`。正确性覆盖三类模数、分批offset、0／大起点、1／2点尾和预期故障拒绝；性能采用2048链、32点链、两种相反顺序，每进程保留3个CUDA事件，无warmup。每调用都核对GMP抽查及跨后端全部输出摘要，正确性模式还要求全量GMP词覆盖。保留exit code、命令、输出、EXE／GMP／manifest、采集器快照及0.2s遥测。

完整曲线工具`bench_stage2_seed_coop.py`复用已完成自动后端矩阵的save、D／布局／链长；默认只取>8192目标bits，显式baby32／giant32，对比seed scalar／coop32。`--case`可重复，`--repeats 1|2|3`控制每后端次数；顺序按格和重复交替。`--checks`另开seed投影、全部设备叶的GMP及目标叶摘要检查，标记diagnostic，不纳入正常计时。`--d`／`--b2`只用于显式固定几何的资格；不在此工具搜索参数。

普通计时与诊断目录必须分开。工具冻结官方build_manifest及EXE／GMP、核对各编译目标、save和原矩阵身份；每格验证NF、N／B1／B2／sigma、真实seed／[D]Q后端、D／P／布局、必需GMP与相同NTT／S4工作量。只读使用不存在的性能文件，关闭学习、短校准、NTT短批调优、relocation、凑整和尾块平衡。保留全部首轮与失败，明显回退另建目录复测；不覆盖原样本。

例如完整小B2对照使用`python tools/bench/bench_stage2_seed_coop.py --study data/experiments/stage2_giant_auto_20261101/matrix01 --exe build_stage2_local/ecm_cuda_stage2.exe --output data/experiments/NEW_STUDY/curves`。运行前核对GPU空闲和准确白名单；不为实验修改时钟或功耗。有效配置和tune分组另由`tools/test/test_stage2_seed_config.py`通过真实Worker #2子进程验证；`test_stage2_runtime_tune.py --fixture point-backends`验证旧标量条件、策略隔离及NTT表复用。

`analyze_stage2_seed_coop.py --study ... --output data/benchmarks/... --figures data/figures/...`拒绝未完成或诊断研究，输出逐格CSV／JSON、完整引擎时间曲线及最高位的互斥阶段堆积图。另从原始console提取`curve_done.wall`，与引擎分别汇总；不把进程墙钟、点检查、S4设备事件或[D]Q嵌套范围重复相加。`--publish docs/performance/attachments`只发布静态PNG；原始证据继续由data忽略规则管理。

### Seed auto资格

`tools/test/test_stage2_seed_auto_gpu.py --study <完成的矩阵目录> --exe <当前Stage2> --output <新证据目录>`复用已验证Stage1输入，顺序验证R2／R1、双方同几何对照、只读复用、未配对保护、baby／giant／seed与有限链长auto。所有曲线保留；这是行为验证，不能当作独立性能矩阵。`test_stage2_seed_config.py`验证真实Worker #2的覆盖／CLI优先／环境回退，并按实际seed_coop核对增量统计。旧文件的非零请求不能作为实际协作证据；格式12写入实际后端，C++／Python加载和聚合均保留未知255。


## S4普通归约验证

生产长除法与warp协作共用`src/ecm/stage2/cuda/device/s4_warp_reduce.cuh`。`tools/build/test/build_stage2_s4_warp_probe.ps1`构建独立GMP探针；`bench_stage2_s4_probe.py --exe ... --output ... --bits ...`默认比较scalar／coop32，支持digit宽度、四类模数、多batch、真实模数文件、空输出窗口及两种故障注入。每调用冻结EXE／DLL／manifest和全部编译源文件，逐词核对所有输出和未使用缓冲哨兵；所有源槽的padding检查包含被省略的系数。`--coeffs 2`为合法空窗口，`--fault 1|2`分别要求输入padding或输出破坏被拒绝，不能将这些调用计入性能均值。

`--backends 0 32 33 --repeats 3 --reverse`比较原标量、生产warp两遍和生产warp合并乘减；两种warp均调用生产分派，33是探针的合并版标识。`--threads 64|96|128`筛查块大小；`--device`指定卡。8192等缓存窗口值仅供旧实验，不是生产配置。全部首轮与事件保留，无warmup。`analyze_stage2_s4_probe.py --study ... --output ... --figures ...`按每进程事件中位数汇总，拒绝故障或未完成研究；原语吞吐不能换算成完整Stage2加速。

完整固定几何对照使用`bench_stage2_seed_coop.py --component reduce`，固定baby32／giant32／seed32，只改变S4策略。`--reduce-kernel 1|2`指定warp版本，默认两遍；`--device`指定卡。`--arithmetic both|ordinary|carrier`可定向复测，默认both；`--checks`单独执行GMP叶检查和完整目标叶摘要比较，不进入正常计时。`analyze_stage2_seed_coop.py`自动识别reduce并使用独立图名及S4设备事件字段，防止覆盖seed图。Worker配置优先级用`test_stage2_seed_config.py --component reduce`验证。`test_stage2_reduce_auto_gpu.py`检查重复门槛、实际身份、相邻D、NTT拐点、强制及未配对保护；`test_stage2_seed_auto_gpu.py --reduce-backend auto`检查各自动后端与链长联合学习。

桌面卡只有获得用户明确授权后才可使用`--background-snapshot`：快照须含`approved_background_by_user`、device、UUID及绑定的PID／路径。采集器复制快照、记录完整遥测，并拒绝新增进程。此类结果须标明桌面背景，不等同独占设备成绩；不关闭桌面程序或改变GPU设置。`stage2_point_backend_fixture --make-save-from-modulus`可为明确给定的高位宽奇模数生成Montgomery B1=20测试点；合成资格输入与生产余因子计时分别汇总。

管理员NCU必须在无其他GPU1任务时串行运行，clock/cache control为none；选定NW256普通归约、跳过200次同类调用后采3次，保留命令、`.ncu-rep`、CSV单位行、进程门禁与遥测。核对版本／输入／检查／工作量后，比较两种内核的对应调用；不将重放后的程序墙钟当作基准，不把sectors当作DRAM字节，也不把每issue的stall ratio当百分比。完整证据与命令位置见[S4性能报告](../../docs/performance/STAGE2_S4_REDUCTION.md)。

树阶段时间线使用`profile_stage2_tree_trace.py`，提供`--reference`已完成调用目录、`--exe`带manifest与冻结sources的程序、`--study`来源矩阵及不存在的`--output`。它开启已有CPU NVTX范围，以管理员Nsight Systems记录CUDA／NVTX，复制配置和保存点，并核对NF、数学输入、D/P、GMP及NTT工作量后导出SQLite。诊断时间不入性能矩阵。

`analyze_stage2_tree_trace.py --trace <trace.sqlite> --device <index> --output <new.json>`按进程限定的提交关联ID归属树内核，支持异步执行跨过CPU范围边界。结果分纯NTT／pointwise、S4、其他kernel与DMA，并单列CPU范围、CUDA API驻留并集和同步API驻留。主机与设备可重叠，嵌套范围也不能相加；同步拷贝API驻留不等于PCIe耗时，API外范围不等于CPU指令时间。单元回归使用内存SQLite构造跨界、同关联ID不同进程和重叠等待，防止错误归因。

`analyze_stage2_seed_coop.py --retest <追加矩阵目录>`可追加同二进制、同设备、同内核版本、同组件／输入矩阵／显存预算／D几何的配对复测；保留首轮，采用全部样本中位数。阶段图取中间真实调用的阶段向量，偶数时平均中间两个，避免独立求各阶段中位数后不能相加。`analyze_stage2_s4_ncu.py --baseline ... --candidate ... --output ...`校验对应真实调用、输入和完整检查／工作量，导出带原始单位依据的诊断摘要。

## S4容量、树追踪与队列回放

容量实验使用S4探针的`--backends 34 35`比较固定九组与按实际W分派的除数缓存，生产固定D对照使用显式coop32及`--reduce-kernel 3`、`--reduce-kernel 4`。树时间线额外输出`Fold.resident_step`和`ntt_shapes`，后者以提交线程上最内层物理NTT长度、batch和kernel名汇总设备时间；旧追踪缺少形状标记时此表为空。形状时间嵌套于所属阶段内。

`replay_stage2_queue_budget.py --plan <完整运行矩阵plan.json> --output <新json>`在同族、同布局已有至少三个合格D的组中回放剩余2、5、20条曲线。共享一个已测基线，最多试前四个备选。首次备选固定允许，后续使用该步原始的测前预测及可选`--allowance`（默认0.25）估计剩余收益，一旦停止即不使用后续预测。试跑取最早R条的父进程墙钟，稳定曲线按该D记录的平均墙钟估算。此回放比较两种有限策略，不重建生产程序的跨族选型、显存准入和120秒预算。

## 同位宽阶段预测的CPU验证

`tools/test/test_stage2_runtime_tune.py --fixture phase-model --output data/experiments/NEW_STUDY/cpu`验证阶段守恒、缺失阶段、硬件与实际策略隔离、NTT拐点、精确测量优先和默认选型隔离，不调用CUDA。独立GMP整数界查询用于CPU打包参考，不替代生产原生准入。

可追加`--profile <仅含一条测量行的tune.toml>`，输出同一训练锚点下的`seconds/rank`和原整曲线修正的`prior_seconds/prior_rank`。结果是D候选的模型审计，不是实测成绩或显存可行性证明；输出保存输入/模型/fixture身份。生产几何仍须使用真实`--plan-only`或完整曲线验证。阶段模型适用条件见[Auto B2与运行时选型](../../docs/architecture/AUTO_B2.md#正常运行累计采样与候选试跑)。
