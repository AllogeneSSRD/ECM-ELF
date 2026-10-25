# Stage2 使用

本页说明 CUDA Stage2。CPU Edwards 续算入口、GW 多线程多项式与 libecm 后端选择见 [Edwards CPU Stage2](EDWARDS_CPU_STAGE2.md)；两份结果文件及提交字段规则统一见 [ECM 结果输出](ECM_RESULTS.md)。

`ecm_cuda_stage2.exe` 读取完成的 Stage1 save，运行 CUDA 多项式 ECM Stage2。手动 B2 可处理实际 N 为 2…16384 bits 的受支持输入；Auto B2 另受 profile 资格限制。设备编号是 CUDA 运行时编号，与构建的 sm 架构不同。

直接读档和队列运行启动时均输出 `ECM-ELF x.y.z (Stage2)` 版本横幅，包括 `--dry-run` 和 `--plan-only`；GUI 从这个横幅读取实际 worker 版本。只输出机器数据的内部辅助入口保持其原有格式。

## 直接读档

基本参数为 `--save <file> --b2 <integer> --device <id>`。`--skip-curves <n>` 跳过记录，`--curves <n>` 限定数量；具体名字以当前 `--help` 为准。B1、sigma 和 N 取自 save，不重新执行 Stage1。`--factor-only` 省去命中素数命名，保留叶乘积、GCD 与必需检查；`--factorize-hits` 开启共享诊断层的附加分解，优先本地 PARI DLL，备选外部 GP，不自动使用 GMP。原始因子先发布，附加诊断未完成不改变成功状态；仅完整且全部已证素才标为分解完成。内部结果保留后端、概率素性及余因子，详见[群阶模块](../architecture/GROUP_ORDER.md#共享后端选择与进程隔离)。

接受归一化 Suyama PARAM0 文本记录，以及 GPU Edwards 输出的 `METHOD=ECM-EDWARDS-AM` 记录。Edwards 使用生产 Stage2 引擎并要求显式 B2；存档身份和常数约定见 [输入与表示](../architecture/STAGE2.md#stage1-save-与承载资格)。CPU Edwards 的 Prime95 二进制 `.tmp` 不属于此入口。先验证选择范围中的所有记录，包括 checksum、N、B1、X 和任务匹配；文件损坏不能当作“没有足够曲线”略过。源 save 不修改。校验和通过只说明文件完整。

直接多记录执行不使用队列自动续跑。每条曲线在独立子进程计算，结束后释放 CUDA 上下文与临时资源。

`--results` 的内部 JSONL 以 `curve=ed|mont` 标明模型，Edwards 的 `param` 为 null，Suyama 的 `param` 为 0。另写 `--submission` 指定的 `results.json.txt`；TOML 的 `user`、`computer` 用于提交身份，标准梅森队列自动传递原始指数、已知因子和可选 AID。直接处理梅森余因子时应提供 `--exponent P --known-factor F`，输出前验证与实际 N 的关系。批次去重和恢复见 [结果输出](ECM_RESULTS.md)。GUI 的因子结果文件使用其独立结果格式，见 [GUI 日志与结果](GUI.md#日志结果与监控)。控制台结果事件也携带模型及完整 sigma；显式 `--gui-events` 会在 quiet 级别保留该结果事件，普通 quiet 控制台规则保持不变。

## 队列语法

save 内部字段及读取限制见 [Save 架构](../architecture/SAVE.md)；队列 checkpoint 的字段、身份摘要与持久化顺序见 [Checkpoint 架构](../architecture/CHECKPOINT.md#7-stage2-队列进度格式)。

任务格式：`ECMSTAGE2=[AID,]k,b,n,c,save[,B2-or-zero][,skip_curves][,num_curves][,"known_factors"]`。

数学目标为 k·bⁿ+c，再按已知因子确定实际 N。可选字段顺序固定：B2、跳过数量、执行数量、已知因子。AID 位于 k 之前，末尾 quoted 字段是因子列表。

例如 `ECMSTAGE2=1,2,3701,-1,"m3701_260e6.save",26000000000,960,10,"xxx"` 中，960 是跳过的 save 记录数，10 是请求数量；`xxx` 所在位置必须填可解析的已知因子或空字符串，不能当作任意备注。

整数支持精确十进制及科学记数法，不能经浮点舍入改变 B2。`b2_ratio` 为非零整数时，执行 Stage2 前用 save 中的 B1 × ratio 覆盖 B2，并检查乘法溢出与 B2 > B1；否则 B2 优先级为 CLI 非零值 → 任务非零值 → `stage2_b2`，最终为 0 且启用 Auto B2 时才自动选择。

## 数量不足与错误

请求 10 条、跳过后仅有 3 条时，提示 requested/available，执行这 3 条。全部成功后按 completed=3 完成原任务，继续下一项；可用记录为 0 时提示并按零曲线完成，不等待 save 追加。

存档缺失、损坏、checksum/N/X 不匹配或任务字段错误属于输入错误：记录原因，队列原行变为错误注释，finished 保留原任务和原因，继续下一项。文件权限、状态对账、CUDA/设备/显存或未知内部故障属于致命错误：停止，保留未完成任务。退出码为正常 0、存在队列输入错误 1、致命错误 2。

## 自动续跑与退出

仅队列记录完成曲线编号，不保存 Stage2 中间状态。

1. 开始曲线前原子写入尝试回执。
2. 子进程验证结果，追加结果和回执并刷新磁盘。
3. 父进程确认成功回执后原子更新完成数。
4. 中断发生在结果写入与进度更新之间时，重启按结果回执补记，避免重复执行。

每项任务使用 `.ecm_ckpt_stage2_<实际N位数>_<任务SHA256>.dat`。身份绑定 AID、实际 N、save 完整内容及 skip；请求数量和有效 B2/Auto B2 是可调整的执行策略，单独写入 schema 3。文件记录完成数、执行策略、结果发布身份及待确认回执，不保存点、多项式或 GPU 算术状态。

重启前可重排任务、移动相同内容的 save、改变设备/显存预算、D/承载选择或日志路径，已有完成数继续有效。当前 worktodo 的数量和有效 B2 覆盖 checkpoint 的执行策略；B2 仍遵循本页配置优先级，不覆盖显式 CLI 或非零 b2_ratio。已完成曲线保留各自真实 B2、D/P 和时间，只有后续曲线使用新 B2。AID、目标 N、save 字节或 skip 改变仍视为新输入；运行期间修改当前队列行仍受完成前一致性检查约束。

例如进度 150/960、目标改为 300 时，从第 151 条续算，完成到 300。目标改为 100 时，不回退完成数，也不再执行曲线；警告后按实际完成的 150 条封存结果并正常推进队列。NF 数量扣除因子命中；若这 150 条均为 NF，提交 curves=150，采用最后完成 NF 的实际参数，不把新请求 B2 写到旧工作上。请求数超过 save 可用数仍按可用数量执行并警告。

旧 V2 独立 checkpoint 和 `worktodo.txt.progress` 可按固定输入匹配导入，保留其原发布身份，使旧、新 B2 的回执仍属于同一次任务；成功后保留原字节为 `.migrated.<摘要>`。V2 未单独保存旧 B2：数量变化可由旧 total 核对；B2 变化时从现存提交回执提取候选界限，并重新计算旧摘要确认匹配，不按文件时间猜测。旧 B2 没有任何可用记录时，应先以原 B2 启动完成升级，再修改 B2；不能从不可逆摘要恢复缺失的参数。多个旧 checkpoint 同时匹配则拒绝自动选择。

调整前优先从原结果文件确认 pending 回执；完整回执使完成数加一，再判断是否已超过新目标。未产生任何持久结果的中断尝试可以放弃。若已有数学结果回执但内部 JSONL 未完整提交，参数调整暂时报错并保留证据，须先按原策略恢复这条曲线。未以换行结束的结果尾行不算成功回执。导入归档失败时报错，不在后台留下可能被再次导入的活动旧档后继续计算。

任务结束先写 finished 回执，再推进队列，最后清除进度。第一次 Ctrl+C 等待当前曲线落盘后退出；第二次立即终止子进程树。强制退出的当前曲线若没有完整回执，下次从头计算。

## TOML 与路径

默认读取 exe 同目录的 `ecm.toml`；隐式文件缺失采用内置默认值，显式 `--config FILE` 缺失或格式错误时报错。配置分区、优先级和逐键覆盖资格见[配置参考](../ECM_TOML_REFERENCE.md)。

Stage2 与 Stage1 共用 `[shared] worktodo`，使用 `[stage2]` 的 `stage2_finished`、`stage2_results_file`、`stage2_submission_file`、`stage2_log_file`、`stage2_progress_dir` 和 `stage2_debug_log_file`。队列只领取 `ECMSTAGE2=` 行；`priority`、`skip` 和 `b2_ratio` 在 `[shared]` 设置，按 worker 覆盖时写入 `[workers.N.shared]`。非零 `b2_ratio` 以 save 中 B1 × ratio 覆盖有效 B2，直接读档也适用。

共用 `worktodo` 及 Stage2 相对路径基于 TOML 目录，其他 Stage1 相对路径基于其程序目录；CLI 相对路径基于当前工作目录。进度目录空值使用队列目录；显式旧进度路径仍可作为导入来源。save 目录空值继承 Stage1 的 `tmp_dir`，`stage2_device = -1` 继承 Stage1 的 `device`。推荐将两个程序与共用 TOML 放在同一目录。

`[stage1] auto_stage2 = 1` 可将无因子的 Stage1 任务转为 Stage2；默认原位改写，也可选择接收 worker 列表或其他队列文件，见 [Stage1 自动交接](STAGE1.md#自动交接到-stage2)。交接由 GUI 在阶段切换时启动对应程序。Edwards Stage2 需要有效 B2，可来自任务、`stage2_b2` 或 `b2_ratio`。GUI 按 `ECMSTAGE1=` / `ECMSTAGE2=` 前缀调度，不读取配置中的阶段字段。

运行时拒绝 Stage1/Stage2 队列冲突及 save、程序、配置、队列、进度、结果和日志之间的危险重叠。

## 日志与内存摘要

可选 `stage2_relocation=1`（或 CLI `--relocation 1`，默认0）在相同 B2 下试算较短 giant 后缀。命中或目标 N 非单位时自动重放完整原区间，完成验证后才发布结果。选 D/内存规划仍按原区间，完整 ECM tune 不支持此模式；详见 [算法、指标及边界](../architecture/STAGE2.md#可选-relocation) 与 [性能实测](../performance/EDWARDS_STAGE2.md)。

`verbose` 控制可读的阶段/批次输出；`stage2_debug_log` 控制独立调试文件。警告和错误始终可见。长阶段每 30 秒的父进程状态提示不等于 kernel 完成百分比。

- `curve_start`：任务曲线编号、save 记录编号、sigma、B1 和文件校验状态。
- `stage2_phase`：`previous` 为上阶段墙钟间隔，`elapsed` 为曲线累计时间。候选计划查询不输出正常阶段日志，也不更新曲线阶段时钟；实际运行的规划阶段每条曲线只打印一次。独立调试日志与规划 JSON 保留诊断信息。
- `real_shape`：B1/B2、算术位宽、D/P、baby/giant 数及 G 批次数。
- `mem`：设备 free/total、reserve 和 arena cap；预计最大变换及树顶形状。
- 曲线结束的 `arena=a/b fold=a/b`：a 为实际完整 NTT 池峰值或启用 fold owner 占用；fold 回退时 a=0。b 保留读取的 TOML 值，可能与 CLI/环境覆盖后的有效预算不同。
- `stage2_ntt_counts=1`（默认 0，CLI `--ntt-counts` 可覆盖）：曲线结束输出按长度排列的卷积、正向、逆向和总次数表，并保留 `ntt_length_counts` 兼容行。一次卷积为两个正向加一个逆向，不是 kernel 启动数；关闭时不采集此表。`ntt_work` 的元素数、逻辑蝶形数、计数边界及开销见 [NTT](../architecture/NTT.md#实际运行的长度与运算量)。
- `stage2_full_wall.total`：完整引擎 init+main，排除 Stage1 与自动 D 扫描。外层 result/curve_done 还含进程和发布结果等开销。
- `stage2_tune_runtime`：只有成功补充缺失算术位宽/执行区间摘要时输出，包含位宽、算术家族、D/B2与实测/预测比值。默认采集已执行的干净完整曲线，可用`stage2_tune_runtime=false`关闭；完整规则见[运行补充](../architecture/AUTO_B2.md#正常运行补充缺失位宽)。

`arena_mb` 是 NTT 池 payload，不是全部显存。独立 giant、S4、owner、下降元数据、按次回退和驱动资源另计。组件峰值不能直接相加；详细含义见 [内存](../architecture/MEMORY.md)。普通日志和结果默认不轮转。

正常离线计算不需要监听网络。Windows Compute Sanitizer 的注入通信可按目标 exe 名称触发防火墙提示；测试工具在 Windows 子进程使用 named pipes。该行为不表示 Stage2 算法需要网络权限。

## 规划与依据

`--plan-only` 查询设备及算法/组件形状，不执行任务曲线、不写成功结果、不推进队列；默认启动短校准可能额外运行小基准并更新性能文件，可用`--short-calibration 0`关闭。`--dry-run`只检查输入。`--carrier-exponent p`要求保存的目标N整除2ᵖ−1，显式0锁定按save的N运算。默认`stage2_auto_carrier=1`允许标准梅森worktodo在已知因子、save和整除验证后比较普通与承载；设置0关闭自动候选，CLI `--auto-carrier 0|1`覆盖TOML，显式carrier锁定仍优先。候选资格、性能保护及指数上限见[Auto B2 / tune](../architecture/AUTO_B2.md)。

`--tune ecm --tune-level 1`优先读取TOML指定的当前worker队列与有效save，采用任务/TOML的生产B2；`--tune-worktodo FILE`可指定队列。`stage2_auto_carrier=1`时先补普通默认基线与经整除验证的承载默认对照，再从模型D/P搜索相邻项；只有平均耗时降低超过2%且越过双方2×样本标准差噪声界限才继续移动。不会推进生产队列。

`--tune-exponents bits,...`明确使用通用位宽网格；默认队列缺失或无Stage2任务时也使用网格。`--tune-save FILE`使用一个保存点，自动承载指数可来自当前worker中重建N相同的任务；也可用`--tune-carrier-exponent p`明确指定。完整梅森N的普通路径已使用梅森归约，自动模式不重复测量相同承载。

默认1800秒软预算，`--tune-budget-seconds`可覆盖。ECM、NTT及启动短校准默认不运行丢弃的预热轮次；`stage2_tune_warmup=1`或CLI `--tune-warmup 1`启用每个新测形状的一次预热。正式`repeats`记录实际次数，保存平均耗时和样本标准差；必要算术检查保持不变。缺NTT资料时补轻量组件校准，真实承载任务的普通/承载参考优先占用预算。

`--tune-d`、`--tune-b2`、`--tune-exponents`使用逗号列表，其中exponents表示实际位数；显式D表限定搜索范围。格式8每个位宽一个`matrix`，首行为列名，后续行与表头对齐；同参数重复执行合并repeats、均值和样本标准差，不同参数保留多行。行内保存来源程序版本与UTC更新时间，版本不影响匹配或累计，普通/承载通过`carrier_exponent`和`modulus_kind`独立记录，算术位宽自动恢复，详见[算术路径字段](../architecture/AUTO_B2.md#toml-中的算术路径)。不产生`[sample.*]`或未访问的模型网格。兼容其他程序版本的format 8，并将旧条件级版本迁入行内；format 8以前的不兼容结构须重新生成，失败不覆盖原文件；队列进度和Stage1 save不受影响。控制台保留`ecm_tune_progress`等进度，内部诊断保存在证据日志。

`--tune ntt`可用`--length-log2 a:b`与`--tune-slices s,...`测指定长度和批量，或用等级1…10预设。输出分别记录每秒域卷积数和每秒批调用数；它们不是ECM曲线吞吐。计时边界、预算跳过和格式见[NTT tune](../architecture/AUTO_B2.md#ntt-tune)。

分批测量可用`--tune ecm --tune-merge A.toml --tune-merge B.toml --tune-file combined.toml`合并，再让`stage2_tune_profile`指向结果。同条件同完整参数按真实次数合并均值与样本标准差，不同参数保留多行；估算不能覆盖实测。可用`--tune-ntt-profile NTT.toml`导入组件。该命令不运行GPU测试，输出不能覆盖输入文件。[合并规则](../architecture/AUTO_B2.md#汇集预计算结果)。

生产用`--tune-profile FILE.toml`或`stage2_tune_profile`加载资料，不要求B1匹配。默认`stage2_prediction_policy=model`：普通按实际N位宽，梅森按p的算术位宽查询各自区间；精确实测优先，其他输入使用基础模型加区间校正。更改D和切换承载分别通过双侧误差保护，NTT/驻留边界及当前显存重新规划。设为`paired`或CLI `--prediction-policy paired`恢复精确位宽/B2/形状的配对实测门禁。

没有用户调优文件时也有内置4060 Laptop/1800MHz参考模型，输出标为builtin_model；范围外可以外推，但不承诺独立精度或未测性能。启动短校准默认保留，可用`--short-calibration 0`查询纯基础预测。邻近算术位宽可复用家族覆盖，不因梅森指数改变而必然再校准。条件忽略默认gpu、driver、cuda、backend、environment；算术、整除和显存检查始终保留。详见[模型与边界](../architecture/AUTO_B2.md#算术位宽区间预测)。

通用位宽网格可用`--tune ecm --tune-exponents 512,2048,4096,8192,12288,15360 --tune-arithmetic both --tune-b2 26000000000,260000000000 --tune-budget-seconds 1800`。它分别收集通用与梅森算术资料；真实队列默认仍用真实save配对。网格的梅森实际余因子位数可能低于算术位数，两者不会混为同一索引。

任务最终B2=0时，`--auto-b2`联合选择B2、D、承载。T1优先使用正的`--stage1-seconds-per-curve`，其次`--stage1-cost-csv FILE.csv`，最后`--stage1-tune-profile FILE.toml`。对应TOML为`stage1_seconds_per_curve`、`stage1_cost_csv`、`stage1_tune_profile`。CSV明确target_bits或container_bits，缺项和跨TPI可估算；`--stage1-cost-mhz`给目标频率，0使用表内参考频率。秒/曲线不会再除batch。Stage1 标量构造默认跟随 TOML 的 `lcm`，可用 `--lcm 1|12` 覆盖。现有成本证据文件中的 `lcm`、`choose12` 标签由程序在读取时映射。

`--auto-min-b2`/`--auto-max-b2`控制搜索边界，默认B1<B2≤2.6×10¹²，不限于已测范围。完整tune优先于旧`stage2_cost_profile`；Auto缺T1/完整成本或没有满足当前显存的候选时明确失败，保留队列。每条worker重新规划。模型精度与当前限制见[Auto B2/tune](../architecture/AUTO_B2.md)。

- [ecm_cuda_stage2_main.cpp](../../src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp#L175)：`records`；[队列字段](../../src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp#L395)：`queue_fields`。
- [Stage2 进度接口](../../include/ecm/io/stage2_checkpoint.h)与[存储实现](../../src/io/stage2_checkpoint.cpp)：原子进度和回执对账。
- [ecm_stage2_console.h](../../src/ecm/platform/ecm_stage2_console.h)：阶段和内存摘要。
- [Stage2 管线](../architecture/STAGE2.md)、[Auto B2](../architecture/AUTO_B2.md)、[构建](BUILD.md)。
