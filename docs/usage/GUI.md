# Windows 图形前端

窗口标题和启动输出显示公共程序名及版本；`ecm_gui.exe --version` 仅查询版本，不打开窗口或初始化 GPU 监控。每个 worker 输出窗只显示该工作程序启动横幅报告的实际版本，例如 `Worker 版本: ECM-ELF x.y.z`；工作程序尚未报告时显示“尚未报告”。GUI 不把自身版本当作 worker 版本。

`ecm_gui.exe` 使用 Dear ImGui docking、Win32/DX11，管理 Stage1 或 Stage2 worker、配置、队列生成、GPU监控、结果展示及ECM概率估算。概率面板使用CPU数值模型；实际曲线计算来自所选worker程序，GUI不链接GMP/OpenCL/CUDA引擎。

## Worker 与配置

一个 worker 对应一个计算进程，并有独立输出窗口和进程监管。Stage1 使用 `ecm_cuda.exe --config <file> --worker N`；Stage2 使用 `ecm_cuda_stage2.exe --config <file> --worker N`。配置文件默认为程序同目录的 `ecm.toml`。GUI 从共用 `worktodo` 按 `priority`、`skip` 选择下一行：`ECMSTAGE1=` 启动 Stage1，`ECMSTAGE2=` 启动 Stage2；同一 worker 同时只运行一个程序。

Worker 管理顶部设置 worker 数量、两个阶段的程序路径和共用 TOML。`[gui]` 中的 `exe` 指向 Stage1，`stage2_exe` 指向 Stage2，`NumWorkers` 控制 GUI worker 数量，`priority` 仅控制进程优先级。`[workers.N]` 保存第 N 个 worker 的 GUI 设置。Stage1/Stage2 键分别放在 `[stage1]`、`[stage2]`，共用键放在 `[shared]`；按 worker 覆盖时写入 `[workers.N.stage1]`、`[workers.N.stage2]` 或 `[workers.N.shared]`。配置文件中的 worker 表与 `worktodo.txt` 的 `[Worker #N]` 队列分区使用同一编号，但语法独立。

GUI 选择配置前检查文件并保存当前窗口设置，选择后不自动运行。语言、字体和布局保持当前会话设置；计算配置、worker 数量及程序路径从所选 TOML 读取。两阶段共用的 `worktodo` 相对 TOML 目录解析；Stage2 输出也基于 TOML 目录，Stage1 的日志和存档等相对路径基于 Stage1 程序目录。生效配置从工作程序同一投影读取。完整分区与每键覆盖资格见[配置参考](../ECM_TOML_REFERENCE.md)。

例如两个 worker 分别选择 Stage1 设备时，可在 `[stage1]` 写 `device = 0`，在 `[workers.2.stage1]` 写 `device = 1`；`worktodo.txt` 中任务仍使用 `[Worker #1]`、`[Worker #2]`。没有 worker 标题的队列行只归 worker 1。GUI 保存 TOML 时保留注释和顺序，并原子替换文件。

### 当前任务摘要

任务行显示数的表达式，例如`M12011 / 已知因子 | assignment ID`。因子字段非空时显示“已知因子”；存在真实AID时只显示固定字样“assignment ID”，不展示其值。空AID或N/A不添加该标记。不展示完整worktodo、save路径或曲线配置尾字段，悬停提示、输出窗口头部和详情使用同一摘要；无法解析时显示摘要不可用，不回退到原始任务行。

该规则仅用于任务摘要。日志窗口的`START:`行保留工作程序输出的原始worktodo，不替换真实因子或AID；原始队列、工作程序日志、原始输出和结果同样保留完整数据。Stage2任务身份由工作程序通过GUI事件传递，不依赖运行过程中可能被重排的队列文件；开始前可读取队列作预览。

## 队列生成与交接

worktodo 生成器解析 PrimeNet ECM/ECM2 任务，过滤、去重、排序、检查 save 名称，输出 `ECMSTAGE1=` 行，并按所选 Stage1 设备档位建议曲线批量。写入前展示预览，追加前重新核对目标队列大小/修改时间，避免覆盖外部编辑。目标 worktodo 已有 worker 标签时，新任务写入对应 worker 最后一个分段的末尾；没有对应标签时在文件末尾新建分段。目标文件没有 worker 标签时沿用文件末尾追加。

“曲线选项”每行两项：推荐曲线数与去重勾选框在第一行，块/SM 与目标显卡在第二行。“输出选项”提供覆盖 B1/B2、曲线类型、存档名格式及排序。曲线类型可选 Edwards（`e`）与 Montgomery param0/2/3（`m0`/`m2`/`m3`），初值来自所选 worker 的 Stage1 配置。默认存档名格式为 `{curve}_{target}_{b1}.save`：梅森任务的 target 为 `m<原始指数>`；其他任务为 `bit<估算的有效位数>`，后者需在涉及边界位数或已知因子时核对，生成器也会提示。`{b1}` 保留输入或覆盖值的原始写法；旧 `m{n}_{b1}.save` 仍可作为自定义格式使用，生成器会规范为所选曲线的前缀。Stage1 队列依文件名前缀选择构造方式；Stage2 仍以 save 内容的 METHOD/PARAM 为依据，详见[存档与队列覆盖](../architecture/SAVE.md)。排序可选输入顺序、单个支持字段或 `n,k,b,c` 组合。覆盖 B1 留空时保留原值；填入后预览和追加结果统一采用覆盖值。覆盖 B2 默认为 `0`，会覆盖输入任务原有的 B2；留空才保留原值。`0` 表示使用 Stage2 的 `stage2_b2`；该值也为零且启用 Auto B2 时才自动选取。修改任一选项后需要重新生成预览。

GPU 批量建议向实际 Stage1 程序查询所选曲线的编译档位：Montgomery 按 param 选择档位，Edwards 使用其独立档位及一位容器余量；曲线数由 blocks/SM、SM 数及每 block 实例数计算。Edwards 当前查询不提供实际占用率，推荐值仅按界面设置的块/SM 计算。CPU Stage1 目标自动关闭并禁用推荐曲线数，保留任务原有曲线数，也不提供 `m2`/`m3`；选择器据此禁用不可执行的组合。生成器不使用 GMP，剥离已知因子后的位宽为估计下界，精确整数仍由驱动校验。

生成器明确选择一个目标 worker，显示当前队列阶段、Stage1 设备及实际队列路径，生成的任务只写入该 worker 分区。即使当前任务正在 Stage2，生成器也依据该 worker 的 Stage1 配置产生 Stage1 行。切换目标 worker 或 TOML 会清除旧预览，需要重新生成。

Prime95 交接状态显示 pending、失败或成功；真实 save/worktodo.add 写入由驱动负责。GUI 不是 Prime95 活动队列的第二个消费者。

## 日志、结果与监控

管道进度用于百分比、速度和 ETA，事件行用于开始、因子、checkpoint 和错误。**Result** 面板直接只读展示 worker 发布的提交 JSON 行；正式内部 JSONL、提交文件及其账本由 CLI worker 生成，见[结果输出](ECM_RESULTS.md)。GUI 不生成因子摘要文件，不合并因子或 NF 曲线数，不改写 Result、checksum 或账本。

### Result

面板根据当前 TOML、worker 数量及每个 worker 的 `stage1_submission_file` / `stage2_submission_file` 查找文件。相对路径基于 TOML 目录；未显式设置时 worker 1 使用 `results.json.txt`，其余 worker 在最后一个扩展名前添加编号（例如 worker 2 为 `results.json_2.txt`）。显式路径不自动添加 worker 后缀。两阶段或多个 worker 指向同一个路径时只读取一次；不同来源文件中的记录分别保留。通过“添加文件”可查看其他目录、历史文件或 CLI `--submission` 指定的输出；手动添加的来源及筛选只保存在当前会话，切换配置或重建 worker 列表后重新按 TOML 定位。

状态（F/NF）、阶段（Stage1/Stage2）、来源文件和文本搜索在同一行并列，支持组合筛选。来源下拉框按文件数量展开，屏幕可用高度足够时显示全部条目，超出时滚动。搜索匹配原始 JSON，可查目标、因子、时间或 assignment ID。表格显示状态、目标、阶段、因子、B1/B2、曲线数、曲线模型、UTC 时间及来源。NF 阶段依据已发布的 B1/B2 判断；F 使用发布行的 stage。NF 未记录具体 Montgomery 参数化时只显示 Montgomery，不根据文件名推测。

“复制筛选结果”复制当前可见的全部记录；勾选行后可“复制所选结果”，仅复制当前筛选内的选中行。点击目标查看原始行并复制单条，右键目标也可复制。复制保留每行的原始 JSON 字节与 checksum，行间使用换行；文件内保持原发布顺序，多文件按来源列表顺序连接。筛选变化会清除选择，避免复制隐藏记录。

Result 面板可见时每 500 ms 检查文件大小和修改时间，仅重读有变化的文件，也可手动刷新。读取、解析与显示数据更新在 GUI 主线程同步执行，大文件重读可能短暂影响界面响应，不阻塞独立的计算 worker。完整 JSON 末行即使没有换行也可显示；尚未写完的末行等待后续更新，完整但无法解析或缺少提交字段的行跳过并在来源文件中提示。这里检查提交结构，不替代 worker 的数学和 CRC 校验。文件截短、替换或删除后更新显示，界面不会用旧记录替代新文件。旧的 `[gui] results_json`、`results_txt` 配置项已移除。

底部“Result 来源文件”折叠区列出完整路径、文件缺失及读取诊断。展开时自动将面板滚动到底部；只在展开时定位一次，之后可自由滚动。该折叠区与上方用于筛选的“来源”下拉框是不同控件。

### 进度与监控

因子事件用于 worker 命中计数、状态提示和高亮；正式 F/NF 展示只依据发布文件。Stage2 GUI 事件携带模型、sigma、B1、记录号和 save 路径，用于当前曲线身份与进度；安静模式启用 GUI 事件时也能更新状态。混合模型的连续曲线逐条更新身份，不继承上一条模型。

Stage2任务进度按**已完成曲线/可用曲线**显示，包括恢复的完成数；当前曲线编号、计算阶段、实际B1/B2/S位宽/D/P另行展示。`mersenne_carrier` 显示本曲线实际承载指数 p；0 表示未使用额外梅森承载，尚未收到引擎的模数信息时不显示。完整梅森数作为 N 使用快速归约时也属于“无额外承载”。大步点批次是当前曲线中一个阶段的进度，阶段剩余时间不作为整个任务的ETA。任务ETA用本轮已完成曲线的平均引擎时间估计，首次完成前不虚构速度或ETA。速度边界优先为`stage2_full_wall.total=init+main`，没有该输出才使用外层曲线墙钟。

Stage2支持启动短校准、存档校验、D/P与内存规划、小步点（baby points）、F乘积树、反转多项式逆、大步点（giant points）/G乘积树/模F折叠、F树下降、叶值分块/GCD、算术检查、因子分解及结果发布的阶段显示。详情和worker输出窗在当前阶段标题下说明该阶段的操作与产物；完整管线见[Stage2架构](../architecture/STAGE2.md)。正式 F/NF 发布行显示在 Result。GUI请求批次级控制台输出以取得状态，但常规输出窗口只保留阶段与曲线事件；形状、NTT统计和批次明细可从原始输出查看，工作程序文件日志仍由其配置维护。

工作进程管理按阶段使用青绿/紫色进度，长路径移到提示，选中worker显示详情。窄面板的表格保留进度列最小宽度并横向滚动，不强行把固定列挤入可视区域；条形与覆盖文字裁剪在同一矩形内，绘制比例约束为0到1。跨列任务摘要单独测量行高及绘制范围，不参与第一列宽度计算。日志提供“自动滚动到底部”开关、错误颜色和可见行绘制，保留原有4000行上限。开启时，首次输出和后续新增输出都会在原先停留底部时继续滚动；向上翻阅时暂停，回到底部恢复；关闭时不随新输出滚动。此设置只控制界面滚动。绘制机制减少长日志的逐帧工作量，没有据此声明已测CPU占用或GPU性能提升。

GPU 面板动态加载 NVML，显示利用率、显存、SM/graphics 时钟、功耗、温度和限频原因。驱动不支持的指标显示不可用；NVML 利用率不是 SM occupancy。监控不改变 GPU 功率/频率配置。

`[gui] refresh_hz` 默认 10 Hz，设置空闲时绘制与 worker 状态处理的目标频率；鼠标、键盘和窗口消息会提前唤醒界面。较低值节省 CPU，但图表与仅由 worker 产生的状态更新可能最多延后一个刷新周期，10 Hz 时约 100 ms。`[gui] gpu_poll_ms` 默认 500 ms，由独立线程控制 NVML 指标采样；即使提高界面刷新频率也不会增加采样次数。未选中的停靠页跳过内容布局与图表处理，NVML 采样仍持续。Prime95 待处理文件每 500 ms 检查一次，收到 worker 交接通知后立即重新检查。

NVML 采样时不持有 UI 读取设备/历史的锁；初始化仍同步，没有进程隔离或超时保证。`--trace` 可记录初始化与窗口恢复。

## ECM 概率面板

从“查看 → ECM 概率”打开或聚焦，也可设置`[gui] start_tab=prob`直接进入。面板作为worker管理旁的独立页签加入现有停靠布局；关闭后可从菜单重开。输入只保存在本次GUI会话中，不写入任务队列或worker计算设置。

输入目标素因子大小、B1、固定B2或B2/B1、曲线数C及曲线概率模型。大小可用bits或十进制位数，不是待分解完整N的位宽：b bits采用代表值2^(b−0.5)，d位十进制采用10^(d−0.5)。支持科学计数法。未启用Stage2时只计算Stage1。

上半部分分为两栏：左侧集中输入因子大小、曲线数C、B1、Stage2开关及固定B2或B2/B1；右侧设置计算方式、概率模型、D_eff、概率口径及目标成功率。

GMP理论预设包括Suyama/param0、param1、param2和param3，采用`tools/ecm_prob/model.py`及`rho.py`的理论有效除子；T=4/8/12/16只是扭转基线。自定义`D_eff`可以输入经验拟合值，但GUI不自动读取忽略目录中的小素数样本，也不把理论预设宣称为实测曲线命中率。`D_eff`与Stage2选型参数D无关。

结果同时给出单曲线、C条曲线累计成功率、未命中率及首次成功的平均曲线数1/p。累计成功率使用Q=1−(1−p)^C，假定各曲线独立且等效；1/p是均值，不能当作达到某个置信概率所需的曲线数。已有Stage1 save可参考条件概率(p₁₂−p₁)/(1−p₁)，表示已未命中Stage1的曲线在Stage2中的成功率；p₁=1时该条件概率无定义。

反算支持曲线数、因子大小、B1、B2及D_eff。目标可为单曲线或累计成功率；曲线数须使用累计目标，B2须启用Stage2。待求输入字段禁用。B1反算可固定B2或固定B2/B1；B2反算改为绝对界限。条件Stage2概率对B1不单调，因此不能按该口径反算B1。结果展示采用参数、实际概率及目标残差；“将反算结果用于输入”会把求得参数填回正向计算表单。

当前反算使用有限范围：因子大小2..2048 bits，B1/B2为2..10¹⁸，D_eff为10⁻⁹..10¹⁸，曲线数不超过10¹⁸。先检查端点，再做80次区间搜索；B1/B2结果向上取整。最小允许界限已达到目标时直接返回边界，目标在模型范围内不可达时显示无解。有限Dickman表及数值截断给出的零概率不表示严格不可能命中。

三张图分别展示因子大小与单曲线概率、因子大小与C条曲线累计成功率、曲线数与累计概率。第二张图固定本次结果的C，并在标题中显示该数值；沿因子大小变化计算Q=1−(1−p)^C，使用与第一张图相同的大小范围。Stage1与完整流程始终可比较，条件Stage2曲线可另行显示；选择该条件口径时自动展示对应曲线。每图使用129点，可悬停读取采样值，虚线标记结果所用参数。支持自定义范围、扩大曲线数图以包含结果、对数概率轴及复制三张图的CSV；CSV中的`factor_size_cumulative`标识固定C的新增图。对数轴跳过零值，条件概率无定义处留空，不伪造零值。

首次打开面板计算默认参数；之后修改输入需点击“计算”。后台CPU任务使用输入快照，修改中的参数标记为待计算；图表与结果摘要均对应已完成的快照。现有概率核心不计入Brent–Suyama或额外Stage1×12收益。当前仅完成构建及接线检查，新增反算数值、边界及界面效果仍需运行验证。

## 窗口与退出

启动检查普通窗口矩形及标题栏是否位于可用显示器；异常尺寸、最小化哨兵或屏幕外位置恢复到工作区，并重建停靠布局。退出保存正常窗口位置，最小化时保留有效布局，避免下次只有任务栏图标。

停止/关闭通过继承的事件请求安全退出：Stage1在安全边界写检查点；Stage2完成当前曲线、发布结果并记录完成数，再停止启动下一条。Stage2不写中间算术存档，也不把曲线计数文件当成Stage1检查点。待自动重启的worker可以直接取消重启。

GUI等待进程自行退出，超过`graceful_stop_ms`仍未退出则强制终止；默认300秒，应覆盖当前完整Stage2曲线的最坏用时，可在TOML调大。用户也可在退出等待界面强制退出，未完成的Stage2曲线会在下次运行时重做，已完成曲线按原生队列进度恢复。Stage1检查点规则见[Stage1使用](STAGE1.md#检查点与交接)，Stage2见[自动续跑](STAGE2.md#自动续跑与退出)。

混合阶段退出提示分别说明Stage1保存检查点和Stage2完成当前曲线；每个worker使用自己的工作目录及停止协议。Prime95交接提示在存在Stage1 worker时保留，并显示两种阶段的运行数量。

Stage2 GUI协议需要支持`--stage2-stop-handle`及`--gui-events`的当前工作程序，GUI与该程序须一起更新。当前验证范围为构建通过；实际GPU运行、交互、显示布局和退出恢复仍需运行验证。

## 代码入口

- [main_win32.cpp](../../src/gui/main_win32.cpp#L87)：窗口矩形验证、恢复和保存。
- [app.cpp](../../src/gui/app.cpp)、[worker_proc.cpp](../../src/gui/worker_proc.cpp)：界面和 worker 生命周期。
- [worktodo_gen.cpp](../../src/gui/worktodo_gen.cpp)、[results.cpp](../../src/gui/results.cpp)、[log_parse.cpp](../../src/gui/log_parse.cpp)：生成器、结果和进度。
- [task_display.h](../../src/gui/task_display.h)：只用于显示的任务摘要与AID/因子替代；`log_parse.cpp::parse_stage2/Stage2Status`汇总阶段及完成曲线。
- [probability_panel.cpp](../../src/gui/probability_panel.cpp)：`ProbabilityPanel::draw`、后台输入快照及概率图；[probability_model.cpp](../../src/gui/probability_model.cpp)：`evaluate`、`calculate`、反算及CSV。数值核心复用[ecm_stage2_probability.h](../../src/core/ecm_stage2_probability.h)。
- [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp)：GUI任务事件和继承的安全停止请求，队列/存档仍由原生驱动处理。
- [gpu_monitor.cpp](../../src/gui/gpu_monitor.cpp)、[toml_file.cpp](../../src/gui/toml_file.cpp)：监控和配置。
- 构建入口：`tools/build/dev/build_gui.ps1`；[构建说明](BUILD.md)。
