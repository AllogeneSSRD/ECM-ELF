# Windows 图形前端

`ecm_gui.exe` 使用 Dear ImGui docking、Win32/DX11，管理 Stage1 或 Stage2 worker、配置、队列生成、GPU监控及结果展示。GUI不执行数论算法，也不链接GMP/OpenCL/CUDA引擎；计算能力来自所选worker程序。

## Worker 与配置

一个worker对应一个计算进程，并有独立输出窗口和进程监管。Stage1使用`ecm_cuda.exe -ini <file> --worker N`；Stage2使用`ecm_cuda_stage2.exe --ini <file> --worker N`。INI/worktodo的`[Worker #N]`使用同一编号。同一个GUI可以同时管理不同阶段的worker，各自使用同一INI中的计算设置和自己的编号。阶段切换不自动启动计算，也不自动串接两个队列。

Worker管理顶部设置默认阶段、worker数量、两个阶段的程序路径和共用INI。`[GUI] stage=[auto|stage1|stage2]`默认auto；auto识别明确命名的Stage2程序，其他情况下优先可用Stage1，再回退Stage2。`[Worker #N] stage=[auto|stage1|stage2]`覆盖默认阶段；auto继承GUI默认值。每行显示实际阶段，星号表示继承默认值。只需停止要修改的worker，即可在其行尾切换阶段；其他worker保持运行和进度。修改默认阶段、worker数量、共享程序或INI须全部worker停止。

`[GUI] exe`用于Stage1或既有明确命名Stage2路径，`stage2_exe`用于Stage2。Stage2默认查找GUI同目录程序、唯一`cuda-stage2*`子包、PATH；存在多个子包时明确选择程序。每种阶段共用一个程序路径，阶段选择不改变直接从命令行启动的程序，也不替代GPU架构支持检查。

使用现有独立Stage2目录时，选择其中的`ecm.ini`；该目录只有Stage2程序而没有Stage1程序时自动选择Stage2。选择INI前检查参数并保存当前窗口设置，选择后不自动运行。当前窗口的语言、字体和布局保持，计算配置、worker数量及程序设置从所选INI读取。INI及实际队列路径在管理/详情面板展示。

Stage2读取`stage2_worktodo`、`stage2_log_file`等独立键，并按Stage2规则相对于INI目录解析路径；`stage2_device`优先于公共device。Stage1读取`worktodo`、`log_file`等键，相对路径基于Stage1程序目录。共享INI与两个程序放在同一目录即可统一工作目录，save无需搬移。GUI显示的生效设置从命令行使用的同一配置投影读取。配置共用规则见[Stage2使用](STAGE2.md)；默认值及范围见[INI参考](../ECM_INI_REFERENCE.md)。

GUI专用设置位于`[GUI]`，worker覆盖位于相应分区。公共键放在任何分区前；共享设置和`[GUI]`全部放在worker分区之前。命令行的普通分组标题不重置worker范围，因此不能把共享`[stage2]`或`[GUI]`放在`[Worker #N]`之后。推荐用注释对公共配置分组。

例如，`[GUI] NumWorkers=2`，`[Worker #1] stage=stage1, device=0`，`[Worker #2] stage=stage2, stage2_device=1`；这里的逗号仅分隔说明项，实际INI每行一个键。两者分别消费`worktodo.txt`和`stage2_worktodo.txt`，两个队列使用相应`[Worker #1]`或`[Worker #2]`标题。没有worker标题的队列只归worker 1；同一GUI可为不同worker配置不同队列，Stage2同一队列文件仍只允许一个消费者。读写保留注释、未知键和顺序，保存使用原子替换。全部默认/范围见 [INI 参考](../ECM_INI_REFERENCE.md)。

### 当前任务摘要

任务行显示数的表达式，例如`M12011 / 已知因子 | assignment ID`。因子字段非空时显示“已知因子”；存在真实AID时只显示固定字样“assignment ID”，不展示其值。空AID或N/A不添加该标记。不展示完整worktodo、save路径或曲线配置尾字段，悬停提示、输出窗口头部和详情使用同一摘要；无法解析时显示摘要不可用，不回退到原始任务行。

该规则仅用于任务摘要。日志窗口的`START:`行保留工作程序输出的原始worktodo，不替换真实因子或AID；原始队列、工作程序日志、原始输出和结果同样保留完整数据。Stage2任务身份由工作程序通过GUI事件传递，不依赖运行过程中可能被重排的队列文件；开始前可读取队列作预览。

## 队列生成与交接

生成器解析 PrimeNet ECM/ECM2 任务，过滤、去重、排序、检查 save 名称，并按所选设备档位建议曲线批量。写入前展示预览，追加前重新核对目标队列大小/修改时间，避免覆盖外部编辑。目标 worktodo 已有 worker 标签时，新任务写入对应 worker 最后一个分段的末尾；没有对应标签时在文件末尾新建分段。目标文件没有 worker 标签时沿用文件末尾追加。

“曲线选项”每行两项：推荐曲线数与去重勾选框在第一行，块/SM 与目标显卡在第二行。“输出选项”的覆盖 B1 与存档名在第一行，排序下拉框在第二行。排序可选输入顺序、单个支持的字段或 `n,k,b,c` 组合。“覆盖 B1”输入框留空时保留各任务原有 B1；填入数值时，预览和追加结果统一使用该 B1。Stage2 工作项从 save 文件名读取 B1，因此生成器会把覆盖值写入默认的 `m{n}_{b1}.save` 名称；自定义存档名仍须包含可解析的 B1。修改任一生成选项后需要重新生成预览。

批量建议基于 blocks/SM、SM 数及每 block 实例数，不等于填满所有理论寄存器槽位。生成器不使用 GMP，剥离已知因子后的位宽为估计下界，精确整数仍由驱动校验。

生成器明确选择一个目标worker，显示该worker的阶段、设备及实际队列路径，生成的任务只写入该worker分区，不因其他worker使用相同GPU而分配给它们。选择Stage2 worker时默认关闭Stage1 GPU批量推荐，保留任务曲线数量；需要时仍可显式开启并从Stage1程序读取档位。实际可用save曲线由Stage2驱动检查。切换目标worker、阶段或INI会清除旧预览，需要重新生成。

Prime95 交接状态显示 pending、失败或成功；真实 save/worktodo.add 写入由驱动负责。GUI 不是 Prime95 活动队列的第二个消费者。

## 日志、结果与监控

管道进度用于百分比、速度和 ETA，事件行用于开始、因子、checkpoint 和错误。结果 JSONL 为原始结果，文本汇总合并同因子的 sigma；两者不可互当 checkpoint。

Stage2任务进度按**已完成曲线/可用曲线**显示，包括恢复的完成数；当前曲线编号、计算阶段、实际B1/B2/S位宽/D/P另行展示。巨步批次是当前曲线中一个阶段的进度，阶段剩余时间不作为整个任务的ETA。任务ETA用本轮已完成曲线的平均引擎时间估计，首次完成前不虚构速度或ETA。速度边界优先为`stage2_full_wall.total=init+main`，没有该输出才使用外层曲线墙钟。

Stage2支持启动短校准、读save、规划、baby点、F树、逆、giant/G树/fold、下降、叶乘积/GCD、算术检查、因子分解及结果发布的阶段显示。正式因子结果进入已有结果汇总。GUI请求批次级控制台输出以取得状态，但常规输出窗口只保留阶段与曲线事件；形状、NTT统计和批次明细可从原始输出查看，工作程序文件日志仍由其配置维护。

工作进程管理按阶段使用青绿/紫色进度，长路径移到提示，选中worker显示详情。窄面板的表格保留进度列最小宽度并横向滚动，不强行把固定列挤入可视区域；条形与覆盖文字裁剪在同一矩形内，绘制比例约束为0到1。跨列任务摘要单独测量行高及绘制范围，不参与第一列宽度计算。日志提供“自动滚动到底部”开关、错误颜色和可见行绘制，保留原有4000行上限。开启时，停留在底部会随新输出滚动；向上翻阅时暂停，回到底部恢复；关闭时不随新输出滚动。此设置只控制界面滚动。绘制机制减少长日志的逐帧工作量，没有据此声明已测CPU占用或GPU性能提升。

GPU 面板动态加载 NVML，显示利用率、显存、SM/graphics 时钟、功耗、温度和限频原因。驱动不支持的指标显示不可用；NVML 利用率不是 SM occupancy。监控不改变 GPU 功率/频率配置。

NVML 采样时不持有 UI 读取设备/历史的锁；初始化仍同步，没有进程隔离或超时保证。`--trace` 可记录初始化与窗口恢复。

## 窗口与退出

启动检查普通窗口矩形及标题栏是否位于可用显示器；异常尺寸、最小化哨兵或屏幕外位置恢复到工作区，并重建停靠布局。退出保存正常窗口位置，最小化时保留有效布局，避免下次只有任务栏图标。

停止/关闭通过继承的事件请求安全退出：Stage1在安全边界写检查点；Stage2完成当前曲线、发布结果并记录完成数，再停止启动下一条。Stage2不写中间算术存档，也不把曲线计数文件当成Stage1检查点。待自动重启的worker可以直接取消重启。

GUI等待进程自行退出，超过`graceful_stop_ms`仍未退出则强制终止；默认300秒，应覆盖当前完整Stage2曲线的最坏用时，可在INI调大。用户也可在退出等待界面强制退出，未完成的Stage2曲线会在下次运行时重做，已完成曲线按原生队列进度恢复。Stage1检查点规则见[Stage1使用](STAGE1.md#检查点与交接)，Stage2见[自动续跑](STAGE2.md#自动续跑与退出)。

混合阶段退出提示分别说明Stage1保存检查点和Stage2完成当前曲线；每个worker使用自己的工作目录及停止协议。Prime95交接提示在存在Stage1 worker时保留，并显示两种阶段的运行数量。

Stage2 GUI协议需要支持`--stage2-stop-handle`及`--gui-events`的当前工作程序，GUI与该程序须一起更新。当前验证范围为构建通过；实际GPU运行、交互、显示布局和退出恢复仍需运行验证。

## 代码入口

- [main_win32.cpp](../../src/gui/main_win32.cpp#L87)：窗口矩形验证、恢复和保存。
- [app.cpp](../../src/gui/app.cpp)、[worker_proc.cpp](../../src/gui/worker_proc.cpp)：界面和 worker 生命周期。
- [worktodo_gen.cpp](../../src/gui/worktodo_gen.cpp)、[results.cpp](../../src/gui/results.cpp)、[log_parse.cpp](../../src/gui/log_parse.cpp)：生成器、结果和进度。
- [task_display.h](../../src/gui/task_display.h)：只用于显示的任务摘要与AID/因子替代；`log_parse.cpp::parse_stage2/Stage2Status`汇总阶段及完成曲线。
- [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp)：GUI任务事件和继承的安全停止请求，队列/存档仍由原生驱动处理。
- [gpu_monitor.cpp](../../src/gui/gpu_monitor.cpp)、[ini_file.cpp](../../src/gui/ini_file.cpp)：监控和配置。
- 构建入口：`tools/build/dev/build_gui.ps1`；[构建说明](BUILD.md)。
