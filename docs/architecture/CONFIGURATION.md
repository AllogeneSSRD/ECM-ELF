# 配置定义与解析

## 单一数据源

[config/ecm_options.json](../../config/ecm_options.json) 定义键名、所属表、类型、默认值、取值范围和双语说明。TOML 键名与 C++ 内部绑定名独立。修改定义后，依次运行 `tools/gen/generate_ecm_config.py`、`tools/gen/generate_gui_v2_state.py` 和 `tools/gen/generate_gui_v2_localization.py`。发布构建同步生成物，`-SkipBuild` 检查现有文件。CMake 用一致性清单核对输入和生成物。[业务构建检查](../../tools/build/internal/check_ecm_config.cmake) · [GUI v2 构建检查](../../tools/build/internal/check_gui_v2_config.cmake)

程序新建文件时使用[默认模板](../../config/ecm.toml.default)，其中列出常用设置。[完整示例](../../config/ecm.toml.example)列出所有配置键，高级选项默认注释。每行键值后用 `#` 标出类型和范围，下方是双语说明。取消行首注释会让该项成为显式设置。

配置定义还用于生成 C++ 绑定、[默认值头](../../src/ecm/platform/generated/configuration_defaults.h)、一致性清单和逐键参考。[中文](../ECM_TOML_REFERENCE.zh-CN.md) · [English](../ECM_TOML_REFERENCE.en.md)

## Worker CPU 资源

`[stage1] stage1_cpu_threads` 控制 CPU 算术及其准备工作，`[stage1] gpu_init_threads` 控制 Edwards CUDA NTT 和 CUDA/OpenCL 一线程一曲线路径的 CPU 曲线构造。两项 `0` 均由各后端决定。GPU 构造线程受可用 CPU、曲线数和适用的初始化内存预算约束。`[stage2] stage2_cpu_threads` 只控制 CPU Stage2，默认 1。三项都可独立在 `[worker.N]` 覆盖。

`[shared] cpu_affinity` 或 `[worker.N] cpu_affinity` 为同一 worker 的两个阶段设置公共 CPU 集合，包括 GPU 的主机工作和 Stage2 曲线子进程。写法为逻辑 CPU 编号和闭区间，例如 `"0,2,4-7"`。去重后应用集合，空串、`auto` 或 `none` 按系统现有调度运行。Windows 单处理器组使用进程 affinity mask，多组使用默认 CPU Sets，编号为 `组号 × 64 + 组内编号`。Windows 系统或外部约束的 affinity mask 优先于 CPU Sets。Linux 在创建计算线程前应用调度掩码，由计算线程继承。不可用的 CPU 编号会停止启动。应用的集合写入 `worker_cpu_affinity` 启动输出。见 [Windows CPU Sets](https://learn.microsoft.com/en-us/windows/win32/procthread/cpu-sets)。

`--stage1-threads` 同时覆盖两项 Stage1 线程设置，`--cpu-threads` 和 `--gpu-init-threads` 分别覆盖。Stage1 CLI 可用 `--affinity` 设置旧式逐线程亲和性。

## 配置版本

主配置在首个表头前声明整数 `config_version = 4`。已有配置可按[完整示例](../../config/ecm.toml.example)重新整理。更改版本号的同时须使用 v4 的表和键名。工作程序与 GUI 使用同一份配置。

### 升级旧配置

从仓库根目录用 Python 3.11 或更高版本转换现有 v3 文件。`--gui-state` 可选导入旧 GUI 偏好，原文件位于 Windows 的 `%LOCALAPPDATA%\ECM-ELF\gui-v2\`。主配置已有的显式值优先，包含已指定的程序路径。

```powershell
python tools/refactor/migrate_ecm_config.py ecm.toml --preview
python tools/refactor/migrate_ecm_config.py ecm.toml --gui-state old-gui-state.toml
```

默认在原目录生成 `ecm.v4.toml`，保留相对路径语义。检查 `[gui]` 的程序路径后，可以通过新版 GUI 和 worker 的 `--config ecm.v4.toml` 启动。

## TOML 分区

- `[shared]` 供两阶段共用，`[stage1]` 和 `[stage2]` 各自管理本阶段的设置。
- `[gui]` 存放界面偏好、布局和计算程序路径。`[worker.N]` 存放第 N 个 worker 的启动设置和覆盖值，N 为 1…1000000。
- 每个键名在整个配置定义中唯一。覆盖时把同一个键写在 `[worker.N]`，例如用 `handoff_mode = "off"` 覆盖 `[stage1]` 的同名设置。点号只用于写表头中的 worker 编号。
- `[shared]` 的队列、跳过规则和调优文件可按 worker 覆盖，同时作用于两个阶段。工作流 worker 数量、共享日志、滚动上限和完成记录是全局配置。

键名和表名区分大小写。键只能放在所属表及允许的 worker 表。运行选项依次合并内置默认值、全局表、当前 worker 覆盖和允许的命令行选项。曲线构造等任务字段采用任务及存档规则，Stage2 的 [B2 优先级](../usage/STAGE2.md) 单独定义。`worktodo.txt` 中的任务仍按 `[Worker #N]` 分区。

[config_format.h](../../src/io/config_format.h) 提供共享 TOML 文法及标量解析，支持带引号的字符串、布尔值、整数、浮点数及窗口整数数组，允许 `#` 行注释和行尾注释。GUI 的 [TOML 编辑器](../../src/ecm/application/gui/toml_file.cpp) 保留未修改行、注释及顺序，改写值时按类型编码，再以临时文件原子替换并保留一份 `.bak`。配置文件默认为程序目录的 `ecm.toml`，显式路径使用 `--config`。不读取旧 `ecm.ini` 格式。

布尔键及其 worker 覆盖接受 `true`、`false`，也接受 `1`、`0`。带自动模式的布尔键还接受 `"auto"`。读取器将数字别名转换为布尔值，同时保留原行以定位错误。GUI 修改布尔键时写入 `true` 或 `false`，未编辑的行保留原来的写法。

## GPU 后端与 worker

统一 Stage2 入口读取当前 worker 的 TOML 快照，先用 `[stage2] device` 选择 CPU 或 GPU，再把同一快照交给对应参数解析器。默认 `auto` 使用 CPU。CPU 支持 Edwards，Montgomery 可在配置中指定 GPU。GPU 曲线子进程直接沿用父进程已确定的 GPU 路径。Stage1 仍由自己的 `target` 选择 CPU 或 GPU。

Windows 的 GPU 路径读取 `stage1_gpu_backend` 或 `stage2_gpu_backend`。两者均接受 `"auto"`、`"cuda"` 和 `"opencl"`，默认 `"auto"`。Stage1 程序为 `ecm_cuda.exe` 和 `ecm.exe`，Stage2 为统一的 `ecm_stage2.exe` 和 `ecm_opencl_stage2.exe`。两者同目录时，auto 优先使用对驱动可见的 CUDA 设备，否则选 OpenCL。GPU 序号依实际后端解释。可在 `[worker.2]` 写 `device = "gpu"`、`stage2_gpu_backend = "opencl"` 和 `stage2_gpu_device = 1` 仅配置该 worker。跨后端切换保留标准流、工作目录、GUI 停止句柄及退出码，由实际执行程序维护队列和结果。

OpenCL Stage2 支持固定 B2、基本 D 规划和显式梅森承载；CUDA tune、Auto B2 及 CUDA 专用内核参数的支持范围见 [OpenCL 生产入口](OPENCL.md#生产-stage2-入口)。

## 按程序职责读取

程序按键名确定所属表，再应用全局和当前 worker 的设置。覆盖项沿用其所属表的校验规则。

- Stage1 校验本 worker 的 `shared` 和 `stage1`。自动交接可用 `[shared] worker_count` 确定候选范围。
- Stage2 校验 `shared` 和 `stage2`。设备、save 目录、日志级别和标量倍数仍可按既有规则从 Stage1 继承。程序先确定实际需要继承哪些字段，再按 worker 优先级读取并校验。队列模式还检查两阶段的完成记录路径是否冲突。
- GUI 校验 `shared`、`gui` 和 worker 启动设置。能识别的计算项用于预览，其他项保留原文。计算程序启动时读取同一份 TOML，校验自己负责的参数。

Stage1 向 Stage2 移交任务时，从启动快照读取接收队列和共用 `finished_file`。同队列自动交接只选实际读取该队列并允许 Stage2 的 worker。配置给出工作流 worker 数量时，GUI v2 使用相同的数量。

GUI 保存时保留未编辑的行、注释、顺序和未知值。编辑已知键时校验本次修改。预览使用的默认值只用于显示。

全文件检查分区和结构。未知分区、重复分区或键、缺少右值、未闭合字符串和非法行结构仍报错。外阶段及其他 worker 的值可保留原文，业务校验由消费该值的程序负责。选中实际执行后端后再启用该后端的预设；读取任务及存档后校验曲线或参数专用的组合约束。

代码入口：[config_format.h](../../src/io/config_format.h)的`TomlReadMode`／`deferred_toml`决定职责，`selected_entries`仅校验实际继承或移交字段；[config.cpp](../../src/io/config.cpp)分别实现Stage1解析和Stage2继承投影；GUI的[TomlFile](../../src/ecm/application/gui/toml_file.h)提供计算预览，[设置编辑](../../src/ecm/application/gui_v2/views/pages.cpp)只检查本次修改。CPU Edwards Stage2与CUDA／OpenCL Stage2均使用Stage2职责。

GUI v2 的配置加载和保存使用主文件中的 `[gui]`。工作程序和界面因此从同一路径读配置，界面保存时保留计算设置、未编辑行和注释。

Stage1 的 [cli.cpp](../../src/io/cli.cpp) 一次转换每个显式选项并保留来源；别名按最后出现生效，非法的较早值不会因覆盖而消失。[config.cpp](../../src/io/config.cpp) 一次读取 TOML 并校验Stage1／共享显式值，保留分区、源行与 typed scope，不补任务默认或改变运行时单例。生成绑定同时提供赋值和已解析字段复制，worker 投影不再往返文本。

[normalize.cpp](../../src/io/normalize.cpp) 的 `normalize_settings()` 从生成默认开始，应用全局、worker 和显式 CLI，生成启动设置；main 安装一次 runtime。`normalize_direct()` 与 `normalize_worktodo()` 分别接收该设置，形成拥有自身数据的冻结任务。启动不回填 CLI，任务循环不重读 TOML；Stage2 接收端投影复用原始启动快照，worktodo 仍每轮重读。Stage2 保留其现有编排。

`checkpoint_backups` 接受 0…16。负数 `checkpoint_interval_seconds` 禁用定期存档。重复指定 sigma 时，后一个值及其可选参数化声明一并生效。独立指定的 `--gpu-param` 仍需与 sigma 一致。

CLI 或当前 worker 配置错误会停止运行。TOML 诊断由 [config_format.h](../../src/io/config_format.h) 的 `config_diagnostic()` 排版，报告路径、行号、worker、键、原始右值及允许值。任务记录本身有误时，队列将该行标为 `# ERROR`。当前配置或后端无法执行合法任务时，队列保留该行，待调整配置后可重试。队列中的任务结构字段由任务提供，sigma 采用单独优先级。见 [Stage1 输入与默认](../usage/STAGE1.md#输入归一化与运行中编辑)。

## 运行时限制

读取配置时检查静态类型和范围。设备、路径冲突、save 与任务是否匹配、D 和 NTT 形状及 Auto B2 的适用条件在实际执行时检查。

- 以 TOML 所在目录为基准解析队列、自动移交目标、结果及提交路径，以及 Stage2 接收端的 `finished_file`。
- 以程序目录为基准解析 Stage1 配置中的 `save_dir`、完成记录、日志、备份及 Prime95 路径，也用于队列行中的相对 save 路径。
- 单次运行的 CLI save 和临时目录以当前工作目录为基准。临时目录省略或留空时使用当前目录。显式的相对 `exponent_cache_dir` 也以当前目录为基准。留空时，仓库内使用根目录的 `cache/`，独立部署使用可执行文件同级的 `cache/`。

Stage1 和 GPU Stage2 按 TOML 目录解析共享的 `log_file`（默认 `ecm.log`）和 `finished_file`（默认 `worktodo.finished.txt`）。每条文件日志以 `[Worker #N] [YYYY-MM-DD HH:MM:SS]` 开头，使用本地时间。活动日志超过 `log_file_max_kb` 后归档，编号取同目录已有归档的最大编号加一。Stage1 的 `--no-log-timestamp` 只关闭控制台时间戳。Stage2 控制台和调优试跑的诊断格式沿用各自的输出协议。

配置路径以程序目录为基准。队列模式缺少配置时创建模板。单次运行缺少默认配置时使用内置默认值，显式指定的文件缺失时报错。`--gpu-info` 缺少配置时沿用查询默认值。各键取值见[配置参考](../ECM_TOML_REFERENCE.zh-CN.md)。
