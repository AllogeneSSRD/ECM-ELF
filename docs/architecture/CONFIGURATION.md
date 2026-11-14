# 配置定义与解析

## 单一数据源

[config/ecm_options.json](../../config/ecm_options.json) 定义各分区的键、类型、默认值、取值范围和双语说明。`v2_key` 与 `v2_conversion` 描述新版路径和旧值转换，C++ 继续使用稳定的绑定名。修改定义后，依次运行 `tools/gen/generate_ecm_config.py`、`tools/gen/generate_gui_v2_state.py` 和 `tools/gen/generate_gui_v2_localization.py`。发布构建会同步生成物，`-SkipBuild` 只检查现有文件。CMake 用一致性清单检查输入和生成物。[业务构建检查](../../tools/build/internal/check_ecm_config.cmake) · [GUI v2 构建检查](../../tools/build/internal/check_gui_v2_config.cmake)

程序新建文件时使用[默认模板](../../config/ecm.toml.default)，其中只列常用设置。[完整示例](../../config/ecm.toml.example)列出所有配置键，高级和兼容选项默认注释。每行键值后用 `#` 标出类型和范围，下方是自动换行的双语说明。取消行首注释会让该项成为显式设置。即使值与内置默认值相同，也可能改变日志后缀、路径或环境变量的优先级。

配置定义还用于生成 C++ 绑定、[默认值头](../../src/ecm/platform/generated/configuration_defaults.h)、一致性清单和逐键参考。旧名和 worker 覆盖见对应语言的参考页。[中文](../ECM_TOML_REFERENCE.zh-CN.md) · [English](../ECM_TOML_REFERENCE.en.md)

## 升级旧配置

新文件在首个表头之前写根级整数 `config_version = 3`。读取 v1、v2 时沿用该版的键、取值和路径规则。无版本标记的文件按 v1 读取。版本未知、声明重复或放错分区时，读取器报告文件及行号。

从仓库根目录执行以下命令预览转换。移除 `--preview` 后，工具默认写入同目录的 `ecm.v3.toml`。使用 Python 3.11 或以上版本。旧日志或完成文件有多个目标时，使用 `--log-file`、`--finished-file` 选定共享目标。需要解析相对的旧 Stage1 路径时提供 `--stage1-directory`。旧配置同时启用两路交接时，使用 `--handoff-mode` 选定新模式。

CUDA 两阶段共用 `[shared] tune_file`，worker 可通过 `shared.tune_file` 为两个阶段一并覆盖。默认使用配置目录的 `tune.toml`。旧配置在同一 worker 上选择了不同的路径时，迁移会报告两阶段的有效值。`--tune-file PATH` 可为所有 worker 指定一个共用路径，原调优数据文件保留在原处。

```powershell
python tools/refactor/migrate_ecm_config.py ecm.toml --preview
python tools/refactor/migrate_ecm_config.py ecm.toml
```

确认新版 GUI 和 worker 能加载新文件后，可用 `--config ecm.v3.toml` 切换。工具会保留原文件，并在同目录导出以保留相对路径语义。需要给 v2 程序转换 v1 文件时，使用 `--to 2`。

## TOML 分区

- `[shared]` 供两阶段共用，`[stage1]` 和 `[stage2]` 各自管理本阶段的设置。
- `[gui]` 存放界面全局设置。`[worker.N]` 存放第 N 个 worker 的启动设置和覆盖值，N 为 1…1000000。
- 专用选项在所属表中使用点号，例如 `[stage1]` 下的 `handoff.mode`。在 `[worker.2]` 下写 `stage1.handoff.mode` 可覆盖该项。`[shared]` 的日志、滚动上限和完成记录供所有 worker 共用。

键名和分区名区分大小写。一个键仅能放在其定义的分区或相应 worker 覆盖分区；同表重复键、未知表，以及当前读取程序负责字段中的未知键或类型错误，会报告文件名和行号。内置默认值 < 全局分区 < 对应 worker 分区 < 命令行。`worktodo.txt` 的 `[Worker #N]` 是独立的队列语法，不是 TOML 配置分区。

[config_format.h](../../src/io/config_format.h) 提供共享 TOML 文法及标量解析，支持带引号的字符串、布尔值、整数、浮点数及窗口整数数组，允许 `#` 行注释和行尾注释。GUI 的 [TOML 编辑器](../../src/ecm/application/gui/toml_file.cpp) 保留未修改行、注释及顺序，改写值时按类型编码，再以临时文件原子替换并保留一份 `.bak`。配置文件默认为程序目录的 `ecm.toml`，显式路径使用 `--config`。不读取旧 `ecm.ini` 格式。

布尔键及其 worker 覆盖接受 `true`、`false`，也接受 `1`、`0`。带自动模式的布尔键还接受 `"auto"`。读取器将数字别名转换为布尔值，同时保留原行以定位错误。GUI 修改布尔键时写入 `true` 或 `false`，未编辑的行保留原来的写法。

## GPU 后端与 worker

Windows 下，两阶段各自设置 `gpu_backend`，取值为 `"auto" | "cuda" | "opencl"`，默认 `"auto"`。在 `[worker.2]` 写 `stage2.gpu_backend = "opencl"` 和 `stage2.gpu_device = 1`，只改变 worker 2 的 Stage2。命令行 `--gpu-backend` 优先于 TOML。Stage1 的 `target = "cpu"` 直接走 CPU。

将同一版本的两个后端放在同目录；Stage1 文件名为 `ecm_cuda.exe`、`ecm.exe`，Stage2 为 `ecm_cuda_stage2.exe`、`ecm_opencl_stage2.exe`。只有一个已安装时 auto 使用该程序。两者均已安装时，auto 先检查指定编号的 CUDA 设备是否对驱动可见，其次选 OpenCL。设备序号属于实际后端，混合厂商系统应按各自的设备清单设置 worker。显式选择缺少的程序会在计算前报错。后端在任务开始前确定，计算、校验、预算或驱动错误沿原路径报告；任务没有隐式重跑。跨后端切换保留命令行、工作目录、标准流、GUI 停止句柄和进程退出码。队列、曲线记录和结果仍由实际计算程序维护。

OpenCL Stage2 支持固定 B2、基本 D 规划和显式梅森承载；CUDA tune、Auto B2 及 CUDA 专用内核参数的支持范围见 [OpenCL 生产入口](OPENCL.md#生产-stage2-入口)。

## 按程序职责读取

程序先按所属分区和 worker 识别设置，再查找字段。两阶段的 `gpu_device` 各有自己的字段。覆盖项沿用其所属分区的校验规则。

- Stage1 校验 `shared`、`stage1` 及路由用的 `gui.worker_count`（v1 为 `gui.NumWorkers`）。
- Stage2 校验 `shared` 和 `stage2`。设备、save 目录、日志级别和标量倍数仍可按既有规则从 Stage1 继承。程序先确定实际需要继承哪些字段，再按 worker 优先级读取并校验。队列模式还检查两阶段的完成记录路径是否冲突。
- GUI 校验 `shared`、`gui` 和 worker 启动设置。能识别的计算项用于预览，其他项保留原文。计算程序启动时读取同一份 TOML，校验自己负责的参数。

Stage1 向 Stage2 移交任务时，从启动快照读取接收队列和完成记录。v3 使用共享 `finished_file`，v1/v2 沿用接收方的阶段配置。

GUI 保存时保留未编辑的行、注释、顺序和未知值。编辑已知键时校验本次修改。预览使用的默认值只用于显示。

忽略范围限于已知阶段分区；未知分区、重复分区／键、缺少右值、未闭合字符串和非法行结构仍报错。外阶段的复杂值可以作为不透明原行保留，其业务有效性由拥有该分区的程序负责。本策略需要各组件先具备按职责读取能力，无法改变旧可执行文件的读取行为。

代码入口：[config_format.h](../../src/io/config_format.h)的`TomlReadMode`／`deferred_toml`决定职责，`selected_entries`仅校验实际继承或移交字段；[config.cpp](../../src/io/config.cpp)分别实现Stage1解析和Stage2继承投影；GUI的[TomlFile](../../src/ecm/application/gui/toml_file.h)提供计算预览，[设置编辑](../../src/ecm/application/gui_v2/views/pages.cpp)只检查本次修改。CPU Edwards Stage2与CUDA／OpenCL Stage2均使用Stage2职责。

共享读取器由 [TOML 夹具](../../tools/test/test_ecm_toml.ps1) 验证。迁移测试覆盖文件导出、路径及注释，worker 对拍覆盖两版配置的真实 Stage1 存档、接收方覆盖和 Stage2 启动。GUI 的 `config_identity_test` 和 `app_test` 覆盖配置身份、点号键编辑和原版本保存。[迁移测试](../../tools/test/test_ecm_config_migration.py) · [worker 对拍](../../tools/test/test_ecm_config_v2_workers.py)

Stage1 的 [cli.cpp](../../src/io/cli.cpp) 一次转换每个显式选项并保留来源；别名按最后出现生效，非法的较早值不会因覆盖而消失。[config.cpp](../../src/io/config.cpp) 一次读取 TOML 并校验Stage1／共享显式值，保留分区、源行与 typed scope，不补任务默认或改变运行时单例。生成绑定同时提供赋值和已解析字段复制，worker 投影不再往返文本。

[normalize.cpp](../../src/io/normalize.cpp) 的 `normalize_settings()` 从生成默认开始，应用全局、worker 和显式 CLI，生成启动设置；main 安装一次 runtime。`normalize_direct()` 与 `normalize_worktodo()` 分别接收该设置，形成拥有自身数据的冻结任务。启动不回填 CLI，任务循环不重读 TOML；Stage2 接收端投影复用原始启动快照，worktodo 仍每轮重读。Stage2 保留其现有编排。

`checkpoint_backups` 接受 0…16。负数 `checkpoint_interval_seconds` 禁用定期存档。重复指定 sigma 时，后一个值及其可选参数化声明一并生效。独立指定的 `--gpu-param` 仍需与 sigma 一致。

CLI/配置错误停止启动。TOML 诊断由 [config_format.h](../../src/io/config_format.h) 的 `config_diagnostic()` 统一排版：`INPUT[config]`、`file=路径:行号`、`worker=`、`key=`、`value=`、错误原因和 `expected 类型 范围` 各占一行。`value` 保留填写的原始右值；`worker=0` 表示全局分区，其他数字表示发生错误的 worker 覆盖分区。布尔范围同时列出true／false和0／1。选中 worktodo 行的错误由队列标记并跳过。队列模式的任务结构字段不能由 CLI 覆盖，sigma 采用单独优先级；完整规则见 [Stage1 输入与默认](../usage/STAGE1.md#输入归一化与运行中编辑)。

## 运行时限制

读取配置时检查静态类型和范围。设备、路径冲突、save 与任务是否匹配、D 和 NTT 形状及 Auto B2 的适用条件在实际执行时检查。

- 以 TOML 所在目录为基准解析队列、自动移交目标、结果及提交路径，以及 Stage2 接收端的 `finished_file`。
- 以程序目录为基准解析 Stage1 配置中的 `save_dir`、完成记录、日志、备份及 Prime95 路径，也用于队列行中的相对 save 路径。
- 单次运行的 CLI save 和临时目录以当前工作目录为基准。临时目录省略或留空时使用当前目录。显式的相对 `exponent_cache_dir` 也以当前目录为基准。留空时，仓库内使用根目录的 `cache/`，独立部署使用可执行文件同级的 `cache/`。

v3 中，Stage1 和 GPU Stage2 按 TOML 目录解析共享的 `log_file`（默认 `ecm.log`）和 `finished_file`（默认 `worktodo.finished.txt`）。每条文件日志以 `[Worker #N] [YYYY-MM-DD HH:MM:SS]` 开头，使用本地时间。活动日志超过 `log_file_max_KB` 后归档，编号取同目录已有归档的最大编号加一。Stage1 的 `--no-log-timestamp` 只关闭控制台时间戳。Stage2 控制台和调优试跑的诊断格式沿用各自的输出协议。

配置路径以程序目录为基准。队列模式缺少配置时创建模板。单次运行缺少默认配置时使用内置默认值，显式指定的文件缺失时报错。`--gpu-info` 缺少配置时沿用查询默认值。各键取值见[配置参考](../ECM_TOML_REFERENCE.zh-CN.md)。
