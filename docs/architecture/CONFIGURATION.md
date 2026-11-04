# 配置定义与解析

## 单一数据源

[config/ecm_options.json](../../config/ecm_options.json) 定义 Stage1、Stage2、GUI 与 worker 的键、类型、默认值、静态范围和英中说明。`shared_with` 使两阶段使用同一个 `[shared]` 定义。运行时读取生成的 C++ 绑定，不读取 JSON。修改定义后依次运行 `python tools/gen/generate_ecm_config.py` 和 `python tools/gen/generate_gui_v2_state.py`，分别同步业务配置绑定/参考与GUI v2状态、业务选项元数据及英中文说明XML；仅修改说明也需要同步两组生成物。release构建入口自动在依赖解析、组件构建及身份冻结之前依次执行这两步，失败立即停止；`-SkipBuild` 只执行各自的 `--check`，不改写源文件。生成器保留内容未变产物的时间戳，避免每次构建触发GUI重编译。[业务构建检查](../../tools/build/internal/check_ecm_config.cmake) 和 [GUI v2构建检查](../../tools/build/internal/check_gui_v2_config.cmake) 继续严格验证哈希，CMake检查本身不生成文件。

生成物包括 [默认配置](../../config/ecm.toml.default)、[示例配置](../../config/ecm.toml.example)、[逐键参考](../ECM_TOML_REFERENCE.md)、C++ 绑定、[独立默认值头](../../src/ecm/platform/generated/configuration_defaults.h)、TOML 键表、嵌入模板和哈希清单。独立默认值头只依赖标准整数/数组类型，供 task 等主机实现消费；完整应用绑定包含该头，不重复生成默认值定义。两份配置按分区及键名排序；每个条目先写 `键 = 默认值`，再写英文、中文注释，最后空一行。显式设置会改变 worker 后缀或环境变量优先级的键保留带注释的默认值，启用时取消注释。参考文档仅由生成器维护。

嵌入模板使用UTF-8字节对应的字符数组初始化，保留 `default_toml` 的constexpr字符数组接口及末尾NUL，避免旧MSVC对超过65535字节的拼接字符串字面量报C1091；不改变默认TOML内容或配置写出方式。

## TOML 分区

- `[shared]`：Stage1 与 Stage2 共用；`[stage1]`、`[stage2]`：对应阶段的专用键。
- `[gui]`：GUI 全局键；`[workers.N]`：第 N 个 GUI worker 的显示与启动设置。
- `[workers.N.shared]`、`[workers.N.stage1]`、`[workers.N.stage2]`：只覆盖同名全局分区中的键。N 为 1…1000000。

键名和分区名区分大小写。一个键仅能放在其定义的分区或相应 worker 覆盖分区；同表重复键、未知表，以及当前读取程序负责字段中的未知键或类型错误，会报告文件名和行号。内置默认值 < 全局分区 < 对应 worker 分区 < 命令行。`worktodo.txt` 的 `[Worker #N]` 是独立的队列语法，不是 TOML 配置分区。

[config_format.h](../../src/io/config_format.h) 提供共享 TOML 文法及标量解析，支持带引号的字符串、布尔值、整数、浮点数及窗口整数数组，允许 `#` 行注释和行尾注释。GUI 的 [TOML 编辑器](../../src/ecm/application/gui/toml_file.cpp) 保留未修改行、注释及顺序，改写值时按类型编码，再以临时文件原子替换并保留一份 `.bak`。配置文件默认为程序目录的 `ecm.toml`，显式路径使用 `--config`。不读取旧 `ecm.ini` 格式。

所有布尔键接受 `true`／`false` 或整数 `1`／`0`，含各worker覆盖；`1=true`、`0=false`。带自动模式的布尔键另接受字符串 `"auto"`。共享读取器仅对schema中声明的布尔类型归一化数值别名，业务投影统一收到true／false，原行继续保留以供错误定位及GUI原样保存。其他整数、小数及数组不能作为布尔值；整数配置键也不会因此接受布尔字面量。GUI读取数值别名时显示相应开关状态，修改后写为true／false；值未改变则保留原来的数字、注释和排版。默认模板仍使用布尔字面量。

## GPU 后端与 worker

Windows 下 `[stage1] stage1_backend` 与 `[stage2] stage2_backend` 分别取 `"auto" | "cuda" | "opencl"`，两个阶段均默认 `"auto"`。例如在 `[workers.2.stage2]` 设置 `stage2_backend = "opencl"` 和 `stage2_device = 1`，只改变 worker 2 的 Stage2。命令行 `--gpu-backend` 优先于对应 worker 和全局配置。Stage1 的 `target = "cpu"` 直接走 CPU。

将同一版本的两个后端放在同目录；Stage1 文件名为 `ecm_cuda.exe`、`ecm.exe`，Stage2 为 `ecm_cuda_stage2.exe`、`ecm_opencl_stage2.exe`。只有一个已安装时 auto 使用该程序。两者均已安装时，auto 先检查指定编号的 CUDA 设备是否对驱动可见，其次选 OpenCL。设备序号属于实际后端，混合厂商系统应按各自的设备清单设置 worker。显式选择缺少的程序会在计算前报错。后端在任务开始前确定，计算、校验、预算或驱动错误沿原路径报告；任务没有隐式重跑。跨后端切换保留命令行、工作目录、标准流、GUI 停止句柄和进程退出码。队列、曲线记录和结果仍由实际计算程序维护。

OpenCL Stage2 支持固定 B2、基本 D 规划和显式梅森承载；CUDA tune、Auto B2 及 CUDA 专用内核参数的支持范围见 [OpenCL 生产入口](OPENCL.md#生产-stage2-入口)。

## 按程序职责读取

各程序按分区承担校验责任，不以键名前缀猜测所属阶段；对应`workers.N.*`遵守同样规则。

- Stage1严格读取`shared`、`stage1`，保留并校验路由后备用的`gui.NumWorkers`；不校验Stage2及纯GUI／worker显示参数。
- Stage2严格读取`shared`、`stage2`。设备、save目录、日志级别或Stage1指数约定缺少专用配置时，仍可继承Stage1的`device`、`tmp_dir`、`verbose`、`lcm`；队列模式还读取Stage1的`finished`以防完成队列路径冲突。只有实际需要继承的字段才在当前worker优先级合并后校验，Stage2／CLI已覆盖的旧默认值不会拦截启动。其他Stage1键和值不参与Stage2校验。
- GUI严格读取`shared`、`gui`及`workers.N`启动／显示参数；不要求Stage1／Stage2键和取值符合本版计算schema。认识且可解析的计算项用于预览，其他项保留原文；预览缺失或失败时使用显示默认，不能据此保证任务能执行。启动时仍把原配置交给对应worker，由其报告自己负责的参数错误。

Stage1向Stage2移交队列时，只从启动快照提取接收端`worktodo`、`stage2_finished`，按当前接收worker的有效值检查路径类型；不借此解析整套Stage2算术选项。其余未使用的Stage2新键、未知枚举或类型变化都不能阻止Stage1运行。

GUI保存保留其他阶段的原行、注释、顺序和未知值；不会用预览默认值回填配置。主动修改某个已知键时，仅校验该项，不因另一个阶段的未知值拒绝保存。含不认识值的配置能够打开，不代表GUI或当前worker已经实现对应功能。

忽略范围限于已知阶段分区；未知分区、重复分区／键、缺少右值、未闭合字符串和非法行结构仍报错。外阶段的复杂值可以作为不透明原行保留，其业务有效性由拥有该分区的程序负责。本策略需要各组件先具备按职责读取能力，无法改变旧可执行文件的读取行为。

代码入口：[config_format.h](../../src/io/config_format.h)的`TomlReadMode`／`deferred_toml`决定职责，`selected_entries`仅校验实际继承或移交字段；[config.cpp](../../src/io/config.cpp)分别实现Stage1解析和Stage2继承投影；GUI的[TomlFile](../../src/ecm/application/gui/toml_file.h)提供计算预览，[设置编辑](../../src/ecm/application/gui_v2/views/pages.cpp)只检查本次修改。CPU Edwards Stage2与CUDA／OpenCL Stage2均使用Stage2职责。

验证入口为`tools/test/test_ecm_toml.ps1`的职责隔离／533项布尔别名夹具、`tools/test/gui_v2/test_worker_config_compatibility.py`的真实空队列矩阵，以及GUI的`config_identity_test`／`app_test`。0.57.10本机验收共53项Worker协议／配置检查通过，包括Stage1带新版Stage2参数自动移交、两侧worker覆盖、专用值覆盖旧默认、保留自身错误检查；队列与原配置摘要不变，不执行ECM计算或修改GPU设置。GUI验证未知计算项保留、Worker继续显示、无关项可编辑保存，共享和自有项按生成绑定校验／规范化。证据为`data/experiments/config_roles_20261102/native_verified04/`、`gui_verified_identity.log`、`gui_verified_app.log`与绑定实际二进制／源码摘要的`final_audit.json`，原始证据由Git忽略。

Stage1 的 [cli.cpp](../../src/io/cli.cpp) 一次转换每个显式选项并保留来源；别名按最后出现生效，非法的较早值不会因覆盖而消失。[config.cpp](../../src/io/config.cpp) 一次读取 TOML 并校验Stage1／共享显式值，保留分区、源行与 typed scope，不补任务默认或改变运行时单例。生成绑定同时提供赋值和已解析字段复制，worker 投影不再往返文本。

[normalize.cpp](../../src/io/normalize.cpp) 的 `normalize_settings()` 从生成默认开始，应用全局、worker 和显式 CLI，生成启动设置；main 安装一次 runtime。`normalize_direct()` 与 `normalize_worktodo()` 分别接收该设置，形成拥有自身数据的冻结任务。启动不回填 CLI，任务循环不重读 TOML；Stage2 接收端投影复用原始启动快照，worktodo 仍每轮重读。Stage2 保留其现有编排。

`ckpt_backups` 的 0..16 范围由配置定义与生成绑定拒绝越界值，不静默截断；负 `ckpt_seconds` 仍归一化为禁用定期存档。重复 sigma 完整替换值和可选 param 声明；独立显式 `--gpu-param` 的矛盾检查继续保留。

CLI/配置错误停止启动。TOML 诊断由 [config_format.h](../../src/io/config_format.h) 的 `config_diagnostic()` 统一排版：`INPUT[config]`、`file=路径:行号`、`worker=`、`key=`、`value=`、错误原因和 `expected 类型 范围` 各占一行。`value` 保留填写的原始右值；`worker=0` 表示全局分区，其他数字表示发生错误的 worker 覆盖分区。布尔范围同时列出true／false和0／1。选中 worktodo 行的错误由队列标记并跳过。队列模式的任务结构字段不能由 CLI 覆盖，sigma 采用单独优先级；完整规则见 [Stage1 输入与默认](../usage/STAGE1.md#输入归一化与运行中编辑)。

## 运行时限制

JSON 中的静态范围与类型由生成绑定检查。设备存在性、路径冲突、save/任务匹配、D/NTT 形状及 Auto B2 标定资格依赖实际运行环境，由消费端验证。队列的 worktodo、自动 Stage2 目标、结果/提交路径及 Stage2 接收端 finished 基于 TOML 目录；Stage1 tmp、finished、日志、备份/P95 与任务 save 基于程序目录。direct 的 CLI save/tmp 基于 CWD，省略或空 tmp 使用 CWD；direct 显式 TOML tmp 基于程序目录。显式相对 exp_cache 基于 CWD，空/省略值默认程序目录，off/none/0/- 保持禁用。配置路径基于程序目录。队列缺少配置时创建模板；direct 缺少默认配置使用默认值，显式缺失报错；gpu-info 缺少配置不创建文件，即使显式指定也保留查询默认。具体键、默认值和 worker 覆盖资格见[配置参考](../ECM_TOML_REFERENCE.md)。
