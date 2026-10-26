# 配置定义与解析

## 单一数据源

[config/ecm_options.json](../../config/ecm_options.json) 定义 Stage1、Stage2、GUI 与 worker 的键、类型、默认值、静态范围和英中说明。`shared_with` 使两阶段使用同一个 `[shared]` 定义。运行时读取生成的 C++ 绑定，不读取 JSON。修改定义后依次运行 `python tools/gen/generate_ecm_config.py` 和 `python tools/gen/generate_gui_v2_state.py`，分别同步业务配置绑定/参考与GUI v2状态、业务选项元数据及英中文说明XML；仅修改说明也需要同步两组生成物。release构建入口自动在依赖解析、组件构建及身份冻结之前依次执行这两步，失败立即停止；`-SkipBuild` 只执行各自的 `--check`，不改写源文件。生成器保留内容未变产物的时间戳，避免每次构建触发GUI重编译。[业务构建检查](../../tools/build/internal/check_ecm_config.cmake) 和 [GUI v2构建检查](../../tools/build/internal/check_gui_v2_config.cmake) 继续严格验证哈希，CMake检查本身不生成文件。

生成物包括 [默认配置](../../config/ecm.toml.default)、[示例配置](../../config/ecm.toml.example)、[逐键参考](../ECM_TOML_REFERENCE.md)、C++ 绑定、[独立默认值头](../../src/ecm/platform/generated/configuration_defaults.h)、TOML 键表、嵌入模板和哈希清单。独立默认值头只依赖标准整数/数组类型，供 task 等主机实现消费；完整应用绑定包含该头，不重复生成默认值定义。两份配置按分区及键名排序；每个条目先写 `键 = 默认值`，再写英文、中文注释，最后空一行。显式设置会改变 worker 后缀或环境变量优先级的键保留带注释的默认值，启用时取消注释。参考文档仅由生成器维护。

嵌入模板使用UTF-8字节对应的字符数组初始化，保留 `default_toml` 的constexpr字符数组接口及末尾NUL，避免旧MSVC对超过65535字节的拼接字符串字面量报C1091；不改变默认TOML内容或配置写出方式。

## TOML 分区

- `[shared]`：Stage1 与 Stage2 共用；`[stage1]`、`[stage2]`：对应阶段的专用键。
- `[gui]`：GUI 全局键；`[workers.N]`：第 N 个 GUI worker 的显示与启动设置。
- `[workers.N.shared]`、`[workers.N.stage1]`、`[workers.N.stage2]`：只覆盖同名全局分区中的键。N 为 1…1000000。

键名和分区名区分大小写。一个键仅能放在其定义的分区或相应 worker 覆盖分区；同表重复键、未知键、未知表及类型错误会报告文件名和行号。内置默认值 < 全局分区 < 对应 worker 分区 < 命令行。`worktodo.txt` 的 `[Worker #N]` 是独立的队列语法，不是 TOML 配置分区。

[config_format.h](../../src/io/config_format.h) 提供共享 TOML 文法及标量解析，支持带引号的字符串、布尔值、整数、浮点数及窗口整数数组，允许 `#` 行注释和行尾注释。GUI 的 [TOML 编辑器](../../src/ecm/application/gui/toml_file.cpp) 保留未修改行、注释及顺序，改写值时按类型编码，再以临时文件原子替换并保留一份 `.bak`。配置文件默认为程序目录的 `ecm.toml`，显式路径使用 `--config`。不读取旧 `ecm.ini` 格式。

Stage1 的 [cli.cpp](../../src/io/cli.cpp) 一次转换每个显式选项并保留来源；别名按最后出现生效，非法的较早值不会因覆盖而消失。[config.cpp](../../src/io/config.cpp) 一次读取 TOML 并校验所有显式值，保留分区、源行与 typed scope，不补任务默认或改变运行时单例。生成绑定同时提供赋值和已解析字段复制，worker 投影不再往返文本。

[normalize.cpp](../../src/io/normalize.cpp) 的 `normalize_settings()` 从生成默认开始，应用全局、worker 和显式 CLI，生成启动设置；main 安装一次 runtime。`normalize_direct()` 与 `normalize_worktodo()` 分别接收该设置，形成拥有自身数据的冻结任务。启动不回填 CLI，任务循环不重读 TOML；Stage2 接收端投影复用原始启动快照，worktodo 仍每轮重读。Stage2 保留其现有编排。

`ckpt_backups` 的 0..16 范围由配置定义与生成绑定拒绝越界值，不静默截断；负 `ckpt_seconds` 仍归一化为禁用定期存档。重复 sigma 完整替换值和可选 param 声明；独立显式 `--gpu-param` 的矛盾检查继续保留。

CLI/配置错误停止启动，诊断包含来源、键、原值和预期范围；选中 worktodo 行的错误由队列标记并跳过。队列模式的任务结构字段不能由 CLI 覆盖，sigma 采用单独优先级；完整规则见 [Stage1 输入与默认](../usage/STAGE1.md#输入归一化与运行中编辑)。

## 运行时限制

JSON 中的静态范围与类型由生成绑定检查。设备存在性、路径冲突、save/任务匹配、D/NTT 形状及 Auto B2 标定资格依赖实际运行环境，由消费端验证。队列的 worktodo、自动 Stage2 目标、结果/提交路径及 Stage2 接收端 finished 基于 TOML 目录；Stage1 tmp、finished、日志、备份/P95 与任务 save 基于程序目录。direct 的 CLI save/tmp 基于 CWD，省略或空 tmp 使用 CWD；direct 显式 TOML tmp 基于程序目录。显式相对 exp_cache 基于 CWD，空/省略值默认程序目录，off/none/0/- 保持禁用。配置路径基于程序目录。队列缺少配置时创建模板；direct 缺少默认配置使用默认值，显式缺失报错；gpu-info 缺少配置不创建文件，即使显式指定也保留查询默认。具体键、默认值和 worker 覆盖资格见[配置参考](../ECM_TOML_REFERENCE.md)。
