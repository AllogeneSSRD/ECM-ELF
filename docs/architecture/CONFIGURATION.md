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

[ecm_toml.h](../../src/ecm/application/configuration/ecm_toml.h) 读取项目使用的 TOML 值类型：带引号的字符串、布尔值、整数、浮点数及窗口整数数组；允许 `#` 行注释和行尾注释。GUI 的 [TOML 编辑器](../../src/ecm/application/gui/toml_file.cpp) 保留未修改行、注释及顺序，改写值时按类型编码，再以临时文件原子替换并保留一份 `.bak`。配置文件默认为程序目录的 `ecm.toml`，显式路径使用 `--config`。不读取旧 `ecm.ini` 格式。

## 运行时限制

JSON 中的静态范围与类型由生成绑定检查。设备存在性、路径冲突、save/任务匹配、D/NTT 形状及 Auto B2 标定资格依赖实际运行环境，由消费端验证。`worktodo` 与 Stage2 相对路径基于 TOML 所在目录，其他 Stage1 相对路径基于程序目录。具体键、默认值和 worker 覆盖资格见[配置参考](../ECM_TOML_REFERENCE.md)。
