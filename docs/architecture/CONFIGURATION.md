# 配置定义与解析

## 唯一数据源

[config/ecm_options.json](../../config/ecm_options.json) 定义 Stage1、Stage2、GUI 和 worker 的键、字段、默认值、类型、静态范围、别名、模板覆盖与英中说明。程序使用已生成的 C++ 文件，不在运行时解析 JSON。

修改定义后调用 `tools/gen/generate_ecm_config.py` 生成并提交全部产物；`--check` 只检查同步。构建使用 `tools/build/internal/check_ecm_config.cmake` 检查输入/生成物哈希，不需要 Python。定义、生成器或产物不一致时停止构建。

## 说明与生成物

`description` 先英文后中文，说明用户能控制什么、各值的效果、空值/0语义及适用条件。显式换行保留，长行按 100 显示列折分；中文全角字符计 2 列。两个语言各起一行，不另写手工键表。

生成物包括配置值/绑定、嵌入 INI 模板、`ecm.ini.example`、[ECM_INI_REFERENCE.md](../ECM_INI_REFERENCE.md) 和哈希清单。参考使用 `key=<domain>; default=<value>`；默认值、范围及说明始终来自定义。生成物固定 LF，避免 checkout 换行改变指纹。

`default` 是内部初始化值；`implicit` 描述未指定时的派生策略；`effective_default` 提供消费端的派生名称；`present` 区分未指定与显式空值。模板的 `release_value`/`first_run_value` 可以有意覆盖内部默认，不能把两者混为同一默认。

## 读取与覆盖

[ecm_ini.h](../../src/core/ecm_ini.h) 提供保留原行的词法读取、worker 识别、层级覆盖及类型转换。worktodo 的 worker 段识别使用同一入口。Stage1/Stage2 使用生成绑定，GUI 保留注释、顺序及未知键并原子写回。

CLI → worker → 全局 → 默认。相同层重复键采用最后一次，worker 层始终优先。支持 `#`/`;` 整行注释。Stage1 键名大小写规则和 Stage2 专用键不区分大小写的规则由各自消费端保留。

命令行仅把 `[Worker #N]` 识别为覆盖范围，普通标题不结束该范围；GUI 把标题当作真实分区。共享模板以注释分组，公共键置于所有分区之前。

## 静态与动态限制

JSON 维护纯静态范围。设备存在性、路径冲突、save 与任务匹配、D/NTT 形状及 Auto B2 标定资格依赖实际运行环境，由消费端验证；不能复制为未经验证的配置业务规则。

新增普通键可以通过定义生成字段、解析、模板和说明，但功能仍需实现消费逻辑。只改文档或生成字段不能视为功能完成。

## 入口

- [生成器](../../tools/gen/generate_ecm_config.py)：`reference`、`template_header` 和生成清单。
- [配置检查](../../tools/build/internal/check_ecm_config.cmake)：原生构建一致性。
- [Stage1 驱动](../../src/core/ecm_driver.cpp)、[Stage2 驱动](../../src/core/ecm_cuda_stage2_main.cpp)、[GUI INI](../../src/gui/ini_file.cpp)：动态接线。
