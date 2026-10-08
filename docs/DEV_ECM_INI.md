# ecm.ini 开发入口

用户配置的默认值、可选范围与逐项用途说明统一见 [配置参考](ECM_INI_REFERENCE.md)。
定义与源码维护流程见 [配置统一维护](DEV_ECM_CONFIG_SCHEMA.md)。

## 修改配置

1. 编辑 `config/ecm_options.json` 的正式键、默认值、范围、用户说明或别名。
2. 执行 `python tools/gen/generate_ecm_config.py`。
3. 提交定义及生成物；构建会拒绝过期内容。

不再在此文重复维护默认值和键表。CLI 参数语法以各程序 `--help` 及驱动代码为准。

## 消费端

- Stage1：`src/core/ecm_queue_config.h/.cpp`，生成结构和赋值分发表，运行接线在 `ecm_driver.cpp`。
- Stage2：`src/core/ecm_cuda_stage2_main.cpp`，生成 Settings 与共享 worker 合并，动态路径/算法校验在驱动。
- GUI：`src/gui/ini_file.h/.cpp` 共享原行读取并保留写回；`src/gui/app.cpp` 消费生成的 GUI/worker 配置。
- 所有输入的段头识别、词法处理和类型转换：`src/core/ecm_ini.h`。

## 行为说明

- [Stage2 发布、队列续跑、日志和退出](ECM_CUDA_STAGE2_RELEASE.md)
- [Stage1 worktodo](DEV_ECM_WORKTODO.md)
- [GUI 监管与 Prime95 交接](DEV_ECM_GUI.md)

后续 TODO：考虑向后兼容，合并两个可执行文件，并配置 Stage1 完成后继续 Stage2，或交给 Prime95/独立 Stage2 程序消费；交接确认、进度归属及防重复消费另行设计。
