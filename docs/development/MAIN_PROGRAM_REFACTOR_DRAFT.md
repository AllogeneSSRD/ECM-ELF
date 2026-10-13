# 主程序共用接口重构开发草稿

此页记录本轮重构和验证；当前架构以 [架构与术语](../architecture/OVERVIEW.md) 为准。

## 范围与决定

- `ecm` 和 `ecm_cuda` 仍使用同一个 Stage1 驱动与不同 GPU 后端；`ecm_cuda_stage2` 保持独立的 save、回执和计算流程。
- Stage1 命令行的重复整数表达式解析器删除，改用 Stage2 与 Android 已使用的 `ecm_parse_expression`。保留原有语法和解析错误文本。
- `ecm_worktodo_read_first_line` 对 Task、Empty、Error 给出不同结果，两个队列驱动共用相同的 worker 分节、BOM 和注释处理。旧 bool 入口保留给现有调用方。Stage1 遇读取错误不再将其作为空队列；Stage2 保留原有报错行为。
- 两个 Stage1 可执行文件的共用源清单在 CMake 中维护一份，各自编译这些源文件并链接自己的 GPU 后端。未合并 Stage2 的计算和回执接口。

## 验证

- MSVC 14.51、CUDA 13.3、sm_89：`ecm` 和 `ecm_cuda` 在 `build_cuda_ckpt_verify_bt/` 编译通过；`ecm_cuda_stage2` 在 `build_cuda_cmake/stage2_width_prediction_20261013/` 复用经身份检查的 CUDA 对象，宿主侧四个单元重新编译并链接通过。
- `worktodo_sections_test` 30/30 通过，覆盖不同 worker、空队列及缺失文件；`ecm` 和 `ecm_cuda` 均以 `(2^127-1)` 完成命令行 CPU Montgomery 任务与单项队列任务。
- 两个 Stage1 程序对非整除表达式 `2^3/3` 均拒绝输入并输出一致的解析错误；源文件差异检查通过。
- 两个 Stage1 程序对缺失 worktodo 均报错且退出码 1；Stage2 对缺失 worktodo 报错且退出码 2。Stage2 的独立队列 dry-run 读取 CPU Stage1 save，识别记录、sigma 与校验值，退出码 0，队列未改写。
- CUDA 队列的 GUI 停止事件回归通过：只启动第一条任务，保留两条原始 worktodo 行并生成检查点。
- 一次性验收输入和生成的 save 位于忽略目录 `data/experiments/architecture_refactor/`。本轮验证未重新测量 GPU 算术性能。

## 后续候选

- Stage1 驱动同时承担命令行、三种计算路径及结果交接，仍是较大的模块。若继续拆分，应先固定每种路径的输入、结果所有权和停止/失败语义，再迁移实现。
- Stage2 驱动同时处理 save 解析、CLI、队列状态和调优入口。save 解析与计算计划是可分别验证的后续抽取点；当前回执顺序及恢复语义应保持在同一处。
