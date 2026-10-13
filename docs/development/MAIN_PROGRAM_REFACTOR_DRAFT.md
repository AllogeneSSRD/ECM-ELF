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

## Stage1 选择项规范化（开发轮次，2026-10-13）

用户决定允许不兼容的接口变更。原 `Method { Gpu, Edwards, Mont }` 混合执行位置与曲线模型；`CpuBackend`、`Field`、`Exponent` 的名称与实际含义不符。因此将选择项拆成曲线模型、执行位置、CPU 实现、SIMD 归约和 Stage1 标量构造五个轴。对外 CLI、INI、GUI 命中记录及使用文档同步改名，旧 Stage1 选择项不再解析；INI 中出现已移除的键会报错。自动选择枚举值为 0，调用已有 IFMA 接口时显式转换常量。具体入口为 [ecm_stage1_selection.h](../../src/core/ecm_stage1_selection.h)、[ecm_driver.cpp](../../src/core/ecm_driver.cpp) 与 [配置定义](../../config/ecm_options.json)。

当前支持 CPU Montgomery、CPU Edwards 和 GPU Montgomery。GPU 参数化可声明 0、2、3，但启动前由链接的 OpenCL/CUDA 后端检查实际编译能力；CUDA PARAM0 保留完整 64 位 sigma 并检查批次不回绕，PARAM2/3 的固定 sigma 连同批量检查 32 位范围。独立 Stage2 的自动规划入口也改用 `--lcm`，在读取现存测量证据时映射其 `lcm`/`choose12` 标签。配置生成器更新 `ecm.ini` 样例、INI 参考及 C++ 绑定；GUI、Android 调用方、工作队列模板和测试脚本同步迁移。

显式 `reduction=montgomery|mersenne-fold` 要求 `impl=simd`。这是为了阻止 `impl=auto` 因曲线数或 ISA 选择标量后悄悄忽略用户指定的归约；默认 `reduction=auto` 可与任何 CPU 实现一起使用。CPU 单次运行的 `--gpu-param` 会拒绝，带参数化前缀的 `-sigma i:s` 仅在 CPU Montgomery 且 `i=0` 时接受。

最后一轮构建包含上述归约检查与重新生成的配置：`ecm`、`ecm_cuda`、`ecm_gui` 和独立 Stage2 均通过。`impl=auto` 配显式 `reduction=montgomery` 在计算前拒绝；选择项测试退出码 0，GUI selftest 36/36 通过。

验证记录：MSVC/CUDA 本机编译 `ecm`、`ecm_cuda`、`ecm_gui` 与 `stage1_selection_test`；独立 `ecm_cuda_stage2` 用构建脚本核对未改动 CUDA 对象的身份，再重新编译四个宿主单元并链接。选择项单元测试、GUI 结果测试 42/42、日志解析测试 71/71、GUI selftest 36/36 均通过。CPU Montgomery 单次及 INI 队列任务完成并生成 save；Stage2 `--lcm 12 --dry-run` 读档通过，非法 `--lcm 7` 被拒绝。GPU Edwards、Edwards `lcm=12`、PARAM2/3 越过 32 位的固定 sigma 和 PARAM0 批次 64 位回绕均在计算前拒绝。实际 CUDA PARAM0 两曲线使用起始 sigma 4294967313 完成，save 中的两个 sigma 未被截断。过时 CLI 与 INI 键被拒绝，错误 INI 的文件哈希不变。此轮 MSVC 编译使用 `/Z7` 避开当前 BuildTools 的调试符号编译故障；未重新测量 GPU 性能。验收文件位于忽略目录 `data/experiments/stage1_selection_smoke/`。

在 INI 队列的 CUDA PARAM0 实跑中发现 `ECM2=` 行完成却不生成最终 save：队列原先传空保存名，GPU 只在显式保存名非空时发布 save。现为该类任务在 `tmp_dir` 生成 `m{n}_{b1}.save`，并在写入失败时保留任务；同步扫描也覆盖 `tmp_dir`。CPU Montgomery 的共享 save 写入结果同样纳入任务成败判断，失败不清理可恢复的检查点。修复后重新编译 `ecm` 与 `ecm_cuda`，CUDA PARAM0 队列两条曲线实际生成 `m127_1e3.save` 并同步到配置目录，save 中保留 sigma 4294967313、4294967314；独立 Stage2 的双曲线 dry-run 读取该文件，两条记录的 checksum 均通过。分别令 GPU 与 CPU 的目标 save 名称被目录占用，两个队列均返回错误、保留原 worktodo，且未写 finished；CPU 的 `.ckpt` 保留供恢复。证据位于忽略目录 `data/experiments/stage1_selection_smoke/queue_param0_64_with_save/`、`queue_save_failure_gpu/` 与 `queue_save_failure_cpu/`。
