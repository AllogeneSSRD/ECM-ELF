# Stage1 中断存档与备份轮换开发草稿

此页记录本轮开发和验证；当前使用规则见 [Stage1 使用](../usage/STAGE1.md)，配置键以 `config/ecm_options.json` 生成的 `docs/ECM_INI_REFERENCE.md` 为准。

## 目标与实现

- Ctrl+C、Ctrl+Break 和 GUI 停止请求只设置停止标志；CUDA/OpenCL 在 GPU 批次边界、CPU 在计算分块边界执行存档 I/O 并退出。Windows GUI 通过继承的事件句柄通知 worker，等待 worker 自行退出；第二次控制台中断保留强制退出路径。
- 存档先写 `.new`，完整关闭后替换主文件。`ckpt_backups` 默认 3，范围 0–16，主文件另计。备份依次编号；损坏主文件恢复后，首次新写入保留已有有效备份。二进制新档附加覆盖头和数据的校验值，旧档按原格式读取。
- 成功完成后删除主文件、临时文件和所有备份代次。CPU Montgomery、CPU Edwards、OpenCL、CUDA 共用停止请求与轮换辅助实现；Android OpenCL 构建也链接该实现。
- 队列处理当前任务时收到停止请求，会在检查点保存后直接退出；未完成任务保留原始 `worktodo` 行，不标为 `# ERROR`，也不启动后续任务。已完成任务之后才收到停止请求时，先完成该行的结果登记，再停止读取下一行。

主要入口：`src/core/ecm_stage1_checkpoint_runtime.cpp`；CPU 驱动 `src/core/ecm_driver.cpp`；OpenCL `src/opencl_ecm_stage1.cpp`、`src/core/ecm_checkpoint.cpp`；CUDA `src/cuda/stage1/montgomery.cu`；GUI `src/gui/worker_proc.cpp`、`src/gui/app.cpp`。

## 本轮验证

- Windows 主程序 `ecm`、GUI `ecm_gui`、存档轮换测试编译通过。`tools/test/stage1_checkpoint_rotation_test.cpp` 通过文本与二进制存档的轮换、损坏回退、缩减备份上限、清理及 GUI 停止事件测试。
- GUI worker 自测 53 项通过；新增用例确认继承事件句柄能让 worker 保存并自行退出。在全档 CUDA 开发构建目录中重新编译 GUI 并复测，日志为忽略目录 `build_cuda_ckpt_verify_bt/ecm_gui_worker_selftest.log`。
- 真实 CPU Montgomery 任务以 B1=10000 完成后，只留下最终 `.save`，没有 `.ckpt` 或备份。结果在忽略目录 `data/experiments/stage1_complete_final/`。
- 配置生成一致性检查与 `git diff --check` 通过。
- 沙箱内 CUDA 13.3 + MSVC 14.51 的编译器检查报 C1902，nvcc 子进程曾报 `ACCESS_VIOLATION`；在沙箱外用仓库的并行构建脚本，先通过 sm_89/5120 单档构建，再通过 sm_89 全档构建，12/12 CUDA 单元、主机代码和链接均成功。完整构建日志在忽略目录 `data/experiments/stage1_build_full.log`，二进制为 `build_cuda_ckpt_verify_bt/ecm_cuda.exe`。
- RTX 4070 Ti 上的 PARAM0 实跑通过：4999 位任务输出 save；GUI 继承事件三次中断均留下检查点，第二次产生 `.bak.dat`，人为损坏主存档后报告校验失败并从较早的备份位号继续。周期存档完成后，主存档及备份均已清理。测试日志在忽略目录 `data/experiments/stage1_cuda_full_verify/`。
- 全档构建对 5240 位任务选用 5632 档。以 5240 位素数执行 B1=10000、8 条 PARAM0 曲线，退出码 0，写出 8 条 save，完成目录无检查点或备份；日志在 `data/experiments/stage1_cuda_5240_prime_verify/`。
- 队列中断回归：原构建在停止后把当前及下一条 `ECM2=` 行都标为 `# ERROR`，仍以退出码 0 结束；原始日志保存在 `data/experiments/stage1_cuda_queue_stop_before/`。修复后重新链接全档 `ecm_cuda.exe`，用 `tools/test/test_cuda_queue_stop.py` 分别通过 GUI 事件和 Ctrl+Break 中断两条真实 CUDA 任务：两次均仅有一次 `START`，原队列两行保持不变，生成检查点且非零退出；日志在 `data/experiments/stage1_cuda_queue_stop_after/`、`data/experiments/stage1_cuda_queue_ctrlbreak_after/`。

## 说明

以上是开发验证构建与本机 GPU 实测；它没有替换发布目录中的二进制。先前用 `2^5240-1` 作为测试输入时，最终点的分母不可逆，程序拒绝写出 save；改用 5240 位素数后成功，该失败样本留在忽略目录 `data/experiments/stage1_cuda_5240_verify/`。
