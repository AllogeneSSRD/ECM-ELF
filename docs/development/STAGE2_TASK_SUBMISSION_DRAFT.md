# Stage2 按任务提交结果：开发草稿

本草稿记录 0.42.0 的开发与验证。当前规则以 [ECM 结果输出](../usage/ECM_RESULTS.md) 为准。

## 需求与实现

- Stage2 完成一条 worktodo 后，将该任务中已完成、未找到因子的曲线汇总为一行 NF；F 仍即时逐曲线提交。
- 即使自动规划使 B2、D、P 因曲线而异，汇总曲线数仍按任务累计，报告参数及 UTC 时间取最后完成的 NF 曲线，重算 checksum。内部 JSONL 保留每条曲线实际参数。汇总 B2 不等于全部曲线均达到该界限。
- `src/core/ecm_result_submission.h` 的 `Submission::publication_order` 保存任务内位置；`src/core/ecm_cuda_stage2_main.cpp` 传入 save 记录序号，`src/core/ecm_cpu_stage2_main.cpp` 传入选择范围内序号。它只写入私有账本分组键，不出现在提交 JSON。
- `src/core/ecm_submission_writer.h` 的 `seal_submissions` 可按任务归并 Stage2 NF，保留不可变批次和因子单行。CPU `--export-results` 对未封存 Stage2 回执也按任务归并。

## 验证

- `tools/test/test_result_output_consistency.py`：Stage1 原批次、Stage2 参数变化归并、最后完成时间、CRC、F 即时单行及 F/NF 共存均通过；证据在忽略目录 `data/experiments/stage2_single_task_20261020_fixture_final/`。
- CPU Stage2 本地构建和四曲线运行：一行 NF、`curves=4`；从未封存的不同 D 回执执行 `--export-results` 后仍为一行、`curves=3`。
- CUDA Stage2 sm_89/CUDA 13.3 生产引擎构建通过；一条四曲线队列在首曲线后中断不导出 NF，恢复后只导出一行 `curves=4`，CRC 正确；证据在 `data/experiments/stage2_single_task_20261020_gpu_final/`。

## 生产文件处理

合并 `D:/code/GIMPS/ecm-win-x86_64-sm89-cuda13.3_param0/results.json_2.txt`。在独占账本锁内核对 968 份逐曲线 NF 回执与 36 个已封存批次一一对应；完整快照保存在生产目录 `data/experiments/stage2_submission_merge_20261020_01/`。合并后为一行、`curves=968`，末条曲线参数 B1=260000000、B2=1300000000000、D=1231230、P=115200，时间 `2026-10-19 23:01:00` UTC，CRC32 为 `1E907441`。历史文件中有两种 B2；这行按末条参数报告，不代表全部 968 条曲线都完成到该 B2。历史 `program.version=0.39.3` 保留，未改写为新程序版本。生产 Stage2 exe 和构建清单已更新为 0.42.0 构建，SHA256=`0830A1A55EAAE42E859045A6F4710AAE7B5509782C58098029B58D93CF25A9BB`。

生产合并先在工作区副本演练：新旧账本均通过 C++ 导出重建的字节一致性检查；生产合并后再核对一行结果、账本只有一个 NF 批次、CRC 有效、原 36 批次已在快照中。
