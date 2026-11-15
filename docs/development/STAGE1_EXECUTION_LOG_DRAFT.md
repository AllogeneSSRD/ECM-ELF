# Stage1 执行详情开发草稿

Stage1 在实际内核或 CPU 执行计划确定后输出一行可读的键值记录。GUI 用这一行更新当前任务，而交接预览来自工作流配置。前者包含自动选型、降档和恢复效果，后者在没有因子且存档可交接的条件下成立。

验证重点是日志先于进度、文件镜像一致、实际 NAF 与配置默认值可区分、NTT 调优结果和运行计划一致、新任务不继承上个任务的详情。正式说明见 [Stage1 执行详情](../usage/STAGE1.md#进度前的执行详情)。

## 验证记录

- CPU Edwards 的自动窗口记录为 12。CPU Montgomery 和 OpenCL PARAM3 的短曲线验证通过，控制台与日志文件均含执行计划。原始输出位于 `data/experiments/stage1_execution_cpu_opencl_mx189c_7/`。
- CUDA sm89、384 位 CGBN 验证 Montgomery ladder、PRAC 搜索宽度、检查点恢复和 PARAM2/3。Edwards 在显存约束下把 NAF 12 降为 9，恢复时仍报告 `naf=9` 和原游标。记录位于 `data/experiments/stage1_actual_cgbn_prac_20261115/` 和 `data/experiments/stage1_actual_cgbn_edwards_resume_20261115/`。
- CUDA 13.3 NTT 验证线性、加权、自动选型及 Edwards NAF 回放。冒号记录与调优格式一致，回放时由 `naf=6` 更新为 `naf=0`。输出位于 `build_dev/s1_native_dep_handoff_384/ntt_shape_smoke_z5u212cs/`。
- Stage1 输入回归 29 项通过。GUI 控制器 56 项通过，覆盖执行记录刷新、新任务清除旧详情和配置交接预览。最终 GUI v2 无窗口测试通过，覆盖按实际曲线判断交接或保留存档。日志解析新增的 4 项 Stage1 检查通过，原有两项 Stage2 汇总 ETA 断言在 Git HEAD 和当前版本都失败。
