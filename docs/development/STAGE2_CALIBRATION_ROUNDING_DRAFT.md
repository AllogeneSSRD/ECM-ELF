# 校准边界与生产B2凑整隔离

本草稿记录0.57.7问题定位与回归，正式行为见[启动短校准](../architecture/AUTO_B2.md#启动短校准)。完成后保留，待用户决定清理。

## 复现和根因

用户0.56.3删除tune后反复报`invalid completed calibration shape`。生产EXE SHA256为`3bcd750c2af5b695153a4a47184244dcbc96251709d5e59c8f84d1c79842b622`；工作配置开启`stage2_b2_round_chain=true`，自动链长及布局，最少重复数1。三份生产失败证据均为D210／P24，实际算术8440 bits，保存B1=110000000、探测B1=20。

删除性能文件触发启动成本探测，期望probe_B2=42020、giant点数⌊42020/210⌋+2=202。校准worker复制生产Options时保留凑整许可，子进程将B2扩大到1723260、giant点数8208，再返回干净的算术结果。父进程严格匹配原先的202点，因而拒绝成绩。失败记录clean=1、bad=0，自检和抽查均有效；这不是据此宣告Stage1保存点失效，也不应通过放宽形状检查接纳该成绩。

冻结0.57.6也复现同一错误，其EXE SHA256为`35691aa62a8c5ca9e67c3d4367d17ca8ec09ad477fba31c76695d07c7f1c577f`。两个隔离对照分别关闭凑整、关闭短校准，都能继续规划；正式ECM tune本来就拒绝凑整，应保持该限制。

## 修复与验证

`portable_measure`只在成本测量worker上清除B2凑整许可；父进程及正常曲线Options不变。保留D、giant点数、路径及算术检查，并在形状错误中按行输出预期D/B2/点数、实际D/点数、clean、bad、自检、抽查及证据路径。

最终0.57.7 EXE SHA256为`f918c73a82a6a1088f8b40ea6f18a905fbe8e22e5f6f493a07372b134df920e4`。GPU1为RTX4060 Laptop，仅做功能与算术回归，不调整硬件设置，也不将这些调用作为性能矩阵。

每组验证5项：缺tune且开启凑整、关闭凑整、关闭短校准、显式ECM tune拒绝凑整、显式ECM tune用CLI关闭凑整。修复前保留失败与通过对照；修复后8440-bit和M127两种宽度均通过。校准回执仍为D210／B2=42020／202点，均clean=1、bad=0，完成自检与抽查。另以真实8440-bit保存点执行完整Stage2：请求B2=26000000、D6006／P720，正常任务凑整到B2=51879828、8640点，最终NF且GMP检查零错误。由此同时锁定成本探测与生产执行两个调用层次。

原始证据根`data/experiments/stage2_calibration_rounding_20261101/`：`before_validated`为修复前完整复现；`after_wide`为规划回归；`after_full_verified`包含完整正常曲线；`after_low`为低位回归。源保存点、二进制、命令、回执和遥测均独立冻结，不修改生产队列、进度或保存点。初版夹具未处理显式tune拒绝条件及控制台过滤规则的失败输出保留，不覆盖原证据。

防止复发的入口为`tools/test/test_stage2_calibration_rounding_gpu.py`，它调用真实父子进程链并同时核对探测和生产边界，而不是只验证已有tune时的正常队列。
