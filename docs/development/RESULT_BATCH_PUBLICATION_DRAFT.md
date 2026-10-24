# 批次结果发布草稿

状态：本轮收尾逻辑已实现并完成离线验收，版本 0.52.2。Stage2 仅 ckpt 方案只做分析，未实施。本文件作为临时开发草稿保留，尚未清理。

用户授权：Stage1 去掉逐曲线磁盘回执，结果在内存聚合后发布最终内部 JSONL 和提交 JSON；save/checkpoint 保留。Stage2 允许磁盘回执，按 save 去掉扩展名分子目录；最终输出路径及文件名保持。每批逐曲线结果按 uint64 sigma 排序；提交 NF 保持单批一条，只有因子行具备可排序的 sigma。

## 可行性测量

证据位于 `data/experiments/stage1_result_memory_20261024/`。Windows 11，MSVC /O2 /MD，GMP；输入 M4423、B1=1000、全 NF、随机打乱 sigma。每种规模分别测内存与最终两文件发布，各 3 次；无 GPU 计算、无历史结果，电源与缓存状态未固定。

- 1000 条：内存聚合中位数约 0.097 秒；含最终两文件约 0.115 秒。私有内存增量约 1–2 MiB。
- 3840 条：含最终两文件中位数约 0.446 秒；排序及合并约 2 毫秒。
- 10000 条：含最终两文件中位数约 1.093 秒；最终进程峰值工作集约 20 MiB。

进程峰值、私有内存增量是不同指标，不能相加。原始 JSON 含完整计时与进程计数器。以上不包括历史读取、故障重试和任务集成，不能当作最终生产性能。

## 发布与恢复设计

Stage1 不生成逐曲线 receipt 或独立 publication.order。内部最终文件的本批首条结果携带小型、带完整性散列的提交计划，包含最终提交目标、已有字节前缀身份及本批提交文本。先原子发布内部结果，再原子发布提交文件；失败保留 checkpoint。重试先核对已提交计划，完成缺失或部分提交，避免整批重复计数。旧账本和原始历史不自动删除。

Stage2 回执在原提交账本前缀下按 save 名称分目录。最终 stage2_results.jsonl 保持完成一条追加一条。新 Stage1 save 批次按 uint64 sigma 排序，因此 Stage2 承接其输入顺序；旧 save 不重排。Stage1/Stage2 共用提交文件时保持原字节前缀，Stage1 因子跨阶段去重。

整个 Stage2 提交账本缺失、但已有内部结果、Stage2 提交记录或恢复进度时，在下一条算术、结果发布和整任务封存前拒绝继续，不自动重建历史信用。局部子目录缺失及所有历史一同丢失不在该检测保证内，不能把 data 当作缓存删除。

实现入口：`src/ecm/service/stage1/task.cpp`、`src/ecm/storage/results/publication.cpp`、Stage2 task 与 CPU Stage2 入口。

## 实际实现测量与验收

证据根目录：`data/experiments/result_batch_impl_20261024/`；正式使用和恢复说明见 [ECM_RESULTS](../usage/ECM_RESULTS.md)，可视化见 [HTML 报告](../web/stage1-result-latency-20261024.html#implemented)。未替换原 build_stage1_local 的程序或结果。

真实发布模块探针：Windows 11、AMD64 Family 26 Model 36、MSVC /O2 /MD、GMP，M4423、B1=1000、全 NF、打乱 sigma，每组 3 次。边界为预检、编码和实际双文件发布，不含算术及 save；电源与缓存未固定。1000/3840/10000 条新输出中位数分别为 0.193/0.730/1.931 秒，整个探针进程峰值工作集分别约 7.1–7.7/12.6–12.7/22.0 MiB。1000 条加 24,960 行、约 14.4 MiB 历史时，中位数 1.093 秒、峰值工作集约 32.3–33.5 MiB。CPU 时间约等于短时单核占用；历史校验及文件替换仍随历史增长。源码、二进制和历史快照 SHA256、各计数器见 performance_final.json，不能当作完整 GPU 进程内存。

完整 GPU 任务：RTX 4060 Laptop GPU，sm_89 的 1024 位测试构建，Edwards 随机 sigma、M521、B1=1000、1000 曲线，0.52.2。新输出与带上述历史各 3 次，save 行至 results 行的单调时钟到达间隔中位数分别为 0.173 和 0.730 秒，范围 0.172–0.177 与 0.729–0.767 秒；包括 checkpoint 完成清理，管道调度可能引入误差，功耗未固定。六批全部核对数值排序、save 与 JSONL sigma 集合、CRC 及无逐曲线回执。证据为 gpu_tail_samples_numeric.json 与各 gpu_tail_*_numeric 目录。先前 b2 构建数据单独冻结，不冒充最终数字版本。

- 新批次发布契约通过：数值排序、uint64 边界、重叠与重复批次、双文件部分/缺失发布、第二文件写失败、损坏计划、内部缺行、旧即时因子兼容、Montgomery PARAM0/2/3、Stage2 子目录和缺失账本预检。入口 tools/test/result_batch_contract.cpp；最终功能证据 contract_loss_guard。
- 18 项真实 Stage1 task/存储协作者检查通过；task_cases_final。23 项旧发布生命周期检查通过；legacy_lifecycle_final，包含保持 Stage2 文件对象的硬链接验证。
- 27 项 CPU/GPU 流程通过；usability_complete/summary.json。覆盖 Edwards scalar/SIMD/GPU binary/NAF、Montgomery scalar/SIMD/GPU PARAM0/2/3、跨 Stage2 共用提交文件、发布失败重试、停止/备份恢复和输入身份。
- 实际 CPU Stage2 的 sealed/pending_nf/fresh_stage2 三项账本丢失检查通过；ledger_loss_checks.json。CUDA Stage2 实际 host task 编译通过；build_stage2_host_acceptance。CUDA Stage2 本轮未重跑完整设备算术，其逐条 JSONL append 调用未修改。

本地 GPU 测试构建最终由已配置工程增量构建并单独记录验收身份。旧测试包装器在对象收集时误选 ecm 而非 ecm_cuda 目标目录，未作为成功构建证据；实际编译/链接和后续运行单独验证。流程测试的旧独占锁断言修正为兼容当前更早生效的队列 ownership 错误。纯数字版本要求保持；用户确认将其他工作的 0.52.2-b2 改回 0.52.2。

仍待验证：按原位宽扫描的全部输入和相同设备/功耗复测，不能把 M521 的时延当作所有位宽保证。冻结的提取脚本 test_result_publication_extraction.py 用于证明旧提交的无行为变化移动；其完整函数体/布局不变断言不适用于本次已授权的发布行为变更，不作为本轮功能验收。

## Stage2 仅 ckpt 方案分析（未实施）

用户要求分析取消中间结果、仅依 ckpt 进度恢复的影响。现有 State V2 不含 NF/F 分类、最后 NF 元数据或提交事务，不能直接替代 receipt。建议取消逐曲线文件、保留逐条最终 JSONL，并把 ckpt 扩展为当前任务累计摘要与单条待提交状态；完成后从最终历史保持去重。CPU 直接读 save 的入口目前没有该队列 ckpt，需另行设计。完整比较及故障窗口见 [HTML 分析](../web/stage1-result-latency-20261024.html#stage2-ckpt-only)。本轮未实施该方案。
