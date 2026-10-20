# 自动 Stage2 目标队列与 worker：开发草稿

## 授权与行为

用户要求在现有 `auto_stage2` 上增加 `auto_stage2_workers` 与 `auto_stage2_worktodo`；确认仍使用 `auto_stage2=1` 总开关。worker 列表留空时不选择其他 worker：同文件原位置改写，显式指定其他文件时沿用当前编号。目标路径留空继承生效的 `worktodo`，文件名和相对路径基于 TOML 所在目录。列表、区间与 P95 共用解析，候选按当前任务数选择，并列取较小编号。`auto` 使用目标已有 worker 分区，无分区时回退本 TOML 的 GUI worker 数。

正式规则维护在 `docs/usage/STAGE1.md`，配置类型、默认值及双语说明由 `config/ecm_options.json` 生成。版本提升为 0.44.0。

## 实现

- `ecm_driver.cpp` 保留原位改写路径，新增可选择的通用 Stage2 投递；找到因子时不产生交接。跨文件任务使用实际 save 的绝对路径，修正配置/程序目录分离时显式 save 的交接定位。
- `ecm_worker_list.h` 共用有界列表/区间解析，P95 调用相同解析器并保留其越界项统计；ECM 接收配置对越界或无有效候选报错。
- `ecm_worktodo.cpp` 提供任务数快照，保留空分区信息。`ecm_stage2_transfer.cpp` 在保存待投递记录时选择 worker，并把确定的目标文件、worker 和 finished 路径写入 outbox；随后移除源行并投递，失败可恢复。
- 目标活动队列与相应 worker 的 finished 用于重放去重。选择时考虑本来源已有 pending；不同来源并发时的任务数快照可能同时选择同一 worker，队列写入仍由编辑锁串行保护，不保证严格的跨进程负载均衡。

## 验证

原始验证输出位于忽略目录 `data/experiments/auto_stage2_routing_20261020*/` 与 `tools/test/_run/stage2_transfer/`。本草稿保留供用户决定整理方式。

- `test_stage2_transfer.ps1`：通过原生队列投递/恢复验证，包括列表与区间、越界输入、待投递负载计数、空 worker 分区、接收方 finished 去重、不可写目标恢复及缺失 finished 路径拒绝。
- `test_auto_stage2_routing.py`：最终 CUDA 主程序的 CPU scalar Stage1 路径通过 13 项端到端用例，包括原位改写、worker 覆盖、最短队列、外部文件同编号/区间、auto、关闭交接、三类无效列表、重启恢复、找到因子不交接和 ECM2。最终输出在 `data/experiments/auto_stage2_routing_20261020_final/`。
- TOML 配置回归通过；配置生成器 `--check` 确认 7 个生成物一致；GUI 自检 33 项通过。
- `ecm`、`ecm_cuda`、GUI、CUDA Stage2 及 CPU Stage2 编译通过，版本为 0.44.0。CUDA Stage1 保留此前完整重建且验证过的 sm89 设备对象，本次重编译主机端并链接；CUDA Stage2 二进制及其清单内 92 个源文件哈希核对一致。
- 最终 Stage1 二进制使用生产检查点副本在 RTX 4070 Ti 上恢复，收到 GUI 同型停止事件后存档正常推进。单次短运行仅验证恢复和停止行为，不作为性能比较；原始输出在 `data/experiments/stage1_stall_20261020/final_044/`。

## 生产同步

2026-10-20 已将 0.44.0 CUDA Stage1、CUDA Stage2、GUI、配置模板、配置参考及本地化文件同步到指定生产目录。旧 INI 的 65 个配置项转换为 TOML 后逐项核对，保留 `auto_stage2 = 0`，新增的两个字符串字段均留空。原生配置读取验证两个 worker 的 Stage1 GPU 编号仍为 0/1，Stage2 GPU 编号均为 1。

生产目录已有的 TOML 是默认模板；部署时与旧 INI 一并备份，生效配置采用生产 INI 的实际设置。旧 INI、INI 示例与旧参考移入备份目录。发布的 13 个文件哈希一致，其余 34 个根目录文件哈希保持不变，包括任务、检查点、结果和调优数据。备份目录为生产根目录下的 `backup/toml_0.44.0_20261020_085606/`，清单保留部署前哈希。没有启动生产任务，没有提交 Git。
