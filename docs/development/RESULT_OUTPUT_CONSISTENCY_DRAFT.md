# 结果排布、批次与恢复开发草稿

版本：0.38.0。此草稿保留开发证据与决策，当前使用规范见 [ECM 结果输出](../usage/ECM_RESULTS.md)。完成后由用户决定如何处理本草稿，不自动删除。

## 授权与诊断

核对生产 Stage1 sigma/save；压缩 Stage2 内部格式；Stage1 NF 按完成任务输出（B2=B1）；因子立即额外单曲线输出；补齐提交 CRC；三份 ledger 移到输出目录 data/，与 experiments 并列。生产程序覆盖最初被自动审批拒绝，用户随后明确批准备份、更新生产 EXE/GMP/身份文件及迁移账本。原始结果文本与 save 不改写。

生产快照：Stage1 JSONL/save 各 960 条，sigma 集合完全相同且无重复。默认 sigma 是随机曲线身份，不要求连续；旧 JSONL 按数学键字符串排序，旧账本写入顺序则与 save 一致。三类账本保存内部去重、不可变提交批次及崩溃恢复信息，不能作为无用缓存删除。Stage2 内部 207 条，提交 12 行均缺 checksum。

已证实的原因：内部导出复用了按键排序的提交渲染器；原生提交序列化没有 CRC。最小回归 baseline01 同时复现顺序错误及 Stage1/Stage2 CRC 缺失。旧文件只读快照及全部失败样本保留在 data/experiments/result_output_audit_20261018/。

## 实现

- ecm_result_paths.h 统一新旧账本路径，CLI、GUI 共用。写锁内整体迁移；双位置冲突拒绝合并。旧 V1 内部回执按时间恢复，新 V2 显式保存递增序号，批次写入只扫描一次。
- ecm_result_json.h 保留有序成员与任意精度数字原文。公共结果模块从最终字段生成 PrMers CRC32；NF 更改时间、曲线数后重算，不覆盖旧的已封存行。
- NF 绑定任务身份，整条任务成功后封存；中断保留待发布曲线。F 验证后立即独立输出 curves=1，保留 sigma；CPU Stage1 worker 和 GPU 末点检查接入即时发布，重放不会重复发布。
- CUDA 内部输出复用公共模型；执行诊断归入 stage2-engine，保留 queue_receipt/tune_observation。梅森目标使用 exponent/known-factors，不重复写 N_hex。回执恢复按 JSON 根字段读取，不依赖成员前缀。
- result_format.py 在内存中兼容旧别名；调优导入及主要基准采集器共用。compact_results.py 只生成新目录审阅副本，不覆盖输入或发送结果。
- INI 描述只改 config/ecm_options.json，六份生成物同步。

## 验证与部署

证据根：data/experiments/result_output_audit_20261018/。validation_manifest.json 保存各测试入口与程序 SHA256，实际身份和 checksum 只在忽略目录。

Stage1 27 项验收通过，覆盖 CPU scalar/SIMD、GPU Edwards/Montgomery PARAM0/2/3、批次存档与检查点、任务恢复、即时因子、跨 Stage2 共用账本、未完成 NF 隔离。完整 local 程序在 GPU1 RTX4060 Laptop 验证 1279/8191/11213/16381 位各两条曲线，与独立 Python 算术校验一致。

CPU Stage2 plain/NTT/gwpoly 各四条 NF 均形成一个完成批次并通过 CRC。22 项双格式检查通过。GPU task_boundary_complete01 验证完成一条后停止没有 NF 导出，恢复仅执行剩余三条，最终一批 curves=4。15 条 GPU 调优采样与 12 条独立观察导入通过，慢样本保留、重复导入不重复计数；各自证据保留对应程序身份，不作为性能基准。

私有审阅副本保留 960/207/12 条结果、原版本和时间，全部独立 CRC 通过；37 条 tune_observation 未变。Stage2 文本 759226 → 492558 字节（约 −35.1%），仅表示格式体积。旧 V1 原生导出 migration02 与 save 的 960 条顺序一致，回执字节未变。

迁移测试识别并修复 Windows 下读取 format 的流未关闭导致替换失败；Stage1 构建另识别 max 宏冲突。C1902 构建失败来自沙箱 MSVC 调试信息环境，正常构建环境通过。失败日志不删除。GUI 新旧路径防覆盖回归通过。

生产三份账本已原样迁移到 data/，逐文件 SHA256 核对；旧程序、账本和原始结果备份位于生产 data/experiments/result_output_0_38_0_backup/，审阅副本位于 result_output_0_38_0_review/。已安装完整位宽 Stage1、CUDA Stage2、GUI local 程序与匹配 GMP，原始 JSONL、提交文本和 save 字节保持不变。未向服务器提交。
