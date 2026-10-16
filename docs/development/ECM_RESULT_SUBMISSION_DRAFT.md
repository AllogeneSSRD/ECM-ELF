# ECM 双结果文件开发草稿

本草稿记录 2026-10-16 结果格式修复与验证过程，属于待处理的临时开发资料。正式使用说明为 [ECM 结果输出](../usage/ECM_RESULTS.md)。完成后保留本草稿，未经用户处理不自动删除。

## 需求与诊断边界

用户在 manual_result 页面提交内部 JSONL 后得到“Parsed 1 lines / Found 0 datestamps / Did not understand 1 lines”。已授权拆分内部 JSONL 与 Prime95 结构的 results.json.txt；INI 读取 user/computer，aid 可选，真实身份与所有校验值不得进入公共提交。security-code 暂省略，公开 CRC32 仅保留在内部文件。

诊断顺序为：先检查程序/字段类型与时间识别，再检查 FFT 和 Edwards 的 F/NF 结构，最后检查新程序身份或校验策略。离线可证实旧输出与正式样本存在结构差异，但没有服务器解析器或接受回执，不能把其中某一个差异认定为服务器拒绝的唯一根因。

用户提供的 PrimeNet v5 API 0.97 已核对；它描述交易接口而非现代手工 JSON 语法。固定参考资料另记 API、Prime95、PrMers、AutoPrimeNet 的不同边界。正式私人样本只读取到内存，不复制身份、AID、校验值或整条原始结果到源码/文档。

## 实现与实验

- 红测试证据 `data/experiments/ecm_submission_20261016/before/`：旧程序不识别新增输出参数，无法满足双文件合同。
- 共享序列化使用真实 ECM-ELF 身份、数字 port、正参考 FFT；CPU/GPU 子进程均保留自己的内部诊断。`mult.asm` 的 zjmptable 转成受生成器检查的公共容量表。
- NF 不含 sigma，同秒多条 curves=1 会变成相同提交行。因此增加私有逐曲线账本，正常结束把新 NF 合并为不可变批次，后续增量不修改已发布行。
- `gates_01/` 出现一次 Windows 文件替换失败，失败日志和临时文件保留。先增加 Win32 错误码，`io_repeat_0/`…`io_repeat_2/` 未重现。随后完整 GPU 回归在原 Stage1 检查点频繁轮换时两次失败，分别保留在 `gpu_regression/` 与 `gpu_regression_02/`。
- 排查顺序为高频文件占用、坏主存档恢复的路径状态、目录权限。独立文件轮换测试 `atomic_probe/` 第 30 次操作观察到 Win32 错误 5；不涉及 GPU 或存档解析，支持原子替换被占用的判断，但未确认占用者。实际共享发布入口的受控占用测试在 `atomic_red/` 失败。
- 共用 `ecm_file_publication.h` 对访问/共享/锁定错误做最多 500 ms 的等待重试，其他错误立即失败；永久占用仍保留旧文件与待发布文件。`atomic_green/` 验证短时占用恢复及持续占用失败，`gpu_regression_03/` 重跑原完整恢复场景，不修改测试门限或检查点频率。
- `gates_04/` 的无算术恢复失败：新模式未提供 save，但路径别名检查仍对空字符串调用 Windows absolute。合同测试先失败，再修复空路径处理；最终 `gates_06/` 的 19 项通过，包括复制账本的未封存 NF 恢复和 M11213 的 FFT 选长。
- CPU 数学回归包含 127/347/2203 位保存点、CPU NTT/plain、GPU1 Stage1、真因子、错误拒绝及中断恢复。最终 CPU/GPU Stage2 完整回归保留在 `cpu_regression_02/`、`gpu_regression_03/`，旧失败与通过证据均保留。
- `review/` 从既有、已核对的 M347 四曲线 Stage1 save 实际完成 CPU plain Stage2，生成两份可审阅文件；没有复制正式样本的计算结果或校验值。

所有实验都在忽略的 data 目录。已有 CPU 性能基准仍引用其独立冻结二进制；本次只验收输出与数学回归，不重新声称性能数据。

## 未验证条件

服务器对 ECM-ELF 身份、省略 security-code 和参考 FFT 的接受需要真实提交反馈。当前只生成文件，没有提交结果。已向用户说明真实提交会写入 PrimeNet 并请求确认；未获授权前不发送。2007 API 文档提到程序注册，但不能据此推断 manual_result 的具体白名单规则。

一般 N 的提交行仍受 2000 字节边界限制；大目标优先使用经过验证的原始梅森指数与 known-factors。CPU 尚无自动队列调度及曲线内部 Stage2 恢复，账本去重不能替代算术进度。
