# ECM 结果输出

CPU Edwards Stage2 与 CUDA Stage2 同时写两份结果：逐曲线内部 JSONL 用于数学和运行诊断，`results.json.txt` 用于手工提交。提交文件只使用 Prime95 ECM 的字段结构，程序名如实写为 **ECM-ELF**。生成文件不触发网络请求；服务器是否接受新程序身份仍待实际验证，不能用离线字段检查代替服务器接受结论。

## 文件与身份

CPU 使用 `--results INTERNAL.jsonl --submission results.json.txt`；CUDA 也支持这两个参数，INI 对应 `stage2_results_file` 和 `stage2_submission_file`。CPU 的 `--results` 目前只从 CLI 设置。提交路径的默认值及范围见生成的 [INI 参考](../ECM_INI_REFERENCE.md)。

两个程序默认读取 exe 同目录的 `ecm.ini`，`--ini FILE` 显式覆盖。身份键为 `user`、`computer`；空值省略对应 JSON 字段。全局键可由 `[Worker #N]` 覆盖，启动时通过 `--worker N` 选择。CPU 的 `--user`、`--computer` 最后覆盖 INI，显式空字符串也生效。CUDA 从选定 worker 的 INI 读取身份并传给每曲线发布路径。

本地配置可使用以下空值模板，实际值只填入被忽略的 INI：

```ini
user=
computer=
stage2_submission_file=results.json.txt
```

INI 相对输出路径基于 INI 目录，显式 CLI 路径基于当前目录。默认提交文件为 worker 1 的 `results.json.txt`，其他 worker 自动添加后缀；显式配置路径需要自行区分。提交文件有独占写锁，不支持多个进程同时共享同一个输出路径。输入 save、内部结果与提交文件必须不同。

AID 可省略；CPU 和 CUDA 直接读档均可用 `--aid`，CUDA 队列沿用任务的 AID。没有 AID 时不生成虚构值。程序名和数字版本由 [ecm_program_version.h](../../src/core/ecm_program_version.h) 统一维护，当前为 `ECM-ELF 0.20.0`；Windows 平台号为数字 4。不再使用占位版本或固定 build=1，也不把 Git 哈希写入 program.version。CPU 内部结果的 `source-build` 保留 Git/入口源码指纹，CUDA 内部结果也记录公共程序身份，精确二进制身份由构建清单保存。

版本按实际功能里程碑维护，不由距离上次发布的天数或提交数量直接生成。0.20.0 对应完整 CUDA 多项式 Stage2、Edwards CPU/GPU 流程、宽位数、存档/队列与调优能力；后续兼容功能递增 minor，兼容修正递增 patch。开发状态可由 Git 标签或发布页表达，提交 JSON 的版本保持纯数字点分格式，不要求先创建 Release。

## 提交字段与参考 FFT

内部 CPU JSONL 每行一条完成曲线，保留 sigma、后端、计时、save 行号和 PrMers 布局 CRC32；其 `fft-length=0` 表示该实现没有执行 GWNUM FFT。CUDA 内部 JSONL 保留引擎原有的点、多项式、NTT 和回执诊断。内部文件不直接粘贴到手工提交页面。

提交文件每行一个 JSON 对象，UTF-8、以换行结束：

- NF：`status=NF`、目标、`worktype=ECM`、B1/B2、曲线数、参考 FFT、程序和 UTC 时间；Edwards 标记为 `Edwards:{}`。
- F：增加十进制字符串因子数组、stage 和原始 sigma，Edwards 使用 `Edwards:{sigma:…}`。Stage1 因子不写 B2。
- user、computer、aid、known-factors 仅在有值时输出。Montgomery 不添加 `Edwards` 字段；不冒用 Prime95 或其他 Stage1 程序名。
- 原生提交序列化不附内部诊断与 CRC；历史转换器按要求附 `checksum:{version:1,checksum:…}` 及其规范串需要的 `errors`/`os`。两者都省略 `security-code`，CRC32 不冒充缺少依据的 Prime95 SEC5。

每个因子均验证 `1<f<N` 且 `f|N`。失败或未完成的当前曲线不生成 NF。NF 的 B2 只声明已完成的请求界限，不把 libecm 内部可能超出的尾部覆盖算作额外工作。没有执行的 Prime95 D、poly-size 或 Stage2 FFT 参数不补造。

参考 FFT 来自 Prime95 31.6 `gwnum/mult.asm` 的循环 AVX-512 `zjmptable`，生成器 [generate_p95_fft_reference.py](../../tools/gen/generate_p95_fft_reference.py) 固化其中 241 个 `(max_exp, fftlen)` 容量对。选择满足 `max_exp≥P` 的第一项；例如 M127/M347 为 128，M2203 为 128，M11057 为 512。它是按原始梅森指数生成的报告参考值，不代表 CPU/GPU 实际执行了该长度 FFT，也不保证等于任意机器上 GWNUM 的运行时选择。

完整梅森数仅在 `N+1` 恰为 2 的幂时自动识别指数。余因子必须提供原始 `--exponent P` 与可重复的 `--known-factor F`，严格验证 `(2^P−1)/∏F=N`。CUDA 标准梅森队列自动携带这些字段。一般 N 没有原始梅森指数时仍使用 N 字段，FFT 参考按 `2·bitlength(N)+128` 的保守零填充容量选择；它不等同于 PrimeNet 梅森任务。提交行限制在 2000 字节以内，较大的一般 N 或很长的可选字段会报错，不能据此宣称支持所有大整数的手工提交。

## 去重、批次与恢复

提交文件旁的 `results.json.txt.ecm-ledger/` 保存私有曲线账本和已发布 NF 批次。曲线键绑定实际 N、模型、sigma、B1、B2、stage 和本次因子。相同数学工作重放不增加提交曲线数；内部 JSONL 仍可能增加重算诊断。首次保存的身份及归属不因重放时更换 INI 而重新分配。

成功 F 立即发布。相同目标、界限和身份的未封存 NF 在正常任务结束或可处理的停止时合并，`curves` 为这批新完成的曲线数。已经发布的 NF 行保持不变；后续新曲线生成独立批次，不反复提交累计数。同秒完全相同的新批次等待真实 UTC 时间推进，避免生成字节相同的两行。

输出使用同目录临时文件及 Windows 原子替换。Windows 的访问/共享/锁定错误最多等待 500 ms 重试，其他错误立即返回；持续占用仍报错并保留待发布文件。Stage1 检查点与内部结果使用同一替换规则。崩溃时可能留下账本中已完成但尚未封存的 NF；可以用 CPU 程序恢复 CPU 或 CUDA 的同格式账本，无需重新执行 Stage2：

```powershell
ecm_cpu_stage2.exe --export-results --submission results.json.txt
```

启动计算也会从已有账本重建导出文件。不要删除账本后继续向同名文件累计工作：没有归属账本的现存结果文件会被拒绝覆盖。需归档时一并保留结果文件与账本；新增独立输出可选全新路径。损坏账本校验失败时停止，原始证据应保留。此账本只管理结果发布，不保存曲线内部算术状态。

## 隐私与提交检查

真实 user、computer、aid、checksum/security-code 值仅留在本地。仓库忽略运行时 `*.ini`、`*.jsonl*` 和 `*.json.txt*`，账本名称也受覆盖；受控模板只含空身份。原始 Prime95 样本只在内存中做结构比较，测试身份在运行时随机产生，原始结果、日志和测试样本均写入忽略的 `data/experiments/`。

提交 Git 前运行 [check_result_privacy.py](../../tools/test/check_result_privacy.py)，通过 `--reference-results` 指定本地正式样本、内部结果，通过可重复的 `--ini` 指定实际配置。它检查暂存新增行是否含这些私有值，并拒绝暂存运行时结果文件；只输出通过或失败，不打印匹配值。这个检查依赖提供的本地输入集合，不能替代对新增内容的审阅。

## 历史 CUDA JSONL 转换

[rewrite_stage2_results.py](../../tools/ecm_report/rewrite_stage2_results.py) 将旧的逐曲线 `stage2_completed` NF 回执转换为提交文件。输入始终保留，输出必须使用全新路径；工具没有网络提交功能。使用方法见 [工具说明](../../tools/ecm_report/README.md#历史-stage2-结果转换)。

目标目录必须给出 exponent、N 与 known-factors，工具逐项精确验证 `(2^P−1)/∏f=N`，并要求实际 N 存在 Miller–Rabin 合数见证；找不到见证时停止供人工复核，不将其直接宣判为素数。这可以拦截已知梅森素数及其他疑似素数测试输入，不能代替对 ECM 工作是否有用的人工判断。

相同 N、sigma、B1、曲线和参数化只保留完成 B2 最大的一条；相同 B2 重放只取首次完成。随后只合并源行相邻、save 记录号递增且数学工作、版本和 worker 一致的连续曲线。不同 B2 或中间的重放/覆盖记录会切开批次，不把整份文件按目标无条件累加。合并时间为末条真实完成时间，保留其 UTC 日期，不改成导出时间。原始版本缺失或为哈希时采用显式迁移版本，并在私有审计记录中注明不能重建逐条原始构建。

实际 D/poly-size 仅在原日志完整关联且整批一致时输出；requested_D 不是实际选型证明。GPU 多层 NTT 不提供单一 Prime95 Stage2 FFT 长度，因此不伪填 stage2-fft-length。通用 fft-length 沿用前述正的容量参考。CRC32 按转换后的实际字段重新计算；它不覆盖所有扩展字段，不是整个 JSON 或曲线数的完整性签名。

2026-10-17 本地转换冻结于 `data/experiments/ecm_submission_rewrite_20261017/`：1056 条 PARAM0 NF 回执，剔除 17 条完全重复和 13 条较小 B2 覆盖记录，得到 1026 条不同曲线、7 个连续批次。M3877/M12011/M12037 的实际余因子分别为 3275/11694/11616 位，已知因子均精确重建目标，合数见证均为基数 3。输出为 0.20.0 迁移标签，身份从快照 INI 读取；原文件不含逐条程序版本。所有数据、身份、实际 CRC 和原始日志只保存在忽略目录；该转换没有提交到服务器。

## 验证与边界

[test_ecm_submission.py](../../tools/test/test_ecm_submission.py) 对本地正式 Prime95 文件的 3309 条 ECM 样本提取字段结构，覆盖 CPU/GPU1 的 F/NF、Stage1/Stage2 因子、INI worker/CLI 优先级、空身份、四曲线 NF 合并、重放去重、增量批次不变、损坏导出恢复、未封存完成曲线恢复、无算术恢复、M11213 FFT 选长、外来文件保护和路径别名拒绝。19 项测试通过，原始证据在 `data/experiments/ecm_submission_20261016/gates_06/`。数学回归证据另见同目录 `cpu_regression_02/` 与 `gpu_regression_03/`；GPU 仅使用设备 1 RTX4060。二进制身份随证据冻结，不用当前文件名替代版本依据。

[atomic_publish_fixture.cpp](../../tools/test/atomic_publish_fixture.cpp) 在真实检查点发布入口模拟短暂与持续占用：短暂占用解除后发布成功，持续占用时旧文件和待发布文件均保留。红/绿测试与两次完整 GPU 流程失败证据保留在同目录；有时限恢复后原 GPU 流程通过。独立循环文件测试曾直接观察到 Win32 错误 5；这不证明占用者的身份。

一份可人工审阅的真实输出在同证据目录 `review/`：M347、sigma=17…20、B1=1000、B2=10^8，四条 CPU plain Stage2 完成的 NF 内部行合并为一条提交行，未填入身份/AID，也未发送网络请求。此输入只用于格式验收，不作为有用 ECM 覆盖量或性能基准。

PrimeNet v5 API 0.97 描述的是 `ar` 请求字段与结果类型，不是现代手工提交 JSON 的完整规范；结构还需对照 Prime95 生成器及正式样本。[固定源码与 API 研究](../reference/ECM_RESULT_JSON.md) 列出依据。当前没有服务器接受回执，也未修改 AutoPrimeNet 的固定程序名扫描规则；省略 security-code 后的新程序接受策略仍为待验证条件。

实现入口：[提交序列化](../../src/core/ecm_result_submission.h) 的 `submission_json`、[账本与原子导出](../../src/core/ecm_submission_writer.h) 的 `publish_submission`/`seal_submissions`，CPU 和 CUDA 主程序在验证完成结果后调用它们。
