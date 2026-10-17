# ECM 结果输出

CPU/CUDA Stage1、CPU Edwards Stage2 与 CUDA Stage2 同时写两份结果：逐曲线内部 JSONL 用于数学和运行诊断，`results.json.txt` 用于手工提交。Stage1 仅向提交文件新增因子；Stage1 NF 只进入内部 JSONL 和 save，整批 NF 且使用全新路径时提交文件为空。Stage2 完成后才发布相应的 NF。提交文件使用 Prime95 ECM 的字段结构，程序名如实写为 **ECM-ELF**。用户已确认一个真实 Montgomery NF 批次被手工提交入口接受，范围见下文；生成文件本身不触发网络请求。

## 文件与身份

Stage1 通过 INI 的 `stage1_results_file`、`stage1_submission_file` 配置两份结果，默认内部文件为 `stage1_results.jsonl`。CPU Edwards Stage2 使用 `--results INTERNAL.jsonl --submission results.json.txt`；CUDA 也支持这两个参数，INI 对应 `stage2_results_file` 和 `stage2_submission_file`。CPU 的 `--results` 目前只从 CLI 设置。提交路径的默认值及范围见生成的 [INI 参考](../ECM_INI_REFERENCE.md)。

各程序默认读取 exe 同目录的 `ecm.ini`，`--ini FILE` 显式覆盖。身份键为 `user`、`computer`；空值省略对应 JSON 字段。全局键可由 `[Worker #N]` 覆盖，启动时通过 `--worker N` 选择。CPU 的 `--user`、`--computer` 最后覆盖 INI，显式空字符串也生效。CUDA 从选定 worker 的 INI 读取身份并传给每曲线发布路径。

本地配置可使用以下空值模板，实际值只填入被忽略的 INI：

```ini
user=
computer=
stage2_submission_file=results.json.txt
```

INI 相对输出路径基于 INI 目录，显式 CLI 路径基于当前目录。默认提交文件为 worker 1 的 `results.json.txt`，其他 worker 自动添加后缀；显式配置路径需要自行区分。提交文件有独占写锁，不支持多个进程同时共享同一个输出路径。输入 save、内部结果与提交文件必须不同。GUI 摘要默认使用 `gui_results.jsonl`，不得与 CLI 输出共用文件。

AID 可省略；CPU/CUDA Stage2 直接读档均可用 `--aid`，Stage1 与 CUDA Stage2 队列沿用任务的 AID。没有 AID 时不生成虚构值。程序名和数字版本由 [ecm_program_version.h](../../src/core/ecm_program_version.h) 统一维护，当前源码版本为 `ECM-ELF 0.35.2`；已发布二进制以自身启动输出为准。Windows x64 平台号为数字 4；其他平台不冒用该编号。program.version 使用纯数字点分版本，Git 哈希放在构建身份中。Stage1 的 `source-build` 从链接后生成的 `ecm_stage1_build_identity.txt` 读取入口/共用源码及构建档位指纹；复制程序时可一并保留该文件，缺失时明确使用 development。精确二进制以构建清单中的 SHA256 为准。CPU Stage2 内部结果的 `source-build` 保留 Git/入口源码指纹，GW 构建另含算术模块与 GW 库指纹；CUDA 内部结果也记录公共程序身份，精确二进制身份由构建清单保存。

版本按 `x.y.z` 维护：新增功能递增 y，并将 z 归零；修复递增 z。版本不随单次构建或 Git 提交自动变化，发布功能或修复时统一更新该头文件；构建依赖会使 CPU/CUDA 输出同步采用新版本。创建 Release 不是生成结果的前提。

## 提交字段与参考 FFT

内部 CPU JSONL 每行一条完成曲线，保留 sigma、后端、计时、save 行号和 PrMers 布局 CRC32；libecm 的 `fft-length=0` 表示未执行 GWNUM FFT；gwpoly 则记录实际选用的 GW FFT 长度，并保留真实 D/P 和引擎误差诊断。CUDA 内部 JSONL 保留引擎原有的点、多项式、NTT 和回执诊断。内部文件不直接粘贴到手工提交页面。

提交文件每行一个 JSON 对象，UTF-8、以换行结束：

- NF：`status=NF`、目标、`worktype=ECM`、B1/B2、曲线数、参考 FFT、程序和 UTC 时间；Edwards 标记为 `Edwards:{}`。
- F：增加十进制字符串因子数组、stage 和原始 sigma，Edwards 使用 `Edwards:{sigma:…}`。Stage1 因子不写 B2。
- user、computer、aid、known-factors 仅在有值时输出。Montgomery 不添加 `Edwards` 字段；不冒用 Prime95 或其他 Stage1 程序名。
- 有实际 Stage2 几何参数时，字段连续排列为 `b2 → d → poly-size`。CUDA 直接使用已完成引擎返回的 D 和 baby 多项式大小，内部 JSONL 也在 B2 后记录这两个字段。Stage1 因子没有这些字段；CPU gwpoly 记录实际 D/P，并在其后输出真实 `stage2-fft-length`；libecm 当前没有导出对应几何参数，保持省略。
- 原生提交序列化不附内部诊断与 CRC；历史转换器按要求附 `checksum:{version:1,checksum:…}` 及其规范串需要的 `errors`/`os`。两者都省略 `security-code`，CRC32 不冒充缺少依据的 Prime95 SEC5。

完整梅森数或带已知因子的目标使用原始 exponent；其他明确的 k·b^n+c 输入写 k/b/n/c 并验证实际模数。其余整数写完整 N；超过 2000 个十进制字符时 N 使用 JSON 字符串。这个长整数扩展保留精确本地结果，不能据此宣称满足 PrimeNet 消息容量或已获手工提交入口认可；梅森数与 k·b^n+c 提交记录仍执行消息容量检查。

每个因子均验证 `1<f<N` 且 `f|N`。失败或未完成的当前曲线不生成 NF。提交 NF 的 B2 只声明已完成的 Stage2 请求界限，不把 libecm 内部可能超出的尾部覆盖算作额外工作。没有执行的 Prime95 D、poly-size 或 Stage2 FFT 参数不补造。Stage1 NF 的 B2=B1 仅用于内部记录。

参考 FFT 来自 Prime95 31.6 `gwnum/mult.asm` 的循环 AVX-512 `zjmptable`，生成器 [generate_p95_fft_reference.py](../../tools/gen/generate_p95_fft_reference.py) 固化其中 241 个 `(max_exp, fftlen)` 容量对。选择满足 `max_exp≥P` 的第一项；例如 M127/M347 为 128，M2203 为 128，M11057 为 512。它是按原始梅森指数生成的报告参考值，不代表 CPU/GPU 实际执行了该长度 FFT，也不保证等于任意机器上 GWNUM 的运行时选择。

完整梅森数仅在 `N+1` 恰为 2 的幂时自动识别指数。余因子必须提供原始 `--exponent P` 与可重复的 `--known-factor F`，严格验证 `(2^P−1)/∏F=N`。CUDA 标准梅森队列自动携带这些字段。一般 N 没有原始梅森指数时仍使用 N 字段，FFT 参考按 `2·bitlength(N)+128` 的保守零填充容量选择；它不等同于 PrimeNet 梅森任务。梅森数与 k·b^n+c 提交行要求少于 2000 字节，过长的可选字段会报错；一般 N 的完整本地导出可以超过此长度，不代表服务器接受该扩展。

## checksum 计算方法

`checksum.version=1` 对应 PrMers ECM 规范串的 CRC32。计算对象是按约定提取的字段值，不是 JSON 文本。当前 CPU 内部结果和历史转换器生成 checksum；原生 `results.json.txt` 序列化不附 checksum。两条带 checksum 的路径都使用下列 16 个槽位，按次序用 ASCII 分号 `;` 连接，空槽保留，最后一个槽后不加分号：

1. 目标：十进制 exponent。一般整数及非梅森 k·b^n+c 扩展使用 `N=` 加实际模数的完整十进制值；转换器只接受经过目标关系验证的梅森任务。
2. 固定字符串 `ECM`。
3. 本次 factors：按数组原顺序用逗号连接十进制因子，无因子时为空。
4. known-factors：按数组原顺序用逗号连接十进制因子，没有时为空。
5. 十进制 b1。
6. 十进制已完成 b2，仅当 b2>B1 时写入；Stage1 因子没有 b2，仅完成 Stage1 的 NF 使用 B2=B1，两者此槽均为空。
7. 曲线及 sigma 标记，取值规则见下文。
8. 十进制 fft-length，直接采用这份结果实际记录的值。
9. 固定字符串 `invariant:0`，对应已验证成功结果中的 invariant 错误计数。
10. program.name，即 `ECM-ELF`。
11. program.version，即该结果记录的数字版本。
12. program.kernel/subversion 槽，本仓库当前为空。
13. program.details 槽，本仓库当前为空。
14. os.os，即 `Windows`。
15. os.architecture，即 `x86_64`。
16. timestamp，保留结果中的 UTC 字符串 `YYYY-MM-DD HH:MM:SS`。

Stage1 与 CPU Stage2 使用共用 [内部序列化模块](../../src/core/ecm_internal_result.h)；提交字段和数学一致性验证使用 [公共结果模型](../../src/core/ecm_result_submission.h)。`stage1-engine.batch-seconds` 是整个批次的引擎及 text-save 发布墙钟时间，包含指数、曲线准备、运行中检查点和算术，排除外层批次档案读取、输入预检与双 JSON 发布；它在逐曲线记录中重复，不能求和当作总耗时。

Stage1 内部 `fft-length=0` 表示没有 GWNUM 变换长度，提交版本使用正的 Prime95 参考容量；`stage1-engine.coverage=B1` 不表示执行过 Stage2。

第 7 槽：Edwards 先写 `E`，再写存在的十进制 sigma；Montgomery 只写存在的 sigma。CPU 逐曲线内部结果始终保留 sigma，故 Edwards NF/F 都为 `E` 加 sigma。转换器当前只处理 NF：Edwards 输出 `Edwards:{}`，此槽为 `E`；Montgomery 没有 Edwards 对象或 sigma，此槽为空。不从原始逐曲线回执额外补入合并结果已省略的 sigma，也不猜测 torsion 后缀。

设规范串为 S，以 UTF-8 编码为字节序列；不包含 BOM、JSON 引号、转义后的 JSON 表达、行尾换行或终止零字节。整数使用普通十进制，无千位分隔符或科学计数法；因子顺序、大小写、空槽和时间空格都参与校验。

CRC 使用标准反射 CRC-32/ISO-HDLC：宽度 32 位，初值 `0xFFFFFFFF`，反射多项式 `0xEDB88320`（正向表示为 `0x04C11DB7`），最终异或 `0xFFFFFFFF`。对每个字节 b，先令 r ← r XOR b，再执行 8 次：若最低位为 1，则 r ← (r >> 1) XOR `0xEDB88320`；否则 r ← r >> 1。全部字节处理后，C = r XOR `0xFFFFFFFF`。将 C 写成大写、左补零至 8 位的十六进制字符串，放入 checksum 对象的 checksum 字段，版本字段为整数 1。Python 的 `zlib.crc32` 对 S 的 UTF-8 字节执行同一计算。

CPU 内部结果使用本行的实际 fft-length（libecm 为 0，gwpoly 为真实 GW 长度）和逐曲线 sigma；转换后提交文件使用正的参考 FFT、NF 模型标记和合并后的真实完成时间，必须从最终记录重新计算 CRC，不能拷贝内部旧值。版本、参与计算的字段或其表示改变时也须重算。单纯调整 JSON 键顺序、空格，以及移动 d/poly-size 不改变规范串。

curves、d、poly-size、stage2-fft-length、status、stage、user、computer、aid、program.port 和 source-build 不在这 16 个槽位中；CRC 不保护这些字段。它也不是 Prime95 的 security-code/SEC5。第三方源码依据见 [PrMers CRC32 研究](../reference/ECM_RESULT_JSON.md#prmers-crc32)，本仓库入口为 [转换器 checksum](../../tools/ecm_report/rewrite_stage2_results.py) 与 [CPU result](../../src/core/ecm_cpu_stage2_main.cpp)；[独立按位校验](../../tools/test/test_stage2_result_rewrite.py) 用于复核库计算结果。真实运行产生的 checksum 值保留在忽略目录。

## 去重、批次与恢复

提交文件旁的 `results.json.txt.ecm-ledger/` 保存私有曲线账本和已发布 NF 批次。曲线键绑定实际 N、模型、sigma、B1、已执行的 B2、stage 和本次因子。Stage1 因子没有执行 B2，改变续算界限或由 Stage2 重读该因子不增加提交记录；旧账本中由 B2 差异形成的同因子回执在导出时合并，保留最早回执身份，不改写原始私有记录。相同数学工作重放不增加提交曲线数；Stage1 内部 JSONL 同样使用独立私有回执去重，逐曲线持久化回执后在批次结束时统一导出两份文件；Stage2 内部 JSONL 仍可能增加重算诊断。首次保存的身份及归属不因重放时更换 INI 而重新分配。

只完成 Stage1 的 NF 在内部 JSONL 中写 B2=B1，不向提交账本写入 NF 回执；找到因子的 Stage1 F 省略 B2。全新提交路径即使没有因子，也创建归属账本与空文件。Stage1 重放保留已有 Stage2 提交批次，也不封存 Stage2 尚未发布的 NF；由 Stage2 自行完成封存与恢复。

Stage1 在批次完成后统一导出因子；Stage2 成功 F 立即发布。相同目标、界限和身份的未封存 Stage2 NF 在正常任务结束或可处理的停止时合并，`curves` 为这批新完成的曲线数。已经发布的 NF 行保持不变；后续新曲线生成独立批次，不反复提交累计数。同秒完全相同的新批次等待真实 UTC 时间推进，避免生成字节相同的两行。

输出使用同目录临时文件及 Windows 原子替换。Windows 的访问/共享/锁定错误最多等待 500 ms 重试，其他错误立即返回；持续占用仍报错并保留待发布文件。Stage1 检查点与内部结果使用同一替换规则。崩溃时可能留下账本中已完成但尚未封存的 NF；可以用 CPU 程序恢复 CPU 或 CUDA 的同格式账本，无需重新执行 Stage2：

```powershell
ecm_cpu_stage2.exe --export-results --submission results.json.txt
```

启动计算也会从已有账本重建导出文件。不要删除账本后继续向同名文件累计工作：没有归属账本的现存结果文件会被拒绝覆盖。需归档时一并保留结果文件与账本；新增独立输出可选全新路径。损坏账本校验失败时停止，原始证据应保留。此账本只管理结果发布，不保存曲线内部算术状态。

## 隐私与提交检查

真实 user、computer、aid、checksum/security-code 值仅留在本地。仓库忽略运行时 `*.ini`、`*.jsonl*`、`*.json.txt*`、worker 后缀 `*.json_*.txt*`、`*.ckpt*` 及私有 `*.ecm-ledger*`；受控模板只含空身份。原始 Prime95 样本只在内存中做结构比较，测试身份在运行时随机产生，原始结果、日志和测试样本均写入忽略的 `data/experiments/`。

提交 Git 前运行 [check_result_privacy.py](../../tools/test/check_result_privacy.py)，通过 `--reference-results` 指定本地正式样本、内部结果，通过可重复的 `--ini` 指定实际配置。它检查暂存新增行是否含这些私有值，并拒绝暂存运行时结果文件；只输出通过或失败，不打印匹配值。这个检查依赖提供的本地输入集合，不能替代对新增内容的审阅。

## 历史 CUDA JSONL 转换

[rewrite_stage2_results.py](../../tools/ecm_report/rewrite_stage2_results.py) 将旧的逐曲线 `stage2_completed` NF 回执转换为提交文件。输入始终保留，输出必须使用全新路径；工具没有网络提交功能。使用方法见 [工具说明](../../tools/ecm_report/README.md#历史-stage2-结果转换)。

目标目录必须给出 exponent、N 与 known-factors，工具逐项精确验证 `(2^P−1)/∏f=N`，并要求实际 N 存在 Miller–Rabin 合数见证；找不到见证时停止供人工复核，不将其直接宣判为素数。这可以拦截已知梅森素数及其他疑似素数测试输入，不能代替对 ECM 工作是否有用的人工判断。

相同 N、sigma、B1、曲线和参数化只保留完成 B2 最大的一条；相同 B2 重放只取首次完成。随后只合并源行相邻、save 记录号递增且数学工作、版本和 worker 一致的连续曲线。不同 B2 或中间的重放/覆盖记录会切开批次，不把整份文件按目标无条件累加。合并时间为末条真实完成时间，保留其 UTC 日期，不改成导出时间。原始版本缺失或为哈希时采用显式迁移版本，并在私有审计记录中注明不能重建逐条原始构建。

实际 d/poly-size 来自已完成回执中的实际字段，或完整关联的原日志；整批一致且证据不冲突时才输出，位置紧随 b2。requested_D 不是实际选型证明。GPU 多层 NTT 不提供单一 Prime95 Stage2 FFT 长度，因此不伪填 stage2-fft-length。通用 fft-length 沿用前述正的容量参考。CRC32 按转换后的实际字段重新计算；它不覆盖所有扩展字段，不是整个 JSON 或曲线数的完整性签名。

本地转换证据冻结于 `data/experiments/ecm_submission_rewrite_20261017/`：1056 条 PARAM0 NF 回执，剔除 17 条完全重复和 13 条较小 B2 覆盖记录，得到 1026 条不同曲线、7 个连续批次。M3877/M12011/M12037 的实际余因子分别为 3275/11694/11616 位，已知因子均精确重建目标，合数见证均为基数 3。该冻结输出使用 0.20.0 迁移标签，身份从快照 INI 读取；原文件不含逐条程序版本。冻结文件保留原版本与原文，已有合法数字版本的回执也不会被转换器改名。所有数据、身份、实际 CRC 和原始日志只保存在忽略目录。

用户于 2026-10-17 提供的服务器回执确认：其中 M3877 的 NF 批次、B1=260,000,000、B2=2,600,000,000,000 已接受 1 行，状态码 0，CPU credit 为 753.2415 GHz-days。该结果使用真实 ECM-ELF 身份、正的参考 fft-length 和 CRC32，省略 security-code。此回执证明这一批提交可用，不扩展为 Edwards 或因子结果的服务器验收结论。已接受批次保持冻结，不能因版本或排版更新再次申报。回执摘要和冻结文件身份保存在 `data/experiments/ecm_submission_v031_20261017/server_acceptance.json`。

## 验证与边界

[test_ecm_submission.py](../../tools/test/test_ecm_submission.py) 对本地正式 Prime95 文件的 3317 条 ECM 样本提取字段结构，覆盖 CPU/GPU1 的 F/NF、Stage1/Stage2 因子、INI worker/CLI 优先级、空身份、四曲线 NF 合并、重放去重、增量批次不变、损坏导出恢复、未封存完成曲线恢复、无算术恢复、M11213 FFT 选长、外来文件保护和路径别名拒绝。版本 0.34.1 的 22 项检查通过，包括 Stage1 因子跨 B2 去重、旧因子账本导出去重、CUDA 实际 D/P、字段顺序及 Stage1 因子省略参数；证据在 `data/experiments/edwards_stage2_opt_20261017/submission_gate_01/`。

[Stage1 流程验收](../../tools/test/test_stage1_usability.py) 的 27 组检查覆盖单批次检查点及全部 CPU/CUDA 选择路径，实际串接 CPU Stage2，验证 Stage1 NF 仅内部记录、因子不重复计入、Stage1 不封存待发布的 Stage2 NF。64 曲线 GPU NF 批次及重放另验证内部和 save 各 64 行、提交为空，并独立复核 147 条 Stage1 内部 CRC32。GPU 仅使用设备 1 RTX4060；输入、二进制与冻结证据位置见 `data/experiments/stage1_usability_20261017/validation_final_manifest.json`，这些检查不作为性能基准或服务器验收。

[转换器回归](../../tools/test/test_stage2_result_rewrite.py) 的 24 项检查通过，涵盖新回执参数、旧日志恢复、冲突/混合参数省略、版本保留、字段顺序和实际 7 批输出的独立按位 CRC。另对当前 CPU 输出的 59 条内部 checksum 复核通过；历史 7 批记录字段值与 CRC 保持一致，已接受导出文件原文未改。证据在同目录 `converter_gates_03/`、`crc_and_history_audit.json`。独立数学回归另见 `data/experiments/ecm_submission_20261016/cpu_regression_02/` 与 `gpu_regression_03/`；测试证据以各自冻结二进制为准。

[atomic_publish_fixture.cpp](../../tools/test/atomic_publish_fixture.cpp) 在真实检查点发布入口模拟短暂与持续占用：短暂占用解除后发布成功，持续占用时旧文件和待发布文件均保留。红/绿测试与两次完整 GPU 流程失败证据保留在同目录；有时限恢复后原 GPU 流程通过。独立循环文件测试曾直接观察到 Win32 错误 5；这不证明占用者的身份。

一份可人工审阅的真实输出在同证据目录 `review/`：M347、sigma=17…20、B1=1000、B2=10^8，四条 CPU plain Stage2 完成的 NF 内部行合并为一条提交行，未填入身份/AID，也未发送网络请求。此输入只用于格式验收，不作为有用 ECM 覆盖量或性能基准。

PrimeNet v5 API 0.97 描述的是 `ar` 请求字段与结果类型，不是现代手工提交 JSON 的完整规范；结构还需对照 Prime95 生成器及正式样本。[固定源码与 API 研究](../reference/ECM_RESULT_JSON.md) 列出依据。上述 Montgomery NF 批次已有用户提供的服务器接受回执；Edwards、因子结果及其他提交入口仍需各自验证。AutoPrimeNet 的固定程序名扫描规则未修改。

实现入口：[提交序列化](../../src/core/ecm_result_submission.h) 的 `submission_json`、[账本与原子导出](../../src/core/ecm_submission_writer.h) 的 `publish_submission`/`seal_submissions`，CPU 和 CUDA 主程序在验证完成结果后调用它们。
