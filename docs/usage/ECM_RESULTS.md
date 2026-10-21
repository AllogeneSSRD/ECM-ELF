# ECM 结果输出

CPU/CUDA Stage1、CPU Edwards Stage2 与 CUDA Stage2 同时维护逐曲线内部 JSONL 和提交格式 `results.json.txt`。**NF 仅在整条 worktodo 任务完成时导出；有 NF 时，Stage2 每条完成的任务生成一行 NF，曲线数是该任务完成且未找到因子的曲线总数，参数取最后完成的 NF 曲线。中途停止只保存内部结果、进度和待发布回执。发现并验证因子时立即额外导出一行 F，固定 curves=1，保留 sigma。** 直接 CLI 运行所选曲线范围视为一个任务。Stage1 NF 的 B2=B1，Stage2 NF 记录最后一条 NF 曲线实际请求的 B2。提交结构参考 Prime95，程序身份如实记录为 ECM-ELF，并附 CRC32；生成文件不触发网络请求。

## 文件与身份

Stage1 通过 TOML 的 `stage1_results_file`、`stage1_submission_file` 配置两份结果，默认内部文件为 `stage1_results.jsonl`。CPU Edwards Stage2 使用 `--results INTERNAL.jsonl --submission results.json.txt`；CUDA 也支持这两个参数，TOML 对应 `stage2_results_file` 和 `stage2_submission_file`。CPU 的 `--results` 目前只从 CLI 设置。提交路径的默认值及范围见生成的 [TOML 参考](../ECM_TOML_REFERENCE.md)。

各程序默认读取 exe 同目录的 `ecm.toml`，`--config FILE` 显式覆盖。身份键为 `user`、`computer`；空值省略对应 JSON 字段。全局键可由 `[Worker #N]` 覆盖，启动时通过 `--worker N` 选择。CPU 的 `--user`、`--computer` 最后覆盖 TOML，显式空字符串也生效。CUDA 从选定 worker 的 TOML 读取身份并传给每曲线发布路径。

本地配置可使用以下空值模板，实际值只填入被忽略的 TOML：

```toml
[shared]
user = ""
computer = ""

[stage2]
stage2_submission_file = "results.json.txt"
```

TOML 相对输出路径基于 TOML 目录，显式 CLI 路径基于当前目录。默认提交文件为 worker 1 的 `results.json.txt`，其他 worker 自动添加后缀；显式配置路径需要自行区分。提交文件有独占写锁，不支持多个进程同时共享同一个输出路径。输入 save、内部结果与提交文件必须不同。GUI 的 [Result 面板](GUI.md#result) 只读这些提交文件，提供筛选和原文复制，不参与结果写入或账本维护。

AID 可省略；CPU/CUDA Stage2 直接读档均可用 `--aid`，Stage1 与 CUDA Stage2 队列沿用任务的 AID。没有 AID 时不生成虚构值。程序名和数字版本由 [ecm_program_version.h](../../src/core/ecm_program_version.h) 统一维护。Windows x64 平台号为数字 4；其他平台不冒用该编号。program.version 使用纯数字点分版本，Git 哈希放在构建身份中。Stage1 的 `source-build` 从链接后生成的 `ecm_stage1_build_identity.txt` 读取入口/共用源码及构建档位指纹；复制程序时可一并保留该文件，缺失时明确使用 development。精确二进制以构建清单中的 SHA256 为准。CPU Stage2 内部结果的 `source-build` 保留 Git/入口源码指纹，GW 构建另含算术模块与 GW 库指纹；CUDA 内部结果的 source-build 直接记录当前可执行文件 SHA256，构建清单另存编译条件。

版本按 `x.y.z` 维护：新增功能递增 y，并将 z 归零；修复递增 z。版本不随单次构建或 Git 提交自动变化，发布功能或修复时统一更新该头文件；构建依赖会使 CPU/CUDA 输出同步采用新版本。创建 Release 不是生成结果的前提。

## 提交字段与参考 FFT

内部 JSONL 的公共顺序为 status、目标、worktype、b1/b2、d/poly-size、curves、曲线/sigma、fft-length、因子、程序/系统/时间与 checksum。CPU libecm 和 CUDA 的实际 fft-length 为 0，gwpoly 记录真实 GW 长度。执行诊断集中于 stage1-engine 或 stage2-engine；CUDA 保留 queue_receipt 和可恢复调优样本 tune_observation，调优计划及检查计数保留在 stage2-engine。关闭的 relocation/Auto B2 不生成整组零值。

梅森目标用 exponent 与 known-factors 精确表示，不重复存储 N_hex，也不重复存储提交路径、参考表名称和模型。一般整数保留必要的完整 N。Python 分析入口通过 tools/ecm_report/result_format.py 在内存中兼容旧的 B1/B2、N_hex 等别名，磁盘新格式不再写这些重复字段。内部文件不直接粘贴到手工提交页面。

提交文件每行一个 JSON 对象，UTF-8、以换行结束：

- NF：`status=NF`、目标、`worktype=ECM`、B1/B2、曲线数、参考 FFT、程序和 UTC 时间；Edwards 标记为 `Edwards:{}`。
- F：增加十进制字符串因子数组、stage 和原始 sigma，Edwards 使用 `Edwards:{sigma:…}`。Stage1 因子不写 B2。
- user、computer、aid、known-factors 仅在有值时输出。Montgomery 不添加 `Edwards` 字段；不冒用 Prime95 或其他 Stage1 程序名。
- 有实际 Stage2 几何参数时，字段连续排列为 `b2 → d → poly-size`。CUDA 直接使用已完成引擎返回的 D 和 baby 多项式大小，内部 JSONL 也在 b2 后记录这两个字段。Stage1 因子没有这些字段；CPU gwpoly 记录实际 D/P，并在其后输出真实 `stage2-fft-length`；libecm 当前没有导出对应几何参数，保持省略。
- 所有原生提交输出与转换器均附 `checksum:{version:1,checksum:…}` 及规范串需要的 `errors`/`os`。均省略 `security-code`，CRC32 不冒充缺少依据的 Prime95 SEC5。

完整梅森数或带已知因子的目标使用原始 exponent；其他明确的 k·b^n+c 输入写 k/b/n/c 并验证实际模数。其余整数写完整 N；超过 2000 个十进制字符时 N 使用 JSON 字符串。这个长整数扩展保留精确本地结果，不能据此宣称满足 PrimeNet 消息容量或已获手工提交入口认可；梅森数与 k·b^n+c 提交记录仍执行消息容量检查。

每个因子均验证 `1<f<N` 且 `f|N`。失败或未完成的当前曲线不生成 NF。提交 NF 的 B2 只声明已完成的 Stage2 请求界限，不把 libecm 内部可能超出的尾部覆盖算作额外工作。没有执行的 Prime95 D、poly-size 或 Stage2 FFT 参数不补造。Stage1 NF 的 B2=B1 同时用于内部和提交记录，仅表示 Stage1 覆盖。

参考 FFT 来自 Prime95 31.6 `gwnum/mult.asm` 的循环 AVX-512 `zjmptable`，生成器 [generate_p95_fft_reference.py](../../tools/gen/generate_p95_fft_reference.py) 固化其中 241 个 `(max_exp, fftlen)` 容量对。选择满足 `max_exp≥P` 的第一项；例如 M127/M347 为 128，M2203 为 128，M11057 为 512。它是按原始梅森指数生成的报告参考值，不代表 CPU/GPU 实际执行了该长度 FFT，也不保证等于任意机器上 GWNUM 的运行时选择。

完整梅森数仅在 `N+1` 恰为 2 的幂时自动识别指数。余因子必须提供原始 `--exponent P` 与可重复的 `--known-factor F`，严格验证 `(2^P−1)/∏F=N`。CUDA 标准梅森队列自动携带这些字段。一般 N 没有原始梅森指数时仍使用 N 字段，FFT 参考按 `2·bitlength(N)+128` 的保守零填充容量选择；它不等同于 PrimeNet 梅森任务。梅森数与 k·b^n+c 提交行要求少于 2000 字节，过长的可选字段会报错；一般 N 的完整本地导出可以超过此长度，不代表服务器接受该扩展。

## checksum 计算方法

`checksum.version=1` 对应 PrMers ECM 规范串的 CRC32。计算对象是按约定提取的字段值，不是 JSON 文本。CPU/CUDA 内部记录、原生提交文件和转换器共用此规则；从最终记录计算，NF 合并修改完成时间后再次计算。所有路径都使用下列 16 个槽位，按次序用 ASCII 分号 `;` 连接，空槽保留，最后一个槽后不加分号：

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

Stage1、CPU/CUDA Stage2 使用共用 [内部序列化模块](../../src/core/ecm_internal_result.h)；提交字段和数学一致性验证使用 [公共结果模型](../../src/core/ecm_result_submission.h)。`stage1-engine.batch-seconds` 是整个批次的引擎及 text-save 发布墙钟时间，包含指数、曲线准备、运行中检查点和算术，包括命中时的即时因子发布，排除外层批次档案读取、输入预检与最终双 JSON 发布；它在逐曲线记录中重复，不能求和当作总耗时。

Stage1 内部 `fft-length=0` 表示没有 GWNUM 变换长度，提交版本使用正的 Prime95 参考容量；`stage1-engine.coverage=B1` 不表示执行过 Stage2。

第 7 槽：Edwards 先写 `E`，再写存在的十进制 sigma；Montgomery 只写存在的 sigma。逐曲线内部结果始终保留 sigma，故 Edwards NF/F 都为 `E` 加 sigma。转换器当前只处理 NF：Edwards 输出 `Edwards:{}`，此槽为 `E`；Montgomery 没有 Edwards 对象或 sigma，此槽为空。不从原始逐曲线回执额外补入合并结果已省略的 sigma，也不猜测 torsion 后缀。

设规范串为 S，以 UTF-8 编码为字节序列；不包含 BOM、JSON 引号、转义后的 JSON 表达、行尾换行或终止零字节。整数使用普通十进制，无千位分隔符或科学计数法；因子顺序、大小写、空槽和时间空格都参与校验。

CRC 使用标准反射 CRC-32/ISO-HDLC：宽度 32 位，初值 `0xFFFFFFFF`，反射多项式 `0xEDB88320`（正向表示为 `0x04C11DB7`），最终异或 `0xFFFFFFFF`。对每个字节 b，先令 r ← r XOR b，再执行 8 次：若最低位为 1，则 r ← (r >> 1) XOR `0xEDB88320`；否则 r ← r >> 1。全部字节处理后，C = r XOR `0xFFFFFFFF`。将 C 写成大写、左补零至 8 位的十六进制字符串，放入 checksum 对象的 checksum 字段，版本字段为整数 1。Python 的 `zlib.crc32` 对 S 的 UTF-8 字节执行同一计算。

内部结果使用本行的实际 fft-length（libecm 为 0，gwpoly 为真实 GW 长度）和逐曲线 sigma；转换后提交文件使用正的参考 FFT、NF 模型标记和合并后的真实完成时间，必须从最终记录重新计算 CRC，不能拷贝内部旧值。版本、参与计算的字段或其表示改变时也须重算。单纯调整 JSON 键顺序、空格，以及移动 d/poly-size 不改变规范串。

curves、d、poly-size、stage2-fft-length、status、stage、user、computer、aid、program.port 和 source-build 不在这 16 个槽位中；CRC 不保护这些字段。它也不是 Prime95 的 security-code/SEC5。第三方源码依据见 [PrMers CRC32 研究](../reference/ECM_RESULT_JSON.md#prmers-crc32)，本仓库入口为 [转换器 checksum](../../tools/ecm_report/rewrite_stage2_results.py) 与 [CPU result](../../src/core/ecm_cpu_stage2_main.cpp)；[独立按位校验](../../tools/test/test_stage2_result_rewrite.py) 用于复核库计算结果。真实运行产生的 checksum 值保留在忽略目录。

## 去重、批次与恢复

三类账本是程序内部数据，位于结果文件所在目录的 `data/` 下，与 `data/experiments/` 并列，保留原文件夹名称。例如：

- `data/stage1_results.jsonl.ecm-ledger/`：逐曲线内部回执、去重及导出顺序。
- `data/results.json.txt.ecm-ledger/`：worker 1 的提交回执与不可变 NF 批次。
- `data/results.json_2.txt.ecm-ledger/`：worker 2 的独立提交回执与批次。

它们用于崩溃恢复与避免重复计算工作被重复计数，不能作为无用缓存删除。每份回执使用绑定数学工作的键及独立文件完整性散列；内部 CRC32 是另一项校验。提交账本中的 `publication.order` 保存带完整性散列的发布历史，包含 F/NF 的实际发布顺序和原始行内容；结果文件缺失时可据此重建。短期独占锁仍位于输出旁，以便和旧程序互斥，关闭后删除。

首次访问时，在独占锁内将旧的输出旁账本整体重命名到 data/；不改曲线回执或已经发布批次内容。两处同时存在时拒绝自动合并，以免丢失归属或重复计数。新内部账本为 V2，记录显式递增导出序号；读 V1 时保留原回执写入时间顺序，时间完全相同的旧回执只能以数学键稳定排序。默认随机 sigma 无需连续；固定 sigma 的序列和曲线身份不因此改变。旧导出曾按 sigma 字符串排序，不表示 save 缺失曲线。

曲线键绑定实际 N、模型、sigma、B1、B2、stage 和因子；Stage1 因子的键不包含续算 B2，Stage2 重读同一因子也只保留首次回执。内部 Stage1 回执不重复追加；Stage2 内部日志可以保留重算诊断。已有身份、时间、版本及已封存 NF 批次不因重放而变更。

NF 回执还绑定任务范围。Stage1 不会封存 Stage2 未完成的 NF，CUDA 停止时不会封存未完成任务；同一任务恢复后完成剩余曲线才生成提交行。Stage2 同一任务内即使自动选出的 B2、D、P 不同，也合并为一行，采用最后完成的 NF 曲线的参数与 UTC 时间，并重新计算 checksum；逐曲线内部 JSONL 保留各自实际参数。参数不同时，汇总行中的 B2 不表示每条曲线都完成到该界限，逐曲线覆盖情况应查内部回执。F 一经验证立即单独导出，永远为 1 曲线。已发布行保持不变，新任务生成独立批次；同秒完全相同的新批次等待真实 UTC 推进。
提交文件以二进制追加方式写入，新结果只加到末尾；已有行的内容、顺序和换行字节保持不变，末行没有换行时先追加分隔换行。F 与 NF 按实际发布顺序混排，不按状态、目标、散列键或结果时间戳重排。首次使用现有提交文件时，以现有内容建立发布历史；尚未导出的回执按回执写入时间补到末尾，同一时间以回执键稳定排序。

在独占锁内，先通过临时文件原子保存发布历史，再向提交文件追加。写入中断后，只要现有文件是发布历史的字节前缀，重试会补齐未写入的后缀，包含最后一行只写了一部分的情况；不会重复追加。文件缺失时按记录顺序恢复；已有文件与历史不一致或历史校验失败时报错，不覆盖文件。不要手动修改提交文件或单独删除发布历史；归档时一并保留提交文件及账本。

账本原子更新、Stage1 检查点与内部结果仍使用同目录临时文件替换规则。Windows 的原子替换遇到访问/共享/锁定错误时最多等待 500 ms 重试；提交文件追加失败时报错，发布历史保留待追加内容供恢复。崩溃时也可能留下尚未封存的 NF。正常应续算原任务直至完整完成；只有确认账本中的全部待发布工作均已完成、希望主动导出时，才使用 CPU 的显式恢复命令。此命令会封存待发布回执，无需重新执行算术，不是自动停止流程：

```powershell
ecm_cpu_stage2.exe --export-results --submission results.json.txt
```

正常任务完成或找到因子时从账本更新导出；启动与未完成 NF 不主动改写提交文件。CPU 显式恢复命令按 Stage2 任务范围封存待发布回执。不要删除账本后继续向同名文件累计工作：没有归属账本的现存结果文件会被拒绝覆盖。需归档时一并保留结果文件与账本；新增独立输出可选全新路径。损坏账本校验失败时停止，原始证据应保留。此账本只管理结果发布，不保存曲线内部算术状态。

## 隐私与提交检查

真实 user、computer、aid、checksum/security-code 值仅留在本地。仓库忽略运行时 `*.ini`、`*.jsonl*`、`*.json.txt*`、worker 后缀 `*.json_*.txt*`、`*.ckpt*` 及私有 `*.ecm-ledger*`；受控模板只含空身份。原始 Prime95 样本只在内存中做结构比较，测试身份在运行时随机产生，原始结果、日志和测试样本均写入忽略的 `data/experiments/`。

提交 Git 前运行 [check_result_privacy.py](../../tools/test/check_result_privacy.py)，通过 `--reference-results` 指定本地正式样本、内部结果，通过可重复的 `--config` 指定实际配置。它检查暂存新增行是否含这些私有值，并拒绝暂存运行时结果文件；只输出通过或失败，不打印匹配值。这个检查依赖提供的本地输入集合，不能替代对新增内容的审阅。

## 历史 CUDA JSONL 转换

[rewrite_stage2_results.py](../../tools/ecm_report/rewrite_stage2_results.py) 将旧的逐曲线 `stage2_completed` NF 回执转换为提交文件。输入始终保留，输出必须使用全新路径；工具没有网络提交功能。使用方法见 [工具说明](../../tools/ecm_report/README.md#历史-stage2-结果转换)。

目标目录必须给出 exponent、N 与 known-factors，工具逐项精确验证 `(2^P−1)/∏f=N`，并要求实际 N 存在 Miller–Rabin 合数见证；找不到见证时停止供人工复核，不将其直接宣判为素数。这可以拦截已知梅森素数及其他疑似素数测试输入，不能代替对 ECM 工作是否有用的人工判断。

相同 N、sigma、B1、曲线和参数化只保留完成 B2 最大的一条；相同 B2 重放只取首次完成。随后只合并源行相邻、save 记录号递增且数学工作、版本和 worker 一致的连续曲线。不同 B2 或中间的重放/覆盖记录会切开批次，不把整份文件按目标无条件累加。合并时间为末条真实完成时间，保留其 UTC 日期，不改成导出时间。原始版本缺失或为哈希时采用显式迁移版本，并在私有审计记录中注明不能重建逐条原始构建。

实际 d/poly-size 来自已完成回执中的实际字段，或完整关联的原日志；整批一致且证据不冲突时才输出，位置紧随 b2。requested_D 不是实际选型证明。GPU 多层 NTT 不提供单一 Prime95 Stage2 FFT 长度，因此不伪填 stage2-fft-length。通用 fft-length 沿用前述正的容量参考。CRC32 按转换后的实际字段重新计算；它不覆盖所有扩展字段，不是整个 JSON 或曲线数的完整性签名。

本地转换证据冻结于 `data/experiments/ecm_submission_rewrite_20261017/`：1056 条 PARAM0 NF 回执，剔除 17 条完全重复和 13 条较小 B2 覆盖记录，得到 1026 条不同曲线、7 个连续批次。M3877/M12011/M12037 的实际余因子分别为 3275/11694/11616 位，已知因子均精确重建目标，合数见证均为基数 3。该冻结输出使用 0.20.0 迁移标签，身份从快照 TOML 读取；原文件不含逐条程序版本。冻结文件保留原版本与原文，已有合法数字版本的回执也不会被转换器改名。所有数据、身份、实际 CRC 和原始日志只保存在忽略目录。

用户于 2026-10-17 提供的服务器回执确认：其中 M3877 的 NF 批次、B1=260,000,000、B2=2,600,000,000,000 已接受 1 行，状态码 0，CPU credit 为 753.2415 GHz-days。该结果使用真实 ECM-ELF 身份、正的参考 fft-length 和 CRC32，省略 security-code。此回执证明这一批提交可用，不扩展为 Edwards 或因子结果的服务器验收结论。已接受批次保持冻结，不能因版本或排版更新再次申报。回执摘要和冻结文件身份保存在 `data/experiments/ecm_submission_v031_20261017/server_acceptance.json`。

## 验证与边界

[结果生命周期回归](../../tools/test/test_result_output_consistency.py) 覆盖内部导出顺序、两阶段跨任务追加、F/NF 混排、原文件对象保留、部分追加恢复、发布历史完整性、NF 任务隔离、完成前不导出、即时单曲线 F、最终 CRC、旧位置迁移、重复回执和不完整 JSON 回执拒绝。[双格式测试](../../tools/test/test_ecm_submission.py) 对照本地正式样本结构，验证 CPU/GPU1 F/NF、TOML、合并、去重、恢复、FFT 和参数顺序。[Stage1 流程验收](../../tools/test/test_stage1_usability.py) 覆盖 CPU/CUDA、检查点恢复及跨 Stage2 共用账本。测试数据与实际身份只保存在忽略目录；这些检查均为离线验证，不代表新增服务器验收。

0.38.0 的本地验收证据在 `data/experiments/result_output_audit_20261018/validation_manifest.json`：27 项 Stage1 流程检查、22 项双格式检查、plain/NTT/gwpoly 各 4 条 NF，以及 GPU1 停止一条后恢复剩余三条的整任务发布检查。完整位宽 local 程序另在 GPU1 上验证 1279、8191、11213、16381 位各两条 Edwards 曲线，与独立 Python 末点/因子校验及 CRC 一致。GUI 结果存储也拒绝新旧位置的 CLI 账本。以上为功能验证，不作性能结论。

生产只读快照中，Stage1 JSONL 与 save 的 960 个 sigma 完全相同；原账本写入时间顺序与 save 一致。207 条旧 CUDA 内部行经审阅工具压缩为 492558 字节，原为 759226 字节，约减少 35.1%；37 条结构化调优观察保持一致。960 条 Stage1、207 条 Stage2 和 12 行补齐 checksum 的提交副本全部通过独立 CRC 校验。副本保留历史版本与时间，原始文件不改写，既有已接受批次不得重复提交。

PrimeNet v5 API 0.97 描述的是 `ar` 请求字段与结果类型，不是现代手工提交 JSON 的完整规范；结构还需对照 Prime95 生成器及正式样本。[固定源码与 API 研究](../reference/ECM_RESULT_JSON.md) 列出依据。上述 Montgomery NF 批次已有用户提供的服务器接受回执；Edwards、因子结果及其他提交入口仍需各自验证。AutoPrimeNet 的固定程序名扫描规则未修改。

实现入口：[提交序列化](../../src/core/ecm_result_submission.h) 的 `submission_json`、[账本与原子导出](../../src/core/ecm_submission_writer.h) 的 `publish_submission`/`seal_submissions`，CPU 和 CUDA 主程序在验证完成结果后调用它们。
