# ECM 结果输出

CPU/CUDA Stage1、CPU Edwards Stage2 与 CUDA Stage2 同时维护逐曲线内部 JSONL 和提交格式 `results.json.txt`。Stage1 整批完成后在内存中聚合，统一发布两份最终文件；中途因子只验证并保留在恢复状态中。Stage2 每完成一条曲线就追加一条内部 JSONL，因子立即额外导出一行 F，固定 curves=1 并保留 sigma。NF 仅在整条任务完成时导出，有 NF 的任务生成一行，曲线数是完成且未找到因子的曲线总数。直接 CLI 运行所选曲线范围视为一个任务。Stage1 NF 的 B2=B1，Stage2 NF 记录最后一条 NF 曲线实际请求的 B2。提交结构参考 Prime95，程序身份如实记录为 ECM-ELF，并附 CRC32；生成文件不触发网络请求。

## 共用发布接口与实现

Stage1、OpenCL worker 与 CPU/CUDA Stage2 共用 [publication.h](../../include/ecm/storage/results/publication.h) 和 [publication.cpp](../../src/ecm/storage/results/publication.cpp)。公开头包含拥有字符串/集合的 Submission 数据和实际序列化、发布、seal、恢复导出入口；不包含 JSON 解析实现、文件锁、操作系统句柄、文件读写或 CRC 函数体。

internal_json/submission_json 接收记录、借用 N 及实际运行元数据，返回协议 JSON 文本。Stage1 的 publish_stage1_batch 接收整批逐曲线结果，负责数值 sigma 排序、去重、双文件发布及恢复。Stage2 的 publish_submission 验证后建立逐曲线 receipt，因子立即发布，NF 按 seal 规则完成任务发布。publish_internal_batch、refresh_internal 保留旧内部账本的读取和恢复能力；refresh_submission_if_owned 恢复提交账本。文件/协议失败抛出异常，任务与队列决定失败策略。

contains_stage1_curves 接收结果路径、N、B1、曲线族/param 和 sigma 集合，优先流式核对最终 Stage1 JSONL；文件缺失时兼容旧内部账本。缺项返回 false，损坏报错。锁、旧目录兼容、发布历史恢复、CRC 与聚合规则集中在 CPP。

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

AID 可省略；CPU/CUDA Stage2 直接读档均可用 `--aid`，Stage1 与 CUDA Stage2 队列沿用任务的 AID。没有 AID 时不生成虚构值。程序名和数字版本由 [ecm_program_version.h](../../include/ecm/platform/program_version.h) 统一维护。Windows x64 平台号为数字 4；其他平台不冒用该编号。program.version 使用纯数字点分版本，Git 哈希放在构建身份中。Stage1 的 `source-build` 从链接后生成的 `ecm_stage1_build_identity.txt` 读取入口/共用源码及构建档位指纹；复制程序时可一并保留该文件，缺失时明确使用 development。精确二进制以构建清单中的 SHA256 为准。CPU Stage2 内部结果的 `source-build` 保留 Git/入口源码指纹，GW 构建另含算术模块与 GW 库指纹；CUDA 内部结果的 source-build 直接记录当前可执行文件 SHA256，构建清单另存编译条件。

版本按 `x.y.z` 维护：新增功能递增 y，并将 z 归零；修复递增 z。版本不随单次构建或 Git 提交自动变化，发布功能或修复时统一更新该头文件；构建依赖会使 CPU/CUDA 输出同步采用新版本。创建 Release 不是生成结果的前提。

## 提交字段与参考 FFT

内部 JSONL 的公共顺序为 status、目标、worktype、b1/b2、d/poly-size、curves、曲线/sigma、fft-length、因子、程序/系统/时间与 checksum。CPU libecm 和 CUDA 的实际 fft-length 为 0，gwpoly 记录真实 GW 长度。执行诊断集中于 stage1-engine 或 stage2-engine；CUDA 保留 queue_receipt 和可恢复调优样本 tune_observation，调优计划及检查计数保留在 stage2-engine。关闭的 relocation/Auto B2 不生成整组零值。

梅森目标用 exponent 与 known-factors 精确表示，不重复存储 N_hex、参考表名称和模型。一般整数保留必要的完整 N。Stage1 每批首行额外携带 stage1-publication 恢复计划，描述配对提交文件及待发布字节；该字段不进入手工提交行。Python 分析入口通过 tools/ecm_report/result_format.py 在内存中兼容旧的 B1/B2、N_hex 等别名。内部文件不直接粘贴到手工提交页面。

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

Stage1、CPU/CUDA Stage2 使用共用 [内部序列化模块](../../src/ecm/storage/results/publication.cpp)；提交字段和数学一致性验证使用 [公共结果模型](../../src/ecm/storage/results/publication.cpp)。`stage1-engine.batch-seconds` 是整个批次的引擎及 text-save 发布墙钟时间，包含指数、曲线准备、运行中检查点、算术与因子验证，排除外层批次档案读取、输入预检与最终双 JSON 发布；它在逐曲线记录中重复，不能求和当作总耗时。

Stage1 内部 `fft-length=0` 表示没有 GWNUM 变换长度，提交版本使用正的 Prime95 参考容量；`stage1-engine.coverage=B1` 不表示执行过 Stage2。

第 7 槽：Edwards 先写 `E`，再写存在的十进制 sigma；Montgomery 只写存在的 sigma。逐曲线内部结果始终保留 sigma，故 Edwards NF/F 都为 `E` 加 sigma。转换器当前只处理 NF：Edwards 输出 `Edwards:{}`，此槽为 `E`；Montgomery 没有 Edwards 对象或 sigma，此槽为空。不从原始逐曲线回执额外补入合并结果已省略的 sigma，也不猜测 torsion 后缀。

设规范串为 S，以 UTF-8 编码为字节序列；不包含 BOM、JSON 引号、转义后的 JSON 表达、行尾换行或终止零字节。整数使用普通十进制，无千位分隔符或科学计数法；因子顺序、大小写、空槽和时间空格都参与校验。

CRC 使用标准反射 CRC-32/ISO-HDLC：宽度 32 位，初值 `0xFFFFFFFF`，反射多项式 `0xEDB88320`（正向表示为 `0x04C11DB7`），最终异或 `0xFFFFFFFF`。对每个字节 b，先令 r ← r XOR b，再执行 8 次：若最低位为 1，则 r ← (r >> 1) XOR `0xEDB88320`；否则 r ← r >> 1。全部字节处理后，C = r XOR `0xFFFFFFFF`。将 C 写成大写、左补零至 8 位的十六进制字符串，放入 checksum 对象的 checksum 字段，版本字段为整数 1。Python 的 `zlib.crc32` 对 S 的 UTF-8 字节执行同一计算。

内部结果使用本行的实际 fft-length（libecm 为 0，gwpoly 为真实 GW 长度）和逐曲线 sigma；转换后提交文件使用正的参考 FFT、NF 模型标记和合并后的真实完成时间，必须从最终记录重新计算 CRC，不能拷贝内部旧值。版本、参与计算的字段或其表示改变时也须重算。单纯调整 JSON 键顺序、空格，以及移动 d/poly-size 不改变规范串。

curves、d、poly-size、stage2-fft-length、status、stage、user、computer、aid、program.port 和 source-build 不在这 16 个槽位中；CRC 不保护这些字段。它也不是 Prime95 的 security-code/SEC5。第三方源码依据见 [PrMers CRC32 研究](../reference/ECM_RESULT_JSON.md#prmers-crc32)，本仓库入口为 [转换器 checksum](../../tools/ecm_report/rewrite_stage2_results.py) 与 [CPU result](../../src/ecm/application/cpu_stage2/ecm_cpu_stage2_main.cpp)；[独立按位校验](../../tools/test/test_stage2_result_rewrite.py) 用于复核库计算结果。真实运行产生的 checksum 值保留在忽略目录。

## 去重、批次与恢复

### Stage1 整批发布

Stage1 不创建逐曲线 `.curve`、`.batch` 或独立 `publication.order`。逐曲线结果在内存中排序、验证和聚合，整批结束时发布 `stage1_results.jsonl` 与 `results.json.txt`；save 和 checkpoint 继续保留原有职责。内部 JSONL 的本批行按无符号 64 位 sigma 升序排列；提交因子行按同一顺序排列，NF 聚合成一行放在本批因子行之后。NF 没有 sigma，不能按逐曲线方式排序。历史批次保持原字节及顺序，不作全文件重排。

原生 Stage1 text-save 的新批次也按数值 sigma 排序；Stage2 依次读取这些点，因此直接承接该顺序。旧 save 历史不重排。Stage2 内部 JSONL 始终完成一条追加一条，旧 save、手工选择范围或重算记录的顺序仍由实际输入和执行决定。

两份最终文件不能作为单个文件系统事务替换。内部本批首行的 `stage1-publication` 保存版本、完整性散列、批次身份与 sigma 集合、相对提交路径、旧提交前缀长度/散列及本批提交文本。持有两个输出锁后，发布器流式验证内部历史与 CRC，先原子替换内部结果，再原子替换提交文件。第二步失败时 checkpoint 不清理；重试验证计划并补齐缺失或部分提交，不重复计数。已提交内部批次缺行、计划损坏或提交文件偏离已确认前缀均报错，不覆盖损坏证据。

短期独占锁及同目录 `.new` 原子替换文件仍会存在；它们不是逐曲线回执。任务成功后才移除算术 checkpoint。内部最终文件同时承担发布恢复依据，需与提交文件一同归档；保持二者相对位置可整体移动目录。更改配对文件名、只保留一份历史或跨磁盘改变配对位置前应另选全新输出。CRC32 只保护规范字段，发布计划的独立散列校验计划自身，不是加密签名。

旧内部与提交账本仍可恢复旧记录；正常 Stage1 发布不遍历旧逐曲线目录。最终内部文件缺失时，预检可从其旧归属账本恢复；旧账本不包含新整批输出，不得用旧账本单独重建新历史。内部最终文件存在时以它为准，不能使用旧版程序的内部导出命令覆盖它。已有数学曲线按 N、曲线族、param、sigma、B1 去重；已在旧提交中即时导出的 Stage1 因子也不会重复发布。

### Stage2 回执与追加

Stage2 保留逐曲线 durable receipt，目录在原提交账本下增加 save 文件名去扩展名的子目录。例如 `e_m10267_1e4.save` 使用：

- `data/results.json.txt.ecm-ledger/e_m10267_1e4/`：单曲线 `.curve` 和封存 NF `.batch`。
- `data/results.json.txt.ecm-ledger/format`、`publication.order`：账本归属和共同提交历史。

worker 后缀、显式提交路径和最终 `stage2_results.jsonl` 路径不变。旧根目录回执和新子目录一起读取，已有首次数学回执保留归属；同一曲线出现在另一个 save 中不重复计数。Stage1 因子身份不包含续算 B2，Stage2 重读同一因子不重复导出。内部 Stage2 JSONL 可保留重算诊断。

整个提交账本丢失时，已有 Stage2 内部结果、Stage2 提交行或恢复进度会使预检报错；CPU/CUDA 在下一条算术前、发布前及任务封存前检查。保留最终文件和队列，不自动按内部 JSONL 猜测已提交批次。已存在合法 Stage1 提交、但尚无任何 Stage2 工作时允许首次建立 Stage2 账本。若所有结果、账本与恢复进度一起丢失，程序无法辨认是否曾完成工作；局部删除子目录也不等同于整个账本缺失检查，不能据此保证自动发现所有漏项。恢复数据后再继续，不能把账本当作可随意清空的缓存。

NF 回执绑定任务范围；CUDA 停止时不封存未完成任务，同一任务恢复后完成剩余曲线才生成提交行。Stage2 同一任务即使各曲线 B2、D、P 不同，也合并一行，采用最后完成的 NF 曲线实际参数与 UTC 时间，并重算 checksum。汇总 B2 不表示每条曲线覆盖相同界限，逐曲线覆盖情况以内部 JSONL 为准。F 立即单独导出，固定一条曲线。完全相同的新 NF 报告等待真实 UTC 秒推进，最多两秒后报错，不虚构时间。

Stage2 重启时可以调整请求曲线数和有效 B2，保持原发布归属，已完成记录不改写。新目标低于已确认完成数时，警告并封存全部实际完成工作；例如150/960改为100，150条均为NF则提交 curves=150。pending 先对账，B2和曲线数覆盖及异常边界见[Stage2自动续跑](STAGE2.md#自动续跑与退出)。

提交文件以二进制追加写入，保持已有文件对象、字节和行顺序。锁内先原子保存带散列的 `publication.order`，再追加未写入后缀。部分追加或文件缺失可恢复；内容与历史冲突或账本损坏时报错。Stage1、Stage2 共用提交文件时，Stage2 可接纳具有程序身份与有效 CRC 的 Stage1 前缀/新增后缀，Stage1 可读取一个已有发布历史文件完成部分提交恢复，均保留已发布字节。

账本属于恢复依据，不能作为缓存删除。旧的输出旁账本首次访问时整体移到结果目录的 `data/`；两处同时存在则拒绝自动合并。旧内部 V1/V2 回执仍沿其导出顺序读取；新 Stage1 行不写入这些账本。Windows 原子替换遇到共享或锁定错误时最多重试 500 ms。Stage2 显式恢复命令按其任务范围封存待发布回执，不执行算术：

```powershell
ecm_cpu_stage2.exe --export-results --submission results.json.txt
```

该命令适用于 Stage2 账本，不替代 Stage1 双文件重试流程；未完成工作应续算原任务。归档保留最终文件、相关账本及尚未完成的 checkpoint。结果发布存储不保存曲线内部算术状态。

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

实现入口：[提交序列化](../../src/ecm/storage/results/publication.cpp) 的 `submission_json`、[账本与原子导出](../../src/ecm/storage/results/publication.cpp) 的 `publish_submission`/`seal_submissions`，CPU 和 CUDA 主程序在验证完成结果后调用它们。
