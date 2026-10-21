# Save：Stage1 完成点、文本格式与交接

本文是当前 save 的格式与发布说明。曲线内状态及 Stage2 队列进度另见 [CHECKPOINT](CHECKPOINT.md)。操作入口见 [Stage1](../usage/STAGE1.md)、[CUDA Stage2](../usage/STAGE2.md)、[CPU Edwards Stage2](../usage/EDWARDS_CPU_STAGE2.md)；配置默认值和范围见 [TOML 参考](../ECM_TOML_REFERENCE.md)。

## 1. Save 保存的是什么

一条记录表示指定 N、曲线族、Montgomery 参数化（仅 Mont）、sigma、B1 下，完成 Stage1 后的点或已经找到的因子。文件通常含多条曲线；没有逐文件曲线数头、二进制索引或全文件 checksum。

内部 GPU/CPU 算术表示在输出前转换为普通整数。没有命中且分母可逆时，保存的是 Montgomery x-only 归一化坐标 X，隐含 Z=1；Edwards 的点也先转换到对应 Montgomery 曲线。消费者由 N、sigma 和曲线方法重建相应常数。

以下对象不属于 save：完整光滑指数、GPU 字典、ladder 中间点、NTT/多项式、剩余任务进度、整批结果账本。文件名或文件存在不证明完成；可写性预检可能留下空文件。

| 数学身份 | 输出/执行信息 |
| --- | --- |
| 实际 N、曲线方法、Montgomery param、sigma、B1、保存点或因子；Edwards 另显式记录固定标量约定 | worker、设备号、GPU 型号、线程/TPI/容器、WHO/TIME、文件名和目录不定义曲线 |

同曲线构造使用不同内核不应改变因子及最小界限的数学语义。完整性校验只核对保存字段，不重新计算 Stage1，不证明 X 是所请求标量乘法的正确结果。

## 2. 命名、路径与队列覆盖

### 2.1 自动 save 名称

CPU/GPU 自动命名由 `save_name_pattern` 统一产生，当前模式结构为 `{curve}_{target}_{b1}.save`，默认值以配置定义为准。

| token | 展开规则 |
| --- | --- |
| `{curve}` | `e`、`m0`、`m2`、`m3`；CPU Montgomery 使用 m0 |
| `{target}` | 梅森任务为 `m<原始指数>`，保留剥离因子前的指数；其他目标为 `bit<实际N位数>` |
| `{n}` | 有原任务元数据时为原始 n；否则识别完整梅森整数的指数，或使用实际 N 位数 |
| `{b1}` | 核心 Stage1 自动命名取四舍五入后的整数 B1，至少3个末尾零时压缩，例如 260000000→26e7；GUI worktodo 生成器保留任务中 B1 的原始写法 |

典型名称为 `e_m5351_260e6.save`、`m0_m5351_260e6.save`、`m2_bit4916_1e6.save`。自定义模式没有匹配的构造前缀时会规范/补充前缀；未知花括号 token 报错。自动路径通常落在 `tmp_dir`；CLI 显式 `-save`/`-savea` 路径按其接口保持，不自动从名称改曲线。

[GUI worktodo 生成器](../usage/GUI.md#队列生成与交接)按所选曲线类型展开同一默认模式；它不链接 GMP，非梅森目标的 `bit` 位数使用估计值，涉及已知因子或位数边界时须核对预览。核心 Stage1 自动命名仍使用实际 N 位数。

来源：[命名器](../../src/ecm/application/cli/ecm_driver.cpp#L1991)、[前缀规范](../../src/ecm/application/cli/ecm_driver.cpp#L777)、[配置源](../../config/ecm_options.json)。

### 2.2 Stage1 以 ECMSTAGE2 行作为输入

这是 Stage1 队列入口的约定，不能与 Stage2 直接读取已有记录的规则混为一谈。

| save 文件名部分 | Stage1 队列行为 |
| --- | --- |
| `e_`，兼容 `ed_` | 覆盖 TOML 曲线为 Edwards；使用固定 48·LCM，兼容名规范为 e_ |
| `m0_`、`m2_`、`m3_` | 覆盖 TOML 曲线为 Montgomery 及对应 gpu_param |
| 无上述前缀 | 沿用 TOML 构造并补充前缀 |
| 最后一个下划线至 `.save` 前 | 提取 B1；支持可解析的正十进制/科学计数形式 |

执行目标、设备仍由 TOML 决定。CPU 不支持 m2/m3，会停止并保留队列；覆盖后仍须通过参数化能力和 sigma 范围检查。每条任务从基础配置复制，前一条覆盖不传播到下一条。`save_name_pattern` 不替代已经在任务中指定的词干和父目录。

控制台 `START:` 保留原始任务行；finished 与受支持的 Prime95 交接使用实际 save 字段，其余 AID、B2、skip、curves、因子字符串不因改名而改写。输入队列行只在任务成功完成后移除。

若旧无前缀/兼容名发生规范化，新目标不存在或为空时，在持有批次锁后检查旧记录的 METHOD/PARAM，复制其已有历史再追加新曲线；原文件保留。这样原 skip 编号仍指向同一历史段。目标已有内容则不重复复制。此复制检查**不是完整 N/B1/checksum 验证器**，不能用来清洗或修复来源错误的 save。

Stage2 消费时，以每条记录内部的 METHOD/PARAM/SIGMA/B1/N 为算术依据；只改文件名前缀不能将一条 Mont 记录转换为 Edwards，也不能改变已完成 B1。

来源：[前缀覆盖](../../src/ecm/application/cli/ecm_driver.cpp#L761)、[队列入口](../../src/ecm/application/cli/ecm_driver.cpp#L3725)、[旧历史复制](../../src/ecm/application/cli/ecm_driver.cpp#L785)、[只改 save token](../../src/ecm/storage/queue/ecm_worktodo.cpp#L134)、[提取 B1](../../src/ecm/storage/queue/ecm_worktodo.cpp#L370)。

## 3. 文本封装

正常完成的 Stage1 使用分号分隔的 key=value 文本，每条曲线一行，行末换行。大整数 X 写为 `0x` 加小写十六进制；SIGMA/B1/checksum 写十进制。N 的编码依曲线族而异。没有要求不同曲线的 X、sigma 或十进制宽度相同。

读取方可以跳过空行和以 `#`/`;` 起始的注释。CUDA Stage2 允许记录开头的 UTF-8 BOM，键名规范为大写；重复键、缺失必要键、非法整数或非归一化 Z 会报错。CPU Edwards 的专用读取器使用严格的方法和值约定，不能假设所有文本宽容规则完全相同。

## 4. Montgomery save

### 4.1 字段表

| 字段 | 写入值 | 是否进入数学 checksum |
| --- | --- | --- |
| METHOD | `ECM` | 方法名本身不进入乘积式校验 |
| PARAM | PARAM0 不写；PARAM2/3 显式写 2/3 | 以 param+1 进入 |
| SIGMA | PARAM0 为每曲线 u64；PARAM2/3 使用受32位窗口约束的起点+i | 是 |
| B1 | 完整十进制整数 | 是 |
| N | 原任务表达式/整数；PARAM3 另有下节规则 | 是，按写入器传入的数值 N |
| X | 未命中时为归一化普通域 x；命中时为因子 | 是 |
| CHECKSUM | [0,4294967291) 的十进制整数 | 此字段自身不进入 |
| PROGRAM | 当前程序名称 | 否 |
| X0、Y0 | 当前写入 `0x0`，不是保存的原始曲线点 | 否 |
| WHO | 操作系统用户名和主机名；可省略 | 否 |
| TIME | 本机本地时间文本 | 否 |

没有 STATUS/HIT 字段，也没有显式 Z。PARAM0 的“省略 PARAM”按其格式表示 param=0；与“不知道参数化”不同。SIGMA 必须属于该条 X 的曲线，不能把随机 sigma 数组误写为一个连续序列。

Montgomery save **不记录 lcm=1/12**，不能从文件内容反推出原 Stage1 是否使用了12倍标量。需要保持生产配置或外部任务依据；checksum 无法辨别这一信息缺失。

### 4.2 校验公式

设 q=4294967291，则当前写入器计算：

**CHECKSUM = B1 · sigma · N · X · (param+1) mod q**。

每次乘法可先取模；PARAM0 的末项为1。该校验不是文件 CRC，不覆盖空格、字段次序、PROGRAM/WHO/TIME，也不保证交换因子的抵消等修改必被发现。它为与既有文本生态兼容保留，不能作为密码学认证。

该公式以整数 B1 为正常输入。当前 writer 的 double→大整数校验输入使用截断，输出 B1 文本使用四舍五入；不应以非整数 B1 构造可交换 save，两种转换的一致性并无通用保证。

### 4.3 N 表达式与因子行

CPU/GPU PARAM0 和 GPU PARAM2 保存计算所使用的目标表达式，保持它与校验输入 N 一致。表达式可包含已知因子除法；“保留目标”指保留该次实际运算目标，不是重新乘回已知因子。

PARAM3 当前路径会收集本批找到的不同有效因子，依次在表达式后追加精确除法；实际能整除剩余数的因子才被剥离。其 writer 的 checksum 仍接收原算术 N。这是与 PARAM0/2 不同的当前行为；**含新因子的 PARAM3 输出，其表达式求值和校验输入一致性须专门验证**，不能宣称所有这类文件都能被外部严格校验的读取器直接续算。无新因子时没有该改写差异。

命中时的 X 是因子，没有点的继续意义。文本没有独立命中标志；消费者需据算术或其他结果判定。Stage1 结果账本是曲线结果的独立依据，见 [ECM 结果输出](../usage/ECM_RESULTS.md)。

来源：[PARAM2/3 写入](../../src/ecm/storage/save/ecm_save.cpp#L153)、[PARAM0 写入](../../src/ecm/storage/save/ecm_save.cpp#L218)、[表达式剥离](../../src/ecm/storage/save/ecm_save.cpp#L106)、[驱动选择](../../src/ecm/application/cli/ecm_driver.cpp#L2904)。

## 5. Edwards save VERSION=1

CPU 与 CUDA Edwards 共用相同文本协议，不使用 Montgomery PARAM。当前归一化转换为 **Qx:Qz = (Z+Y):(Z−Y)**；分母为单位时写 X=Qx/Qz mod N。即使存在同一个数字 sigma，也不能把 Edwards 记录当作 Suyama PARAM0。

### 5.1 字段表

| 字段 | 必需值/含义 |
| --- | --- |
| METHOD | `ECM-EDWARDS-AM` |
| VERSION | 1 |
| SCALAR | `48LCM`，固定 48·lcm(1…B1) |
| SIGMA | 非零 u64 |
| B1 | 完整十进制 Stage1 界限 |
| N | `0x` 加完整实际模数；不依赖原表达式或文件名 |
| X | `0x` 加归一化普通 x，或 STATUS=FACTOR 对应的因子 |
| STATUS | `POINT` 或 `FACTOR` |
| CHECKSUM | 标准 FNV-1a 64 的十进制表示 |

PARAM 被禁止；默认不写 PROGRAM/WHO/TIME，也不写 Z。可选 Z 若提供必须为1。找到构造因子、或末点转换发现 1<gcd(Qz,N)<N 时写 FACTOR；POINT 表示有可使用的归一化点。若合法末点的 gcd(Qz,N)=N，则没有可用 Stage2 点，该曲线完成但不写点行。因此“尝试 C 条曲线”不保证 save 恰有 C 条记录，后续消费者必须根据实际可用记录计数。

### 5.2 校验规范

先把 N、X 规范成无 `0x`、无前导零的小写十六进制；零为字符0。校验输入按以下字段顺序以单个竖线分隔：

**ECM-EDWARDS-AM | 1 | 48LCM | hex(N) | hex(X) | decimal(sigma) | decimal(B1) | STATUS**。

实际输入没有表意空格、引号或末尾换行。FNV-1a 64 初值 **14695981039346656037**，乘数 1099511628211，每字节异或后相乘并模2⁶⁴回绕。它覆盖曲线方法、版本、标量约定及状态，区别于 Montgomery 的乘积式校验。

CPU Edwards 专用读取器要求 N 为大于3的奇数，X∈[0,N)，B1≥2 且不超过2⁵³，并验证 FACTOR 严格大于1且整除 N。CUDA 读取器另受自身位宽上限约束；不能将两个入口的检查集合视为完全相同。

来源：[save_line / save_checksum](../../src/ecm/storage/save/ecm_edwards_io.h#L19)、[坐标与状态判定](../../src/ecm/backend/cuda/stage1/edwards/host_workflow.cpp)、[CPU 专用读取](../../src/ecm/storage/save/ecm_edwards_save_reader.h#L22)。

## 6. 写入、去重与发布

### 6.1 一次发布的步骤

1. 生成本批各条文本记录；CPU/GPU 输出都使用普通域值。
2. 可写性预检检查目录和目标策略。`-save` 避免覆盖已有文件；`-savea`/队列允许追加。具体分支也可能在算术结束时才执行发布检查。
3. 在目标同目录打开 `目标路径.new`；追加时流式读旧文件并写入临时文件，收集已有行的去重键。
4. 追加尚未出现的新记录，关闭临时文件并检查流状态。
5. 通过 Stage1 共用提交器轮换 `.bak.save` 等备份并原子替换目标。
6. 驱动发布内部结果和提交输出；全流程成功后才清理 Stage1 checkpoint，并由队列推进任务。

共用提交器、备份上限、共享冲突重试及掉电持久化边界见 [CHECKPOINT 原子替换](CHECKPOINT.md#52-原子替换和备份)。save 的备份是文件副本；CUDA Stage2 正常读取指定主路径，不自动因校验失败切换到 `.bak.save`。

### 6.2 去重规则

| 文本类型 | 当前去重键 |
| --- | --- |
| Montgomery 当前写入行 | 从行首到 `; PROGRAM=` 之前的字节；忽略后面的程序名、WHO/TIME 等变化 |
| Edwards 当前写入行 | 当前没有 PROGRAM 尾段，因此为完整行 |
| 任意旧文本 | 同一截断规则；只去掉读到的行末 CR，不规范字段顺序、空格或数值写法 |

这使相同 canonical 输出在发布重试时不会再追加一条，但不是按 N/curve/param/sigma/B1 重新解析建立的数学去重索引。字节排列不同的等价记录仍可能并存；旧文件中已有的重复行会被保留，不执行全文件清洗。

Stage1 批次锁防止同名 checkpoint 的并发写入，**没有独立的跨批次 save 文件锁**。两个不同批次/目录若指向同一 save，仍可能争用同一个 `.new` 并丢失追加；需要为并发 worker 使用独立写入目标。备份、文本 save、内部结果、提交 JSON、finished 与队列不构成单个跨文件事务；失败重试分别依靠保留的 checkpoint、文本去重和结果账本处理。

### 6.3 成本和体积

逻辑文件大小精确为各实际 UTF-8 行字节数加各换行之和。N 每行重复，不能按“一个 N+全部 X”估算。若 N 和 X 都写完整十六进制，数值部分每条约有2·ceil(bits(N)/4)个字符，再加 sigma、B1、checksum及标签；Montgomery 的长表达式/已知因子串需按其实际文本另计。

追加旧文件 S bytes、新行 T bytes，单次文本合并本身约读 S、写 S+T；启用备份还会复制旧主文件，增加约读 S、写 S。不能把这是“只追加 T bytes”当成实际磁盘工作量。主机保存新行集合及旧去重键，额外内存随这些文本增长；文件备份不属于GPU显存。

来源：[publish_save](../../src/ecm/storage/save/ecm_edwards_io.h#L30)、[可写性预检](../../src/ecm/storage/save/ecm_save.cpp#L72)、[批次发布和错误保留](../../src/ecm/application/cli/ecm_driver.cpp#L3048)。

## 7. 消费者和交接能力

### 7.1 当前能力矩阵

| 记录 | 本仓库 CUDA Stage2 | 本仓库 CPU Edwards Stage2 | Prime95 交接 | GMP-ECM 文本生态 |
| --- | --- | --- | --- | --- |
| Montgomery PARAM0 | 支持 | 该入口只读 Edwards | 支持相应 Suyama 记录 | 以 PARAM0 解释 |
| Montgomery PARAM2 | 当前读取器拒绝 | 不支持 | Stage1 不自动交接 | 格式面向 PARAM2 读取器 |
| Montgomery PARAM3 | 当前读取器拒绝 | 不支持 | Stage1 允许交接；含新因子改写见4.3 | 格式面向 PARAM3 读取器；同样须注意4.3 |
| Edwards AM VERSION=1 | 支持，重建对应曲线常数；可选本地队列交接 | 支持专用协议，交给 GW/libecm 后端 | Stage1 不向 Prime95 自动交接该文本 | 自定义 METHOD，不能当作普通 PARAM0 文本直接导入 |
| Stage1 私有二进制 checkpoint | 拒绝 | 专用文本入口不接受 | 不是交换 save | 不支持直接交换 |

“Stage1 能生成”不等于“本仓库 CUDA Stage2 能消费所有 param”。消费者按内容检查，不靠文件名前缀放宽能力。

### 7.2 CUDA Stage2 的读取边界

- 必要字段为 METHOD、SIGMA、B1、N、X；PARAM 存在时必须为0。
- N 按项目表达式解析器精确计算，要求奇数、N>3、至多16384 bits；X 要求 [0,N)；Z 必须缺省或等于1。
- Suyama SIGMA≥6、Edwards SIGMA≥1，B1≥2；普通 PARAM0 允许 `0:` 种子前缀。
- PARAM0 的 CHECKSUM 缺失时仍可读取，并标记为未提供校验；存在则必须通过乘积公式。Edwards 的版本、标量约定、状态及 CHECKSUM 都必须存在并通过。
- 读取 Edwards POINT 时重建曲线常数；构造发现因子时按因子处理，退化曲线报错。FACTOR 必须是 N 的非平凡因子。
- 队列重建 worktodo 的实际 N 并与所选每条记录比较；同一队列任务的所选 B1 必须相同，B2 必须满足运行条件。直接文件模式没有任务表达式时，以记录 N 为准。
- `skip_curves` 跳过非注释记录的前若干条；记录编号从1开始。请求数量不足时只执行可用记录并警告；无可用记录可零曲线完成，不把缺少记录当作有算术结果。

这些检查不回算 Stage1，不检测任意一个满足范围的伪造 POINT 是否正确。Stage2 队列恢复使用整个 save 的 SHA256，见 [CHECKPOINT](CHECKPOINT.md#7-stage2-队列进度格式)。

来源：[CUDA 记录解析](../../src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp#L117)、[选定记录](../../src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp#L193)、[队列数学目标检查](../../src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp#L1360)、[CPU 文本入口](../../src/ecm/application/cpu_stage2/ecm_cpu_stage2_main.cpp#L150)。

### 7.3 Prime95、同步与错误

资格按每条 Stage1 任务应用前缀后的实际构造判断：Montgomery PARAM0/3 可交接；Edwards 和 GPU PARAM2 忽略 Prime95 转移路径。save 文件同步与 worktodo.add 交付由驱动执行，GUI 只显示状态。

保存成功后会按配置同步扫描目录中的 `.save`，再交付任务行；增量同步包含与启动标记同一秒写出的文件。不能由此推断所有任意父目录的显式路径都会被扫描；绝对 save 路径和消费者可访问位置需按实际配置安排。同步和交付不是同一个原子事务。

交付失败不回滚完成的 Stage1，任务进入 pending 交付状态供后续重试；不直接改写 Prime95 活动 worktodo.txt。结果/AID/重复交付策略见[Stage1 使用](../usage/STAGE1.md#检查点与交接)和 [ECM 结果输出](../usage/ECM_RESULTS.md)。

Edwards 的本地 Stage2 队列交接由 `edwards_stage2_transfer` 独立控制，不使用 Prime95 `worktodo.add`。完成 save 发布后先记录待投递行，再将原任务移入 Stage1 finished；交付时将实际 save 的绝对路径写入接收 worker 的共用 `worktodo` 分区。队列追加和 Stage2 移除任务共用短时编辑锁；重试前检查活动队列及 Stage2 finished 中的完成行或输入错误回执，避免在队列发布后、待投递记录清除前崩溃导致重复交付。接收 worker 的选择和 B2 要求见 [Stage1 使用](../usage/STAGE1.md#检查点与交接)。

来源：[队列完成与交接](../../src/ecm/application/cli/ecm_driver.cpp#L3236)、[按实际构造启用交接](../../src/ecm/application/cli/ecm_driver.cpp#L3431)、[save 同步](../../src/ecm/storage/queue/ecm_worktodo.cpp#L645)、[交付器](../../src/ecm/storage/transfer/p95_transfer.h)。

## 8. 保留的二进制 MIDSTAGE 接口

`ecm_edwards_write_midstage/read_midstage` 和 `ecm_p95feeder` 保留 Prime95 风格 version 6 的独立二进制接口。**当前 CPU/CUDA Edwards Stage1 正常输出使用第5节文本，不自动生成这类 `.tmp`**。其主公共头与 [CPU Edwards checkpoint](CHECKPOINT.md#44-cpu-edwards-二进制记录和完成记录) 的0..99字节结构共用，但 stage 标签为 `C{curve}S2`、state=2、sigma_type=0、montg_stage1=1。

| 偏移/位置 | 编码 | 内容 |
| --- | --- | --- |
| 0..99 | 公共头与 Stage1 目标字段 | 52-byte 文件头及 curve/average_B2/state/sigma/B1/B2/sigma_type/montg_stage1 |
| 100 | giant | Qx，u32 limb 数+低 limb 在前的 u32 数组 |
| Qx 后 | i32 | Qz flag，当前 writer 写1 |
| flag 后 | giant | Qz，当前 writer 总会写入 |
| Qz 后 | i32 | gg flag，当前 writer 写0，不写 gg giant |

总长度为 **108 + G(Qx)+G(Qz)** bytes，其中 G(U)=4+4·limbs(U)，零也占至少一个 limb。校验位于偏移48，按 CPU 二进制 codec 的加法规则；公共头、average_B2、sigma不计入。它没有完整实际 N 的独立大整数域，k/b/n/c/外部任务必须正确描述目标。

该 writer 直接截断写目标，并非文本 save 的 `.new`+轮换发布器。`read_midstage` 仅实现其 MIDSTAGE 子集；公共头版本数通过不等于任意 Prime95 版本/状态都已支持。不能把这层保留接口当作所有外部二进制 save 的通用恢复器。

来源：[MIDSTAGE 写入/读取](../../src/ecm/storage/save/ecm_edwards_save.cpp#L201)、[feeder 消费入口](../../src/ecm/application/p95_feeder/ecm_p95feeder.cpp#L429)。

## 9. 验证入口与 TODO

- [可用性与保存失败重试](../../tools/test/test_stage1_usability.py)：CPU/GPU、固定种子、完整末点、输出所有权和停止恢复。
- [命名、旧记录复制和混合队列](../../tools/test/test_stage1_checkpoint_compat.py)：原指数保留、显式路径、字段改写、前缀覆盖、构造冲突及交接。
- [Edwards 独立点与 Stage2 校验](../../tools/test/test_edwards_gpu_pipeline.py)、[专用文本读取器](../../src/ecm/storage/save/ecm_edwards_save_reader.h)。
- [Prime95 交付回归](../../tools/test/test_p95_transfer.ps1)、[GUI 新文件名解析](../../tools/test/gui/results_test.cpp)。

待验证/改进：

- PARAM3 本批命中新因子后，N表达式、checksum与外部 Prime95/GMP-ECM 读取的一致性；应同时覆盖多曲线命中和多个可整除因子，不以无命中样本替代。
- 若将来需要多个不同批次共写一个 save，须定义独立锁、冲突及合并策略；当前调用者应隔离写入目标。
- 若将来扩展本仓库 CUDA Stage2 到 PARAM2/3，需实现并验证对应曲线重建和因子语义，不能仅取消 PARAM 检查。
