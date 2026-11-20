# Checkpoint：状态、格式与恢复

本文是当前 checkpoint 的格式与持久化说明。完成 Stage1 后的交换文件另见 [SAVE](SAVE.md)；启动和停止操作见 [Stage1 使用](../usage/STAGE1.md)、[Stage2 使用](../usage/STAGE2.md)。配置键的默认值、范围只在[生成的 TOML 参考](../ECM_TOML_REFERENCE.zh-CN.md)维护。

## 1. 文件职责与范围

| 对象 | 保存内容 | 恢复粒度 | 是否供另一程序执行 Stage2 |
| --- | --- | --- | --- |
| Stage1 任务 checkpoint | 曲线身份、sigma、各后端未完成状态或已完成结果 | CPU 曲线分块、GPU 切片、NTT 已完成子批和当前子批切片 | 否 |
| Stage1 `.save` | 已完成 Stage1 的归一化点或已发现因子 | 一条完成的曲线 | 依曲线族和消费者能力，见 SAVE |
| Stage2 任务 checkpoint | 已完成曲线、待发布结果及确认状态 | 完整曲线 | 否，不含曲线内运算状态 |
| 指数缓存、PRAC 计划、tune profile | 可复用计算或性能资料 | 不代表任务进度 | 否 |

Stage1 外层统一不代表内部算术状态通用。CUDA ladder、CUDA resident/PRAC、OpenCL、CPU Montgomery、CPU Edwards、GPU Edwards 使用不同载荷；改名不能将它们互相转换。CRC、FNV 与结果回执均不能证明标量乘法在数学上正确。

## 2. Stage1 文件名与身份

### 2.1 物理文件

一个任务使用一个主文件。CPU 曲线记录和 NTT 子批进度都封装在其中。

| 曲线 | 任务有原始指数 n | 普通整数、无指数元数据 |
| --- | --- | --- |
| Edwards | `.e_s1_m{n}_{B1}.ckpt` | `.e_s1_bit{bits}_{B1}.ckpt` |
| Montgomery | `.m{param}_s1_m{n}_{B1}.ckpt` | `.m{param}_s1_bit{bits}_{B1}.ckpt` |

param 为 0、2、3，CPU Montgomery 使用 0；bits 为实际模数位数，B1 为完整十进制整数。k·bⁿ+c 任务使用其原始 n 作为命名标签，即使剥离了已知因子；没有这类元数据时，还会识别实际 N 是否为 2ⁿ−1。因此 `m` 命名标签不能代替内容中的完整 N 检查，也不能单独证明原任务是梅森数。

文件位于 `stage1_save_dir`。未配置有效目录的路径不能获得同等的批次持久化保证。文件名不包含 worker、sigma、曲线数量或后端；同名不同任务需使用不同目录。

相关文件如下：

| 名称 | 作用 |
| --- | --- |
| 主文件 | 当前已发布批次 |
| `主文件名.new` | 待发布的完整新文件，不作为自动恢复候选 |
| `.bak.ckpt`、`.bak2.ckpt`… | 后缀前插入备份代数；最多 16 代，实际数量由 `checkpoint_backups` 决定 |
| `主文件名.lock` | 批次独占锁；Windows 使用无共享打开、关闭时删除；POSIX 使用非阻塞 flock |

来源：[驱动命名](../../src/ecm/stage1/task.cpp#L229)、[备份及锁](../../src/io/stage1_checkpoint.cpp)。

### 2.2 内容身份

外层 identity 为 UTF-8/ASCII 文本，以竖线分隔，按完整字符串比较。不是结构体，也不是文件名摘要。

| 顺序 | 内容 | 当前编码 |
| --- | --- | --- |
| 1 | 实际 N | 小写十六进制，无 `0x` |
| 2 | 曲线构造 | `ed`、`mont0`、`mont2`、`mont3` |
| 3 | B1 | 十进制整数 |
| 4 | 标量约定枚举 | 0=LCM，1=12·LCM；Edwards 以自身固定 48·LCM 解释 |
| 5 | 执行目标 | `cpu` 或 `gpu` |
| CPU 附加 1 | 标量编码身份 | Edwards 写入 `edwards-cpu-v2`，Montgomery 写入现有窗口设置 |
| CPU 附加 2 | 实现枚举 | 0=auto、1=IFMA SIMD、2=GMP scalar |
| CUDA NTT 附加 | 曲线后端 | Montgomery 为 `ntt-v1`，Edwards 为 `ntt-edwards-v2` |
| CUDA 拆批 NTT 追加 | 任务载荷版本和总曲线数 | 在 NTT 身份末尾追加 `ntt-task-v1` 和十进制总曲线数 |
| CUDA NTT 梅森承载追加 | 算术模数 | 在公式身份之后追加 `ntt-carrier-v1` 和原始梅森指数 |
| 其他 GPU Montgomery 附加 | CGBN 算法文本 | 由驱动写入有效的 CGBN 算法 |

一般路径用外层头比较曲线数，并在记录头中保存各 sigma。固定 sigma 任务要求每条非零已存 sigma 等于请求起点+i，随机任务采用已存种子。拆批 NTT 的外层固定为一条记录，使用 identity 核对任务总曲线数，从任务载荷恢复起始 sigma。Edwards NTT 的旧身份 `ntt-v1` 和 `ntt-edwards-naf-v1|窗口` 仍可匹配同一任务，写入时使用当前身份。CPU Edwards 也接受包括历史窗口值的旧身份，实际 NAF 窗口从曲线记录恢复。GPU Edwards 从对应载荷恢复适用的窗口、binary/NAF 模式及倍点变体。见 [Stage1 身份构造](../../src/ecm/stage1/task.cpp#L268)。

设备编号、GPU 型号、驱动版本、D、B2、输出路径不在该外层身份中。不能据此推断任意设备或后端之间均可恢复：内部布局、构建包含的内核和算法约束仍须满足。CPU auto 切换实际实现的情况还受各后端的分块恢复条件约束。

NTT 梅森承载任务的首项 N 仍为余因子，算术快照按承载指数编码，完成记录按余因子验证。修改已知因子或承载指数后需重新核对任务身份。见 [梅森承载](NTT.md#stage1-已知因子与梅森承载)。

### Context 与停止信号

`src/io/stage1_checkpoint.cpp/h` 的 Context 拥有本任务的 records、sigma、恢复候选、备份策略和独占锁。task 通过普通函数参数借给 CPU/CUDA/OpenCL 后端，后端回调不读取全局活跃批次。Context 不可复制；活跃期间不能重新 begin 或改路径，begin 部分失败的锁也由析构释放。关闭其他 Context 不改变本批记录，但生产驱动仍只执行单活跃任务。

离线载荷验证在容器互斥量内运行，回调不能重新进入该 Context 的 I/O。进程级停止标志和 GUI event 属于 `platform/stage1_stop`，信号处理只标记停止，不做文件 I/O。数学状态的资格和编码仍由各后端决定；CPU Edwards 使用 `io/prime95_save` 的独立二进制格式适配。

## 3. 当前外层二进制格式 V2

### 3.1 编码约定

- 所有显式 u32/u64 均为无符号小端整数；偏移从文件首字节 0 开始。
- 字符串和载荷采用“字节长度+原始字节”，不附加 NUL 或换行。
- I 为 identity 字节数，C 为外层记录数，第 i 条载荷长度为 Lᵢ。拆批 NTT 的 C 固定为 1，其余路径的 C 为曲线数。
- GPU 批次载荷置于记录 0，其余记录通常 Lᵢ=0，但仍保存各自 sigma。CPU 通常每条记录保存相应曲线状态。拆批 NTT 的唯一外层记录 sigma 为 0。

### 3.2 文件头与尾

| 偏移 | 字节数 | 字段 | 值/含义 |
| --- | --- | --- | --- |
| 0 | 8 | magic | ASCII `ECMS1B2` 加一个 NUL；完整 8 bytes |
| 8 | 4 | version | 2 |
| 12 | 4 | curves | C |
| 16 | 8 | identity_bytes | I |
| 24 | I | identity | 上节的完整身份 |
| 24+I | 变长 | records | 连续 C 条记录，见下表 |
| 文件长度−4 | 4 | file_crc32c | 此字段之前全部字节的 CRC32C |

### 3.3 单条记录

偏移相对于该记录首字节。

| 偏移 | 字节数 | 字段 | 值/含义 |
| --- | --- | --- | --- |
| 0 | 8 | sigma | 本曲线完整 u64 种子；0 是尚未分配种子的哨兵 |
| 8 | 8 | payload_bytes | Lᵢ |
| 16 | 4 | payload_crc32c | 仅载荷字节的 CRC32C；空载荷为 0 |
| 20 | 4 | flags | 当前必须为 0 |
| 24 | Lᵢ | payload | 后端自己的格式；不进行统一数学解释 |

总长度精确为 **28 + I + 24C + ΣLᵢ bytes**。没有自然对齐填充或额外结束标记。

### 3.4 校验及边界

CRC32C 使用 Castagnoli 多项式，反射形式 `0x82f63b78`，初值和最终异或均为 `0xffffffff`。标准输入 `123456789` 的结果为 `0xe3069283`。外层完整 CRC 覆盖头、identity、各记录头及载荷；记录 CRC 只覆盖对应载荷。

读取先核对完整文件 CRC，再检查版本、长度、最小记录头空间、flags、逐记录 CRC 和无多余尾字节。变长字段读取不得越过剩余输入。外层通过后才比较任务身份、曲线数和固定 sigma。文件整体会先读入主机字符串，不能将“长度检查”理解为零拷贝或无需整文件内存。

来源：[V2 写入](../../src/io/stage1_checkpoint.cpp)、[V2/V1 恢复](../../src/io/stage1_checkpoint.cpp)、[小端原语和 CRC32C](../../src/io/stage1_binary.h)。

## 4. 后端载荷总览

| 路径 | 记录内容 | 标记/版本 | 数值表示与进度 |
| --- | --- | --- | --- |
| CPU Montgomery scalar / SIMD | 每曲线一条文本状态或完成结果 | `MPA-MONT-CKPT 1` | 普通域；ladder 已处理位数 |
| CPU Edwards scalar / SIMD | 每曲线一条二进制状态；完成后改为结果记录 | magic `0x1725bcd9`、version 6；或 `ECM_EDWARDS_DONE_V1` | 普通投影坐标；CPU 乘法进度号 |
| CUDA Montgomery ladder | 记录 0 包含整批头和数组 | magic `0x45555047`、version 5 | 切片边界的普通整数；二进制位进度 |
| OpenCL Montgomery PARAM3 | 记录 0 包含整批头和数组 | magic `0x45555047`、version 4 | 设备 Montgomery 域；二进制位进度 |
| CUDA Edwards binary / NAF | 记录 0 包含全部曲线 | `ECMEDB3`／`ECMEDB4`，版本 3／4 | 普通整数，进度为 binary 位或 NAF 列 |
| CUDA thread / OpenCL Edwards | 记录 0 包含全部曲线 | `CUDA_ED_THREAD_V1`／`OCL_ED_NAF_V1` | 设备 Montgomery 域的累加点；NAF 列游标 |
| CUDA resident / PRAC | 记录 0 包含整批驻留状态 | magic `0x31524d444d434545`、version 1 | Montgomery 域；位进度或素数计划位置 |
| CUDA NTT 单批 | 记录 0 包含一批算术状态 | `ECMNTTS1`／`ECMNTTS2`／`ECMNTTE1` | 规范投影坐标；x-only 位游标或 Edwards NAF 列游标 |
| CUDA NTT 拆批任务 | 外层只有记录 0，包含已处理前缀和至多一批算术快照 | `NTT-TASK-1`／`NTT-TASK-2`／`NTT-TASK-3` | 已处理曲线的结果、当前批大小和容量、嵌套算术快照 |

正常 Stage1 驱动通过 `ecm-batch-record:<i>` 虚拟地址访问记录。这个文本不是磁盘文件路径；载荷写入会更新对应记录并重新发布**整份外层文件**。单独调用底层 codec 时存在独立文件接口，不等于驱动会自动扫描这些独立文件。

CUDA thread 和 OpenCL Edwards 使用同一恢复流程。恢复时按原 sigma 重构每个通道。构造退化的通道使用恒等点占位，读取时同时核对 GCD=N 和保存的恒等累加点。[逐曲线 GPU 检查点实现](../../src/ecm/stage1/edwards/thread_workflow.h)

### 4.1 CUDA ladder V5 / OpenCL V4 的 72-byte 头

两者头字段次序相同，版本及数组语义不同。两种 codec 都直接保存本机头和数组字节。以下偏移对应当前小端 Windows x64，CUDA 的 `size_t`、`time_t` 均为 8 bytes；它不是独立于 ABI 的交换协议。OpenCL 头使用固定宽度字段并断言大小为 72 bytes，但同样没有逐字段做跨端序编码。

| 偏移 | 字节数 | 字段 | 含义 |
| --- | --- | --- | --- |
| 0 | 4 | magic | 数值 `0x45555047`，小端字节 `47 50 55 45` |
| 4 | 4 | version | CUDA=5，OpenCL=4 |
| 8 | 8 | s_partial | 已处理的指数位进度 |
| 16 | 8 | s_num_bits | 总指数位数 |
| 24 | 4 | batches_complete | 已执行切片数，有符号 i32 |
| 28 | 4 | curves | C |
| 32 | 8 | sigma | 批次起点；本批各曲线为 sigma+i |
| 40 | 4 | BITS | 容器位宽 K |
| 44 | 4 | TPI | 每个多精度实例线程数 |
| 48 | 4 | gpu_param | 曲线参数化 |
| 52 | 4 | reserved | 新写入为 0 |
| 56 | 8 | data_size | 设备数组字节数 A |
| 64 | 8 | timestamp | Unix 秒，有符号 64 位 |
| 72 | A | data | 按曲线排列的 u32 limb 数组，低 limb 在前 |
| 72+A | 8 | checksum | 头和 data 原始字节的 FNV 形式校验 |

此处校验初值为 **1469598103934665603**，乘数为 1099511628211，每字节先异或再乘、按 2⁶⁴ 回绕。它与 Edwards 使用的标准 FNV-1a 初值不同。当前两种读取器均要求尾部校验和精确文件长度，不接受仅有头和数组的无校验载荷。

| 路径/param | 每曲线的大整数次序 | A |
| --- | --- | --- |
| CUDA PARAM0/2 | N、a24、xdiff、aX、aZ、bX、bZ | 7CK/8 |
| CUDA PARAM3 | N、aX、aZ、bX、bZ | 5CK/8 |
| OpenCL PARAM3 | N、aX、aZ、bX、bZ | 5CK/8 |

单条大整数占 K/32 个 u32。CUDA ladder 在写回前退出 Montgomery 域；OpenCL 在切片间保持设备域。相同头、相同数组长度均不能据此跨两后端恢复。

CUDA 当前载荷检查版本、精确长度、校验、曲线数、参数化、指数总长/进度及数组几何，并恢复保存的 BITS/TPI；相应内核须存在。原始 V5 自动导入另逐 lane 核对完整 N。保存整个指数不必要：当前任务重新生成相应指数。

来源：[CUDA 头和读写](../../src/ecm/stage1/montgomery/cuda/checkpoint.cpp)、[CUDA 恢复布局](../../src/ecm/stage1/montgomery/cuda/runtime.cu)、[普通域写回](../../src/ecm/stage1/montgomery/cuda/arithmetic.cuh#L1148)、[OpenCL codec](../../src/ecm/stage1/montgomery/opencl/checkpoint.cpp)、[OpenCL 状态](../../src/ecm/stage1/montgomery/opencl/engine.cpp#L937)。

### 4.2 GPU Edwards V3/V4

这是外层记录 0 的内部偏移，均采用显式小端编码。

| 偏移 | 字节数 | 字段 | 含义 |
| --- | --- | --- | --- |
| 0 | 8 | magic | `ECMEDB3` 或 `ECMEDB4` 加 NUL |
| 8 | 4 | version | 3 或 4，与 magic 对应 |
| 12 | 4 | window | 0=binary；2..12=NAF |
| 16 | 4 | doubling | binary 为 0；NAF 为 4=HWCD、5=2XY |
| 20 | 4 | count | C |
| 24 | 4 | done | binary 已处理位数，或 NAF 已处理列数 |
| 28 | 4 | original_window | V3 为 0，V4 为内存降级前显式请求的窗口 |
| 32 | 8 | B1 | 完整界限 |
| 40 | 8 | scalar_bits | 48·LCM 指数位数 |
| 48 | 8 | scalar_fingerprint | 指数规范二进制编码的 FNV-1a 64 |
| 56 | 变长 | N | 长度前缀大整数，见下文 |
| N 之后 | 变长 | curves | 连续 C 条曲线 |
| 载荷长度−8 | 8 | checksum | 此前所有载荷字节的 FNV-1a 64 |

一个大整数编码为 u32 字节数 q，接 q 个低字节在前的无符号字节。零的 q=0；非零数最后一个字节不得为 0。读取 N、因子和坐标时，以当前 N 的字节数限制单个数的大小。

| 单曲线字段次序 | 编码 | 含义 |
| --- | --- | --- |
| sigma | u64 | 各曲线独立保存，不假设连续 |
| hit | u32，0 或 1 | 构造/计算已发现因子 |
| factor | 长度前缀大整数 | 对应因子，未命中通常为 0 |
| state[0..4] | 5 个长度前缀大整数 | d、基点 Px、Py、Pz、Pt |
| state[5..8] | 4 个长度前缀大整数 | 当前累加点 X、Y、Z、T |

这里两处 FNV 的初值均为 **14695981039346656037**，乘数为 1099511628211。指数指纹覆盖“u32 指数字节数+规范指数字节”；不只是指数数值文本，也不只是位数。完整指数从 B1 重建或由已验证缓存取得，不写入载荷。

恢复验证完整 N、B1、曲线数、指数位数及指纹、进度上界和固定 sigma。程序用 sigma 重构曲线和基点，核对 d 及全部基点状态、构造因子，并验证累加点范围、非全零、XY=ZT 及 Edwards 方程。构造退化通道使用 `hit=0`、`factor=0`、`d=0` 和恒等点。恢复时要求同一 sigma 重构得到退化结果，且累加点仍为恒等点。V4 仅用于显式窗口因内存不足而缩小的批次，满足 `2 ≤ window < original_window ≤ 12`。V3 和 V4 的坐标均为普通整数。恢复时采用保存的窗口，并按 N 选择容器。

大小精确为 **68 + bytes(N) + 52C + Σbytes(fᵢ) + Σbytes(vᵢⱼ)**，j=0..8。外层仍须再加自身开销。NAF 字典、完整指数、GPU 内存布局均不在该公式内。

来源：[载荷编码和恢复](../../src/ecm/stage1/edwards/cuda/checkpoint.cpp)、[GPU 驱动和安全边界](../../src/ecm/stage1/edwards/cuda/workflow.cpp)。

### 4.3 CPU Montgomery 文本记录 V1

字段各占一行，形式为字段名等号字段值；大整数为无 `0x` 的十六进制，计数为十进制。

| 次序 | 字段 | 含义 |
| --- | --- | --- |
| 首行 | `MPA-MONT-CKPT 1` | 标记和版本 |
| 公共 | N、B1、TORSION、SBITS | 实际模数、界限、1/12 标量倍数、总指数位数或 PRAC 描述符数量 |
| 公共 | CURVE、SIGMA、FIELD | 零基曲线索引、u64 种子、`mpn`/`ifma` 字段层标签或 `prac-<计划身份>` |
| 公共 | STATUS、BITNUM | `INFLIGHT`/`DONE`；已处理的指数位数或 PRAC 描述符数 |
| INFLIGHT | X0、Z0、X1、Z1 | ladder 相邻双点的普通域坐标。PRAC 用 X0/Z0 保存当前点，X1/Z1 为零。BITNUM=0 可作种子记录 |
| DONE | HIT；FACTOR 或 XOUT | 命中标志及真因子，或归一化末点 |
| 倒数第二行 | CHECKSUM | 16 位十六进制；覆盖此前每个字节，包括换行 |
| 最后一行 | END | 完整性结束标记 |

校验初值采用较短的 1469598103934665603。恢复要求 N/B1/TORSION/SBITS 一致，检查标记、结束行、校验和状态所需数据，上层同时核对曲线索引。标量和 SIMD 使用相同数学记录，实际批处理能否继续还取决于 lane 进度及外层 CPU 实现身份。CPU PRAC 的未完成记录仅在 FIELD 的计划身份匹配时继续，宽度变化导致身份不同时，从头执行未完成曲线。7 档 v1 与 v2 的描述符相同则共用计划身份。

来源：[记录结构](../../src/ecm/stage1/montgomery/cpu/checkpoint.h)、[写入](../../src/ecm/stage1/montgomery/cpu/checkpoint.cpp)、[读取](../../src/ecm/stage1/montgomery/cpu/checkpoint.cpp)。

### 4.4 CPU Edwards 二进制记录和完成记录

未完成记录保留 Prime95 风格 version 6 子格式，但正常驱动将它作为**私有外层的单曲线载荷**。这不代表整个 `.ckpt` 是 Prime95 存档。公共头如下；double 为当前宿主的 IEEE-754 64 位表示，整数小端。

| 偏移 | 字节数 | 字段 |
| --- | --- | --- |
| 0 | 4 | magic=`0x1725bcd9` |
| 4 | 4 | version=6 |
| 8 | 8 | k，double |
| 16、20、24 | 各 4 | b、n、c（c 为 i32） |
| 28 | 10 | stage，NUL 结尾的 `C{curve}S1` 标签 |
| 38 | 2 | padding，新写入为 0 |
| 40 | 8 | pct_complete，double |
| 48 | 4 | 内层 checksum，见下文 |
| 52 | 4 | curve；此 CPU 回调写 1，外层索引负责区分 lane |
| 56 | 8 | average_B2，写 0 |
| 64 | 4 | state=1 |
| 68 | 8 | sigma |
| 76、84 | 各 8 | B1、B2 |
| 92、96 | 各 4 | sigma_type=0、montg_stage1=0 |
| 100 | 8 | stage1_start_prime，当前回调写 2 |
| 108 | 4 | exp_buffer_size，当前指数位数 |
| 112 | 4 | bitnum，当前 CPU 乘法进度号 |
| 116 | 4 | dict_size=2^(w−2) |
| 120 | 变长 | dict_x、dict_y、e_x、e_y、e_z 五个 giant |

giant 为 u32 limb 数 m，随后 m 个低 limb 在前的 u32；当前写入零也使用至少一个零 limb。大小为 **140 + 4Σmⱼ** bytes。字典只存基点，不存整个预计算字典；累加点为普通投影 X:Y:Z，恢复时重建所需扩展坐标和字典。

内层 checksum 为模 2³² 的加法和：u32 加自身，u64 加高低两个 u32；每个 giant 加两次 limb 数和各 limb。公共头、average_B2、sigma 不计入该内层和，**外层 CRC32C 则覆盖它们**。完整实际 N 由外层 identity 绑定，不能只用子格式 k/b/n/c 标签检查余因子任务。标量和 SIMD 都使用此 codec；SIMD 恢复要求组合到同一批的 lane 进度兼容，否则该 SIMD 批可能重算。

CPU Edwards 的外层任务身份仍使用原请求窗口，部分记录的 `dict_size` 对应实际运行窗口。恢复时按部分记录固定实际窗口，以保持列号语义。若内存无法容纳该窗口，可降低 CPU 并发数，预算仍不足则保留存档并退出。

CPU 曲线处理完成后，对应记录改写为 `ECM_EDWARDS_DONE_V1`，依次保存 sigma、返回码、十六进制 factor，以及带双引号和转义的完整文本 save 行。返回码 0 表示无因子，1 表示真因子，3 表示构造退化。退化记录的 factor 为 0，文本行为空，恢复时重新验证该 sigma 的构造确实得到 GCD=N。记录依靠外层 CRC。发布失败或批次停止后可直接使用已完成的结果。

来源：[CPU Edwards codec](../../src/io/prime95_save.cpp)、[CPU 批次执行与完成记录](../../src/ecm/stage1/edwards/cpu/workflow.cpp)。Prime95 风格 MIDSTAGE 保留接口另见 [SAVE](SAVE.md#8-保留的二进制-midstage-接口)。

### 4.5 CUDA resident / PRAC

仅在编入实验模块并显式选择时存在，不是默认 ladder 载荷。112-byte 本机结构体头之后是 7CK/8-byte 状态数组。

| 偏移 | 类型 | 字段 |
| --- | --- | --- |
| 0 | u64 | magic=`0x31524d444d434545` |
| 8、16、24、32 | 各 u64 | b1、完整 scalar 的 hash、modulus 的 hash、计划 identity |
| 40、48、56 | 各 u64 | next、total、sigma |
| 64、72 | 各 u64 | bytes、data checksum |
| 80、84、88 | 各 u32 | version=1、algorithm（2=resident ladder，3=PRAC）、domain=1 |
| 92、96、100、104、108 | 各 u32 | bits、tpi、curves、torsion、reserved |
| 112 | bytes 字节 | 7 个大整数/曲线的设备状态，驻留 Montgomery 域 |

next/total 对 ladder 表示位进度，对 PRAC 表示素数计划位置。内层 checksum 仅为 data 的标准 FNV-1a 64，初值14695981039346656037；N/S 的 hash 按低 limb 到高 limb 的 GMP 本机 limb 字节计算，因此不构成跨 ABI 的规范数值摘要。模数、指数和计划身份通过头字段比对，外层 CRC 保护整个载荷。恢复要求版本、算法、域、N/S/计划 hash、B1、总长、BITS/TPI、曲线数及 torsion 等匹配。PRAC window 局部采样不产生完整 Stage1 save 或 checkpoint。

来源：[ResidentCheckpoint 与恢复](../../src/ecm/stage1/montgomery/cuda/prac/runtime.cu#L18)、[window 范围](../../src/ecm/stage1/montgomery/cuda/prac/prac_window.inl#L70)。

### 4.6 CUDA NTT 单批算术快照

原生 Edwards 用 `ECMNTTE1`，依次保存总列数、NAF 窗口、已处理列数、活跃曲线位图和 X/Y/Z。x-only 用 `ECMNTTS1`（批内至多 32 条真实曲线）或 `ECMNTTS2`（更多曲线），依次保存总位数、已处理位数、活跃曲线位图和阶梯的两对 X/Z。

计数和 32 位 limb 均为小端。位图每 32 条真实曲线用一个字，坐标依次按坐标面、活跃曲线及 limb 排列。恢复时使用同一 sigma 重建曲线和基点，并用快照中的模式及窗口重建标量编码和所需字典，重新核对总位数或列数。游标和点状态对应同一编码的标量前缀。后续没有算术快照的子批使用配置中的窗口。快照中的规范整数坐标允许重新选择兼容的卷积、变换和 Graph 参数。见 [NTT 快照读写](../../src/ecm/stage1/cuda/ntt/workflow.cpp#L77)。

### 4.7 NTT 拆批任务载荷

CUDA NTT 的队列任务按工作行的请求量拆批。直接任务超过配置上限或单批预算时也进入拆批调度。恢复时按检查点中的单批或拆批格式继续。调整 `ntt_max_batch` 后，已保存的批次按原宽度完成，后续新批按当前配置和预算选型。尾批默认运行剩余曲线。启用 `ntt_pad_batch` 时，资源和 sigma 范围允许就用新曲线补齐尾批，最多增加本批所选批量减一条。新增曲线与同批其他曲线共用恢复游标，各自保存结果。显存不足且尚无算术快照时，调度器可缩小批量后重试。[生产批量规则](../usage/STAGE1.md#任务曲线数与设备批量) · [拆批入口](../../src/ecm/stage1/task.cpp) · [调度与重试](../../src/ecm/stage1/cuda/ntt/batches.cpp)

所有子批使用同一个标准命名的 `.ckpt` 及其轮转备份。例如 Edwards M4423、B1=1024 使用 `.e_s1_m4423_1024.ckpt`。V2 外层记录 0 封装以下载荷，偏移从载荷首字节算起。

| 偏移 | 字节数 | 字段 | 内容 |
| --- | ---: | --- | --- |
| 0 | 10 | magic | ASCII `NTT-TASK-1`，无 NUL |
| 10 | 8 | first sigma | 整个任务的起始 sigma |
| 18 | 4 | preferred | 调度选出的目标批量 |
| 22 | 4 | curves | 当前子批的真实曲线数，包含已分配的补齐曲线 |
| 26 | 4 | capacity | 新写入等于 curves。旧档中若更大，恢复只采用原有真实曲线范围 |
| 30 | 4 | completed | 已处理的曲线槽位数（含构造退化），也是当前子批的零基起点 |
| 34 | 8 | native bytes | 当前子批算术快照的字节数 |
| 42 | native bytes | native | 上节的单批算术快照。长度为 0 时该子批从头启动 |
| 42 + native bytes | 变长 | completed results | 按任务顺序存放的已完成结果 |

每条完成记录为 1 字节结果类型、u32 数值长度及相应的无符号小端整数。类型 1 保存真因子，类型 2 保存归一化 Stage2 点（Edwards 保存其对应的 Montgomery 坐标 `u`），类型 3 表示算术完成但无可交接点，类型 4 表示 Edwards 构造退化。类型 3 和 4 的数值长度均为 0，类型 4 在恢复时以同一 sigma 重新验证 GCD=N。子批启动前先落盘批次范围和容量，周期写入或正常停止再补入仅含活跃曲线的算术快照。子批验证成功后，一次原子替换同时追加完成记录并清空当前快照，将 `curves`、`capacity` 和 `native bytes` 置零。下一个子批据此领取新的 sigma 区间。见 [任务载荷编码和验证](../../src/ecm/stage1/cuda/ntt/batches.cpp#L18)、[写入及提交子批](../../src/ecm/stage1/cuda/ntt/batches.cpp#L155)。

任务身份仍使用原始请求量。若请求 1000 条，在调整后的尾批上总计完成 1050 条，尾批写入的 `curves` 已含新增曲线，完成后记录 `completed=1050`。恢复使用同一条原始 worktodo 行，按检查点已有范围继续运行或发布。发布的 `finished` 和 Stage2 交接范围由实际写入 `.save` 的记录数确定。

#### 示例：每批 32 条，第五批运行中停止

假设前四批已完成，GPU 正在处理任务的第 129–160 条曲线。主文件记录 `completed=128`、`preferred=32`、`curves=32`、`capacity=32`，并保存前 128 条的完成记录。当前批的 sigma 为 `first sigma + 128` 至 `first sigma + 159`。

1. **正常停止。** GPU 到达下一个已同步的检查点边界后，回读当前批活跃曲线的坐标和共同游标，再原子替换同一个主文件。例如 Edwards NAF 已安全执行 256 列，快照就记下 `cursor=256`。即使 `checkpoint_interval_seconds=0`，正常停止仍会请求这次写入。如果第五批已经完成验证，则持久化为 `completed=160` 的批间状态。见 [GPU 同步边界](../../src/ecm/stage1/cuda/ntt/ladder.cu#L1168)、[停止与周期写入](../../src/ecm/stage1/cuda/ntt/workflow.cpp#L552)。
2. **恢复。** 读取并验证有效代，直接采用前 128 条的已完成结果。第 129–160 条沿用保存的批次范围，从第 257 列继续。自动后端规则改动时，匹配的检查点仍可使任务回到原 NTT 路径。如果强制终止，只能使用最后一次成功发布且仍有效的检查点。该代若只有第五批的范围，第五批从头执行，前 128 条仍可直接恢复。见 [自动后端恢复](../../src/io/normalize.cpp#L716)。
3. **最终发布。** 每个已完成子批的因子可先按单曲线发布。整条任务完成后才统一发布本次的 `.save`、适用的 NF 汇总和 Stage2 交接。若停止发生在末批已持久化与最终发布之间，恢复时直接使用完整结果重试发布。队列提交完成回执后清理检查点和轮转备份。见 [Stage1 结果发布](../../src/ecm/stage1/task.cpp#L398)、[队列回执与清理](../../src/main.cpp#L298)。

恢复时外层或任务载荷损坏可按轮转顺序尝试备份。任务载荷已接受、但嵌套快照未通过算术后端检查时，当前实现报错并保留文件。见 [嵌套 Context](../../src/io/stage1_checkpoint.cpp#L695)。

## 5. 写入、恢复和清理顺序

### 5.1 生命周期

1. 驱动完成输入/输出路径检查，计算完整身份，打开批次独占锁。
2. 尝试读取主文件及已配置备份；没有文件则创建内存中的空记录表。
3. CUDA 先以本后端离线 codec 验证候选载荷，再提交记录和存档 sigma；无载荷时采用已存种子，缺失时才生成。种子选择与参数化分开。种子遍历结束会发布一次外层；CPU codec 也可能写入种子记录。
4. 后端在安全边界按周期或停止请求写入载荷；写入一条虚拟记录会重新序列化整批。
5. 算术完成后发布 `.save`；随后写入内部 JSONL 和对外提交输出，待追加内容与确认写入配套的 `*.publish.ckpt`。
6. 直接运行在输出发布后清理 checkpoint。队列运行保留它，直到完成回执提交 finished 和 worktodo 变更，再清理主文件、备份及发布 ckpt。
7. 外部交接按各自队列或回执执行。这里的原子替换只覆盖单个文件。

只有 sigma、没有非空算术载荷的合法文件是“种子已确定”，**不表示保存了曲线内进度**。异常退出最多退回最后一次完整发布的检查点；0-byte `.save` 可来自可写性预检，不能据此认定 Stage1 完成。

`checkpoint_interval_seconds=0` 关闭周期写入，不关闭正常停止存档。信号处理器只置停止标志；CPU 分块或 GPU 切片结束后才执行 I/O。强制终止、蓝屏或掉电不会调用这一正常停止流程。GPU Edwards 会在最终结果发布前再保留一份完成的批次状态；CPU 完成记录同样支持发布重试。

来源：[批次发布边界](../../src/ecm/stage1/task.cpp#L124)、[种子持久化](../../src/io/stage1_checkpoint.cpp)、[GPU Edwards 完成状态](../../src/ecm/stage1/edwards/cuda/workflow.cpp)。

### 5.2 原子替换和备份

先完整写入 `.new` 并关闭；正常情况下将旧备份向后轮换，把原主文件复制为第一代备份，再用新文件替换主文件。Windows 替换使用 MoveFileExW 的 replace/write-through 标志，对共享冲突等错误有限重试约 500 ms；POSIX 使用 rename。当前 Stage1 写入没有与 Stage2 进度相同的显式 FlushFileBuffers 步骤，不应把原子命名替换等同于所有硬件上的掉电持久化保证。

恢复按主文件、bak、bak2…次序，不按内部进度重新排序。CUDA 在外层通过后暂存一个候选，调用当前曲线/算法的离线 codec 核对载荷，成功后才提交其记录和 sigma。载荷损坏可以继续尝试同身份备份；每次只保留一代候选及剩余路径。**外层有效但任务身份、曲线数或明确指定的 sigma 不匹配时直接拒绝**，显式内部算法/布局不兼容也报错。所有候选均损坏或不可读时返回 Corrupt / IoFailure，保留现场，不自动生成新种子重算。

由备份恢复后会标记主文件损坏，下一次发布不把损坏主文件轮换到良好备份上。调小备份上限后，超额备份只在新主文件成功安装后清除。备份提供另一份可用副本，不执行 Reed–Solomon 等纠错，也不合并两个损坏文件。

CUDA 离线校验不重入 storage，也不发布因子。Edwards 核对原始构造、点方程、算法及进度；Montgomery 核对本机布局、完整逐 lane 模数、规范范围、进度和当前内核可用性，resident/PRAC 另核对域、标量/计划身份。校验不是重新执行整个标量乘法的证明。虚拟地址在候选待验证时禁止读写或导入；失败写入回滚内存记录，失败导入同时回滚种子和载荷。CPU/OpenCL 仍先接纳外层，再按原 codec 处理内部记录；此 CUDA 候选协议不宣称改变它们的恢复语义。

CUDA Montgomery 的运行结果分别报告同步确认的计算游标、已回读的安全快照游标与已成功发布的持久化游标；计算前进不代表存档已提交。周期保存按 steady clock 计时。必要保存失败返回持久化错误并优先于停止，队列保留原行。未配置 checkpoint 路径时不创建隐式回退文件，停止进度不能恢复；末片完成与停止同时发生时完成当前批次的验证和发布，再由队列决定停止后续任务。

来源：[轮换和清理](../../src/io/stage1_checkpoint.cpp)、[文件替换](../../src/io/files.cpp)。

## 6. 仍可读取的兼容格式

| 来源 | 读取条件与处理 |
| --- | --- |
| 外层 `ECM_STAGE1_BATCH_V1` | 文本头、完整 identity、曲线数，随后各行 sigma/载荷长度及原始载荷；尾部十进制 CHECKSUM。按实际长度读取，载荷可为二进制；FNV 初值为较短的 1469598103934665603。下次写入改用 V2 |
| GPU Edwards `ECM_GPU_EDWARDS_BINARY_V1` | 文本完整 N、完整指数、B1/count/done，随后各曲线 sigma、hit、factor 和9个状态值；尾部十进制 FNV。默认恢复 binary，不将位号解释为 NAF 列 |
| GPU Edwards `ECM_GPU_EDWARDS_NAF_V2` | 同上，在标记后增加 window、doubling；完整指数为十六进制文本。读取时检查精确指数和状态，下一次写入 V3 |
| 独立 CUDA V5 `.ecm_ckpt_<bits>_<N首8hex>_<N尾8hex>.dat` | 仅 CUDA/CGBN 默认 ladder 路径在新批次没有算术载荷时尝试；短 N 的名称可能省略末尾片段。检查长度、原校验、布局/参数、进度、sigma 范围，并逐 lane 检查完整 N；V4 及更早不导入 |
| 独立 Edwards `ed_<旧模式>.save.gpu.ckpt` | 没有当前非空载荷时，尝试默认旧模式及当前 pattern 对应名称和备份；完整恢复验证后转入 V3，保留原文件 |
| 独立 CPU codec 文件 | codec 保留单文件读写能力；正常驱动只读批次记录，不承诺扫描并自动收集旧的每曲线散文件 |

外层 V1 的载荷长度行后、以及每个载荷后各有一个换行；尾校验覆盖尾行前的全部字节。GPU Edwards 旧文本的 FNV 使用标准初值 14695981039346656037。两类文本协议不能仅凭都有 CHECKSUM 就共用同一个校验函数。

原始独立 GPU 来源只读保留。若内部 JSONL 已记录其全部 sigma 完成，则跳过该独立旧档的自动导入。当前非空算术载荷优先，不用独立旧档覆盖它。

来源：[V1 外层读取](../../src/io/stage1_checkpoint.cpp)、[CUDA V5 导入](../../src/ecm/stage1/montgomery/cuda/checkpoint.cpp)、[Edwards 旧格式读取](../../src/ecm/stage1/edwards/cuda/checkpoint.cpp)、[独立 Edwards 导入](../../src/ecm/stage1/edwards/cuda/workflow.cpp)。

## 7. Stage2 队列进度格式

Stage2 不保存 NTT、多项式、树、fold 或单曲线内部状态。worktodo 和直接 `--save` 都保存已验证的曲线结果、进度及发布状态。dry-run 和 plan-only 只检查输入。

### 7.1 文件名和任务身份

文件名为 `.ecm_ckpt_stage2_<实际N位数>_<B1>.dat`。同位数和 B1 的不同任务依次加 `-1`、`-2` 等后缀；重启按档内身份选回原文件。任务身份与完整性校验仍使用 SHA256。目录由 `checkpoint_dir` 和使用说明中的路径优先级决定。

| 参与任务摘要 | 不参与任务摘要 |
| --- | --- |
| AID、规范实际 N、save **完整文件字节**的 SHA256、skip | 请求曲线数、有效 B2/Auto B2、worktodo 路径及行顺序、save 路径、worker 编号、设备、显存预算、D、承载模式、日志/结果路径 |

因此相同内容的 save 改名、任务重排或重启调整 B2/曲线数可保留进度；save 追加曲线、换行或元数据变化会改变完整文件 SHA256，从而产生新任务身份，即使选定曲线数学字段相同。skip 仍固定。运行期间仍须满足完成前的队列一致性检查。B2 和 Auto B2 模式作为可调整策略保存，不是输入身份，也不是逐曲线预测结果。

直接 `--save` 的文件名为 `.ecm_ckpt_stage2_direct_<任务摘要>.dat`。摘要另外区分所选曲线范围、请求 B2/Auto B2 和提交路径，同一调用可在异常后继续。完成后保留该 ckpt。

任务摘要输入是以下五项以单个 ASCII 空格连接的文本，无末尾换行：

| 次序 | 摘要输入项 | 编码 |
| --- | --- | --- |
| 1 | 固定域标记 | `stage2-curves-v3` |
| 2 | AID | quoted 字符串，空 AID 也保留引号 |
| 3 | 实际 N | quoted 的无 `0x` 小写十六进制 |
| 4 | save_sha256 | 原文件字节的64字符小写十六进制 SHA256 |
| 5 | skip | 无前导零十进制 |

对上述文本计算 SHA256 才得到文件名中的任务ID；它不是原始 worktodo 行的 SHA256。quoted 只转义双引号和反斜杠，与下一节字符串编码相同。

### 7.2 文本 schema 6

父进程独占任务 ckpt。CPU 曲线直接交回结果，GPU 曲线子进程经管道传回结果。父进程将结果和完成状态写入 ckpt。并行曲线乱序完成时，保存连续完成前缀和其后的完成位图。续跑只补缺口。缩小任务范围时须保留缺口后的已完成曲线。

- 公共头保存连续完成数、目标数、run、任务身份、B2 和 Auto B2。
- 结果区保存提交路径、最后一条用于汇总的 NF、已冻结的最终 NF、完成位图、NF 曲线键和完整因子行。
- 发布状态保存已确认的因子数、NF 确认标记和当前待追加批次的位置及锚点。
- 文件末尾保存此前全部字节的 SHA256。同目录临时文件先刷盘，再原子替换。

### 7.3 恢复与发布

1. 加载匹配的 ckpt，恢复尚待确认的因子和最终 NF。续算沿用 ckpt 中的提交路径。
2. 跳过已确认的曲线，按当前有效 B2 和曲线数继续。缩减目标数时保留实际完成的工作。
3. 单条因子经验证后立即发布。全部曲线完成后，用计划顺序中最后一条唯一 NF 的参数汇总。发布器先保存当前追加位置，再原子追加，最后将确认写回 ckpt。若在替换和确认之间异常退出，重试在保存位置后的新增区间查找该批完整字节。
4. 队列任务写入 finished、移除 worktodo 行之后清理 ckpt。直接任务保留完成 ckpt。

正常停止时，队列等待已运行曲线完成。强制退出后仅重算尚未写入 ckpt 的曲线。原始 Stage1 save 用于读取未完成的点，诊断 JSONL 可以归档。归档提交文件的条件见 [提交文件与归档](../usage/ECM_RESULTS.md#提交文件与归档)。

### 7.4 旧档的一次性迁移

schema 5 的结果与曲线进度可升级到 schema 6。升级时保留当前提交文件，程序在其中核对 ckpt 已保存的确切因子和最终 NF 行，随后写入新版确认。采用旧文件名的来源在迁移完成后改名为 `.migrated`。

schema 1–4 的已完成曲线或预约没有完整结果载荷。带有这类进度的任务须用原版本发布完成后再升级。Stage1 已开始发布的旧批次也应先用原版本完成发布，再切换到新版本。

来源：[State/Task](../../include/ecm/io/stage2_checkpoint.h)、[读写与兼容发现](../../src/io/stage2_checkpoint.cpp)、[恢复与运行顺序](../../src/ecm/stage2/queue.cpp)。

## 8. 存储、写入成本和边界

设 H=28+I+24C。下列为**一个当前主文件**大小；备份、临时文件、save 和发布 ckpt 另计。

| 内容 | 主文件字节数 |
| --- | --- |
| GPU 只有 sigma、无算术载荷 | H |
| CUDA PARAM0/2 ladder | H+80+7CK/8 |
| CUDA PARAM3 / OpenCL PARAM3 | H+80+5CK/8 |
| GPU Edwards V3/V4 | H+68+bytes(N)+52C+Σbytes(fᵢ)+Σbytes(vᵢⱼ) |
| CPU 各曲线记录 | H+Σ实际编码长度；文本十六进制不能按 limb 二进制大小计算 |
| 可选 resident/PRAC | H+112+7CK/8 |
| CUDA NTT 拆批 | H+42+当前子批快照字节数+Σ(5+已完成结果的数值字节数)，其中 C=1 |

GPU 保存前的回传量由后端决定：Montgomery ladder/OpenCL 要取回其整批数组；GPU Edwards 进度回调接收普通坐标，再序列化状态。不能把磁盘大小直接解释为 PCIe 流量。CPU 单曲线更新也发布整批外壳，并非就地修改24-byte记录头。

若当前批次文件大小为 S、备份上限为 k，稳定占盘量约为各代文件实际大小之和；等大小时约 (k+1)S，写入期间还可能有一份新临时文件。轮换和原主文件复制带来额外 I/O。各代可处于种子/算术不同状态，不能不加条件地用此估算代替实测峰值。

## 9. 验证入口与维护边界

- [容器/旧档离线检查](../../tools/test/stage1_checkpoint_container_test.cpp)：CRC 标准向量、长度、损坏、身份、备份、Edwards 全坐标往返与 V5 导入。
- [Stage1 完整/停止恢复](../../tools/test/test_stage1_usability.py)、[兼容及队列](../../tools/test/test_stage1_checkpoint_compat.py)。
- [独立外层解码](../../tools/test/stage1_checkpoint_decode.py)、[Edwards 点与载荷检查](../../tools/test/test_edwards_gpu_pipeline.py)。
- [NTT 拆批恢复](../../tools/test/test_stage1_ntt_task_batches.py)覆盖保留已完成前缀、按保存的 NAF 窗口恢复批内游标、后续子批切换到新窗口、尾批补齐与单次 Stage2 交接，并逐字节对比中断续跑和连续运行的 save。[NTT 生产测试](../../tools/test/test_stage1_ntt_production.py)覆盖原生 NAF 和 x-only 互换配置后的单批恢复。
- [GUI 检查点识别](../../src/ecm/application/gui/platform_win32.cpp#L107)：只识别主文件更新时间作为辅助信号，不能代替 worker 完成安全停止。

这些入口说明可复核的范围，不意味着每次构建都已经执行全部验证。新增格式或语义须同时更新写入、读取、身份、错误路径和本文；不要只修改 magic 或文件名。损坏资料和原始实验保留在忽略的 `data/`，不作为正式格式的新规则来源。
