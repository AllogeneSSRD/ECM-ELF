# Checkpoint：状态、格式与恢复

本文是当前 checkpoint 的格式与持久化说明。完成 Stage1 后的交换文件另见 [SAVE](SAVE.md)；启动和停止操作见 [Stage1 使用](../usage/STAGE1.md)、[Stage2 使用](../usage/STAGE2.md)。配置键的默认值、范围只在[生成的 TOML 参考](../ECM_TOML_REFERENCE.md)维护。

## 1. 文件职责与范围

| 对象 | 保存内容 | 恢复粒度 | 是否供另一程序执行 Stage2 |
| --- | --- | --- | --- |
| Stage1 批次 checkpoint | 曲线身份、sigma、各后端未完成状态或已完成结果 | CPU 曲线分块、GPU 切片 | 否 |
| Stage1 `.save` | 已完成 Stage1 的归一化点或已发现因子 | 一条完成的曲线 | 依曲线族和消费者能力，见 SAVE |
| Stage2 队列 checkpoint | 已完成曲线数、任务身份、待确认结果回执 | 完整曲线 | 否，不含曲线内运算状态 |
| 指数缓存、PRAC 计划、tune profile | 可复用计算或性能资料 | 不代表任务进度 | 否 |

Stage1 外层统一不代表内部算术状态通用。CUDA ladder、CUDA resident/PRAC、OpenCL、CPU Montgomery、CPU Edwards、GPU Edwards 使用不同载荷；改名不能将它们互相转换。CRC、FNV 与结果回执均不能证明标量乘法在数学上正确。

## 2. Stage1 文件名与身份

### 2.1 物理文件

一个任务使用一个主文件；CPU 曲线记录也封装在其中，不由正常驱动逐曲线创建文件。

| 曲线 | 任务有原始指数 n | 普通整数、无指数元数据 |
| --- | --- | --- |
| Edwards | `.e_s1_m{n}_{B1}.ckpt` | `.e_s1_bit{bits}_{B1}.ckpt` |
| Montgomery | `.m{param}_s1_m{n}_{B1}.ckpt` | `.m{param}_s1_bit{bits}_{B1}.ckpt` |

param 为 0、2、3，CPU Montgomery 使用 0；bits 为实际模数位数，B1 为完整十进制整数。k·bⁿ+c 任务使用其原始 n 作为命名标签，即使剥离了已知因子；没有这类元数据时，还会识别实际 N 是否为 2ⁿ−1。因此 `m` 命名标签不能代替内容中的完整 N 检查，也不能单独证明原任务是梅森数。

文件位于 `tmp_dir`。未配置有效目录的路径不能获得同等的批次持久化保证。文件名不包含 worker、sigma、曲线数量或后端；同名不同任务需使用不同目录。

相关文件如下：

| 名称 | 作用 |
| --- | --- |
| 主文件 | 当前已发布批次 |
| `主文件名.new` | 待发布的完整新文件，不作为自动恢复候选 |
| `.bak.ckpt`、`.bak2.ckpt`… | 后缀前插入备份代数；最多 16 代，实际数量由 `ckpt_backups` 决定 |
| `主文件名.lock` | 批次独占锁；Windows 使用无共享打开、关闭时删除；POSIX 使用非阻塞 flock |

来源：[驱动命名与批次边界](../../src/ecm/service/stage1/task.cpp#L138)、[备份及锁](../../src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp#L97)。

### 2.2 内容身份

外层 identity 为 UTF-8/ASCII 文本，以竖线分隔，按完整字符串比较。不是结构体，也不是文件名摘要。

| 顺序 | 内容 | 当前编码 |
| --- | --- | --- |
| 1 | 实际 N | 小写十六进制，无 `0x` |
| 2 | 曲线构造 | `ed`、`mont0`、`mont2`、`mont3` |
| 3 | B1 | 十进制整数 |
| 4 | 标量约定枚举 | 0=LCM，1=12·LCM；Edwards 以自身固定 48·LCM 解释 |
| 5 | 执行目标 | `cpu` 或 `gpu` |
| CPU 附加 1 | 有效 NAF 窗口设置 | 十进制；由驱动加入身份，即使某 CPU 路径不使用它 |
| CPU 附加 2 | 实现枚举 | 0=auto、1=IFMA SIMD、2=GMP scalar |
| GPU Montgomery 附加 | GPU 算法选择文本 | `ECM_GPU_STAGE1_ALGO` 的非空值；未设置时为 `ladder` |

曲线数另在外层头中比较；各 sigma 在记录头中保存。固定 sigma 任务要求每条非零已存 sigma 等于请求起点+i；随机任务采用已存种子。GPU Edwards 的窗口、binary/NAF 模式及倍点变体在载荷内验证。

设备编号、GPU 型号、驱动版本、D、B2、输出路径不在该外层身份中。不能据此推断任意设备或后端之间均可恢复：内部布局、构建包含的内核和算法约束仍须满足。CPU auto 切换实际实现的情况还受各后端的分块恢复条件约束。

## 3. 当前外层二进制格式 V2

### 3.1 编码约定

- 所有显式 u32/u64 均为无符号小端整数；偏移从文件首字节 0 开始。
- 字符串和载荷采用“字节长度+原始字节”，不附加 NUL 或换行。
- I 为 identity 字节数，C 为曲线数，第 i 条载荷长度为 Lᵢ。
- GPU 批次载荷置于记录 0，其余记录通常 Lᵢ=0，但仍保存各自 sigma。CPU 通常每条记录保存相应曲线状态。

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

来源：[V2 写入](../../src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp#L178)、[V2/V1 恢复](../../src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp#L205)、[小端原语和 CRC32C](../../src/ecm/storage/checkpoint/ecm_stage1_binary.h#L10)。

## 4. 后端载荷总览

| 路径 | 记录内容 | 标记/版本 | 数值表示与进度 |
| --- | --- | --- | --- |
| CPU Montgomery scalar / SIMD | 每曲线一条文本状态或完成结果 | `MPA-MONT-CKPT 1` | 普通域；ladder 已处理位数 |
| CPU Edwards scalar / SIMD | 每曲线一条二进制状态；完成后改为结果记录 | magic `0x1725bcd9`、version 6；或 `ECM_EDWARDS_DONE_V1` | 普通投影坐标；CPU 乘法进度号 |
| CUDA Montgomery ladder | 记录 0 包含整批头和数组 | magic `0x45555047`、version 5 | 切片边界的普通整数；二进制位进度 |
| OpenCL Montgomery PARAM3 | 记录 0 包含整批头和数组 | magic `0x45555047`、version 4 | 设备 Montgomery 域；二进制位进度 |
| CUDA Edwards binary / NAF | 记录 0 包含全部曲线 | `ECMEDB3`、version 3 | 普通整数；binary 位或 NAF 列 |
| CUDA resident / PRAC（可选构建） | 记录 0 包含整批驻留状态 | magic `0x31524d444d434545`、version 1 | Montgomery 域；位进度或素数计划位置 |

正常 Stage1 驱动通过 `ecm-batch-record:<i>` 虚拟地址访问记录。这个文本不是磁盘文件路径；载荷写入会更新对应记录并重新发布**整份外层文件**。单独调用底层 codec 时存在独立文件接口，不等于驱动会自动扫描这些独立文件。

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

来源：[CUDA 头和读写](../../src/ecm/backend/cuda/stage1/montgomery/host_io.cpp)、[CUDA 恢复布局](../../src/ecm/backend/cuda/stage1/montgomery/adapter.cu)、[普通域写回](../../src/ecm/backend/cuda/stage1/montgomery/arithmetic.cuh#L1148)、[OpenCL codec](../../src/ecm/storage/checkpoint/ecm_checkpoint.cpp#L131)、[OpenCL 状态](../../src/ecm/backend/opencl/stage1/montgomery/engine.cpp#L937)。

### 4.2 GPU Edwards V3

这是外层记录 0 的内部偏移，均采用显式小端编码。

| 偏移 | 字节数 | 字段 | 含义 |
| --- | --- | --- | --- |
| 0 | 8 | magic | `ECMEDB3` 加 NUL |
| 8 | 4 | version | 3 |
| 12 | 4 | window | 0=binary；2..12=NAF |
| 16 | 4 | doubling | binary 为 0；NAF 为 4=HWCD、5=2XY |
| 20 | 4 | count | C |
| 24 | 4 | done | binary 已处理位数，或 NAF 已处理列数 |
| 28 | 4 | flags | 必须为 0 |
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

恢复验证完整 N、B1、曲线数、指数位数及指纹、进度上界和固定 sigma；用 sigma 重构曲线和基点，核对 d 及全部基点状态、构造因子，并验证累加点范围、非全零、XY=ZT 及 Edwards 方程。默认使用所存算法；显式 binary/NAF、窗口或倍点选择不符时报错。坐标为普通整数，因此 V3 不保存 BITS/TPI，恢复重新选择可用容器。

大小精确为 **68 + bytes(N) + 52C + Σbytes(fᵢ) + Σbytes(vᵢⱼ)**，j=0..8。外层仍须再加自身开销。NAF 字典、完整指数、GPU 内存布局均不在该公式内。

来源：[V3 编码和恢复](../../src/ecm/backend/cuda/stage1/edwards/legacy/ecm_edwards_gpu_checkpoint.h#L38)、[GPU 驱动和安全边界](../../src/ecm/backend/cuda/stage1/edwards/host_workflow.cpp)。

### 4.3 CPU Montgomery 文本记录 V1

字段各占一行，形式为字段名等号字段值；大整数为无 `0x` 的十六进制，计数为十进制。

| 次序 | 字段 | 含义 |
| --- | --- | --- |
| 首行 | `MPA-MONT-CKPT 1` | 标记和版本 |
| 公共 | N、B1、TORSION、SBITS | 实际模数、界限、1/12 标量倍数、总指数位数 |
| 公共 | CURVE、SIGMA、FIELD | 零基曲线索引、u64 种子、`mpn`/`ifma` 字段层标签 |
| 公共 | STATUS、BITNUM | `INFLIGHT`/`DONE`；ladder 已处理进度 |
| INFLIGHT | X0、Z0、X1、Z1 | 相邻 ladder 点的普通域坐标；BITNUM=0 可作种子记录 |
| DONE | HIT；FACTOR 或 XOUT | 命中标志及真因子，或归一化末点 |
| 倒数第二行 | CHECKSUM | 16 位十六进制；覆盖此前每个字节，包括换行 |
| 最后一行 | END | 完整性结束标记 |

校验初值采用较短的 1469598103934665603。恢复要求 N/B1/TORSION/SBITS 一致，检查标记、结束行、校验和状态所需数据；上层同时核对曲线索引。标量和 SIMD 使用相同数学记录；实际批处理是否能继续还取决于 lane 进度等调用条件，不能仅根据 FIELD 文本推断可以切换实现。外层 CPU 实现身份也会限制切换。

来源：[记录结构](../../src/ecm/storage/checkpoint/ecm_mont_ckpt.h#L41)、[写入](../../src/ecm/storage/checkpoint/ecm_mont_ckpt.cpp#L96)、[读取](../../src/ecm/storage/checkpoint/ecm_mont_ckpt.cpp#L147)。

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

CPU 曲线完成后，对应记录改写为 `ECM_EDWARDS_DONE_V1`：依次保存 sigma、返回码 0/1、十六进制 factor，以及带双引号/转义的完整文本 save 行；无可用点时该行可以为空。记录没有独立尾校验，依靠外层 CRC。保存已完成结果是为了发布失败或批次停止后不重复计算已完成曲线。

来源：[CPU Edwards codec](../../src/ecm/storage/save/ecm_edwards_save.cpp#L279)、[驱动回调](../../src/ecm/application/cli/ecm_driver.cpp#L878)、[完成记录](../../src/ecm/application/cli/ecm_driver.cpp#L941)。Prime95 风格 MIDSTAGE 保留接口另见 [SAVE](SAVE.md#8-保留的二进制-midstage-接口)。

### 4.5 可选 CUDA resident / PRAC

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

来源：[ResidentCheckpoint 与恢复](../../src/ecm/backend/cuda/stage1/montgomery/prac/prac_runtime.inl#L18)、[window 范围](../../src/ecm/backend/cuda/stage1/montgomery/prac/prac_window.inl#L70)。

## 5. 写入、恢复和清理顺序

### 5.1 生命周期

1. 驱动完成输入/输出路径检查，计算完整身份，打开批次独占锁。
2. 尝试读取主文件及已配置备份；没有文件则创建内存中的空记录表。
3. 逐曲线确定 sigma。种子遍历结束会发布一次外层；CPU codec 也可能写入种子记录。
4. 后端在安全边界按周期或停止请求写入载荷；写入一条虚拟记录会重新序列化整批。
5. 算术完成后发布 `.save`；随后写入内部结果账本及对外提交输出。
6. 这些发布成功后，驱动才结束 completed 批次并删除主 checkpoint、临时文件和各代备份，释放锁。
7. 队列层再记录 finished、推进 worktodo 和进行受支持的外部交接。它们不是与三个输出文件组成一个跨文件原子事务。

只有 sigma、没有非空算术载荷的合法文件是“种子已确定”，**不表示保存了曲线内进度**。异常退出最多退回最后一次完整发布的检查点；0-byte `.save` 可来自可写性预检，不能据此认定 Stage1 完成。

`ckpt_seconds=0` 关闭周期写入，不关闭正常停止存档。信号处理器只置停止标志；CPU 分块或 GPU 切片结束后才执行 I/O。强制终止、蓝屏或掉电不会调用这一正常停止流程。GPU Edwards 会在最终结果发布前再保留一份完成的批次状态；CPU 完成记录同样支持发布重试。

来源：[批次发布边界](../../src/ecm/service/stage1/task.cpp#L138)、[种子持久化](../../src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp#L277)、[GPU Edwards 完成状态](../../src/ecm/backend/cuda/stage1/edwards/host_workflow.cpp)。

### 5.2 原子替换和备份

先完整写入 `.new` 并关闭；正常情况下将旧备份向后轮换，把原主文件复制为第一代备份，再用新文件替换主文件。Windows 替换使用 MoveFileExW 的 replace/write-through 标志，对共享冲突等错误有限重试约 500 ms；POSIX 使用 rename。当前 Stage1 写入没有与 Stage2 进度相同的显式 FlushFileBuffers 步骤，不应把原子命名替换等同于所有硬件上的掉电持久化保证。

恢复按主文件、bak、bak2…次序，采用首个外层有效且任务匹配的候选，不按内部进度重新排序。损坏主文件可以回退备份；**外层有效但任务身份不匹配时直接拒绝，不继续找另一个身份的备份**。所有存在候选都无效时保留现场并报错。

由备份恢复后会标记主文件损坏，下一次发布不把损坏主文件轮换到良好备份上。调小备份上限后，超额备份只在新主文件成功安装后清除。备份提供另一份可用副本，不执行 Reed–Solomon 等纠错，也不合并两个损坏文件。

外层有效不代表后端语义有效。进入虚拟记录 codec 后，候选只有该记录；不会因内部状态/显式算法不匹配而自动再遍历整批旧备份。CUDA 当前路径会拒绝不兼容的非空载荷；CPU 部分 codec 的拒绝可导致曲线/批次重算，具体按调用入口处理。

来源：[轮换和清理](../../src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp#L121)、[文件替换](../../src/ecm/platform/ecm_file_publication.h#L13)。

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

原始独立 GPU 来源只读保留。若内部结果账本已经确认其全部 sigma 完成，则跳过自动导入；仅将旧文件复制到全新目录而不携带账本，无法从该旧文件得知它是否已在别处完成。当前非空算术载荷优先，不用独立旧档覆盖它。

来源：[V1 外层读取](../../src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp#L254)、[CUDA V5 导入](../../src/ecm/storage/checkpoint/ecm_stage1_legacy_gpu.h#L14)、[Edwards 旧格式读取](../../src/ecm/backend/cuda/stage1/edwards/legacy/ecm_edwards_gpu_checkpoint.h#L97)、[独立 Edwards 导入](../../src/ecm/backend/cuda/stage1/edwards/host_workflow.cpp)。

## 7. Stage2 队列进度格式

Stage2 不保存 NTT、多项式、树、fold 或单曲线内部状态。只在队列模式持久化完整曲线的完成数；直接 `--save`、dry-run 和 plan-only 不使用该队列恢复合同。

### 7.1 文件名和任务身份

文件名为 `.ecm_ckpt_stage2_<实际N位数>_<64字符小写十六进制任务SHA256>.dat`；SHA256 本身为256 bits。目录由 `stage2_progress_dir` 和使用说明中的路径优先级决定。

| 参与任务摘要 | 不参与任务摘要 |
| --- | --- |
| AID、规范实际 N、save **完整文件字节**的 SHA256、skip、实际可执行曲线数、有效 B2/自动 B2 模式 | worktodo 路径及行顺序、save 路径、worker 编号、设备、显存预算、D、承载模式、日志/结果路径 |

因此相同内容的 save 改名或任务重排可保留进度；save 追加曲线、换行或元数据变化也会改变完整文件 SHA256，从而产生新任务身份，即使选定曲线的数学字段相同。运行期间仍须满足完成前的队列一致性检查。自动 B2 身份记录“自动且显式 B2=0”的模式，而不是逐曲线预测出来的 B2 数值。

任务摘要输入是以下八项以单个 ASCII 空格连接的文本，无末尾换行：

| 次序 | 摘要输入项 | 编码 |
| --- | --- | --- |
| 1 | 固定域标记 | `stage2-curves-v2` |
| 2 | AID | quoted 字符串，空 AID 也保留引号 |
| 3 | 实际 N | quoted 的无 `0x` 小写十六进制 |
| 4 | save_sha256 | 原文件字节的64字符小写十六进制 SHA256 |
| 5、6 | skip、total | 无前导零十进制；total 是实际可执行数量 |
| 7 | b2 | 最终采用的显式/配置 B2；自动模式为0 |
| 8 | auto_b2 且 b2=0 | 十进制0或1 |

对上述文本计算 SHA256 才得到文件名中的任务ID；它不是原始 worktodo 行的 SHA256。quoted 只转义双引号和反斜杠，与下一节字符串编码相同。

### 7.2 文本 schema 2

文件是单行文本，空格分隔，最后有换行。下表文本字段均采用带双引号、反斜杠转义的 quoted 编码。

| 次序 | 字段 | 编码/含义 |
| --- | --- | --- |
| 1 | schema | 十进制 2 |
| 2、3 | done、total | u64，0≤done≤total |
| 4 | run | 非空随机运行标识，用于 finished 去重 |
| 5 | pending | 当前曲线随机回执标识；无待确认结果时为空 |
| 6 | identity | 上节任务 SHA256，64 个小写十六进制字符 |
| 7 | pending_results | 待确认回执对应的结果文件路径；空值使用当前结果路径 |

它没有逐记录 CRC 或尾部 checksum；依赖结构检查、任务摘要、原子发布和持久结果回执。done=total 时 pending 必须为空。临时文件为 `.tmp.<PID>`，先 WriteFile 和 FlushFileBuffers，再用 MoveFileExW 替换；独占 `.lock` 阻止同一任务同时消费。Stage2 不调用 Stage1 的多代备份轮换。

### 7.3 结果回执与恢复顺序

1. 执行曲线前，保存 pending 随机标识及结果文件路径。
2. 子进程完成曲线，验证并写入包含该标识的结果回执。
3. 父进程确认回执存在，才令 done 加一、清空 pending 并原子保存。
4. 若中断发生在步骤2和3之间，启动时从原结果文件找完整回执并补记；没有回执则重新执行当前曲线。未以换行结束的尾行不算有效回执。
5. 所有曲线完成后，先写带 run 标记的 finished，再移除原 worktodo 行，最后删除进度主文件。失败保留可恢复状态。

旧 schema 1 / 单文件 `.progress` 仅作为兼容导入来源。按任务目标、AID、save 内容和范围匹配，成功后保留原字节为 `.migrated.<摘要>`；不匹配的旧任务不阻止当前新任务。不是任意 `.dat` 文件都能导入。

来源：[State/Task 声明](../../include/ecm/storage/queue/stage2_progress.h)、[schema 与结果回执实现](../../src/ecm/storage/queue/stage2_progress.cpp)、[恢复与运行顺序](../../src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp#L1395)。

## 8. 存储、写入成本和边界

设 H=28+I+24C。下列为**一个当前主文件**大小；备份、临时文件、save 和结果账本另计。

| 内容 | 主文件字节数 |
| --- | --- |
| GPU 只有 sigma、无算术载荷 | H |
| CUDA PARAM0/2 ladder | H+80+7CK/8 |
| CUDA PARAM3 / OpenCL PARAM3 | H+80+5CK/8 |
| GPU Edwards V3 | H+68+bytes(N)+52C+Σbytes(fᵢ)+Σbytes(vᵢⱼ) |
| CPU 各曲线记录 | H+Σ实际编码长度；文本十六进制不能按 limb 二进制大小计算 |
| 可选 resident/PRAC | H+112+7CK/8 |

GPU 保存前的回传量由后端决定：Montgomery ladder/OpenCL 要取回其整批数组；GPU Edwards 进度回调接收普通坐标，再序列化状态。不能把磁盘大小直接解释为 PCIe 流量。CPU 单曲线更新也发布整批外壳，并非就地修改24-byte记录头。

若当前批次文件大小为 S、备份上限为 k，稳定占盘量约为各代文件实际大小之和；等大小时约 (k+1)S，写入期间还可能有一份新临时文件。轮换和原主文件复制带来额外 I/O。各代可处于种子/算术不同状态，不能不加条件地用此估算代替实测峰值。

## 9. 验证入口与维护边界

- [容器/旧档离线检查](../../tools/test/stage1_checkpoint_container_test.cpp)：CRC 标准向量、长度、损坏、身份、备份、Edwards 全坐标往返与 V5 导入。
- [Stage1 完整/停止恢复](../../tools/test/test_stage1_usability.py)、[兼容及队列](../../tools/test/test_stage1_checkpoint_compat.py)。
- [独立外层解码](../../tools/test/stage1_checkpoint_decode.py)、[Edwards 点与载荷检查](../../tools/test/test_edwards_gpu_pipeline.py)。
- [GUI 检查点识别](../../src/gui/platform_win32.cpp#L107)：只识别主文件更新时间作为辅助信号，不能代替 worker 完成安全停止。

这些入口说明可复核的范围，不意味着每次构建都已经执行全部验证。新增格式或语义须同时更新写入、读取、身份、错误路径和本文；不要只修改 magic 或文件名。损坏资料和原始实验保留在忽略的 `data/`，不作为正式格式的新规则来源。
