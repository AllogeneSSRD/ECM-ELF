# ECM 结果字段与校验研究

研究对象为本地 `.refactor/p95v3106b01.source`、`.refactor/gpuowl`（PRPLL-NTT）、`.refactor/PrMers` 和 `.refactor/AutoPrimeNet` 的源码快照，核对日期 2026-10-16。以下结论只描述这些副本；服务器接受策略及其他版本不由本地代码推断。本仓库当前输出及限制见 [CPU Stage2](../usage/EDWARDS_CPU_STAGE2.md)。

## Prime95 ECM

`ecm.cpp:9942–9972` 构造 NF，`10063–10098` 构造 F：使用 `worktype=ECM`、指数或 N、B1、完成 B2、曲线数、FFT 长度、程序和时间。Edwards 用嵌套对象标记，F 记录含原始 sigma 和 stage。不能把未完成 Stage2 或只得到 Stage1 点的曲线计入完成 B2 的 NF。

Prime95 用 `SEC5(exponent,B1,B2)` 产生 8 位十六进制 `security-code`。本地 `commonb.h` 引用的 `security.h` 不在源码包中；没有该文件时 SEC5 宏回退为 0。这个回退不是生产校验算法。若干输出样本同为 8 位十六进制，无法据此确定它与 CRC32 的关系。

PRPLL-NTT 的 `src/Task.cpp:128–165` 构造结果公共字段和程序尾字段，未生成 `security-code` 或结果 checksum；`common.cpp` 的 CRC32 用于 `File.h` 的保存数据校验，不能据此当作 ECM 结果校验。

## PrMers CRC32

`PrMers/src/io/JsonBuilder.cpp:519–593` 拼接规范字段并生成 `checksum:{version:1,checksum:XXXXXXXX}`，底层为 `src/util/Crc32.cpp:96–113`。标准反射 CRC32 的多项式为 0xEDB88320，初值与末尾异或均为 0xFFFFFFFF，按 UTF-8 字节更新，输出大写 8 位十六进制。它与 Prime95 的字段名称和调用参数不同，不能视为 SEC5 的实现依据。

ECM 的规范串按以下顺序以分号连接，空字段仍占位置，最后时间后无分号：

1. exponent、`ECM`、逗号连接的本次 factors、逗号连接的 known-factors；
2. B1、已完成的 B2（没有则为空）、Edwards sigma 标记；
3. fft-length、`invariant:<计数>`；
4. program.name、program.version、program.kernel/subversion、program.details；
5. os.os、os.architecture、UTC timestamp。

PrMers Edwards sigma 标记以 `E` 开头，随后为十进制 sigma 或 `0x` 加小写十六进制；可附 `_TSG<torsion>`。没有提供 torsion 时不得猜测补入。每个字段的空值、大小写、因子顺序和时间都影响 CRC。它未覆盖所有 JSON 扩展字段，因此不能当作整个 JSON 文本的完整性签名。

PrMers 的 `recomputeChecksumFromSubmittedJson` 所用 `buildCanonicalStringFromSubmitted` 在本地副本仅重建 PRP/LL 特有字段，未重建 ECM 的 B1/B2/sigma；不能用该函数作为 ECM 的通用验证器。其生成器对 ECM invariant 计数使用的变量也需按具体输入核对。公开实现本身不证明服务器已接受 ECM 扩展布局。

一般整数 N 没有指数槽。本仓库为离线一般整数结果约定第一槽为 `N=<完整十进制N>`，其余使用上述 ECM 布局；这是本地扩展。面向梅森任务应显式使用经过目标关系检查的 exponent/known-factors。

## AutoPrimeNet 解析边界

`autoprimenet.py:7781` 的 `parse_result` 按 ECM 的 F/NF、curves、b1/b2、stage 和 factors 解析；此处 `security-code` 只用于可选的人类可读描述，没有生成算法。`RESULT_PATTERN` 的扫描程序名集合与解析函数是不同层，直接解析成功不表示扫描器能发现新程序。离线检查不调用报告或网络函数，不能替代服务器验收。
