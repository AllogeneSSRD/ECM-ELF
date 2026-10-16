# Edwards统一存档、CPU Stage2与提交结果草稿

临时开发记录，完成后保留待用户处理。当前阶段目标为CPU/GPU Edwards Stage1统一多曲线文本save，CPU Stage2直接续算已完成点，并形成符合Prime95 ECM结构的JSONL提交方案。

## 2026-10-16：用户确认及范围

已确认：CPU/GPU统一多曲线文本save；旧CPU Edwards二进制存档及Prime95 feeder仅作为备选，无需新CPU Stage2兼容。CPU Stage2目标B2范围参考当前GPU Stage2，需考虑1e12量级；允许接入libecm，并实测CPU NTT/非NTT多项式后端。逐素数ladder仅适合小界限正确性参考，现有工具中的全量prime/giant数组不能直接作为大B2生产实现。

已确认：AutoPrimeNet仅作格式和校验流程参考，不做适配补丁，不发送结果。用户要求检查本地PRPLL-NTT和PrMers，且确认 `.refactor/gpuowl` 就是PRPLL-NTT。核对后同意省略security-code，只实现公开checksum。程序使用README中的正式项目名ECM-ELF；目录名MPA-OpenCl不作为提交身份。

## 2026-10-16：现状核对

GPU Edwards普通CLI会将空tmp_dir设为当前目录，常规路径已有默认文本save；其底层接口允许无文件输出。CPU Edwards忽略savefilename/saveappend，只在本地写单曲线Prime95二进制存档，尚无统一多曲线文本交接。存档发布失败必须让任务保持未完成，不能只记录错误后推进队列。

Prime95 `ecm.cpp:9942–9972` 定义NF结构，`10063–10098` 定义F结构；Edwards模型标记为嵌套 `Edwards` 对象，因子结果包含完整sigma。不能把Stage1末点、未完成或被中断的Stage2当成已完成B2报告。AutoPrimeNet的 `RESULT_PATTERN` 仅扫描固定程序名；本轮仅验证ECM解析前缀，不修改扫描器。

CPU Stage2将沿用Edwards的48LCM身份，重构其Montgomery常数 `(A+2)/4=1/(1-d)`，直接消费已完成Stage1点，不重复Stage1。大B2算法、依赖构建、内存及实际覆盖界限仍待验证后写入正式文档。

## 2026-10-16：实现与定位

CPU/GPU完成后统一使用原子文本发布。CPU实现默认路径、save/savea、顺序保存、POINT/FACTOR检查，以及发布失败保留队列；GPU底层取消无文件分支。旧CPU二进制仍是备选，不属于新读取器支持范围。标量和SIMD区分停止返回值2与真因子，避免中断被当成因子。

固定样本 `N=1009*(2^127−1), sigma=19, B1=1000` 复现CPU漏因子，GPU找到1009。排查假设依次为构造求逆、末点GCD、保存转换；先将构造接入共享检查器后仍失败，进一步确认大NAF字典的批量求逆失败使整份字典零化。修复为提取真因子；乘积GCD=N时逐项检查，不能找到真因子则报错。恢复归一化同步检查；SIMD不再继续使用失效字典，而转逐曲线检查。标量、SIMD与GPU均通过四个sigma因子回归。构造及CPU检查点基点重构统一使用共享检查器。

libecm依赖原生MSVC构建成功，使用本地源码7.0.6-dev、提交17b5cdf1536b96fce4a7b92832f57f9028fbe570。独立CPU Stage2设置A、x及B1done跳过Stage1；S=1，支持NTT/plain，按上游30-limb阈值auto选择。输入与输出别名、模型/校验损坏及元数据不一致均拒绝；中断、错误和饱和因子不发NF。

小内存测试曾错误地预期1MiB规划预算会立即拒绝B2=10^12，实际libecm选择大量分块。该测试子进程已停止，失败证据保留在gates_03；没有把人工终止作为内存拒绝资格。正式门禁改为小B2低预算正常完成及非法预算拒绝；文档明确maxmem不是RSS硬上限。

## 2026-10-16：验证、结果与资料

最终功能门禁为 `data/experiments/edwards_full_pipeline_20261016/gates_08/summary.json`。独立48LCM末点、CPU标量/多线程/SIMD、GPU1对照、两个Stage2后端、Stage1/2因子、梅森余因子元数据、结果CRC、AutoPrimeNet解析前缀、输入拒绝、Stage1保存失败队列保持、真实停止恢复及Stage2中断已覆盖。旧失败轮次与编译失败日志均保留，不改写原始证据。

`bench_01` 和 `bench_1e12_01` 记录14个后端格、每格1次预热加3次正式样本，共42个正式完整曲线；B2=10^12在127位的两种后端均完整结束。该范围不外推2203位的10^12成本。计时与单进程峰值工作集、CPU功耗未测量条件见正式主题。图表位于 `data/figures/edwards_cpu_stage2_20261016/`。

Prime95本地源码缺security.h、SEC5回退0；PRPLL-NTT没有结果security-code；PrMers的公开CRC32与SEC5没有等价依据。JSON使用ECM-ELF身份与Prime95 ECM字段结构，公开checksum独立校验通过。没有服务器提交，且没有修改AutoPrimeNet。服务器对新身份和ECM校验扩展的接受情况仍待外部确认。

正式主题为 [统一存档与CPU Stage2](../usage/EDWARDS_CPU_STAGE2.md)，固定资料为 [ECM结果字段与校验研究](../reference/ECM_RESULT_JSON.md)。GUI/队列、曲线内部Stage2恢复、发布打包和大位宽10^12资格尚未实现。本临时草稿保留待用户处理，不自动删除。
