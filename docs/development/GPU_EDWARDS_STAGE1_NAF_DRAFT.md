# GPU Edwards Stage1 窗口优化草稿

本草稿记录当前开发、验证及原始证据位置，完成后保留待用户处理。

## 目标与验收

- GPU1 RTX 4060 Laptop 上测量二进制 Edwards、Montgomery PARAM0/PARAM3 在常用不同位宽的完整 Stage1 性能，冻结优化前版本。
- 接入窗口 NAF，核对完整标量结果、复合数因子、所有容器档、任意分块停止恢复及 Stage2 交接；不能用性能样本替代正确性验证。
- 研究 Prime95 倍点算法，按实测选择有效实现；更新当前主题说明并提交 Git。

## 参考与当前问题

基线为 `fc66536945ca9dc796781236f9df827f66a284e0`，GPU Edwards 使用二进制倍加、256 位切片与普通域检查点。CPU Edwards 已有窗口 NAF，可作独立执行路径的末点核对。

第三方资料为 `.refactor/p95v3106b01.source/undoc.txt:367` 与 `ecm.cpp:2106`。文档列算法 4/5/6，但同版源码 `ecm.cpp:2143` 明确说明算法 6 已废弃，实际执行分支为 5、5a、7、4；不能据旧说明声称当前实现了算法 6。算法 5 源自 [Atnashev/Woltman, ePrint 2021/1061](https://eprint.iacr.org/2021/1061)，其 FFT 合并操作需要与 CGBN 模乘成本分开分析。

## 测量约定

同一 GPU、N、B1、曲线数及固定 sigma 起点；Edwards 标量为 48·lcm，Montgomery 为 lcm，比较的是各曲线族实际 Stage1 工作。不同容器/TPI 逐项记录。完整进程墙钟包含进程启动、指数/曲线准备、GPU、最终 GCD/归一化及保存，不能与 kernel 计时混称。每组先预热，再正反交错重复；保留失败样本。设备功耗/时钟使用原设置，不调整硬件。测量期间不并行运行 GPU1 测试。

TODO：冻结基线与源依赖；完成测量、窗口/倍点选择、正确性门禁和正式说明。

## 2026-10-16：NAF 编码门禁

`src/core/ecm_edwards_naf.h` 以两个 32-bit 位平面保存非零位置和有符号字典索引；非零间隔至少 w，因此索引不会重叠。最高位进位独立纳入列数，不与旧二进制 bitnum 混用。

`tools/build/test/build_edwards_naf_test.ps1` 使用实际 MSVC/GMP 构建后通过 46574 组精确重建：窗口 2..12、1..4095、2 的幂及前后邻值、31/32/33 等 limb 边界与确定性随机大整数。覆盖 603364 个负数位及 126047 个跨 limb 索引。测试还逐项检查数字范围、奇数性、非相邻间隔、最高非零位和加法计数；这只证明编码正确，GPU 执行尚待验证。构建身份在 `build_cuda_cmake/edwards_naf_test/manifest.json`。

## 2026-10-16：GPU NAF 算术门禁

独立 CUDA 模块 `src/cuda/stage1/edwards_naf.cpp` 已通过 GPU1 探针：35 个用例、441 个逻辑曲线组合，其中 NAF 384 条、原二进制对照 57 条。NAF 覆盖窗口 2/3/5/6/8/12、两个倍点变体、全部七档容器（256..17408）、部分 block 的 33 曲线、完整 Mersenne 和 M127·M89 复合输入；B1 为 20 或 1000。GPU 内部编号 4 指 HWCD 四平方路径，并非 Prime95 算法 4；编号 5 指采用 Prime95 算法 5 的 2XY 多项式，但不模拟 FFT 复合指令。每条核对现有 CPU Edwards 的投影末点与 GCD、构造参数及扩展点不变量，完成态与第 17 列停止后改用 113 列切片恢复逐坐标一致。B1=1000 的窗口 6 案例实际有 1445 列，覆盖原标量 1444 位之外的 NAF 进位。

证据为 `data/experiments/edwards_naf_20261016/probe_01.log`、`probe_01/manifest.json`。探针 SHA256 为 `0e07c0f366fced6383654d4af5c8d55ee02845127ff297368825cb1c1c7b0a52`，sm89/CUDA13.3，正式性能样本数为 0。首轮构建遇到 CGBN load 不接受 const 指针，已按其接口修正；原始失败日志保留于 `compile_01/naf.log`。

生产驱动与 V1/V2 磁盘恢复仍待门禁，正式性能矩阵尚未完成。优化前基线 exe 已冻结于 `baseline_build/bin/`，SHA256=`3777416c9026027c88d2d56f78e6fbfa28a3886a7d78aa25ce9a500b66a534cb`；编译对象和扩建档位身份见 `baseline_build/identity_v2.json`，原始代码及 CGBN/GMP 头文件在 `sources_v2.zip`。M4003 的三曲线族试跑仅用于核对采集工具，不计正式性能。

## 2026-10-16：生产入口与恢复验收

当前 Stage1 测试 exe SHA256=`d853b4ce80c78853a4fdafa5f3ad3abc51cbe267cd5231c64d2a523e299f9a89`，GPU1 RTX 4060 Laptop。`validation_01.json` 的六组完整通过；补充后的 `restart_02/summary.json` 通过 18 条命令。合计采用最新恢复组的 88 条命令：流水线 29、上层位宽/发布边界 9、恢复 18、小素数 9、大界限 11、GUI 元数据 12。

- CLI、队列、完整 Stage2 叶值、保存校验、损坏主文件后的备份恢复及发布失败保留任务均通过。
- 实际停止的 V1 二进制检查点由新 worker 按原算法继续；V2 保存窗口和倍点编号，默认配置采用原设置。显式窗口或 binary/NAF 模式冲突报错且原检查点和 worktodo 不变。随机 sigma 恢复及全部末点由独立整数梯形参考核对。
- M347 的 B1=1e6/1e7、M677 的 B1=1e6 完整结果与现有 CPU NAF 投影和 GCD 一致。M347/B1=1e6 同时匹配仓库冻结的历史 Prime95 数值；没有声称重新运行了外部 Prime95 二进制。M347 两条曲线继续 B2=1e8，全部 2880 个叶值与直接乘积一致。
- 新路径保留原 13 个 CUDA 对象的逐字节身份，仅新增 NAF 模块；原对象对照见 `object_identity_02.json`。主机对象变化为选择/计时驱动及嵌入配置说明的队列配置。`bench_smoke_02/`、`bench_smoke_03/` 验证二进制/NAF、PARAM0/PARAM3、两倍点变体的采集与末点核对，不列入正式性能矩阵。

正式测量使用完整进程批次、B1=10000、sigma=20260922、GPU1。`bench_initial_01` 保留 347/677/991/2203/4003 位的 60 个有效样本；8191 位的 PARAM3 被缺失 9216 容器拦住，原报告保持 incomplete。Montgomery 预留六位进位，不能误用 8192 容器。已从冻结的优化前源码补建原算法（`baseline_native_identity.json`，exe SHA256=`cfa6c4d675a33fddca72797254c02ff2170593c2a7507d1d78c5e8000cbb5dde`），独立补测该位宽。完整矩阵、NAF 配对、窗口和倍点对照仍在采集；未将失败组标记为已完成。

## 2026-10-16：性能矩阵完成与证据审计

`benchmarks_01.json` 七组顺序采集全部exit0。加上初始五个完整范围，共192个完整批次（144正式、48预热）。六个位宽的原始Edwards/PARAM0/PARAM3基准、同一exe的binary/NAF配对、窗口6/8/10/12以及HWCD/2XY配对已完成。正式数字和条件只维护在 [Stage1性能](../performance/STAGE1.md#gpu-edwards-初始基准与-naf-对照)。

`tools/bench/analyze_edwards_gpu_stage1.py` 在 `data/benchmarks/edwards_naf_20261016/final_02/` 完成独立范围/身份/计时审计。原日志与JSON计时逐项一致；初始 incomplete 报告仅取五个完整范围，8191资格由完整补测提供。四项私有元数据损坏（缺失位宽、错误二进制摘要、修改进程计时、修改GPU区间）均拒绝，证据为 `audit_guards_01/summary.json`，未写原始证据。审计首次处理CRLF日志时匹配失败，已规范化行尾；失败输出目录 `final_01/` 保留，最终使用新目录，不覆盖。

本轮SM离散采样为1500MHz，功耗上限查询N/A；没有把历史1800MHz/55W条件用于当前结论。M2203/M4003/M8191同exe配对的完整进程中位数降低约34.6…36.2%，算术入口为1.64…1.70倍。991位小幅差异仍在样本离散内。窗口10的所测event区间较小，但存在字典显存和冷启动代价；未全局替换默认窗口6。2XY路径在大位宽没有足够证据支持全局切换，默认仍为HWCD。Prime95算法6已按源码确认废弃。

当前编译依赖、原CUDA对象身份、88条命令、46574组NAF重建、441个逻辑探针曲线及144个正式样本的完成审计均通过，冻结在 `completion_01/identity_and_completion.json`，对应exe/GMP/单元测试及原始输出同时保留。

功能和初始性能阶段均已完成；生产B1性能资格、更多设备/批次、容器细化和GUI位进度属于后续验证。此草稿保留待用户处理。

## 2026-10-16：已有实测数据可视化

`tools/bench/plot_edwards_gpu_stage1.py` 读取 `final_02/summary.json`，生成三曲线族初始成本、同一exe的binary/NAF、窗口/字典分配及HWCD/2XY四组PNG/SVG，图与解释维护在 [Stage1性能](../performance/STAGE1.md#gpu-edwards-初始基准与-naf-对照)。没有新增GPU测量，144个正式样本全部保留，48个预热仍排除；误差线为实测范围而非置信区间，嵌套计时独立展示。来源、变换、构建及图表摘要保存在 `data/benchmarks/edwards_naf_20261016/plots_01.json`，输出位于 `data/figures/edwards_naf_20261016/`。原始证据保持不变，开发草稿继续保留。

数值/绘图核对见同目录的 `plot_validation_01.json`：234个绘制的计时观测点均未被裁出坐标范围（含同批次不同计时边界），归一化结果匹配原审计，36个窗口正式样本的字典分配匹配原日志；7项损坏输入均被拒绝。8个PNG/SVG可解析并在重复生成时逐字节相同，图表链接存在，四组图已查看排版。
