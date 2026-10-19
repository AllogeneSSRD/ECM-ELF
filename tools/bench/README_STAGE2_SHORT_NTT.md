# 短 NTT 分块 A/B

用于验证实验开关 `--short-ntt-batch-tune` 的完整Stage2效果。冻结实验的输入和统计边界见[短NTT草稿](../../docs/STAGE2_SHORT_NTT_BATCH_DRAFT.md)，当前全范围结果见[Stage2性能](../../docs/performance/STAGE2.md)。工具不会调整显卡功耗或频率。

## 输入

`benchmark_stage2_short_ntt.py` 接受 `--exe`、`--profile`、`--cases`、`--output`、`--device`。`--repeats`默认3，`--warmups`默认1，单次`--timeout`默认600秒。输出必须是新目录，避免覆盖证据。

cases为JSON数组，每项包含：

- `name`：唯一文件名标签；使用字母、数字和下划线。
- `save`：有效单条Stage1保存点的路径，可相对仓库根目录。
- `b2`、`d`：两侧固定的Stage2边界和D。
- `batch_mb`：实际分块预算MiB。
- `fold_mb`：设备owner预算MiB，可省略，默认640；0用于回退对照。

两侧固定arena自动预算、carrier-exponent=0，禁止启动短校准。开关为off/on，同一输入、同一二进制交替运行。为复用NTT表到不同分块预算，工具显式忽略profile的memory条件；当前预算和显存准入仍执行，结果不能作为忽略显存限制的授权。

## 输出与检查

输出包含冻结exe/DLL/manifest/源码/工具/输入、各次命令与结果、可读完整日志、500ms显卡遥测、`runs.jsonl`、`summary.json`和`summary.csv`。性能使用引擎total中位数，同时记录result与进程墙钟、MAD、分阶段时间、归约、分块及NTT组件峰；预热不进入中位数。

每次要求完成曲线、clean=1、GMP自检及抽查无错误；A/B检查因子集合、G根摘要、数学NTT长度计数和卷积数守恒。开关开启必须加载到有效NTT记录。changed=0表示没有改变分块，时间差不能归因于分块优化。

发生失败立即退出并保留日志；没有完整summary不能声明A/B通过。程序源码必须与构建manifest一致。长耗时进程不应与本GPU的其他计算同时运行；其他显卡任务及CPU竞争也应记录。

## 附加算术/计划门禁

`tools/test/test_stage2_short_ntt_gpu.py` 接受 `--experiment`、`--exe`、`--output`及`--cases <name> ...`，重放上述冻结命令。

仅选择满足驻留请求模型的输入。门禁先读取native计划，再运行独立诊断，比较稠密拓扑、五阶段请求/分块/容量、请求与S4顺序签名、NTT保留上界及off/on全叶子指纹。它开启额外leaf fixture，预期clean=0，诊断时间不进入性能结果；GMP门限保持不变。

CPU策略边界检查包含在`tools/test/test_stage2_portable.py`。`tools/test/test_stage2_ntt_sort.py`接受`--profile`、该portable门禁编译出的`--fixture-exe`、`--output`，测试真实表的数值排序与值保留。

## 汇总与图

`analyze_stage2_short_ntt.py` 接受重复的`--experiment`、`--ntt-profile`、`--output`、`--figure`。推荐汇总放`data/benchmarks/`，PNG放`data/figures/`。

生成完整耗时/收益、纯NTT Conv/s和S4分块数对比，时间误差线为MAD；同时导出含遥测的汇总和逐NTT长度计数。逻辑蝶形工作量不是GPU时间占比；NTT组件峰不能与其他模块峰相加作为进程峰。
