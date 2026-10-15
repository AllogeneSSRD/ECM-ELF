# NTT outer 计划对照

使用生产运行时、PTX3 Goldilocks 归约及 canonical 减法，在同一探针中比较默认自动表和强制 M5–M8。完整 Stage2 对照使用生产引擎，固定 save、B2、D 和预算。当前结论见[outer 性能](../../docs/performance/NTT_OUTER.md)。

## 构建和精确门禁

构建入口：`tools/build/test/build_ntt_outer_plan_probe.ps1 -Build <独立目录> -Arch sm_89`。记录二进制、源码、CUDA、MSVC、GMP 身份，构建不运行测试。

探针入口：

- `<exe> <device> --check`：独立 GMP 正向和逆向检查，包含 batch、M5–M8、compact、缓存和驱逐；另验证 Goldilocks 算术。
- `<exe> <device> --fault`：故意翻转一个结果，要求 bad=1、退出码3。
- `<exe> <device> --policy`：检查中段开关、k 边界、tile、warp、compact、原 radix 覆盖及强制/自动模式，要求672项 bad=0。
- `<exe> <device> --bench <k> <batch> <repeats> <iterations> <operation>`：交错五种计划；operation=0/1/2 分别为正向、融合逆向、完整卷积。mode=0 为自动默认，mode=1/2/3/4 为强制 M5/6/7/8。
- `--profile` 使用相同位置参数，末尾增加 mode，只捕获指定计划；cudaProfilerStart/Stop 包围计算，所有输出仍需通过检查。剖析时使用 repeats=1；不将剖析时长计入性能。

`benchmark_ntt_outer_plans.py --exe <exe> --output <新目录> --device 1` 先运行独立门禁与故障注入，再测 k20–24、batch1/4。默认每形状、每模式10个样本，每样本串联8次卷积；不丢弃 warmup。正式时间排除输入填充、初始化、根表构建和输出校验。每样本最终全部 A/B 与同串联次数的默认结果比较；该对照不代替独立 GMP 门禁。

可用 `--lengths`、`--batches`、`--operations`、`--repeats`、`--iterations` 缩小范围。工具冻结全部构建源并保存原始输出、GPU采样和统计，输出目录须为新目录。

## 完整 Stage2

`benchmark_stage2_graph.py --variant mid-outer --exe <Stage2 exe> --cases <cases.json> --output <新目录> --device 1 --repeats 4`。

该复用工具仍以 Graph 命名；`mid-outer` 模式关闭 Graph，对 `NTT_FUSE_MID_COOP=0/1` 做同二进制交错配对。cases 为对象数组，字段为 `name`、`save`、`b2`、`d`、`batch_mb`，可选 `arena_mb=6300`、`fold_mb=640`、`carrier=0`。固定 D，关闭短校准、运行数据写回和短 batch tune。

逐条核对 GMP、自检、因子、root fold 指纹、数学 NTT 数、长度分布和 S4 分块一致。`--diagnostic` 每侧各一次，另核对全部目标 N 叶子指纹；这些运行的 clean=0，不进入性能汇总。`mid-outer` 的有效性还要求至少一个输入含 k21–23；该策略只在支持设备、tile12、warp 和 compact 条件下生效。

管理员 Systems/Compute 采集继续使用 `profile_stage2_graph.py --reference <已完成 command.json> --output <新目录> --kind systems|compute --run-admin`；Compute 的 `--kernel` 取自 Systems 精确形状。保持 `--clock-control none --cache-control none`，不修改设备设置。

## 分析

`analyze_ntt_outer_plans.py --operator <算子 summary.json> --stage2 <完整曲线 summary.json> --output data/benchmarks/<名称> --figure data/figures/<名称>.png`。

汇总输出每格中位数/MAD、NTT模块峰值、两侧完整曲线样本数和各格中位数之和。后者只评价所选矩阵，不是一个生产任务的时长。算子吞吐与完整曲线分别绘图；组件峰不能作为进程显存峰。
