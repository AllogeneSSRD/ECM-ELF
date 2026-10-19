# Edwards GPU Stage2 relocation 测量

算法和当前限制的权威说明在 [Stage2 架构](../../docs/architecture/STAGE2.md#可选-relocation)，冻结实验结论在 [relocation 草稿](../../docs/development/GPU_STAGE2_RELOCATION_DRAFT.md)。本工具比较同一生产二进制的原算法与默认关闭的 relocation；只增加 Edwards 测量。

## 正确性与冻结

使用 `tools/build/test/build_edwards_gpu_workers.ps1 -Stage stage2 -Build <候选目录> -SplitCompile 1` 构建，再以 `tools/build/test/build_stage2_relocation_probe.ps1 -Build <候选目录>/stage2` 链接同一生产引擎对象。SplitCompile 可按环境选择；若当前工具链出现 ptxas 内部 INVALID_HANDLE，可用1重建，并保留失败日志。

运行 `tools/test/test_edwards_stage2_relocation.py --exe <Stage2程序> --probe <探针> --output <全新证据目录> --wide-reference <既有输入计划>`。门禁包括纯整数几何、独立仿射完整叶、配置覆盖、单G/非满尾、普通/承载、真实因子、精确 gcd=N 反例、宽位数 chain 逐点和短尾重分。它还调用 `test_stage2_relocation_baby_zero.py`，以有效 Edwards d=−1 点验证目标零 baby 坐标：正常设备和强制分配失败的主机回退均须跳过 probe，原区间最终叶一致。诊断开关遵守自身支持范围；这些运行不计入性能。

`bench_edwards_stage2_relocation.py prepare --exe <程序> --gates <门禁summary.json> --reference <旧优化矩阵plan.json> --output <全新研究目录> --device 1 --repeats 3` 冻结生产 EXE、GMP、81份编译依赖、清单、采集器及辅助脚本。要求门禁与 EXE 哈希匹配。四份 NF 输入沿用既有 Edwards 精确余因子；另构造两份独立 CPU Stage1 末点。试算命中样本的小因子剩余点阶由独立有限域 BSGS 确认，并大于 baby 范围及 giant 索引终点，避免把它当成 baby 非单位样本。

## 采样、审计与绘图

1. `bench_edwards_stage2_relocation.py run --study <研究目录>`：固定 D、普通模数、arena6300/owner640/batch256 MiB、factor-only、关闭短标定和 runtime tune、开启 NTT 次数表。清除继承的 `NTT_*` 环境变量，不启用诊断；仅设置 CUDA 非阻塞启动。每格两侧一次预热、三次正式运行，交替侧顺序，保留所有48次调用。原始日志、私有双结果、账本、命令和整卡200ms遥测按调用保存，禁止覆盖。
2. `tools/test/check_edwards_stage2_relocation.py --study <研究目录> --output <全新审计JSON>`：独立核对全套哈希、样本覆盖、计时边界、算术检查、叶重复一致性、因子整除与跨模式一致性、重放最终范围及 NTT 表总数。NF 两种算法的叶本来不同，不要求跨模式相等；F 重放最终叶必须等于原算法。
3. `plot_edwards_stage2_relocation.py --study <研究目录> --audit <审计JSON> --output data/benchmarks/<目录> --figures data/figures/<目录>`：使用 Matplotlib，生成中位数/min-max图、CSV和统计JSON；要求审计与研究哈希匹配。

采样期间可对检查器加 `--partial` 提前核对已完整入账的调用；输出始终 `complete=false`，不能作为绘图门禁。正式完成后必须另跑不带该参数的全矩阵审计。

主指标是 `stage2_full_wall.total=init+main`，包含共享 baby/F 初始化、既有自检、probe、必要重放、GCD 和检查收尾；排除 Stage1、进程启动、外层选 D 和计时前预检。shape 已在 init 中；probe/replay 已在 main 中，不重复相加。进程墙钟另存，退出轮询最多多计200ms。NTT计数包括实际probe与replay。遥测是整卡观察值，不表示进程显存峰值。

输入、失败试验、证据、图表及统计均在忽略的 `data/`。采集器或输入变化时准备全新研究目录；不删除失败调用或选择性移除慢样本。恢复仅跳过已经完整入账且哈希可核对的调用；发现未入账目录时停止供核查，不自动覆盖。
