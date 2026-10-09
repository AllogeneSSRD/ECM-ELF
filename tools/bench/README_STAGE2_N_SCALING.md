# Stage2 位宽 / B2 测量工具

以数据库中的梅森指数和已知因子为输入，测量实际余因子的完整 Stage2 时间。数据库只读；实验记录写入指定目录，不增加数据库运行历史。

## 环境

- Python 3；测量脚本使用标准库。分析需要 NumPy，绘图需要 Matplotlib。
- 与当前源码一致的生产 Stage2 程序，以及它旁边的 `build_manifest.json`、`gmp-10.dll`。
- GMP-ECM：用于独立校验 Stage1 点，路径可用 `--gmp-ecm` 指定。
- runner 限定 RTX 4060 Laptop / `sm_89`，通过 `--device` 显式选卡，默认1。移除其他显卡后4060可能变为GPU0，应指定 `--device 0`；准备和续跑记录并核对UUID及设备编号，不能仅凭旧编号选卡。

## 1. 构建与准备

仓库根目录执行，输出目录必须为新目录：

```powershell
.\tools\build\build_stage2_local.bat -Arch sm_89 -Engine production `
  -Build build_cuda_cmake/n_scaling/native

python tools/bench/bench_stage2_n_scaling.py `
  --exe build_cuda_cmake/n_scaling/native/ecm_cuda_stage2.exe `
  --output data/experiments/n_scaling_study --full-controls --prepare-only
```

默认指数 `503 1009 2003 3001 4001 5003 6011 7001 8011`；默认 B2 为 `26000000000 260000000000 2600000000000`，即 `260e8 260e9 260e10`。默认 B1=20，每个余因子/B2 测三遍。可用 `--exponents`、`--b2`、`--b1`、`--repeats` 覆盖。

`--full-controls` 为每个完整梅森数/B2 加一次对照；它与余因子使用相同 B1、sigma。sigma 从26开始，寻找能在完整梅森数上正常完成 Stage1 的点。小因子可能在较大的 B1 下必然提前检出，这时应减小 B1，不应把已剥离的数冒充完整梅森数对照。

保存点由 Python 参考与 GMP-ECM 独立计算，要求实际 N、归一化 X 完全一致，再写 native checksum。GMP-ECM 调用明确使用 `B2=0`：`B2=B1` 仍可能因 Stage2 区间取整检出小因子并改写 save 的 N，不能作为“禁用 Stage2”。Stage1 生成、校验和源码快照均不计入 Stage2 时间。

准备结果：

- `database_rows.json`：本轮只读提取的数据库核心记录。
- `inputs/`：每个输入的 GMP-ECM 原始 save、日志和 Stage2 save。
- `bin/`、`sources/`：冻结的 exe、DLL、构建身份和源码。
- `measurements.json`：冻结计划及逐条测量结果。
- `commands.json`、`stage2_commands.ps1`：全部显式 Stage2 命令，便于人工复现。手动脚本仅在全新 `manual/` 输出目录执行一次，避免向旧结果文件重复追加。

## 2. 执行和继续

```powershell
python tools/bench/bench_stage2_n_scaling.py --output data/experiments/n_scaling_study --resume
```

`--resume` 使用原计划，不重新读取数据库，也不使用新传入的 B1/B2、预算和指数列表覆盖它；改变实验设置应重新准备新目录。二进制、DLL、INI、保存点哈希变化时拒绝继续。

预算默认 `--arena-mb 6300 --fold-mb 640 --batch-mb 256`，首次准备时可修改。D=0 保留当前生产程序的自动选 D 行为；每条实际 D/P/G、归约路径、NTT 容量都会记录。arena 不是全进程显存硬上限，各模块的容量峰不能相加当作同时峰值。

执行串行，不与编译、profiler 或重型 CPU 分析并行。每个位宽先跑一次最小 B2 预热，预热单独保留。正式三轮按正序/逆序/正序遍历，完整梅森对照插入第二轮。108条正式数据之外还有9条预热。每条使用独立子进程，保留默认必需算术检查；`--factor-only` 关闭素数命名和 GP 因子拆解，仍做叶乘积/GCD及因子合法性验证。

主时间为 `stage2_full_wall.total=init+main`。日志 `shape` 是 baby 索引构造时间，已经包含在 init 中，不能重复相加；真正的自动选 D 扫描由 `d_scan_wall` 记录，不包含在该 total 中。外层进程墙钟另记。NVML每约250ms记录整张 GPU 的利用率、SM时钟、温度、功率、显存观测值；这些不是纯 kernel 时间，也不是精确的进程显存峰，短曲线的采样尤其有限。

`runs/` 保留正常日志、调试日志、完整 stdout/stderr、result JSONL 和 GPU 采样。失败记录不进入均值；重跑时保留原始失败文件。Ctrl+C中断后先确认本工具启动的曲线进程已结束，再继续，避免并发占用 GPU。

## 3. 分析与绘图

全部完成后执行：

```powershell
python tools/bench/analyze_stage2_n_scaling.py `
  --study data/experiments/n_scaling_study --output data/benchmarks/stage2_n_scaling

python tools/bench/plot_stage2_n_scaling.py `
  --summary data/benchmarks/stage2_n_scaling_summary.csv `
  --output data/figures/stage2_n_scaling --tick 1000
```

分析输出：逐遍 `_runs.csv`、按输入/B2分组的 `_summary.csv`、包含公式拟合与原始文件哈希的 `_analysis.json`。均值、样本标准差、CV、最小/最大值全部保留；单次对照的标准差留空。重复之间的 D、叶指纹、因子和工作量不一致会列为警告，不会被静默删除。

绘图默认横轴为实际 N bits，1000 bits一格，纵轴为秒；B2各一条线。生成 `time`（余因子均值及观测范围）、`controls`（完整梅森单次对照）、`reduction`（嵌套的S4设备归约事件耗时），每张同时输出PNG/SVG。`--x exponent` 可改为原梅森指数，但不能将它称为余因子的实际位宽。

另外生成 `phases_percent`（余因子均值）与 `phases_percent_controls`（完整梅森单次对照）的100%堆积柱图，每图按B2分面。柱间距是类别间距，标签为实际N bits；这样不会让318/383 bits两个点重叠。百分比使用“阶段秒数均值/完整时间均值”，不是逐遍百分比的简单均值。

互斥分区为 baby点/归一化、F树及其余初始化/检查、inverse、giant点、G树、fold、G叶准备/回退、下降、叶乘积/GCD、其余记账残差。baby秒数取 `real_baby.ladder+affine`；F树及初始化项取 `init-baby`；其他项补齐完整墙钟。它们不是纯GPU kernel分区，G叶项包含设备准备、CPU和传输。`ntt_seconds`和`s4.t_reduce`嵌套在这些阶段内部，不能加入100%堆积图。毫秒级日志取整限制了极短阶段的百分比精度。

按现有日志边界，设备叶填充及patch上传在G树构建回调内，计入G树；`gleaves`记录构树之前的驻留准备、归一化及非单位回退。图中的模块时间不应重新按操作名称归类后再相加。

## 4. 对比已有完整扫描

准备新目录后，先核对新旧 `measurements.json` 的输入整数、B1、sigma、save/Q 哈希、预算、DLL 和运行顺序。原始实验无需 Git 跟踪，必须保留；旧数据不重新生成或覆盖。绘图和分析沿用本页脚本。

```powershell
python tools/bench/analyze_stage2_n_scaling.py `
  --study data/experiments/new_study --output data/benchmarks/new_study `
  --baseline-analysis data/benchmarks/previous_study_analysis.json

python tools/bench/plot_stage2_n_scaling.py `
  --summary data/benchmarks/new_study_summary.csv `
  --output data/figures/new_study --tick 1000 `
  --version-comparison data/benchmarks/new_study_comparison.json `
  --cpu-comparison data/experiments/prime95_comparison_current/comparison.json `
  --previous-label "GPU previous (55 W)" --current-label "GPU current (1800 MHz)"
```

`--baseline-analysis` 要求两轮全部完成、每格样本数相同、N/B1/sigma/save/Q 与请求预算一致，GPU名称和UUID相同，输出 `_comparison.{json,csv}`。重启或移除其他显卡后允许同一UUID的设备编号改变，两个编号独立保留。D/P/G 可随实现改变，独立记录；因子比较忽略输出次序。时间变化为 `100·(current/previous−1)%`，速度比为 `previous/current`。同时记录模块容量和阶段秒数差额。

`--version-comparison` 生成 `_versions_times`、`_versions_speedup`、`_versions_clocks`、`_versions_phase_delta` 的 PNG/SVG。阶段差额按互斥墙钟分区，正值表示增加、负值表示减少；所有分区之和必须等于完整时间差。可选 `--cpu-comparison` 使用 [Prime95 比较器](../log_parser/README_PRIME95_ECM_BENCH.md) 对当前 GPU 分析生成的精确 N 配对，仅画匹配点。CPU 未匹配目标不进入速度比。

输入及预算相同不代表频率、功率或后台负载相同。必须依据原始遥测与实际设置注明约束；固定55W与固定1800MHz属于不同条件，耗时差不能直接归因于软件，不按频率或功率比例修正。

拟合 `T=C(S/1000)^alpha(B2/2.6e10)^beta`，另按B2独立拟合N指数；报告数据内误差和按原exponent整组留出的误差。默认另拟合 `S>=1000` 子集，避免固定启动开销主导最小两个点。公式只是指定硬件、版本、预算和测量区间内的经验近似；NTT长度、自动D、owner回退及模数类型改变时，不能无条件外推。
