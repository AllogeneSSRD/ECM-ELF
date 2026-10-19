# 数字版本 Stage2 全范围复测

`bench_stage2_version.py` 复用已验收的 Edwards / Prime95 精确输入矩阵，测 GPU1 的普通余因子与梅森承载。默认 13 个输入、两档实际 B2、两条 GPU 路径、各三次，共 156 次。CPU 和 Montgomery 不重跑。当前结果见 [Stage2 性能](../../docs/performance/STAGE2.md)，输入与历史参考见 [参考页](../../docs/performance/EDWARDS_STAGE2.md)。

## 构建与测量

使用正式 local 构建入口，测量前完成编译；不要同时运行本任务的 GPU 诊断：

```powershell
tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/local_stage2_0_38_4 `
  -Arch sm_89 -Engine production -SplitCompile 8
python tools/bench/bench_stage2_version.py `
  --exe build_cuda_cmake/local_stage2_0_38_4/ecm_cuda_stage2.exe `
  --reference data/experiments/edwards_stage2_reference_20261017/matrix_01 `
  --version 0.38.4 --output data/experiments/stage2_v0_38_4_20261018/full_matrix
```

新输出目录必须不存在；参考目录须含已完成准备的 `plan.json`、13 份已独立验证的 save、26 个 Edwards Prime95 条件。工具复制 EXE/GMP、输入及采集器，核对精确余因子整除、save 状态和校验。程序在每次结果中报告的数字版本必须与 `--version` 一致。新增研究元数据和图注使用版本号，历史未记录数字版本的程序明确标注未知，不补造版本。

设备固定为 GPU1，操作者须确认它仍为 RTX 4060 Laptop。工具记录实际设备名称，不改变时钟、功耗、优先级或 GPU0 的工作。预算为 arena6300/fold owner640/batch256 MiB；D 自动选择，短标定/runtime tune/自动承载/relocation/NTT 计数关闭。每轮遍历全部精确条件，偶数轮交换普通/承载顺序；每次为独立进程，没有丢弃的预热或首轮。所有调用保留正常算术检查。

主计时为 `stage2_full_wall.total=init+main`，包括引擎内必需检查和 GCD，排除 Stage1、进程启动、外层选 D 及计时前预检。进程墙钟另存，250 ms 轮询包含最多一周期的退出观察延迟。NVML 观测整张选中显卡，不代表进程峰值或纯内核功耗。

## 续跑与审计

```powershell
python tools/bench/bench_stage2_version.py --resume --version 0.38.4 `
  --output data/experiments/stage2_v0_38_4_20261018/full_matrix
python tools/test/check_stage2_version.py `
  --study data/experiments/stage2_v0_38_4_20261018/full_matrix `
  --output data/experiments/stage2_v0_38_4_20261018/full_matrix/audit.json
```

计划在每条接受后原子保存。`--max-runs N` 可在完成 N 条新调用后正常停下，样本保留且矩阵仍未完成；续跑跳过已接受 ID。采集器改变、已有失败、存在未入账调用日志时拒绝续跑，须先调查并保留证据，不能覆盖原文件后补成成功样本。改变版本、输入或协议应使用新目录。

正式审计要求 156 个唯一 ID、52 格各三次，重新核对输入/save、结果版本和身份、实际 B2/D/P、φ(D)/2、完整区间点数与 G 批数、NF 状态、原始计时守恒、强制 GMP 检查、分项记录和调用设置。提前出因子的运行保留，但不满足完整 NF 时间矩阵。该审计不声称提供每个叶值的独立 oracle；正常 GMP 抽样与独立末点验证各自有明确范围。

## 汇总、绘图与公开副本

```powershell
python tools/bench/plot_stage2_version.py `
  --study data/experiments/stage2_v0_38_4_20261018/full_matrix `
  --output data/benchmarks/stage2_v0_38_4_20261018/final `
  --figures data/figures/stage2_v0_38_4_20261018/final `
  --publish docs/performance/attachments
```

需要 Matplotlib 和 NumPy。新统计/图目录必须不存在；绘图先严格审计当前矩阵，逐格核对旧 Edwards 参考的身份，再生成 CSV/JSON 与 PNG/SVG。绝对耗时和 NTT 模块显存图以相同数据同时生成 `-log`、`-linear` 两版（含 PNG/SVG），线性纵轴从0开始；百分比图保持线性坐标。当前 GPU 使用中位数及 min/max，版本比较阴影组合两侧观测范围，不是置信区间。init/main 按逐调用比例取均值；NTT 模块峰单列，不加组件峰、不当作进程峰。

同一工具重绘已验收历史参考：Prime95 31.4b05 全部两曲线族、32 条 libecm 样本、CPU GW 0.33.0 的 36 次实测，以及 Montgomery 55 W 及两种频率设置下的三份 9 位宽×3 B2×3 次矩阵。不将缺测 CPU 单元插值，不按频率、B2 或线程数归一化。历史参考路径集中在绘图脚本顶部，原始研究不改写。`--references-only` 可在等待 GPU 时单独重绘这些参考，不能生成当前版本完整性能结论。

`--publish` 只复制选定 PNG 到指定 attachments 目录；原始结果、账户字段、运行校验值、EXE、遥测和全部数据只在忽略的 data 目录。公开提交只包含工具、文档和审查过的图。
