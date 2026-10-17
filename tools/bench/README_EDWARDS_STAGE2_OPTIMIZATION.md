# Edwards / Montgomery PARAM0 Stage2 优化测量

`bench_edwards_stage2_optimization.py` 在同一 GPU 上比较两个生产构建和两种曲线族。每格固定实际 N、sigma、B1、实际 B2 和 D；曲线族保持各自构造，不把相同 sigma 当作同一曲线。Edwards save 复用已验证的精确输入基准，PARAM0 save 由本地 GMP-ECM 生成，输入准备排除在 Stage2 计时之外。

采集与审计使用 Python 标准库；绘图环境另外需要 Matplotlib 和 NumPy，可以与采集使用不同解释器。输出目录始终使用全新路径。

## 普通输入预热与承载验证

先只准备四个普通模数格，并完成四侧各一次声明的预热。暂不开始正式轮次：

```powershell
python tools/bench/bench_edwards_stage2_optimization.py prepare `
  --reference-study data/experiments/edwards_stage2_reference_20261017/matrix_01/plan.json `
  --baseline BASELINE/ecm_cuda_stage2.exe `
  --candidate CANDIDATE/ecm_cuda_stage2.exe `
  --gmp-ecm D:/code/GIMPS/gmp-ecm/ecm-2025.10.28-win.multiarch/ecm-zen3.exe `
  --device 1 --repeats 3 `
  --cells 1009:low:normal 10267:low:normal 10267:high:normal 16319:low:normal `
  --output data/experiments/edwards_stage2_opt_new/ordinary_warmups
python tools/bench/bench_edwards_stage2_optimization.py run `
  --output data/experiments/edwards_stage2_opt_new/ordinary_warmups --max-runs 16
python tools/test/test_stage2_carrier_projection.py `
  --study data/experiments/edwards_stage2_opt_new/ordinary_warmups `
  --output data/experiments/edwards_stage2_opt_new/carrier_proof
```

输出目录必须全新。两个构建需要匹配的 `build_manifest.json`、GMP DLL 和依赖源码；工具冻结 EXE、DLL、编译清单及源码，核对源文件哈希。参考 study 的输入 save 哈希必须一致，原 M 与已知因子必须精确恢复 N。PARAM0 准备保留原始 GMP-ECM 命令输出，失败时停止并保留证据。

承载诊断逐点比较 chain/独立 ladder，并检查精确段积、组积、设备叶及 Γ 校正；将完整下降叶投影到目标 N，与相同 N/B1/B2/D 的普通模数参考比较。两种构建都必须得到相同的目标环叶指纹。M 上的原始表示可以不同：单位性和 Γ 的逆元以 N 为准，不能要求不同投影坐标产生相同的 M 代表元。这些额外诊断的耗时不进入性能统计。

## 完整计划

用通过的承载证明准备最终六格，复用逐字节相同的 Stage1 输入，并导入已经完成的预热：

```powershell
python tools/bench/bench_edwards_stage2_optimization.py prepare `
  --reference-study data/experiments/edwards_stage2_reference_20261017/matrix_01/plan.json `
  --baseline BASELINE/ecm_cuda_stage2.exe --candidate CANDIDATE/ecm_cuda_stage2.exe `
  --gmp-ecm D:/code/GIMPS/gmp-ecm/ecm-2025.10.28-win.multiarch/ecm-zen3.exe `
  --device 1 --repeats 3 `
  --input-study data/experiments/edwards_stage2_opt_new/ordinary_warmups `
  --carrier-proof data/experiments/edwards_stage2_opt_new/carrier_proof/summary.json `
  --output data/experiments/edwards_stage2_opt_new/matrix
python tools/bench/bench_edwards_stage2_optimization.py import-warmups `
  --from-study data/experiments/edwards_stage2_opt_new/ordinary_warmups `
  --output data/experiments/edwards_stage2_opt_new/matrix
```

默认六格为 M1009/M10267/M16319 的普通余因子低档 B2、M10267 普通余因子高档 B2，以及 M10267/M16319 的低档承载。实际 N 分别为 383/10232/16272 位，B1=50000、sigma26；B2 和 D 从参考 study 的完成样本读取。每格两种曲线族、两个构建；默认 24 次预先声明的预热和 72 次正式调用，共 96 次。导入只允许正式采样前的已完成预热，逐项核对设备、二进制、输入、参数和原始文件哈希；不会导入未入账调用。原计划保持不变，复制的预热带有来源哈希。

`--cells` 可显式改变研究范围；含承载格时必须提供覆盖这些准确 N/B1/B2/D、模型和构建的独立证明，并用 `--input-study` 指定该证明使用的冻结输入计划。纯普通模数矩阵不需要承载证明。

预算固定为 arena6300/owner640/batch256 MiB，关闭短标定和运行时 tune。普通和承载使用相同指定 D，避免将重新选形混入算术路径对比；这不代表 D 是各自模式的最优选择。

## 执行与恢复

```powershell
python tools/bench/bench_edwards_stage2_optimization.py run `
  --output data/experiments/edwards_stage2_opt_new/matrix
```

每格每侧先保留一次预热，随后按轮次循环变换四个执行臂顺序。所有样本均留档，正式统计只使用声明的正式轮次。每条调用使用新的输出目录；完成一条即更新计划，可以再次运行跳过已验收的调用。存在未入账目录时停止供检查，不覆盖失败或中断的原始日志。`--max-runs N` 可在一批完成 N 条后正常返回，便于分批执行。

采集器固定自身哈希；准备后不能更改采集器继续同一计划。原始日志、结果、逐调用命令和每 200 ms 的 NVML 遥测都保留。设备显存和功率是整卡观察值，不作为进程峰值。工具不修改时钟、功率、CPU 亲和性或系统设置。

每次必须完整 NF、输入身份一致、D/B2/承载一致、`clean=1`、GMP 自检与抽查数为正且 bad=0，完整引擎时间满足 init+main。普通格的完整叶指纹跨版本/轮次相同；承载格的原始指纹须与该构建的独立诊断一致，跨版本相等性在诊断中的目标环核对。找到因子的调用保留后停止，不混入 NF 性能统计。

## 审计与图表

```powershell
python tools/test/check_edwards_stage2_optimization.py `
  --study data/experiments/edwards_stage2_opt_new/matrix `
  --output data/experiments/edwards_stage2_opt_new/audit.json
python tools/bench/plot_edwards_stage2_optimization.py `
  --study data/experiments/edwards_stage2_opt_new/matrix `
  --audit data/experiments/edwards_stage2_opt_new/audit.json `
  --output data/benchmarks/edwards_stage2_opt_new `
  --figures data/figures/edwards_stage2_opt_new
```

审计要求完整调用数、每侧正式重复数和预热数正确，所有冻结输入、承载证明、预热来源和证据哈希有效，目标环叶指纹与 S4 数学工作/检查覆盖一致。图表提供引擎时间中位数及 min/max、版本加速比，以及互斥 init/main 的均值堆叠；均值之和在日志精度内等于完整均值，不将嵌套算子时间重复相加。PNG/SVG、CSV、统计 JSON 与图表哈希清单均输出到忽略的 `data/`。

原生 `stage2_full_wall` 包含 baby 集合枚举、引擎内 NTT/S4 检查、主计算与 GCD；shape 是枚举时间，已计入 init，不能再加。它排除 Stage1、进程启动、外层 D 选型和计时前的 Montgomery/xADD 预检；进程墙钟另存，包含这些进程内工作及至多一个 200 ms 退出轮询周期。两种边界不能混用。分组图只比较实现成本，不比较曲线族找到因子的概率。

正确性入口另包括 `test_stage2_wide_tail_chain.py`：复用 `test_stage2_wide_native.py --prepare-only` 的独立 GMP 输入，覆盖阈值边界、逐点 chain/ladder、非单位分组和非单位基点；完整叶和因子集合均需与基线一致。这些诊断运行不计入正式性能统计。

当前结果和适用范围维护在 [Edwards Stage2 性能](../../docs/performance/EDWARDS_STAGE2.md)，开发过程保留在相应草稿。
