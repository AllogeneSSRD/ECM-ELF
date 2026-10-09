# Stage2 carrier / memory planning analysis

用于比较目标余因子 `N` 与梅森承载模数 `M=2^p−1` 的几何代价，以及不同D/P在
现有布局下的显存下界。下述“离线规划”不运行ECM或CUDA；“承载实验”会启动指定GPU。

研究结论与实现设计见
[研究报告](D:/code/MPA-OpenCl/docs/STAGE2_MERSENNE_CARRIER_MEMORY_PLAN_20261009.md)。

## 离线规划

在仓库根目录运行，Python 3；分析脚本使用标准库及同目录
`calibrate_stage2_d.py` 中的整数 `phi/shape` 计算。绘图需要matplotlib和numpy。

```powershell
python tools/bench/analyze_stage2_carrier_plan.py --output data/stage2_carrier_research_20261009
python tools/bench/plot_stage2_carrier_plan.py --input data/stage2_carrier_research_20261009/analysis.json --output-prefix docs/figures/stage2_carrier_plan_20261009
```

指定输入与预算：

```powershell
python tools/bench/analyze_stage2_carrier_plan.py --bits 7995 --carrier-bits 8011 --b2 2600000000000 --d 690690 810810 1021020 --free-mib 7106 --reserve-mib 768 --arena-mib 6300 --fold-mib 640 --output data/stage2_carrier_custom
```

- `--bits`：目标N位宽；默认7995。
- `--carrier-bits`：承载位宽p；默认8011，必须大于等于目标位宽，最大16384。
- `--b2`：正整数；默认2600000000000。
- `--d`：一个或多个不小于6的偶数；默认报告中的7个候选。
- `--free-mib`：默认7106，来自历史日志，**不会查询当前GPU**。
- `--reserve-mib`、`--arena-mib`、`--fold-mib`：默认768、6300、640。
- `--gpu-analysis`：可选的已完成位宽扫描分析JSON。未指定时，若本地存在
  `docs/benchmarks/stage2_n_scaling_20261008_analysis.json` 则自动使用；不存在
  仍可计算本次几何，省略跨位宽比较图。
- `--output`：结果目录，必填。

本工具接收位宽，不能验证实际整数的整除关系。生产启用承载前必须另行验证 `N|M`。

## 输出口径

`analysis.json`：输入预算、源码SHA256、NTT临界P、候选数据；`measured=false`。

`geometry.csv`：每个目标/承载位宽、D组合一行，主要字段：

- `P/I/G`：baby多项式度数、巨点数、G树数，使用现有GPU覆盖公式。
- `fold_length/packing_bpw/slot_words`：fold的NTT长度、digit位宽和每系数digit数。
- `nominal_tree_length/padded_tree_length`：名义P/2子树与补齐后实际最大子树的NTT。
- `legacy_arena_mib`：现有保守arena估计，便于与共享池比较。
- `fold_big_mib`：单次最大fold所需A/B/Q池 `24L`。
- `owner_mib/legacy_owner_mib`：当前reuse=3与默认reuse=0布局的占用。
- `raw_g_mib/coord_mib`：compact G raw A/B与当前256MiB向上取整策略的坐标chunk。
- `concurrent_lower_mib`：上面四项并存的下界，不含表、S4输出、seed等。
- `owner_budget_fits`：当前owner布局是否在fold预算内。
- `legacy_arena_fits`：旧arena估计是否在arena预算内。
- `lower_fits_free/lower_fits_after_reserve`：下界是否低于输入free及free−reserve。

`carrier_cases.csv`：仅提供/找到既有分析时生成。原D不变，比较输入N与原梅森p
的fold长度和owner变化。`ntt_length_ratio`不是速度比。

PNG/SVG：候选并存下界、G数量随P变化，以及可选的跨位宽NTT长度比。

绘图可加 `--canvas <绝对路径.canvas.tsx>` 输出自包含交互看板；全部数据内嵌，
可切换目标/承载位宽，不联网。

## 适用限制

- 同时存活下界仅建模 `I>P`、当前compact raw、resident owner reuse=3和当前巨点
  chunk策略；其它运行策略需要独立生命周期模型。
- 大池只计算最大单fold形状；若某个批量树调用的 `L×batch` 更大，实际峰值会增加。
- 下界超预算可以排除当前布局；通过预算不能证明运行时能申请成功。
- 生产 `ecm_cuda_stage2_shape_query` 仍为最终形状依据；源码策略更新后应同步公式。
- `data/`、`docs/figures/` 按仓库规则不提交，保留脚本即可重建。

## 承载实验

生产引擎新增实验命令 `--carrier-exponent <p>`。默认0；非零时验证保存点的
目标整数 `N | (2^p−1)`，用梅森模数承载设备算术，求逆/GCD继续针对保存点N。
当前需要显式B2，不能混用未校准的Auto B2。开发引擎不支持此选项。

### 独立小规模参考输入

```powershell
python tools/bench/prepare_stage2_carrier_inputs.py --output data/carrier_small_inputs
python tools/bench/bench_stage2_carrier.py --exe build_cuda_cmake/carrier_stage2/ecm_cuda_stage2.exe --save data/carrier_small_inputs/m37_cofactor.save --carrier-exponent 37 --b2 13230 --d 210 --fixtures data/carrier_small_inputs/fixtures.json --mode check --output data/carrier_m37_check
```

准备工具使用既有Python Montgomery参考实现生成有效Stage1归一化保存点，并在N域
逐点计算baby/giant及完整monic叶子摘要。M37案例包含仅在已剥离因子中不可逆的分母；
M67案例跨过64-bit limb边界；M29、M253案例包含真实非单位因子，后者满足
`gcd(N,M/N)=23`。非单位回退的X代表元不能作为一般monic叶子oracle，使用独立因子证据。
M16384案例取目标`N=2^8192+1`，覆盖256个算术limb及`p%64=0`的边界；
其24个monic叶子也有独立CPU参考。

### 固定形状正确性与计时

```powershell
python tools/bench/bench_stage2_carrier.py --exe build_cuda_cmake/carrier_stage2/ecm_cuda_stage2.exe --save <stage1.save> --carrier-exponent 8011 --b2 26000000000 --d 180180 --mode check --output data/carrier_check
python tools/bench/bench_stage2_carrier.py --exe build_cuda_cmake/carrier_stage2/ecm_cuda_stage2.exe --save <stage1.save> --carrier-exponent 8011 --b2 2600000000000 --d 810810 --mode timing --telemetry --output data/carrier_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/carrier_timing/measurements.json --output docs/benchmarks/carrier_timing_analysis.json --figure-prefix docs/figures/carrier_timing
```

- 同一二进制分别传0与p；N、存档、B2、D、预算和其它策略相同。
- `check`开启baby、seed、segment和chain检查，比较所有叶子投影到N后的摘要。
  开启了额外检查的耗时不属于正式计时。
- `check --projection-only`仅增加完整目标叶子摘要，保留强制算术检查，省去逐点
  ladder/seed诊断；适合在正式计时规模上补充跨后端对照，仍不计为性能样本。
- `--fixtures`可增加独立CPU叶子oracle或已知非单位因子检查。
- `timing`先各预热1遍，再执行ABBA+BAAB，共8个正式样本，每条路径n=4。
  关闭额外投影与逐点检查，保留生产强制算术检查。全部慢样本保留。
- `--telemetry`每2秒只读查询选定GPU的功耗、频率、温度与利用率，不修改设备设置。
- 默认GPU1；arena=6300 MiB、fold=640 MiB、batch=256 MiB，可用同名参数调整。
  `batch`是S4输出分块预算；巨点坐标chunk的256 MiB目标是另一项独立策略。
- 输出目录必须为空；保存原始可读日志、独立调试日志、结果、命令、环境和SHA256。
  二进制及编译源闭包会冻结到构建目录。不要在运行中修改采集器或重编译该二进制。
- `--resume-check`只用于采集器解析失败后的恢复：重读原始成功调用，保留原采集器；
  不允许重用失败曲线，不允许恢复/筛选正式计时矩阵。
- 分析器只接受完整ABBA+BAAB矩阵，输出均值、标准差、范围、父阶段和GPU采样。
  S4 `t_reduce`单列为嵌套事件；workspace/owner峰值单列，不相加冒充进程显存峰值。
