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
- `--workspace-buffers [2|3]`：共享大池的物理缓冲数，默认3；导出与回退仍需3份。
- `--baby-mib`：GPU baby独立预算，默认512；输出其准确临时payload与预算是否足够。
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
- `legacy_arena_mib`：三缓冲时代的保守arena估计，便于与共享池比较；不随本工具的布局选项改变。
- `fold_big_mib`：单次最大fold所需物理大池 `8×buffers×L`。
- `baby_mib/baby_budget_fits`：GPU baby临时payload与独立预算判断；不加入主循环并存下界。
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

## 两缓冲工作区实验

实验环境变量`NTT_WORKSPACE_REUSE_BQ=1`令共享大池的Q在inverse完成后复用B；
默认0。只作用于允许pool且不导出digit指针的调用。导出接口、关闭pool和每调用
分配的回退路径保留三个独立缓冲。分配器、原生几何和fold/frontier headroom已共用
物理缓冲数量策略；完整生命周期MemoryPlan和Auto B2新成本尚未接入，测试需要
显式D/B2。Auto B2拒绝未经校准的两缓冲配置。

在**同一二进制、同一算术后端**上比较环境开关0/1：

```powershell
python tools/bench/bench_stage2_carrier.py --comparison workspace-bq --exe build_cuda_cmake/workspace_bq_stage2/ecm_cuda_stage2.exe --save data/carrier_small_inputs/m37_cofactor.save --carrier-exponent 37 --b2 13230 --d 210 --fixtures data/carrier_small_inputs/fixtures.json --workspace-fixture --mode check --output data/bq_m37_check
python tools/bench/bench_stage2_carrier.py --comparison workspace-bq --exe build_cuda_cmake/workspace_bq_stage2/ecm_cuda_stage2.exe --save <stage1.save> --carrier-exponent 8011 --b2 2600000000000 --d 810810 --mode check --projection-only --telemetry --output data/bq_large_check
python tools/bench/bench_stage2_carrier.py --comparison workspace-bq --exe build_cuda_cmake/workspace_bq_stage2/ecm_cuda_stage2.exe --save <stage1.save> --carrier-exponent 8011 --b2 260000000000 --d 810810 --mode timing --telemetry --output data/bq_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/bq_timing/measurements.json --output docs/benchmarks/bq_analysis.json --figure-prefix docs/figures/bq_analysis
```

- `--carrier-exponent`在两条路径中相同；省略或0代表通用N算术。
- `--workspace-fixture`只用于check，验证两种布局与pool开/关的容量、失败回滚、
  交叉A/B/Q输入、导出生命周期、每调用回退、dRes隔离及延迟carry。
- `--single-arm two_buffer`只允许check，适合三缓冲可能超显存时探索更大D。
  输出记录这是单路径实验，不能作为两布局A/B性能结论。
- `--require-resident`要求fold、scaled root、frontier都实际启用；任何回退拒绝矩阵。
  完成曲线的原始日志、返回值和已解析结果仍保留，`complete=false`不算通过。
- `--baby-mb`默认512，是baby生成的独立临时显存预算，和fold/arena/batch分开设置。
- `--trim-phase-raw`在两条路径中同时设置`NTT_PHASE_TRIM_RAW=1`，在Newton完成后、
  驻留下降前释放已失效的raw A/B。默认关闭；不缩小1 GiB headroom余量，必要回退可
  重新申请raw。日志单列释放字节和耗时，旧Auto B2成本不接受该策略。
- 正式timing仍为各一次预热加ABBA+BAAB；两布局的乘法、归约和强制检查覆盖必须相同。
- 新增`ntt_workspace_layout`记录容量、物理缓冲数、复用调用与省去的Q峰值字节。
- 分析器兼容先前carrier矩阵。两缓冲模式额外绘制`*_memory.png/svg`，分别展示
  big、完整NTT模块容量峰值和每2秒采样的设备已用显存；三项互相包含，不能相加。
  采样的最大值也不等于精确的进程分配峰值。

### 原生规划检查与候选D计时

```powershell
python tools/test/test_stage2_workspace_plan.py --exe build_cuda_cmake/workspace_bq_headroom_v2_stage2/ecm_cuda_stage2.exe --save <stage1.save> --carrier-exponent 8011 --output data/bq_plan_check
python tools/test/test_stage2_phase_trim.py --exe build_cuda_cmake/workspace_bq_phase_trim_stage2/ecm_cuda_stage2.exe --reference-check data/stage2_bq_20261009/phase_trim_m37_check/measurements.json --output data/phase_trim_fallback_gate
python tools/bench/bench_stage2_carrier.py --comparison workspace-bq --exe <ecm_cuda_stage2.exe> --save <stage1.save> --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --mode check --projection-only --single-arm two_buffer --output data/d1381380_check
python tools/bench/bench_stage2_carrier.py --comparison plan --exe <ecm_cuda_stage2.exe> --save <stage1.save> --carrier-exponent 8011 --b2 2600000000000 --d 810810 --candidate-d 1381380 --fold-mb 1024 --mode timing --telemetry --output data/d_plan_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/d_plan_timing/measurements.json --output docs/benchmarks/d_plan_analysis.json --figure-prefix docs/figures/d_plan
```

带阶段释放的最终实验构建为`build_cuda_cmake/workspace_bq_phase_trim_stage2/`；
需要验证此策略时，在check与timing命令中均添加`--trim-phase-raw`，并保持其它预算一致。
`test_stage2_phase_trim.py`要求同二进制已通过的workspace check，且含独立单位
案例CPU oracle并开启阶段释放；分别强制fold/frontier分配失败，验证完整目标
叶子与回退重申请路径。它执行真实小曲线，与只查计划的workspace_plan工具不同。

`plan`只接受timing：固定同一二进制、目标N、carrier、保存点、B2及所有预算，
两臂均使用两缓冲pool，仅切换D；每臂预热一次，正式ABBA+BAAB，各n=4。调用前
分别完成每个D的独立正确性检查。不同D的baby集合不同，不能要求叶子摘要相等，
也不要求跨D乘法计数相同；每个固定D内部的工作与强制检查覆盖必须一致。
允许并记录实际驻留回退；需要全部驻留时显式增加`--require-resident`。

## 物理工作区分块实验

`NTT_S4_WORKSPACE_BUDGET=1`将S4分块的NTT请求预算从固定三缓冲改为分配器的
实际两/三缓冲数量，并计入每slice两个carry诊断字。默认0，保持原分块序列。
它不改变`batch_mb`为全进程显存限制；NTT表缓存、保留容量、S4归约输出、raw、
坐标与owner仍分别占显存。一块也超预算时仍执行一块，日志明确记录。
pool关闭、无arena时按三个缓冲；pool申请失败仍保留三缓冲每调用回退，
`request_peak_bytes`是请求策略的名义量，不代表失败回退的实际瞬时分配量。

```powershell
python tools/bench/bench_stage2_carrier.py --comparison chunk --exe build_cuda_cmake/workspace_chunk_stage2/ecm_cuda_stage2.exe --save <stage1.save> --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode check --projection-only --output data/chunk_check
python tools/bench/bench_stage2_carrier.py --comparison chunk --exe build_cuda_cmake/workspace_chunk_stage2/ecm_cuda_stage2.exe --save <stage1.save> --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode timing --telemetry --output data/chunk_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/chunk_timing/measurements.json --output docs/benchmarks/chunk_analysis.json --figure-prefix docs/figures/chunk_analysis
python tools/test/test_stage2_chunk_budget.py --exe <ecm_cuda_stage2.exe> --reference-check <completed_m37_chunk_check/measurements.json> --output data/chunk_controls
```

- `chunk`两臂均启用B/Q复用，同D、同carrier、同预算；仅分块开关变化。
- 两臂数学工作量`poly_muls/coeffs_reduced`和selftest覆盖必须相同；归约launch、
  GMP抽样数因每调用抽样规则可能变化，各臂内部必须稳定，原始计数分别保留。
  不关闭强制检查来制造加速；check另对全部目标叶子做投影对照。
- `s4_chunk_plan.chunks`是外层分块的NTT子调用数；采集器另对各形状
  `s4_reduce_stats.launches`求和得到`reduce_hook_calls`，记录真实归约hook调用数。
  `s4_multiply_stats.launches`与`real_batched_breakdown.ntt_launches`都是父批次
  调用数，不是kernel数；内部grid-y切分可能令hook数多于外层分块数。
- `owned_subset_observed_peak_bytes`是在每个完成的NTT调用边界，读取实际保留的
  arena（含fuse base）和S4 raw/pack/output容量所得的同时存活子集峰值；不是模块
  峰值求和。它不覆盖临时每调用分配、reduction常量/oracle、点/树/owner等；
  `process_peak_complete=0`，不可作为总进程峰值或显存可行性保证。
- 门禁使用同构建M37的独立CPU叶子oracle；强制单slice分块，覆盖pool关闭、
  三缓冲、arena拒绝以及延迟carry污染。污染必须在发布结果前报错退出。
- `--workspace-fixture`额外运行580例分块整数计算检查，覆盖非二次幂batch、
  两/三缓冲、预算阈值、单块超预算、非法缓冲数和整数溢出。
- 新策略不使用旧Auto B2成本profile；Auto B2拒绝该开关，直到重新标定。

## 阶段归约输出容量释放实验

`NTT_PHASE_TRIM_OUTPUT=1`（默认0）在Newton的逆多项式已保存在主机之后、驻留下降
开始之前，同步释放已经失效的S4归约输出并重置容量。后续hook按需重建。与
`NTT_PHASE_TRIM_RAW`独立；此对照同时启用已有raw释放，两缓冲pool与原分块策略
在两臂相同，仅输出释放0/1变化。保持1 GiB future reserve与全部必要回退。

```powershell
python tools/bench/bench_stage2_carrier.py --comparison phase-output --exe build_cuda_cmake/phase_output_stage2/ecm_cuda_stage2.exe --save <stage1.save> --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode check --projection-only --output data/output_trim_check
python tools/bench/bench_stage2_carrier.py --comparison phase-output --exe build_cuda_cmake/phase_output_stage2/ecm_cuda_stage2.exe --save <stage1.save> --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode timing --telemetry --output data/output_trim_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/output_trim_timing/measurements.json --output docs/benchmarks/output_trim_analysis.json --figure-prefix docs/figures/output_trim
```

- `stage2_phase_trim.output_released_bytes`读取释放时的实际容量，不能用模块全程峰替代。
- 同步`cudaFree`完成已排队的输出scatter/readback与oracle主机快照；延迟CPU检查
  持有自己的主机副本。不释放NTT digits、carry诊断、reduce常量或fold/frontier owner。
- 驻留状态可能因释放发生变化，数学工作路径/检查计数可跨臂不同；每个固定策略
  内计数须稳定，两臂对照全部目标叶子/因子，保持生产强制检查。
- `test_stage2_phase_trim.py`与`test_stage2_chunk_budget.py`也接受同构建的已完成
  `phase-output`小规模独立单位案例参考；用`trimmed_output`臂验证重申请与carry安全。
- 正式计时为ABBA+BAAB、每臂n=4。不要加`--require-resident`拒绝作为基线的合法回退；
  实际owner/root/frontier字段分别保留。小规模驻留门禁可使用该参数。
- Auto B2拒绝未经标定的输出释放策略；未加入INI/发布默认。

## 树形状与共享容量模型

原生`--plan-only`新增`geometry_version=3`和`tree_workspace`。最大树operand为
`h+1`，`h`是严格小于P的最大二次幂；它不是一般意义上的`P/2+1`。
同一组shape/chunk整数合同用于计划和S4分块，完整树的partial group及最后short
chunk都计入。实际调用输出每阶段一行`s4_phase_memory`（debug级）。

`tree_workspace`是**单棵树、无缓存淘汰**的组件模型：

- `shared_big_peak_bytes`：开启pool时A/B[/Q]的最大`8*buffers*N*slices`；关闭pool时
  此字段仅是最大单请求，使用`keyed_big_retained_bytes`查看三缓冲按key保留合计。
- `digit_retained_bytes`：每个`(N,slices)`保留最大的digits输出及两个carry诊断字。
- `output_peak_bytes`：按实际短operand结果长度和chunk计算的S4输出峰。
- `supported`要求S4/device pack、output window/chunk output有效；host pack、最终
  readback等控制路径不在此模型scope。`pool`明确指出是否启用共享池。
- `table_retained_bytes`、`base_retained_bytes`按NTT长度去重；`ntt_retained_bytes`
  包含单树完整NTT保留容量（详见下节），不包含其它阶段遗留缓存、raw/坐标/owner或失败回退；
  `process_peak_complete=false`，不能把这些字段相加作为全进程预算或驻留承诺。

`arena_estimate_bytes`仍是用于旧准入的保守求和，现明确标为
`arena_estimate_kind=legacy_additive`；只修正了其树尺寸，没有将组件下界冒充完整
MemoryPlan，也没有改用旧Auto B2 profile为新布局排序。

```powershell
python tools/test/test_stage2_workspace_plan.py --exe <ecm_cuda_stage2.exe> --save <stage1.save> --carrier-exponent 8011 --output data/tree_plan_gate
python tools/test/test_stage2_workspace_plan.py --exe <ecm_cuda_stage2.exe> --save <stage1.save> --carrier-exponent 8011 --d 1531530 --runtime-check <completed_same_build_check/measurements.json> --output data/tree_plan_runtime_gate
```

首条默认检查5个D、2种分块策略、4种pool/reuse组合；Python独立构造dense padded
tree，并通过native二次幂anchor查询每种NTT长度。第二条另外比较真实F树的group、
pair、chunk、请求大池/输出峰和物理pool峰；不接受构建/输入身份不匹配或回退。
`--workspace-fixture`新增dense tree整数门禁，覆盖131072附近及溢出。

## 驻留准入前回收冷 NTT 缓存

`NTT_OWNER_TRIM_FUSE=1`（默认0）在fold/frontier的1 GiB headroom检查失败时，
回收其它NTT长度的完整fuse上下文（表+base），保留检查的目标N。每次先选容量
最大的冷上下文，同步释放后重新查询真实free；达到申请+growth+原余量即停止。
之后请求相同shape时按原分配器重建一次。它不删除digits/verdict、大池或owner，
不降低余量；没有足够冷缓存时仍正常回退。日志`stage2_cache_trim`记录边界、
前后free、实际释放、context数、耗时；`ntt_phase_cache_stats`记录累计回收。

```powershell
python tools/bench/bench_stage2_carrier.py --comparison owner-cache --exe <ecm_cuda_stage2.exe> --save <stage1.save> --carrier-exponent 8011 --b2 2600000000000 --d 1531530 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode check --projection-only --output data/owner_cache_check
python tools/bench/bench_stage2_carrier.py --comparison owner-cache --exe <ecm_cuda_stage2.exe> --save <stage1.save> --carrier-exponent 8011 --b2 2600000000000 --d 1531530 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode timing --telemetry --output data/owner_cache_timing
```

两臂均为BQ1、output trim1及相同raw/分块/预算；只改变冷缓存策略0/1，key为
`kept_cache/trimmed_cache`。正式ABBA+BAAB每臂n=4，保持必要算术检查。M37独立
参考同时用于分配失败、pool/三缓冲/arena拒绝与carry污染门禁。原workspace
fixture另验证冷缓存释放、保留目标、完整payload/dRes隔离和原shape重建。
Auto B2拒绝该未标定策略，未加入INI/发布默认。

单树计划`tree_workspace`现在包含`cache_shapes`、`table_retained_bytes`、
`base_retained_bytes`、`ntt_retained_bytes`，cache描述直接来自分配器当前device
策略。无eviction且没有其它阶段遗留/额外carry scratch时，这是单树完整NTT
保留容量；仍不是进程峰或全生命周期MemoryPlan。原生门禁另外和真实F树
`ntt_payload_peak_bytes`对照。

**统计更正**：此前NTT+S4子集观测把已包含在arena `bytes`中的fuse base又加一次。
修复后`s4_phase_memory`新增`subset_accounting_version=2`；
`ntt_arena_accounting`输出独立组件重算的`calculated_bytes/mismatch`。旧版
`owned_subset_*`量偏大，不与新版直接作显存收益对比；NTT自身`full_peak_bytes`、
计时、驻留及数学结果均不受此观测错误影响。

## 完整设备分配生命周期台账

`--memory-ledger`使两臂均设置`NTT_MEMORY_LEDGER=1`（默认0）。统一记录源码闭包
成功的cudaMalloc/free，包含临时分配、持久ladder缓存和容量重建；峰值保存当时
同时存活的allocation-site组成。final允许显式persistent缓存，其余必须释放；
unknown free、live/interval/global peak组成与申请/释放守恒均门禁检查。

这是**本实现owned CUDA payload**，不含driver context/module开销、其它GPU进程、
pinned host或驱动分配粒度，不能叫完整物理VRAM峰，也不代替真实free准入。
插桩有额外CPU记账/debug I/O成本，默认关闭，不将检查时间用于正式性能结论。
常规控制台不增加日志；原阶段/模块峰与台账存在重叠，不求和。

```powershell
python tools/bench/bench_stage2_carrier.py --comparison owner-cache --exe <exe> --save <save> --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --memory-ledger --mode check --projection-only --output data/ledger_check
python tools/bench/analyze_stage2_memory_ledger.py --input data/ledger_check/measurements.json --output docs/benchmarks/ledger.json --figure docs/figures/ledger
python tools/test/test_stage2_memory_ledger.py --exe <exe> --matrix data/ledger_check/measurements.json --output data/ledger_gate.json
```

门禁可同时接受多份完成矩阵；`--fallback-dir`接受phase/chunk回退门禁目录。
`stage2_memory_ledger.py`为共享解析器，采集器记录其SHA并检查运行中未变化。
台账来源指向冻结binary的文件/行号；跟踪设备申请API的源码闭包审计不包含
CUDA运行时内部开销。预测式MemoryPlan与普通D/Auto B2联合准入仍在推进。

## Giant点chunk预算取整

`--comparison giant-chunk`使用同binary、固定D与预算，两臂均BQ/raw（显式
`--trim-phase-raw`）/output释放及冷cache准入，只有`NTT_GIANT_CHUNK_FLOOR`改变：
`legacy_points=0`保留向上整批，`bounded_points=1`向下整批，至少保留一个P。
`--giant-point-kb`（默认262144 KiB）传给`NTT_GIANT_POINT_BUDGET_KB`，两臂相同。
这只是X/Z坐标预算，seed、segment、归约、NTT、fold/frontier另行计费。
最低1P超预算时`minimum_over_budget=1`，不能宣称完整显存满足该预算。

```powershell
python tools/bench/bench_stage2_carrier.py --comparison giant-chunk --exe <exe> --save <save> --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --memory-ledger --mode check --projection-only --require-resident --output data/giant_check
python tools/bench/bench_stage2_carrier.py --comparison giant-chunk --exe <exe> --save <save> --carrier-exponent 8011 --b2 2600000000000 --d 1381380 --fold-mb 1024 --baby-mb 640 --trim-phase-raw --mode timing --telemetry --require-resident --output data/giant_timing
python tools/bench/analyze_stage2_carrier_bench.py --input data/giant_timing/measurements.json --output docs/benchmarks/giant_timing.json --figure-prefix docs/figures/giant_timing
python tools/test/test_stage2_giant_chunk.py --exe <exe> --matrix data/giant_check/measurements.json data/giant_timing/measurements.json --previous-check <previous_high_check.json> --output data/giant_gate.json
```

采集验证原生`giant_chunk_plan/done`的预算、整批公式、实际chunk与chain/ladder数量；
workspace fixture包含298项点预算边界与溢出检查。使用`--giant-point-kb 1`可在
小输入上触发多个ladder chunk和最低工作集超预算。production chain阈值仍固定32768，
不绕过其配置门禁。可生成真实跨阈值的独立CPU参考：

```powershell
python tools/bench/prepare_stage2_carrier_inputs.py --exponents 37 --giant-count 32790 --output data/giant_boundary_inputs
python tools/bench/bench_stage2_carrier.py --comparison giant-chunk --exe <exe> --save data/giant_boundary_inputs/m37_cofactor.save --carrier-exponent 37 --b2 6885480 --d 210 --fixtures data/giant_boundary_inputs/fixtures.json --giant-point-kb 512 --workspace-fixture --trim-phase-raw --memory-ledger --mode check --output data/giant_boundary_check
```

`test_stage2_phase_trim.py`和`test_stage2_chunk_budget.py`均支持该comparison的独立
unit参考，可继续验证fold/frontier失败、pool关闭、三缓冲、arena拒绝与carry污染。
正式计时关闭台账及额外叶子/seed检查，保留必要算术检查；ABBA+BAAB每臂n=4。
该策略默认0，尚未进入INI/发布默认；旧Auto B2 profile拒绝新取整和非默认点预算。
