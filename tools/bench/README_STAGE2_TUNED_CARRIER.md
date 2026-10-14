# tune 后的余因子普通/梅森承载对照

## 受限预算的高位宽补充

`prepare_stage2_wide_carrier.py` 从本地 PrimeNet HTML 选择实际位宽不超过指定档且最接近的余因子，逐项验证因子并剥离重复幂。高位宽 Stage1 保存点用 Python 与 GMP-ECM 独立核对；`--lower-study` 可复用已完成扫描的原九个低位宽保存点。维护中的 `ecm_dataset` 数据库不变，临时输入数据库只写入忽略目录。

```powershell
python tools/bench/prepare_stage2_wide_carrier.py --exe build_cuda_cmake/stage2_auto_carrier_20261012/ecm_cuda_stage2.exe --lower-study data/experiments/stage2_d_compression_20261011/paired_default/candidate --output data/experiments/wide_carrier/inputs --b2 26000000 260000000
```

`bench_stage2_tuned_carrier.py --prepared-inputs` 接受这种尚未计时的独立有效输入。`--tune-defaults-only` 将搜索限制为普通和承载两条路径的默认 D，`--b2` 可覆盖边界，`--auto-arm` 增加从 worktodo 推断候选的第三条计时路径。普通和承载路径仍显式锁定算术；三条路径共用同一冻结 TOML。

单批 G=1 的默认计划超出当前 ECM tune 合同，工具保留运行数据；自动选型缺乏配对实测时保留普通默认。`complete_stage2_carrier_tune.py` 补齐软预算截断的多批窗口，保持默认 D 限定，将单批范围列为排除项而非成功校准。预算到期完成当前测量，不能将软预算当作强制墙钟上限。

```powershell
python tools/bench/bench_stage2_tuned_carrier.py --exe build_cuda_cmake/stage2_auto_carrier_20261012/ecm_cuda_stage2.exe --reference-study data/experiments/wide_carrier/inputs --prepared-inputs --profile data/experiments/wide_carrier/seed.toml --output data/experiments/wide_carrier/timing --device 0 --tune-defaults-only --auto-arm --tune-budget-seconds 90
```

`seed.toml` 可由小范围 `--tune ntt` 生成；不会写入生产 tune 文件。`analyze_stage2_tuned_carrier.py` 同时审计三条路径并生成 CSV、JSON、PNG/SVG，图注使用实测活跃时钟，不假设频率设置。所有实验应串行占用 GPU，功能门禁和采样不要并发。

`bench_stage2_tuned_carrier.py`复用一轮已完成位宽扫描的有效余因子save。先调用生产程序现有的局部D搜索，分别测普通模数和梅森承载；所有训练结束后冻结同一性能文件，交替运行两条路径。两侧都保持D=0，由生产选型器读取冻结文件选择D；这项比较包含承载及其选型效果，不是固定D下单个归约内核的比较。

输入仍为相同目标N。显式`carrier_exponent=0`锁定普通路径，显式p锁定承载路径，工具先验证N整除2ᵖ−1。完整梅森数的独立控制任务不是这个A/B的替代输入。

## 使用

```powershell
python tools/bench/bench_stage2_tuned_carrier.py `
  --exe build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe `
  --reference-study data/experiments/reference_study `
  --profile build_cuda_cmake/production_stage2/stage2_ecm_tune.toml `
  --output data/experiments/tuned_carrier --device 0

python tools/bench/analyze_stage2_tuned_carrier.py `
  --study data/experiments/tuned_carrier `
  --output data/benchmarks/tuned_carrier `
  --figures data/figures/tuned_carrier
```

- `--exponents 503 2003 8011`只选择指定原指数；省略则取全部余因子输入。
- 复用参考扫描的三个B2和arena/fold/batch预算。必须使用相同GPU UUID和名称、已完成且无失败的参考扫描。
- 训练每格单独给180秒软预算，`--tune-budget-seconds`可覆盖。调用`--tune-level 10 --tune-repeats 1`；宽D候选表由原生局部搜索逐步访问，复用符合条件的旧实测摘要，预算到期可能留下未完成的邻域。训练时间不计入正式Stage2耗时，分析保留每个窗口的完成状态。
- 正式默认每格每侧一次；`--repeats 3`可增大重复。各位宽/路径先单独预热最小B2，预热不进入统计。重复之间交替两侧顺序。
- 保持必需GMP自检、算术收尾和因子整除验证；关闭额外诊断、启动短校准及运行采集（实验INI的`stage2_tune_runtime=false`），避免正式运行更新性能文件。固定D的`bench_stage2_carrier.py`也关闭短校准，诊断标记不应进入校准的干净计时样本检查。
- 不修改原生产profile、旧扫描或GPU功率/时钟。输出目录必须为空；`--resume`只继续未完成且采集器身份未变的实验，不覆盖未登记原始调用。

若训练窗口因软预算截断，使用`complete_stage2_carrier_tune.py --study <已完成首轮目录> --output <新目录>`继续未完成的位宽/B2。默认每次600秒，复用有效摘要，最多三次逐步增加预算，直到普通和承载两个局部窗口都完成；失败不宣称全局最优。输出的性能文件和`report.json`保存在新目录，不更新首轮证据。

补测完成后用`bench_stage2_tuned_carrier.py --skip-training --profile <补测性能文件>`在另一个新目录重跑独立正式矩阵。该模式直接冻结已经训练好的文件，不在正式矩阵中调优。报告应引用补测来源及profile哈希，首轮截断结果不得替代最终选型验证。可另用`bench_stage2_carrier.py --mode check --projection-only`固定相同D，检查两侧投影到目标N后的完整叶指纹；诊断调用不作为计时样本。

## 输出与口径

`measurements.json`、`bin/`源码/二进制身份、`inputs/`保存点、`seed.toml`、`training.toml`、冻结的`measured.toml`及训练/正式日志均留在忽略目录。正式调用有正常日志、debug、结果JSONL、命令及每约500ms的只读NVML采样。

分析器重新核对输入、普通/承载锁定值、实际算术位宽、配置/profile和原始日志哈希，再输出`_analysis.json`、`_comparison.csv`及时间/速度比/NTT容量PNG和SVG。记录两侧实际D/P/G、阶段时间、S4归约事件、驻留fold及NTT组件峰、显存观测最大值和模型选择摘要。

主时间为`stage2_full_wall.total=init+main`，包括初始化、必需自检和GCD，不含Stage1、选型查询、tune及进程启动。进程秒数另列。S4/NTT是嵌套计时；组件容量峰不能相加作为进程峰。采样显存最大值只是整卡观测峰，不是精确分配峰。单次测量没有样本标准差或置信区间。

## 从正常运行恢复性能摘要

新生产程序默认将缺失算术位宽/执行区间的干净完整曲线摘要写回ECM性能TOML。失败的缓存写回不影响曲线完成，结果JSONL仍含`tune_observation`。`import_stage2_benchmark_tune.py --runtime-results FILE --output data/benchmarks/runtime_import.toml`可提取这些回执；参数可重复，输出须为新文件。再使用生产程序的`--tune ecm --tune-merge data/benchmarks/runtime_import.toml --tune-file combined.toml`原子汇集，目标不得与输入相同。

导入核对目标N、承载整除关系、B1/B2、无因子、自检及抽查、互斥阶段总和和预测比值；忽略末尾没有换行的未提交行。保留每个候选最快的单曲线摘要、原条件、测量前预测及二进制身份；摘要不转成正式重复调优参考。旧结果中缺少完整结构时跳过，不从曲线外层seconds猜测引擎时间。已有冻结全位宽扫描通过同一工具的`--study --condition-profile --audit-analysis`入口导入，原始证据不修改。

`measured_model_ratio=median_seconds/predicted_seconds`只描述完整ECM计时；字段及回溯留一预测规则统一见[Auto B2 / tune](../../docs/architecture/AUTO_B2.md#toml-中的算术路径)。
