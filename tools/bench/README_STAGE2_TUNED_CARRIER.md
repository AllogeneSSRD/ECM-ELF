# tune 后的余因子普通/梅森承载对照

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
- 保持必需GMP自检、算术收尾和因子整除验证；关闭额外诊断及启动短校准，避免正式运行更新性能文件。固定D的`bench_stage2_carrier.py`也关闭短校准，诊断标记不应进入校准的干净计时样本检查。
- 不修改原生产profile、旧扫描或GPU功率/时钟。输出目录必须为空；`--resume`只继续未完成且采集器身份未变的实验，不覆盖未登记原始调用。

若训练窗口因软预算截断，使用`complete_stage2_carrier_tune.py --study <已完成首轮目录> --output <新目录>`继续未完成的位宽/B2。默认每次600秒，复用有效摘要，最多三次逐步增加预算，直到普通和承载两个局部窗口都完成；失败不宣称全局最优。输出的性能文件和`report.json`保存在新目录，不更新首轮证据。

补测完成后用`bench_stage2_tuned_carrier.py --skip-training --profile <补测性能文件>`在另一个新目录重跑独立正式矩阵。该模式直接冻结已经训练好的文件，不在正式矩阵中调优。报告应引用补测来源及profile哈希，首轮截断结果不得替代最终选型验证。可另用`bench_stage2_carrier.py --mode check --projection-only`固定相同D，检查两侧投影到目标N后的完整叶指纹；诊断调用不作为计时样本。

## 输出与口径

`measurements.json`、`bin/`源码/二进制身份、`inputs/`保存点、`seed.toml`、`training.toml`、冻结的`measured.toml`及训练/正式日志均留在忽略目录。正式调用有正常日志、debug、结果JSONL、命令及每约500ms的只读NVML采样。

分析器重新核对输入、普通/承载锁定值、实际算术位宽、配置/profile和原始日志哈希，再输出`_analysis.json`、`_comparison.csv`及时间/速度比/NTT容量PNG和SVG。记录两侧实际D/P/G、阶段时间、S4归约事件、驻留fold及NTT组件峰、显存观测最大值和模型选择摘要。

主时间为`stage2_full_wall.total=init+main`，包括初始化、必需自检和GCD，不含Stage1、选型查询、tune及进程启动。进程秒数另列。S4/NTT是嵌套计时；组件容量峰不能相加作为进程峰。采样显存最大值只是整卡观测峰，不是精确分配峰。单次测量没有样本标准差或置信区间。
