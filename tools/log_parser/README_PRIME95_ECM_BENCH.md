# Prime95 ECM 性能日志分析

`analyze_prime95_ecm.py`读取正在追加的`screen.log`及当前`worktodo.txt`，保存一个可追溯快照，导出逐曲线、分组及阶段数据。仅依赖Python标准库，不启动、停止或修改Prime95，也不改动其目录中的文件。

## 使用

在仓库根目录执行：

```powershell
python tools/log_parser/analyze_prime95_ecm.py `
  --screen "D:\code\GIMPS\p95v3104b05.win64\screen.log" `
  --worktodo "D:\code\GIMPS\p95v3104b05.win64\worktodo.txt" `
  --output data/prime95_ecm_20261009/latest `
  --b2-targets 26000000000 260000000000 2600000000000
```

程序运行过程中或结束后均可重复执行；每次从快照重建输出，不追加重复记录。`--worktodo`省略时默认读取日志旁的`worktodo.txt`。输出目录必须在输入目录之外。

可选参数：

- `--encoding <name>`：默认`utf-8-sig`；替换字符数记录在元数据中。
- `--b2-targets <B2>...`：显式给出比较档位，支持科学计数法。未提供时按精确的请求B2分组。
- `--max-b2-overshoot <x>`：默认`0.10`，范围`[0,1)`；档位关联要求`target ≤ reported ≤ target*(1+x)`。这只是分组容差，不修改实际B2或耗时。

不安装监控或定时任务。快照只读取打开文件时观察到的长度；日志末尾未换行的半条记录暂不解析，下次重跑时再纳入。两个输入分别读取，并非同一时刻的原子快照；是否发生追加、替换及各自哈希均有记录。

## 输出

- `analysis.json`：完整结果、快照身份、CPU/亲和性日志、诊断提示、任务、曲线、FFT候选和分组。
- `runs.csv`：每一次`ECM on`是一行；保存worker、原行号、sigma、曲线类型、B1、请求/实际B2、FFT、D、poly degree、内存、计时、因子及状态。`run_id=w<worker>:line<line>`，重复出现curve #1不会覆盖旧记录。
- `groups.csv`：分组均值、中位数、最小/最大、样本标准差、完整样本数和阶段百分比。仅纳入`benchmark_eligible=true`的记录；不足两条时标准差留空。
- `phases.csv`：nQx、F树构建层、PolyR、压缩、PolyG、PolyH、H缩放、F树up/down及叶子乘积的逐事件计时；保留行号和估计标记。
- `worktodo.csv`：当前剩余任务，含ECM/ECM2/ECMSTAGE2、可选参数和已知因子；可以验证的梅森余因子另有`input_bits`。
- `screen.snapshot.log`、`worktodo.snapshot.txt`：本次实际解析的输入前缀。

CSV采用UTF-8 BOM。嵌套字段编码为JSON；没有计时或无法确定的位宽保留为空，不填0。`0.000 sec`仍是Prime95的实际打印值，不表示真实时间严格等于0。

内存列对应Prime95打印的Available/Using配置与计划值，不是进程内存峰值。逐曲线另存已观察到的polymult helper逻辑CPU编号和最大roundoff；未观察到的配置不推断。transforms为Prime95原始计数，不与GPU NTT调用次数直接等同。

## 状态与统计

- `complete_stage2`：init、main、GCD三项均有完成计时，且无检测到的错误或恢复标记；进入基准均值。
- `factor_early`：找到因子时缺少完整Stage2计时；不进入Stage2均值。
- `open_at_snapshot`：该worker最后一条未观察到终结记录；可能正在执行，不能仅凭日志证明进程存活。
- `incomplete` / `interrupted`：被后续曲线替代但未完成，或有停止标记。
- `error` / `resumed_or_restarted`：观察到错误或恢复/重启标记，独立保留。

完成GCD后才发现因子的曲线仍具有完整计时，可以进入均值。提前结束的短样本不会因拥有Stage1计时而被当作完整Stage2。工具不自动剔除“首遍”，也不将Curve编号当作全局唯一编号。

## 计时口径

```text
S2 = S2_init + S2_main + GCD
logged_curve = S1 + S2
S2_init_% = mean(S2_init) / mean(S2) * 100
S2_main_% = mean(S2_main) / mean(S2) * 100
GCD_% = mean(GCD) / mean(S2) * 100
```

日志的`Stage 2 complete ... Total time`对应这里的`S2_main`，不是包含初始化的完整Stage2。三个百分比的分母是完整Stage2，**不含Stage1**。选形、存档I/O、PRP和其他未被这些打印计时覆盖的工作不能自动算入；`logged_curve`不是完整生产墙钟。日志时间戳仅精确到秒，另存的`timestamp_elapsed_seconds`只作粗略对照。

详细阶段是父计时内部的字段，不能再次加到S2上。尤其`PolyF up/down`在仓库中的Prime95参考源码里使用首个切片计时乘切片数量，属于外推，不能直接拼成实测百分比堆积图。工具保留原值及负残差，不强行归一化到100%。

源码口径参照：`.refactor/p95v3106b01.source/ecm.cpp`：8826（初始化计时起点）、9335（init输出）、9350（主计时重置）、9673/9742（切片外推）、9842（main输出）、9877（独立GCD）。本次运行程序为31.4b05；参考源码为31.6b01，不能据此宣称两个二进制所有实现细节相同。

## B2、实际位宽和任务关联

Prime95会向上调整B2；同一档位的实际B2还可能随曲线轻微变化。因此同时保留`b2_requested`、`b2`及可选的`b2_target_bucket`，分组导出实际B2最小/最大值。任务完成行中的B2按参考源码是多条曲线的调整平均值，独立记为`reported_average_b2`，不覆盖曲线Actual B2。

找到因子后，Prime95可以改写worktodo、减少剩余曲线数并以curve #1继续。当前队列的已知因子不能倒推此前每次测试的初始输入，所以：

- `exponent`只是`M<p>`标签，不能直接当作历史实际余因子位宽。
- 历史曲线的`actual_input_bits`默认空；当前任务的`input_bits`只代表当前worktodo快照。
- `queue_candidate_lines`是当前队列候选，不是历史任务身份的确定匹配。
- 分组按worker、B1、曲线类型、B2档位、D/poly、FFT、内存与`modulus_epoch`隔离。epoch在因子和任务边界处分开，是保守区段，不代表恢复了真实N。

后续拟合CPU耗时与实际N、比较GPU时，应补充测试开始前的worktodo/已知因子记录，或使用明确冻结的输入。工具会保留现有证据，不以指数替代未知位宽。

## 完成后的CPU/GPU比较

`compare_prime95_gpu.py` 可利用 `results.json.txt` 的sigma与known-factors恢复历史N，
逐次验证整除，并用NF结果校验最终N。只有**精确整数相等**及B2档位相同才配对；
不把相同指数/近似位宽视为相同输入，不对功耗或B2进行比例校正。

```powershell
python tools/log_parser/compare_prime95_gpu.py `
  --cpu-analysis data/prime95_ecm_20261009/completed/analysis.json `
  --results D:/code/GIMPS/p95v3104b05.win64/results.json.txt `
  --gpu-analysis data/benchmarks/stage2_n_scaling_20261008_analysis.json `
  --output data/prime95_ecm_20261009/comparison

python tools/log_parser/plot_prime95_gpu.py `
  --input data/prime95_ecm_20261009/comparison/comparison.json `
  --output-prefix data/figures/prime95_gpu_stage2_20261009
```

输出比较JSON、逐曲线N恢复CSV、分组CSV、精确配对CSV，以及完整耗时、实际位宽趋势、
速度比、阶段百分比的PNG/SVG。绘图需要NumPy/Matplotlib；可选
`--canvas <绝对路径.canvas.tsx>` 生成自包含交互看板。
比较器拒绝GPU未完成快照，输出目录必须位于各输入目录之外。

日志曲线GCD/因子终结之后的 `Resuming.` 停机消息不再把该完整曲线标为恢复执行。
本轮详细口径及结果见 `docs/performance/STAGE2.md`。
