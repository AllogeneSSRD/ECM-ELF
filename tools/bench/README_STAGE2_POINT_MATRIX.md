# Stage2链长与尾块矩阵

`bench_stage2_point_matrix.py`沿用已完成小B2矩阵的有效保存点和最快D／G树布局，对13位宽×普通／承载26格补链长，再冻结链长做尾块开关对照。学习、只读选择和独立验证分别保存，不将训练计时计入持出均值。复制同一Stage1点仅作计时，不增加独立ECM覆盖。

源研究须有完整`plan.json`、`defaults`及`frozen_tune.toml`；程序须来自已完成、复用对象哈希可核对的production host overlay。测试不改生产目录，不改GPU设置。逐次检查GPU1外部进程及用户批准的精确PID／路径白名单，连续写入NVML遥测。

```powershell
python tools/bench/bench_stage2_point_matrix.py `
  --study data/experiments/stage2_layout_matrix_20261031/low_warm2 `
  --exe build_stage2_point_matrix_0_56_0_01/ecm_cuda_stage2.exe `
  --output data/experiments/new_point_matrix --phase learn
python tools/bench/bench_stage2_point_matrix.py `
  --resume --output data/experiments/new_point_matrix --phase evaluate
```

学习配置最小观察数1、无warmup。每格固定D、carrier、布局、B2和预算，建立64/32对照；原生有界链长选择仍执行全部算术门禁。计划三条复制保存点，在第二条开始时用原生stop event请求本条完成后停止，得到每条链长一次实际NF。当前条件与旧成绩不完全相同时先建立当前64参考，不改写旧成绩的条件。

完成全部链长学习后复制冻结profile，逐格只读确认采用已准入的最快均值。验证固定所选链长，只开关尾块平衡，顺序按格交替，均为新调用；profile必须保持原SHA。B2凑整关闭。主指标为完整引擎`init+main`，另存曲线内部时间与进程外层时间，不能将嵌套timer相加。

```powershell
python tools/bench/recheck_stage2_point_matrix.py `
  --root data/experiments/new_point_matrix --output data/experiments/new_point_rechecks
python tools/bench/bench_stage2_point_control.py `
  --root data/experiments/new_point_matrix --output data/experiments/new_point_control
python tools/bench/analyze_stage2_point_matrix.py `
  --root data/experiments/new_point_matrix --output data/benchmarks/new_point_matrix `
  --figures data/figures/new_point_matrix --publish docs/performance/attachments
```

复查工具对首遍正负差异均采用同一条件：绝对差>0.05秒且相对差>3%，追加off/on/on/off四条；全部初始与复查调用保留，不能挑最小值替换。正对照固定D180180，分别检查64和32链的开关效果，不加入优选D矩阵总数。

分析器重新核对程序／依赖／保存点／冻结profile、全部原始NF回执、算术计数、实际点块及NTT长度直方图，再生成完整表和两种时间轴及阶段占比图。显著性不能由单次微差推导；点块和NTT工作量未变时应明确报告未触发调度，而不是把噪声计为收益。

失败目录完整保留。采集器恢复须先审核已完成原始调用，在新续跑目录记录来源plan和日志哈希；不得删除失败条目伪装首轮通过，也不得重写原始回执来改变结论。
