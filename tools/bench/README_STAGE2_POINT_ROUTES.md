# Stage2点路线的固定D对照

当前结论见[点生成瓶颈](../../docs/performance/STAGE2_PREPARATION.md#点生成固定d瓶颈验证)。工具只读旧矩阵的有效输入，另建私有输出，不消费生产队列，不生成或更新tune。

## 性能实验

`bench_stage2_point_routes.py`以已完成的有界布局研究为来源，选择指定余因子的承载默认格，冻结EXE／GMP／save。保持原实际B2与D/P、布局、算术、预算及必需检查。只允许GPU1，逐调用检查外部上下文、保留NVML；界面白名单沿用来源研究中用户批准的PID及完整路径，不允许任意共享设备。

```powershell
python tools/bench/bench_stage2_point_routes.py `
  --study data/experiments/stage2_layout_matrix_20261031/low_warm2 `
  --case m8011_n7995_s26 --sequence baseline,split6,split6,baseline `
  --output data/experiments/stage2_point_routes_20261031/tail_abba
```

`split6`仅修改坐标点预算，使实际容量为6P，不修改NTT batch预算或D。原始回执记录具体chunk及chain／ladder路线。每组两次是筛查，不代表全范围收益。

正式程序默认链长64，也支持32及有限auto选择。生产host overlay可以直接用同一二进制比较：

```powershell
python tools/bench/bench_stage2_point_routes.py `
  --study data/experiments/stage2_layout_matrix_20261031/low_warm2 `
  --case m8011_n7995_s26 `
  --exe build_stage2_point_policy_0_55_0_03/ecm_cuda_stage2.exe `
  --sequence baseline,split6,balanced,rounded,rounded,balanced,split6,baseline `
  --output data/experiments/new_point_policy/tail_abba
```

`balanced`保持点容量和B2，仅借整P点消除短尾；`rounded`补最小整P且使尾块达到chain门槛，记录请求／实际B2。`rounded_reference`显式指定相同实际B2，与round路径做规范叶对照。比较摘要时按实际B2分组；不能比较不同覆盖范围的叶hash。

`chain32`在生产overlay下使用正式CLI，不需要测试适配。`chain128`仍只能使用独立探针；下面命令保留为有意的128链研究入口：

```powershell
tools/build/test/build_stage2_point_routes_probe.ps1 `
  -Overlay build_stage2_layout_tune_format9_20261031 `
  -Build build_stage2_point_routes_20261031_b
python tools/bench/bench_stage2_point_routes.py `
  --study data/experiments/stage2_layout_matrix_20261031/low_warm2 `
  --case m16319_n16272_s26 `
  --exe build_stage2_point_routes_20261031_b/ecm_cuda_stage2.exe `
  --sequence baseline,chain32,chain128,chain128,chain32,baseline `
  --output data/experiments/stage2_point_routes_20261031/wide_chain_probe
```

构建脚本核对来源overlay的源码和43个复用对象，重编译一个主机适配器，不重新编译CUDA。不同程序SHA分别记录，不把该探针的成绩标为正式程序的新默认。

## 诊断与正确性

- `--nsys <nsys.exe>`：启用CPU NVTX范围并采集CUDA事件、分配和拷贝，自动导出SQLite。按仓库规则用管理员启动，不同时运行其他GPU测试。诊断计时不并入性能均值。
- `--trace-checks`：显式开启GMP设备叶检查和目标环规范叶摘要，对各设置比较摘要；要求所有组／设备叶word均被检查。引擎按原合同标为`clean=0`，只能用于额外算术诊断。正常性能始终要求`clean=1`。
- 工具保留首轮和原始失败目录；重试必须新建输出目录。失败的半份研究不加入完整交错顺序的平均数。
- 规范叶摘要不是独立逐叶oracle，既有GMP门禁继续保留。NF点重放不增加独立ECM曲线覆盖，也不替代退化点／命中因子语义验收。

NCU单kernel诊断为本轮一次性辅助脚本，保存在忽略的实验目录。使用`--clock-control none`、一次真实NW256 chain launch和明确指标，导出原始`.ncu-rep`及离线CSV；不得把kernel重放时间用作完整曲线成绩。

## 汇总与图表

```powershell
python tools/bench/analyze_stage2_point_routes.py `
  --root data/experiments/stage2_point_routes_20261031 `
  --output data/benchmarks/stage2_point_routes_20261031/final `
  --figures data/figures/stage2_point_routes_20261031/final `
  --publish docs/performance/attachments
```

输入为完整的`tail_abba`、`wide_chain_probe`、`tail_nsys`、`wide_nsys`、`tail_leafcheck_verified`、`wide_leafcheck_verified`与`wide_ncu`。逐次重读NF、构建SHA、N／B1／B2／sigma／D/P、阶段闭合、实际路线与检查计数；缺测即拒绝。统计图只用10次未插桩性能样本，Giant与叶准备合并避免同步归属误读。图为绝对秒数，黑线标原始min/max，不是置信区间。

输出JSON、绘图器快照和PNG／SVG在忽略的data目录；`--publish`只复制审查后的PNG到文档attachments。原始证据、失败日志与冻结程序不公开提交。

生产选项的汇总使用 `analyze_stage2_point_policy.py --root <study-root> --output <fresh-summary> --figure <png>`，输入`tail_abba02/`、`wide_abba01/`、`tail_leafcheck01/`、`wide_leafcheck01/`、`final_tail_holdout01/`及`final_tail_holdout02/`，分别为四种分块对照、生产32/64对照、两档独立叶诊断和最终集成单遍复测。最后发布PNG之前检查原始min/max及遥测，黑线不是置信区间。固定输入名称是这组冻结研究的范围，不代表自动测试其他位宽。

`tools/test/test_stage2_point_policy_native.py`生成独立lcm保存点，检查素数NF、非单位基点、块GCD=N及原路径重放。`tools/test/test_stage2_chain_runtime_native.py`在六份复制的真实保存点上验证64/64/32/32及最快采用；`--reuse-run`只读审核已完成原始调用并补一次规划，适用于采集器在完成GPU工作后失效的情况。它保留旧目录，不能将失败或未完成的计算改判为完整研究。
