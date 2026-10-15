# 短 NTT Graph 完整曲线验证

此组工具测试实际 Stage2 中的主机提交融合。生产内核、pack、carry、S4、检查和分块保持原路径；默认关闭 Graph。当前结果与限制见 [性能说明](../../docs/performance/NTT_GRAPH.md)。

## 构建与正确性

- 完整程序：`tools/build/build_stage2_local.ps1 -Engine production -Build <独立目录> -Arch sm_89 -SplitCompile 8`。
- 独立门禁：`tools/build/test/build_ntt_graph_probe.ps1 -Build <独立目录> -Arch sm_89`；输出程序 `ntt_graph_probe.exe <device>`。`--fault` 故意翻转一个输出 digit，预期恰好一个 GMP 错误且返回非零。
- 门禁使用实际生产 pack→NTT→carry 路径，对完整多项式大整数乘积做 GMP 对照；覆盖不同位宽、batch 尾部、Q 借用 B、表驱逐、工作区释放和重建。

## 同二进制完整 A/B

`benchmark_stage2_graph.py --exe <exe> --cases <cases.json> --output <新目录> --device 1 --repeats 4`。

cases 为对象数组。每项必须提供 `name`、`save`、`b2`、`d`、`batch_mb`；可选 `arena_mb=6300`、`fold_mb=640`、`carrier=0`。name 只接受字母、数字和下划线。不同曲线形状用独立项；普通／承载使用相同 save、B2、D 和预算。

- 每侧 4 次，顺序 off/on、on/off 交替；没有丢弃的 warmup。计时包括 Graph 建立和 mandatory selftests。
- 主要指标是 `stage2_full_wall.total=init+main`，排除 Stage1 与进程启动；进程墙钟另记。
- 检查结果、rootfold 指纹、数学 NTT 数量和 S4 分块数一致；保留全部原始日志、命令、输入、构建身份及 500ms 遥测。
- `--max-k 19 --max-words 4194304 --min-calls 4` 对应实验默认准入。`max-words` 是单次 batch×L，不是单条 L，也不是新增显存分配量。
- Graph 未命中项仍保留，以控制组标识，不能将其正差异归因于 Graph。
- `--diagnostic` 单独运行 off/on 完整叶子指纹对照，设置额外检查，标记 clean=0；不混入正式计时。

## 管理员 Nsight

`profile_stage2_graph.py --reference <成功的.command.json> --output <新目录> --kind systems --run-admin`。

工具复制该样本的实际命令与环境，仅改输出位置；通过生成的 `admin.ps1` 提升权限，采集 Systems 2026 的 Graph node 时间线。`--collect-only --output <目录>` 可重新导出与核验既有 trace，不再运行 GPU。

Compute 使用 `--kind compute --kernel <形状.json>`。形状从 Systems summary 的 kernels 项选择，含 mangled 名称、grid/block/shared/registers 和 first_invocation；须核对 Graph 中选中实例及原路径对应实例。采集只重放一个 kernel，默认 16 passes 左右；到达较晚实例前的插桩可能很慢。

Compute 明确使用 `--clock-control none --cache-control none --graph-profiling node`。不修改功耗、频率或缓存策略；因此重放频率可能低于正式负载。比较吞吐/占用和 stall 需保留实测频率，不能把重放秒数当作完整曲线提速。导入报告也可能需要非沙箱执行，以读取工具的用户 Sections 目录。

采集后再次核对完整曲线结果、NTT 数量和分块；Compute 另核对实际 kernel、grid/block、动态 shared 和 registers。Systems 的无本进程 GPU 事件空隙不等于整卡空闲；CUDA API 时间包含等待，与 GPU 时间重叠。

## 汇总与图

`analyze_stage2_graph.py --experiment <A/B目录> [--experiment <另一目录>] [--profile <Systems目录>] --output <新汇总目录> --figure <图片.png>`。

输出汇总 JSON/CSV、完整时间与变化图、分阶段百分比图；提供Systems摘要时另画真实batch吞吐及本进程GPU事件窗口。`--profile` 也可记录Compute核验摘要。误差线为 MAD，非置信区间。相同固定 D 的不同 batch 可以比较整体代价；不以独立算子 Conv/s 替代完整 Stage2 验收。

原始证据放 `data/experiments/`，统计放 `data/benchmarks/`，图放 `data/figures/`，均由 Git 忽略。
