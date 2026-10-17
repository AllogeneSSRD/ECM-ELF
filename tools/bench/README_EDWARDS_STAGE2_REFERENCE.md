# Edwards Stage2 精确输入基准

`bench_edwards_stage2_reference.py` 冻结 Prime95 屏幕日志和正式结果，在恢复出的同一 N、Edwards sigma、B1、实际 B2 上测量 CPU NTT/plain 与 GPU 普通余因子/梅森承载。性能结论与计时限制见 [Edwards Stage2 性能](../../docs/performance/EDWARDS_STAGE2.md)。工具不启动 Prime95，也不提交结果。

## 准备

先按 `tools/build/test/` 构建三个计算程序，以及独立末点验证器：

```powershell
tools/build/test/build_edwards_save_ladder_check.ps1
python tools/bench/bench_edwards_stage2_reference.py prepare `
  --prime95 D:/code/GIMPS/p95v3104b05.win64/4Core@4GHz `
  --results D:/code/GIMPS/p95v3104b05.win64/results.json.txt `
  --prime95-exe D:/code/GIMPS/p95v3104b05.win64/prime95.exe `
  --stage1 build_cuda_cmake/edwards_fine_tiers/stage1/ecm_cuda.exe `
  --cpu build_cpu_stage2/native/ecm_cpu_stage2.exe `
  --gpu build_cuda_cmake/edwards_gpu_workers/stage2/ecm_cuda_stage2.exe `
  --threads 1 --repeats 3 --device 1 `
  --output data/experiments/edwards_stage2_reference_new/matrix
```

输出目录必须全新。准备阶段复制日志、结果、EXE 和 GMP DLL，按正式记录的校验关联和已知因子精确恢复余因子；不按 M 标签猜模数。两档请求 B2 固定为 2.6×10^10、2.6×10^11，允许日志中不超过 10% 的规划上调。新计算使用每格的调整后 B2。当前脚本针对这组日志，不是任意 Prime95 基准格式的通用导入器。

Stage1 为 CPU 标量 Edwards `48LCM`，每个 N/sigma/B1 只准备一次，四条 Stage2 路径共享 save。首次准备当前使用 Python 独立 ladder；`--ladder-checker` 目前只在 `prepare --resume-prepare --output ... --ladder-checker ...` 续准备时接入，不能在首次命令中依赖该参数。独立 GMP 验证器使用 Montgomery x-only ladder 重新计算完整标量，限制 B1≤10^6。续准备保留已冻结快照和已验证 save。

## 测量

```powershell
python tools/bench/bench_edwards_stage2_reference.py run `
  --output data/experiments/edwards_stage2_reference_new/matrix `
  --engines gpu gpu-carrier
python tools/bench/bench_edwards_stage2_reference.py run `
  --output data/experiments/edwards_stage2_reference_new/matrix `
  --engines cpu-ntt cpu-plain
```

GPU 是当前基准的完整测量范围，CPU 命令用于可选参考采集。需要同时测量时，两批顺序执行，避免本工具的 CPU/GPU 测量相互竞争。每批按轮次轮换实现顺序，每格默认三次，首个调用也保留；没有丢弃的预热。省略 `--engines` 执行计划中的全部路径。已接受的调用会跳过；存在未入账的调用日志时停止，需先检查失败或主动中断的证据，不能覆盖后当作新样本。`--exponents`、`--repeats` 可以执行子集，但未完成全矩阵时不会声明完整。

CPU 默认线程上限 1、4096 MiB 规划预算，进程亲和性为 `0xAA`，即逻辑 CPU 1/3/5/7；`--cpu-affinity 0` 在准备时声明不设置亲和性。`--threads` 仅能在尚无正式样本时修改。GPU 默认设备 1、arena6300/fold640/batch256 MiB，D 自动选择，关闭短标定、运行时 tune 和自动承载，分别显式指定普通余因子与原梅森指数承载。两侧规划可能不同；单独归因算术承载的实验需另外固定 D。

工具保留原生全 Stage2 时间、进程墙钟、CPU 时间、CPU 单进程峰值工作集和 GPU 设备遥测。进程退出按 250 ms 轮询，墙钟包含退出观察延迟，不用来精确估计启动开销。GPU 的父进程 CPU 时间和工作集不代表父子进程合计；设备显存采样也不是本进程峰值。CPU 内存参数是规划预算，不能当作实测峰值。

工具不设置时钟或功率。可在实验目录的 `cpu_conditions.json` 记录 `declared_fixed_clock_mhz`、`source`、`effective_clock_telemetry` 等现有条件；用户声明与实际时钟遥测须区分。CPU 时间/墙钟表示占用，不能证明有效并行加速。

## 以 GPU 为范围的汇总与图表

`--scope gpu` 要求声明的普通余因子与梅森承载两条 GPU 路径全部完成。它单独核验 GPU 范围，不改写原计划，也不把未完成的 CPU 矩阵标成完成。审计仍检查已入账的 CPU 参考结果；图表主图只显示完整 GPU 对照，CPU 已测单元单列为参考图，缺测处不插值。

```powershell
python tools/test/check_edwards_stage2_reference.py `
  --study data/experiments/edwards_stage2_reference_new/matrix `
  --scope gpu `
  --output data/experiments/edwards_stage2_reference_new/audit.json
python tools/bench/plot_edwards_stage2_reference.py `
  --study data/experiments/edwards_stage2_reference_new/matrix `
  --scope gpu `
  --thread-probe data/experiments/edwards_stage2_reference_new/thread_probe/summary.json `
  --output data/benchmarks/edwards_stage2_reference_new/final `
  --figures data/figures/edwards_stage2_reference_new/final
```

需要 Matplotlib。图表保留精确 B2，不按线程、B2 或频率换算时间；三次样本显示中位数和最小/最大值，Prime95 每格只有一个历史样本。省略 `--scope` 时沿用完整原矩阵要求；`--allow-partial` 仅生成所选范围标明 PARTIAL 的检查图，不能替代范围完整性检查。提前出因子的样本保留，但不混入完整 NF 的耗时统计。图表附原计划快照、报告范围和脚本身份，CSV 保留 GPU 与 CPU 所有已测单元的实际/名义 B2、位宽和样本数。

主动停止采集时，保留原计划、未入账调用日志及停止原因；中断调用不作为完整样本。当前 2026-10-17 基准的 GPU 范围为 156 次，CPU 参考为 32 次，CPU 已停止；最终结果路径见性能主题，不再自动补齐 CPU。

并行能力探针为 `bench_cpu_stage2_threads.py --exe ... --save ... --b2 1000000000 --repeats 3 --output data/experiments/.../thread_probe`，分别测 NTT/plain 的 1/4 线程。正式结论应结合源码中的 OpenMP 范围和 CPU 占用，不把 `--threads 4` 写成四核 Stage2 实现。

所有结果、账户字段和原始校验值只写入 Git 忽略的 `data/`；分享数据前另行审查。公共提交只包含工具、汇总结论与证据位置。
