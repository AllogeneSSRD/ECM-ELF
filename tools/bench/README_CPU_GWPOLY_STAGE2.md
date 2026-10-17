# CPU GW 多项式 Stage2 基准

使用已验证的 Edwards Stage1 save，重新冻结输入、EXE/GMP DLL、GW 库、项目头文件、算术源码及收集器。先按 [CPU Stage2](../../docs/usage/EDWARDS_CPU_STAGE2.md) 构建带 `-GwPoly` 的版本；精确 N、sigma、B1、B2 的源计划来自 [Edwards 参考准备工具](README_EDWARDS_STAGE2_REFERENCE.md)。工具不启动 Prime95，不提交结果，也不调整 CPU 时钟。

```powershell
python tools/bench/bench_cpu_gwpoly_stage2.py `
  --exe build_cpu_stage2/gwpoly_fixed/native/ecm_cpu_stage2.exe `
  --study data/experiments/edwards_stage2_reference_20261017/matrix_01 `
  --output data/experiments/cpu_gw_poly_stage2_new/fixed4g `
  --exponents 1009 10267 16319 --threads 1 4 --repeats 3 `
  --memory-mib 4096 --affinity 0xAA --cpu-ghz 4
```

`--cpu-ghz` 必填，记录操作者声明，不能当成有效时钟遥测。CPU 功耗和温度未采样。输出目录必须全新；执行顺序逐轮交替线程数，保留首轮。默认两档 B2；`--bounds low|high` 为各指数独立选择较低/较高请求档，不按全体实际 B2 的最大最小值筛选。每次使用日志调整后的精确 B2，不按比例换算。

原生时间从曲线构造至完整 Stage2 清理结束，不包含读档、进程启动和结果发布；另存进程墙钟、CPU 时间/墙钟占用，以及该进程的峰值工作集。规划预算不是内存硬限制。额外 `--gw-check` 不在性能调用中启用；正确性先使用 [门禁](../test/test_cpu_gwpoly_stage2.py)。提前出因子的运行保留原始记录，不进入完整 NF 中位数；失败保留证据并停止，不自动覆盖或续采。

门禁的 `--reference-study PLAN_DIR` 可额外消费同一冻结计划，在 10232/16272 位余因子上比较三处完整 GMP 投影；`--gpu` 另检查 GPU1 Stage1 save 交接。正确性调用与性能采样顺序执行，避免 CPU 竞争。两类原始证据分目录保存，带额外 GMP 检查的耗时不进入本性能图。

```powershell
python tools/bench/plot_cpu_gwpoly_stage2.py `
  --study data/experiments/cpu_gw_poly_stage2_new/fixed4g `
  --output data/benchmarks/cpu_gw_poly_stage2_new/final `
  --figures data/figures/cpu_gw_poly_stage2_new/final
```

绘图需要 Matplotlib 和 NumPy，要求两档 B2、1/4 线程及声明的所有重复完成。输出 CSV/JSON、耗时 PNG/SVG 和线程/内存 PNG/SVG；误差线为实测最小/最大值，Prime95 是单次历史四线程样本。没有按频率或线程数归一化。性能结论、版本和已完成证据见 [CPU 性能主题](../../docs/performance/CPU_GWPOLY_STAGE2.md)。

原始结果、账户字段和校验值仅保留在忽略的 data 目录，公共 Git 只包含工具及提炼后的结论。
