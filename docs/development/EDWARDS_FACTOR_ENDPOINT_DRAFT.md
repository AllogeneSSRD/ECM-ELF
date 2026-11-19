# Edwards 命中终点、余因子交接与随机 σ

正式行为见 [Edwards 存档](../usage/EDWARDS_CPU_STAGE2.md#存档与数学边界)、[余因子交接](../usage/STAGE2.md#edwards-命中后续算余因子) 和 [Stage1](../usage/STAGE1.md)。本草稿记录验证条件。

## 随机数计时

Windows/MSVC x64 `/O2`，AMD Ryzen AI 9 HX 370，计时线程固定在逻辑 CPU 23，测量前确认无 `cl`、`nvcc` 或 `link` 进程。预热后 301 组，每组调用实际的 `random_sigma53()` 1000 次，包含将值写入预分配数组。中位数 19.7 μs，P95 39.5 μs，最短 19.6 μs。测量范围为随机函数及数组写入，功耗与频率未固定，原始程序和结果位于 `data/experiments/stage1_sigma_sampler/`。

`random_sigma53()` 用时间种子初始化 C 运行库的 `rand()`，拼接随机位。MSVC x86/x64 路径还混入 `RDTSC`，拒绝 0 至 5。NTT 拆批用一次抽样播种，再通过 53 位域上的置换按曲线索引派生互异的 σ。保留串行取种子，构造阶段应用 `gpu_init_threads`。

## 正确性验证

`tools/test/test_edwards_factored_handoff.py` 对照独立的整数点乘，检查 Stage1 终点命中 1009 和构造阶段命中 137 后的余因子端点。开启交接时，使用原 σ=17、B1=20 在余因子上续算并找到 10007。关闭时，直接读档与队列读取均跳过 `FACTOR`。恒等记录留在 Stage1 存档。

在 RTX 4060 Laptop GPU 上，CUDA CGBN 和 NTT 各运行 12 条 M4423 曲线，对照逐曲线随机和随机起点后递增的 σ，确认两条路径均使用配置的 3 个主机构造线程。另暂停 8 条连续 σ 的 NTT 任务，切换为逐曲线随机后恢复，最终 save 保留了检查点里的连续序列。CPU Edwards 在同一进程内的两个新任务抽到不同起点，显式 σ=17 的任务从 17 递增。证据位于 `data/experiments/edwards_sigma_random_start/`。
