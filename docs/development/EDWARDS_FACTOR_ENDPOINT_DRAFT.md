# Edwards 命中终点、余因子交接与随机 σ

正式行为见 [Edwards 存档](../usage/EDWARDS_CPU_STAGE2.md#存档与数学边界)、[余因子交接](../usage/STAGE2.md#edwards-命中后续算余因子) 和 [Stage1](../usage/STAGE1.md)。本草稿记录验证条件。

## 随机数计时

Windows/MSVC x64 `/O2`，AMD Ryzen AI 9 HX 370，计时线程固定在逻辑 CPU 23，测量前确认无 `cl`、`nvcc` 或 `link` 进程。预热后 301 组，每组调用实际的 `random_sigma53()` 1000 次，包含将值写入预分配数组。中位数 19.7 μs，P95 39.5 μs，最短 19.6 μs。测量范围为随机函数及数组写入，功耗与频率未固定，原始程序和结果位于 `data/experiments/stage1_sigma_sampler/`。

`random_sigma53()` 是上述计时使用的采样器，用时间种子初始化 C 运行库的 `rand()`，MSVC x86/x64 路径还混入 `RDTSC`。当前 Edwards 统一从高分辨率时间、worker 编号和进程内计数采种，再用原 NTT 的 53 位置换派生 σ。采种使用独立的 MurmurHash3 终结混合。见 [Stage1 σ 配置](../usage/STAGE1.md#队列与交接)。

## 正确性验证

`tools/test/test_edwards_factored_handoff.py` 对照独立的整数点乘，检查 Stage1 终点命中 1009 和构造阶段命中 137 后的余因子端点。开启交接时，使用原 σ=17、B1=20 在余因子上续算并找到 10007。关闭时，直接读档与队列读取均跳过 `FACTOR`。恒等记录留在 Stage1 存档。

CPU Edwards、命令行直跑、CUDA CGBN 和 NTT 生成的 σ 均通过逆置换核对。三个 worker 并发运行同一 M127，36 条曲线使用互异的 σ。RTX 4060 Laptop GPU 上的 CUDA CGBN 和 NTT 各运行 12 条 M4423 曲线，均使用配置的 3 个主机构造线程。续算测试保留了切换随机模式前的连续序列，也逐条复现了 0.69.14 检查点中的派生 σ。证据位于 `data/experiments/edwards_sigma_worker_mixed/`。
