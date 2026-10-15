# 精确 Tensor NTT 长度基准

本工具直接包含当前`src/cuda/stage2/ntt_runtime.cuh`，在同一二进制中比较生产CUDA tile、Tensor64及Tensor40候选。outer计划、根表、Goldilocks PTX归约和pointwise/scale保持相同。计算的是完整NTT或三变换域卷积，不包括Kronecker pack、整数carry和S4模N归约，不输出ECM曲线吞吐。

## 工具

- `tools/build/test/build_ntt_tensor_length_probe.ps1`：编译sm80+探针，默认sm89；固定PTX3、当前生产规范减法、per-thread默认流，生成源码/EXE/GMP身份。per-thread流用于Graph捕获，所有路线都在同一主机线程顺序运行。
- `ntt_tensor_length_probe.cu`：GMP全变换门禁、故意损坏门禁、完整算子计时、全部A/B输出设备端比较、资源与容量报告。
- `benchmark_ntt_tensor_lengths.py`：顺序运行不同length/batch/CTA，检查二进制/源码/DLL身份、每模式样本数、全部输出及释放，保存原日志、CSV/JSON与500ms遥测。
- `plot_ntt_tensor_lengths.py`：从上述结果生成时间/速度比与短batch图。若输入含重复形状，主图采用命令行靠前数据；汇总JSON仍保留全部记录。默认图放调用者指定位置，汇总放仓库`data/benchmarks/`。

## 使用

在仓库根目录构建：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/test/build_ntt_tensor_length_probe.ps1
```

完成独立GMP门禁及native长度扫描，输出目录必须为空：

```powershell
python tools/bench/benchmark_ntt_tensor_lengths.py --exe build_cuda_cmake/ntt_tensor_length_probe/ntt_tensor_length_probe.exe --device 1 --sizes 8 10 12 14 16 18 19 20 21 22 23 24 25 26 --repeats 11 --output data/experiments/tensor_ntt_native
```

同一冻结EXE可以补充短batch或CTA扫描。已核对相同EXE的GMP门禁后，可显式`--skip-gate`避免重复：

```powershell
python tools/bench/benchmark_ntt_tensor_lengths.py --exe build_cuda_cmake/ntt_tensor_length_probe/ntt_tensor_length_probe.exe --device 1 --sizes 12 16 18 19 --batches 8 32 --operations 2 --skip-gate --output data/experiments/tensor_ntt_batch
python tools/bench/benchmark_ntt_tensor_lengths.py --exe build_cuda_cmake/ntt_tensor_length_probe/ntt_tensor_length_probe.exe --device 1 --sizes 12 18 24 --threads 128 256 512 --operations 2 --skip-gate --output data/experiments/tensor_ntt_cta
```

`--operations 0|1|2`分别为forward、inverse融合pointwise/1/L、两次forward加一次融合inverse的完整卷积。`--sizes`输入k，真实长度为2ᵏ；`--batches`是独立数学操作数量。Tensor CTA可取128/256/512，CUDA对照保持生产512。默认tile_bits=12，短变换自动使用min(k,12)。

Graph诊断以相同一次卷积为首选；需要摊薄短算子的测量分辨率时，才提高串联次数：

```powershell
python tools/bench/benchmark_ntt_tensor_lengths.py --exe build_cuda_cmake/ntt_tensor_length_probe/ntt_tensor_length_probe.exe --device 1 --sizes 8 12 18 24 --operations 2 --execution graph --iterations 1 --skip-gate --output data/experiments/tensor_ntt_graph
python tools/bench/plot_ntt_tensor_lengths.py --inputs data/experiments/tensor_ntt_native data/experiments/tensor_ntt_graph data/experiments/tensor_ntt_batch --output-prefix data/figures/tensor_ntt
```

Graph捕获、分配和参考构造在计时外；每样本报告事件时间除以串联次数，并检查最终全部输出。多次串联重复使用前一次结果，属于吞吐诊断，不代表完整Stage2请求序列。

## 输出口径

- `summary.csv`：中位时间、MAD、host墙钟、batch操作/s、独立操作/s、数学变换/s、相同形状下的CUDA速度比、逐字检查计数和同寿命缓冲容量。
- `operations_per_s=batch/median_seconds`。仅operation=2时是Conv/s；batch_iter_per_s=1/median_seconds。不要把卷积或变换吞吐称为ECM曲线/s。
- `measurements.json`保留每个样本及计划；`*.log`为原始输出；`manifest.json`在EXE旁；`telemetry.csv`为整张GPU采样，不能逐kernel归因。
- 每模式每形状默认7次，建议11次；模式交错执行，没有丢弃的warmup。构造CUDA参考会先执行CUDA路线，因此不声称三条路线具有完全相同的冷缓存状态。
- `host_median_seconds`含提交/同步，以及start事件提交时仍可能存在的填充尾部等待，不是纯CPU开销。主性能比较采用CUDA事件的`median_seconds`。
- 默认batch×length≤2²⁶，四份主数组占32·batch·length bytes。超过限制记为skipped，不静默运行超预算配置。根表、计划表、上下文和Graph对象另计；NVML峰不能与模块峰相加。
- 工具不更改功耗/频率。默认功耗墙下实际频率仍可能下降，结论应标明约束与实测范围。小于几个微秒的差异或零MAD不构成稳定收益证明。

当前实测、限制与后续工作见[Tensor NTT性能](../../docs/performance/NTT_TENSOR.md)。
