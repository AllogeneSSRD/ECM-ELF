# CUDA FFT／NTT 吞吐优化试验

## 已执行

1. **减少 Stage1 NTT 的重复算术。** 混合 radix-5 用 PFA 去除交叉旋转因子。直接置换使多曲线变慢，改为本线程共享内存置换后，四档长度的多曲线吞吐均提高。已通过独立 GMP、生产入口和 tune 测试，作为 0.61.5 混合长度的默认数据流。[GPUowl 的 radix-5 PFA](https://github.com/gwoltman/gpuowl/commit/a5d68380)、[Stage1 性能](../performance/STAGE1_MERSENNE_NTT.md#混合-radix-5-吞吐与寄存器)
2. **统一比较单位，再调寄存器。** 两边在同一设备、同一指数上计时完整模平方，曲线吞吐仍用各自真实阶梯步。GPUowl 的原生寄存器调优按形状和 kernel 编译，M262139 的一次短扫报告默认 107.0 µs、选中组合 97.4 µs。ECM-ELF 的整个编译单元限制为 32 或 48 个寄存器均未产生稳定收益，32 还导致 spill。进一步把逆变换的输入指针表改为只读 grid 参数，在 2560 点、30 曲线提高约 19% 吞吐。 [同卡模平方对照](../performance/STAGE1_MERSENNE_NTT.md#同卡完整模平方与-prpll)
3. **验证 Stage2 双目标。** 用两个独立 worker 隔离上下文、输入和输出，小档串行、同时、固定错峰各测 3 次，大档串行与同时各测 2 次。并行改善两目标吞吐，个别组件因争用变慢。先扩充有显存和因子校验的输入矩阵，再决定是否引入生产调度。 [Stage2 双目标数据](../performance/STAGE2.md#双目标吞吐试验)
4. **按热点减少 Stage1 tile 的域乘。** 8 组 Graph node 轨迹给出 pack、tile、outer 和 carry 的耗时。30 曲线的 tile 占 46%–54%。0.61.6 把长度归一化并入逆权重，六档加权形状的多曲线吞吐提高 0.88%–1.91%。[剖析与固定形状对照](../performance/STAGE1_MERSENNE_NTT.md#tile-剖析与逆变换缩放)
5. **复测 GPUowl 最短原生形状。** 固定 `256:1:256`，在 39 万和 52 万位时，纯 FP32 在这四种候选中最快。GIMPS 波前的优先参考顺序仍按 3161、3261、323161 记录，另在 ECM 位宽设计更短形状做独立比较。[GPUowl 最短形状实测](../reference/GPUOWL_NTT.md#最短原生形状复测)
6. **把逆向 warp 的输入留在寄存器。** 0.61.7 成对读取频谱，直接做逐点运算和首六层 DIT。缩小逆向线程块后，64、256、2560 点的 30 曲线吞吐提高 11.53%、9.74%、4.90%。Nsight Systems 已定位这部分耗时变化。Nsight Compute 因设备计数器权限未采到 bank 指标。[正式性能记录](../performance/STAGE1_MERSENNE_NTT.md#逆向-warp-输入融合与块宽)
7. **验证 Stage2 显存准入和阶段错峰。** M5167／M8387 同时启动提高 6.01% 吞吐，同位宽 M8011 两曲线下降 9.71%。按 giant 主循环入口错峰在这两组分别提高 3.33%、下降 2.21%。预算拒绝按预期串行，已知因子与下降结果一致。[Stage2 准入记录](../performance/STAGE2.md#双目标准入与阶段错峰)
8. **按执行阶段错开双目标。** 0.61.8 测得 baby 与 giant 的 kernel 时间覆盖率都在 94% 以上，F 树和求逆有较多可重叠空档。两 worker 进入所选阶段时共享互斥锁，等各自 CUDA 事件完成后释放。宽档仅互斥 baby 和 giant 时吞吐比串行高 11.33%。M8011 将 G 树/fold 与下降也纳入后高 13.50%。同组复测显示主要额外收益来自互斥 G 树/fold。生产默认串行。[阶段时间线与配对对照](../performance/STAGE2.md#双目标阶段互斥)
9. **接入持续双槽及自动准入。** 0.62.0 在每条完成后补位，队列计时从独立的串行与双槽窗口取得并保存到共享 tune。GPU1 三个位宽的 9 条中段吞吐提高 68.79%、31.07%、11.88%，7995-bit 的 17 条复测提高 12.32%。默认 auto 先完成单曲线选型，再比较同路线窗口，复用已有决策。[双槽队列对照](../performance/STAGE2.md#双槽持续队列)

## 下一轮顺序

1. **Stage1 按 kernel 和形状调优。** 优先剖析 30 曲线的前向 tile，以及单曲线的 2560 点 carry 和 16384 点外层变换。变更前后继续对照固定形状、不同批量与整条曲线。评估第三方后端时优先参考 GPUowl 的 3161、3261、323161。原生形状从 393,216 位起。评估数千至数十万位时，需实现更短的形状后再比较进位、完整模平方和批吞吐。[GPUowl 类型与范围](../reference/GPUOWL_NTT.md)
2. **Stage2 按实测收益筛选双目标。** 扩充不同位宽、D/P、实际点块与显存余量的配对，按阶段互相争用和实测收益确定准入与掩码。每个 worker 单独保留日志、检查点、结果与预算，同时限制单任务尾延迟。已有时间线可识别空档，设备性能计数器可用后再测活跃 SM、warp 和内存带宽，确定同一 CUDA context 方案的空间。进入该方案前先清点 `cudaDeviceSynchronize()` 等全设备同步和跨任务共享资源。[GPUowl 的 Windows 显存分页探测](https://github.com/gwoltman/gpuowl/commit/c48a4304)
3. **Stage2：减少固定多项式的重复变换。** 同一模数和卷积形状内，测量缓存 F 与逆多项式频谱的命中率、显存开销和每条曲线节省的前向变换数。按完整 shape 隔离缓存，在预算不足时回退。用固定 D/P、下降叶值、GCD 和完整曲线耗时确认收益，并检验是否挤占双目标的显存余量。

原始证据位于 `data/experiments/fft_ntt_20261111/`、`data/experiments/fft_ntt_20261112/`、`data/experiments/fft_ntt_20261112_tile/`、`data/experiments/stage2_phase_gate_20261112/` 和 `data/experiments/stage2_parallel_0620/`。
