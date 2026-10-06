# Auto B2：等待诊断、并发与 NTT 拐点实验

日期：2026-10-06。延续[精确树/G1实现](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_G1_EXACT_TREE.md)。Auto B2 的完整精度/排名门禁仍未通过；本轮没有发布新的 `.cprof`，没有启用未经证实的性能默认值。

## 1. 对照范围与二进制

全部运算用 GPU1 RTX4060 Laptop、PTX3/outer0。三种核心对照分别使用同一二进制，不将不同构建的绝对秒数当作性能改进：

- 原5f4c候选：8次Nsight重放与16条同步/异步诊断。
- pinned实验95f89c…7da4：独立完整CUDA编译498.8秒，25个原始依赖冻结；16条pinned/pageable ABBA。
- 撤回pinned后的33c25f…6c8：CUDA原始依赖/object/toolkit核验后HostOnly复用，仅增加Auto B2配置门禁；16条串行/并发、16条tile ABBA，以及6轮NTT tune。

生产893f6e…f69d未替换。pinned实现、额外测试与原采集工具保存在实验快照，已从生产算术源码撤回；不能把实验gate当作当前源码新功能。当前NTT与上一阶段原始字节相同。

本轮合计96条实际曲线（含8条Nsight采集），均完成算术/必需检查，共943,520个独立GMP系数检查；不把432次纯field卷积、60项实验gate或4次合成率plan-only算作ECM曲线。

## 2. 长等待：未确定根因

相同M8191/D30030/B2=52.5亿/owner0的8次Nsight重放，full为10.190198～10.221345秒，最大无自身GPU活动间隙75.020～76.452毫秒，没有复现此前5秒长尾。此现象只说明跟踪条件下更稳定，不能证明跟踪器修复了调度。

候选判断始终区分CPU/驱动等待、GPU抢占/频率与carry传输路径。正常trace里较长的D2H API主要等待发生在DMA开始之前，而复制完成后通常很快返回；它包含之前排队的GPU工作，不能全部算PCIe传输。尚未捕获异常trace，不能把正常trace的结论外推为长尾根因。

[时间线分析器](D:/code/MPA-OpenCl/tools/bench/analyze_stage2_trace.py:15)按PID/context/address重建malloc/free生命周期，按correlation关联API与DMA；查询与GPU间隙重叠的API，包括跨越间隙边界的调用。此前仅查完全落在间隙内的调用，会漏掉等待。

## 3. 三个没有收益的方向

### 3.1 强制CUDA同步

原5f4c同binary、seed固定的交叉顺序；两B2、默认/`CUDA_LAUNCH_BLOCKING=1`、各4次，共16条。

- G1满根B2=86,426,340：默认中位2.285888秒，同步2.357310秒，同步最长5.893052秒。
- 多G B2=5,250,000,000：默认中位12.457934秒，同步13.712466秒，同步最长17.073609秒。

同步没有消除长尾。增加[原生环境门禁](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:515)，自动规划要求`CUDA_LAUNCH_BLOCKING=0/未设置`；手动对照仍可使用诊断配置。4次plan-only合成率fixture确认旧binary允许0/1，新binary允许0、拒绝1；未执行curve，fixture不是可用成本模型。完整实际auto/manual/queue验收仍须等新profile通过发布门禁。

### 3.2 Pinned同步读回

原型只改变carry counters的读回方式：复用最多1MiB pinned主机缓冲、异步D2H、事件等待，然后读取原计数；检查频率和同步判定保留。所有三处读回都接入，分配失败/过大请求走原路径，异步错误不静默回退。

原型真实NTT gate三种路径（原路径、pinned、强制分配失败）均20项/253,923输出字/bad0，覆盖非零carry毒化、清零、容量/字节溢出和释放。性能ABBA+ABBA共16条，各B2/路径4条：

- G1：2.275054→2.287558秒，慢0.550%；
- 多G：12.387458→13.002520秒，慢4.965%；pinned最长15.387796秒。

实际pin峰值分别5,760与8,096字节，无回退；多G每条1,788次读回。NTT设备payload、叶指纹与因子一致。pinned没有解除延迟瓶颈，不能解释为数据量不足或漏走新路径；生产实现已撤回。

### 3.3 双进程并发

[并发试验](D:/code/MPA-OpenCl/tools/bench/bench_stage2_concurrency.py:89)使用已独立验证的Stage1×12/lcm存档中sigma26、27两条记录，分别提取单条save。固定M8191/B1=1000/D30030/B2=52.5亿、arena512MiB、owner0、chain_min8192、检查策略。串行/双进程/双进程/串行重复两轮，共8批/16curve；逐条叶指纹及因子与串行一致。

每批两条curve，包含进程/上下文启动：

- 串行wall中位24.089505秒，范围23.648517～24.821253；吞吐中位0.083034 curve/s。
- 双进程wall中位25.631433秒，范围24.754008～25.996200；吞吐中位0.078032 curve/s，下降6.024%。

没有启用双进程作为生产默认。该结论对应此几何/进程方式，不能否定其他形状、共享上下文或CPU/GPU流水；也没有证明性能差异全部来自context切换。

## 4. 同时存在的容量与采样内存

8个完整trace均有381次device allocation与381次free，结束余额0。重建所有同时存活的设备malloc：

- 峰值539,494,576 B = **514.502121 MiB**；
- CUDA pinned主机分配峰值24,685,576 B = **23.542000 MiB**；
- 单个NTT模块统计不能代替该生命周期峰值；该峰值也不含module/static、driver/context资源，不能当NVML/进程完整峰值。

两个最大同时存活坐标buffer各135,659,520 B。此几何W=ceil(8191/64)=128、P=phi(30030)/2=2880：

\[
C=P\left\lceil\frac{\max(P,\lfloor 256\mathrm{MiB}/(16W)\rfloor)}P\right\rceil
 =132480,\qquad M_{XZ}=16WC=258.75\mathrm{MiB}.
\]

源码的256MiB点预算按P向上对齐，因此不是严格256MiB上限；该公式不包括其他giant/叶子临时容量。

并发试验每200ms观察GPU1 NVML及自身parent/worker进程。当前private/working set在同次采样求和，再取最大，没有将各进程单独峰值相加：

- 串行最大观察GPU总used 1,384.617 MiB；自身private合计1,602.137 MiB，working-set合计310.672 MiB。
- 双进程最大观察GPU总used 2,537.234 MiB；自身private合计3,041.629 MiB，working-set合计621.547 MiB。

private是Windows private commit，working set包含共享页，以上不是唯一物理RAM用量；采样可能漏过瞬时峰。NVML是整卡量，包含baseline/driver等，不能和malloc峰值直接相减得到精确context成本。

试验按每活跃worker各2GiB VRAM/RAM估算留量，并在外留768MiB VRAM/2GiB可用RAM；仅对D30030/已测三宽度的小工作区实验。低VRAM、低RAM预算两个负例均在启动curve前拒绝。**这不是生产分配lease，不是任意P/B2的总峰值保证**；`process_peak_guaranteed=false`保留。

## 5. NTT tune：tile与outer路径必须一起看

同33c25f、顺序12/11/13/13/11/12，k=16..27、每L各5次测量+1预热。72个长度样本、432次field convolution均每次检查全部L输出，bad0。单batch，测量forward(A)+forward(B)+inverse(product)，不含整数carry、mod-N归约、点生成与GCD。

代表性中位数（跨同tile的两轮中位）：

- L=2^17：t12 0.203264ms，t11 0.171520ms，t11快15.62%；
- L=2^22：t12 3.451392ms，t11 3.253248ms，快5.74%；
- L=2^23：t12 8.710656ms，t11 7.849472ms，快9.89%；
- L=2^24：t12 12.810752ms，t11 19.580416ms，慢52.84%；
- L=2^27：t12 111.693825ms（约8.953 iter/s），t11 155.557381ms，慢39.27%。

改变tile同时改变outer路径。[现有auto策略](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1481)仅对指定GPU/t12及k24..27启用shared radix；其余用M4。t12/k24有2个outer radix（6/6）加tile，共3 forward passes；t11/k24退到M4（4/4/1/4）加tile，共5 passes。不能把速度差全归因于16/32/64KiB tile shared-memory占用。

全局t11必须过整曲线：另16条tile12/tile11 ABBA，G1中位2.260671→2.275932秒（慢0.675%）；多G 11.007255→12.104106秒（慢9.965%，保留全部长尾）。NTT full_peak降低3,406,976 B（3.249 MiB），不构成吞吐改善。stage2大量不同L、nbatch的整数乘法不能直接套单batch field最快配置；生产tile12保持。

## 6. 下一轮与复现

追加的shared radix初探也保留负结果：默认mode2与强制mode1/M8的8条ABBA，多G中位10.524302→10.345478秒（−1.699%），但NTT peak增加44,197,280 B；G1仅快0.643%。再对照M8/M6，M6减少约42MiB表/工作区峰，但M8有长尾，不能把其8.49%中位差当稳定收益。

直接默认策略与mode1/M6再做8条ABBA+ABBA：默认中位10.556921秒，强制M6中位10.780120秒；前一组M6约10.15～10.17秒，后一组出现11.389250/13.153983秒。全部运算/叶/因子一致，NTT peak仅比默认增加72,352 B，但完整中位慢2.114%。未发布新策略，未删除坏样本或以“正常样本”算生产提速。总计24条shared radix初探仍不足以解决Auto B2精度问题。

下一项优先考察连续save的CUDA上下文复用：G1的GPU活动窗口与全进程时长差显著，但需要验证跨sigma缓存/资源释放和队列行为，不能直接复用非重入全局状态。继续定位异常等待，shared radix后续必须扩大位宽/形状与重复覆盖。

Auto B2仍须解决全范围精度/排名、新binary重新标定/发布、G2桥接、生产B1/choose12/泛型N、显存/RAM租约及实际吞吐。不会用部分scope、正常重放或纯NTT提速替代整条曲线的验收。

```powershell
$base = 'build_cuda_cmake/_auto_b2_wait_20261006'
$exe = "$base/native_base/ecm_cuda_stage2.exe"
$study = 'build_cuda_cmake/_auto_b2_g1_20261006/study_latency/measurements.json'
python tools/bench/bench_stage2_concurrency.py --exe $exe --study $study `
  --output run/new_concurrency --repeats 2
python tools/bench/bench_stage2_variants.py --exe $exe --study $study `
  --bits 8191 --d 30030 --b2 86426340 5250000000 `
  --variant tile --values 12 11 --output run/new_tile_ab
python tools/bench/analyze_stage2_trace.py --trace "$base/nsys_00/trace.sqlite" `
  --output run/allocation_wait_audit.json
```

使用新输出目录；候选旁需原始冻结源码manifest。合成profile仅用于配置门禁，pinned重放须使用已撤回的实验快照。完整原始日志、有效保存点、源码/工具快照、Nsight摘要及trace指纹见[证据](D:/code/MPA-OpenCl/docs/data/ecm_stage2_wait_20261006_evidence.json)。
