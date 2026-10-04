# ECM Stage2：精确整数 Tensor Core NTT 实验

2026-10-04 采集、2026-10-05 收尾（Europe/London），接续 [xADD6/D](D:/code/MPA-OpenCl/docs/STAGE2_XADD_D_OPTIMIZATION.md) 与 [NTT 尺寸策略/新 D 模型](D:/code/MPA-OpenCl/docs/STAGE2_NTT_SHAPE_D_CALIBRATION.md)。实验在 GPU1 RTX4060 Laptop 8GiB/sm89 串行执行；构建、普通计时和 Nsight 采集分开。GPU0 运行用户的外部生产任务。

## 1. 结果与生产状态

实现了独立的 u8 整数 MMA、132bit 精确累加/Goldilocks 归约，以及符合当前 tile 布局的四层 DFT16。GMP 门禁 **35/0**；数值正确，实际 tile 性能下降，因此保留为实验工具。

主要结果：

- 单独自然序 DFT16，65536 个 16×8 batch：Tensor 相对寄存器 CUDA butterfly 快约 1.2–3.2%，具体取决于方向/CTA。这一接口的收益未延续到实际 tile。
- 实际 t12 tile，最佳 Tensor CTA256：独立复测 N=2²⁴、2²⁵、2²⁶，前向慢 **18.23–18.30%**，带 B/scale 的逆向慢 **5.55–5.60%**，roundtrip 慢 **10.90–10.93%**。
- 两个独立前向任务，CUDA+Tensor 双流比 CUDA+CUDA 双流慢 **9.13–9.17%**。Nsight Systems 中平均 overlap 约 **47.34 μs**，仅占 pair span 的 **0.786%**。

生产仍为 `868c449` 的实现：xADD6、驻留 D 模型、warp tile 与 outer 尺寸策略。`ecm_cuda_stage2.exe` SHA256 `3cf38065e5f88347063f365ac85468475a6624f6e98b52804834d9aef3219f0e`，4042752 bytes；14 项构建依赖与原生产 manifest 复核一致。生产源码没有引用本实验 header。

此前固定工作量结果分别为：xADD full72.830970→68.270377 s（−6.26%）；outer 尺寸策略 full68.813051→67.098667 s（−2.49%）。两轮有各自的对照，百分比不能相加。此次 Tensor 测量范围为独立算子，不含整曲线、carry、REDC、树调度或 CPU 准备；没有新增整 Stage2 性能结论。

## 2. 参考实现与适配边界

固定阅读 Terminus-IMRC/tensor-core-ntt 提交 `6f407daa8a4cef96331511ae86b922b520d7aa33`，Apache-2.0。其 `include/polyarith/cuda/ntt.cuh:71` 将 64bit 值拆成 byte，以矩阵 MMA 累加；`include/polyarith/modular.cuh:326` 的归约器构造函数拒绝 ≥63bit 模数。Goldilocks `q=2⁶⁴−2³²+1` 是 64bit，因此需要独立证明累加与归约范围。[固定矩阵源码](https://github.com/Terminus-IMRC/tensor-core-ntt/blob/6f407daa8a4cef96331511ae86b922b520d7aa33/include/polyarith/cuda/ntt.cuh#L71)、[模数限制](https://github.com/Terminus-IMRC/tensor-core-ntt/blob/6f407daa8a4cef96331511ae86b922b520d7aa33/include/polyarith/modular.cuh#L326)。本仓库实现未导入该库源码。

使用 PTX `mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32`。lane 的矩阵 fragment 依照 NVIDIA 定义组织，整 warp 执行相同 MMA；寄存器结果随后由 CUDA 指令重构并归约。[PTX 整数 fragment 合同](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-integer-type)。该实验适用 sm80+，本轮只测 sm89。

## 3. 精确计算：64 个 byte MMA 与 132bit 和

对 `A[16×16] · B[16×8]`，一个输出为 `S=Σ(k=0..15) Aik·Bkj`。输入 canonical，均在 `[0,q)`：

```text
S ≤ 16(q−1)² < 2^132
A = Σ(a_u · 2^(8u)), B = Σ(b_v · 2^(8v)), u,v=0..7
s_d = Σ(u+v=d) Σ(k=0..15) a_ik,u · b_kj,v, d=0..14
s_d ≤ 16·8·255² = 8,323,200
```

每对 byte 调用一次 MMA，共 `8×8=64` 次 warp MMA；按 15 个对角依次累加、传播 base256 carry。carry 最大不超过32767，`s_d+carry` 仍低于 signed32 上界。无需同时保留15个对角的全部 accumulator。

第14个对角处理后，剩余 carry 的 low8 写入 high64 的最高 byte；`top=carry>>8` 为完整和的 bits128..131，范围0..15。保持 `S=lo+2⁶⁴·hi+2¹²⁸·top`，利用：

```text
2^64  ≡ 2^32−1 mod q
2^128 ≡ −2^32   mod q
result = gl_sub_dev(gl_reduce(lo,hi), top<<32)
```

`gl_reduce` 复用当前任意128bit Goldilocks fold，结果 canonical；减数 `top<<32<q`。丢弃 top 会得到错误余数，roundtrip 单独成功不足以证明正确。GMP 直接保存完整和，并独立比较最终余数及 `floor(S/2¹²⁸)`。

代码：[MMA:14](D:/code/MPA-OpenCl/tools/bench/ntt_tensor_goldilocks.cuh:14)、[对角/carry:28](D:/code/MPA-OpenCl/tools/bench/ntt_tensor_goldilocks.cuh:28)、[132bit 归约:45](D:/code/MPA-OpenCl/tools/bench/ntt_tensor_goldilocks.cuh:45)、[GMP dot:115](D:/code/MPA-OpenCl/tools/test/ntt_tensor_goldilocks_probe.cu:115)。

## 4. 实际 tile：布局、融合和访存

定义 `L=2^k` 为 NTT word 数，`T=2^t` 为 tile word 数，避免与待分解整数 N 混淆。当前目标 t12，即4096 word/CTA。

每 warp 处理128 word，相当于两个64点变换：低6层中的高2层继续用 CUDA shuffle；低4层映射为一个 `16×16 · 16×8`。A 为共同根矩阵，B 为8个输入向量。forward 的高于6层部分沿用生产 shared radix4，然后 CUDA 高2层、MMA 低4层，直接写出原 DIF 的 bit-reversed 顺序。inverse 先在读入 tile 后融合 `a·B·scale`，MMA 低4层，再做 CUDA 高2层和其余 shared DIT，输出自然序。

A 根离线预排为 `[byte][half][lane]`，每方向512个u32，即2048 B。运行中每 lane 连续载入自己的根 fragment，只拆分 B；预排能省根拆分，仍需要64次 MMA及 carry/归约。t6 测试会将第二个64点组补零并丢弃其输出，确保 warp 一致参与；下面计量公式针对 t≥7 的完整128 word组。

静态地址分析还有一个布局候选：forward读B的固定i对应 `sm[offset+4·lane+i]`。按32个4B bank模型，其64bit word的起始bank为 `(8·lane+2i) mod32`，只有4个起始bank，每个对应8lane；原CUDA warp6读连续lane位置。实际64bit指令分拆和冲突程度须由计数器或布局A/B验证，不能将这个静态映射直接记为实测bank冲突率。下一TC版本可先调整shared输入排列或warp重排，单独衡量其成本。

tile 的跨 batch stride、B 只读、原位 a 和原 DIF/DIT 排列均保留。独立 tile 参考与 benchmark 的 scale 为 `1/T`，方便验证每个独立 tile；生产完整 NTT 使用 `1/L`，接入时应传生产 scale。当前实验没有接入 outer planner 或 arena cache。

代码：[packed roots:67](D:/code/MPA-OpenCl/tools/bench/ntt_tensor_goldilocks.cuh:67)、[warp6:93](D:/code/MPA-OpenCl/tools/bench/ntt_tensor_goldilocks.cuh:93)、[真实 tile:148](D:/code/MPA-OpenCl/tools/bench/ntt_tensor_goldilocks.cuh:148)、[根生成:158](D:/code/MPA-OpenCl/tools/test/ntt_tensor_goldilocks_probe.cu:158)、[GMP tile oracle:175](D:/code/MPA-OpenCl/tools/test/ntt_tensor_goldilocks_probe.cu:175)。

## 5. 计算量、容量和传输公式

### 5.1 运算计量

一个16×8输出块有128个word、2048个完整64bit乘积项。64次 MMA 合计执行 `64·16·8·16=131072` 个 byte MAC，即 **1024 byte MAC/输出word**。整次 tile pass 为 `L/2` 次 warp MMA、`1024L` byte MAC、`L` 次132bit重构/归约。这是源码操作数，未测硬件 cycle 或实际 issued instruction 数。

生产 warp tile 的 stage0 已去单位根乘法，stage1 已将 ±2⁴⁸ 变为 shift/fold。因此 forward 通用 `gl_mul` 调用数为 `(t−2)L/2`，带B/scale的 inverse 为 `(t−2)L/2+2L`。Tensor 替换 stage0..3 后分别为 `(t−4)L/2` 与 `(t−4)L/2+2L`，另增加上述 MMA/重构。t12 三变换卷积的 tile 部分：生产 **17L** 次通用模乘；候选 **14L** 次通用模乘 + **1.5L** 次 warp MMA + **3072L** byte MAC + **3L** 次累加和归约。运算类型不同，不能把它们的计数直接相加评价速度。

若一次多项式乘法有 b 个 slice，将上述计数乘 b。完整曲线按实际乘法集合 `C(B2,D,S)` 求和，`P=φ(D)/2`、当前 `I=floor(B2/D)+2`、`G=ceil(I/P)`；`S=bitlen(N)`。各乘法 `m=max(na,nb)`、`slot_bits=2S+max(1,ceil(log2 m))`，选择满足 `m·sw·(2^bpw−1)²<q` 的 bpw，`sw=ceil(slot_bits/bpw)`，`L=nextpow2(2m·sw+1)`。例如总 byte MAC 为 `3072·Σ(c∈C)b_cL_c`。B2 通过 I/G 和树/折叠调用数影响工作，不能用 π(B2) 代替。

### 5.2 Global/shared 与显存

主数组 forward 读写 `16L B`，inverse 读A、读B、写A为 `24L B`。两个forward+inverse的 tile payload 为 **56L B/slice**，两实现相同，根表读取另计。这是逻辑payload，cache命中与实际DRAM流量没有计数器证据。t12 shared数组为 **8T=32768 B/CTA**；两实现同容量，当前布局保留同样的高层 shared 交换。

Probe 的 `TileBuffers` 持有 A/B 两个数组、双向 tile 表、两个 packed root 表，以及各4个tile的input/B/expected池：

```text
单任务设备requested payload = 16L + 16(T−1) + 4096 + 96T + 8
                            = 16L + 112T + 4088 bytes
双任务设备requested payload = 2·(16L + 112T + 4088)
```

t12单任务 L=2²⁴/2²⁵/2²⁶实测peak为268898296/537333752/1074204664 B，即256.441/512.441/1024.441 MiB；双任务2²⁴/2²⁵为537796592/1074667504 B，即512.883/1024.883 MiB。结束 `live_bytes=0`。计量涵盖本工具的cudaMalloc请求，未包含context/module/driver，未测NVML总峰。

tile Tensor forward/inverse寄存器 **74/76**，LOCAL0；生产CUDA两个方向 **40**，LOCAL0。资源API在32KiB shared下允许TC CTA128/256各3块/SM、CTA512只有1块/SM；生产CUDA CTA512为3块/SM。TC CTA256即使可驻留3块，也只有768线程，CUDA CTA512对应1536线程。上述是容量上限，实际occupancy/cycle/带宽仍未测。前阶段 NCU 2026.2.1 返回 `ERR_NVGPUCTRPERM`，本阶段未重复请求相同权限条件。

### 5.3 Host 数据与边界传输

性能probe使用GPU将4个GMP参考tile周期复制到全部L输出，host长期向量payload **112T+4080 B**，t12为462832 B；GMP初始化临时对象及vector allocator开销另计。初始化H2D也为该payload，和L无关；每次输入fill向设备A/B写16L B。计时内H2D/D2H均为0；计时外逐字校验A/B所有L输出，比较器扫描至少16L B，D2H仅8 B错误计数。该构造适合隔离算子吞吐，真实Stage2的数据生成/传输成本见流水线报告。

自然序矩阵性能probe的device payload为 `2048·batches+37112 B`，包含两个主数组、矩阵/根表和16组参考pattern；batches65536约128.035MiB。完整门禁使用更小的fixture并回传全部结果及top，不能把其传输口径混入性能probe。

代码：[计账:7](D:/code/MPA-OpenCl/tools/test/ntt_tensor_goldilocks_probe.cu:7)、[TileBuffers:258](D:/code/MPA-OpenCl/tools/test/ntt_tensor_goldilocks_probe.cu:258)、[资源查询:447](D:/code/MPA-OpenCl/tools/test/ntt_tensor_goldilocks_probe.cu:447)。

## 6. 正确性门禁与 shared 屏障修复

最终独立门禁35/0，原fixture保持：

- 矩阵/自然序频谱/逆向/roundtrip共1920 cases、**1536000 word**。另外对614400个输出验证完整top，其中370400个top非零；覆盖0、1、q−1、最高位、边界交错、dense、sparse、carry8类，batches1/3/4/17，CTA32/64/128/256，独立output和B=C alias。
- tile forward/inverse/roundtrip、B只读、stride保护共2560 cases、**11228160 word**。t6/8/10/12、batches1/3、tiles1/3、padding17；TC CTA128/256/512，CUDA直接调用当前生产warp tile CTA512。GMP DIT oracle使用独立整数算术，不依赖GPU变换结果建立参考。
- 矩阵和tile各故意翻转一个输出；均比较出bad1并返回3，正常执行bad0、退出0、结束无live分配。

首次真实tile实现有随机数值错误，集中于CTA512/t10、t12；原生产tile对照正确。按diagnose流程提出并区分shared时序、低层置换、packed roots/scale三个假设，最小日志显示t12全1样本index1024应为0却有0x40/0x60等变化。

根因：复用的 `tile_radix4_block` / inverse helper将CTA屏障交给caller；新wrapper遗漏了每次调用后的 `__syncthreads()`。原生产caller已有屏障。仅补caller屏障后所有原fixture通过；CTA512/t10/t12 fixture保留，调试日志移到ignored debug证据目录并从最终源码删除。预防方式是明确helper的caller同步合同，并继续保留真实tile调用层的回归覆盖。[修复屏障:164](D:/code/MPA-OpenCl/tools/bench/ntt_tensor_goldilocks.cuh:164)、[inverse:174](D:/code/MPA-OpenCl/tools/bench/ntt_tensor_goldilocks.cuh:174)、[生产caller:1219](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1219)。

脚本另外核对exe/构建source SHA、完成行数、case/word计数、资源条目和LOCAL0；bench校验实际shape/operation/order，mixed限定forward。初期CRLF日志解析把`bad=0\r`误判，改用非空白字段解析后重跑；没有放宽数值比较。

## 7. 性能：独立自然序接口、真实 tile、留出尺寸

所有性能测量使用同一个最终exe、固定数据/检查；每个run先1warm，再3次CUDA-event，逐次在计时外检查全部输出。两个模式采用ABBA+BAAB共8run，每模式4个样本。数据生成、GMP、规划、分配、校验不计入event。没有统计置信区间；不以单次kernel或最优样本代表整Stage2。

### 7.1 自然序 DFT16

batches=65536、CTA64：forward CUDA .606379 ms / Tensor .596053 ms（1.70%）；inverse .609536/.594773 ms（2.42%）；roundtrip1.195435/1.181355 ms（1.18%）。其他CTA的完整结果保存于matrix_sweep。batches1024时约16–32 μs，变异明显，部分CTA变慢；不据此建立阈值策略。

naive dense CUDA DFT16单样本约3.06ms，约5倍差距来自控制算法本身的16项dot与logarithmic butterfly区别。优化候选的主对照采用寄存器/shuffle butterfly和实际生产tile。

### 7.2 实际 tile CTA sweep

L=2²⁴/t12，CUDA对照始终CTA512；下面列TC CTA和各组mean，顺序为CUDA/TC，单位ms：

- TC128：forward2.774613/3.595349（慢29.58%）；inverse3.707136/4.558592（慢22.97%）；roundtrip6.469632/8.111786（慢25.38%）。
- TC256：forward2.769066/3.277568（慢18.36%）；inverse3.705685/3.914069（慢5.62%）；roundtrip6.472875/7.177643（慢10.89%）。
- TC512：forward2.775467/4.231510（慢52.46%）；inverse3.706027/5.058219（慢36.49%）；roundtrip6.470400/9.212075（慢42.37%）。

锁定最佳TC256后，在独立运行中复测k24，并留出k25/26；每种shape仍8run：

- k24：forward2.775296/3.283200；inverse3.706795/3.914325；roundtrip6.470400/7.177642。
- k25：forward5.514581/6.520149；inverse7.363840/7.772587；roundtrip12.874411/14.278571。
- k26：forward10.994091/13.006251；inverse14.682965/15.499947；roundtrip25.693696/28.493824。

三个尺寸稳定为负收益。额外MMA、拆分/carry/归约、低层输入输出组织及寄存器压力可能共同贡献；缺少计数器时不能判定各项比例。k27未测，生产全NTT/整曲线未使用该内核。

## 8. CUDA/Tensor 并发吞吐与 Nsight

两个任务持有独立A/B，使用nonblocking stream，以共同start event开始；serial在第二流等待第一流end，parallel不设该依赖，最终join两个end。模式0/1为CUDA+CUDA serial/parallel；2/3为CUDA+TC serial/parallel。顺序 `0,2,3,1,1,3,2,0,1,3,2,0,0,2,3,1`，每模式4run，每run1warm+3次pair，两个任务全部校验。[调度:316](D:/code/MPA-OpenCl/tools/test/ntt_tensor_goldilocks_probe.cu:316)。

TC CTA256独立holdout（模式0/1/2/3，单位ms）：

- L2²⁴：5.552565 / 5.527723 / 6.059008 / 6.032470。CC并发省.447%，C+TC并发比自身serial省.438%，相对CC并发慢9.13%。
- L2²⁵：11.045728 / 11.006981 / 12.050347 / 12.015800。对应.351%、.287%、慢9.17%。

吞吐为 `2/pair_seconds` 个独立tile任务/s；这些任务不能换算为ECM curves/h。每个任务本身已覆盖许多CTA，第二个kernel共享SM的register/shared/调度/访存资源，独立指令管线并不自动形成可叠加的吞吐。

Nsight Systems2026.1.3独立采集同exe/k24/t12/TC256，128个目标kernel组成64对；每run除warm后，每模式12对：

- CC serial：pair span5.543177ms，overlap0；parallel5.516544ms，平均overlap47.456μs（.860%）。
- C+TC serial：pair span6.053774ms，overlap0；parallel6.025034ms，平均overlap47.342μs（.786%）。
- C+TC的TC kernel平均serial3.278802ms、parallel3.304931ms；两种parallel模式都只在小段尾部重叠。此观察支持当前任务主要顺序消耗资源，并发有轻微争用的推断。

定义overlap为两kernel时间区间交集，pair span为最早start至最晚end。它证明kernel事件重叠；不证明同一个SM的CUDA/Tensor指令同时issue。无trace污染的event测量和profile单次时长分别报告。

## 9. 冻结产物、入口和证据

最终probe SHA256 **`2dd2c23055547b500786a98cb5288c2053a6f344d151a86679fb97396a88b52a`**，2797056 bytes，sm89/CUDA13.3.73，构建34.24s。[manifest及源码快照](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_tensor_20261004/release/manifest.json)包括5项编译依赖；另保留最终Python driver快照。最后修正的是mixed CLI非forward参数拒绝，内核未改；全部数据已用此exe重新采集。

复现入口（仓库根目录，选择空的新输出目录）：

```powershell
.\tools\build\build_ntt_tensor_probe.ps1 -Build build_cuda_cmake\ntt_tensor_probe -Arch sm_89
python tools\test\test_ntt_tensor_goldilocks.py --exe build_cuda_cmake\ntt_tensor_probe\ntt_tensor_goldilocks_probe.exe --device 1 --output build_cuda_cmake\tc_gate_new
python tools\bench\bench_ntt_tensor_goldilocks.py --exe build_cuda_cmake\ntt_tensor_probe\ntt_tensor_goldilocks_probe.exe --device 1 --kind tile --sizes 24 25 26 --threads 256 --operations 0 1 2 --output build_cuda_cmake\tc_tile_new
python tools\bench\bench_ntt_tensor_goldilocks.py --exe build_cuda_cmake\ntt_tensor_probe\ntt_tensor_goldilocks_probe.exe --device 1 --kind mixed --sizes 24 25 --threads 256 --output build_cuda_cmake\tc_mixed_new
```

`--kind matrix`的sizes是log2 batches，缺省16；tile/mixed缺省24。mixed仅forward，其他operation会拒绝；tile支持0forward/1inverse(B)/2roundtrip。原始CLI有同名模式，参数由[main:461](D:/code/MPA-OpenCl/tools/test/ntt_tensor_goldilocks_probe.cu:461)解析。工具的数学reference/填充/校验不进入event。

本机证据：[35项](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_tensor_20261004/release_gate/summary.json)、[matrix sweep](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_tensor_20261004/release_matrix_sweep/measurements.json)、[tile sweep](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_tensor_20261004/release_tile_sweep/measurements.json)、[tile holdout](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_tensor_20261004/release_tile_holdout/measurements.json)、[mixed sweep](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_tensor_20261004/release_mixed_sweep/measurements.json)、[mixed holdout](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_tensor_20261004/release_mixed_holdout/measurements.json)、[trace分析](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_tensor_20261004/release_profile_analysis.json)、[Systems原始trace](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_tensor_20261004/release_nsys_mixed_24_256.nsys-rep)。build/测量/调试快照为ignored，随Git保存的是源码、driver与此报告。

## 10. 下一轮优先级

1. **CUDA低层根专用算术。** 当前只特化stage0/1；stage2/3的8/16次单位根也有移位结构。生成的 `r16=7^((q−1)/16)=q−2⁶⁰`，`r16^5=2¹²`；根可用±2的幂或少量移位差表示，例如 `2⁷²≡2⁴⁰−2⁸`。先独立证明所有forward/inverse根的canonical运算，再测真实warp tile；保留任意table的通用fallback。这能减少一般64×64乘法，额外fold/branch仍需实测，不能预设收益。
2. **逐项解释host提交/准备空隙。** P3 trace近似Stage2无本进程GPU事件9.57s、14.41%；定位实际CPU准备、allocation、检查依赖和同步边界后再决定GPU化/异步化。同步copy的host等待约20s大部分在等GPU，不能全部计为PCIe传输或可删除CPU工作。
3. **多曲线流水及共享workspace。** 两个当前约5GiB工作集不能直接复制到8GiB GPU。方案需要独立Q/Γ/F/finv/H/oracle/carry与共享NTT workspace lease；当前全局模数/统计/default stream入口尚不支持并发调用。一个root owner按 `8W(9P+8)+48` B计账，W70/P126720为638673328B；两owner、共享3GiB主workspace、所有表/临时/检查容量及pinned/host池均需按峰值生命周期求和。测量目标是相同工作/检查的curves/h。
4. **Tensor后续结构化方案。** 若继续研究，先验证§4的shared输入布局，再利用固定根稀疏/符号/移位结构减少64个byte MMA，并降低寄存器压力。当前通用64bit矩阵映射已有明确负证据；扩大batch或直接混合两流不会消除其成本。

公平Prime95对照仍需要相同N、sigma/Q、B1/B2、实际覆盖、检查强度、线程和冷热启动条件。旧CPU90.460s单样本不作为本阶段已经超过Prime95的证据；长期目标继续推进。
