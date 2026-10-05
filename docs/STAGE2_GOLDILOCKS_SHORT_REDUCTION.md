# Stage2：低层单位根实验与 Goldilocks 短归约

日期：2026-10-05。GPU1：RTX 4060 Laptop / sm89 / CUDA 13.3.73。延续六模乘 xADD、按 NTT 长度选择 outer 的优化。这里的归约模数是 NTT 素数 `q=2^64−2^32+1`；ECM 的模数记为 `N_ecm`，其 S4 长除法是另一条算术路径。

## 1. 本轮结论与边界

低层单位根特化的三种实现分别测量。按根值分支比原内核慢；按指数移位仍慢；为移位引入短归约后，真实 tile roundtrip 在独立复测和 k25/26 留出长度上快 2.19%–2.31%。随后将短归约用于所有 device Goldilocks 运算，完整卷积在 k24..27 快 **34.06%–36.63%**，没有改变长度、packing、pass 数或检查量。

新开关 `NTT_GL_SHORT_REDUCE=0/1` 支持同 binary 比较。当前缺省 **0**，保留原四次 fold；host 参考计算继续用原算术。低层根特化只在独立探针中，尚未接入生产 tile。现有生产 exe 是上一阶段的 `868c449` 构建，不代表本轮修改后的工作区源文件。

固定 D/Q 的八条 Stage2 交叉测量：完整均值 **69.778377→59.482459 s，快14.76%**；主循环快17.40%。D 模型适用范围与阶段验收见后续章节。旧 D 系数不能直接用于新归约。

## 2. 源文件与调用边界

- 任意 128 位短归约：[ntt_goldilocks_reduce.cuh:5](D:/code/MPA-OpenCl/tools/bench/ntt_goldilocks_reduce.cuh:5)。这是生产 NTT 也可复用的最小数学原语。
- device 模式和每设备配置：[ntt_poly_probe.cu:97](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:97)、[100](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:100)。统一分派：[gl_reduce:144](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:144)；模乘仍保留低/高 64 位乘积：[178](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:178)。fuse 初始化、arena 获取及 selftest 均配置选定模式，包括缓存命中。
- 三种低层根方法：[ntt_small_roots.cuh](D:/code/MPA-OpenCl/tools/bench/ntt_small_roots.cuh:1)。它不被生产引擎包含。复制当前 tile 的 warp/shared 边界，仅改变 stage2/3 twiddle 算术。
- 独立 GMP 根/128 位/tile 参考：[ntt_small_roots_probe.cu:43](D:/code/MPA-OpenCl/tools/test/ntt_small_roots_probe.cu:43)、[91](D:/code/MPA-OpenCl/tools/test/ntt_small_roots_probe.cu:91)、[108](D:/code/MPA-OpenCl/tools/test/ntt_small_roots_probe.cu:108)。参考复用 Tensor 实验的 **host** oracle，通过 main guard 包含；本轮 CUDA 候选没有发射 MMA。
- 完整卷积同 binary 开关：[ntt_coop_outer_probe.cu:24](D:/code/MPA-OpenCl/tools/test/ntt_coop_outer_probe.cu:24)、[bench_ntt_short_reduce.py](D:/code/MPA-OpenCl/tools/bench/bench_ntt_short_reduce.py:1)；真实 tile 测量：[bench_ntt_small_roots.py](D:/code/MPA-OpenCl/tools/bench/bench_ntt_small_roots.py:1)。
- 构建依赖包含新 header；probe 的 manifest 保存 exe 和原始源码 SHA256。运行前后校验 hash，编译、正确性检查和性能测量串行执行，均选择 GPU1。

## 3. 任意 128 位输入的短归约证明

令 `ε=2^32−1`，`x=lo+hi·2^64`，`hi=h1·2^32+h0`，其中 `0≤lo,hi<2^64`，`0≤h0,h1<2^32`。利用：

```
2^64 ≡ ε (mod q)
2^96 ≡ −1 (mod q)
x ≡ lo − h1 + h0·ε (mod q)
```

先计算 `minus=lo−h1` 的 unsigned64 差。如果借位，实际加了 `2^64`；再减 `ε`，就变成加 `q`，保持同余。检测借位用 `minus>lo`。得到 `a=minus−borrow·ε`。

`b=h0·ε=(h0<<32)−h0` 可直接用 64 位表达，且 `0≤b≤ε²<q`。计算 `sum=a+b`，若进位则补回 `ε`，因为丢弃的 `2^64` 与 `ε` 同余。检测进位用 `sum<a`。

进位时 wrapped sum 至多 `b−1≤ε²−1=2^64−2ε−2`，所以加 `ε` 不会再次进位。无论是否进位，`value` 都是一个 unsigned64 值；由于 `2^64<2q`，最后至多减一次 `q` 即得到 `[0,q)` 的唯一结果。此证明不要求输入是两个 canonical field word 的乘积，覆盖任意 `(lo,hi)`。

原函数源级运行四次 fold，每次保留 carry/borrow 与高半；新函数用高 32 位一次减法、低 32 位一次移位差、两处 carry/borrow 修正和一次规范化。模乘的 `a*b` 与 `__umul64hi(a,b)` 保持。减少的是归约依赖链，不能写成减少所有模乘次数。没有可用 NCU 硬件计数器，因此不报告周期数或实际指令吞吐。

## 4. 低层单位根方法及负结果

这里的 stage2/3 指 tile 内 `st=2/3`、half=4/8 的蝶形层。`γ=2^12` 的阶为 16，`γ^8=−1 mod q`。当前标准根由 7 生成，`r16=γ^13=q−2^60`，逆根为 `γ^3`。stage3 的指数是 `(13j 或 3j) mod16`；stage2 的指数翻倍。将指数分成符号和 `12·(e&7)` 位左移，可去掉相应乘法和根表读取。

最大移位 84 位，`v·2^84` 需要 **148 位**。保留最高 20 位，利用 `2^128≡−2^32 mod q` 进行修正。截掉超过 128 位部分会错误，不能只调用普通 128 位归约。

方法1按实际根值分支，支持任意根表，其他权值回 `gl_mul`。方法2按索引算指数，要求标准根，使用原归约。方法3采用方法2的索引方式，但低 128 位改用短归约。

v1 / k24 / t12 / CTA512，串行 ABBA+BAAB，每模式四个 run、每 run 一次 warm 和三次 event 采样：

- 方法1：forward 2.772309→4.602197 ms，慢66.01%；inverse 3.705173→5.473707 ms，慢47.73%；roundtrip 6.470315→10.065493 ms，慢55.56%。分支省乘法没有转化成实际收益。
- 方法2：forward 2.768896→2.939221 ms，慢6.15%；inverse 3.705771→3.786581 ms，慢2.18%；roundtrip 6.476288→6.727253 ms，慢3.88%。移位仍调用原归约，新增指数/宽移位准备抵消了省乘法。
- 方法3锁定后独立复测：k24 roundtrip 6.469803→6.320640 ms，快2.31%；留出 k25 12.884565→12.596309 ms，快2.24%；k26 25.672789→25.110443 ms，快2.19%。forward 快1.66%–1.68%，inverse 快2.49%–2.76%。

这些是 tile 局部实验，尚未与全局短归约叠加；不能将两组百分比相加。下一步根特化需以全局短归约为对照重新测量。

证据：[v1](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/sweep_v1/measurements.json)、[v2](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/sweep_v2/measurements.json)、[锁定方法3留出复测](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/holdout_v2/measurements.json)。对应二进制/源码快照分别在 `v1`、`v2_reduce` 目录；它们不会随着当前源码修改而变化。

## 5. 全局短归约、寄存器生命周期与完整 NTT

先用非 volatile 的 device constant 保存模式。全部数值检查通过，但真实 forward tile 的 `LOCAL=8B`，触发资源门禁：35通过/1失败；没有进入性能采样。仅将模式改为 volatile 后，forward 的 local spill 归零。40 register / CTA512 / 32KiB shared 的容量 API 上限仍是 3 CTA/SM。模式寄存器被长时间保留是原因推断；这里证明的是单变量改变消除了 local allocation，未获得运行时 occupancy。

最终真实 tile 同 binary 模式0/1，t12/CTA512：

- k24 forward 2.882816→1.910870 ms，快33.72%；inverse 3.732736→2.375424 ms，快36.36%；roundtrip 6.604203→4.246699 ms，快35.70%。
- k25 roundtrip 13.239979→8.454912 ms，快36.14%；k26 26.236757→16.939691 ms，快35.44%。

完整稀疏卷积包含两个 forward、pointwise、inverse 和归一化，使用原生产 warp tile 与现有尺寸策略。每次在 event 外验证全部 `L=2^k` 输出，模式顺序0/1/1/0/1/0/0/1：

- k24：21.062912→13.347670 ms，快36.63%。
- k25：44.785579→29.371136 ms，快34.42%。
- k26：90.802007→58.692352 ms，快35.36%。
- k27：190.308439→125.495125 ms，快34.06%。

固定 outer2 / warp1 / t12 / M4 / compact，planner 在 k24 选 M6、k25..27 选 M8。两个模式的实际 pass、M、coop 一致。初始化、填充、计划和检查排除于 event 时间，不能代表完整曲线。新同 binary 控制包含 runtime 模式判断，历史未带此分派的 binary 需要另行测量，以避免控制变慢放大收益。

上一阶段保留的 probe 独立串行复测（同GPU1、同实际pass/M），其 k24/25/26/27 模式2均值为20.447659/43.468715/88.130135/184.223915 ms。新模式0比它慢3.01/3.03/3.03/3.30%；新短归约相对它仍快 **34.72/32.43/33.40/31.88%**。这是两套二进制的独立采样，不是交叉顺序或置信区间。旧probe SHA由上一阶段GMP验收记录复核；[完整参考数据与限制](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/old_probe_reference.json)。保留这个分派开销记录，后续默认提升和优化不能只引用36.6%的同binary数字。

最终 tile exe SHA256 `79745b0c7e2da53eee704c476dcc94e0f9bdec4e4172da2edf94d52b1c1a914e`；完整 probe `5bcdf7913232f1cbc6dd1c5c0cd2050ddcdf1d12adb4d76ed31c873d8577d756`。构建分别39.69/31.58 s。源码/构建 manifest 和 snapshot 保存在各自目录。

证据：[失败的资源门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/tile_gate_global_0/summary.json)、[最终完整卷积](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/volatile_full_sweep/measurements.json)、[最终 tile](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/volatile_tile_sweep/measurements.json)。

## 6. 正确性与空间/传输合同

最终短归约模式0和1各通过：

- 独立低层根/tile 门禁 **36/0**。三种根方法各98,544字，任意权值回退2,048字，任意128位100,064字；4种tile方法总计11,315,456输出/只读/保护字。t5/6/8/10/12、batch1/3、tile1/3、padding17、八类边界/随机数据；故意损坏被拒绝，live0/local0。
- 完整 NTT 门禁 **13/0**。独立GMP forward/DIT 96组合/27,131,904字，缓存切换3,145,728字，88项策略边界与覆盖检查；device 模乘 selftest 200,000项；故意损坏被拒绝，所有8个outer template local0。

设 `L=2^k` 为 field word 数，`T=2^t` 为 tile word 数：

- 真实 forward tile 全局主数据读写 `16L B`；inverse 加 pointwise 读 `24L B`。共享数据每 CTA `8T B`，t12为32KiB。短归约没有新增表、全长数组或 pass，主访存量保持。
- 每方向一个 radix2 tile 的 butterfly 数为 `Lt/2`。forward 理论 twiddle 调用同数量，inverse另有 `2L` 次 pointwise/scale 模乘；省单位根等既有特化会降低实际 `gl_mul` 数。全变换 butterfly 数 `Lk/2`，卷积三变换总数 `3Lk/2`；新全局归约不改变这些数量。
- 独立 probe 为比较保留 A/B 两全长数组，并复用 Tensor probe 的参考表/周期样本：逻辑显存 `16L+112T+4088 B`。k24/t12为256.441MiB，k26为1024.441MiB。probe 保留的两张 packed Tensor 根表只是 oracle fixture 的复用开销，不是新生产内存。
- probe 的 host 小样本/表量随 `T` 增长，不保存完整 host `L` 数组；初始化 H2D `112T+4080 B`，event内 host传输0，计时外每次只回传8B错误计数，全部输出由device比较。完整卷积探针的输入/检查方式另记在其源码和 provenance。
- 生产新状态仅每设备4B constant 模式，host 小型设备→模式 map；首次或模式切换 H2D4B，无新分配。每次获取arena查询当前device并查模式缓存，这一host开销必须由完整Stage2墙钟包含。

NCU 2026.2.1仍无有效硬件计数权限；上述字节为算法 payload 或明确分配量，不是 DRAM 实测流量、NVML 总峰或周期数。阶段计时有嵌套，不能将分项节省重复相加。

### 6.1 完整 Stage2 门禁及超时记录

新归约的完整门禁 **188/0** 通过，涵盖冻结因子、普通模数、packing、alias、carry、尾块、workspace、内存拒绝后的回退、output window、所有返回 word 与故障注入。另有小曲线确认实际模式1。

原模式完整门禁在182项通过、无失败时达到协调脚本900秒时限，未完成最后六项；日志保留，不记作188/0。延长完整门禁时限后，新模式完成全部188项。两模式的独立36项与13项NTT门禁均完整通过；完整大形状A/B还独立核对原模式和新模式的强制自检与GMP样本。计时门禁的CPU回退耗时排除于性能数据。

证据：[新模式188项](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/stage2_full_gate_1_run.log)、[原模式超时前182项](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/stage2_full_gate_0_run.log)、[实际模式确认](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/stage2_mode_1_run.log)。测量前冻结Stage2 exe SHA256 `17bf4813000e674a2046cb3765fa39e30085c816ea46fa0ac18b884fa78e58a2`，compile578.0/link3.5s；[六项编译依赖manifest与快照](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/stage2/manifest.json)。

## 7. 固定 D/Q 的完整 Stage2 A/B

M4423=`2^4423−1`，sigma26，Stage1 extra12，B1=1000，B2=2011326186870，D1231230。`P=φ(D)/2=115200`，`I=floor(B2/D)+2=1633592`，`G=ceil(I/P)=15`。固定 batch64、chain64、arena6300MiB、NTT outer2/warp1/t12/M4/compact；D模型关闭。仅改变 `NTT_GL_SHORT_REDUCE`。Stage1 Q由独立参考摘要约束，SHA256 `33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`。

串行ABBA+BAAB，共八条、每模式四样本。原归约 full：69.968788/72.586422/68.123219/68.435079 s；短归约：61.246181/59.259161/58.873466/58.551029 s。均值 **69.778377→59.482459 s（快14.7552%）**；main **55.356895→45.725436（快17.3988%）**；init14.421482→13.757024（快4.6074%）。ABBA组71.277605→60.252671，BAAB组68.279149→58.712248；两个顺序组均改善。样本存在初始化波动，没有建立置信区间。

分项均值：giant11.785750→11.797750 s保持；G树21.285000→15.638500（快26.53%）；fold9.912500→7.586000（快23.47%）；descent8.204250→6.923250（快15.61%）；inverse2.448250→2.017750（快17.58%）；accum.179500→.179250基本保持。S4模`N_ecm`归约未改，事件均值2.075250→1.990750存在波动，不宣称本轮优化了S4长除法。

以新full为分母，init约23.13%、giant19.83%、G树26.29%、fold12.75%、descent11.64%、inverse3.39%、accum.30%。阶段计时有嵌套/桥接，以上用于定位占比，不能相加得到精确GPU忙碌率。

`ntt_seconds`是引擎累计调用耗时，含相应准备/等待，38.598000→28.891500（快25.15%）；不是Nsight中的纯NTT kernel池。carry总回读的host计时36.635766→27.666271，较小的oracle-copy host计时.118571→.075772。传输字节保持，host耗时减少主要不能解释成PCIe带宽提升，也不能与NTT节省重复相加。

八条都保持403批NTT/1,979,251对多项式乘法/40,218,760归约系数，2400 mandatory自检、66,139 GMP样本/1126 jobs/4 full checks；8241 carry块/252 finishes/1,893,402 slices。bad0/pending0/fallback0/overflow0/clean1；oracle签名`b9cbd2041266767a`，根sum`033a77303713f3a6`/xor`d16047fb14d39b21`，最终115200叶/8064000字/FNV`10619321735931855904`一致。

NTT arena完整payload峰3,341,512,288 B（3186.714MiB），其中共享A/B/Q workspace3,221,225,472 B（3GiB）、table77,550,752 B（73.958MiB）、mandatory FuseCtx峰37,920,368 B（36.164MiB）；均保持。不要把后两项再加到完整峰上。CUDA workspace分配24次、扩容8次、复用8890次保持。host private提交峰均值7314→7318MiB，未减少；private提交不是物理RAM驻留。没有为这八条采集NVML总峰。

host S4摘要H2D4.71/D2H1.34GiB保持，ALL S4输出window回读2.417GiB保持；口径不同，不相加。精确全进程copy字节和事件空隙见Systems章节。

ABBA四条逐条检查通过并已写CSV后，脚本汇总误用outer模式名筛选，发生除零。修正为long_fold/short_fold后跑BAAB；没有修改前四条记录或计算路径。最终分析按明确的`short_reduce`列核对8条顺序、全部工作量/摘要/容量，再计算均值。原错误日志与原driver快照保留。

证据：[八条完整量化](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/stage2_quantitative.json)、[ABBA CSV](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/stage2_reduce_abba/results.csv)、[BAAB CSV](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/stage2_reduce_baab/results.csv)、[最终同binary开关及检查脚本](D:/code/MPA-OpenCl/tools/bench/bench_stage2_reduce_ab.ps1:69)。

### 7.1 旧 Stage2 二进制的独立参考

上一阶段保留的 `b91f7cab…1322`，保持同D/Q/NTT尺寸策略/检查配置，八条A/B结束后串行执行两次：full **67.450250/66.897070 s，均值67.173660 s**；main53.417585/53.261016 s。新短归约八条均值59.482459 s相对该参考快约 **11.45%**。新模式0均值69.778377 s高于此参考，故只给同binary14.76%是不完整的说明。

两次旧参考没有与八条新binary样本交错，样本少且时间段不同；不能把整曲线控制差额全部归因于模式分派，不能视作置信区间。旧pure NTT的分派开销见§5。两条参考逐次复核exe SHA、Q、mandatory自检/GMP/pending/clean、根和最终叶摘要。实验目录中的旧manifest/源码快照未改：[独立参考记录](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/old_stage2_reference.json)。

## 8. Systems：NTT池缩短，CPU间隙仍在

相同`17bf4813…58a2`、相同参数/检查、模式0与1各独立采样一次。采样时间与八条未插桩wall分开。两侧均113,739个kernel、全部GPU1；各26,694个warp tile，trace记录40reg/local0、CTA512、dynamic shared16/32KiB，证实完整Stage2二进制也没有前述forward spill。资源字段不是实测occupancy。

Stage1 chain之后的kernel池：NTT **28.545614→19.095016 s（减少33.107%）**；point19.216676→19.254773 s；carry3.638176→3.769757；S4 reduce1.930276→1.941451。新NTT与点运算已各约19s，继续只盯NTT会忽略相近的点运算预算。

近似窗口从Stage1 chain之后第一kernel到最后GPU事件：67.069531→57.863991 s。kernel/copy/memset事件并集57.514780→48.350420；**无本进程GPU事件间隙9.554751→9.513572 s**，基本保持，占比14.246%→16.441%。该窗口不是精确Stage2计时，也不代表整卡idle或低occupancy。NTT更短使间隙比例上升，没有消除CPU准备/提交依赖。

全采集范围包含Stage1与初始化。两侧传输完全一致：H2D4689次/6,556,779,956 B，D2H6473次/3,037,648,120 B，D2D92次/937,993,280 B。仅新模式配置的有效payload为4B symbol，不新增生产数据数组。1089次同步copy关联的host API合计21.182848→16.099176 s，相关GPU copy仅.034059→.034212 s。host时间包含计算等待、同步和暂存，不用它计算PCIe吞吐，也不与kernel节省相加。

证据：[Systems完整分项/范围定义](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/profile_quantitative.json)、[真实Stage2 tile资源字段](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/stage2_tile_resources.json)、[模式0 trace](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/nsys_0.nsys-rep)、[模式1 trace](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/nsys_1.nsys-rep)。Compute权限未改变，没有新增有效硬件cycle/stall/带宽计数。

## 9. D 模型保护与发布边界

旧`resident_xadd6_v1`和`resident_shape_v1`按四fold归约测量，不能用在新NTT成本上。已在实际selector增加模式保护：[stage2_tree_gpu.cu:10782](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10784)。短归约开启时`calibrated=false`，两个旧profile均回`legacy_56_1`；显式D优先和原预算逻辑保持。日志增加`gl_short`字段：[10791](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10793)。原模式和缺省模式仍可使用匹配的旧模型。

上述性能binary在该保护之前冻结，所有测量都显式D1231230、`NTT_D_MODEL=0`；最终保护只改host selector/诊断，没有修改GPU归约或测量计算路径。最终单独重编译并验证保护，不能把最终exe SHA冒充为已测的17bf版本。

最终guard exe SHA256 `48a183d153c86b346a1072fb9f6d80978ad03a38ce21c1d5fb18f157aa12c846`，compile609.1/link4.3s。实际selector六种模式/请求/outer组合，**36/0**；默认0、显式0/1、旧original与shape模型、显式D保持和不执行曲线均确认。最终二进制另跑shared/warp两模式独立GMP频谱/逆向、cached switch、生命周期及resource，**14/0**；每侧216组/6,854,400字，均选择short1、local0。此前188/0是在冻结性能binary上，未把guard规划门禁等同于重跑188项。

源码与证据：[D保护门禁脚本](D:/code/MPA-OpenCl/tools/test/test_stage2_gl_reduce_scope.py:1)、[36项](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/final_scope_gate/summary.json)、[最终14项](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/final_ntt_gate/summary.json)。`test_stage2_ntt_warp.py --short-reduce 1`显式选新归约，并核对实际mode；不传该选项仍保持原12项验收接口。

本轮保留`NTT_GL_SHORT_REDUCE`缺省0，生产默认仍由上一阶段生产构建提供。新的短归约当前用于显式D实验；新D拟合与独立holdout完成后再提升生产默认。这样阶段报告可以明确区分已测后端、选择模型和已发布exe。原归约回退具有已量化的分派开销，不能宣称重建新源默认0与旧生产性能完全相同。

## 10. 接续优先级

1. **重新标定新NTT的D成本。** 冻结不同长度的卷积权重；保持真实backend的packing/精确性/内存查询，采多D曲线，重新拟合正系数，留出B2/D检验排序。P跨NTT长度边界的跳变仍存在；不能把所有D的秒数统一乘.66。
2. **减少已确认的CPU准备/提交空隙。** 对9.5s间隙及同步copy的关联kernel定位具体调用；优先处理可驻留的准备、无依赖的小kernel提交和安全复用，检查oracle ring/owner/alias的生命周期。同步copy的host耗时并不等于纯传输，不能仅换异步API就宣称节省全部时间。
3. **缩短剩余CUDA算术。** `gl_mod(x)`的128位高半恒0，数学上只需至多一次减q，可尝试直接规范化，减少无须调用完整归约时的模式加载。再以全局短归约为对照复测低层根方法3；此前2.2%的收益尚不支持叠加或默认接入。
4. **多曲线吞吐和Tensor布局另做预算。** 当前单曲线主workspace3GiB、总arena3.11GiB以外还有S4/点/驻留owner及context；历史单曲线时GPU1的NVML总峰约5.4GiB，粗估两份会超过8GiB。该总峰不是进程分配账本。需要共享workspace lease、独立曲线状态、明确pinned/普通RAM上限，再测curves/h。此前Tensor真实tile为负结果，尚未证明同SM同时issue；独立tile任务的吞吐尚不能换算整曲线吞吐。
5. **新Prime95公平对照继续。** 同N、Stage1 Q、B1/B2、覆盖/检查口径和线程配置重新测；旧90.460s记录不足以证明本轮相对CPU的加速。当前报告证明了GPU自身的改进，长期目标仍在推进。

## 2026-10-05 后续发布状态

短归约重新标定D并完成独立留出验证，生产默认已提升short1；旧归约0保留。实际生产入口33/0、S4后端选择器30/0，同exe/save固定D1381380 ABBA均值65.014116→56.834263 s（快12.58%）。本报告前面的默认关闭/旧产物数据是历史阶段记录。最新SHA、源文件行号、scope、容量和门禁边界见[短归约 D 标定与生产报告](D:/code/MPA-OpenCl/docs/STAGE2_SHORT_REDUCTION_D_CALIBRATION.md)。
