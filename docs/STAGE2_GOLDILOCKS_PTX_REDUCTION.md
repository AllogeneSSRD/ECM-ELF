# ECM CUDA Stage2：显式进位链 Goldilocks 归约

日期：2026-10-05，起点 `2cece93`。接续逆 NTT 移位归一化的完整曲线负结果，改进每次通用 Goldilocks 模乘使用的归约链。当前阶段使用 GPU1 RTX4060 Laptop/sm89/CUDA13.3，实验串行，GPU0 用户生产保持。

## 1. 数学和实现

NTT 素数 `q=2^64−2^32+1`，`epsilon=2^32−1`。ECM 模数另记 `n_ECM`。令 `hi=h1·2^32+h0`，则任意 unsigned128 输入满足：

```text
lo + hi·2^64 ≡ lo − h1 + h0·epsilon    (mod q)
a = lo − h1 − borrow·epsilon          (unsigned64)
b = h0·epsilon
sum = a+b                            (unsigned64)
value = sum + carry·epsilon
result = value≥q ? value−q : value
```

这是已验证短归约的同一恒等式，本轮将借位/进位检测直接接到 PTX 的 CC 链，避免重新做 64bit 关系比较和选择。

`sub.cc/subc` 提供借位，提取成 0/−1 掩码；减该掩码并传播低 word 借位等价于条件减 epsilon。`b` 由低 word `−h0` 和高 word `h0−[h0≠0]` 组成。合并 a/b 后提取 carry，再条件加 epsilon。最后仅当高 word 是 `0xffffffff`、低 word 非零时减 q。

CC 的 carry/borrow 语义以及不跨函数调用保存的规则来自 [NVIDIA PTX 扩展精度算术文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#extended-precision-arithmetic-instructions-subc)。本实现把整个依赖链放在单个 asm 块内。[sppark 的 CUDA Goldilocks 类型](https://github.com/supranational/sppark/blob/9e5c795/ff/gl64_t.cuh#L130)也是显式进位算术的调研来源；此处使用本仓库 canonical 余数和任意 128bit 输入合同。

与旧短归约相同，进位后 wrapped sum 有上界，补 epsilon 不会再次进位；最终至多减一次 q。函数不依赖输入是两个 canonical word 的乘积。

### 实现索引

- [任意128位 PTX 归约](D:/code/MPA-OpenCl/tools/bench/ntt_goldilocks_ptx.cuh:6)。
- [额外的显式64×64乘积探针](D:/code/MPA-OpenCl/tools/bench/ntt_goldilocks_ptx.cuh:40)：保留为独立实验，真实 NTT 采用原 `a*b/__umul64hi` 加 PTX 归约。
- [真实 NTT 配置](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:101)、[归约分派](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:151)：device mode0=原四fold、1=普通短归约、3=PTX短归约。
- [D保护](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10790)：请求新算术时旧经验成本不启用，实际日志输出 `gl_ptx`。
- [GMP/依赖链探针](D:/code/MPA-OpenCl/tools/test/ntt_goldilocks_ptx_probe.cu:1)、[驱动](D:/code/MPA-OpenCl/tools/bench/bench_ntt_goldilocks_ptx.py:1)。
- [真实NTT交叉测量](D:/code/MPA-OpenCl/tools/bench/bench_ntt_scale.py:1)、[完整Stage2交叉测量](D:/code/MPA-OpenCl/tools/bench/bench_stage2_save_reduce.py:16)。

`NTT_GL_PTX_REDUCE=1` 仅在 `NTT_GL_SHORT_REDUCE=1` 时选择 PTX。short0 保留原四fold；host 参考仍用原四fold。PTX 默认0，逆归一化开关固定0。构建依赖增加新 header，包括原生生产、Stage2 树、普通 NTT、cooperative、small-root、Tensor 和 D model probe。

## 2. 独立算术证据

primitive 探针同 binary 比较三法：0=普通短归约；1=原低/高乘积加 PTX 归约；2=显式乘积加 PTX 归约。

每法各检查 **200144 个任意128位余数、200144 个完整模乘**：200000 随机对，包含 0、q±1、全1、2^32、2^63 等 12×12 边界。另每法 **256 个独立依赖序列**，各256步，与 GMP 对照；共1201632个输出 word，全 bad0。故意损坏被拒绝。驱动检查 **11/0**。

本轮性能规模为 **2^20 个线程、每线程512次依赖模乘**，合计 `2^29` 次模乘/计时样本。每法8次 ABBA+BAAB，每次 warm+3 event 样本；全部输出在 event 外与已经由 GMP 验证的基线比对。

- 方法1：4.556016 → 3.984168 ms，快 **12.5515%**。
- 方法2：4.553514 → 3.981331 ms，快 **12.5658%**。
- chain kernel REG16→15，LOCAL/STACK/spill0。方法2没有建立额外收益，因此真实NTT只接归约方法1。

上述时间是整卡并行执行的 kernel 时间，不是单线程依赖链延迟。没有新增有效硬件周期/stall计数。

SASS 三个 chain 均占176条静态指令（含展开、循环与padding）；旧链的 64bit 比较/SEL 与新链的 IADD carry/predication 分布不同。不能根据 PTX 源码行数或静态总数宣称少了固定周期。原始 SASS/opcode 计数保留在 [指令证据](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/primitive/sass_quantitative.json)。

原始 [GMP/依赖链/八样本](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/primitive_results/measurements.json)、[构建资源](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/primitive/build.log)。

## 3. 真实 NTT 及独立旧二进制对照

公共 fused NTT 实现的 GMP selftest、cooperative 96组合/27131904 word、cached切换4调用/3145728 word，以及 shared/warp各216组合/6854400 word、生命周期与故障拒绝全部通过。PTX0/1各自检查，驱动共8组/0失败。warp forward/inverse REG40、LOCAL0；M5..8 cooperative LOCAL0，M8 forward/inverse REG48/46、shared36096/36864B。

新二进制内部固定short1、t12/warp1/outer2，8次 ABBA+BAAB 完整卷积：

- k24：13.791915 → 12.514731 ms，快9.2604%。
- k25：31.014144 → 28.315136 ms，快8.7025%。
- k26：61.984341 → 56.316843 ms，快9.1434%。
- k27：134.099199 → 122.210220 ms，快8.8658%。

**新增分派会拖慢新二进制中的普通短归约基线。** 因此另以 `2cece93` 的冻结旧探针作为基准，每个长度按 old/new/new/old 进程顺序交叉，选旧scale0和新PTX1；每个进程的8次序列取目标模式的4个 event 均值，16个目标样本/长度。两边的pass数、M、coop和实际输出检查相同，旧原始源码快照与新源码分别复核。

- k24：旧13.421910 → 新12.514048 ms，快 **6.7640%**。
- k25：旧29.368619 → 新28.314069 ms，快 **3.5907%**。
- k26：旧58.700032 → 新56.329045 ms，快 **4.0392%**。
- k27：旧125.770070 → 新122.421248 ms，快 **2.6627%**。

这些独立参考数据限制了内部 A/B 的解释；不能把内部8.7%–9.3%都当成相对已发布实现的改善。每个长度只有两个目标进程/后端，没有建立置信区间。

证据：[同二进制完整NTT](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/ntt_results/measurements.json)、[冻结旧参考交叉日志](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/old_reference/measurements.json)。

## 4. 计算量、容量和传输

该轮不改变模乘调用数量。对于各 NTT 调用 `N_i=2^k_i`、slice数 `J_i`，标准warp tile低两层根特化下，三次变换加pointwise/scale的通用模乘数仍为 `Σ_i J_i[(3k_i/2−1)N_i+T_outer,i]`。改变的是每次128位归约的依赖指令。B2/D决定giant规模和树/下降调用；`P=φ(D)/2`、`W=ceil(bitlen(n_ECM)/64)` 与packing决定 N_i。固定 D 的 A/B 保持整个调用集合。

持久 GPU/CPU 算法数组增量0B；新增device配置仍复用原4B mode symbol，每次实际模式改变才传一次4B。workspace、twiddle、S4 owner、点阵列、pinned oracle ring和主机数据生成量保持。每个outer pass的global流量为16N B/slice；两个forward tile加pointwise/scale/inverse tile仍为56N B/slice。曲线H2D/D2H/D2D数据payload没有因算术变化而改变。

primitive门禁device请求payload `3·8·200144=4803456 B`，性能probe为 `2·8·2^20+8=16777224 B`，退出live0；上下文/driver另计。实际完整Stage2的NVML/host-private/PCIe峰没有在本轮重新采集；不把源码0B增量当成总进程显存证明。


## 5. 完整 Stage2 与已发布 A191 对照

独立 native SHA256 `c19e2f572ad66c6fac6d7295ef23a5668048009fa3751fea62daf4c3d2972e1c`，4405760 B；CUDA 编译354.5s，四个C++对象3.8/2.4/3.0/2.7s。18个原始依赖在编译前冻结，运行前后逐字节复核。

GPU1，M4423、sigma26、存档声明B1=1000、B2=2011326186870、D1381380、P126720；固定恢复的Q、short1/outer2/warp1/xADD6/baby1/resident/check配置。Stage1全部跳过，存档SHA `0fe48106563dc727c092f4baf7b4c0f2f3bd57ce3bae9fd9f8a5989f2ec324d4`。恢复的X为存档中的大整数，X位数和原始文本摘要已写入native_provenance；日志restored_X=1表示恢复已启用。显式D且D_MODEL=0，逆移位归一化0。

同 C19E2F57…72E1C 八次0/1/1/0/1/0/0/1：

- 普通短归约：50.963014/50.713041/51.193557/50.819498 s，均值 **50.922277s**。
- PTX短归约：49.136881/49.032649/50.003358/49.357778 s，均值 **49.382666s**。
- 内部full快 **3.0235%**，ABBA/BAAB分别 **3.4487%/2.5996%**。
- init 10.745587→10.616019s；main 40.176691→38.766647s。
- `ntt_seconds` 均值 26.416750→24.925250s。该字段是调用侧墙钟，包含准备/同步，不能当成Systems kernel池。

然后另跑相同save/D/检查的 old/new/new/old：

- 已发布A191：51.188215/51.241748 s，均值 **51.214982s**。
- 新C19/显式PTX1：50.912411/50.645343 s，均值 **50.778877s**。
- 相对已发布实现full快 **0.8515%**。只有两条/后端，未建立统计置信区间；该数据与内部A/B分别保留，避免把分派基线开销误计为收益。

每条结果叶hash4244971527793015097、oracle c85031f6149bae11、因子集合一致，自检/GMP bad0、pending0、clean1。S4每条397launches/1836241poly_muls/36615543coefficients、2400selftest、60474GMP样本；oracle1010selected/queued/compared、60474samples。PTX没有删掉检查。

selector两侧各96/0，模式1拒绝沿用旧经验模型；是观察决策后早停的规划门禁。实际native另在普通模数/frozen因子存档上分别运行PTX0/1，并调用各216组合的真实warp NTT GMP/缓存/生命周期检查，kernel资源local0；因子59649589127497217均正确。

当前阶段没有发布新生产默认：工作区生产仍A1910CB4…56EC6A；新PTX默认0，显式1用于固定D实验。请求PTX时旧D经验模型回legacy。下一阶段需冻结新后端卷积权重、用实际多D曲线重新拟合及独立holdout，再决定默认发布。已有库中的分派开销也应继续以编译特化消除。

证据：[实际源码/存档](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/native_provenance.json)、[内部八条](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/stage2_ab/measurements.json)、[A191独立交叉](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/old_native_reference/measurements.json)、[完整覆盖复算](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/quantitative.json)、[scope0](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/scope_0/summary.json)、[scope1](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/scope_1/summary.json)、[实际NTT/frozen因子](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_ptx_20261005/factor_gate.json)。

## 6. 复现和后续

```powershell
tools/build/build_ntt_goldilocks_ptx_probe.ps1 -Build build_cuda_cmake/gl_ptx_primitive
python tools/bench/bench_ntt_goldilocks_ptx.py --exe build_cuda_cmake/gl_ptx_primitive/ntt_goldilocks_ptx_probe.exe --output <fresh-directory> --device 1
tools/build/build_ntt_coop_probe.ps1 -Build build_cuda_cmake/gl_ptx_ntt
python tools/bench/bench_ntt_scale.py --exe build_cuda_cmake/gl_ptx_ntt/ntt_coop_outer_probe.exe --output <fresh-directory> --toggle ptx --device 1
tools/build/build_ecm_cuda_stage2.ps1 -Build build_cuda_cmake/gl_ptx_native
python tools/bench/bench_stage2_save_reduce.py --exe build_cuda_cmake/gl_ptx_native/ecm_cuda_stage2.exe --save <Stage1-save> --output <fresh-directory> --toggle ptx --device 1 --d 1381380 --runs 8
```

继续优先消除热kernel重复mode分派，再按最新算术重标定D。point Montgomery的依赖MAC、CPU准备、多曲线共享workspace lease仍需分别推进。之前Tensor真tile/双流为负结果，本轮没有重新证明Tensor并行issue；也没有完成同n_ECM/Q/B1/B2/覆盖/线程的Prime95新对照，长期目标继续。
