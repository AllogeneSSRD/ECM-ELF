# ECM GPU Stage2：逆 NTT 的 2 幂归一化实验

日期：2026-10-05。起点 `dcc0e32`。本轮接续已经完成的 xADD6、D 重标定、协作式 outer、短归约和 GPU baby 归一化，减少逆 NTT 中每个系数的依赖模乘。实验只使用 CUDA Core。

## 1. 优先级和当前状态

六模乘 xADD 已经是生产默认，使用两个模 2 除法保持旧坐标比例；此前同二进制八条整曲线改善 6.26%，见 [xADD/D 报告](D:/code/MPA-OpenCl/docs/STAGE2_XADD_D_OPTIMIZATION.md:21)。本轮没有重新测量或改变 xADD。

匹配当前 GPU baby/short/尺寸策略的 D 模型也已发布：大界 D=1381380、小界 D=330330，owner512MiB 时 D=1141140，参见 [GPU baby D 报告](D:/code/MPA-OpenCl/docs/STAGE2_GPU_BABY_D_CALIBRATION.md:1)。这些是历史阶段结果，不能与本轮百分比相加。

源码搜索表明，`gl_mod/gl_mod_dev` 的调用在旧逐层变换的指数辅助函数中，当前 fused tile/outer 热路径直接使用 `gl_mul/add/sub`。因此优先试验每次卷积都实际执行的 `1/N` 归一化。

开关 `NTT_GL_SHIFT_SCALE=1` 启用候选，默认 0。新开关位于公共 NTT 实现，独立 Stage2 save/ini/worktodo 入口可以继承它。请求新算术时 D 选择器明确回退到 `legacy_56_1`，避免把旧经验系数当作新后端的标定结果；性能比较必须显式固定 D。

## 2. 数学与精确性合同

以下 `q=2^64−2^32+1` 是 NTT 素数，ECM 模数另记为 `n_ECM`。设变换长度 `N=2^k`，`1≤k≤32`，输入 `0≤x<q`。

因为 `2^64 ≡ 2^32−1 (mod q)`，且 `(2^32−1)·2^32 ≡ −1`，有：

```text
2^(-k) ≡ 2^(32-k) − 2^(64-k)                 (mod q)
x = h·2^k + r,  h=x>>k,  r=x & (2^k−1)
x·2^(-k) ≡ [h+r·2^(32-k)] − r·2^(64-k)     (mod q)
```

令 `a=h+(r<<(32−k))`、`b=r<<(64−k)`。二者均是 canonical：

- `a < 2^(64−k)+2^32 ≤ 2^63+2^32 < q`；
- `b ≤ 2^64−2^(64−k) ≤ 2^64−2^32 = q−1`。

因此只需一次模减法。若 `a<b`，无符号减法已经加了 `2^64`，再减 `epsilon=2^32−1` 即得到正确余数；结果仍在 `[0,q)`，不需要一般 128bit 归约。

该方法并没有要求原多项式系数、ECM 模数或点坐标可逆。使用它之前的 pointwise `gl_mul(a,b)` 保证 x 为 canonical Goldilocks 余数。

实际 kernel 对 k 和 n_scale 做完整检查：仅当 `1≤k≤32` 且 `n_scale=q−2^(64−k)+2^(32−k)` 时使用移位。自定义 scale、k=0 或范围外继续使用原通用模乘。范围判断先于移位，避免无效移位。

## 3. 实现位置

- [移位归一化原语](D:/code/MPA-OpenCl/tools/bench/ntt_goldilocks_reduce.cuh:8)：canonical 输入、余数拆分、借位修正。
- [有效 k/scale 判断](D:/code/MPA-OpenCl/tools/bench/ntt_goldilocks_reduce.cuh:17)：自定义 scale 保持原语义。
- [实际 tile kernel](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1220)：新增默认 false 的统一开关参数，在原 pointwise/scale 循环内选择算法；蝶形、排列、屏障和写回保持。
- [完整逆变换接入](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1682)：每次主机调用读开关，兼容 shared/warp 两种 tile。
- [D 模型保护](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10789)：请求新后端时拒绝沿用旧标定，日志输出 `gl_shift_scale`。
- [独立 GMP 和真实卷积探针](D:/code/MPA-OpenCl/tools/test/ntt_coop_outer_probe.cu:64)、[串行门禁/A/B 驱动](D:/code/MPA-OpenCl/tools/bench/bench_ntt_scale.py:1)、[整曲线 A/B](D:/code/MPA-OpenCl/tools/bench/bench_stage2_save_reduce.py:16)、[规划保护检查](D:/code/MPA-OpenCl/tools/test/test_stage2_production_scope.py:15)。

本报告 line 对应本轮工作树。旧报告中的 NTT 行号属于其历史源码快照；本轮 probe/native manifest 分别冻结实际源码哈希。

## 4. 计算量、数据量与容量

每个 NTT slice 原先 pointwise/scale 为 `2N` 次通用 Goldilocks 模乘：N 次 pointwise，N 次乘 `1/N`。新路径变为 N 次通用模乘和 N 次移位归一化，删除 **N 次 64×64→128 乘积及一般归约**。

若真实 Stage2 各次逆变换长度为 `N_i`、slice 数为 `J_i`，总减少量为 `Σ_i J_i N_i` 次模乘。依赖 `B2,D,n_ECM` 的部分通过实际树/下降/fold 调用序列体现：baby 数 `P=φ(D)/2`，giant 数和分组由 B2/D 决定，packing slot 与 `W=ceil(bitlen(n_ECM)/64)` 决定 N_i。不能只用最大的 N 乘所有调用次数。

在标准 t12/warp 表下，三次变换的普通蝶形共 `3Nk/2` 次根乘，低两层特化删除 `3N` 次通用根乘。加上 pointwise 和 scale，原调用量为 `(3k/2−1)N + T_outer`，新量为 `(3k/2−2)N + T_outer`；T_outer 是协作 outer 构造/平方 twiddle 的额外调用。k27 时忽略该额外项，删除比例约 `1/39.5=2.53%`，这不是时长改善比例。

移位原语每个系数主要为一个右移、两个左移、掩码、加法、减法和比较/借位修正。64bit 操作在设备上可能拆成多个指令；没有新的有效硬件周期/stall 计数，不将源码操作数或 `N log N` 写成 GPU 周期。

新增算法数组、持久显存和 pinned/普通 RAM payload 均为 **0 B**；主 NTT workspace、twiddle 表、arena、S4 owner、oracle ring 保持。每个 inverse tile pass 的 global 流量仍为 `24N` B/slice：读 A、读 B、写 A 各 `8N`。两次 forward 加该 inverse 的 tile 流量仍为 `56N`；outer 各 pass 的读写不变。

算法没有新增 H2D/D2H/D2D payload，没有新增 symbol 配置或完整数组 pass，也不改变 save 解码、点生成和 baby 传输。已有 kernel 参数末尾增加一个 bool，probe 的参数常量区从 408 到 409 B；这与曲线数组传输是不同口径。总 VRAM/host-private/context 峰不能从上述 0 B 推断，当前轮未重新采集完整分配账本或 PCIe/NVML 峰。

## 5. 独立验证和纯 NTT

probe `7f358913…17585`，sm89/CUDA13.3，编译 19.862858 s。构建脚本检查编译前后五个依赖的 SHA；驱动在每次运行前后复核 exe 和源码。

GMP 归一化检查每种 short 后端 **263357 word、bad0**：k1..32、8192 随机/每 k、q/2/q−1/2^32/2^63 等边界，以及余数低位归零前后；自定义 scale 0/1/q−1 和 k−1/0/33/64 验证回退。

新开关 0/1 分别运行：cooperative GMP 96 组合/27131904 word；cached 模式切换 4 调用/3145728 word；原 shared 和 warp 各 216 组合/6854400 word，以及 lifecycle 14 调用与 cached warp 切换 4 调用/98304 word。故意损坏的 cooperative 输出均被拒绝。驱动共 **8 组检查、0 失败**，不是仅做往返自洽。

8 次 ABBA+BAAB，每次 warm+3 个 event 样本；两次 forward+pointwise/scale/inverse 的真实完整卷积，填充、检查和规划在 event 外。每次检查全部 N 个输出，四样本/模式，未建立统计置信区间。short 固定1、outer 策略固定2、t12/M4/warp1。

- k16：0.000152747 → 0.000150187 s，快 1.676%；此形状没有启用 cooperative。
- k24：0.013784662 → 0.013727061 s，快 0.418%，M6。
- k25：0.029366016 → 0.029271808 s，快 0.321%，M8。
- k26：0.058690475 → 0.058515115 s，快 0.299%，M8。
- k27：0.125595648 → 0.125188864 s，快 0.324%，M8。

warp forward/inverse 均 REG40/STACK0/LOCAL0；普通 shared forward/inverse REG48/46，无新增 spill。t12/CTA512/shared32KiB 的容量 API 上限分别为 warp3块和 shared2块，不是实测 occupancy。该纯卷积的小改善不能直接换算成完整曲线加速。

证据：[probe manifest](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_scale_20261005/probe/manifest.json)、[原始日志/完整八样本](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_scale_20261005/initial/measurements.json)、[编译资源](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_scale_20261005/probe/build.log)。


## 6. 原生存档完整 Stage2

独立 native exe SHA256 `a374fdb78c12e61d4753609c7c82512cc504f0e1dd61f9e29daadeb419407cb3`，4351488 B；CUDA 编译 288.4 s，其余四个 C++ 对象分别 3.9/2.6/3.4/3.3 s。17 个依赖在编译前冻结、运行前后复核并保留原始字节快照。

同 exe/save，GPU1，M4423、sigma26、B1=1000、B2=2011326186870、D=1381380、P=126720、short1/outer2/warp1/xADD6/baby1、resident/check 配置保持。存档 SHA `0fe48106563dc727c092f4baf7b4c0f2f3bd57ce3bae9fd9f8a5989f2ec324d4`；Stage1 全部跳过。D_MODEL=0 且显式 D，两侧没有模型选择差异。八条顺序为 0/1/1/0/1/0/0/1。

- 模式0 full：50.992757/52.407893/52.561458/51.117962 s，均值 **51.770018 s**。
- 模式1 full：52.172205/51.497284/52.592073/51.600190 s，均值 **51.965438 s**。
- 候选比模式0 **慢 0.3775%**；ABBA 和 BAAB 组分别为 **-0.2600% / -0.4946%**。
- init 10.931041 → 10.981509 s；main 40.838976 → 40.983928 s。
- baby ladder 7.899500 → 7.899000 s；affine 0.305500 → 0.258250 s。这些算子没改，差异用于观察运行波动。

四条/模式、每个顺序组两条/模式，无置信区间；当前数据没有建立完整曲线的稳定加速，不能把均值差全部归因于 NTT。纯卷积约0.3%收益受到完整流水线其他阶段和运行波动的稀释。因此没有重标定新 D 或提升新开关的生产默认。

八条叶值都为 `4244971527793015097`，oracle signature 都为 `c85031f6149bae11`，自检/GMP bad0、pending0、clean1，因子结果相同。实际每条 S4 覆盖为 397 launches、1836241 poly_muls、36615543 reduced coefficients、2400 selftests、60474 GMP 检查；没有删除检查换取时长。

实际 selector 0/1 各 **96/0**：模式1 明确拒绝旧经验模型，模式0 保持匹配的原 scope；这些检查在观察决策后控制停止，不冒充完成曲线。另两个实际普通模数/frozen 因子曲线都得到 `59649589127497217`，bad_factors0。

证据：[native 原始依赖/存档](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_scale_20261005/native_provenance.json)、[8 条完整日志索引](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_scale_20261005/stage2_ab/measurements.json)、[阶段/覆盖复算](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_scale_20261005/quantitative.json)、[关闭 scope](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_scale_20261005/scope_0/summary.json)、[开启 scope](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_scale_20261005/scope_1/summary.json)、[实际因子检查](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_scale_20261005/factor_gate.json)。

## 7. 复现与接续

```powershell
tools/build/test/build_ntt_coop_probe.ps1 -Build build_cuda_cmake/ntt_scale_probe
python tools/bench/bench_ntt_scale.py --exe build_cuda_cmake/ntt_scale_probe/ntt_coop_outer_probe.exe --output build_cuda_cmake/ntt_scale_results --device 1

# 新目录构建，不覆盖已发布产物
tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/ntt_scale_native
python tools/bench/bench_stage2_save_reduce.py --exe build_cuda_cmake/ntt_scale_native/ecm_cuda_stage2.exe --save <Stage1-save> --output <fresh-directory> --device 1 --toggle scale --d 1381380 --runs 8
```

当前已发布工作区生产仍为 **A1910CB4…56EC6A**；本轮 A374FDB7…07CB3 是独立实验产物，默认 scale0。GPU0 的用户生产没有改动。

下一优先级：

1. 对实际热路径的 64×64 乘积/短归约检查 SASS，评估 32bit limb 合并以及受控编译特化，避免模式加载延长寄存器生命期。
2. 继续减少 point Montgomery 的依赖 MAC。历史 A5 Systems 的正确 Stage2 范围中 point18.54s、NTT16.94s；本轮归一化只削减 NTT 的一小部分，点运算仍值得优先投入。Mersenne 专用乘积折叠需要显式模数合同和独立精确性验证。
3. 剩余 CPU 准备空隙与多曲线 workspace lease 分开处理；共享 scratch 必须保证曲线独立状态和 RAM/VRAM 生命周期预算。继续保持 Prime95 相同 n_ECM/Q/B1/B2/覆盖/线程的公平对照目标，当前数据不证明已经超过 Prime95。
