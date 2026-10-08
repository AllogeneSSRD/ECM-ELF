# Stage1 param0：TPI16 共享 DBL 的寄存器上限 128 实验

## 1. 目标与范围

基线为 `6682b74`，本地只读快照在 `build_cuda_cmake/prac/after_shared_dbl_tpi32_20261007/`，exe SHA256 为 `2a7db11e451bcda16640a3d30d9582edce814aed7a7da93bcd9e3d2a28989232`。上一轮 N4423 强制 TPI32 的匹配 grid 实验未提高吞吐；本轮固定 N4423、4608-bit 容器、TPI16、TPB128，比较同一共享 seed/loop DBL 与单点 xADD 算法的 natural255、cap168、cap128。

TPI 是每条曲线的协作线程数，TPB 是每个 block 的线程数。寄存器上限不改变 TPB。GPU1 有 65,536 个寄存器/SM；只考虑寄存器时，容量上界为 `floor(65536/(128*R_allocated))`，硬件分配 168/128 对应 3/4 blocks/SM。C384/768/1536 分别提交 48/96/192 blocks；相同 grid 不保证实际驻留数量相同。

事先提出三个可证伪假设：降低寄存器分配有助隐藏依赖延迟；新增 spill 可能抵消收益；收益可能依赖批量和尾波。以实际编译资源、完整 Q、生产标量子窗口、普通前缀投影及管理员 NCU 判断。

## 2. 实现与边界

新增 [MODE14 定义和寄存器限制](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:11)，使用独立 `__maxnreg__(128)` 实例，复用现有 normalized Montgomery、共享 DBL 和单点 ADD。[候选实例](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cu:21)仅支持 4608/TPI16 或默认 TPI；[host 配置](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:59)要求 `ECM_PRAC_REG_TARGET=128` 同时选择 `ECM_PRAC_VARIANT=single-compact`，其他 variant、TPI32、其他正常容器拒绝。不会为了满足实验策略将 N 静默提升到其他容器。

CPU 模型用于确认数学路径；原生门禁用于验证实际编译的 MODE14。不同寄存器策略可以在相同容器/TPI 的 checkpoint 上恢复，因为数据域、标量和点表示未改变。窗口诊断必须保持已有 checkpoint 字节不变，并以 CPU oracle 检查输出。

## 3. 测量方法

- 同二进制三策略 A/B：C384/768/1536，两个反序重复。
- 固定窗口：B1=260m、tail32、chunk4/32、两个排除的 warmup 轮、6 秒采样，共 36 样本。
- 普通前缀：B1=10m/260m、目标50ms、15秒采样、warmup5秒，共36样本；只读取精确 `s/curve` 投影，正常 sample-limit/checkpoint-only 退出。
- GPU1 串行运行；计时期间不编译、不导出大 SASS、不运行 profiler。逐样本验证 exe SHA、缓存命中、真实 TPI/容器/C/grid。
- NCU 使用管理员权限；C768/C1536 的 tail32/grid96/192/TPB128 窗口分别比较 cap168/cap128。重放耗时不用于性能排名。

`W=6*ADD+5*DBL` 及每条曲线的 WMAD 主循环源码求值代理 `4*144²*W` 不变。data/seed 各 `7*C*4608/8` bytes，计划 `16P` bytes，边界逻辑量 `6*C*4608/8*launches` 不变。新增 spill 若存在，会影响设备 local/cache 访问；不能将累计 local sectors 字节代理当作显存容量或 PCIe 流量。

## 4. 编译、正确性与测量结果

七个相关 TU 并行编译完成，最终 CMake 只链接。串行耗时合计 1,451.8s，critical path 为 TPI16 TU 的 735.7s，候选 TU 为 240.8s；本次公共头改变触发完整 PRAC 实例重编，实际超出了六分钟，记录真实耗时。after exe SHA256 为 `d1a5b6476c17c95762283fd0d30205e5175b3d37ccf49198fa2a99c2476d7370`。

MODE14 实际 128 寄存器、stack208 bytes，ptxas 静态 spill stores808/loads972 bytes；28,248 条静态指令、451,968 bytes text。相对 cap168 的 27,784 条、444,544 bytes 增加约1.67%。静态 spill 字节是编译器记录，不是累计运行访存量或分配容量。完整 SASS（含调度行）及资源核对：旧 TPI16 MODE10–13、旧显式 TPI32 MODE12/13 均不变；未新增 outlined callee。

CPU 模型通过：17,066 输入链、六模型共102,396求值、264,538角色步骤；归一化/别名模型804正例、4,824公共别名对照。私有数学头 SHA仍为 `a0f3b9ae91474384d03b8ac644c509fc716236f205798b5a4e806d62a592bafc`，公共头 SHA为 `86abc67cdc088670023a9956aea3158d9c669a179a62ec0802c697fc42ebaf8d`，公共头本轮只扩展策略和编译选择。

GPU1 完整 Q/save 门禁全部通过：3,568 项对照、27 类，其中 cap128 主案例及其 checkpoint 恢复为1,184 Q，其余包括 ladder/resident 与公共结果/目标边界，不将总数全算作候选覆盖。

窗口门禁224条结果、1,792 Q、16恢复Q、29拒绝全部通过；MODE14实际覆盖19窗口/152 Q，几何记录均为4608/TPI16/寄存器限制容量4blocks/SM。176项同TPI完整XZ字节对照包含66跨切片；另109项single-compact跨TPI完整XZ字节对照通过。CPU oracle184miss/1,608hit，每个输出仍验证；cap128 checkpoint不被诊断窗口修改，随后可用baseline255恢复。

同二进制72个性能样本与四份管理员NCU均完成并审计。生产推荐仍为本批最佳TPI16/cap168/C1536/50ms。原始证据在忽略的 `docs/data/stage1_shared_dbl_cap128_*_20261007`；40项源码/工具SHA已冻结。

### 4.1 固定窗口：36个样本完成并审计

以下为两个反序重复的墙钟投影中位，s/curve，顺序 natural255 / cap168 / cap128：

- C384，chunk4：146.424509 / 147.604638 / 160.377261。
- C384，chunk32：145.052985 / 146.246310 / 159.031853。
- C768，chunk4：144.830351 / 145.521171 / 139.431754。
- C768，chunk32：147.188922 / 147.698902 / 141.183239。
- C1536，chunk4：132.363926 / **131.298137** / 134.574535。
- C1536，chunk32：135.803690 / **134.973491** / 137.536531。

cap128相对同C的cap168吞吐：C384为−7.964%/−8.040%，C768为+4.367%/+4.615%，C1536为−2.435%/−1.864%。本批结果只在中间批量档改善吞吐，未超过大批量cap168最佳吞吐；驻留和访存作用结合下面的NCU分析。

36/36矩阵无缺项/重复，真实几何、缓存命中、完整first/prime/work、seed/边界逻辑bytes及原始日志均核对；事件/墙钟投影独立按 `W_full*t/(W_window*measured*C)` 重算，保持两种口径分离。固定窗口W=8,053、完整B1=260m W=3,369,476,895。GPU采样528点/437忙点（利用率≥70%），忙时SM1800MHz，温度52..67°C，设备memory.used采样最大323MiB；不是进程完整峰值。证据为 `stage1_shared_dbl_cap128_pairs_fixed_20261007/audit.json`。

### 4.2 普通前缀：36个样本完成并审计

以下为两个反序重复的精确投影中位，s/curve，顺序 natural255 / cap168 / cap128：

- B1=10m，C384：5.540328 / 5.586193 / 6.073710。
- B1=10m，C768：5.509651 / 5.536534 / 5.355761。
- B1=10m，C1536：5.042464 / **5.004494** / 5.087899。
- B1=260m，C384：145.037352 / 146.238070 / 158.999363。
- B1=260m，C768：144.211251 / 144.906029 / 140.225628。
- B1=260m，C1536：131.951888 / **130.967987** / 134.074840。

cap128相对cap168：C384两个B1均约−8.03%吞吐，C768为+3.375%/+3.338%，C1536为−1.639%/−2.317%。本批最佳仍为cap168/C1536；与上轮约0.02%内相近，不宣称新增最佳吞吐提升。cap128可作为C768受限批量的显式选项，不改默认策略或扩大到其他N。

36/36矩阵无缺项/重复，日志重建每个精确projection样本及其最后五秒中位、范围和数量；全部正常exit1、sample-limit/checkpoint-only，无强制终止/最终save。40项源码SHA及exe身份核对通过。GPU采样1,123点/1,050忙点，忙时SM1800MHz，温度58..67°C，设备memory.used最大317MiB。完整计划W分别为128,750,918和3,369,476,895；没有完成整个生产batch，不将这些投影作为已认证的Auto B2 T1。证据为 `stage1_shared_dbl_cap128_pairs_prefix_20261007/audit.json`。

### 4.3 管理员NCU：驻留增加与spill代价

四份采集/CSV导出均exit0，实际内核分别MODE13/14、TPI16、4608，GPU1/TPB128，C1536 grid192、C768 grid96；原始输入、first/prime/work、exe SHA与wrapper设置均核对。C1536两份19passes，C768两份18passes；不乘重放pass，也不把重放时间当作吞吐。

以下顺序为 C1536 cap168 / cap128 / C768 cap168 / cap128：

- 实际寄存器162 / 128 / 162 / 128，硬件分配168 / 128 / 168 / 128，寄存器限制容量3 / 4 / 3 / 4blocks/SM。
- 活跃warp/SM：11.023639 / 14.262034 / 9.337548 / 13.516335。
- eligible warps/scheduler/cycle：0.482810 / 0.570250 / 0.424307 / 0.551310。
- issue active：33.829546 / 33.264519 / 31.021973 / 32.444758%。
- wait/issue-active：4.158946 / 4.592938 / 3.998378 / 4.551971。
- no_instruction：0.460946 / 1.097408 / 0.272744 / 0.949249。
- short_scoreboard：0.326041 / 0.710975 / 0.327051 / 0.713745；long_scoreboard：0.000049 / 0.005374 / 0.000060 / 0.005442。
- math_pipe_throttle：0.892368 / 1.485069 / 0.769874 / 1.453356。
- local load sectors：0 / 322,449,408 / 0 / 161,224,704；store：21,600 / 160,745,004 / 10,724 / 80,357,960。
- local sectors×32累计字节代理：0.659180 / **14,745.923218** / 0.327271 / **7,372.517822 MiB**。这是该捕获窗口的缓存访问量，绝不是新增14GiB显存或PCIe拷贝。
- cap128 local load命中98.210%/98.268%，store命中94.884%/94.958%；DRAM throughput四份仅0.000032%..0.000152% of peak，无该窗口DRAM饱和证据。
- L1TEX throughput为25.135760 / 25.920187 / 22.956752 / 25.228127% of peak sustained active；没有接近峰值的L1TEX吞吐证据。

cap128的活跃warp增加得到验证，动态local访问也与静态spill一致。C1536的更多warp未改善issue或吞吐；C768原cap168活跃warp更低，cap128增加驻留后issue和吞吐有所改善，但仍慢于C1536/cap168。本结论支持“驻留收益与spill/调度代价的权衡”，未隔离证明某个stall解释全部差异。

不能把stall比值当作墙钟比例、no_instruction当作I-cache miss、short_scoreboard全部归因于local load；也不能仅凭累计local访问量宣称L1带宽饱和。证据为 `stage1_shared_dbl_cap128_ncu_analysis_20261007.json`，其中保留实际内核、尺寸、pass数、原始CSV SHA及指标。

### 4.4 下一步

优先继续cap168的批量几何测试：GPU1的寄存器容量为3blocks/SM，每block8曲线、24SM，满驻留一轮为576曲线。C1536并非576的整数倍；候选C576/1152/1728/2304与C1536同二进制比较，可检验尾波是否仍限制吞吐。容量公式只用于设计实验，不能替代实际驻留/计时。

之后再检验private xADD的临时值复用，例如把disjoint输出Z用作中间U，减少源级局部bn；必须保持输出与全部输入不别名、normalized修正和6M等价计数，并用CPU别名模型、原生完整Q/生产窗口、SASS与同二进制/冻结版本A/B判断编译器是否已做相同合并。若无收益，再推进normalized PRAC专用紧凑容器或Montgomery核心；先证明边界，避免降低全局carry headroom。

## 5. 复现

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/internal/parallel_nvcc.ps1 `
  -BuildDir build_cuda_cmake/prac -Only 'cgbn_stage1(_prac_[^.]+)?\.cu$' -Jobs 6
python tools/test/test_cuda_prac.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --bits 4423 --tpi 16 --registers 128 --variant single-compact --curves 384 `
  --target-ms 50 --device 1 --output docs/data/cap128_fullq
python tools/test/test_cuda_prac_windows.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --single-add --single-compact --single-compact-reg128 --single-compact-tpi32 `
  --production --production-counts 32 --production-chunks 7 32 --device 1 `
  --output docs/data/cap128_windowsq
python tools/bench/bench_stage1_prac_tpi_pairs.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --mode windows --configs shared16 shared16cap168 shared16cap128 `
  --b1 260000000 --seconds 6 --exp-cache build_cuda_cmake/prac `
  --output docs/data/cap128_fixed
python tools/bench/bench_stage1_prac_tpi_pairs.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --mode prefix --configs shared16 shared16cap168 shared16cap128 `
  --b1 10000000 260000000 --target-ms 50 --seconds 15 --warmup 5 `
  --exp-cache build_cuda_cmake/prac --output docs/data/cap128_prefix
```
