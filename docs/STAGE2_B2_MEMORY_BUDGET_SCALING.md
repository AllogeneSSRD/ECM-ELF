# ECM Stage2：B2、模数位数与两类显存预算的独立实验

2026-10-05。本报告接续用户提供的「分析B2耗时关系」，使用当前冻结生产版本，分别约束 NTT 大工作区与驻留 fold owner，并测量 2203/4423/8191 bits 的完整 Stage2。

## 1. 要检验的关系

原报告的观察是：owner 在默认 640 MiB 内驻留时，时间近似随 √B2 增长；超过预算后，时间近似随 B2 增长。这是有价值的现象，但原记录中 **D/P 停止增长与 owner 回退同时发生**，单靠这组数据不能认定增长指数变化由 owner 回退造成。

本轮拆开两种作用：

- 固定相同 N、B2、D/P、大工作区预算，仅改变 owner 预算，观察回退的时间和传输代价。
- 固定 owner 预算足够驻留，只改变允许的大工作区形状，观察 D/P 饱和后的 B2 增长关系。
- 三种位数采用相同 B1、sigma 与指数约定，测量 NTT 长度和 owner 容量边界随位数的变化。

**驻留不保证 √B2。** 当 D/P 固定时，驻留路径也需执行随 B2 增长的 G 树和 fold 批次。近似平方根增长来自可继续调整 D/P 时的工作量平衡；驻留主要减少每批的搬运和准备成本。下面用真实完整曲线检验这一点。

## 2. 输入、版本和测量口径

原附件 `ecm-test.txt` 来自所引用对话：GPU0 RTX4070 Ti、3367 bits、B1=260000000、sigma=8457690939091846，B2 从 8×10¹⁰ 到 2.6×10¹³。用户所列旧 NTT 峰值 877.97/1747.96/3487.75 MiB 与 owner 115.29/188.66/335.39/565.97/0/0 MiB 对应旧版本；旧模块数据不替换为本轮的结果。

[原始附件副本](D:/code/MPA-OpenCl/build_cuda_cmake/_budget_scaling_20261005/original_report/ecm-test.txt) 可核验以下几何和引擎时间（没有混用外层进程时间）：

|B2|D|P|G|owner|stage2_full_wall.total/s|
|---:|---:|---:|---:|---|---:|
|8×10¹⁰|330330|31680|8|驻留|6.209680|
|2.6×10¹¹|570570|51840|9|驻留|10.534447|
|8×10¹¹|1021020|92160|9|驻留|20.541540|
|2.6×10¹²|1711710|155520|10|驻留|36.969517|
|8×10¹²|2042040|184320|22|预算回退|72.505860|
|2.6×10¹³|2042040|184320|70|预算回退|189.036115|

原附件另有同一2.6×10¹¹的10.491305s记录。最后两档 D/P 完全相同，G 从22增至70；这个事实本身就能解释线性项变大，仍需同 D 的 owner 对照来分离额外搬运。

本轮配置：

- GPU1 RTX4060 Laptop / sm89，约 8 GiB 显存；GPU0 外部生产任务保持原状。
- 生产 `ecm_cuda_stage2.exe` SHA256 `893f6e907c17803ed90b09b98ffbf6b85e08b7deb1335a9d6c6efd1f8a01f69d`；19 个原始依赖逐文件冻结。固定 PTX3、outer0、GPU baby、xADD6、Mersenne 点乘；carry 诊断融合关闭。
- N=2^S−1，S=2203/4423/8191；B1=1000、sigma=26、param0、**lcm** Stage1。每个位数的 affine X 由 CPU 参考生成，再与独立 GMP-ECM save 逐字比较；native checksum 包含 X。之后只读取 save 执行 Stage2。
- `NTT_D_MODEL=0`，外部规划器使用 legacy 排序作为统一的实验策略，不使用仅在特定位数/设备/预算范围标定的 profile6。**这里的 D 不宣称是当前后端的性能最优 D。**
- 必需 GMP/selftest/oracle 检查保持默认。相同 `(S,B2,D)` 的对照要求最终 leaf hash 和因子列表一致；不同 D 的 leaf/采样覆盖本来不同，不要求相同 hash。
- 时间取 `stage2_full_wall.total = init+main`；保留独立的进程墙钟。Stage1、形状查询和 CPU/GMP 参考在正式计时前完成；D 扫描不计入该 total。完整曲线串行运行，不与编译或重型 CPU 分析并发。

这是不同设备、B1、sigma、点和后端上的新实验，**不能把新旧秒数之差解释为生产版本加速**。不同位数也是不同 N，不是对同一模数补零。

特别是本轮B1=1000小于D，会包含对应的小素数桥接/命名等收尾工作；原B1=2.6亿大于这些D，该部分负担不同。它包含在本轮full total中，不从完整时间里人为扣除。

## 3. 两类预算分别控制什么

单位均为 MiB=2²⁰ B。基础矩阵三种策略：

|策略名|外部大工作区形状上限|现有 arena 缓存预算|owner 预算|
|---|---:|---:|---:|
|large_resident|3072|6300|1024|
|large_owner128|3072|6300|128|
|small_big|768|6300|1024|

`large_resident/large_owner128` 使用同一 D/P，只变 owner；`small_big` 重新选择满足较小 NTT 形状上限的 D/P，并保留足够的 owner 预算。预算未实际生效的小规模点是对照点，不能据其微小波动宣称收益。

### 3.1 Owner：现成独立分配上限

[FoldDeviceState::init](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7548) 检查 `NTT_FOLD_DEVICE_MAX_MB`，缺省 640。所需 payload 超预算时 `fallback=budget`，回到原路径；0 表示禁止该 owner 分配。它还会检查真实剩余显存，保留可能的大工作区增长及 1 GiB 余量，所以预算足够不等于保证驻留，应记录具体 fallback 原因。

Owner 有两块多项式数组、map/length、模数和 digest；容量为：

```
W = ceil(S/64)
M_owner(P,S) = 8W(9P+8)+48 bytes
P_owner_max(M,S) = floor(((M−48)/(8W)−8)/9)
```

640 MiB 的 P 上限约为：2203 bits 26.6 万；4423 bits 13.3 万；8191 bits 7.28 万。旧 3367 bits 使用 W=53，其上限约 17.59 万：P=155520 可以驻留，P=184320 会超限。因此跨位数比较回退拐点，应比较 `(S,P,M_owner)`，不能只比较 B2。

### 3.2 大工作区：形状过滤与实际峰值核验

当前生产没有单独的“大数组分配硬上限”参数。本轮新增外部实验工具，以实际 packing planner 查询值过滤候选 D，满足：

```
L_f = NTT_length(P+1,S)
L_t = NTT_length(P/2+1,S)
M_big(shape,batch) = 3 × 8 × L × batch bytes
24L_f <= M_big_budget
M_plan = 8[(3L_f+2P+1)+2(3L_t+2(P/2+1)−1)] <= M_arena_budget
```

`M_plan` 对应 [real_run_words](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10680) 的保守规划估计，不是实际 NTT 峰值。选择后用冻结生产的 C++ `ntt_shape_query` 核验 Python 几何结果，并在每条完整曲线结束后要求 `big_peak_bytes <= M_big_budget`。较低层 batch 也包含在实际峰值核验中。

对输入最大系数数 m，当前 packing 几何可用下式重现，见 [Python exact shape](D:/code/MPA-OpenCl/tools/bench/calibrate_stage2_d.py:99) 与 [真实 C++ 查询](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3052)：

```
slot_bits = 2S + max(1, ceil(log2 m))
Q_GL = 2^64−2^32+1
选最大的 b<=62，使 m·ceil(slot_bits/b)·(2^b−1)^2 < Q_GL
digits_per_slot = ceil(slot_bits/b)
L = 2^(bit_length(2m·digits_per_slot))
```

最后一式选择严格大于 `2m·digits_per_slot` 的2幂长度，不应擅自改成遇到等号不进位的ceil-power。fold取m=P+1，tree取m=P/2+1；完整API还有可支持shape等检查，因此本轮仍用真实查询复核。大工作区随 S/P增加呈阶梯增长，owner则按W/P增长，二者的拐点不会重合。

这是 **候选形状约束＋事后实际容量检查**，没有改 GPU 分配器。实际 big 三数组由 [ntt_arena_bufs](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2065) 复用；twiddle、fuse_base、输出和其他模块独立计账。

### 3.3 `--arena-mb` 不能当作全进程显存硬上限

[CLI](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:469) 把该值转换成 `NTT_ARENA_CAP_KB`；规划器也使用它筛选 D。缓存申请超预算可能淘汰其他形状，仍不够则退到 per-call cudaMalloc，见 [缓存拒绝](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2119)。因此它约束缓存，不包括 owner、坐标、raw G、CUDA context，也不能保证拒绝后的临时分配不超同一数值。

本轮保持 arena6300，不用 arena 参数同时改变多个因素；记录 `arena_overflow`，避免把缓存回退误当成单独 big 预算的效果。

**另一个容易误读的字段是 `real_batched_breakdown.arena_mb`。** 它打印 `arena.bytes` 缓存预算账本，不是`ntt_workspace_stats.full_peak_bytes`实际payload。例如本轮S4423/B2=8e12网格的账本约6149MiB，而NTT完整payload峰约3187.03MiB。原因可从 [ntt_arena_fuse](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2243) 看出：表项按旧保守式 `8[L+L/2+4·FUSE_MAX_PASSES·(coop?256:64)] B` 计入bytes；[实际payload统计](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1930) 按具体pass的table/base容量求和。[淘汰](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2001) 又按实际table words减账，base仍由FuseCtx持有。预算计账、真实分配量和峰值统计目前并非同一公式。不能把两个字段混为一谈；本轮overflow0，不把该计账问题冒充已经测出的回退损失。

此外，正常生产的匹配 scope 下 profile6 会按 owner 等路径条件过滤 D；本轮显式关闭经验模型，并把 owner 预算从外部 D 排序中拿出，才可以保持相同 D 做因果对照。本轮的宽预算曲线不是“所有生产默认自动选D”的预测，不能直接外推默认640MiB时自动规划出的D。

## 4. 计算量、数据和预期增长关系

[实际 giant 数和形状](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10998)、[G 批次](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9694)：

```
P = phi(D)/2
I = floor(B2/D)+2
G = ceil(I/P)
fold 次数 = G−1
```

记 `C_G(S,P)` 是一批 G 树时间，`C_fold(S,P)` 是一次 fold 时间，`C_setup(S,P)` 包含 baby/F树、inverse、descent等随 P 的固定工作，则固定 D/P 的近似模型为：

```
T ≈ C_setup(S,P) + I·C_point(S) + G·C_G(S,P)
    + (G−1)·C_fold(S,P) + 检查/收尾
```

固定 D/P 时，驻留与回退两条路径都趋向线性 B2，区别在各项常数。调节 D 时，增大 D 减少 I，却增大 P、F树和 inverse/descent成本；忽略 logarithm、取 phi(D)/D 在局部近似常数时，平衡 `aD+bB2/D` 给出 `D~√B2`、`T~√B2`。实际还包含 NTT 长度的 2 的幂跳变、47-smooth D、批次数取整和不同位数的 packing，不能期待精确幂律。

正固定开销也会压低有限区间的幂指数：对 `T=A+C·B2`，局部指数为 `d lnT/d lnB2 = C·B2/(A+C·B2)`，小于1，且随B2增大趋向1。因此固定D的两三点幂拟合恰好接近0.5，不能单独证明平方根复杂度；应同时检查批次线性模型、截距和更高B2。

- 一次长度 L 的 radix2 NTT 约有 `(L/2)log₂L` 个蝶形；每个完整多项式卷积通常涉及两个 forward、pointwise 和 inverse，实际复用、低层根特例、scale与batch由具体路径决定。这里只是操作数量级，未采集 cycle/stall，不能换算固定 GPU 周期。
- 点算术的当前 Mersenne 模乘主要乘积为 W² limb MAC，加 O(W) 折叠/旋转。six-multiply xADD 每步主导约 6W² MAC；giant 主体随 I 增长，种子和分段修正另计。
- G 叶 `[-X,Z]` 的逻辑系数数据约 `16WI B`；点 X/Z 坐标主体也是 `16WI B`，含分段 seed/scratch 时更大。device leaf frontend 可使这些字节留在设备，不能把逻辑生成量直接当 PCIe 传输量。
- 一个 P 阶 owner 多项式为 `8W(P+1) B`。回退路径需要的 host snapshots/结果回读随批次增加；其数量级为 `O(GPW)`，固定 D/P 时也是 `O(B2·W/D)`。驻留统计的 avoided 字节表示相对旧路径省下的逻辑 payload，不是全曲线实测 PCIe 总量。
- owner 初始化上传 F、inverse、模数约 `8W[2(P+1)+1] B`；实际后续 seed/root handoff与最终读回以日志为准。owner 未启用时 owner 自己的计数为0，**不表示整个 fold 的 H2D/D2H 为0**。
- small/output、raw G A/B 和坐标按窗口、batch和最大层宽分配，不会随全部 B2 线性保留。`outputwindow.d2h_words×8` 是该输出接口实际记录的 payload，不包含其他接口。

更具体地，对一批 n≤P 个 G 叶，[raw frontier](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4387) 需要 A=`16nW B`，紧凑 B=`8[n+ceil(n/2)]W B`；缓存容量可能保留更早的较大层，所以最终字段是历史容量峰。[坐标窗口](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8994) 为 `16CW B`，C是一次生成窗口的 giant点数，可能跨多个G批次。[输出窗口](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3898) 的设备容量为 `8·d_out_cap B`，两 host pinned buffer 合计为 `8(cap0+cap1) B`；后者是主机锁页容量，不能列入显存。实际输出words计数仅在host_output时增加。

阶段比例直接使用每条曲线的 `init/main/total`；`real_batched_split` 与NTT归属计时有嵌套，不把它们相加恢复total。尤其 [real_batched_breakdown.ntt_seconds](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11506) 包含乘法wrapper/host工作，不是纯GPU event耗时；不能据该字段比例直接归因GPU计算或PCIe瓶颈。

## 5. 实验顺序和复现

基础 48 条：3 条预指定 warmup；27 条三位数×三 B2×三预算；4423 bits 再测 2.6×10¹²、8×10¹²的三预算（6条）；固定 D=1021020/P=92160，三个 B2 各按 owner1024/128/128/1024 跑 ABBA（12条）。固定 D 起点为 B2=2.6×10¹¹，确保 G>1，确实使用 fold owner。

另加预先定义的 640 MiB 边界：4423 bits、B2=8×10¹²、D=1531530/P=138240，owner 所需约 664.46 MiB；同 D 按1024/640/640/1024运行，直接跨过用户提出的默认边界。基础网格每点一次，固定 D 和边界每模式每点两次，不报告置信区间，也不把小于波动的差值当稳定收益。

[运行工具](D:/code/MPA-OpenCl/tools/bench/bench_stage2_budget_scaling.py:1)、[真实形状查询](D:/code/MPA-OpenCl/tools/test/stage2_budget_shape_probe.cu:1)、[查询构建](D:/code/MPA-OpenCl/tools/build/build_stage2_budget_shape_probe.ps1:1)、[640边界扩展](D:/code/MPA-OpenCl/tools/bench/extend_stage2_budget_boundary.py:1)、[结果分析](D:/code/MPA-OpenCl/tools/bench/analyze_stage2_budget_scaling.py:1)。路径参数可替换为其他冻结版本；需要对应的19原始依赖及 manifest。

可复用工具允许单独指定 `--big-mb`、`--small-big-mb`、`--owner-mb`、`--small-owner-mb` 和 `--arena-mb`；值以MiB计。默认值即本轮三策略。`--prepare-only` 先完成CPU/GMP保存点和真实形状核验，不执行完整Stage2；`--resume` 按既有plan继续，新的预算参数不会改写已冻结plan。修改预算须使用新目录。历史策略名large_resident/large_owner128在自定义预算下仅为配置标识，是否驻留应读enabled/fallback。640边界扩展是本轮默认形状的专用控制，只对其48条基础矩阵使用。

完成计时后额外核验了 `big3072/small_big512/owner640/small_owner0/arena6300` 的48条**规划**及真实C++shape；并非另外48条性能测量，见 [custom_budget_plan](D:/code/MPA-OpenCl/build_cuda_cmake/_budget_scaling_20261005/custom_budget_plan/plan.json)。所有48规划满足对应big/arena约束；owner不参与D排序，记录超预算时预期回退。

[预算参数门禁](D:/code/MPA-OpenCl/tools/test/check_stage2_budget_options.py:1) 另验5种无效参数均exit2、48规划全部满足预算，且只改owner预算时D/几何保持原矩阵值。此门禁不启动性能曲线。

```powershell
tools/build/build_stage2_budget_shape_probe.ps1 -Build build_cuda_cmake/budget_probe
python tools/bench/bench_stage2_budget_scaling.py `
  --exe build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe `
  --sources build_cuda_cmake/_point_scratch_20261005/native_calibrated/sources `
  --shape-probe build_cuda_cmake/budget_probe/stage2_budget_shape_probe.exe `
  --output build_cuda_cmake/budget_study
python tools/bench/extend_stage2_budget_boundary.py --study build_cuda_cmake/budget_study
python tools/bench/bench_stage2_budget_scaling.py `
  --exe build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe `
  --sources build_cuda_cmake/_point_scratch_20261005/native_calibrated/sources `
  --shape-probe build_cuda_cmake/budget_probe/stage2_budget_shape_probe.exe `
  --output build_cuda_cmake/budget_study --resume
python tools/bench/analyze_stage2_budget_scaling.py `
  --study build_cuda_cmake/budget_study --output build_cuda_cmake/budget_analysis
```

本机证据根目录：[budget_scaling_20261005](D:/code/MPA-OpenCl/build_cuda_cmake/_budget_scaling_20261005)。`study_v3` 是正式数据；初版 Stage1 预检查误要求新 lcm 点与旧基准 Q 相等，失败目录保留，未计入 GPU 时间。另一预备版本固定 D 最小 B2 会只有一批 G，不适合作为 owner 因果对照，也保留但未测量。

采样工具曾因 NVML 利用率读取失败退出：已完成17条保留；第18条计算最终完成，但采样链中断，原日志保存在 `telemetry_aborted`，重新完整测量。修复只把采样缺失标记为空值和接口状态码，未改 Stage2 数学或预算。运行驱动每个版本按 hash 归档，准备版本、修复前、修复后的来源分别保留。

## 6. 结果

<!-- RESULTS_START -->
**52条接受运行：3条预热＋49条正式测量；源码/结果/检查审计52/0。** 另有1条因采样中断而排除并重跑的完整计算，保留原日志。正式网格33条、固定D重复12条、640边界重复4条。所有同`(S,B2,D)`组的leaf、因子、S4工作量和GMP/oracle覆盖一致；21个不同几何/输入组，arena overflow全部为0。

### 6.1 三位数 × 两类预算的完整时间

时间单位s。宽预算两个策略使用相同D/P/G；小工作区策略列出自己的D/P/G。每个网格单元一次完整曲线。

|S bits|B2|宽预算 D/P|G宽|T owner1024|T owner128|小工作区 D/P|G小|T small_big|
|---:|---:|---|---:|---:|---:|---|---:|---:|
|2203|8e10|330330/31680|8|3.148890|3.194285|330330/31680|8|3.157610|
|2203|2.6e11|570570/51840|9|5.988147|6.011848|570570/51840|9|5.989811|
|2203|8e11|1021020/92160|9|11.027070|11.745797|746130/69120|16|11.412476|
|4423|8e10|330330/31680|8|7.072743|7.437019|330330/31680|8|7.112054|
|4423|2.6e11|570570/51840|9|13.611051|14.249673|390390/37440|18|14.632169|
|4423|8e11|1021020/92160|9|25.300168|26.357189|390390/37440|55|35.650186|
|4423|2.6e12|1531530/138240|13|47.598484|49.628891|390390/37440|178|106.106017|
|4423|8e12|1531530/138240|38|109.105587|116.805879|390390/37440|548|317.557874|
|8191|8e10|330330/31680|8|20.719035|21.315393|210210/20160|19|24.552731|
|8191|2.6e11|570570/51840|9|36.476892|37.658658|210210/20160|62|59.538724|
|8191|8e11|746130/69120|16|70.989118|73.170801|210210/20160|189|164.405675|

1024MiB owner在本轮所有形状均驻留；small_big也全部驻留。128MiB在2203bits前两档驻留、第三档回退；4423/8191bits三档起点即回退。具体enabled/fallback与容量见逐条CSV。

![三位数B2时间与预算](D:/code/MPA-OpenCl/docs/figures/stage2_b2_budget_runtime.png)

### 6.2 分段增长指数

`alpha = ln(T_high/T_low)/ln(B2_high/B2_low)`；这是有限区间实测指数，不是渐近复杂度证明。

|S bits|策略|8e10→2.6e11|2.6e11→8e11|8e11→2.6e12|2.6e12→8e12|
|---:|---|---:|---:|---:|---:|
|2203|large_resident|0.5453|0.5432|—|—|
|2203|large_owner128|0.5365|0.5959|—|—|
|2203|small_big|0.5432|0.5736|—|—|
|4423|large_resident|0.5554|0.5516|0.5362|0.7380|
|4423|large_owner128|0.5517|0.5472|0.5369|0.7616|
|4423|small_big|0.6121|0.7923|0.9254|0.9753|
|8191|large_resident|0.4799|0.5924|—|—|
|8191|large_owner128|0.4829|0.5910|—|—|
|8191|small_big|0.7515|0.9037|—|—|

**两个独立反例：** 4423bits的large_owner128始终非驻留，但D从330330→570570→1021020→1531530增长时，前三段alpha为0.5517/0.5472/0.5369，仍接近平方根；small_big始终驻留，但D390390/P37440固定后，最后两段alpha为0.9254/0.9753，接近线性。因此不能把“owner回退”当成增长指数切换的充分条件，也不能把“驻留”当成平方根增长的保证。

宽工作区驻留4423bits到8e12为109.11s，小工作区驻留为317.56s，约2.91倍；观察整卡显存最大used从5444.6→2284.6MiB。8191bits到8e11为70.99→164.41s，约2.32倍；观察整卡used5324.6→2278.6MiB。此处D/批次和检查工作量随形状变化，比较的是实际完整生产路径成本，不是只比较单次NTT。

### 6.3 固定 D 的 ABBA

S4423，D1021020/P92160；big峰3072MiB；驻留owner442.97MiB。每模式每B2两条；括号为观察min～max，非置信区间。

|B2|G|驻留均值s（min～max）|回退均值s（min～max）|回退额外耗时/驻留|
|---:|---:|---|---|---:|
|2.6e11|3|15.457714（15.446695～15.468733）|15.747537（15.738237～15.756837）|1.875%|
|8e11|9|25.280504（25.273013～25.287995）|26.384395（26.350503～26.418287）|4.367%|
|2.6e12|28|59.024554（59.001167～59.047941）|63.897976（63.717608～64.078344）|8.257%|

三档均值对G做说明性线性拟合：

- 驻留：`T ≈ 9.9112 + 1.7507·G s`，R²=0.999765。
- 回退：`T ≈ 9.5072 + 1.9377·G s`，R²=0.999596。

只有三个横坐标，R²不能替代独立验证；没有将这些系数发布为D模型。结果支持固定开销＋重复批次成本的解释，owner回退增加每批成本。

![固定D与不同内存统计范围](D:/code/MPA-OpenCl/docs/figures/stage2_fixed_d_memory.png)

### 6.4 直接跨过默认640MiB边界

S4423/B2=8e12/D1531530/P138240，big3072MiB，owner需求664.4574MiB；只变owner预算。

|预算MiB|实际路径|两次total/s|均值/s|
|---:|---|---|---:|
|1024|驻留|111.444409, 110.450216|110.947312|
|640|fallback=budget|119.963008, 119.644525|119.803766|

预算回退额外8.856454s（相对驻留 **7.983%**）；恢复驻留相对回退省7.392%。两模式NTT full峰相同；整卡观察最大used分别约5444.6/4778.6MiB，不能称为本进程峰值。

### 6.5 模块容量与主机/整卡记录

**下表均为宽预算驻留网格的模块历史容量/峰，不是同时存活清单；不相加为进程峰。** MiB。完整三策略逐条记录在CSV。

|S|B2|NTT big|NTT full_peak|owner|rawA / rawB|坐标|S4设备输出|
|---:|---:|---:|---:|---:|---|---:|---:|
|2203|8e10|384.00|424.45|76.14|16.92 / 12.69|129.34|16.92|
|2203|2.6e11|768.00|833.38|124.59|27.69 / 20.76|243.36|27.69|
|2203|8e11|1536.00|1626.32|221.49|49.22 / 36.91|295.31|49.22|
|4423|8e10|768.00|831.69|152.27|33.84 / 25.38|258.68|33.84|
|4423|2.6e11|1536.00|1624.61|249.17|55.37 / 41.53|276.86|55.37|
|4423|8e11|3072.00|3185.97|442.97|98.44 / 73.83|295.31|98.44|
|4423|2.6e12|3072.00|3187.15|664.46|147.66 / 110.74|295.31|147.66|
|4423|8e12|3072.00|3187.03|664.46|147.66 / 110.74|295.31|147.66|
|8191|8e10|1536.00|1623.39|278.45|61.88 / 46.41|309.38|61.88|
|8191|2.6e11|3072.00|3184.30|455.63|101.25 / 75.94|303.75|101.25|
|8191|8e11|3072.00|3184.58|607.51|135.00 / 101.25|270.00|135.00|

以下单独列另一统计范围，依然不与上表相加。host commit字段是日志快照中的累计峰；pinned为模块容量。

|样本|整卡基线used MiB|整卡观察最大used MiB|日志host peak commit MiB|host输出pinned MiB|
|---|---:|---:|---:|---:|
|S2203, B2=8e11, owner1024|232.0|2880.6|3697|49.22|
|S4423, B2=8e11, owner1024|232.0|5086.6|6663|98.44|
|S8191, B2=8e11, owner1024|232.0|5324.6|7597|135.00|
|S4423, B2=8e12, owner1024|232.0|5444.6|7905|147.66|
|S4423, B2=8e12, owner640|232.0|4778.6|7903|295.31|

全批已观察整卡used最大5444.6172MiB；日志host peak commit最大7912MiB。完成曲线中有6个NVML占用率缺失点，标记为空值；显存读数可用，不把缺失占用率当0% idle。

### 6.6 阶段比例和传输量示例

宽预算驻留、B2=8e10三位数，均D330330/P31680/G8。各子phase是其自身计时区间，比例以full total作分母，不能将嵌套项相加。

|S|total/s|init占比|main占比|giant占比|G树占比|fold占比|
|---:|---:|---:|---:|---:|---:|---:|
|2203|3.148890|19.23%|80.77%|6.67%|28.30%|12.42%|
|4423|7.072743|21.97%|78.03%|11.27%|25.86%|12.02%|
|8191|20.719035|20.06%|79.94%|20.44%|19.63%|8.05%|

S4423/B2=8e12网格驻留样本（第34条，非边界重复均值），owner传输与避免的旧路径payload：

|统计接口|字节数|GiB|含义|
|---|---:|---:|---|
|fold owner H2D|154833144|0.144200|实际owner接口上传，含初始化/小metadata|
|fold owner D2H|77414712|0.072098|实际owner接口读回|
|fold avoided H2D|17152945600|15.974925|相对旧路径的逻辑避免量|
|fold avoided D2H|11424198240|10.639614|相对旧路径的逻辑避免量|
|root→fold D2D|2925201440|2.724306|实际设备root handoff payload|
|root→fold avoided H2D|2925201440|2.724306|相对旧路径的逻辑避免量|
|root→fold avoided D2H|2925201440|2.724306|相对旧路径的逻辑避免量|
|S4 outputwindow D2H|3356916080|3.126372|实际该输出接口payload；不含其他接口|

**以上避免量不等于实测PCIe事务，几个接口计数也不替代全进程传输统计。** giant group/bad-group及叶frontend避免量、NTT各缓存组件、检查覆盖、结果hash见CSV。

### 6.7 可保存的数据与审计

[52条CSV](D:/code/MPA-OpenCl/docs/data/stage2_budget_scaling_20261005.csv)、[回归/指数JSON](D:/code/MPA-OpenCl/docs/data/stage2_budget_scaling_20261005_summary.json)、[审计JSON](D:/code/MPA-OpenCl/docs/data/stage2_budget_scaling_20261005_audit.json)。原始每条engine/driver/JSONL/NVML及Stage1/GMP证据在study_v3；两个运行driver版本逐字归档，19原始源、probe生成源/二进制hash均核验。新命令行预算参数是在完成本轮计时后加到可复用工具的，实际测量driver以归档版本为准。

静态图在所有GPU计时结束后生成。当前图工具依赖Matplotlib3.11.2/NumPy2.5.3，安装在本机ignored实验目录；图表/表格/分析源/measurements hash在analysis_manifest.json中。
<!-- RESULTS_END -->

## 7. 内存峰值和局限

**各模块的容量/峰值不是同一时刻的分配清单，不能直接相加作为进程峰值。** NTT 的 `full_peak_bytes` 是该模块内部维护的完整 payload 峰值；big/small/table/fuse_base各自峰值也不应在不同生命周期下擅自相加。

本轮独立用 NVML 每约200 ms采样 GPU1 的整卡 used/free 和占用率。报告的是 **整卡已观察最大 used**，包含 context、驱动、其他应用、启动和收尾；采样可错过短峰，既不是保证捕获的真实峰，也不是本进程的显存峰。基线单独保留，不把“减基线”解释为精确进程分配。

主机方面，已有 [s2g_state](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:161) 用 `GetProcessMemoryInfo` 的 `PagefileUsage/PeakPagefileUsage` 报host commit；日志虽然标记 `host private`，实际读取的是前述字段。分析保留日志快照中两个字段的最大值，原输出按MiB取整。OS累计峰只截至相应快照，未保证在进程最终峰时再次采样；**这也不是物理RAM驻留集峰值**。host输出窗口pinned容量另列，不能把它加到commit峰上重复计数。本轮未测全接口PCIe事务或物理RAM峰。

三位数基础 B2 仅三个点，且每点一次；更高 B2 只扩展4423。GPU时钟、温度和CPU后台活动可能影响微小差值。固定 D 的重复样本只能支持该范围的近似线性关系，不构成无限 B2 的复杂度证明。未复刻原3367bits/B1=260m输入；若提供原save，可继续在独立空闲设备重复原生产范围。

## 8. 后续实现方向

1. 先统一缓存计账、真实payload和淘汰减账的公式，覆盖cache hit/grow/evict/rebuild/release及cap边界；当前约6149MiB账本不等于约3187MiB实际NTT payload。然后把big shape上限作为独立planner参数接入，并明确是否也禁止per-call超限分配。精确计账可能减少提前拒绝/淘汰，但GPU还要容纳owner/坐标/raw/output，不能据账本差值直接承诺大一档NTT一定装得下。
2. 为不同 S、big上限、驻留/回退路径分别标定 D；当前 legacy排序仅作实验控制。模型应允许在 owner 预算不足时继续增加 D，比较搬运代价与减少 G 的收益，不能把“非驻留”直接判定为无效候选。
3. 优先核验owner临时多项式别名。当前[生产step_loaded](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5276)的q在反转复制后不再用于数学计算，可考虑qb复用q；g在首次乘入t后不再用于数学计算，可考虑reverse复用g。按[实际布局](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5192)，q槽为(P+1)W、qb为PW，单独q/qb复用使9P+8→8P+8，省8WP B；g和reverse各为(P+1)W，单独复用使9P+8→8P+7，省8W(P+1) B；两项同时成立则为7P+7，省8W(2P+1) B。修正此前把两类槽位都近似按P计算的7P+8公式。本轮P=138240/W=70形状，664.457MiB理论变为590.629（q/qb）或516.801MiB（两项），可回到640MiB预算；当前M4423大界P126720已有驻留，不应仅凭节省容量预告时间收益。**尚未实现或验证**，必须先证明异步pack、oracle snapshots、digest和诊断读取已结束，再进行GMP/故障与真实曲线对照，并更新所有owner容量和D规划公式；同时核验NTT工作区与owner实际同时存活的分配。
4. 后续 B2 采样需同时记录 D/P/L/G/驻留原因，并按 D饱和、NTT长度跳档和owner回退分段分析；同一幂指数不能覆盖所有区间。

本轮新增实验和报告，不修改生产内核或发布默认。

文中源行号链接指当前工作区对应实现的位置；实测版本以冻结生产19源码和manifest为准。较新工作区NTT文件另含默认关闭的carry融合候选，本轮没有编译或启用该候选。
