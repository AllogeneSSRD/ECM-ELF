# ECM GPU Stage2：xADD6 与驻留流水线 D 重标定

日期：2026-10-04。此轮先减少椭圆曲线点运算，再修正自动选 D 对当前流水线的成本估计。xADD/D测量使用已提交的warp tile/常量根实现，随后完成cooperative outer v2；各阶段数据分别标注，均未使用Tensor Core。

## 1. 范围与约定

令模数为 `n`，位数 `S=ceil(log2(n+1))`，64bit limb 数 `W=ceil(S/64)`，Montgomery 基数 `R=2^(64W)`。NTT 素数为 `q=2^64−2^32+1`，变换长度用 `N(m,S)` 表示，避免将 NTT 长度与 ECM 模数混用。

生产规模测量在 GPU1 RTX4060 Laptop 8GiB/sm89 上串行执行。大形状为 M4423、sigma26、Stage1 choose12、B1=1000、B2=2011326186870。驻留 root/fold、设备 leaf、缩放下降、异步 oracle、carry 合并及 warp tile 都启用，batch64MiB、arena6300MiB，检查覆盖保持。GPU0 的外部生产进程未改。

算法调用数、payload 容量和观测墙钟分别报告。没有有效硬件周期/NCU 计数器，不把 `N log2 N` 当作 GPU 周期数。阶段时间存在嵌套，不能逐项相加制造新的端到端结论。xADD6固定D锚点中init约20.1%、giant17.3%、G树30.3%、fold15.4%、下降11.4%、inverse3.1%；CPU准备/传输隐藏在这些阶段内，不能把它们独立累计。

## 2. 六次模乘的差分加法

旧 xADD 的坐标是：

```
Xr = Zd · (XpXq − ZpZq)^2
Zr = Xd · (XpZq − ZpXq)^2
```

每个坐标需要两次交叉积、一次平方和一次差分点乘法，共 8 次 Montgomery 乘法。新路径使用：

```
u = (Xp+Zp)(Xq−Zq)
v = (Xp−Zp)(Xq+Zq)
a = (u+v)/2 mod n = XpXq−ZpZq
b = (u−v)/2 mod n = ZpXq−XpZq
Xr = Zd · a^2
Zr = Xd · b^2
```

两次求 u/v、两次平方、两次差分点乘法，共 **6 次**。b 与旧公式的符号相反，平方消除符号。两个模 2 除法恢复原坐标比例，使输出 canonical limb 与旧版逐字相同；因此原 Γ、projective leaf、存档恢复和根摘要的约定继续成立。

`halfmod(a)` 在 a 偶数时右移；a 奇数时先加奇模数 n，再对 W+1 limb 精度的和右移。它保留最高进位，处理接近 R 的模数。2 在奇模数环上可逆；函数不依赖点坐标非零或可逆，也不依赖 n 为素数。模 2 除法是线性运算，对 Montgomery image 同样成立。

源码位置见本报告末尾的实现索引。xADD 使用 4 个原有 `NW` 局部数组，所有输出写入延迟到读取 p/q/diff 之后。内核以 bool 模板生成两套版本，由主机选择；梯形循环内部没有环境开关判断。`NTT_XADD6=0/1` 提供同二进制比较与回退，独立实验 exe 缺省 0。

### 2.1 可量化计算量、容量与传输

- 现有 SOS 乘法加 REDC 的主导工作约 `2W²` 次 64bit MAC/模乘，忽略线性尾部。xADD 从约 `16W²` 降到 `12W²`，减少 `4W²`；补充加减与两个 halfmod 为 `O(W)`，不能仅凭公式换算成 GPU 周期。
- xdbl 仍为 5 次模乘。每个非首位 ladder bit 的 xADD+xdbl 从 13 次降为 11 次（−15.38%）；长 giant chain 的每次续点从 8 次降为 6 次（−25%）。种子 ladder、域转换与 segment 修正不属于每次续点的这 6 次。
- xADD 源码局部容量仍为 `4·8·NW` B/线程；Montgomery 临时数组与 ladder 状态保持。编译器可改变寄存器及 local memory 分配，源码数组容量不是实际显存使用量。
- 点批输入输出、固定点和 segment 产品的元素个数保持，新增算法持久显存为 0，H2D/D2H/D2D payload 没有因 xADD6 改变。NVML 的8次运行峰值均为 **5535MiB**；与上轮5544MiB的差别不构成算法容量下降证据。

### 2.2 正确性与性能证据

独立 primitive 门禁 **67/0**。覆盖 63/64/65/129/257/513/1025/2049/4097/4423/8192bit 的奇模数，普通域和 Montgomery 域、0/1/n−1/n−2、最高加法进位、Z=0及非单位元素。每次 fixture 包含 1280 xADD 比较、2560 坐标字组及1280 halfmod，分别覆盖旧/新两个模板和 separate/p/q/diff/cross 五类输出 alias；GMP 独立计算旧多项式坐标，比较原始 canonical 结果。故意损坏新模板首字时必须拒绝。

同一实验 exe 的 ABBA+BAAB 共8次，每模式4次，只有 `NTT_XADD6` 改变。D 固定1231230、P115200、I1633592、G15。

- 完整 Stage2 `init+main`：**72.830970→68.27037675 s，少4.56059325 s / 6.2619%**。
- main：57.05080775→54.54354725 s（−4.3948%）；init：15.7801625→13.7268295 s（−13.0121%）。
- giant：14.21325→11.78925 s（−17.0545%）。G树20.4955→20.666、fold10.4655→10.4905、归约1.974→1.97825 s，未显示相应加速。
- 旧版 full 样本为72.845061/72.648254/73.043687/72.786878 s；新版68.067077/68.859117/68.549726/67.605587 s。样本数有限，无置信区间，不保证其他模数/设备/形状同幅度收益。
- 外部进程墙钟82.5165→77.202 s，包含 Stage1 和驱动工作，不用它替代 Stage2 指标。

8次保持：403批NTT、1979251对乘法、40218760归约系数、2400 mandatory 自检、66139 GMP 样本/1126 oracle jobs/4 full checks；carry8241块/252 finishes/1893402 slices/max_group222。错误0、pending0、fallback0，oracle签名`b9cbd2041266767a`，根sum`033a77303713f3a6`/xor`d16047fb14d39b21`，最终115200叶/8064000字/FNV`10619321735931855904`一致。

仿射 Q 的独立参考 SHA256 为 `33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`。host传输摘要均为H2D4.71/D2H1.34GiB，pack峰2GiB、临时pack payload峰0，private提交峰均约7.6GiB；private不是物理RAM驻留。

证据：[8次量化](D:/code/MPA-OpenCl/build_cuda_cmake/_xadd6_20261004/quantitative.json)、[ABBA](D:/code/MPA-OpenCl/build_cuda_cmake/_xadd6_ab_20261004/results.csv)、[BAAB](D:/code/MPA-OpenCl/build_cuda_cmake/_xadd6_baab_20261004/results.csv)、[primitive 门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_xadd6_20261004/primitive_gate/summary.json)。测量用 immutable exe SHA256 为 `b09b96f49c22f373ed814cc2e1408ef62b098495ad25258b679a9a6981d18302`；对应 xADD-only CU `d11e25e8a0cf1a95eeadb85e4477398314222f6055791e5ae0bdc719b7e500c6`，NTT `f7c98f0bed2b00661f480f40b447098a0947654b53a5f31b794ebe5bf3d7acd7`。后续主机选 D 源码修改另有编译产物，不覆盖这份测量二进制。

## 3. D 不能只按 giant 数或 P 的线性成本选

`P=φ(D)/2`，`I=floor(B2/D)+2`，`G=ceil(I/P)`，`I=aP+r`。D 增大降低 giant 数，增加 baby/F树/下降工作。当前驻留流水线与旧§56.1的大量拷贝路径阶段权重不同，需要重新拟合。

φ(D)/D 是单位剩余类密度，不能作为素数覆盖概率加入速度评分。对于 p>D 的素数，p 自动与 D 互素，±baby residue 覆盖相应单位类。当前 giant i≥1、baby j≤D/2，对应整数候选 iD±j；最低边界约D/2，D的素因子由Stage1负责。改变 D 仍会改变下界附近和 overshoot 的几何范围，不能宣称不同 D 的测试集合逐项相同，也不能要求它们的叶哈希/采样数相同。ECM覆盖规则和算法未因成本模型修改。日志的below_first_giant仅估计(B1,D/2)中的整数素数个数，并非遗漏因子个数：点的阶还可能在已测试倍数处命中。

### 3.1 实际 NTT 尺寸与树工作特征

对 m 系数乘法，`slot_bits=2S+max(1,ceil(log2 m))`。选择最大可用 bpw，令 `sw=ceil(slot_bits/bpw)`，满足精确性界 `m·sw·(2^bpw−1)^2<q`；`N(m,S)=nextpow2(2m·sw+1)`。C++模型直接查询真实 multiply backend；Python工具复现整数界供离线交叉检查。

定义 `U(m)=N(m,S)·log2 N(m,S)`。它是经验拟合特征，包含蝶形和全数组pass的尺度；不表示单次卷积的准确算术量或访存量。树与 Newton 的特征为：

```
T(P) = sum over h=1,2,4,...<P: floor((P+h)/(2h)) · U(h+1)
V(k) = sum over m=min(2m,k), from m=1 until k: 2 · U(m)
```

T 计数每层 full pair 与最后一个 partial pair，passthrough 节点不乘。当前 multiply 将不等长操作数补齐到共同 m，最后 partial pair 也按该实际调用 shape 计费。V 对应每个 Newton 步两次多项式乘法，最后一步按截断目标 k 计。

顶部子树长度由 **严格小于 P 的最大2次幂 h** 决定，不能直接使用 P/2。P132480 超过2¹⁷，顶层一侧有131072叶，operand m131073 的 N 达2²⁷；旧P/2估算会错报2²⁶。这解释 D1411410 相比 D1381380 的尺寸跳变。

内存筛选保留现有 conservative arena 估计 `A(P+1)+2A(floor(P/2)+1)`，其中 `A(m)=8(3N(m,S)+2m−1)` B。它是筛选量，不是 live VRAM；当前 full buffers 可共享/驱逐，不能把真实 top N 的两份完整workspace简单叠加。模型日志另报真实tree/fold N，实际 arena预算、live headroom、allocation/eviction仍决定可用性。

驻留 fold owner 的最大 payload为 **`8W(9P+8)+48` B**，按 `NTT_FOLD_DEVICE_MAX_MB`（默认640）过滤自动候选。该预算不包含共享NTT workspace和小控制表；不能与NVML峰混为一谈。D1381380/P126720 的 owner为638673328 B（约609.09MiB）。

### 3.2 分阶段预测公式

成本由以下正系数项相加，单位为秒：

```
init    = cb·P·max(1,log2 D−2) + ca·P + cf·T(P)
giant   = cg·I·(6 + 22·log2 B2/64)
gtrees  = cgt·(a·T(P) + T(r))
fold    = cfold·(G−1)·U(P+1)
descent = cd·T(P)
inverse = ci·V(P+1)
accum   = cp·P
glue    = cr·G
total   = init+giant+gtrees+fold+descent+inverse+accum+glue
```

giant 特征近似两个种子ladder/64点chain段及6模乘续点，非逐点精确指令数。F树拟合时间取init减baby/affine，包含固定初始化和自检残差；glue为main减已报告阶段。改变mandatory检查强度会改变权重，因此新模型检查采样参数也受scope约束。

10个系数通过单特征非负最小二乘 `c=max(0,sum(x·t)/sum(x²))` 获得，来自6条显式D曲线和4条xADD6基线曲线：

```
cb=3.4253945078511067e−6    ca=2.639374706676053e−5
cf=1.7447305084417047e−10   cg=3.706877015843896e−7
cgt=7.758513401074988e−11   cfold=2.1353197194212299e−10
cd=3.9597124012586936e−10   ci=1.5652213657735564e−10
cp=1.486006614455004e−6     cr=.0382151508709496
```

阶段权重受调用规模/缓存/CPU准备/设备时钟影响，尤其partial tree的单shape近似并不精确。本模型服务于排序；不能据系数承诺具体秒数。

### 3.3 测量与适用边界

大 B2 显式D按正序后反序测量，每个D两次：D570570为106.000549/109.658017 s，D1381380为64.743013/62.405344 s，D1411410为72.267586/70.961979 s。D1231230的4条xADD6锚点均值68.270377 s。D1381380相对该手动锚点的观测均值63.574179 s，少约6.88%；这组重新选shape的比较不属于固定工作量 xADD A/B。

fit内误差约−4.20..+2.35%，leave-D-out约−4.16..+4.18%。之后冻结系数，以B2=1e11做独立holdout：D330330为13.966437/13.997466 s，D510510为15.189146/15.374536 s；分别预测12.984006和14.360625 s。误差−7.24..−5.45%，排序方向通过，绝对秒数偏低。更早的D330330/510510首测13.797811/15.061152 s没有进入fit；D1231230在该小界下G1直接余式路径26.784068 s，亦不用于驻留fold拟合。

47-smooth D≤200000000共379419候选。auto只在当前校准scope启用经验模型：RTX4060 Laptop、精确M4423、B1=1000、B2=1e11..2011326186870、单曲线、xADD6/warp/current resident/check配置、t12/M4、batch64/chain64。还要求G≥2和owner/arena预算可容纳。其他设备/模数/算法开关/检查参数，以及显式G1或owner超预算D，日志报fallback并使用原模型。显式D仍按用户指定执行。

大界离线首选 **D1381380/P126720/I1456028/G12**，预测63.869231 s。小界首选 **D330330/P31680**，预测12.984006 s。二者是该候选集合和预算内的经验排名，并非数学意义的全局最优；其他架构需要新测量。

`NTT_D_MODEL=0` 可恢复旧模型，实验缺省0。`--d-plan-only` 仅实验入口提供：创建CUDA上下文后打印候选、features、最终D并返回，`curves_executed=0`；生产驱动没有这一参数，防止把规划误当曲线完成。`d_scan_wall` 单独报告CPU选D时间，明确不在既有`stage2_full_wall`中；端到端验收必须检查驱动时间。

数据：[D六次测量](D:/code/MPA-OpenCl/build_cuda_cmake/_d_calibration_20261004/measurements.json)、[fit与leave-D-out](D:/code/MPA-OpenCl/build_cuda_cmake/_xadd6_20261004/final_d_fit.json)、[小界holdout](D:/code/MPA-OpenCl/build_cuda_cmake/_xadd6_20261004/holdout.json)。

## 4. 验收、生产接入和下一项 NTT

xADD/D最终实验编译496.8 s、link2.8 s，exe SHA256 `def0019dd87f6c57c97de3c65193e823004b2c1d4829a9fdfce6c69877468d5a`。该二进制复测primitive67/0、完整Stage2 188/0，20项D规划检查通过。P2最终生产版sm89/CUDA13.3编译成功，CUDA501.7 s，exe4038144 bytes、SHA256 `f85ead72d6e68a5952c5affcd9df1be32002428c08b2376a2f2b396a0e55f7f1`。生产入口30/0，涵盖基础21项、CUDA失败队列保留、已有因子、实际大save、warp/xADD默认和显式回退及自动D两种界；另一次M8 cooperative小界save恢复通过。P2收尾时默认xADD6=1、D模型=1、cooperative=0。生产回退开关为`NTT_XADD6=0`与`NTT_D_MODEL=0`，可以独立设置。每条save曲线仍使用自己的sigma/B1/Q，worktodo解析与存档格式保持。之后的尺寸策略和新D系数见[P3报告](D:/code/MPA-OpenCl/docs/STAGE2_NTT_SHAPE_D_CALIBRATION.md)。

最终save恢复：显式D1231230为init14.395585/main53.738988/full68.134574 s；自动大界D1381380为13.269998/48.459621/61.729618 s，选D另.144891 s；小界D330330为3.345562/10.091565/13.437128 s，选D另.150954 s。叶哈希分别为10619321735931855904、4244971527793015097、7549663880496122317。它们是接入验收单次时间，不替代A/B。用户示例仍选择961–970，xxx拒绝并保留队列；冻结因子59649589127497217正确。cooperative实际save启用时旧D系数明确回退，最终小界叶哈希同上。

证据：[最终生产30项](D:/code/MPA-OpenCl/build_cuda_cmake/_xadd6_20261004/production_accept_v2/summary.json)、[cooperative恢复](D:/code/MPA-OpenCl/build_cuda_cmake/_xadd6_20261004/production_accept_v2/coop_accept.json)、[F85编译来源快照](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/production_f85_before_shape/build_manifest.json)。

## 5. 实现索引

- [halfmod:747](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:747)、[xADD6:769](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:769)、[模板分派:953](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:953)、[GMP fixture:4188](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4188)。
- [DPhaseModel:24](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:24)、[scope:10758](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10758)、[选D计时:10938](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10938)。
- [采集/整数shape:1](D:/code/MPA-OpenCl/tools/bench/calibrate_stage2_d.py:1)、[拟合:1](D:/code/MPA-OpenCl/tools/bench/fit_stage2_d.py:1)、[离线候选排名:1](D:/code/MPA-OpenCl/tools/bench/plan_stage2_d.py:1)。
- [xADD门禁:1](D:/code/MPA-OpenCl/tools/test/test_stage2_xadd6.py:1)、[D规划门禁:1](D:/code/MPA-OpenCl/tools/test/test_stage2_d_model.py:1)、[生产默认:6](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6)。
- [协作outer:9](D:/code/MPA-OpenCl/tools/bench/ntt_coop_outer.cuh:9)、[launch:83](D:/code/MPA-OpenCl/tools/bench/ntt_coop_outer.cuh:83)、[planner:1421](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1421)、[arena key:2169](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2169)、[cooperative GMP:4301](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:4301)。


## 6. Cooperative outer：v1回退、v2收益与P3计划

CTA256线程负责V个相邻coarse offset和R=2^M个radix坐标。M5..7取V32，M8取V16，radix数据在shared交换，global访问沿offset连续。保留DIF自然序→bit-reversed、DIT反向、批stride、B只读及tile中的pointwise/1N融合；没有新增全长8N scratch。NTT_FUSE_COOP_OUTER=1启用实验，M缺省8；t<5沿用原planner，M<5的尾pass仍走原kernel。arena键加入t/M/cooperative/compact以防缓存错误复用。

第一版减少pass但重复计算每个butterfly的twiddle，N2²⁷/M8纯卷积反而.197979→.229372 s（慢15.86%）。v2将twiddle按u复用；d>=ROWS由同row循环group，d<ROWS把至多128个根写入shared，保持所有row参与蝶形。coarse power只由row0生成，forward使用ping-pong根缓冲避免跨warp读取旧根时被覆盖，inverse预计算M组根。v2不改变数学变换。

每coarse offset/pass的twiddle模乘由MR/2降为R−1；coarse root平方由forward M·ROWS降为M−1、inverse (M−1)·ROWS降为M−1。数据蝶形MR/2次模乘保持。k27/t12/M8+7的两forward+inverse outer源码gl_mul调用约47.6875N→28.6875N；此为源码调用尺度，非硬件指令/周期。旧M4外层约36.625N（末尾无用平方可能被编译器删除）。

主数组每pass读写16N B/变换，两forward+inverse及额外读取B约48Np+8N B。k27/t12的pass由5→3，省96N B，即N2²⁷时12GiB/单slice卷积。增加的shared数据payload约16MN B/变换，shared/ALU/同步会抵消部分global节省，不能线性换算墙钟。

v2资源：M5 forward48reg/9984 B shared、inverse46/10752；M6为46/18432与46/19456；M7为53/35328与46/36608；M8为52/36096与45/36864。所有8个模板STACK/LOCAL0；M7/M8容量API允许2CTA/SM，实际occupancy需剖析。未增加持久点/多项式payload，根表按实际radix计入arena生命周期。

独立GMP门禁8/0：96组、27131904字前向频谱和独立逆结果；缓存0/1/0/1额外3145728字，compact/wide、cached/uncached/evicted、M5..8、nbatch3及故障拒绝。最新v2原路径warp0/1各216组/6854400字、4次模式切换98304字和14次生命周期/leaked0亦通过。完整188项是xADD/D阶段的二进制证据；v2使用上述独立NTT门禁和最终生产30项/实际save验收，未重复声称v2跑过188项。

纯NTT卷积同binary ABBA+BAAB，每模式4样本，每样本1warm+3次CUDA event计时；三稀疏系数、所有N输出在计时外与GMP卷积检查。未包含carry/REDC/pack和CPU Stage2工作：

- k24：M6 .021786539→.020449878 s（6.14%），M8约3.92%。
- k25：M8 .050125397→.043473323 s（13.27%），M6约4.11%。
- k26：M8 .099538433→.088127404 s（11.46%），M6约4.47%。
- k27：M8 .198025983→.184210092 s（6.98%）；M6约4.15%，M5仅.17%，M7慢4.29%。宽度并非越大越好。

最终生产exe用同一实际Stage1 save、显式D1231230、D模型0、检查保持，串行ABBA四条：69.595031/67.429160/67.296980/68.752405 s。full均值**69.173718→67.363070 s（2.61754%）**，main55.025652→53.2897545 s（约3.15%），init14.148066→14.073315 s。每模式仅2样本，没有置信区间。外部driver第一控制样本有额外开销，不能把它算成NTT收益。根sum3cf2f49cf1972d5d/xor3fafa10f6f7f6f62、最终叶10619321735931855904、403批/1979251pairs/40218760coeffs/2400自检/66139GMP/8241carry全部一致，错误0。

P2收尾时cooperative仍缺省0，自动D使用旧NTT权重；手动开启cooperative会回旧D模型。P3限定已测设备/t12/warp配置，按N选择M6(k24)、M8(k25..27)，其余shape保持原路径，再冻结新二进制重新采集多个D/独立小界holdout；实施结果见[P3报告](D:/code/MPA-OpenCl/docs/STAGE2_NTT_SHAPE_D_CALIBRATION.md)。Tensor Core与多个Stage2并行属后续独立实验，需要精确整数范围、VRAM/RAM预算及实际吞吐证据。

证据：[纯NTT量化](D:/code/MPA-OpenCl/build_cuda_cmake/_xadd6_20261004/coop_quantitative.json)、[v2门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_coop_v2_final_20261004/gate/summary.json)、[v2旧路径门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_xadd6_20261004/coop_v2_legacy_gate.log)、[生产A/B](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_coop_stage2_ab_20261004/quantitative.json)。v2纯probe SHA256 fd31e39a7fe9c6754fa8f6c2adc49b7f06dfabc1ab14df77a5af3f5ad14bc045；生产A/B使用F85EAD72…F7F1。build目录证据ignored，源码、门禁和构建入口随Git提交。
