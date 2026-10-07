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
- [DPhaseModel:52](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:52)、[scope:10762](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10762)、[选D计时:10952](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10952)。
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


## 7. Giant 相邻 seed：复用 `[D]Q` 与 ladder 双输出（2026-10-07）

### 7.1 已复现的问题与本轮算法

上一轮的小批次策略把每线程链长从64降至8，在4096点等短输入上有效，但完整坐标块后接4096点尾段时，seed 数量增加会抵消收益。GPU1、M8191/B1=1000/lcm/sigma26、D120120、I4096，本轮重放原 fe709 同二进制 ABBA，full 均值中位4.548666→4.347248秒（4.43%）；该对照仅复现原短链策略，不属于下面配对 seed 的收益。

原实现为每段独立求 `[iD]Q`、`[(i+1)D]Q`，还为每个坐标 chunk 求一次 `[D]Q`。新候选先在 GPU 缓存 `H=[D]Q`，再在 H 上执行标量 i 的 ladder。其末端本来同时持有 `[i]H` 与 `[i+1]H`，保留两者即可提供一段的两个起点；每段不再执行两次 ladder，标量也不再带 D 因子。

- [双输出 kernel](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1014)；[有界 launch](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1052)仍按 ladder cap 分批，维持 watchdog 边界。
- [曲线私有 base 生命周期](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8937)：按 workspace/Q/D 缓存，每条曲线只构建一次。D 改变时重新构建，workspace 析构释放。无跨 sigma 共享。
- CPU 读取 H 的 Z 并计算 gcd；不可逆或 infinity 时，整个 seed chunk 回退原算法，不在新 base 上继续放大非单位尺度。真实 `N=103×65537`、sigma26/B1=2 的 lcm 保存点，CPU 已证明 gcd(Z_H,N)=103；实际候选回退并保持原因子103。
- [交接](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9189)保持原 `2B+1` 个 Montgomery `(X,Z)` seed 的交错布局、末尾差分点、单点尾段以及后续 chain/段积/G树接口。代表元的尺度会改变，跨算法不能要求叶 hash 相等；同算法重复必须相等。
- `NTT_GIANT_SEED_PAIR=1` 默认关闭，并要求设备 seed 路径。旧 D profile和 native Auto B2 配置明确拒绝复用，见[设备模型范围](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10950)、[原生配置门禁](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:537)。本阶段没有发布新 cprof。

### 7.2 计算、数据与容量公式

记 W=ceil(Nbits/64)，C 为每线程的连续点数，第 q 个 chunk 有 n_q 点、B_q=ceil(n_q/C) 段，段起点 `i_b=clo_q+bC`。下面只计算 seed，假定标量非零、xADD6启用；一次 xDBL 用5次模乘，xADD 用6次模乘，标量 s 的旧 ladder 用 `5+11 floor(log2 s)` 次模乘。

原 seed 模乘数为：

\[
M_{old}=\sum_q\left[5+11\lfloor\log_2D\rfloor+
\sum_b\{10+11\lfloor\log_2(i_bD)\rfloor+
11\lfloor\log_2((i_b+\epsilon_b)D)\rfloor\}\right],
\]

其中 epsilon_b=1，只有单点尾段为0。候选单位 base 路径为：

\[
M_{pair}=5+11\lfloor\log_2D\rfloor+
\sum_q\sum_b[5+11\lfloor\log_2i_b\rfloor].
\]

后续 chain 的 `6 Σ_qΣ_b max(0,min(C,n_q-bC)-2)` 次模乘保持。精确 Mersenne 点乘每次主导 SOS 工作约 W² 个 limb MAC；以上公式减少模乘次数，不改变单次模乘后端。不能直接把模乘数比例当作墙钟比例：并行度、依赖、local 访存和 launch 都参与实际时间。

- 缓存新增 **16W bytes** 设备 payload；首次每曲线上传 D 共8B、读取 Z 共8W bytes并在 CPU 做 gcd。M8191 的缓存为2048B、Z读回1024B。
- 每个配对 chunk 不再上传 `(2B_q+1)` 个64bit索引，少 `8 Σ_q(2B_q+1)` bytes；扣除首次 D 上传后才是该接口净减少量。不是坐标或主体 PCIe 流量下降。
- seed 输出仍为 `16W(2B_q+1)` bytes，原 workspace 容量和坐标输出保持。候选目前仍构造原 js 主机向量以保留检查/回退合同；尚未减少这部分 CPU 数据生成或临时显存。
- 新 kernel 的 local/register 资源不包含在上述 payload 中。模块峰值不能相加成进程峰值，需结合实际分配生命周期和 profiler。

### 7.3 当前证据与状态

实际候选 native SHA256 `d53f2de116271586f956dfd51d11652e8d60c22b6cb6d5c3419dba412b7d7930`，PTX3/outer0/sm89/CUDA13.3；CUDA编译509.4s。[编译 receipt](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/native/build_manifest.json)及同目录 sources/raw SHA快照保留。原生产893字节未变。测试工具会在每条曲线前后验证 binary、save、依赖和采集工具身份。

[正式矩阵](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/matrix/measurements.json)：M2203/4423/8191、B1=1000/lcm/sigma26、D30030、I4096/24977、C8/64、owner0/arena4096。12种形状，每种原/配对各4条，ABBA再BAAB，共96条计时、2条预热；每种另做完整 seed、segment、affine 门禁，共12条。全部必需算术检查通过；同策略叶指纹及跨策略 proper factor 集合保持。每模式4条，未提供置信区间。

C8 的 full 均值减少：2203位分别6.44%/15.45%，4423位6.31%/17.66%，8191位17.41%/20.90%。M8191/I24977 的 giant 为1.37575→0.668秒（51.44%），full为3.306406→2.615402秒（20.90%）。C64 full收益范围0.88%..13.75%，小输入噪声及固定开销不可忽略；没有据此更改全局链长。

[块尾对照](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/chunk_tail/measurements.json)：M8191/D30030/C64，I8192与I136576（132480完整块＋4096尾段），各原/配对4条及一个独立全点门禁。完整块尾场景 full 减少12.20%，giant减少30.68%；base_builds=1证明跨两个 chunk 使用同一 base。这里两侧强制 chain/C64，不等于已经验证默认 ladder尾段与自适应C8的最终策略。

[原生门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/native_gate/summary.json)26/0：CPU独立生成的 lcm Stage1点、2/65/66点尾段、M127/521/1279、generic129、真实 nonunit base、此前两个独立 CPU 已验证的 known-factor save。覆盖 seed/段积/全点比较、proper factor 和不可逆 base 回退。它不构成16384位或任意模数的完整验收。

采集工具：[A/B及身份核验](D:/code/MPA-OpenCl/tools/bench/bench_stage2_seed_pair.py:1)、[实际原生门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_seed_pair_native.py:1)。初版两次收尾被拒，原因分别是错误要求诊断 clean=1，以及将 clean=0 诊断传入正式计时 parser；原日志保留，未改变算法检查或任何计时。矩阵的 collector_initial.py、collector_continuation_r1.py和continuation身份明确记录修正。继续门禁前重新核验完整96条的计划、分组、原始 SHA及原始计时，未删除样本或重跑挑选更快结果。

### 7.4 大 B2 同二进制对照与默认路径回归

[大界对照](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/large_b2/measurements.json)使用 GPU1 RTX4060 Laptop、M4423/B1=1000/lcm/sigma26、精确 B2=2011326186870、D1381380、I1456028、C64、owner640MiB/arena6300MiB。两侧使用同一个 d53 二进制，仅切换 seed_pair，ABBA+BAAB各4条，预热另列，全部必需GMP/oracle检查保留：

- full 均值 **38.9105725→36.7176005秒，减少5.63593%**；相应 curves/s 增加5.97254%。原范围38.784093..39.054437秒，候选36.527450..37.096580秒。样本数量有限，不作跨设备/生产B1的普遍收益承诺。
- giant 均值 **4.9210→2.7455秒，减少44.20849%**。候选6个坐标chunk只构建一次base；22751个配对ladder省去364064B标量H2D，新base为1120B、Z读回560B。主体NTT/坐标传输没有等比例减少。
- 两侧NTT模块 `full_peak_bytes=3341481200`（3186.684MiB），legacy_mallocs=0；此为模块容量统计，不是进程峰值。不能与历史生产39.04秒相减再计算一次独立收益。
- 另做一个计时外完整门禁：六个chunk的 **1456028个仿射点全部比较，零失配**，45508个seed和91002个段积检查通过。约270秒的诊断运行不进入上述均值；同策略叶指纹一致。

候选大界阶段均值用于判断剩余工作：G树12.5795秒（34.26%）、fold5.65425秒（15.40%）、下降6.21075秒（16.91%）、baby3.723秒（10.14%）、F树约2.61819秒（7.13%）、giant2.7455秒（7.48%）、inverse1.652秒（4.50%）。这些是引擎阶段墙钟，包含各阶段准备/等待，不能都归为纯NTT kernel时间。`init/main` 与这些子项有包含关系，不相加；F树由init扣除baby/affine估算，init其余开销包含在估算内。

[默认点后端门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/default_point_gate_r1/summary.json) **18/0**：七位宽/两种模式的Mont2048及xADD1280 primitive检查、generic128/130分派、两个实际M4423 save模型范围检查。初次误用了另一条有效Q，因固定旧leaf指纹不匹配被拒；保留失败目录，恢复原 `_fixed_d_20261005/native_accept/m4423.save` 身份后通过，没有放宽leaf检查。新 `--payload-accounting-v2` 仅把已有ledger使旧D profile失效这一事实纳入门禁预期，不改变算法。primitive的X=2 fixture不宣称是有效Stage1存档。

本阶段合计 **120条正式计时、6条预热、15条单独的全点门禁**；另有26项配对原生门禁、18项默认后端门禁和4条管理员profile曲线，按类别独立记录，不合并成同一种性能样本。初版采集器和错误锚点导致的拒绝记录也保留。

### 7.5 管理员 Nsight Systems / Compute 的硬件证据

两种profiler均通过管理员隐藏进程启动并正常退出，针对GPU1、M8191/D30030/I4096/C8、owner0/arena4096、同d53和同save。profile曲线检查通过，但profile会改变执行，性能结论采用上面的无profile A/B。

[Systems原路径](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/nsys_original/summary.json)与[配对路径](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/nsys_paired/summary.json)：原seed ladder grid17为0.776943秒；候选一次base ladder grid1为0.116387秒，加paired seed grid8为0.283781秒，合计0.400168秒。baby ladder约0.4506秒、后续chain约0.0655秒和段积约0.0709秒基本不变，符合减少seed计算的解释。

- 自身GPU事件window为2.276861→1.892554秒，无本进程GPU事件间隙 **0.422103→0.414444秒**；绝对间隙基本未变，百分比18.54→21.90%因有效计算缩短而上升。不能解释为整卡空闲比例或证明所有间隙来自CPU计算。
- 设备分配生命周期审计峰值 **260175088→260177136B**，正好多2048B；tracked device end_live=0。pinned host峰值两侧均21223944B。该trace范围含启动/自检/收尾，非单一Stage2计时窗口；动态local流量不等于这些显式malloc容量。

[NCU原seed](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/ncu_original/summary.json)明确跳过baby ladder，捕获grid17/block64；[候选seed](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/ncu_paired/summary.json)捕获唯一paired grid8/block64。18-pass replay，clock-control/cache-control均none，[原始指标摘录](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/ncu_selected.json)：

- 寄存器64→72/thread；achieved warp occupancy约4.11→4.16%，issue active约15.33→14.62%。这是小grid样本，不能仅根据寄存器数归因于occupancy限制。
- local load sectors **3627452056→561401456**，store sectors **2133572511→329772239**，分别减少84.52%/84.54%。这不是DRAM或PCIe字节，也不按18次replay相乘。
- long-scoreboard/issue-active为3.7320→4.0305，wait约1.4733→1.5090，依赖访存/执行等待仍突出。配对kernel采集时长774.390→284.813ms不含候选base计算，不能直接称完整seed或Stage2收益。
- Systems的 `localMemoryPerThread=0` 不证明没有local访存；NCU明确显示大量local操作，下一轮应以硬件计数器和完整A/B判断局部数组改写。

### 7.6 收尾与下一阶段

[最终审计](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_seed_pair_20261007/final_audit.json)重新检查完整ABBA/BAAB计划、原始log/result SHA、每条实际输入和检查、全部计时重解析、采集器初版/修正版身份、两类profiler报告、25项编译依赖。该配对seed阶段收尾于ca94c13时，源文件raw bytes与冻结编译来源完全一致；原生产893不变。证据打包在ignored `build_cuda_cmake/_stage2_seed_pair_20261007/evidence.zip`，同目录 `evidence_manifest.json` 记录逐文件与压缩包SHA；不加入用户已排除的data目录。

优先项：

1. **单点base的CPU GMP预计算**。GPU一个线程求H花0.116秒，独立Python数学探针20次中位约0.01939秒，提示CPU方案值得实现并A/B。Python数字不是native GMP时间或已实现收益；需保留单位检查、正确Montgomery转换/尺度和非单位回退，再比较总墙钟。
2. **按实际点数/位宽校准C和尾段分派**。配对减少seed代价，旧C8/64拐点失效；覆盖完整块、短尾、退化base后才决定默认策略。
3. **准备/等待及NTT热shape**。G树/fold/下降仍合计约66.6%的大界full；绝对事件间隙未消失，须区分主机准备、同步等待和kernel local/NTT成本，再考虑批处理或多曲线重叠，先约束RAM/VRAM总预算。
4. **生产16384位、独立精简cu、日志粒度**。逐项处理入口/几何、256-limb分派、除数表和非模板fold数组，使用有效宽位数保存点验收。稳定算法之后重新标定Auto B2成本；当前无新cprof，seed_pair仍默认0，未发布新的生产二进制。

## 8. 单点 `[D]Q` 的CPU GMP预计算（2026-10-07）

### 8.1 算法、作用范围与成本

§7中的配对seed减少了重复ladder，但在GPU以单线程生成一次base H仍占较大固定延迟。本轮在配对算法内增加 `NTT_GIANT_BASE_CPU=1`，只把这一点交给CPU GMP。配对seed、chain、坐标、段积和全部NTT仍由原GPU路径执行；同时要求 `NTT_GIANT_SEED_PAIR=1`。两个开关均默认0，原生产893未替换。

[CPU helper](D:/code/MPA-OpenCl/tools/bench/stage2_giant_base_host.cuh:1)从曲线已有的Montgomery images解码Q/a24，以普通域GMP求 `[D]Q`，再编码回相同radix。使用原八乘xADD公式，其尺度与xADD6严格相同；不做点的仿射归一化。所有点坐标输入读完后才写输出，保持ladder别名安全。唯一求逆是已知为单位的Montgomery radix，不求点Z的逆。

记 W=ceil(Nbits/64)，L=floor(log2D)，D≥6。本helper每次base用 **10+13L次GMP乘法、11+12L次模约简**，外加一次radix inverse、一次gcd及导入/导出。五次乘法/约简来自三项解码和两项编码；初始化一次xDBL为5乘，后续每bit为xADD8+xDBL共13乘。GPU原base采用xADD6，为5+11L次点模乘。CPU算术次数较多，实际时间取决于GMP的大整数算法和单线程GPU的依赖/访存，不能仅按次数或GPU峰值估算周期。

[base生命周期与交接](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8937)仍按每条曲线/Q/D缓存；Z的gcd非1时保留完整原seed回退。诊断 `NTT_GIANT_SEED_CHECK=1` 额外运行GPU base并逐字比较 **2W个Montgomery words**，涵盖单位和非单位base；这些额外kernel/读回只进入独立门禁，不进入clean计时。`real_giant_base`分别报告CPU/GPU构建次数、CPU算术时间、含传输/分配的构建墙钟、上传量和检查覆盖。

- 设备缓存两侧仍是 **16W bytes**，没有减少NTT或主体坐标显存。helper输出与转换向量合计24W bytes；GMP整数与算法内部scratch另计，不能把向量payload当作进程RAM峰值。
- 普通CPU路径一次上传 **16W bytes**，不再上传8B的D或读取8W bytes的Z。该base接口的双向总payload实际多 **8W−8 bytes**，但少一个GPU kernel和一次同步读回。它是固定延迟优化，不是主体PCIe传输量优化。
- 与GPU base相同的projective images使后续叶指纹应逐字一致，本轮明确要求跨CPU/GPU base也保持相同leaf，而不仅是proper factor集合。
- 新binary需重新标定成本，native Auto B2对CPU开关也明确拒绝旧profile。没有导出新cprof；16384位入口/设备分派仍待下一阶段。

### 8.2 三位宽同二进制计时

独立native SHA256 `48b322a885c09afbc129958048545dd0f20bba8773dd34998aae4a21254abd11`，PTX3/outer0/sm89/CUDA13.3；CUDA编译469.8秒，[编译与26项raw依赖](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/native/build_manifest.json)冻结。提前固定[实验计划](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/experiment_plan.json)，两侧都开启pair，仅切CPU/GPU base。

[矩阵](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/matrix/measurements.json)为GPU1、三精确Mersenne位宽/B1=1000/lcm/sigma26、D30030、I4096/24977、C8/64、owner0/arena4096；**强制CHAIN_MIN=0**，每case ABBA+BAAB，各4条/模式，共96条正式计时、2预热、12独立seed/段积/全点门禁。全部必需算术、实际输入、base/cache覆盖和跨base精确leaf检查通过。默认32768点阈值以下仍走直接ladder，不能只设pair/CPU开关就套用这里的短输入结果；最终默认链长/阈值策略尚须单独验收。

全部正式样本的base构建中位数：2203位 **9.696→0.309ms**，4423位 **35.708→0.7915ms**，8191位 **117.4415→1.9445ms**。其中CPU算术分别约0.272/0.737/1.8775ms；独立Python探针的19.39ms不是native GMP成本。新路径的固定构建时间已减少约96.8%..98.3%，其收益只计一次/曲线。

完整Stage2的有限样本结果：

- M8191/C8：4096点full **1.83425825→1.7272515秒（−5.83379%）**；24977点 **2.6082225→2.4943505秒（−4.36589%）**。giant分别少23.24%/17.25%。
- M8191/C64：full分别少4.07%/4.39%；M2203四种shape少2.73%..4.87%。
- M4423三种shape full少0.73%..4.89%，但24977点/C64 **反而慢1.25%**；该case giant仍少17.74%。全样本保留，说明NTT/准备/等待波动可盖过约35ms的固定节省。

以上无置信区间，不宣称所有输入或生产B1的总时长稳定改善；不把本轮比例与§7的比例相加。

### 8.3 实际门禁与8192位泛型边界

[原生门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/native_gate_r1/summary.json) **30/0**：每个配对CPU base都额外逐字对照GPU base；有效CPU lcm Stage1保存点、短尾、原有known-factor和nonunit base回退全部通过，所有case的raw proper factor集合也与原路径相同。

新增 `N=2^8192−143` 满limb泛型输入，CPU独立证明65个giant Z均为单位，GPU完整seed/段积/仿射比较通过。另保留 `N=2^8192−17` 的非单位giant案例：Stage1和H本身为单位，但CPU在 `[43D]Q` 得gcd1019，实际两侧也返回1019及另外proper factors。

首次把后一案例用于全仿射比较，在原mode0尚未执行CPU base时因21个非单位点失败。不能把这类点当成可逆仿射坐标；[独立诊断](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/generic_boundary_diagnosis.json)保留原错误/输入/log SHA。该案例改按known-factor/nonunit检查并保留，另加入单位案例做全点比较；没有修改算法、删计时或用单位案例替换掉非单位覆盖。此30项不构成16384位验收。

### 8.4 大界无稳定总时长收益；管理员Systems确认计算移除

[大B2对照](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/large_b2/measurements.json)保持M4423/B1=1000/lcm/sigma26、B2=2011326186870/D1381380/I1456028/C64、owner640/arena6300、同48b二进制，2预热、8正式ABBA+BAAB、1完整门禁：

- base构建中位 **51.9165→1.068ms**，giant均值 **2.73775→2.689秒（−1.78066%）**。
- full均值 **36.5268005→36.51862775秒**，差仅8.17ms/0.02237%；原范围36.460569..36.613161秒，CPU36.474773..36.600646秒，**未建立稳定总墙钟收益**。不能把base约50ms的节省直接加到整曲线预测或把较小输入的5.8%推广到大界。
- 六chunk只构建一次CPU base，全部1456028个仿射点零失配；额外GPU base逐字140words、seed6371120words和91002段积检查通过。NTT模块full_peak两侧均3341481200B、legacy_mallocs=0，跨CPU/GPU的精确leaf一致。

管理员[Systems GPU base](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/nsys_gpu/summary.json)/[CPU base](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/nsys_cpu/summary.json)使用M8191/D30030/I4096/C8/owner0/arena4096。两侧均pair1，只切base CPU0/1；进程exit0，曲线算术/输入和Q/leaf身份核验通过：

- 原有base grid1 ladder **0.116356秒消失**；baby grid45 ladder两侧0.450814/0.450808秒，配对seed0.285158/0.283606秒、其后段积/chain基本不变。资源仍REG72的配对kernel；本轮不改GPU算术kernel。
- 自身GPU事件window **1.871349→1.752383秒**；无本进程GPU事件绝对间隙 **0.391826→0.390781秒**，仍基本不动。百分比20.94→22.30%因计算减少而上升，不能解释为GPU占用回归或整卡空闲结论。
- [生命周期审计GPU](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/nsys_gpu/audit.json)/[CPU](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/nsys_cpu/audit.json) tracked device峰值同为260177136B、末尾live0；pinned host峰值同为21223944B。
- 实际H2D增 **2040B**，D2H少 **1024B/一次copy**，D2D不变，净增1016B正好符合8W−8公式。不能称主体传输削减。本轮没有重新采集NCU；§7的GPU local/依赖证据保留，不能根据Systems local=0推翻它。

[默认后端回归](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/default_point_gate/summary.json) **18/0**，原实际save模型范围指纹保持。全阶段为104条正式计时、4预热、13独立全点门禁、30项原生门禁、18项默认门禁、2条管理员profile，分别记账。

### 8.5 收尾判断与下一阶段

[最终审计](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_cpu_base_20261007/final_audit.json)核对26项当前raw编译依赖、全计划/样本/log/result SHA及计时重解析、CPU/GPU base覆盖、跨base精确leaf、实际profile kernel数量和传输差额；所有门禁通过。证据和采集器快照在ignored `build_cuda_cmake/_stage2_cpu_base_20261007/evidence.zip`，逐文件/压缩包SHA见同目录 `evidence_manifest.json`。原始非单位检查失败也保留，没有修改算法或删除样本来通过门禁。开发standalone构建的header增量依赖同步加入。

CPU base作为可选延迟优化保留，**默认仍关闭，生产893未替换**。大界剩余收益重点仍为G树/fold/下降和准备等待；本轮不改变旧Auto B2精度门限或发布成本文件。

下一阶段先把生产的 **16384位支持、独立精简cu与日志控制** 做成实际可验收结果，同时对比原默认、配对CPU seed及C8短尾策略，选择最终保留的算法。8192位泛型通过不能替代256-limb设备分派/归约/数组/几何和宽位数存档验收；8192-point ladder cap独立保留。此后再针对NTT热shape采集管理员NCU，避免继续把单个固定延迟的局部收益当作大界主要提升来源。
