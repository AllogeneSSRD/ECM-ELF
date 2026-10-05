# ECM CUDA Stage2：GPU baby 批量归一化（2026-10-05）

本轮接续 [短归约 D 标定与生产发布](D:/code/MPA-OpenCl/docs/STAGE2_SHORT_REDUCTION_D_CALIBRATION.md)。目标是减少 baby 阶段 CPU 大整数准备和 X/Z 回读，保留原单首项 F 树、非单位 Z 因子记录及 small-prime cache 合同。当前实验开关 `NTT_BABY_DEVICE=1`；生产默认尚未提升。

## 1. 原路径与替换范围

原 baby ladder 输出普通整数域的 X、Z，并将两个 P×W 数组读回 CPU。CPU 每256点计算前缀积、求一次逆元、反向传播每点逆元、计算负仿射值并物化 `[-X/Z,1]`。上轮 D1381380 的 CPU affine 均值3.4555秒，占 full 的6.10%；上一轮 Systems 最大2.784秒无本进程GPU事件的空隙与此处准备关联，未采CPU调用栈，属于源码关联推断。

新路径使用已有 ladder 的 `normal_output=false`，保留 Montgomery 图像 X·R、Z·R，省去每点两个末尾域转换。GPU 用8层二叉乘积树替代 CPU 前缀积；每个最多256点的根由 CPU 判断可逆性并求逆，然后 GPU 向下传播逆元、直接输出普通域负仿射常数。CPU 为每片常数补上首项1，继续使用原 F 树 frontend。正常组不回读 X/Z；坏组回读并转换为普通域，使用既有仿射/GCD语义。

实现索引：

- [设备乘积层](D:/code/MPA-OpenCl/tools/bench/stage2_baby_device.cuh:7)、[逆元传播](D:/code/MPA-OpenCl/tools/bench/stage2_baby_device.cuh:20)、[负仿射值](D:/code/MPA-OpenCl/tools/bench/stage2_baby_device.cuh:36)、[底层与坏组跳过](D:/code/MPA-OpenCl/tools/bench/stage2_baby_device.cuh:48)。
- [预算、buffer生命周期和实际调用](D:/code/MPA-OpenCl/tools/bench/stage2_baby_host.cuh:22)、[根逆元/域转换](D:/code/MPA-OpenCl/tools/bench/stage2_baby_host.cuh:65)、[坏组恢复与因子记录](D:/code/MPA-OpenCl/tools/bench/stage2_baby_host.cuh:86)、[逐字与cache检查](D:/code/MPA-OpenCl/tools/bench/stage2_baby_host.cuh:109)、[真实host调用fixture](D:/code/MPA-OpenCl/tools/bench/stage2_baby_host.cuh:132)。
- [同binary曲线A/B工具](D:/code/MPA-OpenCl/tools/bench/bench_stage2_baby_ab.py:1)、[独立GMP与host门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_baby_device.py:1)、[Python/CPU完整baby/F参考](D:/code/MPA-OpenCl/tools/test/test_stage2_real_baby.py:1)。

## 2. 域与精确性证明

N 是 ECM 的奇数复合候选，不假设为素数。W=ceil(bit_length(N)/64)，R=2^(64W) mod N，Mont(a,b)=a·b·R^-1 mod N。现有 Montgomery 模乘产生规范化的 `[0,N)` 输出。

每个普通 Z_i 的设备图像为 Z_iR。乘积树父节点为 Mont(AR,BR)=ABR；短尾缺少右子节点时直接复制左图像，相当于乘以单位元R。因此每个根为 `(∏Z_i)R`。CPU 求该根的逆元后乘R，上传普通域的 `(∏Z_i)^-1`，无须逐点求逆。

父逆元为普通域 `1/(AB)`，右子乘积为 BR，则 Mont(1/(AB),BR)=1/A；另一子对称。到叶层，用 Mont(X_iR,1/Z_i)=X_i/Z_i，一次模乘直接得到普通域仿射坐标；负值按 N−x 计算，x=0时保持0。CPU设置leading=1，所以与旧 F 树输入逐字一致。

只有可逆根的组执行传播。复合环中，一组乘积可逆当且仅当每个 Z_i 可逆；不能对失败根使用费马逆元或将其视为异常丢弃。坏组在CPU逐点处理：Z=0仍生成x=0并记录cache的gcd=N；非零非单位Z保留普通X为叶值，记录gcd(Z,N)及原因子列表。缓存按原baby索引递增记录，曲线/Q/N/D/B1/B2键及ready时机不变。

## 3. 计算量与依赖

记P=φ(D)/2、c0=P、c_l=ceil(c_(l−1)/2)、G=c8=ceil(P/256)，T=Σ[l=1..8]c_l。每层一次发射，每线程处理一对子节点；上下层仍按依赖顺序执行。

全为可逆组时，模乘数量精确为：

```text
up   = Σ[l=1..8] floor(c_(l−1)/2)
down = 2·Σ[l=2..8] floor(c_(l−1)/2)
leaf = 4·floor(P/2) + (P mod 2)
Mtree = up + down + leaf
```

P为256的倍数时 Mtree=4P−3G。坏组仍计算up，但不执行该组的down/leaf。所有尺寸固定8层up、7层down和1层leaf，共16个归一化kernel；短尾复制避免额外单位模乘。CPU正常组仍G次求逆，与原256点分段相同，另有G次根逆元乘R及一次R求逆。其余原约4P次CPU模乘/模约减与逐点负值转字移到GPU。

ladder原路径额外2P次Mont域转换被省略，因此相对于原路径，GPU净增加约2P−3G次模乘，CPU删除约4P次普通大整数模乘（根种子转换增加G次）。当前SOS/REDC每次Mont的两个主双循环为2W²个依赖MAC，另有W次低字乘法、carry传播和规范化；这里只计源码计算量，不声称GPU周期数。D138/P126720/W70/G495：归一化505395次Mont，主循环4952871000次MAC；扣除省略的ladder域转换，净增加251955次Mont。

## 4. 显存、内存与传输

显式设备payload为 `8[(3P+5)W+P+TW]+G` B：X/Z/负值三个P×W数组、五个W字常量、P个索引、T个W字内部树节点和G字节组mask。root上传复用内部树根位置，无额外根设备数组。`NTT_BABY_DEVICE_MAX_MB`默认512，另检查实时free留64MiB；预算不足或实际cudaMalloc OOM时，在生成叶值/改变因子状态前返回旧路径。所有本路径buffer在F树构建前释放，不与后续大NTT workspace长期并存。

P126720/W70时 T126225、G495，显式payload284592655 B /271.409MiB。这不包括编译器local backing、CUDA context/driver，也不代表整个进程的VRAM峰。逐字检查会额外执行旧ladder并建立其静态buffer，不能把CHECK路径的峰当作生产峰。

正常归一化D2H为 `8W(P+G)` B，即P个普通负仿射常数加G个根；旧X/Z为16PW。D138例分别71240400与141926400 B，减少70686000 B（49.80%）。种子/mask H2D为8WG+G，另有同样所需的索引与5个常量上传。坏组额外回读16W·Bbad，其中Bbad为落在坏组中的点数。CHECK路径另有整批X/Z参考读回，单列诊断成本。

主机侧新负值数组8PW、根/种子8WG、mask G，加坏组当前临时X/Z；leaf容器及后续F树仍是原布局。原两个整批X/Z主机数组被省略；新负值物化完成后释放。未测新版本进程private峰、NVML总峰或全曲线PCIe总字节，不将此局部公式外推为整曲线账本。

## 5. 开关、模型和验收状态

实验开关默认0，生产wrapper未增加默认。`NTT_BABY_DEVICE_CHECK=1`独立执行原普通域ladder，逐字比较所有负仿射值并检查非单位数及cache GCD；`NTT_BABY_DEVICE_TEST=1`使用实际host调用fixture，覆盖零Z及复合N的非单位Z；`NTT_BABY_DEVICE_TEST_BAD=1`故意损坏首输出，应被检查拒绝。`NTT_BABY_DEVICE_ALLOC_FAIL=1`注入分配前回退。

请求新路径时旧经验D模型回legacy，避免用CPU baby成本为新准备路径选择D。本轮性能比较使用固定D、同N/Q/B1/B2、同binary与检查覆盖；若提升默认，需要重新验证D模型与CPU/VRAM预算。

已完成独立kernel GMP门禁：42组、379552字，覆盖W=1/2/3/4/8/16/32/64/70/83/128、单点/奇尾/跨256边界、多组、单位/零/非单位混合、M4423；两个顶层门禁（正常和故意损坏）通过。probe使用从实际stage2_tree_gpu.cu逐字提取的Montgomery算术和实际设备header，构建前后哈希核对；该独立门禁不冒充实际host因子/cache或完整Stage2验收。

证据：[kernel门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/kernel_gate/summary.json)、[probe manifest](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/probe_v2/manifest.json)、[公式样例](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/formula_sample.json)。

## 6. 最终实验验收与同binary性能

完整实验exe SHA256 `a5addcd9229cf61a6c1203142abdf2b25323904968b4ef4644506fed9b30e7f8`，CUDA编译581.4秒、链接3.2秒；[8项构建依赖与原始源快照](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/stage2/manifest.json)在构建和每条A/B前后核对。生产E0139328…BE331保持原发布状态，新功能未提升生产默认；实验算术、模型和已发布产物明确分开。

- 实际host/因子/cache路径、正常与故意损坏、分配和预算回退：**9通过/0失败**，其中独立kernel部分为42组/379552字。
- 独立Python普通域点/F系数、CPU dump、真实Stage1 Q及extra1/12，129/4423/5261位、多段、不同gfinv模式：**12通过/0失败**。
- 实际D选择器确认请求GPU baby时拒绝旧CPU baby经验模型，**1通过/0失败**。它不执行完整曲线，不能替代算术门禁。

八条串行ABBA+BAAB（CPU/GPU/GPU/CPU/GPU/CPU/CPU/GPU），同N=2^4423−1、sigma26、extra12、B1=1000、B2=2011326186870、D1381380/P126720、GPU1及short1/outer2、sample96/check_every8等所有NTT控制。full依次 **58.553092、52.814794、52.528683、55.765622、54.949292、55.503839、59.387116、52.885733秒**。每模式4样本，没有建立置信区间；两组顺序分别快7.85%和6.14%。

完整Stage2均值 **57.302417→53.294626秒（快6.99%）**；init 15.583379→11.317235，main 41.719038→41.977390。baby ladder 7.973750→7.903250秒；affine **3.576000→0.276000秒（快92.28%）**。main均值略高，不能把本轮说成NTT/giant内核提速；init剩余F树/检查/准备也有波动，不将完整4秒差额全部归因于同一个操作。

所有曲线最终叶FNV4244971527793015097、oracle signature c85031f6149bae11、采样数、因子/命名结果相同；GMP bad0/pending0/clean1，small-prime cache matched1。Q与之前独立验证的完整hex逐条匹配，Stage1排除于full；新路径全部bad_groups0，读回计数精确等于§4公式。曲线先固定D以隔离准备路径收益，不能与上一轮D/归约的不同条件百分比相加。

证据：[host9项](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/host_gate/summary.json)、[独立baby/F12项](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/real_baby_gate/summary.json)、[模型scope](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/model_scope.json)、[八条A/B](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/ab/measurements.json)。

## 7. Nsight时间线、传输与分配账本

在上述冻结binary/D/控制上各采一次Nsight Systems 2026.1.3，GPU1所有kernel；这是独立profiling，不把其计时与无观察器A/B合并。采样与CPU context switch关闭，没有CPU栈或有效NCU硬件cycle/occupancy计数。Stage2近似范围为Stage1 chain之后第一个GPU kernel到最后一个本进程GPU事件，不等同于精确stage2_full_wall。

本进程kernel/copy/memset事件并集：span 53.462374→50.709207秒；无事件间隙 **8.674279→6.068857秒**，占比 **16.23%→11.97%**。旧baby Z回读后的2.352594秒间隙缩为新负仿射值回读后的.236172秒；新最大间隙为其他阶段.653794秒（此前16B D2H，随后70963200B H2D），后续还需定位该CPU准备。这里的无本进程事件不代表整卡idle，也不测SM实际利用率。

NTT kernel池17.011000→16.944177秒、原point kernel池18.581757→18.542850保持近似；新增16个baby归一化kernel共.158233秒。全部kernel数102389→102405。完整profile范围的D2H **3279806872→3209120872 B**，减少70686000 B，正好对应公式；H2D7151130460→7151408155 B，增加277695 B/2次，正好是根种子+mask。D2H次数6132不变、D2D40次/841498560 B不变。传输GPU耗时单次有波动，不宣称PCIe带宽提升。

按CUPTI Device-kind分配/释放逐地址核对，无未匹配释放：跟踪payload峰 **4816521168→4673578208 B**，减少 **142942960 B /136.321MiB**。旧路径退出前仍保留8个static ladder buffer，合计同样142942960 B；新路径跟踪余额0。生命周期回收减少后续主阶段常驻量，新的271.409MiB临时payload在F树之前已释放。此账本排除driver/context、静态模块及隐式stack backing；没有采新NVML或host private峰。

实际NW128（W70使用此模板）资源：product/inverse/leaf REG40/47/40，shared0、LOCAL字段0，但**STACK分别4112/5136/6160 B，非零**。CUPTI local=0不能用来宣称没有线程私有存储；资源容量也不是实际occupancy。原始full-exe cuobjdump与trace资源交叉核对。

第一次直接启动目标的采集得到CUDA事件，但应用stdout未进入CLI日志，结果核验因此失败；该记录保留于profiles目录，未计入配对结论。随后用cmd子进程将应用日志独立写文件，在profiles_checked全新目录重采两模式；实际mode/Q/叶值/GMP/clean/signature均确认，SQL只含GPU1。没有将日志失败描述为算法失败，也没有混用两个采集目录。

证据：[配对采集manifest](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/profiles_checked/manifest.json)、[事件/传输/资源分析](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/profile_quantitative.json)、[Device分配账本](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/tracked_memory.json)、[精确binary资源](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_device_20261005/baby_resources.json)。

## 8. 下一优先级与发布条件

1. 对GPU baby重新标定D并用独立B2/D验证排名，然后通过实际save/ini/worktodo入口验收后提升生产默认。现有resident_short系数测量的是CPU baby，不能直接延用。
2. 定位剩余.654秒准备及多个中等间隙；考虑负仿射常数的设备F树frontend，继续保留单首项及非单位语义。当前只回读常数，未实现零leaf传输。
3. point约18.5秒、NTT约17秒现成为主要GPU池；继续减少依赖MAC及无须全128位归约的NTT规范化，收益须各自完整A/B。Tensor真实NTT tile此前为负结果，仍不提升默认。
4. 多曲线必须共享workspace lease并建立逐生命周期RAM/VRAM预算。两个独立进程仅已跟踪Device payload峰就约9.35GB，已超过8GiB设备，不能直接复制两个当前实例。共享workspace后的可行性尚未验证，不把独立池峰相减或相加当作正式多曲线预算。
5. 新Prime95对照仍要求相同N/Q/曲线族/B1/B2/覆盖、线程及检查口径，本轮仅证明GPU自身改进，尚未证明长期CPU对照目标完成。

