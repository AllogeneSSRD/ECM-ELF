# ECM CUDA Stage2：Mersenne 点模乘的折叠与旋转

日期：2026-10-05；起点 `9c4024f`。目标是在保持原 Montgomery 表示和精确坐标的前提下，删除点模乘中的二次归约 MAC。GPU1 RTX4060 Laptop/sm89/CUDA13.3，GPU0外部生产未调度。前一阶段生产为DCF，见[固定PTX D标定](D:/code/MPA-OpenCl/docs/STAGE2_FIXED_PTX_D_CALIBRATION.md)。

## 1. 当前时间线决定优先级

以DCF、真实M4423 Stage1存档、sigma26/B1=1000、D1381380、B2=2011326186870、默认检查和fixed3采一次Nsight Systems2026.1.3。Stage1跳过；canonical Q SHA `33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`，leaf `4244971527793015097`、oracle `c85031f6149bae11`，GMP bad0/pending0/clean1。source18已冻结。

- 自有GPU事件窗口48.154451s，kernel/copy/memset并集42.887115s，无自有事件间隙5.267335s（10.9384%）。这包含启动检查和尾部，非精确stage2_full_wall或整卡idle。
- ladder22次合计14.958864s；chain6次3.000892s。它们合计17.959757s，是本轮优先处理的主要GPU池。
- tile7.987254s，cooperative outer两组2.860311/1.505467s；S4 reducer1.750898s。不能用kernel分类之和当作wall或occupancy。
- H2D7151405143B/4870次/.545955s，D2H3209119752B/6130次/.252715s，D2D841498560B/40次/.007413s。Runtime等待不等于实际传输时长；无事件间隙也不能全归因于PCIe。

第一次采集Q输出开关拼错，缺少Q证据而拒绝。新目录重采后，分析器的leaf标签拼错；保留trace和日志，修复后只重新分析，未重复运行曲线。恢复分析核验记录command wrapper字节、Q/save/exe/source/叶/GMP以及GPU1事件。没有CPU栈或有效硬件cycle/occupancy计数。

证据：[原始trace](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/profile_checked/trace.nsys-rep)、[manifest](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/profile_checked/manifest.json)、[事件与API间隙](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/profile_checked/summary.json)。采集/分析入口：[profile_stage2_points.py](D:/code/MPA-OpenCl/tools/bench/profile_stage2_points.py:1)。

## 2. 数学与计算量

约定 `N=2^s−1`，`W=ceil(s/64)`，`R=2^(64W)`，canonical `0≤a,b<N`，`d=(64W−s) mod s`。原接口返回 `abR⁻¹ mod N`。小于64位时也采用d模s，不假定N的最低字总是全一。

1. 计算完整schoolbook积u=ab，约W²个64bit MAC。
2. 分解u=l+2^s·h。由于u<N²，有l+h<2N。计算v=l+h，一次条件减N即可获得canonical v；s为64倍数时保留末尾进位，部分顶字时保留越过s的位。
3. `2^s≡1 mod N`，所以R⁻¹等价于s位右旋d。输出 `(v>>d) | ((v mod 2^d)<<(s−d))`。两部分位域不重叠；全一表示已在第2步被规范化，右旋继续canonical。d=0直接复制，避免64位无效移位。

这保持了Montgomery表示，不需要重新生成R、a24、Q、Γ、inverse或leaf。普通/Montgomery混合调用（例如Mont(xR,1/z)=x/z）也保持原语义。最终写r延后至a/b已完全读入局部积与规范化out，因此允许r=a或r=b。

通用SOS+REDC约2W² MAC/模乘，新方法W²+O(W)，不是GPU周期公式。M4423的W70/d57，从约9800降至4900 MAC。六模乘xADD仍为6次调用：主导项12W²→6W²；xdbl5次10W²→5W²；一个普通ladder bit的xADD+xdbl为22W²→11W²；chain续点12W²→6W²。这些不包括线性加减/halfmod、初始化与segment修正。

NTT形状、poly mul数、S4 Kronecker算法不因该原语改变。不能将本轮与之前8→6次xADD收益重复累计。

实现：[折叠/旋转原语](D:/code/MPA-OpenCl/tools/bench/stage2_point_mersenne.cuh:6)、[当前分派](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:704)、[host精确模数证明与状态重设](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4137)、[D模型保护](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10803)。

这里的N是ECM大模数；NTT的Goldilocks素数q及其PTX归约保持。NTT_S4_MERSENNE是此前对Kronecker宽系数做普通域归约的另一条路径，不能代替这里Montgomery结果所需的旋转。

## 3. 原语验证与局部速度

独立probe从真实stage2_tree_gpu源码提取通用SOS/REDC/helpers作为对照；不会维护另一手抄旧算法。最终通用reference提取时移除新分派，以单独测原语。所有源文件及实际提取字节有SHA；旧版本测量闭包在probe/sources保留，不把之后工具修正当作当时编译输入。

初次probe SHA `dc7fe3e1a1f5c74d572b8bf9fea4816c84d11ffafec7e68763719fb538fb8889`，build约7.0s。18个bit宽2/3/31/32/63/64/65/127/128/129/257/513/1025/2049/4097/4423/5261/8192 × 两mode × 三alias，每项128输入、3次Montgomery递推，GMP独立计算canonical值。0/1/N−1/N−2交叉与顶位、确定性随机包含在输入内。另故意损坏首字，必须bad>0/exit1：**109进程检查通过/0失败**。Mersenne包含合数，不依赖N素性。

局部性能8进程ABBA+BAAB：M4423/W70，8192输入，每线程16次依赖模乘、128 threads，alias=a；每进程3次warm与3次event均值，所有输入在计时外GMP检查。原语均值22.873928ms→12.270750ms，减少**46.3549%**。它不是完整Stage2收益；每mode4样本，无CI。

probe NW128旧/新寄存器40/42，stack均6160B、spill store/load均0。两者线程局部源码数组容量相同，零spill不等于零线程私有存储，也不证明实际occupancy相同。此资源数据来自probe，不能代替真实ladder内核资源。

证据：[原语109](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/primitive_gate/summary.json)、[局部8条](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/primitive_ab/summary.json)、[probe编译资源](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/probe/build.log)。[probe源码](D:/code/MPA-OpenCl/tools/test/stage2_point_mersenne_probe.cu:1)、[编译与提取](D:/code/MPA-OpenCl/tools/build/test/build_stage2_point_mersenne_probe.ps1:1)、[门禁/计时](D:/code/MPA-OpenCl/tools/test/test_stage2_point_mersenne.py:1)。

## 4. 运行约束与容量

新增NTT_POINT_MERSENNE，缺省0。host以popcount等于bit长度、W匹配证明精确N=2^s−1；不匹配时enabled0回通用算法。每次native曲线初始化都写device symbol（包含关闭时写0），启动2048个Mont GMP检查使用实际后端。请求新路径时拒绝旧D经验速率，固定D比较；后续需新D拟合与留出，不能直接延用profile5。

新增device constant int为4B，新增持久点/NTT数组0B。临时原语源码t/out均为`(3NW+2)*8`B，NW为模板容量，W为实际字数；与旧原语相同，但双路径可改变实际kernel stack/寄存器，需真实资源核对。device scalar初始化每曲线4B H2D；原点批、F/Γ/NTT、oracle数据payload保持。没有删除已有主机数据生成和传输。

device模数状态是每设备共享，当前native执行串行。多曲线并发必须在workspace/模数lease内保障状态不被另一曲线改写，或迁移为显式curve/context参数；本轮没有实现并发安全接口。RAM/VRAM容量必须按生命周期和实际allocator账本评估，不将逻辑arena或零spill当作总峰。


## 5. 函数结构试验、完整程序与门禁

原强制内联完整Mersenne乘法适配器局部快46.35%，但native编译前端运行757.0s仍未结束，主动中止所属cicc（CPU733.88s），exit255由终止引起，未产生可用于性能结论的native。随后共享整个模乘函数的版本又通过109项；M4423/count128/repeats3的mode1约2.73ms、mode0约2.57ms，单次小批次有回退，不推断完整生产吞吐。

最终生产实现**共用原SOS乘积，仅将O(W)折叠/旋转放在noinline设备函数**。最终probe `c190657dbaf5ff6fdc2a6153bdcaf3674c8072af084a36a3fda1374d5ab101c3`，再次109/0；8进程ABBA+BAAB为22.872992→12.334422ms，快46.0743%。probe NW128 REG40→44，累计stack均6160B，无spill。这组取代前述两种结构作为当前实现证据。

native SHA `7cbc7f1cabf2596b1a7f7eebcf3e0f3e5efde97a59e7c6446d9482235a242bca`，4113408B；CUDA编译441.8s，另C++约4.0/2.5/3.2/3.0s，19份raw编译依赖冻结。原始源码容量相同不保证ABI/调用栈相同；真实exe STACK/REG/LOCAL及函数列表在[资源对照](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/point_resources.json)，不将CUPTI local0当作无局部存储。NW128实际选中的xADD6 ladder REG43→58、STACK16400→17424B；chain REG42→64、STACK13328→14352B；baby product/inverse/leaf的STACK4112/5136/6160→5136/6160/7184B。各增加1024B/线程，寄存器也有增长；额外implicit backing尚无总峰测量。这是实现代价，不能宣传为临时显存减少或occupancy提升。

native路径门禁18/0：七个Mersenne位宽61/127/521/607/1279/2203/4423 × 开关0/1，实际2048 Mont/GMP和1280 xADD/普通与Montgomery/五alias/halfmod检查；另两个非Mersenne2^128+1、2^130+1准确回退，最后真实Stage1存档的B2=1e11/D390390两曲线证明mode0启用profile5、mode1拒绝旧fit，并保持leaf1689529688547722991/GMP/因子一致。前16项使用synthetic saved X=2，仅用于算术/分派，**未宣称真实生产Stage1曲线**；启用xADD诊断所以clean0。首次harness误要求clean1而拒绝，修正后用native_gate_v2新目录重跑，未改CU。

原生save/ini/worktodo/CRC/因子/队列等入口另21/0，默认开关0；它是兼容性验收，与18项新路径检查分开。[native18](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/native_gate_v2/summary.json)、[入口21](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/native_accept/summary.json)、[最终原语109](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/fold_gate/summary.json)、[最终局部8](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/fold_ab/summary.json)。[实际native门禁源码](D:/code/MPA-OpenCl/tools/test/test_stage2_point_mersenne_native.py:1)。

## 6. 完整曲线：内部开关与旧生产两组分别报告

实际存档Q/σ/B1/B2/D/GPU1及默认检查保持；Stage1跳过，D1381380、B2=2011326186870，模型关闭；inv shift0/fixedPTX3/GPU baby1/xADD6。没有trace采集器与额外诊断。full包含init/main，不包含约.11s D扫描及shape。

内部开关8条顺序0/1/1/0/1/0/0/1：

- mode0：47.267354 / 46.948500 / 46.898147 / 46.875519s，均值46.997380s。
- mode1：39.779114 / 39.871482 / 39.738362 / 39.813919s，均值39.800719s，减少**15.3129%**。
- init均值9.890883→6.718526s；main 37.106497→33.082193s。新init/main各占新full 16.88%/83.12%。4样本/模式，无CI。

另旧生产DCF/候选mode1/候选mode1/DCF，四条完整交叉：49.393826 / 39.800018 / 39.827902 / 49.512051s；均值49.452939→39.813960s，减少**19.4912%**。2样本/版本，无CI；与内部开关及上一阶段3.13%分别报告，不相加。旧生产源码按其18份immutable snapshots核验，候选按当前19份，不拿旧manifest比改后的working tree。

所有12条均exact Q、leaf4244971527793015097、oracle c85031f6149bae11、factor空/bad0/pending0/clean1。S4覆盖均397 launches、1836241 poly mul、36615543 reduced coeff、2400 primitive cases、60474 GMP samples、3次full checks；没有降低检查数或改变poly/coeff工作量。[内部8](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/internal_ab/measurements.json)、[生产对照4](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/production_ab/measurements.json)。[完整计时工具](D:/code/MPA-OpenCl/tools/bench/bench_stage2_point_mersenne.py:1)。

## 7. 配对profile、传输与后续

各版本独立采一条Nsight，非无观察器A/B：ladder+chain池17.959757→8.574207s；NTT命名池15.206951→15.273277s。各自GPU窗口48.154451→38.748970s，无自有GPU事件5.267335→5.393368s，占10.938%→13.919%。点池缩短后的间隙占比可能升高，不等于CPU工作增加或整卡idle。每条均核验Q/叶/oracle/GMP/source/device。[配对汇总](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/paired_profile.json)、[新trace](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/candidate_profile/trace.nsys-rep)、[新API间隙](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/candidate_profile/summary.json)。

复制payload逐项核验：H2D7151405143→7151405147B、4870→4871次，恰好新增4B设备状态初始化；D2H3209119752B/6130次和D2D841498560B/40次保持。NTT/点/leaf数组几何不变。workspace/arena的逻辑容量与allocator、隐式stack backing不可混为总VRAM；本轮未重采NVML/host-private峰。减少的是MAC工作量。CPU准备/检查以及传输等待仍需要单独定位。

测量时Python驱动字节另冻结在measured_drivers；收尾仅将文件读取显式设为UTF-8，以避免Windows默认GBK误读，未改计时或校验逻辑。最终工具再次109/0，native19原始源码及资源14项、拷贝差量、所有28个报告链接/line核验见[最终审计](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/final_audit.json)。

当前生产仍DCF，实验默认关闭；新路径未拟合D，不覆盖旧生产文件。重建与固定D实验：

```powershell
tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/point_fold_native -Arch sm_89 -GlBackend ptx -Rebuild
$env:NTT_POINT_MERSENNE='1'
build_cuda_cmake/point_fold_native/ecm_cuda_stage2.exe --save YOUR_STAGE1.save --b2 2011326186870 --d 1381380 --device 1 --results results.jsonl
```

回退设NTT_POINT_MERSENNE=0；通用奇数N会自动使用旧REDC。下一步先评估将callee规范化out复用caller scratch以消除新增1024B栈，以及W70采用更贴近实际的模板容量；随后重拟新baby/affine/giant速率与D、独立B2/D留出及save入口，再评估生产默认。原NTT四个共同依赖与尺寸策略未变，权重可在严格来源核验后作为冻结输入，而非直接沿用旧全曲线rates。之后继续准备空隙、共享workspace/curve私有所有权与预算；Tensor旧负结果保留，公平Prime95同Q/B2/线程完整对照仍待完成。
