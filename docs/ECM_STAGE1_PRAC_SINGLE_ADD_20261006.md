# Stage1 PRAC：统一内联 xADD 调用点

日期：2026-10-06。基线a11f895，接续[共享xADD调用成本实验](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_OUTLINED_ADD_20261006.md:1)。上一轮outline-add静态text小约15%，却产生约109GiB/kernel的累计local sector字节代理，生产前缀投影慢约55.7%。本轮保留内联，以固定点角色归一三处xADD调用，不使用通用引用参数的device-call。

## 1. 假设与范围

三个假设：单一内联位置缩小指令展开并改善长片；新增输出点与点搬移增加寄存器/指令成本、抵消收益；编译器重新复制公共体，实际text未缩小。先检查资源，再测同子乘积与普通生产前缀，不以静态体积直接推导速度。

`ECM_PRAC_VARIANT=single-add`，仅选定4608/TPI16容器，要求PRAC及register policy255/168。MODE10为自然策略、MODE11为cap168。工具首轮N4423；默认算法/TPI/寄存器/100ms目标不变。独立CUDA TU实例化两个候选，避免继续增加原TPI16 TU的关键路径。

## 2. 点角色与控制流

每个奇素数的初始状态仍为A=P、B=2P、C=P，`e=p-r`、`d=r-e`。每次非终结迭代先按原逻辑保证d≥e。四规则及终结都归一为唯一内联调用 `T=ADD(A,B,C)`；T使用两个新的bn变量tx/tz。以下A₀/B₀/C₀指迭代开始并完成d/e交换后的逻辑点：

| 规则 | ADD前物理A/B/C | ADD结果T | 恢复后的逻辑A/B/C | d/e更新 |
| --- | --- | --- | --- | --- |
| d≤2.96e | A₀/B₀/C₀ | ADD(A₀,B₀,C₀) | A₀/T/B₀ | d−e,e |
| 同奇偶 | A₀/B₀/C₀ | ADD(A₀,B₀,C₀) | 2A₀/T/C₀ | (d−e)/2,e |
| d为偶数 | A₀/C₀/B₀ | ADD(A₀,C₀,B₀) | 2A₀/B₀/T | d/2,e |
| e为偶数 | B₀/C₀/A₀ | ADD(B₀,C₀,A₀) | A₀/2B₀/T | d,e/2 |
| d=e=1 | A₀/B₀/C₀ | ADD(A₀,B₀,C₀) | A=T，终结 | 不变 |

非减法规则只执行一次DBL，作用于经过准备交换的物理A。消费完输入后使用固定变量set恢复角色，避免动态bn指针或数组索引。终结ADD交换了原实现的两个加数；规范化整数域下sum平方不变、difference符号平方消失，因此应保持完整X/Z字节一致，此性质需GPU逐字节门禁确认。p=2仍直接DBL；prime幂重复、choose12扭率、规范化4M+2S/3M+2S及Montgomery域不变。

源文件：[唯一内联ADD与角色恢复](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cuh:8)、[独立实例化](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cu:8)、[显式分派](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernels.cu:12)、[主机策略](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:60)。

## 3. 计算量与容量

同计划记录的ADD/DBL次数不变。`W=6*ADD+5*DBL`（CGBN当前S=M）；同32-record生产tail为1193ADD、179DBL、W8053，完整260m计划F3369476895。数学等价模乘工作量没有减少；预期收益来自代码与指令调度。

显式设备曲线数组仍`7*C*B/8`bytes，PRAC计划`16*P`bytes；窗口seed额外`7*C*B/8`、每轮恢复一次，片边界逻辑量`6*C*B/8*launches`。没有新增持久GPU数组或PCIe上传。T的两个域值潜在每线程`2*B/(32*TPI)`个32-bit limb；B4608/TPI16为18个，但实际存活期与寄存器复用由编译器决定，不可直接把18加到资源计数。若物化为local，流量需运行计数验证。

每线程stack L乘`C*TPI`只作逻辑栈容量代理，不是实际显存承诺。GPU1/C1536的48/80/96bytes分别对应1.125/1.875/2.25MiB，不能与模块容量或整卡memory.used相加作为进程峰值。

## 4. CPU角色模型

独立模型覆盖≤10000素数的前7个Prime95候选比例，以及缓存B1=10m/260m的前/中/后窗口；8533组prime/d、sigma26及64-bit sigma，共17066条链、264538次点角色步骤。使用M127的规范化整数点运算，逐步d/e与完整A/B/C X/Z、终结X/Z和ADD/DBL次数均与baseline模型完全相同，并与独立ladder交叉相等。覆盖减法190626次、同奇偶29206、d偶17918、e偶9722、终结17066。

这是CPU控制/角色模型，不是GPU正确性证明。证据本地 `stage1_single_add_roles_20261006.json`；[逐步点模型](D:/code/MPA-OpenCl/tools/test/test_prac_single_add.py:19)。窗口ladder oracle增加maxsize1024的有界缓存，键为完整(sigma,scalar,N)，重复输入复用独立结果，但每份GPU输出仍逐点检查规范性及cross-product；数学oracle未换成GPU结果。

## 5. 构建及静态资源

七个TU以6并发完成，critical path仍为普通TPI16，720.7s；serial sum1302.4s。候选独立TU110.3s，主机15.5s；最终只链接，无其他算术TU重编。新exe SHA256：`8aeb7854bf56712ef8f4a2b18a8e224397c2131e12a412a699cc1de1a38652ef`。上一版exe/DLL/配置与TPI16对象保存在本地 `build_cuda_cmake/prac/before_single_add_20261006/`。

候选对象SHA及全部text导出在本地 `stage1_single_add_sass_20261006/`；导出前后对象SHA相等。新binary的baseline MODE4/5的完整56352/55928条SASS指令行（含编码与寄存器操作数）分别与a11f895实验对象逐行相同，资源及text也相同，证据为 `stage1_single_add_baseline_sass_20261006/baseline_identity.json`。候选实例仅在新TU中，不混入普通TPI16分派。

| 策略 | register/thread | stack bytes/thread | spill store/load bytes | 静态指令条数 | text bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline natural MODE4（基线） | 172 | 48 | 0/0 | 56352 | 901632 |
| baseline cap168 MODE5（基线） | 168 | 80 | 36/28 | 55928 | 894848 |
| single-add natural MODE10 | 170 | 48 | 0/0 | 36152 | 578432 |
| single-add cap168 MODE11 | 168 | 96 | 56/44 | 35960 | 575360 |

候选没有独立/内嵌prac_add_outlined函数符号；text分别小约35.85%/35.70%。新增T并未让自然策略寄存器/stack增加，cap168仍有额外spill。静态SASS不是退休指令数或指令缓存工作集；实际寄存器分配与blocks/SM要由CUDA/NCU确认。自然策略170不应直接写成硬件分配170。

## 6. GPU正确性与计时

同SHA8aeb二进制的完整Q/save门禁已完成：自然策略184条、cap168/target50/C384为3568条比较均通过。包括CPU/GMP、lcm/choose12、64-bit sigma、正常checkpoint恢复与损坏重建、退化结果及目标边界。这些计数包含ladder/resident及baseline边界对照，不全部是候选运行。

证据：本地 `stage1_single_add_natural_q_gate_20261006/summary.json` 与 `stage1_single_add_cap168_q_gate_20261006/summary.json`。

独立整数ladder窗口门禁exit0：1064条窗口Q、8条候选checkpoint由baseline恢复Q、15拒绝用例（含不支持的2203/8191容器），前/中/后32-record生产子乘积及chunk7/32均通过。36跨切片逐字节检查通过；额外按同输入/容器/TPI分组，共92次跨variant/register/chunk的完整X/Z CSV逐字节相等，包含前述36次，不重复累计。候选窗口不改变已有生产checkpoint且不发布最终save。

oracle缓存统计为184misses/880hits/current184/max1024，共1064逐点验证。独立CPU参考不是来自GPU；缓存只复用完全相同数学输入。证据：本地 `stage1_single_add_windows_q_gate_20261006/summary.json`、`bitwise_audit.json`。[窗口范围及候选恢复](D:/code/MPA-OpenCl/tools/test/test_cuda_prac_windows.py:170)

### 6.1 固定tail32：48样本

GPU1/N4423/4608/TPI16/TPB128/sigma26/lcm/B1=260m，固定相同seed及prime259999307..259999991（1193ADD+179DBL/W8053）。C384/768/1536 × baseline/single-add × natural/cap168 × chunk4/32 ×两次反序，48个样本全部命中指数/PRAC缓存。每样本至少6s、排除前两恢复轮，关闭坐标导出。每格为两次中位数，CUDA event / 有效轮墙钟投影，单位s/curve：

| C / register | baseline chunk4 | baseline chunk32 | single-add chunk4 | single-add chunk32 |
| --- | ---: | ---: | ---: | ---: |
| 384 / natural | 147.848241 / 148.868252 | 147.284733 / 147.497863 | 147.959067 / 148.994453 | 147.426112 / 147.664758 |
| 384 / cap168 | 149.937898 / 151.071462 | 149.318684 / 149.520735 | 150.975581 / 151.883349 | 150.477978 / 150.694962 |
| 768 / natural | 147.492227 / 148.094051 | 147.127593 / 147.241835 | 147.600464 / 148.144654 | 147.289684 / 147.399872 |
| 768 / cap168 | 147.832091 / 148.365461 | 182.495068 / 182.613424 | 149.468212 / 150.092485 | 153.508064 / 153.622824 |
| 1536 / natural | 147.119315 / 147.472051 | 146.908545 / 146.984981 | 147.279637 / 147.588051 | 147.099695 / 147.159468 |
| 1536 / cap168 | 132.697989 / 133.050499 | 173.835785 / 173.898474 | 133.511511 / 133.811608 | 139.858938 / 139.917894 |

自然策略几乎无收益，C384 cap168也变慢。C768/1536的cap168长片墙钟投影分别减少15.875%/19.540%；但C1536的single-add短片比既有baseline短片增加0.572%，不能将长片改善说成整体最佳吞吐已改善。计时期间没有编译或NCU重放。证据：本地 `stage1_single_add_slicing_matrix_20261006/summary.json`、`gpu.csv`。

显式数组和片边界量与baseline相同：C1536的数据/seed各6193152bytes；chunk4八次launch的边界逻辑量42467328bytes/round，chunk32一次为5308416bytes。它们不是PCIe传输量；候选保持这些量，用动态local计数核验是否避免了outline-add的ABI成本。

### 6.2 普通生产前缀

同SHA8aeb/C1536/grid192，B1=10m/260m，resident及两register策略的baseline/single-add，PRAC target50/100ms，两次反序，共36样本。15s采样、排除前5s、取末5s报告中位数。36/36指数缓存命中，全部PRAC样本计划缓存命中；均由native sample limit正常退出，没有工具强制终止或最终save。以下为两次投影中位数，单位s/curve：

| 策略 | B1=10m / 50ms | B1=10m / 100ms | B1=260m / 50ms | B1=260m / 100ms |
| --- | ---: | ---: | ---: | ---: |
| resident ladder | — | 5.364959 | — | 139.474377 |
| baseline natural | 5.617430 | 5.612368 | 146.934346 | 146.817114 |
| baseline cap168 | **5.061943** | 5.252344 | **132.441969** | 138.923316 |
| single-add natural | 5.627939 | 5.621000 | 147.234199 | 147.037615 |
| single-add cap168 | 5.100183 | 5.118210 | 133.432336 | 135.036177 |

在相同100ms目标下，cap168候选减少2.554%/2.798%；在相同50ms下却增加0.755%/0.748%。本批最快仍为baseline cap168/50ms。单点ADD改善了长片的退化程度，但尚未提高已知最优配置的吞吐量。resident不使用PRAC target，表内放在100ms列仅为对照，不表示它有相同控制器。

证据：本地 `stage1_single_add_prefix_matrix_20261006/summary.json`。这些投影不是整条生产曲线墙钟，也不直接发布为Auto B2的T1；本批生产窗口从新初始点开始，未覆盖成熟生产曲线的中/后段点分布。

### 6.3 管理员NCU：相同tail32动态计数

四份报告均成功完成19passes及CSV导出。GPU1、grid192、TPB128、MODE4/5/10/11；prime259999307..259999991、W8053、warmup2、chunk32，跳过初始化kernel后捕获同一恢复窗口。app.log、command.json及NCU kernel签名逐项核对；全部同SHA8aeb，指数/计划缓存命中。clock-control/cache-control均none，采集与吞吐计时分开。

| 计数 | baseline natural | baseline cap168 | single-add natural | single-add cap168 |
| --- | ---: | ---: | ---: | ---: |
| 硬件分配register/thread | 176 | 168 | 176 | 168 |
| 寄存器限制blocks/SM | 2 | 3 | 2 | 3 |
| eligible warps/scheduler/cycle | 0.357316 | 0.349032 | 0.358152 | 0.471857 |
| issue active % | 30.660797 | 26.689415 | 30.774866 | 33.526672 |
| wait / issue-active ratio | 3.889513 | 3.988495 | 3.957127 | 4.379531 |
| no_instruction / issue-active ratio | 0.058939 | 2.480815 | 0.049389 | 0.622256 |
| short_scoreboard / issue-active ratio | 0.274091 | 0.365523 | 0.192730 | 0.302681 |
| long_scoreboard / issue-active ratio | 0.000063 | 0.000115 | 0.000055 | 0.000163 |
| local load sectors | 0 | 688128 | 0 | 1081344 |
| local store sectors | 21456 | 525220 | 21684 | 1017200 |
| local load L1命中率 % | — | 99.913969 | — | 98.963142 |
| local store L1命中率 % | 68.251305 | 95.444195 | 68.123962 | 94.989383 |
| 32×(load+store sectors)，MiB | 0.654785 | 37.028442 | 0.661743 | 64.042480 |
| DRAM throughput，% of peak | 0.000047 | 0.000084 | 0.000074 | 0.000053 |

自然策略没有local load，local store量与baseline相近，避免了上一轮outline-add的巨大ABI成本。cap168累计local字节代理37.03→64.04MiB，说明额外spill确实有代价；仍以L1命中为主。它是32B/sector换算的累计local请求，不是唯一数据、显存容量、DRAM或PCIe字节，也不乘19次重放。

cap168的no_instruction比值2.480815→0.622256，同时eligible及issue active提高，支持长片指令供给/调度成本减少的判断；此计数不是直接指令缓存miss数。wait比值反而增加，不能把各stall比值相加或转换成墙钟占比。驻留数完全相同，改善不能归因于更多blocks/SM。重放耗时不是吞吐基准。

证据：本地 `stage1_single_add_ncu_{baseline_255,baseline_168,single-add_255,single-add_168}_20261006/` 中的 `trace.ncu-rep`、`metrics.csv`、`quantitative.json`、`app.log`、`exit.txt`；[计数摘要工具](D:/code/MPA-OpenCl/tools/bench/summarize_stage1_profile.py:10)。

### 6.4 设备状态及TPB/TPI解释

固定矩阵755行、忙采样567行；生产前缀1161行、忙采样1059行。util≥90%的有效SM时钟全部1800MHz，温度分别56～63°C和55～66°C，没有观察到忙时降频。缺失值分别6/2个，按字段排除而非置零。整卡memory.used忙采样最大323/317MiB；它不是本进程分配峰值，轮询边界偶见0也不能证明零显存。

当前默认TPB为128，历史256已于2026-09-25调整，见[CMake配置](D:/code/MPA-OpenCl/CMakeLists.txt:434)及[变更记录](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:104)。寄存器cap不改变TPB。GPU1上本轮自然策略分配176、cap168分配168，使TPB128下驻留2→3；若TPB256则这两个寄存器档均只能驻留1block/SM（仅资源容量推算，未在本轮重新计时）。

N4423默认4608/TPI16。4423/TPI32是[显式替代实例](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_alternate32.cu:15)，不是默认分档变化。本轮single-add仅TPI16，不包含TPI32候选；先前TPI32曲线减半用于匹配提交grid，不保证实际驻留数相等。证据：本地 `stage1_single_add_telemetry_20261006.json`、两份 `gpu.csv`。

## 7. 复现

```powershell
python tools/test/test_prac_single_add.py --output docs/data/my_single_roles.json
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/internal/parallel_nvcc.ps1 `
  -BuildDir build_cuda_cmake/prac -Only 'cgbn_stage1_prac.*\.cu$|cgbn_stage1\.cu$' `
  -Jobs 6 -Reconfigure
python tools/test/test_cuda_prac.py --bits 4423 --tpi 16 --registers 255 `
  --variant single-add --device 1 --output docs/data/my_single_q
python tools/test/test_cuda_prac_windows.py --single-add --production `
  --production-counts 32 --production-chunks 7 32 --device 1 `
  --output docs/data/my_single_windows
python tools/bench/bench_stage1_prac_slicing.py --curves 384 768 1536 `
  --variants baseline single-add --registers 255 168 --chunks 4 32 `
  --count 32 --b1 260000000 --seconds 6 --warmup 2 --repeats 2 --device 1 `
  --exp-cache build_cuda_cmake/prac --output docs/data/my_single_fixed
python tools/bench/bench_stage1_prac_variants.py --help
python tools/bench/profile_cuda_stage1.py --help
```

前缀矩阵使用variants工具的resident/natural/cap168/single/single168配置，分别target50/100、B1=10m/260m、C1536、15s、warmup5、repeats2。NCU通过profile工具的`--memory`生成管理员采集脚本；必须提升权限执行，四策略串行采集，与计时分开。具体参数及管理员命令保存在各证据目录的command.json与admin.ps1。

## 8. 采用决定

GPU门禁、静态资源、48固定窗口样本、36生产前缀样本及4份NCU均已完成。最终只读证据审计exit0，重新验证当前exe/对象/角色源码SHA、所有门禁、92份逐字节对照、每组两重复及全部NCU输入；本地 `stage1_single_add_final_audit_20261006.json`。保留single-add为显式长片候选；不提升为默认或当前最佳吞吐配置。当前生产优化对照仍采用baseline cap168/50ms，而程序默认算法、TPI、寄存器策略及100ms目标保持原配置。适用范围为4608/TPI16，不能推广其他位宽或硬件。

下一轮优先减少候选点搬移及临时值存活，检验能否把自然策略170寄存器降到硬件168分配门槛，同时避免cap168多出的spill。终结条件可并入rule状态作为一个具体候选，但未测得收益前不能采用。保留normalized Montgomery归约，不用概率上很少出现的进位作为省略归约的依据。

另一个准备成本问题是公共kernel头直接include候选实现：候选body修改仍会让所有7TU重编，关键路径720.7s。可将body限定在候选TU，公共头仅保留声明，先验证非候选SASS完全相同，再改善后续迭代编译成本。这不是Stage1运行吞吐收益，目前尚未实施。
