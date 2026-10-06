# Stage1 PRAC：single xADD 与 compact DBL 组合

日期：2026-10-06；基线49d2dc7，接续[单点inline xADD实验](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_SINGLE_ADD_20261006.md:1)。目标仍为param0 Stage1吞吐优化；本轮先检验临时值存活与寄存器门槛，不以静态资源推导运行收益。

## 1. 假设及候选

1. 两临时量DBL替换原DBL，可能让single-add自然策略从170寄存器降到硬件168分配档，提高TPB128驻留数。
2. 新顺序也可能增加spill或依赖等待；若更多驻留但同子乘积更慢，则不能采用。
3. 组合可能只改善长片；需与现有baseline cap168/50ms的普通生产前缀直接比较。

新增显式`ECM_PRAC_VARIANT=single-compact`，仅4608/TPI16，要求register255/168；MODE12/13。旧baseline/compact/outline-add/single-add保留同二进制对照，默认算法、TPI、TPB128、寄存器策略及100ms目标不变。

## 2. 算法与源码

PRAC计划、d/e规则、素数幂重复、choose12、ADD次数及normalized Montgomery算术保持原有语义。单点ADD的点角色仍与上一轮完全相同，只把seed B=2P及奇素数循环内DBL选为两临时量版本；p=2直接DBL也选相同版本。

compact DBL先以AA/BB计算X，再将AA覆写为E、Z暂存a24*E、BB覆写为BB+a24*E，最后计算Z。输出X/Z可以别名输入X/Z：输入和/差已先读取，输出提交不覆盖后续需要的输入。仍为3M+2S，与ADD的4M+2S组合，CGBN当前S=M。

实现：[compact DBL](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:15)、[单点链选择](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cuh:7)、[专用实例化](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cu:8)、[主机选择](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:60)。

候选body移出公共kernel头，仅保留模板声明，由专用TU在实例化前include定义。普通TU通过if constexpr丢弃候选调用；本轮公共接口修改仍需重编，未来仅改候选body时不应重编其他TU。需要核对baseline及旧single-add完整SASS，不能仅依赖编译成功或对象时间戳声称机器码不变。

## 3. 计算量与容量约定

记B为容器位宽、C为曲线数、P为计划记录数、TPI为每曲线协作线程数，W=6ADD+5DBL。固定260m tail32：1193ADD+179DBL，W8053；完整计划F3369476895。候选不减少数学模乘次数。

设备曲线数组与窗口seed各`7*C*B/8`bytes，计划`16*P`bytes，片边界逻辑量`6*C*B/8*launches`；这些显式数组不因组合改变。C1536/B4608：data/seed各6193152bytes，chunk4八launch边界42467328bytes，chunk32一launch为5308416bytes；不是PCIe字节。

compact DBL源级少一个bn临时值，潜在每线程`B/(32*TPI)`个32-bit limb，本例9个；实际寄存器取决于全链存活和编译器复用，不能直接从170减9。每线程stack乘`C*TPI`仅为逻辑容量代理，不能作为显存分配峰或与整卡memory.used相加。

## 4. 正确性与证据状态

CPU角色模型已完成：8533组prime/d、两sigma，共17066数学输入链，三模型实际51198次链求值；264538角色步骤。原版、single-add、single-compact的每步d/e及完整A/B/C X/Z、终结X/Z、ADD/DBL次数一致，最终与独立整数ladder交叉相等。覆盖四规则与终结。源SHA在本地`stage1_single_compact_roles_20261006.json`，包括helper及DBL公共头；这不是GPU正确性证明。

同新SHA6e07二进制的完整Q/save门禁已完成：single-compact自然策略184条、cap168/50ms/C384为3568条比较，全部通过。覆盖CPU/GMP、lcm/choose12、64-bit sigma、正常checkpoint与损坏重建、退化/因子及目标边界；计数包括ladder/resident/baseline对照，不全部为新候选执行。证据：本地`stage1_single_compact_{natural,cap168}_q_gate_20261006/summary.json`。

同SHA的前/中/后窗口门禁已完成：1352条窗口Q、8条新候选checkpoint由baseline恢复Q、20拒绝用例全部通过，包含single-add与single-compact两策略。48次跨切片逐字节检查通过；只读审计按相同数学输入/容器/TPI分组，128次完整X/Z CSV逐字节相等（包含前述48，不重复累计）。独立CPU oracle缓存184miss/1168hit，每份输出仍逐点核验。证据：本地`stage1_single_compact_windows_q_gate_20261006/{summary.json,bitwise_audit.json}`。

批次末尾第一次只读审计调用漏写`--input`，CLI在读取数据前拒绝；GPU门禁summary已成功写出。修正调用后128比较全部通过，没有重跑或替换门禁样本，也未将该CLI错误当算术失败。

固定子乘积及普通前缀计时、四份管理员NCU均已完成。旧版完整exe/DLL/配置/两个对象已冻结在本地`build_cuda_cmake/prac/before_single_compact_20261006/`，不覆盖上一轮证据。

### 4.1 构建与静态证据

七TU六并发构建及最终链接exit0；critical path为普通TPI16的691.8s，serial sum1370.2s，候选专用TU205.9s。新exe SHA256 `6e07c486405627c6c1d901d6224cc96aa3f4942d895ff789a26746ccd5073d41`。对象导出前后SHA相等。旧single-add MODE10/11完整函数文本（含指令及调度编码行）分别72306/71922行，与49d2dc7对象完全相同；baseline MODE4/5的112706/111858完整函数行也完全相同，证明候选body隔离及模板参数化未改变这四份既有内核。

| 策略 | compiler register/thread | stack bytes/thread | spill store/load bytes | SASS指令条数 | text bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-add natural MODE10 | 170 | 48 | 0/0 | 36152 | 578432 |
| single-add cap168 MODE11 | 168 | 96 | 56/44 | 35960 | 575360 |
| single-compact natural MODE12 | 170 | 48 | 0/0 | 36128 | 578048 |
| single-compact cap168 MODE13 | 168 | 88 | 48/40 | 36000 | 576000 |

第一假设的静态部分未成立：自然策略仍170，不能声称达到168分配门槛。cap168少8bytes stack及8/4bytes静态spill，需动态local计数/吞吐确定意义；text仅微小改变，不重复获得上一轮约36%的代码缩小。C1536/TPI16下88bytes/thread的逻辑stack代理2.0625MiB，原96为2.25MiB，差0.1875MiB；不是运行显存分配差值。

证据：本地`stage1_single_compact_sass_20261006/{summary.json,resources.txt,all.sass,old_single_identity.json}`、`stage1_single_compact_baseline_sass_20261006/{summary.json,baseline_identity.json}`及编译日志。源码/配置8个SHA已冻结于`stage1_single_compact_build_sources_20261006.json`。

## 5. 首轮实验

GPU1/N4423/B4608/TPI16/TPB128/sigma26/lcm，串行GPU任务。新策略完整Q/save及窗口门禁已通过，以同SHA比较baseline、single-add、single-compact，C384/768/1536、natural/cap168、chunk4/32及两次反序，共72样本，复用B1/PRAC缓存。普通前缀覆盖B1=10m/260m、50/100ms，resident及三种PRAC的natural/cap168，共52样本，以15s进度s/curve投影取末5s中位；均不当作完整曲线墙钟或Auto B2 T1。

### 5.1 固定tail32：72样本已完成

同SHA6e07，三种曲线批量×三variant×两register×chunk4/32×两次反序，72/72全部完成，36组均两次重复，全部指数/PRAC缓存命中、无坐标导出或最终save。每次恢复相同初始点，prime259999307..259999991、W8053/F3369476895；每样本至少6s，排除前两恢复轮。以下为有效轮墙钟投影中位，单位s/curve；CUDA event投影及两次原值在原始summary中。

| C / registers | baseline /4 | baseline /32 | single-add /4 | single-add /32 | single-compact /4 | single-compact /32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 384 / 255 | 148.847347 | 147.485098 | 148.986300 | 147.608133 | 149.312182 | 148.060498 |
| 384 / 168 | 150.687950 | 149.533347 | 152.077794 | 150.678239 | 151.037390 | 149.665606 |
| 768 / 255 | 147.989332 | 147.243216 | 148.136238 | 147.390470 | 148.499685 | 147.813319 |
| 768 / 168 | 148.422449 | 184.738241 | 149.965907 | 153.528244 | 149.031550 | 153.408449 |
| 1536 / 255 | 147.400594 | 146.950598 | 147.569818 | 147.162330 | 147.983552 | 147.564567 |
| 1536 / 168 | 132.921008 | 172.657750 | 133.760498 | 139.757697 | 133.230461 | 139.764788 |

相较旧single-add，组合版cap168短片在C384/768/1536减少0.684%/0.623%/0.396%；但相较baseline短片仍增加0.232%/0.410%/0.233%。自然策略未获收益。相较旧single-add的cap168长片，组合版变化为-0.672%/-0.078%/+0.005%；小于1%的差异应结合两次原值及后续前缀，不当作跨硬件稳定结论。

整体仍未超过C1536 baseline cap168短片。没有计时期间编译/NCU重放；窗口从新初始点开始，不能当真实生产tail成熟点分布。证据：本地`stage1_single_compact_slicing_matrix_20261006/{summary.json,gpu.csv}`。

### 5.2 普通生产前缀：52样本已完成

同SHA6e07/C1536/grid192，52/52完成，26组均两次反序重复；52指数缓存命中、48个PRAC样本计划缓存命中。15s采样、排除前5s、取末5s进度报告中位数，再取两次中位。均native sample limit正常退出，无强制终止或最终save。以下单位s/curve：

| 策略 | 10m /50ms | 10m /100ms | 260m /50ms | 260m /100ms |
| --- | ---: | ---: | ---: | ---: |
| resident ladder | — | 5.364405 | — | 139.488556 |
| baseline natural | 5.616064 | 5.611327 | 146.959179 | 146.821943 |
| baseline cap168 | 5.060764 | 5.233893 | 132.429740 | 139.151724 |
| single-add natural | 5.625595 | 5.620437 | 147.168337 | 147.025494 |
| single-add cap168 | 5.098886 | 5.118753 | 133.439305 | 134.689543 |
| single-compact natural | 5.641321 | 5.635943 | 147.588928 | 147.441751 |
| single-compact cap168 | 5.078480 | 5.103231 | 132.901355 | 134.610068 |

组合版cap168/50ms比旧single-add同目标减少0.400%/0.403%，但比baseline cap168/50ms仍增加0.350%/0.356%。在100ms同目标下，相较baseline减少2.496%/3.264%；相较旧single-add的100ms只减少0.303%/0.059%，不能把这组的小差异当稳定提升。自然策略均略慢。

本批最佳仍baseline cap168/50ms。resident没有PRAC target，放在100ms列只是参照。原始两次值/实际片长在本地`stage1_single_compact_prefix_matrix_20261006/summary.json`；短前缀不是完整生产曲线墙钟，也不发布Auto B2 T1。

### 5.3 设备状态

固定矩阵1145行、849忙采样；前缀1679行、1528忙采样。util≥90%有效SM时钟全部1800MHz，温度分别55～62°C及60～66°C，未观察到忙时降频。缺失值11/9个，按字段排除，不置零。整卡memory.used忙采样最大323/317MiB，不是本进程分配峰；轮询边界偶见0不能证明零显存。监控由各驱动正常终止，与benchmark worker被强制终止是不同事件。

证据：本地`stage1_single_compact_telemetry_20261006.json`及两个矩阵的`gpu.csv`。

### 5.4 管理员NCU：相同tail32

四份采集均19passes、管理员exit0、CSV导出exit0。逐项核对同SHA6e07、GPU1、grid192、TPB128、tail first14195828/count32、prime259999307..259999991、W8053/F3369476895、chunk32/warmup2；跳过INIT后捕获第一份恢复窗口。clock-control/cache-control为none，计时和重放串行分开。以下每份捕获一次，不能把重放耗时当吞吐：

| 计数 | baseline cap168 MODE5 | single-add cap168 MODE11 | single-compact natural MODE12 | single-compact cap168 MODE13 |
| --- | ---: | ---: | ---: | ---: |
| 硬件分配register/thread | 168 | 168 | 176 | 168 |
| 寄存器限制blocks/SM | 3 | 3 | 2 | 3 |
| eligible warps/scheduler/cycle | 0.359616 | 0.463139 | 0.356788 | 0.461576 |
| issue active % | 27.169780 | 32.986475 | 30.685667 | 32.767545 |
| wait / issue-active ratio | 4.013637 | 4.353395 | 3.965913 | 4.353699 |
| no_instruction / issue-active ratio | 3.377078 | 0.609239 | 0.053570 | 0.644848 |
| short_scoreboard / issue-active ratio | 0.365736 | 0.302339 | 0.190063 | 0.252765 |
| long_scoreboard / issue-active ratio | 0.000116 | 0.000163 | 0.000053 | 0.000149 |
| local load sectors | 688128 | 1081344 | 0 | 983040 |
| local store sectors | 525164 | 1016436 | 21488 | 819828 |
| local load L1命中率 % | 99.911063 | 98.927261 | — | 99.872640 |
| local store L1命中率 % | 95.443709 | 94.973417 | 68.968727 | 96.300687 |
| 32×(load+store sectors)，MiB | 37.026733 | 64.019165 | 0.655762 | 55.019165 |
| DRAM throughput，% of peak | 0.000094 | 0.000058 | 0.000053 | 0.000037 |

组合版cap168相对旧single-add累计local字节代理减少9MiB，约14.06%，以L1命中为主。short_scoreboard比值降低，wait基本相同；eligible/issue active没有提高，no_instruction略高。支持“spill/local访问减少”的局部结论，不能据此宣称显著吞吐提升或自然策略达到168门槛。组合自然策略仍176分配、2blocks/SM；cap168仍3，未增加驻留数。

local sectors×32是累计缓存访问代理，不是唯一数据、显存容量、DRAM或PCIe字节，不乘19次重放。stall比值不是墙钟占比，no_instruction不是直接I-cache miss数；本批baseline该比值与上批不同，比较使用本批同输入，保留全部记录，不把不同采集条件的数拼成吞吐结论。

证据：本地`stage1_single_compact_ncu_{baseline_168,single-add_168,single-compact_255,single-compact_168}_20261006/`中的`trace.ncu-rep`、`metrics.csv`、`quantitative.json`、`app.log`、`command.json`、`exit.txt`。[计数摘要](D:/code/MPA-OpenCl/tools/bench/summarize_stage1_profile.py:10)。

新候选不支持TPI32，拒绝门禁已覆盖TPI32、错误寄存器档、非PRAC及不支持位宽。此前TPI32减半曲线用于匹配提交grid，不代表驻留blocks/SM一定相同；本轮比较均为N4423默认4608/TPI16。

## 6. 复现

```powershell
python tools/test/test_prac_single_add.py --output docs/data/my_single_compact_roles.json
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/parallel_nvcc.ps1 `
  -BuildDir build_cuda_cmake/prac -Only 'cgbn_stage1_prac.*\.cu$|cgbn_stage1\.cu$' -Jobs 6
python tools/test/test_cuda_prac.py --bits 4423 --tpi 16 --registers 255 `
  --variant single-compact --device 1 --output docs/data/my_single_compact_q
python tools/test/test_cuda_prac_windows.py --single-add --single-compact --production `
  --production-counts 32 --production-chunks 7 32 --device 1 `
  --output docs/data/my_single_compact_windows
python tools/bench/bench_stage1_prac_slicing.py --curves 384 768 1536 `
  --variants baseline single-add single-compact --registers 255 168 --chunks 4 32 `
  --count 32 --b1 260000000 --seconds 6 --warmup 2 --repeats 2 --device 1 `
  --exp-cache build_cuda_cmake/prac --output docs/data/my_single_compact_fixed
python tools/bench/bench_stage1_prac_variants.py --configs resident natural cap168 `
  single single168 singlecompact singlecompact168 --target-ms 50 100 `
  --b1 10000000 260000000 --curves 1536 --seconds 15 --warmup 5 --repeats 2 `
  --device 1 --exp-cache build_cuda_cmake/prac --output docs/data/my_single_compact_prefix
```

NCU通过profile_cuda_stage1.py的`--prepare-only --memory`生成admin.ps1，以管理员权限串行执行；精确参数在各证据目录command.json中。未来只修改候选body时可用parallel_nvcc.ps1的`-Only 'cgbn_stage1_prac_single_add\.cu$'`；脚本SkipUpToDate采用全目录最新时间，不应把它作为按TU依赖隔离的证明。当前隔离证据是include关系与四份既有内核完整机器码一致，尚未独立测量未来仅body改动的构建时长。

## 7. 采用决定

CPU模型、GPU完整Q/save及窗口门禁、128逐字节对照、72固定窗口样本、52普通前缀样本及4份管理员NCU均完成。保留single-compact显式候选，不提升默认或当前最佳吞吐推荐；默认算法、TPI、寄存器、TPB128及100ms均保持原配置。已知最佳对照仍baseline cap168/50ms；未发布Auto B2 T1或推广其他位宽/硬件。

最终只读审计exit0：核对当前exe、8个源码/配置SHA、CPU模型源码、两门禁全部case、128份逐字节原文件、72/52样本与每组两次重复、四份NCU的输入/模式/寄存器/驻留/exit、对象SHA及四份既有内核完整机器码；9个工具AST及报告源码链接行号检查通过。证据本地`stage1_single_compact_final_audit_20261006.json`。这是本轮证据审计，不代表长期Stage1优化目标完成。

第一假设的寄存器门槛未成立；第二假设中“减少spill/local访问”得到静态及动态证据，但只带来约0.4%的旧single-add/50ms前缀改善；第三假设也未产生新的最佳配置。小于1%的差异仅两次重复，不作普遍性能保证。

下一候选优先利用单点链的输出T与全部ADD输入不别名这一具体契约：将ADD的中间v暂存于输出T.x，检验能否减少一个bn临时量，保持normalized 4M+2S。此专用版本只能用于已证明输出不别名输入的调用，不能直接替换允许输出覆盖任意输入的公共prac_add。先做别名/规范化与逐步点门禁，再检查168分配门槛和固定/普通前缀吞吐；不能跳过Montgomery规范化。
