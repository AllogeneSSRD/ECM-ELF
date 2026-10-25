# 双 CUDA 发布与 JIT 验证

发布目标、入口、完整载荷及复用合同以[构建与发布](../usage/BUILD.md)为准。本文区分静态镜像、实际加载模块、数学对照与耗时；构建成功不自动获得运行资格。完整两套包已生成，运行覆盖仅以以下实际证据为准。

下述Stage2证据对应0.52.1的 `kernel_layout=isolated`，适用于该冻结构建及隔离备选，不认定当前默认grouped布局已通过逐容量隔离或取得相同JIT耗时。grouped的数学、冷/热加载及稳定吞吐须按新二进制身份另验。

## 环境与证据

证据根为 `data/experiments/release_build_redesign/`。冻结计算基线为0.52.0，位于 `baseline_20261024/`；新计算版本0.52.1，GUI独立0.52.0。两份发布组件的精确程序、构建路径及源码/工具链/对象身份由 `build_cuda_release/components/cu<版本>.stageN.json` 指向的清单绑定。

本机使用Windows x64、Ryzen AI9 HX370、32GiB内存。GPU1为RTX4060 Laptop GPU、sm89、驱动610.88；设备功耗上限报告不可用，未改变功耗或频率。工具链为CUDA11.8.89/MSVC14.16.27023和CUDA13.3.73/MSVC14.51.36231，vcpkg GMP6.3.0#3通用x64-windows。实际GMP DLL SHA256为 `88eed87ffb237b627df76d011a5267db832045bbdc4705a1d98382d50edbc6a9`。每次运行保存命令、输入、私有配置/缓存、程序摘要、设备信息及原始输出。

## 完整包与已验行为

`release/ecm-elf-0.52.1/`包含两个独立目录及各自ZIP/SHA256。每包83个文件；cu11.8 ZIP为116647612字节，cu13.3为121221693字节。两包对应源码快照相同，共1291个文件，排除私有测试输出。`build_cuda_release/components/package-manifest.json`保存完整文件摘要、四份计算程序身份及逐模块报告摘要；其`tests_run=false`表示打包过程未执行运行测试，运行证据另存，不回写成构建成功即运行通过。

- `bundle_final_verified/result.json`：两包文件/ZIP摘要与成员完全相符；复制到带空格和中文的目录，PATH仅System32，两个worker版本查询、GMP群阶/点阶助手及GUI v2启动通过。GUI识别相邻配置，未启动worker，配置与队列摘要不变。本机已安装的系统CRT仍存在，此项不等于干净Windows验收。`negative_bundle_payload_verified/result.json`另在私有副本实际移走Stage2、改动GMP DLL字节，两种负例均拒绝，随后复原并确认原发布目录未变；损坏DLL未执行。
- `switching_verified_v3/result.json`：单进程8项私有队列任务、32条末点记录与冻结基线一致，实际旧包、现代包及现代PTX副本均通过。只观察到六个所需曲线/param/位宽模块，重复任务没有重复加载模块。
- `handoff_cu11.8/summary.json`、`handoff_cu13.3/summary.json`：每套5组Stage1末点及5组Stage2对照通过，包含Edwards交接、承载254、relocation和owner预算回退；目标环叶指纹与结果字段相同。
- `arithmetic_legacy.log`、`arithmetic_modern.log`：每套独立GMP探针覆盖1536个模数、152064次比较，错误0。探针链接实际发布对象，身份为`build_release_probes/cu<版本>/stage2/manifest.json`。
- `independent_math_verified/result.json`：两套发布对象的Edwards独立GMP对照各131个用例、537条曲线通过，覆盖全32档、NAF窗口、点不变量、17/113分段恢复与自适应重启。汇总保存两阶段探针构建清单/日志，并重新核对程序、GMP、源码和生产对象摘要。
- `runtime_stage1_legacy_full/result.json`、`runtime_stage1_modern_full/result.json`：完整Stage1覆盖Edwards全32档NAF/binary、Montgomery全31档param0/2/3，每条计算路径比较四条完整末点记录。旧包314条PTX/默认路径、现代包471条PTX副本/SASS/默认路径全部通过，各另有2项纯元数据查询无设备模块/JIT活动。每次实际加载入口均核对曲线、param和容器，没有无关实例。
- `runtime_stage2_verified/result.json`：8193位通用、16381位Mersenne、16384位通用三个独立CPU/GMP准备输入，共15条镜像路径通过；强制检查、GMP对照与目标环叶指纹一致，实际选用容量256。
- `stage2_all_capacities_v4/result.json`：Edwards和Montgomery保存点分别覆盖4/8/16/32/64/128/256个64位字容量，共14个输入、70条镜像路径通过。每次请求四条曲线，逐条比较实际结果与基线，包含发现因子后的正常停止。观察器按PID、模块ID与事件序号保存每次加载镜像，7130个事件文件逐个核对入口；各子进程没有无关容量模块，且订阅/JIT活动状态均成功。`final_evidence_checks.json`核对事件清单、文件成员及各PID状态，防止子进程复用模块编号覆盖证据。
- `ledger_lifecycle_green/result.json`与`ledger_gpu_verified/result.json`：旧MSVC真实头文件双TU生命周期回归通过；两套实际Stage2开启内存账本后，144次分配/释放配平、最终live_bytes=0、unknown_frees=0，数学检查及退出通过。
- `factors_verified/result.json`：独立CPU/GMP准备的两个16384位非单位输入，在冻结基线、两个实际包及现代PTX副本中均返回相同合法因子；固定B1=20、B2=13230、param0、sigma26，预期因子分别1019和2621。

CPU合同6项、profile检查30项、构建身份变异检查54项通过，证据为`cpu_contracts_source_archive_final_v2.log`、`profile_final.log`、`identity_verified.log`。源码归档合同覆盖私有测试队列排除、未提交源码和固定vendor纳入；即使缺少忽略规则，`tools/test/_*`目录也不进入源码快照。所有数学/范围测试均使用私有输入和输出，不修改业务队列。

## 离线构建计时

Stage1两套完整CUDA队列各实际编译128个单元，8路并行。11.8队列墙钟4011.1秒、并发单元经过时间之和20697.7秒；13.3分别为3069.1秒和16557.6秒。证据为`stage1_full_retry.log`、`stage1_cu133.log`及各叶目录`timing/*/command.json`、`nvcc.csv`。这是同机其他构建/验证并发期间各一次队列观测，未计GUI、Stage2、打包与重试；并发单元之和不能解释为串行构建时间。随后正式入口复用128个CUDA单元，未因唯一计时路径改变而重编。

## 镜像与模块观测

发布静态检查分别要求唯一PTX50/SASS60、PTX75/SASS80；Stage1全125个曲线/参数化/容器模块及Stage2全176个设备模块逐对象记录PTX入口和目标。本轮11.8两阶段实际PTX ISA均为7.8，13.3均为9.3，见`components/cu<版本>.stageN.modules.json`。CUPTI测试观察器记录驱动加载的cubin、入口和JIT活动。查询使用纯元数据，任务只接触选中模块；公共模块与算术必需自检另按实际执行解释。

本机13.3程序在设置 `CUDA_FORCE_PTX_JIT=1` 后仍曾加载sm80 SASS，新增 `CUDA_FORCE_JIT` 或禁用缓存亦未使该样例改走JIT。证据为 `modern_force.log`、`force_both/` 以及 `modern_force_rejection.log`；门禁按实际CUDA ELF架构拒绝将其标为PTX通过。此结果限于本机程序/运行时/驱动组合，不推断所有CUDA13.3环境的行为。

现代PTX验证使用明确标记的测试副本：`tools/build/test/build_release_ptx_fixture.ps1` 读取冻结对象，在独立证据目录用nvprune去除SASS，逐对象提取并核对PTX原始字节完全相同，再链接副本。Stage2测试链接去除nvcc生成空SASS占位镜像的原生目标。原发布EXE、对象、清单均只读；测试副本有独立身份，不能替换发布程序或冒充实际包。运行入口通过 `--ptx-fixtures <ptx-fixtures.json>` 显式选择；SASS/默认路径继续测试实际发布EXE。副本观察到sm89驱动生成镜像后才认定PTX路径。未来GPU上的原包自动回退仍须对应实体环境验证。

检查器还须能读取当前驱动生成的cubin。本机11.8的cuobjdump不能解析部分R610新ELF，运行时观察显式使用13.3检查器并记录其摘要；原11.8构建对象和发布镜像仍由对应工具链检查。

重跑时先冻结基线与生产组件，再用`build_release_probes.ps1`构建独立数学探针、`build_cuda_jit_trace.ps1`构建观察器；需要现代PTX路径时由`build_release_ptx_fixture.ps1`生成显式副本。`test_release_runtime.py`以`--stage stage1 --tiers all --skip-sampling`验证完整Stage1，以`--stage stage2 --fixtures <冻结保存点清单> --skip-sampling`验证Stage2；`--sampling-only --samples 5`在新的私有目录采样。共同参数为`--baseline`、`--observer`、`--output`，本机显式`--device 1`；现代PTX附`--ptx-fixtures`。功能测试结束后再采样，保留后台负载、设备快照和每次精确命令。

## 完整程序冷/热计时

证据为`runtime_final_sampling/result.json`，共80个独立进程：2阶段×2个profile×带/不带观察器×冷/热×5组。11.8使用实际完整发布程序，13.3的PTX测量使用上述PTX字节相同的测试副本。每组冷样本使用新私有`CUDA_CACHE_PATH`，热样本复用本组磁盘缓存启动新进程；不同组、观察器开关分别隔离缓存，没有清理全局CUDA缓存。

- Stage1：122位`N=2^121+17`，Montgomery param0、sigma26起始、4曲线，B1=20/B2=0，选择128位容器，指数缓存关闭。
- Stage2：独立CPU/GMP准备的8193位`N=2^8193-3`保存点，param0、sigma26、1曲线，B1=20/B2=13230、D=210、carrier=0，选择256字容量，关闭校准及运行调优。

以下为秒。进程墙钟取无观察器样本的中位数，括号为最小—最大；JIT列取独立观察器样本的编译活动总和中位数，各项样本数均为5。

| 阶段/profile | 冷进程墙钟 | 热进程墙钟 | 冷JIT编译 | 热JIT编译 |
| --- | --- | --- | --- | --- |
| Stage1/cu11.8 | 1.5245（1.2147—1.7001） | 0.2532（0.2355—0.3423） | 1.3104 | 0 |
| Stage1/cu13.3 PTX副本 | 1.1336（1.0696—1.3898） | 0.2659（0.2450—0.3762） | 0.7272 | 0 |
| Stage2/cu11.8 | 14.6955（13.2550—15.3513） | 5.2391（5.1472—5.3330） | 9.6077 | 0 |
| Stage2/cu13.3 PTX副本 | 14.6004（12.1571—14.9694） | 5.0887（2.2985—5.3754） | 9.5718 | 0 |

热样本均记录了缓存活动且没有编译活动；不是将缺失记录填零。`final_evidence_checks.json`检查340个编译区间，没有嵌套或重叠，允许报告该活动总和。驱动编译活动不包含整个进程初始化、文件处理和GPU计算；缓存加载/存储另记。两类中位数来自不同进程，不能相加或把冷暖墙钟差称为纯JIT时间。

采样在本轮功能测试结束后串行进行。GPU1仍有其他已有上下文，未干预；前后快照见`sampling_background_before.csv`、`sampling_background_after.csv`。未固定时钟/功耗，小样本只报告中位数及范围，不给稳定P95或吞吐提升结论。nvcc `--time`为离线编译计时，与运行JIT分开。

## 旧目标 PTX 在 Ada 上的持续吞吐

旧包的兼容目标会影响持续计算性能，不能只比较首次JIT耗时。CGBN在C++编译阶段按 `__CUDA_ARCH__` 默认选择三种整数乘加后端：小于500为IMAD，500到6xx为XMAD，700及以后为WMAD；上游Kepler构建目标为sm35。依据为 `third_party/cgbn/include/cgbn/cgbn.h:67` 和 `core/core.cu:315`。compute50 PTX已经包含XMAD实现，驱动生成sm89机器码时不会重新运行C++条件编译选择WMAD。[NVCC虚拟架构宏说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-compiler-driver-nvcc/index.html#virtual-architecture-identification-macro)

2026-10-25在RTX4070 Ti（GPU0、sm89、60 SM、驱动610.88）复现：输入 `(2^5323-1)/2372950817`、实际5292位/5632位容器、Montgomery param3、sigma26、960曲线、B1=20000、B2=0，关闭定时checkpoint及指数缓存。冻结cu11.8和cu13.3实际发布版0.52.3、本地sm89版0.52.2，分别第二次复用私有CUDA缓存运行；每个程序热样本1次。返回的阶段gputime为8.74327、3.74783、3.76254秒，旧包约为本地的2.32倍。该阶段计时包含主机/设备流程，不等同于纯内核event；首轮含JIT样本不混入比较。六次执行各960条保存记录的数学字段完全一致，仅TIME不同。

同一CUDA11.8.89/MSVC14.16.27023编译的独立CGBN Montgomery乘法探针进一步隔离后端因素：相同模数、5632位/TPI16/TPB128、960组固定随机全宽操作数、每组50000次依赖乘法，加载与预热后用CUDA event计时，每变体5次。中位内核时间为：默认PTX50/XMAD 1701.755毫秒、sm89/WMAD 726.152毫秒、同sm89仅强制XMAD 1645.600毫秒、默认PTX75/WMAD 704.768毫秒。每变体另有4组独立GMP幂模核对，错误0。由同编译器、同sm89目标、只改变后端的对照可知，旧架构算术路径是主要差距来源，不能将其归因于CUDA11.8不支持WMAD。

强制WMAD的sm50 SASS及PTX50也编译通过；PTX50在4070 Ti上JIT为sm89后，中位709.212毫秒（5次event）、4组GMP检查错误0。因此sm70是WMAD的默认启用分界，不能直接当作其硬性编译下限。此项不证明Maxwell/Pascal实体硬件的数学或性能已经通过，也不据此改变正式旧包后端。

同一sm89探针交替WMAD/XMAD各运行3轮，每轮5次event，GPU利用率不低于95%的遥测样本分别22/51条：每轮忙时平均功耗再取三轮均值为202.134/261.866W，SM时钟相应约2802.946/2788.529MHz。该方向与用户报告的旧包高功耗、低吞吐一致；遥测是轮询样本，不将功耗数值直接当作完整任务平均能耗。全部试验未锁功耗/频率，存在桌面上下文，串行执行且不干预其他进程；这些小样本只定位本设备、本输入和冻结二进制的原因，不作为其他位宽、曲线或GPU的稳定加速比。

证据根为 `data/experiments/release_jit_perf_20261025/`。`inputs.json` 保存三个EXE/GMP摘要及版本，`repro.json`、`math_comparison.json` 保存完整程序对照，`source_comparison.json` 核对对应源码归档与实际CGBN/生产算术模板字节相同，`probe_results.json`、`probe_wmad_result.json`、`power_ab.json` 和逐轮日志保存原始测量；`evidence_manifest.json` 保存探针二进制、脚本及汇总身份，`device-after.csv` 保存设备快照。nvcc编译计时仍与运行event分开保存。

本例4070 Ti使用现有cu13.3现代包即可获得接近本地sm89的吞吐。若要旧包同时高效服务现代GPU，需另行验证现代目标镜像或后端分派；全局强制WMAD前须覆盖旧实体显卡。当前发布矩阵和按需模块要求保持不变。

## 兼容性边界与待验条件

用户机器安装的Toolkit版本与运行所需驱动版本不同。实际cu13.3 Stage1/Stage2发布EXE的PE导入表没有外部cudart DLL依赖，使用静态CUDA runtime；运行不要求安装本机13.3 Toolkit。若驱动仅报告CUDA13.0—13.2，官方CUDA13.x小版本兼容机制允许在R580及以后驱动上使用受支持的SASS/既有功能，但PTX JIT及依赖新驱动的功能不在该保证内。当前现代包内置sm80 SASS，Ampere/Ada可使用同大版本兼容镜像；上述4070 Ti/610.88以 `CUDA_DISABLE_PTX_JIT=1` 完成960曲线且结果一致，见 `modern_sass_no_observer/result.json`。Turing及需要其他大版本镜像的设备可能必须回退到13.3生成的PTX75，不能仅凭R580条件宣称兼容。R580/R590/R595实体驱动尚未验收；支持完整13.3 PTX JIT应使用相应R610或更新驱动，不能通过安装Toolkit替代驱动升级。[CUDA小版本兼容条件](https://docs.nvidia.com/deploy/cuda-compatibility/610/minor-version-compatibility.html)、[13.3发布说明](https://docs.nvidia.com/cuda/archive/13.3.1/cuda-toolkit-release-notes/)

旧包Maxwell sm50/52需要PTX50；Pascal sm61可使用sm60同大版本相容SASS。Windows452.39是11.x小版本兼容条件，不能保证11.8 PTX ISA被识别；522.06是11.8对应Windows驱动，可作候选测试基线。LAZY要求R515及以上用户态驱动；显式EAGER或缺少该能力的环境未获得按需范围资格。[11.8发布说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html)、[LAZY版本要求](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-c-programming-guide/lazy-loading.html)。

当前没有Maxwell/Pascal、原生sm60/sm80或旧驱动实体环境；没有宣布最低已验旧驱动版本。已验范围为本机sm89/610.88上的上述镜像路径；原现代包在未来GPU上的自动PTX回退、旧驱动的PTX ISA/LAZY能力、显式EAGER范围仍待实机验证。可接受JIT延迟预算与稳定吞吐门限需另按目标工作负载建立。0.44.6的LOCAL性能数据见[CUDA模块验收](CUDA_STRUCTURE_QUALIFICATION.md)，不作为本次双CUDA/JIT的性能资格。
