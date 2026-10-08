# ECM Stage2：NTT 外层蝶形展开与 ILP 实验

2026-10-05，GPU1 RTX4060 Laptop / sm89 / CUDA13.3。接续 [六模乘 xADD 与 D 标定](STAGE2_XADD_D_OPTIMIZATION.md)、[点折叠与新 D 发布](STAGE2_POINT_FOLD_D_CALIBRATION.md) 和 [固定 PTX 单位根负结果](STAGE2_NTT_SMALL_ROOTS_FIXED_PTX.md)。本轮先测独立完整有限域卷积，再接入真实保存点的 Stage2。

## 1. 优先级与参考实现

六模乘 xADD 和点折叠/D 重标定已完成：当前生产893的完整 Stage2 对照为48.8331495→39.0421205s。大界D1381380，小界D390390，profile6只覆盖已测设备/模数/算术/策略/预算。不能在改变NTT调度后继续声称这些经验权重已标定。

本轮优化 [outer_coop_kernel](D:/code/MPA-OpenCl/tools/bench/ntt_coop_outer.cuh:15)。每个u的根乘积 `w=base_root*radix_root` 被随后的蝶形模乘消费，形成依赖链；不同u的蝶形独立。展开u循环可能让编译器交错这些独立链，同时增加寄存器和代码体积。它没有减少模乘次数，也没有删除pass或同步。

参考仓库内冻结源码：GPU-NTT [多层shared butterfly](D:/code/MPA-OpenCl/.refactor/ntt_sources_20261004/GPU-NTT-d03c5eaeadaa780d153496afcb3b6a9b79a13a63/src/lib/ntt_merge/ntt.cu:75) 展开层循环并保留层间同步；sppark [寄存器/shared交换与展开](D:/code/MPA-OpenCl/.refactor/ntt_sources_20261004/sppark-9e5c7951d4ff4992f78af26f48d3c9230b8c4136/ntt/kernels.cu:77) 同时包含循环/模板约束防止spill的说明。本轮是在本仓库现有Goldilocks蝶形上独立测试展开宽度，没有移植这些项目的较小模数算术，也没有引入Tensor Core。

## 2. 计算量、同步、访存与容量

记 `q=2^64−2^32+1`、NTT长度 `L=2^k`、合作radix `R=2^M`、连续offset数 `V=(M==8?16:32)`、`ROWS=256/V`，CTA256线程。每CTA覆盖RV个field word。

- 蝶形模乘：每层RV/2，共 `MRV/2`；两次forward加一次inverse的该类pass为 `3MRV/2`。这是算法调用计数，不是有效硬件周期。
- 合成根乘积：`V Σ_(i=0..M−1) 2^i=(R−1)V`。d较小时stage_roots只让一组线程生成，避免按group重复计算。
- coarse根平方：每方向 `(M−1)V`。inverse提前生成M层base_roots；forward使用两个槽交替发布。
- barrier：一次初始发布、M次层末、`min(M,log2ROWS)`次小d根发布，共 `1+M+min(M,log2ROWS)`。M5/6/7/8分别9/10/11/13；展开不改变这个计数。
- 主数组每outer pass读写各L个8B word，逻辑量 `16L B/pass`；每CTA另读 `8(R−1)` B radix roots与 `8V` B coarse roots。不是实测DRAM字节数，缓存和事务放大另计。
- shared声明payload为 `8[RV+(R−1)+128+(INVERSE?M:2)V] B`；实际编译另有8B对齐。M8 forward/inverse为36096/36864B；M7为35328/36608B；M6为18432/19456B；M5为9984/10752B。

展开没有新增host/设备数据数组、根表、持久workspace或H2D/D2H量；显式payload增量0B，长度/packing/pointwise/scale/输入输出排列保持。寄存器和GPU代码体积增加，不能将“payload不增加”表述为所有设备资源不增加；没有新NVML峰值、DRAM事务或NCU cycle计数。

k24实际采用M6合作outer，12个外层stage分两pass，含tile共3个forward pass；k25/26/27采用M8合作outer，外层分别13/14/15层，另一合作pass为M5/6/7，含tile仍3个forward pass。k23的策略使用普通M4外层，共4个forward pass，是不经过该候选的独立控制长度。所有比较逐行核验真实geometry。

## 3. 独立探针和来源合同

[builder](D:/code/MPA-OpenCl/tools/build/test/build_ntt_outer_ilp_probe.ps1:1) 复制实际NTT源码，仅给u循环加入指定pragma并给入口加入实际宽度/设备声明，固定PTX3。原始六依赖、生成五源及binary编译前后hash核验；默认0表示原编译器调度，1/2/4表示指定展开宽度。原始数学文件在这次sweep期间没有修改。旧u0初版未测量，修正输出目录解析后的u0_v2才是所有对照基线。

[完整卷积工具](D:/code/MPA-OpenCl/tools/bench/bench_ntt_outer_ilp.py:1) 核验生成文件只能包含上述两项差异；每候选先运行原有cooperative/GMP门禁。每个长度/候选独立四进程ABBA，每进程8个run，每run一warm加三个CUDA event样本，初始化/CPU参考/全L个输出比较在event外。16个run/版本属于两个进程，没有将它们当成16条独立曲线，没有CI。

四种宽度各13/0门禁：96组DFT/DIT、27131904个word，缓存淘汰/生命周期；4次同arena模式切换、3145728 word；88次shape边界/覆盖；200000次Goldilocks自检；故意损坏要求exit3；八个outer内核LOCAL0且容量API可驻留。重复轮的0/4再各13/0。

所有原始源冻结在 [original_sources](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_outer_ilp_20261005/original_sources)，初轮 [完整数据](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_outer_ilp_20261005/sweep/measurements.json)，独立重复 [完整数据](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_outer_ilp_20261005/repeat_u4/measurements.json)。这些artifact在Git忽略目录中；文档和复现工具随Git提交，raw日志/二进制由本机保留。

## 4. 卷积结果与编译资源

初轮耗时减少率，按k24/25/26/27顺序：

- u1：2.4983 /0.0010 /0.0025 /0.0158%。
- u2：0.4281 /−0.8353 /−0.4589 /−0.0713%。
- u4：0.7657 /0.7204 /0.6679 /0.7409%。

独立重复0→4均值，单位ms：k23 7.946285→7.945025（0.0159%），k24 12.233772→12.138817（0.7762%），k25 26.806571→26.606293（0.7471%），k26 53.649963→53.274327（0.7002%），k27 112.366657→111.499200（0.7720%）。各组全输出bad0。u2多数回退，不提升；u4四种作用长度方向一致，进入完整Stage2实验。

[SASS核对](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_outer_ilp_20261005/sass_summary.json) 对八个outer实例提取指令正文并hash：u0/u1全部相同。**u1的早先2.50%不能归因于优化**，它没有改变GPU指令；初轮启动/状态波动是可能原因。u4八个实例确实不同。M8 forward静态指令3272→6568，inverse3208→6464，BAR均13保持；静态分支数量也增加，不能据展开推断动态分支或cycle减少。

u0/u1所有M的forward/inverse REG48/46。u2的M7/8 forward升至55/56，inverse仍46。u4的M5/6/7/8 forward REG46/54/76/78，inverse46/48/47/47；LOCAL均0，shared保持。容量API：M5保持5CTA/SM；M6 forward5→4、inverse仍5；M7/8正逆保持2。M6每块寄存器需求从256×48=12288到256×54=13824个32bit register，五块需求超过65536，可解释容量下降。容量上限不是实测occupancy；ILP改善依赖链是源码/结果支持的推断，尚无stall/cycle证据。

粗略Amdahl估计：若此前NTT pool15.27s的全部工作都改善0.75%，整曲线只约省0.115s，即39.04s的0.29%。实际小尺寸并不经过候选，因此不能将0.75%的完整卷积收益直接套到所有Stage2 phase。

## 5. 原生接入与 D 保护

[可选pragma](D:/code/MPA-OpenCl/tools/bench/ntt_coop_outer.cuh:7) 使用编译期 `NTT_OUTER_UNROLL_U=0|4`，默认0；没有每次模乘的运行时分派。[native builder](D:/code/MPA-OpenCl/tools/build/build_stage2_local.ps1:12) 新增 `-OuterUnrollU`，编译签名与manifest绑定实际选择，修改选择不能复用旧object。

[实际D guard](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10809) 在宽度4下禁用已有经验profile；[调度日志](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10825) 明确实际编译宽度。默认0继续原profile5/6 scope。候选固定D测量，未将旧12权重冒充新标定，也未改变所有profile数值。[原生point/scope门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_point_mersenne_native.py:23) 核验manifest、实际日志及两种点模式的legacy回退。

复现候选：`tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/reproduce_outer_u4 -Arch sm_89 -GlBackend ptx -OuterUnrollU 4 -Rebuild`；固定保存点运行时显式 `--d 1381380`。宽度4暂未有新成本模型，自动D会使用legacy，不应将其选择结果当成优化后的D校准。

完整曲线工具为 [bench_stage2_ntt_outer.py](D:/code/MPA-OpenCl/tools/bench/bench_stage2_ntt_outer.py:1)，与冻结893/19源对照，只允许outer调度、D guard/声明和builder三文件变化。恢复相同M4423/B1=1000/sigma26 Q，B2=2011326186870/D1381380，两侧point1/XADD6/PTX3/检查保持。

## 6. 真实 Stage2 对照与决策

候选native SHA256 `48bd8842dfc4317097d1ba94ae90a2b85a7a0f84aa05ba3c298fd783bf0360ef`，CUDA编译330.0s，19个manifest原始依赖逐字冻结并在每条曲线前后核验。[构建与依赖](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_outer_ilp_20261005/native_u4/build_manifest.json)，[实际native资源](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_outer_ilp_20261005/native_resources.txt)。八个outer实例REG/shared与探针完全一致，STACK0/LOCAL0。

[native门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_outer_ilp_20261005/native_gate/summary.json) 18/0：七个Mersenne位宽×两种点模式、2048 Mont/GMP和1280 xADD逐字/别名检查；两种generic N fallback；两条真实小界保存点恢复及点模式0/1下的D legacy保护。前16条中的合成X=2仅用于算术和分派，不宣称有效Stage1生产曲线；两条真实保存点均leaf1689529688547722991、GMP bad0/pending0/无factor。

完整8条顺序0/4/4/0/4/0/0/4，full分别39.281450 /38.610765 /38.419246 /38.375742 /38.346580 /38.342189 /38.379841 /38.436073s。均值full **38.5948055→38.4531660s（减少0.36699%）**，main32.19406325→32.05513225s（减少0.43154%），init6.4007420→6.3980335s（减少0.04232%）。各版本4条，无CI。

分组检查：前四条0/4/4/0的full均值38.828596→38.5150055s，减少0.80763%；后四条4/0/0/4为38.361015→38.3913265s，候选**慢0.07902%**。第1条baseline明显较长，后续两侧共同下降；这与状态/暖机漂移相容，尚未测量具体原因。不能把八条总均值的小幅改善解释为稳定kernel加速，也不能将首条作为事后异常值删除再挑选有利均值。[全8条原始log/coverage/命令/phase](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_outer_ilp_20261005/whole_ab/measurements.json)。

八条恢复相同Q SHA256 `33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`，save SHA256 `0fe48106563dc727c092f4baf7b4c0f2f3bd57ce3bae9fd9f8a5989f2ec324d4`；leaf4244971527793015097、oracle c85031f6149bae11、空factor集合/bad0/pending0/clean1保持。S4 397launches、1836241poly_muls、36615543coeff_reduced、2400selftest、60474GMP、3full_checks保持。两侧显式D model0，candidate实际调度声明4；old893未有新声明，宽度0由其冻结source/manifest核验。没有取消强制检查。

**结论：保留可选实验实现，不提升生产默认。** 完整有限域卷积的小幅收益已独立复现，完整Stage2稳定收益尚未建立。当前生产仍893，默认调度0及已验证profile6；生产exe字节没有替换。没有为未提升的候选进行新的多D拟合。若继续这个候选，下一步需先设置明确暖机并重复完整对照，再冻结新NTT权重与D数据，不能直接将0.75%乘进旧模型。

下一优先项应提高收益规模：针对outer的根乘积与同步依赖，评估更深入的跨层融合/访存组织；同时用新点算术后的profile继续定位CPU准备和提交空隙。扩大radix或引入完整twiddle表必须先算shared/REG/VRAM及总pass量。多曲线吞吐仍需私有owner与共享NTT workspace lease；公平Prime95同Q/B2/线程对照仍待完成。此轮不证明达到或超过Prime95。

## 7. 接入后收尾验证

最后另建两份probe，使用 `-IntegratedSchedule -UnrollU 0|4` 直接编译当前正式header中的宏路径，而非早期插入pragma的生成路径；manifest记录compiled_outer_u，所有门禁和计时log逐条核验实际编译宏。0/4再次各13/0；k24/25/26/27完整卷积均bad0，名义减少2.7331/0.6976/0.6723/0.7786%。k24首次baseline较长，保留全数据，不用2.73%替换§4的独立稳定重复0.78%；k25–27继续约0.7%。[接入后数据](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_outer_ilp_20261005/integrated_ntt/measurements.json)。

指令正文逐实例hash核验：新宏0八个outer内核与冻结原0全部相同；新宏4八个内核与早期u4探针相同；真实native4八个内核也与探针相同。这样既验证实际接入了候选，也证明默认0没有改变outer GPU指令。其余NTT算术源未改。[最终审计](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_outer_ilp_20261005/final_audit.json)。

native基本入口 [21/0](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_outer_ilp_20261005/native_accept/summary.json)：save校验、ini/worker配置、worktodo可选B2/skip/count、用户原示例、非法known factor拒绝、真实保存点已知factor、不同sigma、队列成功迁移和失败保留。测试只操作新建fixture目录，GPU1；外部GIMPS生产目录和GPU0任务未改变。
