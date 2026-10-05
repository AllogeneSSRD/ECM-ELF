# ECM Stage2：进位与残余诊断融合

2026-10-05，GPU1 RTX4060 Laptop / sm89 / CUDA13.3，固定 Goldilocks PTX3，outer unroll0。接续 [六模乘 xADD / D](STAGE2_XADD_D_OPTIMIZATION.md)、[点折叠与 D 发布](STAGE2_POINT_FOLD_D_CALIBRATION.md)、[outer ILP 留出](STAGE2_NTT_OUTER_ILP.md)。本轮优化真实 NTT 乘法的进位末端，不使用 Tensor Core。

## 1. 已完成的前置项与本轮依据

六模乘 xADD 已接入；精确 Mersenne 点乘共用 SOS 乘积再线性折叠/旋转，保持 Montgomery 坐标，主导 limb MAC 从 `2W²` 降到 `W²`。匹配范围的 profile6 已按新点算术独立标定；生产893在此前八条对照中48.8331495→39.0421205s。上述完成状态与原始行号/数据见前置报告，不能重复算成本轮收益。

此前 [点折叠 Systems 数据](D:/code/MPA-OpenCl/build_cuda_cmake/_point_mersenne_20261005/candidate_profile/summary.json) 中，残余诊断7959次、0.973829497s；进位 cone R5为7416次、1.134357552s，R6为543次、1.249723649s，三者合计约3.358s。设备1事件覆盖约38.749s，无自身 GPU 事件的空隙约5.393s；这只是该进程事件空隙，不能据此宣称 GPU 全局闲置或全部空隙都来自 PCIe。

原 [carry_residual_kernel](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:593) 再遍历全部进位输出，块内八个 warp 更新 shared 计数，块间更新相同 slice 的两个全局计数。先测“独立分块诊断 + 最终汇总”：大数组局部快约17%，小数组因额外 launch 明显变慢。最终候选直接把块摘要放进已有进位 kernel，省去第二次全数组读入。

## 2. 实际算法与源码位置

[原 cone helper](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:879) 展开既有 carry recurrence；原 [cone kernel](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:903) 用 warp ballot 找到最近 generate/stop，lane0必要时向前查找，解决任意长度全 mask 的二进制进位链。

新 [carry_cone_check_kernel](D:/code/MPA-OpenCl/tools/bench/ntt_carry_partial.cuh:33) 调用相同 helper，保持 recurrence、二进制传播和实际输出值。它从算出的最终 digit 生成超界计数/最大 bit height，然后调用 [carry_partial_store](D:/code/MPA-OpenCl/tools/bench/ntt_carry_partial.cuh:6)：warp shuffle归约，八个 warp 的32bit摘要写64B shared，一次 barrier，warp0写本块独占的两个32bit数。尾部无效线程参加 block barrier，只有有效 prefix 参加原 warp 传播；不能直接保留原 kernel 的 early return。

[carry_partial_finish_kernel](D:/code/MPA-OpenCl/tools/bench/ntt_carry_partial.cuh:57) 每 slice 一个256线程块，遍历该 slice 摘要，以64bit累加 count、32bit取最大 height，最终对原 `dRes` 执行 additive/max 更新。非零历史 verdict 保留，不能在每个 interior chunk 清零。这与原 [deferred check](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3589) 的错误累积合同一致。

[实际调度](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3492) 在 R1..10 使用 fused cone；超出 cone 上限时仍执行原 extract/add 降高度，最后 R1 cone 可融合。末端 [诊断分派](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:3583) 使用摘要 finish 或原 residual。forward/inverse、pointwise、scale、slot packing及顺序保持。

[NttArena 临时区](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1804) 仅在 `NTT_CARRY_CHECK_FUSED=1` 且 `L×m≥2²⁰` 时使用；默认0。新临时区由当前 arena 独占、复用并计入硬预算；切换 shape 淘汰或 release 释放。预算不足、分配失败、未使用 arena 的路径回退原检查。`NTT_CARRY_CHECK_ALLOC_FAIL=1` 是故障门禁开关。此实现没有提供多曲线共享 workspace lease。

[统计](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1945) 报实际 fused/skipped/refusal/grow及当前/峰值 payload；[D guard](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10810) 在请求新模式时停用旧经验 profile，即使某次分配回退，也不会把混合路径冒充已校准路径。候选用显式 D 比较。

## 3. 计算量、访存、内存与时间口径

记 `L=2^k` 为每 slice NTT word 数，`m` 为 batch slice 数，`T=ceil(L/256)`；每个 word 8B。这不是 ECM 模数 N，也不是 B2；B2、D决定 polynomial 数/尺寸，实际每次调用的 L/m来自已选 shape。

- 原 residual 逻辑读取 `8Lm B`。候选写入加读入摘要 `16Tm B`，净减少 `8Lm−16Tm B`；L为256整数倍时为原检查读量的 `127/128`。这是算法字节计数，不是实测 DRAM 事务或带宽。
- 新显存 payload `8Tm B≈Lm/32 B`；L=2²⁴/2²⁶/2²⁷、m1为0.5/2/4MiB。真实完整曲线最大4MiB。只有一个当前 shape 临时区；未新增 L-word 数组或 CPU 数据镜像。
- 原 cone每次helper展开 `R(R+1)/2` 次 mask/shift/add表达式。记跨warp lookback额外helper调用数为H，则主helper表达式数为 `(Lm+H)R(R+1)/2`，每调用至多R+1次word读取，输出8LmB。全mask病理输入H可达O(mL²/32)；本轮保持该算法，不把它冒充严格O(Lm)进位。算法操作数不等于硬件周期，未采集cycle/stall计数。
- 新 H2D/D2H payload增量0B，CPU数据生成量增量0；原每 slice 两个64bit verdict 的 readback保持。临时区增长产生 cudaMalloc/free，预算、生命周期和峰值在统计中体现。
- 原进位计算的 R、lookback和输出写入保持；不是减少 NTT 蝶形或模乘的优化。新增每 CTA 两组32bit shuffle reduction、八warp摘要及一个 barrier，finish需 O(Tm)工作。原 carry+check 两次 launch变为 fused carry+finish 两次 launch；不是减少 launch 数。
- 原块间诊断竞争约 O(Tm)个全局 counter 更新；候选最终 O(m)个更新，bad count为0时不执行 atomicAdd。finish声明96B shared，fused cone声明64B shared。
- 编译资源 R5 REG35→39、R6 REG39→40，原 cone shared0→64B；finish REG24/shared96B。真实 native R1..10及finish均STACK0/LOCAL0。容量 API 对探针 R5/R6 新旧均6CTA/SM；容量上限不能替代实测 occupancy。

`t_check` 的原 event仍只包末端诊断，现在仅包 finish；摘要工作已转入 cone。因此不能把 `t_check` 的下降单独当成总收益。需要合计 carry+check或完整曲线。沿用此前3.358s基数，假设所有相关调用省18%–23%，粗估完整39s曲线省0.60–0.77s（1.5%–2.0%）；尺寸、分配和其他工作会改变实际值。

## 4. 独立实验与负结果保留

[primitive builder](D:/code/MPA-OpenCl/tools/build/build_ntt_carry_partial_probe.ps1:1) 从实际源码逐字提取原 residual 和 cone，编译原/候选。同一探针八次 ABBA+BAAB，每次三次 warmup、20次 CUDA event重复，所有输出逐字比较及 CPU参考在 event外；没有 CI。

[原始测量](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/cone_shapes/measurements.json) 使用 R5/bpw26，实际 batch形状 k11×990 /k12×495 /k16×30 /k17×15，两步均值 ms分别0.247770→0.20815225 /0.24834575→0.204787 /0.24106225→0.195699 /0.24091925→0.19522575，减少15.99/17.54/18.82/18.97%。k24/26/27、m1为2.1280515→1.637837 /8.385573→6.52318725 /16.7945715→13.07456ms，减少23.04/22.21/22.15%。

R6/bpw26，k24/26/27两步为2.49498875→2.03174375 /9.851456→8.10668575 /19.67849825→16.20839475ms，减少18.57/17.71/17.63%。R5 k16×1仅约3.66%、R6约2.03%；不据此为所有小调用增加工作。实际小调用阈值使用 Lm，而非仅k。

初期独立两级 residual 的 k12/16单 slice约翻倍，大数组才改善；因此没有作为通用诊断路径提升。首次 sweep因 Python scratch公式括号位置错误被 evaluator 拒绝，保留 [被拒数据](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/sweep)，修正为 `8*((L+255)//256)*m` 后使用新目录。初期/v2/v3与最终接入的源分别冻结，不能用当前改过的源宣称早期 exe闭包一致。

主要独立测量四raw依赖及 driver冻结在 [cone_original_sources](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/cone_original_sources)；实际生成源/二进制在cone_probe_v3。验收时逐条核验 binary、原始/生成源码 hash与真实shape参数。原始日志/二进制在Git忽略目录，本机保存；工具和本报告进入Git。

## 5. 实际路径、错误门禁与复现

[CPU/原核对照探针](D:/code/MPA-OpenCl/tools/test/ntt_carry_partial_probe.cu:28) 覆盖随机64bit输入、任意长 mask链、warp/CTA尾部、输入只读和输出 canary，非零初始count/height及三次累积。独立诊断180cases/657000words/bad0，损坏摘要传播成功；主要测量时 cone七组bpw/R配置、18种shape、新旧两种模式均bad0。最终工具增加R1/4/7/8/9，使R1..10都有CPU对照；最终12配置×18shape×2模式均bad0，见 [最终原语门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/final_primitive_gate_driver.log)。

[实际 NTT runner fixture](D:/code/MPA-OpenCl/tools/test/ntt_carry_fused_probe.cu:1) 使用GPU生成稀疏多项式，经实际完整NTT乘法，验证122850个输出；16cases/bad0。覆盖同arena模式切换、复用、三次deferred检查、上一 interior错误在后续良好调用后仍exit4、nondeferred reset、allocator/cap回退、小调用/乘法overflow拒绝、release计账清零。预期错误输出是故障注入结果，不是失败门禁。

候选 native SHA256 `8a789739a355b0185ea48f5f50aeeef692482d7b3341ff1c3382b1d8857955da`，CUDA编译295.2s，20个raw构建依赖。复现构建：`tools/build/build_ecm_cuda_stage2.ps1 -Build build_cuda_cmake/reproduce_carry_fused -Arch sm_89 -GlBackend ptx -OuterUnrollU 0 -Rebuild`。运行前设 `NTT_CARRY_CHECK_FUSED=1`，显式 `--d 1381380`。

[完整比较工具](D:/code/MPA-OpenCl/tools/bench/bench_stage2_carry_check.py:1) 固定同exe、M4423/B1=1000/sigma26保存点、B2=2011326186870/D1381380、point1/unroll0/PTX3、默认全部检查；清理继承的NTT开关。预先指定两条完全验收的warmup（0/1），再八条0/1/1/0/1/0/0/1，全部日志保存；warmup不进入均值，不事后删除不利样本。每条执行前后验证20原始依赖、冻结副本、exe、driver及save hash。

## 6. 完整曲线结果与原生验收

[全部原始命令/日志/覆盖/计账](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/whole_ab/measurements.json)。预热0/1为38.523433/37.967179s；八条测量full依次38.224616 /37.894490 /37.987322 /38.170698 /37.862764 /38.259463 /38.251149 /37.964662s。

- full均值 **38.2264815→37.9273095s，减少0.299172s /0.782630%**。
- main均值31.89238125→31.592670s，减少0.939758%；init6.3341005→6.33463925s，慢0.008506%，视为近似持平。
- 前ABBA组38.197657→37.940906s，减少约0.672%；后BAAB组38.255306→37.913713s，减少约0.893%。每模式4条、无CI，只覆盖该Q/B2/D/设备。
- 每条候选7866次fused、93次小尺寸回退、8次scratch增长、0次预算/分配拒绝；scratch峰值4194304B。默认0无fused、无scratch分配。
- arena full payload峰值3341481200→3345675504B，增加恰好4MiB；这是源码计账，不是NVML整卡或所有私有owner之和。现有pool malloc/free统计不计这8次独立scratch增长，新增carry统计明确报告增长。

十条含预热均恢复相同Q SHA256 `33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`，save SHA256 `0fe48106563dc727c092f4baf7b4c0f2f3bd57ce3bae9fd9f8a5989f2ec324d4`；leaf4244971527793015097、oracle c85031f6149bae11、空factor/bad0/pending0/clean1。S4覆盖397launches/1836241poly_muls/36615543coeff_reduced，2400selftest/60474GMP/3full_checks，两侧保持，没有减少检查。

[默认0原生门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/native_gate_0/summary.json) 与 [候选1门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/native_gate_1/summary.json) 各18/0。七位宽×两point模式、GMP Mont2048与xADD1280逐字/别名、两generic N、两真实小界保存点/D scope；默认0继续profile5/6，候选1两point模式均legacy回退。合成X=2只用于算术/分派，不声称有效Stage1生产曲线。真实小界两模式均leaf1689529688547722991、无factor、GMP bad0/pending0。

[实际NTT接入门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/final_ntt_gate/summary.json) 13/0：DFT/DIT 96cases/27131904words，same arena四切换/3145728words，policy88，Goldilocks200000，cache生命周期、故意损坏、LOCAL0和容量。该门禁验证transform，不代替§5的实际carry runner及完整曲线。

实际native `NTT_S4_CARRY_TEST_BAD=1` 在首个 interior chunk注入后，于14个累计chunk的deferred finish检出：内核rc4，公开driver返回2（[映射源码](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:539)），未吞错误。初次收尾脚本错误地期待CLI返回4而被拒；保留原脚本/日志，按实际driver合同重新验证为exit2，未修改生产代码绕过失败。实际native强制scratch分配失败也完成真实小界曲线，使用原诊断、相同leaf、GMP bad0/pending0/无factor。[纠正后故障日志](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/carry_fault_v2_engine.log)、[回退日志](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/alloc_fallback_engine.log)。

## 7. Systems机制核验、决策与下一步

同一candidate exe、保存点、固定D/check各一次serial Systems采样，[原诊断](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/profile_0/summary.json)、[融合候选](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/profile_1/summary.json)。只含GPU1，无CPU采样/hardware stall计数；profile会有观测扰动，不能用这两条替代八条A/B的均值。

原R5 cone1.134371451s、R6 cone1.246342064s、residual0.973227829s，合计 **3.353941344s**。新R5 fused1.320364996s、R6 fused1.414617616s、finish0.046814744s，加93个小调用cone/residual0.004768529s，合计 **2.786565885s，减少16.9167% /0.567375459s**。新cone确实更慢，消除后续全word诊断后组合才更快；不能单列finish下降当作收益。kernel总数及7959个乘法检查调用保持，7866个改为fused+finish。

H2D两侧4871次/7151405147B，D2D40次/841498560B；D2H6126→6130次、3209119720→3209119752B，多4次共32B。该单次profile不能说明这4次微小读回的来源；不将算法新增payload0B写成实测每个readback计数完全一致。主体PCIe量未减少，H2D约0.545–0.546s、D2H约0.252s。包括startup/tail的自身GPU事件窗口38.605→37.949s，自身无GPU事件5.464→5.125s（14.153→13.505%）；这不是整卡闲置比例，也不能由两次样本证明CPU准备空隙改善。

**保留可选候选，默认0；当前生产893保持。** 在该真实scope下，组合kernel和两组完整曲线均有正向结果，可进入新NTT成本权重测量与多D/独立holdout。相对同binary默认算法省0.783%，不是新生产发布，也不是达到Prime95的证据。候选flag1自动D目前使用legacy，应给出显式D。

下一步：冻结包含融合的新12种NTT shape成本，重新拟合适配的D profile，并以小界、不同预算及未参与拟合的D留出验证；随后决定是否提升默认。CPU准备/提交空隙另需CPU栈或阶段标记定位；多曲线吞吐需私有owner、共享NTT workspace lease与总RAM/VRAM硬预算。跨层NTT融合/根依赖优化继续保留，尚未声称所有长度和模数获益。

[最终原始依赖/生成源/二进制/生产字节审计](D:/code/MPA-OpenCl/build_cuda_cmake/_carry_partial_20261005/final_audit.json)；native20源在native_sources、最终探针源在final_probe_sources、复现驱动在final_tools。所有重编译入口加入carry header的hash/mtime依赖，防止修改header后误用旧object。
