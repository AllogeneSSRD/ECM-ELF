# NTT 低层单位根：固定 PTX 后端重新评估

2026-10-05，GPU1 RTX4060 Laptop/sm89/CUDA13.3。先完成六模乘xADD与点折叠/D标定，再返回NTT内核。此处 `st=2/3` 指warp内蝶形层（half=4/8），不是ECM Stage2的步骤编号。

## 1. 为什么需要重测

原 [单位根实验](STAGE2_GOLDILOCKS_SHORT_REDUCTION.md#4-低层单位根方法及负结果) 的方法3相对四fold归约快约2.2%。当前NTT生产已固定PTX3，移除了四fold和热模式分派；旧收益没有证明单位根特化在新基线上有效。完整Stage2的点pool缩短后，NTT仍约15.27s，值得优先重新排序。

本轮保留生产NTT四个共同文件及所有pass/packing/尺寸不变，只在独立probe测试单位根候选。参考当前冻结sppark的warp通信思想；本实现的Goldilocks整数修正和标准根合同独立推导。原开源调研的GPU-NTT和sppark提供融合/访存组织思路，不能直接搬用较小模数的Shoup/lazy范围到接近2^64的q。

## 2. 数学、计算量与数据量

NTT模数 `q=2^64−2^32+1`，`2^64≡2^32−1`、`2^96≡−1`、`2^128≡−2^32`。`γ=2^12` 的阶为16；标准generator7生成的r16=γ^13，逆根γ^3。st3根指数 `(13j或3j) mod16`，st2再乘2。用 `e&7` 提供 `bits=12(e&7)`，e&8决定取负。最大bits84，`v·2^84`需要148bits，必须保留最高20bits。

令s=bits mod64，a=v<<s，b=v>>(64−s)，s=0时b强制0。如果bits<64，归约(a,b)；否则归约(0,a)，再减 `b·2^32`，因为最高word乘2^128≡−2^32。最后按e符号取负。所有结果canonical，任意128bit归约继续通过GMP检验。不能截掉超过128bits的部分。

warp6-pair每lane持有两个word，每层做一个有效蝶形。st2/3候选每64个数据word替代64次64×64乘积和64次8B根读取；仍做归约、指数、宽移位和规范模减。省掉的是乘积和逻辑table load，不等价于64次完整模乘成本归零；根往往cache命中，实际DRAM减少量未知。

无新增持久数组、NTT长度/pass或H2D/D2H payload。t12 tile共享空间仍32768B/CTA，CTA512。一次tile pass数据主数组读+写16L B；inverse另读乘积数组8L B，pointwise/scale仍融合。总显存还包括probe用于初始化/全输出参考的buffer，和生产workspace不能混为一谈。

## 3. 代码与测量合同

- [small_root_reduce](D:/code/MPA-OpenCl/tools/bench/ntt_small_roots.cuh:39)：fixed3下直接用gl_reduce128_ptx，runtime保持原short方法3。避免候选仍用旧C++ short而对照用PTX的混合。
- [独立tile/原语probe](D:/code/MPA-OpenCl/tools/test/ntt_small_roots_probe.cu:1)：fixed后端不再在每个benchmark run强制short0；runtime reducer A/B方法4对fixed构建拒绝。正逆、roundtrip、只读数组与padding合同保持。
- [builder](D:/code/MPA-OpenCl/tools/build/build_ntt_small_roots_probe.ps1:1)：新增GlBackend及编译前后九份raw source hash，manifest记录实际固定模式；默认runtime。
- [计时工具](D:/code/MPA-OpenCl/tools/bench/bench_ntt_small_roots.py:1)：核验实际fixed3声明、每行模式和来源。每形状/operation同binary顺序0/1/1/0/1/0/0/1，各四run，每run一warm加三event样本；填充/初始化/GMP参考/全输出比较在event外。

首先完整原语和tile门禁，资源LOCAL0且容量API给出可驻留CTA才进入计时。k24是主要测量长度，k25/26为独立长度复测；forward、inverse与roundtrip分别报告。这些是tile局部延迟，不等价于完整卷积或完整Stage2收益。

## 4. 实测结果与决策

固定PTX probe编译13.0s，SHA256 `d9700c74cd5c4f4eb95462a727b3d1b5aebdf808232befa4dcf53a66360f4f42`，九份raw依赖冻结；36/0门禁通过。三种根方法各98544 primitive字，任意权值fallback2048字，任意128bit归约100064字，四种tile方法各160组合，forward/inverse/roundtrip/readonly各705856字，padding各5440字；故意损坏被拒绝，live0。selected baseline与method3正逆均REG40/LOCAL0，共享32768B，容量API上限3CTA/SM。[manifest](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/ntt_roots_fixed/manifest.json)、[完整36组合同](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/ntt_roots_gate/summary.json)。

每种长度的0→3均值，单位ms：

- k24：forward1.659136→1.798997，慢8.43%；inverse2.173269→2.321926，慢6.84%；roundtrip3.809280→4.098901，慢7.60%。
- k25：forward3.276288→3.549099，慢8.33%；inverse4.362069→4.660992，慢6.85%；roundtrip7.591680→8.159488，慢7.48%。
- k26：forward6.572629→7.126102，慢8.42%；inverse8.764416→9.399894，慢7.25%；roundtrip15.300011→16.452950，慢7.54%。

九组各8次交叉、全输出比较，均bad0。[完整样本](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/ntt_roots_ab/measurements.json)。逻辑buffer峰k24/25/26分别268898296/537333752/1074204664B，每组结束live0；不是NVML峰。没有改变PCIe数据布局，只省kernel逻辑根表读/乘积。

结论：**不接入生产**。旧2.2%局部收益依赖旧四fold基线，在当前固定PTX下方向相反；不是位宽/工作量变化或local spill导致，因为长度、CTA/shared与资源合同保持。指数/宽移位/修正成本抵消乘积节约是源码和结果支持的推断，没有硬件stall/cycle证据。生产继续原warp tile与新点折叠/D模型。完整卷积和Stage2没有发射这个候选，不能将上述tile退化称为生产退化。

## 5. 后续优先级

下一项检查 [outer_coop_kernel](D:/code/MPA-OpenCl/tools/bench/ntt_coop_outer.cuh:9) 的根乘积与shared/barrier成本。每CTA覆盖R·V数据，M层实际蝶形约M·R·V/2次；当前w=base_root·radix_root总逻辑乘积约(R−1)V次，已在d<ROWS层通过stage_roots共享。可先评估unity根跳过和低层根的编译特化，保持custom root/table fallback，不额外存整个L级twiddle数组。之后再考虑pass融合；扩大R/V节约global pass的同时会增加shared/寄存器，必须保持资源与完整卷积的独立门禁。

无新NCU硬件cycle/实际occupancy/DRAM计数；容量API上限不代表实测occupancy。CPU准备空隙和多曲线workspace lease仍需单独推进。当前生产来源/公式/完整性能见 [点折叠与D发布](STAGE2_POINT_FOLD_D_CALIBRATION.md)。
