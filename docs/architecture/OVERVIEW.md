# 架构与术语

## 当前能力与职责

| 模块 | 职责 | 主要入口 |
| --- | --- | --- |
| Stage1 驱动 | 参数、队列、指数、检查点、结果和交接 | [ecm_driver.cpp](../../src/core/ecm_driver.cpp) |
| CUDA Stage1 | CGBN 曲线批处理、分档和点乘 | [cgbn_stage1.cu](../../kernels/cuda/cgbn_stage1.cu) |
| CPU Stage1 | Montgomery/Edwards，GMP 与可选 SIMD 算术 | [src/cpu](../../src/cpu/) |
| OpenCL Stage1 | 设备配置、路径注册、动态内核组装 | [opencl_ecm_entry.cpp](../../src/opencl_ecm_entry.cpp) |
| Stage2 驱动 | save、独立队列、回执、日志、Auto B2 | [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp) |
| CUDA Stage2 | 点生成、多项式树、NTT、归约、下降、GCD | [ecm_cuda_stage2.cu](../../src/cuda/ecm_cuda_stage2.cu) |
| GUI | 配置、worker 进程监管、队列生成、监控、结果展示 | [src/gui](../../src/gui/) |
| 工具 | 构建、参考计算、数据集、计时、剖析和图表 | [tools](../../tools/) |

生产 Stage2 源码独立于 `tools/bench/stage2_tree_gpu.cu` 原型；开发 wrapper 使用实验引擎。两者的构建、后端和源闭包须分别核对，不能以原型文档推断生产行为。

## 数学与数据术语

| 记号 | 含义 |
| --- | --- |
| N | 实际待分解整数；余因子与完整梅森数是不同输入 |
| M | 算术承载模数；普通模式 M=N，承载实验 M=2ᵖ−1 且 N∣M |
| S | 算术模数位数；承载时用 bit_length(M)，另记目标 N bits |
| W | Stage2 每系数的 64-bit 字数，⌈S/64⌉ |
| sigma | 参数化的曲线种子，不能跨参数化解释 |
| B1/B2 | 两阶段的光滑性/单大素数界限 |
| Q | Stage1 完成点；与 N、sigma、B1、指数模式绑定 |
| D | Stage2 baby/giant 间距参数 |
| P | baby 集大小 φ(D)/2，也是 F 的次数 |
| I/G | giant 点数 ⌊B2/D⌋+2，G=⌈I/P⌉ 批 G 树 |
| F/G/H | baby 积树根、当前 giant 多项式、累计模 F 的乘积 |
| L | NTT 长度，取受支持的 2 的幂；与目标整数 N 区分 |
| TPI/TPB | 每实例协作线程数/每 block 线程数；不等于实际驻留数量 |
| save/checkpoint | 完成的 Stage1 点/未完成的 Stage1 中间状态 |
| owner | 有独立分配与释放责任的设备存储；借用/别名不重复计账 |

Montgomery 表示为 xR mod M，乘法返回 abR⁻¹ mod M。普通整数、Montgomery 点坐标、Goldilocks digit 和普通多项式系数必须在入口明确区分。

GCD=1 表示本项未找到因子；GCD=N 是饱和/退化结果，需要细分或恢复处理，不能仅凭等于 N 就判定算术错误。有效输出必须满足 1<f<N 且 f∣N；复合因子仍是有效因子，素性证明另计。

## 证据边界

功能存在、通过构建、算术正确、性能提升和发布资格是不同结论。当前实现以源码与已完成验证为依据；性能数据绑定输入及冻结二进制。规划查询不执行曲线，组件 payload 不代表全进程显存峰。

模块行为分别见 [Stage1](STAGE1.md)、[Stage2](STAGE2.md)、[NTT](NTT.md)、[内存](MEMORY.md)、[配置](CONFIGURATION.md)。未完成工作集中在 [TODO](../TODO.md)。
