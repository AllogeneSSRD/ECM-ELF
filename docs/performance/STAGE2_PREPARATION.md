# Stage2 准备、同步与补零开销

## 输入与证据范围

GPU1 RTX4060 Laptop、24SM、8GiB，UUID `8a67b1f8ef1c3177a822813a7ac2224d`，驱动610.88、CUDA13.3.73、MSVC14.51.36231。保持用户55W/1800MHz上限设置，不修改设备控制；GPU0外部任务保持运行。设置上限不代表实际恒频恒功耗。

主要剖析输入为M4423、B1=20、sigma26有效Stage1 save，B2=2.6×10¹⁰、D60060、P5760，batch256、arena6300、fold640 MiB；Graph、短batch选型和短校准关闭，中段M6开启。Nsight Systems2026.1.3以管理员运行，保留全部CUDA API和CPU NVTX范围，采样/CPU上下文切换采集关闭。所有剖析运行均核对因子、rootfold、NTT工作量、S4分块与未插桩输入一致，不进入性能统计。

正式联合计划矩阵、重复数、全部位宽和B2见[outer性能](NTT_OUTER.md#tileouter-联合组织)。性能二进制为 `3a6da7e504b80f3da45cb9cf9c1879444e209b5f14e39d7ba9c5a191fd427a2e`；扩充CPU标记的剖析二进制为 `6e42ab02480240a9512e40c5ad72b4a9f6f16a5319a2fb38984076dbb1112f90`。两版完整SASS文本SHA256均为 `8973bd25d87e5ff566e7bb80e6a5f7e415abcb085373d70c926a64dfc33a2831`，GPU指令相同；后者另外通过完整曲线工作量/输出核对。它们的主机标记范围不同，不混合计时。

## 计时边界

`stage2_full_wall.total=init+main`。init已包含单独打印的shape时间，不能再加shape；这里shape是baby索引枚举，D/显存候选规划在计时之外。

- init：索引枚举，加Montgomery启动自检之后的上下文/S4初始化、baby生成与F树；包含S4 pack/division和形状自检。
- main：inverse、giant/G树/fold、下降、累乘、GCD、命名与最终oracle排空。
- Montgomery启动自检、保存点验证、进程启动、D/显存规划及结果发布位于引擎total之外。

本页GPU窗口从本进程**第一个到最后一个GPU事件**，包含kernel、copy和memset并集。它不覆盖启动前和最后GPU事件后的全部进程活动，不能称为完整进程墙钟或全卡空闲率。

原二进制同一矩阵off侧各4样本中位数为：

| 输入 | 引擎total | 进程墙钟 | 差值 |
|---|---:|---:|---:|
| M521、短B2 | 0.149092 s | 0.616537 s | 0.467445 s |
| M2203、短B2 | 0.596838 | 1.082648 | 0.485810 |
| M4423、B2=2.6×10¹⁰ | 6.299864 | 6.869408 | 0.569545 |
| M2203、P155520 | 19.540058 | 20.218640 | 0.678582 |

差值包含读取、规划、CUDA初始化、计时外检查和收尾，不是纯CPU准备时间。短曲线会明显暴露这些固定成本；长曲线应分别评价引擎和进程指标。

## GPU空隙的互斥分解

扩充标记的一条诊断曲线GPU窗口6.349079秒，本进程没有kernel/copy/memset事件的区间合计0.598152秒，即9.42%。算法将这些区间与同一worker的CUDA runtime调用做区间交集，按不重叠片段计账；不同API或NVTX范围的嵌套时长不重复相加。

| 空隙所在位置 | 秒 |
|---|---:|
| 已记录runtime API之外 | 0.256011 |
| cudaMemcpy调用内 | 0.174105 |
| kernel提交调用内 | 0.044563 |
| cudaDeviceSynchronize内 | 0.024775 |
| cudaMemsetAsync内 | 0.021675 |
| cudaMemcpyAsync内 | 0.019838 |
| cudaFree内 | 0.018067 |
| cudaMalloc内 | 0.011546 |
| cudaHostAlloc / cudaFreeHost内 | 0.008958 |
| cudaFuncSetAttribute内 | 0.003155 |

未列小项保留在JSON中；以上表格不是全部API持续时间。空隙落在某调用内不证明该API是唯一原因，可能包含驱动调度和剖析影响。

runtime API外的0.256011秒，按最内层CPU范围归属为：

| 范围 | 秒 |
|---|---:|
| 尚未标注 | 0.079498 |
| S4请求准备 | 0.045451 |
| S4常量初始化 | 0.031888 |
| 主机树驱动 | 0.030125 |
| Montgomery自检 | 0.019810 |
| NTT形状/缓存规划 | 0.016544 |
| NTT请求准备 | 0.008868 |
| S4形状自检 | 0.006561 |
| GMP pack自检 | 0.004281 |
| baby准备 | 0.003307 |
| G树level准备 | 0.003155 |
| GMP快照校验 | 0.002554 |

其余小项在原始JSON。范围覆盖函数的CPU执行和等待，不能直接等同于该函数的可消除成本。Montgomery启动自检处于引擎total之外；各项优化价值还要按上一节计时边界评价。

![GPU空隙及CPU范围归属](../../data/figures/stage2_prep_gaps_20261015.png)

## 传输与等待

相同剖析中的真实GPU copy事件为：

| 方向 | 事件数 | bytes | GPU累计时间 |
|---|---:|---:|---:|
| H2D | 1549 | 155,880,627 | 13.703 ms |
| D2H | 2146 | 125,198,016 | 12.165 ms |
| D2D | 309 | 324,326,240 | 1.809 ms |

参考CPU范围追踪中的1843次同步cudaMemcpy API累计约4.56秒，大部分与GPU工作重叠。默认流上读回结果依赖前面的计算；API持续时间不能当作PCIe传输时间，也不能与kernel持续时间重复相加。保留同步前须先识别其结果消费者、缓冲复用和算术断言依赖。

当前样本不支持“传输带宽是主要瓶颈”的结论。常规及全API追踪中，属性设置在空隙中约3–4ms，pinned申请/释放约9–23ms；单独缓存它们无法解释数秒API等待。

## pack与补零的候选空间

GPU memset共4391次，写入84,325,430,832 bytes（78.53GiB），累计0.298869秒。按发起调用的correlationId关联CPU范围，NTT.request内4315次贡献84,325,430,216 bytes和0.298839秒；该范围包含A/B工作区补零及诊断计数器，不能把所有次数都归为大数组清零。

同一曲线pack kernel累计0.256848秒。当前direct/gather输入路径先对A/B整个NTT数组补零，再写占用的digits。对一个m-slice请求，长度L、系数位宽S、digit宽b、两输入系数数ma/mb，当前写入量近似为：

- 全数组补零：16·m·L bytes。
- 有效digit覆写：8·m·(ma+mb)·⌈S/b⌉ bytes；contiguous路径按padding后的P打包。

若采用完整输出覆盖的pack/zero融合，目标是把这两次写入组织成一次，并减少两次独立memset提交。槽内高位、空系数、NTT尾部仍必须写零；0.299秒不能全部视为可消除收益。需比较coalescing、索引除法、每线程工作量、reverse-B和尾slice成本，再以完整曲线验收。

## 当前实现与下一步

`NTT_HOST_TRACE=1` 默认关闭，启用CUDA附带NVTX3的CPU范围标记；不改变算术检查数量或日志。没有NVTX3头文件的构建在请求该功能时明确拒绝。正式矩阵不启用此开关。

优先级：

1. pack/zero/gather融合：用完整脏缓冲、slot尾部、不同ma/mb、reverse-B、借用B/Q、arena增长/回退以及独立GMP验证后，再做固定D对照。
2. tile11/CTA256与outer联合组织：当前测量均为CTA512，只作为待验证候选。
3. 受限显存、host fold及生产高B2的准备/同步范围，扩大到不同位宽；当前9.42%不是通用上限。
4. 评估同N连续曲线的初始化/计划复用；需要完整的模数/表示身份、状态重置和验证合同，不减少现有必需检查。

## 复现与代码依据

- [工具入口](../../tools/bench/README_NTT_OUTER_PLANS.md)：联合扫描、完整曲线、管理员范围追踪与空隙分解。
- 原始证据：`data/experiments/stage2_tile_sync_20261015/`，其中`nsys_all/`为常规全API追踪，`nsys_nvtx_off/`、`nsys_nvtx_on/`、`nsys_nvtx_full/`为CPU范围追踪。
- 汇总：`data/benchmarks/stage2_tile_sync_20261015/gaps_nvtx_full.json`；剖析和正式性能使用独立的构建身份。
- [ntt_host_trace.cuh](../../src/cuda/stage2/ntt_host_trace.cuh)：默认关闭的范围标记；[ntt_runtime.cuh](../../src/cuda/stage2/ntt_runtime.cuh)中的`ntt_shape_plan`、`ntt_run_passes`、`ntt_poly_mul_batch_dev`为NTT范围入口。
- [ecm_cuda_stage2.cu](../../src/cuda/ecm_cuda_stage2.cu)：`s4_reduce_init`、`s4_compare_snapshot`、`poly_mul_batch_modN`、`build_tree_flat`和`giant_chunk_chain`；`s4_gather_final_input`及`s4_pack_final_input`为补零与pack入口。
- 同文件`stage2_init_begin`、`stage2_init_seconds`及`run_batched`返回处定义引擎计时；[ecm_stage2_phase_times.h](../../src/core/ecm_stage2_phase_times.h)定义互斥阶段，不能将旧嵌套timer重新相加。
