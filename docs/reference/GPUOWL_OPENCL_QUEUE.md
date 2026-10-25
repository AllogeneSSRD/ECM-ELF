# GPUOWL / PRPLL：OpenCL 队列与低 CPU 等待

固定第三方源码研究，核对日期为2026-10-24。对象是本地 `.refactor/gpuowl/`：该目录的README自称PRPLL，同时包含原生OpenCL和CUDA兼容后端，不能仅凭目录名当作某个上游gpuowl发行版。该副本没有可用的独立版本标识；源码身份和冻结副本保存在 `data/experiments/gpuowl_stage2_queue_20261024/source_index.json`。本文不声明ECM-ELF已实现该调度器；应用分析见[Stage2等待研究草稿](../development/STAGE2_HOST_WAIT_DRAFT.md)。

## 1. 核心结论

低CPU占用来自三个配合的动作：异步提交、限制在途工作、查询事件后休眠。关键是**提交下一批以后，等待上一批的marker**，从而让CPU休眠时GPU仍有工作，而不是每批提交完就排空全部队列。

它仍有状态轮询，但每次未完成查询后都会休眠，区别于持续占用CPU的紧密轮询。OpenCL接口本身不保证所有驱动的阻塞调用都低CPU；该程序专门绕开其作者观察到的NVIDIA等待行为。

## 2. 队列确实如何创建和提交

- [clwrap.cpp:488](../../.refactor/gpuowl/src/clwrap.cpp#L488)：创建队列时只可选开启profiling，没有设置乱序执行标志，因此是有序队列。
- [clwrap.cpp:508](../../.refactor/gpuowl/src/clwrap.cpp#L508)：kernel通过 `clEnqueueNDRangeKernel` 提交；没有在每次提交后执行 `clFinish`。
- [Queue.cpp:54](../../.refactor/gpuowl/src/Queue.cpp#L54)：`add`统计提交量，达到阈值才进入marker管理。
- [clwrap.cpp:556](../../.refactor/gpuowl/src/clwrap.cpp#L556)：marker由 `clEnqueueMarkerWithWaitList` 创建；空等待列表的marker位于本队列已提交工作之后。

单个队列保持设备依赖顺序，主机不必逐个等待相邻kernel。主机停止提交并不意味着GPU已停止执行先前提交的工作。

## 3. 两批在途与旧marker

[Queue.cpp:81](../../.refactor/gpuowl/src/Queue.cpp#L81)的 `finish` 与 [Queue.cpp:90](../../.refactor/gpuowl/src/Queue.cpp#L90)的 `queueMarkerEvent` 是两个不同操作。NVIDIA常态路径可以抽象为：

```text
提交 A → marker A → 提交 B → 查询/等待 marker A → marker B → 提交 C
                               CPU可休眠
                               GPU继续执行A，之后执行已排队的B
```

第一次达到阈值时没有旧marker，只放置marker A；第二次达到阈值时，B已经排在A之后，此时才等marker A。A完成后马上放置marker B并继续提交C，不要求B也完成。因此存在约两个提交窗口的在途工作。它不是两个必须并行执行的kernel批次，也不是严格的“两块显存”协议。

`MAX_QUEUE_COUNT` 在 [Queue.cpp:32](../../.refactor/gpuowl/src/Queue.cpp#L32)设为：AMD 3200，其他设备500。它统计Queue封装中的提交项，**不是曲线数、NTT数或固定GPU毫秒数**。源码的每次平方4个kernel只是注释中的示例；实际还会有7、10个kernel、读写、队列依赖和Graph提交。

这同时限制主机跑得过远，避免将大量任务压入驱动内部队列。作者记录的Linux CPU百分比没有完整机器、版本、分母和测试协议，不能移作Windows性能承诺，也不能据此断言本机500项最优。

## 4. 休眠时间从哪里来

[Queue.cpp:108](../../.refactor/gpuowl/src/Queue.cpp#L108)的等待循环执行三件事：

1. 查询旧marker状态；完成则结束。
2. 状态为负数则报告OpenCL错误，不能继续等到完成。
3. 未完成则休眠，再次查询。

请求的休眠时间为：

τ_us = integer((squareCount + 1) × squareTime / 2)。

其中squareCount在放置marker时清零，之后由 [Gpu.cpp:1355](../../.refactor/gpuowl/src/Gpu.cpp#L1355)随完成提交的平方/乘法增加。因此它近似反映**旧marker之后已提交的后续工作量**，不是直接测得的旧marker剩余时间。

[Queue.cpp:127](../../.refactor/gpuowl/src/Queue.cpp#L127)维护squareTime：初值50 µs，忽略首次更新时间，后续限制在30…3000 µs；[Gpu.cpp:3106](../../.refactor/gpuowl/src/Gpu.cpp#L3106)等入口从阶段平均迭代时间更新它。这不是每个kernel的精确耗时预测，也不是硬实时定时器。

例如，假设后续已提交125次平方、每次估算50 µs，则后续工作约6.25 ms，请求休眠约3.15 ms。这只是公式示例；实际唤醒可能更晚，且每次平方kernel数、驱动提交和GPU速度都会改变结果。

**为什么不直接睡完整预测时间？** 源码使用约一半的后续工作时间留出余量，意图让主机在GPU耗尽队列前恢复提交。预测只决定何时再查，真正的完成条件仍是事件状态。

## 5. 另外两种队列不要混淆

### 辅助GPU队列

[Gpu.cpp:1338](../../.refactor/gpuowl/src/Gpu.cpp#L1338)的 `splitQueue/mergeQueue` 在主队列和辅助队列之间放置marker及设备等待；[Queue.h:54](../../.refactor/gpuowl/src/Queue.h#L54)封装跨队列依赖。辅助队列在 `add` 中不触发上述主机休眠，Graph录制期间也跳过主机marker管理。

作用是表达GPU工作依赖和尝试不同kernel组织。**多队列不是低CPU等待的必要条件**，一个有序队列加旧marker已能使用这种节流方式。

### Background主机任务队列

[Background.h:16](../../.refactor/gpuowl/src/Background.h#L16)另有默认容量2的主机任务队列。工作线程无任务、生产者遇到满队列以及 `waitEmpty` 等待完成时，使用条件变量挂起线程；不是不停查询任务列表。

[Gpu.cpp:3145](../../.refactor/gpuowl/src/Gpu.cpp#L3145)把部分保存工作交给它，并在最终证明/对象销毁前排空。它减少I/O对提交线程的干扰，但不能替代GPU事件等待，也不意味着GPUkernel在这个CPU任务队列中运行。

## 6. 原生OpenCL与CUDA兼容层

CUDA兼容后端把每个OpenCL队列映射为一个CUDA非阻塞stream，见 [clwrap_cuda.cpp:835](../../.refactor/gpuowl/src/cuda/clwrap_cuda.cpp#L835)。其 `clFinish` 映射为 `cuStreamSynchronize`，状态查询映射为 `cuEventQuery`，见[同文件:1073](../../.refactor/gpuowl/src/cuda/clwrap_cuda.cpp#L1073)。因此上层旧marker加休眠的组织也能用于CUDA，而不依赖OpenCL独有的算术内核机制。

`CU_STREAM_NON_BLOCKING`描述与默认stream的隐式依赖规则，**不等于**主机等待时自动进入低CPU阻塞。

## 7. 移植时必须保留的语义与不确定性

### 7.1 低CPU不覆盖所有路径

[Queue.cpp:37](../../.refactor/gpuowl/src/Queue.cpp#L37)的写入、[Queue.cpp:64](../../.refactor/gpuowl/src/Queue.cpp#L64)的同步读取仍使用阻塞OpenCL拷贝；`finish`最终仍调用 `clFinish`。这些调用可能等待剩余队列。AMD分支达到阈值后直接finish，依据的是作者特定环境的经验，不是OpenCL标准对AMD的保证。

### 7.2 提交与完成可见性

本地 `enqueueMarker` 和纯状态轮询路径没有显式 `clFlush`。不能仅凭该副本在某驱动工作，就认为所有OpenCL驱动都会及时提交此marker。通用移植需在开始纯轮询或跨队列依赖前明确保证相关命令已flush。

OpenCL还区分轻量状态查询和主机同步点：查到完成状态不应被擅自当成任意共享主机内存的同步协议。同步读写、 `clWaitForEvents` 或 `clFinish` 的原有可见性边界应保留。依据为[Khronos事件查询及Flush/Finish规范](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html#clFlush)，核对页面为v3.1.2。

### 7.3 注释不是实际策略开关

`Gpu.cpp:2943`附近的注释把 `setSquareTime(0)`称作tune忙等设置；实际setter会忽略首次更新并将后续值下限限制为30 µs，不能把传0解释为彻底禁止sleep。`Queue.h`中“读写前marker”的概括也不能替代实际 `add` 的阈值逻辑。

### 7.4 错误必须作为终态处理

OpenCL主路径显式检查负事件状态。该副本CUDA兼容层的 `clGetEventInfo` 对 `cuEventQuery` 只专门处理NOT_READY，其他错误未在这个分支中完整映射；不宜照抄其错误处理。CUDA应用应明确区分success、not-ready和失败，不能将失败当作尚未完成，也不能假定为成功。

## 8. 可迁移的原则

1. 在同一有序stream内连续提交依赖工作，不为每个kernel增加主机同步。
2. 主机必须等待时，优先等最早需要回收的事件，保留后续在途工作。
3. 估计GPU时间而非只计kernel数，用有限休眠让出CPU。
4. 缓冲复用、最终校验与错误传播仍以真正完成为条件。
5. 将GPU提交、设备依赖、CPU后台任务和主机睡眠分别建模；它们解决不同问题。

这些原则可以迁移到Stage2；500项阈值和平方时间公式不能直接移植为NTT默认参数。
