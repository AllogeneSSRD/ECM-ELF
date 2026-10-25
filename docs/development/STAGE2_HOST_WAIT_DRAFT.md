# Stage2 低 CPU 等待与有界提交研究草稿

## 1. 初始范围和结论（2026-10-24冻结）

2026-10-24对当前ECM-ELF 0.52.0、提交 `2f91021` 的源码检查。本文是应用研究和待验证方案，未修改生产实现、配置默认值或算术检查，也未新增GPU计时。源码身份和冻结资料在 `data/experiments/gpuowl_stage2_queue_20261024/`。固定第三方分析见 [GPUOWL / PRPLL队列](../reference/GPUOWL_OPENCL_QUEUE.md)。

Stage2已经异步提交部分工作，并有延后计时、pinned双缓冲及GMP快照环。最直接的机会是**改变等待方式及回收粒度，保留已有在途工作**，不需要先引入多stream或重写NTT。

仅从源码不能证明当前CPU占用由哪一项贡献最多，也不能给出收益百分比。必须将真实CPU准备、GMP校验、驱动忙等和提交阻塞分别测量。

## 2. 初始源码核对（2026-10-24冻结）

### 2.1 设备调度模式已有，但不是完整的事件等待策略

[host_policy.cpp:23](../../src/ecm/backend/cuda/stage2/host_policy.cpp#L23)读取环境变量 `NTT_CUDA_WAIT_MODE`：0自动、1主动轮询、2让出调度、4阻塞同步。未设置变量时直接返回，不主动更改设备调度。实际设置后读取flags校验，debug日志可显示 `stage2_cuda_wait`。调用位于[task.cpp:163](../../src/ecm/service/stage2/task.cpp#L163)，在完整Stage2引擎执行之前，部分规划已发生。

`cudaDeviceScheduleBlockingSync`控制设备/stream等待的调度；`cudaEventSynchronize`是否使用阻塞等待还取决于事件的 `cudaEventBlockingSync` 标志。普通 `cudaEventCreate` 以及单独的 `cudaEventDisableTiming` 都不包含此标志。依据：[NVIDIA设备管理](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html)、[事件管理](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)，核对为13.4文档。

因此把环境变量改为4可以作为已有对照入口，**不能保证全部事件等待都停止忙等**。旧cprof还在[planning/stage2.cpp:1571](../../src/ecm/service/planning/stage2.cpp#L1571)绑定模式0，非零冲突会拒绝；正常tune环境由[同文件:1643](../../src/ecm/service/planning/stage2.cpp#L1643)采集NTT变量。比较等待策略时必须核对统计条件，不能忽略错误或悄悄复用不同条件的成本观察。

### 2.2 S4计时环：已有延后回收，满环却等待全部事件

- [s4.h:45](../../src/ecm/backend/cuda/stage2/s4.h#L45)：每个形状的DTRING=8，保存成对的GPU计时事件。
- [s4.cpp:33](../../src/ecm/backend/cuda/stage2/s4.cpp#L33)：`dt_reap`查询已完成事件，累计GPU时间并回收。
- [s4.cpp:48](../../src/ecm/backend/cuda/stage2/s4.cpp#L48)：`dt_slot`找不到空槽就调用 `dt_flush`。
- [s4.cpp:58](../../src/ecm/backend/cuda/stage2/s4.cpp#L58)：`dt_flush`遍历全部在用槽并同步等待。
- [s4.cpp:310](../../src/ecm/backend/cuda/stage2/s4.cpp#L310)：事件通过默认 `cudaEventCreate` 创建，没有事件阻塞标志。

所以满环等待的实际行为是“等完当前形状全部在途计时事件”，不只是注释概括的“等最旧标记一次”。它不一定排空整个设备，但会追到该形状较新的事件，可能消耗用于隐藏主机唤醒延迟的在途工作。是否频繁触发，需读取各形状的 `ring_waits` 并结合API轨迹判断。

优先候选：为计时槽记录提交顺序，满环只等待最旧槽，回收它和其他已经完成的槽；最终收尾仍完整flush。计时start/end位置保持不变，继续保留timing能力，不能为了等待而给这些事件加DisableTiming。

### 2.3 GMP快照环：已经是等最旧项，主要改等待原语

[checks.cpp:551](../../src/ecm/backend/cuda/stage2/checks.cpp#L551)的 `s4_oracle_one`查询head事件；无须阻塞时返回，须等待时调用 `cudaEventSynchronize`。在[同文件:660](../../src/ecm/backend/cuda/stage2/checks.cpp#L660)环满时只消费head，组织方式已经接近gpuowl的旧marker，而不是每次清空整个环。

快照完成事件在[同文件:673](../../src/ecm/backend/cuda/stage2/checks.cpp#L673)仅带DisableTiming；默认环容量4、上限8，见[options.cpp:207](../../src/ecm/backend/cuda/stage2/options.cpp#L207)。容量扩大还会增加pinned快照占用，不能仅凭减少等待次数直接扩大。

此处适合先替换主机等待方式；GMP比较本身仍是CPU工作，休眠不能消除它。必须保留样本选择、signature、selected=compared及最终drain，禁止覆盖尚未校验的快照。

### 2.4 pinned输入/输出：已有一个窗口的重叠空间

[product.cpp:473](../../src/ecm/backend/cuda/stage2/product.cpp#L473)在复用输入槽前查询并等待对应上传完成事件；创建事件在492行，仅DisableTiming。输出在[同文件:611](../../src/ecm/backend/cuda/stage2/product.cpp#L611)异步读回，先提交当前块再等待上一块输出，收尾仍等最后一块。

这已经具备“等待旧工作，GPU还有新工作”的基础。可以沿用原有槽和事件，先研究如何让等待不烧CPU，不能在未完成时覆盖输入或读取输出。pinned申请失败时的同步拷贝回退仍需保留。

[tree.cpp:251](../../src/ecm/backend/cuda/stage2/tree.cpp#L251)和[remainder.cpp:918](../../src/ecm/backend/cuda/stage2/remainder.cpp#L918)也使用这些事件保护上传槽，因此通用等待策略必须覆盖所有使用点，而不只是乘法出口。

### 2.5 无关问题和必要同步

规划、GMP、主机打包和文件I/O属于真实CPU工作。同步D2H若要立即供GCD或主机计算使用，也有真实依赖；不能统一删除。设备分配、同步拷贝和驱动内部队列背压仍可能阻塞，即使代码中的显式等待改为休眠。

Graph录制在[ntt/graph.cpp:33](../../src/ecm/backend/cuda/ntt/graph.cpp#L33)进入capture；它持有单独的capture stream，而不是已证明独立的生产计算stream。不要把host轮询/休眠插入capture，也不要把Graph内部每个节点当成额外CPU等待点。

## 3. 本机已有证据限制

[Stage1主机等待草稿](GPU_STAGE1_HOST_WAIT_DRAFT.md#单变量诊断)记录了Windows/WDDM、RTX4060 Laptop、驱动610.88/CUDA13.3.73的单变量试验：对该冻结Stage1路径，单独设置设备阻塞或事件阻塞标志，进程CPU仍约占一个逻辑核；阻塞设备调度配合显式设备等待才明显降低CPU，同时带来局部墙钟代价。

这说明本机不能仅凭API设置成功宣称低CPU问题解决；它不证明当前Stage2、其他驱动或全部事件类型都具有相同行为。也不适合把Stage1的设备级同步直接塞到每次Stage2多项式乘法后，那会破坏现有异步流水。

[Stage2准备与同步性能](../performance/STAGE2_PREPARATION.md#传输与等待)另记录了同步拷贝调用时间与GPU工作重叠。那些API墙钟不是CPU忙等时间，也不是PCIe纯传输成本；本次没有将旧数字重新标作0.52.0结果。

## 4. 建议的推进优先级

### P0：取得实际CPU归属，不改算术或默认值

沿用 `tools/bench/bench_stage2_host_wait.py` 的自身worker进程CPU采集能力，以及管理员Nsight Systems的CUDA API/CPU线程轨迹。工具的旧save、配置和入口假设先对当前版本核对，不能把文件存在当作本轮已验证可运行。

记录完整生命周期CPU时间、用户态/内核态、墙钟，另在热执行区间采集同口径计数；CPU当量核数为 `CPU_seconds / wall_seconds`。多个worker相加时必须使用对应的共同测量区间，不能把样本不同的进程秒数直接混加，也不能与任务管理器整机百分比混淆。

把事件等待、同步拷贝、提交API、内存分配和API之外的GMP/准备分开，并记录oracle等待、S4满环、输入槽等待及实际GPU空隙。只看GPU利用率或API持续时间不足以确定CPU忙等位置。

### P1：统一完成等待接口，先在既有回收边界接入

保留当前有序stream、资源和事件边界。对需要等事件的地方使用可对照的策略：

1. 快速查询一次，已完成直接继续。
2. 未完成时让出CPU，采用有上限的自适应短休眠，再查询。
3. CUDA错误立即传播；只有NotReady允许重试。
4. 消费数据和复用缓冲必须等真实完成，预测只控制下一次查询时间。

事件阻塞标志可作为单变量对照；是否采用，应由本机CPU时间与吞吐共同决定。完成事件可DisableTiming，性能计时事件必须保留timing。保持事件记录位置不变，避免把CPU休眠/唤醒时间写进新的GPU时间边界。

在途GPU工作为B、预计唤醒延迟为J、恢复提交的CPU工作为H时，只有 `休眠时间 + J + H < B` 才有足够余量不饿空GPU。该式是调度预算，不是对操作系统唤醒的保证。尾批没有后续工作时应单独处理，不能继续套用稳态休眠时间。

### P2：S4满环只等待最旧槽

在提交顺序明确的有界FIFO中回收一个最旧槽，而不是调用完整dt_flush。保持最终排空、每项计时只累计一次、错误传播和跨形状所有权。它比先扩环更容易隔离因果；环容量和快照预算暂不同时改变。

### P3：按GPU时间控制提交窗口

Stage2小NTT与大NTT耗时相差很大；一个Graph提交还可能包含多个kernel，因此不用固定500条命令作为通用默认。可以按算术位宽、NTT长度/batch、路径和阶段的历史GPU事件时间估计每段工作，维持两个有界提交窗口。

窗口中保留旧完成事件，提交下一段后再等待较早事件。起步候选可比较数毫秒级窗口，具体阈值待校准，不写成已批准默认。睡眠量应受剩余后续工作、停止响应、计时粒度及最坏唤醒抖动限制。独立NTT很长时只能在合法边界等待，这不等于缩短单个GPUkernel，也不改变显示驱动watchdog限制。

工作量估计错误只允许改变休眠/节流，不得导致跳过算术、检查或释放在途资源。等待策略改变也应参与性能观察身份及准备缓存失效；不能用忽略environment掩盖真实成本变化。

### P4：确有余量后，再研究后台校验或多stream

GMP快照可以研究有界后台消费者，但必须明确不可变数据、modulus/shape生命周期、取消/异常传播及发布前join。后台仍消耗CPU，只可能减少提交线程被占用的时间。当前NTT工作区、fold owner和默认流顺序不能因另开stream就视为可安全共享；多stream需要另行证明读写依赖和显存租约。

## 5. 未来验证条件

- 固定save、D/P/B2、算术路径、batch及显存预算，关闭自动探索；不同等待方案使用相同算术与检查次数。先做等待原语单变量，再做FIFO回收，最后做窗口组织，不把多项变化混为一次收益。
- 512/约4K/约8K/约16K位宽，短/长NTT和普通/梅森路径分别覆盖；resident及回退、pinned失败、短尾和中断需要保留。小输入只验证功能，不外推生产吞吐。
- 同时核对GMP、自检、oracle signature、末端hash/因子及最终排空，失败时不发布成功；性能核对CPU当量核数、完整曲线/队列墙钟、GPU空隙和主机内存峰。
- 生产数据至少交替重复；预热、采样、设备频率/功耗条件、二进制身份和计时边界写入证据。低CPU不能单独作为吞吐改善的证明。

## 6. 本轮交付和TODO

本轮完成第三方队列机制、当前源码对应及应用优先级分析；没有运行新测试或修改生产调度。后续优先做P0/P1，再评估P2。正常状态文档仍以现有实现为准；本草稿按规则保留，研究落实后交用户处理，不自动删除。

## 7. 2026-10-25 实施记录

当前功能规则改写于 [Stage2架构](../architecture/STAGE2.md#主机等待与提交回收)，性能结论以 [等待验证](../performance/STAGE2_HOST_WAIT.md) 为准。本节保留研究实施过程，不作为另一个运行规范。

- 从提交6f98875工作区构建0.52.3当前源码基线，未使用较早的现存二进制充当当前基线。固定D、N、save和算术检查，以进程GetProcessTimes核对CPU当量。
- CPU测试先复现满环全排空；实现按序号回收最旧槽，保留最终全排空和每项计时恰好累计一次。事件查询不再把CUDA错误吞成未完成。
- 原有显式事件等待约占首个4667-bit样本0.1秒，单独替换几乎没有整进程CPU收益。加入既有阻塞D2H之前的事件等待，保留原同步拷贝和完成条件；NTT通过中立的每线程回调接入。
- 退避请求32..256微秒，Windows私有高精度timer失败回退原等待。实测降低CPU但有吞吐代价，默认关闭；配置stage2_low_cpu_wait、CLI和专家开关均有子进程覆盖测试。
- 初个大P样本暴露满环频次升高后冗余查询的开销；FIFO回收同时改为最旧项未完成即停止查询。CPU夹具验证该路径一次查询，原全扫描最多8次。大P输入B1=260000000、B2=260000000000、D510510交替3次/侧，整条中位数20.915641→20.901611秒，未测出明显速度差；计时槽等待减少并不等于完整曲线节省同样时间。
- 先期诊断曾与构建或另一GPU实验重叠，保留但不进入正式统计。一次legacy侧CUDA700失败未长期复现，不宣称已经修复；冻结原基线7995-bit独立3次通过，后续串行矩阵继续核对。原始失败、Python大整数限制导致的工具中断及重跑位置记录在data/experiments/stage2_host_wait_20261025/exclusions.json。
- 补测原生设备阻塞4与事件退避组合：4667/7995-bit各2次，CPU0.972/0.969核降为0.228/0.135核，整条耗时增加1.34%/0.86%。最终用户开关选择这一组合，读回前休眠保留为独立实验；默认关闭。配置描述由JSON生成，最终父/子进程验证同时核对实际设备调度flags。
- 平台timer实现隔离为普通C++模块，Windows头不进入共享NTT头，避免系统宏与NTT字段冲突；正式HostOnly构建逐项验证CUDA依赖和对象。
- CPU回归覆盖最旧项、环绕、已经完成项、原策略开关、部分事件创建后的释放、错误、最终排空、重复排空及退避/计时器回退。GPU验证不把相同因子集合当作完整检查证明，另核对叶子hash、oracle选择signature、selected=compared与自检/GMP覆盖。

P3尚未接入。休眠在多数强制读回边界后面没有足够已提交工作，唤醒延迟会暴露为GPU空隙。下一优先项是把carry诊断读回做成有界延后窗口，先证明计数器清零/复用前的快照、工作区寿命、错误传播和发布前排空；不能直接在每次NTT后增加设备同步或统一扩大快照环。P4后台GMP与多stream继续等待独立所有权证明。

草稿按仓库规则保留，交付后由用户决定是否整理；不自动删除。
