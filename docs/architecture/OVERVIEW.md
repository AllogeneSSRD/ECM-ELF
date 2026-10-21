# 架构与术语

## 当前能力与职责

| 模块 | 职责 | 主要入口 |
| --- | --- | --- |
| Stage1 应用 | 配置合并、输入、模式、依赖组装、展示和退出 | [ecm_driver.cpp](../../src/ecm/application/cli/ecm_driver.cpp) |
| Stage1 单任务 | 批次寿命、调用执行、逐曲线身份、发布与 seal | [task.cpp](../../src/ecm/service/stage1/task.cpp) |
| Stage1 队列 | 领取任务、处理任务状态、finished、源行改写及跨阶段交接 | [queue_workflow.cpp](../../src/ecm/service/work/queue_workflow.cpp) |
| CUDA Stage1 | CGBN 曲线批处理、实际设备资源、分档和点乘 | [Montgomery](../../src/ecm/backend/cuda/stage1/montgomery/)、[Edwards](../../src/ecm/backend/cuda/stage1/edwards/) |
| CPU Stage1 | 完整 CPU 执行流程与 Montgomery/Edwards 算术 | [CPU 后端](../../src/ecm/backend/cpu/stage1/) |
| OpenCL Stage1 | 设备配置、路径注册、动态内核组装 | [opencl_ecm_entry.cpp](../../src/ecm/backend/opencl/stage1/montgomery/entry.cpp) |
| Stage2 应用 | 配置、CLI、查询/调优/父任务/子进程模式和退出 | [main](../../src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp) |
| Stage2 服务 | 唯一父任务循环、完整曲线 worker、规划与调优协作 | [task.cpp](../../src/ecm/service/stage2/task.cpp)、[planning](../../src/ecm/service/planning/stage2.cpp) |
| 进程适配 | 曲线子进程的启动、pipe、Job、停止与寿命 | [stage2_curve_process.cpp](../../src/ecm/platform/process/stage2_curve_process.cpp) |
| CUDA Stage2 | 点生成、多项式树、NTT、归约、下降、GCD | [模块职责](STAGE2.md#cuda-模块边界)、[engine.cpp](../../src/ecm/backend/cuda/stage2/engine.cpp) |
| CPU Stage2 | Edwards save 续算，GW 多线程快速多项式或 libecm NTT/plain | [ecm_cpu_stage2_main.cpp](../../src/ecm/application/cpu_stage2/ecm_cpu_stage2_main.cpp)、[GW 后端](../../src/ecm/backend/cpu/stage2/gwpoly/ecm_stage2_gwpoly.cpp) |
| GUI | App 绘制/交互；Controller 编排 worker；WorkerProc 管理本机进程 | [GUI](../../src/ecm/application/gui/) |
| 工具 | 构建、参考计算、数据集、计时、剖析和图表 | [tools](../../tools/) |

生产 Stage2 源码独立于 `tools/experiments/cuda/stage2_tree/` 实验引擎；开发 wrapper 位于该目录的 `worker.cu`。实验树、BSGS、NTT 变体和 Tensor 候选均使用独立源组，生产闭包拒绝实验和测试目录。具体职责及探针接口见 [Stage2 模块边界](STAGE2.md#cuda-模块边界)。

生产 giant chain 核在 `src/ecm/backend/cuda/stage2/device/giant_chain.cuh`；Stage2 主机流程、算术和 NTT 的依赖方向见 [模块边界](STAGE2.md#cuda-模块边界)。G 根树的分块空位布局和压缩请求形状由 `src/ecm/core/stage2/ecm_stage2_giant_tree.h` 定义。它们分别服务点生成和根多项式调度，F 树及其有序下降仍由原布局控制。

`ecm` 与 `ecm_cuda` 编译同一组 Stage1 驱动、CPU 算术、配置、队列和存档源文件；两者在 `ecm_backend_*` 接口分别接入 OpenCL 与 CUDA 实现。`ecm_cuda_stage2` 保持独立的 save/回执执行流程。三个程序共用 [整数表达式解析](../../src/ecm/application/configuration/ecm_expr.cpp)与 [worktodo 读写](../../src/ecm/storage/queue/ecm_worktodo.cpp)；队列读取结果区分任务、空队列和 I/O 错误，避免把读档失败视为处理完成。构建的共用 Stage1 源清单在 [cuda_sources.cmake](../../tools/build/internal/cuda_sources.cmake) 中维护，由根 CMake 消费；不同后端仍分别编译这些源文件。

CUDA 实现集中在 `src/ecm/backend/cuda/`：Stage1 按曲线组织主机实现、设备算法及独立实例；可选 PRAC 属于 Montgomery 私有家族。发射头、CGBN 类型和 dispatch/tiers.def 为私有实现。公开请求与选择合同在 include/ecm/service 和 include/ecm/core；实际后端准备/执行合同留在 service/stage1 私有头，应用只在组装处消费。模块职责和位宽边界以 [Stage1](STAGE1.md#cuda-模块边界) 为准。

## 目录与依赖

生产实现全部位于 `src/ecm/`；公开接口在 `include/ecm/` 的对应家族。私有协作头、模板、设备发射和 OpenCL 运行资源均留在 src，不为每个实现文件制造一个公开头。

```text
include/ecm/
├── application/configuration/       runtime.h：已解析运行环境
├── core/{stage1,stage2}/            选择、标量、概率模型
├── service/{stage1,work,execution}/  请求/拥有结果、队列合同、链接后端接缝
├── storage/{queue,results}/         曲线进度、结果发布
└── platform/                       日志、进度、指纹、版本、process 请求
src/ecm/
├── application/
│   ├── cli/                        Stage1 driver、输入、queue_source、群阶诊断
│   ├── configuration/              配置合并、生成配置、表达式和 Stage2 身份
│   ├── cuda_stage2/                CUDA Stage2 CLI 入口
│   ├── cpu_stage2/                 CPU Stage2 CLI 入口
│   └── gui/                        App、Controller、WorkerProc 和 GUI 模型
├── service/
│   ├── stage1/                     task、拥有权、准备与唯一执行分派
│   ├── execution/                  libecm/C 执行类型与兼容声明，非第二个执行器
│   ├── work/                       QueueWorkflow
│   ├── stage2/                     父任务、曲线 worker、输入与私有合同
│   └── planning/                   Stage2 方案、校准、profile 与测量
├── core/{curve,stage1,stage2}/      共享曲线构造、平台中立算法和数据模型
├── backend/
│   ├── cpu/{stage1,stage2,arithmetic}/
│   ├── cuda/{stage1,stage2,ntt,arithmetic,planning,runtime,dispatch}/
│   └── opencl/{stage1,arithmetic,runtime,resources,diagnostics}/
├── storage/{cache,checkpoint,save,queue,transfer,results,profile}/
└── platform/                       文件发布、哈希、日志、进程与轻量默认值
```

顶层按职责，后端内部按技术再按曲线。CPU/CUDA/OpenCL 分别实现实际支持的曲线组合；没有统一所有点运算的虚函数层。算法差异、状态域、实例和 checkpoint 载荷仍属于具体后端。普通 CXX 的主机数学、文件编码和显示可独立修改；实际 CUDA 分配、发射、同步和回读继续依赖 CUDA。

driver → task → 唯一执行分派 → 完整后端流程。队列模式在 task 外包一层唯一 QueueWorkflow，处理领取和完成协议；它与单次模式共用 task。Stage1TaskSession 仅负责批次/锁清理，不执行第二遍流水线。已有 CUDA factory.cpp 只实现能力查询、设备准备和实例接缝，不新增一层 task factory。存储模块执行文件机制，不决定队列推进策略。

Stage1Request 借用模数、选项和命名至同步调用结束，TaskResult 拥有逐曲线结果。队列回调交付拥有输入数据的 QueueItem。Stage2 服务接收已解析选项和窄策略，构造显式 CurveProcessRequest；进程适配返回退出状态并转送输出，不承担曲线规划。具体输入输出、失败顺序和限制分别见 [Stage1](STAGE1.md#单任务编排与资源拥有权)、[Stage2](STAGE2.md#父任务与曲线子进程)、[GUI](../usage/GUI.md)。

`tools/test/test_main_flow_boundaries.py` 检查当前项目内包含关系、公开/私有方向、旧目录和唯一入口；实际编译依赖与运行行为由构建及 worker 回归验证。编译影响验证使用 `test_host_incremental_boundaries.py`，须在无其他生产构建时运行，它保存并恢复原始源码字节。

## 主要接口的输入与输出

| 入口 | 输入与寿命 | 输出与副作用 |
| --- | --- | --- |
| `run_stage1_task` | Stage1Request借用N、B1/B2、曲线数、save/表达式与已解析选项至同步返回；三个私有依赖回调 | TaskResult独立拥有逐曲线结果和失败来源/旧返回码；任务内发布、seal及批次清理 |
| `execute_stage1`（私有） | 已解析执行选项、N/界限、曲线批量、命名；可写执行结果 | 选择一次完整CPU/GPU流程，准备/恢复/安全点协作后填充结果；不重复最终task发布 |
| `run_queue_workflow` | QueuePaths、NextQueueTask、TaskDependencies；回调填充拥有N/字符串/选项的QueueItem | 兼容退出码；唯一finished、源行推进和交接协议；失败策略不交给文件store |
| `run_tasks` / `curve_worker`（私有） | 已解析Options和窄路径/队列/默认值；worker重读save/hash | 父任务进度、曲线结果、receipt和退出状态；校准共用同一请求投影 |
| `run_curve_process` | 拥有协议字段的CurveProcessRequest；同步selection通知，不保留调用方引用 | 0/3/1退出状态及输出转送；资源/I/O/停止异常向调用方传播，清理本机句柄 |
| `internal_json` / `submission_json` / `publish_*` | 记录、借用N、运行元数据与目标路径 | 文本或durable文件/账本/receipt；存储异常由task/queue解释 |
| `WorkerController` | WorkerSettings、借用TomlFile、启动/停止/tick命令和同步通知接收器 | `workers()`只读借用视图与退出阶段；进程寿命由WorkerProc持有；App拥有显示状态 |

这些接口对应现有完整职责。序列化返回文本，持久发布修改文件，设备协作负责真实同步/回读；三者不通过同参转发层串成另一套任务流水线。

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
| save/checkpoint | [完成的 Stage1 点或因子](SAVE.md) / [Stage1 中间状态或 Stage2 曲线进度](CHECKPOINT.md) |
| owner | 有独立分配与释放责任的设备存储；借用/别名不重复计账 |

Montgomery 表示为 xR mod M，乘法返回 abR⁻¹ mod M。普通整数、Montgomery 点坐标、Goldilocks digit 和普通多项式系数必须在入口明确区分。

GCD=1 表示本项未找到因子；GCD=N 是饱和/退化结果，需要细分或恢复处理，不能仅凭等于 N 就判定算术错误。有效输出必须满足 1<f<N 且 f∣N；复合因子仍是有效因子，素性证明另计。

## 证据边界

功能存在、通过构建、算术正确、性能提升和发布资格是不同结论。当前实现以源码与已完成验证为依据；性能数据绑定输入及冻结二进制。规划查询不执行曲线，组件 payload 不代表全进程显存峰。

本机LOCAL完整位宽、可选路径及配对性能的实际覆盖与证据见 [CUDA 模块验收](../performance/CUDA_STRUCTURE_QUALIFICATION.md)。多架构发布暂缓；构建、兼容镜像或PTX JIT结果与目标实体GPU的原生运行分别记录。

模块行为分别见 [Stage1](STAGE1.md)、[Stage2](STAGE2.md)、[NTT](NTT.md)、[内存](MEMORY.md)、[配置](CONFIGURATION.md)。未完成工作集中在 [TODO](../TODO.md)。
