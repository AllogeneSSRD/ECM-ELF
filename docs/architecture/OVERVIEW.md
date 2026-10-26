# 架构与术语

## 当前能力与职责

| 模块 | 职责 | 主要入口 |
| --- | --- | --- |
| Stage1 应用 | 配置合并、输入、模式、依赖组装、展示和退出 | [main.cpp](../../src/main.cpp) |
| Stage1 单任务 | 批次寿命、调用执行、逐曲线身份、发布与 seal | [task.cpp](../../src/ecm/stage1/task.cpp) |
| Stage1 队列 | 领取任务、处理任务状态、finished、源行改写及跨阶段交接 | [main.cpp](../../src/main.cpp) |
| CUDA Stage1 | CGBN 曲线批处理、实际设备资源、分档和点乘 | [Montgomery](../../src/ecm/montgomery_stage1/cuda/)、[Edwards](../../src/ecm/edwards_stage1/cuda/) |
| CPU Stage1 | 完整 CPU 执行流程与 Montgomery/Edwards 算术 | [Montgomery](../../src/ecm/montgomery_stage1/cpu/)、[Edwards](../../src/ecm/edwards_stage1/cpu/) |
| OpenCL Stage1 | 设备配置、路径注册、动态内核组装 | [engine.cpp](../../src/ecm/montgomery_stage1/opencl/engine.cpp) |
| Stage2 应用 | 配置、CLI、查询/调优/父任务/子进程模式和退出 | [main](../../src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp) |
| Stage2 任务 | 唯一父任务循环、完整曲线 worker、规划与调优协作 | [task.cpp](../../src/ecm/stage2/task.cpp)、[planning](../../src/ecm/stage2/planning/planning.cpp) |
| 进程适配 | 曲线子进程的启动、pipe、Job、停止与寿命 | [process.cpp](../../src/ecm/stage2/process.cpp) |
| CUDA Stage2 | 点生成、多项式树、NTT、归约、下降、GCD | [模块职责](STAGE2.md#cuda-模块边界)、[engine.cpp](../../src/ecm/stage2/cuda/engine.cpp) |
| CPU Stage2 | Edwards save 续算，GW 多线程快速多项式或 libecm NTT/plain | [ecm_cpu_stage2_main.cpp](../../src/ecm/application/cpu_stage2/ecm_cpu_stage2_main.cpp)、[GW 后端](../../src/ecm/stage2/cpu/gwpoly.cpp) |
| GUI | App 绘制/交互；Controller 编排 worker；WorkerProc 管理本机进程 | [GUI](../../src/ecm/application/gui/) |
| 工具 | 构建、参考计算、数据集、计时、剖析和图表 | [tools](../../tools/) |

`src/` 根目录只保留 `main.cpp`；其余生产代码位于 `io/`、`sync/` 与 `ecm/`。独立 OpenCL 测试、微基准和 ISA 工具分别位于 `tools/test/`、`tools/bench/` 与 `tools/disasm/`。桌面工具和 Android 共用的 manifest 实现在 `tools/bench/`，由各自构建显式链接。生产流程需要的设备自检仍由 OpenCL 后端拥有。

生产 Stage2 源码独立于 `tools/experiments/cuda/stage2_tree/` 实验引擎；开发 wrapper 位于该目录的 `worker.cu`。实验树、BSGS、NTT 变体和 Tensor 候选均使用独立源组，生产闭包拒绝实验和测试目录。具体职责及探针接口见 [Stage2 模块边界](STAGE2.md#cuda-模块边界)。

生产 giant chain 核在 `src/ecm/stage2/cuda/device/giant_chain.cuh`；Stage2 主机流程、算术和 NTT 的依赖方向见 [模块边界](STAGE2.md#cuda-模块边界)。G 根树的分块空位布局和压缩请求形状由 `src/ecm/stage2/giant_tree.h` 定义。它们分别服务点生成和根多项式调度，F 树及其有序下降仍由原布局控制。

`ecm` 与 `ecm_cuda` 编译同一组 Stage1 驱动、CPU 算术、配置、队列和存档源文件；两者在 `ecm_backend_*` 接口分别接入 OpenCL 与 CUDA 实现。`ecm_cuda_stage2` 保持独立的 save/回执执行流程。三个程序共用 [整数表达式解析](../../src/ecm/application/configuration/ecm_expr.cpp)与 [worktodo 读写](../../src/io/worktodo.cpp)；队列读取结果区分任务、空队列和 I/O 错误，避免把读档失败视为处理完成。构建的共用 Stage1 源清单在 [cuda_sources.cmake](../../tools/build/internal/cuda_sources.cmake) 中维护，由根 CMake 消费；不同后端仍分别编译这些源文件。

Stage1 按曲线阶段组织，CPU/CUDA/OpenCL 实现在各自曲线目录下；共有任务、执行合同、标量、选择与进度位于 `stage1/`。`stage2/` 集中任务、输入、规划、进程及 CPU/CUDA 续算。`cpu/`、`cuda/`、`opencl/` 只保留实际跨阶段或跨曲线的技术支持；没有第二套隐藏的阶段执行流程。公开请求与选择合同在 `include/ecm/stage1/`，阶段私有协作头留在 `src/ecm/stage1/`。模块职责和位宽边界见 [Stage1](STAGE1.md#cuda-模块边界)。

## 目录与依赖

Stage1 入口位于 `src/main.cpp`，主机输入与队列读写位于 `src/io/`，备份与 P95 同步位于 `src/sync/`；计算与后端实现位于 `src/ecm/`；公开接口在 `include/ecm/` 的对应家族。私有协作头、模板、设备发射和 OpenCL 运行资源均留在 src，不为每个实现文件制造一个公开头。

共享 Edwards GMP 构造位于 `curve/`，供 Stage1、Stage2 和群阶计算复用。GKL 构造保留为实验资料，不接入生产曲线选择。生成物、设备实现、第三方引擎和 OpenCL 资源目录分别表达维护来源或实现职责。

```text
include/ecm/
├── integer.h                       共享 GMP 拥有类型
├── stage1/                         请求/拥有结果、选择、标量、进度、链接后端接缝
├── stage2/                         概率模型、曲线进程请求
├── application/configuration/      runtime.h：已解析运行环境
├── io/                             保存范围、恢复容器、结果发布
└── platform/                       日志、停止、指纹和版本
src/
├── main.cpp                        Stage1 模式、唯一任务循环与收尾顺序
├── io/                             输入、文件、命名、最终save、恢复容器和结果
├── sync/                           备份、P95 和 Stage2 持久交接
└── ecm/
    ├── application/                CLI入口、配置和GUI
    ├── stage1/                     共同任务、合同、选择、标量、准备与进度
    ├── edwards_stage1/{cpu,cuda}/   Edwards完整流程和设备实现
    ├── montgomery_stage1/          Montgomery完整流程，共享param分派
    │   └── {cpu,cuda,opencl}/
    ├── stage2/                     任务、输入、几何、进程与控制台
    │   └── {planning,cpu,cuda}/     纯主机规划和各后端续算
    ├── curve/                      跨阶段共享曲线构造
    ├── group_order/                群阶数学
    ├── diagnostics/                PARI/GP诊断与独立host
    ├── cpu/                        共享IFMA算术
    ├── cuda/                       共享算术、NTT、运行时与tiers
    ├── opencl/                     运行时、算术、诊断与源码资源包
    └── platform/                   停止、哈希、日志和生成默认值
```

`src/io` 保持平铺：files 提供路径、持久写入、短锁与消费租约；save 管理最终文本格式、校验、批量发布及连续 Range；results 管理结果格式与发布；stage1_checkpoint 和 stage2_checkpoint 分别管理恢复容器与曲线进度。prime95/prime95_save 是外部队列与二进制格式适配，exponent_cache 遵循可重建的可选失败政策，json 提供共享的严格解析实现。只为生成物保留 generated 子目录。

`src/sync` 的 p95 集中冻结 intent、文件交付、AID 处理、worktodo.add 发送及恢复；stage2 管理独立的本地交接 outbox；backup 只负责备份。两种队列格式不混用，短编辑锁与消费租约保持不同类型和粒度。数学 checkpoint codec 留在对应曲线与后端目录，成本模型归 `stage2/planning`。

Stage1共享文件按任务、计算合同、sigma、纯标量、进度和平台运行辅助组织；execution.h集中私有入口与结果合同，computation.cpp独立实现结果拥有权和验证。sigma集中抽样和范围规则，checkpoint仍拥有恢复状态；libecm_compat保留历史C类型与符号，Android保持独立窄链接。

计算目录按曲线阶段聚合，每个曲线阶段内部再按执行技术组织。CPU/CUDA/OpenCL 分别实现实际支持的曲线组合；没有统一所有点运算的虚函数层。算法差异、状态域、实例和 checkpoint 载荷仍属于具体后端。普通 CXX 的主机数学、文件编码和显示可独立修改；实际 CUDA 分配、发射、同步和回读继续依赖 CUDA。

main → parse_cli → read_config（一次启动快照）→ normalize_settings → direct / live worktodo → 冻结任务 → main 选择执行入口 → task 直接调用完整后端流程。`src/io/cli.cpp` 生成 `ParsedCli`：每次出现的选项立即转换和校验，保存 canonical presence、原始拼写与 argv 位置，不安装运行时单例或写文件。`config.cpp` 校验所有显式计算值，包括其他 worker 和被覆盖的值；`ParsedConfig` 保留原文位置、已解析的 worker scope、选择枚举和 affinity。生成绑定的 `copy` 投影已解析字段及显式标志，不再解析同一文本。

`normalize_settings()` 从生成默认开始，投影全局与所选 worker，再应用显式 CLI，产生启动时的 `Stage1Settings`（配置、计算选项与运行环境）。main 只安装一次有效 runtime；不把结果回填 `ParsedCli`。`gpu-info` 复用相同投影，只取查询需要的字段，不构造虚拟任务。Stage2 接收端配置与 GUI worker 数使用同一启动快照。共享 TOML 文法位于 `config_format.h`，供 GUI 保真编辑和 Stage2 使用；`worktodo.cpp` 集中实时读取、CSV/worker 解析、选择、短编辑锁、首条同文匹配、原子改写及完成恢复。

队列只有 worktodo 一个来源。main 持有 worker 消费租约，计算期间不持有队列编辑锁；每次完成重新读取文件，按当前 worker 中第一条完全相同的任务匹配，不依赖旧行号。`normalize_direct()`与`normalize_worktodo()`将CLI/stdin和两种worktodo语法收口为拥有N、字符串和选项的`NormalizedStage1Task`；main在归一化后不再补写计算字段。队列票据、原行和交接文本独立保存在`WorktodoStage1Task`，不会进入计算请求。

`Stage1Request` 借用规范化输入至同步调用结束，`TaskResult` 拥有逐曲线结果及实际保存范围。task调用保存Module编码最终批次，并由发布器在短文件锁内显式返回已提交的METHOD记录区间，完成时强制重读；历史追加、去重或恢复不能用“末尾总数减请求数”代替区间证明。队列成功任务保留 checkpoint 至完成记录与源行提交成功；中断的完成记录优先恢复，无法判断队列提交版本时停止并保留证据。

日志镜像的寿命和外部日志逐行读取由日志 Module 管理，P95 业务解释仍由同步代码负责。`save_sync_dir_1/2` 是独立备份；P95 使用 `p95_sync_dir`，先持久记录交接责任，再验证和发布目标 save，最后发布 worktodo.add。跨进程消费确认的限制和恢复方式见 [Stage1 使用说明](../usage/STAGE1.md#prime95-持久交接)。Stage2 的父任务与曲线子进程职责保持，见 [Stage2](STAGE2.md#父任务与曲线子进程)。

`tools/test/test_main_flow_boundaries.py` 检查当前项目内包含关系、公开/私有方向、旧目录和唯一入口；实际编译依赖与运行行为由构建及 worker 回归验证。编译影响验证使用 `test_host_incremental_boundaries.py`，须在无其他生产构建时运行，它保存并恢复原始源码字节。

## 文件并发与持久责任

`src/io/files.cpp` 的 `FileLock` 为合作写入者提供短 gate，覆盖完整读取、合并、替换或 append。worktodo 编辑使用 `<资源>.edit.lock`；save、结果、outbox、P95 pending/ledger 和 worktodo.add 发布使用 `<资源>.publish.lock`。Stage1 与 CPU/CUDA Stage2 的真实结果入口共用这一协议；显式共用 internal JSONL 时保留另一阶段的完整记录，仍严格验证校验和及阶段标记。Stage1 仅对本批按 sigma 排序，Stage2 继续逐条 append。

资源身份解析绝对路径和已有父目录别名，再处理剩余路径；Windows 多资源使用大小写无关的固定顺序。最终文件是符号链接或具有多个硬链接时拒绝写入并提示使用实际路径，避免原子替换使别名与 gate 分离。internal/submission 等不同用途不得解析为同一资源。结果双文件按稳定身份排序取锁；P95 源 save 释放后再取得目标 save；outbox 在内部取得 queue edit，计算期间不持有这些短 gate。

取得锁使用单调时钟：默认最多等待 5 秒；worktodo.add 发布最多 3 秒，GUI 编辑和低层 P95 pending gate 立即失败。多个资源的等待预算分别计算，不代表一次任务总上限。Windows 的共享冲突、锁定冲突和访问拒绝按预算重试：零访问权限的属性观察句柄可能使已关闭的载体暂时处于待删除状态，此时也返回访问拒绝。真正的权限错误仍有界失败并保留原错误码；其他不可重试的 I/O 错误立即返回。诊断包含资源、等待时间和系统错误码。Windows 短 gate 使用内核独占句柄并在关闭时删除载体；POSIX 使用 flock 并保留稳定载体，禁止 unlink 后重新创建竞争 inode。不能根据 mtime 或年龄抢占仍由活跃进程持有的锁。

worker `OwnerLease` 和任务 checkpoint 锁属于长期拥有权，分别保持整个消费进程和任务寿命；它们不是每次读取的短锁。共享字节 owner 载体及已有独立 Stage2 worker gate 保持各自生命周期，残留文件名本身不证明仍被占用。P95 的旧 `worktodo.add.lock` 纯 marker 已不参与新协议，不依据其年龄删除或抢占；它与 intent、pending、ready、sending 等业务恢复材料不同。

P95 发布前持久保存完整 attempted 行集，发布可见但确认失败时保留 `.sending` 和 intent；恢复逐条核对 add/活动队列/发送账本，先补齐 sent/recovered 账本，再写 done。责任文件存在但无法读完必须报错，不能当作空文件覆盖；无法计数的失败日志显示 `lines=unknown`。已经持久接管交付责任后允许源队列按既有合同推进，后续 flush 失败保留 intent，不回滚为重新计算。

这些 gate 只约束参加协议的程序，Prime95 本身不参加，模糊状态仍需保留证据。升级时停止所有共享资源写入者，成组替换 Stage1、Stage2、feeder 与 GUI 后再启动；不支持新旧写入者混跑。save、checkpoint、结果和交接责任文件的读取兼容保留，不需要为本轮迁移业务数据。回归入口为 `tools/test/shared_writer_contract.cpp`、`stage1_host_io_contract.cpp`、`file_lock_delete_pending.cpp` 和真实 worker 输入矩阵；共同锁协议的回归证据见[锁实施草稿](../web/stage1-fourth-refactor-review-20261025.html#implementation-result)，当前领域目录、删除待完成竞态与平台验证范围见[领域目录验收](../web/ecm-domain-layout-review-20261026.html#implementation-result)。

## 主要接口的输入与输出

| 入口 | 输入与寿命 | 输出与副作用 |
| --- | --- | --- |
| `run_stage1_task` | Stage1Request借用N、B1/B2、曲线数、save/表达式与已解析选项至同步返回；三个私有依赖回调 | TaskResult独立拥有逐曲线结果和失败来源/旧返回码；任务内发布、seal及批次清理 |
| `select_stage1_entry`（私有） | 当前任务的已归一化Selection | 只返回普通函数入口，无设备操作；task直接调用 |
| `run_mont_stage1` / `run_edwards_stage1`（私有） | 同步Stage1Request与执行副本 | 拥有的ComputationResult；计算与checkpoint安全点，最终文件由task发布 |
| `run_worktodo`（main 私有） | CLI 显式输入、配置快照、唯一 worktodo；逐任务拥有归一化输入 | 唯一任务循环、完成记录、finished、源行推进及交接；没有 source 回调层 |
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
