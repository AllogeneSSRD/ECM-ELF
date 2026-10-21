# ECM 主流程重构方案与分析草稿

分析与实施日期：2026-10-21。范围：当前桌面生产程序的主流程、接口、计算实现和运行资源。状态：已获实施授权；P0、P1a、P1b 已完成并通过 local 验证，Stage2 程序版本 `0.45.21`；P2、P3a/b/c、P4a–i 已完成；P5a–d 已完成存储、规划、父子流程和进程请求；P6–P7 待实施，GUI 按 P6 执行。详见[实施进度与阻塞记录](#实施进度与阻塞记录)。

阅读顺序：[完整目标与约束](#完整重构目标与实施约束) → [构建复用与输出隔离](#首先调整构建复用与输出隔离) → [完整目标目录](#完整目标目录) → [模块职责](#模块职责依赖与构建目标) → [职责划分与重复封装检查](#职责划分与重复封装检查) → [Interface输入输出](#interface-的数据输入输出与所有权) → [driver伪代码](#driver-主流程伪代码) → [正式阶段顺序](#分阶段正式重构顺序)。前半部分保留当前运行、数学与源码依赖的核对依据。

建议采用**顶层按职责分层、计算实现按后端组织、后端内部按阶段和曲线组织**的混合结构。曲线身份和数学合同集中在核心模块；CPU、CUDA、OpenCL 分别拥有自己的资源、数据布局和执行策略。CUDA 主机代码、设备代码和实例文件全部进入 `src/`，不再保留单独的 `kernels/cuda/` 实现树。`include/` 与 `src/` 使用对应的模块路径，前者只保留其他模块需要知道的接口。

降低复杂度的重点是提取有完整职责的 Module，而不只是把大文件拆小：任务执行、完成与恢复、后端执行、存档编解码和 GUI worker 编排各自形成可独立理解和验证的模块。本文用 Interface 表示完整使用合同，用 Seam 表示可替换接口所在位置，用 Adapter 表示具体实现；接口要隐藏状态、顺序和资源管理，形成有实际 Depth 的模块。

## 分析基准与范围

前期粒度复核开始时的工作区 HEAD 为 `8017ac48e0962da149b269f3bb022ff438b8c20c`，分析包含未提交文件，当时读取的版本头为 `0.44.6`。各节行号和版本对应各自的读取快照，不能把分析等同于干净提交或某个已验证二进制。结果写入器的并行更新已核对；旧源码摘要保留在 [source_snapshot.json](../../data/experiments/main_flow_granularity_20261021/source_snapshot.json)。整合方案另有本轮快照，见“完整重构目标与实施约束”。这些摘要用于识别变化，不是生产构建清单。

纳入 Stage1 的 `ecm`、`ecm_cuda`，独立 CUDA Stage2、独立 CPU Stage2，以及 Windows GUI 的生产调用。构建清单只作为确认生产源文件和依赖的证据。测试程序、基准工具、实验引擎、Android 工程和第三方内部重构不属于本次设计范围。

生产运行必需的算术检查、自检、错误检查仍属于主流程；不能按文件名包含 `test` 或 `selftest` 就排除。例如 OpenCL Stage1 编入并调用生产自检，见 [构建入口](../../CMakeLists.txt) 第 158–167 行及 [opencl_ecm_stage1.cpp](../../src/opencl_ecm_stage1.cpp) 第 929 行。

本文件维护分析、目标方案与获授权后的分阶段实施记录，不将拟议目录写成已实现状态。实施验收采用 local build，不运行 release；已知沙箱编译限制通过沙箱外执行同一正式构建脚本处理，参数保持不变。其他预期外重大问题仍停止，不以修改算术或验收门限绕过。阶段提交使用 `[0.45.z]` 前缀；P0 基线保留 `0.44.9` 身份，P1a 程序版本为 `0.45.1`；`[0.45.0]` 为基线文档提交前缀，不是重新命名基线程序，也不产生性能提升结论。当前实现的权威入口仍为 [架构与术语](../architecture/OVERVIEW.md)、[Stage1](../architecture/STAGE1.md)、[Stage2](../architecture/STAGE2.md)、[Save](../architecture/SAVE.md) 和 [Checkpoint](../architecture/CHECKPOINT.md)。已有 [CUDA 结构草稿](GPU_CUDA_STRUCTURE_REFACTOR_DRAFT.md) 描述其实施范围。

## 当前主流程

### Stage1

`ecm` 与 `ecm_cuda` 使用同一组 Stage1 驱动、CPU 计算、队列和存档源文件，链接不同的 GPU Adapter。共享源清单位于 [cuda_sources.cmake](../../tools/build/internal/cuda_sources.cmake) 第 5–27 行；这里的名字已经不能准确表达内容。

主干流程为：解析 CLI 与配置 → 选择直接输入或 worktodo → 形成待分解整数和曲线批次 → 选择 CPU 或 GPU 执行路径 → 准备有效标量及恢复状态 → 执行曲线 → 校验因子与完成点 → 写 checkpoint/save/结果 → 按任务规则交接或推进队列。CPU 的线程与 SIMD 批处理、CUDA Edwards 和 GPU Montgomery 在同一个驱动中分支。

直接输入与队列共用 `run_stage1_once`：`main` 在驱动第 3969 行，队列入口 `run_queue_manager` 在第 3465 行，队列任务经 `queue_run_one`（第 3201 行）进入 `run_stage1_once`（第 3030 行），直接 CLI 也在第 4551 行调用它。真正的执行分派在 `run_stage1_impl`（第 2753 行）。这些是现成的 Seam 起点，但当前参数、结果所有权和文件发布仍未收口。

[ecm_driver.cpp](../../src/core/ecm_driver.cpp) 共 4676 行；它既是应用入口，也承担批次调度、结果发布和多种实现的运行控制。CUDA Montgomery 的 [montgomery.cu](../../src/ecm/backend/cuda/stage1/montgomery/adapter.cu) 仍有 1705 行，并直接包含 checkpoint、进度、日志及文件系统相关设施（第 45–58 行），说明后端内部仍承载部分上层职责。

### CUDA Stage2

独立程序读取 Stage1 save 或 Stage2 队列，处理配置、Auto B2 与调优、进度和回执，再调用 CUDA Stage2 引擎。生产引擎已经分成 context、plan、engine、batch、points、product、tree、remainder、s4、factors 等实现文件，不能把它描述成尚未拆分的单个设备程序。

当前有父子进程分工：父进程 `driver`（main 文件第 1152 行）选择、规划和恢复任务，经 `child_run`（第 617 行）创建每曲线子进程（第 701 行）；子进程 `curve_worker`（第 898 行）重读原 save 并校验 fingerprint，再在第 967 行调用计算引擎。父进程根据持久化 receipt 确认完成，而不是只根据子进程退出码。重构后继续保留该隔离方式。

[ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp) 共 1606 行，并结合 [ecm_stage2_portable_driver.inl](../../src/core/ecm_stage2_portable_driver.inl) 使用运行编排。引擎接口位于 [ecm_cuda_stage2.h](../../src/core/ecm_cuda_stage2.h)，执行入口位于 [engine.cpp](../../src/ecm/backend/cuda/stage2/engine.cpp) 第 929 行。

引擎中的主干计算是：校验模数与输入点 → 规划 D、形状及内存 → baby 点与 F 树 → giant 点与 G 树 → 乘积、模 F 累积和下降 → GCD 与命中处理 → 完成异步算术校验 → 返回结果。规划与执行已经复用 `prepare_stage2_plan`；规划仍会初始化 CUDA 并查询实时内存，不能称为完全无设备副作用的纯计算。

### CPU Stage2 与 GUI

[ecm_cpu_stage2_main.cpp](../../src/core/ecm_cpu_stage2_main.cpp) 是独立的 Edwards save 续算程序，可使用 libecm NTT/plain 或可选 GW 多项式实现。它没有与 CUDA Stage2 等同的队列和调优流程，未来应共享有一致语义的请求与结果，保留自己的执行方式。

GUI 通过 worker 子进程使用计算程序。已有 [worker_proc.cpp](../../src/gui/worker_proc.cpp) 负责进程操作，但 [app.cpp](../../src/gui/app.cpp) 仍集中编排、状态维护、日志与结果处理以及绘制。重构应先提取 worker 编排职责，继续使用现有进程执行模式。

## Stage1 实际运行顺序与后端行为核对

本节核对当前实现，区别于后文拟议的重构结构。核对开始时 HEAD 为 2e64b75e5d14287e2ee25bdaff27e3b1128c1d93，版本头为 0.44.6；源码身份见 [运行流程源摘要](../../data/experiments/stage1_runtime_flow_20261021/source_snapshot.json)。细粒度流程以生产可达代码为准，未运行计算程序，也未修改实现。

### 行为标记

- **主机计算**：CPU 上的整数、曲线、NAF、GCD、验证、归一化和调度。CPU IFMA 仍属于主机计算。
- **GPU 计算**：设备内核的字典预计算、点乘、域转换等。
- **设备资源**：设备选择、上下文、buffer/event 分配和释放；调用 runtime 不等于执行点乘。
- **设备传输**：H2D 上传与 D2H 回读。它不属于输出或文件 I/O。
- **同步等待**：主机等待设备工作完成、检查错误或读取计时；等待完成不意味着已回读所有点。
- **输出 I/O**：stdout/stderr、GUI pipe、控制台进度和镜像日志。镜像日志本身还会产生文件写入。
- **文件 I/O**：TOML、worktodo、指数/内核缓存、checkpoint、dump、save、ledger、JSON 结果、finished 和交接文件。

以下各表是已发生行为的顺序描述，不要求将每一行为拆成独立 Module。

### 对用户流程的逐项修正

| 原理解 | 当前实现的修正或补充 |
| --- | --- |
| 默认值 → 配置文件 → 命令行，后者统一覆盖 | 不能作为所有模式的现状。先解析 CLI 并分流；队列重新读取 TOML，普通输入运行主要用 CLI/内建值，只另取 TOML 的结果输出设置。gpu-info 才显式实现部分 CLI 覆盖 TOML 的查询规则。 |
| 一次检查所有合法性和路径 | 检查分散在 CLI、配置解析、任务接纳、输出账本、后端能力、checkpoint 恢复及实际打开文件时；部分输出目录会创建，不要求启动时全部存在。 |
| 输入运行 / worktodo | 基本正确，但 version/help、kernel 列表和 gpu-info 是提前返回的查询分支。有位置 B1/B2 进入 stdin 输入运行；无位置参数进入队列。 |
| 读取 worktodo，再根据配置选择任务 | 还要限定 worker 段、priority、skip 与消费阶段，区别空队列、其他阶段应执行和 I/O 错误；不能把读取失败当完成。 |
| 根据配置和 worktodo 选择 CPU/device | CPU/GPU 由任务使用的配置确定，带类型的 save 名还可覆盖曲线/param；GPU 的 CUDA/OpenCL 由可执行程序链接选择，不是每个任务动态加载。设备编号是另一项选择。 |
| 初始化曲线结构体，再取得 S | 只有部分路径符合。任务容器先打开；各后端分别建立结果数组、运行状态和曲线点，具体恢复/构造/标量准备次序不同。不存在共用的一份完整曲线运行结构体。 |
| Ed 计算 NAF，再寻找 checkpoint | NAF 不是所有 Edwards 路径必经；恢复载荷与字典、NAF 的相对顺序也不同，见后端细表。 |
| 每次获得当前 s/进度 | K 通常是不可变大整数；循环消费它的 bit、NAF column 或 PRAC descriptor，另维护进度游标，而不是每片重新计算 K。 |
| 发送 GPU batch kernel，阻塞取结果 | 只适用于 GPU。常规 GPU 路径按多 bit/column 分片并等待；CUDA Montgomery/OpenCL 普通片不回读曲线点，CUDA Edwards 每片回读。CPU 用线程和 scalar/IFMA 循环。 |
| I/O 输出进度 | 是独立于点计算的输出行为，按节奏触发；checkpoint 是另一个文件分支。控制台/pipe 与日志文件节流也可能不同。 |
| 完成后得到 a^s mod N | 应为曲线点 Q=[K]P，所有坐标算术模 N。后处理可能得到规范化 x、非平凡因子、饱和状态或错误，不总有可续算点。 |
| 最后逐曲线 GCD | 是重要收尾，但 Edwards 的构造、字典逆元和恢复也可能更早提取因子；有的后端先尝试求逆，失败才做 GCD。 |
| 每曲线拆分合数因子 | 当前没有统一的素因子分解步骤；有效 GCD 可以是复合数，不能把它标成已证明素因子。 |
| 汇总因子去重，再验证整除 | 结果按各曲线身份保留，并逐候选校验范围和整除。ledger 去重发布身份；param3 save 另按不同因子值剥除余因子表达式。不是统一跨曲线去重后才验证。 |
| 可选 save，再可选逐曲线群阶 | save 是否可省略因路径而异，不能统一称为可选；群阶是直接 CLI 的可选诊断，在统一结果发布完成之后，而且只对命中的 probable-prime 因子继续。不是所有曲线/后端的统一步骤。 |
| 输出内部逐曲线 data、result、合并 result | 没有必经的独立逐曲线 data 文件。正式结果是逐曲线 submission ledger、内部结果批次、submission seal；坐标 dump 是可选调试文件，完成点属于 save。 |
| 最后改写 worktodo、发送 save | 还需区分停止、失败、命中与无因子；完成后先暂存必要交接记录，再写 finished、推进源队列、同步 save、flush 交接。所谓发送是文件协议，不是 GPU 行为。 |

### 数学公式与数据表示

本文用 K 表示源码中的 s/batch_s，避免与架构文档中表示算术位数的 S 混淆。对有效整数界限 B，定义：

$$
L(B)=\operatorname{lcm}(1,\ldots,B)
=\prod_{\ell\ \mathrm{prime},\,\ell\le B}
\ell^{\max\{e:\ell^e\le B\}},
\qquad
K=
\begin{cases}
tL(B),\quad t\in\{1,12\}, & \text{Montgomery},\\
48L(B), & \text{Edwards}.
\end{cases}
$$

正常输入按整数 B1 理解；公共 compute_batch_s 的实际转换是 floor(B1+0.0001)。Edwards CPU 先缓存/构造 L 再乘 48，CUDA Edwards 直接请求 torsion=48 的缓存条目；有效 K 相同，不代表缓存文件身份相同。指数缓存不保存曲线、NAF 或运行进度，也不依赖 N/sigma。

Stage1 计算是：

$$
Q=[K]P,\qquad\text{曲线坐标运算模 }N,
$$

不能写成普通整数模幂。Montgomery x-only 结果是 (X:Z)，在 gcd(Z,N)=1 时才有：

$$
x=XZ^{-1}\bmod N.
$$

若不可逆则检查 g=gcd(Z,N)：g=1 表示此项没有提取到因子；1<g<N 且 g 整除 N 才是有效非平凡因子；g=N 是饱和，处理随当前后端而异。逆元是满足 ZZ^{-1}=1 mod N 的值，合数 N 下不能用 Z^(N−2) 解释。CUDA findfactor 第 227 行调用 mpz_invert 是正确操作，其邻近模幂注释不能作为数学依据。

a=1 的 Edwards 扩展点满足：

$$
XY=ZT,\qquad X^2+Y^2=Z^2+d_E T^2\pmod N.
$$

当前 Stage1 完成点映射到 Montgomery x-only 状态：

$$
(U:W)=(Z+Y:Z-Y),\qquad
g=\gcd(W,N),\qquad
x_M=UW^{-1}\bmod N\quad(g=1).
$$

因此不是直接对 Edwards 的原始 Z 求最终 GCD；恢复或字典过程中对 Z 求 GCD 是另一项检查。Edwards 的 d_E、Montgomery 的 a24=(A+2)/4、param3 的整数种子不是同一变量。

当前三种 Montgomery 构造应分别解释：

- **param0**：u=σ²−5，v=4σ，A=(v−u)³(3u+v)/(4u³v)−2，a24=(A+2)/4，P=(u³:v³)。
- **param3**：δ_i=(σ+i)·2^(-32) mod N，a24=δ_i，A=4δ_i−2，P=(2:1)，2P=(9:64δ_i+8)。kernel 的 32 位整数 d=σ+i 通过特殊乘法实现 δ_i 的作用，并不等于普通域 δ_i。
- **param2**：在辅助曲线 y²=x³+36 上取 (u,v) 为点 (-3,3) 的 σ 倍，令 x3=(3u+v+6)/(2(v−3))，再取 A=−(3x3⁴+6x3²−1)/(4x3³)，a24=(A+2)/4，正式起点 P=(2:1)。这里 σ 倍点属于构造，不是 Stage1 的 K 倍点。当前代码对 σ<2 有调整，不能按任意种子直接套用公式。

上述分式都要求分母为单位。Edwards checked construction 会分类返回因子或退化；当前 Montgomery param0/2 的部分构造逆元失败分支只是警告或置零继续，不能把数学上的异常检测自动写成所有实现已具备的行为。

NAF 是 K=Σ d_j·2^j 的有符号表示，非零 d_j 为奇数且 |d_j|<2^(w−1)，字典含 P、3P、…、(2^(w−1)−1)P，共 2^(w−2) 项。NAF 列数可能比 K 的二进制位数多一列；bit、NAF column、PRAC prime-record 与完成曲线数必须分开记录。

### 启动 配置与队列接纳

1. **主机控制、输出 I/O**：检查帮助/版本请求，建立内建值，解析 CLI；部分查询在这里或随后直接返回。配置及路径错误并非都发生在此时。
2. **主机控制**：根据是否存在位置 B1/B2 选择模式。普通输入路径读 stdin 到 EOF，再解释 N；队列路径不读这个 stdin 数字。见 driver 第 3969、4331–4332、4406–4437 行。
3. **文件 I/O、主机计算**：队列构造 EcmQueueConfig，读取全局和 worker 覆盖；缺失默认配置时写模板，存在却非法/不可读则失败。默认值、全局 stage/shared 表、workers.N 覆盖由配置读取器组合。见第 3479–3493 行及 ecm_queue_config.cpp。
4. **配置差异**：队列执行通过 run_queue_manager(ini_path,worker) 重建 opt，直接取 cfg 的曲线、CPU/GPU、设备、实现、窗口等（第 3668–3721 行）；普通 CLI 的同名执行参数没有在此统一重放覆盖。普通输入运行在第 4531–4534 行只另外装入 TOML 结果输出设置；不能写成全字段统一的“CLI 最后覆盖”。gpu-info 的部分覆盖规则见第 4295–4322 行。运行时单例上的 CLI 开关又是另一条路径，需按字段说明。
5. **输出 I/O、文件 I/O**：队列打开镜像日志，解析任务/结果/同步/交接路径，恢复先前 pending，做启动 save 同步。输出配置、缓存、恢复和日志可以在第一条曲线之前已经读写文件。见第 3506–3521、3590–3595、3755–3758 行。
6. **文件 I/O、主机计算**：每轮先检查停止，再按 worker、priority、skip 和 Stage1 consumer 读取任务；读取错误退出，无可执行任务结束或交给其他阶段。解析 ECM/ECM2/ECMSTAGE1，计算实际余因子 N=(k·b^n+c)/F_known，并核对已知因子整除、B1、曲线数量、sigma、曲线/param 和存档身份。带类型的 save 名可改变当前曲线选择。见第 3763–3808、3916–3929 行及 ecm_worktodo.cpp 第 577–607 行。
7. **运行方式差异**：CPU/GPU 是执行目标，CUDA/OpenCL 是二进制中链接的 GPU Adapter。Stage1 队列不会直接调用本程序中的 Stage2 主循环；Stage2 是独立程序，交接及 GUI 进程切换是另一条流程。

### 共同任务外壳先于具体后端

run_stage1_once 的真实顺序是：验证结果元数据与路径、已有账本所有权 → 确定任务 checkpoint 路径并建立 batch identity/lock → 读取 batch 主/备容器与记录 → 必要时复制已有 save 历史 → 进入具体 Adapter → 统一结果发布 → 成功结束 batch。见 driver 第 3047–3099 行，checkpoint runtime 第 205–250 行。

这意味着**任务容器的装载早于 K 构造，具体曲线载荷的解释通常晚于 K**。生产 curve_record 是 batch 内的逻辑记录；read_blob 可直接读取已载入内存，write_blob 更新记录并重发 batch 容器，并非每次都独立读写一份逐曲线物理文件（runtime 第 286–299 行）。

### CPU Montgomery 标量与 IFMA

1. **主机计算、文件 I/O**：取得 K=tL；校验指数缓存，未命中才筛素数和乘积树构造，按条件尝试缓存回写。缓存写失败可以仅降低复用能力，不等于 K 构造失败。见 driver 第 2179–2193 行。
2. **主机计算**：选 scalar/IFMA，按 ISA 与配置判定；K 一次展开为 MSB-first bits。随后逐曲线解释恢复记录，采用 checkpoint 的真实 sigma，识别 DONE/INFLIGHT，并按恢复位置分组。见第 2221–2358 行。
3. **主机计算**：scalar worker 一次处理一曲线；IFMA 一组最多八条且起始 bit 相同，不足 lane 可填充但只消费有效 lane。构造/重建 Suyama 参数，载入普通域 checkpoint，转入所选算术表示。IFMA 可用 REDC 或精确 Mersenne fold，均无 GPU 行为。
4. **主机计算、输出 I/O**：二进制 ladder 消费 bit，维护相邻倍点；在 chunk 安全点转出状态并回调进度/停止。这里没有 launch、H2D、D2H 或设备同步。
5. **文件 I/O**：callback 请求暂停后，ladder 返回 PAUSED，外层再写 INFLIGHT；若只是定时保存，随后从新 start 重新调用 ladder，若停止则不进入正常完成。见第 2470–2494、2576–2595 行。
6. **主机计算**：完成后取 (X:Z)，求 GCD 并尝试归一化；hit 时**文件 I/O**即时发布因子，再写 DONE。最后 join worker，汇集 save 并返回。见第 2503–2526、2612–2638、2697–2728 行。
7. **当前限制**：最终 g=N 在 CPU Montgomery 不作为 proper factor，也无统一饱和细分；scalar 在不可逆时退回 X，IFMA 退回 0。不能把所有 miss 的输出都称为已验证可供 Stage2 续算的 affine x。

### CPU Edwards 标量与 IFMA

1. **主机计算、文件 I/O**：取得 L 再乘 48，建立 per-curve 结果和 save_lines，生成或恢复 sigma；DONE 曲线可直接取得既有结果。
2. **scalar 主机计算**：读取 INFLIGHT → 构造并检查 Atkin–Morain 曲线/起点 → 建算术上下文及奇数倍字典并批量归一化 → 构造 NAF → 初始化或恢复累计点 → 按 NAF 列倍点/加点。构造或字典求逆期间已经可能找到因子。见 ecm_edwards_cpu.cpp 第 370–476、606–643 行。
3. **IFMA 主机计算**：整组 DONE 检查 → 逐 lane 构造预检 → 建 SoA 曲线与字典 → 才载入各 lane INFLIGHT → 同起始位置恢复，否则按当前规则重启批次 → 构造共享 NAF 并运行。构造/字典异常可回退 scalar 分类；尾部一条也走 scalar。见 driver 第 1415–1565、1584–1612 行。
4. **输出 I/O、文件 I/O**：在 callback 安全点输出进度并直接写 checkpoint，区别于 CPU Montgomery 的返回 PAUSED 后再写。scalar chunk 随 K 位长调整，IFMA 每 16384 个 digit 前检查一次；短 IFMA 标量可能没有批内 callback。不能声称所有长度都按固定墙钟周期响应停止。
5. **主机计算**：最终映射 (U:W)=(Z+Y:Z−Y)，检查 GCD；无因子且 W 可逆才生成规范化 save 点。最终 g=N 清空该条点记录，不向 Stage2 输出这个恒等元。
6. **文件 I/O**：新算 scalar 曲线是规范化/捕获 save_line → DONE → 即时因子发布；IFMA 是即时因子发布 → 捕获 save_line → DONE。全任务完成后才合并发布共享 save。恢复 DONE 的路径可由最终统一 publication 补齐结果，不能套用新算路径每一步。
7. **例外**：正常 scalar/IFMA 是 NAF；scalar 超出固定 MPN 容量会回退 GMP binary double-and-add，该分支没有相同的 checkpoint 回调/恢复能力。字典逆元失败的逐分母 GCD 是异常因子提取，不是对复合因子做素因子分解。

### CUDA Montgomery 默认 ladder

1. **主机计算、文件 I/O**：取得 K，必要时导入兼容 checkpoint。**设备资源**：prepare 设备与上下文，再核对 save 可写、分配 host 结果并选 sigma；进入 CUDA entry。见 driver 第 2817–2909 行。
2. **设备资源、H2D**：导出 K words，创建 event，分配并上传 K；之后才解释具体曲线 checkpoint。无恢复时选已编译 tier，主机/GMP 构造 param0/2/3 的 P、2P 与常数；恢复时采用保存的 sigma、布局、tier 和进度。见 montgomery.cu 第 1088–1112、1166–1328 行。
3. **设备资源、H2D**：上传整批曲线状态。普通 ladder 的 global buffer 在片边界为普通 residue，内核每片进入/退出 Montgomery 域；fold 构建是普通域 Mersenne 运算的专门路径，限制为实际支持的 param0/2。
4. **主机控制 → GPU 计算 → 同步等待**：选一个含多 bit 的 slice → launch → cudaDeviceSynchronize 并检查 CGBN error → event 计时。初片约 200 bit，后续以约 100 ms 为目标调整，不是每 bit launch。见第 1478–1485、1523–1557 行。
5. **输出 I/O**：按完成 bit、时间计算速度与 ETA 并输出；普通片**没有曲线点 D2H、没有 GCD、没有文件写入的必经步骤**。见第 1559–1613 行。
6. **可选 D2H、文件 I/O**：dump 开启时在片前/后回读并写 CSV；checkpoint 到期、停止或最后一片时 D2H 整批普通域点 → 主机 header/checksum → 写 checkpoint。停止则返回未完成，不能继续正常 save/result。
7. **完成 D2H、主机计算**：末片 checkpoint 后当前实现还会再次 D2H 做 process_results；校验模数、逐曲线取 (X:Z)。Z 可逆时直接求 affine x；否则 GCD，proper factor 才命中，g=N 令该批错误并拒绝最终 save。见第 1661–1665 行与 findfactor 第 222–237 行。
8. **设备资源、文件 I/O**：释放 buffer/report/event，返回 driver；逐命中即时因子发布，再写最终 save，最后进入统一结果发布。kernel 不计算宿主 GMP 的最终 GCD。
9. **remove 调用与实际清理不同**：后端收尾虽调用 checkpoint remove，生产 active batch 的逻辑记录会被 runtime 第 154–156 行保护；真正删除任务容器发生在两路结果与 seal 成功后的 end_batch(true)。

### CUDA Montgomery resident 与 PRAC

这些是受编译能力和选项约束的路径，不是 param0/2/3 的统一替换。当前仅可选实例、param0、非 fold 等满足条件时接纳。

- **主机计算、文件 I/O**：resident 仍按 K bits；PRAC 额外取得按 prime/repetition/chain 描述的计划及缓存，K 仍参与身份校验，不能把 PRAC 进度叫 bit 或 NAF column。
- **恢复、H2D、GPU 计算**：恢复载荷绑定算法、域、计划、K/N 身份等；新曲线上传后先 GPU INIT 进入 Montgomery 域，恢复态已在该域则跳过 INIT。
- **GPU 计算、同步等待**：resident 按 bit、PRAC 按 prime-record 分片；每片等待并检查错误。普通片不回读点，片边界继续保留 Montgomery 域状态。
- **D2H、文件 I/O**：checkpoint 保存 Montgomery 域载荷，不先 EXPORT。
- **完成 GPU 计算 → 同步 → D2H → 主机计算**：显式 EXPORT 到普通域，再复用最终结果处理。PRAC_WINDOW/采样限额等诊断路径返回未完成，不应画入成功的 Stage1 主线。

依据：montgomery.cu 第 1007–1032 行，prac_runtime.inl 第 132–198、241–295 行。点乘策略可有不同执行轨迹，但不能改变相同构造及有效 K 的数学身份。

### CUDA Edwards

1. **主机计算、文件 I/O**：取得 K=48L，先分配 host Batch 并选择算法，再探测/恢复载荷；新曲线在主机 checked construction，可能先找到因子或退化。构造命中使用占位 lane 维持批次布局，因子不是在构造现场即时发布。
2. **算法**：缺省 NAF、自动窗口 6；可选 binary。恢复必须核对窗口/算法/倍点图和标量身份；done 是 MSB-first binary bit 或 NAF column，不能相互解释。
3. **主机计算、设备资源、H2D**：选容器与 device，正式 NAF plan 在 API 内于恢复后建立；打包状态/位平面并分配上传。**GPU 计算、同步**：NAF 额外构建奇数倍字典，准备完成回调输出初始化/字典时间。完全完成的合法 NAF 恢复态可跳过设备运行。
4. **GPU 计算 → 同步等待 → D2H**：每片运行若干 bit/column，等待并检查错误，**每片都取回 Q 的四个普通坐标**。binary 取回整份 data 后提取 Q；NAF 的设备累计点和字典保留算术域，只回传 publication 坐标。
5. **输出 I/O、文件 I/O**：回调更新进度；首次、停止、末片或约 200 ms 时输出；checkpoint 到期/停止则将已回读状态写盘，无须为该次 checkpoint 再做一轮 D2H。失败或停止不进入正常完成。
6. **文件 I/O、主机计算**：完成态先保留 checkpoint，再逐曲线检查扩展点 → (Z+Y:Z−Y) → GCD → 归一化或命中。g=N 不产生有效 Stage2 点，该条 save 跳过。
7. **文件 I/O**：构造命中和最终命中在逐曲线收尾时发布；累积 text save 后原子发布整批，再返回统一结果层。移除逻辑 checkpoint 的调用同样受 active batch 保护。

依据：ecm_edwards_gpu_driver.inl 第 85–199 行，edwards_runtime.inl 第 31–114 行，edwards_naf.cpp 第 10–18 行，device 的 instances.cuh 与 naf.cuh。

### OpenCL Montgomery param3

1. **主机计算、文件 I/O、设备资源**：共用驱动先取得 K、prepare OpenCL、检查 save 并选 sigma；后端打包 K → 读取 checkpoint → 建立/复用 context → 路径与程序缓存/编译 → 新运行执行所需自检 → 恢复或主机构造 P/2P。仅支持 param3，不创建 Edwards NAF。
2. **设备资源、H2D、主机计算**：分配并上传 K/data；新状态经主机转换 Montgomery 域后上传，恢复态已经是该域。这里的主机域转换不是 GPU kernel。
3. **GPU 计算 → 同步等待**：初片约 10 bit，随后按约 80–120 ms 调整；每片 enqueue kernel 后 clFinish。普通片不回读点。
4. **可选 D2H、文件 I/O**：checkpoint 到期则阻塞下载 raw Montgomery 状态并写盘，然后进度输出/检查停止。sync_each_batch 或 dump 开启时，每片回读并在主机转普通域；dump 另写文件，非末片重新转域并上传。
5. **停止**：片完成后同步、下载状态、写 checkpoint、释放资源并返回错误/未完成，不进入正常 GCD/save/result。
6. **完成 D2H、主机计算**：回读并由主机转普通域；逐曲线先尝试 Z 逆元，失败求 GCD；可选 verify 另作整除与一致性核对。正常主流程最终由统一 submission 校验严格的 proper factor 范围。
7. **当前差异**：OpenCL findfactor 对逆元失败直接标记命中，并未像 CUDA Montgomery 在此拒绝 g=N；后续即时 publication 的范围校验会拒绝 N 本身。不能把两后端的饱和分支描述成完全一致。
8. **资源释放、文件 I/O**：返回 driver 后即时发布因子、可选 text save，再统一结果。后端 remove 调用不等于提前删除 active batch 容器。

依据：opencl_ecm_stage1.cpp 第 380–475、787–1028、1056–1297 行；opencl_backend_glue.cpp 第 22–26、57–89 行。OpenCL sigma 上界存在前述待核对差异，不在文档中替实现确定新规则。

### 最终结果与队列交接

必须区分五类文件：指数缓存只复用标量；checkpoint 保存中途/已完成恢复事实；text save 提供完成点或因子；内部结果保存逐曲线计算身份；submission 合并已完成的无因子覆盖并即时保留因子。dump 是另一个可选调试文件，不等于正式逐曲线 data。

save 的生成条件也不同：普通输入的 GPU Montgomery 仅在给出 save 路径时写最终 text save；队列 ECM/ECM2 会为 GPU 生成确定的路径。CPU Montgomery 在使用 checkpoint/tmp_dir 或显式 save 时写共享 save；CPU Edwards 和 CUDA Edwards 在未给文件名时自动命名并发布。因此不能把“可选 save”写成所有运行方式共同的开关。见 driver 第 2709–2716、2856–2865、2925–2973、3823–3838 行，Edwards 收尾第 1969–1978 行与 GPU Edwards 第 184–189 行。

Adapter 正常返回后，run_stage1_once 的实际顺序为：

1. **主机计算**：核对 per-curve sigma/factor/status 已齐备，形成任务及曲线发布身份；逐项校验元数据与因子。
2. **文件 I/O**：逐曲线 publish_submission(...,false)，写 submission 的不可变曲线账本；命中 F 可以马上追加到外部输出，NF 先留账本。
3. **文件 I/O**：publish_internal_batch，一次写内部逐曲线账本并输出内部 JSONL。
4. **文件 I/O**：seal_submissions，将符合分组规则的已完成 NF 曲线形成批次、重新计算其字段/校验和并追加到发布输出；F 不混入 NF 计数。
5. **文件 I/O、输出 I/O**：end_batch(true) 清理任务 checkpoint 并解锁，打印结果文件位置。若前面失败，end_batch(false) 保留恢复条件。见 driver 第 3081–3104 行与 submission_writer 第 184–240、262–293、320–336 行。

去重不是对所有数学因子值做一次 set：submission 身份包含 N、曲线族/param、sigma、B1、阶段及因子；不同曲线相同因子仍保留曲线来源。param3 save 的不同因子值排序/依次整除仅用于构造剩余 N 表达式；param0/2 保留原 N。参见 ecm_save.cpp 第 106–150 行。

**队列成功路径**：打印统一 hit 信息 → 无因子且需要外部 Stage2 时先 park pending → 写 finished → 无因子且同队列自动 Stage2 时替换为 Stage2 任务，否则移除源任务 → 同步 save → flush 外部 Stage2 pending → 按已接纳构造处理 Prime95 交接 → 读取下一条。命中会结束这项搜索，不把命中曲线作为干净 Stage2 点发送。Prime95 的支持范围与本程序 Stage2 交接不完全相同，不应共用“所有后端都可发送”假设。见 driver 第 3287–3376 行。

**失败/停止路径**：准备或发布失败保留 worktodo 并终止当前队列运行；停止保留队列和恢复条件；其他被归类的执行失败可把任务标为 ERROR。文件失败、算术失败和没有因子不是同一结果。交接失败按对应协议保留 pending，不回滚已完成的曲线。

### 群阶诊断与最小界限公式

群阶不在 Stage1 成功提交的必经步骤中。直接 CLI 的 --go 先检查 PARI/gp 可用性，但实际命中诊断在 run_stage1_once 已发布结果之后。只对命中的 probable-prime 因子 p 调用 gp 计算 #E(F_p) 并分解该群阶，复合因子直接跳过；这不是在分解 Stage1 找到的合数因子。见 driver 第 4353–4363、4551–4609 行。

不能把 #E(F_p) 与起始点阶 r_p=ord(P mod p) 混为一谈。理想非奇异情形下：

$$
r_p\mid \#E(\mathbb F_p),\qquad
[K]P=O\iff r_p\mid K.
$$

若 K=tL(B1)，令 r_eff=r_p/gcd(r_p,t)。忽略其他模分量同时饱和、表示退化及输入范围限制时，使该点被 Stage1 消去的最小整数界限为：

$$
B_{1,\mathrm{point}}=
\max_{\ell\mid r_{\mathrm{eff}}}
\ell^{v_\ell(r_{\mathrm{eff}})},
$$

空集合取 1，再受程序实际允许的 B1 下界约束。用群阶代替点阶只能给出充分的消去界限，未必是该点严格最小值；遗漏 t=12/48 也会高估所需界限。

理想单大素数 Stage2 情形，应检查：

$$
m_p=\frac{r_p}{\gcd(r_p,K)},\qquad
m_p=\ell\text{ 为素数且 }B1<\ell\le B2.
$$

这表达候选数学条件，不取代具体 Stage2 覆盖、检查和非平凡 GCD 验证。不存在只按“群阶最大的素数幂”就能统一保证的实际最小 B1/B2 公式。

当前 go_min_bounds 使用群阶因子，而且把“最大素数幂对应的底素数”作为 largest_prime 继续计算（driver 第 4610–4660 行）；它不能被文档背书为严格最小界限。该诊断还使用 32 位 firstsigma+i 和 Montgomery 族重建 helper；GPU Edwards 明确拒绝 --go，但 CPU Edwards、非连续或 64 位 sigma 的适用性仍需专门核对。这里修正数学解释并列入待验证事项，不修改当前代码或输出字段。

## 非 GPU 修改的编译隔离与 driver 顺序编排

本节为可行性评估与拟议设计，尚未实施。用户要求的是：配置、主机曲线准备、存档和展示的修改不应触发无关 CUDA 编译；**设备同步、回读、上传、资源和内核调度继续属于 CUDA Adapter，不要求脱离 CUDA runtime，也不要求全部改成 .cpp。** 这里的普通代码指项目现有的 C++17/.cpp，不是改写为 C。主机对象变化后仍需重新链接程序。

源码核对基于 HEAD 2e64b75e5d14287e2ee25bdaff27e3b1128c1d93 加工作区。证据见 [主机与 CUDA 依赖源摘要](../../data/experiments/host_cuda_dependency_20261021/source_snapshot.json)：其中只读扫描了四个 CUDA 入口的字面本地 include 闭包，跨预处理分支保守处理，未读取完整 SDK/compiler depfile 闭包；因此它不是生产对象复用证明。本轮未启动构建，未验证某次实际构建日志的失效原因，也没有编译耗时结论。

### 结论：可实现，但职责隔离和构建策略必须同时成立

以下常见改动可以做到：仅编译受影响的普通 C++ 源文件与其主机依赖，再链接；若 .cu 的依赖、调用合同、编译参数和工具链保持不变，则复用已验证的 CUDA 对象。不能承诺任意公共头修改都免 CUDA 重编。

| 用户关心的功能 | 可独立为普通 C++ 的 Implementation | 继续留在 CUDA Adapter 的部分 | 可复用 CUDA 对象的条件 |
| --- | --- | --- | --- |
| 配置文件读写 | TOML 解析与写入、默认值、值域与组合校验、路径解释；把配置转换为明确的执行选项 | 设备能力查询与 runtime 配置执行；后端自己校验所选能力 | 解析和描述头不再进入 .cu；只改配置文件内容本来就不需要编译。新增设备编译期开关或改变共有执行合同仍会影响对应 .cu。 |
| 初始化曲线及 CPU 所需数据 | 基于 GMP 的曲线构造、sigma 处理、K 与缓存、bits/NAF 编码、CPU 字典、GCD 与规范化 | GPU 字典预计算、设备初始化、设备专用布局打包和传输；可保留现有 runtime Implementation | 主机构造函数通过声明调用，Implementation 在 .cpp；修改计算步骤不改变设备布局/公式合同。不能把所有 GPU 预计算也称为 CPU 初始化。 |
| checkpoint | identity、接纳与校验、容器和载荷 codec、校验和、备份与文件发布 | 设备安全点、必要的同步与 D2H、从设备布局提取恢复状态 | writer 接收已经有效的主机快照；格式和文件策略不内联进 .cu。不改变恢复位置含义、曲线身份或设备快照合同。 |
| save | 完成点验证、曲线格式转换、命名、编码、追加/替换、兼容写出 | 取得完成点所需的设备同步与回读 | save writer 只认识曲线身份和主机点，不认识 CUDA handle；改变文件格式不改变点与执行 Interface。 |
| data / result | 逐曲线记录、验证、ledger、去重、submission/internal 编码与 seal | 后端提供已分类结果、有效点和性能事实；即时提交时机仍在安全点 | 格式、路径和发布策略藏在 .cpp；可靠提交的失败仍能反馈给执行方。 |
| tune.toml | profile 解析、合并、选择策略、记录编码、身份检查与文件发布 | 调优测量、真实内存查询、event 计时、运行 GPU 算子 | 改字段或序列化不改变 GPU 计算；新增内核、改变设备编译变体需要重新编译。选择已有变体可为运行时行为。 |
| screen / 日志 | 文案、布局、单位换算、ETA 展示、stdout/stderr 与镜像文件输出 | 采集设备计时与进度事实、控制安全点 | .cu 只传进度数值和计量单位，不包含格式化 Implementation。展示刷新不额外要求回读全部曲线状态。 |

这里的 Locality 是“修改格式只需理解对应 writer，修改曲线构造只需理解对应数学规则”；Leverage 是同一套普通 C++ Implementation 被 CPU、CUDA、OpenCL 调用。无需为每一种文件格式、进度字段或 CPU helper 额外建立公共 Module。

### 当前重编的三个来源

**第一类：真实的头文件或文本包含依赖。** [montgomery.cu](../../src/ecm/backend/cuda/stage1/montgomery/adapter.cu) 第 45–52 行直接包含日志、checkpoint runtime、进度头；其曲线准备、GCD、checkpoint、dump 与 screen Implementation 也位于同一翻译单元。进度格式在 [ecm_stage1_gpu_progress.h](../../src/core/ecm_stage1_gpu_progress.h) 第 8–20 行内联，[CMakeLists.txt](../../CMakeLists.txt) 第 151–152 行还显式声明此头为该 .cu 的 OBJECT_DEPENDS。因此改进度文案会使这个 runtime .cu 失效；这不等于所有独立 kernel instances 都因该头失效。

Edwards 的 [edwards_runtime.cu](../../src/ecm/backend/cuda/stage1/edwards/edwards_runtime.cu) 仅包含 runtime .inl；后者第 31–115 行同时做 GMP 打包、CUDA 分配、同步、回读和进度回调。已有 [edwards_launch.h](../../src/ecm/backend/cuda/stage1/edwards/launch.h) 和 [montgomery_launch.cuh](../../src/ecm/backend/cuda/stage1/montgomery/launch.h) 提供 launch Seam，应优先利用；本轮不要求为消除 runtime 的 CUDA 依赖再重造一套框架。与设备相关的 runtime 可以继续为 .cu，把不相关 Implementation 移入 .cpp 就能缩小其变动范围。

可选 resident/PRAC 路径也要同步收口：[prac_runtime.inl](../../src/ecm/backend/cuda/stage1/montgomery/prac/prac_runtime.inl) 第 35–41、48–172、264–295 行包含 checkpoint、环境配置、计划缓存、恢复与输出；[prac_window.inl](../../src/ecm/backend/cuda/stage1/montgomery/prac/prac_window.inl) 第 13–50、126–142 行包含配置解析与 CSV 输出。不能仅清理默认 ladder 的依赖后就声明 GPU 主机行为已全部隔离。

CUDA Stage2 已经有所分离。[arithmetic_kernels.cu](../../src/ecm/backend/cuda/stage2/device/arithmetic_kernels.cu) 第 1–8 行只引入 launch 与设备算术头；[transform_kernels.cu](../../src/ecm/backend/cuda/ntt/device/transform_kernels.cu) 第 1–7 行引入 NTT 设备头。本次本地 include 扫描没有发现它们依赖主机配置解析、结果或文件 writer。NTT 的 tune_checks.cuh 是设备调优校验代码，不是 tune.toml 编码。配置和调优输出已在普通 C++ 侧，例如 [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp) 第 1284–1356 行。不能把 Stage2 当前的脚本全量重编误写成这些 kernel 都包含输出代码。

**第二类：构建入口主动重编。** 这是独立于目录和 Interface 的问题：

- Stage1 local 入口只有启用 Incremental 才向并行 CUDA 编译传 SkipUpToDate，见 [build_stage1_local.ps1](../../tools/build/build_stage1_local.ps1) 第 102–105 行；[parallel_nvcc.ps1](../../tools/build/internal/parallel_nvcc.ps1) 第 122–151 行在未启用时仍把各 .cu 纳入编译。即使只改不在 CUDA 闭包里的 .cpp，默认入口也可能重编所有 .cu。
- Stage2 local 普通入口以整体 signature 判断是否 fresh；只要整体变化，就遍历全部源重新编译，只有 HostOnly 分支跳过已验证的 CUDA 对象，见 [build_stage2_local.ps1](../../tools/build/build_stage2_local.ps1) 第 112–135 行。
- Stage2 HostOnly 已证明“主机变化、复用 CUDA 对象”的构建路径存在；但第 78–80 行目前为所有源都生成 nvcc 命令，.cpp 也由 nvcc 驱动，链接第 162–163 行亦使用 nvcc。因此现状能称为“不重新编译 .cu”，不能称为“已全部用普通 C++ 编译器构建”。用户不要求 CUDA runtime 脱离后端，这一事实不妨碍将不相关普通 .cpp 改为正常 CXX 编译。

**第三类：共有输入和编译命令变化。** [cuda_build_state.ps1](../../tools/build/internal/cuda_build_state.ps1) 第 29–62、159–198 行同时考虑当前递归 include、保守预处理分支、compiler depfile、response file、编译命令、工具链及对象 hash。即使改动只涉及注释，只要文件仍在依赖闭包中，也可能使对象失效。换 arch、tiers、TPI/TPB、编译宏、GMP/CUDA 相关头、ABI 或编译器也可能合法触发 CUDA 重编。不能靠忽略依赖或关闭身份检查满足隔离目标。

已有保护应保留：[CMakeLists.txt](../../CMakeLists.txt) 第 118–155 行把 Stage1 源身份放在 host sidecar 中，避免把变化的报告 hash 注入所有 CUDA TU 的编译命令。CMAKE_CONFIGURE_DEPENDS 可触发重新配置和身份更新，**重新配置不等于 CUDA 代码必须重新编译**。主机 Implementation 改变后仍须更新程序身份与 EXE，不能借对象复用把整程序冒充为未变化。

### 拟议编译结构：只隔离会一起变化的 Implementation

当前关注的是构建 Seam，不要求增加新的公开分层。顶层继续使用前文的任务、执行、checkpoint、发布和队列 Module；其 Implementation 的编译依赖收敛为：

1. **普通主机 Implementation，CXX 编译。** 配置、任务会话、曲线与标量准备、CPU 算法、存档、输出和队列；不包含 CUDA 设备头或需要 NVCC 解析的文本。与 GPU 无关的功能可独立开发和验证。
2. **CUDA Adapter，保留 CUDA runtime。** 选择容量、资源、pack/unpack、上传、执行、安全点、回读和计时；可以继续使用 .cu。通过稳定声明调用主机准备与 writer 能力，不把其 Implementation 文本包含进来。
3. **CUDA 内核 Implementation。** 点公式、算术、设备字典、实例和 launch 实现；只依赖实际设备合同。迁入 src/ 后仍可保留独立翻译单元，不能为了取消 kernels/cuda 目录而把所有实例合成一个巨型 .cu。

依赖图的期望是：driver → 任务 Interface → 普通主机准备与执行 Interface；CUDA Adapter → 主机数据/协作声明 + 设备 launch 声明；writer.cpp 的函数体不再是 CUDA TU 的依赖。相应公开 Interface 放 include/ecm/ 对应路径，私有 host/device 协作声明放 src/ecm/backend/cuda/，保持与后文目标目录一致。无需把 CUDA launch 细节暴露给 driver。

控制依赖的具体做法：

- **实现移出头，而不是只更名目录。** 进度格式化、save/checkpoint codec、result/TOML 编码与构造算法放 .cpp；头只保留调用所需类型、声明和不变量。不把完整配置 schema、filesystem writer、结果 JSON 或业务选项大结构从新头传递回 .cu。
- **共享的是事实与合同。** 曲线身份、sigma、标量/算法选择、普通点、进度计量单位、借用快照有效期和错误状态属于必要 Interface。CLI 文案、文件路径策略、ledger 内部格式和展示布局属于 Implementation。新增主机配置字段若不改变执行输入，不应触碰 CUDA 合同。
- **区分普通点和设备布局。** 不让 checkpoint/save writer 理解 CGBN container、NAF GPU 字典或 CUDA pointer。GPU Adapter 在现有安全点完成必要同步与转换；writer 得到主机快照后才编码。不要为展示而在每批额外 D2H。
- **稳定 Interface 不代表永久冻结。** 布局、标量语义、位置单位、指针所有权或设备调用合同真的变了，相关 .cu 应重新编译。只有函数体或文件编码改变时才承诺复用。
- **保留已有 launch Seam。** 不是先把所有 runtime 改成 .cpp 才能隔离。可选的进一步 CXX 迁移还需处理完整 CGBN 头与现有 nvcc 编译限制，不属于本轮必要目标，也未做普通编译器可用性验证。
- **不增加一层无意义转发。** 一个 task 协作上下文即可承载进度/停止、快照确认和即时因子提交；不为它们各建一套继承树。Device handle 与模板只在 CUDA Adapter 内部可见。单体静态/对象库已经足够，暂不需要动态插件或稳定跨版本二进制 ABI。

### task内部顺序编排，driver只选择入口

利用现成的run_stage1_once形成唯一的run_stage1_task：校验→打开会话→准备→执行→核对→最终发布→完成会话。application driver调用task或QueueWorkflow，不再重复这七步。具体职责、删除/合并条件与伪代码统一见后文“职责划分与重复封装检查”和“driver 主流程伪代码”。

prepare内部保留各曲线/后端合法次序，不把K、恢复、字典、线程或device准备拆成driver逐一拼接的公开操作。队列提交在task之外由QueueWorkflow拥有。

建议保留一个粗 TaskHooks / ExecutionContext Interface，包含三类有完整合同的协作：

- **安全点与控制/进度。** Adapter 报告已确认的位置和计量单位，取得继续、请求快照或停止决定。screen 文案由普通 .cpp 处理；控制作用和展示作用应在合同中区分，不把所有进度回调都当作可丢弃日志。
- **同步主机快照提交。** GPU 先完成必要同步、回读和状态转换；CPU/IFMA 先固定曲线/lane 身份。存储返回成功/失败后才复用借用 buffer；如要异步写盘，必须另行拥有快照并设计完成确认，不能把 device pointer 或临时 scratch 直接交给 writer。
- **即时因子提交。** 准备/字典求逆或逐曲线结束时都可能发现因子。提交要有失败反馈和现有可靠性；不改成 void observer 或任务结束后的统一缓存。复合 proper factor 仍按当前合同处理，不先强制拆到素数。

这些协作调用发生在 prepare/execute 内部，**顶层顺序清晰和循环内即时 I/O 可以同时成立**。同一上下文里几个函数引用/方法就足够，CPU 并发回调继续由现有会话/发布 Implementation 负责串行化与所有权，不在此承诺多个任务可并发复用现有全局 checkpoint runtime。

当前顺序依据：run_stage1_once 第 3047–3056 行校验输出，第 3064–3075 行打开 checkpoint batch，第 3077 行承接 save history，第 3079 行执行，第 3095–3099 行完成 submission → internal batch → seal → end_batch(true)。错误走第 3080、3104 行 end_batch(false)。各曲线即时因子的调用位置和 lane/快照差异见前文“拆分必须保留的时间顺序”。GPU 安全点示例见 [montgomery.cu](../../src/ecm/backend/cuda/stage1/montgomery/adapter.cu) 第 1545–1580 行与 [edwards_runtime.inl](../../src/ecm/backend/cuda/stage1/edwards/edwards_runtime.inl) 第 97–114 行。

队列完成仍为一个完整 workflow：[ecm_driver.cpp](../../src/core/ecm_driver.cpp) 第 3307–3364 行依次处理 Stage2 outbox park、finished、source replace/remove、save sync、Stage2 flush 和 Prime95 delivery。Prime95 失败可以保留 pending，不倒退已完成计算；源队列更新失败不能静默进入下一轮重算。拆分前后均须保留这些具体行为。

### 实施与验收的统一入口

编译策略、目录隔离、源码提取和验收顺序统一见后文“首先调整构建复用与输出隔离”“分阶段正式重构顺序”“验收矩阵与待确认事项”。本节只定义编译隔离的目标和合同，不另设第二套阶段编号。

验收中的“CUDA编译为零”指不重新编译.cu/生成设备代码，不禁止对象验证、库链接或工具链检查。真正使用runtime的Implementation继续留在CUDA。共享Interface变化导致的必要重编是正确行为；要消除的是无关依赖和默认全量策略造成的重编。

## 当前曲线和后端能力

下表区分生产入口的接纳范围与底层算术能力。它不是未来必须补齐的功能矩阵。

| 阶段与后端 | 当前生产接纳范围 | 限制与代码依据 |
| --- | --- | --- |
| CPU Stage1 | Montgomery Suyama param0；Edwards Atkin–Morain；标量及 IFMA 路径 | Montgomery 接口见 `src/cpu/ecm_mont_cpu.h` 第 2–17、54–66 行；Edwards 见 `ecm_edwards_cpu.h` 第 2–24 行。CPU 路由并未把 `gpu_param` 当成统一曲线参数，驱动第 892–893 行。 |
| OpenCL Stage1 | Montgomery param3 | param0/2 明确拒绝，见 `src/opencl_backend_glue.cpp` 第 22–26、66–77 行；Edwards GPU 路由要求 CUDA，见驱动第 869–873 行。 |
| CUDA Stage1 | Montgomery param0/2/3；Edwards | 实际支持受编译档位、`ECM_NO_PARAM2`、`ECM_MERS_FOLD` 等条件影响，见 `src/cuda/backend.cpp` 第 30–41 行。Edwards 使用独立入口，驱动第 2758–2761 行。 |
| CPU Stage2 | 当前入口读取 Edwards AM 文本存档 | VERSION、SCALAR、Z 的检查见 `src/core/ecm_edwards_save_reader.h` 第 34–47 行；引擎选择见 CPU Stage2 main 第 165–201 行。底层 GW 接口接受 N、x、a24；不能从入口限制推出算术不支持 Montgomery x-only。 |
| CUDA Stage2 | Suyama param0 和 Edwards AM 存档 | 接纳和拒绝规则见 CUDA Stage2 main 第 148–187 行；拒绝 param2/3，要求归一化 X，Z=1 或省略。Edwards 校验后重建 a24，进入同一 x-only 管线。 |
| OpenCL Stage2 | 未发现生产入口 | 不因存在 OpenCL Stage1 就推定支持 Stage2。 |

当前 CUDA Stage2 生产进程入口的 CMake 构建要求 Windows，且 CPU Stage2、CUDA Stage2 都是单独的可选构建目标；见 [CMakeLists.txt](../../CMakeLists.txt) 第 71–73、259–263 行。目录重构不隐含平台移植。

## 主要复杂度来源

### 任务驱动同时拥有输入 执行与完成协议

**建议强度 Strong。** `ecm_driver.cpp` 既知道 CLI 和队列，也知道 CPU 线程、SIMD、CUDA Edwards、save、结果发布和交接。修改一种执行路径时，需要理解与它无关的完成及恢复流程。

提取一个Stage1任务Module，集中单次计算及publication成功条件；QueueWorkflow集中来源队列、完成与交接条件。checkpoint和publisher提供实际存储能力，不在task外再包一套完整任务完成流程。应用入口只组装依赖并选择命令。

这两个 Module 的 Depth 来自集中隐藏批次与完成协议：如果删除它们，复杂度会重新散落回每条 CPU/GPU 路径；因此不是只改名的转发层。验证可以从同一 Interface 覆盖成功、停止、恢复和失败，提高 Locality。

### GPU 接口无法覆盖实际执行路径

**建议强度 Strong。** [ecm_backend.h](../../include/ecm_backend.h) 第 42–71 行主要是 GPU Montgomery Stage1 的链接期替换接口，长参数列表混有 OpenCL 路径、CUDA B1/torsion、计时和 checkpoint 控制。CUDA Edwards 则从驱动直接调用另一组接口。

重新定义阶段级执行 Interface，分别表达能力查询、准备和运行。保留两个真实 Seam：Stage1 批执行与 Stage2 完成点续算。CPU、CUDA、OpenCL 在自己确实支持的阶段提供 Adapter，不要求创建空实现补矩阵。

现有 OpenCL entry 的再次转发见 [opencl_ecm_entry.cpp](../../src/opencl_ecm_entry.cpp) 第 26–39 行。仅转发同一参数列表的 Shallow Module 应在新接口生效后合并；有格式校验、资源管理或错误翻译职责的 Adapter 应保留。

### 头文件和全局状态扩大了理解范围

**建议强度 Strong。** `src/cuda/stage2/context.h` 第 2–16 行传递引入多个规划模型、NTT 运行时、启动声明和设备相关头。`options.h` 同时公开设置项、统计、pinned buffer 和 event；`product.h` 的 `PolyLayer` 同时暴露模数、Arena、S4 和大量计数。它们已分文件，但很多调用者仍需要知道彼此的内部结构。

将不可变选项、一次运行的状态、可复用的设备资源和结果统计分别归属明确的 Module。外部调用者只持有执行会话；Stage2 内部按点生成、多项式运算和 NTT 的职责共享最小私有接口，不把整个 context 当成通行证。

### 核心目录混有数学 设备模型和文件协议

**建议强度 Strong。** `src/core/` 包含数学构造和纯规划，也包含程序 main、worktodo、Prime95 文件、结果 JSON、CUDA 显存模型、控制台输出以及包含实现的 `.inl`。目录名无法回答依赖什么。

按职责分配到 application、service、core、storage、backend。特别是“没有 CUDA 调用”不等于“属于领域核心”：模拟 CUDA Arena、S4、驻留 owner 生命周期的模型应由 CUDA Module 所有，仍可以在无 GPU 环境下验证。

### 主机与设备文件分离增加了平台修改跨度

**建议强度 Strong。** 同一个 CUDA 实现跨 `include/ecm/cuda/`、`src/cuda/` 和 `kernels/cuda/`。当前 `include/ecm/cuda/launch.h` 直接公开 `cuda_runtime.h`、dim3 和 stream；这些是 CUDA Module 内部协作接口，不是主流程应依赖的公共能力。

把主机执行、启动声明、设备模板、实例和常量所有者放在同一后端 Module 下，并保留各自编译单元。调整物理位置不等于合并 `.cu`，也不等于让普通 C++ 编译器解析设备实现。

### GUI 编排与界面状态混合

**建议强度 Worth exploring。** GUI 已有进程 Adapter，但 worker 重启、停止、日志归并和结果处理仍与界面绘制共处。提取 worker 编排 Module，让界面读取状态快照并提交动作。计算程序继续通过现有进程和文件协议运行；不在本轮同时改成 GUI 进程内计算库。

## core 文件粒度与驱动耦合

### 结论与判断标准

**需要同时合并和拆分，但应优先减少调用者必须理解的内容。** 当前既有 `ecm_driver.cpp` 的多职责耦合，也有大量实现头在 core 顶层互相包含。后一种情况不能全部归为“文件太小”：有些是同一 Module 的实现被过度公开，有些是放错层，有些保留了有效的共享不变量。

目录顶层有 96 个文件、73 个 `.h`，其中 34 个头文件不超过 80 行。这是目录结构统计，包含目录内非主流程文件，不是生产闭包或复杂度评分。合并依据是职责与依赖，不能按行数设置拆分或合并门限。

判断一个 Seam 是否值得保留，应回答：调用者能否只凭这个 Interface 正确使用它；它是否隐藏了一组真实规则；它的状态和不变量能否独立定义；修改它是否通常不要求同时修改相邻实现。若调用者仍需组装内部步骤、传递同一大上下文并了解先后次序，拆出的文件还没有形成独立 Module。

必须区分三个动作：**归属调整**把实现放入正确 Module，**接口收口**把无须被调用者知道的类型收回私有实现，**物理合并**才是把多个文件合成一个。很多文件只需要前两个动作。一个深 Module 可以有多个私有源文件，公开文件数量也不等于 Interface 数量。

### 建议合并职责或收回私有实现的文件

| 文件簇 | 当前问题与依据 | 调整建议 |
| --- | --- | --- |
| `ecm_result_submission.h`、`ecm_internal_result.h`、`ecm_submission_writer.h` | 三个生产程序共同包含格式与账本实现。writer 第 2–23 行连带引入曲线 helper、文件发布、锁和平台头；调用方同时知道编码与持久化。 | 归为结果发布 Module，对调用者提供结果记录、发布、已有结果查询及 seal 等少量必要操作。submission/internal 编码与 ledger 保留为私有实现文件，不拼成新的巨型公共头。 |
| `ecm_stage1_legacy_gpu.h` 与 checkpoint codec/runtime | legacy header 第 14–58 行包含完整旧格式导入、文件探测和发布；当前主流程调用在 driver 第 2831 行。它是恢复流程的一种实现。 | 导入归 checkpoint 恢复 Module 私有实现，在打开或恢复会话时协调；不为旧格式再建立一套公开的 importer、validator、locator 类。载荷编码仍按具体算法私有区分。 |
| `ecm_stage1_exp.h`、`ecm_stage1_exp_cache.h` | 生产调用者为取得同一有效指数，需要理解算法与缓存。它们也有不同的纯计算和文件 I/O 职责。 | 任务层提供一次“取得符合语义的标量计划”的入口，缓存路径、读取、校验和回写私有化；纯指数算法与缓存可继续保留两个实现文件，不把文件系统依赖带回 core。 |
| `ecm_stage2_giant_memory.h`、`ecm_stage2_giant_state.h` | 前者表达一段资源计划，后者包含前者并模拟同一组资源的时间线；由 workspace 汇总使用。 | 可合为 CUDA 资源模型内的 `giant_resources` 私有实现；若为了读懂静态计划与动态回放继续保留两文件，也只暴露统一规划结果。此项是合并候选，不改变分配/释放事件。 |
| `ecm_stage2_ntt_memory.h`、`s4_memory.h`、`s4_program.h`、`owner_memory.h`、`initial_memory.h`、`workspace_memory.h` | workspace 第 2–6 行组合多种模型，CUDA context 又引入这些完整模型。各子模型有不同资源寿命，不是重复公式。 | 归入 CUDA 资源规划 Module，对外只暴露请求、规划结果和诊断摘要。保留内部子模型，避免把全部状态机合成一份巨型 `memory.h`；让计算点或多项式的调用者不必包含整套模型。 |
| `ecm_stage2_tune_*`、`portable_model`、`width_prediction` | 活跃 portable 路径借用头链中的字段解析和拟合基础，连带引入完整模型。`portable_model.h` 第 225、230 行只需 `fit_route_cost` 的相关能力，却经 components/prediction/ecm 头链取得它。 | 收敛为 profile 数据与读写、成本预测、运行搜索三个完整职责，先抽出仍使用的基础实现，再让当前路径只依赖所需内容。被连带包含的模型与兼容接纳是否保留需独立核对，不按头文件整片删除。 |
| `ecm_stage2_portable_driver.inl` | 在 Stage2 main 内文本包含，依赖其 Options、Record、child_run 等外层状态；长度 1487 行并未形成独立 Interface。 | 移为调优 Module 的普通实现，明确请求、测量执行 Adapter、profile 与结果。候选生成和重复测量是内部流程，不给每个 helper 建立新 Module。 |
| `ecm_stage1_gpu_progress.h` 与 driver 内进度 helper | GPU 两条曲线确实共享 stdout 规则；driver 又把锁、计数、速率和 bar 分散为全局状态。 | GPU 协议 helper 可保留私有小头；driver 内进度聚合为一次执行的私有 reporter/state，不为计数、锁和速率分别设计 Interface。 |

对于这些 tune 接口，生产源全文搜索未发现 `auto_candidates`、`tune_tail_grid`、`EcmProfile::load`、`NttProfile::load`、`ComponentEstimator` 的直接调用，而当前 `portable_driver.inl` 第 229–230 行等位置实际创建 `pt::Estimator` 和 `pt::WidthPredictor`。这说明“被包含”不能直接解释为“整套模型都被当前运行使用”。但共享解析、特征和拟合仍有真实调用；删除资格需进一步核对全部生产配置、模板实例与取址引用，本次只提出收口和清理候选。

profile 读取可以共享语法解析，验证仍应区分完整 Stage2、纯域卷积和 Stage1 每曲线进程时间等不同计时单位，不能合成一个宽松的通用校验器。成本估计、实时准入、成功准入缓存也保留各自合同；预测值不构成显存承载证明。

### 应保留的小文件和不变量

以下反例说明，减少文件数量本身可能增大耦合：

- **`ecm_expr.h` 与其实现。** 小头表达整数表达式解析，Stage1 与 CUDA Stage2 共用；它把约 299 行实现藏在小 Interface 后，是合理 Depth。
- **`ecm_stage1_binary.h`。** 38 行提供小端读写、长度检查、CRC32C 和容器标识，供 checkpoint runtime、legacy import 和 Edwards 载荷等共同使用。应作为 checkpoint 的私有共享基础保留，不能因小而吞进只属于某条曲线的 codec。
- **`ecm_result_paths.h`。** 9 行规则由 writer 与 GUI 的 `src/gui/results.cpp` 第 1 行共同使用。保留一个轻量路径契约，避免 GUI 为检查结果位置引入 GMP、账本锁和 writer 全部实现。
- **`ecm_result_json.h`。** 除结果编码外，`ecm_toml.h` 第 5 行和 `ecm_stage2_queue_state.h` 第 2 行也使用它。它可以归入受限的共享解析实现，不能仅为合并结果文件而迫使 TOML/queue state 依赖结果发布 Module。
- **`ecm_stage2_giant_chunk.h`。** 保证点块包含完整 G-tree 批次，由 CUDA `plan.cpp` 第 368 行及 `batch.cpp` 第 181 行共用。保留统一规则；不能分别复制进规划与执行。
- **`ecm_stage2_giant_tree.h`。** G 树布局同时服务 `ecm_stage2_requests.h` 第 66 行的请求拓扑与 CUDA `tree.cpp` 第 211–212 行的实际执行。该独立布局不变量不能被 giant 内存模型吞并。
- **`ecm_stage2_initial_memory.h` 中的 `BabyMemoryLayout`。** 同一布局被 CUDA `points.cpp` 第 627–643 行的实际分配与 `plan.cpp` 第 391 行的规划复用。事件模拟器可以私有化，布局合同要继续共享。类似地，`giant_work.h` 表达路由工作量，不应因为同名 giant 就被并入显存状态机。
- **`ecm_stage2_runtime_cache.h`。** 缓存身份、容量和实时内存快照有效性与预测数学不同，可以留作运行搜索 Module 内的独立私有实现；不因 runtime tune 包含它就合并两种职责。
- **`ecm_stage2_width_reference.h`、`tune_columns.def`、`tune_rows.def`。** 分别是有来源身份的生成数据和统一字段清单。它们不是新的架构层，保留数据或生成物与算法分离，迁移不能手写一份替代数据。

`ecm_file_publication.h` 的原子替换规则也被 checkpoint runtime 与结果 writer 共同使用。更适合保留平台能力并把 Windows 实现藏入 `.cpp`，不宜吞进其中一个业务 Module。

### ecm_driver 的合适拆分粒度

建议收敛为六个完整职责，**不是规定只剩六个文件，也不是六个职责各建一层类继承**。

| Module | 当前应归入的主要入口 | 应隐藏的内容 |
| --- | --- | --- |
| Stage1 命令入口 | `main` 第 3969 行，usage 第 536 行，sigma 解析第 116、153 行，设备信息第 3412 行 | CLI 文本、帮助、路径解释和展示。PARI 群阶诊断第 214–535 行可作为应用私有完整功能文件，不另建公共数学 Module。 |
| 队列运行与完成 | `run_queue_manager` 第 3465 行，`queue_run_one` 第 3201 行，Stage2 交接行构造第 3162 行 | worker、任务选择、finished/source queue、save sync、Stage2/Prime95 交接的完整次序；不把当前二十多个参数扩散到新头文件。 |
| Stage1 任务会话 | `Stage1RunOptions/Result` 第 715、751 行，任务验证第 869、905 行，`run_stage1_once` 第 3030 行 | 数学请求与执行选择、一次运行的拥有权、checkpoint 会话和任务成功条件。输出记录的具体编码交给发布实现。 |
| Stage1 批执行 Adapter | CPU Edwards 第 959–1994 行，CPU Montgomery 第 2085–2749 行，GPU Montgomery 分支第 2773–2991 行，以及 Edwards GPU `.inl` | 每种曲线的 scalar/SIMD、worker、快照安全点、lane 身份和回退集中在完整实现中。ISA 与 affinity 执行归 CPU Adapter；字符串解析留应用。 |
| Checkpoint 恢复与存储 | checkpoint runtime、binary、legacy、Montgomery/Edwards 载荷 codec | batch 生命周期、主备、记录、兼容接纳和文件发布；点状态何时安全和如何从设备布局取出仍由对应 Adapter 决定。 |
| 结果与完成点发布 | save 命名第 770–848、2008–2083 行，结果设置与即时发布第 2994–3027 行，最终发布第 3083–3098 行 | save codec、submission/internal 编码、ledger 和 seal；中途 checkpoint 与完成 save 仍是不同操作。 |

直接CLI和队列都经同一run_stage1_task入口执行，Session是该入口的私有状态。task使用执行、checkpoint和发布能力；QueueWorkflow处理待办推进及跨阶段交接。task不反向理解worktodo/Prime95，执行Adapter不删除源队列，结果writer不决定SIMD切片。

**结果拥有权应在拆文件前明确。** 当前 `Stage1RunResult` 暴露 `mpz_t*`、`int*`、`uint64_t*`，队列分支需清理这些数组。请求里的 `stage1_t0_ms/stage1_curves` 又被当作运行状态，Edwards 分支第 1930–1931 行通过转换修改 const options。应先分清请求值、一次执行状态和有自动清理能力的结果，避免把同一大结构搬到公共头后扩大耦合。

**CPU scalar/SIMD 与回调不先分成独立 Module。** `EdwardsWorkerCtx` 第 1111–1127 行共同持有任务下标、sigma、checkpoint、因子、save、停止及完成状态；启动、join 和最终判断属于同一次执行。适合按曲线形成私有完整实现，不能先把线程、回调、lane 转换和计数分别公开，再让调用者拼接生命周期。

### 拆分必须保留的时间顺序

执行过程并非先纯计算、全部返回后才发布。Edwards scalar/SIMD 在驱动第 1438、1559、1615、1863 行，Montgomery CPU 在第 2509、2624 行，GPU Montgomery 在第 2923 行，GPU Edwards 在 `.inl` 第 180 行即时发布因子。发布失败会影响当前执行的失败或停止状态。

因此任务会话应向执行 Adapter 提供**能返回失败的即时结果提交能力**；它隐藏文件和账本细节，并按现有安全点同步确认。不能换成只收集事件、任务结束后统一落盘的实现。最终 submission/internal batch、seal 和 checkpoint 结束仍由任务会话统一控制。

任务成功顺序集中在 `run_stage1_once`：校验输出 → 建立 batch identity/lock → 继承已有 save → 执行并处理即时输出 → submission → internal batch → seal → `end_batch(true)`。失败保留 `end_batch(false)` 的恢复条件。队列的 park/finished/移除源任务/sync/flush/delivery 则保持在队列完成 Module，不能交给通用发布器自动执行。

### Module 内的提取优先级与停止拆分条件

以下是模块内部的提取优先级；整体实施先完成后文 P1 构建复用与目录隔离。

1. **先收敛任务请求、结果拥有权和成功条件。** 利用已有 `run_stage1_once` Seam，暂保留内部计算循环，避免一开始就暴露巨型 options。
2. **整段提取队列运行与完成流程。** 保持发布和交接次序，不按每个 append/advance/sync 各造一层。
3. **按曲线整体迁出 CPU 批执行，再接入 GPU Adapter。** 保留 scalar/SIMD 与 checkpoint 安全点共同拥有的状态，之后再判断有无值得独立的内部 Seam。
4. **将结果与 checkpoint 的实现收回私有源文件。** 调用者只取得有完整含义的操作；轻量共享不变量继续保留。
5. **最后按真实调用收口 Stage2 规划与调优头链。** 先分离活跃基础、生成数据和兼容接纳，再决定是否物理合并或清理；不以删文件数作为结果。

一个职责能在少量入口后隐藏完整规则，调用者不需要读它的全部内部头，测试也能从同一 Interface 覆盖其外部合同，就应停止继续拆公开 Module。若拆出的新文件仍共享同一巨大上下文、必须总一起修改或只原样转发参数，应撤销这个浅 Seam 或将其保持为私有实现文件。

## 曲线优先与后端优先的选择

### 对每种曲线实现不同后端

结构可表现为 `curve/montgomery/{cpu,cuda,opencl}` 与 `curve/edwards/{cpu,cuda}`。优点是同一曲线的构造、点公式与不同设备实现容易横向查找，适合主要变化来自曲线种类的项目。

对本仓库，它会分散相同设备的上下文、内存、算术、实例化与编译设置。CPU 两条曲线共享 IFMA 算术，CUDA Stage2 的多项式和 NTT 按计算状态工作；这些实现很难自然地归属于某一种来源曲线。最终容易再次形成跨曲线的巨型 common 目录。

### 对每种后端实现不同曲线

结构可表现为 `backend/{cpu,cuda,opencl}/stage1/{montgomery,edwards}`。一个设备的资源、布局、编译和性能知识集中，适合当前实现的实际耦合关系。新增一种后端可以独立实现已有数学合同。

风险是同一曲线的规则在各后端复制。因此曲线身份、参数校验、有效标量语义和规范化完成点必须有共同的 Interface 与等价性验证，不能让目录布局决定业务语义。

### 推荐的混合方式

**core 按曲线和算法表达语义，backend 按平台表达实现。** 这两种组织方式同时存在于各自适合的位置，无需选择一个覆盖所有代码。

必须区分以下正交概念：

- 曲线构造身份：Edwards 或 Montgomery；param 仅用于 Montgomery；sigma 是构造种子。
- 执行后端：CPU、CUDA、OpenCL。
- 点乘算法：binary ladder、NAF、PRAC 等，是实现策略。
- 算术与约减策略：GMP/MPN、IFMA、CGBN、通用 REDC、Mersenne fold 等。
- 运行选择：设备、线程、批宽、位宽容器、功耗环境及内存预算。

“Montgomery 曲线”与“Montgomery 模乘表示”分别属于第一项与第四项，命名和目录应明确区分。`ecm_edwards_mont.h` 的通用 CPU 域运算被 Montgomery 曲线复用，应放入 CPU arithmetic，而不是 Edwards 子目录。

## 完整重构目标与实施约束

本节是整合后的**目标实施方案**；前文保留分析时的现状、数学和耦合依据。用户已授权按方案实施；当前已完成 P0 local 基线冻结与 P1a/P1b 构建复用和输出隔离，实际交付与验证限制见下节；P2 已完成普通 CXX、任务拥有权与请求/状态合同；P3 完成 CUDA/主机/OpenCL 家族归位，P4a–P4i 已完成 Stage1 职责整理，P5–P7 继续实施。不把目标目录写成当前结构，后续以本节的阶段顺序为准，不再并行维护一套目录迁移顺序。

整合方案沿用HEAD为2e64b75e5d14287e2ee25bdaff27e3b1128c1d93、当时包含工作区修改且版本头为0.44.7的[读取快照](../../data/experiments/main_flow_refactor_plan_20261021/source_snapshot_final.json)，记录633个源码/构建输入hash，早期采集文件另行保留。职责与封装的文档细化不代表重新审计当前并行源码，原始证据不改写。当时版本与worktodo修改非本文档工作所作；已核对的队列投递规则为复用目标worker最后一段、必要时才建标签、压缩同一连续分区重复标签并保留任务顺序。该规则归storage/queue与service/work的现状合同，迁移保持；前文0.44.6细粒度依据不被改写成同一源码身份。

目标为：顶层按 application、service、core、backend、storage、platform 划分职责；计算 Implementation 按后端、阶段、曲线组织；include 只暴露跨 Module 必需的 Interface；CUDA 的主机、设备和实例统一进入 src。配置、主机曲线准备、文件编码与展示可普通 CXX 编译；需要 CUDA runtime 的资源、上传、执行、同步与回读继续留 CUDA Adapter。

以下保持现有行为：独立 Stage1/Stage2 程序及 GUI 子进程模式、曲线身份与有效标量、受支持组合、配置优先级、文件格式与兼容范围、即时因子提交、checkpoint 安全点、队列完成顺序、生产算术检查和设备常量所有权。组织重构不同时升级算法、默认值、语言标准或文件协议。

### 决定模块粒度的规则

- 一个 Module 隐藏一套完整规则，如一次任务、一次执行、恢复存储或结果发布。不是一文件一模块，也不是一函数一类。
- core 中频繁一起修改的构造/归一化/结果分类收回同一数学职责；稳定的进度单位、算术不变量和文件协议定义可以继续是小文件。
- 公开 Interface 隐藏顺序、所有权和失败处理，形成 Depth；私有 .cpp、.cu、.cuh、.inl 可以按编译与阅读需要拆分。
- 用阶段/批次级运行时选择连接 Adapter；域乘、点加倍和逐位热循环不新增跨 Module 虚调用。无需动态插件和深继承树。
- 文件位置回答“功能在哪里”；依赖合同回答“改动影响谁”；构建与行为验收共同证明 Locality，文件行数不能单独证明复杂度下降。

## 实施进度与阻塞记录

### P0：local 基线与生产输入已冻结

本次从干净提交 `57351082d0d909985f0c2c47580de94359d17e0b`（程序版本 `0.44.9`）开始。冻结 1959 个受版本控制文件的摘要和归档，覆盖生产入口、CUDA/OpenCL 资源、构建清单、配置定义、GUI/CPU 源码及现有协议说明。文档更新前复核源文件未变。设备、命令、日志、冻结程序、构建清单及样本位于本机忽略目录 [P0 验证记录](../../data/experiments/main_flow_refactor_implementation_20261021/p0/qualification.json)；原始数据不随 Git 提交，草稿与网页提交状态及摘要。

| 项目 | 实际结果 | 证据与限制 |
| --- | --- | --- |
| Stage1 local | `build_stage1_local.ps1 -Incremental` 成功；31 个 CU 全部复用，重新编译 5 个 CXX 文件并链接 | [构建日志](../../data/experiments/main_flow_refactor_implementation_20261021/p0/stage1_local.log)；不是冷构建，也不是默认复用已完成的证明。 |
| Stage2 local | 同一正式脚本 `build_stage2_local.ps1` 在沙箱外完整编译 25 个 TU、链接并写入清单成功 | [沙箱外构建日志](../../data/experiments/main_flow_refactor_implementation_20261021/p0/stage2_local_outside_sandbox.log)；源码与编译参数不变。 |
| Stage1 工厂查询 | GPU1、完整 32 档的 260 项注册、选择及拒绝检查通过 | [检查结果](../../data/experiments/main_flow_refactor_implementation_20261021/p0/registry_02/result.json)。第一次调用误用手写的不完整期望档位，日志保留；第二次从 `tiers.def` 读取，不修改测试或生产实现。 |
| 两阶段运行样本 | Stage1 5 例、Stage2 5 例各重复运行两次，完成点、非计时结果及目标叶值一致 | [运行记录](../../data/experiments/main_flow_refactor_implementation_20261021/p0/runtime/summary.json)；这是同一基线的可重复性与样本冻结，不是重构前后等价或完整支持矩阵运行验收。 |
| 验收工具限制 | `test_stage1_usability.py` 首例因仍调用已移除的 `-ini` 被程序拒绝 | [失败日志](../../data/experiments/main_flow_refactor_implementation_20261021/p0/usability.log)；未放宽门限或修改实现。该脚本的配置/CLI 样本须在任务提取验收前适配，再验证暂停、恢复、发布重试。 |

Stage1 样本覆盖 Montgomery PARAM0/2/3、Edwards NAF/binary；Stage2 覆盖普通 Montgomery/Edwards、承载、重定位和 host 回退，保留其生产算术检查。曲线身份、有效标量、checkpoint 安全点、即时因子发布、队列完成次序及 Stage2 pending/receipt 合同仍以本草稿的已核对流程和冻结源码为准。CPU/OpenCL/GUI 本次只冻结输入与已有支持限制，实际运行验收留在对应实施阶段，不把 CUDA 构建成功外推到这些后端。

首次沙箱内 Stage2 构建在第 16/25 项 `kernels/cuda/stage2/arithmetic_kernels.cu` 失败：`Internal error`，随后 `nvcc error: 'ptxas' died with status 0xC0000008 (INVALID_HANDLE)`。失败记录和部分对象状态保留于 [原始失败记录](../../data/experiments/main_flow_refactor_implementation_20261021/p0/result.json)，不覆盖为成功。用户确认沙箱限制为已知问题并授权继续；随后沙箱外同脚本、同参数成功，旧失败不再阻塞 P0。**这只能证明沙箱外构建可用，具体错误根因未进一步诊断。** 后续 CUDA 编译沿用该执行方式，不为避开沙箱错误改变工具链、算术宏或编译并行度。

工具链为 CUDA 13.3.73、MSVC 14.51.36231、sm_89。Stage1 保持完整 tiers、压缩开启、无 PTX、PRAC 关闭；Stage2 保持 SplitCompile=8、ptx/fixed mode3、outer unroll0、add/sub mask1。未运行 release、跨架构构建或性能比较。设备清单见 [devices.csv](../../data/experiments/main_flow_refactor_implementation_20261021/p0/devices.csv)。

冻结程序 SHA256：Stage1 为 `22327aae26f81cde2bf8b95d075e50697649521890f5d7d97f71e85cd3980094`，Stage2 为 `6c1e5a4df7c813a17e561935dfa30068a689e7cdd2c439048d8a0beca9c8ccd5`；GMP DLL 为 `969f0e96ac587dedfecf611ca3442fc7211f9218f1a553214c83548dcdbcfd6e`。源摘要文件 SHA256 为 `8793675558067C2EA53D1AC3236B95F2F639297BC0AE1E12EAB2C6D19FB33570`，源归档 SHA256 为 `50A99E7C930EE9EF8F640493BC8711A4BA0FD66FFE819446AE319A746F0B91A4`。成功构建后另行复制两阶段程序与匹配清单，失败构建记录不作为当前基线。

### P1a：默认增量与 Stage2 逐 TU 复用已完成

对应程序版本 `0.45.1`，证据为 [P1a 验证记录](../../data/experiments/main_flow_refactor_implementation_20261021/p1a/qualification.json)。本子阶段保留原有目录、编译器、编译/算术宏和链接参数，先改变复用决策；P1b 继续处理用途/profile 目录隔离及构建模式收口。

Stage1 local/dev 及内部并行编译器默认验证并复用 CUDA 对象；`Incremental` 保留兼容，显式关闭时转为强制编译。原有 `FullRebuild` 继续删除所选 CUDA 对象。Stage1 主机 CXX 仍由原生 CMake/NMake 管理，不能据此宣称已经统一全部 CXX 对象的内容身份。

Stage2 整体签名变化后按当前命令、依赖内容和对象摘要分别判断每个 TU，仅重编失效单元；输出重编原因。另存链接命令、GMP 导入库及对象摘要，程序自身摘要匹配时可跳过链接。`HostOnly` 仍先验证全部 CUDA 输入/对象，和 `Rebuild` 互斥；结束时同时核对输入指纹及对象摘要，避免普通复用路径把构建期间被替换的对象直接接纳为新清单。

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 同 profile 无修改重复构建 | Stage1 零 CU、零 CXX、零链接；Stage2 零 TU、零链接 | [Stage1](../../data/experiments/main_flow_refactor_implementation_20261021/p1a/stage1_final.log)、[Stage2](../../data/experiments/main_flow_refactor_implementation_20261021/p1a/stage2_final.log)。Stage2 最终轮还覆盖脚本身份变化而实际链接输入未变。 |
| 独立主机源修改及还原 | 对 `ecm_expr.cpp` 临时追加注释，再原字节还原；两阶段两轮均各编译 1 个主机 TU、0 个 CU，并链接 | [受控变异记录](../../data/experiments/main_flow_refactor_implementation_20261021/p1a/integration.json)。源码已还原，最终程序及清单与还原后的输入匹配。 |
| 身份/依赖变异 | 原有 40 项 CPU 门禁通过，含跨 PowerShell 指纹一致性 | [门禁记录](../../data/experiments/main_flow_refactor_implementation_20261021/p1a/identity_final/result.json)。 |
| HostOnly 严格拒绝 | 真实算术 CUDA 对象内容变异在编译前被拒绝，对象和时间戳已还原；CUDA 源摘要变异也在编译前被指定诊断拒绝 | [对象变异日志](../../data/experiments/main_flow_refactor_implementation_20261021/p1a/hostonly_object_tamper.log)、[源摘要变异记录](../../data/experiments/main_flow_refactor_implementation_20261021/p1a/hostonly_rejection_fixed/result.json)。 |
| 两阶段运行对照 | 与冻结的 `0.44.9` local 基线比较，Stage1 5 例、Stage2 5 例通过 | [运行记录](../../data/experiments/main_flow_refactor_implementation_20261021/p1a/runtime/summary.json)。覆盖范围和忽略的计时/身份字段沿用 P0 工具，不构成性能或完整后端资格。 |

同步修正一个旧验收样本：`test_stage2_phase_build_deps.py` 原来修改仅属主机的 phase-times 头摘要，却要求 CUDA 依赖错误；复制到不同叶目录后实际先遇到编译路径身份错误。现在改为变异真实 CUDA 源的已录入摘要，仍要求指定 CUDA 输入诊断、零编译且原构建清单不变；没有放宽为任意失败即通过。同时为 Windows PowerShell 子进程明确执行策略和系统模块路径，避免测试在依赖检查前因运行环境失败。旧失败日志保留。

实际修改点见 [Stage1 local](../../tools/build/build_stage1_local.ps1)、[并行编译器](../../tools/build/internal/parallel_nvcc.ps1)、[身份校验](../../tools/build/internal/cuda_build_state.ps1) 与 [Stage2 local](../../tools/build/build_stage2_local.ps1)。源码/程序/清单摘要统一记录在上述 qualification.json。尚未实现新用途目录、Stage1 严格 HostOnly/RelinkOnly、Stage2 普通 CXX 迁移，也没有把四类非 GPU 变更的真实依赖全面隔离；这些分别属于 P1b、P2、P4。未运行 release 或性能比较。

### P1b：用途/profile 隔离与严格构建入口已完成

对应程序版本 `0.45.2`。[完整验证记录](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/qualification.json)保存源输入、程序、清单及 marker 摘要；[精确叶目录](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/locations.json)由实际入口输出记录。构建用法见[用途与 profile 目录](../usage/BUILD.md#用途与-profile-目录)。

默认目录按 local/dev/release、阶段、引擎、架构、工具链和 profile 隔离。完整字段摘要保存在 profile.json，前 16 位用于目录；源码/HEAD/程序版本不进入 key。显式本机/开发目录是最终叶目录，不追加层次；公开多架构发布参数保持根语义。包目录与构建目录必须独立且不嵌套。旧目录只在正式清单及实际 cache 相容时原位接纳，不迁移缓存。

Stage1 新增严格 HostOnly：要求已配置的同 profile 目录，在配置/主机编译前检查全部 CUDA 对象、内容身份和原生依赖时间戳。关闭旧 RelinkOnly 绕过入口并给出明确诊断，没有保留一个名为 relink 却可隐式重编 CUDA 的入口。Stage2 保留逐 TU 与链接身份门禁，增加 role/profile 对照；dev 默认仍为实验引擎。

| 验证 | 结果与范围 | 证据 |
| --- | --- | --- |
| CPU profile/路径/身份门禁 | 29 项 profile、9 项发布路径、40 项既有身份检查通过；含 PS5/PS7 一致性和 marker 篡改拒绝 | [profile](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/profiles_final/result.json)、[路径](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/release_paths/result.json)、[身份](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/identity_final/result.json) |
| 旧完整档目录原位复用 | build_cuda_cmake 保留 32 档；严格 HostOnly 下零 CU，版本头引起必要主机编译/链接 | [日志](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/stage1_legacy_hostonly.log) |
| 新隔离叶目录冷构建 | Stage1 限定 384 档；Stage2 production 完整 25 TU；均使用 local 原有编译/算术参数，沙箱外执行 | [Stage1](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/stage1_new_leaf.log)、[Stage2](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/stage2_new_leaf.log) |
| 同 profile 再次构建 | 两阶段均零编译、零链接 | [Stage1](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/stage1_repeat.log)、[Stage2](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/stage2_repeat.log) |
| 入口拒绝 | 9 项通过：跨 role/profile/engine、无效生产参数、无缓存 HostOnly、互斥参数及旧绕过入口；零编译且原产物/清单/marker 未变 | [记录](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/cli_rejections.json) |
| 数学/文件运行对照 | 与冻结 P0 基线比较，Stage1 5 例和 Stage2 5 例一致 | [记录](../../data/experiments/main_flow_refactor_implementation_20261021/p1b/runtime/summary.json) |

Stage1 主机 CXX 仍由 CMake/NMake 管理，Stage2 host CPP 仍由 nvcc 驱动，后者留待 P2。release 包装/marker/路径消费者完成静态和 CPU 检查，**没有执行 release 构建或打包验收**；新 Stage1 冷 profile 不代表全档重新验收。GUI 继续读取明确配置的程序路径，不递归选择最新叶目录；CPU/OpenCL/GUI 的正式 role 入口族仍属后续阶段。没有性能结论。首次 profile 测试的辅助变量类型错误已修正，失败证据保留，不影响原编译产物。

### P2a：Stage2 主机源切换普通 C++ 编译已完成

对应程序版本 `0.45.3`。[完整验证记录](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/qualification.json)保存程序、清单、源输入和各项证据；单任务合同与拥有权尚待 P2 后续子步骤，没有宣称 P2 整体完成。

生产 Stage2 的 23 个主机 CPP 改由选定 MSVC 的 cl.exe 直接编译，两个 CU 仍经 nvcc，链接也保留原 nvcc 命令。已用原 nvcc dry-run 和对象 directives 核对实际参数：保留 `/MT`、C++17、`/O2`、`/EHsc`、NTT 宏、默认布局及 legacy 默认流；未伪造 CUDA 编译宏。主机真实 runtime 调用继续包含/链接 CUDA，这不要求脱离 SDK。

MSVC 的 sourceDependencies JSON 捕获实际 SDK/STL/宏包含并进入既有内容指纹、对象摘要、最终核对及 HostOnly 门禁；不再把 CPP 交给 CUDA 前端。静态扫描仍保守检查本项目分支，项目内无法解析的引号包含仍报错；外部 SDK 的非活动引号包含允许缺失，实际启用的 SDK 依赖必须由成功的编译器记录。首轮遇到 CUDA SDK 的 NVRTC-only target 包含，编译前已明确失败；适配后成功，原始日志保留。

host_driver、主机命令与策略进入 profile，产生新 local 叶目录 `stage2/production/sm89/eda4f0d985c23039`。原 P1b profile 未重标；新 profile 首次完整建档，未复制对象或放宽跨目录身份。两个 CU 的源摘要、编译前缀及链接前缀与 P1b 一致；不要求不同输出路径下对象字节相同。

| 验证 | 结果与范围 | 证据 |
| --- | --- | --- |
| 正式 local 构建和重复 | 新 profile 23 CXX + 2 CU 成功；同 profile 再次零编译、零链接 | [构建](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/stage2_local_02.log)、[重复](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/stage2_repeat.log) |
| ABI | 新主机对象和设备对象均为 MT_StaticRelease | [对象指令](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/object_directives.log)；原参数依据 [dry-run](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/nvcc_dryrun_02.log) |
| 依赖身份 | 46 项通过，新增 MSVC SDK/宏包含、保留时间戳变异、错误源/格式/模块拒绝；29 项 profile 通过 | [身份](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/identity_final/result.json)、[profile](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/profiles/result.json) |
| 主机变更隔离 | ecm_expr.cpp 受控注释修改及原字节还原，各只编译 1 个普通 CXX，0 CU；两 CUDA 对象摘要不变。真实 CUDA 对象篡改在编译前拒绝并还原 | [记录](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/integration.json) |
| 真实生产算术对象 | GMP 对照 1536 个模数、152064 次比较，bad=0 | [记录](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/arithmetic_probe_run.log) |
| 真实生产 NTT 对象 | 368 检查、14779392 words，bad=0；Graph 8 builds/16 hits/8 invalidations；fuse 26 allocations/26 frees、残留 0 | [记录](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/ntt_probe_run.log) |
| 两程序运行对照 | Stage1 5 例、Stage2 5 例与 P0 冻结基线一致 | [记录](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/runtime_final/summary.json) |
| Stage1 恢复/发布门禁 | 冻结基线和本阶段完整档 local 程序各通过 24 项；涵盖 scalar/SIMD、Edwards GPU、Montgomery PARAM0/2/3、停止/排他锁/身份拒绝/主备恢复/发布重试 | [基线](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/usability_baseline/summary.json)、[当前](../../data/experiments/main_flow_refactor_implementation_20261021/p2a/usability_candidate/summary.json) |

同步适配既有 usability 输入：旧 -ini/--ini 与平面配置改为当前 --config、shared/stage1/workers 层次和合法 TOML 字符串，不改原断言或生产配置规则。可选 CPU Stage2 分支的 pending 工作副本额外移除已完成的 publication.order，才真实表示未发布状态；源账本不动，原 4→8 curves 断言保留。该可选跨阶段分支未执行，不列入 24 项通过范围；执行版本源码和 hash 另行保存。没有运行 release、性能采样或新增后端能力。

### P2b：唯一任务入口与结果/会话拥有权已完成

对应程序版本 `0.45.4`。[验证记录](../../data/experiments/main_flow_refactor_implementation_20261021/p2b/qualification.json)保存实际源码、程序和对象身份。当前实现说明见 [Stage1 单任务编排与资源拥有权](../architecture/STAGE1.md#单任务编排与资源拥有权)。本子步骤仍在现有驱动翻译单元内落地，没有增加空转发 task.cpp；请求/状态合同和物理模块提取继续实施，不宣称 P2 整体完成。

原 run_stage1_once 直接改为唯一 run_stage1_task，queue/direct 两处共同调用，任务时序保留。Stage1TaskResult 禁止复制、支持移动，统一清理三个独立分配；CPU Montgomery 停止时可能只有 sigmas 而 factors 为空，现在同样释放，群阶诊断提前退出也不再绕过结果清理。后端交付之前的局部分配仍由后端负责，未整体改写内存分配算法。

私有 Session 只拥有 batch/锁寿命，没有 run。先建立清理责任，再调用可能在取得锁后抛出的 begin；正常提交完成、失败保留及异常兜底都收口到同一结束入口。end_batch 移走路径字符串后清状态，失败清理不额外复制路径。已解析选项不再保存计时/总曲线数；每次执行的私有副本持有这两个可变字段，去掉 CPU 路径的三组 const 强制转换。现有全局 batch 和进度状态仍存在，尚未支持同进程多任务并行。

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 算法及任务次序核对 | CPU Edwards/Montgomery、分派正文在明确类型/const-cast 替换后逐字一致；GPU Edwards 仅结果类型名变化；task 元数据、执行和发布顺序保持 | [源码核对](../../data/experiments/main_flow_refactor_implementation_20261021/p2b/source_audit.json) |
| 正式 local 构建 | Stage1 完整档 HostOnly 成功，0 CU；Stage2 仅共享版本头引起 1 个 CXX，0 CU。与 P2a 相比 31+2 个 CUDA 对象摘要全部不变 | [Stage1](../../data/experiments/main_flow_refactor_implementation_20261021/p2b/stage1_local_hostonly.log)、[Stage2](../../data/experiments/main_flow_refactor_implementation_20261021/p2b/stage2_local_hostonly.log) |
| 重复构建 | Stage1 同 profile 0 CU、0 CXX、0 链接 | [日志](../../data/experiments/main_flow_refactor_implementation_20261021/p2b/stage1_repeat.log) |
| 恢复与发布 | 24 项现有门禁全部通过，含八种后端/曲线/CPU实现路径的停止、锁、身份拒绝和主备恢复 | [记录](../../data/experiments/main_flow_refactor_implementation_20261021/p2b/usability/summary.json) |
| 最终程序对照 | Stage1 5 例、Stage2 5 例与 P0 冻结基线一致，记录摘要与最终冻结程序匹配 | [记录](../../data/experiments/main_flow_refactor_implementation_20261021/p2b/runtime/summary.json) |

没有运行 release 或性能采样，也没有声称通过堆剖析证明整个旧执行块无泄漏；本步拥有权证据与运行退出路径范围如上。即时 factor、DONE/save、最终 submission→internal→seal→end_batch 及任务外队列推进顺序不变；具体 codec、CPU/GPU 算法块与输入处理仍待后续职责提取。

### P2c：公开请求、拥有结果与独立任务模块已完成

对应程序版本 `0.45.5`，本步完成 P2 的任务合同，P3–P7 继续实施。[验证记录](../../data/experiments/main_flow_refactor_implementation_20261021/p2c/qualification.json)冻结最终程序、实际输入和范围；前述 P2a/P2b 记录描述各自提交时的中间状态。当前合同说明见 [Stage1](../architecture/STAGE1.md#单任务编排与资源拥有权)。

公开接口位于 include/ecm/service/stage1/task.h，唯一任务流程与完整辅助闭包位于对应 src/ecm/service/stage1/task.cpp。请求分执行偏好、身份、持久化策略三组；TaskResult 以私有状态独立拥有结果，公开只读访问器、完成/停止/失败及失败来源。Selection 的原有值声明移到 include/ecm/core/stage1/selection.h；解析和名字转换仍是私有实现。CPU/GPU 算法块没有重新实现，只通过三项私有依赖接回原执行入口、构建身份和 NAF 默认窗口。driver 只组装一次，两处 task 调用共用；没有 TaskRunner、Session.run 或重复选择器。

| 验证 | 结果与边界 | 证据 |
| --- | --- | --- |
| 实际源码等价 | 11 项核对：三个执行块及 Edwards GPU 正文仅显式类型/分组路径替换，四个 host helper 整块移动；执行/发布次序与 typed-save 请求副本保持 | [源码核对](../../data/experiments/main_flow_refactor_implementation_20261021/p2c/source_audit.json) |
| 独立 CXX 合同 | 8 项通过：成功、因子、准备失败、执行异常、发布失败、仅 sigma 的停止、停止伴发布失败、损坏批次开始；真实 task/checkpoint/文件模块，替换计算入口；实际编译器依赖中 0 CUDA SDK 头、0 nvcc | [最终运行](../../data/experiments/main_flow_refactor_implementation_20261021/p2c/task_contract_final/result.json)、[构建身份](../../data/experiments/main_flow_refactor_implementation_20261021/p2c/task_contract_build_03/manifest.json) |
| 正式 local | Stage1 最终 HostOnly 成功，Stage2 成功；31+2 个 CUDA 对象与 P2b 完全一致。Stage1 重复 0 CXX、0 CU、0 链接 | [Stage1](../../data/experiments/main_flow_refactor_implementation_20261021/p2c/stage1_local_hostonly_04.log)、[Stage2](../../data/experiments/main_flow_refactor_implementation_20261021/p2c/stage2_local_hostonly.log)、[重复](../../data/experiments/main_flow_refactor_implementation_20261021/p2c/stage1_repeat.log) |
| 实际恢复/发布 | 最终分组选项接口程序 24 项通过，含八种实际后端/曲线路径 | [记录](../../data/experiments/main_flow_refactor_implementation_20261021/p2c/usability_final/summary.json) |
| 最终基线对照 | Stage1/Stage2 各 5 组与冻结 P0 一致，程序摘要匹配最终冻结件 | [记录](../../data/experiments/main_flow_refactor_implementation_20261021/p2c/runtime_final/summary.json) |
| 构建依赖图 | 真实 CMake fixture 五项通过；46 项依赖身份门禁通过 | [目标图](../../data/experiments/main_flow_refactor_implementation_20261021/p2c/owner_graph/result.json)、[身份](../../data/experiments/main_flow_refactor_implementation_20261021/p2c/identity/result.json) |

新 src/ecm 路径暴露出旧 owner graph 收集器把源目录名 ecm 误认为同名 OpenCL target 的问题：首次编译/链接完成后身份核对拒绝未解析的 cgbn_opencl.h。修正为只有链接属性引用可形成 target 边，显式 TARGET_OBJECTS 在源码和链接两处继续识别；没有放宽依赖验证。首次失败日志保留，修正后正式 HostOnly 和真实 fixture 通过。请求分组时另一次编译发现 typed-save 副本的五个旧成员访问，按同一字段映射补齐后通过；失败原始日志未覆盖。

本步尚未把 driver 缩成纯 application、拆完 writer 或建立最终 Executor；这些属于 P3/P4/P6。全局单批次限制、可选 CPU Stage2 尚未验收、只覆盖当前 local 设备的限制仍有效。未运行 release 或性能采样。

### P3a：CUDA 完整家族物理归位已完成

对应程序版本 `0.45.6`。本子步骤仅处理完整 CUDA 家族及其必要消费者，不宣称 P3 整体或职责提取完成。[验证记录](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/qualification.json)保存最终源码、对象与程序身份。原 src/cuda 的 54 个文件、kernels/cuda 的 71 个文件、include/ecm/cuda 的 15 个文件及混合主机 PRAC 计划两文件，共 142 个文件迁入 src/ecm/backend/cuda。

Stage1 按 Montgomery/Edwards 组织主机实现、设备算法、独立实例；PRAC 保留完整可选家族。Stage2 的 15 个主机引擎、独立算术 CU 和 NTT CU 边界不变，设备头分别进入 stage2/device 和 ntt/device；CGBN 与 Goldilocks 原语单独归私有 arithmetic。launch/types/tiers/descriptor 不再留在公开 include 树。旧 backend.cpp 归 factory.cpp，但保留现有函数和组装方式，没有引入新工厂框架。核心 driver 仍通过过渡期私有接口消费执行实现；最终 Executor/prepare 和 writer 提取属于后续阶段。

| 核对 | 结果与边界 | 证据 |
| --- | --- | --- |
| 物理迁移 | 142 项一一对应，原三处 CUDA 实现目录无文件；原始 BOM、全部非 include 正文、include 实际指向、源组次序及逐源 CMake 属性保持 | [完整核对](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/layout_final/result.json) |
| 设备算术 | Stage1 55 项设备闭包、四组原始算术正文；Stage2 93 项算术函数保持；21+7 实例 TU 不合并 | [Stage1](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/stage1_structure/result.json)、[Stage2](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/stage2_structure/result.json) |
| 主机及实验合同 | 94 项 NTT 函数/方法，32 个 Stage2 模块清理证明，27 项实验迁移和三项非法包含拒绝，五项探针合同通过；只恢复已核对的 include 拼写用于旧基线比较 | [NTT](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/ntt_host/result.json)、[清理](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/p4_cleanup_02/result.json)、[隔离](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/p4_isolation_03/result.json)、[探针](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/p4_probe_contracts/result.json) |
| 构建身份 | 46 项通过，包括 PS5.1/PS7 内容指纹一致性 | [记录](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/identity_cross/result.json) |
| 正式 local | 完整 Stage1 与生产 Stage2 构建通过；路径身份变化进行必要重编。随后两阶段重复构建均 0 CXX、0 CU、0 链接 | [Stage1](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/stage1_local_02.log)、[Stage2](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/stage2_local_fresh.log)、[Stage1重复](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/stage1_repeat.log)、[Stage2重复](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/stage2_repeat.log) |
| 实际对象独立对照 | Stage2 GMP：1536 模数/152064 比较，bad=0；NTT Graph：368 检查/14779392 字，bad=0，26 次分配/释放，live=0 | [GMP](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/arithmetic_run.log)、[NTT](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/ntt_run.log) |
| 最终程序 | 24 项恢复/发布，完整位宽工厂查询，以及两阶段各 5 组 P0 对照通过 | [可用性](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/usability/summary.json)、[工厂](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/registry/result.json)、[对照](../../data/experiments/main_flow_refactor_implementation_20261021/p3a/runtime/summary.json) |

原 Stage2 目录的 MSVC 依赖记录绑定旧源路径，正常拒绝搬迁后的同名对象复用；保留旧目录，正式输出改到显式 local 叶目录 build_cuda_local/stage2/production/sm89/p3a-eda4f0d985c23039，profile 本身仍为 eda4f0d985c23039，未复制对象或改写身份。首次 Stage1 搬迁工具去掉两份 CUDA 文件的 BOM，复现仓库已记录的 nvcc/GBK 注释吞行，导致 CHECKPOINT_VERSION 未定义；终止该失败轮次，恢复两份原有 BOM 后重新核对所有失效对象。首次失败日志和中止结果保留，未修改宏、数学或编译标准绕过错误。

当前正式架构/使用文档的 CUDA 链接和离线图谱已同步；前期报告仍保留其冻结源码摘要，只将跳转链接映射到当前文件。core/CPU/application/配置生成器/OpenCL/GUI 的剩余物理归位继续 P3，私有混合实现的职责深化继续 P4–P6。没有新增 PRAC 运行资格、release、多架构或性能结论。

### P3b：主机、核心、CPU 与存储家族归位已完成

对应程序版本 `0.45.7`。本子步骤迁移 120 个文件：原 core 98 项、CPU 15 项和共享兼容文件 7 项；与 P3a 合计 262 项。原 src/core、src/cpu 已无文件。当前实现正文保持，只调整物理所有者、引用和构建清单；没有新增转发文件或空层。[最终资格记录](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/qualification.json)保存源码和程序摘要。

driver、独立 Stage2 入口及参数加载归 application；曲线、Stage1 标量及 Stage2 几何归 core；CPU 算术、SIMD 和曲线执行归 backend/cpu；CUDA 硬件内存规划与执行约束归 backend/cuda。checkpoint、save、results、queue、transfer、cache、profile 归 storage；原子发布、指纹、日志和版本归 platform。混合调优保留 service/planning/legacy，PARI 因子拆分归 service/stage2/legacy。配置生成物移入 application/configuration/generated，并同步唯一生成器及清单。

公开 include 与对应模块使用一致路径：Stage1 scalar/selection/task、Stage2 probability、platform logging/version、application runtime 和过渡执行 C ABI。它们不包含 CUDA SDK 或私有 CUDA 类型。probability 目前仍为原有完整头实现，runtime 仍保留全局配置，inline writer、driver 的 CPU/GPU/队列及 PARI 分支也仍存在；这些是 P4–P6 的深化对象，不能因迁移而宣称已完成接口与实现分离。

| 核对 | 结果及范围 | 证据 |
| --- | --- | --- |
| 物理与依赖 | 合计 262 项正文/BOM、211 个包含消费者、785 条实际引用；TU 顺序和逐源编译属性保持。唯一生成测试 fixture 引用按已有生产脚本核对 | [布局检查](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/layout_final_02/result.json) |
| 配置及头所有权 | 默认值、范围、JSON、默认/示例配置及生成参考内容不变；8 个公开主机头不泄漏 CUDA；CPU Stage2 的外部 libecm.h 保持真实所有者 | [主机检查](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/host_layout/result.json) |
| local 构建 | Stage1 选用 384/1024 两档，覆盖全部既有恢复/发布输入；Stage2 生产 local 构建通过。两者重复构建均 0 CXX、0 CU、0 链接 | [Stage1](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/stage1_local.log)、[Stage2](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/stage2_local.log)、[重复1](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/stage1_repeat.log)、[重复2](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/stage2_repeat.log) |
| 主流程 | 原 24 项通过；真实 CPU Stage2 联动共 27 项通过，覆盖跨阶段提交与 pending Stage2 NF 恢复；两阶段各 5 组 P0 对照通过 | [24项](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/usability/summary.json)、[27项](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/usability_cpu/summary.json)、[对照](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/runtime/summary.json) |
| 独立任务 | 8 种成功/失败/停止/发布组合通过；实际编译依赖为 0 nvcc、0 CUDA SDK include、0 CUDA SDK 头 | [合同](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/task_contract/result.json)、[构建清单](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/task_contract_build/manifest.json) |
| 其他消费者 | GUI 构建及 33 项无窗口自检、26 项结果读取、85 项日志解析通过；CPU Stage2 用真实 libecm 构建，GWPoly 关闭 | [GUI](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/gui_build.log)、[结果](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/gui_results.log)、[日志](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/gui_log_parse.log)、[CPU](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/cpu_stage2_build.log) |
| 构建与隔离合同 | 46 项内容身份核对、32 个 Stage2 清理模块、27 项实验隔离及三项负向边界通过；选定两档工厂查询 20 项通过 | [身份](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/identity/result.json)、[清理](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/p4_cleanup/result.json)、[隔离](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/p4_isolation/result.json)、[工厂](../../data/experiments/main_flow_refactor_implementation_20261021/p3b/registry/result.json) |

完整 32 档 local 已在 P3a 冻结；后续组织调整以覆盖既有输入的 384/1024 档验证，P7 再做完整位宽终验。本阶段未运行 CUDA release，不新增性能、多架构、PRAC 或 GWPoly 资格。OpenCL 主机、运行资源与 GUI 文件的物理归位继续 P3c；实际 CUDA 同步、回读和资源管理保持其 CUDA 依赖。

### P3c：OpenCL 主机与生产资源归位已完成

对应程序版本 `0.45.8`。新增隔离的 OpenCL local 入口，在迁移前先构建 P3b 的实际 OpenCL 基线；随后将 18 个主机/头文件及 172 份生产资源统一迁入 backend/opencl，共 190 项。与 P3a/b 合计 452 项。曲线 engine/entry/packing、自检、路径描述与组装、context/program/cache、设备枚举各自保留完整职责与原有函数。limb24 只随现有工具/Android 消费归位，未新增桌面生产执行路径。

CUDA 以外的生产 .cl 全部进入 resources；原 bench/mont.cl 作为生产自检迁入 diagnostics/mont.cl。旧逻辑名称优先按原加载顺序查找，缺失后才映射到诊断位置，保留自定义旧资源根。自检条件和失败处理不变，缓存继续使用设备/厂商/设备版本/驱动版本、完整源码与编译选项，未改变算法或键。Android 保持已有逻辑资产根，适配来源与 CMake 路径，未进行 Android 构建。

新 build_opencl_local.ps1 使用普通 CXX、原有 /MD 优化合同和独立 build_opencl_local/stage1/profile 叶目录，关闭 CUDA 编译器、GUI、工具与独立 CPU Stage2；profile 拒绝用途/配置混用，清单记录程序与资源。生产资源由 runtime 库依赖的 CMake 目标同步到 opencl_resources，资源修改无需编译计算对象。现有 OpenCL 工具的目标另同步纯 bench 输入，保留共享 loader 的可用性；这些输入不出现在生产 local 的 172 项清单中。

| 核对 | 实际结果与范围 | 证据 |
| --- | --- | --- |
| 全集与编译属性 | 合计 452 项一一对应、220 个 include 消费者、801 条引用；数学正文/BOM、TU 顺序、逐源编译属性保持。仅对 loader 的路径/诊断别名及资源 CMake 规则列出精确允许变更 | [完整检查](../../data/experiments/main_flow_refactor_implementation_20261021/p3c/layout_final_02/result.json) |
| OpenCL local | 迁移前基线和候选均以 33 条普通 CXX 命令构建；未启用 CUDA 编译器。候选重复构建 0 CXX、0 CU、0 链接 | [基线](../../data/experiments/main_flow_refactor_implementation_20261021/p3c/opencl_baseline.log)、[候选](../../data/experiments/main_flow_refactor_implementation_20261021/p3c/opencl_candidate.log)、[重复](../../data/experiments/main_flow_refactor_implementation_20261021/p3c/opencl_repeat_final_02.log) |
| 生产资源与计算 | 172 项随附资源逐字节匹配；当前设备 PARAM3、N=2^347−1、sigma=20260922、B1=B2=1000、4 曲线的完成点与结果保持，结果 CRC 有效；迁移前两份缓存被候选直接命中 | [核对](../../data/experiments/main_flow_refactor_implementation_20261021/p3c/opencl_resources_final_02/result.json) |
| CUDA 增量 | 严格 HostOnly 构建成功；Stage1 仅重编 3 个 CXX，Stage2 仅重编 1 个 CXX；31+2 个设备对象摘要保持 | [Stage1](../../data/experiments/main_flow_refactor_implementation_20261021/p3c/cuda_stage1_host.log)、[Stage2](../../data/experiments/main_flow_refactor_implementation_20261021/p3c/cuda_stage2_host.log)、[最终资格](../../data/experiments/main_flow_refactor_implementation_20261021/p3c/qualification.json) |
| 主流程对照 | CUDA 两阶段各 5 组 P0 固定输入对照通过；本阶段无性能采样 | [对照](../../data/experiments/main_flow_refactor_implementation_20261021/p3c/runtime/summary.json) |

本轮未扩大 OpenCL 参数族、位宽或设备资格；未运行 release、Android 或性能验收。OpenCL 默认资源根绑定构建叶目录，部署到其他位置使用显式 --kernel-root；没有声称已有可任意移动的 OpenCL release 包。GUI 按既定 P6 完成物理归位和 Controller 提取，P3 不新增 GUI 转发层。P4 继续提取 CPU/GPU Stage1 host workflow、QueueWorkflow 及配置/数学/存档/结果/screen 的普通 .cpp 实现，P5 继续收口 Stage2。

### P4a：共享标量与进度实现隔离已完成

对应程序版本 `0.45.9`。先解除两条真实 CU 包含链：Edwards runtime 经 internal 头包含完整 NAF 构造，Montgomery adapter 包含完整 screen/速度统计。NAF 的 read、构造和 digit 并入现有 core/stage1/ecm_stage1_exp.cpp，公开 scalar.h 合并原两份标量/NAF 声明；没有新增 NAF 转发或标量 TU。旧私有 NAF 头删除。共享 screen 与 SpeedWindow 的五个方法进入 platform/stage1_progress.cpp，公开头只留声明和原私有字段布局；旧 legacy 进度头删除，没有新增 Pimpl 或改变分配行为。

CPU/GPU 数学、NAF 的非零/符号码两位图、50 样本窗口、格式文本、调用时序和错误字符串保持。声明头位置改变导致首次两个 host runtime CU 合法失效；后续代表性函数体改写只重编 CXX。Montgomery 对进度头的显式 OBJECT_DEPENDS 已移除，实际 include 依赖继续保留。共享实现纳入普通 CXX 清单及 source-build 身份；独立 Edwards probe/worker 取同一个标量实现，host-wait 已有清单本就包含标量 CPP，补齐其漏写的 instances/ 实例路径。本轮不新增 host-wait 计时结论。

| 核对 | 结果及范围 | 证据 |
| --- | --- | --- |
| 实现与布局 | 8 个函数正文逐 token 与 P3c 冻结版本比较，保留字符串；NAF/SpeedWindow 布局不变，原 LCM 实现正文保持 | [源码合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/source_contract_final/result.json) |
| CPU 数学与 screen | 46574 个 NAF 用例，603364 个负数字、126047 个跨字编码，窗口 2..12；11 项 screen/速度统计通过 | [NAF](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/naf_build.log)、[进度](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/progress_run.log) |
| 设备正文 | 55 项设备闭包和四组原始算术正文保持，21+7 独立实例不合并 | [检查](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/device_structure/result.json) |
| 实现改动的实际增量 | NAF read 与 screen filled 计算各作等价 cast 改写；严格 HostOnly 仅重编 2 CXX、0 CU、链接一次，31 个 CUDA 对象摘要保持；随后恢复原函数正文并重建 | [构建](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/implementation_hostonly.log)、[身份](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/implementation_dependency_result.json) |
| 最终 local | 最终无修改重复构建为 0 CXX、0 CU、0 链接；Stage2 HostOnly 仅版本相关 1 CXX，两个设备对象保持 | [Stage1](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/stage1_repeat.log)、[Stage2](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/stage2_hostonly.log) |
| 独立主机合同 | 8 项任务合同通过；真实 task/持久化、标量和进度用普通 CXX 编译，实际记录为 0 nvcc、0 CUDA SDK include、0 CUDA SDK 头 | [运行](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/task_contract/result.json)、[依赖记录](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/task_contract_build/manifest.json) |
| 主流程 | 27 项 CPU/GPU/独立 CPU Stage2 恢复与发布通过，CUDA 两阶段各 5 组 P0 对照通过 | [27项](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/usability/summary.json)、[对照](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/runtime/summary.json) |
| OpenCL 消费 | 普通 CXX local 构建、实际 PARAM3 完成点/结果与 CRC、172 资源和两份旧缓存命中通过 | [构建](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/opencl_local.log)、[运行](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/opencl_resources/result.json) |

[最终资格记录](../../data/experiments/main_flow_refactor_implementation_20261021/p4a/qualification.json)冻结全部源码和程序身份。P3 的完整物理移动正文检查属于迁移阶段证据；进入获授权的实现提取后，新源码合同核对实际提取，source_layout 的 current_overrides 只供测试读取当前位置，生产不加载。没有通过还原丢失实现或放宽数学比较伪装迁移后正文不变。

P4 尚未整体完成：下一步是共享主机准备、完整 CPU 执行闭包、GPU host workflow 和唯一分派，再深化 task 与 QueueWorkflow；仍需提取 Montgomery 构造/结果/checkpoint 和各 writer。后端现有 factor/DONE 与发布失败分类先后必须逐路径保持，不统一 CPU/GPU 的旧差异。实际 CUDA 同步/回读不要求脱离 CUDA；完整位宽终验留 P7，不运行 release。

### P4b：完整 CPU Stage1 执行闭包与共享准备已提取

对应程序版本 `0.45.10`。driver 从 4469 行缩至 2635 行；约 1800 行 CPU 状态、线程、ISA 选择、恢复与执行逻辑整体进入 backend/cpu/stage1/legacy_executor.cpp。两个实际入口接收原参数并填充原结果，不新增类、同参转发或第二套 task。affinity 文本解析保留 application，CPU 模块只应用已有集合。新文件使用基线 CXX，三个 SIMD 实现原 ISA 设置保持。

九个共用函数集中到 service/stage1/preparation.cpp：标量/缓存入口、随机 sigma、save 前缀与格式、固定 sigma 范围、目录、主机时钟和终端检测。随机状态与调用次数不变。首次普通 CXX 构建发现 GPU 文本实现也消费时钟/终端函数，且应用仍需 CPU 默认 NAF 窗口声明；补齐共享声明和实际头后重建通过，没有修改算法、门限或运行失败分类。首次失败日志保留在 p4b/stage1_hostonly.log。

源码合同不仅比较入口，还逐 token 比较完整 CPU 闭包中的状态、结构体、预处理分支、函数正文和先后顺序；九个共享函数正文与上一提交一致。CPU Edwards 标量、IFMA 与 Montgomery 原有 GCD→factor→DONE/save 次序分别保持；CPU 发布失败的既有 Preparation 分类也保持，不借重构改变队列失败策略。

- [源码合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4b/source_contract/result.json)：完整 CPU 闭包与九函数保持，0 新 wrapper 类。
- [独立 CPU 编译](../../data/experiments/main_flow_refactor_implementation_20261021/p4b/cpu_compile/manifest.json)：两个实际 CPP，0 nvcc、0 CUDA SDK include、0 SDK 头、无 AVX512 编译选项；此项为编译验证，运行行为由生产 worker 验证。
- [任务合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4b/task_contract/result.json)：8 项通过，真实 task/持久化支持仍为无 SDK 普通 CXX。
- [完整运行检查](../../data/experiments/main_flow_refactor_implementation_20261021/p4b/usability/summary.json)：27 项 CPU/GPU 恢复、停止、队列、发布及独立 CPU Stage2 交接通过。
- [P0 对照](../../data/experiments/main_flow_refactor_implementation_20261021/p4b/runtime/summary.json)：CUDA 两阶段各 5 组通过，保存点、因子和结果语义保持。
- [OpenCL](../../data/experiments/main_flow_refactor_implementation_20261021/p4b/opencl_resources/result.json)：36 个普通 CXX 编译项，172 资源，完成点/结果与 CRC 保持，两份旧缓存命中。
- [重复构建](../../data/experiments/main_flow_refactor_implementation_20261021/p4b/stage1_repeat.log)：0 CXX、0 CU、0 链接；31 个 Stage1 与 2 个 Stage2 CUDA 对象摘要与 P4a 相同。

[最终资格记录](../../data/experiments/main_flow_refactor_implementation_20261021/p4b/qualification.json)冻结三个生产 worker、依赖记录与本阶段输入。未运行 release、性能采样或全 32 位宽终验。后续继续 GPU host workflow、唯一分派、配置/构造/codec/writer 和 QueueWorkflow；preparation 对 application lowercase helper 的临时依赖也待配置实现收口时消除。

### P4c：GPU 主机流程与唯一分派已提取

对应程序版本 `0.45.11`。driver 从 2635 行缩至 2390 行，不再文本包含 GPU Edwards inl，也不再包含 CPU/GPU 执行正文。task 的真实 execute 依赖直接绑定 service/stage1/execution.cpp 的 execute_stage1；该函数按原次序选择 GPU Edwards、CPU Montgomery、CPU Edwards 或链接时选择的 GPU Montgomery。共享 Montgomery 主机流程保留完整参数准备、执行、factor 与 save 规则，没有新 factory 类、执行注册框架或 driver 同参留壳。

GPU Edwards 的完整状态、初始化/恢复、进度回调和完成发布进入 backend/cuda/stage1/edwards/host_workflow.cpp，普通 CXX 编译，私有头仅声明真实入口。生产主机流程只加入 STAGE1_CUDA，不污染被独立 GPU probe/worker 复用的 STAGE1_EDWARDS 低层清单。checkpoint codec 私有头仍供主机流程和容器验证使用。CMake 中原为文本包含补偿的 driver 强制 GPU OBJECT_DEPENDS 删除，实际 include 依赖保持。

首次 CXX 构建发现队列归一化仍使用四个共享准备函数，补齐实际 using 声明后通过；失败日志保留。P4b 搬迁保留的一处尾随空格在本轮清理，完整 CPU token 与行为保持。独立身份测试首次给出的备用 PowerShell 路径不存在，改用已安装运行时路径重新完成 46 项；未放宽测试或生产验证。

- [完整源码合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4c/source_contract/result.json)：Montgomery 主机正文、Edwards 完整状态/回调及分派次序与 P4b 保持。
- [真实独立主机编译](../../data/experiments/main_flow_refactor_implementation_20261021/p4c/host_compile/manifest.json)：准备、CPU 执行、唯一分派和 GPU Edwards 主机流程四个 CPP，0 nvcc、0 CUDA SDK include、0 SDK 头；不编译实际设备运行代码。
- [构建身份](../../data/experiments/main_flow_refactor_implementation_20261021/p4c/identity_02/result.json)：46 项依赖、命令、对象及篡改拒绝检查通过。
- [运行检查](../../data/experiments/main_flow_refactor_implementation_20261021/p4c/usability/summary.json)：27 项 CPU/GPU 恢复、停止、队列、发布及独立 CPU Stage2 交接通过。
- [P0 对照](../../data/experiments/main_flow_refactor_implementation_20261021/p4c/runtime/summary.json)：CUDA 两阶段各 5 组通过；[OpenCL](../../data/experiments/main_flow_refactor_implementation_20261021/p4c/opencl_resources/result.json)的完成点/结果与 CRC、172 资源、两份旧缓存保持。
- [重复 local 构建](../../data/experiments/main_flow_refactor_implementation_20261021/p4c/stage1_repeat.log)：0 CXX、0 CU、0 链接；Stage1 31 个、Stage2 2 个 CUDA 对象摘要与 P4b 相同。

[最终资格记录](../../data/experiments/main_flow_refactor_implementation_20261021/p4c/qualification.json)冻结输入、三个生产 worker 与依赖。未运行 release、性能采样或全 32 位宽终验。接下来仍须提取 Montgomery adapter 的构造/结果/codec、配置与 writer，并将队列协议集中到 QueueWorkflow。新审查确认当前队列只在启动时加载 TOML，没有逐任务热重载；此前设计中“当前配置重读”的说法已纠正，后续提取应保留启动快照。

### P4d：Montgomery 主机数学、结果与持久化输出已隔离

对应程序版本 `0.45.12`。adapter.cu 从约 1700 行缩至 994 行；556 行 host_math.cpp 集中完整 param0/2/3 构造、param2 Jacobian、转换、标量打包、np0 和结果处理；227 行 host_io.cpp 集中 ladder V5、resident/PRAC codec、路径、TTY/screen/节流和 CSV。只暴露实际跨 TU 入口与原载荷类型，内部 helper 保留私有，不新增执行类或 logger 框架。

两份 CPP 只需标准库、GMP、已有主机日志/持久化声明，没有 CUDA/CGBN SDK 类型。outputf/test_verbose 仍使用 factory 的唯一 verbose 状态。FULL_BUILD 的旧诊断条件与 NDEBUG 断言语义保持，find_np0 的 mpz_invert 仍在断言之外。CARRY_BITS、真实 tier/dispatch 与设备资源保持运行层；没有复制 CGBN report ABI、伪造 __CUDACC__ 或修改第三方源码。

global event 与标量上传先于恢复/构造、checkpoint 在回读之后编码、resident 完成后 EXPORT 再处理普通域结果的顺序保持。源码审查收紧 resident restore 的 sigma 输出合同，使回写继续发生在恢复日志之前；随后重编、重新冻结和回归。首次提取辅助脚本遇到原文件 LF/CRLF 匹配差异时尚未写生产文件，修正读取匹配后继续，未修改 BOM 或设备算术。实际临时等价改写已恢复，并清理搬迁来的尾随空格。

- [最终源码合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4d/source_contract_final_02/result.json)：22 项函数检查、两种 checkpoint 字段布局、ladder/resident 执行次序及资源结构保持。
- [设备闭包](../../data/experiments/main_flow_refactor_implementation_20261021/p4d/device_structure/result.json)：55 项设备闭包、四组原始数学正文与 21+7 实例边界保持。
- [实际增量](../../data/experiments/main_flow_refactor_implementation_20261021/p4d/implementation_dependency_result.json)：数学解包和 screen 计算各作等价 cast 改写，只重编 2 CXX、0 CU、链接一次，31 个 CUDA 对象摘要保持；随后恢复代表函数原 token。声明层提取首次合法更新 adapter，一个 CU，30 个其他设备对象保持。
- [无 SDK 独立编译](../../data/experiments/main_flow_refactor_implementation_20261021/p4d/host_compile_final_02/manifest.json)：真实准备、CPU、分派、GPU Edwards、Montgomery math/IO 共六个 CPP，PRAC 主机 codec 开启编译，0 nvcc、0 SDK include、0 SDK 头。此项为编译资格，不宣称新 PRAC GPU 运行或性能资格。
- [最终运行检查](../../data/experiments/main_flow_refactor_implementation_20261021/p4d/usability_final_02/summary.json)：27 项 CPU/GPU 恢复、停止、发布、队列与独立 CPU Stage2 交接通过；[两阶段 P0 对照](../../data/experiments/main_flow_refactor_implementation_20261021/p4d/runtime_final_02/summary.json)各 5 组通过。
- [最终 OpenCL](../../data/experiments/main_flow_refactor_implementation_20261021/p4d/opencl_resources_final_02/result.json)：完成点、结果/CRC、172 资源及两份旧缓存保持；[最终重复 local](../../data/experiments/main_flow_refactor_implementation_20261021/p4d/stage1_repeat_02.log)为 0 CXX、0 CU、0 链接。

[最终资格记录](../../data/experiments/main_flow_refactor_implementation_20261021/p4d/qualification.json)冻结最终源码和三个 worker。未运行 release、性能采样或全位宽终验。P4 继续收口配置/selection/writer、环境策略与 QueueWorkflow，不能把本子阶段写成整个 Stage1 架构已完成。

### P4e：共享选择与轻量默认值合同已收口

对应程序版本 `0.45.13`。11 个选择词汇/合法性/名称函数从应用私有 inline 头整体进入 core/stage1/selection.cpp，公共 selection.h 只留类型与声明，旧头删除。task 与 preparation 不再包含应用配置实现。Stage1 CUDA/OpenCL 和 Stage2 的真实配置消费者链接同一份普通 CXX 实现；独立选择测试的 CMake 目标同步接入，没有转发头、helper TU 拆分或配置服务类。

生成器从原 JSON 同一次生成 platform/generated/configuration_defaults.h；完整应用绑定引用它，task 只读取这份无 TOML 解析依赖的常量。默认值声明、绑定、范围、别名、配置模板、示例和逐键参考逐 token/文本对照保持。首次独立编译发现默认值中的 GUI 数组需要显式 std::array 头，修正生成器并重新生成。生成器按既有 .gitattributes 使用 LF 原始指纹，重新生成哈希清单后两阶段重复构建通过。未直接修改生成默认值或复制业务规则。

- [源码合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4e/source_contract_final/result.json)：11 函数、Selection 布局、默认值与完整绑定保持；6 份定义/模板/参考文件保持，service 对应用配置实现包含数为 0。
- [配置运行检查](../../data/experiments/main_flow_refactor_implementation_20261021/p4e/toml.log)：真实 TOML 解析与生成默认值通过；[选择目标构建](../../data/experiments/main_flow_refactor_implementation_20261021/p4e/selection_build.log)和独立执行通过。该目标构建复用了 GUI 构建环境脚本，但未重建或重新验收 GUI。
- [任务合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4e/task_contract/result.json)：8 项真实 task/持久化检查通过；[实际依赖](../../data/experiments/main_flow_refactor_implementation_20261021/p4e/task_contract_build/manifest.json)为 0 nvcc、0 SDK 头，且无 TOML/完整配置绑定头。
- [完整运行](../../data/experiments/main_flow_refactor_implementation_20261021/p4e/usability/summary.json)：27 项恢复、停止、队列、发布与独立 CPU Stage2 交接通过；[两阶段 P0 对照](../../data/experiments/main_flow_refactor_implementation_20261021/p4e/runtime/summary.json)各 5 组通过。
- [OpenCL](../../data/experiments/main_flow_refactor_implementation_20261021/p4e/opencl_resources/result.json)：38 个普通 CXX 编译项，172 资源、点位/结果与 CRC、两份旧缓存保持。
- [Stage1 重复构建](../../data/experiments/main_flow_refactor_implementation_20261021/p4e/stage1_repeat.log)和 [Stage2 重复构建](../../data/experiments/main_flow_refactor_implementation_20261021/p4e/stage2_repeat.log)均为 0 编译、0 链接；31+2 个 CUDA 对象与 P4d 最终版本相同。

[最终资格记录](../../data/experiments/main_flow_refactor_implementation_20261021/p4e/qualification.json)冻结源码、生成/构建输入和三个 worker。未运行 release、性能或全位宽终验。结果 writer、环境策略、QueueWorkflow 与私有 legacy 执行载荷继续按后续 P4 子步骤收口。

### P4f：结果编码、ledger 与发布实现已集中

对应程序版本 `0.45.14`。四份结果实现头合并为 554 行 storage/results/publication.cpp；公开 publication.h 只留原 Submission 数据布局和实际入口，Windows/POSIX 文件锁、CRC、解析、读写与迁移不再通过头文件进入消费者。Stage1 CUDA/OpenCL 和 CPU/CUDA Stage2 全部链接同一份普通 CXX 实现。没有给 SubmissionLock 增加 Pimpl/分配，也没有复制锁或状态。

task 中直接取得 SubmissionLock、prepare_internal_ledger 和 read_submission_entry 的完整已完成曲线扫描，移到 writer 的 contains_stage1_curves(path,N,B1,curve,param,sigmas)。GPU 两条旧载荷导入路径直接调用该入口；原 task 同参 helper 和 legacy 声明删除，未新增转发层。driver 不再包含结果编码实现，版本声明改为显式包含。

源码合同完整比较 33 个原函数、Submission、锁和 entry 布局及 receipt 扫描正文。默认空字符串参数含花括号，首次源码检查误将它当作函数体；本轮测试改为在平衡参数列表之后定位定义，校验真实全文。断言、输出顺序、异常字符串、文件/CRC 协议均未修改。

阶段中保留的失败与核对：结果 fixture 补齐其自身 filesystem 头；原硬链接追加检查在沙箱被拒绝后，于正常本机环境完成，未删除或放宽检查。首次 worker 构建补齐 driver 原先间接取得的版本声明。Stage2 的旧 MSVC 依赖集合在正文移走后减少，guard 把不再使用的 corecrt_startup.h 报成内容变化；只读核对确认实际 SHA256 与旧记录一致。保留旧记录及新编译器依赖，再用正式脚本重编建立当前记录，重复构建复用通过，没有修改对象/依赖 sidecar 或绕过 guard。该误报条件列入构建工具 TODO，不能写成 SDK 真正变化。

- [源码合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4f/source_contract_02/result.json)：33 函数与原布局保持；公开 OS 句柄/实现正文为 0。
- [结果生命周期](../../data/experiments/main_flow_refactor_implementation_20261021/p4f/results_03/result.json)：23 项顺序、F/NF、聚合、CRC、硬链接追加、partial append、重放和损坏拒绝通过。
- [任务合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4f/task_contract/result.json)：8 项通过；[真实编译依赖](../../data/experiments/main_flow_refactor_implementation_20261021/p4f/task_contract_build/manifest.json)包含实际 publication.cpp，0 nvcc、0 CUDA SDK 头。
- [完整运行](../../data/experiments/main_flow_refactor_implementation_20261021/p4f/usability/summary.json)：27 项恢复、停止、队列、发布及新编译真实 CPU Stage2 交接通过；[P0 对照](../../data/experiments/main_flow_refactor_implementation_20261021/p4f/runtime/summary.json)两阶段各 5 组通过。
- [OpenCL](../../data/experiments/main_flow_refactor_implementation_20261021/p4f/opencl_resources/result.json)：39 个普通 CXX 项、172 资源，点位/结果/CRC 及两份旧缓存保持。
- [Stage1 重复构建](../../data/experiments/main_flow_refactor_implementation_20261021/p4f/stage1_repeat.log)和 [Stage2 重复构建](../../data/experiments/main_flow_refactor_implementation_20261021/p4f/stage2_repeat.log)均无重编或链接；31+2 个 CUDA 对象与 P4e 相同。[依赖误报证据](../../data/experiments/main_flow_refactor_implementation_20261021/p4f/dependency_guard/result.json)保留。

[最终资格记录](../../data/experiments/main_flow_refactor_implementation_20261021/p4f/qualification.json)冻结四个生产 worker 及输入。未运行 release、性能采样或全位宽终验。后续继续 QueueWorkflow、环境/tune 策略与私有执行合同，P4 仍未整体完成。

### P4g：应用输入与唯一队列完成协议已分离

对应程序版本 `0.45.15`。driver 从上一阶段 2390 行减至 1419 行，保留 CLI/单次输入、依赖组装、模式选择、展示和退出；9 个应用输入函数落入 stage1_input.cpp。queue_source.cpp 持有启动配置快照、两种任务语法、路径/选择规范化和 Prime95 按实际任务惰性初始化；它向 QueueWorkflow 提供一个 next 回调，不调用 task，也不推进完成队列。

[公开队列合同](../../include/ecm/service/work/queue_workflow.h)提供拥有模数、选项和字符串的 QueueItem，转交配置只借用至同步流程返回；QueuePaths 与 NextQueueTask 不携带完整应用 Settings。唯一 [queue_workflow.cpp](../../src/ecm/service/work/queue_workflow.cpp)持有外循环、task 调用和完成协议：先检查准备/发布失败，展示因子，再处理停止/执行失败；成功按 Stage2 pending → finished → 原票据删除/替换 → save 同步 → Stage2 flush → Prime95 delivery 的顺序提交。文件机制继续由现有 QueueStore/transfer 实现提供。无 TaskRunner、QueueRunner 类或重复 factory，单任务恢复/发布/封存仍只属于 task。

保留启动 Stage2 pending flush、save skip 预读和 Prime95 lazy/recovery 的原顺序；启动配置仍只读一次。旧解析错误的 MarkError 写入失败会结束循环并返回 0，本轮明确保留，没有把组织重构混成失败策略变更。普通执行失败标记原行；准备/发布失败与停止保留原行。因子成功不生成本地自动 Stage2，Prime95 原交接政策保持。

- [源码合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4g/source_contract_03/result.json)：9 个输入函数、7 个 source helper、两种完整解析流程、启动/Prime95 顺序和完整完成协议对照一致。
- [独立队列合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4g/queue_contract_02/result.json)：13 用例覆盖空队列、读错、错误标记/写失败、完成、因子、自动 Stage2、准备/发布失败、读前停止/执行中停止、finished 写失败与票据不匹配；只替换算术，运行真实 task/checkpoint/publication/queue 文件模块。实际 MSVC 依赖无 CUDA SDK。
- [9 个主机 CPP 编译](../../data/experiments/main_flow_refactor_implementation_20261021/p4g/host_compile/manifest.json)无 nvcc、CUDA SDK 路径或头，包含应用 queue source 和 service workflow；PRAC codec ON 仅是主机编译覆盖，不新增 GPU 运行资格。
- [Prime95 实际队列验收](../../data/experiments/main_flow_refactor_implementation_20261021/p4g/p95_transfer_02.log)83 断言通过；旧 fixture 改为当前 TOML/绝对隔离路径，保留路由、锁、AID 与恢复断言并增加进程返回检查。首次旧路径失败证据保留。[Stage2 transfer](../../data/experiments/main_flow_refactor_implementation_20261021/p4g/stage2_transfer.log)修正旧构建路径后通过。
- [27 项运行](../../data/experiments/main_flow_refactor_implementation_20261021/p4g/usability/summary.json)、[5+5 基线](../../data/experiments/main_flow_refactor_implementation_20261021/p4g/runtime/summary.json)、[OpenCL 172 资源/2 缓存命中](../../data/experiments/main_flow_refactor_implementation_20261021/p4g/opencl_resources/result.json)通过。CPU Stage2 互操作使用已冻结的 0.45.14 worker，本轮未重建它。

首次主程序编译暴露新增 include 留下中段 BOM 的编辑错误，已保留原始开头 BOM 并删除中段重复项；独立队列 fixture 补齐应用通常提前创建的同步目录，未修改生产文件机制。失败证据与修复记录均保留。[最终资格](../../data/experiments/main_flow_refactor_implementation_20261021/p4g/qualification.json)冻结三个 worker；31+2 个 CUDA 对象未变，重复构建 0 CXX/0 CU/0 link。后续继续环境/tune 与私有执行合同审查，P4 尚未整体结束；未运行 release、性能采样或全位宽终验。

### P4h：Montgomery 环境策略与窗口输出已隔离

对应程序版本 `0.45.16`。[host_policy.cpp](../../src/ecm/backend/cuda/stage1/montgomery/host_policy.cpp)集中算法、PRAC register/variant/TPI/constants、计划缓存目录、切片目标、采样和窗口参数解析；在原调用位置读取环境，不提前缓存或改变优先级/错误顺序。host_policy.h 是后端私有数据/声明，domain.h 只收纳原 launch/constants 两处的相同枚举值，不引入 CUDA 类型。真实 kernel 选择、资源、启动、同步与回读仍留 CUDA。窗口诊断 CSV 写入归现有 host_io.cpp，GPU export → 同步 → 回读 → CSV 顺序保持。

- [源码对照](../../data/experiments/main_flow_refactor_implementation_20261021/p4h/source_contract_02/result.json)：8 个策略函数和 ladder/resident 完整主循环、窗口导出顺序及域值保持。设备闭包新增一个纯枚举头，4 份数学正文和 21+7 实例组未变。
- [主机比较](../../data/experiments/main_flow_refactor_implementation_20261021/p4h/policy_contract.json)：2,656 组输入与从冻结版本提取的 oracle 比较返回值、异常和诊断文本。窗口/数值溢出、无效组合、特殊常量条件均覆盖；10 个主机 CPP 的真实依赖无 CUDA SDK。fixture 缓存 getter 的返回类型声明已修正，首次编译失败证据保留。
- [27 项运行](../../data/experiments/main_flow_refactor_implementation_20261021/p4h/usability/summary.json)、[5+5 基线](../../data/experiments/main_flow_refactor_implementation_20261021/p4h/runtime/summary.json)、[OpenCL 172 资源/2 缓存](../../data/experiments/main_flow_refactor_implementation_20261021/p4h/opencl_resources/result.json)通过。
- 独立 local/384/PRAC ON 构建实际编译可选协作路径；[resident](../../data/experiments/main_flow_refactor_implementation_20261021/p4h/resident_runtime/summary.json)和[PRAC](../../data/experiments/main_flow_refactor_implementation_20261021/p4h/prac_runtime/summary.json)均通过完成点以及停止/损坏接纳/恢复检查。[窗口 CSV](../../data/experiments/main_flow_refactor_implementation_20261021/p4h/prac_window/result.json)验证真实 GPU 导出/回读与两条 CSV，部分乘积未发布为完整 Stage1。此资格仅为实测 384 档、baseline 策略和 GPU1，不外推其他 TPI/variant/constants 或性能。

首次默认 OFF 构建发现 CSV 提取后缺少直接 stdexcept 声明，补齐后通过；既有 ON 编译独立覆盖仍通过。默认 Stage1 仅 adapter 合法重编，其他 30 个及 Stage2 两个 CUDA 对象不变；重复构建 0 CXX/0 CU/0 link。[最终资格](../../data/experiments/main_flow_refactor_implementation_20261021/p4h/qualification.json)冻结三个默认 worker 与可选 PRAC worker。CPU Stage2 互操作沿用 0.45.14。后续 P4 收尾为私有执行合同与应用 PARI 归位；未运行 release、性能或完整位宽终验。

### P4i：Stage1 私有计算合同与应用诊断收尾已完成

对应程序版本 `0.45.17`，P4 的 Stage1 主流程职责整理完成。私有 [execution_contract.h](../../src/ecm/service/stage1/execution_contract.h)以 ExecutionOptions/ExecutionResult 表达一次计算的可变进度与拥有结果，删除三组重复别名；六个构造/移动/释放/状态方法落 [execution_contract.cpp](../../src/ecm/service/stage1/execution_contract.cpp)。TaskResult::State 只在 task.cpp 定义，后端不再看到其内部拥有类型。结果字段/布局、独立清理 sigma 和准备/发布失败优先级保持。TaskDependencies 仍是应用组装的三个私有回调，execute 是唯一粗计算入口，不为实现目录强行新增 Executor/Runner/factory 或重复 prepare/run 层。

PARI helper 和整段命中诊断移入应用私有 [diagnostics/group_order.cpp](../../src/ecm/application/cli/diagnostics/group_order.cpp)，driver 从 1419 行减至 1001 行，只在原预检查、任务完成后调用真实诊断函数。保留 gp 优先级、执行/超时、probable-prime 过滤、输出与旧界限算法；删除没有调用者的旧 32 位 sigma 解析函数。未修订群阶/点阶的数学行为，旧 go_min_bounds、32 位诊断 sigma 和 Edwards 适用性限制仍见前述数学分析，不背书为严格最小 B1/B2。runtime 配置注释改为当前 OpenCL/应用选项职责，未改环境读取行为。

- [源码合同](../../data/experiments/main_flow_refactor_implementation_20261021/p4i/source_contract_04/result.json)：10 个诊断 helper、完整诊断与主入口顺序、六个拥有权方法、结果布局，以及 10 个 CPU/GPU/准备/队列实现单元仅发生既定词汇/包含路径调整。
- [8 项任务](../../data/experiments/main_flow_refactor_implementation_20261021/p4i/task_contract/result.json)、[13 项队列](../../data/experiments/main_flow_refactor_implementation_20261021/p4i/queue_contract/result.json)通过真实 task/存储合同；[12 个主机 CPP](../../data/experiments/main_flow_refactor_implementation_20261021/p4i/host_compile/manifest.json)编译无 CUDA SDK。源码比较工具补齐构造 initializer/noexcept 的识别，失败检查记录保留，未改生产行为或放宽相等断言。
- [PARI 实际对照](../../data/experiments/main_flow_refactor_implementation_20261021/p4i/group_order_02/result.json)：CPU scalar 与 CUDA param0 均对同一固定 N、sigma/B1 得到与 0.45.16 相同因子、群阶分解与诊断字段；记录 gp 的真实 SHA。首次 fixture 未剥离日志时间戳，修正投影后通过。
- [27 项运行](../../data/experiments/main_flow_refactor_implementation_20261021/p4i/usability/summary.json)、[5+5 基线](../../data/experiments/main_flow_refactor_implementation_20261021/p4i/runtime/summary.json)、[OpenCL 172 资源/2 缓存](../../data/experiments/main_flow_refactor_implementation_20261021/p4i/opencl_resources/result.json)通过。31+2 个 CUDA 对象全部不变，重复构建 0 CXX/0 CU/0 link。

[最终资格](../../data/experiments/main_flow_refactor_implementation_20261021/p4i/qualification.json)冻结三个当前 worker，CPU Stage2 互操作仍用 0.45.14；本阶段未重建可选 PRAC profile，其 0.45.16 的资格保持为历史证据。Stage1 仍限每进程一个活跃任务，实际 CUDA 同步/回读不脱离 SDK。Stage2 规划/持久化继续 P5，GUI 与 CPU 入口深化继续 P6，跨类别增量及完整位宽综合终验继续 P7；未运行 release 或性能采样。

提交后的格式修复：移除诊断文件一行继承的尾空白，完整规范化函数/入口对照通过，local 仅 1 CXX/0 CU/1 link，31 个 CUDA 对象不变。版本仍为 0.45.17；[修复资格](../../data/experiments/main_flow_refactor_implementation_20261021/p4i_format/qualification.json)区分原运行验收二进制与格式后重链二进制，不把原运行记录改写成重跑。后续提交在 staging/空白检查成功后才执行。

### P5a：Stage2 进度、回执与指纹存储已收口

对应 Stage2 程序版本 `0.45.18`。公开 [stage2_progress.h](../../include/ecm/storage/queue/stage2_progress.h)只保留 State/Task 的原数据布局与实际读写/回执声明，完整实现落 [stage2_progress.cpp](../../src/ecm/storage/queue/stage2_progress.cpp)。公开 [fingerprint.h](../../include/ecm/platform/fingerprint.h)声明 SHA 接口，Windows CNG、动态加载和句柄清理全部进入 [fingerprint.cpp](../../src/ecm/platform/fingerprint.cpp)。不新增链接依赖、Pimpl 或 Runner。旧 SHA 实现头已删除；旧进度私有头只保留真实 legacy 身份适配与泛型匹配，不作为无价值转发保留。

State/Task 字段和默认值、schema 1/2 接纳、pending_results、SHA 身份、临时文件/Flush/Move 原子提交、完整行 receipt/finished 确认，以及 legacy 导入匹配均保持。公开头不包含 Windows、bcrypt、JSON 或文件实现。完整 Stage2 父子、队列、规划和 tune 正文仍未变，后续继续 P5 深化。

- [源码对照](../../data/experiments/main_flow_refactor_implementation_20261021/p5a/source_contract_02/result.json)：8 个存储方法、3 个 SHA 函数、状态布局、legacy 匹配和完整 driver 相同。比较工具补齐 const 方法识别，原失败记录保留。
- [18 项独立存储检查](../../data/experiments/main_flow_refactor_implementation_20261021/p5a/storage_contract/manifest.json)：实际 MSVC/MT 编译并运行真实 CPP；CUDA SDK 路径/头和 nvcc 调用均为零。
- [23 项结果生命周期](../../data/experiments/main_flow_refactor_implementation_20261021/p5a/results_02/result.json)通过，保留撕裂/损坏/重复回执和结果恢复断言；fixture 补齐其自身直接使用的 JSON 声明，未修改生产解析或放宽断言。
- [5+5 基线运行](../../data/experiments/main_flow_refactor_implementation_20261021/p5a/runtime/summary.json)通过。Stage2 local 为 4 CXX/0 CU/1 link，两个 CUDA 对象不变；重复构建 0 CXX/0 CU/0 link。

[最终资格](../../data/experiments/main_flow_refactor_implementation_20261021/p5a/qualification.json)冻结当前 Stage2 worker。Stage1 对照协作者仍为 0.45.17，本步未重建 Stage1/OpenCL/CPU Stage2，不把这些写成 0.45.18 验收。P5 父子流程、规划与运行状态继续实施；未运行 release、性能或全位宽终验。

### P5b：Stage2 规划成为独立编译单元

Stage2 版本 `0.45.19`。原 1487 行文本包含已删除，选择、校准、测量及运行缓存整体进入 [planning/stage2.cpp](../../src/ecm/service/planning/stage2.cpp)。共享私有 [runtime_types.h](../../src/ecm/service/stage2/runtime_types.h)保留原 7 个数据/错误类型及默认值；TaskDefaults 只传 B2 默认值和原控制台 arena/owner 预算，未透传完整 Settings。原 Options 暂留私有调用合同，不作为公开任务 API。规划中的真实设备选择和显存快照仍使用 CUDA runtime，符合实施范围。

- [完整源码核对](../../data/experiments/main_flow_refactor_implementation_20261021/p5b/source_contract/result.json)：44 个规划函数、36 个 main 函数、完整规划作用域及 7 个布局一致；3 处配置投影保持原调用位置、原预算来源。缓存、作用域指针和 stop 状态不新增拥有者。
- [实际规划运行](../../data/experiments/main_flow_refactor_implementation_20261021/p5b/planning_runtime_06/result.json)：0.45.18 与当前程序各通过配置 B2、校准、测量方案选择、单曲线发布及不使用有效设备的 profile 合并，共 8 项。修正 fixture 对表达式输入、当前 CLI、format 8、默认方案保护及 NF 结果字段的假设；生产逻辑未随之修改，失败证据保留。时间值不作性能比较。
- [5+5 基线](../../data/experiments/main_flow_refactor_implementation_20261021/p5b/runtime/summary.json)通过。两个构建尝试累计实际更新 3 CXX/0 CU/1 link；首次发现缺少私有前置声明，补齐后成功。两个 CUDA 对象相同，重复构建零编译/链接。真实编译依赖中无完整 TOML/Settings 实现头。

[资格与冻结 worker](../../data/experiments/main_flow_refactor_implementation_20261021/p5b/qualification.json)。Stage1 对照仍为 0.45.17；本步未重建 OpenCL/CPU Stage2。下一步收口唯一父流程、曲线 worker 和进程适配，不添加重复 Runner；尚未进行全位宽和性能终验。

### P5c：Stage2 父流程、曲线 worker 与进程职责归位

Stage2 版本 `0.45.20`。application main 从 1508 行减至约 500 行，保留 CLI、配置合并、模式选择和退出；[task.cpp](../../src/ecm/service/stage2/task.cpp)拥有唯一父任务/队列流水线及完整曲线 worker，[input.cpp](../../src/ecm/service/stage2/input.cpp)集中 save/队列校验与持久诊断 append，[stage2_curve_process.cpp](../../src/ecm/platform/process/stage2_curve_process.cpp)拥有 Job、管道、停止和输出投影。环境/真实 CUDA 等待策略进入 backend host_policy，结果身份读取由普通配置 CPP 返回 3 个实际字段，仍在原发布位置读取。旧 console 头整体归 platform，未修改格式。不新增 TaskSession/Runner 的重复循环。

父流程只接收路径、4 项队列策略及 3 项默认值，未透传完整 Settings。pending→子进程→持久 receipt→done→seal→源行复核→finished→移除队列→删除进度保持；输入错误先落原因再标错。worker 重读 save/校验 hash 后才规划，外部发布→内部持久结果→可选调优更新保持。进程单实例 stop/event 状态只保留一个拥有者，不宣称同进程多会话。私有 Options 仍覆盖多个模式，后续继续收窄曲线进程请求及双向 hints。

- [源码合同](../../data/experiments/main_flow_refactor_implementation_20261021/p5c/source_contract_02/result.json)：34 个原函数、44 个既有规划函数、完整父/worker 流水线、应用配置与模式选择及 console 字节一致。
- [5 个真实 CPP 的无 SDK 编译](../../data/experiments/main_flow_refactor_implementation_20261021/p5c/host_compile_03/manifest.json)：application、task、input、身份读取和进程实现均不需 CUDA SDK 头或 nvcc；实际 CUDA policy/显存查询仍保留 SDK。
- [12 项运行对比](../../data/experiments/main_flow_refactor_implementation_20261021/p5c/workflow_runtime_02/result.json)：完成、dry、损坏输入、非法承载、save 改动、发布失败；[停止/恢复](../../data/experiments/main_flow_refactor_implementation_20261021/p5c/stop_publication/report.json)在第 1/4 曲线后停止，恢复不重算，最后只发布一条 4 曲线 NF，CRC 正确。
- [调优 8 项](../../data/experiments/main_flow_refactor_implementation_20261021/p5c/planning_runtime/result.json)及 [5+5 基线](../../data/experiments/main_flow_refactor_implementation_20261021/p5c/runtime/summary.json)通过；8 个主机对象更新，两个 CUDA 对象保持，重复构建零编译/链接。

首次构建再次触发已知退役 include 误报，核对 9 个 SDK 文件未变并[保存证据](../../data/experiments/main_flow_refactor_implementation_20261021/p5c/dependency_guard_02/result.json)，正式重试未改保护/缓存。另修正提取时字段名文本匹配、私有默认参数、缓存类型直接引用；全部失败日志保留。测试使用当前 TOML/CLI，未回退生产解析器。[最终资格](../../data/experiments/main_flow_refactor_implementation_20261021/p5c/qualification.json)。本阶段未重建 Stage1/OpenCL/CPU2，也未运行 release 或性能验收。

### P5d：进程请求收窄，P5 完成

Stage2 版本 `0.45.21`。[公开进程请求](../../include/ecm/platform/process/curve_process.h)对应 [process 实现](../../src/ecm/platform/process/stage2_curve_process.cpp)，不引用 Options、Settings、Windows/GMP/CUDA 类型或 service/backend 头。请求明确列出原协议实际使用的 49 项配置及 save/record/结果/日志/预算；不把现有协议复杂度藏在完整 CLI Options 后面。同步 selection 回调仍在原输出行位置、写日志之前执行，缓存 hints 的校验与变更留任务层。Job/pipe/process Handle 保持原 RAII 实现，留 private header；无空转发兼容头。

`child_run` 现在是实际域到进程协议的投影，由普通任务和校准两个消费者共用，只有一处进程循环。没有新建 factory.cpp、Runner.run 或 TaskSession.run。父流程、完整 worker、校准/选择/cache 作用域均未变，现有 15 个 host 引擎模块继续保留；真实 CUDA runtime、异步检查/重放/回读仍由原后端拥有。环境和 stop/event 继续按原进程作用域读写，不声称同进程多会话。

- [完整源码合同](../../data/experiments/main_flow_refactor_implementation_20261021/p5d/source_contract/result.json)：参数投影、全部进程协议、反馈原时点、父/worker 和规划作用域一致。
- [独立真实进程测试](../../data/experiments/main_flow_refactor_implementation_20261021/p5d/process_contract_02/manifest.json)：只编译 process.cpp 与 fixture，不链接引擎/GMP/CUDA；实际 5 个子进程、30 项 wire 检查，覆盖引号/空格、0/3/其他退出码、双管道尾部、禁用/启用反馈和回调异常清理。首次因 PS 将预期 stderr 当错误而中断，修正测试启动方式并保留原证据。
- [5 CPP 无 SDK 编译](../../data/experiments/main_flow_refactor_implementation_20261021/p5d/host_compile_02/manifest.json)、[12 项队列与故障](../../data/experiments/main_flow_refactor_implementation_20261021/p5d/workflow_runtime/result.json)、[8 项调优](../../data/experiments/main_flow_refactor_implementation_20261021/p5d/planning_runtime/result.json)、[停止恢复](../../data/experiments/main_flow_refactor_implementation_20261021/p5d/stop_publication/report.json)及 [5+5 基线](../../data/experiments/main_flow_refactor_implementation_20261021/p5d/runtime/summary.json)通过。7 个主机对象更新、0 CU，重复构建零编译/链接。

[最终资格](../../data/experiments/main_flow_refactor_implementation_20261021/p5d/qualification.json)。退役依赖误报的 9 个 SDK 内容再次核对未变，正式重试未改保护/缓存；该已知构建问题在 P7 修复。P5 完成，接下来实施 P6 GUI Controller；Stage1/OpenCL/CPU2 本步未重建，完整位宽与性能终验仍未执行，未运行 release。

### P6a：GUI 生产目录与测试构建隔离完成

GUI 版本 `0.45.22`，37 个文件完成归位：[生产入口](../../src/ecm/application/gui/main_win32.cpp)、App、进程 Adapter、配置/结果/生成器/监控及语言资源位于 `src/ecm/application/gui/`；4 个 fake/测试 fixture 移至 `tools/test/gui/`。原目录已删除，CMake 和直接编译入口同步，无兼容目录或源码副本。默认 `build_gui_local/` 只构建 GUI，测试目标 EXCLUDE_FROM_ALL，必须显式选择；不混入 CUDA 构建目录。

[源码核对](../../data/experiments/main_flow_refactor_implementation_20261021/p6a/source_contract/result.json)证明 36 个内容文件只有路径/include 调整，CMake 仅调整位置和显式测试归属。[首次默认产物](../../data/experiments/main_flow_refactor_implementation_20261021/p6a/production_payload.json)只有 ecm_gui.exe，随后显式构建测试 helper。实际无窗口自检 33 项、日志解析 85 项、结果 26 项、生成器 110 项、真实 WorkerProc 54 项通过；checkpoint 扫描 5 类 primary 接纳/3 类 backup-new-Stage2 排除通过。

首次自检 fixture 将整数 autostart 写成 bool，被当前解析器拒绝；改正 fixture 为 0 后通过，原日志保留。发现旧 GUI 脚本仍有 `-ini`、旧分区/裸字符串及“看到 checkpoint 即强杀”的过期断言，不能作为本轮资格；P6b 使用当前 TOML/CLI 和正确退出策略验证 Controller。

[最终资格](../../data/experiments/main_flow_refactor_implementation_20261021/p6a/qualification.json)冻结 GUI/测试程序。本步未提取 Controller，不宣称界面已与生命周期解耦；CPU Stage2 已归位且约 224 个实质行，不为目录整齐新增转发层。未重建计算 worker，未运行 CUDA release、GPU 性能或 GUI 混合阶段资格。

## 首先调整构建复用与输出隔离

### 当前样例说明了什么

用户提供的 [完整原始文本与构建日志](../../data/experiments/main_flow_refactor_plan_20261021/user_build_observation.txt) 记录一份 Stage1 LOCAL 构建：sm_89、压缩 ON、PTX OFF、tiers=all、resident/PRAC OFF、Jobs=8、CUDA 13.3、MSVC 14.51.36231；31 个 CUDA TU 全部编译，CUDA 并行编译阶段墙钟 1535.6 秒，最长 TU 1535.0 秒，并发 TU elapsed 之和为 8477.9 秒。之后还执行 CXX 编译和链接；不能把 1535.6 秒作为完整脚本耗时，也不能用 8477.9/1535.6 推导实测加速比。

这是用户提供的单个样本，未复现。源码/二进制完整身份、CPU 型号、内存、系统负载与功耗条件未提供；sm_89 不是实体 GPU 型号。这里据此确定验收重点为“同配置复用”和“无关主机修改零 CU 编译”，不宣称重构后的耗时或性能提升。原始文本按字节保留，SHA256 为 A044EE0B4891EF4F86AF2F586BCA39275D9571D01133FFDF6F02F8B3710B9D25。

前期快照中的策略见“当前重编的三个来源”：当时 Stage1 仅 Incremental 启用逐对象跳过，Stage2 普通 local 在整体 signature 变化后重编全部源。P1a 已修正这两项策略，验证见“实施进度与阻塞记录”；后续提取继续缩小真实依赖。

### 目标输出目录

dev/local/release 是用途，不等同于 Debug/Release 编译配置。目录名、profile marker 和实际编译参数必须同时一致。下表是目标，不是当前默认值；CPU/GUI/OpenCL 尚无完整正式入口族，不能写成已经具备。

| 程序与用途 | 目标构建叶目录模式 | 约束 |
| --- | --- | --- |
| CUDA Stage1 local | build_cuda_local/stage1/sm89/<profile-key>/ | 本机生产策略；默认 all tiers、压缩 ON、无 PTX、PRAC OFF。架构可显式选择。 |
| CUDA Stage2 local | build_cuda_local/stage2/production/sm89/<profile-key>/ | 生产引擎；与 Stage1 独立 cache/对象/清单。 |
| CUDA Stage1 dev | build_cuda_dev/stage1/sm89/<profile-key>/ | 可缩小 tiers 或显式改参数；不借用 local/release 叶目录。 |
| CUDA Stage2 生产主流程开发 | build_cuda_dev/stage2/production/sm89/<profile-key>/ | 编译生产引擎做开发，不因用途为 dev 自动切实验引擎。 |
| 现有 Stage2 development engine | build_cuda_dev/stage2/experimental/sm89/<profile-key>/ | 明确保留实验身份；不进入生产源闭包和发布包。实验实现本身不在本次重构范围。 |
| CUDA Stage1 release | build_cuda_release/stage1/<toolkit-group>/<arch-set>/<profile-key>/ | 保留 CUDA12.6/13.3 分组与现有拆架构策略，不强用 local 参数。 |
| CUDA Stage2 release | build_cuda_release/stage2/<toolkit-group>/<arch>/<profile-key>/ | 每架构独立。当前 60/70 使用 12.6，其余目标使用 13.3；以实际支持检查为准。 |
| CPU Stage2 dev/local/release | build_cpu_<role>/stage2/<libecm-or-gwpoly>/<profile-key>/ | 新增正式入口时使用；当前入口在 tools/build/test，仍构建生产程序。CPU Stage1 继续随主 worker，不新增独立程序。 |
| OpenCL Stage1 dev/local/release | build_opencl_<role>/stage1/<profile-key>/ | 若补齐入口族，使用独立根；运行时 kernel cache 另按设备与内容身份管理。 |
| GUI dev/local/release | build_gui_<role>/<profile-key>/ | 正式 local/release 默认只编生产 GUI；fake worker/独立测试显式选择。 |

叶目录统一容纳以下构建产物，不与用户运行文件混放：

```text
<build-leaf>/
├── profile.json                 # 用途、engine、工具链、ABI和编译策略
├── cmake/                       # CMakeCache、生成规则、owner graph（如使用CMake）
├── objects/                     # 或由CMake管理的对象目录；一份实际所有者
├── state/                       # 每TU指纹、depfile索引、构建/链接/拆分/打包清单
├── logs/                        # configure、逐TU、link及汇总
└── bin/                         # exe、所需DLL、运行时资源与身份sidecar
```

objects/state 是职责名称，不要求第一步强行改变 CMake 私有对象布局；统一位置由构建 profile 解析器返回，消费者不得再猜目录或单独拼旧路径。日志、清单、DLL 与 exe 均属于同一 profile。发布包继续放现有 dist/cuda、dist/cuda-stage2-* 对应目录，与 build 根分离；保持当前包内容和多架构发布资格限制，重构本身不恢复尚未完成的发布资格。

用户 TOML、worktodo、checkpoint、save、result 和实验数据不因 build 根改名而自动搬移或覆盖。更改 exe 输出位置时同步 GUI/启动器的定位规则、资源根和显式路径处理，验证工作目录变化不会意外改读另一份配置/队列。

### profile-key 与身份

profile-key 基于**配置兼容性**：用途、程序/engine、toolkit 组和版本、架构集合、CMake generator/config、MSVC/CRT、默认流、优化/设备宏、tiers/实例选择、PTX/压缩、PRAC、GMP 等 ABI/依赖策略。它不包含每个业务源码 hash 或每次 HEAD，否则修改一个 .cpp 就新建目录，失去增量价值。

profile.json 保存可读字段和完整配置 hash，短 key 冲突时核对完整字段。显式 BuildDir/Build 同样检查 marker；不得静默把 local 叶目录改标成 release，或把实验叶目录改标为 production。配置组合变更使用另一叶目录；同一组合的源码变化由逐 TU 指纹处理。

完整构建身份仍绑定源码、生成物、编译对象与 exe。区分“配置兼容身份”“逐 TU 编译身份”“链接/发布身份”，不是删去任一层校验。Stage1 host sidecar 继续避免整程序源码身份污染所有 CUDA 命令。

### 统一构建模式与编译规则

| 模式 | 精确定义 | 不满足时 |
| --- | --- | --- |
| 默认 reuse | 逐 TU 校验命令、include/depfile/response、工具链和对象 hash；只编译失效单元；链接输入变化才链接 | 只构建需要更新的单元，给出失效原因。 |
| rebuild / FullRebuild | 显式重编当前 profile 的必要单元 | 不删除其他 profile、用途、架构或原始证据。 |
| reconfigure | 更新生成配置和构建图 | 不等于清空对象。 |
| HostOnly | 强约束不重编 .cu；有效 CUDA 对象必须全部可复用；主机按失效状态编译并链接 | 缺对象、合同/依赖/命令变化时提前失败，不静默回退全量 CUDA 编译。 |
| SkipBuild | 完全不编译，只对完整身份匹配的冻结产物继续拆分/打包 | 失配即拒绝，不能补写清单冒充新产物。 |
| RelinkOnly | P1b 已关闭旧绕过入口，使用时明确报错 | 正常增量处理必要主机编译/链接；严格 HostOnly 先验证全部 CUDA 对象，不承诺所有主机对象均不编译。 |

互斥选项如 HostOnly+rebuild 明确拒绝。Incremental 可保留为旧入口兼容参数；新默认已经 reuse，不再需要用户记忆该开关。BAT 仅转发，默认值继续只维护在 PS1，生产源清单与 owner graph 保持唯一来源。

普通 .cpp 逐步用兼容 CXX 编译器；.cu 用 NVCC。Stage2 现有 .cpp 也由 nvcc 驱动，先完成对象复用，再独立验证切 CXX 的 CRT、宏、异常/分配、GMP 与 CUDA 链接；设备 runtime 无需脱离 CUDA。SDK 只由实际选择的后端查找，根 CMake 的无条件 OpenCL 查找应收紧。共用源不自动共用对象：Stage1 两个 worker 的后端宏不同，只有编译合同一致才共用库。

概念构建流程：

```text
profile = resolve_and_validate_role_program_options()
leaf = locate_isolated_leaf(profile)
assert_profile_marker_compatible(leaf, profile)
graph = generate_from_single_source_inventory(profile)

for unit in graph.required_translation_units:
    state = fingerprint(current_command, dependencies, toolchain, object)
    if state.valid:
        reuse(unit)
    else:
        if mode forbids_compilation_of(unit):
            fail_with_reason(unit, state.reason)
        compile_with_language_compiler(unit)
        verify_inputs_unchanged_during_compile(unit)
        record_object_identity(unit)

if mode permits_link and link_inputs_or_command_changed:
    link_and_record_identity()
verify_final_source_object_exe_identity()
write_summary(cxx_compiled, cu_compiled, reused, invalidation_reasons, timing_boundaries)
```

当前内容指纹会保守扫描预处理分支；改注释、移动路径也可能失效，这是有依据的重编。不能以“看起来不影响 GPU”为由忽略依赖。

### 旧目录和已有对象的处理

保留 build_cuda_cmake 等旧目录及清单，不移动、删除或重写原始证据。过渡期间可显式用旧目录进行**原用途**的验证复用；它不能成为 release/其他 engine 的缓存。

新默认目录首次可能需要完整编译。现有指纹包含绝对命令、working directory、response 和对象路径；不能直接复制/改名 CMakeCache、obj、.ecm-state.json，再宣称等价。跨目录 artifact import 是可选后续优化，必须验证旧对象、输入/ABI等价和来源，不作为本次默认增量的前提。源码目录迁移导致的合法一次重编也要记录，不许刷新 hash 绕过。

## 完整目标目录

这是**模块级完整目标树**，列出关键 Interface 和实现家族；不是把每个旧 helper 固定为一个新文件。只创建有实际使用的公开头，不要求空目录镜像。配置定义、第三方、测试和工具保持原领域；本次只调整必要构建入口与消费者路径。

```text
ECM-ELF/
├── CMakeLists.txt
├── config/                              # ecm_options.json是配置唯一事实源
├── include/ecm/                         # 跨Module Interface；无设备模板
├── src/ecm/                             # 全部生产主流程Implementation
├── tools/build/                         # dev/local/release及内部构建引擎
├── tools/build/test/                     # 现有验证/探针；必要路径适配
├── docs/                                # 正式状态文档与本设计草稿
├── data/                                # 忽略的原始证据/运行与实验数据
├── third_party/                         # 第三方依赖，不做内部重构
├── .refactor/                           # 外部参考
├── build_<backend>_<role>/               # 隔离构建；具体层次见前节
└── dist/                                # 发布包
```

公开表面与实现使用对应路径：

```text
include/ecm/
├── application/
│   ├── configuration/
│   │   ├── settings.h                   # CLI/GUI共享模型和归一化入口
│   │   └── generated/                   # 仅确需跨Module的生成声明
│   └── gui/worker_controller.h           # 仅worker编排的窄接口
├── core/
│   ├── integer.h                        # 现有GMP的拥有值/借用合同
│   ├── curve/{identity,construction,point}.h
│   ├── stage1/scalar.h
│   ├── stage2/{model,geometry}.h
│   └── result/outcome.h
├── service/
│   ├── stage1/task.h                     # 数学请求、执行偏好、任务结果
│   ├── stage2/task.h                     # 独立续算与父子编排合同
│   ├── execution/{stage1,stage2}.h        # 粗执行Interface、能力、协作
│   ├── planning/planner.h                # Auto B2 / D / 调优流程的输入输出
│   └── work/{checkpoint,publication,queue}.h # queue.h声明workflow与文件合同
├── backend/
│   ├── cpu/factory.h
│   ├── cuda/factory.h
│   └── opencl/factory.h
├── storage/{save,profile,cache,factory}.h # 数据合同与文件Adapter创建入口
└── platform/{logging,process,file_commit}.h
```

```text
src/ecm/
├── application/
│   ├── cli/                             # Stage1 main/命令、stdin、帮助
│   │   └── diagnostics/                 # 当前PARI群阶诊断等私有功能
│   ├── cuda_stage2/                      # 独立main/命令与程序组装
│   ├── cpu_stage2/                       # 当前CPU Edwards续算入口
│   ├── gui/                             # 视图、UI状态、worker_controller
│   │   └── models/                      # 现有概率、结果、生成等纯模型
│   └── configuration/                   # 解析、生成物适配、兼容归一化
│       └── generated/                   # 默认值/键描述/解析表等私有生成物
├── core/
│   ├── integer.cpp
│   ├── curve/
│   │   ├── identity.cpp
│   │   ├── montgomery/                  # param0/2/3数学构造、检查
│   │   └── edwards/                     # 数学构造、映射与完成点检查
│   ├── stage1/                          # K构建、共享的NAF/PRAC数学语义
│   ├── stage2/                          # bounds、geometry、D候选、概率/relocation
│   └── result/                          # proper factor、退化/饱和分类
├── service/
│   ├── stage1/task.cpp                   # 唯一任务编排；Session默认私有同文件
│   ├── stage2/                          # 独立任务、父子运行、receipt恢复
│   ├── execution/                       # 只在有共享Implementation时建；不包转发层
│   ├── planning/                        # 搜索/采样编排，不包含文件编码
│   └── work/queue_workflow.cpp           # 循环与完成/失败/交接的完整私有实现
├── backend/
│   ├── cpu/
│   │   ├── factory.cpp
│   │   ├── arithmetic/{scalar,ifma}/    # GMP/MPN与IFMA；非曲线身份
│   │   ├── stage1/{montgomery,edwards}/ # scalar/SIMD批执行和私有恢复
│   │   └── stage2/{libecm,gwpoly}/
│   ├── cuda/
│   │   ├── factory.cpp
│   │   ├── runtime/                    # device/memory/stream/event/error
│   │   ├── dispatch/                   # tiers.def、能力和实例注册
│   │   ├── arithmetic/{cgbn,goldilocks}/
│   │   ├── stage1/
│   │   │   ├── montgomery/
│   │   │   │   ├── prepare.cpp         # 仅CUDA所需的主机布局准备
│   │   │   │   ├── adapter.cu          # 资源、chunk、launch、同步/回读
│   │   │   │   ├── launch.h            # 私有设备调用合同
│   │   │   │   ├── point.cuh           # 私有设备算法
│   │   │   │   ├── instances/          # 保留param/位宽/TPI独立TU
│   │   │   │   └── prac/               # 当前可选resident/PRAC
│   │   │   └── edwards/
│   │   │       ├── prepare.cpp         # 主机NAF编码与布局准备
│   │   │       ├── adapter.cu          # 允许保留runtime.cu形式
│   │   │       ├── launch.h
│   │   │       ├── point.cuh
│   │   │       └── instances/
│   │   ├── stage2/                     # 现有host引擎、private context、
│   │   │                               # points/product/tree/remainder/S4/设备算术
│   │   ├── ntt/                        # shape/runtime/graph/transform及实例
│   │   └── planning/                   # CUDA形状、精度、内存/驻留模型
│   └── opencl/
│       ├── factory.cpp
│       ├── runtime/                    # context/program/cache/资源加载
│       ├── arithmetic/                 # 生产路径、manifest与组装
│       ├── stage1/montgomery/          # 当前param3执行与私有恢复
│       └── resources/                  # 生产.cl与include资源
│           └── diagnostics/mont.cl     # 当前bench/mont.cl的生产自检用途
├── storage/
│   ├── factory.cpp
│   ├── queue/                          # worktodo格式、锁、原子改写
│   ├── checkpoint/                     # 外壳、主备、codec、legacy接纳
│   ├── save/                           # 完成点读写和兼容
│   ├── results/                        # data/ledger/submission/internal/seal
│   ├── profile/                        # tune.toml读写、合并与schema
│   ├── cache/                          # K/计划等主机缓存文件
│   └── transfer/                       # Stage2 outbox/Prime95协议
└── platform/                           # 日志、进程、路径/原子文件操作
```

映射关系为 include/ecm/<module-path>/ 与 src/ecm/<module-path>/；纯 Interface 可以无同名 .cpp，私有实现不创建公共镜像。CUDA 的 launch.h、.cuh、CGBN 类型、tiers 和实例留 src；其他 Module 只使用阶段合同与 factory。现有 include/ecm.h 等项目头通过私有兼容层过渡，避免与 libecm 的 <ecm.h> 混淆。

配置字段、默认值、范围及说明仍只在config/ecm_options.json维护。生成器只作必要的输出路径适配，公开模型与私有解析表由同一来源生成，参考文档继续为生成物；不手工维护第二套默认值，不把完整生成schema传入设备编译闭包。

原 kernels/cuda 的生产内容全部迁入 src/ecm/backend/cuda，不把实例合成巨型 .cu、不改变常量单一所有者或默认流、不因此引入 RDC。生产 OpenCL 资源也进入 src；[opencl_ecm_selftest.cpp](../../src/opencl_ecm_selftest.cpp) 第 32–36 行读取的 bench/mont.cl 属于生产自检输入，迁入 diagnostics 时同步 loader 与打包。不能按 bench/test 文件名删除生产必需行为；纯测试/实验资源不进入生产清单。

## 模块职责、依赖与构建目标

| Module / 建议构建目标 | 输入 → 输出 | 拥有的 Implementation；禁止承担的职责 |
| --- | --- | --- |
| application / 现有各 exe | CLI/TOML/GUI命令/队列来源 → 规范化请求、退出码或UI快照 | 文本、路径解释、组装factory、展示；不拆装设备状态。不同程序命令语义保留。 |
| core / ecm_core | 数学身份、整数、bounds → 构造/标量/几何/分类结果 | GMP主机数学和不变量；不读文件、配置、设备或Windows状态。 |
| service/stage1 / ecm_service | Stage1Request + 执行/存储能力 → TaskResult | session、恢复/准备/运行/最终发布；不读取worktodo语法或持有CUDA指针。 |
| service/stage2 / ecm_service | save任务、选择与进程能力 → receipt确认的续算结果 | 当前父子进程/恢复流程；不把CPU入口虚构为同等能力。 |
| service/execution / 声明与少量共享CXX | 任务、编译能力、执行偏好 → 接纳或明确拒绝 | 粗执行Seam；没有逐域乘/逐位虚接口，不创建总设备框架。 |
| service/planning / ecm_service | 数学候选、profile、后端资源快照 → PlanDecision | Auto B2/D/调优采样编排；纯数学归core，设备模型归backend，TOML归storage。 |
| service/work / ecm_service | 来源ticket、任务结果/发布凭据 → 完成、保留、pending或失败 | 完整队列/交接顺序；不由writer自动推进源任务。 |
| backend/cpu / ecm_backend_cpu | 阶段请求、计划、恢复态 → 结果/安全点事实 | CPU线程、scalar/SIMD、lane与私有布局、ISA检测；公共构造数学不因运行在CPU就归此处。 |
| backend/cuda / ecm_backend_cuda | 粗执行请求、窄协作Interface → 结果/快照/设备统计 | 设备runtime、数据布局、内核、NTT与模型；无配置/文件格式/展示实现。真实runtime可继续.cu。 |
| backend/opencl / ecm_backend_opencl | 当前支持任务、执行选项 → 结果/快照 | OpenCL资源、组装、cache、生产自检；不改变当前支持范围。 |
| storage / ecm_storage | 身份、主机点/记录/文件请求 → 已校验输入或可靠写入确认 | 编解码、ledger、锁、备份与发布；不决定GPU同步、算法或队列成功。 |
| platform / ecm_platform | 日志/进程/文件操作请求 → 操作结果、进程/句柄owner | 系统机制；不承载TaskSession、总Supervisor或业务重试规则。 |

依赖方向：application 组装 service 与 Adapter；service 依赖 core 和抽象执行/持久化 Interface；backend/storage 实现这些 Interface，并依赖 core/platform。core 不反向依赖它们。service/work 持久化合同由 service 拥有，filesystem Implementation 在 storage；源码声明依赖不应演变为 service/storage 静态库循环链接。

只创建少量职责完整的库；CUDA Stage1/Stage2/NTT 可为其私有目标，每目录不强制成库。factory和session等文件名不强制形成类或独立Module；service/execution若只有合同声明，不建运行时转发对象。PUBLIC include 仅公开 include/；各 backend 私有 src 路径不传播。目标相关宏未消除前，共享源清单仍可按 exe 编译，不能为共享对象误改行为。

曲线与后端两条轴采用组合：

- 曲线身份为 target N + family + Montgomery param（Edwards不适用）+ sigma；B1/有效标量和执行选择分别记录。CPU旧配置的兼容映射先按实际路径解释，不能直接用旧GPU默认param3替代CPU实际param0。
- core/curve 共享 checked 构造、普通点和分类合同；GPU上传前 CPU packing 属于该GPU backend的普通 .cpp。CPU scalar/IFMA/CGBN/OpenCL 的运行布局各自私有。
- 共享NAF数学语义不等于共享编码/字典/进度：CPU scalar、IFMA、CUDA默认窗口、字典坐标和位平面不同；中途状态不随意互换。
- Montgomery曲线与Montgomery模乘表示分别命名；Edwards也使用后者。Stage2 target N 与算术carrier M 保持不同含义。
- Stage2 CUDA已用x-only状态复用引擎；不按来源曲线复制整套多项式流程。工厂返回真实能力，不增加未实现的组合。

## 职责划分与重复封装检查

本节细化前述目标设计，仍未实施。driver、task、factory、QueueWorkflow 是职责名称，**不是必须依次经过的四层，也不是四个必建的类**。文件名只用于定位 Implementation；独立文件、类和构建目标都须有实际职责依据。

### 四个职责分别决定什么

| 职责与建议落点 | 决定什么；输入 → 输出 | 不承担的职责 |
| --- | --- | --- |
| application driver；main.cpp及CLI私有实现即可 | 程序怎样运行：解析命令/配置、规范化输入、组装本exe已链接能力、选择直接/队列/查询模式、展示与退出码。输入为用户命令和运行环境，输出为任务调用或程序结果。 | 不再复制一次任务的准备/执行/恢复/发布，不处理设备状态或逐文件提交；无需单独Driver类。 |
| task；service/stage1/task.cpp中的run_stage1_task | 一次任务怎样才算成功：请求不变量、session、恢复、执行、完整结果与发布确认、checkpoint结束。Stage1Request+窄依赖 → TaskResult。 | 不解析CLI/TOML、不循环读取worktodo、不更新finished/源队列、不包含kernel或文件编码。 |
| factory；backend或storage的创建函数 | 怎样创建所需实现：按已规范化选择创建当前exe支持的Adapter，返回拥有对象或明确错误。存储factory创建文件能力。 | 不运行任务、不打开TaskSession、不推进队列；曲线数学构造不是factory。若直接构造已足够，不必另建factory.cpp或Factory类。 |
| QueueWorkflow；service/work/queue_workflow.cpp中的run_queue_workflow | 来源任务怎样继续或完成：读取下一项、调用同一task、解释停止/失败、完成交接与源队列协议，再继续循环。归一化任务来源+任务窄依赖+队列存储 → QueueResult。 | 不复制task七步；不拥有点状态、设备资源或结果编码。不再外包给第二个完整QueueService。 |

driver只需要知道“调用哪条流程”。task的七步是一个Module内部可读的顺序；QueueWorkflow增加的是来源任务的循环、完成与交接语义，二者不是同一套编排。

### 调用关系：组装与执行分开

```text
application driver / main
├── 解析与组装：持有必要owner、存储能力和本exe的创建函数
├── 直接运行 ───────────────────────→ run_stage1_task
└── 队列运行 → run_queue_workflow ───→ run_stage1_task
                   ├── 获取下一条规范化任务（含当前配置重读）
                   └── 私有完成/失败处理 → QueueStore、Transfer Adapter

run_stage1_task                         # 唯一一次任务编排
├── create_executor(selection)          # 每次任务一处创建/选择
├── 私有TaskSession                     # 身份、锁、上下文及生命周期
├── executor.prepare / execute          # prepare内接纳实际组合与恢复态
└── CheckpointStore / Publisher         # 受控提交与最终确认

executor → ExecutionContext             # 安全点、同步快照、可靠即时提交
```

application组装可供调用的创建函数，不在组装后先为同一任务create一次，再由task经过registry.select重选。task调用该函数一次，取得阶段级Executor；具体曲线/算法的实际支持、设备约束与恢复接纳由Executor.prepare负责。capabilities可用于查询或prepare内部校验，不要求每个task再额外查询一套全局能力。

QueueWorkflow请求的next_task是窄函数引用或轻量来源对象，由应用适配启动配置快照、QueueStore读选和请求归一化。它只返回QueueItem/Empty/ReadError，不执行task或完成队列；不需要为这个回调再建立SourceService。这样service/work不用包含application的TOML/CLI实现。QueueItem保留ticket、规范化请求与完成所需的队列策略。

### 状态、发布与文件机制各有一个所有者

- **TaskSession是task的私有状态和资源owner。** 不再提供另一套公开run。默认与task.cpp同文件或使用私有头；只有复杂状态值得独立实现时才拆session.cpp，不强制拆成第二个Module。
- **finalize_outcomes只核对完整性、身份与已分类状态。** 不重新计算已完成GCD、归一化或再次即时提交factor。
- **Publisher.finish_batch负责最终发布确认与seal。** 按既有ledger/幂等规则补齐所需输出，不另做数学后处理；session.finish_completed只结束checkpoint与释放会话，不再次调用publisher。
- **QueueWorkflow拥有完成顺序，QueueStore拥有文件机制。** complete_queue_task是同一workflow的私有函数，维持park→finished→source更新→sync→flush→delivery。QueueStore只读写、锁、按expected ticket更新和执行协议去重，不调用task、不决定业务完成。
- **状态返回不新增I/O。** 如果RAII已经默认保留恢复材料，retain_and_fail等方法只是包一层Failed返回，应直接返回。只有额外记录错误来源或执行真实状态规则时才保留私有helper。

结果与checkpoint可以使用不同文件协议，但同一任务成功条件只有task负责；来源队列完成条件只有QueueWorkflow负责。逐曲线save/factor/DONE原有次序与失败反馈继续通过ExecutionContext和Adapter保持。

### 依赖和校验如何避免重复

TaskDependencies只包含创建Executor的函数，以及CheckpointStore、Publisher、PlanCache等必要引用；没有QueueStore、完整Settings、GUI状态或可随意查询任意能力的service locator。QueueWorkflow另接收队列/交接存储，只把TaskDependencies传给task。可以用几个显式参数或轻量结构表达，不再增加BindingsView、TaskBindingsAdapter、QueueBindingsAdapter等剪裁转发层。

校验按依据分配：driver处理文本语法、输入来源和配置覆盖；task校验数学请求与单次任务不变量；factory判断所选实现是否存在/可创建；prepare检查该实现的组合、设备与恢复状态。只做一次实现选择，不在每层重复解析或筛选同一个请求。必要的文件现状检查、恢复接纳、算术自检和发布前proper-factor/身份确认继续保留，它们检查的是不同事实。

### 已检查出的重叠及精简决定

| 重叠或风险 | 本方案采用的精简 | 删除或合并条件 |
| --- | --- | --- |
| driver和task同时列出七步；可能形成main→Driver.run→Task.run→Session.run | 七步只放run_stage1_task；driver选择直接或队列入口；Session仅拥有状态 | 只转交同一请求、没有自己的程序级决策的Driver/TaskRunner层不建立。 |
| task.cpp、session.cpp被目录清单误解为两个公开执行Module | task.cpp默认含私有Session；不规定拆文件 | 若只是Task.run调用Session.run，则保留唯一入口与一份编排。 |
| compose选中实现、factory.create、executors.select重复选择 | application只组装创建能力；task一处create；prepare接纳实际执行组合 | 不保留同一请求上的create+registry.select双路径。 |
| Queue.next/complete与run_worktodo都被称为完整workflow | 公开入口统一run_queue_workflow；完成/失败处理收为私有函数；QueueStore只做文件机制 | workflow外再包完整QueueService，或store反向调用task的设计不建立。 |
| Bindings从入口一路全量透传 | 应用owner留在应用；task与queue各接收有名窄依赖 | 不增加仅剪裁或转交Bindings的新Module。 |
| ExecutionService只转发factory/Executor；每个phase/helper各建类 | service/execution可只定义合同；私有函数组织阶段细节 | 没有共享规则、状态、资源或错误翻译时，不建运行转发层/独立实现库。 |
| 旧backend兼容链、新factory或provider只改调用名 | 有真实创建/兼容责任才保留；单一固定实现可直接构造 | 所有生产调用迁完且无校验、资源、ABI/错误翻译职责后删除纯转发；不能按名字直接删除Adapter。 |

验收用删除测试（deletion test）：删除一层后若只少一次同参转发，没有规则散落，应合并；若使直接运行和队列各自复制恢复/发布规则，task应保留；若使多个调用者自行拼接finished/源队列/交接顺序，QueueWorkflow应保留。该判断用于是否建立独立公开Module/调用层，不因私有helper或访问器很短就全部内联。这是Implementation提取与P2/P4审核条件，不是本轮已删改源码的声明。

上述调用图以Stage1为例。Stage2保留自己的父子receipt与来源完成workflow，可共用QueueStore机制，不为统一命名再建立通用Task基类或强套Stage1循环。

## Interface 的数据、输入输出与所有权

下列是 C++17 概念合同，不是拟立即发布的SDK或稳定跨版本ABI。GMP继续是数学依赖；不引入大整数虚接口、std::span、std::expected 或 C++20 concepts。

### 关键数据

| 数据 | 必需内容 | 所有权与不变量 |
| --- | --- | --- |
| Stage1Request | target N、来源metadata、B1/B2现有语义、CurveBatchSpec、执行偏好、输出策略 | 接纳后不可变；输出策略不传入CUDA执行请求。时间/进度不再塞回const请求。 |
| TaskDependencies / 私有TaskSession | 创建Executor函数、checkpoint/publisher/cache与控制/进度引用 / 一次任务身份、锁、上下文、状态 | 前者是窄借用集合，不是service locator；后者只在task内拥有运行状态，不提供第二个公开run。 |
| CurveBatchSpec / ResolvedCurveSet | family、仅Mont存在的param、sigma策略/数量；prepare接纳恢复载荷后记录实际sigma | session拥有逐曲线身份目录；缺失身份可在prepare内按现有策略确定，一旦绑定不再改变。恢复以已保存sigma为准，不假定Edwards为first+i。 |
| ScalarPlan | K、1/12/48 LCM等现有构造身份、精确有效范围/计划身份 | 不变、共享借用；后端编码私有。缓存属于主机存储，不能把文件读取放进core。 |
| ExecutionInput | 数学请求视图、CurveBatchSpec/会话身份目录、ScalarPlan、后端执行配置 | 没有worktodo、TOML对象、结果路径或GUI字段；准备/执行期间有效。RestoreView单独传入，载荷解释留在prepare。 |
| PreparedExecution | 实际选择、后端私有曲线状态/资源、恢复位置 | 移动拥有的不透明对象；由Adapter释放，不能把CUDA/IFMA内部结构暴露给driver。 |
| CheckpointEnvelope / RestoreView | task/curve identity、kind/version、进度单位、后端兼容标记、载荷 | 外壳由存储校验；Adapter判断载荷能否恢复。view借用session，不能跨session持有。 |
| SnapshotView | 身份、已完成安全点位置、host载荷/坐标、状态域与版本 | 仅同步提交调用内借用；writer返回后不得继续读取。无device pointer。 |
| ProgressFacts | 阶段、曲线/批次、done/total、单位、wall/device计时边界 | 不带格式文本；binary bit、NAF column、PRAC step不混用；不重复相加嵌套时间。 |
| CurveOutcome / BatchOutcome | 已分类状态、proper factor、规范化完成点、实际sigma、统计 | 拥有整数/数组，避免手工mpz_clear/free；有factor与有Stage2点是不同维度。 |
| TaskResult / PublishedTask | Completed/Stopped/Failed、错误来源、已可靠提交结果、恢复信息；完整发布凭据 | Completed才有完整批次确认；Stopped不代表无因子，不能为未完成曲线发布NF。PublishedTask不宣称多文件原子事务。 |
| Stage2Input / PlanDecision | 来源完成点、target N/carrier M、bounds、实际执行选择/资源模型 | 保留当前save接纳范围；plan注明设备副作用与预测/实测来源。 |
| QueueTicket / QueueItem | 原始来源/worker/选取身份 / ticket、规范化请求与队列完成策略 | ticket提交前重核来源，不能只保存行号；QueueItem不把完整TOML模型带进service。 |
| WorkerSnapshot | 只读worker状态 | UI不拥有执行线程、设备或进程句柄。 |

有效因子仍满足 1<f<N 且 f整除N，复合proper factor可输出；GCD=N按当前退化/饱和合同处理。规范化完成点与中途checkpoint不是一个类型。没有通过检查的miss不能被包装为有效Stage2点。

### 粗 Interface

| Interface / 操作 | 输入 | 输出与失败合同 |
| --- | --- | --- |
| Configuration.load/normalize | 运行模式、CLI/TOML/来源值 | Settings或输入错误；兼容现有优先级，不借重构改变默认值。 |
| CurveMath.construct / ScalarBuilder.build | target、CurveSpec / B1及构造策略 | checked普通点/构造因子/退化结果；K与数学计划。无文件或设备副作用。 |
| run_stage1_task | Stage1Request、TaskDependencies | 拥有结果的TaskResult；唯一一次任务入口；无需再建Stage1Task/TaskRunner转发类。 |
| create_executor（工厂函数） | 阶段、已解析的后端选择与窄配置 | 拥有的Executor或创建错误；公共声明无SDK类型；每任务一处创建，不与registry.select重复。 |
| Stage1Executor.capabilities（按需） | 查询/诊断请求 | 实际编译能力；不强制每任务额外全局查询。组合接纳可由prepare内部共享校验完成。 |
| Stage1Executor.prepare | ExecutionInput、RestoreView、ExecutionContext | 接纳组合、设备约束和恢复态，返回PreparedExecution或Stopped/Failed；准备期间也能发现因子并提交。 |
| Stage1Executor.execute | PreparedExecution、ExecutionContext | BatchOutcome；返回前worker已join、设备已到所需安全状态，停止/错误包含恢复信息。 |
| ExecutionContext.safe_point | ProgressFacts | Continue / CheckpointThenContinue / CheckpointThenStop / Failed；控制与展示分清合同。 |
| ExecutionContext.store_checkpoint | SnapshotView | 同步Ack或错误；成功返回后才允许复用借用buffer。 |
| ExecutionContext.publish_factor / accept_curve | curve identity + proper factor / 已分类完成点 | 同步Ack或错误；普通主机publisher执行即时ledger/save职责，Adapter不认识文件格式。保持各路径原时机。 |
| CheckpointStore.open/write/finish | identity / host record / 成功标记 | 锁与RestoreView / Ack / 清理或保留；不会因单条active record remove删除整个batch。 |
| Publisher.finish_batch | 完整BatchOutcome与发布上下文 | PublishedTask或输出错误；确认save，完成submission→internal batch→seal。 |
| run_queue_workflow | 下一条规范化任务的窄回调、TaskDependencies、队列/交接存储 | QueueResult；唯一循环与完成协议入口，私有函数处理失败和来源提交。 |
| QueueStore.read_next/update_expected | 读选条件 / expected ticket及文件改写值 | 数据、Empty或I/O结果；仅文件机制，不调用task或决定业务成功。 |
| Save/Profile/PlanCache | 读取选择、校验身份或待写的host值 | 已校验模型、CacheMiss或I/O错误；miss与损坏/不可接纳明确区分。 |
| Stage2Planner / Stage2Executor | 候选+profile+设备快照 / 已验证续算输入与计划 | PlanDecision / 受检查的执行结果；不支持能力明确返回，不伪造CPU/GPU等价参数。 |
| WorkerController | 配置、start/stop命令、tick | 只读WorkerSnapshot和操作错误；拥有worker编排，底层进程机制由Process Adapter负责。 |

ExecutionContext 只有三组协作：安全点与控制/进度、同步checkpoint、可靠publication（含即时factor和完成点）。可以是一个私有上下文的几个方法/函数引用，不需要ProgressService、StopService、SnapshotService、FactorObserver多套框架。各调用不是热路径逐点调度。

CPU并发回调由session/相应writer保持串行化规则；借用寿命覆盖prepare/execute；禁止重入同一session。恢复容器打开不代表已知全部sigma：Adapter在prepare内解码载荷、按现有规则补足缺失身份，并绑定到session后才构造/发布该曲线，后续不得重抽。身份目录访问是上下文的数据合同，不新增事件框架。

对象化前仍保持当前单实例checkpoint与现有进程隔离，未验证前不宣称同进程多任务并发安全。设备失效等现有fatal退出不自动变成可恢复错误，单独核对后才调整。

## driver 主流程伪代码

这里将程序入口、单次任务和队列workflow分开表达。它们不是依次包装同一个run；所有类名/辅助函数都只是C++17职责示意。compose_runtime是应用私有组装代码，可以直接写成局部构造，不新增Runtime/CompositionService。

### driver：只做入口与组装

```cpp
int main(Args args) {
    auto command = parse_command(args);
    if (command.is_query()) return run_query(command);

    auto input = load_and_normalize_for_mode(command);
    auto runtime = compose_runtime(input);       // 应用内owner与创建函数
    TaskDependencies tasks {
        runtime.create_executor, runtime.checkpoints, runtime.publisher,
        runtime.plan_cache, runtime.signals
    };                                          // 仅必要引用，不透传runtime

    if (input.is_queue_mode())
        return exit_code(run_queue_workflow(
            runtime.next_queue_task, tasks, runtime.queue_stores));

    for (auto target : input.direct_targets()) {
        auto request = make_stage1_request(target, input);
        auto result = run_stage1_task(request, tasks);
        if (!result.completed()) return exit_code(result);
    }
    return existing_success_exit_policy();
}
```

next_queue_task是应用提供的窄回调：按现有规则重读当前TOML，经QueueStore读选并归一化，返回QueueItem/Empty/ReadError。它不运行计算或完成队列。TaskDependencies不包含queue、完整配置和应用owner；这里只组装可用factory，不提前为某任务创建Executor。

### task.cpp：唯一一次任务编排

```cpp
TaskResult run_stage1_task(const Stage1Request& request, TaskDependencies& deps) {
    auto admitted = validate_task_invariants_and_outputs(request);
    if (!admitted.ok()) return Failed(admitted.error());

    // 私有Session只拥有状态；open建立锁、恢复容器和发布上下文。
    auto opened = TaskSession::open(
        request, deps.checkpoints, deps.publisher, deps.signals);
    if (!opened.ok()) return Failed(opened.error());
    auto session = opened.take();
    // 默认析构保留可用恢复材料；这里不再调用Session.run。

    auto created = deps.create_executor(request.execution_selection());
    if (!created.ok()) return Failed(created.error());
    auto executor = created.take();             // 每任务一处创建，无二次select

    auto scalar = prepare_scalar_with_cache(session.scalar_request(), deps.plan_cache);
    if (!scalar.ok()) return Failed(scalar.error());

    auto prepared = executor->prepare(
        session.execution_input(scalar.value()),
        session.restore_view(), session.context());
    if (!prepared.ok()) return as_task_result(prepared.status());

    auto batch = executor->execute(prepared.value(), session.context());
    if (!batch.completed()) return as_task_result(batch.status());

    auto checked = session.finalize_outcomes(batch.take());
    if (!checked.ok()) return Failed(checked.error());

    auto published = deps.publisher.finish_batch(checked.value());
    if (!published.ok()) return Failed(published.error());

    session.finish_completed();                 // seal后结束checkpoint，不二次发布
    return Completed(published.take());
}
```

这七步只在task内存在：**校验→打开会话→准备→执行→核对→最终发布→完成会话**。as_task_result只是状态/错误来源转换，不触发另一轮恢复或I/O。直接返回成立的前提是Adapter在Stopped前已经完成合同要求的安全快照，Session默认释放资源并保留材料；不为显式“retain”名称增加无行为包装。

finalize_outcomes不重复GCD/归一化或即时factor提交；finish_batch按当前ledger规则完成最终发布；finish_completed不隐藏第二次finish_batch。prepare保留各后端局部次序与实际能力/设备/载荷检查，driver不会逐步拼接CUDA buffer、SIMD lane或字节编码。

### QueueWorkflow：唯一队列循环与完成协议

```cpp
QueueResult run_queue_workflow(NextTask next_task,
                               TaskDependencies& tasks,
                               QueueStores& stores) {
    for (;;) {
        auto selected = next_task();            // 包含按现有策略重读配置与选取
        if (selected.empty()) return QueueEmpty;
        if (selected.read_error()) return Failed(selected.error());
        auto item = selected.take();             // ticket + request + queue policy

        auto result = run_stage1_task(item.request, tasks);
        if (result.stopped()) return Stopped;    // 不推进未完成来源任务

        if (result.failed()) {
            auto action = handle_task_failure(item, result.error(), stores);
            if (action != RetryOrContinueAllowedByCurrentPolicy)
                return Failed(result.error());
            continue;
        }

        auto committed = complete_queue_task(
            item, result.published_task(), stores);
        if (!committed.ok())
            return Failed(committed.error());   // 不静默重算并重复append
    }
}
```

NextTask只是可调用入口，不规定新的来源类。handle_task_failure与complete_queue_task是QueueWorkflow同一Implementation内的私有函数，不是额外公开Service。后者拥有park→finished→source更新→save sync→Stage2 flush→Prime95 delivery；QueueStore/Transfer Adapter只执行具体文件协议。Prime95 pending与计算失败不同；原失败/停止策略不变。

### 执行期间的三组协作

```cpp
// 只示意安全点合同，不规定每个backend必须采用相同机械循环。
auto decision = context.safe_point(progress_facts);

if (decision.needs_checkpoint()) {
    auto host_view = adapter.capture_safe_host_state();
    // CUDA在这里或此前完成必要同步/D2H；CPU完成lane身份转换。
    if (!context.store_checkpoint(host_view))
        return Failed(SnapshotPublicationError);
}
if (decision.stop_requested()) return Stopped;

// 也可发生于prepare的构造/字典求逆；具体先后保持现有路径。
if (proper_factor_found &&
    !context.publish_factor(curve_identity, factor))
    return Failed(ResultPublicationError);

if (curve_completed &&
    !context.accept_curve(classified_host_outcome))
    return Failed(SaveOrPublicationError);
```

主机快照writer无需CUDA；同步/D2H仍是Adapter行为。不能把暂态device pointer交给writer，也不为screen刷新额外下载全部曲线。CPU callback可直接持久化或先PAUSED返回再由外层持久化，不为整齐统一循环结构。

必须保留当前细节：Edwards scalar的capture/DONE可先于factor发布，IFMA和Montgomery的具体顺序不同；最终submission→internal→seal→end_batch(true)不变；停止不得为未完成曲线发布NF。当前依据为 [ecm_driver.cpp](../../src/core/ecm_driver.cpp) 第 3079–3104 行及前文逐后端细表。

### 独立 Stage2 与 GUI

Stage2有自己的状态机，不套用Stage1的checkpoint循环。其owner默认保留progress/pending；CUDA父进程流程示意中，checked表示失败即停止推进并由会话保留状态的错误检查：

```cpp
auto task = checked(stage2.open_and_validate(source));
checked(task.recover_completed_pending_receipts());

for (auto curve : task.remaining_curves()) {
    if (stop.requested()) return Stopped;

    auto pending = checked(task.persist_pending_before_launch(curve));
    auto exited = worker.run(task.worker_request(curve, pending));
    if (!exited.success()) return Failed(exited.error());

    if (!publisher.has_durable_receipt(pending))
        return Failed(MissingReceipt);
    checked(task.persist_curve_completed(pending));
}
checked(publisher.seal_task(task.identity()));
checked(complete_stage2_source_after_recheck(task, queue_stores)); // workflow私有
checked(task.remove_completed_progress());
```

子进程重读save并验证fingerprint后调用Stage2Executor；完成异步算术检查，再发布结果和receipt。exit=0不能替代receipt。非零退出沿用当前保留/报错，后续重启才按durable receipt恢复，不能悄悄改成当场成功。依据为 [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp) 第 617、898、914–920、1505–1508、1551–1585 行。

CPU Stage2保留当前Edwards save直接续算，不新增CUDA式队列/Auto B2/父子receipt。GUI的WorkerController负责configure→选择exe→启动/监管→事件归并→停止/重启→快照；视图只提交命令并读快照。Stage1安全点停止、Stage2当前曲线结束后停止和等待子进程退出的现有语义保留；看见checkpoint文件不表示可以强杀。GUI监管与Stage2父子编排可复用进程机制，不强行合成总Supervisor。

## 现有文件到目标模块的迁移映射

| 当前家族 | 目标 | 迁移约束 |
| --- | --- | --- |
| ecm_driver.cpp | application/cli、service/stage1、service/work、backend/cpu | 先按完整职责提取；不把原二十多个参数搬到公共巨型options。 |
| ecm_edwards_gpu_driver.inl | service/stage1 + CUDA host prepare/Adapter + storage | 拆掉对driver static helper的文本依赖；发布时机保留。 |
| ecm_stage1_selection、curve、exp、NAF/PRAC计划 | core/curve、core/stage1；cache归storage | 数学与后端选择/用户解析分开，曲线身份不随backend变化。 |
| ecm_checkpoint、mont_ckpt、Ed checkpoint helpers | storage/checkpoint + backend私有载荷codec | 外壳/发布共享；算法状态与版本接纳由Adapter拥有。 |
| ecm_save、Ed io/save reader | storage/save + core点检查 | 兼容格式和消费者范围不变。 |
| ecm_result/submission/internal/file_publication | storage/results + platform + service/work | 编码/账本细节收回.cpp，可靠确认与完成次序仍独立。 |
| ecm_queue_config、worktodo、p95_worktodo/transfer | application/configuration、storage/queue/transfer、service/work | 解析、文件机制和推进策略按职责分；不扩展为工具内部重构。 |
| src/cpu/ecm_mont_cpu、ecm_edwards_cpu、simd_* | backend/cpu/stage1与arithmetic | scalar/SIMD状态、worker/join、lane转换保持完整；共享构造数学归core。 |
| src/cuda/stage1 + kernels/cuda/stage1/optional/prac | backend/cuda/stage1 | 包括resident/PRAC的配置、计划cache、checkpoint与screen隔离。 |
| src/cuda/stage2 + kernels/cuda/stage2 | backend/cuda/stage2 | 延续已有引擎分工；设备常量单一所有者不变。 |
| src/cuda/ntt + kernels/cuda/ntt/shared | backend/cuda/ntt与arithmetic/所属阶段 | Graph先于借用buffer释放；按真实复用归属，不制造泛化common。 |
| include/ecm/cuda的launch/types/tiers | backend/cuda私有src | 必要factory与阶段合同才留include；不暴露设备模板。 |
| ecm_stage2_geometry/catalog/probability/relocation | core/stage2；设备筛选归CUDA planning | target/carrier、树布局、区间和算术不变量保持权威定义。 |
| ecm_stage2_tune_*、portable_model | service/planning、backend模型、storage/profile | 搜索/预测/采样与TOML分开；生成字段表仍可私有小文件。 |
| ecm_cuda_stage2_main/portable_driver | application/cuda_stage2、service/stage2/planning/work | 保留父子进程、pending、receipt、seal和源队列复核。 |
| ecm_cpu_stage2_main、gwpoly | application/cpu_stage2、backend/cpu/stage2 | 保留第三方生命周期和当前接纳范围。 |
| src/opencl_*、cl_probe、kernels/opencl/impl_opencl.cpp及生产资源 | backend/opencl；中立日志归platform | 包括生产自检所需bench/mont.cl，保持动态组装/cache键。 |
| src/gui生产文件 | application/gui、platform/process、已有纯模型 | 保留WorkerProc；提取完整Controller，独立fake/test目标不迁入主流程。 |
| 项目ecm.h/cgbn_stage1旧入口 | core/service/私有兼容Adapter | 先保留窄兼容转发，调用迁完后删除无用转发；不动第三方内部。 |

## 分阶段正式重构顺序

每阶段独立可审核、可构建并有退出条件。物理迁移、拥有权变化、算法/格式变化分开；后两类未经本方案授权新增的行为不混入组织重构。以下为各阶段交付目标；实际进度为 P0–P5 完成，P6–P7 未完成，详见“实施进度与阻塞记录”。

### P0 冻结现状与生产闭包

交付：当前主程序/源清单、完整支持矩阵、构建参数与工具链、对象/exe/文件样本身份、曲线/标量/恢复/发布顺序；保留并行未提交状态的独立快照。记录本轮附件但不把其当作干净提交或性能基线。

通过条件：能定位每个生产入口和资源；强制自检、OpenCL诊断资源、设备常量、默认流、环境变量读取时机、错误/停止路径均纳入。现有已知限制单列，不能在重构时掩盖。

### P1 首先完成构建复用和目录隔离

P1a保留当前编译器/算术/链接策略，统一逐TU默认reuse、失效理由、HostOnly/SkipBuild规则和owner graph。Stage2去掉“整体变更就全组重编”，Stage1 local/dev/release不再默认主动重编所有CU。

P1b落实build_cuda_local/dev/release各stage/engine/toolkit/arch/profile叶目录、marker、日志和清单位置，更新包装入口与消费者定位；保留旧目录原用途兼容，不直接搬缓存。CPU/GUI/OpenCL补齐入口时遵循相同role规则，当前缺失能力明确标注。

通过条件：同profile无修改二次构建零CXX/零CU且无不必要link；独立host源码改变零CU；改设备依赖/命令正确失效；HostOnly对失效对象提前拒绝；显式跨用途目录拒绝；包与build不混用。新目录首次冷构建和旧目录增量分开记录。P1通过后才做主体源码提取。

### P2 收口合同、拥有权和普通 CXX 编译

交付：Stage1Request、拥有结果的TaskResult、私有TaskSession与粗ExecutionContext；以run_stage1_task为唯一任务入口。现有入口先由私有Adapter承接，不同时重写算法。移除请求被当作运行状态/强制const转换和手工数组拥有权扩散；按“职责划分与重复封装检查”去除同参转发、重复factory/select与Bindings全量透传。

独立子步骤让Stage2 host .cpp切普通CXX；先校验CRT/宏/异常分配/GMP/CUDA链接，再改消费者。此步骤不要求runtime脱离CUDA，不能把“少CU重编”与“无需SDK”混为门槛。

通过条件：CLI/queue共用一次任务入口，直接与队列语义不变；公开头自包含且不含设备模板/大writer实现；错误来源、Stopped与Completed明确；旧构建与发布清单仍可核对。

### P3 完成目标目录与后端实现归位

交付：按目标树迁入src/ecm；CUDA生产实现完全离开kernels/cuda；生产OpenCL主机与资源一并归位；更新源清单、include、owner graph、资源loader和打包规则。保留必要旧入口转发以便逐步接入。

迁移只改位置/引用，不改点公式、模板实例、默认流、常量所有者、强制检查和释放顺序。路径变化导致的合法重编在此集中记录，预期可能需要一次完整CUDA编译，不能为了复用放宽指纹。

通过条件：生产闭包不再引用kernels/cuda和实验实现；实际tiers/参数族/实例注册一致；OpenCL资源可加载；NTT/Stage2常量和Graph寿命保持；相关消费者仅作必要路径适配。

### P4 深化 Stage1 Module 并隔离非 GPU Implementation

交付：整段提取CPU曲线Adapter、GPU Montgomery和Edwards host workflow；配置/构造数学、checkpoint/save/result/tune/screen函数体落普通.cpp；.cu只保留必要协作声明和设备职责。resident/PRAC同步处理。

task实现收敛为前述七步；application driver仅保留输入、组装、模式选择与展示/退出。QueueWorkflow拥有任务外循环和完成协议，QueueStore只做文件机制。先保持各后端GCD、save、factor、DONE与快照时机，再删过时转发；不按线程/lane/helper继续过拆。

通过条件：四类非GPU功能的代表改动只重编受影响CXX并链接；screen不再因内联格式使Montgomery runtime.cu失效；恢复、即时提交失败、暂停和queue推进与旧版一致；.cu无法通过新头重新包含完整配置/writer。

### P5 收口 Stage2 规划、持久化和运行状态

交付：唯一父任务流水线、完整曲线 worker、独立进程协议与实现，明确纯数学/设备模型/profile文件归属；不新增只有转发职责的 Stage2TaskSession/CurveWorkerRunner。现有15个host引擎.cpp继续保留完整职责，不重新造单体引擎。

环境读取与全局状态实例化单独实施，先保留现有进程隔离；可恢复错误翻译与fatal CUDA错误行为分别核对，不声称已支持同进程多会话。

通过条件：pending先落盘、子进程重读save校验、receipt确认、seal和source复核顺序不变；异步oracle在成功/重放决策前排空；Auto B2/D与tune字段/身份/计时合同一致；CPU入口不被扩大能力。

### P6 GUI 与独立 CPU 入口归位

交付：WorkerController与只读快照，绘制不再编排worker；共享配置/队列读选和现有纯模型；CPU Stage2后端与正式构建入口清晰。Process Adapter保留具体句柄/pipe/Job/thread管理。

通过条件：启动、停止、重启、崩溃/日志和阶段切换贯通；不因看见checkpoint强杀；测试/fake-worker不默认进入正式包；CPU/GW/libecm支持与依赖不变。

### P7 综合验收、文档同步与收尾

交付：新结构与构建说明、迁移映射、当前支持/限制、依赖报告与验收证据；调用全部迁完后删除无价值兼容转发。正式现状文档改写为实际实现，草稿提示用户处理，不自动删除。

通过条件：公开依赖无环，生产清单和包身份完整；主流程、恢复/故障、各实际支持后端与位宽覆盖通过；性能比较满足冻结条件且未引入未经说明的退化。多架构镜像构建、PTX/JIT和实体GPU原生验收分开记录，未具设备不宣称通过。

## 验收矩阵与待确认事项

| 维度 | 必须验证的结果 | 所属阶段 |
| --- | --- | --- |
| 职责与封装 | 一份task编排、一处Executor创建、一份队列完成协议；Session不再run；窄依赖且无只转发同参的新增层 | P2、P4 |
| 构建增量 | 无修改0编译；host-only修改0CU；设备header/宏变更仅正确依赖失效；对象篡改拒绝 | P1、P4 |
| 输出隔离 | role/engine/toolkit/arch不混用；旧目录不自动迁移；exe/资源/DLL/清单一致；运行文件不被覆盖 | P1、P3 |
| 数学与结果 | 固定N、curve、param、sigma、B1/有效K比较；proper/composite factor、退化与有效点合同不变 | P0、P2–P5 |
| 恢复与故障 | 保存/读回位置、损坏/主备接纳、停止、发布失败、已有因子、未完成NF与queue行为 | P2、P4、P5 |
| CUDA/OpenCL执行 | tiers/实例/常量、资源、同步/计时、自检、Graph借用寿命；无新增逐点虚调用 | P3–P5 |
| Stage2 | save fingerprint、pending/receipt、target/carrier、异步检查、重复发布与source复核 | P5 |
| GUI/CPU | 子进程状态、平滑停止、实际支持范围、普通CXX和第三方运行依赖 | P6 |
| 性能资格 | 相同输入、设备/功耗、源码与二进制、样本数和计时边界；原始证据不改写 | P7 |

这些是未来验证工作，不在本轮修改测试/工具内部的范围。必要的现有测试/工具路径适配随生产移动同步进行，不能通过删除测试、放宽门限或改变正确实现消除表面不一致。

仍需在实施时核对的条件：

- 最终profile字段与短key长度由真实命令/清单决定；不得把源码hash作为目录key。跨目录artifact import尚未设计与验证，初期不承诺。
- GUI/打包/现有脚本对exe根和运行时资源路径的全部消费者，P1/P3生成清单后逐一核对。
- 共享源能否跨exe共用对象取决于目标宏与ABI；无证明时继续按目标编译。
- 同进程多会话、跨后端checkpoint互换、错误恢复升级、结构化GUI进程协议属于另行验证/设计的能力，不由本次组织重构自动获得。
- 实际支持组合以当前生产接纳和本机资格为准；当前CPU Stage2/GUI缺少的正式构建入口族是计划交付，不是现状声明。

## C++ Core Guidelines 参考

采用其接口显式化、资源所有权和依赖控制原则：I.1/I.4、I.2、C.129、R.1、SF.9/SF.11。I.27 的 Pimpl 适合隐藏大型设备对象，不要求每个值类型增加间接层。本文的目录、粗Interface、profile和阶段顺序是本仓库的设计，不是规范规定的模板。

来源：[C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)，前文查阅日期 2026-10-21。
