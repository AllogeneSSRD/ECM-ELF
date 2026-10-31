# ECM 项目文档

本仓库提供 CUDA/OpenCL GPU Stage1、CPU Stage1、独立 CUDA/OpenCL 多项式 Stage2、GWNUM/polymult 与 libecm CPU Edwards Stage2，以及 Windows 图形前端和数据集工具。文档描述当前源码；本机已有发布包是否包含某功能，应核对其构建身份。

CUDA 支持 Atkin–Morain Edwards Stage1 与同曲线的 NTT Stage2 交接；使用、存档身份、检查点和当前限制见 [GPU Edwards](architecture/STAGE1.md#gpu-edwards)。

## 使用

- [Stage1](usage/STAGE1.md)：曲线计算、指数模式、保存点、统一批次检查点、双结果输出和 Prime95 交接。
- [Stage2](usage/STAGE2.md)：读 Stage1 save、任务字段、数量不足、队列续跑、日志和结果。
- [Edwards CPU Stage2](usage/EDWARDS_CPU_STAGE2.md)：统一文本存档、GW 多线程快速多项式、libecm NTT/plain 续算及大 B2 验证。
- [ECM 结果输出](usage/ECM_RESULTS.md)：内部 JSONL、Prime95 结构提交文件、TOML 身份、参考 FFT 长度、批次去重与本地恢复。
- [GUI](usage/GUI.md)：worker、队列生成、监控及结果。
- [GUI v2 候选](usage/GUI_V2.md)：默认DX11、可迁移侧栏、独立分组设置/语言、交互图表、群阶/点阶分解与结果文件导入、[图标图鉴](web/gui-v2-icons.html)、Ely宠物与[三款装甲皮肤预览](web/ely-skins-20261022.html)；Vulkan显式诊断路径及验收限制；[草稿与进度](development/GUI_MODERNIZATION_DRAFT.md)、[HTML 阅读版](web/gui-v2-vulkan-plan-20261021.html)。
- [构建与发布](usage/BUILD.md)：双 CUDA 完整发布入口、分阶段构建、[GPU 型号 / SM 与选包速查](usage/BUILD.md#gpu-型号与发布包选择)、工具链/PTX兼容边界、CUDA/GMP 依赖和发布内容。
- [Release 改造草稿](development/RELEASE_BUILD_REDESIGN_DRAFT.md)：双 CUDA 完整包、GUI v2、统一 vcpkg GMP、nvcc 计时和按需 JIT 的需求与实施验收记录；当前使用方式见[构建与发布](usage/BUILD.md)。
- [ptxas 沙箱编译诊断草稿](development/PTXAS_SANDBOX_DIAGNOSIS_DRAFT.md)：CUDA 13.3.73 单文件并行与严格句柄检查的最小复现、因果对照及已验证的串行编译处理。
- [因子数据集](usage/DATASET.md)：PARAM0数据库、最优sigma、扫描与生产验证，以及Edwards独立群阶/点阶分析。
- [ECM 报告工具](../tools/ecm_report/README.md)：PrimeNet 进度导入、两日变化图和按日期及指数范围统计的增长热度图。
- [TOML 配置参考](ECM_TOML_REFERENCE.md)：所有键的符号默认值、范围和英中说明；由配置定义生成。

## 实现

- [架构与术语](architecture/OVERVIEW.md)：模块职责、数据表示和入口。
- [CPU/GMP 群阶与点阶模块](architecture/GROUP_ORDER.md)：整数分解、BSGS/Mestre、点阶、资源与证明边界，以及 Stage1/Stage2 的 GP 优先自动回退。
- [目录精简规划草稿](development/DIRECTORY_LAYOUT_SIMPLIFICATION_DRAFT.md)：方案 A 已完成，版本 0.45.28；31 个路径调整、14 个公开头保留，GUI 单一 BAT/local 短默认目录与零 CU 验证通过。[网页阅读版](web/directory-layout-plan-20261021.html)。
- [主流程重构方案草稿](development/MAIN_FLOW_REFACTOR_ANALYSIS_DRAFT.md)：P0–P7已完成，源码/local版本0.45.24；保留设计、分阶段记录和证据，当前目录与职责以正式架构为准。[网页阅读版](web/architecture-review-20261021-ecm-main-flow.html)同步实施状态与源码依据，不替代当前架构与构建规范。
- [CUDA 可视化架构图谱](../web/index.html)：代码分层、编译清单、位宽与运行时流程的离线交互网页；维护入口见 [网页说明](../web/README.md)。
- [Stage1 算法](architecture/STAGE1.md)：曲线、标量、ladder/PRAC、域转换和存档。
- [Stage1 主机重构开发草稿](web/stage1-intake-architecture-20261025.html)：首轮0.53.0提交、第二轮0.53.1输入与执行入口、拥有结果及统一发布；保留讨论和验收证据，待用户决定草稿处理。
- [Stage1 第四轮重构开发草稿](web/stage1-fourth-refactor-review-20261025.html#implementation-result)：0.53.3 共同写锁、竞态复现、输入单次转换与归一化；保留实施合同和验收证据，完成后待用户处理草稿。
- [领域目录重构开发草稿](web/ecm-domain-layout-review-20261026.html#implementation-result)：0.53.4 的完整目录集中、保存／发布合同与必要合并；保留 [逐文件清单](development/ECM_DOMAIN_REFACTOR_MANIFEST.json)、复现和验收证据。当前目录以[架构与术语](architecture/OVERVIEW.md)为准，草稿保留待用户处理；延期候选见 [评估页](web/stage1-fifth-refactor-review-20261026.html)。
- [Stage1 下一轮重构候选](web/stage1-next-refactor-review-20261025.html)：包含 0.53.2 计算结果、checkpoint 拥有权及主机 IO 实施结果，保留后续候选与验收证据。
- [Checkpoint](architecture/CHECKPOINT.md)：Stage1 统一外壳、各后端载荷字段与偏移、校验、备份恢复及 Stage2 曲线进度。
- [Stage1 checkpoint 跨后端方案草稿](web/stage1-checkpoint-interop-review-20261026.html)：公共数学状态、计划与游标、CPU/CUDA/OpenCL 能力范围、曲线重组、旧格式导入和验证建议；仅分析，尚未实施。
- [GPU Stage1 数学可视化分析](web/gpu-stage1-visualization-20261026.html)：真实小整数算例的有限域点、ladder 双点、点阶覆盖与因子提取，以及生产数学快照接入建议；尚未连接真实 worker。
- [Stage1 共享文件与sigma优化开发草稿](web/stage1-shared-modules-review-20261026.html#implementation-result)：0.53.5共享文件收口、sigma归一和执行头合并；[阶段目录归位](web/stage1-shared-modules-review-20261026.html#stage-nesting)已完成112文件迁移及CPU/CUDA/OpenCL/Stage2回归及Android NDK构建与导出验证，版本保持0.53.5。[下一轮候选](web/stage1-shared-modules-review-20261026.html#next-candidates)及platform/群阶集中尚未实施。保留讨论与验收证据，checkpoint互通未实施，草稿待用户处理。
- [Stage1 CUDA实例与编译排布草稿](web/stage1-cuda-instance-layout-review-20261026.html#implementation-result)：0.53.6统一构建生成的实施、位宽归一化、PRAC稀疏实例、实时速度窗口及实际验收；保留初始评估和性能测量条件。
- [Save](architecture/SAVE.md)：完成点与因子文本、校验公式、命名前缀、发布去重和消费者/交接边界。
- [Stage2 管线](architecture/STAGE2.md)：每一步的输入、输出、计算量、CPU/GPU 分工及可选 relocation/原区间重放。
- [NTT](architecture/NTT.md)：精确卷积、Goldilocks 算术、布局和设备归约。
- [内存与规划](architecture/MEMORY.md)：容量公式、生命周期模型、准入边界和证据。
- [Auto B2 / tune](architecture/AUTO_B2.md)：收益目标、profile合同、正常运行累计采样、承载优先试跑、实测波动选型与标定资格。
- [配置维护](architecture/CONFIGURATION.md)：定义、生成物、解析器与构建一致性。
- [OpenCL Stage1 与 Stage2](architecture/OPENCL.md)：原生NTT/多项式流水线、生产save/队列接入、CPU尾部及设备资格；[点算子实验](web/opencl-stage2-port-plan-20261027.html#points-closure)。

## 性能与后续工作

- [双 CUDA 发布与 JIT 验证](performance/RELEASE_JIT.md)：0.52.1 发布身份、实际加载模块、数学对照、冷/热 JIT 计时及旧 GPU/驱动的未测边界。

- [本地群阶与点阶分析研究草稿](development/LOCAL_GROUP_ORDER_DRAFT.md)：GP 职责、算法取舍、独立模块数据集验收与主程序回退接入；[HTML 结果](web/cpu-group-order-validation-20261023.html)。

- [主流程重构验收](performance/MAIN_FLOW_QUALIFICATION.md)：0.45.24本机local、完整位宽、非GPU正文零CU、实际恢复/交接与8个代表样例的配对性能范围。

- [GKL family (iv) 曲线实验](performance/ECM_FRIENDLY_IV.md)：有理挠群、约 420 万组独立群阶/点阶、条件 D_eff、GMP NAF 及 GPU Stage2 因子接续与性能边界。
- [Stage1 性能](performance/STAGE1.md)：GPU Edwards细分实例与Montgomery三曲线族同exe成本、冻结初始基准、NAF窗口/倍点图表、PRAC投影和资源约束。
- [Stage2 性能](performance/STAGE2.md)：0.38.4 完整 GPU 矩阵、普通/承载、版本对照、分项分析与 Montgomery 全范围参考。
- [Stage2 精确输入与 Prime95 / CPU 参考](performance/EDWARDS_STAGE2.md)：13 个精确余因子、两档实际 B2、Prime95 两曲线族全范围和保留的 libecm 样本。
- [CPU GW 多项式 Stage2](performance/CPU_GWPOLY_STAGE2.md)：4 GHz、相同精确输入的 Prime95 对照、1/4 线程实测和进程内存。
- [Tensor NTT 性能](performance/NTT_TENSOR.md)：精确Tensor64/40/N96r64、长度/batch、同卡CUDA/Tensor并发及硬件计数器。
- [短NTT Graph性能](performance/NTT_GRAPH.md)：缓存生命周期、固定D完整曲线、batch吞吐及管理员Nsight分析。
- [NTT outer 性能](performance/NTT_OUTER.md)：自动 radix 表、中段 M6 候选、精确门禁和固定 D 对照。
- [Stage2 主机等待](performance/STAGE2_HOST_WAIT.md)：FIFO 回收、可选低 CPU 模式、固定 D 的耗时与 CPU 对照。
- [Stage2 准备与同步](performance/STAGE2_PREPARATION.md)：CPU范围、GPU空隙、真实拷贝与等待、pack/补零融合及计时边界。
- [Stage2 G树布局](performance/STAGE2_GTREE_LAYOUT.md)：固定D/P的NTT长度、收益与回退、算术检查及当前自动资格；[实验草稿](development/STAGE2_GTREE_LAYOUT_DRAFT.md)保留实施与失败证据说明。
- [Stage2 CPU尾部成本](performance/STAGE2_CPU_TAIL.md)：CUDA归一化、少量GCD、块乘积的CPU/GPU对照，完整曲线占比及收益上限。
- [OpenCL Stage2性能](performance/OPENCL_STAGE2.md)：13位宽普通／承载与CUDA快速矩阵、0.54.4默认宽位反馈、显式巨步递推的固定D及固定提交对照、阶段成本和未覆盖范围。
- [算法参考](reference/ECM_ALGORITHMS.md)：论文、Prime95、PrMers 和开源 NTT 的适配要点。
- [固定数学与第三方源码资料](reference/README.md)：保留完整推导、引用代码、伪代码及研究条件，独立于本仓库实现状态。
- [TODO](TODO.md)：尚未完成或需要重新验证的工作。

1094项共享D候选表和跨NTT区间tune已接入源码，当前规则见 [Auto B2 / tune](architecture/AUTO_B2.md)。冻结实验、有限预算查询、379419项参考枚举及未完成边界见 [D候选表草稿](development/STAGE2_D_CATALOG_DRAFT.md)。

候选是否可继续精简、几何包络的局限，以及按P比例跨步与跨区间持续探索，见 [D候选压缩与搜索研究草稿](development/STAGE2_D_SEARCH_DRAFT.md)。0.39.0源码已接入有界搜索，当前验证范围与尚待完成的性能验收见草稿；运行规则以Auto B2 / tune为准。

CUDA 目录与模块重构的冻结实施记录和验证证据见 [CUDA 结构重构草稿](development/GPU_CUDA_STRUCTURE_REFACTOR_DRAFT.md)；当前模块职责以 Stage1/Stage2 架构说明、构建规则以构建与发布说明为准。

## 数据位置

原始运行日志、保存点、剖析和冻结证据在 `data/experiments/` 或工具指定的 `data/` 子目录；统计结果在 `data/benchmarks/`，图表在 `data/figures/`。这些目录不提交 Git，链接到本机证据的页面不能代替原始证据分发。复用脚本保留在 `tools/`，可按其说明重新生成结果。

原论文、译述与论文分析保留在 `docs/paper/`；外部源码与上游工程资料在 `.refactor/`。项目的文档维护规则见根目录 [AGENTS.md](../AGENTS.md)。

- [CUDA Stage1 统一架构实施草稿](development/CUDA_STAGE1_UNIFICATION_DRAFT.md)：获批方案、分阶段状态及验收证据。
