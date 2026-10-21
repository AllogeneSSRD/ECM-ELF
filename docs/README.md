# ECM 项目文档

本仓库提供 CUDA/OpenCL GPU Stage1、CPU Stage1、独立 CUDA 多项式 Stage2、GWNUM/polymult 与 libecm CPU Edwards Stage2，以及 Windows 图形前端和数据集工具。文档描述当前源码；本机已有发布包是否包含某功能，应核对其构建身份。

CUDA 支持 Atkin–Morain Edwards Stage1 与同曲线的 NTT Stage2 交接；使用、存档身份、检查点和当前限制见 [GPU Edwards](architecture/STAGE1.md#gpu-edwards)。

## 使用

- [Stage1](usage/STAGE1.md)：曲线计算、指数模式、保存点、统一批次检查点、双结果输出和 Prime95 交接。
- [Stage2](usage/STAGE2.md)：读 Stage1 save、任务字段、数量不足、队列续跑、日志和结果。
- [Edwards CPU Stage2](usage/EDWARDS_CPU_STAGE2.md)：统一文本存档、GW 多线程快速多项式、libecm NTT/plain 续算及大 B2 验证。
- [ECM 结果输出](usage/ECM_RESULTS.md)：内部 JSONL、Prime95 结构提交文件、TOML 身份、参考 FFT 长度、批次去重与本地恢复。
- [GUI](usage/GUI.md)：worker、队列生成、监控及结果。
- [构建与发布](usage/BUILD.md)：六组一键入口、GPU 架构、CUDA/GMP 依赖和发布内容。
- [因子数据集](usage/DATASET.md)：PARAM0数据库、最优sigma、扫描与生产验证，以及Edwards独立群阶/点阶分析。
- [ECM 报告工具](../tools/ecm_report/README.md)：PrimeNet 进度导入、两日变化图和按日期及指数范围统计的增长热度图。
- [TOML 配置参考](ECM_TOML_REFERENCE.md)：所有键的符号默认值、范围和英中说明；由配置定义生成。

## 实现

- [架构与术语](architecture/OVERVIEW.md)：模块职责、数据表示和入口。
- [主流程重构方案草稿](development/MAIN_FLOW_REFACTOR_ANALYSIS_DRAFT.md)：完整目标目录、模块与Interface、driver伪代码及构建先行阶段顺序；当前 P0 local 基线与运行样本已冻结，P1–P7 尚未实施。[网页阅读版](web/architecture-review-20261021-ecm-main-flow.html)同步实施状态与源码依据，不替代当前架构与构建规范。
- [CUDA 可视化架构图谱](../web/index.html)：代码分层、编译清单、位宽与运行时流程的离线交互网页；维护入口见 [网页说明](../web/README.md)。
- [Stage1 算法](architecture/STAGE1.md)：曲线、标量、ladder/PRAC、域转换和存档。
- [Checkpoint](architecture/CHECKPOINT.md)：Stage1 统一外壳、各后端载荷字段与偏移、校验、备份恢复及 Stage2 曲线进度。
- [Save](architecture/SAVE.md)：完成点与因子文本、校验公式、命名前缀、发布去重和消费者/交接边界。
- [Stage2 管线](architecture/STAGE2.md)：每一步的输入、输出、计算量、CPU/GPU 分工及可选 relocation/原区间重放。
- [NTT](architecture/NTT.md)：精确卷积、Goldilocks 算术、布局和设备归约。
- [内存与规划](architecture/MEMORY.md)：容量公式、生命周期模型、准入边界和证据。
- [Auto B2 / tune](architecture/AUTO_B2.md)：收益目标、profile合同、正常运行累计采样、承载优先试跑、实测波动选型与标定资格。
- [配置维护](architecture/CONFIGURATION.md)：定义、生成物、解析器与构建一致性。
- [OpenCL 算子](architecture/OPENCL.md)：路径注册、内核组装和设备约束。

## 性能与后续工作

- [Stage1 性能](performance/STAGE1.md)：GPU Edwards细分实例与Montgomery三曲线族同exe成本、冻结初始基准、NAF窗口/倍点图表、PRAC投影和资源约束。
- [Stage2 性能](performance/STAGE2.md)：0.38.4 完整 GPU 矩阵、普通/承载、版本对照、分项分析与 Montgomery 全范围参考。
- [Stage2 精确输入与 Prime95 / CPU 参考](performance/EDWARDS_STAGE2.md)：13 个精确余因子、两档实际 B2、Prime95 两曲线族全范围和保留的 libecm 样本。
- [CPU GW 多项式 Stage2](performance/CPU_GWPOLY_STAGE2.md)：4 GHz、相同精确输入的 Prime95 对照、1/4 线程实测和进程内存。
- [Tensor NTT 性能](performance/NTT_TENSOR.md)：精确Tensor64/40/N96r64、长度/batch、同卡CUDA/Tensor并发及硬件计数器。
- [短NTT Graph性能](performance/NTT_GRAPH.md)：缓存生命周期、固定D完整曲线、batch吞吐及管理员Nsight分析。
- [NTT outer 性能](performance/NTT_OUTER.md)：自动 radix 表、中段 M6 候选、精确门禁和固定 D 对照。
- [Stage2 准备与同步](performance/STAGE2_PREPARATION.md)：CPU范围、GPU空隙、真实拷贝与等待、pack/补零融合及计时边界。
- [算法参考](reference/ECM_ALGORITHMS.md)：论文、Prime95、PrMers 和开源 NTT 的适配要点。
- [固定数学与第三方源码资料](reference/README.md)：保留完整推导、引用代码、伪代码及研究条件，独立于本仓库实现状态。
- [TODO](TODO.md)：尚未完成或需要重新验证的工作。

1094项共享D候选表和跨NTT区间tune已接入源码，当前规则见 [Auto B2 / tune](architecture/AUTO_B2.md)。冻结实验、有限预算查询、379419项参考枚举及未完成边界见 [D候选表草稿](development/STAGE2_D_CATALOG_DRAFT.md)。

候选是否可继续精简、几何包络的局限，以及按P比例跨步与跨区间持续探索，见 [D候选压缩与搜索研究草稿](development/STAGE2_D_SEARCH_DRAFT.md)。0.39.0源码已接入有界搜索，当前验证范围与尚待完成的性能验收见草稿；运行规则以Auto B2 / tune为准。

CUDA 目录与模块重构的 P0～P4 实施范围、验证证据和后续阶段见 [CUDA 结构重构草稿](development/GPU_CUDA_STRUCTURE_REFACTOR_DRAFT.md)；当前模块职责以 Stage1/Stage2 架构说明、构建规则以构建与发布说明为准。

## 数据位置

原始运行日志、保存点、剖析和冻结证据在 `data/experiments/` 或工具指定的 `data/` 子目录；统计结果在 `data/benchmarks/`，图表在 `data/figures/`。这些目录不提交 Git，链接到本机证据的页面不能代替原始证据分发。复用脚本保留在 `tools/`，可按其说明重新生成结果。

原论文、译述与论文分析保留在 `docs/paper/`；外部源码与上游工程资料在 `.refactor/`。项目的文档维护规则见根目录 [AGENTS.md](../AGENTS.md)。
