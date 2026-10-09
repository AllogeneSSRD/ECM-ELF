# ECM 项目文档

本仓库提供 CUDA/OpenCL GPU Stage1、CPU Stage1、独立 CUDA 多项式 Stage2，以及 Windows 图形前端和数据集工具。文档描述当前源码；本机已有发布包是否包含某功能，应核对其构建身份。

## 使用

- [Stage1](usage/STAGE1.md)：曲线计算、指数模式、保存点、检查点和 Prime95 交接。
- [Stage2](usage/STAGE2.md)：读 Stage1 save、任务字段、数量不足、队列续跑、日志和结果。
- [GUI](usage/GUI.md)：worker、队列生成、监控及结果。
- [构建与发布](usage/BUILD.md)：六组一键入口、GPU 架构、CUDA/GMP 依赖和发布内容。
- [因子数据集](usage/DATASET.md)：两表数据库、最优 sigma、扫描和生产验证。
- [INI 配置参考](ECM_INI_REFERENCE.md)：所有键的符号默认值、范围和英中说明；由配置定义生成。

## 实现

- [架构与术语](architecture/OVERVIEW.md)：模块职责、数据表示和入口。
- [Stage1 算法](architecture/STAGE1.md)：曲线、标量、ladder/PRAC、域转换和存档。
- [Stage2 管线](architecture/STAGE2.md)：每一步的输入、输出、计算量及 CPU/GPU 分工。
- [NTT](architecture/NTT.md)：精确卷积、Goldilocks 算术、布局和设备归约。
- [内存与规划](architecture/MEMORY.md)：容量公式、生命周期模型、准入边界和证据。
- [Auto B2 / tune](architecture/AUTO_B2.md)：收益目标、profile 合同、搜索和标定资格。
- [配置维护](architecture/CONFIGURATION.md)：定义、生成物、解析器与构建一致性。
- [OpenCL 算子](architecture/OPENCL.md)：路径注册、内核组装和设备约束。

## 性能与后续工作

- [Stage1 性能](performance/STAGE1.md)：已测范围、吞吐量投影和资源约束。
- [Stage2 性能](performance/STAGE2.md)：CPU/GPU 比较、位宽/B2关系、阶段占比和容量收益。
- [算法参考](reference/ECM_ALGORITHMS.md)：论文、Prime95、PrMers 和开源 NTT 的适配要点。
- [固定数学与第三方源码资料](reference/README.md)：保留完整推导、引用代码、伪代码及研究条件，独立于本仓库实现状态。
- [TODO](TODO.md)：尚未完成或需要重新验证的工作。

## 数据位置

原始运行日志、保存点、剖析和冻结证据在 `data/experiments/` 或工具指定的 `data/` 子目录；统计结果在 `data/benchmarks/`，图表在 `data/figures/`。这些目录不提交 Git，链接到本机证据的页面不能代替原始证据分发。复用脚本保留在 `tools/`，可按其说明重新生成结果。

原论文、译述与论文分析保留在 `docs/paper/`；外部源码与上游工程资料在 `.refactor/`。项目的文档维护规则见根目录 [AGENTS.md](../AGENTS.md)。
