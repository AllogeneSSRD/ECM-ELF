# 项目领域与阅读入口

ECM-ELF 提供多后端 Stage1、独立 CUDA 多项式 Stage2、Windows GUI 和相关工具。Stage1/Stage2 仍为独立程序，配置定义共享，同一 worker 的 worktodo 按 Stage1/Stage2 前缀选择程序。

当前领域术语与模块职责以 [架构与术语](docs/architecture/OVERVIEW.md) 为准；功能和限制从 [文档目录](docs/README.md) 查阅，未完成工作见 [TODO](docs/TODO.md)。不要从外部源码或冻结实验记录推断当前生产行为。

文档维护及冲突处理遵循根目录 [AGENTS.md](AGENTS.md)。配置键在 `config/ecm_options.json` 定义，参考说明由生成器维护。

主机读写（`src/io/`）负责 CLI、配置和 worktodo 的来源解释与队列编辑；归一化负责默认、覆盖和完整任务。CLI 每次出现一次转换，TOML 保留启动时的已解析快照；normalize_settings 生成有效设置，不回填 CLI。当前任务输入冻结，未来任务可编辑；同一 worker 内第一条完全同文的行是匹配对象。保存范围是 save 中连续 METHOD 记录的 skip/count，不是物理行号。完成记录保存已完成数学工作的队列提交责任；P95 交接记录保存冻结的文件版本、范围、目标和路由，二者不充当新的 worktodo 任务 ID。

Stage1最终输入由CLI/stdin和worktodo两个适配入口产生，归一化后计算字段冻结，队列票据与交接元数据分离。main按当前任务选择普通执行函数，task直接调用，后端返回拥有的ComputationResult；task统一最终save与结果发布，保存区间显式返回。Montgomery各param共用一条workflow，param是构造身份，算法和设备布局受其约束。

计算目录按领域聚合：`src/ecm/stage1` 拥有跨曲线共同流程，`edwards_stage1`、`montgomery_stage1` 拥有各自完整的 CPU/CUDA/OpenCL 能力；`stage2` 拥有共享续算引擎和纯主机规划，`curve` 拥有跨阶段构造。真实共享的 `cpu`、`cuda`、`opencl` 技术支持继续独立。保存命名、旧历史快照和持久发布由 IO 拥有，IO 不依赖 Stage1Options。没有旧路径转发头，CLI、导出符号和磁盘格式保持兼容。目录、依赖和具体入口以[架构与术语](docs/architecture/OVERVIEW.md)为准。实施草稿与逐文件证据保留在[领域目录重构](docs/web/ecm-domain-layout-review-20261026.html#implementation-result)。

Stage1 的 sigma Module 拥有随机生成、独立/连续分配和整数范围；各后端先按原合同验证 checkpoint，再消费确定的构造身份。共同标量/cache/log 编排和保存 Naming 投影归 task；纯 scalar、结果 computation、CPU runtime 与 libecm 兼容实现保持各自可独立链接的职责。
