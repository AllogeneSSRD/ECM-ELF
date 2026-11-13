# Application 重构开发草稿

本轮以 `6db9826f`（0.62.11）为实施基线，更新为 0.62.12。目录盘点和确认过程保存在本机的 `docs/web/application-architecture-review-20261113.html`。本文件保留实施与验收记录。现行说明分别在[架构总览](../architecture/OVERVIEW.md)、[群阶](../architecture/GROUP_ORDER.md)、[Stage2](../usage/STAGE2.md)和[构建说明](../usage/BUILD.md)。

- 群阶：`application/cli` 的界限推导归 `src/ecm/group_order/group_order.cpp/.h`，终端报告留 Stage1 入口，GP 参数不再经运行时单例。共享 Session 归 `group_order/backends.cpp/.h`；独立 GMP 构建和 PARI 宿主不引入 CLI 或 GUI。
- Android：旧标量按原实现移入 `Android/ECM/app/src/main/cpp/ecm_android_run.cpp`，只有 Android 编译；共用整数表达式归 `src/io/expression.cpp/.h`。
- Stage2：父进程读取一份 TOML 启动快照，用同一解析结果投影 Stage2 和实际继承的 Stage1 键。每项任务从启动选项和当前任务行确定有效值。子进程携带已解析的发布路径、用户名和计算机名，原 TOML 不再参与发布。Edwards CPU 入口原已在计算前固定配置，继续使用。未新增热重载或 checkpoint 字段。
- 源码包：GUI v2 和诊断输入清单改用 `src/io`，GUI 源码归档跳过已从工作区删除的跟踪文件。诊断源码包可禁用再次打包，用包内 GMP 重建宿主和探针。GUI 布局、进程行为和群阶前端未重构。
- Feeder：正式 CMake、发布及使用入口退出旧 feeder。旧源码、MIDSTAGE 编解码与资料归 `tools/p95feeder`；正式 P95 同步及 CPU Edwards Stage1 checkpoint 保留。

## 验收

原始证据位于忽略目录 `data/experiments/application_refactor_20261113/`。本轮无性能结论。

- 独立 GMP 自测覆盖群阶历史界限和乘 48 后的点阶界限。Stage1 同一 CPU 命中对照了旧、本轮二进制的 GMP、PARI、GP 三种后端，七条 `go*` 报告逐条相同；CUDA Stage1 的 PARI 命中也通过同样的对照。共享诊断 44 项通过；Stage2 因子分解前端两项通过。旧 GO 整套回退脚本的 1 秒 GP 限制在本机曾超时，保留该失败日志；未放宽该脚本。
- CUDA Stage1/Stage2 本机构建复用经验证的设备对象，重编主机并链接；关闭工具目标后的 OpenCL Stage1 构建也通过。Stage2 主机及进程合同通过。Android NDK 28.2 ARM64 构建通过；未在 Android 设备执行。
- CUDA Stage2 双槽队列和直跑、AMD OpenCL 队列均以 M521、四条 Edwards 曲线验证：任务开始后破坏 TOML，仍按原用户和目标路径发布；下次启动会拒绝损坏配置。另验证 CPU Edwards Stage2 四曲线发布。RTX 4060 Laptop 的 OpenCL 初始化失败在同设备同输入的旧、新二进制均可复现；AMD 设备的对照因子为 257。
- CUDA Stage2 checkpoint 发布受阻后重启，NF 汇总和因子记录仍使用保存时的用户身份；回执重放保持逐字节一致。CPU Edwards Stage1 从冻结 checkpoint 恢复，八个完成点相同。正式 P95 同步生成 save 和 worktodo.add，共享写入合同通过。
- GUI v2 和诊断源码包分别解压重建，GUI 能显示版本，诊断探针能完成 GMP 群阶和点阶分析。完整双 CUDA 发布包和 GUI 交互回归未在本轮重跑。
- Stage2 配置和发布采用当前运行回归验收；依赖旧源码路径和旧函数清单的冻结提取检查继续作为历史记录。
