# 配置 v4 开发验证

0.69.0 的主配置显式声明 `config_version = 4`。全局键名唯一，worker 覆盖沿用同一个键名。`[worker.N]` 由键所属的全局分区确定覆盖的阶段。生成器检查键名和唯一性，运行程序和 GUI 编辑器使用同一份映射。

GUI v2 的程序路径、布局、显示、窗口和宠物设置从主配置读取。保存界面偏好时重新读取主文件，只写入本次修改的偏好。历史计时和结果复制记录是运行数据。离线迁移可合并 v3 主文件与旧 GUI 偏好，主文件的显式值优先。

## 本机验证

- CUDA Stage1 与统一 Stage2 增量构建复用设备对象。Worker 3 在独立队列完成了 CPU Stage1、保存点交接和 CPU Stage2。
- RTX 4060 Laptop（CUDA 设备 1）与 CPU 使用统一版运行同一 Edwards Stage2 输入，两条路径都找到因子 10007。测试还覆盖设备选择的大小写、全局和 worker 优先级。
- Stage1 输入、交接、NTT 余因子和 NAF 内存回归通过。GPU1 的 NTT 结果与 CGBN 一致，断点续跑生成相同存档，GPU Stage2 可消费余因子保存点。
- GUI 启动和进程路径回归通过。实际统一版在 Worker 3 显示 CPU Stage2，GPU 冲突提示未将该 worker 计入。
- 本机生产 v3 配置及旧 GUI 偏好通过离线迁移预览。迁移后的 worker 3 选择 CPU，Stage2 可执行文件来自主配置。生产原文件保留。

构建、配置格式和离线迁移见 [配置架构](../architecture/CONFIGURATION.md)。
