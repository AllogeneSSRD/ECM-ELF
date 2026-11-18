# CPU worker 资源配置验证

新增 `[stage1] cpu.threads` 与 `gpu.init_threads`，分别控制 CPU 算术和受支持 GPU 后端的 CPU 曲线构造。`[stage2] cpu.threads` 继续独立配置。新增 `[shared] cpu_affinity`，并允许 `[worker.N] shared.cpu_affinity` 为两阶段进程统一覆盖。最终行为和迁移规则见 [配置架构](../architecture/CONFIGURATION.md#worker-cpu-资源)。

验证重点为新旧键同层优先级、worker 覆盖、GPU 构造线程裁剪、跨阶段绑核和非法 CPU 拒绝。Windows 多处理器组编号使用组号与组内索引，Windows CPU Sets 受外部 affinity mask 约束。

Windows 本地开发构建通过 `tools/test/test_worker_cpu_resources.py` 验证。CPU Stage1 以 1 个线程执行，CUDA NTT、CUDA 一线程一曲线和 OpenCL 的 GPU 主机构造均按两核集合将请求的 3 个线程裁剪为 2 个。CPU GWNUM、CUDA 和 OpenCL Stage2 均以同一 worker 覆盖启动。测试还覆盖旧键回退、显式 CLI 优先、`none` 取消共享覆盖、倒序区间及不存在的 CPU。通过 Win32 单独查询运行中的 CPU Stage1 进程，得到掩码 `0x3`，对应逻辑 CPU 0 和 1。证据位于 `data/experiments/cpu_worker_config_0670_all/`。

配置迁移 9 项、GUI App 回归、CPU 与 CUDA Stage2 双向队列恢复均通过。构建验证覆盖 CUDA Stage1、CPU/CUDA Stage2、OpenCL Stage1/Stage2 和 GUI。机器是单处理器组，Windows 多组和 Linux 未实机验证。开发草稿保留，当前配置行为以正式文档和生成参考为准。
