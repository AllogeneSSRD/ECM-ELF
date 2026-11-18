# 统一 Stage2 入口

0.68.0 的 CUDA 构建将 CPU Edwards worker、libecm 和 CUDA 后端链接到 `ecm_stage2.exe`。CPU 专用构建和 OpenCL 进程入口仍可独立使用。GUI 识别统一版标记，从当前 worker 的配置显示执行设备。

统一入口从命令行读取配置位置和 worker 编号，加载 TOML 快照，再把该快照和其余参数交给选中的后端。两种工作模式都沿用各自的参数解析。GPU 的内部曲线进程沿用父进程已确定的路径。配置及用法见 [Stage2 运行设备](../usage/STAGE2.md#选择运行设备)。

## 验证

- Windows 本机构建核对并复用了 CUDA 对象。统一版、OpenCL 入口和 GUI v2 成功构建。在设备 1 上经 `--cost-device-info` 验证两种 GPU 入口可双向移交。
- 在 RTX 4060 Laptop（CUDA 设备 1）上，统一版根据 Worker 3 的混合大小写配置分别走 CPU libecm plain 与 CUDA。两者对同一 Edwards save 均找到因子 10007 并完成各自队列。测试也覆盖缺省 CPU、`auto`、无效值、CLI 覆盖和 GUI 队列协议。
- 同一程序对 CPU GWNUM → CUDA 与 CUDA → CPU GWNUM 各验证了一次整曲线续跑。任务从第 2 条曲线继续，每条 sigma 只记录一次，完成文件和提交文件各有一份任务结果。
- 发布包结构测试覆盖 GMP-ECM 许可和对应源码。另使用本机的发布 GMP 前缀生成过源码 ZIP。

复现入口为 `tools/test/test_combined_stage2.py`、`tools/test/test_cpu_stage2_queue.py` 和 `tools/test/test_release_package_layout.py`。原始记录位于 `data/experiments/stage2_combined_0680_cross/`。

## 按任务自动选设备

当前 `auto` 使用 CPU。若扩展为按输入选择，队列模式应在取得同一 worker 的所有权、解析实际任务并读取所选 save 行之后选择执行设备。这样两种 worktodo 格式可共用逻辑，混合队列也能逐任务选择。选择依据包含曲线模型、实际 N、有效 B2 和后端资格。显式设备继续固定。这个扩展需把两边共用的队列选项与后端专属选项进一步分离。
