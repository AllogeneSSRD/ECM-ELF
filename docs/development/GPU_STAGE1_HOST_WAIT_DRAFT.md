# GPU Stage1 主机等待诊断草稿

本轮只调查 Edwards 与 Montgomery 的 CPU 占用差异。生产算术、等待策略、配置和二进制均未修改；诊断工具使用实际生产模板，只替换主机等待控制。当前行为的入口说明见 [Stage1 架构](../architecture/STAGE1.md#gpu-edwards)。本草稿完成后保留，由用户决定如何处理，不自动删除。

## 已确认原因

默认 GPU Edwards NAF 的高 CPU 占用主要来自 `cudaEventSynchronize` 中的主机等待，不是 NAF 点运算在 CPU 执行。两条驱动的入口不同：

- `src/core/ecm_driver.cpp` 的 GPU Edwards 分支直接进入 `run_edwards_gpu_stage1`，未执行 Montgomery 使用的 `ecm_backend_prepare`。
- `src/cuda/ecm_cuda_backend.cu:178` 为 Montgomery 设置阻塞设备调度；`kernels/cuda/cgbn_stage1.cu:1595` 在读 CGBN 报告、记录结束事件前调用设备同步。
- `src/cuda/ecm_cuda_edwards_naf.cu` 只选择设备；`src/cuda/ecm_cuda_edwards_naf.inl:4` 默认创建事件，字典及每片完成时在第 30/41 行直接等待结束事件。
- Edwards 的每片规范坐标回读、GMP 导入和主机回调确实有 CPU 工作，但不是本次接近一核的主要来源。二进制 Edwards 另使用设备同步且共享缺少阻塞调度的入口，本轮没有将 NAF 的量化结果外推为 binary 的测量结果。

[NVIDIA Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html) 说明默认事件与阻塞事件的等待语义；本地 CUDA 13.3 头文件含同一接口约定。公开页面为 13.4 文档。本机实测中，成功以阻塞标志创建事件仍未降低 CPU 占用，因此不能把“加一个事件标志”作为已经验证的修复，也不能仅凭本试验证明具体驱动内部缺陷。

## 原生程序复现

证据根目录：`data/experiments/stage1_host_wait_20261018/`。`baseline02/report.json` 为有效原生对照；baseline01 的 Edwards 命令错误地带了 GPU PARAM 参数，未完成，保留但不用于结论。

使用冻结的 ECM-ELF 0.38.0 `ecm_cuda.exe`，SHA256 前缀 `07a26b7b5d84`，完整身份在报告中。当前源码的相关 CUDA API、模板和 Montgomery 初始化与该冻结版本相同。输入为 M1279、B1=B2=200000、64 条曲线、sigma=17…80、GPU1；各在独立目录运行一次，没有网络提交。该小梅森素数仅作本地诊断，不能上传为 ECM 覆盖结果。

进程 CPU 时间由 Windows `GetProcessTimes` 的用户态与内核态之和得到，分母为进程创建至退出的完整时间；包含上下文、指数/曲线准备和文件发布，不与下节 API 局部时间混用。占用单位为一个逻辑核心的比例，不是整机 CPU 百分比。

| 曲线 | 进程墙钟秒 | 进程 CPU 秒 | 平均逻辑核心 |
| --- | ---: | ---: | ---: |
| Montgomery PARAM0 | 16.2766 | 0.7344 | 0.0451 |
| Edwards 默认 NAF | 11.4154 | 11.1094 | 0.9732 |

这组数据只复现 CPU 占用，不能作为两种曲线算法的性能排名。

## 单变量诊断

`tools/test/stage1_host_wait_probe.cu` 实例化生产 `run_1280`，复用原字典、NAF 切片、256 列分片和读回，直接进入同一生产模板。事件创建与 API 计时通过仅在诊断翻译单元中生效的包装器接入；不修改生产头文件。报告记录的是 CUDA API 成功接受的创建参数，不声称查询了驱动内部事件状态。

第一轮 API 分解位于 `api_probe01/report.json`：默认情况下约 0.947 秒调用中，事件同步累计约 0.920 秒墙钟、0.922 秒进程 CPU；回读约 0.008 秒墙钟。Windows 进程 CPU 计数有调度粒度，逐 API 的 CPU 数字仅作归属诊断，不将它们当作独立精确线程采样，也不与外层 CPU/wall 重复相加。

正式控制组位于 `controlled01/report.json`、`summary.json`，构建清单为 `build_cuda_cmake/stage1_host_wait_probe/manifest.json`。探针 SHA256 前缀 `c57f3db2ae05`；CUDA 13.3.73、MSVC 14.51.36231、sm89、Zen3 GMP。环境为 Windows 11 10.0.26200 / WDDM、Ryzen AI 9 HX 370、RTX4060 Laptop GPU、驱动 610.88。没有更改 CPU 或 GPU 时钟、优先级和功耗设置；有效 CPU 频率未独立测量。GPU 功率上限接口为 N/A，运行采样功耗范围为 8.27…11.10 W，SM 采样为 1500 MHz，表示低占用诊断工况的观测值，不是功率预算或进程峰值。GPU0 有其他工作，未操作它。

控制输入固定为 M1279、标量 2^131072−1、64 条相同 sigma 曲线、NAF w=6、1280-bit/TPI8/TPB128。此标量用于隔离设备等待，不冒充某个 B1 的 ECM 覆盖。计时仅含生产模板入口至返回，排除上下文、曲线构造及最终坐标文件导出，包含设备分配、字典、循环、回读和主机 GMP 导入。CUDA event 区间仍可能包含提交间隙，不作为 profiler 的纯核时间。

| 设备调度 | 事件 | 显式设备等待 | 样本数 | CPU 核心中位数 | 调用墙钟中位数秒 |
| --- | --- | --- | ---: | ---: | ---: |
| Auto | 默认 | 无 | 3 | 0.9996 | 3.7551 |
| BlockingSync | 默认 | 无 | 1 | 1.0001 | 3.7966 |
| Auto | BlockingSync | 无 | 1 | 1.0004 | 3.7797 |
| BlockingSync | BlockingSync | 无 | 1 | 0.9971 | 3.7922 |
| Auto | 默认 | 记录结束事件后等待 | 1 | 1.0011 | 3.7615 |
| BlockingSync | 默认 | 记录结束事件后等待 | 3 | 0.0441 | 3.8951 |

主要两组按 A/B、B/A、A/B 顺序重复，全部十个运行的完整四坐标文件 SHA256 相同，曲线不变量均通过。每项统计分别取中位数。基线墙钟范围为 3.7517…3.7673 秒，候选为 3.8736…3.9015 秒；候选中位数约增加 3.73%，不能据此声称 GPU 算术加速。其累计 CPU 时间中位数由 3.7656 降至 0.1719 秒。此候选在记录结束事件之后等待，不将主机唤醒间隙加入新的结束事件位置。

`short_oracle01/report.json` 使用 M1279、sigma=17、标量 2^32−1，原等待和候选等待的 GPU 末点均通过独立 Python Montgomery ladder 投影比对，且全部坐标一致。该短标量校验与长标量十组逐字相同的检查属于不同证据，不能互相扩大验证范围。

## 复现入口与后续边界

先构建独立探针，再运行控制组；所有输出必须使用新目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/test/build_stage1_host_wait_probe.ps1
python tools/bench/diagnose_stage1_host_wait.py --probe build_cuda_cmake/stage1_host_wait_probe/probe.exe --output data/experiments/stage1_host_wait_repeat
```

脚本仅查询自身样本的 CPU 与 GPU1 遥测，不修改生产队列或 GPU 设置。算术完全一致检查不依赖耗时阈值；未来驱动的阻塞事件行为可能变化，应重做控制组，不能固定假设所有 Windows/CUDA 版本都忽略事件阻塞。

生产修复尚未实施。已验证的方向是为 Edwards 明确设置阻塞设备调度，并在结束事件记录后通过设备同步等待，再读事件与 CGBN 报告。若改用流同步、改变 chunk 或减少回读，需要另外测量与检查点/停止延迟验证。后续 main 程序修复应按仓库规则更新修复版本号；本轮仅新增诊断工具与说明，不更新主程序版本、不部署二进制。
