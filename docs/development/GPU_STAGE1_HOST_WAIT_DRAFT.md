# GPU Stage1 主机等待诊断草稿

本草稿保留主机等待诊断及后续生产等待、自适应切片优化的验证过程。第一阶段未修改生产实现；当前 0.38.4 候选已修改等待与切片调度，点公式和标量不变。诊断工具使用实际生产模板。当前行为的入口说明见 [Stage1 架构](../architecture/STAGE1.md#gpu-edwards)。本草稿完成后保留，由用户决定如何处理，不自动删除。

## 已确认原因

0.38.0 GPU Edwards NAF 的高进程 CPU 占用主要集中在 `cudaEventSynchronize` 等待区间，不是 NAF 点运算在 CPU 执行。原始两条驱动的入口不同；以下为该冻结版本的代码依据，当前实现见后文：

- `src/core/ecm_driver.cpp` 的 GPU Edwards 分支直接进入 `run_edwards_gpu_stage1`，未执行 Montgomery 使用的 `ecm_backend_prepare`。
- `src/cuda/ecm_cuda_backend.cu:178` 为 Montgomery 设置阻塞设备调度；`kernels/cuda/cgbn_stage1.cu:1595` 在读 CGBN 报告、记录结束事件前调用设备同步。
- `src/cuda/ecm_cuda_edwards_naf.cu` 只选择设备；`src/cuda/ecm_cuda_edwards_naf.inl:4` 默认创建事件，字典及每片完成时在第 30/41 行直接等待结束事件。
- Edwards 的每片规范坐标回读、GMP 导入和主机回调确实有 CPU 工作，但不是本次接近一核的主要来源。二进制 Edwards 另使用设备同步且共享缺少阻塞调度的入口，本轮没有将 NAF 的量化结果外推为 binary 的测量结果。

[NVIDIA Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html) 说明默认事件与阻塞事件的等待语义；本地 CUDA 13.3 头文件含同一接口约定。公开页面为 13.4 文档。本机实测中，成功以阻塞标志创建事件仍未降低 CPU 占用，因此不能把“加一个事件标志”作为已经验证的修复，也不能仅凭本试验证明具体驱动内部缺陷。

## 原生程序复现

证据根目录：`data/experiments/stage1_host_wait_20261018/`。`baseline02/report.json` 为有效原生对照；baseline01 的 Edwards 命令错误地带了 GPU PARAM 参数，未完成，保留但不用于结论。

使用冻结的 ECM-ELF 0.38.0 `ecm_cuda.exe`，SHA256 前缀 `07a26b7b5d84`，完整身份在报告中。诊断第一阶段的相关 CUDA API、模板和 Montgomery 初始化与该冻结版本相同。输入为 M1279、B1=B2=200000、64 条曲线、sigma=17…80、GPU1；各在独立目录运行一次，没有网络提交。该小梅森素数仅作本地诊断，不能上传为 ECM 覆盖结果。

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

第一阶段仅提交诊断工具与说明，未更改生产程序。后续 0.38.4 等待及切片优化的证据见下节。单独事件标志的试验只说明本机进程 CPU 占用未降低，没有逐线程采样证明驱动内部线程状态。


## 0.38.4 等待及切片优化

长时间运行是否损失吞吐，取决于同步边界的密度。设一次切片的 GPU 工作为 g，原主机开销为 h，阻塞唤醒比忙等增加 δ，稳定工况的吞吐比约为 `(g+h)/(g+h+δ)`。增加 B1 会按比例增加 GPU 工作和同步次数；若片长保持不变，占比不会因运行了几小时自动消失。把切片调整到约 100 ms，才会摊薄每次唤醒、回读和提交的固定成本。该模型不假设 δ 在所有驱动、系统负载下恒定。

CUDA 官方区分事件创建标志与设备调度标志；`cudaEventBlockingSync` 控制事件等待，`cudaDeviceScheduleBlockingSync` 控制设备等待。[设备管理接口](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html)也说明主动轮询可减少等待延迟，但可能影响其他 CPU 线程。本机采用实测有效的阻塞设备同步路径，不能把两个标志混为同一开关。

### 当前代码与响应边界

Edwards 两个入口共用 `ecm_edwards_cuda_detail::prepare_device`，设置阻塞调度且保留非调度标志。NAF 在原结束事件记录后执行设备同步，然后读取事件、报告及普通坐标；等待不移动结束事件位置。binary 原有设备同步在相同调度设置下执行。当前模块只有默认流；若未来并发独立流，须重新评估设备级等待范围。

生产传入 chunk=0，`src/cuda/ecm_cuda_edwards_slice.h` 从 32 列/位开始，依据上一片时间选择约 100 ms 的下一片，范围 16…4096，每次最多增减 4 倍。正数 chunk 保留固定边界。NAF 使用原 CUDA event 计时，binary 使用提交至设备同步返回的墙钟；CPU 构造、错误检查、回传和检查点不计作纯 GPU 时间。最后不足一片的尾部只执行剩余列。目标 100 ms 不构成停止响应的硬上限。

回调、停止和检查点继续发生在已完成切片及普通坐标回读之后。切片长度不是算法身份，也不写入检查点；恢复仍以完整标量、窗口、倍点方法、列进度及普通域累加点为依据，重新从小片采样。窗口和倍点冲突检查保持原门限。

### 验证口径

构建使用正式 local 入口、sm89、CUDA 13.3.73、MSVC 14.51.36231、Zen3 GMP，测试 worker 编入 512/768/1280/10240 四档。`build_cuda_cmake/edwards_host_wait_0_38_4/stage1/edwards_build_manifest.json` 记录源与对象身份；本轮未部署至生产目录。CPU/GPU 环境沿用上节，未更改功耗、时钟或优先级。

`long_m1279_c64_b1e6/` 与 `long_m9689_c768_b4e4/` 为探索校准，部分与编译重叠，不进入正式吞吐结论。后者仅首个约 84 秒样本完成，下一探针按精确可执行路径结束；原始不完整日志保留。正式采集在全部编译结束后进行，GPU0 原有工作保持运行，GPU1 没有并发其他本轮测试。

正式三路对照使用同一探针二进制比较：原固定片忙等、固定片阻塞、自适应片阻塞。每路 3 次，按 A/B/C、C/B/A、A/B/C 顺序交错，不丢弃首轮。每个输入内部的完整 XYZT 输出须逐字相同，曲线不变量全部通过。探针仅用测试包装器抑制或接入设备等待，不改点公式；API 计时包装器有自身成本，因此还要用未插桩生产 exe 完整进程对照校核吞吐。

四位宽 CPU/GPU 门禁为 M347/M677/M1279/M9689，每个 binary/NAF 组合 3 条 sigma，共 8 组。完整运行、固定 17/113 切片恢复、自适应中断/恢复的全部状态相同，投影末点与独立 CPU 实现一致；证据 `gates_0_38_4/point_probe.log`。CPU 调度控制的变速收敛、正向推进及短尾测试与 46574 组 NAF 编码测试通过。

真实进程恢复测试位于 `restart_0_38_4_02/summary.json`，18 项检查通过。旧 binary V1 与 NAF V2 均恢复成功；固定/随机 sigma 保持原序，显式冲突仍拒绝。旧 binary 停止信号到退出约 47.7 ms，新 NAF 两例约 93.6/181.5 ms，包含检查点写入及进程退出；样本很少且输入较小，不声明生产最大响应时间。第一次恢复测试误将现代基线当作默认 binary，已改为显式选择 binary，原失败证据 `restart_0_38_4/` 保留，未放宽状态校验。


### 正式吞吐结论与后续

原固定切片忙等、固定切片阻塞、自适应阻塞三路各3次，另以未插桩旧/新生产exe各3次核对完整批次，均已完成。正式数据、源/二进制身份、计时边界、原始证据、范围和复现入口集中维护于[Stage1性能说明](../performance/STAGE1.md#gpu-edwards-主机等待与自适应切片)，本草稿不另建当前性能表。功耗遥测出现590.01 W异常，未删改原始记录，本轮不作功耗结论。

`factor_mont_0_38_4/summary.json` 还验证了 Edwards 非平凡因子命中与 Montgomery PARAM0 回归：旧/新各4条记录逐项一致，因子整除和独立 Montgomery CPU末点检查通过。

当前版本为0.38.4；源代码等待与切片优化可进入生产构建，本轮已完成四档local规则构建与测试，未替换生产目录。数小时/生产B1=260e6、全发布架构和不同驱动下的速度/能耗仍未测量；若将来新增独立流，须重新选择同步范围。草稿已完成但按规则保留，不自动清理。
