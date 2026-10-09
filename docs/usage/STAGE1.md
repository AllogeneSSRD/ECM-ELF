# Stage1 使用

## 程序与输入

`ecm_cuda.exe` 使用 NVIDIA CUDA/CGBN 并行处理曲线。`ecm.exe` 提供 CPU 和 OpenCL 路径；CPU 支持 Montgomery/Suyama 与 Edwards，OpenCL 的设备和算子能力按实际构建确定。选项以程序帮助及 [INI 参考](../ECM_INI_REFERENCE.md) 为准。

生产任务通常使用 Suyama PARAM0。sigma 是曲线种子，不是曲线序号；不同参数化的 sigma 不能互换。GPU 一批曲线共享 N、B1 和指数，分别维护点状态。

`exponent=lcm` 使用 lcm(1…B1)；`exponent=choose12` 在该标量上乘 12。后者会改变保存点和因子发现范围，比较程序或生成测试存档时必须明确口径。驱动、队列及命令行选择均由配置消费端处理。

## 共用 INI 与 worker

两个 CUDA 程序可共用 exe 目录中的 `ecm.ini`。Stage1 使用 `worktodo`、`finished`、`log_file`；Stage2 使用相应 `stage2_*` 键，分别消费队列。

`--worker N` 同时选择 INI 与 worktodo 中的 `[Worker #N]`。worker 设置覆盖全局设置，再使用默认值。每个队列只能有一个消费进程；不同 worker 使用独立输出路径。键的默认值和适用范围只在配置参考维护。

## 计算与输出

主机生成曲线和指数，GPU 分片推进点乘，主机验证末点、求 GCD 并写入结果。`gpucurves` 是批量，不表示设备同时驻留的曲线数。批量不足可能降低吞吐；适合的批量由位宽、TPI、TPB、SM 数和实际资源占用共同决定。

普通进度中的 `s/curve` 是当前速度的每曲线摊销；部分运行的投影不是完成一条生产曲线的实测时间。文件日志的进度频率由 `progress_log_seconds` 控制，GUI 所用管道进度独立保留。

最终 save 保存可继续 Stage2 的点；checkpoint 保存尚未完成 Stage1 的状态，二者不能互用。save 的归一化 X 必须属于记录中的 N、参数化、sigma、B1 和指数模式；checksum 用于检查文件完整性，不重新证明点乘正确。

## 检查点与交接

Stage1 支持按切片保存中间进度。恢复时检查任务身份、格式和实际计算布局，不能通过改名绕过检查。未完成采样只输出 checkpoint，不应当作有效最终 save。

配置 Prime95 交接后，驱动将完成任务的 save 同步到目标位置，并向同目录 `worktodo.add` 追加原任务行；失败交付进入 pending 文件，后续重试。不要直接由两个程序共同改写 Prime95 的活动 `worktodo.txt`。CPU 本地存档的独立交接工具为 `ecm_p95feeder`。

## 算法与构建范围

CUDA 默认路径为 ladder。resident/PRAC 必须在构建时启用，再显式选择；发布入口默认不编译这些实验模块。TPI 和寄存器实验的适用条件见 [Stage1 实现](../architecture/STAGE1.md) 与 [性能](../performance/STAGE1.md)。

本程序与 Stage2 仍为两个独立可执行文件。自动串接两阶段、统一消费策略属于 [TODO](../TODO.md)，不能由共用 INI 推断为已经实现。

## 代码入口

- [ecm_driver.cpp](../../src/core/ecm_driver.cpp)：命令行、INI、队列、进度和交接。
- [cgbn_stage1.cu](../../kernels/cuda/cgbn_stage1.cu)：CUDA 主机调度、曲线布局和结果处理。
- [ecm_stage1_exp.cpp](../../src/core/ecm_stage1_exp.cpp)：指数生成与缓存。
- [CPU 实现目录](../../src/cpu/)：Montgomery、Edwards 和保存点。
