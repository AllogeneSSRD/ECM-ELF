# Stage1 使用

## 程序与输入

`ecm_cuda.exe` 使用 NVIDIA CUDA/CGBN 并行处理曲线。`ecm.exe` 提供 CPU 和 OpenCL 路径；CPU 支持 Montgomery/Suyama 与 Edwards，OpenCL 的设备和算子能力按实际构建确定。选项以程序帮助及 [INI 参考](../ECM_INI_REFERENCE.md) 为准。

生产任务通常使用 Suyama PARAM0。sigma 是曲线种子，不是曲线序号；不同参数化的 sigma 不能互换。GPU 一批曲线共享 N、B1 和指数，分别维护点状态。`ECM=`/`ECM2=` 队列行没有 save 名称；GPU Montgomery worker 会按默认的 `m{n}_{b1}.save` 名称写入 `tmp_dir`，CPU Montgomery 也在该目录写入共享 save。保存失败时任务留在 worktodo，不会进入 finished。

Montgomery 路径的 `lcm=1` 使用 lcm(1…B1)，`lcm=12` 在该标量上乘 12。后者会改变保存点和因子发现范围，比较程序或生成测试存档时必须明确口径。Edwards 路径固定使用 48·lcm(1…B1)，不接受 `lcm=12`。`ecm_cuda.exe` 可用 `--curve ed --target gpu` 或同名 INI 键运行 GPU Edwards；OpenCL 不支持该组合。CUDA PARAM0 和 Edwards 可使用完整 64 位 sigma；PARAM2/3 的批次受 32 位窗口限制。

GPU Edwards 使用不带参数化前缀的 sigma；CLI 不接受 `--gpu-param` 和 `--go`。默认使用窗口 NAF，`--naf-w 0` 在 GPU 选择窗口 6，显式允许 2..12；每片最多 256 个 NAF 列，在 CPU 上构造曲线、检查结果及求 GCD。窗口越大，GPU 字典显存和准备成本越高。旧 GPU 二进制检查点会按原算法恢复；显式选择与检查点不符时会报错并保留进度。默认保存名在 `m{n}_{b1}.save` 前添加 `ed_`，显式 `-save` 可指定路径。存档格式、对照选项和适用限制统一见 [GPU Edwards 实现](../architecture/STAGE1.md#gpu-edwards)。GPU Edwards 文本存档交给独立 CUDA Stage2；CPU Edwards 的 Prime95 二进制 `.tmp` 格式保持独立。

## 共用 INI 与 worker

两个 CUDA 程序可共用 exe 目录中的 `ecm.ini`。Stage1 使用 `worktodo`、`finished`、`log_file`；Stage2 使用相应 `stage2_*` 键，分别消费队列。

`--worker N` 同时选择 INI 与 worktodo 中的 `[Worker #N]`。worker 设置覆盖全局设置，再使用默认值。每个队列只能有一个消费进程；不同 worker 使用独立输出路径。键的默认值和适用范围只在配置参考维护。

队列无待运行任务时正常结束；`worktodo` 不存在或读取失败时报错退出，不能把它当成空队列。

## 计算与输出

主机生成曲线和指数，GPU 分片推进点乘，主机验证末点、求 GCD 并写入结果。`gpucurves` 是批量，不表示设备同时驻留的曲线数。批量不足可能降低吞吐；适合的批量由位宽、TPI、TPB、SM 数和实际资源占用共同决定。

普通进度中的 `s/curve` 是当前速度的每曲线摊销；部分运行的投影不是完成一条生产曲线的实测时间。文件日志的进度频率由 `progress_log_seconds` 控制，GUI 所用管道进度独立保留。

Auto B2的可复用Stage1成本可用[tune_stage1_cost.py](../../tools/bench/tune_stage1_cost.py)测量完整批次并独立复核保存点。当前限CUDA ladder PARAM0、自动TPI、关闭指数缓存；可使用已知梅森素数目录，或通过`--target-n <N...>`测量指定余因子。所有测量曲线必须完成且通过无因子/独立末点校验，不接受部分运行速度投影。等级、参数、计时边界及使用方式统一见[完整Stage1成本预计算](../architecture/AUTO_B2.md#完整stage1成本预计算)。

最终 save 保存可继续 Stage2 的点；checkpoint 保存尚未完成 Stage1 的状态，二者不能互用。save 的归一化 X 必须属于记录中的 N、参数化、sigma、B1 和指数模式；checksum 用于检查文件完整性，不重新证明点乘正确。

## 检查点与交接

Stage1 支持按切片保存中间进度。恢复时检查任务身份、格式和实际计算布局，不能通过改名绕过检查。未完成采样只输出 checkpoint，不应当作有效最终 save。

Ctrl+C、Ctrl+Break 或 GUI 的正常停止请求会在当前后端的下一个安全边界触发检查点写入，然后结束当前任务。CUDA/OpenCL 在 GPU 批次边界保存，CPU 在曲线计算的分块边界保存；这不是在信号处理函数中直接写文件。队列模式会随即退出，当前未完成任务及后续行留在 `worktodo`，不会被标为 `# ERROR` 或写入 `finished`；重启时从检查点恢复。`ckpt_seconds=0` 只关闭定期写入，不关闭中断存档。GUI 超时后仍会强制终止 worker，因此应给大批次和慢速存储足够的 `graceful_stop_ms`。

检查点先写入同目录的 `.new` 临时文件，完整关闭后再替换主文件。`ckpt_backups` 控制轮换备份的份数，主文件另计；例如 `.ecm_ckpt_….dat` 的首份备份为 `.ecm_ckpt_….bak.dat`，后续为 `.bak2.dat` 等，CPU `.ckpt` 文件同理。恢复按主文件、备份序号的顺序检查，跳过读不到、截断、校验失败或参数不符的文件；历史二进制检查点没有新附加的完整性校验，仍按旧格式读取。Stage1 成功完成后清理该任务的主检查点、临时文件和全部备份。存档写入或轮换失败会报错，不能把失败消息视作已保存进度。

配置受支持的 Prime95 交接后，驱动将完成任务的 save 同步到目标位置，并向同目录 `worktodo.add` 追加原任务行；失败交付进入 pending 文件，后续重试。不要直接由两个程序共同改写 Prime95 的活动 `worktodo.txt`。CPU 本地存档的独立交接工具为 `ecm_p95feeder`；GPU Edwards 的独立文本格式交给 CUDA Stage2，不支持该 Prime95 自动交接。

## 算法与构建范围

CUDA 默认路径为 ladder。resident/PRAC 必须在构建时启用，再显式选择；发布入口默认不编译这些实验模块。TPI 和寄存器实验的适用条件见 [Stage1 实现](../architecture/STAGE1.md) 与 [性能](../performance/STAGE1.md)。

本程序与 Stage2 仍为两个独立可执行文件。自动串接两阶段、统一消费策略属于 [TODO](../TODO.md)，不能由共用 INI 推断为已经实现。

## 代码入口

- [ecm_driver.cpp](../../src/core/ecm_driver.cpp)：命令行、INI、队列、进度和交接。
- [cgbn_stage1.cu](../../kernels/cuda/cgbn_stage1.cu)：CUDA 主机调度、曲线布局和结果处理。
- [ecm_stage1_exp.cpp](../../src/core/ecm_stage1_exp.cpp)：指数生成与缓存。
- [CPU 实现目录](../../src/cpu/)：Montgomery、Edwards 和保存点。
