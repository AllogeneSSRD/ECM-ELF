# Stage1 使用

## 程序与输入

`ecm_cuda.exe` 使用 NVIDIA CUDA/CGBN 并行处理曲线。`ecm.exe` 提供 CPU 和 OpenCL 路径；CPU 支持 Montgomery/Suyama 与 Edwards，OpenCL 的设备和算子能力按实际构建确定。选项以程序帮助及 [INI 参考](../ECM_INI_REFERENCE.md) 为准。

生产任务通常使用 Suyama PARAM0。sigma 是曲线种子，不是曲线序号；不同参数化的 sigma 不能互换。GPU 一批曲线共享 N、B1 和指数，分别维护点状态。`ECM=`/`ECM2=` 队列行没有 save 名称；GPU Montgomery worker 会按默认的 `m{n}_{b1}.save` 名称写入 `tmp_dir`，CPU Montgomery 也在该目录写入共享 save。保存失败时任务留在 worktodo，不会进入 finished。

Montgomery 路径的 `lcm=1` 使用 lcm(1…B1)，`lcm=12` 在该标量上乘 12。后者会改变保存点和因子发现范围，比较程序或生成测试存档时必须明确口径。Edwards 路径固定使用 48·lcm(1…B1)，不接受 `lcm=12`。`ecm_cuda.exe` 可用 `--curve ed --target gpu` 或同名 INI 键运行 GPU Edwards；OpenCL 不支持该组合。CUDA PARAM0 和 Edwards 可使用完整 64 位 sigma；PARAM2/3 的批次受 32 位窗口限制。

GPU Edwards 使用不带参数化前缀的 sigma；CLI 不接受 `--gpu-param` 和 `--go`。已知素因子的Edwards群阶、点阶及分解可用[独立PARI入口](DATASET.md#edwards群阶与点阶)计算。默认使用窗口 NAF，`--naf-w 0` 在 GPU 选择窗口 6，显式允许 2..12；每片最多 256 个 NAF 列，在 CPU 上构造曲线、检查结果及求 GCD。窗口越大，GPU 字典显存和准备成本越高。批次内的 GPU 二进制状态会按保存的算法恢复；显式选择与检查点不符时会报错并保留进度。默认保存名在 `m{n}_{b1}.save` 前添加 `ed_`，显式 `-save` 可指定路径。GPU 算术和检查点限制见 [GPU Edwards 实现](../architecture/STAGE1.md#gpu-edwards)。CPU/GPU Edwards 完成后统一生成多曲线文本 save，供 CUDA Stage2 或 [CPU Stage2](EDWARDS_CPU_STAGE2.md) 读取。

## 共用 INI 与 worker

两个 CUDA 程序可共用 exe 目录中的 `ecm.ini`。Stage1 使用 `worktodo`、`finished`、`log_file`；Stage2 使用相应 `stage2_*` 键，分别消费队列。

`--worker N` 同时选择 INI 与 worktodo 中的 `[Worker #N]`。worker 设置覆盖全局设置，再使用默认值。每个队列只能有一个消费进程；不同 worker 使用独立输出路径。键的默认值和适用范围只在配置参考维护。

队列无待运行任务时正常结束；`worktodo` 不存在或读取失败时报错退出，不能把它当成空队列。

## 计算与输出

主机生成曲线和指数，GPU 分片推进点乘，主机验证末点、求 GCD 并写入结果。`gpucurves` 是批量，不表示设备同时驻留的曲线数。批量不足可能降低吞吐；适合的批量由位宽、TPI、TPB、SM 数和实际资源占用共同决定。

CPU Edwards 单曲线、SIMD 和 GPU 路径均在分块边界刷新进度；重定向到文件或 GUI 管道时输出普通文本百分比，不依赖终端控件。GPU NAF 使用已有的列计数，不为进度重新编码标量。

普通进度中的 `s/curve` 是当前速度的每曲线摊销；部分运行的投影不是完成一条生产曲线的实测时间。文件日志的进度频率由 `progress_log_seconds` 控制，GUI 所用管道进度独立保留。

Auto B2的可复用Stage1成本可用[tune_stage1_cost.py](../../tools/bench/tune_stage1_cost.py)测量完整批次并独立复核保存点。当前限CUDA ladder PARAM0、自动TPI、关闭指数缓存；可使用已知梅森素数目录，或通过`--target-n <N...>`测量指定余因子。所有测量曲线必须完成且通过无因子/独立末点校验，不接受部分运行速度投影。等级、参数、计时边界及使用方式统一见[完整Stage1成本预计算](../architecture/AUTO_B2.md#完整stage1成本预计算)。

最终 save 保存可继续 Stage2 的点；checkpoint 保存尚未完成 Stage1 的状态，二者不能互用。save 的归一化 X 必须属于记录中的 N、参数化、sigma、B1 和指数模式；checksum 用于检查文件完整性，不重新证明点乘正确。

内部 JSONL 记录全部曲线的 F/NF；提交文件仅新增 Stage1 找到的因子。Stage1 NF 保留在内部结果和 save 中，供后续 Stage2 使用；全新提交路径的整批 NF 生成空文件及私有归属账本。

## 检查点与交接

Stage1 支持按切片保存中间进度。恢复时检查任务身份、格式和实际计算布局，不能通过改名绕过检查。未完成采样只输出 checkpoint，不应当作有效最终 save。

Ctrl+C、Ctrl+Break 或 GUI 的正常停止请求会在当前后端的下一个安全边界触发检查点写入，然后结束当前任务。CUDA/OpenCL 在 GPU 批次边界保存，CPU 在曲线计算的分块边界保存；这不是在信号处理函数中直接写文件。队列模式会随即退出，当前未完成任务及后续行留在 `worktodo`，不会被标为 `# ERROR` 或写入 `finished`；重启时从检查点恢复。`ckpt_seconds=0` 只关闭定期写入，不关闭中断存档。GUI 超时后仍会强制终止 worker，因此应给大批次和慢速存储足够的 `graceful_stop_ms`。

每个任务只使用一个批次检查点，CPU 各曲线的进度和完成结果都封装在该文件中。主文件名严格为：

- Edwards：`.e_s1_m{n}_{B1}.ckpt` 或 `.e_s1_bit{bits}_{B1}.ckpt`。
- Montgomery：`.m{param}_s1_m{n}_{B1}.ckpt` 或 `.m{param}_s1_bit{bits}_{B1}.ckpt`，param 为 0、2、3；CPU 使用 0。

有 k·b^n+c 元数据的任务使用原始 n，即使剥离了已知因子；直接输入完整梅森整数也能识别 n。其他整数使用实际位宽 bits。B1 使用完整十进制整数。例如 `.e_s1_m127_1000.ckpt`、`.m3_s1_bit512_1000.ckpt`。不追加 worker 或曲线序号，同名任务需要独立的 `tmp_dir`。

`ECM_STAGE1_BATCH_V1` 容器绑定完整模数、曲线族、参数化、B1、指数约定、曲线数及精确 sigma；CPU 另绑定实现和 NAF 窗口，GPU 载荷检查自己的算法与状态布局。原子写入使用 `.new` 临时文件；`ckpt_backups` 轮换备份，例如 `.e_s1_m127_1000.bak.ckpt`、`.bak2.ckpt`。恢复检查完整校验，主档损坏时尝试备份；有效主档属于不同任务时直接拒绝，不覆盖它。没有任何有效备份时保留损坏文件并报错。

同一批次有进程独占锁，防止两个 worker 同时写同名文件。CPU Edwards 已完成曲线保留结果，CPU Montgomery 保留完成结果及尚未完成的 ladder 状态；SIMD Edwards 在各 lane 的恢复位号一致时继续该批，否则重算该 SIMD 批。GPU 存放整个设备批次的状态。只支持本容器中的对应后端载荷，不自动读取 Prime95 的检查点。Stage2 的曲线计数文件并非曲线内算术检查点；本节命名和续算保证适用于 Stage1。

最终多曲线 save、内部 JSONL 和提交 JSON 均成功发布后才删除检查点。写入失败时保留批次状态和原队列任务；恢复发布不会重复增加数学工作回执或重复追加相同 save 记录。内部与提交文件的配置、字段和校验见 [ECM 结果输出](ECM_RESULTS.md)。

Edwards 与 GPU Montgomery PARAM2 自动忽略配置的 Prime95 转移路径，仅输出说明，不访问该路径或转移账本。受支持的 Montgomery PARAM0/3 仍可将 save 同步并向 Prime95 `worktodo.add` 追加原任务；失败交付进入 pending 文件，后续重试。不要由两个程序共同改写 Prime95 活动 `worktodo.txt`。Edwards 文本 save 由本仓库 CPU/CUDA Stage2 消费。

## 算法与构建范围

CUDA 默认路径为 ladder。resident/PRAC 必须在构建时启用，再显式选择；发布入口默认不编译这些实验模块。TPI 和寄存器实验的适用条件见 [Stage1 实现](../architecture/STAGE1.md) 与 [性能](../performance/STAGE1.md)。

本程序与 Stage2 仍为两个独立可执行文件。自动串接两阶段、统一消费策略属于 [TODO](../TODO.md)，不能由共用 INI 推断为已经实现。

## 代码入口

- [ecm_driver.cpp](../../src/core/ecm_driver.cpp)：命令行、INI、队列、进度和交接。
- [cgbn_stage1.cu](../../kernels/cuda/cgbn_stage1.cu)：CUDA 主机调度、曲线布局和结果处理。
- [ecm_stage1_exp.cpp](../../src/core/ecm_stage1_exp.cpp)：指数生成与缓存。
- [CPU 实现目录](../../src/cpu/)：Montgomery、Edwards 和保存点。
