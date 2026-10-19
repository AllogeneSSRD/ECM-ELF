# Stage1 使用

## 程序与输入

正常计算/队列启动输出公共程序名和版本；`--version` 只输出版本后退出，不启动计算。`--gpu-info` 等机器查询保持原输出结构。

`ecm_cuda.exe` 使用 NVIDIA CUDA/CGBN 并行处理曲线。`ecm.exe` 提供 CPU 和 OpenCL 路径；CPU 支持 Montgomery/Suyama 与 Edwards，OpenCL 的设备和算子能力按实际构建确定。选项以程序帮助及 [INI 参考](../ECM_INI_REFERENCE.md) 为准。

生产任务通常使用 Suyama PARAM0。sigma 是曲线种子，不是曲线序号；不同参数化的 sigma 不能互换。GPU 一批曲线共享 N、B1 和指数，分别维护点状态。`ECM=`/`ECM2=` 队列行没有 save 名称；CPU/GPU 的自动 save 共用 `save_name_pattern`，默认 `{curve}_{target}_{b1}.save`，写入 `tmp_dir`。curve 为 e、m0、m2、m3；target 保留梅森任务的原始指数，例如 m5351，即使已经剥离因子。普通整数使用 bit 加实际位宽。B1 使用紧凑写法，例如 `m0_m5351_260e6.save`、`e_m5351_260e6.save`。旧自定义模式自动补充曲线前缀；CLI 显式 `-save` 路径保持原样。保存失败时任务留在 worktodo，不会进入 finished。

Stage1 启动器也接受 `ECMSTAGE1=` 队列：从 save 文件名末尾的 `_B1.save` 提取 B1。文件名前缀 `e_`、`m0_`、`m2_`、`m3_` 分别指定 Edwards、Montgomery PARAM0/2/3，覆盖该任务的 INI `curve`、`gpu_param`；`ed_` 兼容识别为 Edwards，并规范为 `e_`。Edwards 自动采用固定的 48·lcm 标量，忽略该任务的 Montgomery `lcm` 设置。执行设备和 CPU/GPU 选择仍由 INI 决定，CPU 遇到 `m2_`/`m3_` 会报错并保留队列。每条任务独立应用前缀，不影响下一条无类型前缀任务的 INI 默认值。无类型前缀的旧名称沿用 INI，并自动补充类型前缀；父目录和剩余文件名保留。

控制台 `START:` 保留原始 worktodo 行；`finished` 和受支持的 Prime95 交接行使用实际 save 名称，其他字段保持原样。补前缀时，若新目标不存在或为空，将旧 save 中同曲线族、同参数化的记录复制到新文件后追加，保持原 `skip_curves` 编号；旧文件保留。记录类型冲突时停止并保留队列。新目标已有内容时不重复复制旧历史。`save_name_pattern` 只生成没有显式 save 名称的任务路径。

Montgomery 路径的 `lcm=1` 使用 lcm(1…B1)，`lcm=12` 在该标量上乘 12。后者会改变保存点和因子发现范围，比较程序或生成测试存档时必须明确口径。Edwards 路径固定使用 48·lcm(1…B1)，不接受 `lcm=12`。`ecm_cuda.exe` 可用 `--curve ed --target gpu` 或同名 INI 键运行 GPU Edwards；OpenCL 不支持该组合。CUDA PARAM0 和 Edwards 可使用完整 64 位 sigma；PARAM2/3 的批次受 32 位窗口限制。

GPU Edwards 使用不带参数化前缀的 sigma；CLI 不接受 `--gpu-param` 和 `--go`。已知素因子的Edwards群阶、点阶及分解可用[独立PARI入口](DATASET.md#edwards群阶与点阶)计算。默认使用窗口 NAF，`--naf-w 0` 在 GPU 选择窗口 6，显式允许 2..12；切片会根据设备执行时间调整到约 100 ms，CPU 使用阻塞等待，并负责构造曲线、检查结果及求 GCD。窗口越大，GPU 字典显存和准备成本越高。批次内的 GPU 二进制状态会按保存的算法恢复；显式选择与检查点不符时会报错并保留进度。自动文件名使用共用模式中的 e 前缀，显式 `-save` 可指定路径。GPU 算术和检查点限制见 [GPU Edwards 实现](../architecture/STAGE1.md#gpu-edwards)。CPU/GPU Edwards 完成后统一生成多曲线文本 save，供 CUDA Stage2 或 [CPU Stage2](EDWARDS_CPU_STAGE2.md) 读取。

## 共用 INI 与 worker

队列的 `tmp_dir` 相对可执行文件所在目录解析；路径生成时清除 `.` 等冗余目录组件，并使用当前平台的分隔符。检查点文件名自身的前导 `.` 属于命名的一部分。

两个 CUDA 程序共用 `ecm.ini` 和 `worktodo`；任务前缀 `ECMSTAGE1=`、`ECMSTAGE2=` 分别指定 Stage1、Stage2。`worktodo` 相对 INI 目录解析。Stage1 使用 `finished`、`log_file`，Stage2 使用 `stage2_finished`、`stage2_log_file` 等独立输出。GUI 根据下一个可运行任务启动对应程序；单独启动 CLI 程序时，该程序只领取自己阶段的任务。

`--worker N` 同时选择 INI 与 worktodo 中的 `[Worker #N]`。worker 设置覆盖全局设置，再使用默认值。同一 worker 分区同时只由一个程序处理；不同 worker 可以共用队列文件，队列改写受文件锁保护。`priority=0` 按文件顺序，`1` 优先处理全部 Stage1，`2` 优先处理全部 Stage2；优先阶段无任务时回退另一阶段。`skip=1` 完全忽略 Stage1 行，`skip=2` 完全忽略 Stage2 行，`0` 不跳过。空值等于 0。

当前程序没有符合 `priority`、`skip` 和阶段要求的任务时正常结束；`worktodo` 不存在或读取失败时报错。`auto_stage2=1` 时，Stage1 完成后照常将原任务写入 `finished`；未找到因子时再把 worktodo 原位置改写为 `ECMSTAGE2=`，已找到因子时直接移除任务。0 或空值仍删除已完成行。改写的 save 使用绝对路径，避免两程序工作目录不同。Stage2 执行时，非零 `b2_ratio` 将有效 B2 覆盖为 save 中 B1 × ratio；0 或空值保留原 B2 选择。该选项范围为 1…1000000，任务仍须满足 B2 > B1。四项队列设置均可在 `[Worker #N]` 覆盖。

## 计算与输出

主机生成曲线和指数，GPU 分片推进点乘，主机验证末点、求 GCD 并写入结果。`gpucurves` 是批量，不表示设备同时驻留的曲线数。批量不足可能降低吞吐；适合的批量由位宽、TPI、TPB、SM 数和实际资源占用共同决定。

CPU Edwards 单曲线、SIMD 和 GPU 路径均在分块边界刷新进度；重定向到文件或 GUI 管道时输出普通文本百分比，不依赖终端控件。

GPU Edwards 先输出 `GPU Edwards initialization` 和 `GPU Edwards dictionary` 耗时，再进入与 CUDA Montgomery 相同格式的 `GPU:` 实时进度条。NAF 的累计量和单片增量单位为 `columns`；binary 为 `bits`，二者不可互换。NAF 直接使用已有列计数，不为进度重新编码标量。终端原地刷新；GUI 管道保留完整文本行，常规刷新间隔约 200 ms，首片、末片及安全停止时立即输出。

这里的 `bits` 是 Stage1 标量 S 的二进制位置数，不是目标 N 的位宽。`columns` 是同一 S 的窗口 NAF 位置数，包含零列；每列仍对应一个 2 的幂次位置，窗口 6 不代表每列处理 6 bits。每列执行一次倍点，非零列再加减一个预计算奇数倍点。NAF 列数可能与二进制位数不同，也不等于非零项数或点加法次数。比较两种路径的耗时应看同任务的完成时间及 `s/curve`；数字表示与例子见 [NAF 原理](../reference/ECM_EDWARDS_MATH.md#3-naf-标量乘法)。

Edwards 的初始化涵盖指数、曲线构造或恢复、标量编码及设备分配/上传，排除单独报告的设备字典构建。进度条 `elapsed` 从准备完成后开始，包含本次主循环的回传、输出和检查点耗时；恢复前的工作不计入本次耗时。速度取最近最多 50 个切片的推进量/时间，包含切片及状态回传，排除进度回调中的输出和检查点写入。续算以已保存位置为采样基准，避免把历史推进量计入当前速度。完整计时边界见[实现说明](../architecture/STAGE1.md#gpu-edwards)。

普通进度中的 `s/curve` 是当前速度的每曲线摊销；部分运行的投影不是完成一条生产曲线的实测时间。文件日志的进度频率由 `progress_log_seconds` 控制，GUI 所用管道进度独立保留。

Auto B2的可复用Stage1成本可用[tune_stage1_cost.py](../../tools/bench/tune_stage1_cost.py)测量完整批次并独立复核保存点。当前限CUDA ladder PARAM0、自动TPI、关闭指数缓存；可使用已知梅森素数目录，或通过`--target-n <N...>`测量指定余因子。所有测量曲线必须完成且通过无因子/独立末点校验，不接受部分运行速度投影。等级、参数、计时边界及使用方式统一见[完整Stage1成本预计算](../architecture/AUTO_B2.md#stage1-成本)。

最终 save 保存可继续 Stage2 的点；checkpoint 保存尚未完成 Stage1 的状态，二者不能互用。save 的归一化 X 必须属于记录中的 N、参数化、sigma、B1 和指数模式；checksum 用于检查文件完整性，不重新证明点乘正确。

内部 JSONL 记录全部曲线的 F/NF，并按实际批次曲线顺序导出。整条任务完成后，NF 合并为提交行（B2=B1）；验证后的因子立即额外输出为一条 curves=1 的 F，保留 sigma。账本位于输出目录的 data/ 下，字段与恢复规则见 [ECM 结果输出](ECM_RESULTS.md)。

## 检查点与交接

完整内部格式、偏移、校验范围和兼容条件见 [Checkpoint 架构](../architecture/CHECKPOINT.md)；完成点字段、校验公式、发布去重及消费者能力见 [Save 架构](../architecture/SAVE.md)。本节说明正常使用与恢复操作。

Stage1 支持按切片保存中间进度。恢复时检查任务身份、格式和实际计算布局，不能通过改名绕过检查。未完成采样只输出 checkpoint，不应当作有效最终 save。

Ctrl+C、Ctrl+Break 或 GUI 的正常停止请求会在当前后端的下一个安全边界触发检查点写入，然后结束当前任务。CUDA/OpenCL 在 GPU 批次边界保存，CPU 在曲线计算的分块边界保存；这不是在信号处理函数中直接写文件。队列模式会随即退出，当前未完成任务及后续行留在 `worktodo`，不会被标为 `# ERROR` 或写入 `finished`；重启时从检查点恢复。`ckpt_seconds=0` 只关闭定期写入，不关闭中断存档。GUI 超时后仍会强制终止 worker，因此应给大批次和慢速存储足够的 `graceful_stop_ms`。

每个任务只使用一个批次检查点，CPU 各曲线的进度和完成结果都封装在该文件中。主文件名严格为：

- Edwards：`.e_s1_m{n}_{B1}.ckpt` 或 `.e_s1_bit{bits}_{B1}.ckpt`。
- Montgomery：`.m{param}_s1_m{n}_{B1}.ckpt` 或 `.m{param}_s1_bit{bits}_{B1}.ckpt`，param 为 0、2、3；CPU 使用 0。

有 k·b^n+c 元数据的任务使用原始 n，即使剥离了已知因子；直接输入完整梅森整数也能识别 n。其他整数使用实际位宽 bits。B1 使用完整十进制整数。例如 `.e_s1_m127_1000.ckpt`、`.m3_s1_bit512_1000.ckpt`。不追加 worker 或曲线序号，同名任务需要独立的 `tmp_dir`。

当前写入 `ECMS1B2` 二进制容器：固定小端字段、逐记录长度与 CRC32C，以及完整文件 CRC32C。任务身份绑定完整模数、曲线族、参数化、B1、指数约定、曲线数及精确 sigma；CPU 另绑定实现和 NAF 窗口，GPU 载荷检查自己的算法与状态布局。Montgomery CUDA ladder 保留 V5 头和设备状态。Edwards GPU 的二进制 V3 载荷保存状态、算法和指数指纹；指数从 B1 重建或从已验证缓存取得，不将完整大指数写进检查点。

文本批次 V1 及其旧 Edwards GPU 文本载荷仍可读取。CUDA Montgomery ladder 可以从同目录旧 `.ecm_ckpt_<bits>_<first8>_<last8>.dat` 及其备份导入带校验的 V5 状态：逐 lane 核对完整 N、参数化、曲线数、布局、指数位数和进度，保留原文件。只有新批次没有算术载荷时才尝试导入；已有算术载荷优先。随机 sigma 从有效旧档继承，固定 sigma 不得被不同曲线替换；V4 及更早版本不导入。旧 GPU Edwards 独立文件 `ed_<旧模式>.save.gpu.ckpt` 也可在没有现有算术载荷时导入默认或当前模式对应的文件，继续检查完整 N、B1、指数、sigma 及曲线状态，原文件保留。

保留的旧源档不会单凭存在就反复续算：同一内部结果账本已经确认全部曲线完成时，自动导入跳过该来源。迁移检查点时应保留相应结果账本；若只将旧档复制到全新工作目录，程序无法从旧文件本身得知它是否已在其他目录完成。

原子写入使用 `.new` 临时文件；`ckpt_backups` 轮换备份，例如 `.e_s1_m127_1000.bak.ckpt`、`.bak2.ckpt`。恢复检查完整校验，主档损坏时尝试备份；有效主档属于不同任务时直接拒绝，不覆盖它。没有任何有效备份时保留损坏文件并报错。CRC 用于发现字节损坏，恢复依靠有效备份，不执行纠错码修复。只有 sigma、没有非空算术载荷的文件保留了曲线身份，但不包含曲线内进度；定期存档尚未到时可能处于该状态。

同一批次有进程独占锁，防止两个 worker 同时写同名文件。CPU Edwards 已完成曲线保留结果，CPU Montgomery 保留完成结果及尚未完成的 ladder 状态；SIMD Edwards 在各 lane 的恢复位号一致时继续该批，否则重算该 SIMD 批。GPU 存放整个设备批次的状态。只支持本容器中的对应后端载荷，不自动读取 Prime95 的检查点。Stage2 的曲线计数文件并非曲线内算术检查点；本节命名和续算保证适用于 Stage1。

最终多曲线 save、内部 JSONL 和提交 JSON 均成功发布后才删除检查点。写入失败时保留批次状态和原队列任务；恢复发布不会重复增加数学工作回执或重复追加相同 save 记录。内部与提交文件的配置、字段和校验见 [ECM 结果输出](ECM_RESULTS.md)。

Prime95 交接资格按每条任务应用文件名前缀后的实际曲线判断。Edwards 与 GPU Montgomery PARAM2 自动忽略配置的 Prime95 转移路径，仅输出说明，不访问该路径或转移账本。受支持的 Montgomery PARAM0/3 可将 save 同步并向 Prime95 `worktodo.add` 追加任务，save 字段指向实际输出文件；失败交付进入 pending 文件，后续重试。不要由两个程序共同改写 Prime95 活动 `worktodo.txt`。

`edwards_stage2_transfer=1` 可把已完成的 Edwards Stage1 任务作为 `ECMSTAGE2=` 投递到共用 worktodo 的另一 worker 分区；`edwards_stage2_worker=0` 沿用当前编号，正整数指定接收 worker。该机制与原位 `auto_stage2` 分别适用于跨 worker 和同 worker 交接；同时启用时由 `auto_stage2` 接管当前任务。投递行使用实际 save 的绝对路径，并保留 B2、曲线范围、AID 和已知因子。

跨 worker 投递前先将记录写入队列旁的 `.stage2_pending_workerN`；目标不可写时保留待投递记录并在下次启动时重试。已经存在于活动队列或 Stage2 finished 的相同行不会重复入队。已有 pending 记录在关闭交接开关后仍会重试。

## 算法与构建范围

CUDA 默认路径为 ladder。resident/PRAC 必须在构建时启用，再显式选择；发布入口默认不编译这些实验模块。TPI 和寄存器实验的适用条件见 [Stage1 实现](../architecture/STAGE1.md) 与 [性能](../performance/STAGE1.md)。

Stage1 与 Stage2 仍为独立可执行文件。GUI 按队列行自动选择并在当前阶段任务耗尽后切换；直接启动 CLI 时应分别运行两个程序。

## 代码入口

- [ecm_driver.cpp](../../src/core/ecm_driver.cpp)：命令行、INI、队列、进度和交接。
- [cgbn_stage1.cu](../../kernels/cuda/cgbn_stage1.cu)：CUDA 主机调度、曲线布局和结果处理。
- [ecm_stage1_exp.cpp](../../src/core/ecm_stage1_exp.cpp)：指数生成与缓存。
- [CPU 实现目录](../../src/cpu/)：Montgomery、Edwards 和保存点。
