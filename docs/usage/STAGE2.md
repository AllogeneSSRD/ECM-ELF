# Stage2 使用

统一的 `ecm_stage2.exe` 可让同一 worker 按任务使用 CPU 或 GPU。CPU Edwards 续算及 GWNUM／libecm 选项见 [Edwards CPU Stage2](EDWARDS_CPU_STAGE2.md)。两份结果文件及提交字段规则见 [ECM 结果输出](ECM_RESULTS.md)。

实验、归因和性能对照使用的进程环境开关见 [Stage2 点与队列](ENVIRONMENT_STAGE2.md)和 [CUDA NTT 与多项式](ENVIRONMENT_CUDA_NTT.md)。

## 选择运行设备

`device = "auto"` 是统一程序的默认值。每条 worktodo 任务按存档的实际模数（扣除已知因子后的 N）匹配 `stage2_algorithm`，直接使用 `--save` 时以首条选中的曲线为准。同一个 save 批次应使用相同的模数和曲线族。以下是默认规则，Worker 3 单独固定为四线程 CPU。

```toml
config_version = 4

[stage2]
device = "auto"
stage2_algorithm = "gwnum[4096:], gpu[:16385], cpu[]"
stage2_gpu_device = 1

[worker.3]
device = "cpu"
stage2_cpu_threads = 4
```

规则从左到右取首个可用项，区间含下界、不含上界，`[]` 匹配任意位宽。也可以填写单个算法名。算法名不区分大小写。规则可在 `[worker.N]` 覆盖。

- Edwards 从 4096 bit 起优先使用 `gwnum`（GWNUM/polymult）。4096 bit 以下优先用 GPU。后续规则为缺少 DLL 或 GPU 的 Edwards 环境提供回退。
- `cpu` 读取 `cpu_backend`，`cpu-ntt` 和 `cpu-plain` 直接选择 libecm。CPU 目前支持 Edwards。兼容的 Montgomery 存档由 GPU 续算。
- `gpu` 读取 `stage2_gpu_backend`，`cuda` 和 `opencl` 直接指定 GPU 后端。GPU 支持的上限为 16,384 bit，worktodo 中的原始指数也受此上限约束。卡号由 `stage2_gpu_device` 或 `--device N` 指定。

没有可用规则时，程序报告原因并保留当前 worktodo 条目。选型结果打印为 `stage2_algorithm: bits=… selected=… device=…`，CPU 的设备编号为 `-1`。续算仍使用原有的整曲线检查点。

`device = "cpu"` 和 `device = "gpu"` 固定设备，`--target cpu|gpu|auto` 可覆盖 TOML。在自动模式下，`--backend`、`--gpu-backend` 或专用诊断命令显式选择对应后端。GPU 调优示例为 `ecm_stage2.exe --target gpu --tune ntt --device 1`。独立的 CPU 或 OpenCL 程序使用其原有后端配置。

## Edwards 命中后续算余因子

`[stage2] stage2_after_factored = true` 让 Stage1 在 Edwards 命中后继续交接本地 Stage2。Stage1 按产生存档的 worker 读取此选项，默认关闭。

Stage1 从原模数剥离本批记录的因子，把可用曲线的终点约化到同一个余因子，然后另存为 `.cofactor-*.save` 并交接。命中曲线自身的终点也参与投影，前提是它在目标余因子上的分母可逆。新队列行保留 B2、AID 与已知因子，并附加本次剥离的因子。原存档保留原模数和 `FACTOR` 标记。余因子无可用点时不生成任务。

直接读取原始 save 时，Stage2 跳过 `FACTOR` 和 `IDENTITY`。命中后续算使用上述余因子存档。

## 原生 OpenCL 入口

`ecm_opencl_stage2.exe` 共用本页的文本 save、`ECMSTAGE2` 队列任务、结果发布和停止策略。同目录安装两套 worker 时，在 `[stage2]` 设置 `stage2_gpu_backend`，默认 `"auto"` 优先使用可见的 CUDA 设备。先运行 `ecm_opencl_stage2.exe --list-devices`，再按输出填写 `stage2_gpu_device`。OpenCL 和 CUDA 分别编号，OpenCL 还可能在多个 ICD 平台上列出同一设备。

直接续算可用 `ecm_opencl_stage2.exe --config ecm.toml --save FILE --device N --b2 B2 --d D`；队列入口为 `ecm_opencl_stage2.exe --config ecm.toml --worker N --once`。D=0使用独立基本模型和owned-payload预算。支持显式 `--carrier-exponent P`，先验证N整除2^P−1；不自动比较承载路线。

须提供非零 B2，可来自 CLI、worktodo、`b2` 或 `b2_ratio`。OpenCL 支持本后端的运行时点计划调优。离线 NTT/ECM tune、Auto B2、自动承载比较及 relocation 仍限定 CUDA。两个后端分别维护实测文件。OpenCL 忽略 TOML 中的 CUDA 加权 NTT 预设。显式请求 CUDA 加权 NTT 或通过 `--parallel 2` 请求双槽会报错。OpenCL 采用阻塞传输和事件等待，`low_cpu_wait` 只控制 CUDA 的主机等待方式。

`arena_mib` 是整个OpenCL owned-buffer硬预算，包含F树、结果、NTT临时区和表；它与CUDA arena口径不同。0表示自动payload预算；`opencl_payload`日志记录本模块实际账本峰值，不是进程显存峰或设备空闲显存。预算不足会失败并保留队列，不自动放宽。

NVIDIA 可由原生 OpenCL 计划选择设备点或 CPU 点；AMD目标超过512 bits使用CPU点生成，继续使用OpenCL多项式流水线。归一化和最终GCD由CPU/GMP完成。功能边界见 [OpenCL架构](../architecture/OPENCL.md#生产-stage2-入口)，构建与性能见 [OpenCL Stage2性能](../performance/OPENCL_STAGE2.md)。

### OpenCL 性能对照开关

同一构建下的点计划对照见 [OpenCL Stage2 环境变量](ENVIRONMENT_STAGE2.md#原生-opencl-stage2)。

生产 TOML 可在 `[stage2]` 设置 `opencl_point_plan="auto"`、`opencl_tune_file="stage2_opencl_tune.toml"`。打开 `tune_runtime_enabled` 后通过真实任务学习；`tune_min_repeats=1` 可快速试用，默认 2。使用独立路径可分离各 worker。非零的 `opencl_point_chunk`、`opencl_point_group`、`opencl_point_window`会锁定对应参数；设备短批次另由 `opencl_short_point_batch` 控制。完整键名、取值及默认值见[配置参考](../ECM_TOML_REFERENCE.zh-CN.md#stage2_opencl_point_plan)，CUDA 与 OpenCL 的区别见[运行时点计划](../architecture/OPENCL.md#opencl-运行时点计划)。

在已验证的 HX 370／RTX 4060 Laptop 生产形状上，`auto` 从 GMP 点递推开始，启用调优后与设备 giant 递推对照。日志中的 `stage2_opencl.point_backend` 给出实际执行位置，`opencl_cpu_chain` 给出 CPU 递推的点数与恢复窗口数。其他形状可显式选择 `"cpu"` 做整曲线对照；多项式运算仍使用 OpenCL。[生产实测与边界](../performance/OPENCL_STAGE2.md#生产-b2-的同几何采样)

点chunk的初始化值按真实目标N的64-bit limb数w选择：NVIDIA在w≤8时1024、9≤w≤128时256、w>128时16；其他厂商依次为w≤8:1024、w≤32:256、w≤64:128、w≤128:64、其余16。自动模式从实际首条命令学习，不另做warmup。9～128 limbs的升档目标为compute units×32且不超过4096，目标不高于初始值时保留初始值；1秒是带15%裕量的升档门限，不是命令时长硬保证。129～256 limbs采用下述渐进策略，最多256点。两者保留当前owned-buffer余额检查；AMD宽位CPU回退不提交点kernel。

窗口是主机坐标暂存范围，chunk是单次设备命令上限，两者需配合。小P时仅增大chunk仍受原G批点数限制。自动窗口优先取G批与点上限的最小公倍数，预算不足则取预算内的整G批；至多65536点。额外坐标准入按 `6×算术limbs×8×点数≤batch MiB` 估算，不包含GMP分配器开销。`opencl_point_window`打印请求、有效点数、估算cap、G批及来源；`opencl_point_plan`打印资格、设备CU、初始／最大chunk、有效反馈数、实际升档和预算裁剪。对照时用`ECM_OPENCL_POINT_AUTO=0`关闭自动计划；显式窗口0只锁定窗口，仍允许符合资格的baby批量升档。实测与范围见[自动点执行计划](../performance/OPENCL_STAGE2.md#自动点执行计划)。

`log_level="phases"`开启点内核、CPU检查/归一化、树阶段与程序缓存的汇总；默认批进度级别不新增这些明细。日志字段与嵌套计时见 [OpenCL架构](../architecture/OPENCL.md#生产-stage2-入口)。

设备点路径默认开启宽位升档；符合短标量计划时自动启用巨步递推，其余形状可显式开启递推对照。宽位计划最多256点，完整32点组的单位点／bit成本超过16点seed的1.10倍时退回seed；其后每bit命令成本增加超过35%时拒绝升档并停止增长。它没有1秒硬时限。巨步递推的提交点数与段长另受设备规模、预算和手动chunk限制；不能由POINT_AUTO的1秒门限推断其最长命令。两者保持原G批、NTT工作和独立检查，CPU／AMD宽位资格回退不变。

## CUDA 入口

选择 GPU 和 CUDA 后，`ecm_stage2.exe` 读取完成的 Stage1 save，运行 CUDA 多项式 ECM Stage2。手动 B2 可处理实际 N 为 2…16384 bits 的受支持输入。Auto B2 另受 profile 资格限制。设备编号是 CUDA 运行时编号，与构建的 sm 架构不同。

直接读档和队列运行启动时均输出 `ECM-ELF x.y.z (Stage2)` 版本横幅，包括 `--dry-run` 和 `--plan-only`；GUI 从这个横幅读取实际 worker 版本。只输出机器数据的内部辅助入口保持其原有格式。

CUDA点算术使用`[stage2]` 中的 `baby_backend="auto"`、`giant_backend="auto"`默认配置。Baby可强制scalar／coop32，giant可强制scalar／coop16／coop32；CLI为`--baby-backend`／`--giant-backend`。协作仅支持实际算术8193…16384 bits，承载时按承载位宽判断；giant还需真实chain。auto先保留标量，在当前D／布局／链长上先比较baby、再逐项比较giant，受`tune_min_repeats`控制；达到重复数后选最快合格均值，再继续D与链长搜索。短队列或关闭运行时探索时不会为收集数据额外重复任务。配置范围见[TOML参考](../ECM_TOML_REFERENCE.zh-CN.md#stage2_giant_backend)，学习、回退和性能文件复用见[协作点后端选型](../architecture/AUTO_B2.md#协作点后端选型)。

多卡共用工作目录时，可为每个 worker 指定自己的设备和性能文件。

```toml
config_version = 4

[worker.1]
device = "gpu"
stage2_gpu_device = 0
tune_file = "tune_gpu0.toml"

[worker.2]
device = "gpu"
stage2_gpu_device = 1
tune_file = "tune_gpu1.toml"
```

相对路径基于 `ecm.toml` 所在目录。每个 worker 的 CUDA Stage1 和 Stage2 共用选定文件。省略 worker 覆盖时继承 `[shared] tune_file`，默认 `tune.toml`。CLI `--tune-profile` 可为单次 Stage2 命令显式指定输入。示例 GPU 编号按实际设备调整。

CUDA 完整区间和 CPU gwpoly 默认启用 `b2_round_chain`。设为 false 或传入 `--b2-round-chain 0` 可使用精确 B2。CUDA 凑整支持运行时自动调优：各D按实际凑整B2准入、比较和记录，均覆盖请求B2。不同实际B2的行保持独立，现有重复数和搜索预算继续生效；细则见[点块与有限链长身份](../architecture/AUTO_B2.md#点块与有限链长身份)。

CPU gwpoly 同样读取这个开关，按自己的高位后缀补满最后一个 P 点 G 块。直接读档和队列的结果均使用实际 B2，内部结果另保留请求值。[CPU 整块扩展](EDWARDS_CPU_STAGE2.md#将-b2-扩展到整块)

CUDA的`seed_backend`默认auto，也可强制scalar／coop32；CLI为`--seed-backend`。它控制8193…16384实际算术bits的paired seed及GPU构建[D]Q，其他情况仍为标量。运行时先完成同形状、R=`tune_min_repeats`次的标量／协作对照，再选最快合格均值；正式tune可复用配对区域。只读模式继续采用已有合格胜者，单独强制协作的未配对记录不足以启用auto。实际`seed_coop`按行记账，NTT表继续共享。详见[配置参考](../ECM_TOML_REFERENCE.zh-CN.md#stage2_seed_backend)与[自动选型合同](../architecture/AUTO_B2.md#giant-seed自动选型)。

普通高位宽S4可显式设置`reduce_backend="coop32"`，CLI为`--reduce-backend coop32`；默认`scalar`。仅CUDA实际算术8193…16384 bits的普通归约启用协作，小位宽和梅森归约继续原内核。它与seed／baby／giant的auto独立；ECM性能文件分开记录两种归约策略，独立NTT表可共用。当前设备资格、固定D收益和限制见[S4性能](../performance/STAGE2_S4_REDUCTION.md)与[配置参考](../ECM_TOML_REFERENCE.zh-CN.md#stage2_reduce_backend)。

## 直接读档

GPU 直接读档可用 `--target gpu --save <file> --b2 <integer> --device <id>`。`--skip-curves <n>` 跳过记录，`--curves <n>` 限定数量；其余参数见 `--target gpu --help`。B1、sigma 和 N 取自 save，不重新执行 Stage1。`--factor-only` 省去命中素数命名，保留叶乘积、GCD 与必需检查；`--factorize-hits` 开启共享诊断层的附加分解，优先本地 PARI DLL，备选外部 GP，不自动使用 GMP。原始因子先发布，附加诊断未完成不改变成功状态；仅完整且全部已证素才标为分解完成。内部结果保留后端、概率素性及余因子，详见[群阶模块](../architecture/GROUP_ORDER.md#共享后端选择与进程隔离)。

接受归一化 Suyama PARAM0 文本记录，以及 GPU Edwards 输出的 `METHOD=ECM-EDWARDS-AM` 记录。Edwards 使用生产 Stage2 引擎并要求显式 B2；存档身份和常数约定见 [输入与表示](../architecture/STAGE2.md#stage1-save-与承载资格)。CPU Edwards 的 Prime95 二进制 `.tmp` 不属于此入口。先验证选择范围中的所有记录，包括 checksum、N、B1、X 和任务匹配；文件损坏不能当作“没有足够曲线”略过。源 save 不修改。校验和通过只说明文件完整。

直接多记录执行不使用队列自动续跑。每条曲线在独立子进程计算，结束后释放 CUDA 上下文与临时资源。

`--results` 的内部 JSONL 以 `curve=ed|mont` 标明模型，Edwards 的 `param` 为 null，Suyama 的 `param` 为 0。另写 `--submission` 指定的 `results.json.txt`；TOML 的 `user`、`computer` 用于提交身份，标准梅森队列自动传递原始指数、已知因子和可选 AID。直接处理梅森余因子时应提供 `--exponent P --known-factor F`，输出前验证与实际 N 的关系。批次去重和恢复见 [结果输出](ECM_RESULTS.md)。GUI 的因子结果文件使用其独立结果格式，见 [GUI 日志与结果](GUI.md#日志结果与监控)。控制台结果事件也携带模型及完整 sigma；显式 `--gui-events` 会在 quiet 级别保留该结果事件，普通 quiet 控制台规则保持不变。

Stage2 父进程启动时读取 TOML，任务开始时从该快照和当前任务行确定有效设置。并行曲线与结果发布沿用同一配置，包括提交身份和解析后的路径。运行中修改 TOML 不会影响当前进程的任务；需要应用新设置时请重新启动。重启时仍校验 checkpoint 的提交路径，已经记录的发布内容保持原身份。

## CUDA 双曲线调度

在 `[stage2]` 设置 `curve_concurrency`。单个 worker 可在 `[worker.N]` 用 `curve_concurrency` 覆盖，命令行仍接受 `--parallel auto|0|1|2`。双槽要求同一任务中的 N、B1 和曲线族一致，并已确定有效 B2。

- `"auto"` 按实测自动选择。无匹配成绩时，先用真实曲线测串行，再测可准入的双槽候选。窗口由 `parallel_tune_curves` 指定，默认 5，范围 2–16。5 条及以上取第 2 条完成到倒数第 2 条完成之间的吞吐，要求提高至少 3% 且整窗墙钟没有增加。2 至 4 条用整窗吞吐，要求提高至少 3%。合格候选按整窗每条耗时择优。不同条数的 tune 记录独立匹配。工作区为 auto 且双工作区预算不足时，直接采用可准入的 F 树前租约。
- `"serial"` 单曲线串行，默认。固定参数对照可以显式传 `--parallel 1 --tune-runtime 0 --tune-profile FILE`，将合格的串行窗口作为比较基准保存。
- `"dual"` 用已选路线尝试双槽。完成一条即补下一条，显存或计划不满足准入时串行处理。

各 worker 保持独立 CUDA 上下文。父进程可在 GPU 计算期间预读下一条，或让上一条继续发布。按算术位宽，单槽和仅报告因子的双槽默认预读至 4096 bit，保留因子归因的双槽至 2048 bit。调试日志按进程槽附加 `.slot1` 至 `.slot3`，内部 JSONL 按实际完成写入。中途命中因子的窗口仍发布结果，该窗口不计入调优。

曲线级日志用 `stage2_queue: workers=1|2` 显示当前窗口的调度模式。双槽和启用预读的串行窗口中，子进程的阶段、批次、结果和警告带有 `[curve N]` 前缀。N 是任务内从 1 开始的曲线序号，`record` 是 save 文件中的记录号。`curve_prepare` 表示开始预读，`curve_start` 表示获准计算。`curve_done` 中的 `completed` 是任务中已有完成记录的总数，包含续跑恢复的结果。曲线可乱序完成，完成数独立于曲线序号。

双槽准入日志中的 `required_free_bytes` 和 `free_bytes` 分别是所需和查询到的空闲显存，单位为字节。`insufficient_device_memory` 表示显存不足，`device_memory_query_failed` 表示查询失败，`device_queue_busy` 表示另一队列已占用设备调度锁，`device_queue_lock_failed` 表示锁操作失败。候选试跑同时记录 `D`、实际 `B2` 和 `requested_B2`。

`curve_done` 的 `eta` 以秒为单位，是按当前调度窗口估算的任务剩余墙钟。串行使用本窗口已完成曲线的平均墙钟。双槽队列有后续曲线时用完成速率估计，收尾时用活跃曲线中最晚的预计完成时间。双槽积压窗口需至少两个完成样本才报告 ETA，切换窗口时重新估计。停机请求后不再估算后续任务，最后一条完成时 ETA 为零。`wall` 保留单条曲线的墙钟耗时。双槽从窗口内第二条曲线完成起输出 `queue_s_per_curve`，用窗口墙钟除以本窗口实际完成的曲线数。GUI v2 用它显示双槽的队列吞吐。

高级对照可设置 `ECM_STAGE2_QUEUE_PIPELINE=0` 关闭预读、`1` 仅预读固定路线、`2` 同时预读运行时调优。`ECM_STAGE2_EARLY_GPU_HANDOFF=0` 禁用块积后的提前交接，`NTT_STAGE2_RESIDENT_PRODUCTS=0` 强制上传叶值后计算块积。`ECM_STAGE2_QUEUE_D_SEARCH=1` 在首选路线显存充足时也搜索备选。`ECM_STAGE2_GIANT_GATE=loop` 实验性地把整个 giant、G 树和 fold 循环合成一次持锁，默认仍逐批交接。

在 `[stage2]` 设置 `workspace_policy`。默认 `"auto"`。`"independent"` 让两槽各持一套工作区，`"leased"` 让下一槽先做准备和 Baby 点，到 F 树前再申请独占大工作区。并发为 `"auto"` 时按实测比较可准入的模式。并发为 `"dual"` 时使用指定模式，工作区也为 `"auto"` 时优先双工作区，预算不足时使用可准入的租约模式。工作区模式和阶段掩码分别参与 tune 匹配。见 [阶段调度](../architecture/STAGE2.md#双目标阶段调度)。

`ECM_STAGE2_QUEUE_PHASE_MASK` 可给队列双槽指定阶段互斥掩码，默认 `57`。用 `0` 允许所有阶段交错，`56` 放开 Baby，`32` 只锁下降，`25` 放开下降，`49` 只移除 giant 点生成的锁。各掩码保留各自的双槽成绩。实时显存检查、设备队列互斥和显式启用的工作区租约仍生效。

## 队列语法

save 内部字段及读取限制见 [Save 架构](../architecture/SAVE.md)；队列 checkpoint 的字段、身份摘要与持久化顺序见 [Checkpoint 架构](../architecture/CHECKPOINT.md#7-stage2-队列进度格式)。

队列接受 `ECMSTAGE2=` 和 `ECMSTAGE2N=`，语法和样例见 [README 的 Worktodo 格式](../../README.md#worktodo-格式)。有原始表达式时用前者校验存档的实际模数，并可利用原梅森指数选择承载。只有 save 时用后者，采用第一条所选记录的完整 N，并核对其余所选记录的 N 和 B1。尾部已知因子用于结果记录，无需再次从 N 中剥除，也不用于推断原梅森指数。

整数支持精确十进制和科学记数法。CLI 和任务行的 B2 字段也接受 `B2min-B2`。例如 `--b2 5000-8000`、`ECMSTAGE2=1,2,127,-1,"point.save",5000-8000,0,1` 和 `ECMSTAGE2N="point.save",5000-8000,0,1` 都要求覆盖 `(5000, 8000]`。区间要求 `存档 B1 ≤ B2min < B2`。

Stage2 继续使用 save 中的曲线终点和真实 B1。B2min 只用于确定巨步起点及小素数搜索下界。多项式按整个轮格处理，边缘格点、倍点和归一化仍可能发现区间外的有效因子。支持 B2 凑整的后端以区间内的巨步数量计算对齐，只提高 B2 上界。原有后端选项的适用条件继续生效。

非零 `b2_ratio` 按已有规则用存档 B1 × ratio 覆盖上界，区间下界继续参与校验。其余情况下依次采用非零的 CLI 值、任务值、`b2`。CUDA 在有效 B2 为 0、`b2_ratio` 为 0 且启用 Auto B2 时才自动选择。`stage2_plan` 的 `B2_source` 记录本次选择的来源。不同请求区间各自持有检查点，两端相同的任务可继续使用整曲线级检查点。

## 数量不足与错误

删除性能文件会触发默认启动短校准。成本探测固定使用小B2，不继承`b2_round_chain`；正常任务之后仍可按该开关凑整。遇到`invalid completed calibration shape`时查看完整错误中的预期／实际点数及证据路径，这类错误不能据此判定Stage1 save损坏。实现与严格校验条件见[启动短校准](../architecture/AUTO_B2.md#启动短校准)。

请求 10 条、跳过后仅有 3 条时，提示 requested/available，执行这 3 条。全部成功后按 completed=3 完成原任务，继续下一项；可用记录为 0 时提示并按零曲线完成，不等待 save 追加。

存档缺失、损坏、checksum/N/X 不匹配或任务字段错误属于输入错误。队列原行标记为错误注释，finished 保留原任务和原因，随后继续下一项。当前配置或后端无法为任务提供有效 B2 时（例如 OpenCL 收到只有 Auto B2 可用的任务），停止并保留该任务。文件权限、状态对账、设备、显存和未知内部故障也会停止并保留未完成任务。退出码为正常 0、存在队列输入错误 1、致命错误 2。

## 自动续跑与退出

新队列任务使用 `.ecm_ckpt_stage2_<实际N位数>_<B1>.dat`，记录完成状态、NF 汇总和已验证因子。续跑需要原队列、Stage1 save 和 ckpt；逐曲线 JSONL 可选，提交结果也可以从尚存的 ckpt 补发。提交路径在任务存续期间保持一致。阶段内的 NTT、树和多项式状态不会写入 ckpt。

重启可以更换设备、显存预算、D、承载设置和日志路径，或移动内容完全相同的 save。AID、实际 N、save 字节和 skip 绑定任务身份。当前数量和有效 B2 可应用到剩余曲线。例如 150/960 改为 300 时从第 151 条续算；改为 100 时保留已完成的 150 条并发布。遇到双槽已完成曲线之间的缺口，先补齐再缩到缺口之前。

第一次 Ctrl+C 停止启动后续曲线，已运行的曲线可以提交结果后退出。第二次会终止子进程树，尚未写入 ckpt 的曲线下次重算。最终提交完成后才写 finished、移除队列行并清理 ckpt。

Stage2 schema 6 ckpt 保存进度、待发布结果和发布确认。直接 `--save` 也使用该恢复流程，并保留完成 ckpt。旧档升级条件见 [Checkpoint](../architecture/CHECKPOINT.md#74-旧档的一次性迁移)。


## TOML 与路径

默认读取 exe 同目录的 `ecm.toml`；隐式文件缺失采用内置默认值，显式 `--config FILE` 缺失或格式错误时报错。配置分区、优先级和逐键覆盖资格见[配置参考](../ECM_TOML_REFERENCE.zh-CN.md)。

Stage2 与 Stage1 共用 `[shared]` 的 `worktodo_file`、`finished_file` 和 `log_file`。Stage2 的结果、提交、检查点和调试日志在 `[stage2]` 单独设置。队列领取 `ECMSTAGE2=` 和 `ECMSTAGE2N=`。`[stage2] b2_ratio` 在直接读档和队列模式下按存档 B1 × ratio 覆盖有效 B2，worker 使用 `b2_ratio`。

共用 `worktodo` 及 Stage2 相对路径基于 TOML 目录，其他 Stage1 相对路径基于其程序目录；CLI 相对路径基于当前工作目录。进度目录空值使用队列目录；显式旧进度路径仍可作为导入来源。`stage2_save_dir` 为空时继承 `stage1_save_dir`，`stage2_gpu_device = -1` 继承 `stage1_gpu_device`。推荐将两个程序与共用 TOML 放在同一目录。

在 `[stage1]` 中设置 `handoff_mode = "ecm-elf"`，可将无因子的 Stage1 任务转为 Stage2。默认追加到原 worker 队列末尾，也可指定接收 worker 列表或其他队列文件。Edwards Stage2 需要有效 B2，可来自任务、`b2` 或队列比例 `b2_ratio`。见 [Stage1 自动交接](STAGE1.md#自动交接到-stage2)。

运行时拒绝 Stage1/Stage2 队列冲突及 save、程序、配置、队列、进度、结果和日志之间的危险重叠。

## 日志与内存摘要

可选 `relocation=1`（或 CLI `--relocation 1`，默认0）在相同 B2 下试算较短 giant 后缀。命中或目标 N 非单位时自动重放完整原区间，完成验证后才发布结果。选 D/内存规划仍按原区间，完整 ECM tune 不支持此模式；详见 [算法、指标及边界](../architecture/STAGE2.md#可选-relocation) 与 [性能实测](../performance/EDWARDS_STAGE2.md)。

`verbose` 控制可读的阶段/批次输出；`debug_log` 控制独立调试文件。警告和错误始终可见。长阶段每 30 秒的父进程状态提示不等于 kernel 完成百分比。

- `curve_start`：任务曲线编号、save 记录编号、sigma、B1 和文件校验状态。
- `stage2_phase`：`previous` 为上阶段墙钟间隔，`elapsed` 为曲线累计时间。候选计划查询不输出正常阶段日志，也不更新曲线阶段时钟；实际运行的规划阶段每条曲线只打印一次。独立调试日志与规划 JSON 保留诊断信息。
- `real_shape`：B1/B2、算术位宽、D/P、baby/giant 数及 G 批次数。
- `mem`：设备 free/total、reserve 和 arena cap；预计最大变换及树顶形状。
- 曲线结束的 `arena=a/b fold=a/b`：a 为实际完整 NTT 池峰值或启用 fold owner 占用；fold 回退时 a=0。b 保留读取的 TOML 值，可能与 CLI/环境覆盖后的有效预算不同。
- `ntt_length_counts=1`（默认 0，CLI `--ntt-counts` 可覆盖）：曲线结束输出按长度排列的卷积、正向、逆向和总次数表，并保留 `ntt_length_counts` 兼容行。一次卷积为两个正向加一个逆向，不是 kernel 启动数；关闭时不采集此表。`ntt_work` 的元素数、逻辑蝶形数、计数边界及开销见 [NTT](../architecture/NTT.md#实际运行的长度与运算量)。
- `stage2_full_wall.total`：完整引擎 init+main，排除 Stage1 与自动 D 扫描。外层 result/curve_done 还含进程和发布结果等开销。
- 运行时调优摘要只有成功补充缺失算术位宽/执行区间摘要时输出，包含位宽、算术家族、D/B2与实测/预测比值。默认采集已执行的干净完整曲线，可用`tune_runtime_enabled=false`关闭；完整规则见[运行补充](../architecture/AUTO_B2.md#正常运行补充缺失位宽)。

`arena_mib` 指定 NTT 池 payload，不是全部显存。独立 giant、S4、owner、下降元数据、按次回退和驱动资源另计。组件峰值不能直接相加；详细含义见 [内存](../architecture/MEMORY.md)。普通日志和结果默认不轮转。

正常离线计算不需要监听网络。Windows Compute Sanitizer 的注入通信可按目标 exe 名称触发防火墙提示；测试工具在 Windows 子进程使用 named pipes。该行为不表示 Stage2 算法需要网络权限。

## 固定参数G树实验

生产配置 `groot_balanced = "auto"` 默认为自动选择；`true`强制分块均衡布局，`false`强制左侧连续布局。CLI `--groot-layout auto|true|false`覆盖当前worker配置，保留0/1别名；auto也会传给曲线子进程，以免重新读取TOML强制值。tune矩阵的 `groot_balanced` 仍记录实际布局0/1，不记录第三种auto布局。OpenCL接受auto，强制布局限于CUDA。

只比较 G 树布局时，可以用 `--groot-layout 0` 和 `--groot-layout 1` 做 A/B，固定 D 和 B2，并关闭 Auto B2、自动承载、运行时采样、短校准和短 NTT 调优。使用独立实验目录与调参文件隔离已有的性能记录。[复现实验](../performance/STAGE2_GTREE_LAYOUT.md)

不同形状有收益也有回退，结果与受控复现入口见[G树性能](../performance/STAGE2_GTREE_LAYOUT.md)。

## CUDA 加权 NTT

`--weighted-ntt 1` 启用根加权多项式卷积，也可在 `[stage2]` 或 `[worker.N]` 中设置 `weighted_ntt = true`。默认 `false` 使用全长线性卷积。加权要求 `pack_zero = true`，可用于 Edwards 和 Montgomery；直接设备打包或缓存根表不可用的调用走全长路径。曲线结束的 `cuda.weighted_ntt` 日志列出实际执行的加权卷积数。完整曲线配对结果见 [根加权半长卷积](../performance/NTT_OUTER.md#根加权半长卷积)。

## CUDA点块与链长选项

可用 `giant_tail_balance` 对完整giant区间末两块借整P点，避免可消除的ladder短尾；它保持B2与点总数。未实际借块时继续使用原执行与缓存身份，不增加策略重放或额外同步；日志及结果分别保留请求和实际值。默认仍为false，真实借块的原分块重放预算尚待完整扩展。`b2_round_chain` 则增加实际B2到最小整P且使尾块可走chain，日志和结果保留请求／实际边界；它会增加覆盖量和可能的G树工作。

`giant_chain` 可固定32或64，或以auto在正常任务中对当前最优项补有限32/64对照。默认保留原点块和64链；这些CUDA选项尚未获得全设备／全位宽默认启用资格。配置取值见[生成参考](../ECM_TOML_REFERENCE.zh-CN.md)，完整约束及回退见[点策略](../architecture/STAGE2.md#完整区间点块策略)，实测见[固定D报告](../performance/STAGE2_PREPARATION.md#完整尾块平衡b2凑整与3264生产选项)。

## 规划与依据

`--plan-only` 查询设备及算法/组件形状，不执行任务曲线、不写成功结果、不推进队列；默认启动短校准可能额外运行小基准并更新性能文件，可用`--short-calibration 0`关闭。`--dry-run`只检查输入。`--carrier-exponent p`要求保存的目标N整除2ᵖ−1，显式0锁定按save的N运算。默认`auto_carrier=1`允许标准梅森worktodo在已知因子、save和整除验证后比较普通与承载；设置0关闭自动候选，CLI `--auto-carrier 0|1`覆盖TOML，显式carrier锁定仍优先。候选资格、性能保护及指数上限见[Auto B2 / tune](../architecture/AUTO_B2.md)。

`--tune ecm --tune-level 1`优先读取TOML指定的当前worker队列与有效save，采用任务/TOML的生产B2；`--tune-worktodo FILE`可指定队列。`auto_carrier=1`时先补普通默认基线与经整除验证的承载默认对照，再从模型D/P搜索相邻项；只有平均耗时降低超过2%且越过双方2×样本标准差噪声界限才继续移动。不会推进生产队列。

`--tune-exponents bits,...`明确使用通用位宽网格；默认队列缺失或无Stage2任务时也使用网格。`--tune-save FILE`使用一个保存点，自动承载指数可来自当前worker中重建N相同的任务；也可用`--tune-carrier-exponent p`明确指定。完整梅森N的普通路径已使用梅森归约，自动模式不重复测量相同承载。

默认1800秒软预算，`--tune-budget-seconds`可覆盖。ECM、NTT及启动短校准默认不运行丢弃的预热轮次；`tune_warmup=1`或CLI `--tune-warmup 1`启用每个新测形状的一次预热。正式`repeats`记录实际次数，保存平均耗时和样本标准差；必要算术检查保持不变。缺NTT资料时补轻量组件校准，真实承载任务的普通/承载参考优先占用预算。

`--tune-d`、`--tune-b2`、`--tune-exponents`使用逗号列表，其中exponents表示实际位数；显式D表限定搜索范围。格式8每个位宽一个`matrix`，首行为列名，后续行与表头对齐；同参数重复执行合并repeats、均值和样本标准差，不同参数保留多行。行内保存来源程序版本与UTC更新时间，版本不影响匹配或累计，普通/承载通过`carrier_exponent`和`modulus_kind`独立记录，算术位宽自动恢复，详见[算术路径字段](../architecture/AUTO_B2.md#toml-中的算术路径)。不产生`[sample.*]`或未访问的模型网格。兼容其他程序版本的format 8，并将旧条件级版本迁入行内；format 8以前的不兼容结构须重新生成，失败不覆盖原文件；队列进度和Stage1 save不受影响。控制台保留`ecm_tune_progress`等进度，内部诊断保存在证据日志。

`--tune ntt`可用`--length-log2 a:b`与`--tune-slices s,...`测指定长度和批量，或用等级1…10预设。输出分别记录每秒域卷积数和每秒批调用数；它们不是ECM曲线吞吐。计时边界、预算跳过和格式见[NTT tune](../architecture/AUTO_B2.md#ntt-tune)。

用 `--tune ecm --tune-merge A.toml --tune-merge B.toml --tune-file combined.toml` 汇集合格摘要，再让 `tune_file` 指向结果。同形状的实测摘要优先保留样本数较多的一行，次数相同时采用较新的记录。不同形状各自保留。可用 `--tune-ntt-profile NTT.toml` 导入组件。该命令不运行 GPU 测试，输出应使用独立文件。[合并规则](../architecture/AUTO_B2.md#汇集预计算结果)。

生产用`--tune-profile FILE.toml`或`tune_file`加载资料，不要求B1匹配。默认`prediction_policy=model`：普通按实际N位宽，梅森按p的算术位宽查询各自区间；精确实测优先，其他输入使用基础模型加区间校正。更改D和切换承载分别通过双侧误差保护，NTT/驻留边界及当前显存重新规划。设为`paired`或CLI `--prediction-policy paired`恢复精确位宽/B2/形状的配对实测门禁。

没有用户调优文件时也有内置4060 Laptop/1800MHz参考模型，输出标为builtin_model；范围外可以外推，但不承诺独立精度或未测性能。启动短校准默认保留，可用`--short-calibration 0`查询纯基础预测。邻近算术位宽可复用家族覆盖，不因梅森指数改变而必然再校准。条件忽略默认gpu、driver、cuda、backend、environment；算术、整除和显存检查始终保留。详见[模型与边界](../architecture/AUTO_B2.md#算术位宽区间预测)。

通用位宽网格可用`--tune ecm --tune-exponents 512,2048,4096,8192,12288,15360 --tune-arithmetic both --tune-b2 26000000000,260000000000 --tune-budget-seconds 1800`。它分别收集通用与梅森算术资料；真实队列默认仍用真实save配对。网格的梅森实际余因子位数可能低于算术位数，两者不会混为同一索引。

任务最终B2=0时，`--auto-b2`联合选择B2、D、承载。T1优先使用正的`--stage1-seconds-per-curve`，其次`--stage1-cost-csv FILE.csv`，最后`--stage1-tune-profile FILE.toml`。对应 TOML 为 `tune_stage1_seconds_per_curve`、`tune_stage1_cost_csv`、`tune_stage1_cost_file`。CSV明确target_bits或container_bits，缺项和跨TPI可估算；`--stage1-cost-mhz`给目标频率，0使用表内参考频率。秒/曲线不会再除batch。Stage1 标量构造默认跟随 TOML 的 `montgomery_lcm_multiplier`，可用 `--lcm 1|12` 覆盖。现有成本证据文件中的 `lcm`、`choose12` 标签由程序在读取时映射。

`--auto-min-b2`/`--auto-max-b2`控制搜索边界，默认B1<B2≤2.6×10¹²，不限于已测范围。完整tune优先于旧`tune_legacy_cost_profile`；Auto缺T1/完整成本或没有满足当前显存的候选时明确失败，保留队列。每条worker重新规划。模型精度与当前限制见[Auto B2/tune](../architecture/AUTO_B2.md)。

- [存档与队列字段](../../src/ecm/stage2/input.cpp) 由 `records` 和 `queue_fields` 读取；[共享队列驱动](../../src/ecm/stage2/queue.cpp) 负责恢复和发布。
- [Stage2 进度接口](../../include/ecm/io/stage2_checkpoint.h)与[存储实现](../../src/io/stage2_checkpoint.cpp)：原子进度与发布状态。
- [ecm_stage2_console.h](../../src/ecm/stage2/console.h)：阶段和内存摘要。
- [Stage2 管线](../architecture/STAGE2.md)、[Auto B2](../architecture/AUTO_B2.md)、[构建](BUILD.md)。
