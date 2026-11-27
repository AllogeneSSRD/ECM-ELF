# ecm.toml 配置参考

<!-- Generated from config/ecm_options.json. -->

[English](ECM_TOML_REFERENCE.en.md)

在首个表头前写必填的 `config_version = 4`。见[配置迁移](architecture/CONFIGURATION.md#升级旧配置)。

## 分区与优先级

- `[shared]` 存放两阶段共用的设置。
- `[stage1]` 和 `[stage2]` 各自存放本阶段的设置。
- `[gui]` 存放 GUI 全局设置，worker 无法覆盖。
- `[worker.N]` 存放第 N 个 worker 的设置，N 为 1–1000000。覆盖项使用与全局分区完全相同的键名。

程序依次应用内置默认值、全局分区、对应 worker 分区和命令行参数，后者优先。重复键或表、未知分区、缺失值及未闭合字符串会报错。`worktodo.txt` 使用独立的 `[Worker #N]` 分区。程序不读取 INI 文件。

worker 校验共享和本阶段设置。GUI 校验共享、GUI 和 worker 设置，并预览能识别的计算参数。见[配置读取职责](architecture/CONFIGURATION.md#按程序职责读取)。

共用 worktodo 与 Stage2 的相对路径基于 TOML 所在目录，其他 Stage1 路径通常基于可执行文件目录。具体规则见[运行时限制](architecture/CONFIGURATION.md#运行时限制)。

## [shared]

### computer

```toml
computer = ""
```

类型 `string` · 范围 `<local-computer-name>` · Worker 覆盖 `[worker.N] computer` · 级别 `常用`。

结果输出的可选本地计算机名；实际配置值及生成结果不得进入公共Git提交。

<a id="shared_cpu_affinity"></a>

### cpu_affinity

```toml
cpu_affinity = ""
```

类型 `string` · 范围 `[none|auto|<n>,...|<a>-<b>]` · Worker 覆盖 `[worker.N] cpu_affinity` · 级别 `高级`。

worker 进程使用的逻辑 CPU，包括 GPU 后端的主机工作。编号从 0 开始，范围含两端，例如 0,2,4-7。空值或 auto 让 CPU Stage2 每个物理核选一个逻辑 CPU，优先选性能核。显式列表中的不同物理核用完后，可以使用列表内的 SMT 兄弟线程。none 由系统调度，CPU Stage2 停用调优复用和采样。其他后端遇到空值、auto 或 none 时沿用已有亲和性。可按 worker 同时覆盖两阶段。

<a id="shared_finished_file"></a>

### finished_file

```toml
finished_file = "worktodo.finished.txt"
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `不支持` · 级别 `常用`。

两阶段共用的完成记录。相对路径基于配置目录。

<a id="shared_log_file"></a>

### log_file

```toml
log_file = "ecm.log"
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `不支持` · 级别 `常用`。

共用运行日志。相对路径基于配置目录。空字符串关闭文件日志。

### log_file_max_kb

```toml
log_file_max_kb = 10000
```

类型 `u64` · 范围 `[1000..1000000]` · Worker 覆盖 `不支持` · 级别 `常用`。

共享运行日志超过此上限后归档。1 KB 等于 1000 字节。

<a id="priority"></a>

### queue_order

```toml
queue_order = "file_order"
```

类型 `string` · 范围 `[file_order|stage1_first|stage2_first]` · Worker 覆盖 `[worker.N] queue_order` · 级别 `常用`。

file_order 按文件顺序选任务。stage1_first 和 stage2_first 优先执行对应阶段，直到该阶段没有剩余任务。

<a id="skip"></a>

### queue_skip

```toml
queue_skip = "none"
```

类型 `string` · 范围 `[none|stage1|stage2]` · Worker 覆盖 `[worker.N] queue_skip` · 级别 `常用`。

none 允许两阶段执行。stage1 和 stage2 将对应阶段的任务留在队列中。

<a id="shared_tune_file"></a>

### tune_file

```toml
tune_file = "tune.toml"
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] tune_file` · 级别 `常用`。

CUDA Stage1、CUDA Stage2 和 CPU GWNUM Stage2 共用此文件，默认使用 ecm.toml 同目录的 tune.toml。worker 可覆盖路径。显式留空时关闭自动持久化。

### user

```toml
user = ""
```

类型 `string` · 范围 `<local-user-id>` · Worker 覆盖 `[worker.N] user` · 级别 `常用`。

Stage1与Stage2结果的可选本地用户标识；真实账号值不得写入提交的模板或测试样本。

<a id="workflow_worker_count"></a>

### worker_count

```toml
worker_count = 1
```

类型 `int` · 范围 `<x:Z,1..64>` · Worker 覆盖 `不支持` · 级别 `高级`。

供 GUI v2 与同队列 ECM-ELF 自动交接共用的 worker 数量，编号从 1 到该数量。自动交接排除跳过 Stage2 的 worker。

<a id="worktodo"></a>

### worktodo_file

```toml
worktodo_file = "worktodo.txt"
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] worktodo_file` · 级别 `常用`。

共用队列。ECM、ECM2 和 ECMSTAGE1 交给 Stage1，ECMSTAGE2、ECMSTAGE2N 和 ECMSTAGE2PART 交给 Stage2。同一 worker 同时由一个进程消费队列。

## [stage1]

<a id="save_sync_dir_1"></a>

### backup_directory_1

```toml
backup_directory_1 = ""
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] backup_directory_1` · 级别 `高级`。

指定第一个 Stage1 存档同步目录。留空不启用该目标；同步范围和方式由 backup_mode 决定。 仅用于文件备份；此路径不作为 Prime95 任务发布依据。

<a id="save_sync_dir_2"></a>

### backup_directory_2

```toml
backup_directory_2 = ""
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] backup_directory_2` · 级别 `高级`。

指定第二个 Stage1 存档同步目录。可同时向两个目标同步；留空只关闭此目标，不影响第一个目标。 仅用于文件备份；此路径不作为 Prime95 任务发布依据。

<a id="sync_mode"></a>

### backup_mode

```toml
backup_mode = "incremental"
```

类型 `string` · 范围 `[incremental|full]` · Worker 覆盖 `[worker.N] backup_mode` · 级别 `高级`。

控制 Stage1 任务完成后的存档同步。incremental 仅同步本次任务期间更新的文件；full 同步全部符合条件的存档。程序启动时仍执行一次完整同步。

<a id="ckpt_backups"></a>

### checkpoint_backups

```toml
checkpoint_backups = 3
```

类型 `int` · 范围 `<x:Z,0..16>` · Worker 覆盖 `[worker.N] checkpoint_backups` · 级别 `高级`。

Stage1 检查点轮换备份的最大份数；主存档另计，0 表示只保留主存档。恢复时跳过损坏或无法读取的代次；Stage1 完成后清理主存档及所有备份。

<a id="ckpt_seconds"></a>

### checkpoint_interval_seconds

```toml
checkpoint_interval_seconds = 600
```

类型 `double` · 范围 `<x:R,finite,x>=0>` · Worker 覆盖 `[worker.N] checkpoint_interval_seconds` · 级别 `高级`。

设置 Stage1 定期写检查点的间隔，单位为秒。0 关闭定期检查点，负数按 0 处理；仍可通过正常停止或 Ctrl+C 请求保存。实际写盘时机受当前计算后端的安全检查点限制。

<a id="impl"></a>

### cpu_implementation

```toml
cpu_implementation = "auto"
```

类型 `string` · 范围 `[auto|simd|scalar|gwnum|slice list|empty]` · Worker 覆盖 `[worker.N] cpu_implementation` · 级别 `常用`。

选中 CPU 时使用的 Stage1 算术策略。单值使用对应后端，auto 或空字符串按 CPU 能力与批量自动选择，实际模数超过 10240 位时要求 AVX512-IFMA。支持 simd[:6144], scalar[:4096], gwnum[4096:], auto[] 这样的切片列表。位宽按剥除已知因子后的实际模数计算。允许区间重叠，依次选择首个满足位宽且可用的后端。切片含起点、不含终点，省略起点按 1 位处理，省略终点表示无上限，[] 等同于 [:]，忽略空白。可用 auto[] 显式回退到自动选择。simd 要求 AVX512-IFMA，scalar 使用 GMP，超过 10240 位时仅支持 Edwards。gwnum 要求 Montgomery PARAM0 和 Windows x64 worker 同目录的本地 gwnum.dll。GPU worker 可继承此 TOML 策略，运行时不启用。

<a id="cpu_mont_chain"></a>

### cpu_montgomery_chain

```toml
cpu_montgomery_chain = "ladder"
```

类型 `string` · 范围 `[ladder|prac]` · Worker 覆盖 `[worker.N] cpu_montgomery_chain` · 级别 `高级`。

Montgomery CPU 点乘。prac 要求 target=cpu、curve=mont、impl=simd 和 B1 在 2..4294967295 之间。它与 ladder 使用相同的标量约定及最终 save 格式，各自使用不同的运行中检查点状态。

### cpu_prac_search

```toml
cpu_prac_search = 7
```

类型 `int` · 范围 `[1..50]` · Worker 覆盖 `[worker.N] cpu_prac_search` · 级别 `高级`。

CPU IFMA PRAC 对每个素数尝试的种子数量。恢复 PRAC 检查点时应保持相同搜索宽度。

<a id="reduction"></a>

### cpu_simd_reduction

```toml
cpu_simd_reduction = "auto"
```

类型 `string` · 范围 `[auto|montgomery|mersenne-fold]` · Worker 覆盖 `[worker.N] cpu_simd_reduction` · 级别 `高级`。

CPU SIMD 模约减：auto 按输入数选择；mersenne-fold 要求 N=2^p-1；montgomery 使用通用约减。显式归约要求 impl=simd，避免 auto 选中标量时悄悄忽略设置。

### cuda_low_cpu

```toml
cuda_low_cpu = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] cuda_low_cpu` · 级别 `常用`。

设为 1 或 true 后，CUDA CGBN Stage1 主机线程在等待 GPU 完成时休眠，对所有切片深度优先生效。此模式用于降低 CPU 占用，唤醒可能稍慢。

### cuda_montgomery_chain

```toml
cuda_montgomery_chain = "ladder"
```

类型 `string` · 范围 `[ladder|prac]` · Worker 覆盖 `[worker.N] cuda_montgomery_chain` · 级别 `高级`。

为 CUDA CGBN Montgomery PARAM0 选择点乘链。其他曲线族与后端忽略此项。省略时允许使用旧环境变量 ECM_GPU_STAGE1_ALGO 覆盖。

### cuda_prac_search

```toml
cuda_prac_search = 7
```

类型 `int` · 范围 `[1..50]` · Worker 覆盖 `[worker.N] cuda_prac_search` · 级别 `高级`。

CUDA PRAC 对每个素数尝试的种子数量。恢复 PRAC 检查点时应保持相同搜索宽度。

### cuda_slice_depth

```toml
cuda_slice_depth = 3
```

类型 `int` · 范围 `[2|3|8]` · Worker 覆盖 `[worker.N] cuda_slice_depth` · 级别 `常用`。

CUDA CGBN Stage1 的切片提交方式。2 和 3 分别滚动提交至多两片和三片。8 按最多八片一组等待组尾。

### curve

```toml
curve = ""
```

类型 `string` · 范围 `[mont|ed|empty]` · Worker 覆盖 `[worker.N] curve` · 级别 `高级`。

param 为 auto 时可用此项指定曲线。mont 为 Montgomery，ed 为 Edwards。省略或留空时按运行模式取默认值。桌面 OpenCL 支持 1024 bit 以内的 Edwards 和 Montgomery PARAM3。

### edwards_doubling

```toml
edwards_doubling = 1
```

类型 `int` · 范围 `[0|1]` · Worker 覆盖 `[worker.N] edwards_doubling` · 级别 `高级`。

新 CUDA CGBN 与 OpenCL Edwards NAF 批次的倍点公式。0 选择 HWCD，1 选择 2XY（默认）。续跑沿用检查点中的公式。省略此键时，CUDA CGBN 还支持 ECM_EDWARDS_DBL_ALGO。

<a id="naf_w"></a>

### edwards_naf_window_bits

```toml
edwards_naf_window_bits = 0
```

类型 `int` · 范围 `[0|2..12]; CPU: [0|3..12]` · Worker 覆盖 `[worker.N] edwards_naf_window_bits` · 级别 `高级`。

Edwards NAF 请求窗口。0 在 CPU 从 12 开始，在 CUDA CGBN 和 OpenCL 从 6 开始。CUDA NTT 中 0 按兼容实测吞吐选择窗口，无记录可用时回退 x-only，2～12 固定 2XY NAF 窗口。CPU、CGBN 和 OpenCL 可根据内存预算为新任务自动降窗。

<a id="exp_cache"></a>

### exponent_cache_dir

```toml
exponent_cache_dir = ""
```

类型 `string` · 范围 `[off|none|0|-|<dir>]` · Worker 覆盖 `[worker.N] exponent_cache_dir` · 级别 `高级`。

指定 Stage1 指数及 PRAC 计划缓存目录。留空使用仓库根目录的 cache/，独立部署使用可执行文件同级的 cache/。off、none、0 或 - 关闭缓存。显式相对路径以进程当前工作目录为基准。较小的 B1 可能不写指数缓存。

### gpu_algorithm

```toml
gpu_algorithm = "ntt[16384:], cgbn[]"
```

类型 `int|string` · 范围 `[1|2|3|cgbn|ntt|thread|slice list]` · Worker 覆盖 `[worker.N] gpu_algorithm` · 级别 `高级`。

Stage1 GPU 算法，不区分大小写。单值 1/cgbn、2/ntt 和 3/thread 显式指定算法。切片列表按实际模数位宽从左到右尝试可用的算法，含下界、不含上界，空括号匹配任意位宽。CUDA NTT 要求完整梅森模数，支持 Edwards 和 Montgomery PARAM0。CUDA thread 支持最高 1024 bit 的 Edwards。OpenCL worker 的 cgbn 规则使用其原有 OpenCL 算法。

<a id="stage1_gpu_init_threads"></a>

### gpu_init_threads

```toml
gpu_init_threads = 0
```

类型 `u32` · 范围 `<x:Z,0..2^32-1>` · Worker 覆盖 `[worker.N] gpu_init_threads` · 级别 `高级`。

Edwards CUDA CGBN、CUDA NTT 和 CUDA/OpenCL 一线程一曲线 Stage1 的主机构造线程数。0 由后端自动选择。实际线程受可用 CPU、曲线数和适用的内存预算限制。命令行 --gpu-init-threads 可覆盖。

### handoff_mode

```toml
handoff_mode = "ecm-elf"
```

类型 `string` · 范围 `[off|ecm-elf|prime95]` · Worker 覆盖 `[worker.N] handoff_mode` · 级别 `常用`。

Stage1 无因子完成且有可用保存点时，默认在原 worker 队列末尾追加 ECM-ELF Stage2 任务。可配置其他接收方。off 只保留存档，prime95 交给 Prime95。

### handoff_partial

```toml
handoff_partial = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] handoff_partial` · 级别 `高级`。

Stage1 继续运行时，将已完成的批次交接给 ECM-ELF Stage2。

### handoff_unsupported

```toml
handoff_unsupported = "ecm-elf"
```

类型 `string` · 范围 `[ecm-elf|keep-save]` · Worker 覆盖 `[worker.N] handoff_unsupported` · 级别 `高级`。

Prime95 协议不支持该存档时使用的后备方式。

<a id="auto_stage2_workers"></a>

### handoff_workers

```toml
handoff_workers = ""
```

类型 `string` · 范围 `[<n>[slice]|<a>-<b>[slice]|comma list|auto]` · Worker 覆盖 `[worker.N] handoff_workers` · 级别 `高级`。

Stage2 接收 worker，可填 1..1000000 内的编号或闭区间。逗号分隔的每项可加位宽切片，如 2[:4096], 3[4096:]。按实际余因子位宽匹配，包含起点、排除终点。省略方括号、[]、[:] 都匹配任意位宽。选匹配候选中任务最少者，并列取较小编号。无匹配时保留源任务。同队列 auto 使用 worker_count 或已有分区，候选须读取该队列并允许 Stage2。外部队列 auto 优先采用已有分区。留空保留当前 worker。目标不可写时保留待投递记录。

<a id="auto_stage2_worktodo"></a>

### handoff_worktodo_file

```toml
handoff_worktodo_file = ""
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] handoff_worktodo_file` · 级别 `高级`。

auto_stage2=1 时的目标队列。留空使用当前 worker 生效的 shared worktodo_file；可填写绝对文件路径，或相对 ecm.toml 所在目录的文件名/路径。投递行中的 save 使用绝对路径。完成去重读取本 TOML 中接收 worker 的 stage2_finished；worker 分区可覆盖。

<a id="lcm"></a>

### montgomery_lcm_multiplier

```toml
montgomery_lcm_multiplier = "1"
```

类型 `string` · 范围 `[1|12]` · Worker 覆盖 `[worker.N] montgomery_lcm_multiplier` · 级别 `高级`。

Montgomery Stage1 标量是 montgomery_lcm_multiplier(1..B1) 乘 1 或 12。Edwards 固定使用 48·montgomery_lcm_multiplier，不接受 12。此选择改变输出点，须与 Stage2 存档约定一致。

### naf_memory_mb

```toml
naf_memory_mb = 0
```

类型 `int` · 范围 `[0|1..1048576]` · Worker 覆盖 `[worker.N] naf_memory_mb` · 级别 `高级`。

Edwards Stage1 内存上限，1 MB = 1,000,000 字节。约束 CPU NAF 工作集、CUDA CGBN 或 OpenCL 设备缓冲、CUDA NTT 字典及设备乘积树。0 使用后端预算。OpenCL 以报告的全局内存一半为预算，并检查单个缓冲的最大容量。

### ntt_allow_known_factors

```toml
ntt_allow_known_factors = 1
```

类型 `int` · 范围 `[0|1]` · Worker 覆盖 `[worker.N] ntt_allow_known_factors` · 级别 `高级`。

1 允许带已知因子的梅森任务使用完整 2^p-1 进行 NTT 运算。曲线构造、因子检查和发布存档使用剩余余因子。0 禁用此承载路径。算法切片仍按余因子的位宽匹配。

### ntt_batch_target_percent

```toml
ntt_batch_target_percent = 95
```

类型 `int` · 范围 `[1,100]` · Worker 覆盖 `[worker.N] ntt_batch_target_percent` · 级别 `高级`。

CUDA Stage1 NTT 选批的吞吐门槛，以可行峰值的百分比表示。请求批量达标时优先整批执行。需要拆批时，选择达到可行子批峰值该比例的最小实测批量。

<a id="ntt_bpw"></a>

### ntt_bits_per_word

```toml
ntt_bits_per_word = 0
```

类型 `int` · 范围 `0 or [16,31]` · Worker 覆盖 `[worker.N] ntt_bits_per_word` · 级别 `高级`。

每个数字的位数。0 选择最宽精确线性表示或最短安全加权变换。加权模式下，显式值限制最大 digit 位宽。

### ntt_convolution

```toml
ntt_convolution = "auto"
```

类型 `string` · 范围 `[auto|linear|weighted]` · Worker 覆盖 `[worker.N] ntt_convolution` · 级别 `高级`。

auto 按测量选型，缺失时短调优；也可显式选择精确线性卷积或加权循环卷积（IBDWT）。两者都使用 Goldilocks，共用 Stage1 任务及检查点格式。

### ntt_length

```toml
ntt_length = 0
```

类型 `int` · 范围 `0, 2^k in [2,524288], 3*2^k in [768,3072] (weighted), or 5*2^k in [2560,163840] (weighted)` · Worker 覆盖 `[worker.N] ntt_length` · 级别 `高级`。

变换长度。0 推导安全长度。加权卷积支持所列 radix-3 和 radix-5 长度。

### ntt_max_batch

```toml
ntt_max_batch = 512
```

类型 `int` · 范围 `[1,512]` · Worker 覆盖 `[worker.N] ntt_max_batch` · 级别 `高级`。

CUDA Stage1 NTT 调优和生产每个新批次的曲线数上限。各变换长度和 NAF 窗口的实际容量还受内存预算限制。

### ntt_memory_mib

```toml
ntt_memory_mib = 0
```

类型 `int` · 范围 `[0,1048576]` · Worker 覆盖 `[worker.N] ntt_memory_mib` · 级别 `高级`。

Stage1 NTT 总显存预算（MiB）。0 使用 CUDA 初始化后空闲显存的 80%；显式预算也受空闲显存限制。

### ntt_naf_gpu_min_bits

```toml
ntt_naf_gpu_min_bits = "auto"
```

类型 `int|string` · 范围 `[auto|0..5242881]` · Worker 覆盖 `[worker.N] ntt_naf_gpu_min_bits` · 级别 `高级`。

CUDA NTT 生产和调优使用 GPU 构造 Edwards NAF 字典的最小梅森指数。auto 或 0 使用内置门槛，正整数表示指数达到该值时优先 GPU，仍受内存预算限制。5242881 表示所有支持的指数都用 GMP。调优按一条曲线准备字典。

<a id="ntt_outer_max"></a>

### ntt_outer_radix_limit

```toml
ntt_outer_radix_limit = 0
```

类型 `int` · 范围 `[0,4]` · Worker 覆盖 `[worker.N] ntt_outer_radix_limit` · 级别 `高级`。

每个外部 pass 最多融合的 radix-2 层数。0 使用所选配置。

### ntt_pad_batch

```toml
ntt_pad_batch = 0
```

类型 `int` · 范围 `[0|1]` · Worker 覆盖 `[worker.N] ntt_pad_batch` · 级别 `高级`。

0 按任务请求的曲线数运行 Stage1 NTT。1 在资源允许时用真实曲线把尾批补齐到优选批量。新增曲线使用独立 sigma，计入结果和符合条件的 Stage2 交接。

### ntt_profile

```toml
ntt_profile = 0
```

类型 `int` · 范围 `[0|1|2]` · Worker 覆盖 `[worker.N] ntt_profile` · 级别 `高级`。

按长度选择的变换配置：0 自动，1 大 tile（至多 12 层），2 分块 tile（至多 10 层）。显式 tile 和外层设置优先。

### ntt_tight_weighted_bound

```toml
ntt_tight_weighted_bound = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] ntt_tight_weighted_bound` · 级别 `高级`。

CUDA Stage1 的原生 Edwards NAF 使用更紧的无符号系数界。自动选型将通过该界的短加权计划与其他合法计划一同实测。false 使用保守界。

### ntt_tile

```toml
ntt_tile = 0
```

类型 `int` · 范围 `[0,12]` · Worker 覆盖 `[worker.N] ntt_tile` · 级别 `高级`。

融合 tile 的 radix-2 层数。0 使用所选配置。

### ntt_warp_tail

```toml
ntt_warp_tail = "auto"
```

类型 `bool|string` · 范围 `[auto|false|true|0|1]` · Worker 覆盖 `[worker.N] ntt_warp_tail` · 级别 `高级`。

用 warp shuffle 处理 tile 变换的局部尾部。auto 使用所选配置。

<a id="ntt_window"></a>

### ntt_window_bits

```toml
ntt_window_bits = -1
```

类型 `int` · 范围 `[-1,64]` · Worker 覆盖 `[worker.N] ntt_window_bits` · 级别 `高级`。

每次 CUDA Graph 重放处理的阶梯位数。-1 使用所选配置，0 使用直接启动。

<a id="kernel_add"></a>

### opencl_kernel_add

```toml
opencl_kernel_add = ""
```

类型 `string` · 范围 `[auto|<id>|<alias>]` · Worker 覆盖 `[worker.N] opencl_kernel_add` · 级别 `高级`。

指定 OpenCL 模加内核的标识或别名。空值或 auto 由后端选择；通常保持自动，CUDA 后端忽略此键。

<a id="kernel_mul"></a>

### opencl_kernel_mul

```toml
opencl_kernel_mul = ""
```

类型 `string` · 范围 `[auto|<id>|<alias>]` · Worker 覆盖 `[worker.N] opencl_kernel_mul` · 级别 `高级`。

选择 OpenCL 模乘。PARAM3 接受注册的内核 ID 和别名。Edwards 默认 cios，最高 512 bit 可选 fips 或 NVIDIA 专用的 fips_ptx。CUDA 忽略此键。

<a id="kernel_special_mult"></a>

### opencl_kernel_special_mult

```toml
opencl_kernel_special_mult = ""
```

类型 `string` · 范围 `[auto|<id>|<alias>]` · Worker 覆盖 `[worker.N] opencl_kernel_special_mult` · 级别 `高级`。

指定 OpenCL 特殊乘法内核的标识或别名。空值或 auto 由后端选择；通常保持自动，CUDA 后端忽略此键。

<a id="kernel_sqr"></a>

### opencl_kernel_sqr

```toml
opencl_kernel_sqr = ""
```

类型 `string` · 范围 `[auto|<id>|<alias>]` · Worker 覆盖 `[worker.N] opencl_kernel_sqr` · 级别 `高级`。

选择 OpenCL 模平方。PARAM3 接受注册的内核 ID 和别名。Edwards 的 auto 复用所选模乘，cios 强制使用 CIOS，fips 和 NVIDIA 专用的 fips_ptx 在 512 bit 以内使用对称 FIPS。一个 Edwards 任务使用同一种 FIPS 实现。CUDA 忽略此键。

<a id="kernel_sub"></a>

### opencl_kernel_sub

```toml
opencl_kernel_sub = ""
```

类型 `string` · 范围 `[auto|<id>|<alias>]` · Worker 覆盖 `[worker.N] opencl_kernel_sub` · 级别 `高级`。

指定 OpenCL 模减内核的标识或别名。空值或 auto 由后端选择；通常保持自动，CUDA 后端忽略此键。

<a id="tpi"></a>

### opencl_threads_per_integer

```toml
opencl_threads_per_integer = 8
```

类型 `u32` · 范围 `[1|2|4|8|16|32]` · Worker 覆盖 `[worker.N] opencl_threads_per_integer` · 级别 `高级`。

设置 OpenCL 每个大整数使用的线程数，需满足有效 TPI 与整数 limb 数的布局约束。CUDA 后端自行按位宽选择 TPI，不使用此键；不要用它调整 CUDA 的线程块大小。

<a id="wg_size"></a>

### opencl_workgroup_size

```toml
opencl_workgroup_size = 0
```

类型 `int` · 范围 `<x:Z,x>=0>` · Worker 覆盖 `[worker.N] opencl_workgroup_size` · 级别 `高级`。

设置 OpenCL 工作组大小，0 由后端自动选择。显式值必须满足当前设备和内核的限制；CUDA 后端不使用此设置。

### param

```toml
param = "auto"
```

类型 `string` · 范围 `[auto|e|m0|m2|m3]` · Worker 覆盖 `[worker.N] param` · 级别 `常用`。

Stage1 曲线构造。auto 时，队列任务默认用 Edwards，直接输入在 CPU 上默认用 m0，在 GPU 上默认用 m3。curve 可在 auto 时指定曲线。ECMSTAGE1 的存档类型前缀优先。

<a id="p95_keep_aid"></a>

### prime95_keep_aid

```toml
prime95_keep_aid = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] prime95_keep_aid` · 级别 `高级`。

控制交接任务是否保留 PrimeNet assignment ID。1 保留未被判定失效的 AID，0 一律去掉；已在 Prime95 日志中确认失效的 AID 会去掉，以免任务再次被丢弃。

<a id="p95_recover_lost"></a>

### prime95_recover_lost

```toml
prime95_recover_lost = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] prime95_recover_lost` · 级别 `高级`。

控制是否补投被 Prime95 因失效 AID 删除的交接任务。1 根据交接记录和 Prime95 日志检查并去掉 AID 补投，0 关闭自动补投；只对程序能够确认的已交接任务生效。

<a id="p95_sync_dir"></a>

### prime95_save_dir

```toml
prime95_save_dir = ""
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] prime95_save_dir` · 级别 `高级`。

Prime95 交接专用存档目录，启用 prime95_worktodo_file 时必须设置。备份目录独立；已发布存档前缀保持不变，只允许追加完整记录。

<a id="p95_add_workers"></a>

### prime95_workers

```toml
prime95_workers = ""
```

类型 `string` · 范围 `[<n>|<n>,...|<a>-<b>|auto]` · Worker 覆盖 `[worker.N] prime95_workers` · 级别 `高级`。

选择接收交接任务的 Prime95 worker，可指定单个编号、编号列表或区间；程序在候选 worker 中选择队列较短者。auto 从同目录 prime.txt 读取 NumWorkers；留空追加时不写 worker 段头，按 Prime95 的默认 worker 处理。

<a id="p95_worktodo_path"></a>

### prime95_worktodo_file

```toml
prime95_worktodo_file = ""
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] prime95_worktodo_file` · 级别 `高级`。

用于受支持的 Montgomery Stage1 交接的 Prime95 worktodo_file 路径。Edwards 与 GPU PARAM2 自动忽略此路径，存档保留在本地，不访问 Prime95 转移文件。 需同时配置 prime95_save_dir；先验证并发布存档，再写入 worktodo_file.add。

### progress_average_seconds

```toml
progress_average_seconds = 0
```

类型 `double` · 范围 `[0..60]` · Worker 覆盖 `[worker.N] progress_average_seconds` · 级别 `高级`。

Stage1 CPU/CUDA 实时 s/curve 与剩余时间估计的平均窗口，单位为秒。0 使用最近一次有效采样，不做平均；正值按采样时长累计已完成工作量，最多 60 秒。CUDA 采样排除准备和 checkpoint 写盘；不改变计算、切片、最终计时或日志刷新频率。

### progress_color

```toml
progress_color = "cyan"
```

类型 `string` · 范围 `[none|red|green|yellow|blue|magenta|cyan|white|grey]` · Worker 覆盖 `[worker.N] progress_color` · 级别 `高级`。

选择 Stage1 控制台进度文字颜色。none 关闭着色；此设置不改变普通文件日志内容或计算方式。

### progress_log_seconds

```toml
progress_log_seconds = 60
```

类型 `double` · 范围 `<x:R,finite>` · Worker 覆盖 `[worker.N] progress_log_seconds` · 级别 `高级`。

控制 Stage1 进度行写入日志文件的间隔，单位为秒。正数按间隔记录，0 不记录中间进度，负数记录每次进度更新；完成时的 100% 行仍会记录。此键不限制控制台刷新频率。

<a id="stage1_random_sigma"></a>

### random_sigma

```toml
random_sigma = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] random_sigma` · 级别 `常用`。

未指定 σ 时，支持的路径逐曲线随机取值。false 时随机抽取一个起点，之后每条曲线递增 1。GPU Montgomery 批次本身就使用连续 σ。显式 σ 优先。

### save_name_pattern

```toml
save_name_pattern = "{curve}_{target}_{b1}.save"
```

类型 `string` · 范围 `<pattern>` · Worker 覆盖 `[worker.N] save_name_pattern` · 级别 `高级`。

统一设置 CPU/GPU Stage1 自动 save 文件名。{curve}=e|m0|m2|m3；{target}=梅森任务的 m<指数>，否则为 bit<实际位数>；{n}=原任务指数，无任务元数据时为实际位数；{b1}=紧凑 B1。缺少曲线前缀时自动补充。ECMSTAGE1 队列中，类型前缀覆盖 TOML 曲线/param，末尾文件名字段给出 B1；无类型前缀时沿用 TOML。完成/交接行使用实际带前缀的 save；CLI 显式 save 路径保持原样。留空恢复默认。

### sigma

```toml
sigma = 0
```

类型 `u64` · 范围 `<x:Z,0..2^64-1>` · Worker 覆盖 `[worker.N] sigma` · 级别 `高级`。

曲线起始 sigma；0 表示随机生成，连续曲线使用 sigma+i 且不得在 64 位处回绕。CUDA PARAM0 接受完整 64 位范围，PARAM2/3 批次须位于 2^32 以下。CPU 路径的 sigma 超出 Prime95 有符号读取范围时会警告，这不是 Stage1 的限制。ECM/ECM2 任务行的非零 sigma 覆盖此键。

<a id="stage1_cpu_compute_threads"></a>

### stage1_cpu_threads

```toml
stage1_cpu_threads = 0
```

类型 `u32` · 范围 `<x:Z,0..2^32-1>` · Worker 覆盖 `[worker.N] stage1_cpu_threads` · 级别 `高级`。

CPU Stage1 算术与准备线程数。0 由后端自动选择。命令行 --cpu-threads 可覆盖。

<a id="stage1_backend"></a>

### stage1_gpu_backend

```toml
stage1_gpu_backend = "auto"
```

类型 `string` · 范围 `[auto|cuda|opencl]` · Worker 覆盖 `[worker.N] stage1_gpu_backend` · 级别 `常用`。

Windows GPU 工作程序。auto 优先同目录的 CUDA 程序及所选编号可见的 CUDA 设备，其次 OpenCL。显式选择要求安装对应程序。设备编号按选中后端解释。--gpu-backend 优先。

### stage1_gpu_device

```toml
stage1_gpu_device = 0
```

类型 `int` · 范围 `<x:Z,0..device_count-1>` · Worker 覆盖 `[worker.N] stage1_gpu_device` · 级别 `常用`。

选择 GPU 设备编号，从 0 开始。用于 Stage1；Stage2 未显式指定 stage2_device 时也继承此编号。编号对应各可执行文件实际使用的 GPU 后端设备列表。

### stage1_results_file

```toml
stage1_results_file = "stage1_results.jsonl"
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] stage1_results_file` · 级别 `常用`。

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

统一格式的逐曲线 Stage1 内部 JSONL，包含 CRC32 与执行元数据，与提交文件分开。

<a id="tmp_dir"></a>

### stage1_save_dir

```toml
stage1_save_dir = "."
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] stage1_save_dir` · 级别 `常用`。

指定 Stage1 曲线存档的输出目录。Stage2 未设置 stage2_save_dir 时也在此目录查找任务引用的 save；此键不是 Stage2 全部临时显存或内存的预算。

### stage1_submission_file

```toml
stage1_submission_file = "results.json.txt"
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] stage1_submission_file` · 级别 `常用`。

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

Stage1 提交 JSON：整条任务完成后合并 NF（B2=B1）；验证后的因子立即单曲线输出。私有账本位于输出目录的 data/ 下。

### target

```toml
target = "gpu"
```

类型 `string` · 范围 `[cpu|gpu]` · Worker 覆盖 `[worker.N] target` · 级别 `常用`。

选择 Stage1 在 CPU 或 GPU 执行。stage1_gpu_backend 选择 CUDA 或桌面 OpenCL。两者均在各自位宽范围内支持 Edwards。

<a id="cuda_thread_block_size"></a>

### thread_block_size

```toml
thread_block_size = 128
```

类型 `int` · 范围 `[32|64|128|256]` · Worker 覆盖 `[worker.N] thread_block_size` · 级别 `高级`。

仅用于 CUDA Edwards thread 的线程块大小，可选 32、64、128 或 256。CLI --wg 可覆盖，--wg 0 选择 128。

### verbose

```toml
verbose = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] verbose` · 级别 `常用`。

控制普通控制台输出的详细程度。Stage2 中 true 显示阶段切换，false 保留曲线摘要；调试日志由 debug_log 独立控制。Stage1 按各计算后端的普通详细输出规则处理。

## [stage2]

<a id="stage2_arena_mb"></a>

### arena_mib

```toml
arena_mib = 0
```

类型 `u64` · 范围 `<x:Z,0..2^20>` · Worker 覆盖 `[worker.N] arena_mib` · 级别 `高级`。

设置 Stage2 大工作区预算，单位为 MiB，用于规划可用的 NTT 工作形状。0 由运行环境或自动规划决定，不表示不使用显存；过小可能迫使较小工作形状或使任务无法运行。表、坐标和其他模块仍可能另占显存。

<a id="stage2_auto_b2"></a>

### auto_b2

```toml
auto_b2 = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] auto_b2` · 级别 `常用`。

所有来源均为零 B2 时自动选择。1 按计入 Stage1 成本的全流程单位时间收益选取；0 关闭。须有可用 Stage2 成本与正的 Stage1 秒/曲线。B1 不作为 tune 复用门槛；其他条件由 tune_ignore 控制。当前输入仍检查显存及承载整除关系；估算保留不确定性标记，旧 .cprof 接口保持原资格检查。

<a id="stage2_auto_carrier"></a>

### auto_carrier

```toml
auto_carrier = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] auto_carrier` · 级别 `高级`。

对有效ECMSTAGE2=1,2,p,-1保存点，验证N整除M=2^p-1且2<=p<=16384后加入承载候选。普通成本按target_bits、梅森成本按arithmetic_bits=p查询，并遵循prediction_policy；完整M_p已使用原生梅森算术，不重复加入相同候选。显式承载选项优先；余因子save须匹配任务才能推断p。

<a id="stage2_auto_max_b2"></a>

### auto_max_b2

```toml
auto_max_b2 = 0
```

类型 `u64` · 范围 `<x:Z,0..2^63-8193>` · Worker 覆盖 `[worker.N] auto_max_b2` · 级别 `高级`。

Auto B2 搜索上界；0 为 2.6e12。未测值允许估算，显式任务 B2 优先。

<a id="stage2_auto_min_b2"></a>

### auto_min_b2

```toml
auto_min_b2 = 0
```

类型 `u64` · 范围 `<x:Z,0..2^63-8193>` · Worker 覆盖 `[worker.N] auto_min_b2` · 级别 `高级`。

Auto B2 搜索下界；0 为 B1+1。未测值允许估算，显式任务 B2 优先。

<a id="stage2_b2"></a>

### b2

```toml
b2 = 0
```

类型 `u64` · 范围 `<x:Z,x=0|B1<x<=2^63-1-8192>` · Worker 覆盖 `[worker.N] b2` · 级别 `常用`。

设置未在命令行或任务行给出非零 B2 时使用的 Stage2 上界。0 表示继续从其他来源选择；显式 B2 必须大于存档的 B1。增大 B2 通常增加搜索范围、运行时间及显存需求；命令行非零 B2 优先于任务行，任务行优先于此键。

### b2_ratio

```toml
b2_ratio = 0
```

类型 `u64` · 范围 `[0|1..1000000]` · Worker 覆盖 `[worker.N] b2_ratio` · 级别 `常用`。

命令行、任务行和 b2 配置均未提供非零 B2 时，非零比例使用存档 B1 × ratio 作为上界。适用于队列及 GPU 直接读档。乘积超出范围会报错。

<a id="stage2_b2_round_chain"></a>

### b2_round_chain

```toml
b2_round_chain = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] b2_round_chain` · 级别 `高级`。

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

按候选 D 向上凑整 B2。CPU gwpoly 在确定高位后缀后补满已有的最后一个 P 点 G 块；CUDA 还要求尾点块能走链。调优按实际 B2 采样，日志与内部结果记录请求和实际界限。CPU gwpoly ABI 4 和 CUDA 完整区间默认开启，设为 false 保留请求界限。CPU libecm、旧 DLL、CUDA relocation 和标定默认使用固定界限。整数、内存及启用的 AutoB2 上限仍生效。CUDA 启动探测使用固定界限，显式 ECM 标定要求关闭凑整。CLI --b2-round-chain 覆盖 worker 配置。

<a id="stage2_baby_backend"></a>

### baby_backend

```toml
baby_backend = "coop32"
```

类型 `string` · 范围 `[auto|scalar|coop32]` · Worker 覆盖 `[worker.N] baby_backend` · 级别 `高级`。

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

CUDA 默认在实际算术位宽 8193..16384 bits 且设备 baby 生成可用时使用 coop32，其他形状使用标量。显式 auto 从标量起步，依据通过校验的配对 tune 结果选择。CLI --baby-backend 覆盖 TOML。省略此键时优先使用显式 NTT_BABY_COOP。旧成本配置默认使用标量，OpenCL 使用标量。

<a id="stage2_progress_dir"></a>

### checkpoint_dir

```toml
checkpoint_dir = ""
```

类型 `string` · 范围 `<dir>` · Worker 覆盖 `[worker.N] checkpoint_dir` · 级别 `高级`。

直接读档 Stage2 断点的目录。空值使用已配置进度路径的父目录或队列目录。队列任务共享存档旁的断点，并保留每条已完成曲线的实际 B2。

<a id="stage2_progress_file"></a>

### checkpoint_legacy_file

```toml
checkpoint_legacy_file = ""
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] checkpoint_legacy_file` · 级别 `旧版`。

旧进度文件路径。checkpoint_dir 为空时，直接读档断点使用该路径的父目录。队列共享断点位于存档旁。

<a id="stage2_cpu_backend"></a>

### cpu_backend

```toml
cpu_backend = "auto"
```

类型 `string` · 范围 `[auto|gwpoly|ntt|plain]` · Worker 覆盖 `[worker.N] cpu_backend` · 级别 `高级`。

Edwards CPU 续算的算术后端。auto 优先使用 GWNUM/polymult，其次是 libecm。命令行 --backend 可覆盖。

<a id="stage2_cpu_memory_mib"></a>

### cpu_memory_mib

```toml
cpu_memory_mib = 512
```

类型 `u64` · 范围 `[1,1048576]` · Worker 覆盖 `[worker.N] cpu_memory_mib` · 级别 `高级`。

CPU Stage2 算术内存预算，单位为 MiB。命令行 --memory-mib 可覆盖。

<a id="stage2_cpu_tune_candidates"></a>

### cpu_tune_candidates

```toml
cpu_tune_candidates = 5
```

类型 `u64` · 范围 `[1,16]` · Worker 覆盖 `[worker.N] cpu_tune_candidates` · 级别 `高级`。

一个 GWNUM Stage2 条件最多比较的计划数，包含默认计划、备选 D 和 F 树布局。1 只保留默认计划。ABI 4 最多提供三个 D。命令行 --cpu-tune-candidates 可覆盖。

<a id="stage2_cpu_tune_dense_d"></a>

### cpu_tune_dense_d

```toml
cpu_tune_dense_d = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] cpu_tune_dense_d` · 级别 `高级`。

用 ABI 5 调优 GWNUM Stage2 时枚举更细的 D 档位。关闭后使用原有轮候选。命令行 --cpu-tune-dense-d 0|1 可覆盖。

<a id="stage2_parallel"></a>

### curve_concurrency

```toml
curve_concurrency = "serial"
```

类型 `string` · 范围 `[auto|serial|dual]` · Worker 覆盖 `[worker.N] curve_concurrency` · 级别 `高级`。

CUDA Stage2 曲线调度。auto 实测串行和可准入的双槽模式。默认 serial 串行执行，dual 请求双槽并检查显存准入。命令行仍使用 --parallel auto|0|1|2。

<a id="stage2_d"></a>

### d

```toml
d = 0
```

类型 `u64` · 范围 `<x:Z,x=0|x>=6,x%2=0>` · Worker 覆盖 `[worker.N] d` · 级别 `高级`。

手动指定 Stage2 算法步长 D。0 自动选形；非零值除满足偶数且不小于 6 外，还必须满足当前引擎的布局限制。D 会改变树规模、批次数和内存需求，通常保持自动。

<a id="stage2_debug_log"></a>

### debug_log

```toml
debug_log = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] debug_log` · 级别 `高级`。

控制是否将 Stage2 调试细节写入独立日志。true 开启，false 关闭；verbose=true 不会自动开启调试日志，开启本选项也不把全部调试输出刷到控制台。

<a id="stage2_debug_log_file"></a>

### debug_log_file

```toml
debug_log_file = ""
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] debug_log_file` · 级别 `高级`。

指定 Stage2 独立调试日志文件，仅在 debug_log 开启时写入。留空恢复默认文件名；默认名称按 worker 加后缀，显式指定时按原路径使用。

<a id="stage2_factorize_hits"></a>

### decompose_factors

```toml
decompose_factors = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] decompose_factors` · 级别 `高级`。

控制命中因子的附加分解。1 使用共享诊断层，优先本地 PARI DLL，备选外部 GP，不自动使用 GMP 算法；0 只保留 Stage2 已得到并验证的因子。仅完整且全部已证素的分解标为完成；诊断未完成仍保留原始因子与曲线成功状态。

<a id="stage2_execution_device"></a>

### device

```toml
device = "auto"
```

类型 `string` · 范围 `[auto|cpu|gpu]` · Worker 覆盖 `[worker.N] device` · 级别 `常用`。

统一 Stage2 程序的执行设备。auto 按 stage2_algorithm 选型，cpu 和 gpu 固定设备。CPU 支持 Edwards 存档。可用 --target 覆盖。

<a id="stage2_factor_timeout"></a>

### factor_timeout_seconds

```toml
factor_timeout_seconds = 30
```

类型 `u64` · 范围 `<x:Z,1..600>` · Worker 覆盖 `[worker.N] factor_timeout_seconds` · 级别 `高级`。

设置备选外部 GP 分解调用的超时秒数。本地 PARI 单次进程限 5 秒、单曲线 worker 会话累计限 10 秒；显式选择 GMP 时单次限 2 秒。附加诊断不改变原始因子或 Stage2 成功状态。

<a id="stage2_fold_mb"></a>

### fold_resident_mib

```toml
fold_resident_mib = 640
```

类型 `u64` · 范围 `<x:Z,0..2^20>` · Worker 覆盖 `[worker.N] fold_resident_mib` · 级别 `高级`。

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

设置 fold 累积结果驻留 GPU 的预算，单位为 MiB。0 关闭这部分驻留；所需空间超过预算时可退回分批路径，可能显著增加处理和传输开销。此预算与大工作区、批次缓冲分别控制，不是总显存限制。

<a id="stage2_giant_backend"></a>

### giant_backend

```toml
giant_backend = "coop32"
```

类型 `string` · 范围 `[auto|scalar|coop16|coop32]` · Worker 覆盖 `[worker.N] giant_backend` · 级别 `高级`。

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

CUDA 默认在实际算术位宽 8193..16384 bits 且可走递推链时使用 coop32。coop16 使用固定 16384-bit 容器，coop32 按位宽分档。显式 auto 从标量起步，依据通过校验的配对 tune 结果选择。CLI --giant-backend 覆盖 TOML。省略此键时优先使用显式 NTT_GIANT_COOP。旧成本配置默认使用标量，OpenCL 使用标量。

<a id="stage2_giant_chain"></a>

### giant_chain

```toml
giant_chain = "64"
```

类型 `string` · 范围 `[64|32|auto]` · Worker 覆盖 `[worker.N] giant_chain` · 级别 `高级`。

每条CUDA线程链连续生成的点数，与CUDA线程块大小无关。64保留基线，32显式使用短链；auto在满足tune_min_repeats门槛的准入实测支持32前保留64，并进行有界正常曲线探测。各策略具有独立tune身份。命中或非单位时按旧64点链重放完整区间。CLI --giant-chain覆盖当前worker配置。

<a id="stage2_giant_tail_balance"></a>

### giant_tail_balance

```toml
giant_tail_balance = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] giant_tail_balance` · 级别 `高级`。

允许完整giant区间的末两点块借整P批次，使两块均可走chain；不改变B2、容量、覆盖及块数。默认false。未实际借块时沿用原成本及缓存身份，不增加重放或同步；实际借块独立记录tune，命中或非单位时重放原区间，重放可能需要更大工作区。CLI --giant-tail-balance覆盖当前worker配置。

<a id="stage2_gp"></a>

### gp

```toml
gp = ""
```

类型 `string` · 范围 `<exe>` · Worker 覆盖 `[worker.N] gp` · 级别 `高级`。

指定附加因子分解的外部 PARI/GP 备选程序，默认先使用本地 PARI DLL。gp.stage1_exe 按 Windows 程序搜索规则查找，也接受完整路径。GP 失败保留 PARI 部分结果，不自动使用 GMP；数据集脚本继续使用各自的 --gp 路径规则。

<a id="stage2_groot_balanced"></a>

### groot_balanced

```toml
groot_balanced = "auto"
```

类型 `bool|string` · 范围 `[auto|false|true|0|1]` · Worker 覆盖 `[worker.N] groot_balanced` · 级别 `高级`。

G根多项式布局："auto"按已验证的NTT拐点模型及实测tune选型，true强制分块均衡布局，false强制连续布局，默认"auto"。不改变giant点块划分或B2/D/P。仅CUDA支持强制布局，OpenCL接受"auto"。CLI --groot-layout auto|true|false覆盖当前worker配置，并保留CLI 0/1别名；tune行记录实际布局0/1。

<a id="stage2_groot_subtree"></a>

### groot_subtree

```toml
groot_subtree = "1[:3072], 0[8192:], auto[]"
```

类型 `int|string` · 范围 `[auto|-1|0|1|slice list]` · Worker 覆盖 `[worker.N] groot_subtree` · 级别 `高级`。

CUDA Stage2 G 树子树策略。0 关闭，1 对支持的原生或承载梅森算术开启四根子树，auto 或 -1 按匹配的 ECM tune 结果选择。自动模式在两种模式均有合格结果前走原路径。切片按实际余因子位宽依次匹配，含起点、不含终点。命令行 --groot-subtree 优先。其他算术路径沿用原树。

<a id="stage2_log_level"></a>

### log_level

```toml
log_level = -1
```

类型 `int|string` · 范围 `[quiet|curve|phases|batches|debug|0..4]` · Worker 覆盖 `[worker.N] log_level` · 级别 `高级`。

兼容旧配置的控制台日志等级：quiet、curve、phases、batches、debug 分别对应 0..4。未配置时由 verbose 决定；debug 在生产程序中开启独立调试日志，普通控制台最高保持 batches。一般只需使用 verbose 和 debug_log。

<a id="stage2_low_cpu_wait"></a>

### low_cpu_wait

```toml
low_cpu_wait = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] low_cpu_wait` · 级别 `高级`。

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

可选低CPU策略：CUDA设备阻塞调度（4）配合事件退避轮询；同步读回保持原路径，可能增加耗时。
默认false：自动调度（0）和原生事件等待。命令行--low-cpu-wait 0|1优先，FIFO回收独立生效。
显式配置设置NTT_CUDA_WAIT_MODE、NTT_HOST_EVENT_WAIT，并关闭NTT_HOST_READBACK_WAIT。
未配置时保留专家环境覆盖；读回前休眠可单独实验。

<a id="stage2_ntt_counts"></a>

### ntt_length_counts

```toml
ntt_length_counts = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] ntt_length_counts` · 级别 `高级`。

生产 Stage2 每条曲线结束时输出 NTT 长度计数表：1 开启，0 关闭采集与输出。统计数学变换次数，每次卷积含两个正向与一个逆向；包含多项式初始化及主计算，不含 Stage1 或底层独立算术自检。Edwards/Montgomery 通用，不增加设备计数或额外同步。CLI --ntt-counts 覆盖选定 worker 的 TOML。

<a id="stage2_ntt_two_axis"></a>

### ntt_two_axis

```toml
ntt_two_axis = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] ntt_two_axis` · 级别 `高级`。

统一控制 CUDA Stage2 的 F 树、G 树、Newton 倒数、H fold 和下降是否使用双轴 NTT。双轴路径要求梅森承载、受支持的形状和可用显存。其余乘积及旧成本配置使用单轴。命令行 --two-axis-ntt 0|1 优先。此选项在 OpenCL 上不生效。

<a id="stage2_opencl_point_chunk"></a>

### opencl_point_chunk

```toml
opencl_point_chunk = 0
```

类型 `u64` · 范围 `[0..4096]` · Worker 覆盖 `[worker.N] opencl_point_chunk` · 级别 `高级`。

每条 OpenCL 命令的点数。0 自动；正值锁定分块并关闭运行时计划试探。

<a id="stage2_opencl_point_group"></a>

### opencl_point_group

```toml
opencl_point_group = 0
```

类型 `u64` · 范围 `[0..4096]` · Worker 覆盖 `[worker.N] opencl_point_group` · 级别 `高级`。

OpenCL 点工作组大小。0 使用后端默认；正值必须符合设备与内核上限。

<a id="stage2_opencl_point_plan"></a>

### opencl_point_plan

```toml
opencl_point_plan = "auto"
```

类型 `string` · 范围 `[auto|device|chain|cpu]` · Worker 覆盖 `[worker.N] opencl_point_plan` · 级别 `高级`。

OpenCL 点生成。auto 从已验证的默认计划起步，可用完整曲线继续调优。device 使用设备点计划（含已验证短批次）；chain 选择连续巨步种子；cpu 使用 GMP 点及 OpenCL 树。手动计划跳过运行时探索。

<a id="stage2_opencl_point_window"></a>

### opencl_point_window

```toml
opencl_point_window = 0
```

类型 `u64` · 范围 `[0..65536]` · Worker 覆盖 `[worker.N] opencl_point_window` · 级别 `高级`。

巨步坐标暂存窗口点数。0 由后端选型；正值按预算和 G 批次边界调整。

<a id="stage2_opencl_short_point_batch"></a>

### opencl_short_point_batch

```toml
opencl_short_point_batch = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] opencl_short_point_batch` · 级别 `高级`。

允许已验证的短设备批次。关闭后不使用该策略合并 baby 提交及配对 giant。

<a id="stage2_opencl_tune_profile"></a>

### opencl_tune_file

```toml
opencl_tune_file = "stage2_opencl_tune.toml"
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] opencl_tune_file` · 级别 `高级`。

原生 OpenCL 点计划计时文件。相对 ecm.toml，可按 worker 覆盖；空路径关闭学习。共用 tune_runtime_enabled、tune_min_repeats 和 tune_runtime_sample_limit。

<a id="stage2_pack_zero"></a>

### pack_zero

```toml
pack_zero = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] pack_zero` · 级别 `高级`。

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

每个操作数用一个 GPU 内核完成 digit 打包和补零，覆盖连续输入、设备驻留 gather 及反向输入。其余形状使用原打包路径。命令行 --pack-zero 优先于本键。旧 .cprof 成本配置默认使用 false，显式配置时也须设为 false。

<a id="stage2_parallel_tune_curves"></a>

### parallel_tune_curves

```toml
parallel_tune_curves = 5
```

类型 `u64` · 范围 `[2..16]` · Worker 覆盖 `[worker.N] parallel_tune_curves` · 级别 `高级`。

CUDA Stage2 每个队列调优窗口的曲线数。2 至 4 条按整窗吞吐比较，5 条及以上还检查中段完成吞吐。

<a id="stage2_prediction_policy"></a>

### prediction_policy

```toml
prediction_policy = "model"
```

类型 `string` · 范围 `[model|paired]` · Worker 覆盖 `[worker.N] prediction_policy` · 级别 `高级`。

model：优先使用精确配对实测；其他输入分别按通用、梅森算术位宽区间预测，满足双侧误差保护及当前显存准入才替换。缺调优资料时使用明确标为估算的4060 Laptop/1800MHz基础模型；未覆盖的NTT/驻留区间保留默认。paired恢复原精确位宽/B2实测收益门禁。B1不作为复用门槛。

<a id="stage2_ratio_adjust"></a>

### ratio_adjust

```toml
ratio_adjust = 1
```

类型 `double` · 范围 `<x:R,finite,x>0>` · Worker 覆盖 `[worker.N] ratio_adjust` · 级别 `高级`。

将 Auto B2 模型中的 Stage2 时间乘以该系数。1 不调整，大于 1 表示估计 Stage2 更贵，小于 1 表示更便宜；通常会相应影响所选 B2。此键只调整选形成本，不会让 GPU 按比例变快或变慢。

<a id="stage2_reduce_backend"></a>

### reduce_backend

```toml
reduce_backend = "auto"
```

类型 `string` · 范围 `[auto|scalar|coop32]` · Worker 覆盖 `[worker.N] reduce_backend` · 级别 `高级`。

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

普通S4逐系数归约：默认auto，在同形状重复对照合格前保留scalar，再按含点启动检查的实测总耗时选择最快项。每个算术家族固定D／布局／链长／点后端，每次仅增加一个缺少的协作变体（两遍／合并），完成有效对照后再继续D搜索。CUDA实际算术8193..16384 bits可使用共享余数；较小位宽和精确梅森归约保持原内核。实际reduce_coop、线程块与内核身份按ECM每行独立记账，即使忽略environment也不混合，独立NTT表继续复用。CLI --reduce-backend覆盖当前worker；省略时保留显式NTT_S4_COOP_REDUCE=0|32。OpenCL实际为scalar。

<a id="stage2_relocation"></a>

### relocation

```toml
relocation = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] relocation` · 级别 `高级`。

可选的生产 GPU Stage2 保守 relocation。保持 B2 和原 giant 终点，干净 NF 使用较短后缀；命中或出现目标 N 非单位（包括 GCD=N）时重放原区间。短点块按完整 G 边界重分。默认 0 保留原算法，CLI --relocation 覆盖当前 worker 的 TOML。D/AutoB2 和内存准入使用原区间估算；本模式不采集 ECM 调优样本、不执行启动 ECM 标定，并禁用显式 --tune ecm；纯 NTT 样本仍可复用。

<a id="stage2_batch_mb"></a>

### s4_batch_mib

```toml
s4_batch_mib = 64
```

类型 `u64` · 范围 `<x:Z,1..2^20>` · Worker 覆盖 `[worker.N] s4_batch_mib` · 级别 `高级`。

设置 Stage2 S4 批次的工作缓冲预算，单位为 MiB。增大预算可能减少分批开销，但会占用更多显存；它只约束对应批次缓冲，不是整个进程的显存上限。

<a id="stage2_seed_backend"></a>

### seed_backend

```toml
seed_backend = "coop32"
```

类型 `string` · 范围 `[auto|scalar|coop32]` · Worker 覆盖 `[worker.N] seed_backend` · 级别 `高级`。

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

CUDA 默认在实际算术位宽 8193..16384 bits 且使用成对种子时，以 coop32 生成种子和设备端 [D]Q。显式 auto 从标量起步，依据通过校验的配对 tune 结果选择。CLI --seed-backend 覆盖 TOML。省略此键时优先使用显式 NTT_GIANT_SEED_COOP。旧成本配置默认使用标量，OpenCL 使用标量。

<a id="stage2_short_calibration"></a>

### short_calibration

```toml
short_calibration = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] short_calibration` · 级别 `高级`。

无合适 ECM tune 文件、必要组件缺失或任务位宽完全没有可复用校准覆盖时短校准；部分覆盖允许估算。

<a id="stage2_short_calibration_seconds"></a>

### short_calibration_seconds

```toml
short_calibration_seconds = 10
```

类型 `u64` · 范围 `<x:Z,1..600>` · Worker 覆盖 `[worker.N] short_calibration_seconds` · 级别 `高级`。

短校准测量预算，不含首次 CUDA 初始化；到期完成当前测量后停止。

<a id="stage2_short_ntt_batch_tune"></a>

### short_ntt_batch_tune

```toml
short_ntt_batch_tune = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] short_ntt_batch_tune` · 级别 `高级`。

实验性短 NTT 批量选择：读取 stage2_tune_profile，作用于不超过 2^19 的长度。在当前分块预算内按含尾批的预计卷积总时间选型；至少需要两个实测 batch，预计收益须超过 10%。不改变内核配置。纯 NTT 收益不保证完整 Stage2 更快；完成整机验证前默认 false。

<a id="stage2_factor_only"></a>

### skip_prime_attribution

```toml
skip_prime_attribution = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] skip_prime_attribution` · 级别 `高级`。

控制是否跳过命中结果的素数归因诊断，即查找哪些 Stage2 素数对应此次命中。1 跳过此诊断，仍执行因子提取与验证；0 使用引擎默认诊断策略。此键不表示关闭因子分解，也不改变 decompose_factors。

### stage2_after_factored

```toml
stage2_after_factored = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] stage2_after_factored` · 级别 `常用`。

Edwards Stage1 命中后，从可复用的终点生成余因子存档并交接给本地 ECM-ELF Stage2。按产生存档的 worker 读取。默认关闭。

### stage2_algorithm

```toml
stage2_algorithm = "gwnum[4096:], gpu[:16385], cpu[]"
```

类型 `string` · 范围 `[gwnum|cpu-ntt|cpu-plain|cpu|gpu|cuda|opencl|slice list]` · Worker 覆盖 `[worker.N] stage2_algorithm` · 级别 `常用`。

统一 Stage2 在 device 为 auto 时按任务选型。按存档实际模数位宽从左到右匹配，含下界、不含上界，[] 匹配任意位宽。CPU 规则要求 Edwards；gwnum 使用 GWNUM/polymult，cpu-ntt 和 cpu-plain 使用 libecm，cpu 读取 cpu_backend。gpu 读取 stage2_gpu_backend，GPU 输入上限为 16384 bit。

### stage2_cpu_threads

```toml
stage2_cpu_threads = 1
```

类型 `u64` · 范围 `[1,1024]` · Worker 覆盖 `[worker.N] stage2_cpu_threads` · 级别 `高级`。

每个 CPU Stage2 worker 请求的线程数，以所选亲和性范围可用的线程数为上限。命令行 --threads 可覆盖。

<a id="stage2_backend"></a>

### stage2_gpu_backend

```toml
stage2_gpu_backend = "auto"
```

类型 `string` · 范围 `[auto|cuda|opencl]` · Worker 覆盖 `[worker.N] stage2_gpu_backend` · 级别 `常用`。

Windows GPU 工作程序。auto 优先同目录的 CUDA 程序及所选编号可见的 CUDA 设备，其次 OpenCL。显式选择要求安装对应程序。设备编号按选中后端解释。--gpu-backend 优先。

<a id="stage2_device"></a>

### stage2_gpu_device

```toml
stage2_gpu_device = -1
```

类型 `int` · 范围 `[-1|0..2^31-1]` · Worker 覆盖 `[worker.N] stage2_gpu_device` · 级别 `常用`。

Stage2 CUDA 设备编号。-1 继承 Stage1 的 device；0 及以上显式选择 CUDA 设备。

### stage2_results_file

```toml
stage2_results_file = ""
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] stage2_results_file` · 级别 `常用`。

CPU 与 GPU 队列任务的可选逐曲线 JSONL。留空时队列不写，直接运行 save 使用默认文件名。旧 checkpoint 迁移完成前保留原 JSONL。

### stage2_save_dir

```toml
stage2_save_dir = ""
```

类型 `string` · 范围 `<dir>` · Worker 覆盖 `[worker.N] stage2_save_dir` · 级别 `高级`。

指定 Stage2 查找任务所引用 save 的目录。留空继承共享 tmp_dir；任务给出绝对 save 路径时直接使用该路径。程序读取并验证源 save，不修改其内容。

### stage2_submission_file

```toml
stage2_submission_file = "results.json.txt"
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] stage2_submission_file` · 级别 `常用`。

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

仅含提交字段的ECM JSON行，与内部结果JSONL分开；FFT长度参考Prime95 mult.asm。个人标识与校验值仅保存在本地忽略文件中。

<a id="stage2_target_factor_bits"></a>

### target_factor_bits

```toml
target_factor_bits = 0
```

类型 `u64` · 范围 `<x:Z,0|2..16384>` · Worker 覆盖 `[worker.N] target_factor_bits` · 级别 `高级`。

Dickman 半光滑成功概率的目标素因子位数；0 根据 B1 推荐，与完整模数位宽不同。

<a id="stage2_tune_arithmetic"></a>

### tune_arithmetic

```toml
tune_arithmetic = "both"
```

类型 `string` · 范围 `[both|generic|mersenne]` · Worker 覆盖 `[worker.N] tune_arithmetic` · 级别 `高级`。

显式位宽网格ECM tune的算术家族：both同时测量；generic使用通用可能素数；mersenne使用去除真因子指数继承因子及小因子后的2^p-1承载算术。p为网格要求的算术位宽，实际余因子位宽另行记录。真实worktodo_file/save配对仍遵循auto_carrier；无效或产生因子的合成梅森样本跳过，不写成计时数据。

<a id="stage2_tune_budget_seconds"></a>

### tune_budget_seconds

```toml
tune_budget_seconds = 1800
```

类型 `u64` · 范围 `<x:Z,1..86400>` · Worker 覆盖 `[worker.N] tune_budget_seconds` · 级别 `高级`。

调优预算，包含输入准备、测量及显式开启的预热。每个位宽/B2 从默认模型的 D 开始比较邻档，仅沿超过 2% 且跨过 MAD 噪声界限的实测收益移动。复用已测候选；到期完成正在运行的测量并保留有效摘要。

<a id="stage2_tune_condition_tag"></a>

### tune_condition_tag

```toml
tune_condition_tag = ""
```

类型 `string` · 范围 `<text>` · Worker 覆盖 `[worker.N] tune_condition_tag` · 级别 `高级`。

可选测量条件标签，例如固定频率或功耗。不同标签保留独立最快汇总；忽略 environment 时允许复用。

<a id="stage2_tune_error_limit"></a>

### tune_error_limit

```toml
tune_error_limit = 0.08
```

类型 `double` · 范围 `<x:R,0<x<=1>` · Worker 覆盖 `[worker.N] tune_error_limit` · 级别 `高级`。

非负完整曲线成本模型的留一验证相对误差门限。稀疏估算保留不确定性标记，不能覆盖默认 D；局部下降比较实测中位数与 MAD，不用预测误差决定移动。

<a id="stage2_tune_ignore"></a>

### tune_ignore

```toml
tune_ignore = "gpu,driver,cuda,backend,environment"
```

类型 `string` · 范围 `[gpu|driver|cuda|backend|memory|environment],...` · Worker 覆盖 `[worker.N] tune_ignore` · 级别 `高级`。

逗号分隔的复用忽略项；留空核对全部条件。memory 默认匹配执行路径。算术合法性及当前显存准入始终检查。

<a id="stage2_cost_profile"></a>

### tune_legacy_cost_profile

```toml
tune_legacy_cost_profile = ""
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] tune_legacy_cost_profile` · 级别 `旧版`。

stage2_tune_profile 为空时供 Auto B2 使用的旧 .cprof 成本配置。保持原二进制、设备、算术及标定资格检查；新的完整 ECM tune 不放宽旧合同。留空不提供旧配置。

<a id="stage2_tune_min_repeats"></a>

### tune_min_repeats

```toml
tune_min_repeats = 2
```

类型 `u64` · 范围 `[1..1000]` · Worker 覆盖 `[worker.N] tune_min_repeats` · 级别 `高级`。

运行时D、布局与链长候选比较和采用所需的最少已校验观察数，同时作为ECM校准最小重复次数；--tune-repeats可要求更多。默认2；设为1用于快速筛查，但无法测得离散程度。不得超过tune_runtime_sample_limit；已有观察保留，提高门槛后需补足证据。算术验证不变。CLI：--tune-min-repeats N。

<a id="stage2_tune_runtime"></a>

### tune_runtime_enabled

```toml
tune_runtime_enabled = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] tune_runtime_enabled` · 级别 `高级`。

CUDA Stage1 NTT 在生产前补测缺失位宽，关闭时只读选型。Stage2 CUDA 按设备、环境与执行形状累计通过检查的无因子生产计时。完全相同参数合并为一条 matrix 记录，保存 repeats、平均耗时和样本标准差，包含较慢观察。行内版本与 UTC 更新时间仅记录来源，程序升级保留兼容计时。model 选型优先试跑合法梅森承载，再比较普通路径和有希望的 D。达到 tune_min_repeats 后采用实测最快均值。B1 不限制 CUDA 成本复用。CPU GWNUM 把耗时写入 [cpu_stage2] 矩阵。预检确认 FFT 和 F 树方案一致时，可跨位宽和内存预算复用。写入使用共享锁和原子替换，缓存失败不影响已完成的曲线。false 关闭单曲线试跑、采集和写回，CPU 仍可只读复用合格记录。双槽队列由 curve_concurrency 控制。OpenCL 固定几何，在 opencl_tune_file 中学习点计划。

<a id="stage2_tune_runtime_samples"></a>

### tune_runtime_sample_limit

```toml
tune_runtime_sample_limit = 10
```

类型 `u64` · 范围 `[1..1000]` · Worker 覆盖 `[worker.N] tune_runtime_sample_limit` · 级别 `高级`。

完全相同参数（含B1）及相同设备、环境、算术和执行形状的独立运行NF观察上限，包含较慢观察及可兼容的其他程序版本。相同参数合并repeats和均值/标准差；上限统计观察次数，不统计matrix行数；不同参数行保留。达到tune_min_repeats次后可用于实测选型；调低上限不删除已有证据。CLI：--tune-runtime-samples N。

<a id="stage1_batch"></a>

### tune_stage1_batch

```toml
tune_stage1_batch = 1
```

类型 `u64` · 范围 `<x:Z,1..2^20>` · Worker 覆盖 `[worker.N] tune_stage1_batch` · 级别 `高级`。

指定 Auto B2 成本估算对应的 Stage1 批次曲线数。它用于选择或核对 Stage1 的成本条件，不会让 Stage2 驱动启动该批 Stage1 计算，也不改变任务要处理的曲线数。

<a id="stage1_cost_csv"></a>

### tune_stage1_cost_csv

```toml
tune_stage1_cost_csv = ""
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] tune_stage1_cost_csv` · 级别 `高级`。

总流程 Auto B2 的 Stage1 秒/曲线 CSV；列为 container_bits 或 target_bits、opencl_threads_per_integer、curves、b1、mhz、seconds_per_curve；可选 tpb、exponent。缺失位宽可插值或外推，成本已摊销。

<a id="stage1_tune_profile"></a>

### tune_stage1_cost_file

```toml
tune_stage1_cost_file = ""
```

类型 `string` · 范围 `<path>` · Worker 覆盖 `[worker.N] tune_stage1_cost_file` · 级别 `高级`。

已核验的完整 CUDA Stage1 成本锚点，优先级低于显式秒/曲线和 CSV。匹配批次、模数类型和指数模式；B1 与缺失位宽按 B1 和容器位宽平方估算。设备及运行时复用遵守 tune_ignore；此文件不启动 Stage1。

<a id="stage1_cost_mhz"></a>

### tune_stage1_cost_mhz

```toml
tune_stage1_cost_mhz = 0
```

类型 `u64` · 范围 `<x:Z,0..10000>` · Worker 覆盖 `[worker.N] tune_stage1_cost_mhz` · 级别 `高级`。

Stage1 成本反频率缩放的目标频率；0 使用 CSV 参考频率，不修改 GPU 时钟。

<a id="stage1_seconds_per_curve"></a>

### tune_stage1_seconds_per_curve

```toml
tune_stage1_seconds_per_curve = 0
```

类型 `double` · 范围 `<x:R,finite,x>=0>` · Worker 覆盖 `[worker.N] tune_stage1_seconds_per_curve` · 级别 `高级`。

为 Auto B2 提供 Stage1 每曲线秒数，按实际批次摊销，无需再除以 tune_stage1_batch。正值（可填小数）优先于调优数据中的 Stage1 成本。设为 0 或省略时，使用已配置的 Stage1 成本来源。完整 ECM tune 可使用成本 CSV 或匹配的 tune_stage1_cost_file。旧组件 profile 使用其自身匹配的 Stage1 数据。该值用于成本规划。

<a id="stage2_tune_warmup"></a>

### tune_warmup

```toml
tune_warmup = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] tune_warmup` · 级别 `高级`。

0 表示 ECM/NTT 调优及启动短校准直接开始正式测量，不执行丢弃的预热轮次；1 表示每个形状先运行一轮经过核验的预热，占用相同预算但不计入中位数/MAD。CLI --tune-warmup 0|1 覆盖此键；必要算术检查始终执行。

<a id="stage2_weighted_ntt"></a>

### weighted_ntt

```toml
weighted_ntt = false
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `[worker.N] weighted_ntt` · 级别 `高级`。

生产 CUDA 后端可启用的多项式卷积。直接打包时融合首层 DIF，逆变换末层重建高低两半。需启用 pack_zero。没有缓存根表或直接设备打包的调用使用全长路径。OpenCL 不启用该 TOML 预设。命令行 --weighted-ntt 0|1 优先。

<a id="stage2_parallel_workspace"></a>

### workspace_policy

```toml
workspace_policy = "auto"
```

类型 `string` · 范围 `[auto|independent|leased]` · Worker 覆盖 `[worker.N] workspace_policy` · 级别 `高级`。

independent 为两个 CUDA Stage2 槽分别分配工作区。leased 允许下一槽先准备和计算 Baby 点，到 F 树前再租用大工作区。auto 比较两种模式，只有租约可准入时直接使用 leased。

## [gui]

### body_font_size

```toml
body_font_size = 15
```

类型 `int` · 范围 `[10..30]` · Worker 覆盖 `不支持` · 级别 `高级`。

正文与日志字号（逻辑像素），独立于界面 DPI 比例；主要操作与次要说明使用各自偏移，标题保持相对比例。

### display_language

```toml
display_language = "en"
```

类型 `string` · 范围 `[en|zh]` · Worker 覆盖 `不支持` · 级别 `高级`。

界面语言；中文字体不可用时回退英文。

### dock_layout

```toml
dock_layout = ""
```

类型 `string` · 范围 `<escaped_blob>` · Worker 覆盖 `不支持` · 级别 `旧版`。

保存 GUI 面板停靠布局的转义数据。留空使用默认布局；由 GUI 自动写回，手动修改不完整数据可能使布局无法恢复。

### dock_layout_ver

```toml
dock_layout_ver = 0
```

类型 `int` · 范围 `<x:Z,x>=0>` · Worker 覆盖 `不支持` · 级别 `旧版`。

保存已写入布局的版本编号，由 GUI 管理。版本早于当前默认布局时会重建布局；此键不是用户选择 GUI 功能版本的开关。

### exit_confirm

```toml
exit_confirm = "ask"
```

类型 `string` · 范围 `[ask|stop|kill]` · Worker 覆盖 `不支持` · 级别 `高级`。

控制关闭GUI时如何处理运行中的worker：ask先询问，stop请求安全停止，kill立即终止。Stage1写检查点后退出；Stage2完成当前曲线、记录完成进度后退出。强制终止可能丢失未存的Stage1计算或当前Stage2曲线。

### external_windows

```toml
external_windows = false
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

重启后允许原生独立面板窗；默认在主窗口内停靠/浮动，以规避已观察到的慢呈现路径。

### follow_log

```toml
follow_log = true
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

跟随新的 worker 输出。

### font

```toml
font = ""
```

类型 `string` · 范围 `<ttf|ttc>` · Worker 覆盖 `不支持` · 级别 `旧版`。

指定 GUI 使用的 .ttf 或 .ttc 字体文件。留空由程序按界面语言选择系统字体；需要中文界面时应使用包含中文字形的字体。

### font_size

```toml
font_size = 0
```

类型 `float|string` · 范围 `[auto|<x:R,6..96>]` · Worker 覆盖 `不支持` · 级别 `旧版`。

设置 GUI 字号，单位为像素，可使用小数。auto 或空值按 15*DPI 比例自动选择；小于 6 或非有限数恢复自动，大于 96 按 96 处理。手动字号不会再按自动规则选择大小。

### font_snap

```toml
font_snap = true
```

类型 `bool` · 范围 `[false|true|0|1]` · Worker 覆盖 `不支持` · 级别 `旧版`。

控制字体是否对齐像素网格。1 开启，0 关闭；小数字号或不同 DPI 下可比较显示清晰度。兼容 off、false、no 关闭，其他值按开启处理，字符串区分大小写。

### gpu_clock_ranges

```toml
gpu_clock_ranges = ""
```

类型 `string` · 范围 `<GPU-UUID:minMHz-maxMHz[:auto];...>` · Worker 覆盖 `不支持` · 级别 `高级`。

GUI v2 的 GPU 核心频率草稿：GPU-<UUID>:最小MHz-最大MHz，多卡用分号分隔。可选 :auto 启用启动时一次、每个本地整点、以及每次连续越界 5 秒后的单次自动应用；检测复用监控采样。无 :auto 时载入不修改硬件。保存不应用本次草稿；点击应用才启用本次范围和自动选项。恢复默认关闭本次会话自动应用，不删除保存草稿。空值不保存范围，不改变 worker 计算配置。

### gpu_history_seconds

```toml
gpu_history_seconds = 60
```

类型 `int` · 范围 `[15|30|60|120]` · Worker 覆盖 `不支持` · 级别 `高级`。

GPU 图表时间范围（秒），所有设备共用并随界面偏好保存；不改变 500 ms 采样间隔。

### gpu_poll_ms

```toml
gpu_poll_ms = 500
```

类型 `int` · 范围 `<x:Z,100..2^31-1>` · Worker 覆盖 `不支持` · 级别 `高级`。

设置 GPU 监控查询间隔，单位为毫秒，最小为 100。增大可减少 NVML 查询频率；此键只影响监控数据更新，不控制计算 kernel 的调度。

### graceful_stop_ms

```toml
graceful_stop_ms = 300000
```

类型 `i64` · 范围 `<x:Z,1000..3600000>` · Worker 覆盖 `不支持` · 级别 `高级`。

请求安全停止后等待的最长时间，单位毫秒。进程退出结束等待；到期仍运行的worker强制终止。应为Stage1检查点写入或当前Stage2整条曲线留足时间；Stage2曲线计数不是中间算术存档。

### group_order_gp

```toml
group_order_gp = ""
```

类型 `string` · 范围 `<path-to-gp.exe|empty>` · Worker 覆盖 `不支持` · 级别 `旧版`。

群阶分解界面使用的外部 PARI/GP 备选程序路径。Auto 默认先使用本地 PARI DLL；空值在需要 GP 时从 PATH 查找。此路径仅供 GUI 使用，计算 worker 使用相同诊断层，但 GP 路径独立设置。

### inspector

```toml
inspector = true
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

宽窗口显示 worker 检查区。

### language

```toml
language = "english"
```

类型 `string` · 范围 `<xml-stem>` · Worker 覆盖 `不支持` · 级别 `高级`。

选择 GUI 界面语言，值为 localization 目录下语言 XML 文件的文件名主体，不带 .xml。找不到所需资源或字体不能显示该语言时，按 GUI 的回退规则使用可显示的界面。

### language_file

```toml
language_file = ""
```

类型 `string` · 范围 `<string, up to 255 bytes>` · Worker 覆盖 `不支持` · 级别 `高级`。

程序旁localization/v2目录中的界面XML文件名；空值按旧en/zh选择迁移。配套options.*.xml不是可选界面文件。

### localization_dir

```toml
localization_dir = ""
```

类型 `string` · 范围 `<dir>` · Worker 覆盖 `不支持` · 级别 `高级`。

指定 GUI 语言 XML 文件目录。留空使用 GUI 可执行文件旁的 localization 目录；从开发构建目录运行时还会尝试源码中的语言资源目录。

### logs

```toml
logs = true
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

开启Workers管理页日志；其他视图的自动隐藏不改变此保存偏好。

### logs_keep_visible

```toml
logs_keep_visible = false
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

已开启日志跨页保持可见；关闭时跟随可见的Workers管理页，其他视图可临时手动打开。

### nav_collapsed

```toml
nav_collapsed = false
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

将左侧导航缩为图标栏。

### nav_expanded_width

```toml
nav_expanded_width = 0
```

类型 `int` · 范围 `[0..16384]` · Worker 覆盖 `不支持` · 级别 `高级`。

上次展开的导航宽度，以逻辑像素计。0 使用默认宽度。

### nav_hide_tab_bar

```toml
nav_hide_tab_bar = true
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

导航顶部标签栏默认收起，保存的选择在重启及布局重建后保留。

### nav_order

```toml
nav_order = "01237456"
```

类型 `string` · 范围 `<string, up to 8 bytes>` · Worker 覆盖 `不支持` · 级别 `高级`。

稳定页面编号0..7；5打开设置。旧七页排序追加群阶分解页，不重置布局。

### page

```toml
page = 0
```

类型 `int` · 范围 `[0..7]` · Worker 覆盖 `不支持` · 级别 `高级`。

页面：0任务、1结果、2生成器、3概率、4 GPU、6组件、7群阶分解；5打开设置。

### pause_schedule

```toml
pause_schedule = []
```

类型 `string_array` · 范围 `["application, Worker, weekday, time", ...]` · Worker 覆盖 `不支持` · 级别 `高级`。

GUI v2 暂停规则。每项依次填写应用、Worker、星期、时间。例如 ["foobar, [1,3-5], Mon-Fri, 08:30-17:30"]。应用按运行程序路径的子串匹配，忽略大小写。仅按时间暂停时使用 *。空数组关闭计划。

### pet_animate

```toml
pet_animate = true
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

按既有 GUI 节奏动画，不另开渲染循环。

### pet_enabled

```toml
pet_enabled = false
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

显示兼容 ELYTerminal 行为的窗口边缘宠物。

### pet_left

```toml
pet_left = 72
```

类型 `float` · 范围 `[0..16384]` · Worker 覆盖 `不支持` · 级别 `高级`。

宠物相对主窗口左缘的逻辑单位偏移。

### pet_skin

```toml
pet_skin = "classic"
```

类型 `string` · 范围 `[classic|miss_pink_elf|herrscher_of_human_ego|hi_love_elf]` · Worker 覆盖 `不支持` · 级别 `高级`。

ELY 外观，即时切换；选择项使用装甲的完整英文名称。

### primary_font_offset

```toml
primary_font_offset = 1
```

类型 `int` · 范围 `[0..6]` · Worker 覆盖 `不支持` · 级别 `高级`。

主要操作字号相对正文的偏移（逻辑像素）；次要按钮和日志正文保持正文字号。

### process_priority

```toml
process_priority = "below_normal"
```

类型 `string` · 范围 `[idle|below_normal|normal|above_normal|high]` · Worker 覆盖 `不支持` · 级别 `高级`。

设置 GUI 启动的 worker 进程优先级。below_normal 为默认，较低优先级便于同时使用电脑；此键不是 GPU 独占或显卡占用率设置。

### raw_log

```toml
raw_log = false
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

显示原始输出而非分类事件。

### refresh_hz

```toml
refresh_hz = 10
```

类型 `int` · 范围 `<x:Z,1..60>` · Worker 覆盖 `不支持` · 级别 `高级`。

GUI 空闲时绘制帧与 worker 状态处理的目标频率，单位为 Hz。输入消息会及时唤醒界面；提高此值可使图表和状态显示更平滑，但增加界面开销。GPU 指标按 gpu_poll_ms 独立采样，两项设置均不改变计算速度。

### results_last_copy_utc

```toml
results_last_copy_utc = ""
```

类型 `string` · 范围 `<YYYY-MM-DD HH:MM:SS UTC>` · Worker 覆盖 `不支持` · 级别 `高级`。

GUI v2 上次成功复制 JSON 结果的 UTC 时间。留空表示尚未记录复制时间。

### right_hide_tab_bar

```toml
right_hide_tab_bar = false
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

持久记忆右侧停靠窗Hide tab bar选择；多个原生标签保持可见，恢复单标签时恢复该选择。

### right_page

```toml
right_page = 4
```

类型 `int` · 范围 `[0..7]` · Worker 覆盖 `不支持` · 级别 `高级`。

右侧停靠区最后激活的实际页面；读取旧设置编号5时改选实际页面。

### right_pages

```toml
right_pages = 0
```

类型 `int` · 范围 `[0..254]` · Worker 覆盖 `不支持` · 级别 `高级`。

右侧实际页面共用横排标签，关闭后返回导航，至少保留一个实际中央页面；读取旧侧车时移除设置位5。

### right_visible

```toml
right_visible = true
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

保留以读取旧侧车的兼容字段；右侧页面直接显示，关闭标签即可回到导航。

### scale

```toml
scale = 0
```

类型 `float` · 范围 `[0..4]` · Worker 覆盖 `不支持` · 级别 `高级`。

零按窗口 DPI 自动计算；正值指定界面比例。

### secondary_font_offset

```toml
secondary_font_offset = -2
```

类型 `int` · 范围 `[-6..0]` · Worker 覆盖 `不支持` · 级别 `高级`。

次要说明字号相对正文的偏移（逻辑像素）；实际字号最小为 8 逻辑像素。

### selected_worker

```toml
selected_worker = 1
```

类型 `int` · 范围 `[1..1024]` · Worker 覆盖 `不支持` · 级别 `高级`。

选中的 worker 分区。

### smooth_scroll

```toml
smooth_scroll = true
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

主窗口与分离窗口使用平滑滚轮滚动。

<a id="exe"></a>

### stage1_exe

```toml
stage1_exe = ""
```

类型 `string` · 范围 `<exe>` · Worker 覆盖 `不支持` · 级别 `高级`。

Stage1 工作程序。相对路径以 GUI 目录为基准，单独文件名也可经 PATH 查找；留空依次查找 ecm_cuda.stage1_exe、ecm.stage1_exe。每个 worker 根据 worktodo_file 前缀选择此程序或 stage2_exe。

### stage2_exe

```toml
stage2_exe = ""
```

类型 `string` · 范围 `<exe>` · Worker 覆盖 `不支持` · 级别 `高级`。

CPU、CUDA 或 OpenCL 的 Stage2 程序。留空时在工作流和 GUI 目录依次查找 ecm_stage2.stage1_exe、ecm_cuda_stage2.stage1_exe、ecm_cpu_stage2.stage1_exe 和 ecm_opencl_stage2.stage1_exe，也支持唯一匹配的子包。安装多个子包时可用路径指定后端。

### stage2_timing_profile

```toml
stage2_timing_profile = ""
```

类型 `string` · 范围 `<string, up to 4096 bytes>` · Worker 覆盖 `不支持` · 级别 `高级`。

可选的只读Stage2 tune摘要路径。空值使用工作流stage2_tune_profile，再查找工作流/worker旁的tune.toml或stage2_ecm_tune.toml；相对路径基于工作流目录。

### stage2_timing_source

```toml
stage2_timing_source = "auto"
```

类型 `string` · 范围 `[auto|previous|tune]` · Worker 覆盖 `不支持` · 级别 `高级`。

Stage2区段时长来源：auto按上次完成曲线覆盖tune数据；previous或tune仅使用指定来源，缺项明确标为未知。

### start_tab

```toml
start_tab = "workers"
```

类型 `string` · 范围 `[workers|detail|gen|prob]` · Worker 覆盖 `不支持` · 级别 `旧版`。

选择 GUI 启动时优先显示的面板。workers 为 worker 列表，detail 为详情，gen 为任务生成器，prob 为ECM概率计算器；概率面板也可从查看菜单打开。不改变是否自动启动 worker。

### theme

```toml
theme = "dark"
```

类型 `string` · 范围 `[dark|light]` · Worker 覆盖 `不支持` · 级别 `高级`。

外观模式，独立于旧 GUI 设置。

### toasts

```toml
toasts = true
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

显示不抢焦点的操作轻提示；错误提示仍显示。

### window

```toml
window = [120, 80, 1500, 900]
```

类型 `rect` · 范围 `<x:Z>,<y:Z>,<w:Z>,<h:Z>` · Worker 覆盖 `不支持` · 级别 `旧版`。

保存 GUI 主窗口的位置和大小，依次为 x,y,宽,高。GUI 正常退出时会更新；通常无需手动编辑，宽和高应为正数。

### window_height

```toml
window_height = 850
```

类型 `int` · 范围 `[360..16384]` · Worker 覆盖 `不支持` · 级别 `高级`。

恢复窗口的物理像素高度。

### window_maximized

```toml
window_maximized = false
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

恢复最大化状态。

### window_title_bar

```toml
window_title_bar = false
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

显示系统标题栏；默认隐藏，窗口按钮位于应用顶栏；保留显式保存的选择。

### window_width

```toml
window_width = 1280
```

类型 `int` · 范围 `[480..16384]` · Worker 覆盖 `不支持` · 级别 `高级`。

恢复窗口的物理像素宽度。

### window_x

```toml
window_x = -2147483648
```

类型 `int` · 范围 `[-2147483648..2147483647]` · Worker 覆盖 `不支持` · 级别 `高级`。

窗口物理像素 x；INT_MIN 由系统放置。

### window_y

```toml
window_y = -2147483648
```

类型 `int` · 范围 `[-2147483648..2147483647]` · Worker 覆盖 `不支持` · 级别 `高级`。

窗口物理像素 y；INT_MIN 由系统放置。

### worker_chart_auto_y

```toml
worker_chart_auto_y = false
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

曲线耗时Y轴按当前保留窗口的有效样本自动缩放；关闭时从0开始。

### worker_chart_columns

```toml
worker_chart_columns = 1
```

类型 `int` · 范围 `[1|2]` · Worker 覆盖 `不支持` · 级别 `高级`。

曲线耗时图列数：1为列表，2为每行两图；工作区过窄时临时使用单列，不改变保存的选择。

### workspace_hide_tab_bar

```toml
workspace_hide_tab_bar = true
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `高级`。

工作区顶部标签栏默认收起，保存的选择在重启及布局重建后保留。

### workspace_layout

```toml
workspace_layout = ""
```

类型 `string` · 范围 `<string, up to 524288 bytes>` · Worker 覆盖 `不支持` · 级别 `高级`。

十六进制编码的 ImGui 布局，仅存于本侧车。

## [worker.1]

### autostart

```toml
autostart = false
```

类型 `bool` · 范围 `[false|true]` · Worker 覆盖 `不支持` · 级别 `常用`。

设为 true 时，GUI 打开工作流后自动启动此 worker。

### extra_args

```toml
extra_args = ""
```

类型 `string` · 范围 `<argv>` · Worker 覆盖 `不支持` · 级别 `高级`。

向 GUI 启动的 worker 追加两阶段共用的命令行参数。阶段专用选项使用对应的键。工作流配置、worker 编号和队列路径由 GUI 管理。

### gpucurves

```toml
gpucurves = 0
```

类型 `int` · 范围 `<x:Z,x>=0>` · Worker 覆盖 `不支持` · 级别 `旧版`。

保留旧 GUI 配置的曲线数显示字段，可继承全局同名值。Stage1 队列执行的曲线数由各任务行决定，此键不会覆盖任务；直接单次运行的 -gpucurves 是独立命令行参数。

### name

```toml
name = "Worker #{N}"
```

类型 `string` · 范围 `<text>` · Worker 覆盖 `不支持` · 级别 `高级`。

设置此 worker 在 GUI 中显示的名称。未配置时显示 Worker #N，其中 N 为 worker 编号；只改变显示名称，不改变设备或任务分配。

### stage1_extra_args

```toml
stage1_extra_args = ""
```

类型 `string` · 范围 `<argv>` · Worker 覆盖 `不支持` · 级别 `高级`。

Stage1 专用命令行参数，追加在共用 extra_args 之后。参数以空格分隔，含空格的路径使用双引号。

### stage2_extra_args

```toml
stage2_extra_args = ""
```

类型 `string` · 范围 `<argv>` · Worker 覆盖 `不支持` · 级别 `高级`。

Stage2 专用命令行参数，追加在共用 extra_args 之后。参数以空格分隔，含空格的路径使用双引号。

## 维护

修改 `config/ecm_options.json` 后运行 `python tools/gen/generate_ecm_config.py`，并将定义与生成物一同提交。
