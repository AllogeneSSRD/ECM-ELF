# ecm.toml Reference / ecm.toml 配置参考

<!-- GENERATED: config/ecm_options.json; do not edit. -->

Stage1, Stage2 and the GUI read the same `ecm.toml` file. Strings are quoted, booleans are `true`/`false`, and `window` is an integer array.

Stage1、Stage2 和 GUI 共用 `ecm.toml`。字符串须加引号，布尔值使用 `true`/`false`，`window` 使用整数数组。

## Tables and precedence / 分区与优先级

- `[shared]`: settings used by both stages. / 两阶段共用。
- `[stage1]`, `[stage2]`: stage-specific settings. / 阶段专用。
- `[gui]`: GUI-wide settings; these do not have worker overrides. / GUI 全局设置，不支持 worker 覆盖。
- `[workers.N]`: GUI settings for worker N (1..1000000). / 第 N 个 GUI worker 的设置。
- `[workers.N.shared]`, `[workers.N.stage1]`, `[workers.N.stage2]`: worker overrides for the matching table. / 对应分区的 worker 覆盖。

Precedence: built-in default < global table < matching worker table < CLI. A key may appear only once in a table; unknown tables, duplicate keys and invalid TOML values are errors. `worktodo.txt` retains its separate `[Worker #N]` queue sections. INI files are not read.

优先级：内置默认值 < 全局分区 < 对应 worker 分区 < 命令行。同一分区中的键不能重复；未知分区、重复键和无效 TOML 值会报错。`worktodo.txt` 继续使用独立的 `[Worker #N]` 队列分区。程序不读取 INI。

Compute programs skip GUI-only values in `[gui]` and `[workers.N]`, warning about unknown keys in those tables. The exception is `gui.NumWorkers`: it remains validated and available as the Stage1 routing fallback when a destination queue has no worker sections. Unknown compute keys/tables, invalid compute types, malformed TOML and duplicate keys are still errors. GUI readers retain strict schema validation. This compatibility boundary does not permit arbitrary unknown tables or unsupported TOML value syntax.

计算程序跳过 `[gui]` 和 `[workers.N]` 中的纯界面值，对其中未知键给出警告。例外是 `gui.NumWorkers`：目标队列没有 worker 分区时，Stage1 分发用它作为后备数量，仍保留并校验。计算键和分区未知、计算值类型错误、TOML 语法错误和重复键仍拒绝；GUI 读取器继续严格校验 schema。兼容范围不包括任意未知分区或尚不支持的 TOML 值语法。

The shared worktodo path and Stage2 relative paths use the TOML directory; other Stage1 relative paths use the executable directory. CLI relative paths use the current directory.

共用 worktodo 与 Stage2 的相对路径基于 TOML 所在目录；其他 Stage1 相对路径基于可执行文件目录；命令行相对路径基于当前目录。

## [shared]

### b2_ratio

```toml
b2_ratio = 0
```

Type / 类型: `u64`; range / 范围: `[0|1..1000000]`; worker override / worker 覆盖: `[workers.N.shared]`.

At Stage2 execution with this TOML, nonzero ratio replaces B2 with saved B1 multiplied by this integer, including direct save runs. Zero or omitting the key preserves normal B2 selection. Product overflow is rejected.

使用此 TOML 执行 Stage2 时，非零整数将 B2 设为存档 B1 与此值的乘积，直接读档也适用。0 或不填写此键沿用原有 B2 选择；乘积溢出会报错。

### computer

```toml
computer = ""
```

Type / 类型: `string`; range / 范围: `<local-computer-name>`; worker override / worker 覆盖: `[workers.N.shared]`.

Optional local computer name in result output. Keep configured values and generated results outside public Git commits.

结果输出的可选本地计算机名；实际配置值及生成结果不得进入公共Git提交。

### priority

```toml
priority = 0
```

Type / 类型: `int`; range / 范围: `[0|1|2]`; worker override / worker 覆盖: `[workers.N.shared]`.

Queue selection order: 0 follows file order; 1 drains ECMSTAGE1 first; 2 drains ECMSTAGE2 first. When no preferred task remains, the other stage runs. Worker sections can override this setting.

队列选择顺序：0 按文件顺序；1 优先完成 ECMSTAGE1；2 优先完成 ECMSTAGE2。优先阶段暂无任务时执行另一阶段；worker 分区可覆盖。

### skip

```toml
skip = 0
```

Type / 类型: `int`; range / 范围: `[0|1|2]`; worker override / worker 覆盖: `[workers.N.shared]`.

0 processes both stages; 1 leaves ECMSTAGE1 rows untouched; 2 leaves ECMSTAGE2 rows untouched. Worker sections can override this setting.

0 处理两个阶段；1 完全跳过 ECMSTAGE1 行；2 完全跳过 ECMSTAGE2 行。worker 分区可覆盖。

### user

```toml
user = ""
```

Type / 类型: `string`; range / 范围: `<local-user-id>`; worker override / worker 覆盖: `[workers.N.shared]`.

Optional local result user identity for Stage1 and Stage2. Never put real account values in committed templates or fixtures.

Stage1与Stage2结果的可选本地用户标识；真实账号值不得写入提交的模板或测试样本。

### worktodo

```toml
worktodo = "worktodo.txt"
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.shared]`.

Shared Stage1/Stage2 task queue for the current worker. ECMSTAGE1 and ECMSTAGE2 select the engine; Stage1 and Stage2 keep separate finished records. One worker process consumes this queue at a time.

Stage1/Stage2 共用的任务队列。ECMSTAGE1 与 ECMSTAGE2 前缀选择执行程序；两阶段分别记录完成任务。同一 worker 同时只能有一个进程消费此队列。

## [stage1]

### affinity

```toml
affinity = ""
```

Type / 类型: `string`; range / 范围: `[none|auto|<n>,...|<a>-<b>]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Logical CPU IDs or ranges for CPU Stage1 threads, assigned round-robin. Empty, none, or auto leaves scheduling to the OS. The current Windows implementation supports only IDs 0..63 in one processor group; other platforms do not apply this binding. Use IDs available on the machine.

设置 CPU Stage1 计算线程的亲和性，可用逻辑 CPU 编号列表或区间；线程依次循环使用列表中的 CPU。空值、none 或 auto 均不主动绑定，由操作系统调度。当前 Windows 实现只对单处理器组内编号 0..63 生效，其他平台不执行此绑定；编号还须对应实际可用的逻辑 CPU。

### auto_stage2

```toml
auto_stage2 = 0
```

Type / 类型: `int`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Enable Stage1-to-Stage2 handoff after successful Stage1 without a factor. With no receiver list and the same worktodo, replace the source row in place with ECMSTAGE2; otherwise deliver to auto_stage2_worktodo/auto_stage2_workers. Stage1 finished output is retained. A factor hit generates no Stage2 task. Zero disables new handoffs. Worker sections can override this setting.

启用 Stage1 无因子成功完成后的 Stage2 交接。未指定接收 worker 且目标为同一 worktodo 时，原位置改写为 ECMSTAGE2；否则按 auto_stage2_worktodo/auto_stage2_workers 投递。保留 Stage1 finished 记录；找到因子不生成 Stage2 任务。0 关闭新交接，worker 分区可覆盖。

### auto_stage2_workers

```toml
auto_stage2_workers = ""
```

Type / 类型: `string`; range / 范围: `[<n>|<n>,...|<a>-<b>|auto]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Receiving workers when auto_stage2=1: a single ID, comma list or ranges within 1..1000000. Select the shortest candidate queue; ties use the lowest ID. auto uses the destination's existing worker sections, falling back to GUI NumWorkers when none exist. Missing explicit sections are created. Empty keeps the current worker and rewrites in place when the destination is the same file; an external file receives that same worker number. Unreadable destinations remain pending for retry.

auto_stage2=1 时的接收 worker，可填写 1..1000000 内单个编号、逗号列表或区间；选择任务数最少者，并列取较小编号。auto 使用目标队列已有 worker 分区，无分区时使用 GUI NumWorkers。显式指定但尚不存在的分区会创建。留空保留当前 worker，同文件原位改写，其他文件也使用当前编号；目标不可写时保留待投递记录。

### auto_stage2_worktodo

```toml
auto_stage2_worktodo = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Destination queue when auto_stage2=1. Empty inherits this worker's effective shared worktodo. Accepts an absolute file path or a filename/relative path resolved beside ecm.toml. Save filenames in delivered tasks are absolute. Stage2 finished receipts use the receiving worker's stage2_finished setting from this TOML. Worker sections can override this setting.

auto_stage2=1 时的目标队列。留空使用当前 worker 生效的 shared worktodo；可填写绝对文件路径，或相对 ecm.toml 所在目录的文件名/路径。投递行中的 save 使用绝对路径。完成去重读取本 TOML 中接收 worker 的 stage2_finished；worker 分区可覆盖。

### ckpt_backups

```toml
ckpt_backups = 3
```

Type / 类型: `int`; range / 范围: `<x:Z,0..16>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Maximum number of rotated Stage1 checkpoint backups. The primary checkpoint is additional; 0 keeps only the primary. Corrupt or unreadable generations are skipped on resume. A completed Stage1 task removes its primary checkpoint and all backups.

Stage1 检查点轮换备份的最大份数；主存档另计，0 表示只保留主存档。恢复时跳过损坏或无法读取的代次；Stage1 完成后清理主存档及所有备份。

### ckpt_seconds

```toml
ckpt_seconds = 600
```

Type / 类型: `double`; range / 范围: `<x:R,finite,x>=0>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Stage1 checkpoint interval in seconds. 0 disables periodic checkpoints; negative values act as 0. Normal stop or Ctrl+C can still request a checkpoint. Actual writes wait for the backend's safe checkpoint boundary.

设置 Stage1 定期写检查点的间隔，单位为秒。0 关闭定期检查点，负数按 0 处理；仍可通过正常停止或 Ctrl+C 请求保存。实际写盘时机受当前计算后端的安全检查点限制。

### curve

```toml
curve = "mont"
```

Type / 类型: `string`; range / 范围: `[mont|ed]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Stage1 curve model: mont for Montgomery or ed for Atkin-Morain Edwards. Edwards runs on CPU or CUDA GPU, using 48*lcm(1..B1); OpenCL Edwards is unsupported. CUDA Edwards saves use METHOD=ECM-EDWARDS-AM for CUDA Stage2.

Stage1 曲线模型：mont 为 Montgomery，ed 为 Atkin-Morain Edwards。Edwards 支持 CPU 与 CUDA GPU，固定使用 48*lcm(1..B1)；OpenCL 不支持 Edwards。CUDA Edwards 输出 METHOD=ECM-EDWARDS-AM 存档供 CUDA Stage2 使用。

### device

```toml
device = 0
```

Type / 类型: `int`; range / 范围: `<x:Z,0..device_count-1>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Zero-based GPU device ID for Stage1. Stage2 inherits it unless stage2_device is explicitly set. IDs refer to the device list exposed by the executable's actual GPU backend.

选择 GPU 设备编号，从 0 开始。用于 Stage1；Stage2 未显式指定 stage2_device 时也继承此编号。编号对应各可执行文件实际使用的 GPU 后端设备列表。

### edwards_stage2_transfer

```toml
edwards_stage2_transfer = false
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage1]`.

When enabled, a completed Edwards Stage1 task is delivered as ECMSTAGE2 to the receiving worker section of the shared worktodo. Published saves use absolute paths; failed delivery remains pending for retry. auto_stage2 takes precedence when both are set.

启用后，将已完成的 Edwards Stage1 任务作为 ECMSTAGE2 投递到共用 worktodo 的接收 worker 分区。存档使用绝对路径；投递失败时保留待投递记录。两项同时设置时以 auto_stage2 为准。

### edwards_stage2_worker

```toml
edwards_stage2_worker = 0
```

Type / 类型: `int`; range / 范围: `[0|1..1000000]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Receiving worker when edwards_stage2_transfer=1. Zero keeps the Stage1 worker number; a positive value selects that worker section of the shared worktodo.

edwards_stage2_transfer=1 时的接收 worker。0 沿用 Stage1 worker 编号；正数指定共用 worktodo 中的接收分区。

### exp_cache

```toml
exp_cache = ""
```

Type / 类型: `string`; range / 范围: `[off|none|0|-|<dir>]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Directory for the precomputed Stage1 scalar cache, reused for matching B1 and exponent mode. Empty uses the executable directory; off, none, 0, or - disables caching. Explicit relative paths use the current working directory. Small B1 values may be computed without writing a cache.

指定 Stage1 指数缓存目录，复用相同 B1 和指数模式的预计算结果。留空使用可执行文件目录；off、none、0 或 - 关闭缓存。显式相对目录以进程当前工作目录为基准；很小的 B1 可能直接计算而不写缓存。

### finished

```toml
finished = "worktodo.finished.txt"
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Stage1 completed-task record, containing original assignments and results. This is not an intermediate curve checkpoint.

指定 Stage1 已完成任务的记录文件。用于保留原任务行和处理结果，不是曲线计算的中间存档。

### gpu_param

```toml
gpu_param = 3
```

Type / 类型: `int`; range / 范围: `[0|2|3]`; worker override / worker 覆盖: `[workers.N.stage1]`.

GPU Stage1 curve parametrization. CUDA Stage2 consumes param0 saves; select 0 for that handoff. OpenCL uses param3. The default is 3.

选择 GPU Stage1 曲线参数族。当前 CUDA Stage2 消费参数 0 的 save，准备交给它处理时使用 0；OpenCL 路径使用参数 3。默认值为 3。

### impl

```toml
impl = "auto"
```

Type / 类型: `string`; range / 范围: `[auto|simd|scalar]`; worker override / worker 覆盖: `[workers.N.stage1]`.

CPU Stage1 arithmetic implementation. auto selects by CPU capability and batch size; simd requires AVX512-IFMA; scalar uses GMP. Non-auto values are rejected for GPU runs.

CPU Stage1 算术实现：auto 根据 CPU 能力和批量选择；simd 要求 AVX512-IFMA；scalar 使用 GMP。GPU 运行时拒绝非 auto 值。

### kernel_add

```toml
kernel_add = ""
```

Type / 类型: `string`; range / 范围: `[auto|<id>|<alias>]`; worker override / worker 覆盖: `[workers.N.stage1]`.

OpenCL modular-addition kernel ID or alias. Empty or auto lets the backend choose; automatic selection is normally suitable. CUDA ignores this key.

指定 OpenCL 模加内核的标识或别名。空值或 auto 由后端选择；通常保持自动，CUDA 后端忽略此键。

### kernel_mul

```toml
kernel_mul = ""
```

Type / 类型: `string`; range / 范围: `[auto|<id>|<alias>]`; worker override / worker 覆盖: `[workers.N.stage1]`.

OpenCL modular-multiplication kernel ID or alias. Empty or auto lets the backend choose. Intended for comparing or debugging known kernels; automatic selection is normally suitable. CUDA ignores this key.

指定 OpenCL 模乘内核的标识或别名。空值或 auto 由后端选择；用于已知内核的比较和调优，通常保持自动。CUDA 后端忽略此键。

### kernel_special_mult

```toml
kernel_special_mult = ""
```

Type / 类型: `string`; range / 范围: `[auto|<id>|<alias>]`; worker override / worker 覆盖: `[workers.N.stage1]`.

OpenCL special-multiplication kernel ID or alias. Empty or auto lets the backend choose; automatic selection is normally suitable. CUDA ignores this key.

指定 OpenCL 特殊乘法内核的标识或别名。空值或 auto 由后端选择；通常保持自动，CUDA 后端忽略此键。

### kernel_sqr

```toml
kernel_sqr = ""
```

Type / 类型: `string`; range / 范围: `[auto|<id>|<alias>]`; worker override / worker 覆盖: `[workers.N.stage1]`.

OpenCL modular-squaring kernel ID or alias. Empty or auto lets the backend choose; automatic selection is normally suitable. CUDA ignores this key.

指定 OpenCL 模平方内核的标识或别名。空值或 auto 由后端选择；通常保持自动，CUDA 后端忽略此键。

### kernel_sub

```toml
kernel_sub = ""
```

Type / 类型: `string`; range / 范围: `[auto|<id>|<alias>]`; worker override / worker 覆盖: `[workers.N.stage1]`.

OpenCL modular-subtraction kernel ID or alias. Empty or auto lets the backend choose; automatic selection is normally suitable. CUDA ignores this key.

指定 OpenCL 模减内核的标识或别名。空值或 auto 由后端选择；通常保持自动，CUDA 后端忽略此键。

### lcm

```toml
lcm = "1"
```

Type / 类型: `string`; range / 范围: `[1|12]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Montgomery Stage1 scalar is lcm(1..B1) multiplied by 1 or 12. Edwards uses its fixed 48*lcm scalar and rejects 12. This choice changes the output point and must match the Stage2 save convention.

Montgomery Stage1 标量是 lcm(1..B1) 乘 1 或 12。Edwards 固定使用 48·lcm，不接受 12。此选择改变输出点，须与 Stage2 存档约定一致。

### log_file

```toml
log_file = "screen.log"
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

Append normal Stage1 output to this file. Empty disables file logging. When no filename is explicitly set, workers add a numeric suffix, such as screen_2.log.

指定 Stage1 普通运行日志，程序追加写入。留空关闭文件日志；未显式指定文件名时，worker 2 等自动使用 screen_2.log 这类带编号的文件名。

### naf_w

```toml
naf_w = 0
```

Type / 类型: `int`; range / 范围: `[0|3..12]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Edwards NAF window. 0 selects CPU 12 or CUDA GPU 6. CUDA accepts explicit windows 2..12. Larger windows reduce loop additions while the dictionary grows as 2^(w-2), increasing memory and setup time. Typical values are 3..12. CUDA Montgomery/PRAC ignores this setting.

设置 Edwards NAF 窗口。0 在 CPU 选择 12，在 CUDA GPU 选择 6；CUDA 显式窗口允许 2..12。增大窗口减少主循环加法，但字典按 2^(w-2) 增长，会增加内存和准备时间。通常使用 3..12；CUDA Montgomery/PRAC 不使用此设置。

### p95_add_workers

```toml
p95_add_workers = ""
```

Type / 类型: `string`; range / 范围: `[<n>|<n>,...|<a>-<b>|auto]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Prime95 workers eligible to receive handoffs: one ID, a comma list, or ranges. The shortest eligible queue is selected. auto reads NumWorkers from prime.txt in the same directory. Empty omits the worker header and uses Prime95's default worker.

选择接收交接任务的 Prime95 worker，可指定单个编号、编号列表或区间；程序在候选 worker 中选择队列较短者。auto 从同目录 prime.txt 读取 NumWorkers；留空追加时不写 worker 段头，按 Prime95 的默认 worker 处理。

### p95_dir

```toml
p95_dir = ""
```

Type / 类型: `string`; range / 范围: `<dir>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Legacy compatibility setting. A nonempty value only produces an optional hint; it does not enable the current Prime95 handoff mechanism. Use p95_worktodo_path to hand off ECMSTAGE2 work.

保留旧配置兼容性，设置非空值只产生旧选项提示，不启用新的 Prime95 交接流程。需要交接 ECMSTAGE2 任务时使用 p95_worktodo_path。

### p95_keep_aid

```toml
p95_keep_aid = true
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Keep PrimeNet assignment IDs in handoffs: 1 retains IDs not known to be invalid; 0 strips all IDs. IDs confirmed invalid by Prime95 logs are always removed to prevent repeated rejection.

控制交接任务是否保留 PrimeNet assignment ID。1 保留未被判定失效的 AID，0 一律去掉；已在 Prime95 日志中确认失效的 AID 会去掉，以免任务再次被丢弃。

### p95_recover_lost

```toml
p95_recover_lost = true
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Recover handoffs deleted by Prime95 because of invalid assignment IDs. 1 checks handoff records and Prime95 logs, then reposts confirmed lost assignments without an ID; 0 disables recovery. Only confirmed prior handoffs qualify.

控制是否补投被 Prime95 因失效 AID 删除的交接任务。1 根据交接记录和 Prime95 日志检查并去掉 AID 补投，0 关闭自动补投；只对程序能够确认的已交接任务生效。

### p95_worktodo_path

```toml
p95_worktodo_path = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Prime95 worktodo path for supported Montgomery Stage1 handoff. Edwards and GPU PARAM2 ignore this path and keep saves local; they do not access Prime95 transfer files.

用于受支持的 Montgomery Stage1 交接的 Prime95 worktodo 路径。Edwards 与 GPU PARAM2 自动忽略此路径，存档保留在本地，不访问 Prime95 转移文件。

### progress_color

```toml
progress_color = "cyan"
```

Type / 类型: `string`; range / 范围: `[none|red|green|yellow|blue|magenta|cyan|white|grey]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Color of the Stage1 console progress bar. none disables color. This does not change ordinary file logs or arithmetic.

选择 Stage1 控制台进度文字颜色。none 关闭着色；此设置不改变普通文件日志内容或计算方式。

### progress_log_seconds

```toml
progress_log_seconds = 60
```

Type / 类型: `double`; range / 范围: `<x:R,finite>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Interval in seconds between Stage1 progress entries in file logs. Positive values limit frequency; 0 suppresses intermediate progress; negative values log every refresh. Completion at 100% is still logged. Console refresh frequency is unchanged.

控制 Stage1 进度行写入日志文件的间隔，单位为秒。正数按间隔记录，0 不记录中间进度，负数记录每次进度更新；完成时的 100% 行仍会记录。此键不限制控制台刷新频率。

### reduction

```toml
reduction = "auto"
```

Type / 类型: `string`; range / 范围: `[auto|montgomery|mersenne-fold]`; worker override / worker 覆盖: `[workers.N.stage1]`.

CPU SIMD modular reduction. auto selects for the input; mersenne-fold requires N=2^p-1; montgomery uses general reduction. An explicit reduction requires impl=simd, so it cannot be silently ignored when auto chooses scalar.

CPU SIMD 模约减：auto 按输入数选择；mersenne-fold 要求 N=2^p-1；montgomery 使用通用约减。显式归约要求 impl=simd，避免 auto 选中标量时悄悄忽略设置。

### save_name_pattern

```toml
save_name_pattern = "{curve}_{target}_{b1}.save"
```

Type / 类型: `string`; range / 范围: `<pattern>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Automatic Stage1 save filename pattern for CPU and GPU. {curve}=e|m0|m2|m3; {target}=m<exponent> for Mersenne assignments, otherwise bit<actual_bits>; {n}=original assignment exponent when available, otherwise actual bits; {b1}=compact B1. Missing curve prefixes are added. In ECMSTAGE1 queues, typed save prefixes override TOML curve/param and the final filename field supplies B1; untyped names use the TOML construction. Finished/handoff lines name the actual prefixed save. Explicit CLI save paths are preserved. Empty restores the default.

统一设置 CPU/GPU Stage1 自动 save 文件名。{curve}=e|m0|m2|m3；{target}=梅森任务的 m<指数>，否则为 bit<实际位数>；{n}=原任务指数，无任务元数据时为实际位数；{b1}=紧凑 B1。缺少曲线前缀时自动补充。ECMSTAGE1 队列中，类型前缀覆盖 TOML 曲线/param，末尾文件名字段给出 B1；无类型前缀时沿用 TOML。完成/交接行使用实际带前缀的 save；CLI 显式 save 路径保持原样。留空恢复默认。

### save_sync_dir_1

```toml
save_sync_dir_1 = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage1]`.

First destination for Stage1 save synchronization. Empty disables this destination; sync_mode controls the files and synchronization policy.

指定第一个 Stage1 存档同步目录。留空不启用该目标；同步范围和方式由 sync_mode 决定。

### save_sync_dir_2

```toml
save_sync_dir_2 = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Second destination for Stage1 save synchronization. Both destinations may be enabled; empty disables only this destination.

指定第二个 Stage1 存档同步目录。可同时向两个目标同步；留空只关闭此目标，不影响第一个目标。

### sigma

```toml
sigma = 0
```

Type / 类型: `u64`; range / 范围: `<x:Z,0..2^64-1>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Initial curve sigma. 0 chooses a random value; consecutive curves use sigma+i without 64-bit wraparound. CUDA PARAM0 accepts the full 64-bit range, while PARAM2/3 batches must stay below 2^32. CPU runs warn if a sigma exceeds Prime95's signed reader range; this is not a Stage1 limit. A nonzero ECM/ECM2 assignment sigma overrides this setting.

曲线起始 sigma；0 表示随机生成，连续曲线使用 sigma+i 且不得在 64 位处回绕。CUDA PARAM0 接受完整 64 位范围，PARAM2/3 批次须位于 2^32 以下。CPU 路径的 sigma 超出 Prime95 有符号读取范围时会警告，这不是 Stage1 的限制。ECM/ECM2 任务行的非零 sigma 覆盖此键。

### stage1_results_file

```toml
stage1_results_file = "stage1_results.jsonl"
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

Canonical internal per-curve Stage1 JSONL with CRC32 and execution metadata, separate from submission output.

统一格式的逐曲线 Stage1 内部 JSONL，包含 CRC32 与执行元数据，与提交文件分开。

### stage1_submission_file

```toml
stage1_submission_file = "results.json.txt"
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

Stage1 submission JSON: seal NF once the whole task completes (B2=B1); publish verified factors immediately as individual curves. Private ledgers live under the output directory data/.

Stage1 提交 JSON：整条任务完成后合并 NF（B2=B1）；验证后的因子立即单曲线输出。私有账本位于输出目录的 data/ 下。

### stage1_threads

```toml
stage1_threads = 0
```

Type / 类型: `u32`; range / 范围: `<x:Z,0..2^32-1>`; worker override / worker 覆盖: `[workers.N.stage1]`.

CPU Stage1 thread count; 0 selects automatically. Applies only to CPU arithmetic and does not set CUDA threads per block or threads per integer.

设置 CPU Stage1 的线程数。0 自动选择；只影响 CPU 计算路径，不改变 CUDA 的线程块大小或每个大整数使用的线程数。

### sync_mode

```toml
sync_mode = "incremental"
```

Type / 类型: `string`; range / 范围: `[incremental|full]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Save synchronization after a Stage1 assignment: incremental copies files updated during that assignment; full copies all eligible saves. Startup always performs a full synchronization.

控制 Stage1 任务完成后的存档同步。incremental 仅同步本次任务期间更新的文件；full 同步全部符合条件的存档。程序启动时仍执行一次完整同步。

### target

```toml
target = "gpu"
```

Type / 类型: `string`; range / 范围: `[cpu|gpu]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Run Stage1 on CPU or on the GPU backend linked into this executable. Edwards GPU requires ecm_cuda; the target does not select between CUDA and OpenCL.

选择 Stage1 在 CPU 或本程序链接的 GPU 后端执行。Edwards GPU 需要 ecm_cuda；此键不在同一可执行文件内切换 CUDA 与 OpenCL。

### tmp_dir

```toml
tmp_dir = "."
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage1]`.

Output directory for Stage1 curve saves. Stage2 also searches here when stage2_save_dir is unset. This is not a Stage2 RAM or VRAM budget.

指定 Stage1 曲线存档的输出目录。Stage2 未设置 stage2_save_dir 时也在此目录查找任务引用的 save；此键不是 Stage2 全部临时显存或内存的预算。

### tpi

```toml
tpi = 8
```

Type / 类型: `u32`; range / 范围: `[1|2|4|8|16|32]`; worker override / worker 覆盖: `[workers.N.stage1]`.

OpenCL threads per integer, subject to limb-layout constraints. CUDA selects TPI by operand width and ignores this key. This is not the CUDA block size.

设置 OpenCL 每个大整数使用的线程数，需满足有效 TPI 与整数 limb 数的布局约束。CUDA 后端自行按位宽选择 TPI，不使用此键；不要用它调整 CUDA 的线程块大小。

### verbose

```toml
verbose = true
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage1]`.

Normal console detail. In Stage2, true shows phase transitions and false keeps curve summaries; stage2_debug_log independently controls debug logging. Stage1 uses the selected backend's normal verbosity behavior.

控制普通控制台输出的详细程度。Stage2 中 true 显示阶段切换，false 保留曲线摘要；调试日志由 stage2_debug_log 独立控制。Stage1 按各计算后端的普通详细输出规则处理。

### wg_size

```toml
wg_size = 0
```

Type / 类型: `int`; range / 范围: `<x:Z,x>=0>`; worker override / worker 覆盖: `[workers.N.stage1]`.

OpenCL work-group size. 0 lets the backend select automatically; explicit values must satisfy device and kernel limits. CUDA ignores this setting.

设置 OpenCL 工作组大小，0 由后端自动选择。显式值必须满足当前设备和内核的限制；CUDA 后端不使用此设置。

## [stage2]

### stage1_batch

```toml
stage1_batch = 1
```

Type / 类型: `u64`; range / 范围: `<x:Z,1..2^20>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Stage1 batch curve count associated with Auto B2 cost estimation. Selects or checks Stage1 cost conditions; it neither starts a Stage1 batch from Stage2 nor changes the assignment's curve count.

指定 Auto B2 成本估算对应的 Stage1 批次曲线数。它用于选择或核对 Stage1 的成本条件，不会让 Stage2 驱动启动该批 Stage1 计算，也不改变任务要处理的曲线数。

### stage1_cost_csv

```toml
stage1_cost_csv = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage2]`.

CSV Stage1 s/curve anchors for workflow Auto B2. Columns: container_bits or target_bits, tpi, curves, b1, mhz, seconds_per_curve; optional tpb and exponent. Missing widths may interpolate or extrapolate. Costs are already amortized.

总流程 Auto B2 的 Stage1 秒/曲线 CSV；列为 container_bits 或 target_bits、tpi、curves、b1、mhz、seconds_per_curve；可选 tpb、exponent。缺失位宽可插值或外推，成本已摊销。

### stage1_cost_mhz

```toml
stage1_cost_mhz = 0
```

Type / 类型: `u64`; range / 范围: `<x:Z,0..10000>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Stage1 target frequency for inverse-frequency cost scaling. 0 uses the CSV reference frequency; this does not change GPU clocks.

Stage1 成本反频率缩放的目标频率；0 使用 CSV 参考频率，不修改 GPU 时钟。

### stage1_seconds_per_curve

```toml
stage1_seconds_per_curve = 0
```

Type / 类型: `double`; range / 范围: `<x:R,finite,x>0>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Positive Stage1 seconds per curve for Auto B2, already amortized for the actual batch. Does not divide by stage1_batch again. Overrides Stage1 tune measurements. When unset, full ECM tune requires a matching stage1_tune_profile; the legacy component profile uses its own matched Stage1 data. Not a runtime limit or the tune input preparation time.

为 Auto B2 提供正的 Stage1 每曲线摊销秒数，应已包含实际批次收益，不再除以 stage1_batch。优先于 Stage1 tune 实测。未指定时，完整 ECM tune 需匹配 stage1_tune_profile；旧组件 profile 使用其自身匹配的 Stage1 数据。此值不是本次运行时限，也不是调优输入的准备耗时。

### stage1_tune_profile

```toml
stage1_tune_profile = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Checked completed CUDA Stage1 cost anchors, used after explicit seconds/curve and CSV. Matches batch, modulus type and exponent; B1 and missing widths scale by B1 and the dispatched container squared. Device/runtime reuse follows stage2_tune_ignore. This does not run Stage1.

已核验的完整 CUDA Stage1 成本锚点，优先级低于显式秒/曲线和 CSV。匹配批次、模数类型和指数模式；B1 与缺失位宽按 B1 和容器位宽平方估算。设备及运行时复用遵守 stage2_tune_ignore；此文件不启动 Stage1。

### stage2_arena_mb

```toml
stage2_arena_mb = 0
```

Type / 类型: `u64`; range / 范围: `<x:Z,0..2^20>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Stage2 large-workspace budget in MiB, used to plan available NTT shapes. 0 uses environment settings or automatic planning, not zero VRAM. Too little may force smaller shapes or prevent execution. Tables, coordinates, and other modules may allocate additional VRAM.

设置 Stage2 大工作区预算，单位为 MiB，用于规划可用的 NTT 工作形状。0 由运行环境或自动规划决定，不表示不使用显存；过小可能迫使较小工作形状或使任务无法运行。表、坐标和其他模块仍可能另占显存。

### stage2_auto_b2

```toml
stage2_auto_b2 = false
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Choose B2 when every source supplies zero. 1 optimizes workflow benefit per time including Stage1 cost; 0 disables. Requires usable Stage2 costs and positive Stage1 s/curve. B1 never gates tune reuse; stage2_tune_ignore controls other conditions. Memory and carrier divisibility are checked on the current input. Estimates retain uncertainty labels; the legacy .cprof interface keeps its original qualification checks.

所有来源均为零 B2 时自动选择。1 按计入 Stage1 成本的全流程单位时间收益选取；0 关闭。须有可用 Stage2 成本与正的 Stage1 秒/曲线。B1 不作为 tune 复用门槛；其他条件由 stage2_tune_ignore 控制。当前输入仍检查显存及承载整除关系；估算保留不确定性标记，旧 .cprof 接口保持原资格检查。

### stage2_auto_carrier

```toml
stage2_auto_carrier = true
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Add M=2^p-1 for validated ECMSTAGE2=1,2,p,-1 saves when N divides M and 2<=p<=16384. Compare ordinary cost at target_bits with Mersenne cost at arithmetic_bits=p, using stage2_prediction_policy. Full M_p already has native Mersenne arithmetic. Explicit carrier options take priority; a cofactor save needs a matching task to infer p.

对有效ECMSTAGE2=1,2,p,-1保存点，验证N整除M=2^p-1且2<=p<=16384后加入承载候选。普通成本按target_bits、梅森成本按arithmetic_bits=p查询，并遵循stage2_prediction_policy；完整M_p已使用原生梅森算术，不重复加入相同候选。显式承载选项优先；余因子save须匹配任务才能推断p。

### stage2_auto_max_b2

```toml
stage2_auto_max_b2 = 0
```

Type / 类型: `u64`; range / 范围: `<x:Z,0..2^63-8193>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Auto B2 upper search bound; 0 means 2.6e12. Unmeasured values may be estimated; explicit task B2 takes precedence.

Auto B2 搜索上界；0 为 2.6e12。未测值允许估算，显式任务 B2 优先。

### stage2_auto_min_b2

```toml
stage2_auto_min_b2 = 0
```

Type / 类型: `u64`; range / 范围: `<x:Z,0..2^63-8193>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Auto B2 lower search bound; 0 means B1+1. Unmeasured values may be estimated; explicit task B2 takes precedence.

Auto B2 搜索下界；0 为 B1+1。未测值允许估算，显式任务 B2 优先。

### stage2_b2

```toml
stage2_b2 = 0
```

Type / 类型: `u64`; range / 范围: `<x:Z,x=0|B1<x<=2^63-1-8192>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Stage2 upper bound when neither CLI nor task supplies a nonzero B2. 0 defers to other sources. Explicit B2 must exceed the save's B1. Larger B2 usually increases coverage, time, and VRAM demand. Precedence: nonzero CLI B2, then task B2, then this key.

设置未在命令行或任务行给出非零 B2 时使用的 Stage2 上界。0 表示继续从其他来源选择；显式 B2 必须大于存档的 B1。增大 B2 通常增加搜索范围、运行时间及显存需求；命令行非零 B2 优先于任务行，任务行优先于此键。

### stage2_batch_mb

```toml
stage2_batch_mb = 64
```

Type / 类型: `u64`; range / 范围: `<x:Z,1..2^20>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Stage2 S4 batch-buffer budget in MiB. Larger budgets may reduce batching overhead but consume more VRAM. This limits the corresponding buffers, not total process VRAM.

设置 Stage2 S4 批次的工作缓冲预算，单位为 MiB。增大预算可能减少分批开销，但会占用更多显存；它只约束对应批次缓冲，不是整个进程的显存上限。

### stage2_cost_profile

```toml
stage2_cost_profile = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Legacy .cprof cost profile for Auto B2 when stage2_tune_profile is empty. Checks the original binary/device/arithmetic and calibration contract; new full ECM tune does not relax those checks. Empty supplies no legacy profile.

stage2_tune_profile 为空时供 Auto B2 使用的旧 .cprof 成本配置。保持原二进制、设备、算术及标定资格检查；新的完整 ECM tune 不放宽旧合同。留空不提供旧配置。

### stage2_d

```toml
stage2_d = 0
```

Type / 类型: `u64`; range / 范围: `<x:Z,x=0|x>=6,x%2=0>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Manual Stage2 step D. 0 selects the shape automatically. Explicit values must be even, at least 6, and satisfy the engine's layout constraints. D affects tree size, batch count, and memory demand; automatic selection is normally suitable.

手动指定 Stage2 算法步长 D。0 自动选形；非零值除满足偶数且不小于 6 外，还必须满足当前引擎的布局限制。D 会改变树规模、批次数和内存需求，通常保持自动。

### stage2_debug_log

```toml
stage2_debug_log = false
```

Type / 类型: `bool`; range / 范围: `[true|false|1|0|yes|no|on|off]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Write Stage2 debug details to a separate log: true enables and false disables. verbose=true does not enable debug logging; enabling this option does not send all debug details to the console.

控制是否将 Stage2 调试细节写入独立日志。true 开启，false 关闭；verbose=true 不会自动开启调试日志，开启本选项也不把全部调试输出刷到控制台。

### stage2_debug_log_file

```toml
stage2_debug_log_file = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Separate Stage2 debug log, written only when stage2_debug_log is enabled. Empty restores the default filename. Default names gain a worker suffix; explicit paths are used unchanged.

指定 Stage2 独立调试日志文件，仅在 stage2_debug_log 开启时写入。留空恢复默认文件名；默认名称按 worker 加后缀，显式指定时按原路径使用。

### stage2_device

```toml
stage2_device = -1
```

Type / 类型: `int`; range / 范围: `[-1|0..2^31-1]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Stage2 CUDA device ID. -1 inherits Stage1's device; 0 or greater selects a CUDA device explicitly.

Stage2 CUDA 设备编号。-1 继承 Stage1 的 device；0 及以上显式选择 CUDA 设备。

### stage2_factor_only

```toml
stage2_factor_only = false
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Skip diagnostic prime attribution, which identifies Stage2 primes responsible for a hit. 1 skips attribution while retaining factor extraction and validation; 0 follows the engine's diagnostic policy. This does not disable decomposition or change stage2_factorize_hits.

控制是否跳过命中结果的素数归因诊断，即查找哪些 Stage2 素数对应此次命中。1 跳过此诊断，仍执行因子提取与验证；0 使用引擎默认诊断策略。此键不表示关闭因子分解，也不改变 stage2_factorize_hits。

### stage2_factor_timeout

```toml
stage2_factor_timeout = 30
```

Type / 类型: `u64`; range / 范围: `<x:Z,1..600>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Timeout in seconds for each external GP fallback decomposition call. Local PARI calls have a 5-second process deadline and a 10-second cumulative session budget per curve worker. Explicit GMP analysis remains bounded to 2 seconds per call. These diagnostics never change the original factor or successful Stage2 computation status.

设置备选外部 GP 分解调用的超时秒数。本地 PARI 单次进程限 5 秒、单曲线 worker 会话累计限 10 秒；显式选择 GMP 时单次限 2 秒。附加诊断不改变原始因子或 Stage2 成功状态。

### stage2_factorize_hits

```toml
stage2_factorize_hits = false
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Further decompose found factors: 1 uses the shared diagnostic service (local PARI DLL first, external GP second); GMP is never selected automatically. 0 keeps only factors extracted and verified by Stage2. Only complete, proven decomposition sets factorization_complete. Unfinished diagnostics preserve the original factor and successful curve status.

控制命中因子的附加分解。1 使用共享诊断层，优先本地 PARI DLL，备选外部 GP，不自动使用 GMP 算法；0 只保留 Stage2 已得到并验证的因子。仅完整且全部已证素的分解标为完成；诊断未完成仍保留原始因子与曲线成功状态。

### stage2_finished

```toml
stage2_finished = "worktodo.stage2.finished.txt"
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

Record of completed or invalid Stage2 assignments from the shared worktodo. Stores original tasks, result status, and commit receipts. Must be distinct from Stage1 finished and the active queue.

记录共用 worktodo 中已完成或输入错误的 Stage2 任务，保存原任务、结果状态及提交回执。路径必须不同于 Stage1 finished 和活动队列。

### stage2_fold_mb

```toml
stage2_fold_mb = 640
```

Type / 类型: `u64`; range / 范围: `<x:Z,0..2^20>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

Budget in MiB for keeping fold accumulators on the GPU. 0 disables this residency. Exceeding the budget may fall back to batching, substantially increasing processing and transfer costs. Independent of arena and batch budgets; not a total VRAM limit.

设置 fold 累积结果驻留 GPU 的预算，单位为 MiB。0 关闭这部分驻留；所需空间超过预算时可退回分批路径，可能显著增加处理和传输开销。此预算与大工作区、批次缓冲分别控制，不是总显存限制。

### stage2_gp

```toml
stage2_gp = ""
```

Type / 类型: `string`; range / 范围: `<exe>`; worker override / worker 覆盖: `[workers.N.stage2]`.

External PARI/GP fallback executable for optional factor decomposition. The local PARI DLL is tried first. gp.exe follows Windows executable search rules; a full path is accepted. GP failure preserves partial PARI results and never enables GMP automatically. Dataset scripts retain their separate --gp resolution rules.

指定附加因子分解的外部 PARI/GP 备选程序，默认先使用本地 PARI DLL。gp.exe 按 Windows 程序搜索规则查找，也接受完整路径。GP 失败保留 PARI 部分结果，不自动使用 GMP；数据集脚本继续使用各自的 --gp 路径规则。

### stage2_log_file

```toml
stage2_log_file = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

Append normal Stage2 phases, batches, warnings, and final statistics. Empty disables this file log; console output remains. Default filenames gain a worker suffix unless an explicit filename is supplied.

指定 Stage2 普通运行日志，追加记录阶段、批次、警告和结束统计。留空关闭普通文件日志，控制台仍输出；未显式指定时，多 worker 自动给默认文件名加编号后缀。

### stage2_log_level

```toml
stage2_log_level = -1
```

Type / 类型: `int`; range / 范围: `[quiet|curve|phases|batches|debug|0..4]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Legacy console level: quiet, curve, phases, batches, and debug map to 0..4. Unset follows verbose. In production, debug enables the separate debug log while normal console detail remains capped at batches. Usually use verbose and stage2_debug_log.

兼容旧配置的控制台日志等级：quiet、curve、phases、batches、debug 分别对应 0..4。未配置时由 verbose 决定；debug 在生产程序中开启独立调试日志，普通控制台最高保持 batches。一般只需使用 verbose 和 stage2_debug_log。

### stage2_ntt_counts

```toml
stage2_ntt_counts = false
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Print a per-curve NTT length count table at the end of production Stage2: 1 enables; 0 disables collection and output. Counts mathematical transforms (two forwards and one inverse per convolution), including polynomial initialization and main work, excluding Stage1 and independent arithmetic selftests. Shared by Edwards and Montgomery; no GPU counters or extra synchronization. CLI --ntt-counts overrides the selected worker TOML.

生产 Stage2 每条曲线结束时输出 NTT 长度计数表：1 开启，0 关闭采集与输出。统计数学变换次数，每次卷积含两个正向与一个逆向；包含多项式初始化及主计算，不含 Stage1 或底层独立算术自检。Edwards/Montgomery 通用，不增加设备计数或额外同步。CLI --ntt-counts 覆盖选定 worker 的 TOML。

### stage2_pack_zero

```toml
stage2_pack_zero = true
```

Type / 类型: `bool`; range / 范围: `[false|true]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

Fuse input digit packing and zero padding in one GPU kernel per operand. Covers contiguous, resident gather and reverse inputs; unsupported shapes use the original packer. Does not change arithmetic checks, D, NTT length or device workspace capacity. CLI --pack-zero overrides this key; an explicit key overrides NTT_S4_PACK_ZERO. Without an override, the built-in default applies; legacy .cprof cost profiles retain false and require false explicitly if this key is set.

每个操作数用一个GPU内核完成digit打包和补零，覆盖连续输入、设备驻留gather及反向输入；不支持的形状回退原packer。不改变算术检查、D、NTT长度或设备工作区容量。命令行--pack-zero优先于本键，显式TOML优先于NTT_S4_PACK_ZERO；均未指定时使用默认值。旧.cprof成本配置保留false；显式配置本键时须设false。

### stage2_prediction_policy

```toml
stage2_prediction_policy = "model"
```

Type / 类型: `string`; range / 范围: `[model|paired]`; worker override / worker 覆盖: `[workers.N.stage2]`.

model: prefer exact paired measurements, otherwise compare generic and Mersenne arithmetic-width region estimates with two-sided uncertainty and current memory admission. Missing tune data uses an explicitly estimated 4060-Laptop/1800-MHz basic model. Uncovered NTT/residency regions retain the default. paired: require the original exact-width/B2 measured win. B1 never gates reuse.

model：优先使用精确配对实测；其他输入分别按通用、梅森算术位宽区间预测，满足双侧误差保护及当前显存准入才替换。缺调优资料时使用明确标为估算的4060 Laptop/1800MHz基础模型；未覆盖的NTT/驻留区间保留默认。paired恢复原精确位宽/B2实测收益门禁。B1不作为复用门槛。

### stage2_progress_dir

```toml
stage2_progress_dir = ""
```

Type / 类型: `string`; range / 范围: `<dir>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Directory for per-task .ecm_ckpt_stage2_<bits>_<task-id>.dat progress files. Empty uses the legacy progress file's directory if explicitly set, otherwise the queue directory. Task IDs bind AID, N, save contents, selected curve range and B2; reordering tasks, changing device/memory/log paths or tuning D/carrier does not reset completed curves. Files contain counts and receipt metadata only, and are removed after the task is published and advanced.

每任务 .ecm_ckpt_stage2_<bits>_<task-id>.dat 进度文件的目录。空值使用显式旧进度路径的父目录，否则使用队列目录。任务 ID 绑定 AID、N、save 内容、曲线范围和 B2；调整任务顺序、设备/显存/日志路径或 D/承载调优不会重置已完成曲线。文件仅保存计数和回执元数据，任务发布并移出队列后清除。

### stage2_progress_file

```toml
stage2_progress_file = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Legacy single-file progress import path. Empty checks the old queue-path plus worker-suffix .progress file. A matching task's completed count and pending receipt are imported into its own .ecm_ckpt_stage2_<bits>_<task-id>.dat file. An unrelated legacy task is retained and does not stop this task. New progress files use stage2_progress_dir; no arithmetic state is checkpointed.

旧单文件进度的导入路径。空值检查旧的队列路径加 worker 后缀 .progress 文件；匹配任务的完成数和待确认回执迁入独立 .ecm_ckpt_stage2_<bits>_<task-id>.dat 文件。不相关的旧任务保留，当前任务继续。新进度文件使用 stage2_progress_dir，不保存曲线内部算术状态。

### stage2_ratio_adjust

```toml
stage2_ratio_adjust = 1
```

Type / 类型: `double`; range / 范围: `<x:R,finite,x>0>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Multiplier applied to Stage2 time in the Auto B2 model. 1 leaves it unchanged; greater than 1 estimates higher cost, less than 1 lower cost, usually changing selected B2. Adjusts planning cost, not actual GPU speed.

将 Auto B2 模型中的 Stage2 时间乘以该系数。1 不调整，大于 1 表示估计 Stage2 更贵，小于 1 表示更便宜；通常会相应影响所选 B2。此键只调整选形成本，不会让 GPU 按比例变快或变慢。

### stage2_relocation

```toml
stage2_relocation = false
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Optional production GPU Stage2 conservative relocation. Keeps B2 and the original giant endpoint, uses a shorter suffix for clean NF, and replays the original interval on any hit or target-N nonunit (including GCD=N). Rebalances short point chunks on whole G boundaries. Default 0 preserves the original algorithm. CLI --relocation overrides this worker TOML. D/AutoB2 and memory admission use original-interval estimates; relocation does not collect ECM tune samples or run startup ECM calibration, and explicit --tune ecm is unavailable in this mode. Pure NTT samples remain reusable.

可选的生产 GPU Stage2 保守 relocation。保持 B2 和原 giant 终点，干净 NF 使用较短后缀；命中或出现目标 N 非单位（包括 GCD=N）时重放原区间。短点块按完整 G 边界重分。默认 0 保留原算法，CLI --relocation 覆盖当前 worker 的 TOML。D/AutoB2 和内存准入使用原区间估算；本模式不采集 ECM 调优样本、不执行启动 ECM 标定，并禁用显式 --tune ecm；纯 NTT 样本仍可复用。

### stage2_results_file

```toml
stage2_results_file = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage2]`.

JSONL results for successful Stage2 curves, including factors and resume receipts. Empty uses the default filename. Do not independently delete or replace it while resuming an unfinished queue; it is needed to reconcile completed curves.

指定 Stage2 成功曲线结果的 JSONL 文件，包含因子信息和续跑回执。留空使用默认文件名；不要在未完成队列续跑期间单独删除或替换它，否则会失去回执对账依据。

### stage2_save_dir

```toml
stage2_save_dir = ""
```

Type / 类型: `string`; range / 范围: `<dir>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Directory searched for saves referenced by Stage2 assignments. Empty inherits tmp_dir; absolute save paths in assignments are used directly. Source saves are read and validated without modification.

指定 Stage2 查找任务所引用 save 的目录。留空继承共享 tmp_dir；任务给出绝对 save 路径时直接使用该路径。程序读取并验证源 save，不修改其内容。

### stage2_short_calibration

```toml
stage2_short_calibration = true
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Calibrate once when no suitable ECM tune file exists, required components are missing, or the task has no reusable calibrated width coverage. Partial coverage permits estimates.

无合适 ECM tune 文件、必要组件缺失或任务位宽完全没有可复用校准覆盖时短校准；部分覆盖允许估算。

### stage2_short_calibration_seconds

```toml
stage2_short_calibration_seconds = 10
```

Type / 类型: `u64`; range / 范围: `<x:Z,1..600>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Short calibration measurement budget, excluding first CUDA initialization; finish the current measurement before stopping.

短校准测量预算，不含首次 CUDA 初始化；到期完成当前测量后停止。

### stage2_short_ntt_batch_tune

```toml
stage2_short_ntt_batch_tune = false
```

Type / 类型: `bool`; range / 范围: `[false|true]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Experimental short NTT batch selection from stage2_tune_profile, for lengths up to 2^19. Minimizes estimated convolution time including the tail, within the current chunk budget; requires at least two measured batch sizes and a predicted gain above 10%. Does not tune kernel configuration. Pure NTT gains do not guarantee faster full Stage2; default false pending full-engine validation.

实验性短 NTT 批量选择：读取 stage2_tune_profile，作用于不超过 2^19 的长度。在当前分块预算内按含尾批的预计卷积总时间选型；至少需要两个实测 batch，预计收益须超过 10%。不改变内核配置。纯 NTT 收益不保证完整 Stage2 更快；完成整机验证前默认 false。

### stage2_submission_file

```toml
stage2_submission_file = "results.json.txt"
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。

Submission-only ECM JSON lines, separate from the internal results JSONL. Uses a reference FFT size from Prime95 mult.asm. Personal identifiers and checksums belong only in local ignored files.

仅含提交字段的ECM JSON行，与内部结果JSONL分开；FFT长度参考Prime95 mult.asm。个人标识与校验值仅保存在本地忽略文件中。

### stage2_target_factor_bits

```toml
stage2_target_factor_bits = 0
```

Type / 类型: `u64`; range / 范围: `<x:Z,0|2..16384>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Target prime-factor size for Dickman semismooth success probability. 0 derives a recommendation from B1; this is distinct from the full modulus size.

Dickman 半光滑成功概率的目标素因子位数；0 根据 B1 推荐，与完整模数位宽不同。

### stage2_tune_arithmetic

```toml
stage2_tune_arithmetic = "both"
```

Type / 类型: `string`; range / 范围: `[both|generic|mersenne]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Arithmetic families for explicit width-grid ECM tune: both, generic probable-prime N, or Mersenne 2^p-1 arithmetic after stripping proper-exponent inherited factors and small target divisors. p is the requested grid arithmetic width; actual cofactor width is recorded separately. Real worktodo/save pairs continue to follow stage2_auto_carrier. Invalid or factor-producing synthetic Mersenne fixtures are skipped, never saved as timing measurements.

显式位宽网格ECM tune的算术家族：both同时测量；generic使用通用可能素数；mersenne使用去除真因子指数继承因子及小因子后的2^p-1承载算术。p为网格要求的算术位宽，实际余因子位宽另行记录。真实worktodo/save配对仍遵循stage2_auto_carrier；无效或产生因子的合成梅森样本跳过，不写成计时数据。

### stage2_tune_budget_seconds

```toml
stage2_tune_budget_seconds = 1800
```

Type / 类型: `u64`; range / 范围: `<x:Z,1..86400>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Tuning budget including input preparation, measurements and optional warmups. Start each width/B2 at the default model's D, compare neighbors, and follow only measured improvements above 2% plus the MAD noise margin. Reuse completed candidates. Finish an in-flight measurement after the deadline and retain valid summaries.

调优预算，包含输入准备、测量及显式开启的预热。每个位宽/B2 从默认模型的 D 开始比较邻档，仅沿超过 2% 且跨过 MAD 噪声界限的实测收益移动。复用已测候选；到期完成正在运行的测量并保留有效摘要。

### stage2_tune_condition_tag

```toml
stage2_tune_condition_tag = ""
```

Type / 类型: `string`; range / 范围: `<text>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Optional measurement-condition label, e.g. a fixed clock or power setting. Distinct labels preserve separate fastest summaries; ignored environment allows reuse.

可选测量条件标签，例如固定频率或功耗。不同标签保留独立最快汇总；忽略 environment 时允许复用。

### stage2_tune_error_limit

```toml
stage2_tune_error_limit = 0.08
```

Type / 类型: `double`; range / 范围: `<x:R,0<x<=1>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Relative error limit for qualifying the nonnegative full-curve cost model by leave-one-out validation. Sparse estimates keep uncertainty labels and cannot override the default D. Local descent compares measured medians and MAD rather than prediction error.

非负完整曲线成本模型的留一验证相对误差门限。稀疏估算保留不确定性标记，不能覆盖默认 D；局部下降比较实测中位数与 MAD，不用预测误差决定移动。

### stage2_tune_ignore

```toml
stage2_tune_ignore = "gpu,driver,cuda,backend,environment"
```

Type / 类型: `string`; range / 范围: `[gpu|driver|cuda|backend|memory|environment],...`; worker override / worker 覆盖: `[workers.N.stage2]`.

Comma-separated conditions ignored for profile reuse. Empty enforces every condition. Memory normally matches execution path; arithmetic legality and current memory admission are always checked.

逗号分隔的复用忽略项；留空核对全部条件。memory 默认匹配执行路径。算术合法性及当前显存准入始终检查。

### stage2_tune_profile

```toml
stage2_tune_profile = ""
```

Type / 类型: `string`; range / 范围: `<path>`; worker override / worker 覆盖: `[workers.N.stage2]`.

Format-7 final ECM matrix TOML. Arithmetic-family lookup uses arithmetic_bits; generic and Mersenne data are independent, and native full-Mersenne data can calibrate carriers. target_bits and carrier_exponent retain the measurement identity. stage2_prediction_policy controls estimated replacement. Missing data permits the basic model; B1 never gates reuse. Empty selects adjacent stage2_ecm_tune.toml unless a legacy cost profile is supplied.

格式7最终ECM矩阵TOML。按arithmetic_bits分别查询通用、梅森算术数据，原生完整梅森数数据也可校准承载；target_bits和carrier_exponent保留测量身份。估算替换遵循stage2_prediction_policy，缺数据可使用基础模型；B1不作为复用门槛。留空且未指定旧成本配置时使用相邻stage2_ecm_tune.toml。

### stage2_tune_runtime

```toml
stage2_tune_runtime = true
```

Type / 类型: `bool`; range / 范围: `[false|true]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Accumulate checked no-factor production timings by device, environment and execution shape. Identical complete parameters share one matrix row with repeats, mean time and sample standard deviation; slower observations count too. Row version and UTC update time record provenance only; software upgrades retain compatible timings. Model selection tests a legal Mersenne carrier first, then ordinary and promising D candidates. At least two observations enable comparison with switching hysteresis. B1 does not gate cost reuse. Writes are locked and atomic; cache failure does not fail a completed curve. false disables runtime trials, collection and writes.

按设备、环境与执行形状累计通过检查的无因子生产计时。完全相同参数合并为一条matrix记录，保存repeats、平均耗时和样本标准差，包含较慢观察。行内版本与UTC更新时间仅记录来源，程序升级保留可兼容计时。model选型优先试跑合法梅森承载，再比较普通路径和有希望的D；至少两次观察后按均值、标准差及切换余量比较。B1不限制成本复用；锁定后原子写回，缓存失败不使已完成曲线失败。false关闭运行试跑、采集与写回。

### stage2_tune_runtime_samples

```toml
stage2_tune_runtime_samples = 10
```

Type / 类型: `u64`; range / 范围: `[2..1000]`; worker override / worker 覆盖: `[workers.N.stage2]`.

Maximum independent runtime NF observations per identical complete parameters (including B1) and matching device/environment/arithmetic/path, including slower observations and compatible software versions. Identical parameters merge into repeats and mean/stddev; the limit counts observations, not matrix rows. Different parameter rows are retained. At least two observations enable measured selection. Lowering this limit preserves existing evidence. CLI: --tune-runtime-samples N.

完全相同参数（含B1）及相同设备、环境、算术和执行形状的独立运行NF观察上限，包含较慢观察及可兼容的其他程序版本。相同参数合并repeats和均值/标准差；上限统计观察次数，不统计matrix行数；不同参数行保留。至少两次用于实测选型；调低上限不删除已有证据。CLI：--tune-runtime-samples N。

### stage2_tune_warmup

```toml
stage2_tune_warmup = false
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `[workers.N.stage2]`.

0 starts formal ECM/NTT tuning and startup calibration without a discarded warmup. 1 runs one checked warmup per shape before formal repeats; it consumes the same budget and is excluded from the median/MAD. CLI --tune-warmup 0|1 overrides this key. Mandatory arithmetic checks are always performed.

0 表示 ECM/NTT 调优及启动短校准直接开始正式测量，不执行丢弃的预热轮次；1 表示每个形状先运行一轮经过核验的预热，占用相同预算但不计入中位数/MAD。CLI --tune-warmup 0|1 覆盖此键；必要算术检查始终执行。

## [gui]

### dock_layout

```toml
dock_layout = ""
```

Type / 类型: `string`; range / 范围: `<escaped_blob>`; worker override / worker 覆盖: `no / 不支持`.

Escaped GUI docking-layout data. Empty uses the default layout; the GUI writes it automatically. Incomplete manual edits may prevent layout restoration.

保存 GUI 面板停靠布局的转义数据。留空使用默认布局；由 GUI 自动写回，手动修改不完整数据可能使布局无法恢复。

### dock_layout_ver

```toml
dock_layout_ver = 0
```

Type / 类型: `int`; range / 范围: `<x:Z,x>=0>`; worker override / worker 覆盖: `no / 不支持`.

Version of the saved docking layout, managed by the GUI. Older layouts are rebuilt using the current default. This does not select a GUI feature version.

保存已写入布局的版本编号，由 GUI 管理。版本早于当前默认布局时会重建布局；此键不是用户选择 GUI 功能版本的开关。

### exe

```toml
exe = ""
```

Type / 类型: `string`; range / 范围: `<exe>`; worker override / worker 覆盖: `no / 不支持`.

Stage1 worker executable. Relative paths resolve beside the GUI; bare names can use PATH. Empty searches ecm_cuda.exe then ecm.exe. The worktodo prefix selects this executable or stage2_exe for each worker.

Stage1 工作程序。相对路径以 GUI 目录为基准，单独文件名也可经 PATH 查找；留空依次查找 ecm_cuda.exe、ecm.exe。每个 worker 根据 worktodo 前缀选择此程序或 stage2_exe。

### exit_confirm

```toml
exit_confirm = "ask"
```

Type / 类型: `string`; range / 范围: `[ask|stop|kill]`; worker override / worker 覆盖: `no / 不支持`.

Behavior when closing the GUI with active workers: ask prompts, stop requests a safe stop, kill terminates immediately. Stage1 saves a checkpoint; Stage2 finishes its current curve and records completed progress. Forced termination may lose the unfinished Stage1 work or current Stage2 curve.

控制关闭GUI时如何处理运行中的worker：ask先询问，stop请求安全停止，kill立即终止。Stage1写检查点后退出；Stage2完成当前曲线、记录完成进度后退出。强制终止可能丢失未存的Stage1计算或当前Stage2曲线。

### font

```toml
font = ""
```

Type / 类型: `string`; range / 范围: `<ttf|ttc>`; worker override / worker 覆盖: `no / 不支持`.

GUI .ttf or .ttc font file. Empty selects a system font for the interface language. A Chinese interface requires a font with Chinese glyphs.

指定 GUI 使用的 .ttf 或 .ttc 字体文件。留空由程序按界面语言选择系统字体；需要中文界面时应使用包含中文字形的字体。

### font_size

```toml
font_size = 0
```

Type / 类型: `float`; range / 范围: `[auto|<x:R,6..96>]`; worker override / worker 覆盖: `no / 不支持`.

GUI font size in pixels; fractional values are accepted. auto or empty selects 15*DPI scale. Values below 6 or nonfinite values restore auto; values above 96 clamp to 96. Manual sizes bypass automatic sizing.

设置 GUI 字号，单位为像素，可使用小数。auto 或空值按 15*DPI 比例自动选择；小于 6 或非有限数恢复自动，大于 96 按 96 处理。手动字号不会再按自动规则选择大小。

### font_snap

```toml
font_snap = true
```

Type / 类型: `bool`; range / 范围: `[0|1]`; worker override / worker 覆盖: `no / 不支持`.

Snap glyphs to the pixel grid: 1 enables, 0 disables. Compare clarity at fractional sizes or different DPI scales. off, false, and no also disable; other values enable. String matching is case-sensitive.

控制字体是否对齐像素网格。1 开启，0 关闭；小数字号或不同 DPI 下可比较显示清晰度。兼容 off、false、no 关闭，其他值按开启处理，字符串区分大小写。

### gpu_clock_ranges

```toml
gpu_clock_ranges = ""
```

Type / 类型: `string`; range / 范围: `<GPU-UUID:minMHz-maxMHz[:auto];...>`; worker override / worker 覆盖: `no / 不支持`.

GUI v2 GPU core-clock drafts keyed by NVML GPU UUID: GPU-<UUID>:minMHz-maxMHz, separated by semicolons. Optional :auto enables one startup apply, every local whole hour, and a one-shot reapply after each continuous 5-second out-of-range episode. Detection reuses the monitor samples. Without :auto, loading never changes hardware. Saving does not apply drafts in this session; Apply activates the draft range and auto option. Reset disables auto in this session without erasing saved drafts. Empty saves none; worker computation settings are unaffected.

GUI v2 的 GPU 核心频率草稿：GPU-<UUID>:最小MHz-最大MHz，多卡用分号分隔。可选 :auto 启用启动时一次、每个本地整点、以及每次连续越界 5 秒后的单次自动应用；检测复用监控采样。无 :auto 时载入不修改硬件。保存不应用本次草稿；点击应用才启用本次范围和自动选项。恢复默认关闭本次会话自动应用，不删除保存草稿。空值不保存范围，不改变 worker 计算配置。

### gpu_poll_ms

```toml
gpu_poll_ms = 500
```

Type / 类型: `int`; range / 范围: `<x:Z,100..2^31-1>`; worker override / worker 覆盖: `no / 不支持`.

GPU monitoring interval in milliseconds, at least 100. Larger values reduce NVML polling frequency. Affects monitoring updates only, not compute-kernel scheduling.

设置 GPU 监控查询间隔，单位为毫秒，最小为 100。增大可减少 NVML 查询频率；此键只影响监控数据更新，不控制计算 kernel 的调度。

### graceful_stop_ms

```toml
graceful_stop_ms = 300000
```

Type / 类型: `i64`; range / 范围: `<x:Z,1000..3600000>`; worker override / worker 覆盖: `no / 不支持`.

Maximum wait in milliseconds after requesting a safe worker stop. Process exit ends the wait; workers still running at the deadline are terminated. Allow enough time for Stage1 checkpoint writes or the full current Stage2 curve. Stage2 curve counters are not intermediate arithmetic saves.

请求安全停止后等待的最长时间，单位毫秒。进程退出结束等待；到期仍运行的worker强制终止。应为Stage1检查点写入或当前Stage2整条曲线留足时间；Stage2曲线计数不是中间算术存档。

### group_order_gp

```toml
group_order_gp = ""
```

Type / 类型: `string`; range / 范围: `<path-to-gp.exe|empty>`; worker override / worker 覆盖: `no / 不支持`.

External PARI/GP fallback executable for the Group Order Factor GUI. Auto tries the local PARI DLL first; empty searches gp in PATH only when GP is needed. GUI-only path setting; compute workers use the same diagnostic service with independent GP paths.

群阶分解界面使用的外部 PARI/GP 备选程序路径。Auto 默认先使用本地 PARI DLL；空值在需要 GP 时从 PATH 查找。此路径仅供 GUI 使用，计算 worker 使用相同诊断层，但 GP 路径独立设置。

### language

```toml
language = "english"
```

Type / 类型: `string`; range / 范围: `<xml-stem>`; worker override / worker 覆盖: `no / 不支持`.

GUI language, specified by the language XML filename stem in localization, without .xml. Missing resources or unavailable glyphs follow the GUI's display fallback rules.

选择 GUI 界面语言，值为 localization 目录下语言 XML 文件的文件名主体，不带 .xml。找不到所需资源或字体不能显示该语言时，按 GUI 的回退规则使用可显示的界面。

### localization_dir

```toml
localization_dir = ""
```

Type / 类型: `string`; range / 范围: `<dir>`; worker override / worker 覆盖: `no / 不支持`.

Directory containing GUI language XML files. Empty uses localization beside the GUI executable. Development builds also try the source-tree language resources.

指定 GUI 语言 XML 文件目录。留空使用 GUI 可执行文件旁的 localization 目录；从开发构建目录运行时还会尝试源码中的语言资源目录。

### NumWorkers

```toml
NumWorkers = 1
```

Type / 类型: `int`; range / 范围: `<x:Z,1..64>`; worker override / worker 覆盖: `no / 不支持`.

Number of workers managed by the GUI, each using its [workers.N] settings. Does not change task curve counts in worktodo. Concurrent workers should have separate queues and output paths.

设置 GUI 管理的 worker 数量。每个 worker 使用对应的 [workers.N] 设置；此键不会修改 worktodo 中每项任务的曲线数。并行运行时应给各 worker 配置独立队列和输出路径。

### priority

```toml
priority = "below_normal"
```

Type / 类型: `string`; range / 范围: `[idle|below_normal|normal|above_normal|high]`; worker override / worker 覆盖: `no / 不支持`.

Priority of worker processes launched by the GUI. below_normal is the default; lower priority makes concurrent desktop use easier. This is not a GPU exclusivity or utilization setting.

设置 GUI 启动的 worker 进程优先级。below_normal 为默认，较低优先级便于同时使用电脑；此键不是 GPU 独占或显卡占用率设置。

### refresh_hz

```toml
refresh_hz = 10
```

Type / 类型: `int`; range / 范围: `<x:Z,1..60>`; worker override / worker 覆盖: `no / 不支持`.

Idle GUI frame and worker-status refresh target in Hz. Input messages wake the GUI promptly; higher values smooth charts and status updates at additional UI cost. GPU metrics are sampled independently at gpu_poll_ms, and neither setting changes computation speed.

GUI 空闲时绘制帧与 worker 状态处理的目标频率，单位为 Hz。输入消息会及时唤醒界面；提高此值可使图表和状态显示更平滑，但增加界面开销。GPU 指标按 gpu_poll_ms 独立采样，两项设置均不改变计算速度。

### stage2_exe

```toml
stage2_exe = ""
```

Type / 类型: `string`; range / 范围: `<exe>`; worker override / worker 覆盖: `no / 不支持`.

Shared executable path for Stage2 workers. Empty searches ecm_cuda_stage2.exe beside the GUI, then a single cuda-stage2* child package, then PATH. Multiple child packages require an explicit path. Each worker receives the same TOML and its own --worker index, using an inherited Stage2 stop event. A current Stage2 build is required; exe remains the separate Stage1 path.

Stage2 worker共用的程序路径。留空先查找GUI同目录的ecm_cuda_stage2.exe，再查找唯一的cuda-stage2*子包，最后使用PATH；多个子包时须明确指定路径。各worker使用同一TOML和自己的--worker编号，以及继承的Stage2停止事件，需要当前Stage2构建；exe仍保存独立Stage1路径。

### start_tab

```toml
start_tab = "workers"
```

Type / 类型: `string`; range / 范围: `[workers|detail|gen|prob]`; worker override / worker 覆盖: `no / 不支持`.

Panel initially shown by the GUI: workers opens the worker list, detail the details panel, gen the task generator, and prob the ECM probability calculator. The probability panel can also be opened from View. Does not change worker autostart behavior.

选择 GUI 启动时优先显示的面板。workers 为 worker 列表，detail 为详情，gen 为任务生成器，prob 为ECM概率计算器；概率面板也可从查看菜单打开。不改变是否自动启动 worker。

### window

```toml
window = [120, 80, 1500, 900]
```

Type / 类型: `rect`; range / 范围: `<x:Z>,<y:Z>,<w:Z>,<h:Z>`; worker override / worker 覆盖: `no / 不支持`.

Saved main-window geometry: x,y,width,height. Updated on normal GUI exit; manual editing is usually unnecessary. Width and height should be positive.

保存 GUI 主窗口的位置和大小，依次为 x,y,宽,高。GUI 正常退出时会更新；通常无需手动编辑，宽和高应为正数。

## [workers.1]

### autostart

```toml
autostart = 0
```

Type / 类型: `int`; range / 范围: `[0|1]`; worker override / worker 覆盖: `no / 不支持`.

Automatically launch this worker when the GUI starts: 0 disables; nonzero enables. Does not affect starting a worker directly from the command line.

控制 GUI 启动后是否自动启动此 worker。0 不启动，非零值启动；此设置不影响从命令行直接启动 worker 的行为。

### extra_args

```toml
extra_args = ""
```

Type / 类型: `string`; range / 范围: `<argv>`; worker override / worker 覆盖: `no / 不支持`.

Additional command-line arguments for this GUI-launched worker. Separate arguments with spaces; quote paths containing spaces. Overrides depend on options supported by the selected executable.

向 GUI 启动的 worker 追加命令行参数。参数以空格分隔，含空格的路径使用双引号；命令行覆盖配置文件的选项应以实际 worker 支持的参数为准。

### gpucurves

```toml
gpucurves = 0
```

Type / 类型: `int`; range / 范围: `<x:Z,x>=0>`; worker override / worker 覆盖: `no / 不支持`.

Legacy GUI curve-count display field, inheriting the global value if present. Stage1 queue curve counts come from task lines; this field does not override them. The single-run -gpucurves CLI option is independent.

保留旧 GUI 配置的曲线数显示字段，可继承全局同名值。Stage1 队列执行的曲线数由各任务行决定，此键不会覆盖任务；直接单次运行的 -gpucurves 是独立命令行参数。

### name

```toml
name = "Worker #{N}"
```

Type / 类型: `string`; range / 范围: `<text>`; worker override / worker 覆盖: `no / 不支持`.

Worker display name in the GUI. Unset shows Worker #N, where N is the worker index. Changes only the name, not device or task assignment.

设置此 worker 在 GUI 中显示的名称。未配置时显示 Worker #N，其中 N 为 worker 编号；只改变显示名称，不改变设备或任务分配。

## Maintenance / 维护

Edit `config/ecm_options.json`, then run `python tools/gen/generate_ecm_config.py`; commit the schema and generated files together.

修改 `config/ecm_options.json` 后运行生成器，并将定义与生成物一同提交。
