# ecm.ini Reference / ecm.ini 配置参考

<!-- GENERATED: config/ecm_options.json; do not edit. -->

Add or edit the configuration lines below in `ecm.ini`. Stage1, Stage2, and the GUI share this file; each group identifies the applicable settings.<br>
在 `ecm.ini` 中添加或修改下面的配置行。Stage1、Stage2 和 GUI 共用此文件，各选项的适用范围在分组中注明。

Configuration lines use symbolic notation; the accompanying descriptions explain purpose, value effects, and when to adjust a setting.<br>
配置行保留符号写法，后面的文字说明用途、取值效果和需要调整的情况。

## Notation and general rules / 记号与通用规则

```text
<x> = value; [x] = optional; [a|b] = one of; a..b = inclusive range
<x> = 值; [x] = 可选; [a|b] = 任选其一; a..b = 含端点的范围
default = built-in default; "" = empty; @x = inherited or derived value
default = 内置默认值; "" = 空值; @x = 继承或派生值
Z = integer; R = real; N = worker_index (1..1000000)
S1 = Stage1; S2 = Stage2; GUI = ecm_gui; P95 = Prime95
MiB = 2^20 B; s = seconds; ms = milliseconds
key=<domain>; default=<default>; [unit=<unit>]; [empty=<policy>]
INI: key=value; comment=[#|;]...; inline_comment=unsupported; path_quotes=none
key_case: S1=sensitive; S2=insensitive; GUI=sensitive
bool(S1,S2): [true|false|yes|no|on|off|1|0]; case=insensitive
integer(S2): decimal|scientific; exact_integer; >=0
override: built_in < global < Worker#N < CLI
B2: CLI(nonzero) > task(nonzero) > stage2_b2 > AutoB2
CLI.sections: Worker#N=scope; other[]=label; label!=scope_reset
GUI.sections: global|GUI|Worker#N; other[]=distinct_section
global_keys => before_first_section; grouping => #comment
duplicate_key: last_in_layer; unknown_key: ignore
invalid: S1=legacy_fallback; S2=error; GUI=fallback_or_clamp
path_base: S1=exe_dir; S2.INI=ini_dir; CLI=CWD; exceptions=path_base(key)
queue: consumers=1; parallel_workers=>distinct(queue,outputs)
S1.INI => queue_mode; single_run => CLI
AutoB2(current_release)=uncalibrated; required=explicit_B2
memory: arena+fold+batch != process_peak; budget!=total_VRAM_cap
```

Write only `key=value` to the INI. `default`, `unit`, and similar annotations are not additional keys. `release` and `first_run` identify deliberate template values that may differ from built-in defaults; existing INI files are not updated automatically.<br>
只把 `key=value` 写入 INI，`default`、`unit` 等为说明记号，不是额外配置键。`release` 和 `first_run` 表示模板有意采用的值，可能与内置默认值不同；已有 INI 不会自动改为模板值。

Place shared keys before the first section header and worker overrides under `[Worker #N]`. CLI programs treat only worker headers as scopes; other headers are labels and do not leave an active worker scope. The GUI reads actual sections. Use `#` comments to group shared keys for compatibility.<br>
公共键放在任何分区标题之前；专用 worker 值放在 `[Worker #N]` 下。命令行程序只把 worker 标题当作作用域，其他标题仅是标签，也不会退出已有 worker 作用域；GUI 则按真实分区读取。为兼容两者，公共键应使用 `#` 注释分组。

The last duplicate key in each layer wins; worker settings override global settings, and CLI overrides INI. Boolean keys accept the forms above, but Stage2 integer switches accept only `0|1`. Do not quote INI paths or append inline comments to configuration values.<br>
同一层重复键以最后一次出现为准，worker 设置覆盖全局，命令行覆盖 INI。布尔键可以使用上列布尔写法，但标为整数开关的 Stage2 键只接受 `0|1`。路径不要加引号，不支持在配置行末尾追加注释。

Stage1 relative paths normally use the executable directory. Stage2 INI paths use the INI directory; CLI paths use the current working directory. Individual keys document any exceptions. Concurrent processes need separate queues, progress files, and output paths.<br>
Stage1 相对路径通常以可执行文件目录为基准；Stage2 的 INI 相对路径以 INI 目录为基准，命令行相对路径以当前工作目录为基准。各键另有约定时见该项说明。多进程并行运行应使用独立队列、进度及输出路径。

## Stage1 and shared settings / Stage1 与共享选项 — global or [Worker #N] / 全局或 [Worker #N]

### worktodo

```text
worktodo=<path>; default=worktodo.txt
```

Stage1 task queue for the current worker. Completed assignments move to finished. Stage2 uses<br>
stage2_worktodo; each queue must have only one consumer.<br>
指定 Stage1 读取的任务队列。程序按当前 worker 领取任务；完成后将原任务移入 finished。Stage2 使用独立<br>
的 stage2_worktodo，不要让两个程序同时消费同一文件。

### finished

```text
finished=<path>; default=worktodo.finished.txt
```

Stage1 completed-task record, containing original assignments and results. This is not an<br>
intermediate curve checkpoint.<br>
指定 Stage1 已完成任务的记录文件。用于保留原任务行和处理结果，不是曲线计算的中间存档。

### log_file

```text
log_file=<path>; default=screen.log; empty=off; worker(N>1)=_N
```

Append normal Stage1 output to this file. Empty disables file logging. When no filename is<br>
explicitly set, workers add a numeric suffix, such as screen_2.log.<br>
指定 Stage1 普通运行日志，程序追加写入。留空关闭文件日志；未显式指定文件名时，worker 2 等自动使用<br>
screen_2.log 这类带编号的文件名。

### tmp_dir

```text
tmp_dir=<path>; default=.
```

Output directory for Stage1 curve saves. Stage2 also searches here when stage2_save_dir is unset.<br>
This is not a Stage2 RAM or VRAM budget.<br>
指定 Stage1 曲线存档的输出目录。Stage2 未设置 stage2_save_dir 时也在此目录查找任务引用的 save；此键<br>
不是 Stage2 全部临时显存或内存的预算。

### save_sync_dir_1

```text
save_sync_dir_1=<path>; default=""; empty=off
```

First destination for Stage1 save synchronization. Empty disables this destination; sync_mode<br>
controls the files and synchronization policy.<br>
指定第一个 Stage1 存档同步目录。留空不启用该目标；同步范围和方式由 sync_mode 决定。

### save_sync_dir_2

```text
save_sync_dir_2=<path>; default=""; empty=off
```

Second destination for Stage1 save synchronization. Both destinations may be enabled; empty disables<br>
only this destination.<br>
指定第二个 Stage1 存档同步目录。可同时向两个目标同步；留空只关闭此目标，不影响第一个目标。

### sync_mode

```text
sync_mode=[incremental|full]; default=incremental
```

Save synchronization after a Stage1 assignment: incremental copies files updated during that<br>
assignment; full copies all eligible saves. Startup always performs a full synchronization.<br>
控制 Stage1 任务完成后的存档同步。incremental 仅同步本次任务期间更新的文件；full 同步全部符合条件的<br>
存档。程序启动时仍执行一次完整同步。

### p95_worktodo_path

```text
p95_worktodo_path=<path>; default=""; empty=off
```

Path to Prime95 worktodo.txt. Stage1 appends generated ECMSTAGE2 assignments to worktodo.add in the<br>
same directory for Prime95 to import. Empty disables handoff. Prime95 must also be able to access<br>
the save files.<br>
指定 Prime95 的 worktodo.txt 路径。启用后，将 Stage1 生成的 ECMSTAGE2 任务追加到同目录的<br>
worktodo.add，由 Prime95 自行导入；留空关闭交接。save 文件本身还需能被 Prime95 访问。

### p95_add_workers

```text
p95_add_workers=[<n>|<n>,...|<a>-<b>|auto]; default=""; empty=no_worker_header
```

Prime95 workers eligible to receive handoffs: one ID, a comma list, or ranges. The shortest eligible<br>
queue is selected. auto reads NumWorkers from prime.txt in the same directory. Empty omits the<br>
worker header and uses Prime95's default worker.<br>
选择接收交接任务的 Prime95 worker，可指定单个编号、编号列表或区间；程序在候选 worker 中选择队列较短<br>
者。auto 从同目录 prime.txt 读取 NumWorkers；留空追加时不写 worker 段头，按 Prime95 的默认 worker 处<br>
理。

### p95_keep_aid

```text
p95_keep_aid=[0|1]; default=1
```

Keep PrimeNet assignment IDs in handoffs: 1 retains IDs not known to be invalid; 0 strips all IDs.<br>
IDs confirmed invalid by Prime95 logs are always removed to prevent repeated rejection.<br>
控制交接任务是否保留 PrimeNet assignment ID。1 保留未被判定失效的 AID，0 一律去掉；已在 Prime95 日志<br>
中确认失效的 AID 会去掉，以免任务再次被丢弃。

### p95_recover_lost

```text
p95_recover_lost=[0|1]; default=1
```

Recover handoffs deleted by Prime95 because of invalid assignment IDs. 1 checks handoff records and<br>
Prime95 logs, then reposts confirmed lost assignments without an ID; 0 disables recovery. Only<br>
confirmed prior handoffs qualify.<br>
控制是否补投被 Prime95 因失效 AID 删除的交接任务。1 根据交接记录和 Prime95 日志检查并去掉 AID 补投，<br>
0 关闭自动补投；只对程序能够确认的已交接任务生效。

### progress_color

```text
progress_color=[none|red|green|yellow|blue|magenta|cyan|white|grey]; default=cyan
```

Color of the Stage1 console progress bar. none disables color. This does not change ordinary file<br>
logs or arithmetic.<br>
选择 Stage1 控制台进度文字颜色。none 关闭着色；此设置不改变普通文件日志内容或计算方式。

### progress_log_seconds

```text
progress_log_seconds=<x:R,finite>; default=60; unit=s; x>0:period(x); x=0:off; x<0:each_line; 100%:always
```

Interval in seconds between Stage1 progress entries in file logs. Positive values limit frequency; 0<br>
suppresses intermediate progress; negative values log every refresh. Completion at 100% is still<br>
logged. Console refresh frequency is unchanged.<br>
控制 Stage1 进度行写入日志文件的间隔，单位为秒。正数按间隔记录，0 不记录中间进度，负数记录每次进度更<br>
新；完成时的 100% 行仍会记录。此键不限制控制台刷新频率。

### verbose

```text
verbose=[0|1]; default=1; S2:1=phases;0=curve; debug:independent
```

Normal console detail. In Stage2, true shows phase transitions and false keeps curve summaries;<br>
stage2_debug_log independently controls debug logging. Stage1 uses the selected backend's normal<br>
verbosity behavior.<br>
控制普通控制台输出的详细程度。Stage2 中 true 显示阶段切换，false 保留曲线摘要；调试日志由<br>
stage2_debug_log 独立控制。Stage1 按各计算后端的普通详细输出规则处理。

### method

```text
method=[gpu|edwards|mont]; default=gpu; empty=default; [opencl]=>gpu; [atkin-morain]=>edwards; [montgomery|suyama]=>mont
```

Stage1 arithmetic method. gpu uses the backend linked into the executable: CUDA in ecm_cuda, OpenCL<br>
in ecm. edwards and mont select the corresponding CPU implementations. This setting cannot switch an<br>
executable between CUDA and OpenCL.<br>
选择 Stage1 计算方法。gpu 使用当前可执行文件链接的 GPU 后端：ecm_cuda 使用 CUDA，ecm 使用 OpenCL；<br>
edwards 和 mont 使用对应的 CPU 曲线实现。改为 gpu 不会在同一可执行文件内切换 CUDA 与 OpenCL。

### backend

```text
backend=[auto|simd|gmp]; default=auto; empty=default; simd=>AVX512-IFMA; [avx512]=>simd; [scalar|mpn]=>gmp
```

CPU Edwards/Montgomery big-integer backend: auto selects automatically, simd uses supported<br>
AVX512-IFMA operations, and gmp uses GMP. GPU Stage1 ignores this setting.<br>
选择 CPU Edwards/Montgomery 的大整数乘法后端。auto 自动选择，simd 使用受支持的 AVX512-IFMA 路径，gmp<br>
使用 GMP；GPU Stage1 不使用此设置。

### field

```text
field=[auto|mersenne|montgomery]; default=auto; empty=default; mersenne=>N=2^p-1; [mers|on|fold]=>mersenne; [mont|off|cios]=>montgomery
```

CPU SIMD reduction method: auto selects for the input, mersenne requires N=2^p-1, and montgomery<br>
uses general Montgomery reduction. Usually leave auto. GPU Stage1 ignores this setting.<br>
选择 CPU SIMD 的模约减方法。auto 按输入数选择，mersenne 适用于 N=2^p-1，montgomery 使用通用<br>
Montgomery 约减；GPU Stage1 不使用此设置。通常保持 auto。

### stage1_threads

```text
stage1_threads=<x:Z,0..2^32-1>; default=0; zero=auto
```

CPU Stage1 thread count; 0 selects automatically. Applies only to CPU arithmetic and does not set<br>
CUDA threads per block or threads per integer.<br>
设置 CPU Stage1 的线程数。0 自动选择；只影响 CPU 计算路径，不改变 CUDA 的线程块大小或每个大整数使用<br>
的线程数。

### affinity

```text
affinity=[none|auto|<n>,...|<a>-<b>]; default=""; empty=OS; n<logical_CPU_count
```

Logical CPU IDs or ranges for CPU Stage1 threads, assigned round-robin. Empty, none, or auto leaves<br>
scheduling to the OS. The current Windows implementation supports only IDs 0..63 in one processor<br>
group; other platforms do not apply this binding. Use IDs available on the machine.<br>
设置 CPU Stage1 计算线程的亲和性，可用逻辑 CPU 编号列表或区间；线程依次循环使用列表中的 CPU。空值、<br>
none 或 auto 均不主动绑定，由操作系统调度。当前 Windows 实现只对单处理器组内编号 0..63 生效，其他平<br>
台不执行此绑定；编号还须对应实际可用的逻辑 CPU。

### save_name_pattern

```text
save_name_pattern=<pattern>; default=m{n}_{b1}.save; empty=default; tokens:{n},{b1}; suffix:_<B1>.save
```

CPU Montgomery save filename pattern: {n} is the Mersenne exponent and {b1} is B1. Empty restores<br>
the default pattern. Other backends retain their own naming rules.<br>
设置 CPU Montgomery 存档的名称模式，{n} 表示输入指数，{b1} 表示 B1。留空恢复默认模式；此键不改变其他<br>
计算后端自己的文件命名规则。

### exp_cache

```text
exp_cache=[off|none|0|-|<dir>]; default=""; empty=exe_dir; path_base=process_CWD(nonempty); [off|none|0|-]:disabled; case:sensitive
```

Directory for the precomputed Stage1 scalar cache, reused for matching B1 and exponent mode. Empty<br>
uses the executable directory; off, none, 0, or - disables caching. Explicit relative paths use the<br>
current working directory. Small B1 values may be computed without writing a cache.<br>
指定 Stage1 指数缓存目录，复用相同 B1 和指数模式的预计算结果。留空使用可执行文件目录；off、none、0<br>
或 - 关闭缓存。显式相对目录以进程当前工作目录为基准；很小的 B1 可能直接计算而不写缓存。

### naf_w

```text
naf_w=[0|3..12]; default=0; zero=12; recommended; runtime:x=0|x>=2; dict=2^(x-2)
```

CPU Edwards NAF window. 0 uses the built-in choice. Larger windows usually reduce loop additions,<br>
while the precomputed dictionary grows as 2^(w-2), increasing memory and setup time. Typical values<br>
are 3..12. CUDA PRAC ignores this setting.<br>
设置 CPU Edwards 的 NAF 窗口。0 使用内置窗口；增大窗口通常减少主循环加法，但预计算字典按 2^(w-2) 增<br>
长，会增加内存和准备时间。通常使用 3..12；CUDA PRAC 不使用此设置。

### exponent

```text
exponent=[lcm|choose12]; default=lcm; empty=default; lcm:lcm(1..B1); choose12:12*lcm(1..B1); [1|gmp|gmp-ecm]=>lcm; [12|prime95]=>choose12
```

Stage1 scalar: lcm uses lcm(1..B1); choose12 uses 12*lcm(1..B1) when extra multiplication by 12 is<br>
required. This changes the output point. Keep the convention consistent with the curve and save<br>
consumed by Stage2.<br>
选择 Stage1 使用的指数。lcm 为 lcm(1..B1)，choose12 为 12*lcm(1..B1)，用于需要额外乘 12 的流程；该设<br>
置改变计算结果，接续 Stage2 时应与所用曲线和存档约定一致。

### sigma

```text
sigma=<x:Z,0..2^64-1>; default=0; zero=random; curve(i):sigma+i; Suyama:x>=6; P95:sigma+curves<=2^63-1
```

Initial curve sigma. 0 chooses a random value within the backend's valid range; consecutive curves<br>
use sigma+i. A nonzero sigma in an ECM/ECM2 assignment overrides this setting; otherwise this<br>
setting is used. ECMSTAGE2 assignments without a sigma field also use it.<br>
指定曲线起始 sigma；0 由程序随机选择。在采用连续 sigma 的批处理中，后续曲线使用 sigma+i；各曲线后端<br>
仍有自己的有效范围。ECM/ECM2 任务行给出非零 sigma 时覆盖此键，否则使用此键；不带 sigma 字段的<br>
ECMSTAGE2 任务也使用此键。

### p95_dir

```text
p95_dir=<dir>; default=""; empty=off
```

Legacy compatibility setting. A nonempty value only produces an optional hint; it does not enable<br>
the current Prime95 handoff mechanism. Use p95_worktodo_path to hand off ECMSTAGE2 work.<br>
保留旧配置兼容性，设置非空值只产生旧选项提示，不启用新的 Prime95 交接流程。需要交接 ECMSTAGE2 任务时<br>
使用 p95_worktodo_path。

### ckpt_seconds

```text
ckpt_seconds=<x:R,finite,x>=0>; default=600; unit=s; zero=periodic_off; Ctrl+C:save; x<0=>0
```

Stage1 checkpoint interval in seconds. 0 disables periodic checkpoints; negative values act as 0.<br>
Normal stop or Ctrl+C can still request a checkpoint. Actual writes wait for the backend's safe<br>
checkpoint boundary.<br>
设置 Stage1 定期写检查点的间隔，单位为秒。0 关闭定期检查点，负数按 0 处理；仍可通过正常停止或 Ctrl+C<br>
请求保存。实际写盘时机受当前计算后端的安全检查点限制。

### device

```text
device=<x:Z,0..device_count-1>; default=0
```

Zero-based GPU device ID for Stage1. Stage2 inherits it unless stage2_device is explicitly set. IDs<br>
refer to the device list exposed by the executable's actual GPU backend.<br>
选择 GPU 设备编号，从 0 开始。用于 Stage1；Stage2 未显式指定 stage2_device 时也继承此编号。编号对应<br>
各可执行文件实际使用的 GPU 后端设备列表。

### gpu_param

```text
gpu_param=[0|2|3]; default=3; release=0; first_run=0; OpenCL:3; S2_save:0; invalid=>3
```

GPU Stage1 curve parametrization. The current CUDA Stage2 consumes param0 saves; use 0 for that<br>
handoff. OpenCL uses param3. The built-in default is 3, while first-run and release templates<br>
deliberately write 0. Existing INI files are not changed automatically.<br>
选择 GPU Stage1 曲线参数族。当前 CUDA Stage2 消费参数 0 的 save，准备交给它处理时使用 0；OpenCL 路径<br>
使用参数 3。内置默认值为 3，但首次启动和发布模板有意写入 0；旧 INI 不会自动改值。

### tpi

```text
tpi=[1|2|4|8|16|32]; default=8; limbs%effective_TPI=0; CUDA:ignored
```

OpenCL threads per integer, subject to limb-layout constraints. CUDA selects TPI by operand width<br>
and ignores this key. This is not the CUDA block size.<br>
设置 OpenCL 每个大整数使用的线程数，需满足有效 TPI 与整数 limb 数的布局约束。CUDA 后端自行按位宽选择<br>
TPI，不使用此键；不要用它调整 CUDA 的线程块大小。

### wg_size

```text
wg_size=<x:Z,x>=0>; default=0; zero=auto; x<=device_kernel_limit; CUDA:ignored
```

OpenCL work-group size. 0 lets the backend select automatically; explicit values must satisfy device<br>
and kernel limits. CUDA ignores this setting.<br>
设置 OpenCL 工作组大小，0 由后端自动选择。显式值必须满足当前设备和内核的限制；CUDA 后端不使用此设置<br>
。

### kernel_mul

```text
kernel_mul=[auto|<id>|<alias>]; default=""; empty=auto; CUDA:ignored
```

OpenCL modular-multiplication kernel ID or alias. Empty or auto lets the backend choose. Intended<br>
for comparing or debugging known kernels; automatic selection is normally suitable. CUDA ignores<br>
this key.<br>
指定 OpenCL 模乘内核的标识或别名。空值或 auto 由后端选择；用于已知内核的比较和调优，通常保持自动。<br>
CUDA 后端忽略此键。

### kernel_sqr

```text
kernel_sqr=[auto|<id>|<alias>]; default=""; empty=auto; CUDA:ignored
```

OpenCL modular-squaring kernel ID or alias. Empty or auto lets the backend choose; automatic<br>
selection is normally suitable. CUDA ignores this key.<br>
指定 OpenCL 模平方内核的标识或别名。空值或 auto 由后端选择；通常保持自动，CUDA 后端忽略此键。

### kernel_add

```text
kernel_add=[auto|<id>|<alias>]; default=""; empty=auto; CUDA:ignored
```

OpenCL modular-addition kernel ID or alias. Empty or auto lets the backend choose; automatic<br>
selection is normally suitable. CUDA ignores this key.<br>
指定 OpenCL 模加内核的标识或别名。空值或 auto 由后端选择；通常保持自动，CUDA 后端忽略此键。

### kernel_sub

```text
kernel_sub=[auto|<id>|<alias>]; default=""; empty=auto; CUDA:ignored
```

OpenCL modular-subtraction kernel ID or alias. Empty or auto lets the backend choose; automatic<br>
selection is normally suitable. CUDA ignores this key.<br>
指定 OpenCL 模减内核的标识或别名。空值或 auto 由后端选择；通常保持自动，CUDA 后端忽略此键。

### kernel_special_mult

```text
kernel_special_mult=[auto|<id>|<alias>]; default=""; empty=auto; CUDA:ignored
```

OpenCL special-multiplication kernel ID or alias. Empty or auto lets the backend choose; automatic<br>
selection is normally suitable. CUDA ignores this key.<br>
指定 OpenCL 特殊乘法内核的标识或别名。空值或 auto 由后端选择；通常保持自动，CUDA 后端忽略此键。

## Stage2 — global or [Worker #N] / 全局或 [Worker #N]

### stage2_worktodo

```text
stage2_worktodo=<path>; default=stage2_worktodo.txt; empty=invalid
```

ECMSTAGE2 queue consumed by Stage2. Only curves available in the save are processed; shortages are<br>
reported. Processing continues with the next assignment after completion. Only one process may<br>
consume a queue and its progress file.<br>
指定 Stage2 读取的 ECMSTAGE2 任务队列。程序仅执行 save 中可用的曲线并报告数量不足；任务完成后继续下<br>
一项。队列及其进度文件只应由一个消费进程使用。

### stage2_finished

```text
stage2_finished=<path>; default=stage2_worktodo.finished.txt; empty=invalid
```

Record of completed Stage2 assignments and assignments with invalid input. Stores original tasks,<br>
result status, and commit receipts for reconciliation. Must not share a path with Stage1 finished or<br>
an active queue.<br>
指定 Stage2 完成或输入错误任务的记录文件。保存原任务、结果状态及提交回执，用于队列完成后的对账；不能<br>
与 Stage1 finished 或活动队列共用路径。

### stage2_log_file

```text
stage2_log_file=<path>; default=stage2_screen.log; empty=off; worker(N>1)=_N
```

Append normal Stage2 phases, batches, warnings, and final statistics. Empty disables this file log;<br>
console output remains. Default filenames gain a worker suffix unless an explicit filename is<br>
supplied.<br>
指定 Stage2 普通运行日志，追加记录阶段、批次、警告和结束统计。留空关闭普通文件日志，控制台仍输出；未<br>
显式指定时，多 worker 自动给默认文件名加编号后缀。

### stage2_results_file

```text
stage2_results_file=<path>; default=stage2_results.jsonl; empty=default; worker(N>1)=_N
```

JSONL results for successful Stage2 curves, including factors and resume receipts. Empty uses the<br>
default filename. Do not independently delete or replace it while resuming an unfinished queue; it<br>
is needed to reconcile completed curves.<br>
指定 Stage2 成功曲线结果的 JSONL 文件，包含因子信息和续跑回执。留空使用默认文件名；不要在未完成队列<br>
续跑期间单独删除或替换它，否则会失去回执对账依据。

### stage2_debug_log_file

```text
stage2_debug_log_file=<path>; default=stage2_debug.log; empty=default; worker(N>1)=_N
```

Separate Stage2 debug log, written only when stage2_debug_log is enabled. Empty restores the default<br>
filename. Default names gain a worker suffix; explicit paths are used unchanged.<br>
指定 Stage2 独立调试日志文件，仅在 stage2_debug_log 开启时写入。留空恢复默认文件名；默认名称按<br>
worker 加后缀，显式指定时按原路径使用。

### stage2_progress_file

```text
stage2_progress_file=<path>; default=""; empty=queue_path+worker_suffix+.progress; queue_only; empty!=off
```

Stage2 queue resume state. Empty derives the path from the queue plus worker suffix and .progress;<br>
it does not disable resume. Stores completed curve indices and receipts, without checkpointing<br>
arithmetic within a curve. Restart recomputes only curves not confirmed complete.<br>
指定 Stage2 队列续跑状态文件。空值自动使用队列路径加 worker 后缀和 .progress，并不关闭续跑；记录已完<br>
成曲线编号和回执，不保存当前曲线的 Stage2 中间计算。重启时只重算尚未确认完成的曲线。

### stage2_save_dir

```text
stage2_save_dir=<dir>; default=""; empty=tmp_dir
```

Directory searched for saves referenced by Stage2 assignments. Empty inherits tmp_dir; absolute save<br>
paths in assignments are used directly. Source saves are read and validated without modification.<br>
指定 Stage2 查找任务所引用 save 的目录。留空继承共享 tmp_dir；任务给出绝对 save 路径时直接使用该路径<br>
。程序读取并验证源 save，不修改其内容。

### stage2_debug_log

```text
stage2_debug_log=[true|false|1|0|yes|no|on|off]; default=false
```

Write Stage2 debug details to a separate log: true enables and false disables. verbose=true does not<br>
enable debug logging; enabling this option does not send all debug details to the console.<br>
控制是否将 Stage2 调试细节写入独立日志。true 开启，false 关闭；verbose=true 不会自动开启调试日志，开<br>
启本选项也不把全部调试输出刷到控制台。

### stage2_device

```text
stage2_device=<x:Z,0..2^31-1>; default=@device
```

Override Stage2's zero-based CUDA device ID. Unset inherits device. The internal inheritance<br>
sentinel does not mean -1 is accepted as an explicit INI value.<br>
单独指定 Stage2 的 CUDA 设备编号，从 0 开始。未配置时继承共享 device；继承状态不等于可以在 INI 中显<br>
式填写 -1。

### stage2_log_level

```text
stage2_log_level=[quiet|curve|phases|batches|debug|0..4]; default=@verbose?phases:curve; production:debug=>batches+debug_file; normal_file:batches
```

Legacy console level: quiet, curve, phases, batches, and debug map to 0..4. Unset follows verbose.<br>
In production, debug enables the separate debug log while normal console detail remains capped at<br>
batches. Usually use verbose and stage2_debug_log.<br>
兼容旧配置的控制台日志等级：quiet、curve、phases、batches、debug 分别对应 0..4。未配置时由 verbose<br>
决定；debug 在生产程序中开启独立调试日志，普通控制台最高保持 batches。一般只需使用 verbose 和<br>
stage2_debug_log。

### stage2_b2

```text
stage2_b2=<x:Z,x=0|B1<x<=2^63-1-8192>; default=0; zero=task_or_CLI_or_AutoB2
```

Stage2 upper bound when neither CLI nor task supplies a nonzero B2. 0 defers to other sources.<br>
Explicit B2 must exceed the save's B1. Larger B2 usually increases coverage, time, and VRAM demand.<br>
Precedence: nonzero CLI B2, then task B2, then this key.<br>
设置未在命令行或任务行给出非零 B2 时使用的 Stage2 上界。0 表示继续从其他来源选择；显式 B2 必须大于存<br>
档的 B1。增大 B2 通常增加搜索范围、运行时间及显存需求；命令行非零 B2 优先于任务行，任务行优先于此键<br>
。

### stage2_d

```text
stage2_d=<x:Z,x=0|x>=6,x%2=0>; default=0; zero=auto; engine_shape_valid
```

Manual Stage2 step D. 0 selects the shape automatically. Explicit values must be even, at least 6,<br>
and satisfy the engine's layout constraints. D affects tree size, batch count, and memory demand;<br>
automatic selection is normally suitable.<br>
手动指定 Stage2 算法步长 D。0 自动选形；非零值除满足偶数且不小于 6 外，还必须满足当前引擎的布局限制<br>
。D 会改变树规模、批次数和内存需求，通常保持自动。

### stage2_batch_mb

```text
stage2_batch_mb=<x:Z,1..2^20>; default=64; unit=MiB
```

Stage2 S4 batch-buffer budget in MiB. Larger budgets may reduce batching overhead but consume more<br>
VRAM. This limits the corresponding buffers, not total process VRAM.<br>
设置 Stage2 S4 批次的工作缓冲预算，单位为 MiB。增大预算可能减少分批开销，但会占用更多显存；它只约束<br>
对应批次缓冲，不是整个进程的显存上限。

### stage2_arena_mb

```text
stage2_arena_mb=<x:Z,0..2^20>; default=0; unit=MiB; zero=auto_or_env
```

Stage2 large-workspace budget in MiB, used to plan available NTT shapes. 0 uses environment settings<br>
or automatic planning, not zero VRAM. Too little may force smaller shapes or prevent execution.<br>
Tables, coordinates, and other modules may allocate additional VRAM.<br>
设置 Stage2 大工作区预算，单位为 MiB，用于规划可用的 NTT 工作形状。0 由运行环境或自动规划决定，不表<br>
示不使用显存；过小可能迫使较小工作形状或使任务无法运行。表、坐标和其他模块仍可能另占显存。

### stage2_fold_mb

```text
stage2_fold_mb=<x:Z,0..2^20>; default=640; unit=MiB; zero=off; CLI>INI>NTT_FOLD_DEVICE_MAX_MB>default
```

Budget in MiB for keeping fold accumulators on the GPU. 0 disables this residency. Exceeding the<br>
budget may fall back to batching, substantially increasing processing and transfer costs.<br>
Independent of arena and batch budgets; not a total VRAM limit.<br>
设置 fold 累积结果驻留 GPU 的预算，单位为 MiB。0 关闭这部分驻留；所需空间超过预算时可退回分批路径，<br>
可能显著增加处理和传输开销。此预算与大工作区、批次缓冲分别控制，不是总显存限制。

### stage2_factorize_hits

```text
stage2_factorize_hits=[0|1]; default=0
```

Invoke PARI/GP to further decompose found factors: 1 enables; 0 keeps only factors extracted and<br>
verified by Stage2. Decomposition can add time after curve computation and requires a working<br>
stage2_gp.<br>
控制是否调用 PARI/GP 进一步分解命中的因子。1 开启，0 只保留 Stage2 已得到并验证的因子结果；进一步分<br>
解可能增加曲线结束后的时间，需要可用的 stage2_gp。

### stage2_factor_only

```text
stage2_factor_only=[0|1]; default=0
```

Skip diagnostic prime attribution, which identifies Stage2 primes responsible for a hit. 1 skips<br>
attribution while retaining factor extraction and validation; 0 follows the engine's diagnostic<br>
policy. This does not disable decomposition or change stage2_factorize_hits.<br>
控制是否跳过命中结果的素数归因诊断，即查找哪些 Stage2 素数对应此次命中。1 跳过此诊断，仍执行因子提取<br>
与验证；0 使用引擎默认诊断策略。此键不表示关闭因子分解，也不改变 stage2_factorize_hits。

### stage2_auto_b2

```text
stage2_auto_b2=[0|1]; default=0
```

Choose B2 when every source supplies zero. 1 optimizes total workflow benefit per unit time,<br>
including Stage1 cost; 0 disables. Full ECM tune requires matching device/width/B1/memory policy and<br>
a provided or matched measured Stage1 cost. It jointly selects measured or qualified in-range B2, D<br>
and legal carrier. Without full ECM tune, the legacy cost profile must pass its calibration checks.<br>
This switch does not calibrate costs.<br>
所有来源均为零 B2 时自动选择。1 按计入 Stage1 成本的全流程单位时间收益选取；0 关闭。完整 ECM tune 要<br>
求设备、位宽、B1、显存策略匹配，并提供正的 Stage1 每曲线耗时；联合选择实测或合格区间内的 B2、D 和合<br>
法承载。未提供完整 ECM tune 时，旧成本配置必须通过原有标定检查。本开关不执行成本标定。

### stage2_gp

```text
stage2_gp=<exe>; default=gp.exe; empty=gp.exe; path_base=CreateProcess/PATH
```

PARI/GP executable used for further factor decomposition. gp.exe follows Windows executable search<br>
rules; a full path is also accepted. Required only when stage2_factorize_hits is enabled. This INI<br>
key does not use the dataset scripts' --gp environment-variable fallback.<br>
指定进一步分解因子使用的 PARI/GP 可执行文件。默认 gp.exe 按 Windows 程序搜索规则查找；也可填写完整路<br>
径。仅在 stage2_factorize_hits 开启时需要它；此 INI 键不使用数据集脚本的 --gp 环境变量回退规则。

### stage2_factor_timeout

```text
stage2_factor_timeout=<x:Z,1..600>; default=30; unit=s
```

Timeout in seconds for each PARI/GP factor-decomposition call. Limits only optional decomposition,<br>
not Stage2 curve computation. A timeout does not invalidate a factor already found by Stage2.<br>
设置每次 PARI/GP 因子分解调用的超时时间，单位为秒。仅影响附加的因子分解，不限制 Stage2 曲线计算时长<br>
；超时不表示原 Stage2 算法未找到因子。

### stage2_cost_profile

```text
stage2_cost_profile=<path>; default=""; empty=none
```

Legacy .cprof cost profile for Auto B2 when stage2_tune_profile is empty. Checks the original<br>
binary/device/arithmetic and calibration contract; new full ECM tune does not relax those checks.<br>
Empty supplies no legacy profile.<br>
stage2_tune_profile 为空时供 Auto B2 使用的旧 .cprof 成本配置。保持原二进制、设备、算术及标定资格检<br>
查；新的完整 ECM tune 不放宽旧合同。留空不提供旧配置。

### stage2_tune_profile

```text
stage2_tune_profile=<path>; default=""; empty=legacy_selection
```

ECM final-summary TOML for joint B2/D/carrier selection. B1 never gates reuse. Width and B2 may be<br>
estimated with uncertainty labels. Explicit D or carrier locks the choice; saved N must divide<br>
2^p-1. Empty uses stage2_ecm_tune.toml unless a legacy cost profile is supplied.<br>
ECM 最终汇总 TOML，联合选择 B2、D、承载；B1 不作为复用门槛。位宽和 B2 可带不确定性标记估算。显式 D<br>
或承载固定选择；承载须通过 save 模数整除检查。留空且未指定旧成本配置时使用 stage2_ecm_tune.toml。

### stage1_tune_profile

```text
stage1_tune_profile=<path>; default=""; empty=disabled
```

Checked completed CUDA Stage1 cost anchors, used after explicit seconds/curve and CSV. Matches<br>
batch, modulus type and exponent; B1 and missing widths scale by B1 and the dispatched container<br>
squared. Device/runtime reuse follows stage2_tune_ignore. This does not run Stage1.<br>
已核验的完整 CUDA Stage1 成本锚点，优先级低于显式秒/曲线和 CSV。匹配批次、模数类型和指数模式；B1 与<br>
缺失位宽按 B1 和容器位宽平方估算。设备及运行时复用遵守 stage2_tune_ignore；此文件不启动 Stage1。

### stage2_auto_min_b2

```text
stage2_auto_min_b2=<x:Z,0..2^63-8193>; default=0; zero=B1_plus_one; min<=max
```

Auto B2 lower search bound; 0 means B1+1. Unmeasured values may be estimated; explicit task B2 takes<br>
precedence.<br>
Auto B2 搜索下界；0 为 B1+1。未测值允许估算，显式任务 B2 优先。

### stage2_auto_max_b2

```text
stage2_auto_max_b2=<x:Z,0..2^63-8193>; default=0; zero=2600000000000; min<=max
```

Auto B2 upper search bound; 0 means 2.6e12. Unmeasured values may be estimated; explicit task B2<br>
takes precedence.<br>
Auto B2 搜索上界；0 为 2.6e12。未测值允许估算，显式任务 B2 优先。

### stage1_batch

```text
stage1_batch=<x:Z,1..2^20>; default=1
```

Stage1 batch curve count associated with Auto B2 cost estimation. Selects or checks Stage1 cost<br>
conditions; it neither starts a Stage1 batch from Stage2 nor changes the assignment's curve count.<br>
指定 Auto B2 成本估算对应的 Stage1 批次曲线数。它用于选择或核对 Stage1 的成本条件，不会让 Stage2 驱<br>
动启动该批 Stage1 计算，也不改变任务要处理的曲线数。

### stage1_seconds_per_curve

```text
stage1_seconds_per_curve=<x:R,finite,x>0>; default=@profile; unit=s/curve
```

Positive Stage1 seconds per curve for Auto B2, already amortized for the actual batch. Does not<br>
divide by stage1_batch again. Overrides Stage1 tune measurements. When unset, full ECM tune requires<br>
a matching stage1_tune_profile; the legacy component profile uses its own matched Stage1 data. Not a<br>
runtime limit or the tune input preparation time.<br>
为 Auto B2 提供正的 Stage1 每曲线摊销秒数，应已包含实际批次收益，不再除以 stage1_batch。优先于<br>
Stage1 tune 实测。未指定时，完整 ECM tune 需匹配 stage1_tune_profile；旧组件 profile 使用其自身匹配<br>
的 Stage1 数据。此值不是本次运行时限，也不是调优输入的准备耗时。

### stage2_ratio_adjust

```text
stage2_ratio_adjust=<x:R,finite,x>0>; default=1; T2_model=x*(engine+cold)
```

Multiplier applied to Stage2 time in the Auto B2 model. 1 leaves it unchanged; greater than 1<br>
estimates higher cost, less than 1 lower cost, usually changing selected B2. Adjusts planning cost,<br>
not actual GPU speed.<br>
将 Auto B2 模型中的 Stage2 时间乘以该系数。1 不调整，大于 1 表示估计 Stage2 更贵，小于 1 表示更便宜<br>
；通常会相应影响所选 B2。此键只调整选形成本，不会让 GPU 按比例变快或变慢。

### stage2_tune_ignore

```text
stage2_tune_ignore=[gpu|driver|cuda|backend|memory|environment],...; default=gpu,driver,cuda,backend,environment; empty=none
```

Comma-separated conditions ignored for profile reuse. Empty enforces every condition. Memory<br>
normally matches execution path; arithmetic legality and current memory admission are always<br>
checked.<br>
逗号分隔的复用忽略项；留空核对全部条件。memory 默认匹配执行路径。算术合法性及当前显存准入始终检查。

### stage2_tune_budget_seconds

```text
stage2_tune_budget_seconds=<x:Z,1..86400>; default=1800
```

Total ECM tuning measurement budget, including warmups. Finish an in-flight measurement after the<br>
deadline; publish completed summaries for incremental reuse.<br>
ECM 调优的测量总预算，包含预热；到期后完成正在运行的测量并保存有效汇总，供下次增量使用。

### stage2_short_calibration

```text
stage2_short_calibration=[0|1]; default=1
```

Calibrate once when no suitable ECM tune file exists, required components are missing, or the task<br>
has no reusable calibrated width coverage. Partial coverage permits estimates.<br>
无合适 ECM tune 文件、必要组件缺失或任务位宽完全没有可复用校准覆盖时短校准；部分覆盖允许估算。

### stage2_short_calibration_seconds

```text
stage2_short_calibration_seconds=<x:Z,1..600>; default=10
```

Short calibration measurement budget, excluding first CUDA initialization; finish the current<br>
measurement before stopping.<br>
短校准测量预算，不含首次 CUDA 初始化；到期完成当前测量后停止。

### stage2_tune_error_limit

```text
stage2_tune_error_limit=<x:R,0<x<=1>; default=0.08
```

Independent relative prediction error threshold used to prioritize additional full-curve<br>
measurements. Unchecked estimates retain an uncertainty allowance.<br>
独立预测相对误差阈值，用于优先追加完整曲线测量；未验证估算保留不确定性余量。

### stage1_cost_csv

```text
stage1_cost_csv=<path>; default=""; empty=none
```

CSV Stage1 s/curve anchors for workflow Auto B2. Columns: container_bits or target_bits, tpi,<br>
curves, b1, mhz, seconds_per_curve; optional tpb and exponent. Missing widths may interpolate or<br>
extrapolate. Costs are already amortized.<br>
总流程 Auto B2 的 Stage1 秒/曲线 CSV；列为 container_bits 或 target_bits、tpi、curves、b1、mhz、<br>
seconds_per_curve；可选 tpb、exponent。缺失位宽可插值或外推，成本已摊销。

### stage1_cost_mhz

```text
stage1_cost_mhz=<x:Z,0..10000>; default=0
```

Stage1 target frequency for inverse-frequency cost scaling. 0 uses the CSV reference frequency; this<br>
does not change GPU clocks.<br>
Stage1 成本反频率缩放的目标频率；0 使用 CSV 参考频率，不修改 GPU 时钟。

### stage2_target_factor_bits

```text
stage2_target_factor_bits=<x:Z,0|2..16384>; default=0
```

Target prime-factor size for Dickman semismooth success probability. 0 derives a recommendation from<br>
B1; this is distinct from the full modulus size.<br>
Dickman 半光滑成功概率的目标素因子位数；0 根据 B1 推荐，与完整模数位宽不同。

### stage2_tune_condition_tag

```text
stage2_tune_condition_tag=<text>; default=""; empty=none
```

Optional measurement-condition label, e.g. a fixed clock or power setting. Distinct labels preserve<br>
separate fastest summaries; ignored environment allows reuse.<br>
可选测量条件标签，例如固定频率或功耗。不同标签保留独立最快汇总；忽略 environment 时允许复用。

## GUI settings / GUI 配置 — [GUI]

### NumWorkers

```text
NumWorkers=<x:Z,1..64>; default=1
```

Number of workers managed by the GUI, each using its [Worker #N] settings. Does not change task<br>
curve counts in worktodo. Concurrent workers should have separate queues and output paths.<br>
设置 GUI 管理的 worker 数量。每个 worker 使用对应的 [Worker #N] 设置；此键不会修改 worktodo 中每项任<br>
务的曲线数。并行运行时应给各 worker 配置独立队列和输出路径。

### exe

```text
exe=<exe>; default=""; empty=beside_GUI_then_PATH
```

Worker executable launched by the GUI. Empty first searches for the default worker beside the GUI,<br>
then the executable search path. Use this to select a built Stage1 executable.<br>
指定 GUI 启动 worker 使用的程序路径。留空先查找 GUI 同目录的默认 worker 程序，再按程序搜索路径查找；<br>
可用此键选择已构建的 Stage1 可执行文件。

### language

```text
language=<xml-stem>; default=english
```

GUI language, specified by the language XML filename stem in localization, without .xml. Missing<br>
resources or unavailable glyphs follow the GUI's display fallback rules.<br>
选择 GUI 界面语言，值为 localization 目录下语言 XML 文件的文件名主体，不带 .xml。找不到所需资源或字<br>
体不能显示该语言时，按 GUI 的回退规则使用可显示的界面。

### localization_dir

```text
localization_dir=<dir>; default=""; empty=GUI_exe_dir/localization
```

Directory containing GUI language XML files. Empty uses localization beside the GUI executable.<br>
Development builds also try the source-tree language resources.<br>
指定 GUI 语言 XML 文件目录。留空使用 GUI 可执行文件旁的 localization 目录；从开发构建目录运行时还会<br>
尝试源码中的语言资源目录。

### font

```text
font=<ttf|ttc>; default=""; empty=auto
```

GUI .ttf or .ttc font file. Empty selects a system font for the interface language. A Chinese<br>
interface requires a font with Chinese glyphs.<br>
指定 GUI 使用的 .ttf 或 .ttc 字体文件。留空由程序按界面语言选择系统字体；需要中文界面时应使用包含中<br>
文字形的字体。

### font_size

```text
font_size=[auto|<x:R,6..96>]; default=auto; x<6=>auto; auto=15*DPI
```

GUI font size in pixels; fractional values are accepted. auto or empty selects 15*DPI scale. Values<br>
below 6 or nonfinite values restore auto; values above 96 clamp to 96. Manual sizes bypass automatic<br>
sizing.<br>
设置 GUI 字号，单位为像素，可使用小数。auto 或空值按 15*DPI 比例自动选择；小于 6 或非有限数恢复自动<br>
，大于 96 按 96 处理。手动字号不会再按自动规则选择大小。

### font_snap

```text
font_snap=[0|1]; default=1; off:0|off|false|no; case:sensitive
```

Snap glyphs to the pixel grid: 1 enables, 0 disables. Compare clarity at fractional sizes or<br>
different DPI scales. off, false, and no also disable; other values enable. String matching is<br>
case-sensitive.<br>
控制字体是否对齐像素网格。1 开启，0 关闭；小数字号或不同 DPI 下可比较显示清晰度。兼容 off、false、no<br>
关闭，其他值按开启处理，字符串区分大小写。

### refresh_hz

```text
refresh_hz=<x:Z,1..60>; default=10; unit=Hz
```

GUI refresh frequency in Hz. Higher values can smooth charts and status updates at additional UI<br>
cost. Does not change GPU computation frequency.<br>
设置 GUI 刷新频率，单位为 Hz。提高可使图表和状态显示更平滑，但增加界面刷新开销；不改变 GPU 计算迭代<br>
频率。

### gpu_poll_ms

```text
gpu_poll_ms=<x:Z,100..2^31-1>; default=500; unit=ms
```

GPU monitoring interval in milliseconds, at least 100. Larger values reduce NVML polling frequency.<br>
Affects monitoring updates only, not compute-kernel scheduling.<br>
设置 GPU 监控查询间隔，单位为毫秒，最小为 100。增大可减少 NVML 查询频率；此键只影响监控数据更新，不<br>
控制计算 kernel 的调度。

### priority

```text
priority=[idle|below_normal|normal|above_normal|high]; default=below_normal
```

Priority of worker processes launched by the GUI. below_normal is the default; lower priority makes<br>
concurrent desktop use easier. This is not a GPU exclusivity or utilization setting.<br>
设置 GUI 启动的 worker 进程优先级。below_normal 为默认，较低优先级便于同时使用电脑；此键不是 GPU 独<br>
占或显卡占用率设置。

### window

```text
window=<x:Z>,<y:Z>,<w:Z>,<h:Z>; default=120,80,1500,900; w,h>0
```

Saved main-window geometry: x,y,width,height. Updated on normal GUI exit; manual editing is usually<br>
unnecessary. Width and height should be positive.<br>
保存 GUI 主窗口的位置和大小，依次为 x,y,宽,高。GUI 正常退出时会更新；通常无需手动编辑，宽和高应为正<br>
数。

### dock_layout

```text
dock_layout=<escaped_blob>; default=""; empty=default
```

Escaped GUI docking-layout data. Empty uses the default layout; the GUI writes it automatically.<br>
Incomplete manual edits may prevent layout restoration.<br>
保存 GUI 面板停靠布局的转义数据。留空使用默认布局；由 GUI 自动写回，手动修改不完整数据可能使布局无法<br>
恢复。

### dock_layout_ver

```text
dock_layout_ver=<x:Z,x>=0>; default=0; GUI_managed; current=4
```

Version of the saved docking layout, managed by the GUI. Older layouts are rebuilt using the current<br>
default. This does not select a GUI feature version.<br>
保存已写入布局的版本编号，由 GUI 管理。版本早于当前默认布局时会重建布局；此键不是用户选择 GUI 功能版<br>
本的开关。

### start_tab

```text
start_tab=[workers|detail|gen]; default=workers
```

Panel initially shown by the GUI: workers opens the worker list, detail the details panel, and gen<br>
the task generator. Does not change worker autostart behavior.<br>
选择 GUI 启动时优先显示的面板。workers 为 worker 列表，detail 为详情，gen 为任务生成器；不改变是否自<br>
动启动 worker。

### exit_confirm

```text
exit_confirm=[ask|stop|kill]; default=ask
```

Behavior when closing the GUI with active workers: ask prompts, stop requests a graceful stop and<br>
waits for checkpoints, kill terminates immediately. Forced termination may lose progress not yet<br>
checkpointed.<br>
控制关闭 GUI 时如何处理运行中的 worker。ask 先询问，stop 直接请求正常停止并等待检查点，kill 立即终止<br>
；强制终止时未写入的计算进度可能丢失。

### graceful_stop_ms

```text
graceful_stop_ms=<x:Z,1000..3600000>; default=300000; unit=ms
```

Maximum wait in milliseconds after requesting a graceful worker stop. A new checkpoint or process<br>
exit may end the wait early; workers still running at the deadline are terminated. Allow enough time<br>
for current Stage1 checkpoint writes.<br>
设置 GUI 请求正常停止后等待的最长时间，单位为毫秒。检测到新检查点或进程自行退出时可提前结束；到期仍<br>
未退出时强制终止。应为当前 Stage1 检查点耗时留出余量。

### results_json

```text
results_json=<path>; default=results.json.txt; empty=GUI_exe_dir/results.json.txt
```

GUI factor-results JSONL file. Empty uses results.json.txt in the GUI executable directory. This<br>
summary is separate from stage2_results_file, which provides Stage2 resume receipts.<br>
指定 GUI 汇总命中结果的 JSONL 文件。留空使用 GUI 可执行文件目录下的 results.json.txt；这是 GUI 的结<br>
果汇总文件，与 Stage2 用于续跑对账的 stage2_results_file 分开。

### results_txt

```text
results_txt=<path>; default=results.txt; empty=GUI_exe_dir/results.txt
```

Readable GUI result summary derived from JSONL. Empty uses results.txt in the GUI executable<br>
directory. Preserve the source JSONL as well for long-term result storage.<br>
指定 GUI 汇总结果的可读文本文件，由 JSONL 结果派生。留空使用 GUI 可执行文件目录下的 results.txt；需<br>
要长期保留结果时同时妥善保存 JSONL 来源。

## GUI worker settings / GUI worker 配置 — [Worker #N]

### name

```text
name=<text>; default=Worker #{N}; N:worker_index
```

Worker display name in the GUI. Unset shows Worker #N, where N is the worker index. Changes only the<br>
name, not device or task assignment.<br>
设置此 worker 在 GUI 中显示的名称。未配置时显示 Worker #N，其中 N 为 worker 编号；只改变显示名称，不<br>
改变设备或任务分配。

### autostart

```text
autostart=[0|1]; default=0; x!=0:on
```

Automatically launch this worker when the GUI starts: 0 disables; nonzero enables. Does not affect<br>
starting a worker directly from the command line.<br>
控制 GUI 启动后是否自动启动此 worker。0 不启动，非零值启动；此设置不影响从命令行直接启动 worker 的行<br>
为。

### extra_args

```text
extra_args=<argv>; default=""
```

Additional command-line arguments for this GUI-launched worker. Separate arguments with spaces;<br>
quote paths containing spaces. Overrides depend on options supported by the selected executable.<br>
向 GUI 启动的 worker 追加命令行参数。参数以空格分隔，含空格的路径使用双引号；命令行覆盖配置文件的选<br>
项应以实际 worker 支持的参数为准。

### gpucurves

```text
gpucurves=<x:Z,x>=0>; default=0; S1_queue:ignored; CLI:-gpucurves:independent
```

Legacy GUI curve-count display field, inheriting the global value if present. Stage1 queue curve<br>
counts come from task lines; this field does not override them. The single-run -gpucurves CLI option<br>
is independent.<br>
保留旧 GUI 配置的曲线数显示字段，可继承全局同名值。Stage1 队列执行的曲线数由各任务行决定，此键不会覆<br>
盖任务；直接单次运行的 -gpucurves 是独立命令行参数。

## Compatibility aliases / 兼容别名

Legacy names remain readable; use canonical keys in new configurations. Direct aliases share the canonical value rules and have no separate defaults; conversion switches use the mappings below.<br>
旧名称仍可读取，新配置建议使用对应的正式键。直接别名沿用正式键的取值规则，不另设默认值；转换开关按下面列出的映射处理。

### cpu_affinity

```text
cpu_affinity => affinity
```

Compatibility alias for affinity. Use affinity in new configurations.<br>
与 affinity 相同，用于兼容旧配置；新配置建议使用 affinity。

### gpuckpt_seconds

```text
gpuckpt_seconds => ckpt_seconds
```

Legacy GPU checkpoint name. Use ckpt_seconds; the interval and unit remain the same.<br>
旧 GPU 检查点间隔名称，现统一使用 ckpt_seconds；单位和行为相同。

### edwards_backend

```text
edwards_backend => backend
```

Legacy CPU Edwards backend name. Shares the backend setting.<br>
旧 CPU Edwards 后端名称，现与 backend 共用设置。

### mont_backend

```text
mont_backend => backend
```

Legacy CPU Montgomery backend name. Shares the backend setting.<br>
旧 CPU Montgomery 后端名称，现与 backend 共用设置。

### edbackend

```text
edbackend => backend
```

Legacy abbreviation of edwards_backend. Shares the backend setting.<br>
edwards_backend 的旧缩写，现与 backend 共用设置。

### edwards_threads

```text
edwards_threads => stage1_threads
```

Legacy Edwards thread-count name. Shares stage1_threads.<br>
旧 Edwards 线程数名称，现与 stage1_threads 共用设置。

### mont_threads

```text
mont_threads => stage1_threads
```

Legacy Montgomery thread-count name. Shares stage1_threads.<br>
旧 Montgomery 线程数名称，现与 stage1_threads 共用设置。

### edthreads

```text
edthreads => stage1_threads
```

Legacy Edwards thread-count abbreviation. Shares stage1_threads.<br>
旧 Edwards 线程数缩写，现与 stage1_threads 共用设置。

### edwards_naf_w

```text
edwards_naf_w => naf_w
```

Legacy Edwards NAF-window name. Shares naf_w.<br>
旧 Edwards NAF 窗口名称，现与 naf_w 共用设置。

### ednafw

```text
ednafw => naf_w
```

Legacy Edwards NAF-window abbreviation. Shares naf_w.<br>
旧 Edwards NAF 窗口缩写，现与 naf_w 共用设置。

### mont_save_pattern

```text
mont_save_pattern => save_name_pattern
```

Legacy CPU Montgomery save filename pattern. Shares save_name_pattern.<br>
旧 CPU Montgomery 存档名称模式，现与 save_name_pattern 共用设置。

### edwards

```text
edwards => method; domain=<x:Z>; x!=0:edwards; x=0:no_change
```

Legacy method switch. Nonzero selects method=edwards; 0 leaves the method unchanged. Use method<br>
directly in new configurations.<br>
旧方法开关。非零值选择 method=edwards，0 不改变当前方法；新配置直接使用 method。

### mont

```text
mont => method; domain=<x:Z>; x!=0:mont; x=0:no_change
```

Legacy method switch. Nonzero selects method=mont; 0 leaves the method unchanged. Use method<br>
directly in new configurations.<br>
旧方法开关。非零值选择 method=mont，0 不改变当前方法；新配置直接使用 method。

### edwards_mersenne

```text
edwards_mersenne => field; domain=<enum>; on:mersenne,mersenne:mersenne,mers:mersenne,yes:mersenne,1:mersenne,off:montgomery,montgomery:montgomery,mont:montgomery,no:montgomery,0:montgomery; else:auto
```

Legacy CPU reduction switch. on, yes, 1, or mersenne selects Mersenne; off, no, 0, or montgomery<br>
selects Montgomery; other values select auto. Use field in new configurations.<br>
旧 CPU 约减方法开关。on、yes、1 或 mersenne 相关值选择 Mersenne，off、no、0 或 montgomery 相关值选择<br>
Montgomery，其他值自动选择；新配置使用 field。

### emersenne

```text
emersenne => field; domain=<enum>; on:mersenne,mersenne:mersenne,mers:mersenne,yes:mersenne,1:mersenne,off:montgomery,montgomery:montgomery,mont:montgomery,no:montgomery,0:montgomery; else:auto
```

Same conversion as edwards_mersenne. Use field in new configurations.<br>
与旧 edwards_mersenne 开关相同；新配置使用 field。

### mont_torsion

```text
mont_torsion => exponent; domain=<x:Z>; x=12:choose12; else:lcm
```

Legacy scalar switch. 12 maps to exponent=choose12; other integers map to exponent=lcm. Use exponent<br>
in new configurations.<br>
旧指数选择开关。12 对应 exponent=choose12，其他整数对应 exponent=lcm；新配置使用 exponent。

### debug_log

```text
debug_log => stage2_debug_log; only_if(target=absent)
```

Compatibility alias for the Stage2 debug switch. Used only if stage2_debug_log is absent; the<br>
explicit canonical key always takes precedence.<br>
Stage2 的兼容调试开关，仅在未配置 stage2_debug_log 时生效；一旦正式键存在，以正式键为准。

## Repository maintenance / 源码仓库维护

`config/ecm_options.json` -> `tools/gen/generate_ecm_config.py`

See `docs/architecture/CONFIGURATION.md` in the source repository; release packages include only this configuration reference.<br>
维护流程见源码仓库的 `docs/architecture/CONFIGURATION.md`；发布包仅附本配置说明。
