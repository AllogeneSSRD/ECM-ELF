# ecm.toml Reference

<!-- Generated from config/ecm_options.json. -->

[简体中文](ECM_TOML_REFERENCE.zh-CN.md)

Put `config_version = 3` before the first table. A file without a version uses the legacy format. See [configuration migration](architecture/CONFIGURATION.md#升级旧配置).

## Tables and precedence

- `[shared]` contains settings used by both stages.
- `[stage1]` and `[stage2]` contain settings specific to each stage.
- `[gui]` contains GUI-wide settings. Workers cannot override them.
- `[worker.N]` configures worker N (1–1000000). Prefix an override with `shared.`, `stage1.` or `stage2.`.

From lowest to highest priority, the program applies built-in defaults, the global table, the matching worker table and CLI arguments. Duplicate keys or tables, unknown tables, missing values and unclosed strings cause errors. `worktodo.txt` uses separate `[Worker #N]` sections. The program does not read INI files.

Each worker validates shared settings and its own stage. The GUI validates shared, GUI and worker settings and previews recognized compute options. See [configuration ownership](architecture/CONFIGURATION.md#按程序职责读取).

The shared worktodo path and Stage2 relative paths use the TOML directory. Stage1 usually resolves its paths from the executable directory. See [runtime paths](architecture/CONFIGURATION.md#运行时限制) for details.

## [shared]

### computer

```toml
computer = ""
```

Type `string` · Range `<local-computer-name>` · Worker override `[worker.N] shared.computer` · Level `core`.

Optional local computer name in result output. Keep configured values and generated results outside public Git commits.

<a id="shared_finished_file"></a>

### finished_file

```toml
finished_file = "worktodo.finished.txt"
```

Type `string` · Range `<path>` · Worker override `not supported` · Level `core`.

Shared completed-work file for both stages. Relative paths use the configuration directory.

<a id="shared_log_file"></a>

### log_file

```toml
log_file = "ecm.log"
```

Type `string` · Range `<path>` · Worker override `not supported` · Level `core`.

Shared run log. Relative paths use the configuration directory. An empty string disables file logging.

### log_file_max_KB

```toml
log_file_max_KB = 10000
```

Type `u64` · Range `[1000..1000000]` · Worker override `not supported` · Level `core`.

Rotate the shared run log after its size exceeds this limit. 1 KB is 1000 bytes.

<a id="priority"></a>

### queue_order

```toml
queue_order = "file_order"
```

Type `string` · Range `[file_order|stage1_first|stage2_first]` · Worker override `[worker.N] shared.queue_order` · Level `core`.

Legacy v1 key `shared.priority`

file_order follows the queue. stage1_first or stage2_first runs the preferred stage until it has no remaining tasks, then selects the other stage.

<a id="skip"></a>

### queue_skip

```toml
queue_skip = "none"
```

Type `string` · Range `[none|stage1|stage2]` · Worker override `[worker.N] shared.queue_skip` · Level `core`.

Legacy v1 key `shared.skip`

none allows both stages. stage1 or stage2 leaves that stage's tasks in the queue.

<a id="shared_tune_file"></a>

### tune_file

```toml
tune_file = "tune.toml"
```

Type `string` · Range `<path>` · Worker override `[worker.N] shared.tune_file` · Level `core`.

CUDA Stage1 and Stage2 use this file together. Defaults to tune.toml beside ecm.toml. A worker can override the path for both stages. An explicit empty value disables automatic persistence for both stages.

### user

```toml
user = ""
```

Type `string` · Range `<local-user-id>` · Worker override `[worker.N] shared.user` · Level `core`.

Optional local result user identity for Stage1 and Stage2. Never put real account values in committed templates or fixtures.

<a id="worktodo"></a>

### worktodo_file

```toml
worktodo_file = "worktodo.txt"
```

Type `string` · Range `<path>` · Worker override `[worker.N] shared.worktodo_file` · Level `core`.

Legacy v1 key `shared.worktodo`

Shared Stage1/Stage2 task queue for the current worker. ECMSTAGE1 and ECMSTAGE2 select the engine; Stage1 and Stage2 keep separate finished records. One worker process consumes this queue at a time.

## [stage1]

<a id="save_sync_dir_1"></a>

### backup.directory_1

```toml
backup.directory_1 = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage1.backup.directory_1` · Level `advanced`.

Legacy v1 key `stage1.save_sync_dir_1`

First destination for Stage1 save synchronization. Empty disables this destination; sync_mode controls the files and synchronization policy. Backup only; this path never authorizes publishing a Prime95 task.

<a id="save_sync_dir_2"></a>

### backup.directory_2

```toml
backup.directory_2 = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage1.backup.directory_2` · Level `advanced`.

Legacy v1 key `stage1.save_sync_dir_2`

Second destination for Stage1 save synchronization. Both destinations may be enabled; empty disables only this destination. Backup only; this path never authorizes publishing a Prime95 task.

<a id="sync_mode"></a>

### backup.mode

```toml
backup.mode = "incremental"
```

Type `string` · Range `[incremental|full]` · Worker override `[worker.N] stage1.backup.mode` · Level `advanced`.

Legacy v1 key `stage1.sync_mode`

Save synchronization after a Stage1 assignment: incremental copies files updated during that assignment; full copies all eligible saves. Startup always performs a full synchronization.

<a id="ckpt_backups"></a>

### checkpoint_backups

```toml
checkpoint_backups = 3
```

Type `int` · Range `<x:Z,0..16>` · Worker override `[worker.N] stage1.checkpoint_backups` · Level `advanced`.

Legacy v1 key `stage1.ckpt_backups`

Maximum number of rotated Stage1 checkpoint backups. The primary checkpoint is additional; 0 keeps only the primary. Corrupt or unreadable generations are skipped on resume. A completed Stage1 task removes its primary checkpoint and all backups.

<a id="ckpt_seconds"></a>

### checkpoint_interval_seconds

```toml
checkpoint_interval_seconds = 600
```

Type `double` · Range `<x:R,finite,x>=0>` · Worker override `[worker.N] stage1.checkpoint_interval_seconds` · Level `advanced`.

Legacy v1 key `stage1.ckpt_seconds`

Stage1 checkpoint interval in seconds. 0 disables periodic checkpoints; negative values act as 0. Normal stop or Ctrl+C can still request a checkpoint. Actual writes wait for the backend's safe checkpoint boundary.

<a id="affinity"></a>

### cpu_affinity

```toml
cpu_affinity = ""
```

Type `string` · Range `[none|auto|<n>,...|<a>-<b>]` · Worker override `[worker.N] stage1.cpu_affinity` · Level `advanced`.

Legacy v1 key `stage1.affinity`

CPU affinity list/ranges, e.g. 0-7,10. Indices must be 0..1023, with at most 1024 entries. Empty, auto or none leaves threads unpinned. Invalid entries are rejected with their source and allowed grammar, including values hidden by worker/CLI overrides.

<a id="impl"></a>

### cpu_implementation

```toml
cpu_implementation = "auto"
```

Type `string` · Range `[auto|simd|scalar|gwnum|slice list|empty]` · Worker override `[worker.N] stage1.cpu_implementation` · Level `core`.

Legacy v1 key `stage1.impl`

CPU Stage1 arithmetic. A single mode forces its usual behavior; auto or an empty value selects by CPU capability and batch size. Above 10240 modulus bits, auto requires AVX512-IFMA. In config v3, a list such as simd[:6144], scalar[:4096], gwnum[4096:], auto[] tries each candidate in order on the actual modulus after removing known factors. Overlapping slices are allowed. The first available backend whose slice contains the width is selected. Slices include the start and exclude the stop. A missing start means 1, a missing stop means unbounded, and [] is equivalent to [:]. Whitespace is ignored. Use auto[] for an explicit automatic fallback. simd requires AVX512-IFMA. scalar uses GMP and supports widths above 10240 bits only for Edwards. gwnum requires Montgomery PARAM0 and a local gwnum.dll beside the Windows x64 worker. GPU runs accept only automatic mode.

<a id="cpu_mont_chain"></a>

### cpu_montgomery_chain

```toml
cpu_montgomery_chain = "ladder"
```

Type `string` · Range `[ladder|prac]` · Worker override `[worker.N] stage1.cpu_montgomery_chain` · Level `advanced`.

Legacy v1 key `stage1.cpu_mont_chain`

Montgomery CPU point multiplication. prac requires target=cpu, curve=mont, impl=simd and B1 in 2..4294967295. It shares the scalar convention and final save format with ladder, with its own in-flight checkpoint state.

<a id="reduction"></a>

### cpu_simd_reduction

```toml
cpu_simd_reduction = "auto"
```

Type `string` · Range `[auto|montgomery|mersenne-fold]` · Worker override `[worker.N] stage1.cpu_simd_reduction` · Level `advanced`.

Legacy v1 key `stage1.reduction`

CPU SIMD modular reduction. auto selects for the input; mersenne-fold requires N=2^p-1; montgomery uses general reduction. An explicit reduction requires impl=simd, so it cannot be silently ignored when auto chooses scalar.

<a id="stage1_threads"></a>

### cpu_threads

```toml
cpu_threads = 0
```

Type `u32` · Range `<x:Z,0..2^32-1>` · Worker override `[worker.N] stage1.cpu_threads` · Level `advanced`.

Legacy v1 key `stage1.stage1_threads`

CPU Stage1 thread count; 0 selects automatically. Applies only to CPU arithmetic and does not set CUDA threads per block or threads per integer.

### edwards_doubling

```toml
edwards_doubling = 1
```

Type `int` · Range `[0|1]` · Worker override `[worker.N] stage1.edwards_doubling` · Level `advanced`.

Doubling formula for new CUDA CGBN Edwards NAF batches. 0 selects HWCD, 1 selects 2XY (default). Applies to the main loop. Restoring a checkpoint uses the saved formula. If this key is omitted, the legacy ECM_EDWARDS_DBL_ALGO environment variable can override the default. CPU, CUDA NTT, and binary Edwards do not use this setting.

<a id="naf_w"></a>

### edwards_naf_window_bits

```toml
edwards_naf_window_bits = 0
```

Type `int` · Range `[0|2..12]; CPU: [0|3..12]` · Worker override `[worker.N] stage1.edwards_naf_window_bits` · Level `advanced`.

Legacy v1 key `stage1.naf_w`

Requested Edwards NAF window. 0 starts at CPU 12 or CUDA CGBN 6. Larger windows reduce additions and grow the dictionary as 2^(w-2). Memory admission may lower the window on a fresh task. Partial checkpoints retain their saved window.

<a id="exp_cache"></a>

### exponent_cache_dir

```toml
exponent_cache_dir = ""
```

Type `string` · Range `[off|none|0|-|<dir>]` · Worker override `[worker.N] stage1.exponent_cache_dir` · Level `advanced`.

Legacy v1 key `stage1.exp_cache`

Directory for Stage1 scalar and PRAC caches. Empty uses cache/ at the checkout root, or cache/ next to the executable for standalone installs. off, none, 0, or - disables caching. Explicit relative paths use the current working directory. Small B1 values may skip writing the scalar cache.

### gpu_algorithm

```toml
gpu_algorithm = "cgbn"
```

Type `int|string` · Range `[1|2|cgbn|ntt]` · Worker override `[worker.N] stage1.gpu_algorithm` · Level `advanced`.

Stage1 GPU algorithm: 1/CGBN or 2/NTT, case-insensitive. CUDA NTT supports Edwards and Montgomery PARAM0 on full Mersenne moduli, with 1–128 curves. Edwards NTT requires naf_w=0.

<a id="stage1_backend"></a>

### gpu_backend

```toml
gpu_backend = "auto"
```

Type `string` · Range `[auto|cuda|opencl]` · Worker override `[worker.N] stage1.gpu_backend` · Level `core`.

Legacy v1 key `stage1.stage1_backend`

Windows GPU worker. Auto prefers an installed CUDA worker with a visible CUDA device at the selected index, then OpenCL. Explicit selection requires that worker beside the executable. Device indices belong to the selected backend. Override with --gpu-backend.

<a id="device"></a>

### gpu_device

```toml
gpu_device = 0
```

Type `int` · Range `<x:Z,0..device_count-1>` · Worker override `[worker.N] stage1.gpu_device` · Level `core`.

Legacy v1 key `stage1.device`

Zero-based GPU device ID for Stage1. Stage2 inherits it unless stage2_device is explicitly set. IDs refer to the device list exposed by the executable's actual GPU backend.

<a id="handoff_mode"></a>

### handoff.mode

```toml
handoff.mode = "ecm-elf"
```

Type `string` · Range `[off|ecm-elf|prime95]` · Worker override `[worker.N] stage1.handoff.mode` · Level `core`.

Send successful Stage1 tasks with usable points and no factor to ECM-ELF Stage2 by default. Use off to keep the save without queuing Stage2, or prime95 to select Prime95.

<a id="handoff_unsupported"></a>

### handoff.unsupported

```toml
handoff.unsupported = "ecm-elf"
```

Type `string` · Range `[ecm-elf|keep-save]` · Worker override `[worker.N] stage1.handoff.unsupported` · Level `advanced`.

Fallback for saves unsupported by the Prime95 protocol.

<a id="auto_stage2_workers"></a>

### handoff.workers

```toml
handoff.workers = ""
```

Type `string` · Range `[<n>|<n>,...|<a>-<b>|auto]` · Worker override `[worker.N] stage1.handoff.workers` · Level `advanced`.

Legacy v1 key `stage1.auto_stage2_workers`

Receiving workers when auto_stage2=1: a single ID, comma list or ranges within 1..1000000. Select the shortest candidate queue; ties use the lowest ID. auto uses the destination's existing worker sections, falling back to GUI NumWorkers when none exist. Missing explicit sections are created. Empty keeps the current worker and rewrites in place when the destination is the same file; an external file receives that same worker number. Unreadable destinations remain pending for retry.

<a id="auto_stage2_worktodo"></a>

### handoff.worktodo_file

```toml
handoff.worktodo_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage1.handoff.worktodo_file` · Level `advanced`.

Legacy v1 key `stage1.auto_stage2_worktodo`

Destination queue when auto_stage2=1. Empty inherits this worker's effective shared worktodo. Accepts an absolute file path or a filename/relative path resolved beside ecm.toml. Save filenames in delivered tasks are absolute. Stage2 finished receipts use the receiving worker's stage2_finished setting from this TOML. Worker sections can override this setting.

<a id="lcm"></a>

### montgomery_lcm_multiplier

```toml
montgomery_lcm_multiplier = "1"
```

Type `string` · Range `[1|12]` · Worker override `[worker.N] stage1.montgomery_lcm_multiplier` · Level `advanced`.

Legacy v1 key `stage1.lcm`

Montgomery Stage1 scalar is lcm(1..B1) multiplied by 1 or 12. Edwards uses its fixed 48*lcm scalar and rejects 12. This choice changes the output point and must match the Stage2 save convention.

### naf_memory_mb

```toml
naf_memory_mb = 0
```

Type `int` · Range `[0|1..1048576]` · Worker override `[worker.N] stage1.naf_memory_mb` · Level `advanced`.

Edwards Stage1 working-memory cap in MB (1 MB = 1,000,000 bytes). 0 uses the available-memory budget. The runtime takes the smaller of this cap and the free-memory budget after reserve. Applies to the CPU NAF working set or CUDA CGBN device buffers. Scalar construction and CUDA host preparation have separate free-memory checks. A positive value starts at 1 MB.

<a id="ntt_bpw"></a>

### ntt.bits_per_word

```toml
ntt.bits_per_word = 0
```

Type `int` · Range `0 or [16,31]` · Worker override `[worker.N] stage1.ntt.bits_per_word` · Level `advanced`.

Legacy v1 key `stage1.ntt_bpw`

Bits per digit. Zero selects the widest exact linear packing or the shortest safe weighted transform. In weighted mode, an explicit value is an upper bound on the largest digit.

<a id="ntt_convolution"></a>

### ntt.convolution

```toml
ntt.convolution = "auto"
```

Type `string` · Range `[auto|linear|weighted]` · Worker override `[worker.N] stage1.ntt.convolution` · Level `advanced`.

Legacy v1 key `stage1.ntt_convolution`

Auto selects the measured backend or calibrates on a cache miss. Explicit modes select exact linear convolution or weighted cyclic convolution (IBDWT). Both use Goldilocks and the same Stage1 task and checkpoint format.

<a id="ntt_length"></a>

### ntt.length

```toml
ntt.length = 0
```

Type `int` · Range `0, 2^k in [2,65536], or 5*2^k in [2560,20480] (weighted)` · Worker override `[worker.N] stage1.ntt.length` · Level `advanced`.

Legacy v1 key `stage1.ntt_length`

Transform length. Zero derives safe lengths, including shorter supported radix-5 shapes for weighted convolution.

<a id="ntt_memory_mib"></a>

### ntt.memory_mib

```toml
ntt.memory_mib = 0
```

Type `int` · Range `[0,1048576]` · Worker override `[worker.N] stage1.ntt.memory_mib` · Level `advanced`.

Legacy v1 key `stage1.ntt_memory_mib`

Total Stage1 NTT VRAM budget in MiB. Zero uses 80% of free VRAM after CUDA initialization. An explicit cap is also bounded by available VRAM.

<a id="ntt_outer_max"></a>

### ntt.outer_radix_limit

```toml
ntt.outer_radix_limit = 0
```

Type `int` · Range `[0,4]` · Worker override `[worker.N] stage1.ntt.outer_radix_limit` · Level `advanced`.

Legacy v1 key `stage1.ntt_outer_max`

Maximum fused radix-2 layers per outer pass. Zero uses the profile.

<a id="ntt_profile"></a>

### ntt.profile

```toml
ntt.profile = 0
```

Type `int` · Range `[0|1|2]` · Worker override `[worker.N] stage1.ntt.profile` · Level `advanced`.

Legacy v1 key `stage1.ntt_profile`

Per-length transform profile: 0 automatic, 1 full tile (up to 12 bits), 2 split tile (up to 10 bits). Explicit tile and outer overrides take precedence.

<a id="ntt_tile"></a>

### ntt.tile

```toml
ntt.tile = 0
```

Type `int` · Range `[0,12]` · Worker override `[worker.N] stage1.ntt.tile` · Level `advanced`.

Legacy v1 key `stage1.ntt_tile`

Fused tile width in radix-2 layers. Zero uses the profile.

<a id="ntt_warp_tail"></a>

### ntt.warp_tail

```toml
ntt.warp_tail = "auto"
```

Type `bool|string` · Range `[auto|false|true|0|1]` · Worker override `[worker.N] stage1.ntt.warp_tail` · Level `advanced`.

Legacy v1 key `stage1.ntt_warp_tail`

Enable warp shuffles for the local tail of the tile transform. Auto uses the selected profile.

<a id="ntt_window"></a>

### ntt.window_bits

```toml
ntt.window_bits = -1
```

Type `int` · Range `[-1,64]` · Worker override `[worker.N] stage1.ntt.window_bits` · Level `advanced`.

Legacy v1 key `stage1.ntt_window`

Ladder bits per CUDA Graph replay. -1 uses the profile; 0 uses direct launches.

<a id="kernel_add"></a>

### opencl.kernel_add

```toml
opencl.kernel_add = ""
```

Type `string` · Range `[auto|<id>|<alias>]` · Worker override `[worker.N] stage1.opencl.kernel_add` · Level `advanced`.

Legacy v1 key `stage1.kernel_add`

OpenCL modular-addition kernel ID or alias. Empty or auto lets the backend choose; automatic selection is normally suitable. CUDA ignores this key.

<a id="kernel_mul"></a>

### opencl.kernel_mul

```toml
opencl.kernel_mul = ""
```

Type `string` · Range `[auto|<id>|<alias>]` · Worker override `[worker.N] stage1.opencl.kernel_mul` · Level `advanced`.

Legacy v1 key `stage1.kernel_mul`

OpenCL modular-multiplication kernel ID or alias. Empty or auto lets the backend choose. Intended for comparing or debugging known kernels; automatic selection is normally suitable. CUDA ignores this key.

<a id="kernel_special_mult"></a>

### opencl.kernel_special_mult

```toml
opencl.kernel_special_mult = ""
```

Type `string` · Range `[auto|<id>|<alias>]` · Worker override `[worker.N] stage1.opencl.kernel_special_mult` · Level `advanced`.

Legacy v1 key `stage1.kernel_special_mult`

OpenCL special-multiplication kernel ID or alias. Empty or auto lets the backend choose; automatic selection is normally suitable. CUDA ignores this key.

<a id="kernel_sqr"></a>

### opencl.kernel_sqr

```toml
opencl.kernel_sqr = ""
```

Type `string` · Range `[auto|<id>|<alias>]` · Worker override `[worker.N] stage1.opencl.kernel_sqr` · Level `advanced`.

Legacy v1 key `stage1.kernel_sqr`

OpenCL modular-squaring kernel ID or alias. Empty or auto lets the backend choose; automatic selection is normally suitable. CUDA ignores this key.

<a id="kernel_sub"></a>

### opencl.kernel_sub

```toml
opencl.kernel_sub = ""
```

Type `string` · Range `[auto|<id>|<alias>]` · Worker override `[worker.N] stage1.opencl.kernel_sub` · Level `advanced`.

Legacy v1 key `stage1.kernel_sub`

OpenCL modular-subtraction kernel ID or alias. Empty or auto lets the backend choose; automatic selection is normally suitable. CUDA ignores this key.

<a id="tpi"></a>

### opencl.threads_per_integer

```toml
opencl.threads_per_integer = 8
```

Type `u32` · Range `[1|2|4|8|16|32]` · Worker override `[worker.N] stage1.opencl.threads_per_integer` · Level `advanced`.

Legacy v1 key `stage1.tpi`

OpenCL threads per integer, subject to limb-layout constraints. CUDA selects TPI by operand width and ignores this key. This is not the CUDA block size.

<a id="wg_size"></a>

### opencl.workgroup_size

```toml
opencl.workgroup_size = 0
```

Type `int` · Range `<x:Z,x>=0>` · Worker override `[worker.N] stage1.opencl.workgroup_size` · Level `advanced`.

Legacy v1 key `stage1.wg_size`

OpenCL work-group size. 0 lets the backend select automatically; explicit values must satisfy device and kernel limits. CUDA ignores this setting.

### param

```toml
param = "auto"
```

Type `string` · Range `[auto|e|m0|m2|m3]` · Worker override `[worker.N] stage1.param` · Level `core`.

Stage1 curve construction. auto uses m0 on CPU and m3 on GPU. A task save prefix selects its own construction.

<a id="p95_keep_aid"></a>

### prime95.keep_aid

```toml
prime95.keep_aid = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage1.prime95.keep_aid` · Level `advanced`.

Legacy v1 key `stage1.p95_keep_aid`

Keep PrimeNet assignment IDs in handoffs: 1 retains IDs not known to be invalid; 0 strips all IDs. IDs confirmed invalid by Prime95 logs are always removed to prevent repeated rejection.

<a id="p95_recover_lost"></a>

### prime95.recover_lost

```toml
prime95.recover_lost = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage1.prime95.recover_lost` · Level `advanced`.

Legacy v1 key `stage1.p95_recover_lost`

Recover handoffs deleted by Prime95 because of invalid assignment IDs. 1 checks handoff records and Prime95 logs, then reposts confirmed lost assignments without an ID; 0 disables recovery. Only confirmed prior handoffs qualify.

<a id="p95_sync_dir"></a>

### prime95.save_dir

```toml
prime95.save_dir = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage1.prime95.save_dir` · Level `advanced`.

Legacy v1 key `stage1.p95_sync_dir`

Dedicated save destination for Prime95 handoff; required with p95_worktodo_path. Backup destinations are independent. Existing published save prefixes must remain unchanged; only complete appended records may extend them.

<a id="p95_add_workers"></a>

### prime95.workers

```toml
prime95.workers = ""
```

Type `string` · Range `[<n>|<n>,...|<a>-<b>|auto]` · Worker override `[worker.N] stage1.prime95.workers` · Level `advanced`.

Legacy v1 key `stage1.p95_add_workers`

Prime95 workers eligible to receive handoffs: one ID, a comma list, or ranges. The shortest eligible queue is selected. auto reads NumWorkers from prime.txt in the same directory. Empty omits the worker header and uses Prime95's default worker.

<a id="p95_worktodo_path"></a>

### prime95.worktodo_file

```toml
prime95.worktodo_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage1.prime95.worktodo_file` · Level `advanced`.

Legacy v1 key `stage1.p95_worktodo_path`

Prime95 worktodo path for supported Montgomery Stage1 handoff. Edwards and GPU PARAM2 ignore this path and keep saves local; they do not access Prime95 transfer files. Requires p95_sync_dir. The save is validated and published before worktodo.add.

### progress_average_seconds

```toml
progress_average_seconds = 0
```

Type `double` · Range `[0..60]` · Worker override `[worker.N] stage1.progress_average_seconds` · Level `advanced`.

Time window for live Stage1 CPU/CUDA s/curve and remaining-time estimates. Zero uses the latest valid sample without averaging; positive values average completed work over measured sample time, up to 60 seconds. CUDA samples exclude preparation and checkpoint I/O. This does not change computation, slicing, final timing or log refresh frequency.

### progress_color

```toml
progress_color = "cyan"
```

Type `string` · Range `[none|red|green|yellow|blue|magenta|cyan|white|grey]` · Worker override `[worker.N] stage1.progress_color` · Level `advanced`.

Color of the Stage1 console progress bar. none disables color. This does not change ordinary file logs or arithmetic.

### progress_log_seconds

```toml
progress_log_seconds = 60
```

Type `double` · Range `<x:R,finite>` · Worker override `[worker.N] stage1.progress_log_seconds` · Level `advanced`.

Interval in seconds between Stage1 progress entries in file logs. Positive values limit frequency; 0 suppresses intermediate progress; negative values log every refresh. Completion at 100% is still logged. Console refresh frequency is unchanged.

<a id="stage1_results_file"></a>

### results_file

```toml
results_file = "stage1_results.jsonl"
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage1.results_file` · Level `core`.

Legacy v1 key `stage1.stage1_results_file`

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Canonical internal per-curve Stage1 JSONL with CRC32 and execution metadata, separate from submission output.

<a id="tmp_dir"></a>

### save_dir

```toml
save_dir = "."
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage1.save_dir` · Level `core`.

Legacy v1 key `stage1.tmp_dir`

Output directory for Stage1 curve saves. Stage2 also searches here when stage2_save_dir is unset. This is not a Stage2 RAM or VRAM budget.

### save_name_pattern

```toml
save_name_pattern = "{curve}_{target}_{b1}.save"
```

Type `string` · Range `<pattern>` · Worker override `[worker.N] stage1.save_name_pattern` · Level `advanced`.

Automatic Stage1 save filename pattern for CPU and GPU. {curve}=e|m0|m2|m3; {target}=m<exponent> for Mersenne assignments, otherwise bit<actual_bits>; {n}=original assignment exponent when available, otherwise actual bits; {b1}=compact B1. Missing curve prefixes are added. In ECMSTAGE1 queues, typed save prefixes override TOML curve/param and the final filename field supplies B1; untyped names use the TOML construction. Finished/handoff lines name the actual prefixed save. Explicit CLI save paths are preserved. Empty restores the default.

### sigma

```toml
sigma = 0
```

Type `u64` · Range `<x:Z,0..2^64-1>` · Worker override `[worker.N] stage1.sigma` · Level `advanced`.

Initial curve sigma. 0 chooses a random value; consecutive curves use sigma+i without 64-bit wraparound. CUDA PARAM0 accepts the full 64-bit range, while PARAM2/3 batches must stay below 2^32. CPU runs warn if a sigma exceeds Prime95's signed reader range; this is not a Stage1 limit. A nonzero ECM/ECM2 assignment sigma overrides this setting.

<a id="stage1_submission_file"></a>

### submission_file

```toml
submission_file = "results.json.txt"
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage1.submission_file` · Level `core`.

Legacy v1 key `stage1.stage1_submission_file`

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Stage1 submission JSON: seal NF once the whole task completes (B2=B1); publish verified factors immediately as individual curves. Private ledgers live under the output directory data/.

### target

```toml
target = "gpu"
```

Type `string` · Range `[cpu|gpu]` · Worker override `[worker.N] stage1.target` · Level `core`.

Run Stage1 on CPU or on the GPU backend linked into this executable. Edwards GPU requires ecm_cuda; the target does not select between CUDA and OpenCL.

### verbose

```toml
verbose = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage1.verbose` · Level `core`.

Normal console detail. In Stage2, true shows phase transitions and false keeps curve summaries; stage2_debug_log independently controls debug logging. Stage1 uses the selected backend's normal verbosity behavior.

## [stage2]

<a id="stage2_auto_b2"></a>

### auto_b2

```toml
auto_b2 = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.auto_b2` · Level `core`.

Legacy v1 key `stage2.stage2_auto_b2`

Choose B2 when every source supplies zero. 1 optimizes workflow benefit per time including Stage1 cost; 0 disables. Requires usable Stage2 costs and positive Stage1 s/curve. B1 never gates tune reuse; stage2_tune_ignore controls other conditions. Memory and carrier divisibility are checked on the current input. Estimates retain uncertainty labels; the legacy .cprof interface keeps its original qualification checks.

<a id="stage2_auto_max_b2"></a>

### auto_max_b2

```toml
auto_max_b2 = 0
```

Type `u64` · Range `<x:Z,0..2^63-8193>` · Worker override `[worker.N] stage2.auto_max_b2` · Level `advanced`.

Legacy v1 key `stage2.stage2_auto_max_b2`

Auto B2 upper search bound; 0 means 2.6e12. Unmeasured values may be estimated; explicit task B2 takes precedence.

<a id="stage2_auto_min_b2"></a>

### auto_min_b2

```toml
auto_min_b2 = 0
```

Type `u64` · Range `<x:Z,0..2^63-8193>` · Worker override `[worker.N] stage2.auto_min_b2` · Level `advanced`.

Legacy v1 key `stage2.stage2_auto_min_b2`

Auto B2 lower search bound; 0 means B1+1. Unmeasured values may be estimated; explicit task B2 takes precedence.

<a id="stage2_b2"></a>

### b2

```toml
b2 = 0
```

Type `u64` · Range `<x:Z,x=0|B1<x<=2^63-1-8192>` · Worker override `[worker.N] stage2.b2` · Level `core`.

Legacy v1 key `stage2.stage2_b2`

Stage2 upper bound when neither CLI nor task supplies a nonzero B2. 0 defers to other sources. Explicit B2 must exceed the save's B1. Larger B2 usually increases coverage, time, and VRAM demand. Precedence: nonzero CLI B2, then task B2, then this key.

### b2_ratio

```toml
b2_ratio = 0
```

Type `u64` · Range `[0|1..1000000]` · Worker override `[worker.N] stage2.b2_ratio` · Level `core`.

In GPU Stage2, a nonzero ratio sets B2 to the saved B1 times this value for queue and direct save runs. Zero leaves B2 selection to the other settings. Overflow is rejected.

<a id="stage2_b2_round_chain"></a>

### b2_round_chain

```toml
b2_round_chain = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.b2_round_chain` · Level `advanced`.

Legacy v1 key `stage2.stage2_b2_round_chain`

For each runtime candidate D and arithmetic, increase B2 to the smallest whole-P giant count whose last point chunk can use a chain. Runtime tuning includes that candidate's actual work; measurements remain keyed by actual B2. Logs/results keep requested and actual B2. Default 0 preserves bounds. Overflow, live memory and the AutoB2 maximum remain hard limits. Startup cost probes keep fixed bounds; explicit ECM calibration still requires rounding off. CLI --b2-round-chain overrides this worker TOML.

<a id="stage2_progress_dir"></a>

### checkpoint_dir

```toml
checkpoint_dir = ""
```

Type `string` · Range `<dir>` · Worker override `[worker.N] stage2.checkpoint_dir` · Level `advanced`.

Legacy v1 key `stage2.stage2_progress_dir`

Directory for per-task .ecm_ckpt_stage2_<bits>_<B1>.dat progress files. Empty uses the legacy progress file's directory if explicitly set, otherwise the queue directory. Task IDs bind AID, N, full save contents and skip. On restart the current effective B2/Auto B2 and requested curve count override the checkpoint policy; completed curves retain their actual bounds. A target below the completed count warns and seals all completed work. New checkpoints contain completed curve state and pending submission data. They are removed after publication and queue advancement.

<a id="stage2_progress_file"></a>

### checkpoint_legacy_file

```toml
checkpoint_legacy_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage2.checkpoint_legacy_file` · Level `legacy`.

Legacy v1 key `stage2.stage2_progress_file`

Legacy single-file progress import path. Empty checks the old queue-path plus worker-suffix .progress file. A matching task's completed count and pending receipt are imported into its own .ecm_ckpt_stage2_<bits>_<B1>.dat file. An unrelated legacy task is retained and does not stop this task. New progress files use stage2_progress_dir; no arithmetic state is checkpointed.

<a id="stage2_auto_carrier"></a>

### cuda.auto_carrier

```toml
cuda.auto_carrier = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.cuda.auto_carrier` · Level `advanced`.

Legacy v1 key `stage2.stage2_auto_carrier`

Add M=2^p-1 for validated ECMSTAGE2=1,2,p,-1 saves when N divides M and 2<=p<=16384. Compare ordinary cost at target_bits with Mersenne cost at arithmetic_bits=p, using stage2_prediction_policy. Full M_p already has native Mersenne arithmetic. Explicit carrier options take priority; a cofactor save needs a matching task to infer p.

<a id="stage2_baby_backend"></a>

### cuda.baby_backend

```toml
cuda.baby_backend = "auto"
```

Type `string` · Range `[auto|scalar|coop32]` · Worker override `[worker.N] stage2.cuda.baby_backend` · Level `advanced`.

Legacy v1 key `stage2.stage2_baby_backend`

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Baby point arithmetic: auto retains the scalar baseline until checked runtime comparisons qualify coop32; scalar forces one thread per point; coop32 requests 32 cooperating CUDA threads for arithmetic widths 8193..16384 bits. Unsupported widths or device-baby fallback use scalar. Runtime tuning compares one backend alternative at fixed D/layout/chain before expanding D. CLI --baby-backend overrides this worker key; an omitted key preserves an explicit NTT_BABY_COOP override. OpenCL accepts auto/scalar only.

<a id="stage2_parallel"></a>

### cuda.curve_concurrency

```toml
cuda.curve_concurrency = "serial"
```

Type `string` · Range `[auto|serial|dual]` · Worker override `[worker.N] stage2.cuda.curve_concurrency` · Level `advanced`.

Legacy v1 key `stage2.stage2_parallel`

CUDA Stage2 curve scheduling. auto measures serial and eligible dual-slot modes. serial is the default. dual requests two slots subject to device admission. CLI: --parallel auto|0|1|2.

<a id="stage2_giant_backend"></a>

### cuda.giant_backend

```toml
cuda.giant_backend = "auto"
```

Type `string` · Range `[auto|scalar|coop16|coop32]` · Worker override `[worker.N] stage2.cuda.giant_backend` · Level `advanced`.

Legacy v1 key `stage2.stage2_giant_backend`

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Giant chain arithmetic: auto keeps scalar until checked comparisons qualify coop32 or coop16, then selects the fastest measured route including point startup checks. At fixed D/layout/chain, try one alternative at a time after baby selection; reuse paired regions by arithmetic width, chain occupancy and memory route before extending D. coop16 uses a fixed 16384-bit container; coop32 uses width-matched containers. Cooperative modes require CUDA arithmetic widths 8193..16384 bits and an eligible chain; seeds and [D]Q are unchanged. Explicit scalar/coop16/coop32 locks the backend. CLI --giant-backend overrides this worker key; omission preserves explicit NTT_GIANT_COOP. OpenCL accepts auto/scalar only.

<a id="stage2_low_cpu_wait"></a>

### cuda.low_cpu_wait

```toml
cuda.low_cpu_wait = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.cuda.low_cpu_wait` · Level `advanced`.

Legacy v1 key `stage2.stage2_low_cpu_wait`

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Opt-in low CPU policy: CUDA blocking device scheduling (4) and bounded event polling. Blocking reads keep their native path; elapsed time may increase.
Default false selects auto scheduling (0) and native event waits. CLI --low-cpu-wait 0|1 overrides this key.
An explicit key sets NTT_CUDA_WAIT_MODE and NTT_HOST_EVENT_WAIT, and clears NTT_HOST_READBACK_WAIT. Omission preserves expert environment overrides. FIFO recycling is independent.

<a id="stage2_pack_zero"></a>

### cuda.pack_zero

```toml
cuda.pack_zero = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.cuda.pack_zero` · Level `advanced`.

Legacy v1 key `stage2.stage2_pack_zero`

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Fuse input digit packing and zero padding in one GPU kernel per operand. Covers contiguous, resident gather and reverse inputs; unsupported shapes use the original packer. Does not change arithmetic checks, D, NTT length or device workspace capacity. CLI --pack-zero overrides this key; an explicit key overrides NTT_S4_PACK_ZERO. Without an override, the built-in default applies; legacy .cprof cost profiles retain false and require false explicitly if this key is set.

<a id="stage2_parallel_tune_curves"></a>

### cuda.parallel_tune_curves

```toml
cuda.parallel_tune_curves = 5
```

Type `u64` · Range `[2..16]` · Worker override `[worker.N] stage2.cuda.parallel_tune_curves` · Level `advanced`.

Legacy v1 key `stage2.stage2_parallel_tune_curves`

Number of curves in each CUDA Stage2 queue tuning window. Windows of 2 to 4 curves compare full-window throughput. Longer windows also check steady-state completion throughput.

<a id="stage2_reduce_backend"></a>

### cuda.reduce_backend

```toml
cuda.reduce_backend = "auto"
```

Type `string` · Range `[auto|scalar|coop32]` · Worker override `[worker.N] stage2.cuda.reduce_backend` · Level `advanced`.

Legacy v1 key `stage2.stage2_reduce_backend`

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Ordinary S4 coefficient reduction: auto keeps scalar until repeated same-shape comparisons qualify coop32, then uses the fastest checked total including point startup checks. At fixed D/layout/chain/point backends, it adds at most one missing cooperative variant at a time (two-pass or merged), then extends D after checked comparisons. CUDA arithmetic widths 8193..16384 bits can use shared warp remainders; smaller widths and exact Mersenne reduction keep their kernels. Actual reduce_coop, threads and kernel identity stay separate in ECM timing rows even if environment matching is ignored; standalone NTT tables remain reusable. CLI --reduce-backend overrides this worker key; omission preserves explicit NTT_S4_COOP_REDUCE=0|32. OpenCL uses scalar.

<a id="stage2_seed_backend"></a>

### cuda.seed_backend

```toml
cuda.seed_backend = "auto"
```

Type `string` · Range `[auto|scalar|coop32]` · Worker override `[worker.N] stage2.cuda.seed_backend` · Level `advanced`.

Legacy v1 key `stage2.stage2_seed_backend`

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Giant paired-seed and GPU [D]Q arithmetic: auto keeps scalar until checked, repeated same-shape comparisons qualify coop32, then selects the fastest measured route including point startup checks. At fixed D/layout/chain/baby/giant arithmetic, try one alternative per family before extending D; reuse paired regions by arithmetic width, chain occupancy and memory route. Cooperative arithmetic requires CUDA widths 8193..16384 bits and a real chain; other cases remain scalar, preserving coordinates and nonunit fallback. Actual seed_coop is part of each ECM timing row even when environment matching is ignored; NTT tables remain reusable. CLI --seed-backend overrides this worker key; omission preserves explicit NTT_GIANT_SEED_COOP=0|32. OpenCL uses scalar.

<a id="stage2_weighted_ntt"></a>

### cuda.weighted_ntt

```toml
cuda.weighted_ntt = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.cuda.weighted_ntt` · Level `advanced`.

Legacy v1 key `stage2.stage2_weighted_ntt`

Opt-in CUDA polynomial convolution. Fuses the first DIF stage into direct digit packing and reconstructs both halves in the final inverse stage. Requires stage2_pack_zero. Calls without cached roots or direct device packing use the full-length path. CLI --weighted-ntt 0|1 overrides this key.

<a id="stage2_parallel_workspace"></a>

### cuda.workspace_policy

```toml
cuda.workspace_policy = "auto"
```

Type `string` · Range `[auto|independent|leased]` · Worker override `[worker.N] stage2.cuda.workspace_policy` · Level `advanced`.

Legacy v1 key `stage2.stage2_parallel_workspace`

independent gives each CUDA Stage2 slot a workspace. leased lets the second slot prepare and compute baby points before leasing the heavy workspace at the F-tree boundary. auto compares both and uses leased directly if only it fits.

<a id="stage2_d"></a>

### d

```toml
d = 0
```

Type `u64` · Range `<x:Z,x=0|x>=6,x%2=0>` · Worker override `[worker.N] stage2.d` · Level `advanced`.

Legacy v1 key `stage2.stage2_d`

Manual Stage2 step D. 0 selects the shape automatically. Explicit values must be even, at least 6, and satisfy the engine's layout constraints. D affects tree size, batch count, and memory demand; automatic selection is normally suitable.

<a id="stage2_debug_log"></a>

### diagnostics.debug_log

```toml
diagnostics.debug_log = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.diagnostics.debug_log` · Level `advanced`.

Legacy v1 key `stage2.stage2_debug_log`

Write Stage2 debug details to a separate log: true enables and false disables. verbose=true does not enable debug logging; enabling this option does not send all debug details to the console.

<a id="stage2_debug_log_file"></a>

### diagnostics.debug_log_file

```toml
diagnostics.debug_log_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage2.diagnostics.debug_log_file` · Level `advanced`.

Legacy v1 key `stage2.stage2_debug_log_file`

Separate Stage2 debug log, written only when stage2_debug_log is enabled. Empty restores the default filename. Default names gain a worker suffix; explicit paths are used unchanged.

<a id="stage2_factorize_hits"></a>

### diagnostics.decompose_factors

```toml
diagnostics.decompose_factors = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.diagnostics.decompose_factors` · Level `advanced`.

Legacy v1 key `stage2.stage2_factorize_hits`

Further decompose found factors: 1 uses the shared diagnostic service (local PARI DLL first, external GP second); GMP is never selected automatically. 0 keeps only factors extracted and verified by Stage2. Only complete, proven decomposition sets factorization_complete. Unfinished diagnostics preserve the original factor and successful curve status.

<a id="stage2_factor_timeout"></a>

### diagnostics.factor_timeout_seconds

```toml
diagnostics.factor_timeout_seconds = 30
```

Type `u64` · Range `<x:Z,1..600>` · Worker override `[worker.N] stage2.diagnostics.factor_timeout_seconds` · Level `advanced`.

Legacy v1 key `stage2.stage2_factor_timeout`

Timeout in seconds for each external GP fallback decomposition call. Local PARI calls have a 5-second process deadline and a 10-second cumulative session budget per curve worker. Explicit GMP analysis remains bounded to 2 seconds per call. These diagnostics never change the original factor or successful Stage2 computation status.

<a id="stage2_gp"></a>

### diagnostics.gp

```toml
diagnostics.gp = ""
```

Type `string` · Range `<exe>` · Worker override `[worker.N] stage2.diagnostics.gp` · Level `advanced`.

Legacy v1 key `stage2.stage2_gp`

External PARI/GP fallback executable for optional factor decomposition. The local PARI DLL is tried first. gp.exe follows Windows executable search rules; a full path is accepted. GP failure preserves partial PARI results and never enables GMP automatically. Dataset scripts retain their separate --gp resolution rules.

<a id="stage2_ntt_counts"></a>

### diagnostics.ntt_length_counts

```toml
diagnostics.ntt_length_counts = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.diagnostics.ntt_length_counts` · Level `advanced`.

Legacy v1 key `stage2.stage2_ntt_counts`

Print a per-curve NTT length count table at the end of production Stage2: 1 enables; 0 disables collection and output. Counts mathematical transforms (two forwards and one inverse per convolution), including polynomial initialization and main work, excluding Stage1 and independent arithmetic selftests. Shared by Edwards and Montgomery; no GPU counters or extra synchronization. CLI --ntt-counts overrides the selected worker TOML.

<a id="stage2_factor_only"></a>

### diagnostics.skip_prime_attribution

```toml
diagnostics.skip_prime_attribution = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.diagnostics.skip_prime_attribution` · Level `advanced`.

Legacy v1 key `stage2.stage2_factor_only`

Skip diagnostic prime attribution, which identifies Stage2 primes responsible for a hit. 1 skips attribution while retaining factor extraction and validation; 0 follows the engine's diagnostic policy. This does not disable decomposition or change stage2_factorize_hits.

<a id="stage2_giant_chain"></a>

### giant_chain

```toml
giant_chain = "64"
```

Type `string` · Range `[64|32|auto]` · Worker override `[worker.N] stage2.giant_chain` · Level `advanced`.

Legacy v1 key `stage2.stage2_giant_chain`

Points per sequential CUDA thread chain, independent of CUDA block size. 64 preserves the baseline; 32 explicitly selects shorter chains. auto retains 64 until stage2_tune_min_repeats admitted runtime observations support 32, with bounded useful-curve probes. Policies have separate tune identities. Hit/nonunit replays the legacy 64-point full interval. CLI --giant-chain overrides this worker TOML.

<a id="stage2_giant_tail_balance"></a>

### giant_tail_balance

```toml
giant_tail_balance = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.giant_tail_balance` · Level `advanced`.

Legacy v1 key `stage2.stage2_giant_tail_balance`

Allow the last two giant point chunks to borrow whole P batches so both can use chains. B2, capacity, coverage and chunk count are unchanged. Default false. No actual borrow keeps the original cost/cache identity and avoids extra replay or synchronization. Actual borrowing keeps an independent tune identity and replays the original interval on a hit/nonunit; that replay may need more workspace. CLI --giant-tail-balance overrides this worker TOML.

<a id="stage2_backend"></a>

### gpu_backend

```toml
gpu_backend = "auto"
```

Type `string` · Range `[auto|cuda|opencl]` · Worker override `[worker.N] stage2.gpu_backend` · Level `core`.

Legacy v1 key `stage2.stage2_backend`

Windows GPU worker. Auto prefers an installed CUDA worker with a visible CUDA device at the selected index, then OpenCL. Explicit selection requires that worker beside the executable. Device indices belong to the selected backend. Override with --gpu-backend.

<a id="stage2_device"></a>

### gpu_device

```toml
gpu_device = -1
```

Type `int` · Range `[-1|0..2^31-1]` · Worker override `[worker.N] stage2.gpu_device` · Level `core`.

Legacy v1 key `stage2.stage2_device`

Stage2 CUDA device ID. -1 inherits Stage1's device; 0 or greater selects a CUDA device explicitly.

<a id="stage2_groot_balanced"></a>

### groot_balanced

```toml
groot_balanced = "auto"
```

Type `bool|string` · Range `[auto|false|true|0|1]` · Worker override `[worker.N] stage2.groot_balanced` · Level `advanced`.

Legacy v1 key `stage2.stage2_groot_balanced`

G-root polynomial layout. "auto" uses the qualified NTT-cliff model and measured tuning; true forces the balanced tile layout, false forces the contiguous layout. Default "auto". This does not rebalance giant point chunks or change B2/D/P. CUDA only; OpenCL accepts "auto". CLI --groot-layout auto|true|false overrides this worker TOML; numeric CLI aliases 0/1 remain supported. Tune rows store the resolved layout as 0/1.

<a id="stage2_log_level"></a>

### log_level

```toml
log_level = -1
```

Type `int|string` · Range `[quiet|curve|phases|batches|debug|0..4]` · Worker override `[worker.N] stage2.log_level` · Level `advanced`.

Legacy v1 key `stage2.stage2_log_level`

Legacy console level: quiet, curve, phases, batches, and debug map to 0..4. Unset follows verbose. In production, debug enables the separate debug log while normal console detail remains capped at batches. Usually use verbose and stage2_debug_log.

<a id="stage2_arena_mb"></a>

### memory.arena_mib

```toml
memory.arena_mib = 0
```

Type `u64` · Range `<x:Z,0..2^20>` · Worker override `[worker.N] stage2.memory.arena_mib` · Level `advanced`.

Legacy v1 key `stage2.stage2_arena_mb`

Stage2 large-workspace budget in MiB, used to plan available NTT shapes. 0 uses environment settings or automatic planning, not zero VRAM. Too little may force smaller shapes or prevent execution. Tables, coordinates, and other modules may allocate additional VRAM.

<a id="stage2_fold_mb"></a>

### memory.fold_resident_mib

```toml
memory.fold_resident_mib = 640
```

Type `u64` · Range `<x:Z,0..2^20>` · Worker override `[worker.N] stage2.memory.fold_resident_mib` · Level `advanced`.

Legacy v1 key `stage2.stage2_fold_mb`

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Budget in MiB for keeping fold accumulators on the GPU. 0 disables this residency. Exceeding the budget may fall back to batching, substantially increasing processing and transfer costs. Independent of arena and batch budgets; not a total VRAM limit.

<a id="stage2_batch_mb"></a>

### memory.s4_batch_mib

```toml
memory.s4_batch_mib = 64
```

Type `u64` · Range `<x:Z,1..2^20>` · Worker override `[worker.N] stage2.memory.s4_batch_mib` · Level `advanced`.

Legacy v1 key `stage2.stage2_batch_mb`

Stage2 S4 batch-buffer budget in MiB. Larger budgets may reduce batching overhead but consume more VRAM. This limits the corresponding buffers, not total process VRAM.

<a id="stage2_opencl_point_chunk"></a>

### opencl.point_chunk

```toml
opencl.point_chunk = 0
```

Type `u64` · Range `[0..4096]` · Worker override `[worker.N] stage2.opencl.point_chunk` · Level `advanced`.

Legacy v1 key `stage2.stage2_opencl_point_chunk`

Maximum points per OpenCL command. Zero uses automatic sizing. A positive value locks the chunk and disables runtime plan trials.

<a id="stage2_opencl_point_group"></a>

### opencl.point_group

```toml
opencl.point_group = 0
```

Type `u64` · Range `[0..4096]` · Worker override `[worker.N] stage2.opencl.point_group` · Level `advanced`.

Legacy v1 key `stage2.stage2_opencl_point_group`

Local OpenCL point work-group size. Zero uses the backend default. Positive sizes must fit both device and kernel limits.

<a id="stage2_opencl_point_plan"></a>

### opencl.point_plan

```toml
opencl.point_plan = "auto"
```

Type `string` · Range `[auto|device|chain|cpu]` · Worker override `[worker.N] stage2.opencl.point_plan` · Level `advanced`.

Legacy v1 key `stage2.stage2_opencl_point_plan`

OpenCL point generation. Auto starts with the qualified plan and can learn from complete curves. Device selects the device ladder plan (including its qualified short batches); chain requests consecutive giant seeds; cpu uses GMP points with OpenCL trees. Manual settings bypass runtime exploration.

<a id="stage2_opencl_point_window"></a>

### opencl.point_window

```toml
opencl.point_window = 0
```

Type `u64` · Range `[0..65536]` · Worker override `[worker.N] stage2.opencl.point_window` · Level `advanced`.

Legacy v1 key `stage2.stage2_opencl_point_window`

Giant coordinate staging window in points. Zero uses the backend plan; positive requests are clipped by the budget and G batch boundaries.

<a id="stage2_opencl_short_point_batch"></a>

### opencl.short_point_batch

```toml
opencl.short_point_batch = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.opencl.short_point_batch` · Level `advanced`.

Legacy v1 key `stage2.stage2_opencl_short_point_batch`

Allow qualified short device batches. False disables batched baby ladders and paired giants from the short-batch policy.

<a id="stage2_opencl_tune_profile"></a>

### opencl.tune_file

```toml
opencl.tune_file = "stage2_opencl_tune.toml"
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage2.opencl.tune_file` · Level `advanced`.

Legacy v1 key `stage2.stage2_opencl_tune_profile`

Native OpenCL point timing profile. Relative to ecm.toml, with per-worker overrides. Empty disables profile learning. Uses stage2_tune_runtime, stage2_tune_min_repeats and stage2_tune_runtime_samples.

<a id="stage2_prediction_policy"></a>

### prediction_policy

```toml
prediction_policy = "model"
```

Type `string` · Range `[model|paired]` · Worker override `[worker.N] stage2.prediction_policy` · Level `advanced`.

Legacy v1 key `stage2.stage2_prediction_policy`

model: prefer exact paired measurements, otherwise compare generic and Mersenne arithmetic-width region estimates with two-sided uncertainty and current memory admission. Missing tune data uses an explicitly estimated 4060-Laptop/1800-MHz basic model. Uncovered NTT/residency regions retain the default. paired: require the original exact-width/B2 measured win. B1 never gates reuse.

<a id="stage2_ratio_adjust"></a>

### ratio_adjust

```toml
ratio_adjust = 1
```

Type `double` · Range `<x:R,finite,x>0>` · Worker override `[worker.N] stage2.ratio_adjust` · Level `advanced`.

Legacy v1 key `stage2.stage2_ratio_adjust`

Multiplier applied to Stage2 time in the Auto B2 model. 1 leaves it unchanged; greater than 1 estimates higher cost, less than 1 lower cost, usually changing selected B2. Adjusts planning cost, not actual GPU speed.

<a id="stage2_relocation"></a>

### relocation

```toml
relocation = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.relocation` · Level `advanced`.

Legacy v1 key `stage2.stage2_relocation`

Optional production GPU Stage2 conservative relocation. Keeps B2 and the original giant endpoint, uses a shorter suffix for clean NF, and replays the original interval on any hit or target-N nonunit (including GCD=N). Rebalances short point chunks on whole G boundaries. Default 0 preserves the original algorithm. CLI --relocation overrides this worker TOML. D/AutoB2 and memory admission use original-interval estimates; relocation does not collect ECM tune samples or run startup ECM calibration, and explicit --tune ecm is unavailable in this mode. Pure NTT samples remain reusable.

<a id="stage2_results_file"></a>

### results_file

```toml
results_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage2.results_file` · Level `core`.

Legacy v1 key `stage2.stage2_results_file`

Optional per-curve Stage2 JSONL for new GPU queue tasks. Empty disables it for queue runs. Direct save runs use the default filename. Keep the original JSONL until legacy checkpoint migration completes.

<a id="stage2_save_dir"></a>

### save_dir

```toml
save_dir = ""
```

Type `string` · Range `<dir>` · Worker override `[worker.N] stage2.save_dir` · Level `advanced`.

Legacy v1 key `stage2.stage2_save_dir`

Directory searched for saves referenced by Stage2 assignments. Empty inherits tmp_dir; absolute save paths in assignments are used directly. Source saves are read and validated without modification.

<a id="stage2_short_calibration"></a>

### short_calibration

```toml
short_calibration = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.short_calibration` · Level `advanced`.

Legacy v1 key `stage2.stage2_short_calibration`

Calibrate once when no suitable ECM tune file exists, required components are missing, or the task has no reusable calibrated width coverage. Partial coverage permits estimates.

<a id="stage2_short_calibration_seconds"></a>

### short_calibration_seconds

```toml
short_calibration_seconds = 10
```

Type `u64` · Range `<x:Z,1..600>` · Worker override `[worker.N] stage2.short_calibration_seconds` · Level `advanced`.

Legacy v1 key `stage2.stage2_short_calibration_seconds`

Short calibration measurement budget, excluding first CUDA initialization; finish the current measurement before stopping.

<a id="stage2_short_ntt_batch_tune"></a>

### short_ntt_batch_tune

```toml
short_ntt_batch_tune = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.short_ntt_batch_tune` · Level `advanced`.

Legacy v1 key `stage2.stage2_short_ntt_batch_tune`

Experimental short NTT batch selection from stage2_tune_profile, for lengths up to 2^19. Minimizes estimated convolution time including the tail, within the current chunk budget; requires at least two measured batch sizes and a predicted gain above 10%. Does not tune kernel configuration. Pure NTT gains do not guarantee faster full Stage2; default false pending full-engine validation.

<a id="stage2_submission_file"></a>

### submission_file

```toml
submission_file = "results.json.txt"
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage2.submission_file` · Level `core`.

Legacy v1 key `stage2.stage2_submission_file`

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Submission-only ECM JSON lines, separate from the internal results JSONL. Uses a reference FFT size from Prime95 mult.asm. Personal identifiers and checksums belong only in local ignored files.

<a id="stage2_target_factor_bits"></a>

### target_factor_bits

```toml
target_factor_bits = 0
```

Type `u64` · Range `<x:Z,0|2..16384>` · Worker override `[worker.N] stage2.target_factor_bits` · Level `advanced`.

Legacy v1 key `stage2.stage2_target_factor_bits`

Target prime-factor size for Dickman semismooth success probability. 0 derives a recommendation from B1; this is distinct from the full modulus size.

<a id="stage2_tune_arithmetic"></a>

### tune.arithmetic

```toml
tune.arithmetic = "both"
```

Type `string` · Range `[both|generic|mersenne]` · Worker override `[worker.N] stage2.tune.arithmetic` · Level `advanced`.

Legacy v1 key `stage2.stage2_tune_arithmetic`

Arithmetic families for explicit width-grid ECM tune: both, generic probable-prime N, or Mersenne 2^p-1 arithmetic after stripping proper-exponent inherited factors and small target divisors. p is the requested grid arithmetic width; actual cofactor width is recorded separately. Real worktodo/save pairs continue to follow stage2_auto_carrier. Invalid or factor-producing synthetic Mersenne fixtures are skipped, never saved as timing measurements.

<a id="stage2_tune_budget_seconds"></a>

### tune.budget_seconds

```toml
tune.budget_seconds = 1800
```

Type `u64` · Range `<x:Z,1..86400>` · Worker override `[worker.N] stage2.tune.budget_seconds` · Level `advanced`.

Legacy v1 key `stage2.stage2_tune_budget_seconds`

Tuning budget including input preparation, measurements and optional warmups. Start each width/B2 at the default model's D, compare neighbors, and follow only measured improvements above 2% plus the MAD noise margin. Reuse completed candidates. Finish an in-flight measurement after the deadline and retain valid summaries.

<a id="stage2_tune_condition_tag"></a>

### tune.condition_tag

```toml
tune.condition_tag = ""
```

Type `string` · Range `<text>` · Worker override `[worker.N] stage2.tune.condition_tag` · Level `advanced`.

Legacy v1 key `stage2.stage2_tune_condition_tag`

Optional measurement-condition label, e.g. a fixed clock or power setting. Distinct labels preserve separate fastest summaries; ignored environment allows reuse.

<a id="stage2_tune_error_limit"></a>

### tune.error_limit

```toml
tune.error_limit = 0.08
```

Type `double` · Range `<x:R,0<x<=1>` · Worker override `[worker.N] stage2.tune.error_limit` · Level `advanced`.

Legacy v1 key `stage2.stage2_tune_error_limit`

Relative error limit for qualifying the nonnegative full-curve cost model by leave-one-out validation. Sparse estimates keep uncertainty labels and cannot override the default D. Local descent compares measured medians and MAD rather than prediction error.

<a id="stage2_tune_ignore"></a>

### tune.ignore

```toml
tune.ignore = "gpu,driver,cuda,backend,environment"
```

Type `string` · Range `[gpu|driver|cuda|backend|memory|environment],...` · Worker override `[worker.N] stage2.tune.ignore` · Level `advanced`.

Legacy v1 key `stage2.stage2_tune_ignore`

Comma-separated conditions ignored for profile reuse. Empty enforces every condition. Memory normally matches execution path; arithmetic legality and current memory admission are always checked.

<a id="stage2_cost_profile"></a>

### tune.legacy_cost_profile

```toml
tune.legacy_cost_profile = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage2.tune.legacy_cost_profile` · Level `legacy`.

Legacy v1 key `stage2.stage2_cost_profile`

Legacy .cprof cost profile for Auto B2 when stage2_tune_profile is empty. Checks the original binary/device/arithmetic and calibration contract; new full ECM tune does not relax those checks. Empty supplies no legacy profile.

<a id="stage2_tune_min_repeats"></a>

### tune.min_repeats

```toml
tune.min_repeats = 2
```

Type `u64` · Range `[1..1000]` · Worker override `[worker.N] stage2.tune.min_repeats` · Level `advanced`.

Legacy v1 key `stage2.stage2_tune_min_repeats`

Minimum checked observations required before runtime D, layout and chain candidates are compared or adopted. Also sets the minimum ECM calibration repeat count; --tune-repeats may request more. Default 2; 1 enables faster screening with no measured dispersion. Must not exceed stage2_tune_runtime_samples. Existing observations are retained; increasing this minimum requires additional evidence. Arithmetic validation is unchanged. CLI: --tune-min-repeats N.

<a id="stage2_tune_runtime"></a>

### tune.runtime_enabled

```toml
tune.runtime_enabled = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.tune.runtime_enabled` · Level `advanced`.

Legacy v1 key `stage2.stage2_tune_runtime`

Accumulate checked no-factor production timings by device, environment and execution shape. Identical complete parameters share one matrix row with repeats, mean time and sample standard deviation; slower observations count too. Row version and UTC update time record provenance only; software upgrades retain compatible timings. Model selection tests a legal Mersenne carrier first, then ordinary and promising D candidates. Candidates that reach stage2_tune_min_repeats are ranked by the fastest mean. B1 does not gate cost reuse. Writes are locked and atomic; cache failure does not fail a completed curve. false disables single-curve runtime trials, collection and writes. Parallel queue decisions use stage2_parallel. OpenCL learns point plans in stage2_opencl_tune_profile at fixed geometry.

<a id="stage2_tune_runtime_samples"></a>

### tune.runtime_sample_limit

```toml
tune.runtime_sample_limit = 10
```

Type `u64` · Range `[1..1000]` · Worker override `[worker.N] stage2.tune.runtime_sample_limit` · Level `advanced`.

Legacy v1 key `stage2.stage2_tune_runtime_samples`

Maximum independent runtime NF observations per identical complete parameters (including B1) and matching device/environment/arithmetic/path, including slower observations and compatible software versions. Identical parameters merge into repeats and mean/stddev; the limit counts observations, not matrix rows. Different parameter rows are retained. Selection requires stage2_tune_min_repeats observations. Lowering this limit preserves existing evidence. CLI: --tune-runtime-samples N.

<a id="stage1_batch"></a>

### tune.stage1_batch

```toml
tune.stage1_batch = 1
```

Type `u64` · Range `<x:Z,1..2^20>` · Worker override `[worker.N] stage2.tune.stage1_batch` · Level `advanced`.

Legacy v1 key `stage2.stage1_batch`

Stage1 batch curve count associated with Auto B2 cost estimation. Selects or checks Stage1 cost conditions; it neither starts a Stage1 batch from Stage2 nor changes the assignment's curve count.

<a id="stage1_cost_csv"></a>

### tune.stage1_cost_csv

```toml
tune.stage1_cost_csv = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage2.tune.stage1_cost_csv` · Level `advanced`.

Legacy v1 key `stage2.stage1_cost_csv`

CSV Stage1 s/curve anchors for workflow Auto B2. Columns: container_bits or target_bits, tpi, curves, b1, mhz, seconds_per_curve; optional tpb and exponent. Missing widths may interpolate or extrapolate. Costs are already amortized.

<a id="stage1_tune_profile"></a>

### tune.stage1_cost_file

```toml
tune.stage1_cost_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage2.tune.stage1_cost_file` · Level `advanced`.

Legacy v1 key `stage2.stage1_tune_profile`

Checked completed CUDA Stage1 cost anchors, used after explicit seconds/curve and CSV. Matches batch, modulus type and exponent; B1 and missing widths scale by B1 and the dispatched container squared. Device/runtime reuse follows stage2_tune_ignore. This does not run Stage1.

<a id="stage1_cost_mhz"></a>

### tune.stage1_cost_mhz

```toml
tune.stage1_cost_mhz = 0
```

Type `u64` · Range `<x:Z,0..10000>` · Worker override `[worker.N] stage2.tune.stage1_cost_mhz` · Level `advanced`.

Legacy v1 key `stage2.stage1_cost_mhz`

Stage1 target frequency for inverse-frequency cost scaling. 0 uses the CSV reference frequency; this does not change GPU clocks.

<a id="stage1_seconds_per_curve"></a>

### tune.stage1_seconds_per_curve

```toml
tune.stage1_seconds_per_curve = 0
```

Type `double` · Range `<x:R,finite,x>0>` · Worker override `[worker.N] stage2.tune.stage1_seconds_per_curve` · Level `advanced`.

Legacy v1 key `stage2.stage1_seconds_per_curve`

Positive Stage1 seconds per curve for Auto B2, already amortized for the actual batch. Does not divide by stage1_batch again. Overrides Stage1 tune measurements. When unset, full ECM tune requires a matching stage1_tune_profile; the legacy component profile uses its own matched Stage1 data. Not a runtime limit or the tune input preparation time.

<a id="stage2_tune_warmup"></a>

### tune.warmup

```toml
tune.warmup = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2.tune.warmup` · Level `advanced`.

Legacy v1 key `stage2.stage2_tune_warmup`

0 starts formal ECM/NTT tuning and startup calibration without a discarded warmup. 1 runs one checked warmup per shape before formal repeats; it consumes the same budget and is excluded from the median/MAD. CLI --tune-warmup 0|1 overrides this key. Mandatory arithmetic checks are always performed.

## [gui]

### dock_layout

```toml
dock_layout = ""
```

Type `string` · Range `<escaped_blob>` · Worker override `not supported` · Level `legacy`.

Escaped GUI docking-layout data. Empty uses the default layout; the GUI writes it automatically. Incomplete manual edits may prevent layout restoration.

### dock_layout_ver

```toml
dock_layout_ver = 0
```

Type `int` · Range `<x:Z,x>=0>` · Worker override `not supported` · Level `legacy`.

Version of the saved docking layout, managed by the GUI. Older layouts are rebuilt using the current default. This does not select a GUI feature version.

### exit_confirm

```toml
exit_confirm = "ask"
```

Type `string` · Range `[ask|stop|kill]` · Worker override `not supported` · Level `advanced`.

Behavior when closing the GUI with active workers: ask prompts, stop requests a safe stop, kill terminates immediately. Stage1 saves a checkpoint; Stage2 finishes its current curve and records completed progress. Forced termination may lose the unfinished Stage1 work or current Stage2 curve.

### font

```toml
font = ""
```

Type `string` · Range `<ttf|ttc>` · Worker override `not supported` · Level `legacy`.

GUI .ttf or .ttc font file. Empty selects a system font for the interface language. A Chinese interface requires a font with Chinese glyphs.

### font_size

```toml
font_size = 0
```

Type `float|string` · Range `[auto|<x:R,6..96>]` · Worker override `not supported` · Level `legacy`.

GUI font size in pixels; fractional values are accepted. auto or empty selects 15*DPI scale. Values below 6 or nonfinite values restore auto; values above 96 clamp to 96. Manual sizes bypass automatic sizing.

### font_snap

```toml
font_snap = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `not supported` · Level `legacy`.

Snap glyphs to the pixel grid: 1 enables, 0 disables. Compare clarity at fractional sizes or different DPI scales. off, false, and no also disable; other values enable. String matching is case-sensitive.

### gpu_clock_ranges

```toml
gpu_clock_ranges = ""
```

Type `string` · Range `<GPU-UUID:minMHz-maxMHz[:auto];...>` · Worker override `not supported` · Level `advanced`.

GUI v2 GPU core-clock drafts keyed by NVML GPU UUID: GPU-<UUID>:minMHz-maxMHz, separated by semicolons. Optional :auto enables one startup apply, every local whole hour, and a one-shot reapply after each continuous 5-second out-of-range episode. Detection reuses the monitor samples. Without :auto, loading never changes hardware. Saving does not apply drafts in this session; Apply activates the draft range and auto option. Reset disables auto in this session without erasing saved drafts. Empty saves none; worker computation settings are unaffected.

### gpu_poll_ms

```toml
gpu_poll_ms = 500
```

Type `int` · Range `<x:Z,100..2^31-1>` · Worker override `not supported` · Level `advanced`.

GPU monitoring interval in milliseconds, at least 100. Larger values reduce NVML polling frequency. Affects monitoring updates only, not compute-kernel scheduling.

### graceful_stop_ms

```toml
graceful_stop_ms = 300000
```

Type `i64` · Range `<x:Z,1000..3600000>` · Worker override `not supported` · Level `advanced`.

Maximum wait in milliseconds after requesting a safe worker stop. Process exit ends the wait; workers still running at the deadline are terminated. Allow enough time for Stage1 checkpoint writes or the full current Stage2 curve. Stage2 curve counters are not intermediate arithmetic saves.

### group_order_gp

```toml
group_order_gp = ""
```

Type `string` · Range `<path-to-gp.exe|empty>` · Worker override `not supported` · Level `legacy`.

External PARI/GP fallback executable for the Group Order Factor GUI. Auto tries the local PARI DLL first; empty searches gp in PATH only when GP is needed. GUI-only path setting; compute workers use the same diagnostic service with independent GP paths.

### language

```toml
language = "english"
```

Type `string` · Range `<xml-stem>` · Worker override `not supported` · Level `advanced`.

GUI language, specified by the language XML filename stem in localization, without .xml. Missing resources or unavailable glyphs follow the GUI's display fallback rules.

### localization_dir

```toml
localization_dir = ""
```

Type `string` · Range `<dir>` · Worker override `not supported` · Level `advanced`.

Directory containing GUI language XML files. Empty uses localization beside the GUI executable. Development builds also try the source-tree language resources.

### process_priority

```toml
process_priority = "below_normal"
```

Type `string` · Range `[idle|below_normal|normal|above_normal|high]` · Worker override `not supported` · Level `advanced`.

Legacy v1 key `gui.priority`

Priority of worker processes launched by the GUI. below_normal is the default; lower priority makes concurrent desktop use easier. This is not a GPU exclusivity or utilization setting.

### refresh_hz

```toml
refresh_hz = 10
```

Type `int` · Range `<x:Z,1..60>` · Worker override `not supported` · Level `advanced`.

Idle GUI frame and worker-status refresh target in Hz. Input messages wake the GUI promptly; higher values smooth charts and status updates at additional UI cost. GPU metrics are sampled independently at gpu_poll_ms, and neither setting changes computation speed.

<a id="exe"></a>

### stage1_exe

```toml
stage1_exe = ""
```

Type `string` · Range `<exe>` · Worker override `not supported` · Level `advanced`.

Legacy v1 key `gui.exe`

Stage1 worker executable. Relative paths resolve beside the GUI; bare names can use PATH. Empty searches ecm_cuda.exe then ecm.exe. The worktodo prefix selects this executable or stage2_exe for each worker.

### stage2_exe

```toml
stage2_exe = ""
```

Type `string` · Range `<exe>` · Worker override `not supported` · Level `advanced`.

Shared executable path for Stage2 workers. Empty searches ecm_cuda_stage2.exe beside the GUI, then a single cuda-stage2* child package, then PATH. Multiple child packages require an explicit path. Each worker receives the same TOML and its own --worker index, using an inherited Stage2 stop event. A current Stage2 build is required; exe remains the separate Stage1 path.

### start_tab

```toml
start_tab = "workers"
```

Type `string` · Range `[workers|detail|gen|prob]` · Worker override `not supported` · Level `legacy`.

Panel initially shown by the GUI: workers opens the worker list, detail the details panel, gen the task generator, and prob the ECM probability calculator. The probability panel can also be opened from View. Does not change worker autostart behavior.

### window

```toml
window = [120, 80, 1500, 900]
```

Type `rect` · Range `<x:Z>,<y:Z>,<w:Z>,<h:Z>` · Worker override `not supported` · Level `legacy`.

Saved main-window geometry: x,y,width,height. Updated on normal GUI exit; manual editing is usually unnecessary. Width and height should be positive.

<a id="numworkers"></a>

### worker_count

```toml
worker_count = 1
```

Type `int` · Range `<x:Z,1..64>` · Worker override `not supported` · Level `core`.

Legacy v1 key `gui.NumWorkers`

Number of workers managed by the GUI. Each uses its own [worker.N] settings. Task curve counts come from worktodo.

## [worker.1]

### autostart

```toml
autostart = false
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `core`.

Legacy v1 key `workers.1.autostart`

Start this worker when the GUI opens. The setting applies to GUI startup.

### extra_args

```toml
extra_args = ""
```

Type `string` · Range `<argv>` · Worker override `not supported` · Level `advanced`.

Legacy v1 key `workers.1.extra_args`

Additional command-line arguments for this GUI-launched worker. Separate arguments with spaces; quote paths containing spaces. Overrides depend on options supported by the selected executable.

### gpucurves

```toml
gpucurves = 0
```

Type `int` · Range `<x:Z,x>=0>` · Worker override `not supported` · Level `legacy`.

Legacy v1 key `workers.1.gpucurves`

Legacy GUI curve-count display field, inheriting the global value if present. Stage1 queue curve counts come from task lines; this field does not override them. The single-run -gpucurves CLI option is independent.

### name

```toml
name = "Worker #{N}"
```

Type `string` · Range `<text>` · Worker override `not supported` · Level `advanced`.

Legacy v1 key `workers.1.name`

Worker display name in the GUI. Unset shows Worker #N, where N is the worker index. Changes only the name, not device or task assignment.

## Maintenance

Edit `config/ecm_options.json` and run `python tools/gen/generate_ecm_config.py`. Commit the schema and generated files together.
