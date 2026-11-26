# ecm.toml Reference

<!-- Generated from config/ecm_options.json. -->

[简体中文](ECM_TOML_REFERENCE.zh-CN.md)

Put the required `config_version = 4` before the first table. See [configuration migration](architecture/CONFIGURATION.md#升级旧配置).

## Tables and precedence

- `[shared]` contains settings used by both stages.
- `[stage1]` and `[stage2]` contain settings specific to each stage.
- `[gui]` contains GUI-wide settings. Workers cannot override them.
- `[worker.N]` configures worker N (1–1000000). Use the same flat key as in its global table.

From lowest to highest priority, the program applies built-in defaults, the global table, the matching worker table and CLI arguments. Duplicate keys or tables, unknown tables, missing values and unclosed strings cause errors. `worktodo.txt` uses separate `[Worker #N]` sections. The program does not read INI files.

Each worker validates shared settings and its own stage. The GUI validates shared, GUI and worker settings and previews recognized compute options. See [configuration ownership](architecture/CONFIGURATION.md#按程序职责读取).

The shared worktodo path and Stage2 relative paths use the TOML directory. Stage1 usually resolves its paths from the executable directory. See [runtime paths](architecture/CONFIGURATION.md#运行时限制) for details.

## [shared]

### computer

```toml
computer = ""
```

Type `string` · Range `<local-computer-name>` · Worker override `[worker.N] computer` · Level `core`.

Optional local computer name in result output. Keep configured values and generated results outside public Git commits.

<a id="shared_cpu_affinity"></a>

### cpu_affinity

```toml
cpu_affinity = ""
```

Type `string` · Range `[none|auto|<n>,...|<a>-<b>]` · Worker override `[worker.N] cpu_affinity` · Level `advanced`.

Logical CPUs for the worker process, including host work for GPU backends. Use zero-based indices and inclusive ranges, e.g. 0,2,4-7. Empty or auto lets CPU Stage2 select one logical CPU per physical core, preferring performance cores. An explicit list allows SMT if the requested thread count needs siblings in that list. none leaves CPU Stage2 unbound and disables its tune reuse and collection. For other backends, empty, auto and none leave affinity unchanged. A worker may override the shared list for both stages.

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

### log_file_max_kb

```toml
log_file_max_kb = 10000
```

Type `u64` · Range `[1000..1000000]` · Worker override `not supported` · Level `core`.

Rotate the shared run log after its size exceeds this limit. 1 KB is 1000 bytes.

<a id="priority"></a>

### queue_order

```toml
queue_order = "file_order"
```

Type `string` · Range `[file_order|stage1_first|stage2_first]` · Worker override `[worker.N] queue_order` · Level `core`.

file_order follows the queue. stage1_first or stage2_first runs the preferred stage until it has no remaining tasks, then selects the other stage.

<a id="skip"></a>

### queue_skip

```toml
queue_skip = "none"
```

Type `string` · Range `[none|stage1|stage2]` · Worker override `[worker.N] queue_skip` · Level `core`.

none allows both stages. stage1 or stage2 leaves that stage's tasks in the queue.

<a id="shared_tune_file"></a>

### tune_file

```toml
tune_file = "tune.toml"
```

Type `string` · Range `<path>` · Worker override `[worker.N] tune_file` · Level `core`.

CUDA Stage1, CUDA Stage2 and CPU GWNUM Stage2 share this file. Defaults to tune.toml beside ecm.toml. Workers can override the path. An explicit empty value disables automatic persistence.

### user

```toml
user = ""
```

Type `string` · Range `<local-user-id>` · Worker override `[worker.N] user` · Level `core`.

Optional local result user identity for Stage1 and Stage2. Never put real account values in committed templates or fixtures.

<a id="workflow_worker_count"></a>

### worker_count

```toml
worker_count = 1
```

Type `int` · Range `<x:Z,1..64>` · Worker override `not supported` · Level `advanced`.

Workflow worker count for GUI v2 and automatic ECM-ELF handoff within the same queue. Use worker numbers 1 through this count. Automatic handoff excludes workers that queue_skip Stage2.

<a id="worktodo"></a>

### worktodo_file

```toml
worktodo_file = "worktodo.txt"
```

Type `string` · Range `<path>` · Worker override `[worker.N] worktodo_file` · Level `core`.

Shared queue. ECM, ECM2 and ECMSTAGE1 select Stage1; ECMSTAGE2 and ECMSTAGE2N select Stage2. One process consumes a worker queue at a time.

## [stage1]

<a id="save_sync_dir_1"></a>

### backup_directory_1

```toml
backup_directory_1 = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] backup_directory_1` · Level `advanced`.

First destination for Stage1 save synchronization. Empty disables this destination; backup_mode controls the files and synchronization policy. Backup only; this path never authorizes publishing a Prime95 task.

<a id="save_sync_dir_2"></a>

### backup_directory_2

```toml
backup_directory_2 = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] backup_directory_2` · Level `advanced`.

Second destination for Stage1 save synchronization. Both destinations may be enabled; empty disables only this destination. Backup only; this path never authorizes publishing a Prime95 task.

<a id="sync_mode"></a>

### backup_mode

```toml
backup_mode = "incremental"
```

Type `string` · Range `[incremental|full]` · Worker override `[worker.N] backup_mode` · Level `advanced`.

Save synchronization after a Stage1 assignment: incremental copies files updated during that assignment; full copies all eligible saves. Startup always performs a full synchronization.

<a id="ckpt_backups"></a>

### checkpoint_backups

```toml
checkpoint_backups = 3
```

Type `int` · Range `<x:Z,0..16>` · Worker override `[worker.N] checkpoint_backups` · Level `advanced`.

Maximum number of rotated Stage1 checkpoint backups. The primary checkpoint is additional; 0 keeps only the primary. Corrupt or unreadable generations are skipped on resume. A completed Stage1 task removes its primary checkpoint and all backups.

<a id="ckpt_seconds"></a>

### checkpoint_interval_seconds

```toml
checkpoint_interval_seconds = 600
```

Type `double` · Range `<x:R,finite,x>=0>` · Worker override `[worker.N] checkpoint_interval_seconds` · Level `advanced`.

Stage1 checkpoint interval in seconds. 0 disables periodic checkpoints; negative values act as 0. Normal stop or Ctrl+C can still request a checkpoint. Actual writes wait for the backend's safe checkpoint boundary.

<a id="impl"></a>

### cpu_implementation

```toml
cpu_implementation = "auto"
```

Type `string` · Range `[auto|simd|scalar|gwnum|slice list|empty]` · Worker override `[worker.N] cpu_implementation` · Level `core`.

Stage1 arithmetic policy when the target is CPU. A single mode forces its usual behavior; auto or an empty value selects by CPU capability and batch size. Above 10240 modulus bits, auto requires AVX512-IFMA. A list such as simd[:6144], scalar[:4096], gwnum[4096:], auto[] tries each candidate in order on the actual modulus after removing known factors. Overlapping slices are allowed. The first available backend whose slice contains the width is selected. Slices include the start and exclude the stop. A missing start means 1, a missing stop means unbounded, and [] is equivalent to [:]. Whitespace is ignored. Use auto[] for an explicit automatic fallback. simd requires AVX512-IFMA. scalar uses GMP and supports widths above 10240 bits only for Edwards. gwnum requires Montgomery PARAM0 and a local gwnum.dll beside the Windows x64 worker. GPU workers can inherit this TOML policy and leave it inactive.

<a id="cpu_mont_chain"></a>

### cpu_montgomery_chain

```toml
cpu_montgomery_chain = "ladder"
```

Type `string` · Range `[ladder|prac]` · Worker override `[worker.N] cpu_montgomery_chain` · Level `advanced`.

Montgomery CPU point multiplication. prac requires target=cpu, curve=mont, impl=simd and B1 in 2..4294967295. It shares the scalar convention and final save format with ladder, with its own in-flight checkpoint state.

### cpu_prac_search

```toml
cpu_prac_search = 7
```

Type `int` · Range `[1..50]` · Worker override `[worker.N] cpu_prac_search` · Level `advanced`.

Number of candidate seeds per prime for CPU IFMA PRAC. Keep the same width when resuming a PRAC checkpoint.

<a id="reduction"></a>

### cpu_simd_reduction

```toml
cpu_simd_reduction = "auto"
```

Type `string` · Range `[auto|montgomery|mersenne-fold]` · Worker override `[worker.N] cpu_simd_reduction` · Level `advanced`.

CPU SIMD modular cpu_simd_reduction. auto selects for the input; mersenne-fold requires N=2^p-1; montgomery uses general cpu_simd_reduction. An explicit cpu_simd_reduction requires impl=simd, so it cannot be silently ignored when auto chooses scalar.

### cuda_low_cpu

```toml
cuda_low_cpu = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] cuda_low_cpu` · Level `core`.

Set to 1 or true to put the CUDA CGBN Stage1 host thread to sleep while waiting for GPU completions, regardless of slice depth. This mode is intended to reduce host CPU use and may increase wake-up latency.

### cuda_montgomery_chain

```toml
cuda_montgomery_chain = "ladder"
```

Type `string` · Range `[ladder|prac]` · Worker override `[worker.N] cuda_montgomery_chain` · Level `advanced`.

Point multiplication chain for CUDA CGBN Montgomery PARAM0. Other curve families and backends ignore this setting. If omitted, the legacy ECM_GPU_STAGE1_ALGO environment override is accepted.

### cuda_prac_search

```toml
cuda_prac_search = 7
```

Type `int` · Range `[1..50]` · Worker override `[worker.N] cuda_prac_search` · Level `advanced`.

Number of candidate seeds per prime for CUDA PRAC. Keep the same width when resuming a PRAC checkpoint.

### cuda_slice_depth

```toml
cuda_slice_depth = 3
```

Type `int` · Range `[2|3|8]` · Worker override `[worker.N] cuda_slice_depth` · Level `core`.

CUDA CGBN Stage1 slice submission. Values 2 and 3 keep a rolling queue of up to that many slices. Value 8 waits at the end of each group of up to eight.

### curve

```toml
curve = ""
```

Type `string` · Range `[mont|ed|empty]` · Worker override `[worker.N] curve` · Level `advanced`.

Optional curve override when param is auto. Use mont for Montgomery or ed for Edwards. An omitted or empty key uses the mode default. Desktop OpenCL supports Edwards through 1024 bits and Montgomery PARAM3.

### edwards_doubling

```toml
edwards_doubling = 1
```

Type `int` · Range `[0|1]` · Worker override `[worker.N] edwards_doubling` · Level `advanced`.

Doubling formula for new CUDA CGBN and OpenCL Edwards NAF batches. 0 selects HWCD, 1 selects 2XY (default). Resume uses the saved formula. CUDA CGBN also accepts ECM_EDWARDS_DBL_ALGO when this key is omitted.

<a id="naf_w"></a>

### edwards_naf_window_bits

```toml
edwards_naf_window_bits = 0
```

Type `int` · Range `[0|2..12]; CPU: [0|3..12]` · Worker override `[worker.N] edwards_naf_window_bits` · Level `advanced`.

Requested Edwards NAF window. 0 starts at 12 for CPU and 6 for CUDA CGBN or OpenCL. With CUDA NTT, 0 selects the fastest compatible measured NAF window (x-only fallback). Values 2–12 fix the 2XY NAF window. CPU, CGBN and OpenCL can lower the window for a fresh task to fit the memory budget.

<a id="exp_cache"></a>

### exponent_cache_dir

```toml
exponent_cache_dir = ""
```

Type `string` · Range `[off|none|0|-|<dir>]` · Worker override `[worker.N] exponent_cache_dir` · Level `advanced`.

Directory for Stage1 scalar and PRAC caches. Empty uses cache/ at the checkout root, or cache/ next to the executable for standalone installs. off, none, 0, or - disables caching. Explicit relative paths use the current working directory. Small B1 values may queue_skip writing the scalar cache.

### gpu_algorithm

```toml
gpu_algorithm = "ntt[16384:], cgbn[]"
```

Type `int|string` · Range `[1|2|3|cgbn|ntt|thread|slice list]` · Worker override `[worker.N] gpu_algorithm` · Level `advanced`.

Stage1 GPU algorithm, case-insensitive. Single values 1/cgbn, 2/ntt and 3/thread explicitly select an algorithm. A slice list tries compatible algorithms from left to right at the actual modulus width, with inclusive starts and exclusive ends. Empty brackets match any width. CUDA NTT requires a full Mersenne modulus and Edwards or Montgomery PARAM0. CUDA thread requires Edwards up to 1024 bits. The cgbn rule uses the linked OpenCL algorithm on an OpenCL worker.

<a id="stage1_gpu_init_threads"></a>

### gpu_init_threads

```toml
gpu_init_threads = 0
```

Type `u32` · Range `<x:Z,0..2^32-1>` · Worker override `[worker.N] gpu_init_threads` · Level `advanced`.

Host curve-construction threads for Edwards CUDA CGBN, CUDA NTT and CUDA/OpenCL one-thread-per-curve Stage1. 0 uses the backend's default. Work is bounded by the available CPUs, curves and memory where applicable. Override with --gpu-init-threads.

### handoff_mode

```toml
handoff_mode = "ecm-elf"
```

Type `string` · Range `[off|ecm-elf|prime95]` · Worker override `[worker.N] handoff_mode` · Level `core`.

Send successful Stage1 tasks with usable points and no factor to ECM-ELF Stage2 by default. Append the new task at the end of the source worker's queue unless a receiver is configured. Use off to keep the save or prime95 to select Prime95.

### handoff_unsupported

```toml
handoff_unsupported = "ecm-elf"
```

Type `string` · Range `[ecm-elf|keep-save]` · Worker override `[worker.N] handoff_unsupported` · Level `advanced`.

Fallback for saves unsupported by the Prime95 protocol.

<a id="auto_stage2_workers"></a>

### handoff_workers

```toml
handoff_workers = ""
```

Type `string` · Range `[<n>[slice]|<a>-<b>[slice]|comma list|auto]` · Worker override `[worker.N] handoff_workers` · Level `advanced`.

Stage2 receiving workers: IDs and inclusive ID ranges within 1..1000000. Each comma-separated entry can have an optional bit slice, e.g. 2[:4096], 3[4096:]. The actual cofactor width is matched against inclusive starts and exclusive stops. Omitted brackets, [] and [:] match any width. Select the shortest eligible queue, breaking ties by lowest ID. If no slice matches, retain the source task. Auto uses worker_count or existing sections for the same queue and considers workers that read it and allow Stage2. For external queues, auto prefers existing sections. Empty keeps the current worker. Unwritable destinations remain pending for retry.

<a id="auto_stage2_worktodo"></a>

### handoff_worktodo_file

```toml
handoff_worktodo_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] handoff_worktodo_file` · Level `advanced`.

Destination queue when auto_stage2=1. Empty inherits this worker's effective shared worktodo_file. Accepts an absolute file path or a filename/relative path resolved beside ecm.toml. Save filenames in delivered tasks are absolute. Stage2 finished receipts use the receiving worker's stage2_finished setting from this TOML. Worker sections can override this setting.

<a id="lcm"></a>

### montgomery_lcm_multiplier

```toml
montgomery_lcm_multiplier = "1"
```

Type `string` · Range `[1|12]` · Worker override `[worker.N] montgomery_lcm_multiplier` · Level `advanced`.

Montgomery Stage1 scalar is montgomery_lcm_multiplier(1..B1) multiplied by 1 or 12. Edwards uses its fixed 48*montgomery_lcm_multiplier scalar and rejects 12. This choice changes the output point and must match the Stage2 save convention.

### naf_memory_mb

```toml
naf_memory_mb = 0
```

Type `int` · Range `[0|1..1048576]` · Worker override `[worker.N] naf_memory_mb` · Level `advanced`.

Edwards Stage1 memory cap in MB (1 MB = 1,000,000 bytes). Limits the CPU NAF working set, CUDA CGBN or OpenCL device buffers, or CUDA NTT dictionaries and their device product tree. 0 uses the backend budget. OpenCL uses half the reported global memory and also checks the maximum allocation size.

### ntt_allow_known_factors

```toml
ntt_allow_known_factors = 1
```

Type `int` · Range `[0|1]` · Worker override `[worker.N] ntt_allow_known_factors` · Level `advanced`.

1 allows Mersenne assignments with known factors to use NTT arithmetic modulo the full 2^p-1. Construction, factor checks and published saves use the remaining cofactor. 0 excludes this carrier path. Algorithm slice rules still match the cofactor bit width.

### ntt_batch_target_percent

```toml
ntt_batch_target_percent = 95
```

Type `int` · Range `[1,100]` · Worker override `[worker.N] ntt_batch_target_percent` · Level `advanced`.

Select the smallest eligible CUDA Stage1 NTT batch reaching this percentage of the fastest saved throughput.

<a id="ntt_bpw"></a>

### ntt_bits_per_word

```toml
ntt_bits_per_word = 0
```

Type `int` · Range `0 or [16,31]` · Worker override `[worker.N] ntt_bits_per_word` · Level `advanced`.

Bits per digit. Zero selects the widest exact linear packing or the shortest safe weighted transform. In weighted mode, an explicit value is an upper bound on the largest digit.

### ntt_convolution

```toml
ntt_convolution = "auto"
```

Type `string` · Range `[auto|linear|weighted]` · Worker override `[worker.N] ntt_convolution` · Level `advanced`.

Auto selects the measured backend or calibrates on a cache miss. Explicit modes select exact linear convolution or weighted cyclic convolution (IBDWT). Both use Goldilocks and the same Stage1 task and checkpoint format.

### ntt_length

```toml
ntt_length = 0
```

Type `int` · Range `0, 2^k in [2,524288], or 5*2^k in [2560,163840] (weighted)` · Worker override `[worker.N] ntt_length` · Level `advanced`.

Transform length. Zero derives safe lengths, including shorter supported radix-5 shapes for weighted convolution.

### ntt_max_batch

```toml
ntt_max_batch = 512
```

Type `int` · Range `[1,512]` · Worker override `[worker.N] ntt_max_batch` · Level `advanced`.

Maximum curves per new CUDA Stage1 NTT batch in tuning and production. Available memory can lower the limit for each transform and NAF window.

### ntt_memory_mib

```toml
ntt_memory_mib = 0
```

Type `int` · Range `[0,1048576]` · Worker override `[worker.N] ntt_memory_mib` · Level `advanced`.

Total Stage1 NTT VRAM budget in MiB. Zero uses 80% of free VRAM after CUDA initialization. An explicit cap is also bounded by available VRAM.

### ntt_naf_gpu_min_bits

```toml
ntt_naf_gpu_min_bits = "auto"
```

Type `int|string` · Range `[auto|0..5242881]` · Worker override `[worker.N] ntt_naf_gpu_min_bits` · Level `advanced`.

Minimum Mersenne exponent for GPU Edwards NAF dictionary construction in CUDA NTT production and tuning. Auto (or 0) uses built-in thresholds. A positive value selects the GPU at and above that exponent, subject to memory limits. 5242881 selects GMP for all supported exponents. Tune prepares one curve.

<a id="ntt_outer_max"></a>

### ntt_outer_radix_limit

```toml
ntt_outer_radix_limit = 0
```

Type `int` · Range `[0,4]` · Worker override `[worker.N] ntt_outer_radix_limit` · Level `advanced`.

Maximum fused radix-2 layers per outer pass. Zero uses the profile.

### ntt_pad_batch

```toml
ntt_pad_batch = 0
```

Type `int` · Range `[0|1]` · Worker override `[worker.N] ntt_pad_batch` · Level `advanced`.

0 runs exactly the requested number of Stage1 NTT curves. 1 rounds the final batch up to the selected size when resources permit. Each extra curve gets a new sigma and is included in the results and any eligible Stage2 handoff.

### ntt_profile

```toml
ntt_profile = 0
```

Type `int` · Range `[0|1|2]` · Worker override `[worker.N] ntt_profile` · Level `advanced`.

Per-length transform profile: 0 automatic, 1 full tile (up to 12 bits), 2 split tile (up to 10 bits). Explicit tile and outer overrides take precedence.

### ntt_tight_weighted_bound

```toml
ntt_tight_weighted_bound = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] ntt_tight_weighted_bound` · Level `advanced`.

Use the tighter unsigned coefficient bound for native Edwards NAF in CUDA Stage1. Automatic selection benchmarks eligible shorter weighted plans alongside the other safe shapes. False uses the conservative bound.

### ntt_tile

```toml
ntt_tile = 0
```

Type `int` · Range `[0,12]` · Worker override `[worker.N] ntt_tile` · Level `advanced`.

Fused tile width in radix-2 layers. Zero uses the profile.

### ntt_warp_tail

```toml
ntt_warp_tail = "auto"
```

Type `bool|string` · Range `[auto|false|true|0|1]` · Worker override `[worker.N] ntt_warp_tail` · Level `advanced`.

Enable warp shuffles for the local tail of the tile transform. Auto uses the selected profile.

<a id="ntt_window"></a>

### ntt_window_bits

```toml
ntt_window_bits = -1
```

Type `int` · Range `[-1,64]` · Worker override `[worker.N] ntt_window_bits` · Level `advanced`.

Ladder bits per CUDA Graph replay. -1 uses the profile; 0 uses direct launches.

<a id="kernel_add"></a>

### opencl_kernel_add

```toml
opencl_kernel_add = ""
```

Type `string` · Range `[auto|<id>|<alias>]` · Worker override `[worker.N] opencl_kernel_add` · Level `advanced`.

OpenCL modular-addition kernel ID or alias. Empty or auto lets the backend choose; automatic selection is normally suitable. CUDA ignores this key.

<a id="kernel_mul"></a>

### opencl_kernel_mul

```toml
opencl_kernel_mul = ""
```

Type `string` · Range `[auto|<id>|<alias>]` · Worker override `[worker.N] opencl_kernel_mul` · Level `advanced`.

OpenCL modular multiplication. PARAM3 accepts registered IDs and aliases. Edwards defaults to cios and accepts fips or NVIDIA-only fips_ptx through 512-bit moduli. CUDA ignores this key.

<a id="kernel_special_mult"></a>

### opencl_kernel_special_mult

```toml
opencl_kernel_special_mult = ""
```

Type `string` · Range `[auto|<id>|<alias>]` · Worker override `[worker.N] opencl_kernel_special_mult` · Level `advanced`.

OpenCL special-multiplication kernel ID or alias. Empty or auto lets the backend choose; automatic selection is normally suitable. CUDA ignores this key.

<a id="kernel_sqr"></a>

### opencl_kernel_sqr

```toml
opencl_kernel_sqr = ""
```

Type `string` · Range `[auto|<id>|<alias>]` · Worker override `[worker.N] opencl_kernel_sqr` · Level `advanced`.

OpenCL modular squaring. PARAM3 accepts registered IDs and aliases. Edwards auto reuses the selected multiplier. Explicit cios selects CIOS, while fips and NVIDIA-only fips_ptx select symmetric FIPS through 512 bits. On one Edwards task, select the same FIPS implementation for both operators that use FIPS. CUDA ignores this key.

<a id="kernel_sub"></a>

### opencl_kernel_sub

```toml
opencl_kernel_sub = ""
```

Type `string` · Range `[auto|<id>|<alias>]` · Worker override `[worker.N] opencl_kernel_sub` · Level `advanced`.

OpenCL modular-subtraction kernel ID or alias. Empty or auto lets the backend choose; automatic selection is normally suitable. CUDA ignores this key.

<a id="tpi"></a>

### opencl_threads_per_integer

```toml
opencl_threads_per_integer = 8
```

Type `u32` · Range `[1|2|4|8|16|32]` · Worker override `[worker.N] opencl_threads_per_integer` · Level `advanced`.

OpenCL threads per integer, subject to limb-layout constraints. CUDA selects TPI by operand width and ignores this key. This is not the CUDA block size.

<a id="wg_size"></a>

### opencl_workgroup_size

```toml
opencl_workgroup_size = 0
```

Type `int` · Range `<x:Z,x>=0>` · Worker override `[worker.N] opencl_workgroup_size` · Level `advanced`.

OpenCL work-group size. 0 lets the backend select automatically; explicit values must satisfy device and kernel limits. CUDA ignores this setting.

### param

```toml
param = "auto"
```

Type `string` · Range `[auto|e|m0|m2|m3]` · Worker override `[worker.N] param` · Level `core`.

Stage1 curve construction. With auto, queue tasks default to Edwards and direct input defaults to m0 on CPU or m3 on GPU. The optional curve key applies with auto. A typed ECMSTAGE1 save name takes precedence.

<a id="p95_keep_aid"></a>

### prime95_keep_aid

```toml
prime95_keep_aid = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] prime95_keep_aid` · Level `advanced`.

Keep PrimeNet assignment IDs in handoffs: 1 retains IDs not known to be invalid; 0 strips all IDs. IDs confirmed invalid by Prime95 logs are always removed to prevent repeated rejection.

<a id="p95_recover_lost"></a>

### prime95_recover_lost

```toml
prime95_recover_lost = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] prime95_recover_lost` · Level `advanced`.

Recover handoffs deleted by Prime95 because of invalid assignment IDs. 1 checks handoff records and Prime95 logs, then reposts confirmed lost assignments without an ID; 0 disables recovery. Only confirmed prior handoffs qualify.

<a id="p95_sync_dir"></a>

### prime95_save_dir

```toml
prime95_save_dir = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] prime95_save_dir` · Level `advanced`.

Dedicated save destination for Prime95 handoff; required with prime95_worktodo_file. Backup destinations are independent. Existing published save prefixes must remain unchanged; only complete appended records may extend them.

<a id="p95_add_workers"></a>

### prime95_workers

```toml
prime95_workers = ""
```

Type `string` · Range `[<n>|<n>,...|<a>-<b>|auto]` · Worker override `[worker.N] prime95_workers` · Level `advanced`.

Prime95 workers eligible to receive handoffs: one ID, a comma list, or ranges. The shortest eligible queue is selected. auto reads NumWorkers from prime.txt in the same directory. Empty omits the worker header and uses Prime95's default worker.

<a id="p95_worktodo_path"></a>

### prime95_worktodo_file

```toml
prime95_worktodo_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] prime95_worktodo_file` · Level `advanced`.

Prime95 worktodo_file path for supported Montgomery Stage1 handoff. Edwards and GPU PARAM2 ignore this path and keep saves local; they do not access Prime95 transfer files. Requires prime95_save_dir. The save is validated and published before worktodo_file.add.

### progress_average_seconds

```toml
progress_average_seconds = 0
```

Type `double` · Range `[0..60]` · Worker override `[worker.N] progress_average_seconds` · Level `advanced`.

Time window for live Stage1 CPU/CUDA s/curve and remaining-time estimates. Zero uses the latest valid sample without averaging; positive values average completed work over measured sample time, up to 60 seconds. CUDA samples exclude preparation and checkpoint I/O. This does not change computation, slicing, final timing or log refresh frequency.

### progress_color

```toml
progress_color = "cyan"
```

Type `string` · Range `[none|red|green|yellow|blue|magenta|cyan|white|grey]` · Worker override `[worker.N] progress_color` · Level `advanced`.

Color of the Stage1 console progress bar. none disables color. This does not change ordinary file logs or arithmetic.

### progress_log_seconds

```toml
progress_log_seconds = 60
```

Type `double` · Range `<x:R,finite>` · Worker override `[worker.N] progress_log_seconds` · Level `advanced`.

Interval in seconds between Stage1 progress entries in file logs. Positive values limit frequency; 0 suppresses intermediate progress; negative values log every refresh. Completion at 100% is still logged. Console refresh frequency is unchanged.

<a id="stage1_random_sigma"></a>

### random_sigma

```toml
random_sigma = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] random_sigma` · Level `core`.

Choose per-curve random sigmas where supported when no seed is supplied. False draws one random starting sigma and increments it by one per curve. Montgomery GPU batches already use consecutive sigmas. An explicit sigma takes precedence.

### save_name_pattern

```toml
save_name_pattern = "{curve}_{target}_{b1}.save"
```

Type `string` · Range `<pattern>` · Worker override `[worker.N] save_name_pattern` · Level `advanced`.

Automatic Stage1 save filename pattern for CPU and GPU. {curve}=e|m0|m2|m3; {target}=m<exponent> for Mersenne assignments, otherwise bit<actual_bits>; {n}=original assignment exponent when available, otherwise actual bits; {b1}=compact B1. Missing curve prefixes are added. In ECMSTAGE1 queues, typed save prefixes override TOML curve/param and the final filename field supplies B1; untyped names use the TOML construction. Finished/handoff lines name the actual prefixed save. Explicit CLI save paths are preserved. Empty restores the default.

### sigma

```toml
sigma = 0
```

Type `u64` · Range `<x:Z,0..2^64-1>` · Worker override `[worker.N] sigma` · Level `advanced`.

Initial curve sigma. 0 chooses a random value; consecutive curves use sigma+i without 64-bit wraparound. CUDA PARAM0 accepts the full 64-bit range, while PARAM2/3 batches must stay below 2^32. CPU runs warn if a sigma exceeds Prime95's signed reader range; this is not a Stage1 limit. A nonzero ECM/ECM2 assignment sigma overrides this setting.

<a id="stage1_cpu_compute_threads"></a>

### stage1_cpu_threads

```toml
stage1_cpu_threads = 0
```

Type `u32` · Range `<x:Z,0..2^32-1>` · Worker override `[worker.N] stage1_cpu_threads` · Level `advanced`.

CPU Stage1 arithmetic and preparation threads. 0 uses the backend's automatic selection. Override with --cpu-threads.

<a id="stage1_backend"></a>

### stage1_gpu_backend

```toml
stage1_gpu_backend = "auto"
```

Type `string` · Range `[auto|cuda|opencl]` · Worker override `[worker.N] stage1_gpu_backend` · Level `core`.

Windows GPU worker. Auto prefers an installed CUDA worker with a visible CUDA device at the selected index, then OpenCL. Explicit selection requires that worker beside the executable. Device indices belong to the selected backend. Override with --gpu-backend.

### stage1_gpu_device

```toml
stage1_gpu_device = 0
```

Type `int` · Range `<x:Z,0..device_count-1>` · Worker override `[worker.N] stage1_gpu_device` · Level `core`.

Zero-based GPU device ID for Stage1. Stage2 inherits it unless stage2_device is explicitly set. IDs refer to the device list exposed by the executable's actual GPU backend.

### stage1_results_file

```toml
stage1_results_file = "stage1_results.jsonl"
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage1_results_file` · Level `core`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Canonical internal per-curve Stage1 JSONL with CRC32 and execution metadata, separate from submission output.

<a id="tmp_dir"></a>

### stage1_save_dir

```toml
stage1_save_dir = "."
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage1_save_dir` · Level `core`.

Output directory for Stage1 curve saves. Stage2 also searches here when stage2_save_dir is unset. This is not a Stage2 RAM or VRAM budget.

### stage1_submission_file

```toml
stage1_submission_file = "results.json.txt"
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage1_submission_file` · Level `core`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Stage1 submission JSON: seal NF once the whole task completes (B2=B1); publish verified factors immediately as individual curves. Private ledgers live under the output directory data/.

### target

```toml
target = "gpu"
```

Type `string` · Range `[cpu|gpu]` · Worker override `[worker.N] target` · Level `core`.

Run Stage1 on CPU or GPU. The stage1_gpu_backend setting selects CUDA or desktop OpenCL. Both support Edwards within their own width limits.

<a id="cuda_thread_block_size"></a>

### thread_block_size

```toml
thread_block_size = 128
```

Type `int` · Range `[32|64|128|256]` · Worker override `[worker.N] thread_block_size` · Level `advanced`.

CUDA block size for the Edwards thread algorithm only. Use 32, 64, 128 or 256. CLI --wg overrides it; --wg 0 selects 128.

### verbose

```toml
verbose = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] verbose` · Level `core`.

Normal console detail. In Stage2, true shows phase transitions and false keeps curve summaries; debug_log independently controls debug logging. Stage1 uses the selected backend's normal verbosity behavior.

## [stage2]

<a id="stage2_arena_mb"></a>

### arena_mib

```toml
arena_mib = 0
```

Type `u64` · Range `<x:Z,0..2^20>` · Worker override `[worker.N] arena_mib` · Level `advanced`.

Stage2 large-workspace budget in MiB, used to plan available NTT shapes. 0 uses environment settings or automatic planning, not zero VRAM. Too little may force smaller shapes or prevent execution. Tables, coordinates, and other modules may allocate additional VRAM.

<a id="stage2_auto_b2"></a>

### auto_b2

```toml
auto_b2 = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] auto_b2` · Level `core`.

Choose B2 when every source supplies zero. 1 optimizes workflow benefit per time including Stage1 cost; 0 disables. Requires usable Stage2 costs and positive Stage1 s/curve. B1 never gates tune reuse; tune_ignore controls other conditions. Memory and carrier divisibility are checked on the current input. Estimates retain uncertainty labels; the legacy .cprof interface keeps its original qualification checks.

<a id="stage2_auto_carrier"></a>

### auto_carrier

```toml
auto_carrier = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] auto_carrier` · Level `advanced`.

Add M=2^p-1 for validated ECMSTAGE2=1,2,p,-1 saves when N divides M and 2<=p<=16384. Compare ordinary cost at target_bits with Mersenne cost at arithmetic_bits=p, using prediction_policy. Full M_p already has native Mersenne arithmetic. Explicit carrier options take process_priority; a cofactor save needs a matching task to infer p.

<a id="stage2_auto_max_b2"></a>

### auto_max_b2

```toml
auto_max_b2 = 0
```

Type `u64` · Range `<x:Z,0..2^63-8193>` · Worker override `[worker.N] auto_max_b2` · Level `advanced`.

Auto B2 upper search bound; 0 means 2.6e12. Unmeasured values may be estimated; explicit task B2 takes precedence.

<a id="stage2_auto_min_b2"></a>

### auto_min_b2

```toml
auto_min_b2 = 0
```

Type `u64` · Range `<x:Z,0..2^63-8193>` · Worker override `[worker.N] auto_min_b2` · Level `advanced`.

Auto B2 lower search bound; 0 means B1+1. Unmeasured values may be estimated; explicit task B2 takes precedence.

<a id="stage2_b2"></a>

### b2

```toml
b2 = 0
```

Type `u64` · Range `<x:Z,x=0|B1<x<=2^63-1-8192>` · Worker override `[worker.N] b2` · Level `core`.

Stage2 upper bound when neither CLI nor task supplies a nonzero B2. 0 defers to other sources. Explicit B2 must exceed the save's B1. Larger B2 usually increases coverage, time, and VRAM demand. Precedence: nonzero CLI B2, then task B2, then this key.

### b2_ratio

```toml
b2_ratio = 0
```

Type `u64` · Range `[0|1..1000000]` · Worker override `[worker.N] b2_ratio` · Level `core`.

Stage2 fallback when the CLI, task line and b2 setting supply no nonzero B2. A nonzero ratio uses the saved B1 times this value. Applies to queues and GPU direct save runs. Overflow is rejected.

<a id="stage2_b2_round_chain"></a>

### b2_round_chain

```toml
b2_round_chain = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] b2_round_chain` · Level `advanced`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Round B2 per candidate D. CPU gwpoly fills the existing last P-point G block after relocation. CUDA also requires a chain-capable final point chunk. Tuning measures each candidate at its actual B2. Logs and internal results retain both bounds. Enabled by default for CPU gwpoly ABI 4 and CUDA full-interval runs. Set false to keep the requested bound. CPU libecm, older DLLs, CUDA relocation and calibration use fixed bounds by default. Integer, memory and enabled AutoB2 limits apply. CUDA startup cost probes keep fixed bounds and explicit ECM calibration requires rounding off. CLI --b2-round-chain overrides this worker setting.

<a id="stage2_baby_backend"></a>

### baby_backend

```toml
baby_backend = "coop32"
```

Type `string` · Range `[auto|scalar|coop32]` · Worker override `[worker.N] baby_backend` · Level `advanced`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

CUDA defaults to coop32 for baby ladders at arithmetic widths 8193..16384 bits when device baby generation is available. Other shapes use scalar. Explicit auto uses checked paired tune results, starting with scalar. CLI --baby-backend overrides TOML. If omitted, an explicit NTT_BABY_COOP takes precedence. Legacy cost profiles default to scalar. OpenCL uses scalar.

<a id="stage2_progress_dir"></a>

### checkpoint_dir

```toml
checkpoint_dir = ""
```

Type `string` · Range `<dir>` · Worker override `[worker.N] checkpoint_dir` · Level `advanced`.

Directory for per-task .ecm_ckpt_stage2_<bits>_<B1>.dat progress files. Empty uses the legacy progress file's directory if explicitly set, otherwise the queue directory. Task IDs bind AID, N, full save contents and queue_skip. On restart the current effective B2/Auto B2 and requested curve count override the checkpoint policy; completed curves retain their actual bounds. A target below the completed count warns and seals all completed work. New checkpoints contain completed curve state and pending submission data. They are removed after publication and queue advancement.

<a id="stage2_progress_file"></a>

### checkpoint_legacy_file

```toml
checkpoint_legacy_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] checkpoint_legacy_file` · Level `legacy`.

Legacy single-file progress import path. Empty checks the old queue-path plus worker-suffix .progress file. A matching task's completed count and pending receipt are imported into its own .ecm_ckpt_stage2_<bits>_<B1>.dat file. An unrelated legacy task is retained and does not stop this task. New progress files use checkpoint_dir; no arithmetic state is checkpointed.

<a id="stage2_cpu_backend"></a>

### cpu_backend

```toml
cpu_backend = "auto"
```

Type `string` · Range `[auto|gwpoly|ntt|plain]` · Worker override `[worker.N] cpu_backend` · Level `advanced`.

CPU arithmetic for Edwards continuation. Auto uses GWNUM/polymult when available, then libecm. Override with --backend.

<a id="stage2_cpu_memory_mib"></a>

### cpu_memory_mib

```toml
cpu_memory_mib = 512
```

Type `u64` · Range `[1,1048576]` · Worker override `[worker.N] cpu_memory_mib` · Level `advanced`.

CPU Stage2 arithmetic memory budget in MiB. Override with --memory-mib.

<a id="stage2_cpu_tune_candidates"></a>

### cpu_tune_candidates

```toml
cpu_tune_candidates = 5
```

Type `u64` · Range `[1,16]` · Worker override `[worker.N] cpu_tune_candidates` · Level `advanced`.

Maximum GWNUM Stage2 plans considered for one workload, including the baseline and alternative D and F-tree layouts. 1 keeps only the baseline. ABI 4 can return up to three D choices. Override with --cpu-tune-candidates.

<a id="stage2_cpu_tune_dense_d"></a>

### cpu_tune_dense_d

```toml
cpu_tune_dense_d = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] cpu_tune_dense_d` · Level `advanced`.

Enumerate intermediate D values for GWNUM Stage2 tuning on ABI 5. Disable to search the original wheels. Override with --cpu-tune-dense-d 0|1.

<a id="stage2_parallel"></a>

### curve_concurrency

```toml
curve_concurrency = "serial"
```

Type `string` · Range `[auto|serial|dual]` · Worker override `[worker.N] curve_concurrency` · Level `advanced`.

CUDA Stage2 curve scheduling. auto measures serial and eligible dual-slot modes. serial is the default. dual requests two slots subject to device admission. CLI: --parallel auto|0|1|2.

<a id="stage2_d"></a>

### d

```toml
d = 0
```

Type `u64` · Range `<x:Z,x=0|x>=6,x%2=0>` · Worker override `[worker.N] d` · Level `advanced`.

Manual Stage2 step D. 0 selects the shape automatically. Explicit values must be even, at least 6, and satisfy the engine's layout constraints. D affects tree size, batch count, and memory demand; automatic selection is normally suitable.

<a id="stage2_debug_log"></a>

### debug_log

```toml
debug_log = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] debug_log` · Level `advanced`.

Write Stage2 debug details to a separate log: true enables and false disables. verbose=true does not enable debug logging; enabling this option does not send all debug details to the console.

<a id="stage2_debug_log_file"></a>

### debug_log_file

```toml
debug_log_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] debug_log_file` · Level `advanced`.

Separate Stage2 debug log, written only when debug_log is enabled. Empty restores the default filename. Default names gain a worker suffix; explicit paths are used unchanged.

<a id="stage2_factorize_hits"></a>

### decompose_factors

```toml
decompose_factors = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] decompose_factors` · Level `advanced`.

Further decompose found factors: 1 uses the shared diagnostic service (local PARI DLL first, external GP second); GMP is never selected automatically. 0 keeps only factors extracted and verified by Stage2. Only complete, proven decomposition sets factorization_complete. Unfinished diagnostics preserve the original factor and successful curve status.

<a id="stage2_execution_device"></a>

### device

```toml
device = "auto"
```

Type `string` · Range `[auto|cpu|gpu]` · Worker override `[worker.N] device` · Level `core`.

Execution device in the combined Stage2 program. Auto selects by stage2_algorithm; cpu and gpu pin the device. CPU supports Edwards saves. Override with --target.

<a id="stage2_factor_timeout"></a>

### factor_timeout_seconds

```toml
factor_timeout_seconds = 30
```

Type `u64` · Range `<x:Z,1..600>` · Worker override `[worker.N] factor_timeout_seconds` · Level `advanced`.

Timeout in seconds for each external GP fallback decomposition call. Local PARI calls have a 5-second process deadline and a 10-second cumulative session budget per curve worker. Explicit GMP analysis remains bounded to 2 seconds per call. These diagnostics never change the original factor or successful Stage2 computation status.

<a id="stage2_fold_mb"></a>

### fold_resident_mib

```toml
fold_resident_mib = 640
```

Type `u64` · Range `<x:Z,0..2^20>` · Worker override `[worker.N] fold_resident_mib` · Level `advanced`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Budget in MiB for keeping fold accumulators on the GPU. 0 disables this residency. Exceeding the budget may fall back to batching, substantially increasing processing and transfer costs. Independent of arena and batch budgets; not a total VRAM limit.

<a id="stage2_giant_backend"></a>

### giant_backend

```toml
giant_backend = "coop32"
```

Type `string` · Range `[auto|scalar|coop16|coop32]` · Worker override `[worker.N] giant_backend` · Level `advanced`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

CUDA defaults to coop32 for eligible giant chains at arithmetic widths 8193..16384 bits. coop16 uses a fixed 16384-bit container, while coop32 uses width-matched containers. Explicit auto uses checked paired tune results, starting with scalar. CLI --giant-backend overrides TOML. If omitted, an explicit NTT_GIANT_COOP takes precedence. Legacy cost profiles default to scalar. OpenCL uses scalar.

<a id="stage2_giant_chain"></a>

### giant_chain

```toml
giant_chain = "64"
```

Type `string` · Range `[64|32|auto]` · Worker override `[worker.N] giant_chain` · Level `advanced`.

Points per sequential CUDA thread chain, independent of CUDA block size. 64 preserves the baseline; 32 explicitly selects shorter chains. auto retains 64 until tune_min_repeats admitted runtime observations support 32, with bounded useful-curve probes. Policies have separate tune identities. Hit/nonunit replays the legacy 64-point full interval. CLI --giant-chain overrides this worker TOML.

<a id="stage2_giant_tail_balance"></a>

### giant_tail_balance

```toml
giant_tail_balance = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] giant_tail_balance` · Level `advanced`.

Allow the last two giant point chunks to borrow whole P batches so both can use chains. B2, capacity, coverage and chunk count are unchanged. Default false. No actual borrow keeps the original cost/cache identity and avoids extra replay or synchronization. Actual borrowing keeps an independent tune identity and replays the original interval on a hit/nonunit; that replay may need more workspace. CLI --giant-tail-balance overrides this worker TOML.

<a id="stage2_gp"></a>

### gp

```toml
gp = ""
```

Type `string` · Range `<exe>` · Worker override `[worker.N] gp` · Level `advanced`.

External PARI/GP fallback executable for optional factor decomposition. The local PARI DLL is tried first. gp.stage1_exe follows Windows executable search rules; a full path is accepted. GP failure preserves partial PARI results and never enables GMP automatically. Dataset scripts retain their separate --gp resolution rules.

<a id="stage2_groot_balanced"></a>

### groot_balanced

```toml
groot_balanced = "auto"
```

Type `bool|string` · Range `[auto|false|true|0|1]` · Worker override `[worker.N] groot_balanced` · Level `advanced`.

G-root polynomial layout. "auto" uses the qualified NTT-cliff model and measured tuning; true forces the balanced tile layout, false forces the contiguous layout. Default "auto". This does not rebalance giant point chunks or change B2/D/P. CUDA only; OpenCL accepts "auto". CLI --groot-layout auto|true|false overrides this worker TOML; numeric CLI aliases 0/1 remain supported. Tune rows store the resolved layout as 0/1.

<a id="stage2_groot_subtree"></a>

### groot_subtree

```toml
groot_subtree = "1[:3072], 0[8192:], auto[]"
```

Type `int|string` · Range `[auto|-1|0|1|slice list]` · Worker override `[worker.N] groot_subtree` · Level `advanced`.

CUDA Stage2 G-tree subtrees. 0 disables, 1 enables four-root subtrees for supported native or carrier Mersenne arithmetic, and auto or -1 uses matching ECM tune results. Auto uses the original path until both modes have qualified results. Ordered slices match the actual cofactor width, including the start and excluding the end. CLI --groot-subtree overrides this policy. Other arithmetic paths use the original tree.

<a id="stage2_log_level"></a>

### log_level

```toml
log_level = -1
```

Type `int|string` · Range `[quiet|curve|phases|batches|debug|0..4]` · Worker override `[worker.N] log_level` · Level `advanced`.

Legacy console level: quiet, curve, phases, batches, and debug map to 0..4. Unset follows verbose. In production, debug enables the separate debug log while normal console detail remains capped at batches. Usually use verbose and debug_log.

<a id="stage2_low_cpu_wait"></a>

### low_cpu_wait

```toml
low_cpu_wait = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] low_cpu_wait` · Level `advanced`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Opt-in low CPU policy: CUDA blocking device scheduling (4) and bounded event polling. Blocking reads keep their native path; elapsed time may increase.
Default false selects auto scheduling (0) and native event waits. CLI --low-cpu-wait 0|1 overrides this key.
An explicit key sets NTT_CUDA_WAIT_MODE and NTT_HOST_EVENT_WAIT, and clears NTT_HOST_READBACK_WAIT. Omission preserves expert environment overrides. FIFO recycling is independent.

<a id="stage2_ntt_counts"></a>

### ntt_length_counts

```toml
ntt_length_counts = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] ntt_length_counts` · Level `advanced`.

Print a per-curve NTT length count table at the end of production Stage2: 1 enables; 0 disables collection and output. Counts mathematical transforms (two forwards and one inverse per convolution), including polynomial initialization and main work, excluding Stage1 and independent arithmetic selftests. Shared by Edwards and Montgomery; no GPU counters or extra synchronization. CLI --ntt-counts overrides the selected worker TOML.

<a id="stage2_ntt_two_axis"></a>

### ntt_two_axis

```toml
ntt_two_axis = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] ntt_two_axis` · Level `advanced`.

Enable two-axis NTT for eligible CUDA Stage2 products in the F and G trees, Newton inverse, H fold, and descent. Eligibility requires a Mersenne carrier, a supported shape, and available device memory. Single-axis NTT handles other products and legacy cost profiles. CLI --two-axis-ntt 0|1 overrides this key. The setting is inactive on OpenCL.

<a id="stage2_opencl_point_chunk"></a>

### opencl_point_chunk

```toml
opencl_point_chunk = 0
```

Type `u64` · Range `[0..4096]` · Worker override `[worker.N] opencl_point_chunk` · Level `advanced`.

Maximum points per OpenCL command. Zero uses automatic sizing. A positive value locks the chunk and disables runtime plan trials.

<a id="stage2_opencl_point_group"></a>

### opencl_point_group

```toml
opencl_point_group = 0
```

Type `u64` · Range `[0..4096]` · Worker override `[worker.N] opencl_point_group` · Level `advanced`.

Local OpenCL point work-group size. Zero uses the backend default. Positive sizes must fit both device and kernel limits.

<a id="stage2_opencl_point_plan"></a>

### opencl_point_plan

```toml
opencl_point_plan = "auto"
```

Type `string` · Range `[auto|device|chain|cpu]` · Worker override `[worker.N] opencl_point_plan` · Level `advanced`.

OpenCL point generation. Auto starts with the qualified plan and can learn from complete curves. Device selects the device ladder plan (including its qualified short batches); chain requests consecutive giant seeds; cpu uses GMP points with OpenCL trees. Manual settings bypass runtime exploration.

<a id="stage2_opencl_point_window"></a>

### opencl_point_window

```toml
opencl_point_window = 0
```

Type `u64` · Range `[0..65536]` · Worker override `[worker.N] opencl_point_window` · Level `advanced`.

Giant coordinate staging window in points. Zero uses the backend plan; positive requests are clipped by the budget and G batch boundaries.

<a id="stage2_opencl_short_point_batch"></a>

### opencl_short_point_batch

```toml
opencl_short_point_batch = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] opencl_short_point_batch` · Level `advanced`.

Allow qualified short device batches. False disables batched baby ladders and paired giants from the short-batch policy.

<a id="stage2_opencl_tune_profile"></a>

### opencl_tune_file

```toml
opencl_tune_file = "stage2_opencl_tune.toml"
```

Type `string` · Range `<path>` · Worker override `[worker.N] opencl_tune_file` · Level `advanced`.

Native OpenCL point timing profile. Relative to ecm.toml, with per-worker overrides. Empty disables profile learning. Uses tune_runtime_enabled, tune_min_repeats and tune_runtime_sample_limit.

<a id="stage2_pack_zero"></a>

### pack_zero

```toml
pack_zero = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] pack_zero` · Level `advanced`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Fuse input digit packing and zero padding in one GPU kernel per operand. Covers contiguous, resident gather and reverse inputs. Other shapes use the original packer. CLI --pack-zero overrides this key. Legacy .cprof cost profiles default to false and require false when set explicitly.

<a id="stage2_parallel_tune_curves"></a>

### parallel_tune_curves

```toml
parallel_tune_curves = 5
```

Type `u64` · Range `[2..16]` · Worker override `[worker.N] parallel_tune_curves` · Level `advanced`.

Number of curves in each CUDA Stage2 queue tuning window. Windows of 2 to 4 curves compare full-window throughput. Longer windows also check steady-state completion throughput.

<a id="stage2_prediction_policy"></a>

### prediction_policy

```toml
prediction_policy = "model"
```

Type `string` · Range `[model|paired]` · Worker override `[worker.N] prediction_policy` · Level `advanced`.

model: prefer exact paired measurements, otherwise compare generic and Mersenne arithmetic-width region estimates with two-sided uncertainty and current memory admission. Missing tune data uses an explicitly estimated 4060-Laptop/1800-MHz basic model. Uncovered NTT/residency regions retain the default. paired: require the original exact-width/B2 measured win. B1 never gates reuse.

<a id="stage2_ratio_adjust"></a>

### ratio_adjust

```toml
ratio_adjust = 1
```

Type `double` · Range `<x:R,finite,x>0>` · Worker override `[worker.N] ratio_adjust` · Level `advanced`.

Multiplier applied to Stage2 time in the Auto B2 model. 1 leaves it unchanged; greater than 1 estimates higher cost, less than 1 lower cost, usually changing selected B2. Adjusts planning cost, not actual GPU speed.

<a id="stage2_reduce_backend"></a>

### reduce_backend

```toml
reduce_backend = "auto"
```

Type `string` · Range `[auto|scalar|coop32]` · Worker override `[worker.N] reduce_backend` · Level `advanced`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Ordinary S4 coefficient cpu_simd_reduction: auto keeps scalar until repeated same-shape comparisons qualify coop32, then uses the fastest checked total including point startup checks. At fixed D/layout/chain/point backends, it adds at most one missing cooperative variant at a time (two-pass or merged), then extends D after checked comparisons. CUDA arithmetic widths 8193..16384 bits can use shared warp remainders; smaller widths and exact Mersenne cpu_simd_reduction keep their kernels. Actual reduce_coop, threads and kernel identity stay separate in ECM timing rows even if environment matching is ignored; standalone NTT tables remain reusable. CLI --reduce-backend overrides this worker key; omission preserves explicit NTT_S4_COOP_REDUCE=0|32. OpenCL uses scalar.

<a id="stage2_relocation"></a>

### relocation

```toml
relocation = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] relocation` · Level `advanced`.

Optional production GPU Stage2 conservative relocation. Keeps B2 and the original giant endpoint, uses a shorter suffix for clean NF, and replays the original interval on any hit or target-N nonunit (including GCD=N). Rebalances short point chunks on whole G boundaries. Default 0 preserves the original algorithm. CLI --relocation overrides this worker TOML. D/AutoB2 and memory admission use original-interval estimates; relocation does not collect ECM tune samples or run startup ECM calibration, and explicit --tune ecm is unavailable in this mode. Pure NTT samples remain reusable.

<a id="stage2_batch_mb"></a>

### s4_batch_mib

```toml
s4_batch_mib = 64
```

Type `u64` · Range `<x:Z,1..2^20>` · Worker override `[worker.N] s4_batch_mib` · Level `advanced`.

Stage2 S4 batch-buffer budget in MiB. Larger budgets may reduce batching overhead but consume more VRAM. This limits the corresponding buffers, not total process VRAM.

<a id="stage2_seed_backend"></a>

### seed_backend

```toml
seed_backend = "coop32"
```

Type `string` · Range `[auto|scalar|coop32]` · Worker override `[worker.N] seed_backend` · Level `advanced`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

CUDA defaults to coop32 for paired seeds and GPU [D]Q at arithmetic widths 8193..16384 bits when the paired-seed path is used. Explicit auto uses checked paired tune results, starting with scalar. CLI --seed-backend overrides TOML. If omitted, an explicit NTT_GIANT_SEED_COOP takes precedence. Legacy cost profiles default to scalar. OpenCL uses scalar.

<a id="stage2_short_calibration"></a>

### short_calibration

```toml
short_calibration = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] short_calibration` · Level `advanced`.

Calibrate once when no suitable ECM tune file exists, required components are missing, or the task has no reusable calibrated width coverage. Partial coverage permits estimates.

<a id="stage2_short_calibration_seconds"></a>

### short_calibration_seconds

```toml
short_calibration_seconds = 10
```

Type `u64` · Range `<x:Z,1..600>` · Worker override `[worker.N] short_calibration_seconds` · Level `advanced`.

Short calibration measurement budget, excluding first CUDA initialization; finish the current measurement before stopping.

<a id="stage2_short_ntt_batch_tune"></a>

### short_ntt_batch_tune

```toml
short_ntt_batch_tune = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] short_ntt_batch_tune` · Level `advanced`.

Experimental short NTT batch selection from stage2_tune_profile, for lengths up to 2^19. Minimizes estimated convolution time including the tail, within the current chunk budget; requires at least two measured batch sizes and a predicted gain above 10%. Does not tune kernel configuration. Pure NTT gains do not guarantee faster full Stage2; default false pending full-engine validation.

<a id="stage2_factor_only"></a>

### skip_prime_attribution

```toml
skip_prime_attribution = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] skip_prime_attribution` · Level `advanced`.

Skip diagnostic prime attribution, which identifies Stage2 primes responsible for a hit. 1 skips attribution while retaining factor extraction and validation; 0 follows the engine's diagnostic policy. This does not disable decomposition or change decompose_factors.

### stage2_after_factored

```toml
stage2_after_factored = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] stage2_after_factored` · Level `core`.

After an Edwards Stage1 hit, generate a cofactor save and a local ECM-ELF Stage2 assignment from reusable endpoints. Read in the producing worker scope. Default false.

### stage2_algorithm

```toml
stage2_algorithm = "gwnum[4096:], gpu[:16385], cpu[]"
```

Type `string` · Range `[gwnum|cpu-ntt|cpu-plain|cpu|gpu|cuda|opencl|slice list]` · Worker override `[worker.N] stage2_algorithm` · Level `core`.

Algorithm for each task when device is auto in the combined Stage2 program. Rules are tried left to right at the saved modulus width. Starts are inclusive, ends exclusive and [] matches any width. CPU rules require Edwards; gwnum uses GWNUM/polymult, cpu-ntt and cpu-plain use libecm, cpu uses cpu_backend. gpu uses stage2_gpu_backend. GPU input is limited to 16384 bits.

### stage2_cpu_threads

```toml
stage2_cpu_threads = 1
```

Type `u64` · Range `[1,1024]` · Worker override `[worker.N] stage2_cpu_threads` · Level `advanced`.

Requested thread count per CPU Stage2 worker, capped by the selected affinity pool. Override with --threads.

<a id="stage2_backend"></a>

### stage2_gpu_backend

```toml
stage2_gpu_backend = "auto"
```

Type `string` · Range `[auto|cuda|opencl]` · Worker override `[worker.N] stage2_gpu_backend` · Level `core`.

Windows GPU worker. Auto prefers an installed CUDA worker with a visible CUDA device at the selected index, then OpenCL. Explicit selection requires that worker beside the executable. Device indices belong to the selected backend. Override with --gpu-backend.

<a id="stage2_device"></a>

### stage2_gpu_device

```toml
stage2_gpu_device = -1
```

Type `int` · Range `[-1|0..2^31-1]` · Worker override `[worker.N] stage2_gpu_device` · Level `core`.

Stage2 CUDA device ID. -1 inherits Stage1's device; 0 or greater selects a CUDA device explicitly.

### stage2_results_file

```toml
stage2_results_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage2_results_file` · Level `core`.

Optional per-curve Stage2 JSONL for CPU and GPU queue tasks. Empty disables it for queue runs. Direct save runs use the default filename. Keep the original JSONL until legacy checkpoint migration completes.

### stage2_save_dir

```toml
stage2_save_dir = ""
```

Type `string` · Range `<dir>` · Worker override `[worker.N] stage2_save_dir` · Level `advanced`.

Directory searched for saves referenced by Stage2 assignments. Empty inherits tmp_dir; absolute save paths in assignments are used directly. Source saves are read and validated without modification.

### stage2_submission_file

```toml
stage2_submission_file = "results.json.txt"
```

Type `string` · Range `<path>` · Worker override `[worker.N] stage2_submission_file` · Level `core`.

Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.

Submission-only ECM JSON lines, separate from the internal results JSONL. Uses a reference FFT size from Prime95 mult.asm. Personal identifiers and checksums belong only in local ignored files.

<a id="stage2_target_factor_bits"></a>

### target_factor_bits

```toml
target_factor_bits = 0
```

Type `u64` · Range `<x:Z,0|2..16384>` · Worker override `[worker.N] target_factor_bits` · Level `advanced`.

Target prime-factor size for Dickman semismooth success probability. 0 derives a recommendation from B1; this is distinct from the full modulus size.

<a id="stage2_tune_arithmetic"></a>

### tune_arithmetic

```toml
tune_arithmetic = "both"
```

Type `string` · Range `[both|generic|mersenne]` · Worker override `[worker.N] tune_arithmetic` · Level `advanced`.

Arithmetic families for explicit width-grid ECM tune: both, generic probable-prime N, or Mersenne 2^p-1 arithmetic after stripping proper-exponent inherited factors and small target divisors. p is the requested grid arithmetic width; actual cofactor width is recorded separately. Real worktodo_file/save pairs continue to follow auto_carrier. Invalid or factor-producing synthetic Mersenne fixtures are skipped, never saved as timing measurements.

<a id="stage2_tune_budget_seconds"></a>

### tune_budget_seconds

```toml
tune_budget_seconds = 1800
```

Type `u64` · Range `<x:Z,1..86400>` · Worker override `[worker.N] tune_budget_seconds` · Level `advanced`.

Tuning budget including input preparation, measurements and optional warmups. Start each width/B2 at the default model's D, compare neighbors, and follow only measured improvements above 2% plus the MAD noise margin. Reuse completed candidates. Finish an in-flight measurement after the deadline and retain valid summaries.

<a id="stage2_tune_condition_tag"></a>

### tune_condition_tag

```toml
tune_condition_tag = ""
```

Type `string` · Range `<text>` · Worker override `[worker.N] tune_condition_tag` · Level `advanced`.

Optional measurement-condition label, e.g. a fixed clock or power setting. Distinct labels preserve separate fastest summaries; ignored environment allows reuse.

<a id="stage2_tune_error_limit"></a>

### tune_error_limit

```toml
tune_error_limit = 0.08
```

Type `double` · Range `<x:R,0<x<=1>` · Worker override `[worker.N] tune_error_limit` · Level `advanced`.

Relative error limit for qualifying the nonnegative full-curve cost model by leave-one-out validation. Sparse estimates keep uncertainty labels and cannot override the default D. Local descent compares measured medians and MAD rather than prediction error.

<a id="stage2_tune_ignore"></a>

### tune_ignore

```toml
tune_ignore = "gpu,driver,cuda,backend,environment"
```

Type `string` · Range `[gpu|driver|cuda|backend|memory|environment],...` · Worker override `[worker.N] tune_ignore` · Level `advanced`.

Comma-separated conditions ignored for profile reuse. Empty enforces every condition. Memory normally matches execution path; arithmetic legality and current memory admission are always checked.

<a id="stage2_cost_profile"></a>

### tune_legacy_cost_profile

```toml
tune_legacy_cost_profile = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] tune_legacy_cost_profile` · Level `legacy`.

Legacy .cprof cost profile for Auto B2 when stage2_tune_profile is empty. Checks the original binary/device/arithmetic and calibration contract; new full ECM tune does not relax those checks. Empty supplies no legacy profile.

<a id="stage2_tune_min_repeats"></a>

### tune_min_repeats

```toml
tune_min_repeats = 2
```

Type `u64` · Range `[1..1000]` · Worker override `[worker.N] tune_min_repeats` · Level `advanced`.

Minimum checked observations required before runtime D, layout and chain candidates are compared or adopted. Also sets the minimum ECM calibration repeat count; --tune-repeats may request more. Default 2; 1 enables faster screening with no measured dispersion. Must not exceed tune_runtime_sample_limit. Existing observations are retained; increasing this minimum requires additional evidence. Arithmetic validation is unchanged. CLI: --tune-min-repeats N.

<a id="stage2_tune_runtime"></a>

### tune_runtime_enabled

```toml
tune_runtime_enabled = true
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] tune_runtime_enabled` · Level `advanced`.

CUDA Stage1 NTT tunes missing widths before production and uses read-only selection when disabled. Stage2 CUDA accumulates checked no-factor production timings by device, environment and execution shape. Identical complete parameters share one matrix row with repeats, mean time and sample standard deviation. Slower observations count too. Row version and UTC update time record provenance only. Software upgrades retain compatible timings. Model selection tests a legal Mersenne carrier first, then ordinary and promising D candidates. Candidates that reach tune_min_repeats are ranked by the fastest mean. B1 does not gate CUDA cost reuse. CPU GWNUM records timings in the [cpu_stage2] matrix and can reuse a matching FFT and F-tree plan across widths and memory budgets after preflight. Writes are locked and atomic. Cache failure does not fail a completed curve. false disables single-curve runtime trials, collection and writes. CPU can still read qualified observations. Parallel queue decisions use curve_concurrency. OpenCL learns point plans in opencl_tune_file at fixed geometry.

<a id="stage2_tune_runtime_samples"></a>

### tune_runtime_sample_limit

```toml
tune_runtime_sample_limit = 10
```

Type `u64` · Range `[1..1000]` · Worker override `[worker.N] tune_runtime_sample_limit` · Level `advanced`.

Maximum independent runtime NF observations per identical complete parameters (including B1) and matching device/environment/arithmetic/path, including slower observations and compatible software versions. Identical parameters merge into repeats and mean/stddev; the limit counts observations, not matrix rows. Different parameter rows are retained. Selection requires tune_min_repeats observations. Lowering this limit preserves existing evidence. CLI: --tune-runtime-samples N.

<a id="stage1_batch"></a>

### tune_stage1_batch

```toml
tune_stage1_batch = 1
```

Type `u64` · Range `<x:Z,1..2^20>` · Worker override `[worker.N] tune_stage1_batch` · Level `advanced`.

Stage1 batch curve count associated with Auto B2 cost estimation. Selects or checks Stage1 cost conditions; it neither starts a Stage1 batch from Stage2 nor changes the assignment's curve count.

<a id="stage1_cost_csv"></a>

### tune_stage1_cost_csv

```toml
tune_stage1_cost_csv = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] tune_stage1_cost_csv` · Level `advanced`.

CSV Stage1 s/curve anchors for workflow Auto B2. Columns: container_bits or target_bits, opencl_threads_per_integer, curves, b1, mhz, seconds_per_curve; optional tpb and exponent. Missing widths may interpolate or extrapolate. Costs are already amortized.

<a id="stage1_tune_profile"></a>

### tune_stage1_cost_file

```toml
tune_stage1_cost_file = ""
```

Type `string` · Range `<path>` · Worker override `[worker.N] tune_stage1_cost_file` · Level `advanced`.

Checked completed CUDA Stage1 cost anchors, used after explicit seconds/curve and CSV. Matches batch, modulus type and exponent; B1 and missing widths scale by B1 and the dispatched container squared. Device/runtime reuse follows tune_ignore. This does not run Stage1.

<a id="stage1_cost_mhz"></a>

### tune_stage1_cost_mhz

```toml
tune_stage1_cost_mhz = 0
```

Type `u64` · Range `<x:Z,0..10000>` · Worker override `[worker.N] tune_stage1_cost_mhz` · Level `advanced`.

Stage1 target frequency for inverse-frequency cost scaling. 0 uses the CSV reference frequency; this does not change GPU clocks.

<a id="stage1_seconds_per_curve"></a>

### tune_stage1_seconds_per_curve

```toml
tune_stage1_seconds_per_curve = 0
```

Type `double` · Range `<x:R,finite,x>=0>` · Worker override `[worker.N] tune_stage1_seconds_per_curve` · Level `advanced`.

Stage1 seconds per curve for Auto B2, already amortized for the actual batch. A positive value, including a decimal, overrides the Stage1 cost from tuning data. Set 0 or omit this option to use the configured Stage1 cost sources. Full ECM tune can use a cost CSV or a matching tune_stage1_cost_file. The legacy component profile uses its own matched Stage1 data. This value is a planning input.

<a id="stage2_tune_warmup"></a>

### tune_warmup

```toml
tune_warmup = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] tune_warmup` · Level `advanced`.

0 starts formal ECM/NTT tuning and startup calibration without a discarded warmup. 1 runs one checked warmup per shape before formal repeats; it consumes the same budget and is excluded from the median/MAD. CLI --tune-warmup 0|1 overrides this key. Mandatory arithmetic checks are always performed.

<a id="stage2_weighted_ntt"></a>

### weighted_ntt

```toml
weighted_ntt = false
```

Type `bool` · Range `[false|true|0|1]` · Worker override `[worker.N] weighted_ntt` · Level `advanced`.

Opt-in polynomial convolution for the production CUDA backend. Fuses the first DIF stage into direct digit packing and reconstructs both halves in the final inverse stage. Requires pack_zero. Calls without cached roots or direct device packing use the full-length path. The TOML preset stays inactive on OpenCL. CLI --weighted-ntt 0|1 overrides this key.

<a id="stage2_parallel_workspace"></a>

### workspace_policy

```toml
workspace_policy = "auto"
```

Type `string` · Range `[auto|independent|leased]` · Worker override `[worker.N] workspace_policy` · Level `advanced`.

independent gives each CUDA Stage2 slot a workspace. leased lets the second slot prepare and compute baby points before leasing the heavy workspace at the F-tree boundary. auto compares both and uses leased directly if only it fits.

## [gui]

### body_font_size

```toml
body_font_size = 15
```

Type `int` · Range `[10..30]` · Worker override `not supported` · Level `advanced`.

Body and log font size in logical pixels, independently of interface DPI scale. Primary actions and secondary text apply their configured offsets; headings preserve their ratio.

### display_language

```toml
display_language = "en"
```

Type `string` · Range `[en|zh]` · Worker override `not supported` · Level `advanced`.

UI language; CJK font failure falls back to English.

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

### external_windows

```toml
external_windows = false
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Allow detached native panel windows after restart; default keeps docking and floating inside the main window to avoid the observed slow presentation path.

### follow_log

```toml
follow_log = true
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Follow new worker output.

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

### gpu_history_seconds

```toml
gpu_history_seconds = 60
```

Type `int` · Range `[15|30|60|120]` · Worker override `not supported` · Level `advanced`.

GPU chart time span in seconds, shared by all devices and persisted with UI preferences. Does not change the 500 ms sampling interval.

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

### inspector

```toml
inspector = true
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Show the worker inspector on sufficiently wide windows.

### language

```toml
language = "english"
```

Type `string` · Range `<xml-stem>` · Worker override `not supported` · Level `advanced`.

GUI language, specified by the language XML filename stem in localization, without .xml. Missing resources or unavailable glyphs follow the GUI's display fallback rules.

### language_file

```toml
language_file = ""
```

Type `string` · Range `<string, up to 255 bytes>` · Worker override `not supported` · Level `advanced`.

Interface XML filename in localization/v2 beside the executable. Empty migrates the legacy en/zh selection. Companion options.*.xml files are not selectable interface files.

### localization_dir

```toml
localization_dir = ""
```

Type `string` · Range `<dir>` · Worker override `not supported` · Level `advanced`.

Directory containing GUI language XML files. Empty uses localization beside the GUI executable. Development builds also try the source-tree language resources.

### logs

```toml
logs = true
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Enable logs in Workers management; automatic hiding on other views does not change this saved preference.

### logs_keep_visible

```toml
logs_keep_visible = false
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Keep enabled logs visible across pages; otherwise follow visible Workers management, with a temporary manual override on other views.

### nav_collapsed

```toml
nav_collapsed = false
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Show only icons in the navigation sidebar.

### nav_expanded_width

```toml
nav_expanded_width = 0
```

Type `int` · Range `[0..16384]` · Worker override `not supported` · Level `advanced`.

Previous expanded navigation width in logical pixels. Zero uses the default width.

### nav_hide_tab_bar

```toml
nav_hide_tab_bar = true
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

The Navigation tab bar starts hidden by default. The saved choice survives restarts and layout rebuilding.

### nav_order

```toml
nav_order = "01237456"
```

Type `string` · Range `<string, up to 8 bytes>` · Worker override `not supported` · Level `advanced`.

Stable page IDs 0..7; ID 5 opens Settings. Legacy seven-page orders append Group Order Factor without resetting the layout.

### page

```toml
page = 0
```

Type `int` · Range `[0..7]` · Worker override `not supported` · Level `advanced`.

Selected page: 0 workers, 1 results, 2 generator, 3 probability, 4 GPU, 6 components, 7 Group Order Factor; 5 opens Settings.

### pause_schedule

```toml
pause_schedule = []
```

Type `string_array` · Range `["application, Worker, weekday, time", ...]` · Worker override `not supported` · Level `advanced`.

GUI v2 local-time pause rules. Each item has application, Worker, weekday and time groups. Example: ["foobar, [1,3-5], Mon-Fri, 08:30-17:30"]. Application names match a running program's path substring regardless of case. Use * for time-only rules, or an empty array to disable scheduling.

### pet_animate

```toml
pet_animate = true
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Animate at the existing GUI cadence; no extra render loop.

### pet_enabled

```toml
pet_enabled = false
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Show the ELYTerminal-compatible edge companion.

### pet_left

```toml
pet_left = 72
```

Type `float` · Range `[0..16384]` · Worker override `not supported` · Level `advanced`.

Companion offset from the main window left edge in logical units.

### pet_skin

```toml
pet_skin = "classic"
```

Type `string` · Range `[classic|miss_pink_elf|herrscher_of_human_ego|hi_love_elf]` · Worker override `not supported` · Level `advanced`.

ELY appearance. Switches immediately; official battlesuit names are preserved in the selector.

### primary_font_offset

```toml
primary_font_offset = 1
```

Type `int` · Range `[0..6]` · Worker override `not supported` · Level `advanced`.

Primary action font offset from the body size, in logical pixels; secondary buttons and log text keep the body size.

### process_priority

```toml
process_priority = "below_normal"
```

Type `string` · Range `[idle|below_normal|normal|above_normal|high]` · Worker override `not supported` · Level `advanced`.

Priority of worker processes launched by the GUI. below_normal is the default; lower process_priority makes concurrent desktop use easier. This is not a GPU exclusivity or utilization setting.

### raw_log

```toml
raw_log = false
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Show raw output instead of classified event lines.

### refresh_hz

```toml
refresh_hz = 10
```

Type `int` · Range `<x:Z,1..60>` · Worker override `not supported` · Level `advanced`.

Idle GUI frame and worker-status refresh target in Hz. Input messages wake the GUI promptly; higher values smooth charts and status updates at additional UI cost. GPU metrics are sampled independently at gpu_poll_ms, and neither setting changes computation speed.

### results_last_copy_utc

```toml
results_last_copy_utc = ""
```

Type `string` · Range `<YYYY-MM-DD HH:MM:SS UTC>` · Worker override `not supported` · Level `advanced`.

Last successful JSON result copy in GUI v2, recorded in UTC. Empty means no copy time has been recorded.

### right_hide_tab_bar

```toml
right_hide_tab_bar = false
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Remember the right dock Hide tab bar choice. Multiple native tabs remain visible; the choice resumes when one remains.

### right_page

```toml
right_page = 4
```

Type `int` · Range `[0..7]` · Worker override `not supported` · Level `advanced`.

Last active real page in the right dock. Legacy Settings ID 5 is remapped on load.

### right_pages

```toml
right_pages = 0
```

Type `int` · Range `[0..254]` · Worker override `not supported` · Level `advanced`.

Real pages in the right dock share native horizontal tabs. Closing returns a page to navigation; one real page remains central. Settings bit 5 is cleared when opening old sidecars.

### right_visible

```toml
right_visible = true
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Compatibility field retained for old sidecars. Right-side pages are always displayed; close their tabs to return them to navigation.

### scale

```toml
scale = 0
```

Type `float` · Range `[0..4]` · Worker override `not supported` · Level `advanced`.

Zero selects window DPI; positive values select a fixed interface scale.

### secondary_font_offset

```toml
secondary_font_offset = -2
```

Type `int` · Range `[-6..0]` · Worker override `not supported` · Level `advanced`.

Secondary text font offset from the body size, in logical pixels; the rendered role has a minimum of 8 logical pixels.

### selected_worker

```toml
selected_worker = 1
```

Type `int` · Range `[1..1024]` · Worker override `not supported` · Level `advanced`.

Selected worker partition.

### smooth_scroll

```toml
smooth_scroll = true
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Smooth wheel scrolling in main and detached windows.

<a id="exe"></a>

### stage1_exe

```toml
stage1_exe = ""
```

Type `string` · Range `<exe>` · Worker override `not supported` · Level `advanced`.

Stage1 worker executable. Relative paths resolve beside the GUI; bare names can use PATH. Empty searches ecm_cuda.stage1_exe then ecm.stage1_exe. The worktodo_file prefix selects this executable or stage2_exe for each worker.

### stage2_exe

```toml
stage2_exe = ""
```

Type `string` · Range `<exe>` · Worker override `not supported` · Level `advanced`.

Stage2 executable for CPU, CUDA or OpenCL. Empty searches ecm_stage2.stage1_exe, then legacy ecm_cuda_stage2.stage1_exe, ecm_cpu_stage2.stage1_exe and ecm_opencl_stage2.stage1_exe in the workflow and GUI directories. A single matching child package can also be discovered. Set an explicit path to choose between installed packages.

### stage2_timing_profile

```toml
stage2_timing_profile = ""
```

Type `string` · Range `<string, up to 4096 bytes>` · Worker override `not supported` · Level `advanced`.

Optional read-only Stage2 tune summary path. Empty uses the workflow stage2_tune_profile, then tune.toml or stage2_ecm_tune.toml next to the workflow/worker. Relative paths use the workflow directory.

### stage2_timing_source

```toml
stage2_timing_source = "auto"
```

Type `string` · Range `[auto|previous|tune]` · Worker override `not supported` · Level `advanced`.

Stage2 segment duration source: auto overlays previous completed-curve timings on tune data; previous or tune restricts the source. Missing durations are marked unknown.

### start_tab

```toml
start_tab = "workers"
```

Type `string` · Range `[workers|detail|gen|prob]` · Worker override `not supported` · Level `legacy`.

Panel initially shown by the GUI: workers opens the worker list, detail the details panel, gen the task generator, and prob the ECM probability calculator. The probability panel can also be opened from View. Does not change worker autostart behavior.

### theme

```toml
theme = "dark"
```

Type `string` · Range `[dark|light]` · Worker override `not supported` · Level `advanced`.

Appearance mode, independent of legacy GUI settings.

### toasts

```toml
toasts = true
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Show non-activating action notifications; errors always remain visible.

### window

```toml
window = [120, 80, 1500, 900]
```

Type `rect` · Range `<x:Z>,<y:Z>,<w:Z>,<h:Z>` · Worker override `not supported` · Level `legacy`.

Saved main-window geometry: x,y,width,height. Updated on normal GUI exit; manual editing is usually unnecessary. Width and height should be positive.

### window_height

```toml
window_height = 850
```

Type `int` · Range `[360..16384]` · Worker override `not supported` · Level `advanced`.

Restored window height in physical pixels.

### window_maximized

```toml
window_maximized = false
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Restore the maximized state.

### window_title_bar

```toml
window_title_bar = false
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Show the system title bar; hidden by default with window controls in the app top bar. Explicit saved choices are retained.

### window_width

```toml
window_width = 1280
```

Type `int` · Range `[480..16384]` · Worker override `not supported` · Level `advanced`.

Restored window width in physical pixels.

### window_x

```toml
window_x = -2147483648
```

Type `int` · Range `[-2147483648..2147483647]` · Worker override `not supported` · Level `advanced`.

Window x in physical pixels; INT_MIN selects system placement.

### window_y

```toml
window_y = -2147483648
```

Type `int` · Range `[-2147483648..2147483647]` · Worker override `not supported` · Level `advanced`.

Window y in physical pixels; INT_MIN selects system placement.

### worker_chart_auto_y

```toml
worker_chart_auto_y = false
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

Fit curve-time Y bounds to finite samples in the current retained window; off keeps a zero baseline.

### worker_chart_columns

```toml
worker_chart_columns = 1
```

Type `int` · Range `[1|2]` · Worker override `not supported` · Level `advanced`.

Curve-time chart columns: 1 for list, 2 for two charts per row; narrow workspaces temporarily use one column without changing the saved choice.

### workspace_hide_tab_bar

```toml
workspace_hide_tab_bar = true
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `advanced`.

The Workspace tab bar starts hidden by default. The saved choice survives restarts and layout rebuilding.

### workspace_layout

```toml
workspace_layout = ""
```

Type `string` · Range `<string, up to 524288 bytes>` · Worker override `not supported` · Level `advanced`.

Hex encoded ImGui layout; stored only in this sidecar.

## [worker.1]

### autostart

```toml
autostart = false
```

Type `bool` · Range `[false|true]` · Worker override `not supported` · Level `core`.

Start this worker when the GUI opens. The setting applies to GUI startup.

### extra_args

```toml
extra_args = ""
```

Type `string` · Range `<argv>` · Worker override `not supported` · Level `advanced`.

Command-line arguments common to both stages of this GUI-launched worker. Use the stage-specific keys for options that only one stage understands. Workflow configuration, worker number and queue path are managed by the GUI.

### gpucurves

```toml
gpucurves = 0
```

Type `int` · Range `<x:Z,x>=0>` · Worker override `not supported` · Level `legacy`.

Legacy GUI curve-count display field, inheriting the global value if present. Stage1 queue curve counts come from task lines; this field does not override them. The single-run -gpucurves CLI option is independent.

### name

```toml
name = "Worker #{N}"
```

Type `string` · Range `<text>` · Worker override `not supported` · Level `advanced`.

Worker display name in the GUI. Unset shows Worker #N, where N is the worker index. Changes only the name, not device or task assignment.

### stage1_extra_args

```toml
stage1_extra_args = ""
```

Type `string` · Range `<argv>` · Worker override `not supported` · Level `advanced`.

Additional Stage1 command-line arguments appended after the common extra_args. Separate arguments with spaces and quote paths containing spaces.

### stage2_extra_args

```toml
stage2_extra_args = ""
```

Type `string` · Range `<argv>` · Worker override `not supported` · Level `advanced`.

Additional Stage2 command-line arguments appended after the common extra_args. Separate arguments with spaces and quote paths containing spaces.

## Maintenance

Edit `config/ecm_options.json` and run `python tools/gen/generate_ecm_config.py`. Commit the schema and generated files together.
