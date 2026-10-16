# ECM-ELF

[中文](README.md) | English

ECM-ELF is an Elliptic Curve Method (ECM) toolkit: CUDA/CGBN and OpenCL GPU Stage1, GMP and optional AVX-512 IFMA CPU Stage1, a separate CUDA polynomial Stage2, libecm CPU Edwards Stage2, plus a Windows GUI, task queues and factor-data tools. The former name, OpenCL-ECM, described only one of these backends; executable names such as `ecm.exe` and `ecm_cuda.exe` remain in use.

This page retains the purpose of each entry point, runnable commands, configuration workflows, save-file boundaries, build routes and tool index. The generated [INI reference](docs/ECM_INI_REFERENCE.md) is authoritative for configuration defaults, ranges and bilingual descriptions. Check the actual binary and its help output to determine which kernel tiers or experimental algorithms a build includes.

![Language: C](https://img.shields.io/badge/language-C-blue)
![GitHub License](https://img.shields.io/github/license/AllogeneSSRD/ECM-ELF)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/AllogeneSSRD/ECM-ELF)
![GitHub last commit](https://img.shields.io/github/last-commit/AllogeneSSRD/ECM-ELF)

<a id="comparison"></a>
## Compared with established ECM tools

ECM-ELF provides a standalone, multi-backend ECM workflow: CUDA/CGBN, OpenCL or CPU runs batches of Stage1 curves, then this repository's CUDA polynomial Stage2 or an eligible external tool processes the completed save.

- **Compute paths:** CUDA and OpenCL cover different GPU platforms; CPU offers scalar GMP and batched AVX-512 IFMA implementations. The Stage1 paths share a task entry point. Available operand sizes and algorithms depend on the build.
- **Independent Stage2:** CUDA Stage2 uses exact integer polynomial arithmetic and supports direct save input, a separate queue, per-curve resumption, measured tuning and Auto B2. [Stage2 usage](docs/usage/STAGE2.md) lists input and resource limits.
- **Long-running work:** worktodo, workers, the Windows GUI and Prime95 handoff support batch operation. Stage1 writes checkpoints at safe boundaries and rotates backups; Stage2 resumes queued work from completion receipts. See [queues and handoff](#queues).
- **Save interoperability:** Eligible Suyama PARAM0 and batch PARAM3 saves can be handed to GMP-ECM or Prime95.

For performance comparisons and applicable ranges, see [Stage1 performance](docs/performance/STAGE1.md), [Stage2 performance](docs/performance/STAGE2.md) and the [historical performance record](#historical-performance). Each result is bounded by its stated input, device, build and timing scope.

## Contents

| Section | Scope |
| --- | --- |
| [Compared with established ECM tools](#comparison) | Backend, save compatibility and workflow overview; performance data follows below |
| [Programs and capabilities](#programs) · [Quick start](#quick-start) | Choosing a program, runnable commands and common options |
| [CPU Stage1 configuration](#cpu-stage1) · [Queues and handoff](#queues) | Recipes, thread batches, workers, saves and checkpoints |
| [Separate CUDA Stage2](#stage2) · [Building and releasing](#build) | Stage2 input/resuming and build entry points |
| [Android, GUI and bundled tools](#android-tools) | Mobile builds, graphical frontend and helper tools |
| [Historical performance record](#historical-performance) | Earlier README data and its limits |
| [Documentation and repository layout](#docs-layout) · [Dependencies and license](#license) | Topic guides, source entry points and licenses |

<a id="programs"></a>
## Programs and capabilities

| Program | Current purpose | Main input and output |
| --- | --- | --- |
| `ecm_cuda.exe` | NVIDIA CUDA/CGBN Stage1, processing batches of curves; PARAM0, PARAM2 and PARAM3 families when built in | Number/expression or Stage1 queue; completed `.save` files and in-progress checkpoints |
| `ecm.exe` | OpenCL GPU Stage1 or CPU Edwards and Suyama-Montgomery Stage1 | Shared driver input, INI, queue and text saves; selectable OpenCL operators |
| `ecm_cuda_stage2.exe` | Separate CUDA polynomial Stage2 for eligible normalized PARAM0 Stage1 saves | Direct save input or separate `stage2_worktodo`; factors, results and per-curve receipts |
| `ecm_gui.exe` | Windows Stage1 workers, configuration, queue generation, logs, GPU status and results | Manages worker processes; the chosen `ecm_cuda.exe` still performs the computation |
| `ecm_p95feeder` | Prime95 handoff for local CPU saves | Delivers tasks Prime95 can consume; the driver also supports completed-task handoff |

### Stage1 backends

- **CUDA/CGBN** is the native Windows CUDA path. The host builds the exponent and curves, the GPU advances point multiplication in slices, and the host checks final points before writing saves. `gpu_param=0` is the Suyama sigma family shared with CPU Montgomery and GMP-ECM `-param 0`. `gpu_param=2` is GMP-ECM batch 2 and produces saves with `PARAM=2`, which Prime95 does not accept. `gpu_param=3` is the batch family available to both CUDA and OpenCL. Supported operand sizes depend on the kernel tiers included in the binary. Production builds use the ladder by default; resident/PRAC require explicit build and runtime selection. See [Stage1 implementation](docs/architecture/STAGE1.md).
- **OpenCL** is the Windows, Linux, macOS and Android GPU path. Its registry selects Montgomery multiplication/squaring, modular addition/subtraction and special multiplication operators. `ecm.exe --showkernel` lists paths in the current build. CUDA and OpenCL device IDs, runtimes and internal checkpoints are not interchangeable. See the [OpenCL architecture](docs/architecture/OPENCL.md) and [kernel guide](kernels/opencl/README.md).
- **CPU** offers independent Edwards (Atkin–Morain) and Suyama-Montgomery curve families, using scalar GMP or eight-curve AVX-512 IFMA SIMD batches. `reduction=auto` can use Mersenne folding for a full `2^k−1` target and Montgomery reduction for a general integer. `impl=simd` requires that ISA; `impl=auto` may fall back. See [Stage1 implementation](docs/architecture/STAGE1.md) and the [mathematical references](docs/reference/README.md).

`gpucurves` is a task batch size, not a promise that all curves reside on the GPU at once. Choose GPU batches using `--gpu-info`, operand size, TPI, TPB, SM count and actual resource occupancy. CPU SIMD parallelism is bounded by `ceil(curves/8)` batches; the scalar path assigns one curve per task. With a fixed sigma, curve `i` uses `sigma+i`. The same numeric sigma under different parametrizations does not define the same curve.

<a id="quick-start"></a>
## Quick start

Run these commands from the executable directory, or replace the paths with the output locations described in the [build guide](docs/usage/BUILD.md). `B2=0` runs Stage1 only; give the completed save to a compatible Stage2 consumer afterward.

```powershell
# Inspect the actual backend, options and available tiers; --gpu-info runs no curve
.\ecm_cuda.exe --help
.\ecm_cuda.exe --gpu-info --bits 5240
.\ecm.exe --showkernel

# One-off OpenCL / CUDA Stage1; N may be a decimal number or an expression
echo '(2^347-1)' | .\ecm.exe --curve mont --target gpu -gpucurves 1 -d 0 1e4 0
echo '(2^421-1)' | .\ecm_cuda.exe --curve mont --target gpu --gpu-param 0 -gpucurves 32 -d 0 1e5 0

# CPU Suyama-Montgomery; auto selects SIMD on a supported CPU
echo '(2^1277-1)' | .\ecm.exe --curve mont --target cpu --impl auto -gpucurves 8 1e5 0
```

A one-off run has the form `echo '<N>' | <ecm.exe or ecm_cuda.exe> [options] B1 [B2]`. With no positional bound, the program enters Stage1 queue mode. `--curve` selects the curve and `--target` selects CPU or GPU; the default is Montgomery on GPU, and CUDA supports GPU Edwards while OpenCL does not. `-d` is the device index enumerated by that program. `--gpu-info` reports CUDA tiers and suggested batch sizes; an OpenCL build reports that this query is not applicable.

### Run Stage1 from worktodo

This example assumes `ecm_cuda.exe`, `ecm_cuda_stage2.exe` and `ecm.ini` are in the same directory and commands run there. Merge these keys into an existing INI if you have one. Create `ecm.ini`:

```ini
[Worker #1]
curve = mont
target = gpu
gpu_param = 0
device = 0
worktodo = worktodo.txt
finished = worktodo.finished.txt
tmp_dir = .
stage2_worktodo = stage2_worktodo.txt
stage2_finished = stage2_worktodo.finished.txt
```

Create `worktodo.txt`. The final `ECM2` fields are B1, B2 and curve count; `B2=0` runs Stage1 only:

```text
[Worker #1]
ECM2=1,2,3001,-1,10000,0,8
```

```powershell
.\ecm_cuda.exe -ini .\ecm.ini --worker 1
```

With no positional bounds, this command consumes assignments for the selected worker. A Montgomery `ECM2=` queue task writes its final save under `tmp_dir` using the `m{n}_{b1}.save` format, such as `m3001_1e4.save`. Completed assignments go to `worktodo.finished.txt`. On a normal interruption or save write failure, unfinished assignments remain in `worktodo.txt`; rerun the command to resume from a valid checkpoint. `ecm.exe` has the same queue entry point when configured for the chosen CPU/OpenCL path. Use one consumer per queue and separate queues and output paths for different workers. See [queues and handoff](#queues) and [Stage1 usage](docs/usage/STAGE1.md).

### Run and tune CUDA Stage2

After Stage1 finishes and writes `m3001_1e4.save`, create a separate `stage2_worktodo.txt`. This task has an explicit B2; the save name must match the actual output:

```text
[Worker #1]
ECMSTAGE2=1,2,3001,-1,"m3001_1e4.save",1000000
```

```powershell
# Optional: measure complete ECM using the current Stage2 queue and save; leaves the queue untouched
.\ecm_cuda_stage2.exe --ini .\ecm.ini --worker 1 --tune ecm --tune-level 1 --tune-file .\stage2_ecm_tune.toml

# Consume one queued assignment; after tuning, you may add --tune-profile .\stage2_ecm_tune.toml
.\ecm_cuda_stage2.exe --ini .\ecm.ini --worker 1 --once

# Or process the same save directly without the queue (do not run both paths)
.\ecm_cuda_stage2.exe --save .\m3001_1e4.save --b2 1000000 --device 0

# Optional: measure the NTT component only; this is not a complete ECM tune
.\ecm_cuda_stage2.exe --tune ntt --tune-level 1 --tune-file .\stage2_ntt_tune.toml
```

ECM tune reads the selected worker's Stage2 queue by default. If the queue is absent, it measures a generic bit-width grid; `--tune-save FILE --tune-b2 B2` selects one save explicitly. Tuning has a time budget, and the resulting profile can be loaded with `--tune-profile FILE` or the `stage2_tune_profile` INI key. Explicit B2 does not require tuning; Auto B2 also requires qualified Stage1 cost and a Stage2 profile. See [separate CUDA Stage2](#stage2), [Stage2 usage](docs/usage/STAGE2.md) and [Auto B2](docs/architecture/AUTO_B2.md) for queue, direct-save and Auto B2 limits.

Common options are below. Run the relevant binary's `--help` and consult the [INI reference](docs/ECM_INI_REFERENCE.md) for all options and exact defaults.

| Option | Purpose |
| --- | --- |
| `-gpucurves N`, `--gpu-param P`, `-sigma S` | Curve batch, GPU parametrization and fixed seed; PARAM0 can use a full 64-bit sigma |
| `--curve mont|ed`, `--target cpu|gpu` | Curve model and execution target |
| `--impl auto|simd|scalar`, `--reduction auto|montgomery|mersenne-fold` | CPU arithmetic and SIMD reduction |
| `--lcm 1|12` | Montgomery Stage1 scalar construction |
| `--stage1-threads N`, `--affinity LIST` | CPU workers and optional affinity; measure before pinning on hybrid CPUs |
| `--ckpt SEC`, `--tmp-dir DIR`, `--exp-cache DIR` | Checkpoint interval, local Stage1 output and exponent cache |
| `-ini FILE`, `--worker N` | Queue configuration and section selection |
| `--mul/--sqr/--add/--sub/--special-mult ID` | OpenCL operator paths; CUDA does not use these overrides |
| `--go`, `--gp PATH`, `--verify-gpu`, `--profile-ops` | Group-order diagnostics, GP path, GPU cross-check and operator counts |

Current `--help` also lists OpenCL kernel-root, binary-cache, TPI, work-group, logging and diagnostic options.

### OpenCL runtime and operator diagnostics

The former README's environment-variable migration corresponds to current flags. `--kernel-root DIR` selects the `.cl` source tree; `--kernel-cache-dir DIR`, `--no-kernel-cache` and `--kernel-cache-verbose` control the binary cache; `--compile-verbose` reports compilation time. `--tpi N` and `--wg N` support experiments with instance threads and work-group size. `--force-normalize`, `--addsub-fused-unroll`, `--sync-each-batch`, `--gpu-dump`, `--profile-ops` and `--verify-gpu` support diagnostics and cross-checks. They are not CUDA kernel-selection interfaces; validate results again after changing an operator path.

Operator microbenchmarks have their own CLI. `opencl_ecm_addsub` provides `--no-asm`, `--asm-b64`, `--addsub-fused-unroll` and `--csv`; `opencl_ecm_montsqr` provides `--wg-impl`, `--wg-impl4-unroll`, `--csv` and related options. Both take kernel iterations, instances and launch repeats as positional arguments, with `--bits` selecting operand size. These are single-operator measurements and cannot be equated directly with complete ECM seconds per curve. Source entry points are `src/opencl_ecm_addsub_bench.cpp` and `src/opencl_ecm_montsqr_bench.cpp`.

<a id="cpu-stage1"></a>
## CPU Stage1 configuration

Both CPU curve models use the same driver and `ecm.ini` as the GPU, but curve construction, coordinates and checkpoint files differ. `curve=mont` with `target=cpu` selects Suyama sigma / GMP-ECM PARAM0; `curve=ed` selects the Atkin–Morain family. For Montgomery, `lcm=1` uses `lcm(1…B1)`; `lcm=12` multiplies it by 12 for workflows requiring Prime95 scalar semantics. Edwards uses a fixed `48*lcm(1…B1)` scalar and rejects `lcm=12`. The Montgomery choice changes the computed point and must be recorded when comparing saves.

Example local Montgomery configuration:

```ini
curve = mont
target = cpu
impl = auto
reduction = auto
lcm = 1
stage1_threads = 0
tmp_dir = saves
worktodo = worktodo.txt
finished = worktodo.finished.txt
```

Set `curve=ed` for the Edwards path; `naf_w` applies only there. To compare reductions on the same mathematical target, set `reduction` to `mersenne-fold` and `montgomery` in separate runs. The former requires a full `2^k−1` target. A Suyama save handed to Prime95 Stage2 must use compatible parametrization and scalar construction, and Prime95 must be able to access the save. Applicable keys and ranges are in the [INI reference](docs/ECM_INI_REFERENCE.md).

Common CPU recipes:

| Purpose | Change from the example above | Check |
| --- | --- | --- |
| GMP-ECM PARAM0 comparison | `curve=mont`, `lcm=1` | Compare the Stage1 final point for the same N, B1 and sigma; run Stage1 only in the reference program |
| Prime95 Suyama handoff | `curve=mont`, `lcm=12` | Curve and scalar semantics, save location and task fields required by Prime95 |
| Edwards on a full Mersenne number | `curve=ed`, `reduction=auto` | Actual reduction, NAF setting and Edwards-specific save |
| Edwards on a general modulus or for A/B | `curve=ed`, `impl=simd`, `reduction=montgomery` | Avoid forcing folding after known factors have been removed and N is no longer `2^k−1` |

`save_name_pattern` controls only the CPU Montgomery output name; other backends retain their own naming. A GMP-ECM text save, a Prime95 assignment and a local `.ckpt` serve different purposes. To continue this repository's unfinished CPU job, rerun the original command so it can read its checkpoint. An external `-resume` on a completed save still needs the bounds required by that program.

One eight-curve SIMD batch is one CPU scheduling task; 64 curves therefore create at most eight parallel batches. `stage1_threads=0` chooses worker count automatically, and an explicit count is still bounded by available tasks. Do not run multiple processes against the same `(N,B1)` save/checkpoint path; give them separate output directories. A fixed sigma helps reproduce a curve, while a random seed should not be inferred from log order.

For common failures: if explicit `simd` reports a missing ISA, use `auto` or `scalar`; if workers are idle, compare the number of curve batches with the worker count; if `reduction=mersenne-fold` is rejected, verify that the actual N is still a full Mersenne number; if an INI edit has no effect, check the selected worker section and CLI overrides. Stage1 validation should compare final points under identical N/B1/sigma/parametrization/scalar conditions. Passing a checksum alone is not mathematical agreement.

<a id="queues"></a>
## Queues, workers and handoff

Stage1 reads `worktodo.txt` and moves completed tasks into `worktodo.finished.txt`; failed tasks retain an error record. `[Worker #N]` sections in INI and worktodo must correspond: `--worker N` selects both. Give each queue one consumer and each worker separate output paths. The GUI generator can parse PrimeNet `ECM=` / `ECM2=` assignments, preview them and append them to the queue; the worker performs computation and writes the real saves.

For example, put a Stage1 task into the chosen worker section:

```text
ECM2=1,2,3001,-1,100000,0,64
```

Known factors, AID, sigma, curve count and save names have distinct queue semantics. Renaming a save cannot make it belong to another task. For production handoff, configure `p95_worktodo_path`: the driver synchronizes completed saves and appends assignments to Prime95's `worktodo.add`; failed deliveries enter a pending file for retry. The GUI displays status and is not a second consumer of Prime95's active queue. `ecm_p95feeder` can separately handle local CPU saves. See [Stage1 usage](docs/usage/STAGE1.md) and [GUI usage](docs/usage/GUI.md).

### Saves and checkpoints are different

A completed `.save` contains a normalized point and task fields for a compatible Stage2. An in-progress checkpoint contains unfinished point state, exponent position and implementation layout. CUDA, OpenCL, CPU Edwards and CPU Montgomery validate their own checkpoint formats; similar names do not make them cross-backend files. A normal stop or Ctrl+C waits for a safe save boundary. Forced termination can leave a partially written checkpoint. Resume the original task with matching N, B1, parametrization, exponent mode and compatible build; the log reports whether a checkpoint was loaded. See [Stage1 usage](docs/usage/STAGE1.md).

Save compatibility is also bounded. Eligible CUDA/CPU PARAM0 and CUDA/OpenCL PARAM3 text records can feed the corresponding GMP-ECM/Prime95 workflow. PARAM2 saves carry `PARAM=2`; GMP-ECM accepts them, Prime95 does not. This repository's CUDA Stage2 accepts normalized PARAM0 and Edwards AM text records. CPU/GPU Edwards share the same text save, also accepted by the separate [libecm CPU Stage2](docs/usage/EDWARDS_CPU_STAGE2.md). A checksum tests file integrity, not mathematical correctness of GPU point multiplication. Independent acceptance checks should cover N, B1, sigma, exponent mode and final point.

<a id="stage2"></a>
## Separate CUDA Stage2

`ecm_cuda_stage2.exe` consumes a completed Stage1 save; it does not rerun Stage1. Direct invocation and queue examples (replace the filename and B2 with the actual task):

```powershell
.\ecm_cuda_stage2.exe --save .\m3701_260e6.save --b2 26000000000 --device 0
.\ecm_cuda_stage2.exe --ini .\ecm.ini --worker 1 --once
```

The Stage2 queue is selected by `stage2_worktodo`, independently of Stage1's `worktodo`. An assignment has the form `ECMSTAGE2=[AID,]k,b,n,c,save[,B2][,skip_curves][,num_curves][,"known_factors"]`. Direct save input can use `--skip-curves` and `--curves` to select records. The queue records completed curves with receipts and reconciles them after interruption; an unfinished current curve must be recomputed. Missing/corrupt saves, mismatched task fields and device/resource failures have distinct queue handling.

Manual B2 supports eligible 2…16384-bit inputs. Auto B2 additionally needs a suitable cost profile and Stage1 cost, and always remains subject to current memory and arithmetic checks. Give production tasks an explicit B2 when no qualified model is available. `--dry-run` only validates input. `--plan-only` may run a short startup calibration; add `--short-calibration 0` for a pure query. See [Stage2 usage](docs/usage/STAGE2.md), [pipeline](docs/architecture/STAGE2.md) and [Auto B2](docs/architecture/AUTO_B2.md) for arguments, receipts, memory summaries and tuning data.

<a id="build"></a>
## Building and releasing

CUDA Stage1 and Stage2 each have **development, local production and release production** PS1/BAT entry points. The PS1 files under `tools/build/` define their defaults and full parameters:

```powershell
.\tools\build\build_stage1_dev.ps1
.\tools\build\build_stage1_local.ps1
.\tools\build\build_stage1_release.ps1
.\tools\build\build_stage2_dev.ps1
.\tools\build\build_stage2_local.ps1
.\tools\build\build_stage2_release.ps1
```

Release entry points target sm60, 70, 75, 86, 89 and 120 by default: sm60/70 use CUDA 12.6, while the other tiers use CUDA 13.3. Execution still needs a matching NVIDIA GPU, driver and validated CPU dependencies. Local/development builds default to a local Zen3 GMP; releases default to generic x64 GMP. A local DLL is not automatically suitable for redistribution. Production Stage1 defaults to the complete supported tiers/curve families and the ladder; PRAC/resident experiments require explicit build and runtime selection. Successful compilation is not a runtime qualification of every architecture. See [build and release](docs/usage/BUILD.md) and [launcher parameters](tools/build/README.md).

The OpenCL/CPU CMake project needs a suitable C++ toolchain, GMP, OpenCL runtime/headers and required build libraries; CUDA also needs compatible nvcc/MSVC. `ecm.exe` and OpenCL operator microbenchmarks can be built separately. Use the current [build guide](docs/usage/BUILD.md) and CMake options for Windows debug/release or a specific toolchain. The following microbenchmark entries are retained from the earlier README; actual targets and options depend on the build:

```powershell
# Configure the OpenCL/CPU project separately; supply local dependency paths if needed
cmake -S . -B build_opencl -DECM_ENABLE_CUDA=OFF
cmake --build build_opencl --config Release

.\build_rel\Release\opencl_ecm_addsub.exe --bits 512 10000 128 3
.\build_rel\Release\opencl_ecm_montsqr.exe --bits 512 1000 128 1
```

The microbenchmark examples retain their old `build_rel` paths. If built using `build_opencl` above, use the actual executable paths from that directory. `--showkernel` lists operators and `--gpu-info` reports CUDA tiers; neither is an end-to-end throughput benchmark.

<a id="android-tools"></a>
## Android, GUI and bundled tools

The Android app lives in `Android/ECM` and provides OpenCL device probing, ECM Stage1 factoring, operator microbenchmarks, work queues and local saves. Open that subproject in Android Studio; native factoring requires GMP. The app loads the vendor OpenCL library at runtime. Do not package a `libOpenCL.so` pulled from a phone into the APK. The build syncs `.cl` kernels into assets; the first compilation of a large kernel may take time. See the [Android overview](Android/README.md), [factoring guide](Android/ECM/README_ECM_FACTORIZATION.md) and [app guide](Android/ECM/README.md) for entry points, 16 KiB page constraints and device checks.

The Windows GUI uses Dear ImGui docking, Win32/DX11 and NVML. One worker corresponds to one Stage1 child process. It edits configuration, generates/previews assignments and displays worker output, GPU metrics and results. It performs no ECM arithmetic and is not a second Prime95 queue consumer. The `[GUI]` and `[Worker #N]` settings, window recovery and stop behavior are covered in [GUI usage](docs/usage/GUI.md).

- [`ecm_prob`](tools/ecm_prob/README.md): ECM success probability, parametrization and effective-divisor models; interpret predictions under their model conditions.
- [`ecm_report`](tools/ecm_report/README.md): PrimeNet ECM progress and result summaries.
- [`ecm_dataset`](tools/ecm_dataset/README.md): Mersenne and factor data, including sigma management and queries.
- [`tools/bench`](tools/README.md): device arithmetic, CUDA/OpenCL, NTT, Stage2 planning and timing probes; run data goes into ignored `data/`.
- [`tools/test`](tools/README.md): CPU/GPU arithmetic, queue, GUI, save/checkpoint and Stage2 regression checks.

PrimeNet progress example from the original README:

![Example ECM progress for exponents 1–20000](docs/image/ecm_progress_1-20000_factored_overlay.png)

The probability-model and measured-comparison images from that README are retained as **historical examples**. Their samples, models and conclusions are not a fresh validation of current binaries:

![Effective divisor against factor size](docs/image/emp_d_eff_vs_bit.png)

| Historical success-rate sample | Historical success rate versus B1 |
| --- | --- |
| ![Measured success rate versus factor size](docs/image/emp_success_vs_bit.png) | ![Measured and predicted success rate versus B1](docs/image/success_vs_B1.png) |

<a id="historical-performance"></a>
## Historical performance record and limits

The CPU comparison in the older README is not a current size recommendation or performance promise, but remains useful for checking that older implementation. The table preserves **single-thread, B1=1e6, seconds per curve** historical values from 2026-09-24, sourced from `docs/ECM_Montgomery_STAGE1.md` §§14.1–14.6 in commit `9be3d433855856fbb1519ff5e77e0bdc3bcb4789`. Targets are full Mersenne numbers. This implementation used AVX-512 IFMA eight-curve batches on a Ryzen AI 9 HX 370; GMP-ECM was 7.0.6 `ecm-zen3.exe` (`-c 2`); Prime95 was v31.04b05 with one worker. The original record reports pinned big/small core clocks of 3990/2995 MHz and a 90 W condition. Stage1 alone was timed: batch wall time divided by eight for this implementation, and `Step 1 took` for GMP-ECM. Measured batches here used eight curves and the GMP-ECM comparison command used two. Only the Prime95 M3001 value is measured, as a median of seven curves; its other rows use a calibrated FFT-tier model. Some rows for this implementation and GMP-ECM were also fitted or extrapolated by operand size: the table must not be read as ten direct measurements. The original record lacks an independently checkable SHA for this implementation's executable, so this table cannot describe a current build.

| N | This implementation, SIMD | GMP-ECM 7.0.6 | Prime95 v31.04b05 |
| --- | ---: | ---: | ---: |
| M127 | 0.118 s | — | 3.86 s (model) |
| M521 | 0.371 s | — | 3.86 s (model) |
| M1277 | 0.979 s | 2.484 s | 3.86 s (model) |
| M2203 | 2.285 s | 5.804 s | 3.86 s (model) |
| M3001 | 4.137 s | 9.586 s | 5.65 s (measured median of 7 curves) |
| M3500 | 5.164 s | 11.656 s | 5.65 s (model) |
| M4001 | 6.290 s | 14.624 s | 5.65 s (model) |
| M5755 | 12.05 s (fit) | ~21 s (extrapolation) | 8.06 s (model) |
| M8527 | 24.9 s (fit) | ~43 s (extrapolation) | 10.06 s (model) |
| M19701 | ~123 s (fit) | ~215 s (extrapolation) | 20.6 s (model) |

These historical numbers compare only the recorded machine, exponent, binaries and timing boundary. The larger-size rows in the original document were fitted; they do not establish today's performance crossover. Do not extrapolate them to a different power limit, GPU/CPU, curve family or full Stage1+Stage2 wall time. Current repeatable conditions and evidence are in [Stage1 performance](docs/performance/STAGE1.md) and [Stage2 performance](docs/performance/STAGE2.md).

<a id="docs-layout"></a>
## Documentation and repository layout

- The [documentation index](docs/README.md) covers current usage, implementation, performance evidence and TODO. [Stage1](docs/usage/STAGE1.md), [Stage2](docs/usage/STAGE2.md) and [GUI](docs/usage/GUI.md) give detailed workflows.
- The [architecture overview](docs/architecture/OVERVIEW.md) traces the driver in `src/core/`, CPU code in `src/cpu/`, CUDA glue in `src/cuda/`, the OpenCL host in `src/opencl_ecm_stage1.cpp`, and kernels in `kernels/cuda/` and `kernels/opencl/`. [Configuration maintenance](docs/architecture/CONFIGURATION.md) explains generation from `config/ecm_options.json` into templates and references.
- [Fixed mathematical and third-party references](docs/reference/README.md) retain ECM, curve, Goldilocks/NTT, Prime95/PrMers/GPUOWL derivations and research conditions. Original papers and translations remain in `docs/paper/`.
- The [tools index](tools/README.md) covers `build/`, `bench/`, `test/`, `gen/`, `disasm/`, `log_parser/`, `ecm_prob/` and more. NPU microbenchmarks are under [RyzenAI](RyzenAI/README_ADDSUB.md).

Original experiment logs, saves and profiles belong in ignored `data/experiments/`, summaries in `data/benchmarks/` and generated figures in `data/figures/`. Evidence paths from an older commit do not substitute for current local raw data. External sources belong in `.refactor/`. Current project documents, fixed research references and temporary development drafts are maintained under the [repository rules](AGENTS.md).

<a id="license"></a>
## Dependencies and license

The project builds on [GMP-ECM](https://gitlab.inria.fr/zimmerma/ecm), uses CGBN for CUDA multi-precision arithmetic and Dear ImGui for the GUI. Builds need the relevant CUDA/OpenCL, C++ toolchain, GMP and other components. Preserve each dependency's copyright and license when distributing binaries. See [LICENSE](LICENSE) for the project license.
