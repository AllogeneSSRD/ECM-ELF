# How gpuowl/PRPLL implements its transforms on CUDA, and how the built-in paths differ

Read-only analysis of `D:\code\MPA-OpenCl\.refactor\gpuowl` (PRPLL, a gpuowl fork, HEAD `287b736`). Citations are `path:line` inside that tree, quotes are verbatim and short.
  Nothing was built or run. Sibling: `docs/DEV_GPUOWL_NTT_NOTES.md`. Purpose: which of these paths, if any, a stage-2 GPU engine should imitate.

---

## 1. The CUDA path itself: the `.cl` kernels *are* the CUDA kernels

* **No separate CUDA implementation.** `src/cl/*.cl` is the only kernel source; on CUDA it is compiled at run time by **NVRTC** and driven by OpenCL API functions re-implemented
  over the CUDA *driver* API — `src/cuda/clwrap_cuda.cpp:1-3`: *"This replaces clwrap.cpp when building with the native CUDA backend, mapping all cl* calls to cu* equivalents"*.
  The `.cl` text is not read from disk at run time: `genbundle.sh:8-38` (driven by `Makefile:113-114`) embeds every `src/cuda/*.cuh` and `src/cl/*.cl` into `src/bundle.cpp` as
  raw string literals, and `src/KernelCompiler.cpp:181-192,302-317` hands them to `clCompileProgram` as OpenCL "headers".
* **The translation is a text preprocessor plus a macro/type shim, not a compiler front end.** `cudawrap.cpp:221` `preprocessOpenCL()` strips `#pragma OPENCL` (`:226-234`),
  rewrites `#define KERNEL(x)` into `extern "C" __global__ void __launch_bounds__(x)` (`:239-250`), deletes base.cl's OpenCL `typedef`s (`:255-273`), turns
  `__attribute__((reqd_work_group_size(N,1,1)))` into `__launch_bounds__(N)` and drops `overloadable` (`:289-314`), rewrites `(ulong2)(a,b)` → `make_ulong2(a,b)` (`:316-358`),
  converts `local TYPE NAME[` → `__shared__ TYPE NAME[` while deleting `local` in parameter lists (`:360-401`), changes the asm constraint `"n"` to `"r"` (`:403-415`, NVRTC wants
  real constants) and maps `.a[0]`/`.a[1]` → `.a.x`/`.a.y` (`:275-287`). The shim, `src/cuda/opencl_compat.cuh`, is injected into every kernel (`clwrap_cuda.cpp:415-418`) and
  supplies: `#define __kernel extern "C" __global__` (`:16`), `__local`/`local` → nothing (`:23-24`), `__constant` → `const` (`:28`, CUDA `__constant__` cannot be a kernel
  parameter), `restrict` → `__restrict__` (`:32`), `get_global_id(d) ((unsigned)(blockIdx.x*blockDim.x+threadIdx.x))`, `get_local_id(d) threadIdx.x` (`:35-41`), `barrier(flags)`
  → unconditional `__syncthreads()` (`:47-52`), `as_uint2`/`as_double`/`as_ulong2` reinterprets (`:183-247`), vector `+ - * fma mul_hi` (`:115-274`), `atomic_max/add/cmpxchg`
  (`:276-279`), `__asm` → `asm` (`:307`), `sub_group_broadcast` → `__shfl_sync` (`:310`), and forced `NVIDIAGPU 1` + `HAS_PTX 1200` (`:326-331`).
* **Flags and caching** (`clwrap_cuda.cpp:293-393`): `--gpu-architecture` is the device's own `sm_XY` (CUBIN, no JIT) or, for a GPU this NVRTC no longer knows, the newest
  `compute_XY` below it which the driver JITs (`:296-303`). Always `-default-device -std=c++17 -w --fmad=true`; `-cl-std`/`-cl-finite-math-only` are dropped, and `--fmad` is
  called the safe subset — *"no flush-to-zero, no reduced-precision division/sqrt"* — because *"every butterfly is multiply-add pairs"* (`:365-370`). `--restrict` is deliberately
  **off**: *"tested but causes GPU read errors — some PRPLL kernels use in-place operations where in/out buffers alias"* (`:372-374`). A requested `--maxrregcount` forces the PTX
  path, because NVRTC's own ptxas ignores the cap in the CUBIN it emits (`:460-478`). Cached binaries are PTX + CUBIN (`:216-234`, `:236-250`), the PTX kept because `clCreateKernel` reads the work-group size and the PDL wait from it (`:664-696`).
* **What the shim does NOT emulate** (hard limits, not slow paths):
1. **Only dimension 0.** `get_global_id(d)`/`get_local_id(d)` ignore `d` (`opencl_compat.cuh:35-41`), although `workDim == 2` does make a 2-D grid (`clwrap_cuda.cpp:872-884`,
  `src/clwrap.cpp:451-462`). No *live* kernel reads dimension 1 — the only `get_group_id(1)` uses are commented out (`tailmul.cl:24`, `tailsquare.cl:24`) — though the host says the Y coordinate bumps the line number by `WIDTH` (`src/Gpu.cpp:1397-1398`).
2. **No dynamic local memory.** Launches always use `sharedMemBytes = 0` (`:862,869`), and a `local` argument (NULL `cl_mem`) becomes a *null global pointer* (`:760-765`): only the static `local TYPE NAME[` rewrite works.
3. **No FP64 capability query.** `clGetDeviceInfo` has no `CL_DEVICE_DOUBLE_FP_CONFIG` case (`:1116-1252`; unknown keys return `CL_INVALID_VALUE`), so `hasFP64()` swallows the
  error and returns true — *"every CUDA device has FP64"* (`src/clwrap.cpp:187-194`). Hence `NO_FP64` is never defined on CUDA (`src/Gpu.cpp:496`), `T/T2` stay `double/double2`
  (`src/cl/base.cl:266-272`), and the type chooser never excludes FP64 (`src/FFTConfig.cpp:342-343`): FFT64 always compiles, even on cards whose FP64 runs at 1/32–1/64 rate, and is avoided only by timing (`src/tune.cpp:454-484`).
4. **Barriers lose their flags and atomics their order/scope.** `barrier(0)` and `barrier(CLK_LOCAL_MEM_FENCE)` are the same `__syncthreads()` (`:47-52`), making FAST_BARRIER
  (`base.cl:901-908`) meaningless; `memory_order_*`/`memory_scope_device` are `#define`d to 0 and `atomic_load_explicit(p,order,scope)` ignores both (`:285-303`); `atomic_max` is a 32-bit *unsigned* compare (`:277`).

5. **Nothing else of the OpenCL library**: no images/samplers/half/generic address space/`vload`, `clSVMAlloc` is an unused `cuMemAlloc` wrapper (`:1332-1350`), `globalOffset` is
  ignored (`:873`), and a global size that is not a multiple of the group size launches unmasked out-of-range items (`:879-884`; only an `assert` guards it, `src/Kernel.cpp:38`).

* **The `.cl` tree is co-designed with this backend**, which is why so little emulation suffices. `HAS_PTX 1200` selects *inline PTX* inside nominally portable OpenCL:
  `bar.warp.sync` (`base.cl:895-898`), `bar.sync N, count` for sub-group barriers (`:930-931`), and `griddepcontrol.launch_dependents/.wait` for programmatic dependent launch (`:1005-1015`; `-use PDL=1` on sm_90+, listed CUDA-only in `src/Gpu.cpp:264-272`, and the shim detects `griddepcontrol.wait` in the PTX to launch with
  `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`, `clwrap_cuda.cpp:687-696,840-870`). `#if CUDA_BACKEND` also disables the NVIDIA `__constant`-cache weight workaround
  (`carryfused.cl:166-172`) and the register-`bar.sync` path (`base.cl:883-890`).
* **Consequences.** Sources are compiled on the user's machine at first run, needing a matching toolkit/driver (CUDA 13's NVRTC dropped `sm_50..sm_72`,
  `clwrap_cuda.cpp:334-350`); anything the preprocessor does not know fails as an NVRTC error with the source dumped to `prpll_fail_N.cu` (`:438-457`); the group size is the
  kernel's declared work-group size and the grid is `ceil(global/local)`; and "does this card do doubles well?" is answered empirically, never by a query.

**Licence (one paragraph).** `LICENSE:1-3` is *"GNU GENERAL PUBLIC LICENSE Version 3"*; `README.md:47` states the project is *"licensed under the **GNU General Public License
  v3.0**"*. One correction to this document's premise: **this repo's own root `LICENSE` is GPL-3.0 as well**, differing from gpuowl's only in the FSF URL and the `<>` boilerplate
  placeholders. The BSD-style text this project leans on belongs to **gwnum** (`gwnum/readme.txt:86-110`, quoted in `docs/DEV_GWNUM_FEASIBILITY.md:86-100`), a separate component.
  So copying gpuowl/PRPLL source into MPA-OpenCl is GPLv3 → GPLv3 compatible: verbatim kernels *may* be copied if attribution and the GPL notice are kept and the combined work
  stays GPLv3. What may **not** be done is to relicense those lines permissively, or to ship them inside a component distributed under the BSD-style gwnum terms — if this repo
  were ever relicensed permissively, every gpuowl-derived line would have to be removed or reimplemented (algorithms and formulas are not copyrightable; kernel source is).

---

## 2. IBDWT: what it is here, and why it exists

* **The code's own definition**: input words are scaled by a fractional power of two before the transform and by the inverse afterwards — *"Weight is 2^[ceil(qj / n) - qj/n]
  where j is the word index, q is the Mersenne exponent, and n is the number of words."* (`src/cl/fftp.cl:446-448`, `carryfused.cl:861-863`). `N = NWORDS = ND*2` with
  `ND = WIDTH*BIG_HEIGHT` (`base.cl:239-242`); per-word bit length `bitlen(N,E,k) = E/N + isBigWord(N,E,k)`, with `extra(N,E,k) = step(N,E)*k % N` and `step(N,E) = N - E%N`
  (`src/state.h:14-17`), so words alternate between `⌊E/N⌋` and `⌈E/N⌉` bits.
* **Why it exists: zero padding becomes unnecessary.** The transform length is exactly `ND = N/2` complex elements over exactly `N` words — the exponent's word count, no guard
  region — because the cyclic wrap *is* the reduction mod `2^E − 1` that a Mersenne test needs (`2^E ≡ 1`), and the fractional weights keep a word grid that does not divide `E`
  exact. The wrap shows in the host pack/unpack: `compactBits()` carries across word boundaries (`src/state.cpp:22-48`) and `expandBits()` ends with
  `data[0] += u32(bucket.bits); // carry wrap-around.` (`:105`) — that is the folding of the top bits.
* **How the weights are applied.** FP64/FP32: real multiplies, with tables stored as `weight − 1` so one FMA does it — `fancyMul(a,b) = fma(a, b, a)` (`math.cl:426-427`), used as
  `T base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y))` (`fftp.cl:24-30`). Halving is an exponent-bit flip: *"we use inverse weights between 1.0 and
  2.0 because it allows us to implement this routine with a single OR instruction on the exponent"* (`weight.cl:77-100`); the 8 intra-group steps are `2^(k/8) − 1` /
  `2^(−k/8) − 1` (`:35-63`) selected by `weightStepIndex(i) = i*STEP % NW*(8/NW)` (`:29-30`). NTT: because `2` is a 61st (31st) root of unity in `Z61` (`Z31`), weights become
  **cyclic shifts** — `shr(a,k) = (a>>k) + ((a<<(61-k)) & M61)`, `shl(a,k) = shr(a,61-k)` (`math.cl:1083-1088`), `adjust_m61_weight_shift(w) = optional_mod(w, 61)`
  (`weight.cl:172-180`). Setup: `m61_log2_root_two = ((1ULL << 60)/NWORDS) % 61` and `m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m61_log2_root_two % 61`
  (`fftp.cl:449-454`); per-element shifts *and* the big/little-word flags advance together through one packed 64-bit counter (`combo_step`/`combo_bigstep`, `:465-494`). The
  comments calling it "the 60th root" (`fftp.cl:447`, `carryfused.cl:862`) are a typo — every reduction is mod 61.
* **Where the tables are built — correction to the expected answer.** The FP weight tables are built by `Gpu::genWeights()` (`src/Gpu.cpp:91-168`, formulas at `:56-70`), uploaded
  as `bufWeights`/`bufConstWeights` (`:1091-1092`) and bound as fixed kernel arguments (`:1184-1190`); `Gpu.cpp:589-595` compiles `FRAC_BPW_HI/LO = (E % N)/N·2^64` into every
  kernel (note `bpw--; // bpw must not be an exact value`). They are **not** in `TrigBufCache.cpp`, whose only related content is the NTT root constants
  (`_h_0`/`_h_1`/`_h_order`). `src/cl/weight.cl` holds only the 8-entry step tables and the shift adjustments. Consumers: `fftP` (weights on the way in, `src/cl/fftp.cl`), `fftw`
  (final width pass, `src/cl/fftw.cl`), and `carry`/`carryFused` (inverse weights, `carry.cl:29-42`, `carryfused.cl:890-905`). For NTTs the transform's own scale factor is folded
  into the same shift: `weight_shift += log2_NWORDS + 1` *"for the fact that NTT returns results multiplied by 2*NWORDS"* (`carryfused.cl:881-886`).
* **Cost and preconditions.** Cost is a few shifts per word (NTT) or two multiplies in and two out per word (FP) — cheap, and amortised against the zero padding it replaces.
  Preconditions: (i) the modulus must be `2^E − 1`, since the wrap is the reduction and `EXP` enters the weight exponent (`weight.cl:3`, `fftp.cl:453`); (ii) the fractional part
  must be carried across word boundaries, hence `isBigWord`/`bitlen` and `state.cpp:105`; (iii) the length must be a power of two for the twiddle machinery
  (`root_one(n) = h^(2^62/n)`, `TrigBufCache.cpp:824`), enforced for NTTs by `// Reject non-power-of-two NTTs` (`src/FFTConfig.cpp:96`).
* **Is IBDWT useful for a non-Mersenne modulus, e.g. a stage-2 polynomial product? No.** Its whole gain is that the wrap-around *is* the reduction you already wanted; a
  polynomial product wants a linear (or deliberately negacyclic) convolution, so you zero-pad to ≥ `2L−1` (or apply a twist) and the fractional-weight bookkeeping buys nothing —
  it also needs `mod 2^E − 1`, so a general stage-2 modulus has no `E` to feed `weight.cl:3`. What is transferable is not the weights but the discipline: per-word bit lengths
  that vary, one integer counter producing both the shift/twiddle stream and the big/little flags, and the inverse weight folded into the carry kernel rather than a separate
  pass.

---

## 3. FFT64 (floating point)

* **Element type and layout**: `typedef double T; typedef double2 T2;` (`base.cl:267-268`); one `T2` holds two words, so the buffer is `N/2` complex elements = `8N` bytes
  (`src/Gpu.h:390`: `FP64_DATA_SIZE = PAD_ADJUST(W*M*H*2,…)`, unit `sizeof(double)`). Host I/O words are 4-byte (`WordSize == 4` → `Word2 = int2`, `base.cl:315-324`;
  type→`WordSize` at `src/FFTConfig.cpp:297-305`).
* **Radix / middle / width structure**: a 3-D decomposition of a length-`ND` complex transform, `ND = WIDTH*MIDDLE*SMALL_HEIGHT`, `N = 2ND`. Width and height passes use `NW`/`NH`
  ∈ {4,8} (`src/FFTConfig.h:44-46`), one work-group of `G_W = WIDTH/NW` threads holding `NW` elements each, looping
  `for (u32 s = 1; s < WG; s *= RADIX) { fft_RADIX(u); tabMul(...); shufl(...); }` plus a final `fft_RADIX(u)` (`base.cl:1054-1060`; `WG`/`RADIX` are `G_W`/`NW`,
  `src/cl/fftwidth.cl:4-14`). `MIDDLE` is one butterfly of size 2/4/8/16 per thread in the middle kernels (`fft-middle.cl`, `fftmiddlein/out`); twiddles are indexed
  `trig[(i-1)*WG + (me & ~(f-1))]` (`base.cl:1915-1933`).
* **Trig comes from two independent mechanisms.** (a) Device tables generated per shape in `TrigBufCache.cpp` (`genSmallTrigFP64:141`, combo/tail `:242`, middle `:283`, dispatch
  `:931-1011`). (b) An 8-term polynomial evaluated in-kernel, `T2 reducedCosSin(int k, double cosBase)` (`src/cl/trig.cl:7-41`), driven by `TRIG_SCALE/TRIG_SIN/TRIG_COS` compiled
  in from `trigCoefs(fft.shape.size()/4)` (`src/Gpu.cpp:514-517`), with the minimax tables `COS[7]`/`SIN[7]` in `src/Trig.cpp:31-49`; `trigCoefs` asserts the shape factorises as
  `mid·2^twos` with `mid ≤ 15 || mid = 625k` (`:77-91`), and `slowTrig_N` folds `k` into `[0, n/8]` with sign/swap fixups (`trig.cl:44-71`). Which is used is `-use TAIL_TRIGS`,
  default 2 = compute, no memory accesses (`TrigBufCache.cpp:1065`).
* **Roundoff control.** A carried element becomes an integer only after measuring its distance to a rounding boundary:
  `float roundoff = fabs((float) fma(u, invWeight, RNDVALCarry - d)); *maxROE = max(*maxROE, roundoff);` (`carryutil.cl:208-227`); 0.5 means a wrong integer. There is **no
  `gw_passes_safety_margin` in this tree**; the equivalents are the `fftbpw.h` tables, the `fft.maxExp() < E` warning (`src/Gpu.cpp:1130-1132`), the `bitsPerWord < minBpw()`
  throw *"FFT size too large"* (`:1135-1138`; `minBpw() = 3.0`, `src/FFTConfig.h:48`), and the sloppy-carry cut-off `MAXBPW = maxBpw()*100` (`src/Gpu.cpp:505`) with
  `#define SLOPPY_MAXBPW (MAXBPW - 110)` — *"We only allow sloppy results when not near the maximum bits-per-word"* (`carryutil.cl:695-702`). `useLongCarry` is forced below 10.0
  bpw (`src/Gpu.cpp:1140`) and 32-bit carries are illegal at `EXP/NWORDS >= 19` (`carryutil.cl:807-819`).
* **Achievable bpw** (`src/fftbpw.h`; the 6 values per key are variants 000,101,202,010,111,212):

| key | quoted bpw | bits per byte (8 B/word) |
|---|---|---|
| `256:2:256` (`:2`) | `19.204 19.547 19.636 19.204 19.547 19.636` | 2.40 – 2.45 |
| `512:8:512` (`:38`) | `18.256 18.280 18.314 18.319 18.369 18.444` | 2.28 – 2.31 |
| `4K:16:1K` (`:94`) | `16.744 16.887 16.966 16.921 17.048 17.208` | 2.09 – 2.15 |

A variant is worth ~0.35 bpw (≈1.8 %) at a fixed shape. Butterfly cost is tabulated by the repo as `2·FMA + ADD`: radix-4 `0 FMAs + 16 ADDs` (`fft4.cl:15`), radix-8
  `4 MUL + 52 ADD` (`fft8.cl:25`; cost 16 vs 60 in `FFT.md:12,16`).

---

## 4. NTT61, NTT31 and the hybrids

All share the kernels with FFT64, switched by `FFT_TYPE`, `NTT_GF31`, `NTT_GF61`, `FFT_FP32`, `FFT_FP64`, `WordSize` (`src/FFTConfig.cpp:297-305`, `src/Gpu.cpp:931-961`). Types
  (`base.cl:273-278`): `Z31=uint`, `GF31=uint2` (8 B), `Z61=ulong`, `GF61=ulong2` (16 B), `F2=float2` (8 B), `T2=double2` (16 B); one element holds **two** words.

| `FFT_TYPES` | computes | elements (field) | B/elem | `WordSize` | I/O B/word |
|---|---|---|---|---|---|
| `FFT64` 0 | FFT | `double2` | 16 | 4 | 8 |
| `FFT32` 53 | FFT | `float2` | 8 | 4 | 4 |
| `FFT31` 52 | NTT | `GF(M31²)` `uint2` | 8 | 4 | 4 |
| `FFT61` 3 | NTT | `GF(M61²)` `ulong2` | 16 | 4 | 8 |
| `FFT3161` 1 | two NTTs + CRT | `GF(M31²)`+`GF(M61²)` | 8+16 | 8 | 12 |
| `FFT3261` 2 | FFT + NTT | `float2`+`GF(M61²)` | 8+16 | 8 | 12 |
| `FFT6431` 51 | FFT + NTT | `double2`+`GF(M31²)` | 16+8 | 8 | 12 |
| `FFT3231` 50 | FFT + NTT | `float2`+`GF(M31²)` | 8+8 | 4 | 8 |
| `FFT323161` 4 | FFT + two NTTs + CRT | `float2`+`GF(M31²)`+`GF(M61²)` | 8+8+16 | 8 | 16 |

Bytes/word follows from `src/Gpu.h:390-395` (unit `sizeof(double)`; each component contributes `PAD_ADJUST(W*M*H*2,…)` scaled by `sizeof(element)/sizeof(double)`: FP64 `8N`, FP32
  `4N`, GF31 `4N`, GF61 `8N`). `WordSize` is *host I/O* width, not element width — FFT61 stores 16 B per 2 words while reading/writing 4-byte `int2` words (`FFTConfig.cpp:300`,
  `base.cl:316-324`).

* **M31 and M61 are recombined by CRT at carry time, in the same pass.** `weightAndCarryOne(Z31,Z61,…)` applies both inverse weights as shifts, then (`carryutil.cl:446-478`):
```c
u32 n31 = get_Z31(u31);
u61 += make_u64(hi32(M61), lo32(M61) - n31);   // u61 - u31
u61 += shl(u61, 31);                           // u61 + (u61 << 31)
i64 n61 = get_balanced_Z61(modM61(u61));
i96 value = make_i96(n61 >> 1, ((u32)n61 << 31) | n31);   // n61*M31 + n31
```
a 92-bit `n61·M31 + n31` built with multiplies folded into shifts (comments `:452-459` credit Gallot's `mersenne2`). `FFT323161` adds the FP32 component to choose the multiple of
  `M31·M61`: *"Use FP32 data to calculate how many multiples of M31*M61 need to be added to n3161"* (`:487-524`, 128-bit reassembly). Both NTTs run over one buffer, GF61 data at
  offset `DISTGF61` (`src/Gpu.cpp:565-575`), each with its own weight-shift stream.
* **Length limit — the `v2(p−1)` argument.** For `p = M61 = 2^61−1`, `p−1 = 2·(2^60−1)` has `v2 = 1`, so `F(M61)` contains no power-of-two root of unity beyond `−1`: `F_p` alone
  cannot host a length-`2^k` transform. The code works in the quadratic extension — `TrigBufCache.cpp:791-800`: *"GF((2^61 - 1)^2): the prime field of order p^2"* — with
  `static const uint64_t _h_order = uint64_t(1) << 62;` and `static GF61 root_one(const size_t n) { return GF61(Z61(_h_0), Z61(_h_1)).pow(_h_order / n); }`,
  `_h_0 = 264036120304204`, `_h_1 = 4677669021635377` (`:796-800,824`), matching `v2(p²−1) = v2(2^62·(2^60−1)) = 62`. The M31 analogue is `_h_order = 1 << 32`, `_h_0 = 7735`,
  `_h_1 = 748621` (`:618-620,644`) for `v2(31²−1) = 32`. The tree states the *orders* but never writes this derivation (it is mine; the constants corroborate it). Operationally
  every NTT type needs power-of-two `MIDDLE` (`FFTConfig.cpp:96`) and `root_one(n)` divides `2^62` by `n`; reachable lengths are `ND ≤ 2^26` (`WIDTH ∈ {256,512,1024,4096}`,
  `fftwidth.cl:20-22`; `HEIGHT ∈ {256,512,1024}`, `MIDDLE ∈ 2..16`, `FFTConfig.cpp:92-104`), inside `2^62` (M61) and also inside `2^32` (M31).
* **Observed bpw and payload per byte** (prefix = the FFT-type number, `FFTConfig.h:51`):

| type | `…:256:2:256` | bpw | bits/byte | `…:4K:16:1K` | bpw | bits/byte |
|---|---|---|---|---|---|---|
| FFT64 | `256:2:256` (`:2`) | 19.204 – 19.636 | **2.40 – 2.45** | `4K:16:1K` (`:94`) | 16.744 – 17.208 | 2.09 – 2.15 |
| FFT3161 | `1:256:2:256` (`:96`) | 40.54 | **3.38** | `1:4K:16:1K` (`:113`) | 37.12 | 3.09 |
| FFT3261 | `2:256:2:256` (`:115`) | 34.53 | 2.88 | `2:4K:16:1K` (`:132`) | 28.54 | 2.38 |
| FFT61 | `3:256:2:256` (`:134`) | 25.02 | 3.13 | `3:4K:16:1K` (`:151`) | 22.42 | 2.80 |
| FFT323161 | `4:256:2:256` (`:153`) | 50.01 | 3.13 | `4:4K:16:1K` (`:170`) | 44.12 | 2.76 |
| FFT3231 | `50:256:2:256` (`:172`) | 19.57 | 2.45 | `50:4K:16:1K` (`:189`) | 7.05 | 0.88 |
| FFT6431 | `51:256:2:256` (`:191`) | 35.27 | 2.94 | `51:4K:16:1K` (`:208`) | 32.25 | 2.69 |
| FFT31, FFT32 | — | **no entry** | — | — | **no entry** | — |

Densest per byte: **FFT3161 (M31+M61) at ≈3.38 bits/byte**, then FFT61 and FFT323161 at ≈3.13; FFT64 is worst (2.40) yet often fastest in wall-clock, because the metric ignores
  arithmetic. FFT31/FFT32 are uncalibrated: absent from the tables *and* from `allShapes()` (`FFTConfig.cpp:92`), and on an unmapped shape the constructor falls back to
  `bpw = {18.1f,…}` with *"ERROR: BPW info for %s not found, using default of 18.1"* (`:165-169`) — an FP64-derived default that for FFT31's 4 B/word would imply 4.5 bits/byte,
  the best of all if it were real. Nothing measures it.
* **The tables are empirical, not a proof.** The M61 section records the one documented hard failure: *"LL of 100028317 failed (ROEmax=0.294, ROEavg=0.247). Lowering bpw from
  23.94 to 23.84."* (`fftbpw.h:141`, row `3:1K:8:256`). For NTT types "ROE" is not floating-point error but proximity of the reconstructed coefficient to the `M61/2` (resp.
  `M31/2`) wrap boundary — `u32 roundoff = (u32) abs((i32) hi32(value));` with *"calculate roundoff error as proximity to M61/2. 28 bits of accuracy should be sufficient"*
  (`carryutil.cl:322-326`), normalised later by `(float) roundMax / (float)(M61 >> 32)` (`carryfused.cl:958`; M31: `/(float)M31`, `:723`). A word therefore carries `bpw` bits
  exactly only while the coefficient sums stay inside the modulus window with the statistical margin the table encodes; when they did not, the entry was lowered.

---

## 5. The differences that matter

| | FFT64 | FFT61 | FFT3161 (M31+M61) | hybrids (3261/6431/3231/323161) |
|---|---|---|---|---|
| correctness | approximate: FP64 + ROE criterion (`carryutil.cl:208-227`) | **exact mod M61** within its window | **exact**, CRT `n61·M31+n31` (`:446-478`) | exact, window widened by the FP part (`:487-524`) |
| carry | 32-bit allowed < 19 bpw (`FFTConfig.cpp:329`) | 64-bit mandatory | 64-bit mandatory | 64-bit mandatory (`carryutil.cl:807-819`) |
| length | `2^26` by shape space only | `2^62` (`TrigBufCache.cpp:800`) | `2^32` (`:620`) | that of the NTT member |
| radix-8 butterfly | `4 MUL + 52 ADD` FP64 (`fft8.cl:25`) | GF61 `cmul` = 3 wide multiplies (`math.cl:1241-1246`), lazy reductions (`fft8.cl:144-151`) | both, same kernel | FP + NTT mix |
| bytes/word | 8 | 8 | 12 | 8–16 |
| bits/byte @ `256:2:256` | 2.40 | 3.13 | **3.38** | 2.45 – 3.13 |
| best at | GPUs with fast FP64 (`-tune` compares it to M31+M61, thresholds 0.80/1.20, `tune.cpp:454-484`) | densest *single* exact path, least LDS | densest per byte, no FP64 needed | 3231/3261 when FP32 is cheap; 323161 for the largest exponents |

* **Length is where a single prime loses.** `FFT31`'s table collapses as the transform grows (`50:4K:16:1K` → 7.05 bpw, `fftbpw.h:189`): the coefficient window, not the
  arithmetic, binds. `GF(M61²)` has both the deepest 2-adic order (62) and the widest single modulus (61 bits), which is why it is the default NTT that `-tune` measures against
  FP64.
* **Memory traffic is set by the kernel structure, not the element type.** One squaring is `fftP → fftMidIn → tailSquare → fftMidOut → carryFused` (`src/Gpu.cpp:2256-2283`): ~4
  full read+write passes over the transform buffer, against `log2(ND) ≈ 17-26` passes for a naive 1-D Stockham loop. The pointwise product sits *inside* the tail kernel, between
  two half-height passes (`tailsquare.cl:1072-1127`).
* **The tail pairing is the squaring discount, and it does not transfer to a general product.** `pairSq` computes `csqq(a)`, `csq(b)` and the cross term `ab` for a Hermitian line
  pair and folds them with a `t_squared` twiddle (`tailsquare.cl:892-991`); line 0 and line H/2 pair with themselves with the extra factor `TAILTGF61` (`:1099-1110`; pairing
  `line2 = line1 ? H - line1 : H/2`, `:1063-1064`). The general-product twin exists — `pairMul`/`tailMul` (`src/cl/tailmul.cl:50-86`, driven by `Gpu::mul`,
  `src/Gpu.cpp:2003-2016`) — and is the *harder* kernel: `onePairMul` needs a real `cfma`/`cmul` per output, whereas `pairSq` reuses `a²`, `b²` and `ab`.
* **Carry and reconstruction are the least reusable part**: a chain over the whole number with `nBits = EXP/NWORDS + (isBigWord?1:0)` (`carryutil.cl:547-579`, `state.h:16-17`),
  carries forwarded between workgroups through a global shuttle with `write_mem_fence` + `atomic_store(ready)` + spin (`carryfused.cl:915-1040,1726-1755`), and LL's initial `−2`
  (`:904-907`) — all specialised to `X² mod (2^p−1)`.

### What a stage-2 engine should take from this

* **(a) Field for Kronecker-packed products with ~20-30 bit digits: not M61 alone.** `FFT61` gives ~22–25 bits/word at practical shapes (`fftbpw.h:134,151`) and those numbers are
  statistical, with a recorded failure at 23.94 bpw (`:141`). An exact product of two length-`L` polynomials with `d`-bit digits must cover the worst case `log2 P > log2 L + 2d`
  (derived here from the moduli, not quoted). With `P = M61` (~61 bits) that gives `d < (61 − log2 L)/2`: `d ≈ 24` at `L = 2^12`, `d ≈ 22` at `L = 2^16`, and no 30-bit option.
  With `P = M31·M61` (~92 bits, CRT'd at `carryutil.cl:446-478`) any `d ≤ 30` works up to `L = 2^32`. So: **use the M31+M61 pair (FFT3161's field pair) for 20-30 bit digits**;
  M61 alone only for ≤ ~24-bit digits at short lengths; **never M31 alone** (its 7.05 bpw at `4K:16:1K` is the warning).
* **(b) Imitate the decomposition, not the launch topology**: a width×middle×height split where only the transposes touch global memory (`fftMiddleIn`/`fftMiddleOut`); a
  multi-radix butterfly set with `×i`/`×√½` done as swap/negate/delayed scaling (`fft4.cl:16-61`, `fft8.cl:16-23`); twiddles either tabulated or from an 8-term polynomial
  (`trig.cl:7-41`); and, for a general product, the `tailMul` shape (`tailmul.cl:50-86`) so the pointwise multiply happens inside the transform. Do not copy `pairSq`'s squaring
  shortcut for a two-polynomial product.
* **(c) Do not take**: IBDWT (needs `mod 2^E−1` and a wrap you do not want — `weight.cl:3`, `state.cpp:105`); the `n61·M31+n31` CRT carry (`carryutil.cl:446-478`, whose
  multiply-by-shifts and lazy `[0, 2M61+ε]` ranges are hand derived for one modulus pair and one carry chain, `state.h:16-17`); the `−2` LL initial carry
  (`carryfused.cl:904-907`); and the sloppy-carry cut-off (`carryutil.cl:695-702`), which trades worst-case correctness for speed.

---

## 6. Open questions (not determined from the source)

* **Why the NTT bpw tables are as high as they are.** The worst-case coefficient of a length-`ND` cyclic convolution is `~ND·2^(2·bpw)`; at `3:256:2:256` that is
  `2·25.02 + 17 ≈ 67` bits against a 61-bit `M61` window, yet the table claims ROE ≈ 0.35. The empirical framing and the recorded failure (`fftbpw.h:141`) imply a statistical
  criterion, but no derivation, distribution model or safety-factor rationale is written anywhere (searched `src/`, `src/cl/*.cl`, `FFT.md`, `README.md`, `z.txt`). The rule for
  "how many bits `M61` can carry at length `L`" is therefore **not established**; §5(a) is my conservative reconstruction, not the tree's rule.
* **The Y dimension of the 2-D tail launch.** `Gpu.cpp:1398` says the Y coordinate bumps the line number by `WIDTH` and the host passes `kernelsToExecuteY = (MIDDLE+1)/2`
  (`:1399-1411`), but the kernel-side uses are commented out (`tailmul.cl:24`, `tailsquare.cl:24`) and live code derives lines from `get_group_id(0)` alone (`tailmul.cl:13-43`).
  Whether those Y blocks duplicate work (they write identical values, so not a correctness bug) or are consumed elsewhere, I could not determine; the path needs
  `-use L2_STRIPING=n` (`base.cl:232-234` defaults it to 0).
* **How the host interprets NTT ROE samples**: `bufROE` is `Buffer<float>` (`src/Gpu.h:216`) while NTT kernels write a `u32` proximity count, sometimes normalised
  (`carryfused.cl:723,958`) and sometimes raw through an implicit `u32 → float` conversion (`carryfused.cl:259,1764`, `carry.cl:51`). Which sites are live I could not establish.
* **No measured speed comparison exists in the tree**: `z.txt` has no NTT rows and there is no baked-in FP64-vs-M31+M61 timing — it is measured at run time into `tune.txt`
  (`tune.cpp:454-484`). Any "which path is faster" claim must be measured.
* **`bits/byte` here is payload, not allocation**: `PAD_ADJUST` (`src/Gpu.h:376-389`) can inflate the real buffer by up to 1.6× depending on `INPLACE`/`PAD`. And nothing was
  built or run, so every constant above is quoted, not reproduced.
