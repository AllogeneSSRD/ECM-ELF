# PRPLL / gpuowl integer NTT — implementable specification (for a CUDA port)

Source analysed read-only: `D:\code\MPA-OpenCl\.refactor\gpuowl` (PRPLL, a fork of gpuowl). Citations are `file:line`
inside that tree; quotes are verbatim and short. Nothing was built or run — §7 lists what stayed unresolved.

## 0. Two facts to know first

* **There is no `ntt*.cl`.** The NTT lives inside the shared FFT kernels, switched by macros: `NTT_GF31`/`NTT_GF61`/`FFT_FP64`/`FFT_FP32` come per kernel from `Gpu::kernelDefines()` (`src/Gpu.cpp:931-961`), with `FFT_TYPE`, `WordSize`, `TAILTGF31`, `TAILTGF61` (`src/Gpu.cpp:519-528`) and `WIDTH`, `SMALL_HEIGHT`, `MIDDLE`, `CARRY_LEN`, `NW`, `NH` (`src/Gpu.cpp:469-480`); type→flags table `src/FFTConfig.cpp:297-305`. Kernel bodies are `#if NTT_GF31` / `#if NTT_GF61` blocks (e.g. `src/cl/fftbase.cl:1873`, `src/cl/carryfused.cl:815`); the radix butterflies are shared overloads (`fft4`, `fft8` at `src/cl/fft4.cl:205-254`, `src/cl/fft8.cl:116-158`).
* **A CUDA backend already exists in this tree**: `README.md:52`, `README.txt` ("`make CUDA=1`"), `src/cuda/opencl_compat.cuh:1-6` ("Allows .cl kernel files to compile under NVRTC with minimal changes"), `src/cuda/clwrap_cuda.cpp`. These kernels are already compiled for CUDA through that shim; a separate CUDA NTT is only needed to leave the harness.

## 1. Field and modulus

* Element types (`src/cl/base.cl:251-254`, `:266-278`):
  ```c
  typedef ulong Z61;      // A value calculated mod M61.  For a GF(M61^2) NTT.
  typedef ulong2 GF61;    // A complex value using two Z61s.  For a GF(M61^2) NTT.
  typedef uint Z31;       // A value calculated mod M31.  For a GF(M31^2) NTT.
  typedef uint2 GF31;     // A complex value using two Z31s.  For a GF(M31^2) NTT.
  ```
  `Z61`/`Z31` are the prime fields F_p (8 B / 4 B); `GF61`/`GF31` (16 B / 8 B) are F_{p²} = F_p[i]/(i²+1), i.e. the pair is treated exactly as a complex number (`mul`/`sqr` below).
* Moduli: `#define M61 ((((Z61) 1) << 61) - 1)` (`src/cl/math.cl:194`) = 2305843009213693951 = 2^61−1; `#define M31 ((((Z31) 1) << 31) - 1)` (`src/cl/math.cl:709`) = 2147483647 = 2^31−1.
* Host-side reference implementation (clearest statement of the algebra), copied from Yves Gallot's `mersenne2` — `src/TrigBufCache.cpp:747-837`: `Z61::_mul` `:768-774`, `GF61::sqr`/`GF61::mul` `:813-814`, and:
  ```c
  // Primitive root of order 2^62 which is a root of (0, 1). ... PRPLL FFTs use this root.  Thanks, Yves!
  static const uint64_t _h_0 = 264036120304204ull, _h_1 = 4677669021635377ull;
  static const uint64_t _h_order = uint64_t(1) << 62;
  static GF61 root_one(const size_t n) { return GF61(Z61(_h_0), Z61(_h_1)).pow(_h_order / n); }
  ```
  (`src/TrigBufCache.cpp:796-800`, `:824`). A second candidate root, "root of (0, -1)", is commented out at `:798-799` (using it would flip every twiddle sign). GF31 analog: order 2^32, `_h_0 = 7735u, _h_1 = 748621u` (`src/TrigBufCache.cpp:618-620`, `:644`).
* **Why a quadratic extension.** The source states the *order* used (2^62 for M61, 2^32 for M31) but never writes the `v2(p−1)` argument in words — searched `src/`, all `*.md`, `README.txt`, `z.txt` (zero hits for `NTT`). The argument is consistent with the code: v2(p−1)=1 for M61 and p+1 = 2^61 give v2(p²−1) = 62, exactly `_h_order`, and transform lengths here are powers of two up to ND = 2^26 (`4096:16:1024`); likewise M31 (v2(p²−1)=32). So "GF(p²) is needed because v2(p−1)=1" is confirmed-by-construction, not a quoted derivation.
* Types and word sizes (`src/FFTConfig.cpp:297-305`, enum `src/FFTConfig.h:26`): `FFT64` FP64/4; `FFT3161` GF31+GF61/8; `FFT3261` FP32+GF61/8; `FFT61` GF61 only/4; `FFT323161` FP32+GF31+GF61/8; `FFT3231` FP32+GF31/4; `FFT6431` FP64+GF31/8; `FFT31` GF31 only/4; `FFT32` FP32 only/4.
* **M31 and M61 are combined by CRT at carry time — not two independent results.** Both NTTs run in the *same* kernels over one buffer, GF61 data after GF31 at offset `DISTGF61` (`src/Gpu.cpp:565-575`), each with its own weight-shift stream (`src/cl/carry.cl:546-602`). Fusion in `weightAndCarryOne(Z31, Z61, …)` (`src/cl/carryutil.cl:446-478`):
  ```c
  u61 += make_u64(hi32(M61), lo32(M61) - n31);   // u61 - u31
  u61 += shl(u61, 31);                           // u61 + (u61 << 31)
  i64 n61 = get_balanced_Z61(modM61(u61));
  ...
  u64 vlo = ((u32)n61 << 31) | n31;  i96 value = make_i96(n61 >> 1, vlo);
  value = sub(value, n61);                       // n61 * M31 + n31
  ```
  i.e. a 92-bit `n61*M31 + n31` (comments `:452`, `:457-459`). `FFT323161` additionally uses FP32 data to decide the multiple of M31·M61 (`src/cl/carryutil.cl:487-514`).

## 2. Arithmetic primitives (ready to port)

**M61 reduction — the core equation**, `weakModM61` scalar branch (`src/cl/math.cl:1132-1166`):
```c
u64 lo = u128_lo64(a), hi = u128_hi64(a);
u64 lo61 = lo & M61;                                  // Max value is M61
if (num_bits <= 125) {
   hi = (hi << 3) + (lo >> 61);
   return lo61 + hi;                                  // Caller must insure this does not overflow
} else {
   u64 hi61 = ((hi << 3) + (lo >> 61)) & M61;         // Max value is M61
   return lo61 + hi61 + (hi >> 58);                   // Max value is 2*M61 + epsilon
}
```
Fold: `2^64 ≡ 8 (mod M61)` (`hi << 3`) plus the bit-61 carry `lo >> 61`; when the product can reach the full 128 bits, the top 6 bits (`hi >> 58`) are folded a second time. **Lazy: the result is in [0, 2·M61+ε], not canonical.** `num_bits` is supplied by the caller from a bound on the product. A PTX variant using `shf.r.clamp.b32` is at `:1134-1154`.

* Multiplication: `Z61 mul(Z61 a, Z61 b) { return modM61(weakMul(a, b, 2, 2)); }` (`src/cl/math.cl:1197`); `weakMul` = `mul64` + `weakModM61(ab,125)` (`:1172-1179`). `mul64` is one 64×64→128 (`mul.lo.u64`/`mul.hi.u64`, `:303-308`; measured 2 % faster than the C version, `:300-302`), `mad64` fuses the add (`:359-387`). Roughly 6–8 ALU ops per `mul` on the scalar path.
* Simpler reduction: `Z61 modM61(Z61 a) { return (a & M61) + (a >> 61); }` (`src/cl/math.cl:1067`), also lazy ([0, M61+ε]). Canonical extractors: `get_Z61` `:1063`, `get_balanced_Z61` `:1064`.
* Add/sub/neg (`src/cl/math.cl:1073-1080`), using "add enough M61s to stay positive":
  ```c
  Z61 add(Z61 a, Z61 b) { return modM61(a + b); }
  Z61 sub(Z61 a, Z61 b) { return modM61(a + neg(b, 2)); }   // neg(b,2) = 2*M61 - b
  Z61 neg(Z61 a, const u32 m61_count) { return m61_count * M61 - a; }   // :1070
  ```
  There is no canonical `[0,p)` invariant (`src/cl/math.cl:1058-1061`): *"uses faster, sloppier mod M61 reduction where the end result is in the range 0..M61+epsilon … better off using the quick routines and negative intermediate results"*, and multiplication "must use only positive values as `__int128` multiply is very slow". Hence every `weakMul` takes an `m61_count` (how many M61s are folded in) and the `…q` helpers renormalise: `modM61q(GF61 a, m61_count) { if (m61_count) { a.x += m61_count*M61; a.y += m61_count*M61; } return modM61(a); }` (`src/cl/math.cl:1381`); also `addq/subq/addiq/subiq/X2q/X2qconjb` (`:1361-1375`).
* Multiply by 2^k is a **rotation** (2^61 ≡ 1 ⇒ 2 is a 61st root of unity in Z61), `src/cl/math.cl:1083-1088`: `shr(a,k) = (a >> k) + ((a << (61-k)) & M61)`, `shl(a,k) = shr(a, 61-k)`; faster dedicated `shl30`/`shl31` at `:1091-1130`. *Two comments call it "2 is the 60th root GF61" (`src/cl/fftp.cl:447`, `src/cl/carryfused.cl:862`) while all code reduces shifts mod 61 (`adjust_m61_weight_shift`, `src/cl/weight.cl:176-178`) — flagged in §7.*
* Complex (extension) multiply — **3 wide multiplies** (Karatsuba), `src/cl/math.cl:1241-1246`:
  ```c
  GF61 cmul(GF61 a, GF61 b) {
    u128 k1 = mul64(b.x, a.x + a.y);                            // max value is 2*M61^2+epsilon
    Z61 k1k2 = weakMulAdd(a.x, b.y + neg(b.x, 2), k1, 2, 4);    // max value is 6*M61+epsilon
    Z61 k1k3 = weakMulAdd(a.y, neg(b.y + b.x, 3), k1, 2, 4);    // max value is 6*M61+epsilon
    return U2(modM61(k1k3), modM61(k1k2)); }
  ```
  The textbook 4-multiply version is disabled at `:1248-1254`. Squaring = 2 wide multiplies (`csqq`, `:1214-1220`, from `(a+ib)² = ((a+b)(a−b), 2ab)`). Butterfly shortcuts: `X2_internal`/`X2t4_internal`/`X2conjb_internal`/`X2_mul_t8_internal` (`:1331-1355`), `mul_t4` (mul by i, `:1270`), `mul_t8`/`mul_3t8` (mul by (±2^30,−2^30) as two shifts, `:1273-1278`), `mul_t16` family with hard constants (`:1281-1327`), `csqTrig`/`ccubeTrig` (`:1259`, `:1262`).
* **M31 analog** (the sloppy branch, `src/cl/math.cl:801-1035`, `#elif 1` at `:801`): `Z31 mul(Z31 a, Z31 b) { u64 t = a * (u64) b; return modM31(t, 62); }` (`:868`); `modM31(u64 a, u32 maxbits)` splits into 31-bit limbs (`:815-833`); `add`/`sub` at `:844`/`:847`. GF31 `cmul` is also 3 multiplies (`:971-977`), GF31 `csq` = `mul(add)·mul(sub)` (`:747`); `MODM31` picks among three `modM31` implementations (`:804-813`).

## 3. Transform structure

* **Shape.** Host: `N = shape.size() = width*height*middle*2` = number of big-integer *words* (`src/Gpu.cpp:969`, `src/FFTConfig.h:44`); `hN = N/2` (`src/Gpu.cpp:974`); `BIG_H = SMALL_H*MIDDLE` (`:971-973`); `NW`/`NH` = 4 or 8 (`src/FFTConfig.h:45-46`). `src/cl/base.cl:239-249` defines `ND = WIDTH*BIG_HEIGHT`, `NWORDS = ND*2`, `G_W = WIDTH/NW`, `G_H = SMALL_HEIGHT/NH`. The GPU transform is a **3-D decomposition of a length-ND complex NTT** (ND = `WIDTH*MIDDLE*SMALL_HEIGHT` = `NWORDS/2`), one complex element per *pair* of words: ND elements in a buffer of N doubles (8·N bytes) ⇒ 16 B/element, 8 B/word for GF61.
* **Radix and stages.** RADIX = NW (width) or NH (height) ∈ {4, 8}: `for (u32 s = 1; s < WG; s *= RADIX) { fft_RADIX(u); tabMul(trig,u,s,lowMe); shufl(lds,u,s,numWG,lowMe); } fft_RADIX(u);` (`src/cl/fftbase.cl:2064-2072`) ⇒ log_RADIX(WG)+1 stages (WIDTH=256, NW=4 ⇒ four radix-4 stages). Variants use radix-16 tails `fft8_16a/b` (`src/cl/fft8.cl:317-371`) and a fused shuffle+radix-2 `shufl_and_fft2` (`src/cl/fftbase.cl:2058`). The middle transform is one full butterfly of size MIDDLE ∈ {2,4,8,16} per thread (`fft_MIDDLE` = fft2/4/8/16, `src/cl/fft-middle.cl:838-854`).
* **Per thread.** Width pass `GF61 u[NW]`, height `u[NH]`, middle `u[MIDDLE]` (`src/cl/fftmiddlein.cl:476`, `src/cl/ffthin.cl:146`, `src/cl/fftmiddleout.cl:208`, `src/cl/tailsquare.cl:1061`). Workgroup = `G_W = WIDTH/NW` threads (`src/fftwidth.cl:4`), each holding NW elements ⇒ one workgroup covers WIDTH contiguous elements. Vector types are OpenCL `ulong2`/`uint2`/`double2`/`float2` — never float4.
* **Shared memory per stage.** `LDS_SHUFL_BYTES(numWG) = (WG*RADIX + LDSPAD_COUNT)*SHUFL_BYTES`, `LDS_BYTES(numWG) = numWG*LDS_SHUFL_BYTES(numWG)` (`src/cl/fftbase.cl:11-17`); `SHUFL_BYTES` defaults to 8 (`src/cl/base.cl:160-165`), `LDSPAD_COUNT` = 12 (radix 4) or 7 (`src/cl/fftbase.cl:15`) ⇒ for WIDTH=256, NW=4: (256+12)·8 = 2144 B per workgroup per shuffle buffer; double-wide tail kernels take `LDS_BYTES(2)` (`src/cl/tailsquare.cl:1152`). `shufl` is templated on 4/8/16-byte units (`src/cl/shufl.cl:53-60`); `:45-47` warns about the mixed element sizes of an M31+M61 NTT.
* **In-place?** `INPLACE` defaults to 1 on NVIDIA, 0 otherwise (`src/cl/base.cl:98-106`); it selects the swizzled FFT-data layout (`src/cl/middle.cl:689-708`) and whether a `buf3` scratch is used (`src/Gpu.cpp:1669`, `:1737`). The transform is always a *decomposition with transposes* (`fftMiddleIn`/`fftMiddleOut`, `middleShuffle`, `src/cl/fft-middle.cl:912-975`), never a single 1-D Stockham loop.
* **Kernel chain** (recorded, then replayed in order: `src/Gpu.cpp:1663-1751`, `:1776-1822`). Squaring, `Gpu::square` (`src/Gpu.cpp:2224-2290`):
  1. `fftP` (`src/cl/fftp.cl:426-501`): read words, apply IBDWT weights, `fft_WIDTH`, write carryFused layout.
  2. `fftMidIn` (`src/cl/fftmiddlein.cl:474-522`): `middleMul2`, `fft_MIDDLE`, `middleMul`, transpose.
  3. `tailSquare` (`src/cl/tailsquare.cl:1051-1127`): `fft_HEIGHT1` on the line pair u,v, `pairSq`, `fft_HEIGHT2`.
  4. `fftMidOut` (`src/cl/fftmiddleout.cl:207-259`): `middleMul`, `fft_MIDDLE`, `middleMul2`, transpose back.
  5. Either `fftW` + `carryA`/`carryM` + `carryB`, or fused `carryFused` (`src/Gpu.cpp:2262-2283`).
  A multiplication is `fftP → fftMidIn → tailMul → fftMidOut → fftW → carry` (`src/Gpu.cpp:2003-2016`). `carryFused` "is equivalent to the sequence: fftW, carryA, carryB, fftPremul" using "stairway forwarding" (`src/cl/carryfused.cl:131-133`). Kernel work sizes are work-*item* counts (`src/Gpu.cpp:1030-1063`), divided by the local size in `src/Kernel.cpp:17-67`.
* **Twiddle tables** (host, one `ulong2` each, `src/TrigBufCache.cpp`): `for (line = 1; line < radix; ++line) for (col = 0; col < WG; ++col) tab.push_back(root1GF61(root1size, col * line));` (`:859-865`), where `root1GF61(root1N,k) = root1N.pow(k)`, `root1N = GF61::root_one(N)` (`:829-837`); size-256/radix-8 special layout `:845-856`; tail/height combos `genSmallTrigComboGF61` `:872-903` (`TAIL_TRIGS61`, default read-from-memory `:875`); middle table `genMiddleTrigGF61` `:905-920` (`root_one(smallH*middle)` indexed `k*m`, then `root_one(middle*width)` indexed `k`, then `root_one(width*middle*smallH)` indexed `k`). Buffers are bound as fixed args at `src/Gpu.cpp:1170-1181`, with offsets `DISTWTRIGGF61`/`DISTMTRIGGF61`/`DISTHTRIGGF61` (`src/Gpu.cpp:582-586`). **Index formula inside the butterfly** (`src/cl/fftbase.cl:1915-1933`):
  ```c
  void OVERLOAD tabMul(TrigGF61 trig, GF61 *u, u32 f, u32 me) {
    u32 p = me & ~(f - 1);
    if (!TABMUL_CHAIN61) { for (u32 i = 1; i < RADIX; ++i) u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p])); return; }
    chainMul(u, TFLOAD(&trig[p])); }   // TABMUL_CHAIN61=1: w^i by repeated cmul, :1885-1913
  ```
  i.e. element i at stage factor f uses entry `(i-1)*WG + (me & ~(f-1))` = ω^{(me & ~(f-1))·i}. Continued/condensed tables for the radix-8-then-4 pairs: `tabMul8_4a`/`tabMul8_4b` (`src/cl/fftbase.cl:1936-2032`, `trig += 7*WG`, `trig += 6*WG`).
* **Weight transform** (`src/cl/weight.cl`; applied in `fftp.cl` and `carryfused.cl`). Purpose: it is the Crandall–Fagin IBDWT — each word is multiplied by 2^{ceil(qj/n) − qj/n} before the transform and by the inverse afterwards, so the exponent's non-power-of-two word lengths need no extra padding and the convolution stays exact. Here every weight is a *cyclic shift mod 61* because 2 is a 61st root of unity in Z61:
  ```c
  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  const u32 m61_log2_root_two = (u32) (((1ULL << 60) / NWORDS) % 61);
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m61_log2_root_two % 61;
  ...
  u61[i] = U2(shl(make_Z61(in[p].x), m61_weight_shift0), shl(make_Z61(in[p].y), m61_weight_shift1)); // Form a GF61 from each pair of input words
  ```
  (`src/cl/fftp.cl:446-454`, `:487`; same at `src/cl/carryfused.cl:861-877`, `:1030`). Big/little-word flags and shifts advance together through one packed 64-bit counter built from `fracBits`/`comboFracBits` (`src/cl/weight.cl:1-30`) and `FRAC_BPW_HI/LO` (`src/Gpu.cpp:589-595`). After the inverse NTT the shift is corrected by `log2_NWORDS + 1` because "NTT returns results multiplied by 2*NWORDS" (`src/cl/carryfused.cl:881-886`).
* **Middle product is not a pointwise square.** `pairSq(N,u,v,t_squared,special)` (`src/cl/tailsquare.cl:970-991`) calls `onePairSq(pa,pb,t_squared,t_squared_type)` (`:892-968`): `X2qconjb(a,b)`, then `a2 = csqq(a)`, `b2 = csq(b)`, `ab`, `addin = neg(a2+b2,4)`, `d = csqa(ab, addin, 6)`, `b2t2 = cmul(b2, t_squared)`, combining `c = subq(a2,b2t2)` / `addq` / `subiq` / `addiq` per `t_squared_type` (0/1/2/3 = ×1, ×i, ×−1, ×−i) and ending `*pa = SWAP_XY(c), *pb = SWAP_XY(d)`. Lines 0 and H/2 pair with themselves, with `TAILTGF61` as extra factor (`tailSquareZeroGF61`, `:996-1041`; `:1106`).
* **Data layout.** Documented at `src/cl/middle.cl:640-681`; addressing in `readCarryFusedLine`/`writeCarryFusedLine` (`:716-729`), `readMiddleInLine`/`writeMiddleInLine` (`:739-749`), `readTailFusedLine`/`writeTailFusedLine` (`:760-770`), with `SWIZ(a,m) = (m)^(a)` and pads `SIZEBLK = SMALL_HEIGHT`, `SIZEW = 16*SIZEBLK + 16`, `SIZEM = WIDTH/16*SIZEW (+16)` (`:695-708`).

## 4. Bits-per-word and exactness

`src/fftbpw.h` feeds a `map<string, array<float,6>> BPW` (`src/FFTConfig.cpp:22-29`). **Every bpw figure is bits per big-integer *word*, with word count `N = shape.size() = width*middle*height*2`** — the host computes `float const bitsPerWord = E / float(N);` (`src/Gpu.cpp:1124`) and `maxExp() = maxBpw() * size()` (`src/FFTConfig.h:50`). The 6 values per line are variants 000, 101, 202, 010, 111, 212 (`src/FFTConfig.h:13-17`), consumed as `bpw[variant_M*3 + variant_H]`, averaged when W≠H (`src/FFTConfig.cpp:314-330`). They are empirical: header comments say the tables were "Computed by targeting maxROE of ~0.35 over 1000 iterations" (`src/fftbpw.h:95`, `:133`) — a statistical overflow criterion, not a hard bound.

| type | table | representative key | bpw | transform element types | bytes/word | bpw per byte |
|---|---|---|---|---|---|---|
| FFT64 | `fftbpw.h:2-94` (no prefix) | `256:2:256` | 19.204 … 19.636 | `T2` = `double2`, 16 B (`base.cl:267-268`) | 8 | 2.40–2.45 |
| FFT64 | | `512:8:512` | 18.256 … 18.444 | | 8 | 2.28–2.31 |
| FFT64 | | `4K:16:1K` | 16.744 … 17.208 | | 8 | 2.09–2.15 |
| FFT3161 | `fftbpw.h:95-113` (`1:`) | `1:256:2:256` | 40.54 (all 6) | `GF31` uint2 8 B + `GF61` ulong2 16 B (`base.cl:276-278`) | 12 | 3.38 |
| FFT3161 | | `1:512:8:512` | 39.46 | | 12 | 3.29 |
| FFT3161 | | `1:4K:16:1K` | 37.12 | | 12 | 3.09 |
| FFT61 | `fftbpw.h:133-151` (`3:`) | `3:256:2:256` | 25.02 | `GF61` ulong2, 16 B | 8 | 3.13 |
| FFT61 | | `3:512:8:512` | 23.84 (`:141`: cut from 23.94 after a failed LL) | | 8 | 2.98 |
| FFT61 | | `3:4K:16:1K` | 22.42 | | 8 | 2.80 |
| FFT3261 | `fftbpw.h:114-132` (`2:`) | `2:256:2:256` | 34.53 | FP32 `F2` float2 8 B + GF61 16 B | 12 | 2.88 |
| FFT323161 | `fftbpw.h:152-170` (`4:`) | `4:256:2:256` | 50.01 | F2 8 B + GF31 8 B + GF61 16 B | 16 | 3.13 |
| FFT3231 | `fftbpw.h:171-189` (`50:`) | `50:256:2:256` / `50:4K:16:1K` | 19.57 / 7.05 | F2 + GF31 | 8 | 2.45 / 0.88 |
| FFT6431 | `fftbpw.h:190-208` (`51:`) | `51:256:2:256` / `51:4K:16:1K` | 35.27 / 32.25 | T2 + GF31 | 12 | 2.94 / 2.69 |
| FFT31, FFT32 | **no table entries** | — | — | GF31 8 B / F2 8 B per element ⇒ 4 B/word | 4 | — |

Bytes/word derivation: `FP64_DATA_SIZE`/`FP32_DATA_SIZE`/`GF31_DATA_SIZE`/`GF61_DATA_SIZE` return `PAD_ADJUST(W*M*H*2, …)` scaled by `sizeof(element)/sizeof(double)` (`src/Gpu.h:390-393`) — unpadded N doubles, N floats, N/2 floats, N/2 doubles. `PAD_ADJUST` then pads/swizzles on top (`src/Gpu.h:376-389`: ×1.0 at `INPLACE=1`, up to ×1.6 at `inplace=0, pad=512`), so bytes/word here is *payload*, not allocation.

**How bpw is picked.** `FFTConfig::maxBpw()` (`src/FFTConfig.cpp:314-330`) looks up shape `"<type>:<W>:<M>:<H>"`; on a miss `FFTShape::FFTShape` maps the shape onto a pre-computed one (`while (m < 9) { m *= 2; w /= 2; } …`, `src/FFTConfig.cpp:152-181`) and subtracts 0.05, else falls back to `bpw = {18.1f, …}` plus `log("ERROR: BPW info for %s not found, using default of 18.1.\n")` (`:166-169`) — the path FFT31/FFT32 take, since they are also absent from `allShapes()` (`:92`). Consumers: `maxExp()` vs `E` warns (`src/Gpu.cpp:1130-1132`); `bitsPerWord < minBpw()` (3.0 for all types except FFT32 = 1.0, `src/FFTConfig.h:48`) throws "FFT size too large" (`src/Gpu.cpp:1135-1138`); `useLongCarry` is forced below 10.0 bpw (`:1140`); `MAXBPW` (×100) reaches the kernels (`:505`, register selection `numRegisters` `:755-860`) and drives the "sloppy" carry shortcuts, defined as 1.1 bits below maxbpw (`src/cl/carryutil.cl:695-699`). `CARRY32` is illegal above `EXP/NWORDS = 18` (`src/cl/carryutil.cl:807-819`), so the NTT types need 64-bit carries.

## 5. Carry / reconstruction

* `bitlen(b) = EXP / NWORDS + b` is the bit length of a little (b=0) or big (b=1) word (`src/cl/carryutil.cl:92`). `carryStep(x, &outCarry, isBigWord)` returns the signed low `nBits` and passes the rest on (`:547-579`), so the chain is "word := low nBits; carry := x >> nBits" across the whole number; a word pair goes through `weightAndCarryPair(+Sloppy)` (`src/cl/carryinc.cl:20-28`, `:236-245`), and `carryFinal(u, inCarry, b1)` applies the last in-carry to word 0 of a line, leaving the remainder un-normalised (`src/cl/carryinc.cl:5-10`).
* Fused path: one `carryFused` kernel per line does inverse weights → integer → carry propagation over the line's NW pairs, passing carries to the next workgroup through the `carryShuttle` global buffer with `write_mem_fence`/`ready` flags (`src/cl/carryfused.cl:131-133`, `:915-1039`). Split path when `useLongCarry`: `carry` (`src/cl/carry.cl:526-613`) + `carryB` (`src/cl/carryb.cl:8-49`) + `fftP` (`src/Gpu.cpp:2212-2221`).
* Cost relative to the transforms: **no measurement stored in the repo.** The one quantitative hint is `src/Gpu.cpp:1293-1295`: "a 4M GF61+GF31 NTT needs just 32MB L2 cache during GF61 processing of fftMiddleIn, tailSquare, and fftMiddleOut (and only 16MB duing GF31 processing)".
* **FP64 vs M61 vs M31+M61 speed: not documented.** `z.txt` is a `-ztune` log of FP64 variants with zero hits for `NTT`/`GF61`/`M61`; there is no `CHANGELOG`; `README.md`, `README.txt`, `FFT.md`, `tools/README.md` carry no NTT timings (`FFT.md` only tabulates FP64 radix FMAs/ADDs). The repo measures it at runtime instead: `-tune` times `FFT64 512:16:512` against `FFT3161 512:8:512` on exponent 141000001 and branches at 0.80/1.20 ratio thresholds (`src/tune.cpp:454-484`).

## 6. Porting checklist (CUDA)

Must replicate for correctness:
* Field arithmetic exactly as in §2, **including the lazy range convention**: no canonical `[0,p)`; `weakMul` operands must be positive and you must track the folded-in M61 count. Every `modM61q(…, k)` bound in the butterflies (`src/cl/fft4.cl:243-246`, `src/cl/fft8.cl:118-142`, `src/cl/tailsquare.cl:892-968`) is hand-derived; getting one wrong is silently wrong, not crash-wrong.
* M61 fold `2^64 ≡ 8`, the 128-bit path (`mul64`+`weakModM61`), and the second fold of the top 6 bits.
* Root `_h = (264036120304204, 4677669021635377)` of order 2^62, forward direction as chosen at `src/TrigBufCache.cpp:796-799` (the other root flips every twiddle sign).
* Twiddle table contents **and** the index formula `trig[(i-1)*WG + (me & ~(f-1))]` (§3), the table concatenation order, and the `DIST*TRIGGF61` offsets; `TAIL_TRIGS61`/`TABMUL_CHAIN61` select different tables/indexing (`src/Gpu.cpp:1170-1181`, `src/cl/fftbase.cl:1936-2032`).
* IBDWT weights as shifts mod 61 with the `+log2_NWORDS+1` correction, and the big/little-word flags from the same packed counter (`fracBits`/`comboFracBits`, `FRAC_BPW_HI/LO`). A sign or off-by-one shifts every word.
* `pairSq`/`onePairSq` for the middle product (not a pointwise square), the special line 0 / line H/2 handling with `TAILTGF61`, the `SWAP_XY` that replaces a conjugation before the inverse transform (`src/cl/tailsquare.cl:47-49`, `:967`), the line pairing `line2 = line1 ? H - line1 : H/2` (`:1063-1066`), and the layout/padding/swizzle of §3.
* Carry: `nBits = EXP/NWORDS + (isBigWord?1:0)`, the big/little flag sequence, MUL3, the LL `−2` initial carry (`src/cl/carryfused.cl:904-907`), CRT fusion `n61*M31 + n31` for M31+M61 (and the FP32 multiple determination for FFT323161); 64-bit carries are mandatory (`src/cl/carryutil.cl:807-819`).
* `WordSize`/`Word` (4 or 8 bytes host-side) and the fact that two consecutive words become one complex element (`src/FFTConfig.cpp:297-305`, `src/cl/base.cl:316-324`).

Pure performance tuning, safe to defer:
* `INPLACE`, `SHUFL_BYTES_W/H`, `LDSPAD_W/H`, `LDSMUL_W/H`, `WMUL`, `MULTI_Q`, `L2_STRIPING`, `ZEROHACK_*`, `UNROLL_*` (`src/cl/base.cl:5-11`, `:157-234`).
* Shape search/`-tune` (`src/tune.cpp`), CUDA graph replay (`src/Gpu.cpp:2230-2249`), queue splitting (`:1260-1284`), register/VGPR caps (`numRegisters` `:755-860`, `amdRegisterOption` `:914-927`).
* PTX/AMD fast paths: `TRY_SHL30/31`, `DISABLE_MUL64`, `ENABLE_MAD64`, `ENABLE_ALT_MUL64`, `MODM31`, `TABMUL_CHAIN*`, `MIDDLE_CHAIN`, `FUSE_WEIGHT_BUTTERFLY`, `ROE`/`STATS` (`src/cl/math.cl:300-330`, `:804-813`, `:1091-1130`, `src/cl/base.cl:184-234`), and chainMul-vs-table reads for twiddles (`src/cl/fftbase.cl:1918-1932`).

## 7. Explicitly not determined from the source

* No written derivation of the `v2(p−1)=1 ⇒ GF(p²)` argument (§1; searched `src/`, all `*.md`, `README.txt`, `z.txt`).
* No measured FP64-vs-M61-vs-M31+M61 timings, and no NTT rows anywhere in `z.txt` (§5).
* Whether the "60th root" comments (`src/cl/fftp.cl:447`, `src/cl/carryfused.cl:862`) are a typo for 61st; the code reduces shifts mod 61 throughout.
* Effective bpw for `FFT31`/`FFT32`: no table entry and not enumerated by `allShapes()`; the code falls back to the shape-mapping heuristic and then a hardcoded 18.1 (`src/FFTConfig.cpp:152-169`). The value a given shape actually reaches was not computed here (read-only analysis; nothing built or run).
* Carry cost relative to the transforms (§5), and the reasoning behind the `log2_NWORDS + 1` correction beyond the one-line comment at `src/cl/carryfused.cl:881-886`.
