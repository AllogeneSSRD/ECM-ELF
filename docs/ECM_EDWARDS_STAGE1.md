# ECM Edwards Stage-1 CPU 实现 Spec

目标：在现有 `ecm_driver` 体系内实现 **Atkin-Morain Edwards（a=1，Z/2×Z/8）Stage-1 CPU 版**，公式参考 Prime95、但用 limb（GMP `mpz`）而非 FFT，目标 <10000-bit 整数。交叉验证对比 Prime95（同 σ 同曲线同 `[s]P`），stage-2 存档格式后置（本 spec 已读清字节格式）。

## 1. 曲线构造（Atkin-Morain，sigma_type=0）

Prime95 `choose_atkin_morain`（`ecm.cpp:4256`），参考论文 §7（ePrint 2008/016）：

1. 在 Weierstrass 曲线 `T² = S³ − 8S − 32` 上取点 `(12, 40)`。
2. 标量乘 `(s, t) = σ · (12, 40)`（σ 为 uint64）。
3. 归一化 `s = sN/sD`、`t = tN/tD`（tD = sD）。
4. 推导 α、β、d（**等价域运算，避免中间归一化**）：
   - `α = (s−9) / (t+s+16)`
   - `β = 2α(4α+1) / (8α²−1)`
   - `d = (2(2β−1)² − 1) / (2β−1)⁴`
5. 基点 P = (x1, y1)：
   - `x = (2β−1)(4β−3) / (6β−5) = (2β−1)(2(2β−1)−1) / (3(2β−1)−1)`
   - `y = (2β−1)(t²+50t−2s³+27s²−104) / ((t+3s−2)(t+s+16))`

结果：plain Edwards 曲线 `x²+y² = 1 + d·x²y²`（a=1），Z/2×Z/8 扭子群（16 整除 #E）。

> Prime95 为省逆采用 `sN/sD`、`dN/dD` 形式（`ecm.cpp:4356-4433` 的 `#else` 分支），只做 1 次模逆（`inv(sD·dD)`）。用 `mpz` 实现时可直接按上式归一化（多次 `mpz_invert`），正确性优先。

## 2. 扩展 Edwards 域公式（a=1）

扩展坐标 `(X:Y:Z:T)`，`x=X/Z, y=Y/Z, T=XY/Z`。mod N（合数）。

标准 Edwards 加法律（曲线 `x²+y²=1+dx²y²`，恒等点 `(0,1)`）：
`x3 = (x1y2+x2y1)/(1+dx1x2y1y2)`，`y3 = (y1y2−x1x2)/(1−dx1x2y1y2)`。

- **统一加法**（HWCD08 通用 a 公式，a=1，**9M**）：
  ```
  A = X1·X2;  B = Y1·Y2;  C = d·T1·T2;  D = Z1·Z2
  E = (X1+Y1)(X2+Y2) − A − B
  F = D − C;  G = D + C;  H = B − A        # a=1
  X3 = E·F;  Y3 = G·H;  T3 = E·H;  Z3 = F·G
  ```
- **倍点**（`dbl-2008-hwcd`，a=1，**4M+4S**）：
  ```
  A = X1²; B = Y1²; C = 2Z1²; D = A (=aA, a=1)
  E = (X1+Y1)² − A − B;  G = D+B;  F = G−C;  H = D−B
  X3 = E·F;  Y3 = G·H;  T3 = E·H;  Z3 = F·G
  ```
  （倍点不含 d：用曲线方程 `x²+y²=1+dx²y²` 把 `1+dx²y²` 代成 `x²+y²`，故无需 d。）

> ⚠️ 不要用 `add-2008-hwcd-4`（`C=2T1Z2, D=2T2Z1` 那组）：它给出 `x3=(x1y1+x2y2)/(y1y2−x1x2)`，是**错误**的加法律（会把 `[2]P` 算成曲线外点）。Prime95 的 `ed_add`（`ecm.cpp:2016`）实际用的是 Atnashev eprint 2021/1061 的替代公式（替代坐标 `(XZ,YZ,XY,Z²)`，`x3=(x1y1+x2y2)/(y1y2+x1x2)`），与标准加法律等价（同恒等点 `(0,1)`、同曲线），故 `[s]P` 一致，仅投影缩放不同。limb 实现用标准 9M 公式即可。

## 3. NAF 标量乘法

- 指数 `s = 48 · lcm(1..B1)`（`48 = lcm(12,16)`，见 `ecm_calc_exp` 初始 `g=48`；screen.log 实测 B1=1e6 指数长 1442105 bit = ceil(log2(48·lcm(1..1e6)))，与 48·lcm 精确一致）。
- 转 **w-NAF**（有符号、无相邻非零位），非零密度 ~1/(w+2)。（当前实现先用 double-and-add 跑通正确性，NAF 后续接入。）
- **字典**：预计算奇数倍 `3P, 5P, …, (2^(w−1)−1)P`，批量模逆归一化（`dict_normalize`，1 次逆归一化整表）。
- 逐 bit：1 次倍点 +（非零位）1 次加法。
- 命中判定：`[s]P = 恒等点 (0,1)` ⇔ `ord(P)` 为 B1-powersmooth（与 `ecmath.py` 一致）。

> 窗口 w 只影响性能不影响结果，交叉验证用任意 w 均可；Prime95 默认 DictionaryMemory=256MB → w≈14（4675 字典项，见 screen.log）。

## 4. 曲线构造 + 标量乘的 stage-1 结果（交叉验证口径）

stage-1 完成后，Edwards 点 `[s]P=(e.x:e.y:e.z)` 转 Montgomery（`ed_to_Montgomery`），得 `(Qx:Qz)`：

- Montgomery ↔ Edwards 双有理映射：`u=(1+y)/(1−y), v=u/x`；Montgomery `A = 2(1+d)/(1−d) = 2(a+d)/(a−d)`（a=1），`B=4/(1−d)`。
- Prime95 存 `Ad4 = (dD−dN)/dD = 4/(A+2)`（见 `choose_atkin_morain` 与 `ed_to_Montgomery`）。

交叉验证：同 σ（如 20260922）、同 N（M347=2³⁴⁷−1）、同 B1=1000000，计算 `[s]P` → `(Qx,Qz)`，与存档 `e0000347` 中的 `Qx_binary`/`Qz_binary` 比对。

> **关键**：`(Qx,Qz)` 的**投影缩放不唯一**。不同加法律（标准 9M vs Atnashev 替代式）产生同仿射点但不同 Z，故原始 `Qx/Qz` 字节不逐位相等；**投影无关的不变量是 `u = Qx/Qz = (1+y)/(1−y)`**（Montgomery x 坐标），以及 `gcd(Qz,N)`（因子判定）。交叉验证以 `u` 一致 + `gcd(Qz,N)` 一致为准。

## 5. Prime95 存档字节格式（ECM_VERSION=6）

全部 **小端**。实测 `e0000347`（204 字节）如下：

```
偏移    字段            类型        实测(M347) 说明
0x00    magic          u32          0x1725bcd9
0x04    version        u32          6
0x08    k              double       1.0
0x10    b              u32          2
0x14    n              u32          347          (M347 = 2^347 - 1)
0x18    c              i32          -1
0x1c    stage          char[10]     "C1S2"+6×0
0x26    2×pad          byte         0,0
0x28    pct_complete   double       0.0
0x30    checksum       u32          (write_checksum 回填到 offset 48)
0x34    curve          u32          1
0x38    average_B2     u64          0           (不计入 checksum)
0x40    state          u32          2           (ECM_STATE_MIDSTAGE)
0x44    sigma          u64          20260922    (不计入 checksum)
0x4c    B              u64          1000000     (B1)
0x54    C              u64          100000000   (B2)
0x5c    sigma_type     u32          0           (0=Atkin-Morain Edwards)
0x60    montg_stage1   u32          1           (见下方 ⚠️)
-- MIDSTAGE 状态数据 --
0x64    Qx_binary      giant        len=11 + 44B
0x94    Qz flag        i32          1
0x98    Qz_binary      giant        len=11 + 44B
0xc8    gg flag        i32          0
```

**字段读取顺序**（`ecm_restore`，`ecm.cpp:6445-6465` 与 `ecm_save` 一致）：`curve, average_B2, state, sigma, B, C, sigma_type, montg_stage1`。

**giant 编码**（`write_giant`/`commonc.c:4548`）：4 字节 limb 数（u32）+ `len×4` 字节 limb 数组（**32-bit 小端**，即数值 mod N 的标准二进制）。`gwnum` 经 `gwtogiant` 转此格式。

**checksum**（`write_checksum`）：把各 `write_uint32/64`（sum≠NULL 的）与 giant（len+Σlimb）累加（u32 溢出回绕），回填到 **offset 48**（`CHECKSUM_OFFSET`）。`average_B2`、`sigma` 以 `sum=NULL` 写入，不计入。

**footer**（可选）：`"MOREINFOJSONDATA"`(16B) + chunk_size + version=1 + crc32 + JSON，仅当 `morejsoninfo` 非空。

**state 枚举**：`0=STAGE1_INIT, 1=STAGE1, 2=MIDSTAGE, 3=STAGE2, 4=GCD`。
- `STAGE1`（Edwards, montg_stage1=0）：`stage1_start_prime, stage1_exp_buffer_size, stage1_bitnum, NAF_dictionary_size, dict_start.x/y, e.x/e.y/e.z`。
- `MIDSTAGE`（stage1 完成）：`Qx_binary` + 可选 `Qz_binary` + 可选 `gg_binary`。

## 6. 待确认矛盾（⚠️）

实测 `sigma_type=0`（Edwards 曲线）但 `montg_stage1=1`（Montgomery stage 1），与 screen.log "Edwards curve #1 + dictionary size 4675"（Edwards NAF）矛盾。可能：① 用户 Prime95 配置 `MontgStage1=1`（Edwards 曲线但 Montgomery stage 1）；② 31.04 与 31.06 字段语义差异。**不影响交叉验证**（`[s]P` 在同一条曲线上，坐标系统不同但 Qx/Qz 相同），实现时需用 Prime95 实测确认。

## 7. 交叉验证结果（✅ 已通过）

独立可编译单元 `src/cpu/ecm_edwards_cpu.cpp`（MSVC + vcpkg GMP，`build_edwards_test.bat`），用法 `ecm_edwards_cpu <N> <sigma> <B1>`，输出 d、P、`[s]P` 仿射 y、`u=Qx/Qz`、`gcd(Qz,N)`。

- ✅ **构造一致**：同 σ=20260922、M347，d、P.x、P.y 与 Prime95 `choose_atkin_morain` 公式（及 Python 独立实现）逐位一致。
- ✅ **指数一致**：`s_bits = 1442105`，等于 Prime95 screen.log 的 "exponent length 1442105"。
- ✅ **`[s]P` 一致**：`u = Qx/Qz = 65291666393298858753814459376057982063685036782370801855363963899907103095354875531365032313828077159500`，与 `e0000347` 存档的 `Qx_binary/Qz_binary` 计算出的 `u` **完全相等**。
- ✅ **因子命中回归**（`results.txt` 的 Edwards stage-1 命中，B1=1e6）：
  | N | 期望因子 | sigma | 实测 gcd(Qz,N) |
  |---|---|---|---|
  | M677 | 1943118631 | 6581585141005897 | ✅ 1943118631 |
  | M991 | 8218291649 | 105413044550089 | ✅ 8218291649 |
  | M4003 | 16756559 | 2027329164697536 | ✅ 16756559 |
- ✅ **无因子一致**：M347/sigma=20260922 `gcd(Qz,N)=1`，与 `e0000347` 为 mid-stage 存档（未命中）一致。

> 命中时 `u` 无定义（`Qz` 含因子 p，`Qz⁻¹ mod N` 不存在），程序输出 `u=0`，正常。

## 8. 实现状态

- [x] `src/cpu/ecm_edwards_cpu.cpp/.h`：Atkin-Morain 构造 + 标准 Edwards 加/倍 + double-and-add + `ed_to_Montgomery`，全部 mpz，独立可测。
- [x] 接入 `ecm_driver`：`--edwards` flag（单次运行）+ 队列管理器 `edwards = 1`（`ecm.ini`）。64-bit sigma（`-sigma` 与 `ECM2=` 的 `specificsigma`）；随机 sigma 直接移植 Prime95 `ecm.cpp:7397` 的生成式（`(rand()&0x1F)<<48 + (rand()&0xFFFF)<<32 + (rdtsc_lo^rdtsc_hi^(rand()<<16))`，拒绝 `sigma<=5`，`srand(time)` + `__rdtsc` 增熵）。
- [x] Prime95 `ECM2=` worktodo 读取器（`ecm_worktodo.cpp/.h`），保留原 `ECMSTAGE2=` 语义；队列管理器按前缀分发。
- [x] double-and-add → **w-NAF + 仿射字典**（默认 w=8，可 `edwards_set_naf_w` 调）：`naf_digits` 移植 Prime95 `ecm.cpp:4824-4856` 的 O(bits) 单遍算法（`tstbit` + carry，避免 O(bits²) 逐位右移）；字典批量逆归一化到仿射，混合加法 `ed_add_affine`（8M）。M991 标量乘 9.8s→6.7s（standalone，~31%）；driver 端到端 10s→7.8s（~22%）。
- [x] mpz → **mpn Montgomery**（`ecm_edwards_mont.h`，固定 160×64-bit limb=10240 bit，覆盖 <10000 bit）：REDC 用 `mpn_addmul_1` 自实现（`mpn_redc_1` 不在公开接口），域运算全部栈上无堆分配。标量乘主循环完全走 Montgomery，曲线构造仍用 mpz（含逆）。M991 driver 端到端 7.8s→3.0s（累计 ~3.3×）。
- [ ] Prime95 二进制存档读写（字节格式已在本 spec §5 读清）。

### ECM= / ECM2= worktodo 格式（已实现，两者等价）

```
ECM=[<AID>,|N/A,|<nul>][FFT2=<fftl>,|<nul>]<k>,<b>,<n>,<c>,<B1>[,<B2>][,<curves_to_run>][,<specificsigma>][,"comma-separated-list-of-known-factors"]
```

- Prime95 将 `ECM=` 与 `ECM2=` 视为等价（`commonc.c:2934`）；`B2` 默认 0，`curves_to_run` 默认 100。
- `<specificsigma>` 为 64-bit（Atkin-Morain 曲线参数）；缺失/0 = 随机。
- `N = (k*b^n + c) / (已知因子乘积)`，已知因子不整除时报错。
- 队列管理器逐行处理：命中/成功 → 移入 finished；错误 → 原位标记 `# ERROR`。
- 端到端验证（`edwards=1`）：`ECM=1,2,677,-1,1000000,0,1,6581585141005897` → `1943118631`；`ECM=1,2,991,-1,1000000,0,1,105413044550089` → `8218291649`。

> 投影缩放差异：原始 `Qx/Qz` 与 Prime95 相差一个非平凡投影因子 λ（Prime95 的 Atnashev 序列不归一化 Z）。这不影响因子判定（`gcd(Qz,N)`）与跨版本续跑（stage-2 只用仿射 `u=Qx/Qz`）。若需**字节级**复刻 Prime95 存档，须复刻 Atnashev 公式 + 其 NAF 字典归一化序列（非必需，见 §4 说明）。

## 9. 性能调优数据（B1=1e6，`tools/ecm_edwards_bench.cpp`）

### w 窗口扫描（ED_MONT_MAX_LIMBS=160）

| N (bits) | w=2 | w=4 | w=8 | w=10 | w=12 | w=14 | 字典内存(kB) |
|---|---|---|---|---|---|---|---|
| M347 (347) | 0.67 | 0.60 | 0.62 | 0.67 | 0.68 | 0.69 | w↔内存: 2^(w-2)×8.75 |
| M677 (677) | 2.16 | 2.05 | **1.59** | **1.54** | 1.54 | 1.57 | w=8: 560; w=10: 2240; w=14: 35840 |
| M991 (991) | 3.63 | 3.25 | 3.07 | **2.92** | 2.98 | 2.99 | |
| M4003 (4003) | 47.4 | 42.4 | 38.8 | — | 37.5 | **37.0** | |

结论：**w=8~10 为甜点**（默认已设 w=8）。更大 w 减少加法，但字典 2^(w-2) 项、缓存不友好（w≥12 时字典 >9MB 不再受益甚至略慢）。字典内存 ≈ 2^(w-2) × (4+3) × limb×8 字节。

### limb 数扫描（对小 N 减少 ED_MONT_MAX_LIMBS）

| N | 160-limb | 16-limb | 说明 |
|---|---|---|---|
| M347 (6 limbs) | 0.60 | 0.60 | 无差异（曲线构造 mpz 占主导） |
| M677 (11 limbs) | 2.05 (w=4) | 1.67 (w=4) | **~18% 快**（缓存局部性） |
| M991 (16 limbs) | 3.07 (w=8) | 2.98 (w=8) | ~3% 快 |

结论：mpn 热循环已按运行时 `nlimbs` 计算（不按 MAX_LIMBS），故减小 MAX_LIMBS **不改变计算量**，收益来自**结构体更紧凑 → 更好的缓存局部性**，仅对小 N（<~700 bit）显著（10~20%）。默认 160 覆盖 <10000 bit 目标；若部署目标为小 N，可 `-DED_MONT_MAX_LIMBS=16`（≤1024 bit）或 `32`（≤2048 bit）重编译换取该收益。
