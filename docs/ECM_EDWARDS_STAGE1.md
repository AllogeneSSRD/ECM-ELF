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
- [x] **Prime95 二进制存档读写**（`ecm_edwards_save.{h,cpp}`，ECM_VERSION=6）：MIDSTAGE（state=2，Qx/Qz）与 STAGE1（state=1，Edwards checkpoint）写/读，checksum 公式与 `e0000347` **字节级一致**（写出的 204 字节与 Prime95 原文件完全相同）。
- [x] **交接自动化**（§11 已重构：`ecm.exe` 只落本地盘）：原实现写 `e{n:07d}` + 追加 `worktodo.add`。现在 `ecm.exe` 只写本地 `e{n:07d}_c{curve:06d}.tmp`；把结果送进 Prime95（写 `e{n:07d}` + 往 `worktodo.add` 的 `[Worker #N]` 段追加 `ECM=` 行，按 `MaxHighMemWorkers` 限流）由独立程序 **`ecm_p95feeder`** 负责，见 §11。
- [x] **多曲线交接**（§11 已重构）：原为 `ecm.exe` 内后台线程复制 + 轮询 p95 消费；现由 `ecm_p95feeder` 以"在飞计数 + 同 N 唯一 + 空闲 Worker 段"规则完成。
- [x] **自我 checkpoint 恢复**：`ed_mul` 分块可恢复（`edwards_stage1_curve_progress`），每 16384 位回调；时间间隔（复用 `gpuckpt_seconds`）+ SIGINT 触发写 STAGE1 存档；启动时检测并恢复（`resume STAGE1 checkpoint @ bitnum=X`）。
- [x] **多曲线并行（吞吐）**：`curves` 条曲线分派到 `min(curves, #cores)` 个工作线程（每线程独立曲线，原子计数器动态取号）。`ecm.ini` 的 `edwards_threads`（0=自动，1=顺序）/ `--edwards-threads N` 控制；线程内只读共享 `N`/`s`（GMP 支持并发只读），输出按曲线索引隔离；`random_sigma_u64` 用全局 `rand()`，因此 sigma 全部在主线程预生成后分发。多曲线 checkpoint 路径改为每条曲线独立 `e{n:07d}_c{6-digit}`（单曲线仍是 `e{n:07d}`，Prime95 交接路径不变），避免并行写冲突。**24 线程实测 ~12.6× 吞吐**（§10.3）。
- [x] **GMP x86_64/zen3 内核重建 + NAF 窗口重调**：见 §10.2 / §10.4（单曲线 1.17~1.26×、窗口 w 8→12 约 2~5%）。

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

## 9. 性能调优数据（B1=1e6，`tools/bench/ecm_edwards_bench.cpp`）

### w 窗口扫描（ED_MONT_MAX_LIMBS=160）

| N (bits) | w=2 | w=4 | w=8 | w=10 | w=12 | w=14 | 字典内存(kB) |
|---|---|---|---|---|---|---|---|
| M347 (347) | 0.67 | 0.60 | 0.62 | 0.67 | 0.68 | 0.69 | w↔内存: 2^(w-2)×8.75 |
| M677 (677) | 2.16 | 2.05 | **1.59** | **1.54** | 1.54 | 1.57 | w=8: 560; w=10: 2240; w=14: 35840 |
| M991 (991) | 3.63 | 3.25 | 3.07 | **2.92** | 2.98 | 2.99 | |
| M4003 (4003) | 47.4 | 42.4 | 38.8 | — | 37.5 | **37.0** | |

结论：**w=8~10 为甜点**（默认已设 w=8）。更大 w 减少加法，但字典 2^(w-2) 项、缓存不友好（w≥12 时字典 >9MB 不再受益甚至略慢）。字典内存 ≈ 2^(w-2) × (4+3) × limb×8 字节。

> ⚠️ 此表为早期（通用 GMP 内核、单次测量）数据，已被 §10.4 的交替 A/B 复测取代：**默认窗口已从 8 改为 12**，更细的 w 扫描（含 w=14/16/18 的拐点）见 §10.4。

### limb 数扫描（对小 N 减少 ED_MONT_MAX_LIMBS）

| N | 160-limb | 16-limb | 说明 |
|---|---|---|---|
| M347 (6 limbs) | 0.60 | 0.60 | 无差异（曲线构造 mpz 占主导） |
| M677 (11 limbs) | 2.05 (w=4) | 1.67 (w=4) | **~18% 快**（缓存局部性） |
| M991 (16 limbs) | 3.07 (w=8) | 2.98 (w=8) | ~3% 快 |

结论：mpn 热循环已按运行时 `nlimbs` 计算（不按 MAX_LIMBS），故减小 MAX_LIMBS **不改变计算量**，收益来自**结构体更紧凑 → 更好的缓存局部性**，仅对小 N（<~700 bit）显著（10~20%）。默认 160 覆盖 <10000 bit 目标；若部署目标为小 N，可 `-DED_MONT_MAX_LIMBS=16`（≤1024 bit）或 `32`（≤2048 bit）重编译换取该收益。

## 10. 性能优化记录（编译器 / 构建 / GMP 内核 / 并行）

测试机：AMD Ryzen AI 9 HX 370（Zen 5，12 物理核 / 24 逻辑核，TSC 标定核心频率 ≈ 3.99 GHz），MSVC 14.51（VS18），Windows 10.0.26200。

### 10.1 我们自己的代码：MSVC 编译标签基本无效（热点在 GMP 汇编）

`tools/bench/ecm_edwards_speed.cpp`（M991，B1=1e6，w=8，各 3 次）逐个标签测量：

| 标签 | min (s) | median (s) |
|---|---|---|
| `/O2` | 2.908 | 4.224 |
| `/O2 /arch:AVX2` | 2.763 | 3.540 |
| `/O2 /arch:AVX512` | 2.828 | 3.962 |
| `/O2 /GL`（LTCG） | 2.971 | 3.996 |
| `/O2 /favor:AMD64` | 2.934 | 3.668 |
| `/Ox` | 2.838 | 3.587 |
| `/O2 /GL /arch:AVX2` | 2.892 | 4.105 |

**结论：差异全部落在测量噪声内**（同一配置的 run-to-run 波动 2.9→4.2 s，远大于标签间差异）。原因是**热循环 100% 在 GMP 的 `mpn_*` 手写汇编**里（`mpn_mul_n`/`mpn_sqr` → `mpn_mul_basecase` → `mpn_addmul_1`，以及我们 REDC 里的 `mpn_addmul_1`）；编译器只能影响外层 C 代码（循环、NAF 位生成、Montgomery 结构体调度），占比很小。

**AVX2/AVX-512 同样无效**：GMP 的 `mpn` 是**标量 64-bit** 汇编，我们的域运算/标量乘是天然串行的大整数链，没有可向量化的数据并行维度。要在 x86 上真正吃到 AVX2/AVX-512，需要换掉整个多精度层（例如 GPU 端的 CIOS/批量曲线，或专门的 AVX-512 IFMA 实现）——那不是编译标签能解决的。

> 注：本机笔记本在持续满载时有明显降频，因此**任何 <10% 的差异都需要交替 A/B 测量**才可信（见 §10.4 的方法）。单配置连跑 N 次取 min 会被降频趋势污染。

### 10.2 GMP 内核：vcpkg 版用的是"通用 x86_64"内核（这是真正的瓶颈）

反汇编现用的 `D:\code\vcpkg\installed\x64-windows\bin\gmp-10.dll`（GMP 6.3.0）可见 `__gmpn_addmul_1` 是 `mul` + `adc` 的 4× 展开循环：

```
0000000180001780: add  qword ptr [rcx+r8*8],r10
0000000180001784: adc  rdi,rax
0000000180001787: mov  rax,qword ptr [rsi+r8*8]
000000018000178B: adc  r11,rdx
0000000180001794: mul  rax,r9          <-- 通用内核（K8 时代风格）
```

原因：vcpkg 的 gmp port 用 `--build=x86_64-pc-mingw32` 配置且未开 `--enable-fat`，GMP 的 `configure.ac` 对 `$host_cpu = x86_64` 选择 `path_64="x86_64/k8 x86_64"`，即 `mpn/x86_64/` 下最通用的内核；GMP 6.3.0 里其实已经有 `mpn/x86_64/{zen,zen2,zen3,mulx,coreibwl}/`（zen3 的 4 个热内核直接 `include_mpn` 自 `coreibwl`，用 **mulx + adcx/adox**，官方表格标称 zn3 = 1.5 cycles/limb）。

**处理**：用 `third_party/build_gmp_zen3.sh` 重建了一份 GMP，除 CPU 路径外完全复刻 vcpkg 的配置（`CC="compile cl.exe"`、`CCAS=clang --target=x86_64-pc-windows-msvc`、`--enable-shared --disable-static`、msys2 的 m4/make/sed），只把那一支改成：

```
path="x86/k7/mmx x86/k7 x86/mmx x86"
x86_have_mulx=yes
path_64="x86_64/zen3 x86_64/zen2 x86_64/zen x86_64"
```

产物 `third_party/gmp-zen3/dist/`（`bin/gmp-10.dll` + `lib/gmp.lib` + `include/gmp.h`），反汇编确认含 `mulx`×315 / `adcx`×150 / `adox`×176。构建要点：configure 必须**用相对路径调用**（`../src/configure`），否则 GMP 会把 MSYS 风格的绝对路径写进 `#include "$srcdir/gmp-h.in"` 探测程序，cl.exe 无法解析。

`tools/bench/gmp_mpn_microbench.c`（用 8×`add` 依赖链标定核心频率后换算 cycles/limb）：

| n (limb) | `mpn_addmul_1` 通用 | zen3 | 加速 | `mpn_sqr` 通用 | zen3 | 加速 |
|---|---|---|---|---|---|---|
| 8 | 1.872 | 1.765 | 1.06× | 1.258 | 1.137 | 1.11× |
| 16 | 2.007 | **1.591** | **1.26×** | 1.034 | 0.905 | 1.14× |
| 32 | 1.908 | 1.474 | 1.29× | 0.954 | 0.784 | 1.22× |
| 63 | 2.063 | **1.460** | **1.41×** | 0.767 | 0.642 | 1.19× |
| 128 | 2.248 | 1.443 | 1.56× | 0.542 | 0.483 | 1.12× |

（单位为 cycles/limb；`mpn_mul_n` 在 n=16/32/63/128 只快 2~8%，n=8 反而慢 10%（1.664→1.844）——n≥16 时 GMP 走 Toom-2.2 而非纯 basecase，且小 n 下 mulx 版的前导分派开销占比大。这解释了为什么 M347 端到端没有收益。）

**端到端 A/B**（同一份源码、同一组 MSVC 标签，只换 GMP，交替跑以抵消降频）：

| N | 通用 GMP | zen3 GMP | 加速 |
|---|---|---|---|
| M347 (347 bit) | 0.619 s | 0.624 s | ~1.00×（小 n 无收益） |
| M991 (991 bit) | 3.225 s | 2.769 s | **1.17×** |
| M4003 (4003 bit) | 54.04 s | 42.92 s | **1.26×** |

**正确性**：`BUILD_ECM_EDWARDS_STANDALONE` 交叉验证程序在两种 GMP 下输出**逐字节相同**（M347，sigma=20260922，B1=1e6），且 `u` 仍等于 Prime95 `e0000347` 的已知值；三条因子不变式（M677→1943118631、M991→8218291649、M4003→16756559）全部通过；`ecm_edwards_checkpoint_test` / `ecm_edwards_save_test`（含与 `e0000347` 的字节级比对）/ `ecm_worktodo_test` 全绿。

**⚠️ 可移植性**：zen3 内核使用 **BMI2（`mulx`）+ ADX（`adcx`/`adox`）**，需要 Intel Haswell（2013）/ AMD Excavator / Zen 及以上。若要同时保留老机器支持，应改用 `--enable-fat`（运行期 CPUID 分派）；注意 GMP 6.3.0 的 `mpn/x86_64/fat/fat.c` 只认到 AMD family 0x19，Zen 5 是 0x1A，需补一行 `case 0x1a:`，否则回退到通用内核（实测那条路径用的是 `zen/aorsmul_1.asm`，官方标称 2.0 cycles/limb，收益很小）。

集成方式：`CMakeLists.txt` 会**自动检测** `third_party/gmp-zen3/dist`（存在则优先，否则回退 vcpkg GMP），也可用 `-DECM_WINDOWS_GMP_ROOT=<前缀>` 显式指定。注意 vcpkg 工具链会把 `CMAKE_FIND_ROOT_PATH` 钉在自己的 triplet 上，使 `find_path`/`find_library` 忽略外部 HINTS——所以 CMakeLists 在 `ECM_WINDOWS_GMP_ROOT` 有效时直接强制写入 `GMP_INCLUDE_DIR`/`GMP_LIBRARY`。`build_edwards_test.bat` 同样优先使用该前缀。

### 10.3 多曲线并行：真正的吞吐杠杆

同一 N 的多条曲线彼此独立，因此按曲线并行。用法：

```ini
# ecm.ini
edwards = 1
edwards_threads = 0   # 0 = 自动 min(曲线数, 核数); 1 = 顺序; N = 指定
```

```
ecm.exe --edwards --edwards-threads 12 -gpucurves 24 1000000 < N.txt
```

24 条曲线、M991、B1=1e6（`tools/bench/bench_threads.ps1`；最终配置 = zen3 GMP + w=12。1 线程数字含持续满载降频，故"加速比"偏乐观，务必看**绝对曲线耗时**）：

| 线程 | 总墙钟 (s) | 每曲线 (s) | 相对 1 线程 |
|---|---|---|---|
| 1 | 87.74 | 3.66 | 1.00× |
| 4 | 27.50 | 1.15 | 3.19× |
| 8 | 13.87 | 0.58 | 6.33× |
| 12 | 10.03 | 0.42 | 8.75× |
| 24 | **6.97** | **0.29** | **12.58×** |

**净效果**：会话开始时的基线（顺序执行 + 通用 GMP + w=8）跑完这 24 条曲线需 ~111 s；现在是 **6.97 s ≈ 16×**（= 并行 12.6× × zen3 内核 1.22× × w=12 1.05×）。

观测要点：

- 12 条曲线以内近似线性；12→24 线程（物理核→SMT）只多约 40%，与"整数乘加密集"负载的预期一致。
- 早期（通用 GMP、w=8）测得的 16 线程结果**比 12 线程更差**（10.44s vs 10.02s）——12 物理核 + 4 个 SMT 线程造成映射不均；24 线程（正好 2×SMT）反而最好。因此 `edwards_threads=0`（自动取满逻辑核）是合理默认。
- 高并发下 zen3 GMP 相对通用 GMP 只有 ~1.04-1.06×（交替 A/B 实测：24 线程 8.00s→7.68s，12 线程 10.05s→9.51s），说明**多线程满载时瓶颈变成功耗/降频与共享资源，而不是单核 mpn 内核**。单核优化与多核吞吐要分开看。
- 交接（`p95_dir`）与并行正交：多曲线时每条曲线写自己的 `e{n:07d}_c{6-digit}.tmp`，后台线程按序交接；实测 6 曲线 + 4 曲线两个任务（B1=1e6）共 7.7 s 完成，存档/`worktodo.add` 均正确。（§11 重构后该"后台交接线程"已从 `ecm.exe` 移出到独立程序。）

### 10.4 NAF 窗口重新调优：默认 w 从 8 改为 12（~2~5%）

换上 zen3 内核后重扫窗口。方法：每个 w **交替**执行 3 轮取 min/median（单配置连跑会把降频误差当成 w 的差异——早期 §9 的表就是这么得出"w=8~10 甜点"的）。M347/M677/M2203 的 run-to-run 抖动只有 ~0.5%，非常可信：

| N | w=4 | w=6 | w=8 | w=10 | w=12 | w=14 | w=16 | w=18 |
|---|---|---|---|---|---|---|---|---|
| M347 (B1=1e6) | 0.672 | 0.640 | 0.622 | 0.612 | **0.609** | — | — | — |
| M677 (B1=1e6) | — | 1.547 | 1.501 | 1.471 | **1.467** | — | — | — |
| M991 (B1=1e6) | — | — | 2.785 | 2.730 | **2.705** | 2.707 | 2.823 | 3.338 |
| M2203 (B1=2e5) | — | — | 2.120 | 2.090 | **2.076** | 2.130 | — | — |
| M4003 (B1=2e5) | — | — | 6.963 | 7.045 | **6.729** | 7.090 | — | — |

（单位秒；除 M4003 抖动较大外均为 3 轮交替取 min，M4003 的 w=12 在两轮里也都最优。）

结论与改动：

- **w=12 在所有测试尺寸上都最好**（相对 w=8：M347 −2.1%、M677 −2.3%、M991 −2.9%、M2203 −2.1%、M4003 −3.4%），理论侧也吻合：每 bit 代价 = 8C + 8C/(w+1)，w=8→12 只降 3.1%，而字典构造开销可忽略（2^(w-2)×~12 次域乘 vs 主循环 ~1.28e7 次）。
- **w≥14 收益转负**（M991: w=14 持平、w=16 −4%、w=18 −23%）。不是加法变多，而是字典内存：`aff` 表每项 3×`mont_t` = 3840 B（即使只用到前 16 limb），w=16 时 63 MB、w=18 时 251 MB，分配 + 缺页 + 超出 L3 的代价超过了节省的加法。
- **并行下同样有效**：24 曲线 / 24 线程，w=8 → 7.31 s，w=12 → 6.96 s（−4.8%，交替 3 轮）。
- 改动：`src/cpu/ecm_edwards_cpu.cpp` 的 `g_edwards_naf_w` 默认 8→12；新增 `--edwards-naf-w N` / `ecm.ini` 的 `edwards_naf_w`（0 = 内置默认）便于继续调优。

> 由此看到的下一步：w 的上限被 `mont_t` 的**固定 160-limb 尺寸**卡住（真正用到的只有 n limb，却按 160 limb 的跨度排布）。把域运算按 `nlimbs` 模板化后字典能小一个数量级，w=14~16 才可能重新变优——这与 §9 的"小 N 缓存局部性收益"是同一个根因。

### 10.5 仍未尝试 / 已知可做但未做

- **GMP `--enable-fat` + `fat.c` 补 `case 0x1a`**：可同时拿到 Zen 5 内核与老机器兼容性（见 §10.2 注）。
- **按 N 的 limb 数模板化域运算**：把 `ed_*_mont` 栈按 `ED_MONT_MAX_LIMBS ∈ {16,32,64,160}` 各实例化一份、运行期选最小可容纳者。预期有两处收益：(a) §9 所示小 N 的缓存局部性（10~20%）；(b) 让 §10.4 的 NAF 字典按真实 limb 数排布，w=14~16 才可能重新变优。
- **AVX-512 IFMA / 批量曲线向量化**：要绕开 GMP 标量 `mpn` 另写多精度层，属于另一个量级的工程（GPU 端已经在做批量 CIOS）。**2026-09-23 重新评估并启动**，见 §12/§13。

## 11. 职责拆分：`ecm.exe` 只落本地盘，Prime95 交接独立成 `ecm_p95feeder`

### 11.1 为什么拆

原实现把"算 stage-1"和"喂 Prime95"耦合在一个进程里：`ecm.exe` 直接把 MIDSTAGE 存档写进 p95 目录并追加 `worktodo.add`，多曲线时还要一个后台线程按序复制 + 轮询等 p95 消费。问题：

- `ecm.exe` 需要知道 p95 的目录、要处理"p95 是否已消费"的状态机，还要和并行 worker 抢文件；
- `worktodo.add` 是**全局**追加，无法指定哪个 Worker 跑，多个任务会挤在一起；
- 一旦 p95 没运行，交接线程会一直等（或丢下临时文件），状态不透明。

拆分后各管一件事：

```
ecm.exe（stage-1 计算）                    ecm_p95feeder（搬运）
  ↓ 写本地                                  ↓ 轮询本地 + p95
<tmp_dir>/e{n:07d}_c{curve}.tmp  ────►  复制为 <p95_dir>/e{n:07d}
<tmp_dir>/e{n:07d}_c{curve}              + 往 <p95_dir>/worktodo.add 的
 (STAGE1 自检查点)                          [Worker #N] 段追加 ECM= 行
```

### 11.2 `ecm.exe` 侧（已改）

- 所有产物只写 `<tmp_dir>`：`e{n:07d}_c{curve:06d}.tmp`（MIDSTAGE，state=2）与 `e{n:07d}_c{curve:06d}`（STAGE1 自检查点，state=1）。**不再写 p95 目录、不再碰 `worktodo.add`**。
- `p95_dir` 键/`--p95-dir` 参数已废弃（保留仅为向后兼容；用了会打印一行提示）。新增 `tmp_dir`（`ecm.ini`）/`--tmp-dir`，默认 `.`（即 exe 目录 / 当前目录）；设 `tmp_dir =`（空）则完全不落盘（也无自检查点）。
- 删除了原后台交接线程（`handoff_worker`/`handoff_start`/`handoff_stop`/`handoff_enqueue`）与 `read_save_state`，`run_edwards_stage1` 因此更短。
- 单曲线 / 多曲线统一命名：不再有"单曲线写 `e{n:07d}`"的特例，一律 `_c{curve:06d}`，避免并行写冲突与恢复歧义。

验证：M677→1943118631、M991→8218291649 不变式通过；8 曲线并行 4.08 s（8 个 `.tmp`）；`tmp_dir` 为空时工作目录零残留；M4003 上 `gpuckpt_seconds=1` 可见 `e0004003_c000001`（STAGE1）被周期性重写。

### 11.3 `ecm_p95feeder` 侧（新增）

独立可执行文件（`src/core/ecm_p95feeder.cpp` + `src/core/p95_worktodo.{h,cpp}`，只依赖 GMP），循环执行：

1. 读 `<p95_dir>/worktodo.txt` 的 `[Worker #N]` 段、`<p95_dir>/prime.txt` 的 `MaxHighMemWorkers`/`NumWorkers`；
2. 统计"在飞"任务；(a) `worktodo.txt` 里**我们这种** handoff 行（`ECM=`/`ECM2=` 且 `curves==1` 且 `sigma!=0`）＋(b) `worktodo.add` 里同类行；
3. 找**空闲 Worker**（该段有效工作行数 = 0，可被 `worker_allow` 限制）；
4. 扫本地 `*.tmp`（按 mtime 从旧到新），逐个校验后投递。

投递动作：先 `copy <tmp_dir>/x.tmp → <p95_dir>/e{n:07d}`，再往 `<p95_dir>/worktodo.add` 追加

```
[Worker #3]
ECM=1,2,991,-1,1000000,0,1,105413044550089,"8218291649"
```

**为什么走 `worktodo.add` 而不是改 `worktodo.txt`**：Prime95 官方 `undoc.txt` 明确写了"Prime95/mprime will periodically look for worktodo.add and append the entries from **each `[Worker #]` section`**, then the worktodo.add file will be deleted"。所以带 `[Worker #N]` 段的 `worktodo.add` 就是官方支持的、无需与运行中的 p95 抢写 `worktodo.txt` 的投递通道。

安全规则（都会打日志）：

| 规则 | 目的 |
|---|---|
| 存档的 N 与 B1 必须与配置的 `worktodo` 文件里某条 ECM 任务一致 | "保存文件需要 N 和 B1 与 worktodo 对应上"；防止把陈旧/别的 B1 的结果推给 p95 |
| 同时在飞 ≤ `max_in_flight`（默认取 p95 的 `MaxHighMemWorkers`，缺省 1） | stage-2 是高内存阶段，`MaxHighMemWorkers` 就是 p95 对此的限制 |
| **同一 N 同时在飞 ≤ 1** | p95 的 ECM 存档名按指数固定为 `e{n:07d}`，两个在飞会互相覆盖 |
| 只投递到当前有效的 `[Worker #N]` 段 | p95 只从段里取活；段非空说明该 Worker 正忙 |

匹配到 worktodo 条目时，会把该条的**已知因子串**一起写进 `ECM=` 行，保证 p95 算出的 N 与我们 stage-1 时一致。

### 11.4 配置与用法

```
ecm_p95feeder.exe [--ini feeder.ini] [--once] [--dry-run] [-v]
```

`feeder.ini`（缺失时自动生成带注释的模板）：

| 键 | 默认 | 说明 |
|---|---|---|
| `tmp_dir` | `.` | 本地 `*.tmp` 所在目录 |
| `p95_dir` | （必填） | Prime95 工作目录 |
| `worktodo` | `worktodo.txt,worktodo.finished.txt` | 校验 N/B1 用的文件（逗号分隔）；相对路径依次在 **`tmp_dir` → exe 目录 → 当前目录** 里找，取第一个存在的（启动时把命中的路径打印出来）；留空 = 不校验 |
| `poll_seconds` | `5` | 轮询间隔 |
| `max_in_flight` | `0`（= 取 `MaxHighMemWorkers`） | 同时在飞上限 |
| `worker_allow` | 空（= 任意空闲 Worker） | 限定 Worker 号，如 `1,2` |
| `log_file` | exe 目录 `feeder.log` | 日志 |
| `keep_tmp` | `0` | 1 = 投递后保留本地 `.tmp` |
| `dry_run` | `0` | 1 = 只报告不落地 |

### 11.5 测试

- **单测** `tools/test/p95_worktodo_test.cpp`（35 项全过）：段解析（含 `;;MOVED;;[Worker #N]`）、handoff 判据（`curves=1 + sigma!=0`，排除 `curves=664` 的新分配 / 注释行 / `Pminus1` / `ECMSTAGE2=`）、`worktodo.add` 合并写回（新增段 / 追加到已有段 / 截断）、`prime.txt` 键读取、存档名生成。
- **集成测试** `tools/test/test_feeder.ps1`（自包含：先跑真实 `ecm.exe` 生成存档，再驱动沙箱 p95 目录 `[Worker #1..#4]` + `MaxHighMemWorkers=2`）走 8 个周期，退出码即结论，验证
  dry-run 不落地 → 首轮投递 2 条（分别进 `#1`/`#2`，存档 `e0000677`/`e0000991` 落盘，`.tmp` 删除）→ 满载不再投递 → 模拟 p95 消费 `worktodo.add` → 仍在飞 → 模拟完成一个任务后自动补位 → 只剩 B1 不匹配的存档时必须拒绝（`no worktodo entry matches N=991 B1=200000`，不写 `worktodo.add`、保留 `.tmp`，脚本断言 `PASS`）。
- **边界**：B1 不匹配的存档被跳过并保留 `.tmp`；`worker_allow=2` 且 Worker #2 忙时输出 `no empty [Worker #N] section`；B1 能在 `worktodo.finished.txt` 里匹配到时正常投递；用户给的参考格式（`ECM=AID,1,2,12323,-1,55000000,0,664,"f1,f2"`）不会被误判为在飞任务，其所在 Worker 被正确视为忙。
- **实测中修掉的两个字段错位 bug**：尾部的已知因子串（唯一的引号字段）长度可变，若按固定下标取 `curves`/`sigma` 会把 `curves=4,"8218291649"` 误判成 handoff（`sigma` 读到了第一个因子），且因子串无法透传。现在先把引号尾段剥离再按位置解析，因子串原样转发到 `ECM=` 行（实测输出 `ECM=1,2,991,-1,1000000,0,1,105413044550089,"8218291649,340840085969272441649"`）。
## 12. REDC 分派：改用 GMP 自己的归约内核（已落地，6000~10000 bit +15~34%）

### 12.1 问题

`ecm_edwards_mont.h` 的 REDC 是手写二次循环（n 次 `mpn_addmul_1`），而 `mpn_mul_n` 是次二次的。到 8000 bit（125 个 64-bit limb）时 `mul_n ≈ 4,000` limb-mults 而 REDC `= n² = 15,625` → **REDC 占了整个模乘的 ~80%**，且它是唯一还在二次增长的部分。

### 12.2 做法

按 GMP 自己的调度阈值（`mpn/x86_64/gmp-mparam.h`: `REDC_2_TO_REDC_N_THRESHOLD = 79`）分派：

- `n < 79` → `mpn_redc_1`（GMP 的调优实现，其 `MPN_REDC_1` 约定为"返回值非零则再减一次 N"）
- `n ≥ 79` → **`mpn_redc_n`**（GMP 的**次二次**归约，内部走 `mpn_mulmod_bnm1`），需要 `ip = N^{-1} mod B^n`（由 `mpn_binvert` 预计算）

`tools/bench/mont_redc_ab.c` 在同一 GMP 构建上对四种归约做了对照（含与 `mpz` 参照的逐位校验）：

| limbs | bits | 原二次循环 | `mpn_redc_1` | `mpn_redc_n` | 显式 SOS |
|---|---|---|---|---|---|
| 20 | 1280 | 0.36 µs | **0.32 (1.10×)** | 0.38 (0.95×) | 0.39 (0.95×) |
| 47 | 3000 | 1.59 | **1.47 (1.08×)** | 1.54 (1.03×) | 1.60 (0.99×) |
| 63 | 4000 | 2.80 | **2.53 (1.11×)** | 2.60 (1.08×) | 2.53 (1.11×) |
| 94 | 6000 | 5.64 | 5.19 (1.09×) | **4.60 (1.22×)** | 5.35 (1.05×) |
| 125 | 8000 | 9.54 | 8.83 (1.08×) | **7.68 (1.24×)** | 9.01 (1.06×) |
| 154 | 9850 | 13.83 | 13.12 (1.05×) | **10.67 (1.30×)** | 12.81 (1.08×) |

（显式 SOS＝两次次二次乘法 + `mpn_mullo_n`，**实测不划算，已放弃**。）

端到端交替 A/B（`ecm_edwards_speed`，B1=20000，3 轮取 min，同一份源码只改 `ECM_MONT_USE_GMP_REDC`）：

| bits | n64 | 原二次 | 新分派 | 加速 |
|---|---|---|---|---|
| 4003 | 63 | 0.657 s | 0.652 s | 1.01× |
| 6000 | 94 | 1.320 s | 1.145 s | **1.15×** |
| 8000 | 125 | 2.234 s | 1.907 s | **1.17×** |
| 10000 | 157 | 3.650 s | 2.722 s | **1.34×** |

正确性：M677→1943118631、M991→8218291649、M4003→16756559 三条不变式全部通过。

### 12.3 ⚠️ 依赖声明（差异清单条目）

`mpn_redc_1` / `mpn_redc_n` / `mpn_binvert` / `mpn_binvert_itch` **不在 `gmp.h` 里** —— GMP 明确声明这些是 internal、接口可变甚至可能消失。我们用 `__MPN()` 宏声明（注意 C++ 下必须包 `extern "C"`，否则名字修饰会导致 LNK2019），靠 GMP 构建导出这些符号来链接。GMP 6.3.0 自 2023 年起未再更新，风险可接受。

因此保留 `ECM_MONT_USE_GMP_REDC=0` 编译开关可退回原来的二次实现（自检/对照/换 GMP 版本时用）。

## 13. SIMD（AVX-512 IFMA）批量曲线：计划与边界

### 13.1 目标尺寸段（已与用户确认）

用户实际工作集中在 **3000~8000 bit**（= 47~125 个 64-bit limb）；更大 bit 用 Prime95 的 FFT，不在本仓库范围。2000 bit 左右若有足够性能也可接受。

### 13.2 架构结论

- **lane = 一条曲线**，8 lane（单 zmm 一个 limb 号），**端到端 SoA**，只在每条曲线的入口（`mont_to`）与出口（`mont_from`+`gcd`）跨布局。理由：仓库自己的记录（`DEV_CPU_MONT_AVX_PLAN.md`）已判定"垂直 SIMD"无用，实测 0.48×/0.20×。
- radix **2^52 IFMA**（`vpmadd52luq/huq`，8 lane，1/cycle 实测），而非 32-bit `vpmuludq` CIOS（后者每曲线约 4 倍乘法指令数，打不过 GMP）。
- **没有 64×64→128 的 SIMD 乘法**（`vpmullq` 只给低 64 位，IFMA 乘数硬性限 52 位）。64 位优势落在：64 位懒进位累加器（n ≤ ~1000 时整个乘法相位无需中途归一化）、64 位归一化/搬运指令（`vpaddq` 实测 2/cycle）、以及标量侧（`np0`/`R`/批量求逆/逐曲线抽取）。
- 本机实测：AVX512F/DQ/**IFMA**/BW/VL/VBMI2 全部支持，XCR0=0xe7；**MSVC 14.51 `/arch:AVX512` 可编出 IFMA intrinsic**（已编译验证）。AVX-512 代码必须隔离在单独 TU（照 `cpu_addsub_avx512.cpp` 的规矩）。
- **AVX2 变体（4 lane × 32-bit）仅记入 TODO**：AVX2 无 IFMA，每曲线乘法指令数约为 IFMA 的 2.4 倍，大概率过不了闸门。

### 13.3 关键约束：寄存器窗口

CIOS 的累加器需要 `n+2` 个 zmm；加上广播/掩码/进位临时量，**寄存器窗口只在 n ≤ 26（≈1350 bit）时成立**。而用户的目标段是 n₅₂ = 39~154 → **必须采用分块（chunked）实现**：把操作数切成 ~20 limb 的块，块内用寄存器窗口算部分积，再累加进 L1 数组。额外开销 `O(nB)/n²`，n=154 时约 15%。

### 13.4 闸门（可行性闸门，与用户确认）

- **指标**：每曲线-模乘的**有效 cycles**，交替 A/B；SIMD 侧含 SoA 装载、累加器 L1 往返、末尾归一化的全部开销后除以 8；GMP 侧必须是**生产路径的原语**（`mpn_mul_n`/`mpn_sqr` + 上文的新 REDC），**不得用 `mpz_mul`/`mpz_mod` 当基线**。
- **不计入闸门**：字典构造、批量求逆、逐曲线 `mont_from`+`gcd`、存档 I/O、线程调度、NAF 位生成（这些留给端到端"不得回退"检查）。
- **判据**：在 n₅₂ = 77（≈4000 bit）与 125（≈6500 bit）上均 **≥ 2×**；n₅₂ = 154（8000 bit）只记录。不达标即停手，把现状作为终态，内核留作 bench 资产并归档负面结论。
- 现实预期：账面 4n² 给 4.2×，但累加器在 L1 + 每个乘积约 1 次 64B load → 按 IFMA 利用率 u 折算，**u=0.5 时 1.3~1.7×，u=0.8 时 2.4~2.5×**。也就是说 8000 bit 端需要 u ≥ 0.8 才能过闸门 —— 这是 M1 必须回答的问题。

### 13.5 里程碑

1. **M1**：`simd_mont_ifma`（SoA、radix 2^52、分块 CIOS）+ mpz 同-R 参照逐位校验 + 闸门测量（n₅₂ = 77/125/154，另含 n₅₂=20 对照）。不写一次性探针，直接按生产接口写（bench main 用 `#ifdef` 隔离）。
2. **M2**：Edwards 点运算 SoA 化（dbl/add/add_affine）+ 字典 SoA 化 + 与标量路径的 `u`/`gcd`/因子一致性验证。
3. **M3**：接入 driver 作为可切换后端（静态分批、批内 lane 固定、8 个 `_c{k}` 存档/批、resume 从存档读回 sigma），重扫最优 w（字典内存模型变了），端到端"不得回退"检查。

### 13.6 差异清单（验收以"外在行为相同"为准）

1. radix 2^52（`R = 2^52n`）vs 现有 2^64 → 蒙氏域数值不同
2. `(Qx,Qz)` 投影代表相差一个公共因子 λ（与 p95、与标量路径都不同）；**`u = Qx·Qz^{-1} mod N` 与 `gcd(Qz,N)` 相同**
3. 存档**字节**不保证相同 —— 只保证格式/语义兼容、p95 可读可续（"与 p95 字节级一致"由既有的 `e0000347` 合成写测试独立保证，不受影响）
4. checkpoint 粒度：SIMD 一批 8 条曲线 → 8 个 `_c{k}` 文件
5. 曲线数语义：`auto` 默认"整批 SIMD + 余数标量"，**严格等于请求曲线数**；`edwards_simd_pad=1` 才补满
6. 线程模型：批为调度单位 + 批内 lane 静态 + 可选绑核；标量路径仍是曲线为单位 + 原子取号
7. 后端选择：`edwards_backend = auto|simd|gmp`，`simd` 模式下 ISA 不可用**硬报错、不静默降级**
8. 性能口径：闸门看每曲线-模乘 cycles，端到端另算；端到端天花板 = `1.35/0.35 = 3.86×`（stage-2 ≈ 0.35 × stage-1）
9. REDC 依赖 GMP internal 符号（§12.3）
10. 偶数 N：Montgomery 不可用 → **报错**（不复现旧路径的静默错算）

### 13.7 已知边界与待办

- **偶数 N / N ≤ 3**：`mont_init` 原来对偶数 N 会静默算错（`mpz_invert` 返回值未检查 → `nprime0` 为垃圾）。新后端将加硬守卫，**旧路径同一处 bug 也要一起修**（否则违反第 6 条的"外在行为相同"）。
- **`ED_MONT_MAX_LIMBS = 160`（10240 bit）**：超过即回退 mpz double-and-add（`n=12323` 就会触发），且**该回退无 checkpoint、不响应 SIGINT**。用户的 3000~8000 bit 不触发，但这是个运维风险，待修。
- **本地暂存名已加 B1**：`e{n:07d}_B{B1}_c{curve:06d}[.tmp]`（同一个 n 配不同 B1 的多个任务会互相覆盖，p95 参考 worktodo 里 Worker #2/#3/#4 正是这种形状），并加了"头部不一致则拒绝覆盖"的保护。
- **AID 规则**：投递行**永不透传真实 AID**（置空让 p95 重新分配）；源条目是 `N/A` 则原样保留。
- **AVX2 4-lane 变体**：TODO，接口按 `<LANES, LIMBS, FIELD>` 模板化预留，但不在 M1/M2 实现。
- **手写汇编**：M1/M2 只用 intrinsics（MSVC x64 无内联汇编，手写需独立 `.asm` + `ml64`）；仅在测得 IFMA 利用率 < 0.5/cycle 且 profiling 定位到寄存器溢出/调度时才考虑，且保留 intrinsics 版做交叉验证。

