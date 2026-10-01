# 自研 stage 2（N ≤ 10 000 位）可行性：gwnum + 改 `polymult` 路线（2026-09-30）

> 问题（用户提问）：*探究使用 gwnum 库和修改 `polymult64.lib` 实现优化小位宽（10000 bit 内）stage 2 的可行性*。
>
> 前置结论见 `docs/DEV_GWNUM_FEASIBILITY.md`（复用它做 stage 1 域运算的可行性）。
> 本文每条结论带证据：源码给 `文件:行`，性能给"实测 + 复现命令"，推断一律标注。

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| 现在 stage 2 是谁在做、多快？ | **Prime95**（我们的 `p95_worktodo_path` 交接）。用**它自己的 results.json.txt 时间戳**实测：24 h 窗口 2800 条曲线 / B2 均值 4.4e11 ⇒ **28.6 s/曲线**（整机吞吐）⇒ **一个 960 曲线任务 ≈ 7.6 h**，而 stage 1（GPU）≈19.4 h |
| Prime95 为我们这些任务选了什么参数？ | 实测（它的 results.json）：M5153 B1=2.6e8 → B2=2.33e12、**D=1771770、poly_size=167040**、stage2-fft=320；M5261 B1=2.6e8 → B2=1.94e12、D=1411410、poly_size=132480；M8273 B1=1.1e8 → D=510510、poly_size=46080、s2fft=512 |
| `poly_size` 与模数有关吗？ | **无关**。`poly_size = D 的 numrels`（`ecm.cpp:6199-6201`，纯查表 `poly_D_data`，`ecm.cpp:591-680`）；polymult 自己的 FFT 长度是 `polymult_fft_size(2·poly_size)`（`polymult.c:410-426`），**与模数无关**，最小 4。模数只通过 **`EXTRA_BITS` 安全余量**（`gwnum.h:580`、`gwnum.c:8434-8441`）进入 ⇒ **小位宽天然有利**（同一 FFT 档位能容纳更大 poly_size） |
| 用 gwnum + 改 polymult 自研 stage 2 值得吗？ | **不值得**（对 ≤10 000 位）。理由：① 你要写的是**同一算法、同一引擎**（Prime95 的 stage 2 就是 gwnum + polymult），最好只能打平；② 要补的算法量很大：relp/D 代数、nQx、F/G/R 三重乘积树、Bernstein **scaled remainder tree** 下降、roundoff 门，共约 **1050 行**极细的编排（`ecm.cpp:8805-9851`）；③ polymult **不是独立可用的库**（见下）；④ 收益上限是"把 7.6 h 变少"，而它在流水线里排在 19.4 h 的 GPU stage 1 之后 |
| polymult 能改吗？ | **技术上能**：`polymult.c` 只有一个文件、纯 C + intrinsics、**不需要汇编器**，MSVC 编 5 个 ISA 变体（`compil64:211-239`）；也可以编一份**改名私有副本**链进同一个 `gwnum64.lib`。许可允许（BSD 式两条件，`gwnum/readme.txt:86-110`）。**但**：它自带的调优参数被作者标注为未测（`KARAT_BREAK=32 / FFT_BREAK=64`，`polymult.c:532-539` 都写着 `//GW: Fix me`），改这些**只可能带来 10–30% 级别**的差别，不是数量级 |
| 那真正的机会在哪？ | **GPU stage 2**（不是改 polymult）：stage 2 的系数乘法是**成千上万个互相独立的小模数 modmul**（5153 位 ≈ 0.4 µs/次），正是我们 AVX512-IFMA 8 lane 与 CGBN 的强项；而你的 **4060 Laptop 目前完全空闲**（2026-09-30 10:5x 实测 `nvidia-smi`：index 1 = `0 %`, `9.10 W`, `0 MiB` used；同一时刻 index 0 = 4070 Ti `99 %`, `178 W` 在跑 stage 1），stage 1 只占 4070 Ti。把 stage 2 放 GPU 可与 stage 1 **并行**，收益是"整条流水线缩短 28%"而不是"某个引擎快 10%" |
| 一个必须先做的测量 | 用 Prime95 自己跑一个 ≤10 000 位模数、开 `PolyVerbose=1`/`Stage2Estimates=1`，读出 `"Estimated stage 2 vs. stage 1 runtime ratio"`（`ecm.cpp:6217`）与 "Using %uMB of memory. D: …, degree-… polynomials"（`ecm.cpp:8815`）。这两个数决定"自研"到底有没有意义；本文的 §2 给了它当前的实测吞吐作为对照 |

---

## 1. stage 2 到底要做什么（源码事实，全部来自 `p95v3104b02.source`）

### 1.1 结构

* 状态机：`ECM_STATE_MIDSTAGE(2) → ECM_STATE_STAGE2(3) → ECM_STATE_GCD(4)`（`ecm.cpp:1464-1471`），两种实现 `ECM_STAGE2_PAIRING(0)` 与 `ECM_STAGE2_POLYMULT(1)`；`numvals ≥ 200` 时默认选 polymult（`ecm.cpp:6010-6012`）。
* **D 不是算出来的，是查表**：`poly_D_data[]`（`ecm.cpp:591-680`），例如 `{2·3·5·7·11·3, 4·6·10·3, 13, 17} // 6930, 720`（`ecm.cpp:621`）；上限 `POLY_MAX_D = 164894730`、`POLY_MAX_RELPRIMES = 14100480`（`ecm.cpp:682-683`）。`poly_size = numrels`（D 的、小于 D/2 且与 D 互素的个数）。
* 每曲线的工作（`ecm.cpp:8805-9851`）：
  1. `nQx`：D 的互素倍数上的点 `Q^i`（1,5 mod 6 走法，`ecm.cpp:8885-8924`）；
  2. `F(X) = Π(X - Q^{r_i})` 乘积树（`ecm.cpp:9075-9204`，level 0 用自定义 helper，之后用 `polymult(MONIC|MONIC|NO_UNFFT|SAVE_PLAN)`）；
  3. `1/F(X)`（Newton 加倍，`ecm.cpp:9212-9251`）；
  4. **giant-step 外循环**（`ecm.cpp:9334`，迭代次数 `num_polyG = ceil(numDsections/poly_size)`，强制 ≥2，`ecm.cpp:5771-5775`）：生成下一批 `Q^{mD}` → 建 `G(X)` 树 → **3 次全尺寸 `poly_size × poly_size` polymult** 把 `H = G·H mod F`（`ecm.cpp:9443-9461`）；
  5. **scaled remainder tree 下降**（Bernstein，`ecm.cpp:9518-9738`）：两趟、第二趟按 1/8 切片，用 `polymult_several(CIRCULAR|MULHI)`；
  6. **每曲线 1 次 GCD**：把最终 H(X) 的系数乘起来（`gwmul3`，`ecm.cpp:9740-9771` + helper `ecm.cpp:6763-6786`），然后 `gcd(...)`（`ecm.cpp:9858`）。
* 复杂度自评（源码自带）：建 F 树 `log2(P)·P` 次 polymult 等效；建 R 树 ≈ 2 次全尺寸；**每个 outer loop 3 次全尺寸 polymult**；下降 `log2(P)` 层 × 2 次半尺寸（`ecm.cpp:5823-5884`）。polymult 的单价用 `polymult_cost = 2.145 + 0.58·log2(P/105)` 估计，并按"每线程 polymult 内存 vs L2"惩罚最多 2.05×（`ecm.cpp:5801-5809`）。

### 1.2 换成数字（用 §2 实测的 Prime95 参数代入）

以 M5261 / B1=2.6e8 / B2=1.94e12 / **D=1411410、P=132480** 为例：

* `numDsections = (B2_end - B2_start)/D ≈ 1.4e6`；`num_polyG ≈ 11`。
* 每曲线：≈11 次 outer loop ×（P 次 RLP 点加法 + G 树 + 3 次 P×P polymult）+ 一次下降树 + 1 次 GCD。
* ⇒ 每曲线 ≈ **30 次全尺寸 P×P polymult**（`(num_polyG-1)·3`，P=1.3e5；第一轮不需要 H 归约）+ 建 F/R 树与下降树的 `O(P·log P)` 量级系数乘 + 约 `num_polyG·P ≈ 1.5e6` 次 RLP 点加法（上述次数是从 `ecm.cpp:9334-9475` 的循环结构直接数出来的估计，不是实测）。

**这解释了 Prime95 的吞吐**：28.6 s/曲线（整机）。也就是说 stage 2 的每一秒都花在"大量小模数（5153 位）算术"上 —— 与 stage 1 的 GPU 内核在数学上是同一类工作。

---

## 2. Prime95 的真实参数与吞吐（实测，来自它自己的文件）

数据源：`D:\code\GIMPS\p95v3104\results.json.txt`（3258 条 ECM 记录，跨 2026-02-08 … 2026-09-30）与 `prime.txt`。

### 2.1 它为我们交接的任务选的参数（`curves ≥ 300`，即 GPU 交接）

| 指数 | B1 | B2 | D | poly_size | stage2-fft | 曲线 | 参数来源 |
|---|---|---|---|---|---|---|---|
| M5153 | 2.6e8 | 2.325e12 | 1771770 | 167040 | 320 | 960 | `ECMSTAGE2` 交接后由 Prime95 自选 B2 |
| M5261 | 2.6e8 | 1.938e12 | 1411410 | 132480 | 320 | 960 | 同上 |
| M5351 | 1.1e8 | 9.99e11 | 1411410 | 132480 | 320 | 960 | 同上 |
| M8273 | 1.1e8 | 5.29e11 | 510510 | 46080 | 512 | 960 | 同上 |
| M12323 | 1.1e8 | 5.87e11 | 510510 | 46080 | 768 | 480 | 同上 |

（对照组：PrimeNet 发的 3.5M 位任务 B1=1e5、B2=7.57e7 时 `d=6930`、`poly-size=720`、`stage2-fft=204800`。）

**两个可以直接利用的规律**：
1. `stage2-fft` = 该模数的 gwnum FFT 长度（320/384/512/768）——**小位宽的 FFT 很小**（5120 位以下只要 320–384 words，`gwfftlen` 实测 384@5153 位），所以每次系数乘/每次 polymult 都便宜；
2. B2/B1 被 Prime95 推到 **≈7000–9000**（因为它有 12 GB 内存预算：`prime.txt` 的 `Memory=12288`、`MaxHighMemWorkers=6`），这直接决定了 P 与 D 取大档。

### 2.2 吞吐（从完成时间戳反算）

| 窗口 | 完成曲线数 | B2 均值 | 曲线/小时 | s/曲线 |
|---|---|---|---|---|
| 24 h | 2800 | 4.40e11 | 125.7 | **28.6** |
| 72 h | 8350 | 4.04e11 | 119.2 | 30.2 |
| 168 h | 11830 | 4.20e11 | 78.5 | 45.9 |
| 336 h | 15160 | 4.80e11 | 51.3 | 70.2 |

（长窗口含"Prime95 没在跑"的空闲期；24 h 窗口最接近"运行中吞吐"。**运营事实**：2026-09-30 11:00 实测 Prime95 进程未运行（stage 2 靠手动启动），所以交接任务会在 CPU 侧排队 —— 这也是"自研 stage 2 可以并行化"的另一个理由。）⇒ **一个 960 曲线交接任务 ≈ 7.6 h 的 CPU 时间**，而该任务的 GPU stage 1 ≈ 19.4 h（`docs/ECM_CGBN_OPTIMIZATION.md:610-615`）。

---

## 3. 引擎复用分析：gwnum + polymult

### 3.1 API 与**必需**的调用顺序（这是最容易踩的坑）

`polymult` 的 API 是公开且自足的（`polymult.h`）：`polymult_init`、`polymult_set_cpu_flags`、`polymult_default_tuning(L2,L3)`、`polymult_set_max_num_threads`、`polymult_launch_helpers`、`polymult_safety_margin`、`polymult_fft_size`、`polymult_mem_required`、`polymult` / `polymult2` / `polymult_fma` / `polymult_several`、`polymult_preprocess`。

按 `ecm.cpp:7830-7842` 与 `ecm.cpp:7951-7990` 归纳出的**必需顺序**：

```
gwinit → gwset_using_polymult → gwset_num_threads → gwset_polymult_safety_margin
       → gwset_minimum_fftlen / gwset_larger_fftlen_count → gwsetup(1,2,p,-1)
       → [可选 gwuser_init_FFT1] → polymult_init → polymult_set_cpu_flags
       → polymult_set_max_num_threads → polymult_default_tuning(L2,L3)
       → polymult_launch_helpers → polymult*
```

并在 `gwsetup` 之后**自检**：`gw_passes_safety_margin(h, polymult_safety_margin(P,P))`，不满足就把 FFT 加长重来（Prime95 的二分搜索 `max_safe_poly2_size`，`ecm.cpp:5453-5471`）。

**我实测踩到的四个失败模式（都记在 `tools/gwnum_probe/polymult_probe.cpp` 的注释里）**：

1. `gwset_using_polymult` 漏了 / 顺序不对 ⇒ 每个 gwnum 的头部尺寸与 polymult 的假设不一致（`gwnum.h:1122`）；
2. `polymult_default_tuning` 漏了 ⇒ 句柄被 `memset` 清零、所有调优阈值 = 0 ⇒ 规划器**不报错、直接死循环**；
3. `polymult_set_max_num_threads` 必须在**第一次 polymult 之前**调用（`pmdata->num_threads = 0` 同样挂住）；
4. **`polymult_launch_helpers` 是行并行的工作线程来源**（分发点 `polymult.c:4463` `HELPER_POLYMULT_LINE`）——不启动它，第一次 polymult 会派发工作后永久等待。

⇒ **`polymult64.lib` 不是一个"链上就能用"的独立库**；它需要 Prime95 自己那套 helper 线程脚手架（`ecm.cpp:9144-9148` 先设 `helper_work` 再 `polymult_launch_helpers`）。我的探针按官方文档顺序 + 上述四步全做齐之后，**仍停在 `polymult_launch_helpers` 不返回**（未解决；`tools/gwnum_probe/polymult_probe.cpp` 保留全部插桩，可继续查）。

### 3.2 能重建/能改吗

* **能**：`polymult.c` 单文件、纯 C + intrinsics、头文件只有 `stdlib/math/memory/cpuid.h/gwnum.h/gwutil.h/polymult.h`（`polymult.c:11-20`），5 个变体就是 5 条编译命令（`compil64:211-239`）：`/DSSE2`、`/arch:AVX /DAVX`、`/arch:AVX2 /DFMA`、`/arch:AVX512 /DAVX512`。**不需要 UASM**（那是重建 `gwnum64.lib` 才需要的，而我们不必重建它）。
* **更省事的第三方案**：编一份**改名私有副本**（`-Dpolymult=my_polymult …`）链进同一个 `gwnum64.lib`，完全绕开 `polymult_dispatch` 的 if/else 链。
* **许可**：`gwnum/readme.txt:86-110` 的 BSD 式两条件（源码再分发带版权/免责声明 + 用本软件找到梅森素数时发现权归 GIMPS），`gwnum.h:593 gwnum_is_gpl() (0)` ⇒ 改与再分发都允许（与 `docs/DEV_GWNUM_FEASIBILITY.md` §2.4 同一结论）。

### 3.3 小位宽的真实杠杆点

* **有利**：poly FFT 与模数无关（`polymult.c:410-426`），模数只通过 `EXTRA_BITS` 进入；小位宽 ⇒ 同一 FFT 档位下 `EXTRA_BITS` 更宽 ⇒ 可用的 `poly_size` 更大、内存更省、cache 命中更好。实测（`polymult_probe`，p=5153）：P=720 时 `polymult_fft_size(1440)=1440`、`polymult_safety_margin=2.478`、**`polymult_mem_required` 仅 284 KB**（4 线程），而 `gwsetup` 在 `fftlen=384` 下 `EXTRA_BITS=13.144`（P=64 时 256 words 只有 2.097 < 需要的 2×1.566 ⇒ 必须升到 384 才够 —— 这就是 §3.1 那个自检循环的作用）。
* **无专用快路**：全代码**没有**"小模数分支"；只有 FMA3 小 FFT 的搜索 hack（`ecm.cpp:7929-7934`，`SearchFMA3FFTs`）——**这也解释了 Prime95 给 M5153 选 `stage2-fft=320` 而我默认得到 384**：它在主动找更小的 FMA3 FFT（与我在 `DEV_GWNUM_FEASIBILITY.md` §3.3 实测的"3571 位时 FMA3 比 AVX-512 快 40%"是同一件事）。
* **可调空间有限但真实**：`KARAT_BREAK=32`/`FFT_BREAK=64`（`polymult.c:532-539`，两处都写着 `//GW: Fix me`）、`polymult_default_tuning` 的 `two_pass_start/max_pass2_size/mt_ffts_start/mt_ffts_end/strided_writes_end/streamed_stores_start`（`polymult.c:573-618`），作者原话是"almost certainly imperfect"。改这些是**百分比级**收益。

---

## 4. 与我们自己引擎的对比（引用 `DEV_GWNUM_FEASIBILITY.md` 的实测）

| 位宽 | gwnum FFT（FMA3 / AVX-512） | 我们 AVX512-IFMA（8 曲线批摊薄） |
|---|---|---|
| 3571 | 0.36 / 0.51 µs | ≈0.38 µs |
| 12323 | 1.37 / 1.14 µs | ≈4 µs（外推） |

**含义**：stage 2 的系数乘在 **≤3760 位**时我们与 gwnum 打平甚至略快，**≥5755 位**（M5153/M5261/M8273 这一档）gwnum 快 1.5–3×。也就是说，"用小位宽 + 我们自己的域运算替换 gwnum FFT"这条思路，只在 **≤3760 位**（M3571/M3583/M3613 那一档）成立，而那一档恰恰是 P、D 都较小的档。

---

## 5. 结论与建议

1. **不建议**用"gwnum + 改 polymult"自研 stage 2 来优化 ≤10 000 位：
   * 算法与引擎都和 Prime95 相同 ⇒ 上限是打平；要写的是 ~1050 行极细的编排（乘积树 + 下降树 + relp/D 代数）以及**必须自己承担 roundoff 安全**（polymult 自己**不做任何 roundoff 校验**，`polymult.c:29` 的 TODO #22；Prime95 靠 `gwset_polymult_safety_margin` + 每阶段 `gw_maxerr < 0.4375` 门，`ecm.cpp:9470`）；
   * polymult 还需要 Prime95 的 helper 线程协议才不阻塞（本文 §3.1 实测）；
   * 收益天花板 = 7.6 h/任务，且与 19.4 h 的 GPU stage 1 **串行**。
2. **更有价值的方向（建议优先评估）：把 stage 2 放到 GPU 上**，因为：
   * stage 2 的计算是**海量独立的小模数 modmul**（5153 位 × 上百万次），与我们已有的 CGBN/IFMA 内核同构；
   * 你的 **4060 Laptop 目前完全空闲**（实测 util 0–2%、功耗 1.5–9 W），可以**与 4070 Ti 上的 stage 1 并行**，把流水线从 19.4 + 7.6 h 压到 ≈ max(19.4, GPU stage2)；
   * 这条路线**不需要** gwnum/polymult，也不需要坚持 Prime95 的 D/P/B2 政策（可以自己按 GPU 的算术/内存特性重算最优 B2）。
3. **若确实要做（无论是 CPU 自研还是 GPU 自研），先把两个测量做掉**：
   * 用 Prime95 对同一个 ≤10 000 位模数跑一次并开 `PolyVerbose=1` + `Stage2Estimates=1`，读出 `Estimated stage 2 vs. stage 1 runtime ratio`（`ecm.cpp:6217`）与 `Using …MB of memory. D: …`（`ecm.cpp:8815`）——这是自研方案必须打平的**同机、同参数**靶子（本文 §2.2 的 28.6 s/曲线是它的整机稳态版）；
   * 写一个"Bernstein 树 + 我们域运算"的最小原型，量出**每次系数乘的 ns** 与**每曲线需要的次数**（本文 §1.2 给了次数模型），即可在几小时内判定能不能赢，而不必先把 1050 行编排写完。
4. **明确不做**：为了让 polymult 跑得更快而去重编它的 5 个 ISA 变体 —— 除非 §5.3 的第二个测量显示 polymult 的规划/调优就是瓶颈（现有证据不支持：它的内部 FFT 已经很小，内存只有几百 KB）。

---

## 6. 复现与未完成项

* 探针：`tools/gwnum_probe/polymult_probe.cpp`（与 `gwnum_probe.cpp` 同一构建脚本；链接 `gwnum64.lib` + `polymult64.lib` + 自带 GMP）。用法：`polymult_probe.exe <p> <poly_size> [iters] [max_threads]`。**当前状态**：能拿到 `gwfftlen`/`EXTRA_BITS`/`polymult_fft_size`/`polymult_safety_margin`/`polymult_mem_required` 与安全余量循环的判定，**但第一次 `polymult` 调用不返回**（卡在 `polymult_launch_helpers`；四个已知坑已全部按 Prime95 的顺序处理，仍未解决）。所以**本文没有 standalone polymult 的计时**，§2 的吞吐全部来自 Prime95 自己的完成时间戳（这反而更贴近"要打平的靶子"）。
* Prime95 **无法非交互启动**（Windows 版是 GUI 程序，重定向 stdout 后 20 s 内自行退出、不写日志），所以"让 Prime95 跑一个指定模数并读回参数/计时"这一步留给用户在 GUI 里做（`Advanced/Time` 或直接放一条 `ECMSTAGE2` 行）。
* 未验证：`adjusted_max_exponent` 表的具体数值（决定 `EXTRA_BITS`）；polymult 内部规划的完整协议（helper 线程与 `pmdata->scratch` 的关系）。
* 引用到的源码位置：`ecm.cpp:591-680`（D 表）、`:7830-7842`/`:7951-7990`（初始化顺序）、`:8805-9851`（stage 2 主体）、`:5823-5884`（成本模型）、`polymult.c:410-426`（poly FFT 长度）、`:532-539`（断点常量）、`:573-618`（调优）、`:4463-4478`（ISA 分发）、`:462-506`（内存核算）、`gwnum.h:580`/`gwnum.c:8434-8441`（安全余量）、`gwnum/readme.txt:86-110`（许可）。
