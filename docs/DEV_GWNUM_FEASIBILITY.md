# 复用 prime95/gwnum 的 AVX-512 IBDWT FFT：可行性评估（2026-09-30）

> 问题（用户提问）：*研究 prime95 源码下的 gwnum 库，使用其 AVX2 和 AVX512 的 IBDWT FFT 的可行性*。
>
> 本文的每一条结论都带证据：源码给 `文件:行`，性能给"实测 + 复现命令"，推测一律标注为推测。
> 实测数据与原始日志：`tools/gwnum_probe/`（探针源码 + `probe_output.txt`）。

---

## 0. 结论摘要

| 问题 | 结论 | 依据 |
|---|---|---|
| gwnum 有 **AVX2** 的 FFT 路径吗？ | **没有**。`CPU_AVX2` 只被检测（`cpuid.c:418`）与定义（`cpuid.h:58`），gwnum 目录里**再无第二处读取**；浮点 FFT 只有 SSE2(`x*`)、AVX+FMA3(`y*`)、AVX-512(`z*`)。AVX2 只作为 **polymult 的编译开关**（`compil64:230`）。清掉 AVX-512 得到的是 **FMA3** 路径（实测 `fft: FMA3 FFT length 160`） | 源码 grep + 实测 |
| gwnum 有 **AVX512-IFMA** 吗？ | **没有**。整个 gwnum 目录（含 `.asm/.mac` 与 `gwnum64.lib` 符号表）`ifma\|IFMA` **零命中**。IFMA 是我们自己那套 52 位肢域运算用的东西，不是 gwnum 的 | grep |
| gwnum 有 **ECM 专用 API** 吗？ | **没有**。`gwnum.h` 里 `ecm` 只出现 1 次且是注释（`:1198`）。prime95 的 ECM 是**应用层** `ecm.cpp`（16,815 行）用 62 个通用原语（`gwmul3/gwaddmul4/gwsubmul4/…` + `polymult`）自己搭的 | grep + `ecm.cpp:2403-2409` 等 |
| 在**我们生产任务的尺寸**（M3571/M3583/M3613）能更快吗？ | **不能，反而更慢**。gwnum 每次模乘 0.36–0.58 µs，而我们自己的 AVX512-IFMA **8 曲线批处理**等效 ≈0.38 µs/次；gwnum 又是**标量**（无法跨曲线批处理） | 本文 §3、§4 |
| 在 **M12323** 呢？ | **约 3× 收益**（gwnum 1.14 µs vs 我们 ≈4 µs/次），但这条 CPU 路径**不是生产路径** | §4 |
| 对**整体生产任务**有多大意义？ | **≈0**。生产是 GPU stage 1，宿主机大整数工作只占一个任务墙钟的 **≈0.016%**（`s=lcm` 首次 10.7 s / 缓存 0.26 s，曲线生成 <0.6%） | 仓库既有实测，§5 |
| 许可允许链进我们程序吗？ | **允许**（BSD 式两条件），但文本**没有明确写"可以链接"**，且权威规则指向 mersenne.org/legal（未取） | §2.4 逐字引用 |
| 建议 | **不引入生产**。若将来要做，触发条件是"**CPU-only 且 N ≳ 10 000 位**"的场景，或只借 `polymult64.lib` 做 stage-2 多项式乘法 | §7 |

---

## 1. 源码与实测环境

| 项 | 值 |
|---|---|
| gwnum 源码 | `D:\code\GIMPS\p95v3104b02.source\gwnum`（prime95/mprime v31.4，`GWNUM_VERSION "31.4"`，`gwnum.h:58`） |
| 预编译库 | 同目录 `gwnum64.lib`（x64 静态，26.9 MB）+ `polymult64.lib`，另有 32 位与 debug 版 |
| 本机 CPU | AMD Ryzen AI 9 HX 370（Zen 5，家族 26），24 逻辑核；`CPU_FLAGS=0x00d6ffaf` ⇒ AVX2/AVX512F/VL/DQ 全有 |
| 对比基线（我们） | 仓库自带 GMP 6.3.0 **zen3/BMI2+ADX** 定制构建（`third_party/gmp-zen3`）+ 自有 AVX512-IFMA 域运算 |

---

## 2. 源码事实

### 2.1 有哪些 FFT 路径

| 前缀 | 指令集 | 关键文件 | 备注 |
|---|---|---|---|
| `z*` | **AVX-512** | `zr4.asm`、`zr4dwpn.asm/2`、`zr2..zr16.mac`、`zonepass.mac`、`znormal*.mac`、`zmult*.asm/mac`、`zbasics.mac` | 汇编（UASM），表在 `gwtables.c:38-2090`（`zr4_build_onepass_sincos_table` 等） |
| `y*` | **AVX / FMA3** | `yr4.asm`、`yr4dwpn*.asm`、`yr*.mac`、`ymult*.asm` | AVX2 的 FFT **不存在**；FMA3 是 AVX 路径上的 `vfmadd231pd`（`ybasics.mac:16`） |
| `x*` / `hg*` | SSE2 / 上古 | `xmult*.asm`、`hg*.asm/mac`、`r4*.asm` | 32 位与老 CPU |
| C | 调度与表 | `gwnum.c`（`gwmul3:11683`、`gwinfo:964`、`calculate_bif:593`）、`gwtables.c`、`radix.c` | 无 intrinsics；`polymult.c` 是唯一按 ISA 编译 5 份的文件（`compil64:212-239`） |

IBDWT 在头文件里写得明明白白：`gwnum.h:4-6` "…multi-precision **IBDWT** arithmetic routines…"；权重公式在
`gwdbldbl.cpp:852-853`：`b^(ceil(j*n/FFTLEN) - j*n/FFTLEN) * abs(c)^(j/FFTLEN)`；
"irrational" 指非整数位/字的 FFT（`gwnum.h:158-162`、`:979-980`），AVX-512 路径**全部**用 irrational FFT
（`gwnum.c:2676-2678`："AVX-512 FFTs can use irrational FFTs because the weights are handled entirely in assembly code"）。

### 2.2 调度与"强制路径"（这是唯一受支持的实验开关）

* FFT 档位由汇编跳转表给出（`mult.asm:5098-5132`，格式 `PRCSTRT max_exp, fftlen, speed`）。AVX-512 表对
  **Mersenne 形态**（`gwnum.c:1224-1225`：表里的 `max_exp` 按 `k=1,c=-1` 标定）是
  `2905→128`、`5755→256`、`8527→384`、`11309→512`、`14119→640`、`16839→768`、`19701→896`、`22445→1024`。
  **探针实测**：p=3571 用 `fftlen=256`、p=12323 用 `fftlen=640` —— 与表一致。
* 强制路径（官方、文档化）：`gwinit` 之后、`gwsetup` 之前改句柄里的 `cpu_flags`
  （`tutorial.txt:29-36`：`gwdata.cpu_flags &= ~CPU_AVX512F;`）。也可以自己先填 `cpuid.h` 的全局
  （`readme.txt:4-7`；`gwnum.c:2124` 只在 `CPU_FLAGS==0 && CPU_SPEED==0` 时才自己探测）。
  探针用的就是前者。**没有环境变量**（gwnum 目录里 `getenv` 零命中）；只有一个 `gwnum.txt` 基准数据库。
* 反向注意：gwnum **也会自己降级**（`gwnum.c:1008-1012`：`log2b*n < 5*128` 时清掉 AVX-512；`:1150-1153`：
  `b==2 && n<100000` 时清掉），所以"强制"不是绝对的。

### 2.3 没有 ECM 专用 API

gwnum 只提供**模乘/加/减 + FFT 状态管理 + polymult 多项式乘法**。prime95 的 ECM 全在应用层：

* Stage 1 的 Montgomery 加倍（`ecm.cpp:2403-2409`）就是 6 条通用调用：
  `gwsetmulbyconst` + `gwmul3(GWMUL_FFT_S12|GWMUL_MULBYCONST)` + `gwsub3o` + `gwsquare2` + `gwmul3` + 两条 `gwaddmul4`；
* Stage 2 用 `polymult`/`polymult_fma`（`ecm.cpp:9445-9454`）+ `gwsubmul4` 累加（`:8721`）+ 最后 `gwmul3`（`:9755`）与 `gcd`（`:9858`）；
* 模数形态二选一（`ecm.cpp:7199-7200`）：`w->n != 0` → `gwsetup(k,b,n,c)`（快），否则 `gwsetup_general_mod_giant`（慢 3×，`gwnum.h:105-107`）。

**含义**：复用 gwnum ≠ 得到 ECM；要么自己写曲线算术（我们**已经有了**，在 GPU 与 IFMA 上），
要么照抄 `ecm.cpp` 那 16.8k 行。gwnum 能提供的只有"更快的模乘"这一件事。

### 2.4 许可（逐字）

授权文本在 `gwnum/readme.txt:86-110`（**不在** `license.txt` 里；根 `license.txt` 是 GIMPS 的 EULA，全文未提 gwnum）：

```
-> Legal stuff

Copyright (c) 1996-2022, Mersenne Research, Inc.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

(1) Redistributing source code must contain this copyright notice,
limitations, and disclaimer.
(2) If this software is used to find Mersenne Prime numbers, then
GIMPS will be considered the discoverer of any prime numbers found
and the prize rules at http://mersenne.org/prize.htm will apply.
```

* 这是 **BSD 式**骨架（第 2 条被换成 GIMPS 发现权条件），**无 copyleft**、**无"不得商用"**；
  `gwnum.h:592-593` 的 `gwnum_is_gpl() (0)` 说明**这份** gwnum 不是 GPL（全树只有这一处定义、无人调用）。
* **歧义（必须说清）**：文本**从未**明确"可以链进第三方程序"；条件 (1) 字面只约束 **source** 再分发，
  **没有** binary 形式的 notice 条款（同树其他 BSD 文本如 `mt19937ar.c:18` 都有）；预编译 `.lib` 是否算
  "binary form" 未说明；根 EULA 自称不完整并把权威规则指向 `mersenne.org/legal`（**本次未取**）。
* 若要正式引入，建议：保留 `gwnum/readme.txt` 的版权/免责声明原文，并在文档里写明 GIMPS 发现权条件。

---

## 3. 实测

### 3.1 探针

`tools/gwnum_probe/`（新增）：`gwnum_probe.cpp` + `build_and_run.ps1`（MSVC x64，链 `gwnum64.lib` 与仓库自带
`gmp.lib`；**必须 `/MT`** + `advapi32.lib` —— 预编译库是 `/MT` 静态 CRT，大页支持在 `gwutil.c:111-119` 用
`OpenProcessTokenizerToken/AdjustTokenPrivileges`）。复现：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\gwnum_probe\build_and_run.ps1          # 全部尺寸
powershell -NoProfile -ExecutionPolicy Bypass -File tools\gwnum_probe\build_and_run.ps1 -Quick   # 只 3571/12323
tools\gwnum_probe\bin\gwnum_probe.exe --selftest                                                 # 恒等式自检
```

探针做四件事：① 测 `gwmul3`/`gwsquare2`（Mersenne 形态 `gwsetup(h,1,2,p,-1)`）与 `gwsetup_general_mod_64`（无特殊形态的模数）；
② 分别测"自动 / 清 AVX-512 / 再清 FMA3"三条路径并打印 `gwfft_description`；③ 与 GMP `mpz_mul+mpz_mod` 同尺寸对比；
④ **正确性**：先跑代数恒等式（`3*5=15`、`(-1)²=1`、`2·2^(p−1)=1`），再对 50 组随机数把 gwnum 结果与 GMP 逐位比对，并报 `gw_get_maxerr`。

### 3.2 数字（3 次重复的稳定值；日志见 `tools/gwnum_probe/probe_output.txt`）

| p（位） | gwnum 自动（AVX-512） | 清 AVX-512 → **FMA3** | 再清 FMA3 → AVX | 通用模数（Montgomery 2×FFT） | GMP `mul+mod` | GMP 裸 `mpz_mul` |
|---|---|---|---|---|---|---|
| **3571** | 0.51–0.58 µs（fftlen 256） | **0.36–0.37 µs**（fftlen 160） | 0.37 µs | 1.37 µs（**3.7×**） | 3.00 µs | 0.97 µs |
| **12323** | **1.14–1.15 µs**（fftlen 640） | 1.36–1.37 µs | 1.34 µs | 3.24 µs（2.8×） | 22.6 µs | 8.3–9.4 µs |
| 100003 | **18.8 µs**（8 线程 15.8） | 24.6 µs | 23.1 µs | — | 795 µs | 254 µs |
| 1000003 | **285 µs**（8 线程 305） | 454 µs | 467 µs | — | 11 974 µs | 4 529 µs |

正确性：**所有配置 `bad=0`**（50/50 逐位相符），`gw_get_maxerr` = 8.8e-7（3571 位 AVX-512）、0.0625（3571 位 FMA3）、
0.00276（12323 位）、0.188（1 000 003 位，仍 < 0.5 的安全线）。

### 3.3 三个值得记住的观察

1. **3571 位时 AVX-512 反而比 FMA3 慢 ~40%**（0.51 vs 0.37 µs）：AVX-512 的档位从 128 起跳，
   p≤5755 全落在 `fftlen=256` 的 radix-4 上，而 FMA3 用 `fftlen=160`。gwnum 自己也会在很小的形态上清掉
   AVX-512（`gwnum.c:1008-1012`），但那个阈值只到 640 位，**盖不住 3571 位**。要复现 gwnum 的最佳性能，
   小尺寸必须自己关 AVX-512。
2. **通用（无特殊形态）模数慢约 3×**（1.37 vs 0.37 µs @3571）：与教程 `tutorial.txt:25-26` 的说法一致，
   描述串会写 `Montgomery reduction AVX-512 FFT length 2x256`（内存也翻倍）。PrimeNet 给的非 `2^p−1` 任务会吃这个亏。
3. **多线程在我们关心的尺寸上帮不上忙**：p=100003 时 8 线程只有 1.19×（18.8→15.8 µs），p=1000003 时
   8 线程反而略慢（285→305 µs，本机实测，3 次重复同向）。源码层面也解释了：**单趟 FFT 不能多线程**
   （`gwnum.c:6500-6505`），而多线程"最早也要 13 万位才开始"（`tutorial.txt:38`）。

---

## 4. 决定性对比：gwnum 的 FFT vs 我们自己的 IFMA

### 4.1 我们这一侧的实测（端到端）

```powershell
# M3571, B1=1e5, 8 曲线 SIMD 批, 单线程
echo <2^3571-1 的十进制> | build_cuda_cmake\ecm_cuda.exe --method mont -gpucurves 8 100000
```

实测 **0.55 s/曲线**（`s_bits=144344` ⇒ 每条曲线 10×144 344 = **1.443e6 次域乘**）
⇒ 等效 **≈381 ns/次域乘**（8 条曲线在 SIMD 批里并行摊薄后的每曲线成本）。
对照仓库既有实测：M4001/B1=1e6 单线程 6.264 s/曲线 ⇒ 等效 0.435 µs/次，与上一致。

### 4.2 结论

| 尺寸 | 我们（IFMA，8 曲线批） | gwnum FFT（标量） | 谁快 |
|---|---|---|---|
| **M3571 / M3583 / M3613** | ≈0.38–0.44 µs/次 | 0.36 µs（FMA3）/ 0.51 µs（AVX-512，gwnum 默认选它） | **平手到我们略快**；且 gwnum 不能跨曲线批处理 |
| **M12323** | ≈4 µs/次（按 n² 外推） | **1.14 µs** | gwnum ≈**3.5×** |

这与仓库里**已有的独立实测**完全一致（`docs/ECM_Montgomery_STAGE1.md` §14.5/§14.6，固定 B1=1e6 逐曲线）：
交叉点在 **≈2 880 位** 与 **≈3 760 位**；`M3001` 我们快 1.37×、`M3500` 快 1.09×；`M4001` 我们 0.90×；
`M5755` 0.67×、`M8527` 0.40×、`M19701` 0.17×（即 Prime95 分别快 1.5×/2.5×/6×）。

**机理**（两边都能自证）：我们是 **O(n²) 的 schoolbook + 52 位 IFMA + 8 条曲线并行**；
gwnum 是 **FFT，同一档位内每次模乘成本几乎恒定，只在档位边界阶跃**（档位正是 `mult.asm` 里的
2905/5755/8527/11309/14119…，与上面那张交叉点表一一对应）。

---

## 5. 为什么对整体生产任务几乎没有意义

* **生产路径是 GPU stage 1**：`method = gpu`（`ecm_queue_config.cpp:404`，默认 `ecm_queue_config.h:86`），
  CPU 两条 stage-1 路径是**显式 opt-in**（`--method mont|edwards`），而且没有 GPU 时默认路径直接报错退出
  （`ecm_driver.cpp:2785-2800`、`:3015`），**不会**自动回退到 CPU。
* **宿主端大整数工作占比 ≈0.016%**：一个生产任务（M3571/B1=2.6e8/960 曲线，72.9 s/曲线 ≈19.4 h）里，
  `s=lcm(1..B1)` 首次 10.7 s、命中缓存 0.26 s（47 MB），曲线生成 param0 0.071 ms/曲线（<0.6%），
  其余是 GPU 内核。把宿主端模乘换成"无限快"，任务时间也只动 0.0x%。
* **M12323 那条 3× 收益落在最弱的 CPU 路径上**：`mont_t` 上限 160 肢 = 10 240 位
  （`ecm_edwards_mont.h:54-58`），M12323 超过它 ⇒ 标量 Montgomery 直接拒绝（`ecm_mont_cpu.cpp:186`），
  Edwards 标量路径退化成**纯 mpz 双倍-加法**（`ecm_edwards_cpu.cpp:356-371`）。也就是说
  "有 3× 收益"的那条路本来就不是我们在用的路。
* 唯一真正在用的 CPU 大整数工作是 `s=lcm` 的乘积树（顶层 `mpz_mul` 187.5 Mbit×187.5 Mbit = 0.871 s，
  已被 GMP 自己的 FFT 接管），而它**每台机器每个 (B1,torsion) 只算一次**。

---

## 6. 如果要做：集成成本清单

| 项 | 事实 / 要求 |
|---|---|
| 调用序列 | `gwinit(&h)` → 可选 `gwset_num_threads` → `gwsetup(&h,1.0,2,p,-1)`（Mersenne）；通用模数 `gwsetup_general_mod_64`（慢 3×）→ `gwalloc` → `binarytogw`/`u64togw` → `gwmul3`/`gwsquare2` → `gwtobinary` → `gwfreeall`/`gwdone` |
| 自证 | `gwfft_description(h, buf)`（"AVX-512 FFT length 256…"）、`gwfftlen(h)`、`h.FFT_TYPE/ARCH`；`gw_get_fft_count` |
| 线程 | **一个线程一个 `gwhandle`**（`gwnum.h:15-17`），或 `gwclone` 共享表；我们的 CPU 路径是"每线程一批 8 条曲线"，换 gwnum 后**曲线之间无法再批处理** ⇒ 需要每线程 8 份标量计算，SIMD 并行度直接归零 |
| 位精确性 | 我们与 gmp-ecm **逐位一致**（`test_cuda_param2.ps1`）且 `.save` 带 CHECKSUM（`ecm_save.cpp:177-184`）。FFT 是**浮点**：必须常开 `gwerror_checking` + 监控 `gw_get_maxerr`（>0.5 就错），必要时 `gwset_larger_fftlen_count`/`gwset_safety_margin`。gwnum 自己**不会**自动重试 |
| 内存 | 每个数 ≈ `8×FFTLEN + 184` B（推断，见源码 `gwnum.c:1952`+`:7351`）：256→≈2.2 KB、640→≈5.3 KB；每个句柄固定表 ≈8.4/16.4/20.4 KB（`mult.asm` 表列） |
| 构建 | 只需 `gwnum64.lib`（静态 x64，`/MT`）+ `libcmt` + 系统库 + **`advapi32.lib`**；`polymult` 另需 `polymult64.lib`；**不需要** sqlite/GMP/pthread |
| 版本守卫 | `gwnum.h` 必须与所链的库同版本：`gwinit` 传 `GWNUM_VERSION`（"31.4"）与 `sizeof(gwhandle)`，不匹配时 `gwsetup` 报 `GWERROR_VERSION_MISMATCH(1006)` / `GWERROR_STRUCT_SIZE_MISMATCH(1007)`（`gwnum.c:2089-2096`）；汇编侧另有版本自检（`gwnum.c:996-997`）。所以**不能**拿旧头文件配新库 |
| 多线程的真实边界 | 只有**双趟** FFT 支持多线程（`gwnum.c:6500-6505`），而单趟/双趟由 (CPU, FFT 长度) 在跳转表里写死。我们关心的尺寸在 AVX-512 上正好是 **r4 单趟**（p≤5755→256、≤11309→512；探针实测 p=12323 得 `fftlen=640` 且无 `Pass1/Pass2`），到 10 万位才变 `type=3` 双趟（探针：`AVX-512 FFT length 5K`） |
| 实际选中的 arch | 探针里 `h->ARCH = 8` = `ARCH_SKX`（`gwnum.c:590`）。gwnum 另有 `ZENPLUS` 的 AVX-512 变体标签（`zarch.mac:29-30`），但本机（Zen 5）选的是 SKX |
| 许可 | §2.4：允许再分发与修改（两条件），但"链接进第三方程序"未被明文写出 |

---

## 7. 建议

1. **不把 gwnum 引入生产路径**：收益上界是一个任务的 0.016%，而代价是浮点 FFT 的位精确性风险、
   失去 8 曲线 SIMD 批处理、以及许可文本的歧义。
2. **保持"stage 1 我们算（GPU/IFMA）、stage 2 交给 Prime95"**（仓库既有决策）：Prime95 用的正是这套
   AVX-512 IBDWT 引擎，在 ≥5755 位它本来就比我们快 1.5–6×，交给它是**已经吃到**这个红利的方式。
3. **再评估的触发条件**（满足任一再考虑）：
   * 出现"**无 GPU / CPU-only 且 N ≳ 10 000 位**"的真实使用场景（那时 gwnum 有 3–6×）；
   * 或我们决定**自研 stage 2**（那时该看的是 `polymult64.lib` 的多项式 FFT，而不是 `gwmul3`）。
4. **若真的用**：小尺寸（≤5 700 位）**必须**关掉 AVX-512（`h.cpu_flags &= ~CPU_AVX512F`），否则会慢 ~40%；
   并以"每线程 8 份句柄 + 曲线级并行"补偿丢失的 SIMD 批处理。

---

## 8. 附录

* 探针与原始日志：`tools/gwnum_probe/gwnum_probe.cpp`、`tools/gwnum_probe/build_and_run.ps1`、
  `tools/gwnum_probe/probe_output.txt`（3 次重复）。
* 顺手发现（与本议题无关但值得记）：`build_cuda_cmake\cpu_mont_bench.exe` 的**自检在所有尺寸都失败**
  （`--no-overflow` 在 512 位也失败，`1024..4096` 位默认用例失败），因此本文的"我们这一侧"数字取自
  **端到端** stage-1（§4.1）与仓库既有实测，而不是这个基准工具。
* 本次未做：`mersenne.org/legal` 与 prize 页面未取（许可权威文本）；未验证强制 `cpu_flags` 后
  gwnum 内部降级规则（`gwnum.c:1011`、`:1150`）在所有尺寸下的影响；1 000 003 位的多线程数字只测了 1 轮。
