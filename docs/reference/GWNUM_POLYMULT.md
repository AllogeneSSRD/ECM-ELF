# Prime95 gwnum / polymult 表示与源码事实

固定参考资料。来源：`DEV_GWNUM_FEASIBILITY.md §§1–2`。保留数学推导、第三方源码分析及其条件；源码行号针对原文研究的本地副本，第三方上游可能已有变化。本文不声明本仓库当前支持、默认值或性能。涉及实验数字时只按原文条件理解；当前实现见 [架构](../architecture/OVERVIEW.md)，当前计时口径见 [性能](../performance/STAGE2.md)。

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
