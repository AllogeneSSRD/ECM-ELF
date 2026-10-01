# stage-1 存档 → stage 2 → 因子：端到端流程图与数据手册

> 目标读者：接手 stage-2 GPU 引擎的人。本文只回答四个问题：**谁在哪块设备上跑哪一步**、**每步的算法是什么**、**它占多少寄存器/共享内存/显存/内存**、**时间花在哪儿**。
> 主文档：`docs/DEV_STAGE2_GPU_PLAN.md`（下称 §N，引用即该文件的 §N）；术语（蝶形/进位/单位根/权重/载荷位/系数/数字/槽/流/趟/下降）沿用该文件。
>
> **数字纪律**：本文每一个数字都来自仓库内的文件或代码，逐条给 `§NN` 或 `file:line`。**仓库里查不到的写"未测/未知"并说明查过哪里，一律不估。** 若两处来源冲突，给出两处并优先采用更晚的节（该文档 §10.1、§10.5、§10.6、§14.13、§14.15 各自撤回了一条更早的说法）。

---

## 0. 一句话总览

| 角色 | 输入 | 输出 |
|---|---|---|
| GPU0（4070 Ti，60 SM） | `N`、`B1`、σ 序列 | 每曲线一行 GMP 风格存档（`SIGMA/B1/N/X`） |
| CPU | 存档 + planner | σ→曲线参数（`a24`、起点）、GCD、两个 oracle |
| GPU1（4060 Laptop，24 SM） | 存档 + `(B1,B2,D)` | 每曲线（或每块）一个累加器 → 主机 GCD → 因子 |

三条并存路线（§2.3）：**① CUDA 配对/BSGS** = 正确性沙箱 + 无 Prime95 兜底（性能不可行）；**② CPU GMP 树版** = M3 的 oracle；**③ GPU 树版** = 唯一有性能前途的主线（S1–S3 已完成，见 §18）。

---

## 1. 输入：stage-1 存档（GMP 风格）

### 1.1 精确格式

一任务一文件、**每曲线一行、行内自包含**（每行都重复公共字段，理由是参考实现的 reader 逐行解析，抽走字段会破坏 `ecm -resume` 兼容性）：

```
METHOD=ECM; [PARAM=0;] SIGMA=<十进制>; B1=<十进制>; N=<十进制或表达式>; X=0x<hex>;
CHECKSUM=<十进制>; PROGRAM=ECM-ELY; X0=0x0; Y0=0x0; WHO=<host>; TIME=<ctime>;
```

* 字段集合与分隔符（`; ` 分隔、`KEY=VALUE`）是 **gmp-ecm 家族的事实格式**：`docs/ECM_Montgomery_STAGE1.md` §9.1 逐字照录了 `3001_B1e5.save` 的真实一行，§9.2 记录了"一文件多行、每行自包含"的取舍。
* **谁写它**：本仓库的 writer 是 `src/core/ecm_save.cpp:219`（`ecm_append_save_lines_mont`），输出见 `src/core/ecm_save.cpp:256-266`。GPU stage-1 走 param0 时写这一种（`src/core/ecm_driver.cpp:2921-2922`：**省略 `PARAM=`、必须携带原始 N**，否则 gmp-ecm 按 N 校验 CHECKSUM 会把整行判成 bad checksum）；CPU stage-1 走同一函数（`src/core/ecm_driver.cpp:2742`）。
* **`X` 是归一化仿射 x = Qx/Qz**，`stage2_ref.cpp:28` 明确写出这一点，`stage2_ref.cpp:768-772` 与 `tools/bench/stage2_tree_ref.cpp:57` 都注明解析时要**跳过 `0x` 前缀**。
* **历史 bug（静默型）**：§7.3 —— 解析 `X=0x…` 时保留了 `0x`，而 `mpz_set_str(...,16)` 不接受前缀 ⇒ 点静默变成 0，"存档驱动的 stage 2 找不到因子，而同一曲线的合成模式能找到"。修法即跳过 4 个字符。
* **解析失败必须显式报错**（不是静默跳过）：`stage2_ref.cpp:908-912` 打印 `stage2_ref: unparsable X in the save: …` 并 `continue`；`stage2_tree_ref.cpp:1239-1240` 同款。相反 **`X` 字段整段缺失时解析器不报错**——`stage2_ref.cpp:750-781` 只在 `have_x` 为空时退回"自己重算 ladder"（`stage2_ref.cpp:922-934`）。**这是当前一处宽松点，接驱动时要显式化。**

### 1.2 一条曲线的存档携带什么、stage 2 从中取什么

| 存档量 | stage 2 用它做什么 | 出处 |
|---|---|---|
| `SIGMA` | 唯一不可推导的量：重建曲线（Suyama σ→`a24`）与起点 `(u³:v³)` | §3.1、"stage 2 只需要 (N, B1, X, σ)" |
| `B1` | 计算 `s = torsion·lcm(1..B1)`，用于自测/重算 X | `stage2_ref.cpp:228-229`、`kernels/cuda/cgbn_stage2.cu:107-109` |
| `N` | 全部算术的模数（**必须是原始 N**） | `src/core/ecm_driver.cpp:2923-2926` |
| `X` | stage-2 的输入点，直接当 `(x : 1)` 用（**stage 2 全程不需要求逆/域转换**，见 §9.2） | §9.2、§9.3 第 4 项 |
| `CHECKSUM/PROGRAM/WHO/TIME` | 只用于与 gmp-ecm 互操作，stage 2 不读 | `docs/ECM_Montgomery_STAGE1.md` §9.1 |

**验收口径**：存档路径与"从 (N,σ,B1) 独立重算 ladder"必须给出**同一结论**（§7.2 第 2 项：真实驱动写出的 `X=0x294b…` 与参考实现重算的 x 完全相同；§9.3 第 4 项断言 `stage1_point=save` 而不是悄悄回退）。

---

## 2. 设备分工（三处：CPU / GPU0 / GPU1）

| 设备 | 角色 | 实测状态 | 出处 |
|---|---|---|---|
| **GPU0** 4070 Ti（60 SM） | 生产 stage 1，**24/7** | `99 %` / `178 W` 在跑 stage 1 | §0 表、§附录 A（`nvidia-smi` 实测） |
| **GPU1** 4060 Laptop（24 SM） | stage-2 的全部实验实现（配对内核、NTT 乘法、树版） | `0 %` / `9.10 W` / `0 MiB` 完全空闲 | `DEV_STAGE2_SELFHOST_FEASIBILITY.md` §0 表（2026-09-30 10:5x 实测） |
| **CPU**（24 逻辑核） | 曲线参数化、GCD、**两个 oracle**（配对/BSGS 参考、树版 GMP 参考）、planner | — | §7.1、§13、§18 |

**"stage-2 测量只在 GPU1"这条纪律的来源**（两处，都被代价换过）：

1. §8.4 第二轮补测：上一轮写下的"大形状反复超时 ⇒ 显存是限制"**是错的，已撤回**——真因是**设备争用**（同一时刻另一个进程在同一张卡 device 1 上做基准），代价是"一整轮结论被自己的重测推翻"；由此定下操作规则：**本机 GPU 基准必须串行跑**。
2. §17.4 测量纪律（"本会话用血换来的"）：用 device 1、跑前确认没有别的 CUDA 进程；同一形状**连测两次**，偏差 >3 % 判为受争用并重测。§14.11 的两次测量偏差 ≤0.2 %、§14.14 ≤1 %，即按这条纪律执行的结果。

**为什么 stage 2 必须与 GPU0 解耦**：§0/§2.2 —— 960 曲线任务的 stage 1 约 19.4 h（`docs/ECM_CGBN_OPTIMIZATION.md:608-615`），而"把 stage 2 藏进 stage 1 的时间里"正是整个计划的收益来源。

---

## 3. 逐步流程（每步：目的 / 设备 / 算法 / 资源 / 耗时）

### S0 曲线参数化（Suyama σ → `a24`、起点）

| 项 | 内容 |
|---|---|
| 目的 | 由存档里的 σ 唯一重建一条 Montgomery 曲线与其起点 |
| 设备 | **CPU，主机 GMP** |
| 算法 | `u=σ²−5`、`v=4σ`、`A=(v−u)³(3u+v)/(4u³v)−2`、`a24=(A+2)/4`、起点 `(X:Z)=(u³:v³)`（= gmp-ecm `-param 0`） |
| 资源 | 小整数 + 两次模逆；退化（`gcd(den,N)>1`）时**直接吐出因子**并跳过该曲线 |
| 耗时 | **未测/未知**（逐曲线量级为微秒，但仓库里没有这一项的单独计时；查过 §7、§18 与 `stage2_ref.cpp` 的计时输出，均无此分项） |

出处：`tools/bench/stage2_ref.cpp:26-28`、`:190-226`（含两次 `mpz_invert` 与退化处理）；GPU 侧同一套公式在 `kernels/cuda/cgbn_stage2.cu:67-100`；约定固化于 `docs/ECM_Montgomery_STAGE1.md` §4.1/§6（Q2：29 条 PrMers 存档 + gmp-ecm 7 个 σ 双向一致）。

### S1 stage-1（GPU0 的生产路径）

| 项 | 内容 |
|---|---|
| 目的 | 计算 `Q = [s]P`，`s = torsion·lcm(1..B1)` |
| 设备 | **GPU0**（CGBN 内核，`kernels/cuda/cgbn_stage1_kernel.h`） |
| 算法 | x-only Montgomery ladder，MSB-first 逐位；每 bit 2S+2M（+条件差分加法 2S+4M） |
| 容器/线程 | CGBN `<TPI,BITS>`；`TPI` = 每实例协作线程数，`TPB` 默认 128（`cgbn_stage1_kernel.h:113-115`） |
| 寄存器 | 见 §5.1 表（按档位 56…255，实测 `ptxas -v`） |
| 域约定 | `mont_mul` 返回 `a·b·R⁻¹`（`R=2^BITS`）；**`a24` 必须先 `cgbn_bn2mont`** |
| 耗时 | **72.88–72.90 s/曲线**（960 曲线 ≈19.4 h；4070 Ti、B1=260e6、`CGBN<16,5120>`） |

* **"静默算错"的教训（§14.10 与 §9.2）**：`xDBL` 里 `Z2 = K·(BB + a24·K)` 要求两项同域（`K`、`BB` 都是 `R⁻¹` 量级），所以 **`a24` 必须在 Montgomery 域**，否则"算出来的是另一条曲线，而症状只是找不到因子"。同一原因让 stage 1 的 Suyama 路径使用**全宽 `mont_mul`** 而不是 32 位 `special_mult_ui32`（`cgbn_stage1_kernel.h:803-860`）。
* **指数 `s`**：`torsion=1` ⇒ `lcm(1..B1)`（与 gmp-ecm `-param 0` 自洽，本项目验收口径），`torsion=12` ⇒ Prime95 风格；自研 stage 2 固定 `--mont-torsion 1`（§5 风险表）。
* 调度事实：`blocks = curves/(TPB/TPI)` 是吞吐主变量；TPB=512（块数减半）实测 −15 %，96–128 的寄存器上限在 ≤2048 档 +4.7 %；**240 块（4 块/SM）比 120 块（2 块/SM）慢**（73.41 vs 72.88–72.90 s/曲线）——kernel 是 issue bound（ncu: Compute SOL ~83 %、DRAM ~0.5 %）。

### S1′ 存档归一化（为什么存仿射 x）

* 存的是 `x = Qx/Qz`（`src/core/ecm_driver.cpp:2645-2651` 注释即"same normalisation as `mont_stage1_curve_bits_x()`"），代价是**每曲线一次模逆**；收益是 stage 2 可以把点当 `(x:1)` 直接用，**整条 stage-2 链路不需要求逆也不需要域转换**（§9.2）。
* 对照装置：`--print-stage1-x`（`stage2_ref.cpp:820`、`:838`）从 `(N,σ,B1)` 重算 x，供与存档逐位对拍；测试项见 `tools/test/test_stage2_ref.ps1:126`。

### S2 三条实现路线各自的位置与角色

| 路线 | 位置 / 角色 | 结构 | 结论 |
|---|---|---|---|
| **① CUDA 配对 / BSGS** | `kernels/cuda/cgbn_stage2_kernel.h` + `cgbn_stage2.cu`；**正确性沙箱 + 无 Prime95 兜底** | baby 表 + giant 表 + 每素数累乘 `X_iZ_j − X_jZ_i` | **性能上不可行**：要 6 s/曲线需 **≥2.7e9 ops/s**（§2.1），而 GPU 模乘只有 35.7 Mops/s（`<16,4096>`，4060）/ ~9e7（4070 Ti）⇒ 低 30–75×，且比 24 线程 IFMA CPU（95 Mops/s）还慢 **2.7–10×**（§1.5 实测 + `ECM_Montgomery_STAGE1.md` §11） |
| **② CPU GMP 树版参考** | `tools/bench/stage2_tree_ref.cpp`；**M3 的 oracle** | 简单版：一棵 giant 积树 + 一次余式树下降 | 唯一能给出"因子 + `hit_primes` + 分批版成本分解"的独立实现（§13.2 我逐项复测通过；`--selftest` 12 checks / 0 failed） |
| **③ GPU 树版** | `tools/bench/stage2_tree_gpu.cu`（S1–S3 已完成，S4 待做） | S2 简单结构（主机编排）→ **S3 分批结构 + 设备侧编排** | S2 46–49 s → S3 **1.08 s**（42–45×）；乘法**只有一份实现**（`ntt_poly_mul_host()`，`#include` 探针文件，禁止复制） |

三条路线与 **NTT 乘法**的关系：① 完全不使用（全是系数模乘）；② 与 ③ 的每次多项式乘法都落在那一个融合 NTT 上，③ 只是"调用它、不重写它"——§18 的工程约束第 1 条就是为了避免"两处实现各自漂移"（§14.8 的教训）。

### S3 收尾：上升、GCD、命名、结果

| 步骤 | 内容 | 出处 |
|---|---|---|
| 累加 | 树版把下降的叶子值在**设备上**做分块乘积（`s2g_block_prod_kernel`），主机 GMP 只看到"每块一个值" | `stage2_tree_gpu.cu:2100-2116` |
| GCD | 每曲线（树版：每块 + 末尾各一次）一次 `gcd(acc, N)`；配对数版本"每块 gcd 缩小范围、块内逐素数 gcd 指名" | `stage2_ref.cpp:17-18`、`stage2_tree_ref.cpp:29-31` |
| 命名 | 命中素数写进 `hit_primes`（实测冻结向量 **114713**，与 gp 给出的 `#E` 最大素因子一致） | `stage2_ref.cpp:263-295`、`stage2_ref.cpp:8-9`（§8.2 交叉验证） |
| 验证 | 每个报出的因子必须整除 `N`，否则计入 `bad_factors`（全部运行均为 0） | `stage2_ref.cpp:273-280` |
| 结果文件 / 交接 | 设计上：`stage2_engine=self` 时**抑制 `p95_add` 交付**（与 `p95_transfer` 互斥，避免双份上报）；ini 键 `stage2_engine`(p95/self/auto，默认 p95)、`stage2_b2`、`stage2_memory_mb`、`stage2_curves_batch`、`stage2_device`(默认 1)。**这些键在代码里尚不存在**（§7 未完成清单） | §3.1 表、`src/core/p95_transfer.cpp:389-392` |

---

## 4. 算法清单（每个一句话 + 出处）

| 算法 | 一句话 | 出处 |
|---|---|---|
| Suyama 参数化（`-param 0`） | `u=σ²−5, v=4σ` ⇒ `a24=(A+2)/4`，起点 `(u³:v³)`；三次求逆的意义是**在复合 N 上做除法**，逆不存在时直接得到 `gcd(den,N)`（免费因子） | `stage2_ref.cpp:190-226`、`cgbn_stage2.cu:69-100` |
| x-only Montgomery ladder | 只跟踪 `(X:Z)`，MSB-first 逐位 DBL/条件 ADD，差值点固定为 `P` | `stage2_ref.cpp:120-155` |
| BSGS 配对判据 | `X_iZ_j − X_jZ_i ≡ 0 (mod p) ⇔ p \| iD±j`；候选集 `{iD±j} ∩ (B1,B2]` 是素数的**超集**，故 brute 的命中必是 pairing 的子集（这就是两侧一致的判据） | `stage2_ref.cpp:15-21`、§7.1 |
| Bernstein 简单结构 | `F(X)=Π(X−x_j)` 积树 → `1/F` Newton → 对全部 giant 点补一棵积树再做**余式树多点求值** → 累加 → 每曲线一次 GCD | `stage2_tree_ref.cpp:11-31` |
| **分批结构**（GPU 主线） | F 树一次 + `1/rev(F)` Newton **只做一次** + 每轮一棵 G 树 + `H=G·H mod F` **恰好 3 次全尺寸乘法** + **一次**下降 | §18.2；`stage2_tree_gpu.cu:2064-2094` |
| Kronecker + Goldilocks NTT 乘法 | 每个系数放进 `slot_bits = 2S+log2 P` 位的槽（按字对齐的 `slot_stride`），整数组一次整数 NTT 卷积；**槽宽保证进位不跨槽** ⇒ 无全局 carry | `ntt_poly_probe.cu:2259`、§8.4、§11.1 |
| **可证明的精确性界** | `L·(2^bpw−1)² < p`，其中 `L = P·slot_words`（`p = 2^64−2^32+1` 是 Goldilocks 素数）；**由实际数组值算出**（外加"每个 digit `< 2^bpw`"的硬断言），不是由设计意图推出 | §14.15；`stage2_tree_gpu.cu:50-52`、`ntt_poly_probe.cu:2352-2360` |
| Newton 除法 | 反转多项式迭代；`F` 与每个树节点都**首一** ⇒ 首项逆恒为 1 ⇒ 复合 N 下**永远不会失败**（这正是 stage 2 能工作的原因） | `stage2_tree_ref.cpp:24-28`、`stage2_tree_gpu.cu` 的 `poly_inv_series` |

---

## 5. 资源占用

### 5.1 stage 1（CGBN 内核，按档位）

寄存器实测（`ptxas -v`，suyama/param0 家族，TPB=128）与由此得到的块/SM；`REG_TARGET` 的落地规则同样是代码里的实测表（`kernels/cuda/cgbn_stage1_kernel.h:301-303`：`≤2048 → 56`、`2560..5120 → 128`、`≥5632 → 255`）：

| tier（bits） | 2560 | 3072 | 3584 | 4096 | 4608 | 5120 | 5632 | 6144 | 7168 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|
| 寄存器（不限） | 86 | 98 | 109 | 117 | 129 | 141 | 163 | 174 | 186 | 211 |
| 块/SM | 5 | 5 | 4 | 4 | 3 | 3 | 3 | 2 | 2 | 2 |
| `FORCE=128` 的 spill | 0 | 0 | 0 | 0 | 48 B | 144 B | 560 B | 1028 B | 1896 B | 3356 B |

出处：`docs/ECM_CGBN_OPTIMIZATION.md:589-603`（§5.7④c 全档位实测）与 `cgbn_stage1_kernel.h:268-300`（同表的代码注释）。

| 项 | 值 / 说明 | 出处 |
|---|---|---|
| 每 block 线程 `TPB` | **128**（默认；`-DECM_TPB` 可扫） | `cgbn_stage1_kernel.h:113-115` |
| 实例粒度 `TPI` | 4（128–512）/ 8（768–2048）/ 16（2560–8192）/ 32（9216–16384） | `docs/ECM_Montgomery_STAGE1.md:2397-2402` |
| 共享内存 | **0**（`SHM_LIMIT=0`，"no shared mem available"，CGBN 用 shuffle 交换） | `cgbn_stage1_kernel.h:253` |
| 每实例状态 | 2 个域元素（`X,Z`）+ 全宽 `a24` | §14.10、`cgbn_stage1_kernel.h:852-860` |
| 容器填充浪费 | tier 是 n² 关系：3001→3072（+2 %）、3217→3584（+11 %） | `docs/ECM_Montgomery_STAGE1.md:2066-2067` |

**一处文档内部冲突（已标注）**：`docs/ECM_Montgomery_STAGE1.md:2015-2016` 写"kernel 每线程 88/98/105 个寄存器（3072/3584/4096），每 block 512–640 线程"，这来自 2026-09-24 的移植前估算；**更晚的实测**（2026-09-25，`docs/ECM_CGBN_OPTIMIZATION.md:589-603`）给 98/109/117 与 TPB=128。按"取更晚节"的规则，本文采用后者。

### 5.2 NTT 乘法（`ntt_poly_probe.cu`）

| 项 | 值 | 出处 |
|---|---|---|
| 元素类型 | **64 位 Goldilocks 数字**（`p = 2^64−2^32+1`） | §11.4、`ntt_poly_probe.cu:77` |
| 每元素载荷 | **bpw 位/数字**；实测随 P 变化：P=1024/8192/65536/92160 → **22/21/19/19** | §14.15 表 |
| 共享内存 tile | **32 KB**（`t=12` ⇒ `2^12` 个 u64）；twiddle 表放全局（表 32 KB、L2 常驻）；32 KB 下每 SM 驻 **3 块 = 1536 线程 = 48 warp** | `ntt_poly_probe.cu:1041-1046`、`:1169` |
| 寄存器级融合 | 外层 **radix-16**（`NTT_FUSE_M=4`，间距最大的 4 级在寄存器内完成，前向 3 趟/逆向 3 趟）；radix-32/64 实测**溢出且更慢**（M=5 → 0.181、M=6 → 0.224，均 13/10 趟） | §14.14、`ntt_poly_probe.cu:745` |
| 趟数 | `passes=` 打印（一趟 = 整数组一次读 + 一次写）；P=1024/8192/65536/92160 → **10/13/16/16** | §14.15 表、§14.14 |
| 显存（单次乘法的缓冲） | `mem_mb = (4·N + out_slots)·8 B`，`N = nwords`；**P=92160/S=5261 ⇒ 4097 MB**（N=2²⁷） | 公式 `ntt_poly_probe.cu:2352-2353`；数值 §14.12/§14.15 |
| 每字节承载载荷位（与 gpuowl 对照） | 我们的 Goldilocks 单模数 `2d + log2 L < 64` ⇒ L=2²¹ 时 d≈21；gpuowl：`FFT64` **2.40**、`FFT3161`(M31+M61) **3.38** b/B ⇒ **换域只值 1.35×**，胜负手在融合层数 | §11.4、`DEV_GPUOWL_CUDA_FFT_PATHS.md` §5 |
| L2 | **未测/未知**：仓库只有"twiddle 32 KB 是 L2 常驻"这一句定性描述，**没有 L2 容量或命中率的实测**（查过 §11/§12/§14 与探针注释） | — |
| 硬件上限 | device 1 实测拷贝带宽 **200 GB/s** ⇒ **一趟 = 1.34 ms** | §14.14 |

### 5.3 树版（S1–S3）

| 项 | 值 | 出处 |
|---|---|---|
| 系数表示 | `m·W` 个 u64 的**扁平数组**，`W=ceil(bits(N)/64)`，**系数优先**（`poly[i*W+t]` = 第 i 个系数的第 t 个 limb，LE），每系数归约进 `[0,N)` | `stage2_tree_gpu.cu:22-28` |
| 选它的理由 | 与 NTT 乘法的输入布局**逐位相同**（`S=bits(N)` ⇒ 热路径零转换）；系数连续便于下降的逐系数运算 | `stage2_tree_gpu.cu:30-41` |
| 每层尺寸 | 平衡堆 + 零填充：S1 冻结形状 `leaves=24 padded=32 muls=23 ntt_calls=23 slot_checks=135`、`coeffs_gpu=25`；D=2310 形状 `coeffs=241`、`F_degree 240/240` | §18.1（我复跑的 `check_stage2_tree_gpu.ps1` 输出） |
| 最大乘法尺寸 | S3 冻结向量：**`max_mul=25x16`**（另有 `coeff_muls` / `max_mul_bits` 同线打印） | `stage2_tree_gpu.cu:2894`；数值 §18.2 |
| 精确性判据 | **每次调用重新推导并断言** `L·(2^bpw−1)² < p`（`L=P·slot_words`），**绝不从探针的形状继承** | §18.1、`stage2_tree_gpu.cu:50-52` |
| 设备 arena | `NttArena` 按 `(nwords,out_slots)` 缓存 6 个设备缓冲、按 `(nwords,k,omega)` 缓存 `FuseCtx` **与每一趟的 twiddle 表**；单调钉住共享内存上限 | §18.2、`stage2_tree_gpu.cu:2096-2099` |
| **arena 上限（文档与代码不一致）** | **文档**（§18.2）写"真实上限 `min(free/4, 2 GB)`"；**代码**是 `cap = (free − reserve)`（`reserve` 默认 1 GB，free<reserve 时取 `free/2`），另有 `hard = 2 GB` 但**被显式丢弃**（`(void)hard;`）。环境变量 `NTT_ARENA_CAP_KB` / `NTT_ARENA_RESERVE_MB` / `NTT_ARENA_FRACTION` 可覆盖 | 代码 `stage2_tree_gpu.cu:2651-2674`；文档 §18.2。**以代码为准**，文档那句 `min(free/4, 2 GB)` 与实现不符 |
| 装不下时的行为 | 回退到逐调用分配：**结果相同、更慢**（不是失败） | `stage2_tree_gpu.cu:2668-2671` |
| 本轮实测 `arena_mb` / `arena_cap_mb` | **未测/未知**：这两个字段由运行打印，仓库里没有 S3 运行的 stdout 留档（查过 `.bench_tmp/`、`build_cuda_cmake/` 的日志与 §18） | — |

### 5.4 CPU 参考（GMP）

| 项 | 值 | 出处 |
|---|---|---|
| 用法 | 全部多项式运算走 `mpz_*`：`mpz_powm_ui`/`mpz_invert`（参数化）、教科书多项式乘、Newton 反转多项式除法、Horner 求值（独立 oracle） | `stage2_ref.cpp`、`stage2_tree_ref.cpp` |
| 为什么只适合小 P | 它是**教科书法**：乘法 `O(P²)`、逐点 Horner `O(P·#giant)`；§18.2 实测冻结形状（P=24）S2 结构要 46–49 s，而同一 GPU 引擎 1.08 s；§17.2 门禁 2 明说"真实形状 P≈9e4 时 CPU 参考不可行" | §18.1、§18.2、§7（未完成清单） |
| 已知陷阱（都是静默型） | ① `mpz_add_ui` 在 Windows 上按 32 位 `unsigned long` 截断（§14.10）；② `mpz_clear` 之后复用同一 `mpz_t` ⇒ 堆损坏、~5–7 % 无输出猝死（§14.13，修后 100/100 通过）；③ PowerShell 变量名不分大小写（§14.15） | 同左 |

---

## 6. 耗时占比

### 6.1 每曲线：stage 1 vs stage 2 的量级

| 阶段 | 数值 | 条件 | 出处 |
|---|---|---|---|
| stage 1（现状） | **72.88–72.90 s/曲线**；960 曲线 ≈**19.4 h** | 4070 Ti、B1=260e6、`CGBN<16,5120>`、TPB=128 | `docs/ECM_CGBN_OPTIMIZATION.md:608-615` |
| stage 2（现状 = Prime95） | **28.6 s/曲线**；960 曲线 ≈**7.6 h** | B2≈4.4e11，24 h 窗口、2800 曲线样本（整机吞吐） | §0 表、`DEV_STAGE2_SELFHOST_FEASIBILITY.md` §2.2 |
| stage-1 vs stage-2 内部比 | stage 2 ≈ **0.31 × stage 1** | M4001、B1=1e7、B2=1.24e11（12415×B1）、单 worker、Prime95 自报 | `docs/ECM_Montgomery_STAGE1.md` §16.3 |
| 任务级预算 | stage 1 19.4 h + stage 2 7.6 h ⇒ 串行 27 h；目标 **≈19.4–21 h**（stage 2 藏进 stage 1） | 960 曲线 | §0 表 |
| 自研 stage 2 的预期带 | 未融合 0.278 ns/位 ⇒ **120–129 s/曲线**（960 曲线 ~33 h，比 Prime95 慢 ~4×）；融合后 0.08 ⇒ ~44 s；融合 + 整数密度 ⇒ **30–45 s**（8–12 h，parity 到 1.6×） | P1/P2 口径，`4.30e11`(balanced)/`4.63e11`(padded) 操作数位/曲线 | §17.3、§15.1/§15.3 |

### 6.2 stage 2 内部：位数占比与墙钟

**位数占比（冻结向量 = GPU 实际口径 vs CPU 模型）**——两边刻意做成可对照，总计差 **+0.94 %**（对 `ours-balanced` 为 +0.99 %）：

| 分量 | GPU `batched_cost`（操作数位） | CPU 模型 `ours-padded` | 差异与解释 | 出处 |
|---|---|---|---|---|
| `f_tree` | 43 280 | 43 280（冻结形状模型值） | **精确相同** | §18.2 |
| `g_tree` | 8 586 626 | — | **+0.20 %**（GPU 建 199 棵，模型按 `loops=198`） | §18.2 |
| `fold` | 7 700 092 | — | **−1.4 %**（最后一批只有 11 点） | §18.2 |
| `descent` | 277 902 | — | **+221 %**（模型的 `2·f_tree` 只是**约定**，模型自己声明该项无法测量；GPU 每个访问节点都真做一次 Newton 求逆） | §18.2 |
| `inv` | 57 652 | 模型没有此项 | 新增项 | §18.2 |
| `poly_muls` | 5 357（其中 G 树 4 564，都在 `log2m ≤ 5`） | — | 另：`max_mul=25x16` | §18.2、§18.2 遗留① |

M5261 真实形状（P=132480、`num_polyG=11`、B2=1.94e12、S=5261）的三种口径（**以经代码逐项核对的 `ours-padded` 为准**，旧的 `plan-script` 为低计）：

| 口径 | f_tree | g_tree | fold | descent | 总计 | 每曲线 @0.278 ns/位 |
|---|---|---|---|---|---|---|
| `ours-balanced` | 2.664e10 | 2.664e11 | 8.378e10 | 5.328e10 | **4.301e11** | 119.6 s |
| **`ours-padded`** | 2.918e10 | 2.918e11 | 8.378e10 | 5.837e10 | **4.632e11** | **128.8 s** |
| `plan-script`（低计 1.25×，已标注保留） | 2.221e10 | 2.221e11 | 8.378e10 | 4.442e10 | 3.753e11 | 104.3 s |

出处：§15.1（含 `model_tree(24)==f_tree==43280`、`model_tree(4763)==giant_tree==19523226` 三个形状 MATCH）。

**墙钟（S2 → S3，同一二进制、device 1 空闲、各测两次）**：

| 阶段 | 墙钟 | NTT 调用 | 提速 |
|---|---|---|---|
| S2（简单结构 + 逐调用分配） | 46.28 / 48.72 s | 38 721 | 1× |
| **S3（分批 + arena）** | **1.09 / 1.08 s** | **5 357** | **42.5× / 45.1×** |
| S3 分解 | arena **单独**加在不改结构的简单版上：45.18 → **8.17 s（5.5×）** | — | 分批再去掉 **7.7×** 的乘法调用（38 721→5 334）；5.5 × 7.7 ≈ 42 |

* **最重要的读数（§18.2）**：S2 那 45 s 里 **82 % 是每次调用的 `cudaMalloc`/`fuse_init`/表重建，不是算术**。
* **相位级墙钟（giant / g_trees / fold / descent / inv / accum / name）未测/未知**：这些字段由运行打印（`stage2_tree_gpu.cu:2939-2941` 的 `batched_split:` 行），但仓库里没有 S3 运行的 stdout 留档（查过 `.bench_tmp/`、`build_cuda_cmake/` 的日志与 §18）。**以下给的是位数分解，两者不可互换。**

### 6.3 单次乘法内部

| 项 | 值 | 条件 | 出处 |
|---|---|---|---|
| `t_fwd` / `t_inv` / `t_slot` / `host_side` | **0.098 / 0.069 / 0.339 / 0.535 s** | 冻结向量 S3 整轮（5 334 次乘法、1.07 s 中的 97 %） | §18.2 |
| **纯蝶形算术** | **仅 ~1.4 ms** | 1.67e7 操作数位 @ 0.084 ns/位 ⇒ **剩下的 1.06 s 全是每次调用的固定开销** | §18.2 |
| 每次调用的固定开销 | 文档 §18.2 写 **195–209 µs**；**代码注释**给 `host_side = 0.535 s = 100 µs/call`（0.535 s ÷ 5 334 ≈ 100 µs） | 两处口径不同（未注明是否含 `t_fwd/t_inv/t_slot`），**冲突已记录** | §18.2 vs `stage2_tree_gpu.cu:791-793` |
| 13 趟 vs 遍数下界 | **200 GB/s ⇒ 一趟 1.34 ms**；13 趟 + 组装 0.56 ⇒ **18.2 ms 下界 = 0.108 ns/位**（"即使零计算，当前趟数也到不了 0.10"） | device 1 | §14.14 |
| 逐核计时 | 外层趟 1.5–2.4 ms（下界 1.34）、tile 趟 3.4/3.8 ms（下界 1.34）、进位+组装 3 ms | §14.14 文字写"P=8192"，但 1.34 ms 这一档对应的是 **P=92160（N=2²⁷）** 的流量规模；**该标签与数值不自洽，已记录** | §14.14 |
| 融合前后（乘法层，M2 门槛） | 未融合 2.1773 (P=1024)/2.7918 (P=8192) → 融合 0.1806/0.1676 → **最终 0.0843/0.0845**（P=8192/S=5153、13 趟、N=2²³） | device 1 空闲、各测两次、`ok=1`、偏差 ≤1 % | §14.11、§14.14、§14.15 |
| 与门槛的关系 | `P=8192/S=5153` = **0.0843 ≤ 0.10 ⇒ §12.3 的 M2 门槛达成，余量 2.0×**；对同形状 fp64 cuFFT 0.278 是 **3.3×** | 同左 | §14.15 |
| 换算 | 0.0843 ns/位 × 4.3e11 位/曲线 ≈ **36 s/曲线**（对 Prime95 的 28.6 s 是 1.27×，首次进入同量级） | P1 口径 | §14.15 |
| 真实形状 | `P=92160/S=5261`：**0.1275 / 0.1278 ns/操作数位**（16 趟、bpw=19、4097 MB、11 个抽样系数 bad=0） | device 1 | §14.15 |

---

## 7. 尚未完成 / 未验证（诚实清单）

| # | 项 | 状态 |
|---|---|---|
| 1 | **驱动接入** | **未做**：`include/ecm_backend.h` 只有 `ecm_backend_prepare/stage1/query_gpu`，没有 `ecm_backend_stage2`；ini 键 `stage2_engine`/`stage2_b2`/`stage2_memory_mb`/`stage2_curves_batch`/`stage2_device` 与 `--stage2 self` 只存在于 §3.1 的设计表里，**grep 全仓库只在文档中出现**（§9.5、§16.3 第 5 项） |
| 2 | **真实形状的树层验证方式有限** | S1/S2/S3 只在冻结向量（P=24）与 D=2310（P=240）上验过；P≈9e4 时 CPU 参考不可行（§17.2 门禁 2），所以**树层在真实形状上没有独立 oracle**，只剩"跑到结束且 `bad_factors=0`"这种弱证据 |
| 3 | **`bench` 模式仍是 `ok=0`** | `bench 100000/1e6/1e7 20 1` 全 `ok=0`，`abs diff bits` 恒为输入位数的 **62.5 %**（系统性错误）；**M2 的所有数字都来自 `poly` 模式，不受影响**（§14.12） |
| 4 | **`nttcheck` 的"host DIT mirror mismatches=255/256"是误报** | 主机端镜像过期（设备前向 vs 直接 DFT 仍 `bad=0/256`、往返 0），与 §14.8 的"镜像与实现各自漂移"同源，应删镜像或共用同一份 `__host__ __device__` 代码（§14.12） |
| 5 | **S4 未开始** | 真实形状 P=92160/S=5261 + 显存预算；把同层同尺寸的乘法并成一次启动（G 树占 5 334 次里的 4 564 次）；把精确系数的 mod N 归约彻底搬到设备（§18.2 遗留①②，S4 归约内核已落地但 §18.1 第 2 项的文字仍写"目前在主机 GMP"——**文档滞后于代码**） |
| 6 | **§12.5 的真设计选择未定** | 单层 Kronecker vs 两层（Prime95 式外层 + 每系数内层 FFT）：两者显存/流量结构不同，**必须各测一版**才能定（§10.5、§12.5） |
| 7 | **D 的选取** | planner 用的是**模型**（`stage2_shape_model.py --choose-d`），真实形状的"显存装得下的最大 D"没有实测过；P≈1e6 对应的每曲线位数只有插值/模型值（§10.6、§15.2） |
| 8 | **stage-2 结果与 Prime95 的对拍** | §0 要求的"同机、同 save、同参数对拍（wall time / 因子集合 / 峰值显存）"**尚未执行**（缺 S4 与驱动接入） |
| 9 | **`--st` 之外的测量纪律依赖** | 所有 GPU 数字都带"device 1 空闲、各测两次"的条件；脱离该条件的复现结果**不可比**（§8.4、§17.4） |

---

## 8. 一张"从存档到因子"的流程图

```
 [GPU0 4070 Ti, 24/7]                          [磁盘]
   S1 CGBN param0 ladder  ──►  stage-1 save：每曲线一行
   s = 1·lcm(1..B1)            SIGMA=<dec>; B1=<dec>; N=<dec>; X=0x<hex>   (1 行 ≈ 1.5 KB)
   72.88–72.90 s/曲线                    │
   19.4 h / 960 曲线                     ▼
                    ┌────────────────────────────────────────────────────┐
                    │ parse_save()  stage2_ref.cpp:750-781               │
                    │ SIGMA→σ  B1→B1  X=0x…→十六进制（必须跳过 "0x"）    │
                    │ 解析失败 ⇒ 显式报错（§7.3 的历史 bug）             │
                    └──────────────┬─────────────────────────────────────┘
                                   ▼
 [CPU] planner / 曲线参数化：Suyama σ → a24=(A+2)/4、起点 (X:Z)=(u³:v³)
       planner 选 (D,P,B2)（GPU 成本模型；§10.6：D 取"显存装得下的最大"）
                                   │
        ┌──────────────────────────┼───────────────────────────────┐
        ▼                          ▼                               ▼
 ①CUDA 配对（沙箱/兜底）    ②CPU GMP 树版（oracle）        ③GPU 树版（主线）
 kernels/cuda/cgbn_stage2   stage2_tree_ref.cpp           stage2_tree_gpu.cu
 baby[j]=[j]Q (j≤D/2)       F 树(简单版)                   S3 分批：
 giant[i]=[iD]Q             + 余式树多点求值               F 树一次 + 1/rev(F) Newton 一次
 每素数 X_iZ_j−X_jZ_i        + Horner 对拍 oracle           + 每轮 G 树 + H=G·H mod F (3 次)
 ⇒ 35.7 Mops/s ⇒ 448 s/曲线  ⇒ 只适合小 P（教科书法）        + 一次下降；每轮 3 次全尺寸乘法
   （性能不可行，§2.1）                                      S2 46–49 s → S3 1.08 s
        │                          │                               │
        │                          │              乘法唯一来源：ntt_poly_mul_host()
        │                          │              64 位 Goldilocks 数字、每元素 bpw 位载荷
        │                          │              slot_bits=2S+log2 P、按字对齐 slot_stride
        │                          │              32 KB 共享内存 tile、寄存器级 radix-16
        │                          │              13–16 趟；P=92160 ⇒ 4097 MB 显存
        │                          │              判据 L·(2^bpw−1)² < p（按实际数组值断言）
        └──────────────────────────┴───────────────┬───────────────┘
                                                   ▼
 [GPU1 4060, 24 SM]  下降的叶子值 → 设备分块乘积（s2g_block_prod_kernel）
                     主机 GMP 只看到"每 64 叶一个值"
                                                   ▼
 [CPU] gcd(acc, N)  ──► 命中素数命名（hit_primes；冻结向量 = 114713）
                       每个因子都验算整除 N（bad_factors 必须为 0）
                                                   ▼
                    results 双文件 + GUI 通知；无因子 ⇒ 记 "no factor"
                    设计上 stage2_engine=self 时抑制 p95_add（与 p95_transfer 互斥）——未实现
```

**数据量级（全部取自 §15.1/§14.15/§18.1）**：1 行存档 ≈ 1.5 KB（`X` 约 1250 个 hex 字符，见 `docs/ECM_CGBN_OPTIMIZATION.md:587` 的长度读数）；960 曲线 ⇒ 存档 ~1.4 MB；每曲线 NTT 操作数位 4.30e11–4.63e11（P1 口径）；单次乘法搬 `2·P·slot_bits` 位，P=92160 时 N=2²⁷、缓冲 4097 MB；冻结形状每次乘法的精确系数是 `2S+log2 P` 位（真实形状约 10539 位）。

---

## 9. 复现命令（本文引用数字的入口）

```
# 两条 oracle（CPU，无需显卡）
stage2_ref.exe      --n 2^128+1 对应的十进制 --sigma 26 --b1 1000 --b2 1e6 --d 210 --algorithm both
stage2_tree_ref.exe --n <同上> --sigma 26 --b1 1000 --b2 1e6 --d 210 --naive-check --cost
stage2_tree_ref.exe --dump-F F.txt   # 给 GPU 树版做逐系数 oracle

# 存档驱动（必须与 ladder 路径同解）
ecm_cuda.exe --method mont --exponent lcm -sigma 26 -gpucurves 1 --save <path>
stage2_ref.exe      --n <N> --save <path> --b2 1e6 --d 210
stage2_tree_ref.exe --n <N> --save <path> --b2 1e6 --d 210

# 乘法层（device 1 必须先确认空闲）
ntt_poly_probe.exe poly 8192 5153 1 1      # 期望 ok=1，ns_per_operand_bit ≈ 0.084
ntt_poly_probe.exe poly 92160 5261 1 1     # 期望 ok=1，4097 MB，≈ 0.128
powershell -File tools\bench\ntt_vs_cufft.ps1 -Real

# GPU 树版（S1–S3）
tools\build\check_stage2_tree_gpu.ps1      # 一键验收：F 逐系数 + e2e + 锐利性 + 三口径一致

# 成本模型（不跑 GPU）
python tools\bench\stage2_shape_model.py --b2 1.94e12 --bits 5261 --d 1411410
python tools\bench\stage2_shape_model.py --b2 1.94e12 --bits 5261 --choose-d --mem-cap-mb 2048
```

**读这些输出时的四条硬规则**（§16.4 + §17.4）：① 只有 `ok=1` 的构建才报时间；② 报每曲线耗时必须同时报 `bpw`/`slot_bits`/`passes=`；③ 与 Prime95 只能比**墙钟**（比"操作数位吞吐"不成立，§10.5）；④ 手算/期望值必须先**打印真实输入**（§14.8 的教训）。
