# CGBN 优化：实测、结论与探针（2026-09-25）

> 主题：我们 CUDA stage-1 kernel 所依赖的 **CGBN** 还有多少可挖。
> 起因：CGBN 是 Volta（sm_70）时代的库、之后基本没更新，而我们的主力卡是 Ada（sm_89）。
> 姊妹文档：`docs/ECM_Montgomery_STAGE1.md`（param0/param3 kernel 的来龙去脉、§19–§21）、
> `docs/OPENCL_IMPLEMENTATION.md`（与 OpenCL 路径的差异）。

本文件记录 4 条结论 + 两处已落地的改动（§4 删冗余 `normalize_addition`、§5.5 占用率/寄存器默认值）
+ 两套探针工具 + 已被证据关闭的两个方向。**所有数字都是本机实测**，测量装置见 §1、复现命令见 §7。

---

## 0. 结论速览

| # | 结论 | 收益 | 状态 |
|---|---|---|---|
| 1 | CGBN 按架构选的乘法变体（sm_70+ → **WMAD**）在 Ada 上就是最优的，强切 XMAD/IMAD 反而慢 1.5–2.3× | — （**关闭**该方向） | 已测，不再投入 |
| 2 | `mont_sqr` 就是 `mont_mul(a,a)`（α = 1.0），专用平方的**理论上限 12–14%**，且 CGBN 的分布式布局要先解决跨 lane 部分和交换 | ≤ +14% | 可做，优先级低 |
| 3 | 我们每 bit 有 **8 次（param3）/ 6 次（param0）冗余的 `normalize_addition`** —— `mont_mul`/`mont_sqr` 返回时已经 < n；删掉后实测 **param3 +5.4%、param0 +6.7%** | **+5.4% / +6.7%** | **已实现**（本文件 §4） |
| 4 | add-chain / w-NAF **在 x-only 下不合法**（差值是每窗口都不同的新点，表里没有），且一维差分链的加法次数下界 1.44042/bit > 赢所需门槛 1.26/bit；真实链（PRAC，实测 1.388 加法 + 0.135 倍点每 bit）折成算子数 8.90–9.56 vs 梯子 8.24 ⇒ **慢 8–16%**。探针测出的"理想上界" 1.70×（w=2）/1.85×（w=3）/1.94×（w=4）不可达 | **无（方向关闭）** | 已实现探针 + 实测，结论关闭（§5.4） |

**优先级建议**：3（已做，免费）→ 4（研究型，收益最大）→ 2（统一小收益，代价大）。

> **补充（2026-09-25 当天后续）**：结论 4 已被证据**关闭**（§5.4：x-only 窗口不合法 + 加法次数下界
> + PRAC 实测算子数慢 8–16%）。同一天还量出并**落地**了一组与算法无关的收益：**每批曲线数 ≥8192
> （+7.6%）、TPB=128/MAX_ROTATION=1、小档位 `--maxrregcount=56`（合计 +4.7%）**，见 §5.5。

---

## 1. 测量装置与纪律

| 工具 | 用途 |
|---|---|
| `tools/bench/cgbn_op_probe.cu` | 逐 **CGBN 算子**单价（`mont_mul`/`mont_sqr`/compare+cond-sub/add/sub/shift），可切 TPI/BITS 档位，可用 `-DXMP_WMAD/-DXMP_XMAD/-DXMP_IMAD` 切乘法链变体、`-DPROBE_VALUE_MODE=0/1/2` 测值依赖性 |
| `tools/bench/cuda_kernel_ab.ps1` | **整 kernel** A/B 计时：固定 N/B1/曲线数，重复取中位数，输出 `gputime` 与 curve-bits/s |
| `-DECM_PROBE_ADD_DENSITY=k`（§5） | 让融合 double-and-add 只在每 k bit 执行一次加法（k>1 **结果错误**，纯计时探针） |
| `-DECM_PROBE_CHAIN_W=M`（§5.2/§5.4） | 把融合步换成"每 bit 1 次倍点 + 每 M bit 一次**真实**投影差值差分加法"（结果错误，纯计时探针） |

**纪律**（本仓库踩过的坑，见 `docs/ECM_Montgomery_STAGE1.md` §19.4）：

* 计时必须在**空闲** GPU 上做。本次 4070 Ti（GPU 0）被其它任务占用 99%，所有数据都取在
  **RTX 4060 Laptop（GPU 1，空闲）**；比值可跨 Ada 部件迁移，绝对 ns 是 4060 的。
* **绝对 ms 会随"整机状态"漂移 —— 只有同一时段内的比值可信（2026-09-25 新增）**：同一份二进制、
  同一组参数（M511/4096 曲线/B1=1e5），05:0x 测到 10326 ms，05:5x 复测同一配置只有 8173 ms
  （−21%），中间只有 GPU 0 的负载变化。所以本文档的**绝对 ms 一律视为同时段比值**；
  跨时段的绝对数（例如早先那次"加法全关 5403.90 ms"）**不能**与新数据混用 —— 今天在同一时段内
  两套结构分别测到 4251 / 4207 ms，比值 0.41 才是那个结论的依据。
* 固定用 `N = 2^k−1`（k 取素数指数 ⇒ 不会命中因子 ⇒ 每次都跑满全部 bit），避免"提前找到因子"污染计时。
* 每次运行新建/清空临时目录，避免 ckpt/save 残留。
* 同一配置重复 3 次，实测抖动 < 0.1%（例：param3 M511 三次 10880.7 / 10873.9 / 10873.6 ms）。
* **探针只能在"同一套代码结构"内部比较加法密度**（2026-09-25 新增，见 §5.1）。实测两套结构做**同样**
  "每 bit 一次 xz 倍点"，一次测到 4207 ms、另一次 4251 ms（同一结构内可信），但换成融合式的旧结构
  曾测到 5404 ms —— 同名算子数下跨结构能差 20–30%。所以：**跨结构拼边际成本会得出错误结论**，
  定价要么用**算子数**（结构无关），要么在同一结构内做 A/B。
* **CGBN 的算子本身对操作数值不敏感**（`-DPROBE_VALUE_MODE=0/1/2` 实测 0.91/0.91/0.91 ns，
  zero 与 one 操作数同价）⇒ 探针状态退化**不会**让算子变快；探针读数的偏差来自结构，不是值。

---

## 2. 结论 1：CGBN 的"Volta 默认"在 Ada 上是对的（方向关闭）

`cgbn/include/cgbn/cgbn.h:67-76` 按架构自动选乘法链：

```c
#if __CUDA_ARCH__<500   -> XMP_IMAD
#elif __CUDA_ARCH__<700 -> XMP_XMAD
#else                   -> XMP_WMAD     // Volta 及以后，含我们的 sm_89
```

强制切换后测 `mont_mul`（RTX 4060，TPI=16 / BITS=3072，2048 实例，1500 迭代）：

| 变体 | `mont_mul` | 相对 |
|---|---|---|
| **WMAD（sm_89 默认）** | **17.56 ns** | **1.00** |
| IMAD | 25.87 ns | 1.47× 慢 |
| XMAD | 40.59 ns | 2.31× 慢 |

⇒ "CGBN 没为 Ada 调过"这条在**乘法链层面不成立**。关闭该方向，不再投入。

---

## 3. 结论 2：专用平方（`mont_sqr`）——上限 12–14%，且要先解决跨 lane

**事实**：`cgbn/include/cgbn/impl_cuda.cu:1032` 的 `mont_sqr` 直接转发成 `mont_mul(r,a,a)`。
实测确认 α = 1.0（TPI=16/3072：`mont_mul` 17.56 ns vs `mont_sqr` 17.32 ns，差 <2%；tier3 复跑完全一致）。

**上限推导**：CIOS / 交错的 Montgomery 乘法里，一次乘 ≈ 乘积部 n² + 归约部 n² = 2n²，而**归约部不可对称化**
（它不是平方结构）。理想平方 = n²/2 + n² = 1.5n² ⇒ **α_min = 0.75**。我们每 bit 是 4M+4S：

$$\text{提速}=\frac{8}{4+4\alpha}\ \Rightarrow\ \alpha=0.75:1.143\times,\quad \alpha=0.70:1.18\times$$

**CGBN 特有的折扣**：`core_mont_wmad.cu:41` 里每个线程 `for(thread=0; thread<TPI; thread+=2)` 遍历
**所有源 lane**、用 `__shfl_sync` 取 `b` 的字，再对自己那份 `a` 做 LIMBS×LIMBS 全块乘 ——
即**每一对无序 lane {i,j} 被算了两次**（lane i 用 b_j，lane j 用 b_i）。要吃满平方对称性必须跨 lane
交换**部分列和**（不只是操作数）。只对称化对角线块（lane i×i）只省 `(1/TPI)·50%` 的乘积部：
TPI=16 时约 **1–2%**，不值得。

**结论**：可以做，但排在 3、4 之后；若做，先写"跨 lane 部分和交换"的小原型实测 α，只有 α ≤ 0.8 才铺开到全档位。

---

## 4. 结论 3：删掉冗余 `normalize_addition`（已实现，实测 +5.4% / +6.7%）

### 4.1 契约证据

`cgbn/include/cgbn/core/core_mont_wmad.cu:178-189` 的乘法尾部：

```c
c=-fast_propagate_add(c, r);          // 借位
t=n[0]-(group_thread==0);             // -n
r[i]=chain.add(r[i], ~n[i] & c);      // 按掩码条件减 n
fast_propagate_add(c, r);
```

⇒ **`mont_mul`/`mont_sqr` 返回时结果已经 < n**（这就是 CGBN 的后置条件）。
因此 kernel 里紧跟在乘/平方之后的 `normalize_addition`（`cgbn_compare` + 条件 `cgbn_sub`）是**纯开销**。

### 4.2 每个 bit 的分布

| kernel | `normalize_addition`/bit | **冗余**（紧跟 mul/sqr） | 必要（紧跟 add/sub/shift） |
|---|---|---|---|
| param3 `double_add_v2` | 11 → 3 | **8** | 3（两个 add + 一个 shift） |
| param0 `double_add_v2_suyama` | 9 → 4 | **6** | 4 |

### 4.3 实测（dev 构建，M511/M761，4096 曲线，B1=1e5，GPU 1，3 次中位）

| 配置 | 改前 | 改后 | 提升 |
|---|---|---|---|
| param3 M511 | 10880.7 ms | **10322.4 ms** | **+5.4%** |
| param0 M511 | 12340.0 ms | **11566.2 ms** | **+6.7%** |
| param3 M761 | 10875.6 ms | **10323.2 ms** | **+5.4%** |
| param0 M761 | 12335.8 ms | **11568.3 ms** | **+6.6%** |

**为什么只有 5–7%，而算子模型预测 ~20%（512 bit）？** 因为 `normalize_addition` 与乘法**没有依赖关系**，
在 SM 上能与乘法那条长依赖链重叠执行 —— 删掉它省下的是"发射槽"而不是关键路径。
⇒ 教训：**辅助整数运算的独立开销会被乘法链掩盖**，评估时必须实测，不能只按算子单价相加。
（反过来，结论 4 里被删掉的是**乘法**本身，那是关键资源，所以加法次数的减少会近似线性兑现。）

### 4.4 正确性验证

* `tools/stat/ecm_hitrate.ps1 -Engine mont,gpu -GpuParam 0 -Count 16 -Curves 8`：
  CPU 与 GPU **逐曲线命中集一致**（两边都是 37/128，且所有命中因子都等于 p）。
* 全量构建后跑 `tools/test/test_cuda_param0.ps1`（CPU↔GPU 逐曲线一致、64 位 σ、gmp-ecm 互通、
  硬杀+恢复 4096/4096、param3 回归）：**2026-09-25 在删掉冗余 `normalize_addition` 的全量二进制上
  18/18 全 PASS（ALL OK）**。
* **第 7 轮（`__maxnreg__` per-tier 寄存器表 + CLI 冲突检查）在全量二进制上的复测**：
  `test_cuda_param0` **ALL OK（19 PASS / 0 FAIL —— 脚本后来又加过一项，所以是 19 不是 18）**、
  `test_cuda_param2`（M1279）**ALL OK（7 checks）**、
  新增 `tools/test/test_cli_args.ps1` **ALL OK（9 checks）**、
  `ecm_hitrate.ps1 -Engine mont,gpu -GpuParam 0`：CPU `mont/simd` 与 `gpu-param0`
  **完全一致（两边都是 37/128 = 28.9062%，素数 16/16）** ✓。

---

## 5. 结论 4：add-chain / w-NAF —— 结构上不合法，实测链比梯子慢 ~10%

### 5.1 梯子的成本怎么分（两次独立测量）

| 结构 | 每 bit 名义算子 | 实测 gputime（M511 素数，4096 曲线，B1=1e5，GPU 1） |
|---|---|---|
| **现状**：融合 double+add，每 bit 都算加法 | 4M+4S+廉价 | **10326.00 ms** |
| 加法半边**完全关掉**（`-DECM_PROBE_ADD_DENSITY=1e6`，结果错） | 2S+2M+廉价 | **4251.14 ms** |
| 只倍点（**另一套**代码结构：`-DECM_PROBE_CHAIN_W=1e6`） | 2S+2M+廉价 | **4206.77 ms** |

两套互不相干的代码结构给出同一个"只倍点"数字（4251 / 4207，差 1%）⇒ 这个基线可信。
⇒ **加法半边占梯子运行时间的 ~59%**，倍点只占 ~41%（(10326−4230)/10326 = 0.590）。

> ⚠ **勘误（2026-09-25 当天修正）**：本文档早先版本写的是"倍点 4.24 / 加法 3.86 乘当量，
> 模型 `cost(k)=4.24+3.86/k`，天花板 1.91×"。那组数字是对 k=1..6 **最小二乘外推**出来的，
> 而且其中"k=10⁶ → 5403.90 ms"这一行**今天无法复现**（今天两种结构都测到 ~4230 ms，重复 3 次稳定）。
> 按今天的数据：
>
> * 加法半边的**边际**成本 = 10326 − 4230 ≈ **6100 ms**（不是 4917 ms），倍点 ≈ **4230 ms**；
> * "关掉全部加法"的天花板 = 10326/4230 = **2.44×**（不是 1.91×）；
> * **可加模型 `c0 + c1/k` 本身不成立**：k=2..6 实测 7805 / 7024 / 6615 / 6381 / 6218，
>   全部**高于** `4230 + 6100/k` 的预测 7280 / 6263 / 5755 / 5450 / 5247 —— 加法半边的单位成本
>   随其密度**超线性**上升（活跃值、调度、寄存器压力的非线性效应，正是 §4 的教训）。
>
> ⇒ **新增纪律（§1）**：探针只能在**同一套代码结构内部**比较不同加法密度；跨结构比较（哪怕名义算子数相同）
> 会差 20–30%，不能当作算子成本模型用。

k 扫描（同一结构内，加法每 k bit 执行一次，k>1 结果错）——只说明"形状"，不构成可达收益：

| k | 实测 gputime | 相对 k=1 |
|---|---|---|
| 1（现状） | **10320.95 ms** | 1.00 |
| 2 | **7805.06 ms** | 1.32× |
| 3 | **7024.34 ms** | 1.47× |
| 4 | **6614.89 ms** | 1.56× |
| 5 | **6380.67 ms** | 1.62× |
| 6 | **6217.62 ms** | 1.66× |
| 加法全关（今天复测） | **4251.14 ms** | **2.43×**（天花板，不可达） |

### 5.2 实测：把融合步换成"链的算子组合"（w=2 及以上的 w-NAF 排布）

`-DECM_PROBE_CHAIN_W=M` 让 kernel 每 bit 只做 **1 次 xz 倍点**，并且每 M bit 做一次**真实的**
投影差值差分加法（EFD `dadd-1987-m-3` = 4M+2S，参数为伪值，结果错，纯计时）。M = w+1 就是
"宽度 w 的 NAF、且不变量维护免费"这个**上界**：

| M | 相当于 | 每 bit 算子 | 实测 gputime | 相对现状 |
|---|---|---|---|---|
| — | 现状（梯子） | 4M+4S | **10326.00 ms** | 1.00 |
| 1 | PRAC 式（每 bit 一次链加法） | 2S+2M + 4M+2S | **9863.96 ms** | 1.047× |
| 2 | w=1 | + 半次 | **7027.41 ms** | 1.47× |
| 3 | **w=2（NAF 密度 1/3）** | + 1/3 次 | **6079.98 ms** | 1.70× |
| 4 | w=3 | + 1/4 次 | **5594.87 ms** | 1.85× |
| 5 | w=4 | + 1/5 次 | **5332.60 ms** | 1.94× |
| 10⁶ | 只倍点（同结构基线） | 2S+2M | **4206.77 ms** | 2.45× |

**怎么读这张表（重要）**：

* 同一结构内的边际成本 = 4207 + 5649/M（对 M=1..5 拟合误差 ≤ 0.4%）⇒ **链加法的边际成本是
  倍点的 1.34 倍**（5649/4207），与算子数预测（6.0/4.24 = 1.42）吻合到 6% ✓ **算子模型在结构内部成立**。
* 但 M=1 那行**不能**理解成"链比梯子快 5%"：M=1 的每 bit 算子数（10）**多于**梯子（8.24），
  它更快是**结构效应**（这条探针的顺序式代码比融合式 `double_add_v2` 的每算子开销低 ~28%，
  见 §5.1 的两套"只倍点"基线 4207 vs 4251... 同一结构内；跨结构则是 4207 vs 5404 的旧值已被推翻）。
  探针的状态是伪值，不是合法算法 —— 这张表的用途是**上界**，不是收益。
* ⇒ **"宽度 w 的 NAF、维护免费"的上界是 1.70×（w=2）～1.94×（w=4）**。下面 §5.4 说明它**不可达**，
  而且真实链反而更慢。

### 5.3 探针怎么用

```powershell
# 只测时间：结果全是错的（文档 §5.1/§5.2 顶部与 kernel 头部都有警告）
#   A) 加法密度探针（旧）：加法每 k bit 一次
cmake -S . -B build_cuda_dev -DECM_PROBE_ADD_DENSITY=3
#   B) 链算子探针（新）：1 倍点/bit + 每 M bit 一次投影差值差分加法
cmake -S . -B build_cuda_dev -DECM_PROBE_CHAIN_W=3
cmake --build build_cuda_dev --config Release --target ecm_cuda
powershell -NoProfile -ExecutionPolicy Bypass -File tools\bench\cuda_kernel_ab.ps1 -Label k3 `
  -Bits 511 -NExpr (Get-Content .bench_tmp\cuda_p511_prime.txt -Raw).Trim() -Curves 4096 -B1 1e5 -Device 1
# 测完务必两个都还原（-DECM_PROBE_ADD_DENSITY=1 -DECM_PROBE_CHAIN_W=0），否则结果全错
```

> **N 必须取素数**（见 §6.6）：探针的垃圾状态会撞出假因子，驱动按"退化曲线"提前结束批次。
> **且必须两套探针都关掉**：它们互相独立，只关一个仍然错。


### 5.4 为什么这个上界不可达：w-NAF 在 x-only 下"不合法"，真实链在算子数上就输了

**(1) 结构上不可能（不是"贵"，是没有公式）。** x-only 坐标没有一般加法：要算 x(P+Q) 必须先知道
x(P−Q)（差分加法）。窗口法要算 R + dP（d 是奇数倍），需要的差值是 **x(R − dP)** —— 而 R 每过一个
窗口就变成 2^w R + …，于是这个差值是**每个窗口、每个 digit 都不同的新点**：既不在任何预计算表里
（表里只有 {dP}，而且还要 {d_i−d_j}P 整套），也无法从 x-only 数据推出来。**所以 x-only 的
w-NAF / 滑窗 / 固定窗口不是"代价高"，而是根本不合法。** 旁证：EFD 的 Montgomery XZ 页只有 1987 年的
那几条公式、没有任何窗口/字典条目；co-Z 文献全在短 Weierstrass/Jacobian 上，没有 Montgomery x-only 的
co-Z 工作；Goundar–Joye–Miyaji 明确**拒绝 NAF**，只用"零位有符号数字表示"（仍然一位一 bit、无表）。

**(2) 加法次数有下界，而且低于赢的门槛。** 一维差分加法链的加法次数下界是 **1.44042 次/bit**
（Montgomery 1992 Thm 7 / Bernstein 2006）。而在**算子数**口径下赢的门槛是：

```
链:  4.24·d + 6.0·a     <  梯子: 8.24        (d = 倍点数/bit, a = 加法数/bit)
PRAC 实测 d = 0.135  ⇒  a < (8.24 − 0.57)/6.0 = 1.26 次/bit
```

**1.44 > 1.26 ⇒ 任何"差值是投影点"的一维链都不可能赢。** 关键在于：链里每个加法需要的差值点都是
**投影**的（Montgomery 1992 的 PRAC 把 `C = X_{a−b}` 当活寄存器带着走），代价 4M+2S = 6.0；
唯一能让差值保持**仿射且归一化**（2M+2S ≈ 4.3）的结构就是梯子自己（差值 = 固定的起点）。

**(3) 真实链的实测算子密度（两条独立来源，都是加法主导）。**

| 来源 | 倍点/bit | 加法/bit | 折成 mont 算子/bit | 对梯子 8.24 |
|---|---|---|---|---|
| Bernstein 2009 实测 GMP-ECM 6.2.3（B1=10⁶，b=1442099） | 0.135 | **1.388** | 0.57 + 8.33 = **8.90** | **慢 ~8%** |
| `tools/stat/prac_cost.py` 转录 Prime95 `lucas_mul`（B1=1e5） | 0.40 | **1.31** | 1.70 + 7.86 = **9.56** | 慢 ~16% |

（同一脚本的交叉校验：算子数与 Prime95 自己的 `lucas_cost()` 差 3.4%，所以"加法主导"这个结论是稳的。
此前文里写的"1 倍点 + 0.7 加法/bit"是错的。）
再加两个 GPU 特有的扣分项：链要 **3 个活跃 XZ 点**（GMP-ECM 的预计算链路径要 16 个 XZ 寄存器），
寄存器/占用率比梯子的 2 个点贵；且链的**步进是加法**（0.135 倍点/bit 意味着倍点很少、加法很多），
而加法在 GPU 上并不比倍点便宜。

**(4) 不能用"跨结构的实测边际成本"来定价链 —— 本节最重要的方法论结论。** 把 §5.2 里链探针测到的
加法边际成本（5649 ms/次）和同一个探针的倍点基线（4207 ms/bit）代入 PRAC 的密度，会算出
`0.135·4207 + 1.388·5649 = 8412 ms < 10326 ms`，即"PRAC 快 23%" —— **这是错的**：那是把两套
代码结构的边际成本拼在一起（§5.1：跨结构同样算子数也能差 20–30%）。所以本文档**不**据此声称链更快；
稳妥的口径是**算子数**（算子数是结构无关的），结论是链慢 8–16%。要最终定案只能真写一个 PRAC 移植，
但**前提是算子数已经判它输**，投入产出不划算。

**(5) 校准复查（回应"你们的加法 3.86 是哪来的"）。** 我们的加法半边是 EFD 的 x-only 差分加法
`mdadd-1987-m`，但**差值点的仿射 x 是常数 2**（param3/batch 族起点 x0 = 2），这一项被折进
`cgbn_shift_left(v,1)`（免费），所以是 **2M+2S+shift** 而不是教科书的 3M+2S；倍点里 `dK = K·d`
用 `special_mult_ui32`（32 位乘 + 单字归约）代替全宽 a24 乘。教科书严格 EFD 记法
（`mdbl-1987-m` 1M+2S+1a + `mdadd-1987-m` 3M+2S）是 **9.0** 算子/bit，我们实测 **8.10**
⇒ **参数化本身已经替我们省掉约 10%**，两处 kernel 注释都写明了这一点。
命名以 EFD 为准：**`dadd-1987-m-3`（4M+2S）才是投影差值**那一条，仿射的是 `mdadd-1987-m`（3M+2S）。

**⇒ 结论 4 关闭**：梯子（每 bit 1 次倍点 + 1 次"仿射、归一化差值"差分加法）是 x-only 下的最优结构；
w = 2/3/4 的上界 1.70×/1.85×/1.94× **不可达**，真实链在算子数上就慢 8–16%。
**但 §5.1 顺带暴露了一个真实的、与算法无关的杠杆**：加法半边占 59% 的运行时间，而它在融合步里的
**单位算子成本比倍点高 ~44%**（两者名义算子数相同）——这是调度/活跃区间/占用率问题，见 §8。

> 本节结论的完整证据链（候选方案逐项枚举与定价、EFD 公式清单与勘误、原始引文与不确定性标注）见
> **`docs/ECM_XONLY_SCALARMUL_RESEARCH.md`**（附 `docs/ECM_XONLY_SCALARMUL_QUOTES.txt` 的源码原文摘录）。
> 该附录同时给出了"x-only 窗口法不合法"这一结论的**搜索结果边界**：没有找到任何在 Montgomery x-only 上
> 做带表窗口/NAF 的文献，这是"证据 + 上述结构性证明"，不是"穷尽证明"。

---

## 5.5 kernel 工程杠杆：每批曲线数 / TPB / MAX_ROTATION / 寄存器上限（2026-09-25 实测）

全部为 **M511 素数、B1=1e5、param3、GPU 1**，同一时段内比较（跨时段绝对数会漂，见 §1）。

**① 每批曲线数（`-gpucurves`，TPB=256/TPI=4 时）**

| 曲线数 | block 数 | gputime | curve-bits/s | 相对 4096 |
|---|---|---|---|---|
| 2048 | 32 | 4083.57 ms | 72.39 M | 1.00 |
| 4096 | 64 | 8173.03 ms | 72.34 M | 1.00 |
| 8192 | 128 | 15195.45 ms | 77.82 M | **1.076** |
| 16384 | 256 | 30146.75 ms | 78.45 M | **1.084** |
| 32768 | 512 | 59239.70 ms | 79.84 M | **1.104** |

**② TPB（`-DECM_TPB`，8192 曲线，ROT=4）**

| TPB | block 数 | curve-bits/s | 相对 256 |
|---|---|---|---|
| 512 | 64 | 66.19 M | **0.847** |
| 256（历史默认） | 128 | 78.18 M | 1.000 |
| 128 | 256 | 79.15 M | 1.012 |
| 64 | 512 | 79.19 M | 1.013 |

⇒ 主变量是 **block 数 = 曲线数/(TPB/TPI)**：512 线程/块把同一批曲线的 block 数砍到 64，直接掉 15%。

**③ MAX_ROTATION（`-DECM_MAX_ROTATION`，TPB=256，8192 曲线）**：4 → 78.18 M、2 → 78.31 M、
1 → 78.52 M。**差异 ≤0.4%，实际无影响**（CGBN 里这个参数对 4-limb/线程的档位几乎不起作用）。

**④ 寄存器上限（`-DECM_MAXRREG` → `nvcc --maxrregcount`，TPB=128/ROT=1，8192 曲线）**
（⚠ 第 7 轮起机制已换成 **per-tier `__maxnreg__`**，见 §5.7；下表是当时用"按文件"的
`--maxrregcount` 量到的，数值仍有效，但结论只对 ≤2048 bit 档位成立）

| 上限 | curve-bits/s | 相对 256/4/无上限（78.32 M） |
|---|---|---|
| 无（编译器给 72 寄存器） | 79.38 M | 1.014 |
| 80 | 80.66 M | 1.030 |
| 64 | 81.12 M | 1.036 |
| **56** | **82.01 M** | **1.047** |
| 48 | 80.30 M | 1.025（开始溢出） |

⇒ 把 72 个寄存器压到 56：占用率从 7 块/SM（65536/(128×72)）提到 9 块/SM（65536/(128×56)），
**+4.7%**；再压到 48 就出现寄存器溢出，反而掉回去。**寄存器上限是这一组里最大的单项**。

**⑤ TPB 与寄存器上限是同一件事（都通过"块数/占用率"起作用）**：TPB=256/ROT=1/REG=64 也有 81.75 M
（+4.4%），与 TPB=128/REG=56 的 82.01 M 只差 0.3%。⇒ 真正的主变量是 **warps/SM**
（256/64 → 4 块/SM × 8 warps = 32 warps；128/56 → 9 块/SM × 4 warps = 36 warps）。

**⑥ 计算顺序（`-DECM_STEP_VARIANT=2`，把融合步改写成显式 prep 寄存器、消除写后读相关）**：
81.14 M vs 81.12 M ⇒ **完全没差别（+0.02%）**，两者寄存器分配也都是 72 ⇒ ptxas 早就重排好了。
**结论：融合步的"结构开销"（§5.1）不是源码顺序造成的**，而是硬件层面的占用率/延迟，靠"调顺序"拿不到。
variant 2 保留在代码里作为证据（默认 1）。

⇒ 与 ① 合起来：**"每批 ≥8192 曲线"（+7.6%）+ "TPB/寄存器上限"（+4.7%）≈ 12% 的净收益，完全不改算法**。

**⑦ 全量生产二进制的前后对比（M1021，8192 曲线，B1=1e5，param3）**

| 配置 | curve-bits/s | 相对 |
|---|---|---|
| 历史默认（TPB=256 / ROT=4 / 无寄存器上限） | 38.18 M | 1.00 |
| **新默认（TPB=128 / ROT=1 / ≤2048 bit 源文件 `--maxrregcount=56`）** | **39.24 M** | **1.028** |

（1021 bit 用的是 TPI=8 那套实例化；小位宽档位在 dev 构建上量到的是 +4.7%，大位宽这里 +2.8% ——
寄存器压力随每线程 limb 数上升，56 的上限在大档位上收益递减，这也是为什么**≥2560 bit 档位不加上限**。）

**⑧ 占用率告警（已落地）**：`kernels/cuda/cgbn_stage1.cu` 启动时会算
`blocks_per_sm = cudaOccupancyMaxActiveBlocksPerMultiprocessor(...)`，填不满就打印：

```
GPU: CGBN<8, 768> kernel, N is 521 bits (64 blocks x 128 threads)
GPU: warning: 64 blocks fill only 29% of this device (24 SMs x 9 blocks/SM);
              raise -gpucurves to about 3456 (measured +7.6% at 8192 vs 4096 curves)
```

实测设备上限：24 SM × **9 块/SM**（TPB=128、56 寄存器）＝ 216 块；历史配置（TPB=256、72 寄存器）
只有 3 块/SM ＝ **72 块** ⇒ 新配置的可并行块数是原来的 **3 倍**，这正是那 +4.7% 的来源。
（告警走 `OUTPUT_NORMAL`：`ecm.ini` 默认 `verbose = true`，队列流程里可见；手工跑加 `-v`。）

---

## 5.6 结论 4 之外：param2 经济学（2026-09-25 实测，含主机侧生成成本）

**问题**：param2（gmp-ecm 的 "batch 2"，6-挠 batch 族）的成功率与 Suyama(param0) 同档
（实测 D_eff 20.16 vs 20.90），但它每 bit 的算子数是 **5M+4S = 9**，param0 是 **6M+4S = 10**，
param3 是 **4M+4S + 32 位廉价乘 ≈ 8.2**。算子数是不是能兑现成吞吐？还有主机侧建曲线要付多少？

**源码依据**（`.refactor/gmp-ecm/parametrizations.c`）：`get_curve_from_param2` 最后一行是
`mpres_set_ui (x0, 2, n)`（L373）⇒ **param2 的 stage-1 起点也是 x0 = 2**，所以差值点不需要乘法
（和 param3 一样折进 `shift_left`）；而它的 `a24 = (A+2)/4` 是**满宽**剩余类（A 由 x₃ 推出，L350-371），
所以常数乘仍是满模乘 ⇒ 每 bit **5M+4S**。生成路径则最贵：固定曲线 `y²=x³+36` 上的 `σ·(−3:3:1)`
加法链 + **三次模逆**（L323 / L338 / L361）。

**① kernel 算子形状实测**（dev 构建，M511 素数，8192 曲线，B1=1e5，param0 数据路径，取中位）：

> ⚠ **勘误（同一天修正）**：本节初版把 param2 的形状测成了"**32 位** a24 + shift"（用 `special_mult_ui32`），
> 那其实是 **param3 的形状** ✗。param2 的 `a24 = (A+2)/4` 是**满宽**剩余类，只有**差值**是常数 2
> （`parametrizations.c:373` 的 `x0 = 2`）。修正后的形状 = **满宽 a24 乘 + 差值折进 shift** ⇒ 9 算子/bit ✓。

| step 形状 | 算子/bit | curve-bits/s | 相对 param0 |
|---|---|---|---|
| param0（满宽 a24 + xdiff 乘） | 10 | **71.52 M** | 1.000 |
| **param2 形状（满宽 a24 + 常数差值折成 shift）** | **9** | **79.15 M**（79.15–79.42，多次） | **1.107** |
| param3（参考，32 位 d + shift） | 8.2 | 82.01 M | 1.147 |

⇒ **算子数兑现了**：10/9 = 1.111 预测 vs **1.107 实测**（差 0.4%）✓（`-DECM_PARAM2SHAPE=1` 强制走
param2 形状，跑在 param0 数据路径上，结果错、只计时）。
⇒ param2 比 param0 **快 ~11%**，比 param3 慢 ~3.5%。

> ⚠ **陷阱（本轮真踩，且代价 34%）**：我一开始用"运行期判定 `xdiff == 2`（Montgomery 形式）"来自动选
> shift/multiply，理由是"warp-uniform 分支应当免费"。实测 **寄存器从 71 涨到 92**（xdiff 被迫跨整个
> bit 循环保活），param0 吞吐 **71 M → 47 M（−34%）** ✗✗。热循环里不要加这种"看起来免费"的判定；
> **正确做法是让 param2 用独立的 kernel 家族**（同一套实例化、`const_diff = true`），由 dispatch 选家族
> —— 和 param0 现在就是一个独立家族一样。已回退，并把 `const_diff` 参数留在
> `double_add_v2_suyama` 里给下一个家族用。

**② 主机侧建曲线成本实测**（新工具 `tools/bench/param2_gen_cost.cpp`，GMP，N 取梅森素数，
3000 曲线跑 3 次取最快）：

| N | param2（gmp-ecm `get_curve_from_param2`） | param0 形状（模型：1 次模逆 + ~12 次模乘） | param3 形状（对照） |
|---|---|---|---|
| 521 bit | **0.127 ms/curve** | 0.008 | ≈0.000 |
| 1021 bit | **0.227 ms/curve** | 0.014 | 0.000 |
| 4423 bit | **1.916 ms/curve** | 0.071 | 0.001 |

（独立交叉验证：直接用 gmp-ecm `-param 2` 与 `-param 0` 在同一 N 上做同 B1 的批跑，差值 0.31 ms/curve
@1021 bit，与此处 0.23 ms/curve 同量级 ✓。param0 的形状是**成本模型**不是逐行转录，见工具头注释。）

**③ 净收益（①②与 kernel 每曲线时间放一起）**：M1021、8192 曲线、B1=1e5 时 param3 基线的 kernel
每曲线 = 30136/8192 = **3.68 ms**；param0/param2 的 kernel 每曲线按 §5.5 在 M511 实测的比值
（0.866 / 79.15÷82.01 = 0.965）换算：

| 参数化 | kernel | 生成 | 合计 | 相对 param0 |
|---|---|---|---|---|
| param0 | 4.25 ms | 0.014 ms | 4.26 ms | 1.000 |
| **param2** | 3.81 ms | 0.227 ms | **4.04 ms** | **1.054** |
| param3 | 3.68 ms | ~0 | 3.68 ms | 1.158 |

* **对 param0**：B1=1e5 时净 **+5.4%**；B1 ≥ 1e6 时 kernel 每曲线涨 10 倍以上、生成占比掉到 <0.6%，
  净收益回到 **+10.7%**。⇒ 生成成本只在**小 B1**（B1 ≲ 1e5 且 N ≲ 1024 bit）才吃掉收益。
* **对 param3**：每曲线慢 ~10%（4.04 vs 3.68 ms），但 D_eff 高约 3 倍（20.16 vs 6.41），
  **等时成功率**仍是大赢（与此前实测的等时口径 1.08–1.26× 一致）。

**④ 实现与验证状态（2026-09-25 第 4–5 轮：已实现、**已验证**、默认仍不启用）**

已落地（代码在树上，需要 `--gpu-param 2` 显式启用）：
* **kernel 独立家族** `kernels/cuda/cgbn_stage1_kernels_param2.cu`：与 suyama 家族同一套网格，
  但 `kernel_double_add_suyama<params, CONST_DIFF=true>` ⇒ 5M+4S。**这是第 4 轮的重点教训**：
  一开始用"运行期 `xdiff == 2`"判定，寄存器 71→92、param0 吞吐 −34%，已回退（见 ① 的陷阱）。
* **主机侧生成** `set_p_2p_param2()`（`cgbn_stage1.cu`）：把 gmp-ecm 的 `get_curve_from_param2`
  逐行转写（`σ·(−3:3:1)` 于 `y²=x³+36`、x3、A、a24、(2:1)/2P），`σ` 取 32 位标量。
* **路由 + 存档**：`--gpu-param 2` / ini `gpu_param = 2`、7 词 Suyama 缓冲区布局、检查点
  `gpu_param = 2`、存档写 `PARAM=2`（`ecm_save.cpp` 的 writer 现在按 param id 参数化）。
* **OpenCL 后端**对 2 与 0 一样明确拒绝（内核仍是 batch 形状）。
* 回归测试 **`tools/test/test_cuda_param2.ps1`**（7 项，ALL OK）。

**验证：通过与 gmp-ecm 的逐字节对照**（方法是先证明它可信：**param0** 的 stage-1 x 与 gmp-ecm
在同一 σ/B1 下**逐字节相同** ✓，param2 用同一把尺子量）：

| 检查（M521，σ=1000000） | 结果 |
|---|---|
| 我们 param2 的 X == gmp-ecm `-param 2` 的 X（B1=1e3） | **相同** ✓ |
| 同上（B1=1e5） | **相同** ✓ |
| 存档形态（`PARAM=2`、`SIGMA`、原始 N）+ gmp-ecm `-resume` 接受 | 通过 ✓ |
| 生成的 A vs 独立仿射参考实现 | 相同 ✓ |

**定位过程中修掉的三个真 bug（按发现顺序）**：
1. **`p2_jac_dbl` 的输出别名输入**：`Z3 = 2·Y1·Z1` 写在 `Y3` 之后，而 `Y3` 已覆盖 `Y1`
   ⇒ Z3 错 ⇒ 点离开曲线 ⇒ 推出的 **x3 = 1、A = −2**（奇异曲线）。改成"中间量全用临时变量、
   最后统一写回"。
2. **我误判了"曲线要不要重标定"**：一度按 `FindGroupOrderParam2` 注释里的 `a/b` 形式把
   a24 改成 `(A/(4A+10)+2)/4` ✗ —— 用一个**独立的 Python x-only stage-1 参考 + 候选约定搜索**
   证明了 **(A, x0=2) 才是对的**（它精确复现 gmp-ecm 的 X），于是回退 ✓。
3. **真正的元凶：结果字索引用错了 buffer 布局** ✗✗ —— `words_per_curve` 已跟着
   `suyama_layout` 改，但 `p1_word`/`p2_word` 仍是 `param0 ? 3 : 1` ⇒ param2 从**错误的字**
   （word 1 = a24，而不是 word 3 = aX）解码结果 ⇒ 存出来的 X 永远是错的，而曲线与梯形其实
   一直是对的 ✓。改成 `suyama_layout ? ...` 后立刻逐字节对上 ✓。

**端到端吞吐（M511 素数，8192 曲线，B1=1e5，GPU 1，dev 构建）**

| 参数化 | gputime | curve-bits/s |
|---|---|---|
| param3 | 14446 ms | 81.86 M |
| param0 | 16617 ms | 71.16 M |
| **param2** | **15713 ms** | **75.25 M** |

⇒ param2 比 param0 **快 5.7%**（含主机侧生成 ~0.3 s/8192 曲线 ✓）。**注意**：形状探针（§5.2）
给的上界是 +10.7%，而真实路径只有 +5.7% ⇒ **约 5 个百分点在 param2 路径里被吃掉了**，
原因未定（探针跑的是 suyama 家族 + 强制 `const_diff`，真实路径跑 param2 家族；寄存器上限
现已对两者都生效 —— 但这一点是**下一轮要查的开放项**）。
另：param2 家族一开始**没被** `ECM_MAXRREG_SUYAMA` 覆盖（只列了 suyama 源文件），加上后
从 74.12 M 升到 75.25 M。

**⑤ 结论（取舍仍在，但技术验证已完成）**：

> 若 stage 2 交给 Prime95 → 保持 param0（本仓库现状，Prime95 读不了 `PARAM=2`）；
> 若 stage 2 交给 gmp-ecm（或只做 stage 1） → param2 相对 param0 快 **5.7%（B1=1e5）～约 11%
> （大 B1，生成可忽略）**，且成功率同档，值得做。

**⑨ param0 路径的寄存器上限（同一天补测，M511/8192 曲线、param0 数据路径）**

| suyama 源文件的上限 | curve-bits/s | 相对无上限 |
|---|---|---|
| 无（编译器默认） | 71.07 M | 1.000 |
| **64** | **71.68 M** | **1.009** |
| 56 | 71.25 M | 1.003 |

⇒ param0 只拿到 **+0.9%**（param3 是 +4.7%）：它每 bit 要 10 个满模乘（param3 只有 8.2），
寄存器压力本来就高，压到 56 就开始溢出。**param0 与 param3 的差距（71.7 vs 82.0 M）主要是算子数，
不是占用率** —— 这条也是"先量再改"才没走错（我原本以为 param0 缺的是那 +4.7%）。已按实测把
`suyama` 源文件的上限设为 **64**（`-DECM_MAXRREG_SUYAMA`）。

---

## 5.7 寄存器预算：把上限编码进位宽实例化（`__maxnreg__`，第 7 轮）

**问题（用户提出）**：用 `__launch_bounds__` 或 `__maxnreg__(N)`，**在不同位宽的实例化里把寄存器数量编码进去**；
并指出 `--maxrregcount=N` 只能作用于**单个文件**。
**结论：建议成立，已落地**（原来的 `__launch_bounds__(TPB, MIN_BLOCKS)` 已删除）；而且"按文件"的限制
在本仓库是**真的错**，不只是不够灵活。

**① 为什么按文件的上限是错的（实测踩到）**：`-DECM_MAXRREG_SUYAMA=64` 是 §5.6 为 **≤2048 bit** 档位
实测出来的 +0.9%，但它作为 `nvcc --maxrregcount` 会套到 `cgbn_stage1_kernels_suyama.cu` 里
**所有 ≥2560 bit 档位**上（2560…8192），而那些档位从没实测过、且更重 ⇒ 这就是"按文件"的硬伤。
`__maxnreg__` 是**单个 kernel 的属性**，写法上可以取模板常量，于是"每个位宽一个预算"天然成立 ✓。

**② 机制与三个必须知道的细节**

| 事项 | 结论 |
|---|---|
| 可用性 | CUDA 13.3 `crt/host_defines.h` **无条件**定义 `__maxnreg__`（`__attribute__((maxnreg(a)))` / `__declspec(maxnreg(n))`），sm_89 实测可用 ✓ |
| 粒度 | 参数可以是模板常量：`__global__ void __maxnreg__(params::REG_TARGET) kernel_double_add(...)`，`params` 就是该实例化的 `cgbn_params_t<TPI, BITS>` ⇒ **随位宽变化** ✓ |
| 取值 | **`N` 必须 ≥1**：`__maxnreg__(0)` 直接编译失败（`The maximum number of registers that can be allocated per thread must be positive`，§6.20）⇒ 要表达"不限制"只能写 **255**（sm_89 每线程上限）|

落地形式（`kernels/cuda/cgbn_stage1_kernel.h`）：

```cpp
static const uint32_t REG_TARGET = (ECM_REG_TARGET_FORCE > 0) ? ECM_REG_TARGET_FORCE
                                   : ((bits <= 2048u) ? 56u : 255u);
__global__ void __maxnreg__(params::REG_TARGET) kernel_double_add(...)          // param3
__global__ void __maxnreg__(params::REG_TARGET) kernel_double_add_suyama(...)   // param0 / param2
```

`-DECM_REG_TARGET_FORCE=N` 覆盖整张表（0 = 用表；A/B 用）；`-DECM_MAXRREG_SMALL` /
`-DECM_MAXRREG_SUYAMA` **默认都改成 0**（原来 56 / 64）⇒ 全仓库只剩**一个**寄存器机制，不会互相打架。

**③ 实测寄存器（tier 4608、TPB=128、`ptxas -v`，受限档构建）**

| kernel（实例化） | 按表（≥2560 ⇒ 255） | `FORCE=128` | 块/SM（65536/(128·regs)）|
|---|---|---|---|
| `kernel_double_add<16,4608>`（param3） | **122**，0 spill | 116，0 spill | 4 → 4 |
| `kernel_double_add_suyama<16,4608,false>`（param0） | **129**，0 spill | **128**，48 B spill stores / 40 B loads | **3 → 4** ✓ |
| `kernel_double_add_suyama<16,4608,true>`（param2） | **122**，0 spill | 125，0 spill | 4 → 3 |
| 小档位（128 bit，TPB=128，按表 56） | 44（上限没起作用） | — | — |

**④ A/B 吞吐（本机 4060 Laptop、24 SM、TPB=128、TPI=16、N=M4423 素数、B1=1e5、
两个 exe 交替测量取中位数；A/B 期间机器空闲）**

| 曲线数 | A 波数（容量 576） | B 波数（容量 768） | A：按表 129 寄存器 ⇒ 3 块/SM | B：`FORCE=128` ⇒ 4 块/SM | B/A |
|---|---|---|---|---|---|
| 576 | 1.00 | 0.75 | 2.38 M | **2.42 M** | **+1.7%** |
| 768 | 1.33 | 1.00 | 2.51 M | **2.59 M** | **+3.2%** |
| 1152 | 2.00 | 1.50 | 2.53 M | **2.60 M** | **+2.8%** |
| 1920 | 3.33 | 2.50 | 2.57 M | **2.63 M** | **+2.2%** |

（每次 2–3 次重复的运行间离散 ≤0.2% —— 例如 576 曲线的 B 三次是 34381/34364/34365 ms。）

**④b 直接测用户的生产档位 tier 5120**（5000 位素数 N 落在 5120 档、同卡、TPB=128、param0）：

| 曲线数 | A 波数（576） | B 波数（768） | A：141 寄存器 ⇒ 3 块/SM | B：128 寄存器 + 144 B spill ⇒ 4 块/SM | B/A |
|---|---|---|---|---|---|
| 576 | 1.00 | 0.75 | 1.99 M | 1.99 M | ±0%（A 正好整波，没有尾巴可省）|
| 768 | 1.33 | 1.00 | 2.06 M | **2.11 M** | **+2.5%** ✓ |
| 1920 | 3.33 | 2.50 | 2.11 M | **2.14 M** | **+1.4%** ✓ |

⇒ 两个档位结论一致：**128 寄存器（4 块/SM）在有"半空尾波"的批量上稳定赢 1.4–3.2%**，
只有在 A 恰好整波时打平（576 = A 的 1.00 波）。
**正确性**：两种分配跑同一条曲线，存档里的 stage-1 X **逐字节相同**（X 头 40 位 hex 一致、长度 1250）✓。

**④c 全档位寄存器/溢出实测（suyama=param0 家族、TPB=128、`ptxas -v`）**

| tier | 2560 | 3072 | 3584 | 4096 | **4608** | **5120** | 5632 | 6144 | 6656 | 7168 | 7680 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 不限制寄存器 | 86 | 98 | 109 | 117 | **129** | **141** | 163 | 174 | 178 | 186 | 203 | 211 |
| 块/SM | 5 | 5 | 4 | 4 | **3** | **3** | 3 | 2 | 2 | 2 | 2 | 2 |
| `FORCE=128` 后 | 86 | 96 | 110 | 124 | 128 | 128 | 128 | 128 | 128 | 128 | 128 | 128 |
| spill stores | 0 | 0 | 0 | 0 | **48 B** | **144 B** | 560 B | 1028 B | 1512 B | 1896 B | 2472 B | 3356 B |
| ⇒ 块/SM | 5 | 5 | 4 | 4 | **4** | **4** | 4 | 4 | 4 | 4 | 4 | 4 |

（TPI=32 档位同理：9216→129、10240→139、11264→154、12288→166 寄存器，`FORCE=128` 的 spill 从
16 B 涨到 12288 的 1000 B；≤2048 档位在 56 上限下：512 位 508 B、1024 位 556 B、2048 位 3164 B。）

⇒ **据此定表（已落地）**：`bits ≤ 2048 → 56`；`2560 ≤ bits ≤ 5120 → 128`（2560–4096 那里根本不 binding，
4608/5120 那里有实测收益）；`bits ≥ 5632 → 255`（溢出代价陡增、无实测，不猜）。

**④d 纯占用率对照（用户 4070 Ti（60 SM）@1800MHz 实测，B1=260e6、CGBN<16,5120>、param0；每个配置跑 19–31 小时
到稳定后取 s/curve —— s/curve 就是吞吐的倒数，可用）**

| 构建 | 寄存器 | 块/SM | 曲线数 | 块数 | s/curve | 说明 |
|---|---|---|---|---|---|---|
| 9/24 20:16 | 140 | 1 | 960 | 60（256 thr/块）| 76.75 | 删 `normalize_addition` **之前** |
| 9/25 16:28 | 141 | 2 | 960 | 120 | **72.88** | 最好 |
| 9/25 16:28 | 141 | 3 | 1440 | 180 | 77.39 | 反常点，见下 |
| 9/25 16:28 | 141 | 4 | 1920 | 240 | 74.75 | **1.33 波**（容量 60×3=180）✗ |
| 9/25 21:51（本轮）| 128 | 2 | 960 | 120 | **72.90** | 最好 |
| 9/25 21:51（本轮）| 128 | 4 | 1920 | 240 | 73.41 | **正好 1 波**（容量 60×4=240）✓ |

这张表把"占用率"和"波尾"彻底分开了，也**修正了本节 ④ 的归因**：

* **纯占用率**（同样无尾巴、只改驻留块数）＝ 128 寄存器那两行 120 vs 240 块：
  **72.90 → 73.41，即多一倍驻留 warp 反而慢 0.7%** ⇒ **占用率中性偏负** ✓ 与本轮 ncu
  （Compute SOL ~83%、DRAM 0.5%，issue-bound）一致，也**印证了用户的判断"继续深挖寄存器没有价值"** ✓。
  ④ 表里那 +1.7…+3.2% 里**主要成分是"消掉半空尾波"**，不是"块更多"：证据就是 576 曲线那行
  （A 恰好 1 整波、B 0.75 波）两边打平 ✓。
* **波尾**：同样是 240 块（1920 曲线），141 寄存器（容量 180 ⇒ **1.33 波**）74.75 vs
  128 寄存器（容量 240 ⇒ **1 整波**）73.41 ⇒ **+1.8% 就是尾巴的钱** ✓ 这也是本轮寄存器表的真实价值：
  **让 1920 曲线这批正好压成一整波**，不是让你去追占用率。
* 用户的 77.39（1440 曲线、180 块、正好 1 波）**与两条规律都不符**（既无尾巴、warp 数居中），
  是单次 19–31 小时长跑里最可疑的一行；**结论是"需要同窗口交替重测"**，不能据此说"3 块/SM 最差"。
* 76.75 → 72.88 那 5.3% 与 §4 实测的"删 `normalize_addition` +5.4%/+6.7%"**吻合** ✓
  （但那一行同时换了块大小 256→128 线程，严格说是两个变量的混合）。

⇒ **修正后的结论（本节 ④/⑤ 的总口径）**：**占用率不是杠杆，波对齐才是**；寄存器上限的作用是
"把批量的波数对齐到整数"，值 1–2%，**不要指望靠它或靠继续压寄存器拿吞吐**。
要继续拿吞吐只能去**指令数**那边（§8.9：IMAD 44.9% / IADD3 18.6% / MOV 15.8%）。

**⑤ 对用户生产形状（4070 Ti、60 SM、TPB=128、TPI=16、5120 bit param0）的含义**：容量从
`60×3×8 = 1440` 变成 `60×4×8 = 1920` 曲线 ✓ —— 他们一直用的 **1920 曲线**在旧分配下是
**1.33 波**（ncu 的 `Waves Per SM = 1.33`、占用率只有 20.16% 而不是理论 25%，缺口全在尾巴），
在 128 寄存器下正好是 **1 整波** ✓。本机 4060 的 768 曲线正是同一个"1.33 波 vs 1.00 波"形状，
实测就是上表的 **+2.5%**（tier 5120）/ **+3.2%**（tier 4608）✓。

⚠ 仍未做：**5632 及以上**档位该不该压到 128（那里 spill 560 B…3356 B，需要单独 A/B）。

---

## 6. 本次顺带记录的工程陷阱（都真实踩过）

1. **`.cu` 文件里不要写非 ASCII 注释**：文件是 UTF-8 无 BOM，nvcc 按 ANSI(GBK) 读，行尾的中文全角字符
   会把**换行符吃掉**，让下一行代码被并进注释。现象极具误导性（报"break statement may only be used within
   a loop or switch"、"expected a declaration"）。`tools/bench/cgbn_op_probe.cu` 现在强制 ASCII-only 并在文件头写了警告。
   （同一族问题：含中文的 `.ps1` 必须存成 UTF-8 **with BOM**，见 `tools/README.md`。）
2. **CGBN 的 host 侧需要 GMP**：`gmp.h` 必须**先于** `cgbn.h` 包含，否则 `cgbn_cpu.h` 直接 `#error You must use GMP for now`。
3. **`cgbn_load` 要非 const 指针**：`cgbn_mem_t<BITS>*`（`const` 版本匹配不上，报错会列出候选签名）。
4. **设备端 `switch` + 模板 + CGBN 会让 nvcc 的 device 拆分通道出错**：改成模板参数 + `if constexpr` 风格的
   分支（每个算子一个独立 kernel 实例化）即可绕开，顺带去掉运行时分支。
5. **CGBN 默认构建**（不定义 `ECM_CUDA_FULL_BUILD`）是 dev 构建，只到 **768 bit**（不是 1024）：
   用 1021 bit 测会直接报 "No available CGBN Kernel large enough"。
6. **计时探针必须用素数 N**：用 M511 = 2^511−1 这类**合数**在 4096 曲线下跑到 k≥4 时会**提前中止**——
   某条曲线撞出一个假因子（实测 `factor[0]=55753`），kernel 走退化曲线分支提前退出、`gputime` 根本不打出来。
   §5 的 k 扫描因此改用 511-bit **素数**（`cuda_kernel_ab.ps1 -NExpr <prime>`），数据才完整。
   （换句话说：`N = 2^k−1` 这条"防提前找到因子"的经验规则只对 B1 小、曲线少的情况成立。）
   **同一族的坑在 round 5 又踩了一次**：用 `M1021` 跑 param2 对照时"逐字节不一致" ✗，
   真因是 **1021 不是梅森素数指数**（M1021 是合数）⇒ stage-1 正常命中因子 ⇒ 存档的 X
   按设计写的是**因子**而不是 x 坐标 ⇒ 与 gmp-ecm 的 x 当然不同 ✗。对照实验必须选
   **梅森素数指数**（521/607/1279/2203/3217…）；`tools/test/test_cuda_param2.ps1` 现在会检测
   "X 位数过短 = 命中"并直接报出这个原因。
7. **本机 `powershell.exe` 的执行策略是 Restricted**：`tools/` 下的测试脚本要加 `-ExecutionPolicy Bypass`
   才能跑（`pwsh` 在这台机器上不在 PATH 里）。文档 §5.3/§7 的命令都按这个写法给。
8. **不要用 PowerShell 管道把输入喂给原生命令**（2026-09-25 又踩一次）：PS 5.1 会往 stdin 加 UTF-8 BOM，
   `gmp-ecm` 直接回 `Error - invalid number` —— 于是整轮"生成成本"测量其实在量**报错路径**
   （所有 N 尺寸都是 ~0.12 ms/curve，还随 N 变小，明显不合理）。正确写法：
   `cmd /c "prog.exe < file.txt"`，或者干脆按 §5.6 那样写一个独立工具。
9. **没有实测就用"结构解释"下结论会翻车**：`-DECM_STEP_VARIANT=2`（显式 prep 寄存器、消除写后读相关）
   听起来该更快，实测 **+0.02%**；param0 的寄存器上限听起来该像 param3 一样 +4.7%，实测 **+0.9%**
   （因为它算子更多、溢出更早）。两次都靠 A/B 才没有把"合理化猜想"写进文档。
10. **热循环里"看起来免费"的运行期判定不免费（2026-09-25，代价 34%）**：为了省掉一个 kernel 家族，
    我在 suyama kernel 里加了 `const_diff = (xdiff == 2)` 的 warp-uniform 判定，理论上零成本 ——
    实测 **寄存器 71 → 92**（`xdiff` 被迫跨整个 bit 循环保活），param0 吞吐 **71.5 M → 47.5 M（−34%）**。
    凡是往每 bit 的循环里塞东西，先看 `nvcc -Xptxas -v` 的寄存器数。已回退；param2 改用独立家族。
11. **形状测错一次就要承认并重测**：param2 的 kernel 形状我第一版测成了"32 位 a24"（那是 param3 的
    形状），修正成"满宽 a24 + 常数差值"后数字从 78.37 M → 79.15 M、结论从 +10.4% → +10.7%。两次都
    写进文档，别只留后者 —— 结论相同但**理由**不同，而理由决定实现方式。
12. **GMP 辅助函数的"输出别名输入"会静默出错（2026-09-25，param2 卡了一轮）**：`p2_jac_dbl(X,Y,Z,
    X,Y,Z,N)` 里 `Z3 = 2·Y1·Z1` 写在 `Y3` 之后，而 `Y3` 覆盖了 `Y1` ⇒ Z3 错 ⇒ 点离开曲线 ⇒
    推出的 x3 = 1、A = −2（奇异）。写法：**中间量一律用临时变量，最后统一写回**。
    教训：跨实现对照（这里用独立仿射参考 + gmp-ecm）才暴露了它 —— 自己和自己比不会发现。
13. **`outputf` 不支持 `%Zd`（静默什么都不打印）**：`kernels/cuda/cgbn_stage1.cu` 里的调试输出要用
    `mpz_get_str` + `%s`。症状极具误导性：格式串后面的值"消失"，看起来像计算没跑。
14. **改完带中文注释的 `.cu/.cpp` 之后必须补回 UTF-8 BOM**：本仓库的 `cgbn_stage1.cu` 里有中文注释，
    任何一次"读出来再写回去"的编辑都会丢掉 BOM，nvcc/cl 便按 ANSI(GBK) 读 ⇒ 中文注释的最后一个字节
    吃掉换行 ⇒ **下一行的 `#define CHECKPOINT_VERSION 4` 被并进注释**，报 "identifier CHECKPOINT_VERSION
    is undefined"。体检/修复：`tools/diag/fix_bom.py <file...>`（本轮新加，只对"含非 ASCII 且无 BOM"
    的文件补 BOM）。
15. **缓冲区字索引必须跟着"家族标志"，不能跟着主参数标志（2026-09-25，param2 卡了两轮）**：
    `words_per_curve` 已改成 `suyama_layout ? 7 : 5`，但 `p1_word`/`p2_word` 仍是
    `param0 ? 3 : 1` ⇒ param2 从 **word 1（a24）**而不是 **word 3（aX）**读取结果 ⇒ 曲线与梯形
    完全正确、**输出永远是垃圾** ✗。症状：与参考实现怎么都对不上，而每一步"看起来"都没错。
    教训：**同一份布局的多个消费者要一起改**（`words_per_curve` / `p1_word` / `p2_word` / 检查点校验）。
16. **用脚本改源码时，用错缩进的锚点会静默删掉一大块**：我写的一个"删掉警告块"的补丁按
    `      }\n`（6 空格）找块尾，而真正的块尾是 `  }`（2 空格）⇒ 它一路吞到很远处，删掉了
    `s_num_bits`/事件创建/检查点校验等 ~164 行，报的却是"某个 #define undefined"这类**误导性**错误。
    纪律：脚本改完立刻 `git diff --numstat` 看增删行数是否符合预期（本轮就是靠
    "181 deletions"异常才发现），并优先用 `git show HEAD:<file>` 精确还原区间。
17. **"完美 0.0% 差异"是报警信号：先比对两个二进制的哈希（第 7 轮，差点把假数据写进文档）**。
    我用 `-DECM_REG_TARGET_FORCE=128` 做寄存器 A/B，两轮构建的两个 exe **哈希完全相同**
    （`E9A1DD7A6000BEBD`）⇒ 差异当然是 0.0%。真因：**这个变量当时根本没接到 `flags.make` 上**
    （CMakeLists 里只在一句注释里提到过它）⇒ `cmake -D<无人使用的 cache 变量>` 是**静默空操作** ✗。
    已修（现在会打印 `ecm_cuda: forcing __maxnreg__(128) on every kernel tier`），修后两个 exe
    哈希不同、ptxas 寄存器数也不同 ✓。纪律：**改编译宏的 A/B，第一步先确认产物真的变了**
    （哈希或 `-Xptxas -v` 的寄存器行），再看吞吐。
18. **把 exe 拷出构建目录会静默失效（第 7 轮）**：变体 exe 需要与 `gmp-10.dll` 同目录，
    拷到 `.bench_tmp\obj\` 之后运行**什么都不打印**（没有报错、没有 gputime），看起来像"脚本坏了"。
    做法：变体留在构建目录里（`build_nm16\ecm_u.exe` / `build_nm16\ecm_c128.exe`），
    `cuda_kernel_ab.ps1 -Exe <构建目录里的名字>` 直接用。
19. **含中文的 `.ps1` 没有 BOM + `powershell -File` ⇒ 参数被静默吞掉（第 7 轮，作废了一次 A/B）**：
    `tools/bench/cuda_kernel_ab.ps1` 在 HEAD（`0715172`）里**没有 BOM**（PS 5.1 按 GBK 读，
    中文注释把行尾吃掉 ⇒ `param()` 块被破坏）⇒ `-NExpr 2^4423-1` **静默失效**，脚本回落到默认
    `N = 2^Bits-1` = `2^4608-1`（限制档构建里根本没有这个档位）⇒ 拿到 `gputime=0`。
    可复现指纹：`-Bits 4608` 跑完后 `.bench_tmp\kernel_ab\n_4608.txt` 是 **11 字节 `(2^4608-1)`**
    （正常应为 9 字节 `2^4423-1`）—— **凡是用 `-NExpr` 的计时，先看这个文件的字节数**。
    已用 `tools/diag/fix_bom.py` 修复；审计：`tools/**/*.ps1` 里含非 ASCII 的 4 个脚本现在都有 BOM ✓。
    （顺带说明为什么它有时"看起来"能用：同一个文件在 PS7 会话里被 dot-source 时按 UTF-8 解码，
    参数就正常 —— **调用方式不同、解码不同**，所以这类 bug 会时隐时现。）
20. **`__maxnreg__(0)` 不能用来表达"不限制"**：nvcc 直接报
    `error: The maximum number of registers that can be allocated per thread must be positive`。
    sm_89 的每线程上限是 **255**，所以要显式放开就写 255（见 §5.7）。
21. **PowerShell 变量名大小写不敏感：`$jobs` 就是 `[int]$Jobs`（第 7 轮）**。新的并行编译脚本里
    参数是 `[int]$Jobs`，我随后写 `$jobs = @()` 装任务列表 ⇒ 直接抛
    `Cannot convert the "System.Object[]" value of type "System.Object[]" to type "System.Int32"`，
    脚本 0.5 s 就"成功"退出、什么都没编 ✗。改名 `$tuList` 解决。教训：**给脚本参数起名时，别用
    会在正文里当普通名字复用的词**（PowerShell 不区分大小写，`$jobs`/`$Jobs`/`$JOBS` 同一个变量）。
22. **`Start-Process -PassThru` 的 `.ExitCode` 可能是 `$null`，而 `$null -ne 0` 在 PowerShell 里是 `$true`（第 7 轮）**：
    于是**8 个 TU 全部编译成功、产物齐全，却被全部报告成 `FAIL (exit )`** ✗（第一次看到"8/8 FAIL"但
    exe 明明生成了就是这个原因）。修法：`WaitForExit()` 后仍为 `$null` 时，退回"**obj 是否比启动时刻新**"
    这一判据（这才是真正关心的属性）。教训：脚本里的"成功/失败"判定不要依赖单一可疑 API，
    尤其是**用 `$null` 参与比较**时 —— PowerShell 不会替你报错。
23. **`-sigma i:s` 的前缀是"参数声明"，不能丢（第 7 轮，用户指出）**：原本的
    `parse_sigma64_arg()` 只取冒号**后面**的数字，把 `i` 直接扔了 ⇒
    ① `-sigma 3:12345678` 与 `--gpu-param 0` 同时给会被**静默接受**（gmp-ecm 会报
    `Error, conflict between -sigma and -param arguments`）✗；
    ② 单独给 `-sigma 0:12345678` 也**不会**选 param0（会跑默认的 param3）✗ —— 这比 ① 更危险，
    因为用户以为自己指定了参数化。现在按 gmp-ecm 的语义实现：`i` 是参数声明，冲突即报错并返回 1，
    单独给则**采用** `i`；`i` 不在 {0,2,3} 里则报"不支持的参数化"。
    回归测试：`tools/test/test_cli_args.ps1`（8 项）。**顺带修掉工具里的错误写法**：
    `tools/bench/cuda_kernel_ab.ps1` 原来固定写 `-sigma 3:...` 却同时传 `--gpu-param 0/2`，
    `tools/test/test_cuda_param2.ps1` 也写的是 `--gpu-param 2 -sigma 3:...` ✗ ⇒ 都改成前缀与
    参数一致（`-sigma $GpuParam:12345678` / `-sigma 2:$Sigma`）。

---

## 7. 复现命令

```powershell
# 0) 算子单价 + 变体 A/B（GPU 1 空闲卡；需 zen3 GMP 在 PATH）
$env:PATH = "D:\code\MPA-OpenCl\third_party\gmp-zen3\dist\bin;$env:PATH"
nvcc -std=c++17 -O3 -arch=sm_89 -I cgbn/include -I third_party/gmp-zen3/dist/include -DXMP_WMAD `
     -Xcompiler /wd4819 -o .bench_tmp/cgbn/probe_wmad.exe tools/bench/cgbn_op_probe.cu `
     -L third_party/gmp-zen3/dist/lib -lgmp
.bench_tmp\cgbn\probe_wmad.exe 3 2048 1500 1        # tier3 = TPI=16/BITS=3072

# 1) 整 kernel A/B（§4 的表格）
powershell -NoProfile -ExecutionPolicy Bypass -File tools\bench\cuda_kernel_ab.ps1 -Label pos -Bits 511 -Curves 4096 -B1 1e5 -Device 1 -GpuParam 3

# 2) add-density 扫描（§5 的表格；N 必须是素数，见 §6.6）
$p = (Get-Content .bench_tmp\cuda_p511_prime.txt -Raw).Trim()
foreach ($k in 1,2,3,4,5,6,1000000) {
  cmake -S . -B build_cuda_dev -DECM_PROBE_ADD_DENSITY=$k
  cmake --build build_cuda_dev --config Release --target ecm_cuda
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\bench\cuda_kernel_ab.ps1 -Label "k$k" `
    -Bits 511 -NExpr $p -Curves 4096 -B1 1e5 -Device 1 -GpuParam 3
}
cmake -S . -B build_cuda_dev -DECM_PROBE_ADD_DENSITY=1   # 还原

# 3) 正确性（§4.4）：CPU↔GPU 命中集一致 + 三套回归
powershell -NoProfile -ExecutionPolicy Bypass -File tools\stat\ecm_hitrate.ps1 -Engine mont,gpu -GpuParam 0 -Bits 20 -Count 16 -Curves 8 -Backend simd -CudaExe build_cuda_cmake\ecm_cuda.exe -Device 1
powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_cuda_param0.ps1 -CudaExe build_cuda_cmake\ecm_cuda.exe
powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_cuda_param2.ps1 -CudaExe build_cuda_cmake\ecm_cuda.exe -Bits 1279
powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_cli_args.ps1    -CudaExe build_cuda_cmake\ecm_cuda.exe   # -sigma i:s vs --gpu-param（§6.23）

# 4) 链算子探针（§5.2/§5.4）：M=1..5 与"只倍点"基线 M=1e6
$p = (Get-Content .bench_tmp\cuda_p511_prime.txt -Raw).Trim()
foreach ($m in 1,2,3,4,5,1000000) {
  cmake -S . -B build_cuda_dev -DECM_PROBE_CHAIN_W=$m
  cmake --build build_cuda_dev --config Release --target ecm_cuda
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\bench\cuda_kernel_ab.ps1 -Label "chainM$m" `
    -Bits 511 -NExpr $p -Curves 4096 -B1 1e5 -Device 1 -GpuParam 3
}
cmake -S . -B build_cuda_dev -DECM_PROBE_CHAIN_W=0   # 还原

# 5) CGBN 算子的"值依赖性"检验（§1）：zero / one / 通用 操作数应当同价
nvcc -std=c++17 -O3 -arch=sm_89 -I cgbn/include -I cgbn/include/cgbn -DXMP_WMAD -DPROBE_VALUE_MODE=1 `
     -Xcompiler /wd4819 -o .bench_tmp/cgbn/probe_v1.exe tools/bench/cgbn_op_probe.cu `
     -L third_party/gmp-zen3/dist/lib -lgmp
.bench_tmp\cgbn\probe_v1.exe 0 2048 4000 1      # tier0 = TPI4/512

# 6) param2 经济学（§5.6）：kernel 形状（探针，结果错）+ 主机侧生成成本
cmake -S . -B build_cuda_dev -DECM_PARAM2SHAPE=1
cmake --build build_cuda_dev --config Release --target ecm_cuda
powershell -NoProfile -ExecutionPolicy Bypass -File tools\bench\cuda_kernel_ab.ps1 -Label p2shape `
  -Bits 511 -NExpr $p -Curves 8192 -B1 1e5 -Device 1 -GpuParam 0
cmake -S . -B build_cuda_dev -DECM_PARAM2SHAPE=0            # 还原
tools\build_tool.bat tools\bench\param2_gen_cost.cpp
cmd /c "build_vs18\tools\param2_gen_cost.exe 3000 3 < .bench_tmp\paramgen\n1021.txt"

# 7) 占用率杠杆（§5.5）：TPB / ROT / 寄存器上限；还原成生产默认
cmake -S . -B build_cuda_dev -DECM_TPB=128 -DECM_MAX_ROTATION=1 -DECM_MAXRREG=0 `
      -DECM_MAXRREG_SMALL=0 -DECM_MAXRREG_SUYAMA=0
cmake --build build_cuda_dev --config Release --target ecm_cuda

# 8) 并行编译 kernel TU（§8.8）：把 6 个 TU 的 nvcc 并发跑，再让 nmake 只做 host+链接
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\parallel_nvcc.ps1 -BuildDir build_cuda_cmake
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\parallel_nvcc.ps1 -BuildDir build_nm16 -Only tpi16

# 9) 寄存器预算 A/B（§5.7）：受限档构建 + 两个变体 exe 留在构建目录里（gmp-10.dll 必须同目录！）
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\parallel_nvcc.ps1 -BuildDir build_nm16 -Reconfigure
#   A = 按表（tier 4608 的 suyama 是 129 寄存器 ⇒ 3 块/SM）
Copy-Item build_nm16\ecm_cuda.exe build_nm16\ecm_u.exe -Force
#   B = 全档位强制 128（⇒ 4 块/SM，代价 48 B spill）
cmake -S . -B build_nm16 -DECM_REG_TARGET_FORCE=128
Get-ChildItem build_nm16\CMakeFiles\ecm_cuda.dir\kernels\cuda\*.obj | Remove-Item -Force   # 必须删 obj，见 §6.17
cmake --build build_nm16
Copy-Item build_nm16\ecm_cuda.exe build_nm16\ecm_c128.exe -Force
cmake -S . -B build_nm16 -DECM_REG_TARGET_FORCE=0                                          # 还原
foreach ($c in 576,768,1152,1920) {
  foreach ($v in 'ecm_u.exe','ecm_c128.exe') {
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\bench\cuda_kernel_ab.ps1 -Label $v `
      -Exe "build_nm16\$v" -Bits 4608 -NExpr '2^4423-1' -Curves $c -B1 1e5 -Device 1 -GpuParam 0 -Repeats 3
  }
}
# 注意 -NExpr 一定要带引号；无引号的写法在 PS 5.1 下会被算成表达式（§6.19 同族坑）
```

---

## 8. 下一步与取舍

1. **已完成**：删冗余 `normalize_addition`（§4），收益 +5.4%/+6.7%，全档位无需额外成本，
   全量构建的 `tools/test/test_cuda_param0.ps1` **全过**（当时 18 项，现在是 19 项，见 §4.4）。
2. **已关闭：add-chain / w-NAF**（§5.4）。x-only 下窗口法不合法 + 加法次数下界 1.44/bit > 门槛 1.26/bit
   + PRAC 实测算子数慢 8–16%。**不再投入**（只保留两套探针作为以后复用的计时工具）。
3. **已落地（2026-09-25）：占用率与每批曲线数** —— 全部证据见 **§5.5**，第 7 轮的机制更新见 **§5.7**。
   * 默认值改为 `ECM_TPB=128` + `ECM_MAX_ROTATION=1`（原 256/4）；
   * ~~对 ≤2048 bit 的 kernel 源文件加 `--maxrregcount=56`、对 suyama 源文件加 64~~ ——
     **已被 §5.7 取代**：现在统一用 kernel 属性 `__maxnreg__(cgbn_params_t::REG_TARGET)`
     （`bits<=2048 ⇒ 56`，`≥2560 ⇒ 255` / 可被 `-DECM_REG_TARGET_FORCE` 覆盖）；
     `-DECM_MAXRREG_SMALL` 与 `-DECM_MAXRREG_SUYAMA` **默认都是 0**（关），只作为应急的钝器保留，
     因为"按文件"的上限会误伤同文件里未实测的大档位（§5.7 ①）。
   * 启动时会打印**占用率告警**：block 数填不满设备时提示"把 `-gpucurves` 提到约 N"
     （`kernels/cuda/cgbn_stage1.cu`）。
   * ⚠ **已有 build 目录的 CMake cache 会保留旧值**（`ECM_TPB=256` 等），要生效需显式传
     `-DECM_TPB=128 -DECM_MAX_ROTATION=1`，或删掉 cache 重新 configure。
   * **已排除**：`-DECM_STEP_VARIANT=2`（融合步改写为显式 prep 寄存器）实测 **+0.02%**（wash），
     保留在代码里作为证据，默认仍是 variant 1。
   * **部分完成（第 7 轮）**：≥2560 bit 的寄存器表已在测（`__maxnreg__` 后 tier 4608/TPB=128 有
     A/B 实测，见 §5.7）；TPB 在 ≥2560 bit 档位上的复测仍未做。
4. **暂缓：专用平方**（§3）。上限 12–14%（α_min=0.75，需要跨 lane 交换部分和），而在 CGBN 现有
   分布式布局里只对称化对角线块只值 **~1–3%**（TPI=4/8 时 50%/TPI 的乘积部），要动
   `core_mont_wmad.cu` —— 相对上面已经拿到的 ~10% 不值得。**除非**以后要重写 CGBN 的乘法核。
5. **param2：已实现、已与 gmp-ecm 逐字节对齐**（§5.6）。相对 param0 实测快 5.7%（B1=1e5）～约 11%
   （大 B1）。**已完成**：独立 kernel 家族 + `set_p_2p_param2()` + `--gpu-param 2` 路由 +
   `PARAM=2` 存档 + 回归测试 `tools/test/test_cuda_param2.ps1`（7/7）。
   **待查（开放项）**：真实路径只拿到形状探针上界（+10.7%）的一半多 —— 探针跑 suyama 家族 +
   强制 `const_diff`，真实路径跑 param2 家族，~5 个百分点差异原因未定；
   另：per-source-file 的 `--maxrregcount` 会作用于该文件**所有**档位（含未实测的 ≥2560 bit），
   ≥2560 bit 的占用率需要单独复测 —— **第 7 轮已修**：改用 per-kernel 的 `__maxnreg__`，两个
   `ECM_MAXRREG_*` 开关默认归零（§5.7）。
   **产品决策**仍在：param2 存档 Prime95 不能吃、gmp-ecm 能吃。
6. **进行中（第 7 轮）**：在 ≥2560 bit 档位（`tpi16`/`tpi32`）复测寄存器预算与 TPB —— tier 4608 +
   TPB=128 已实测（§5.7，+2–3%）；其余档位的寄存器/spill 表在测，TPB 复测仍未做。
7. **大位宽的另一条路**：CUDA 建议 <12288 bit，更大交给 FFT/NTT 实现 —— 若要推大位宽，
   Karatsuba（把乘积部从 n² 降到 n^1.585，乘与平方一起受益）比"只优化平方"覆盖面更大，但工作量也更大。
8. **编译时间：把 GPU 内核编译并行化（2026-09-25 第 6 轮，进行中）**
   * **根因**：本仓库的构建目录用的是 **NMake Makefiles 生成器 ⇒ 串行编译** ✗（`build_cuda_cmake`
     全量 ≈ 20 min；`build_cuda_dev` 也要数分钟）。
   * **三条并行路线在本机都失败**（都实测过，失败方式各不相同）：
     | 方案 | 结果 |
     |---|---|
     | Ninja（vcpkg 自带 1.13.2） | configure **卡死**在 "Detecting C compiler ABI info" ✗（10 min 无进展）|
     | MSBuild（VS 生成器，`build_vs18`，`-- /m:24`） | CUDA 13.3 target **取消构建**（MSB5021 终止 cmd），未产出 obj ✗ |
     | jom 1.1.7（并行 NMake） | configure 失败："parallel job execution disabled for Makefile" + `try_compile` 失败 ✗ |
   * **并行方案全部失败，根因已定位（第 7 轮）**：三条路线失败的**共同点**是它们都用
     `cmd`/子进程包装 nvcc，而**本环境会终止这类被包装的子进程** ——
     MSBuild 的 CUDA 13.3.targets 直接报 `MSB5021: 正在终止"cmd"及其子进程，以便取消生成` ✓，
     Ninja 在 configure 阶段卡死、jom 的 `try_compile` 失败也符合同一模式 ✓；
     而**直接调用 nvcc 的路径（NMake、以及我用 `cmd /c <bat>` 逐个编译）都正常** ✓✓。
     ⇒ **可用的并行化做法**（第 7 轮已实现：`tools/build/parallel_nvcc.ps1`）：写一个"并行编译脚本"——把 6 个 kernel TU 的 nvcc
     命令行**从 PowerShell 直接并发启动**，再调 `cmake --build` 只做链接 ✓。

      ```powershell
      # 并行编译 kernel TU 并链接（构建目录需已配置过；缺 compile_commands.json 时加 -Reconfigure）
      powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\parallel_nvcc.ps1 -BuildDir build_cuda_cmake
      # 只重编某一个 kernel 家族（改完一个 TU 时最省时间）
      powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\parallel_nvcc.ps1 -BuildDir build_nm16 -Only tpi16
      ```

      每个 TU 的日志写在 `<BuildDir>\par_nvcc\`；脚本打印每个 TU 的墙钟时间、串行时间之和与加速比，
      然后才调 `cmake --build`（此时只剩 host TU 与链接）。两个前提：① 必须有
      `compile_commands.json`（脚本会按需 configure 并加 `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`）；
      ② 脚本**默认总是重编所有被选中的 TU**（不做依赖分析 —— 这正是它可靠的代价），
      省时间要靠 `-Only <正则>`；`-SkipUpToDate` 用时间戳跳过最新的 TU，但改过 CMake 选项时**不要**用。

      **实测（第 7 轮，全部从零重编，本机 = 24 SM 4060 Laptop 的笔记本、**24 逻辑核**）**：

      | 配置 | TU 数 | 串行时间之和 | 并行墙钟 | 加速比 |
      |---|---|---|---|---|
      | `-DECM_NO_PARAM2=1`（param2 编成空桩） | 8 | 1305 s（21.8 min）| **636 s（10.6 min）** | 2.05× |
      | 全量、两个大 TU 未拆 | 8 | **2166.6 s（36.1 min）** | **745.1 s（12.4 min）** | 2.91× |
      | **全量、按 TPI 拆分后（本轮）** | **12** | **2376.6 s（39.6 min）** | **395.4 s（6.6 min）** | **6.01×** ✓ |
      | 全量 + host TU + 链接（拆分后） | 12 | — | 499.1 s（8.3 min）| 串行约 42 min |

      拆分后的单 TU 墙钟：`suyama_tpi16` **395 s**（关键路径）、`param2_tpi16` 368 s、`tpi16` 333 s、
      `suyama_tpi32` 329 s、`param2_tpi32` 311 s、`tpi32` 269 s、`suyama` 119 s、`param2` 114 s、
      `tpi8` 76 s、`tpi4` 36 s、`cgbn_stage1` 14 s、`ecm_cuda_backend` 12 s。
      Σ/6 = 396 s ≈ 395 s 墙钟 ⇒ **`-Jobs 6` 的墙钟正好等于最慢 TU**。再往上加 job **没有用**：

      | `-Jobs` | 墙钟 | 最慢 TU | 说明 |
      |---|---|---|---|
      | 6 | 395.4 s | `suyama_tpi16` 395.4 s | 12 个 TU 分两批跑 |
      | 12 | **394.6 s** | `suyama_tpi16` 394.6 s | 12 个并发、每个 TU 用时几乎不变（≤3%）|

      ⇒ 本机 24 逻辑核，实测**平均只有约 6 个核在忙**（大 TU 跑满全程、小 TU 早早结束），
      所以**瓶颈不是 CPU 也不是 job 数，而是"最慢的单个 TU 的串行编译时间"**。
      唯一的下一步就是**继续拆最长的 TU**（按位宽拆，见下）。

      **下一步怎么拆（成本分析）**：`*_tpi16.cu` 里 12 个档位的编译成本大致 ∝ 位宽²
      （每线程 limb 数 × 展开后的 wmad 数量），所以 **8192 一个档位就占整个文件约 17%**。
      按位宽把 `*_tpi16.cu` 拆成 `2560..6656` / `7168..8192` 两个文件（成本约 46%/46%，剩下的
      小档位是 8%），每个文件 ≈ 182 s；`*_tpi32.cu`（9216…16384）同理在 `13312` 处拆成 48%/52%，
      每个 ≈ 165 s ⇒ **墙钟可望从 395 s 降到约 200 s**（全量 CUDA 构建 ~3.5 min，对串行 40 min 是 ~12×）。
      代价：每个 TPI 家族从 1 个文件变成 2 个，且**顶级 dispatch 要串起来**
      （`cgbn_stage1_kernel_<fam>_tpi16_lo` / `_hi`，`cgbn_stage1.cu` 里两个 dispatcher 各加一次跳转）。

      ⇒ 四个必须记住的结论：
     ① **并行的上限由最慢的单个 TU 决定**：`cgbn_stage1_kernels_suyama_tpi16.cu` 一个文件就编了 **395 s**
      （2560…8192 共 12 个档位）。脚本会打印 `critical path = ...`；**"拆 TU"是这里唯一有效的下一步**
      （本轮把 `suyama`/`param2` 按 TPI 拆成 3 个文件，墙钟从 745 s → 395 s ✓）。
     ② **param2 是第二份全档位拷贝**：它自己的 TU 要 **711 s**（拆分后 368+311+114 s），关掉它能把串行时间
      从 2377 s 砍到 1305 s（**−45%**）。于是新增 `-DECM_NO_PARAM2=1`：param2 的四个查找函数编成返回
      `nullptr` 的空桩，只做 param0/param3 实验时用它；此时 `--gpu-param 2` 会**明确报错**
      （"param2 kernels are NOT compiled into this binary"）而不是回退到别的参数化 ✓。
     ③ **和"串行 20 min"的老印象相比，现在全量串行是 40 min** —— 因为档位（TPI=16/32 的 512 间隔）
      和 param2 家族都是后来加的。**并行是现在唯一实用的全量构建方式** ✓。
     ④ **观察到 4–6 个 nvcc 工具链同时跑、每个进程内部 `cicc` 与 `ptxas` 交替、`ptxas` 占大头**，
      这是**正常现象**：CGBN 的模乘被完全展开，PTX 优化与寄存器分配（ptxas）本来就是这里的主要成本；
      编译**大档位**（≥5120 bit）时 ptxas 的时间占比还会更高。
   * **另有两处必须记下的坑**：① VS 生成器目录里 `--target ecm_cuda` 会**什么都没编就"成功"**
     （93.6 s / 0 个 TU / 无 exe）✗，别把它当成构建成功；② 增量 VS 目录里出现过
     `LNK1181: 无法打开输入文件 ecm_cuda.dir\Release\cgbn_stage1.obj`（host TU 没被编）✗ ——
     两者都只在 VS 目录出现，**受限/全量都用 NMake 目录**最稳。
9. **SASS/PTX 统计（第 6 轮新增工具 `tools/bench/sass_stats.ps1`）**：`cuobjdump -sass` 导出后按
   函数切分并统计 opcode 直方图与 spill 交通。样例（param3 家族，TPI=16、8192 bit、当前未加寄存器上限）：

   | 指标 | 值 |
   |---|---|
   | SASS 指令数 | 61464 |
   | spill | LDL=4 / STL=5（基本没有溢出 ✓）|
   | 前几项 opcode | IMAD 44.9%、IADD3 18.6%、**MOV 15.8%**、SEL 3.8%、CALL 3.7%、SHFL 3.2%、LOP3 3.2% |

   ⇒ 大档位的分配本身是健康的（几乎无 spill），但 **MOV 占 15.8%** 说明寄存器搬运很重 ——
   这是下一轮调 TPI=16 的顺序/上限时最值得盯的指标（目标：把 MOV 换成有用的算术）。
10. **每批曲线数：要按"填满每个 SM"来算，不是按"一波"来算（第 6 轮实测）**：
    4060 Laptop 有 24 SM，`曲线数 = SM × (TPB/TPI)` 只给出**一波**（TPB=128/TPI=4 时 = 768），
    但每个 SM 能同时驻留多块（由占用率决定）。实测 **1536 曲线（48 块 = 2 块/SM）时
    param0 只有 63.5 M、param3 69.5 M**，而 **8192 曲线（256 块）时是 71 / 82 M** ⇒
    真正的推荐值是 **`SM × 驻留块数/SM × (TPB/TPI)`**（= `kernels/cuda/cgbn_stage1.cu` 启动时
    打印的那个建议值，例：TPB=128/TPI=8 时 3456），以及它的整数倍；`SM × (TPB/TPI)` 只是下界。

    ⚠ **第 7 轮修正（重要）**：上面这条是**小档位（≤2048 bit、算子少、延迟敏感）**的结论。
    在**大档位（TPI=16、5120 bit、B1 很大）**上它**不成立**：用户 60-SM 的 4070 Ti 在 B1=260e6 下
    实测 **960 曲线（2 块/SM）72.90 s/curve 最优，240 块（4 块/SM）73.41**（详见 §5.7 ④d），
    即"填满块槽位"**不再带来吞吐**，多驻留的 warp 甚至略亏（issue-bound）。因此启动时的
    占用率提示已经改成：**只有 `块数 < SM 数`（有 SM 完全没活干）才按 `OUTPUT_NORMAL` 警告**；
    "只填了 N% 的块槽位"降级为 `OUTPUT_VERBOSE` 的**提示**，并且明确说"这不是吞吐缺口，
    真正有用的是让批量成为整波（避免半空的最后一波）"。**不要再按"填满 SM"去挑大档位的曲线数。**
11. **大档位（TPI=16/32）的占用率由寄存器决定，且必须用 per-kernel 属性（`__maxnreg__`）而不是全局
    `--maxrregcount`（2026-09-25 第 6–7 轮，ncu 实测驱动；机制评估见 §5.7）**。用户对 4070 Ti（60 SM）生产运行的 ncu 剖析：
    kernel `CGBN<16, 5120>`、block 256（16 实例/块）、grid **60 = 1 块/SM**、
    **寄存器 140/线程** ⇒ `65536/(256×140) = 1.8` ⇒ **被寄存器限制在 1 块/SM = 8 warps = 16.67% 占用率**，
    而 Compute SOL 已达 **80%**、Memory 22%、DRAM 0.6%（纯 compute/issue-bound）；
    1920 曲线那次因此是**串行两波**（13.09 → 26.18 ms，线性 ✓），而不是两块同时在驻。

    本机复测（suyama 家族，TPI=16/5120 bit，TPB=256，ptxas -v）：

    | 寄存器上限 | 寄存器 | spill stores | 块/SM | warps/SM |
    |---|---|---|---|---|
    | 无（编译器自选） | **141** | 0 | **1** | 8 |
    | **128** | 128 | **144 B**（轻微）| **2** | **16** |
    | 96 | 96 | 1412 B（严重）✗ | 2 | 16 |

    ⇒ **128 是 2 块/SM 的最省做法**（只花 144 B spill）；96 换不到更多块、纯粹多花钱 ✗。
    ⇒ 机制上先改用 **tier 感知的 `__launch_bounds__(params::TPB, params::MIN_BLOCKS)`**（`cgbn_params_t`），
    **第 7 轮又换成更合适的 `__maxnreg__(params::REG_TARGET)`**（`__launch_bounds__` 已删除，
    完整评估见 **§5.7**）：
    `bits<=2048` ⇒ 56 寄存器，`bits>=2560` ⇒ 255（不限制）/ 可用 `-DECM_REG_TARGET_FORCE` 覆盖。
    **踩过的坑**（`__launch_bounds__` 时代）：一开始写成固定的"4 块"，ptxas 把 56 寄存器放宽到 63、
    param3 掉 4%（78.84 vs 82.01 M）✗ —— 占用率目标必须用*寄存器预算*表达，不能用块数硬编码；
    而且 `MIN_BLOCKS` 的形式**天生和 TPB 耦合**（`bits>=2560 ⇒ 2 块` 在 TPB=128 时等价于"允许 256
    寄存器"= 空约束 ✗），这正是换掉它的原因 ✓。

    加寄存器预算之后 5120 档（param0、4060、TPB=256、2 块/SM）实测：
    **384 曲线 20.89 s / 768 曲线 41.71 s / 1536 曲线 83.25 s，三者都是 2.65–2.66 M curve-bits/s**
    —— 即 **2 块/SM 之后按波数线性扩展** ✓（对比早期 1 块/SM 的 2.25 / 1.93 / 2.16 M 那种非单调）。
    ⚠ 当时那组对照**不是同一次 A/B**（`-DECM_MIN_BLOCKS_FORCE=1` 那次的 VS 目录增量构建遇到
    `LNK1181`，host TU 没被编 ✗，见 §6.18 同族问题）；**第 7 轮用可验证的 `__maxnreg__` 重做了**：
    tier 4608/TPB=128 上 3 块/SM → 4 块/SM 稳定 **+1.7…+3.2%** ✓（§5.7 ④ 的表）。

    **TPB=128 的 ncu（用户第二次剖析，1920 曲线 / 5120 bit）**：grid 240、block **128**、
    寄存器 **141** ⇒ `Block Limit Registers = 3` ⇒ 理论 **12 warps/SM = 25%**，
    但**实测只有 9.68 warps（20.16%）**，而 `Waves Per SM = 1.33`（240 块 vs 容量 60×3=180）✅
    ⇒ **这 20% 的占用率缺口完全来自"最后一波只有 1 块/SM"的尾巴** ✓；Compute SOL 反而升到 **83.17%**
    （TPB=256 时是 80.23% ✓ 与 §5.5 的 TPB 结论一致），DRAM 0.5%。

    **由此修掉我自己规则里的一个 TPB 依赖 bug**：`MIN_BLOCKS` 原来写成"`bits>=2560` ⇒ 2 块"，
    只有 TPB=256 时等价于"≤128 寄存器" ✗；TPB=128 时它给出 2 块（= 允许 256 寄存器）⇒ 毫无约束 ✗。
    现在**两个档位都改成寄存器预算**：`65536 / (TPB × (bits<=2048 ? 56 : 128))`
    ⇒ TPB=128 时小/大档位分别是 **9 / 4**，TPB=256 时是 **4 / 2** ✓✓（大档位实测 128 寄存器 ✓）。

    **波尾敏感性扫描（TPI=16、tier 4608、N=M4423 素数、TPB=128、4060 的 24 SM、
    容量 = 24×4×8 = 768 曲线/波）**：

    | 曲线数 | 波数 | curve-bits/s |
    |---|---|---|
    | 384 | 0.5 | 3.14 M |
    | 576 | **0.75** | **2.95 M** ✗（唯一明显掉点）|
    | 768 | 1.0 | 3.10 M |
    | 1152 | 1.5 | 3.10 M |
    | 1536 | 2.0 | 3.10 M |
    | 1920 | 2.5 | 3.11 M |
    | 3072 | 4.0 | 3.10 M |

    ⇒ **升到 4 块/SM（16 warps）之后，波尾基本不再影响吞吐**（≥1 波时平坦在 3.10 M ✓，
    只有 0.75 波那种半空波掉 5% ✗）—— 也就是说：**用户那个 20% 的占用率缺口本质上是
    "3 块/SM + 1.33 波"的组合，把寄存器压到 ≤128（⇒ 4 块/SM）就能把它消掉** ✓。
    对 4070 Ti（60 SM、TPB=128、TPI=16）：容量从 60×3×8=**1440** 变成 60×4×8=**1920** ✓
    —— **正好是用户一直在用的 1920 曲线**：在新分配下它从"1.33 波"变成**整整 1 波** ✓。

    **`-DECM_TIERS` 的一个坑**：tier 是按 N 的位长向上取档的，`N = 2^5120−1`（5120 位）
    并不落在 5120 档 ⇒ 受限构建会报 "No available CGBN Kernel large enough" ✗。
    做受限实验要挑一个**落在该档内、且是素数**的 N（例：M4423 落在 4608 档 ✓）。
