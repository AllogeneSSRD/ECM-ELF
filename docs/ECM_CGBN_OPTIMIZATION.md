# CGBN 优化：实测、结论与探针（2026-09-25）

> 主题：我们 CUDA stage-1 kernel 所依赖的 **CGBN** 还有多少可挖。
> 起因：CGBN 是 Volta（sm_70）时代的库、之后基本没更新，而我们的主力卡是 Ada（sm_89）。
> 姊妹文档：`docs/ECM_Montgomery_STAGE1.md`（param0/param3 kernel 的来龙去脉、§19–§21）、
> `docs/OPENCL_IMPLEMENTATION.md`（与 OpenCL 路径的差异）。

本文件记录 4 条结论 + 已落地的两项改动 + 一项待决策的研究型改动。**所有数字都是本机实测**，
测量装置见 §1、复现命令见 §7。

---

## 0. 结论速览

| # | 结论 | 收益 | 状态 |
|---|---|---|---|
| 1 | CGBN 按架构选的乘法变体（sm_70+ → **WMAD**）在 Ada 上就是最优的，强切 XMAD/IMAD 反而慢 1.5–2.3× | — （**关闭**该方向） | 已测，不再投入 |
| 2 | `mont_sqr` 就是 `mont_mul(a,a)`（α = 1.0），专用平方的**理论上限 12–14%**，且 CGBN 的分布式布局要先解决跨 lane 部分和交换 | ≤ +14% | 可做，优先级低 |
| 3 | 我们每 bit 有 **8 次（param3）/ 6 次（param0）冗余的 `normalize_addition`** —— `mont_mul`/`mont_sqr` 返回时已经 < n；删掉后实测 **param3 +5.4%、param0 +6.7%** | **+5.4% / +6.7%** | **已实现**（本文件 §4） |
| 4 | add-chain（PRAC / NAF + 字典）的**时间天花板已实测到下限**：每 2 bit 一次加法 **1.32×**、3 bit **1.47×**、4 bit **1.56×**、5 bit **1.62×**、6 bit **1.66×**，**加法全关（只倍点）1.91×**；模型 `cost(k) = 4.24 + 3.86/k`（乘当量/bit）对全部实测点吻合到 **0.7%** 以内 | 现实预期 **1.4–1.6×** | 探针已落地（§5），实现待定 |

**优先级建议**：3（已做，免费）→ 4（研究型，收益最大）→ 2（统一小收益，代价大）。

---

## 1. 测量装置与纪律

| 工具 | 用途 |
|---|---|
| `tools/bench/cgbn_op_probe.cu` | 逐 **CGBN 算子**单价（`mont_mul`/`mont_sqr`/compare+cond-sub/add/sub/shift），可切 TPI/BITS 档位，可用 `-DXMP_WMAD/-DXMP_XMAD/-DXMP_IMAD` 切乘法链变体 |
| `tools/bench/cuda_kernel_ab.ps1` | **整 kernel** A/B 计时：固定 N/B1/曲线数，重复取中位数，输出 `gputime` 与 curve-bits/s |
| `-DECM_PROBE_ADD_DENSITY=k`（§5） | 让融合 double-and-add 只在每 k bit 执行一次加法（k>1 **结果错误**，纯计时探针） |

**纪律**（本仓库踩过的坑，见 `docs/ECM_Montgomery_STAGE1.md` §19.4）：

* 计时必须在**空闲** GPU 上做。本次 4070 Ti（GPU 0）被其它任务占用 99%，所有数据都取在
  **RTX 4060 Laptop（GPU 1，空闲）**；比值可跨 Ada 部件迁移，绝对 ns 是 4060 的。
* 固定用 `N = 2^k−1`（k 取素数指数 ⇒ 不会命中因子 ⇒ 每次都跑满全部 bit），避免"提前找到因子"污染计时。
* 每次运行新建/清空临时目录，避免 ckpt/save 残留。
* 同一配置重复 3 次，实测抖动 < 0.1%（例：param3 M511 三次 10880.7 / 10873.9 / 10873.6 ms）。

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
* 全量构建后跑 `tools/test/test_cuda_param0.ps1`（18 项：CPU↔GPU 逐曲线一致、64 位 σ、gmp-ecm 互通、
  硬杀+恢复 4096/4096、param3 回归）：**2026-09-25 在删掉冗余 `normalize_addition` 的全量二进制上
  18/18 全 PASS（ALL OK）**。

---

## 5. 结论 4：add-chain（PRAC / NAF + 字典）——天花板实测 1.47×（每 3 bit 一次加法）

### 5.1 为什么这条最有希望

现在的 kernel 每个 bit 都做**一次融合的 double+add**（4M+4S ≈ 8.1 乘当量）。把两部分拆开看：

| 部分 | 算子 | 乘当量（理论） | 实测拟合 |
|---|---|---|---|
| 倍点（AA, BB, q=AA·BB, u=K(BB+dK)，param3 的 dK 是 32-bit 廉价乘） | 2S + 2M + 廉价 | ≈ 4.1 | **4.24** |
| 加法（CB=t·u, DA=v·w, w=(DA+CB)², v=2(DA−CB)²） | 2M + 2S | ≈ 4.0 | **3.86** |

⇒ 用两参数拟合（只用 k=1..6 的实测值最小二乘）得到 **`cost(k) = 4.24 + 3.86/k`**（乘当量 / bit），
k = 每多少 bit 做一次加法。理论拆分（4.1 + 4.0）与拟合（4.24 + 3.86）一致到 2% —— 模型不是凑出来的：

| k | 模型 cost | 实测 gputime（M511，4096 曲线，B1=1e5） | 提速（对 k=1） |
|---|---|---|---|
| 1（现状） | 8.10 | **10320.95 ms** | 1.00 |
| 2 | 6.17 | **7805.06 ms** | **1.32×** |
| 3 | 5.53 | **7024.34 ms** | **1.47×** |
| 4 | 5.21 | **6614.89 ms** | **1.56×** |
| 5 | 5.01 | **6380.67 ms** | **1.62×** |
| 6 | 4.88 | **6217.62 ms** | **1.66×** |
| ∞（只倍点，把加法完全关掉） | 4.24 | **5403.90 ms** | **1.91×** |

模型对**全部** k=1..6 的预测误差 ≤ 0.7%（例：k=3 模型 1.465× vs 实测 1.469×），而且**拟合出的渐近系数 4.24
与单独实跑的"只倍点"（k=10⁶，加法全关）5403.90 ms 完全对上**（8.10/4.24 = 8.10/4.241 = 1.910）⇒ 这条曲线可信。

⇒ 注意 k≥4 之后的**边际收益快速衰减**（5→6 只多 2.5%），而字典点数、寄存器压力随 k 增长，
所以现实里没有理由把 k 推到 6：**k=3～5 是甜区**。

### 5.2 现实可拿多少

NAF 的非零密度：w=2 → 1/3、w=4 → 1/5、w=5 → 1/6；PRAC 量级在 1/3 附近。对应 **1.47×（k=3）～1.62×（k=5）**，
再往上（k=6）只多 2.5%。所以现实目标是 **1.4–1.6×**（探针是"零成本地跳掉加法"，真实链要付字典与索引的代价）。

**关键有利条件**：`s = lcm(1..B1)` 对**所有曲线完全相同** ⇒ NAF 位型/warp 调度**天然 uniform、零 divergence**，
而且不需要为每条曲线重算位型。这是这条路线在我们这个场景下特别划算的原因。

**代价（真实实现要付的）**：

1. **字典 / PRAC 状态**：w-NAF 需要每曲线字典（w=5 → 8 个点）；4096-bit、8192 曲线量级的内存要单独核算，
   TPI=16 时字典点也存在寄存器/共享内存里，可能反过来压占用率。
2. **失去融合**：探针省掉的是"加法那一半"，真实链的每一步仍要按位型分支；分支本身在 SIMT 下无代价（位型统一），
   但需要额外的索引与规范化序列。
3. **探针的 caveat（已核查，结论有利）**：跳过加法会改变活跃变量数、进而改变寄存器分配。实测
   （`nvcc -Xptxas -v` 单编 `cgbn_stage1_kernels_tpi4.cu`，512 档）：**k=1 → 72 寄存器/线程，k=3 → 86 寄存器/线程**——
   跳过加法**反而让寄存器变多**（分支使更多值在两条路径上都活跃）。占用率因此没有变好、只会变差，
   所以**实测的 1.32×/1.47× 不含"占用率提高"的水分**，是保守值。

### 5.3 探针怎么用

```powershell
# 只测时间：k>1 的结果是错的（文档 §5 顶部与 kernel 头部都有警告）
cmake -S . -B build_cuda_dev -DECM_PROBE_ADD_DENSITY=3
cmake --build build_cuda_dev --config Release --target ecm_cuda
powershell -NoProfile -ExecutionPolicy Bypass -File tools\bench\cuda_kernel_ab.ps1 -Label k3 `
  -Bits 511 -NExpr (Get-Content .bench_tmp\cuda_p511_prime.txt -Raw).Trim() -Curves 4096 -B1 1e5 -Device 1
# 测完务必改回 -DECM_PROBE_ADD_DENSITY=1，否则结果全错（只有计时有意义）
```

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
7. **本机 `powershell.exe` 的执行策略是 Restricted**：`tools/` 下的测试脚本要加 `-ExecutionPolicy Bypass`
   才能跑（`pwsh` 在这台机器上不在 PATH 里）。文档 §5.3/§7 的命令都按这个写法给。

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

# 3) 正确性（§4.4）
powershell -NoProfile -ExecutionPolicy Bypass -File tools\stat\ecm_hitrate.ps1 -Engine mont,gpu -GpuParam 0 -Bits 20 -Count 16 -Curves 8 -Backend simd -CudaExe build_cuda_dev\ecm_cuda.exe -Device 1
powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_cuda_param0.ps1
```

---

## 8. 下一步与取舍

1. **已完成**：删冗余 `normalize_addition`（§4），收益 +5.4%/+6.7%，全档位无需额外成本，
   全量构建的 `tools/test/test_cuda_param0.ps1` **18/18 全过**。
2. **待决策：add-chain（PRAC/NAF）原型**。探针已经把天花板测清楚了（k=3 → 1.47×、k=5 → 1.62×、只倍点 1.91×），
   寄存器核查也做了（跳过加法反而多 14 个寄存器 ⇒ 实测值不含占用率水分）。接下来实现 w-NAF + 字典的
   x-only 差分链（**先估算每曲线字典内存**，w=5 是 8 个点），**目标 1.4–1.6×**（k=3～5 是甜区，不要推到 6）。
3. **暂缓：专用平方**（§3）。上限 12–14%，且要跨 lane 交换部分和；在 3、4 都做完之前不值得开工。
4. **未探索但低风险**：TPB / 每 block 实例数（本次探针用 TPB=128，实际 kernel 用更大 TPB）、
   `MAX_ROTATION`、以及 CGBN 的 `cgbn_swap`/`mont2bn` 在收尾阶段的少量开销。
5. **大位宽的另一条路**：CUDA 建议 <12288 bit，更大交给 FFT/NTT 实现 —— 若要推大位宽，
   Karatsuba（把乘积部从 n² 降到 n^1.585，乘与平方一起受益）比"只优化平方"覆盖面更大，但工作量也更大。
