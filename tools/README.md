# tools/ 目录索引

本目录按**用途**分子目录。历史上所有脚本都堆在 `tools/` 根下，现按下面分类整理；
根目录不再放散落脚本。

```
tools/
├── gen/        内核/参数代码生成器（Python，单源 → .cl/.py 产物）
├── refactor/   一次性迁移/重构/校验脚本（Python，历史工程脚本）
├── bench/      基准与 A/B 脚本（.cpp/.c/.ps1）
├── test/       单元测试 + 集成测试 + 夹具
├── stat/       ECM 命中率统计与后端配对比较（Python/.ps1，用 ecm_prob 的素数集）
├── diag/       诊断/canary/崩溃复现（.cpp/.cmd/.py，定位具体缺陷用）
├── disasm/     反汇编 / ISA 检查（含 Windows 工具链安装）
├── ecm_prob/   ECM 参数化概率分析套件（Python，自带 README）
├── ecm_report/ 进度数据库/图表（Python + bat，自带 README）
└── log_parser/ 日志解析（自带 README）
```

> **小工具编译**：`tools\build_tool.bat <tool.cpp> [额外 .cpp ...]`，产物落到
> `build_vs18\tools\<name>.exe`（源码树保持干净）；只给名字时会在 `tools\diag`、
> `tools\bench`、`tools\test` 里找。每个脚本的 scratch 写在自己目录下的 `_run\`。
>
> **不要用 PowerShell 的行过滤重写 UTF-8 源码**：本工作区的 PowerShell 会按 ANSI 读
> UTF-8、再以 UTF-8 写回，整份文件的中文注释会二次编码损坏（本轮真发生过一次，
> 只能 `git checkout` 重做）。按行改源码请用 Python + 显式 `encoding="utf-8"`
> （参考 `tools/diag/enc_diag.py` 的做法；它也能直接体检某个文件）。
>
> **`.ps1` 怎么跑**：本机 `powershell.exe` 执行策略是 Restricted，且 `pwsh` 不在 PATH，
> 所以用 `powershell -NoProfile -ExecutionPolicy Bypass -File tools\...\x.ps1`。含中文的
> `.ps1` 必须存成 UTF-8 **with BOM**；纯 ASCII 的脚本无需 BOM（如 `test_cuda_param0.ps1`）。

## test/ — 单元测试与集成测试

| 文件 | 内容 |
|---|---|
| `p95_worktodo_test.cpp` | Prime95 worktodo/prime.txt 解析与写回（35 项断言） |
| `ecm_worktodo_test.cpp` | `ECM=`/`ECM2=`/`ECMSTAGE2=` 行解析与 N 计算 |
| `ecm_edwards_save_test.cpp` | Prime95 ECM_VERSION=6 存档读写；与真实 `e0000347` 字节级比对 |
| `ecm_edwards_checkpoint_test.cpp` | 分块标量乘的中止/恢复等价性（Qx/Qz 一致） |
| `gen_ckpt.cpp` | 生成一个中途 STAGE1 存档，用于验证驱动恢复 |
| `test_feeder.ps1` | `ecm_p95feeder` 端到端集成测试（沙箱 p95 目录，7 个周期） |
| `test_agreement.ps1` | **三后端互认**：同一 `(N,B1,sigma)` 跑 标量 / SIMD-Montgomery / SIMD-折叠域，断言命中集合、因子值、以及每个 `.tmp` **逐字节相同**。含历史失败用例（M3001 σ=20260922 B1=1e5），是 §15.9/§15.10 两个 bug 的回归 |
| `test_invariants.ps1` | §7 因子不变式回归：M677→1943118631、M991→8218291649、M4003→16756559，两域各一遍 |
| `fixtures/` | 测试派生文件（`_test_*.save`，由 `ecm_edwards_save_test` 写出） |

手工编译示例（GPE/zen3 前缀按需替换）：

```powershell
cl /nologo /O2 /EHsc /utf-8 /I third_party/gmp-zen3/dist/include /I src/cpu /I src/core ^
   tools/test/ecm_edwards_save_test.cpp src/cpu/ecm_edwards_cpu.cpp src/cpu/ecm_edwards_save.cpp ^
   /Fe:tools/test/ecm_edwards_save_test.exe /link third_party/gmp-zen3/dist/lib/gmp.lib
```

```powershell
# 纯 GMP 无关的单测
cl /nologo /O2 /EHsc /utf-8 /I src/core ^
   tools/test/p95_worktodo_test.cpp src/core/p95_worktodo.cpp /Fe:tools/test/p95_worktodo_test.exe
```

## bench/ — 基准与 A/B

| 文件 | 内容 |
|---|---|
| `ecm_edwards_speed.cpp` | Edwards stage-1 单曲线计时（bits/B1/w/reps 参数化） |
| `ecm_edwards_bench.cpp` | N × NAF 窗口扫描 + 字典内存统计 |
| `gmp_mpn_microbench.c` | `mpn_addmul_1`/`mpn_mul_n`/`mpn_sqr` 的 cycles/limb（内联汇编标定主频） |
| `mont_probe.cpp`、`gmp_limb_probe.c` | Montgomery/limb 行为探针 |
| `bench_threads.ps1` | 多曲线并行线程数扫描（吞吐） |
| `build_speed.ps1`、`sweep_flags.ps1` | MSVC 编译标签扫描（单标签 / 批量） |
| `abtest_gmp.ps1` | 两套 GMP（通用 vs zen3/BMI2）交替 A/B |
| `bench_wg_scaling.ps1` | OpenCL work-group 规模扫描 |
| `mers_test.cpp` | 折叠/Montgomery 内核对拍（17 个模数、对抗输入、规范形）+ 扫描计时（ns / madds / Gmadd/s） |
| `mers_loop_bench.cpp` | 乘积主循环形状扫描（单行 0.67 → 两行 1.0 madd/cycle） |
| `mers_chain.cpp` | 30 万步链式域运算对拍（抓"值相关"bug） |
| `simd_mont_tail.cpp` / `simd_mont_notail.cpp` | 尾巴成本拆解（用 `set DEFS=/DIFMA_NOTAIl=1` 选无尾版，避免运行时 `getenv` 污染热路径） |
| `probe2.c` / `probe3.c` | 时钟与 `vpmadd52` 峰值标定（§15.3 数据的来源：4.0 GHz、1 madd/cycle） |
| `ecm_edwards_standalone.cpp` | 单曲线 stage-1 dump（`d`、基点、`s_bits`、`Qx/Qz`、`u`、`gcd`），用于与 Prime95/参考阶梯对拍。原先藏在 `src/cpu/ecm_edwards_cpu.cpp` 的 `BUILD_ECM_EDWARDS_STANDALONE` 里，现已搬出并加 CMake 目标 `ecm_edwards_standalone` |
| `cgbn_op_probe.cu` | **CGBN 逐算子单价**（`mont_mul`/`mont_sqr`/compare+cond-sub/add/sub/shift），可切 TPI/BITS 档位，可用 `-DXMP_WMAD/-DXMP_XMAD/-DXMP_IMAD` 切乘法链变体；用于判断"改哪个算子值多少"（结论见 `docs/ECM_CGBN_OPTIMIZATION.md`）。**文件必须保持 ASCII-only**（中文注释会让 nvcc 按 GBK 读、吃掉换行） |
| `cuda_kernel_ab.ps1` | **整 CUDA kernel A/B 计时**：固定 N（默认 `2^Bits−1`）/B1/曲线数，多次取中位数，输出 `gputime` 与 curve-bits/s；`-Device` 默认 1（计时要在空闲卡上做） |
| `simd_edwards_bench.cpp` | SIMD 批的验证 + 计时（`verify` 子命令与标量逐位对拍；`ED_SOA_FIELD=mont\|mers\|auto`、`ED_SOA_FSELFTEST=<n>`） |
| `ab_mersenne.ps1` | 验收 A/B：8 曲线 M3001 B1=1e6，折叠域 vs Montgomery，交替 min-of-3 + 存档字节一致性 |
| `ab_naf_window.ps1` | NAF 窗口 A/B（w=8/10/12） |
| `cpu_addsub_bench.cpp`、`cpu_addsub_avx512.cpp` | 早期 OpenCL 对照的 CPU add/sub 微基准（**已从 `src/cpu/` 搬到这里**，CMake 目标不变） |
| `cpu_mont_avx.cpp`、`cpu_mont_scalar.cpp`、`cpu_mont_bench.cpp` | 同上，Montgomery CIOS 微基准（**已从 `src/cpu/`、`src/` 搬到这里**） |

## stat/ — ECm 命中率统计与后端配对比较

| 文件 | 内容 |
|---|---|
| `ecm_hitrate.ps1` | 从 `ecm_prob/data/primes/bits<b>.bin` 抽样素数，**嵌进大合数**（`N = p·(2^521-1)`）后逐后端统计命中率，并与 `ecm_prob` 的独立参考值对比。`-Engine edwards\|mont\|gpu`、`-GpuParam 0\|3`、`-BitsFrom/-BitsTo`（或 `-Bits`）、`-Count N` / `-All`、`-Backend`、`-Curves`、`-Threads`、`-Device`、`-Output csv`、`-Quiet` 可调。输出 `hits/(primes×curves)`（逐曲线命中率）和 `primesHit/primePct`（至少命中一条曲线的素数比例）；`failedRuns` 非 0 说明驱动器根本没跑起来（该行的速率无意义） |
| `compare_b1.ps1` | 同一 `(N,sigma)` 上多个 B1 的三后端配对比较（命中数 + 存档最终点是否逐字节相同） |
| `bisect_b1.ps1` | 在已知失败用例上扫 B1 找最小复现档（当初逼出标量 bug 的那张表） |
| `ref_common.py`、`ref_ladder4.py`、`verify_hit4.py` | 独立 Python 参考阶梯（大整数 / 小模数 / 投影二进制造），用来裁定"这个命中是不是真的" |

> **必须嵌大合数**：若 `N = p` 是素数，stage-1 命中的 `gcd` 就是 N 本身，驱动当平凡因子丢掉
> ⇒ 命中率恒为 0，什么都测不出来。

> **数多行输出前先拆行**：`$out = cmd /c "..." 2>&1 | Out-String` 得到的是**一个**多行字符串，
> 而 `$out | Select-String -Pattern ...` 对它只返回**一条**匹配 —— 用 `.Count` 数命中会把
> "每个素数的多次命中"压成 1 次（2026-09-24 真踩到：报 6.25%，真值 30.47%，差 4.9 倍）。
> 要么先 `$out -split "`r?`n"`，要么用 `[regex]::Matches($out, ...)`。

> **含中文的 .ps1 必须存成 UTF-8 带 BOM**：Windows PowerShell 5.1 没有 BOM 时按系统 ANSI
> （中文系统 = GBK）解码脚本，UTF-8 的中文恰好会把**后面的引号/反引号**当成 GBK 尾字节吃掉，
> 于是出现"明明没写错却报语法错误"（2026-09-24 在 `stat/ecm_hitrate.ps1` 上真踩到：
> `"（速率不可信）："` 里的 `：` 把收尾的 `"` 吞了）。写成 UTF-8 **with BOM** 即可，
> `pwsh`（PowerShell 7）默认按 UTF-8 读所以看不出问题，但仓库里其它脚本是给 5.1 用的。
>
> **基准值**（`ecm_prob/out/measure_20_256.json`，独立实现 + 穷举 38635 个 20-bit 素数）：
> Edwards Z/2×Z/8、B1=256 → **32.66 %**。本仓库三后端实测（1000 素数 × 8 曲线）=
> 2611/8000 = **32.6375 %** 且**命中同一批 (素数,sigma)**；B1=1e5（长阶梯，60 素数 × 8 曲线）
> 三后端 = **477/480 = 99.375 %**。

## diag/ — 诊断 / canary / 崩溃复现

| 文件 | 内容 |
|---|---|
| `limb_canary.cpp` | **limb 高位污染**检查：乘/平方结果每个 limb 是否 < 2^52（任意 k × 两域）。锁定 §15.9 的 CIOS 未掩码 |
| `mont_canary.cpp` | **标量** Montgomery 层 canary（mul/sqr/add/sub/neg vs mpz + 原样 limb 规范性）。锁定 §15.10 的 `mpn_redc_1` 返回值误用 |
| `canary.cpp`、`canary_sqr.cpp` | 内核原始 limb 对拍（**不做 `mpz_mod` 掩蔽**）：`raw < N` 且等于期望值 |
| `helper_canary.cpp` | 域运算 `add/sub/neg` 的原样 limb 规范性与正确性 |
| `lane_indep.cpp` | lane 无关性（lane 0 固定、其余 lane 灌垃圾） |
| `selftest_many.cpp` | 域/点自检多轮跑（需先 `set_curves`，否则 `c->d` 未初始化） |
| `dump_tmp.cpp` | 用驱动自己的 reader 解析 `.tmp` 打印 `Qx/Qz`，用于跨后端比对最终点 |
| `crashloop.cmd` | 中止路径崩溃复现（逐次记 exit code，可看出 `0xC0000005`；§15.11 的回归） |
| `enc_diag.py` | 文件编码体检：UTF-8 是否有效、CJK/乱码字符数、与 `HEAD` 版本对比 |

> 三次 bug（§15.4 值非规范、§15.9 limb 非规范、§15.10 limb 非规范）都栽在同一件事上：
> **"读回 mpz 再比较"的测试是空洞的**。`ifma_to_mpz_lane` 先 `& 2^52−1` 再 `mpz_mod`，
> `mpz_import` 之后也总会规约 ⇒ 非规范表示在 mpz 眼里与规范表示一模一样。
> **要查规范形必须读原始 limb**。

> 本机（笔记本）持续满载会降频，**<10% 的差异必须交替 A/B 测量**才可信；
> 单配置连跑取 min 会把降频趋势当成配置差异。见 `docs/ECM_EDWARDS_STAGE1.md` §10。

## disasm/ — 反汇编 / ISA

`DISASM_SETUP.md`（Windows 工具链说明）、`install_disasm_tools.{bat,ps1}`、
`verify_disasm_tools.{bat,ps1}`、`disasm_mont_isa.ps1`、`disasm_addsub_isa.ps1`。

## gen/ — 代码生成器

内核 `.cl` 与参数文件的单源生成器；产物在 `kernels/` 下。
`gen_all.py` 批量跑 mp_addsub 一组生成器；`mp_asm_block_gen.py` 是共享库
（被多个生成器 `import`，因此与它们同目录）。

生成器一律以 `Path(__file__).resolve().parents[N]` 定位仓库根（`tools/gen/*.py` 为
`parents[2]`），或以命令行参数/当前目录为输出根；**不要再把它们移出本目录而不改深度**。

## refactor/ — 一次性工程脚本

`migrate_ecm_stage1_kernel_layout.py`、`migrate_operators_v2.py`、
`refactor_ecm_stage1_macro_iface.py`、`split_ecm_stage1_kernel_tree.py`、
`patch_npu_addsub_mod.py`、`validate_stage1.py`、`_trace_4x2.py`。
保留供追溯/重跑，日常开发不需要。

## ecm_prob/ · ecm_report/ · log_parser/

各自独立，用法见其目录内 `README.md`：

- `ecm_prob/` — 参数化概率/`D_eff` 实测（含 `FindGroupOrder3_example.gp`）
- `ecm_report/` — 进度数据库与图表（`import_ecm.py` / `download_ecm.py`）
- `log_parser/` — `screen.log` 等日志解析
