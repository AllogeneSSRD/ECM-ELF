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
├── p95feeder/  ecm_p95feeder（stage-1 结果投递给 Prime95 的独立进程）
├── ecm_prob/   ECM 参数化概率分析套件（Python，自带 README）
├── ecm_report/ 进度数据库/图表（Python + bat，自带 README）
├── log_parser/ 日志解析（自带 README）
└── ecm_worktodo/ worktodo 管线生成器（Python，Windows 原生：分配行 → stage1 队列 + P95 行）
```

> **`ecm_gui` 图形前端不在 `tools/` 下**：它是 `src/gui/`（与 `src/core`、`src/cuda` 平级的），
> `localization/` 也在 `src/gui/localization/`。见 `docs/DEV_ECM_GUI.md`。

## CMake 布局：`ecm` / `ecm_cuda` 在根，其余都是 tools

根 `CMakeLists.txt` 现在只定义**两个交付物** `ecm`（OpenCL）与 `ecm_cuda`（CUDA/CGBN）
+ 它们需要的两个静态库 `cgbn_opencl`、`opencl_ecm_entry`。所有 bench/test/diag 类可执行文件
都拆到了各自子目录的 `CMakeLists.txt`：

| 定义位置 | 目标 |
|---|---|
| `tools/bench/CMakeLists.txt` | `cpu_addsub_bench`、`cpu_mont_bench`、`simd_mont_gate`、`simd_edwards_bench`、`ecm_edwards_standalone`、`opencl_ecm_addsub`、`opencl_ecm_montsqr` |
| `tools/test/CMakeLists.txt` | `main`、`sliced_cios_test`、`sliced_cios_8192_test` |
| `tools/disasm/CMakeLists.txt` | `opencl_asm_selftest`、`opencl_mont_isa_export`、`opencl_addsub_isa_export` |
| `tools/p95feeder/CMakeLists.txt` | `ecm_p95feeder` |
| `src/gui/CMakeLists.txt` | `ecm_gui`（**仅 Windows**；`third_party/imgui/imgui.h` 缺失或 `-DECM_BUILD_GUI=OFF` 时自动跳过） |
| 根 `CMakeLists.txt` | `ecm`、`ecm_cuda`、`cgbn_opencl`、`opencl_ecm_entry` |

* **`-DECM_BUILD_TOOLS=OFF`**：不配置任何工具目标（configure 更快、构建树更小）；
  `ecm`/`ecm_cuda` 不受影响 —— 没有任何工具目标被它们依赖。默认 `ON`。
* **产物路径不变（这是硬约束）**：工具可执行文件仍然落在**构建根**（VS 生成器下是
  `<build>\Release\`），因为文档与脚本按路径引用它们
  （`build_vs18\Release\simd_mont_gate.exe`、`build_vs18\Release\ecm_p95feeder.exe`、
  `build_cuda_cmake\cpu_mont_bench.exe` …）。根文件里的 `ecm_tool_output_root(<target>)`
  就是干这个的。
* **AVX512 标志要重新声明**：`src/cpu/simd_*.cpp` 的 `/arch:AVX512` 是**源文件属性**，
  而源文件属性只在同一个 `CMakeLists.txt` 目录内可见。所以 `tools/bench/CMakeLists.txt`
  对 `simd_mont_gate` / `simd_edwards_bench` 用
  `set_source_files_properties(<abs path> TARGET_DIRECTORY <target> PROPERTIES COMPILE_OPTIONS ...)`
  重新声明一次（CMake ≥ 3.18）。改这两个目标时别把这段删了。
* 新增工具目标时：源码放 `tools/<group>/`，定义写进该目录的 `CMakeLists.txt`，
  照抄一行 `ecm_tool_output_root(<target>)`。

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
| `worktodo_sections_test.cpp` | **worktodo 段化**（`[Worker #N]`，D2）：段头语法、section-aware `first_line`/`advance`（只动自己段、段头与注释原样保留）、`list_workers`；**带断言、失败非零退出**（`tools\build_tool.bat tools\test\worktodo_sections_test.cpp src\core\ecm_worktodo.cpp`）|
| `test_worker_sections.ps1` | **D1+D2 端到端验收**：临时沙箱里造 `ecm.ini`（全局 + `[Worker #1]`/`[Worker #2]`）与分段 `worktodo.txt`，跑 `ecm_cuda.exe -ini … --worker 1/2/3`，断言段覆盖生效、每 worker 独立日志默认值、只标记自己段的行、ini 不被改写、无段文件保持旧行为（24 项检查）|
| `test_gui_smoke.ps1` | **ecm_gui 真窗口冒烟**：拉起真窗口（Win32+D3D11+ImGui）→ `PostMessage(WM_CLOSE)` → 断言退出码 0、`[GUI] window=`/`dock_layout=` 落盘、driver 键与注释未被改动、`.bak` 存在、**driver 仍能读改写后的 ini**、第二次启动窗口几何与第一次一致；**[1b]/[1c] 最小化→还原后进程存活 + 最小化状态下 post `WM_CLOSE` 必须退出 0**（消息泵回归护栏）；**[6] 四象限布局几何**（都在客户区内／不重叠／左 60 % 右 40 %／上下顺序）；**[7] 字体 ≈15×DPI 且中文 `cjk_ok=1`**（47 项检查；用 `EnumWindows` 找本进程 class=`ecm_gui` 的窗口，并处理 DPI 坐标系差异）|
| `test_gui_workers.ps1` | **ecm_gui M2 监管验收（假 worker，不需要 GPU）**：四个 worker（正常／崩溃一次／挂死／**旧 driver**）autostart，断言 trace 里的 `autostart pid=`、`state Running`→`QueueEmpty`、崩溃→`Restarting`→恰好一次 `restart #1`、关闭时停止仍在跑的 worker、无残留进程、ini 里 driver 键与 `extra_args`/`autostart` 原样保留；第 4 个 worker 用 `--scenario stale-driver`（逐字节复现 D1/D2 之前那份 ecm_cuda 的输出）断言 GUI 打出 `DIAGNOSIS: … --worker …`；另读 `table: workers …` 实测几何断言**进度条宽度、ETA 列在面板内、`fits=1`、任务行跨整行**；关闭时断言**静默终止**（单行 `shutdown: terminated N running worker(s)`、之后不得出现 `Error/Restarting/restart`）（34 项检查）|
| `test_gui_real_workers.ps1` | **ecm_gui M2 监管验收（真 `ecm_cuda.exe` + 双卡）**：同一份 `worktodo.txt` 的两个 `[Worker #N]` 段各一条 M991/M997（B1=1e4、8 曲线），断言两行都被成功移除、无 `# ERROR`、`finished` 两条、两份 `screen_<N>.log` 各自只出现自己的指数与 device（"没串"的逐条证据）、无残留进程（23 项检查）|
| `test_hit_fields.ps1` | **D3 验收**：M677/B1=1e6、8 曲线走**队列模式**，断言命中行含 `curve=/sigma=/param=/method=/save=` 六个字段、`curve` 与下标一致、`sigma-curve` 为同一常数、param/method/save 与任务一致、因子真的整除 `2^677-1`（11 项检查）|
| `test_gui_gpu.ps1` | **M4/NVML 验收**：① 跑 `--gpu-selftest` 并检查报告（退出码 0、无失败项、NVML 已加载、有设备、交叉比对通过）；② 真 GUI 的 trace 必须报 `gpu: nvml ok`，逐卡名称与 `nvidia-smi` 一致、设备数相等；③ `ECM_GUI_NVML` 指向不存在的 DLL 时 GUI **仍能启动**、trace 报 `gpu: NVML unavailable`、干净退出（15 项检查）|
| `test_gui_results.ps1` | **M5 验收（真驱动，两轮）**：同一任务（M677/B1=1e6/8 曲线）跑两遍 GUI 队列，断言 `results.json.txt` 是**追加式 JSONL**（字段齐、因子都整除 `2^677-1`）、`results.txt` 始终**一个因子一行**并带 curve/sigma 列表与命中数、两轮命中数相加等于 JSONL 对象数、已知因子 `1943118631` 的 sigma 列表随第二轮增长、每行都能由 JSONL 复算（27 项检查）|
| `test_gui_cjk_pixels.ps1` | **中文渲染的像素级验收**：抓真窗口（`PrintWindow`），量菜单栏里逐字形格子的**中位宽度**与**不同 ink 值个数**。四轮：[A] 自动字体 → 全宽中文（实测 5 格 / 30 px = 0.75 em / 5 种 ink）；[B] `[GUI] font` 指到画不出中文的字体（`arial.ttf`）→ 必须**被救回**系统 CJK 字体且像素仍是全宽中文（实测 0.43 em / 2 种 ink 是**修好之前**的 tofu 标定值）；[C] `ECM_GUI_CJK_FONT=none` → 必须**切回英文**、绝不出现 `???`；[D] 英文启动 + `--switch-language chineseSimplified`（= Language 菜单）→ 字体必须重挑、切换后像素是全宽中文（31 项检查）|
| `test_gui_gpu_curves.ps1` | **"曲线是不是一条直线"的机器判据**：真放一个 `ecm_cuda` worker 压 device 0 约 40 s，读 `--trace` 的 `gpu: history` 行，断言忙卡 `util/power/clock` 各有 ≥5 个**不同取值**、`flat=0`、`plot=lo..hi` 非退化且覆盖观测区间；关闭后不残留驱动进程。实测忙卡 `power=22 distinct`、`clock=21 distinct`（空闲卡只有 1~2 个不同值 —— 那才是"真·直线"）（15 项检查）|
| `test_gui_exit_checkpoint.ps1` | **退出流程验收（真驱动）**：`exit_confirm = ask` 时 `WM_CLOSE` **不关窗**而是弹确认框（trace `exit: confirmation requested`），Esc 取消后 GUI 继续跑；回车确认后 GUI **等到新的检查点落盘**（`worker 1: checkpoint written (…), safe to stop`）才终止 worker，并**独立核对磁盘上 `.ecm_ckpt_*.dat` 的 mtime 比关闭前更新**；`exit_confirm = kill` 仍可立即退出。脚本用 `PostMessage(WM_CLOSE/VK_RETURN/VK_ESCAPE)` 驱动，就像用户在动手（27 项检查）|
| `test_gui_all.ps1` | **一键跑完整套件**：15 个入口（headless 自测/单测 + 上面所有真窗口脚本）按顺序跑，逐项打印 `passed/failed` 与耗时，汇总表 + 每项完整日志写到 `tools/test/_run/suite_<时间戳>/`，任一失败非零退出。`-SkipGpu` 跳过需要显卡的项、`-Only '*smoke*'` 挑着跑、`-List` 只列清单、`-KeepGoing` 失败后继续；**每项有超时**（默认 900 s，超时算失败），并且**开始/结束/每项之后都清掉 `tools/test/_run` 下遗留的 ecm_gui/fake worker/ecm_cuda 进程**（只杀沙箱副本，绝不动真实构建或生产目录）|
| `ecm_edwards_save_test.cpp` | Prime95 ECM_VERSION=6 存档读写；与真实 `e0000347` 字节级比对 |
| `ecm_edwards_checkpoint_test.cpp` | 分块标量乘的中止/恢复等价性（Qx/Qz 一致） |
| `gen_ckpt.cpp` | 生成一个中途 STAGE1 存档，用于验证驱动恢复 |
| `test_feeder.ps1` | `ecm_p95feeder` 端到端集成测试（沙箱 p95 目录，7 个周期） |
| `test_agreement.ps1` | **三后端互认**：同一 `(N,B1,sigma)` 跑 标量 / SIMD-Montgomery / SIMD-折叠域，断言命中集合、因子值、以及每个 `.tmp` **逐字节相同**。含历史失败用例（M3001 σ=20260922 B1=1e5），是 §15.9/§15.10 两个 bug 的回归 |
| `test_invariants.ps1` | §7 因子不变式回归：M677→1943118631、M991→8218291649、M4003→16756559，两域各一遍 |
| `test_cuda_mers_fold.ps1` | **梅森折叠域验收**（`-DECM_MERS_FOLD=1` 构建）：折叠 ↔ Montgomery ↔ CPU 三方差分逐行存档对照（M991/M3217/M4999/2^4400−1，param0 + param2）、非梅森 N 与 `--gpu-param 3` 的守卫、以及 M4999 的吞吐 A/B（`-ExpectFoldFaster` 才把"更快"当门限，默认现状是**折叠更慢**，见 `docs/ECM_CGBN_OPTIMIZATION.md` §9） |
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

## ecm_worktodo/ — stage1/stage2 任务分配管线（Windows 原生）

`ecm.py` 把分配行（Prime95 原生 `ECM=`/`ECM2=`，两种前缀等价）加工成两份 worktodo：
**stage 1 由我们的驱动跑**（`ECMSTAGE2=` 行），**stage 2 由 Prime95 跑**（`ECM=`/`ECM2=` 行）。
它**只生成、从不执行**，也从不改写运行中的 Prime95 拥有的文件。

```powershell
# 生成 stage1 队列（ECMSTAGE2=，B1 藏在 save 名里）+ P95 原生行
python tools\ecm_worktodo\ecm.py --input tools\ecm_worktodo\sorted.csv `
    --set-b1 110e6 --gpu-curves 960 --sort-by n `
    --out-ecmstage2 worktodo_add.csv --out-ecm p95_worktodo.txt
# 只看统计不写文件；逐任务命令行脚本；旧开关名（--out-windows/--out-prmers/--out-linux）仍然可用
python tools\ecm_worktodo\ecm.py --input sorted.csv --dry-run
python tools\ecm_worktodo\ecm.py --input sorted.csv --emit-cli todo.ps1 --emit-cli-kind ps1 --device 1
```

要点（都是实测/按驱动实现定的，别猜）：

| 规则 | 说明 |
|---|---|
| **去重判据** | `(k,b,n,c)` **按数值**比较；冲突保留 **B1 最大 → curves 最大 → 真 AID 优先 → 先出现者**；known factors 取**并集** |
| **管线顺序** | 解析 → 过滤 → 去重 → 重写（`--set-b1/--set-b2/--set-has-na`）→ 排序。过滤在去重**之前**，否则胜出行被过滤掉会让整个数消失 |
| **save 名契约** | `ECMSTAGE2=` 没有 B1 字段，驱动用 `ecm_extract_b1_from_save_name()` 从名字里抽 ⇒ 名字必须匹配 `…_<B1>.save`。工具**强校验并非零退出**；给别的消费者（P95 侧自己从存档读 B1）时用 `--allow-invalid-save-name` |
| **`B2=0` 的语义** | Prime95 里 0 = **自动选取 B2**，不是"不做 stage 2"；`--set-b2` 就是写这个字段 |
| **换行/编码** | 读自动识别（UTF-8 有/无 BOM → GBK 兜底、无视 BOM、CRLF/LF 都吃）；写 **UTF-8 无 BOM + CRLF**（`--newline lf` 可切）；统计输出是 ASCII |
| **可选的校验开关** | `--sort-factors`（因子按数值升序规范化，默认关以保持逐字节可比）、`--verify-factors`（精确整数整除校验，默认关） |

入口脚本 `ecmcuda.bat` / `ecmpr.bat` 是薄封装（**GBK 编码**，因为里面有中文路径；换机器只改文件头的 `set` 变量）。
回归测试：`tools/test/test_worktodo_pipeline.ps1`（33 项，含"输出能被我们的 C++ 解析器接受"的跨实现验收）。

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
| `fix_bom.py` | **编码体检/修复**：对"含非 ASCII 且缺 UTF-8 BOM"的源文件补回 BOM（用法 `python tools\diag\fix_bom.py <file...>`）。任何一次"读出来再写回去"的编辑都会丢掉 BOM，而 nvcc/cl 会把无 BOM 的中文注释按 GBK 读、**吃掉换行**，导致下一行的 `#define` 被并进注释（本轮真踩：`CHECKPOINT_VERSION is undefined`）。改完 `cgbn_stage1.cu` 之类的文件请顺手跑一次 |
| `cgbn_op_probe.cu` | **CGBN 逐算子单价**（`mont_mul`/`mont_sqr`/compare+cond-sub/add/sub/shift），可切 TPI/BITS 档位，可用 `-DXMP_WMAD/-DXMP_XMAD/-DXMP_IMAD` 切乘法链变体、`-DPROBE_VALUE_MODE=0/1/2` 检验算子对**操作数值**是否敏感（实测不敏感：通用/0/1 都是 0.91 ns）；用于判断"改哪个算子值多少"（结论见 `docs/ECM_CGBN_OPTIMIZATION.md`）。**文件必须保持 ASCII-only**（中文注释会让 nvcc 按 GBK 读、吃掉换行） |
| `cgbn_mers_fold_probe.cu` | **梅森折叠 vs Montgomery 的逐算子探针**：8 档位 × `mont_mul`/`mul_wide`/`mul_wide+reduce`/`fold_gen`/`fold_align`，每档两遍、每个算子一条 1000 步链，全部用 GMP `mpz_powm` 校验，另有 18 个特殊值 × 自身的边界电池（`ktest=edge`）。结论：折叠在 ≥4608 bit 每模乘便宜 21–31%，但**整 kernel 反而慢 15–19%** —— 链口径排不了 throughput，见 `docs/ECM_CGBN_OPTIMIZATION.md` §9 |
| `cuda_kernel_ab.ps1` | **整 CUDA kernel A/B 计时**：固定 N（默认 `2^Bits−1`）/B1/曲线数，多次取中位数，输出 `gputime` 与 curve-bits/s；`-Device` 默认 1（计时要在空闲卡上做）。探针实验必须 `-NExpr <素数>`，否则垃圾状态会撞出假因子、批次提前结束 |
| `param2_gen_cost.cpp` | **主机侧建曲线成本**（GMP 独立工具，N 走 stdin，如 `cmd /c "build_vs18\tools\param2_gen_cost.exe 3000 3 < n.txt"`）：对比 param3 形状与 gmp-ecm `get_curve_from_param2`（加法链 + 3 次模逆）的 ms/curve。结论见 `docs/ECM_CGBN_OPTIMIZATION.md` §5.6。**注意别用 PowerShell 管道喂 N**（会加 BOM，gmp-ecm 报 invalid number） |
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
| `ensure_bom.ps1` | **BOM 守卫（构建前自动跑）**：把 `kernels/`、`src/` 下每个源文件与 git HEAD 的 BOM 状态比对并恢复（`-NoFix` 只报告）。与 `bench/fix_bom.py` 的区别：后者是"含非 ASCII 就补 BOM"的钝器（会给本来就无 BOM 的文件制造 diff），前者只在**编辑往返把 BOM 抹掉**时修回去。`tools/build/local_build.ps1` 已内置调用（2026-09-26：`cgbn_stage1.cu` 的 BOM 被抹掉 ⇒ nvcc 按 GBK 读 ⇒ `#define CHECKPOINT_VERSION` 被中文注释吃掉） |
| `grab_window.ps1` | **窗口截图**（GUI 无头验收的补充）：`-ProcessName ecm_gui -Class ecm_gui -Out shot.png`。按窗口类找顶层窗、用 `PrintWindow(PW_RENDERFULLCONTENT)` 抓客户区存 PNG（D3D11 窗口不加该 flag 会抓到空帧）。人不在屏幕前时用它看界面 |
| `row_ink_profile.ps1` | **文字行定位**：对截图逐行统计"墨"量（背景亮度取全图直方图众数，`-Delta` 默认 140），把连续的行归成文字带并打印前 60 行明细。用来把 `text_ink_probe.ps1` 对准真正的文字（ImGui 菜单栏不一定在 y=0，抓图还含标题栏） |
| `text_ink_probe.ps1` | **字形像素体检**：自动找第一条文字带 → 按空列切成逐字形格子 → 报每格宽/高/墨量/中心墨量、中位宽度、不同墨值个数（`-Json` 给脚本用）。判"真字形 vs tofu 方块"的**有效**判据是宽度（0.75 em vs 0.43 em）与"格子是否互相雷同"，不是"中心空不空"（fallback 方块中心也有笔画） |
| `check_printf_calls.ps1` | **格式串审计**：扫 `src/gui/*.cpp` 里所有 `ImGui::Text*` 调用，数格式串的转换符个数 vs 实参个数，不一致就报行号。起因是一次真实的 0xC0000005：`ImGui::Text("… %s … %s …", …, s.clock_mem_mhz)` 里 `%s` 收到整数，`vsnprintf` 把它当指针解引用，而且**只在显存频率非 0 时才崩**（为 0 时 MSVC 打印 `(null)`），于是"偶尔崩一次"躲过了很久的测试。当前 0 处不符 |

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

## src/gui/ — 图形前端 `ecm_gui`（M1 骨架 + M2 worker 管理 + M3 解析 + M4 GPU 监控 + M5 results + M8 反馈修正 已落地）

> 代码在 **`src/gui/`**（不在 `tools/` 下，与 `src/core`、`src/cuda` 平级）。本节留在这里是因为
> tools 文档承担"所有可执行目标怎么跑、怎么验"的索引职责。

多 worker 进程管理 + 每 worker 输出窗 + GPU 监控（NVML）+ 命中因子汇总（`results.json.txt` / `results.txt`）。
**配置全部在 `ecm.ini` 里**（全局键 = 所有 worker 的默认值，`[Worker #N]` = 覆盖，`[GUI]` = 前端自身设置），
worker 进程用 `ecm_cuda.exe -ini ecm.ini --worker N` 启动。

```powershell
cmake --build build_gui --target ecm_gui ecm_gui_fake_worker ecm_gui_log_parse_test ecm_gui_results_test
build_gui\ecm_gui.exe                      # 正常启动（配置来自 <exe 目录>\ecm.ini）
build_gui\ecm_gui.exe --selftest           # 无窗口自测：ini/本地化/字体/状态键（36 项）
build_gui\ecm_gui.exe --worker-selftest    # 无窗口自测：worker 进程管理（48 项，不需要 GPU）
build_gui\ecm_gui.exe --gpu-selftest       # 无窗口自测：NVML + 与 nvidia-smi 交叉比对（27 项）
build_gui\ecm_gui_log_parse_test.exe       # 解析层单测（49 项，含"旧 driver"分类）
build_gui\ecm_gui_results_test.exe         # results 双文件单测（42 项）
build_gui\ecm_gui.exe -ini <path> --trace  # 生命周期写 <exe 目录>\ecm_gui_trace.log
build_gui\ecm_gui.exe -ini <path> --switch-language chineseSimplified
                                           # 诊断：等于在 Language 菜单里切一次语言（脚本用）
                                           # 另有 ECM_GUI_CJK_FONT=<path|none> 覆盖 CJK 字体查找
powershell -File tools\test\test_gui_smoke.ps1        # 真窗口冒烟（57 项：含最小化/还原、布局几何、表格几何、字体与中文）
powershell -File tools\test\test_gui_workers.ps1      # 假 worker 监管 + 表格几何 + 静默关闭（34 项，含"旧 driver 不认识 --worker"诊断）
powershell -File tools\test\test_gui_real_workers.ps1 # 真 ecm_cuda + 双卡（23 项，需 GPU）
powershell -File tools\test\test_gui_gpu.ps1          # NVML 面板 + 降级（15 项）
powershell -File tools\test\test_gui_results.ps1      # results 双文件（27 项，真 ecm_cuda，两轮）
powershell -File tools\test\test_gui_cjk_pixels.ps1   # 中文渲染像素验收 + 救回/英文回退/运行中切语言（31 项）
powershell -File tools\test\test_gui_gpu_curves.ps1   # 真 worker 压卡：功率/频率曲线确实在变（15 项）
```

要点：

| 项 | 事实 |
|---|---|
| 构建开关 | `-DECM_BUILD_GUI=ON/OFF`（默认 ON）、`-DECM_IMGUI_DIR=<dir>`；非 Windows 或目录里没有 `imgui.h` → `gui: skipped`，**其它目标不受影响** |
| 目标 | `ecm_gui`（WIN32 可执行，产物在构建根）、`ecm_gui_imgui`（vendored ImGui）、`ecm_gui_fake_worker`、`ecm_gui_log_parse_test`、`ecm_gui_results_test` |
| 依赖 | 只链系统库（`d3d11 dxgi d3dcompiler dwmapi shell32`）；**不**链 `ecm`/`ecm_cuda`/GMP/OpenCL |
| worker 进程 | `CreateProcessW` + `CREATE_NO_WINDOW`、stdout/stderr **共用同一管道写端**、Job object（`KILL_ON_JOB_CLOSE`）→ 关界面/崩溃不留孤儿占卡；崩溃 5 s 后重启，5 分钟内 3 次则熔断停住 |
| GPU 监控 | NVML **运行时动态加载**（`nvml.dll`；测试可用 `ECM_GUI_NVML=<dll>` 指到别处验证降级），采样在独立线程。每卡：util/功耗与上限/SM 与显存时钟/温度/显存/节流原因 + 三条曲线 + 全机合计功耗。**注意**：`nvmlDeviceGetNumGpuCores` 给的是 CUDA 核数（不是 SM 数），SM 数由 driver 的 `--gpu-info`（D4，待做）提供 |
| 布局 | 默认停靠布局（`DockBuilder`，版本 3）：**左列 60 % 宽 = 上 `Workers`（`详情` 与之同节点做标签页）+ 下 每 worker 输出（标签页）；右列 40 % 宽 = 上 `GPU` + 下 `Results`**，每列上下各 50 %。面板用固定 ID（`###workers` 等）→ 切语言不会打乱布局。布局写进 `[GUI] dock_layout` + `dock_layout_ver`（版本变旧会自动重建一次，修掉"面板堆在中间"的历史 ini）。`--trace` 会打每面板矩形，`test_gui_smoke.ps1` 据此断言"都在客户区内、互不重叠、左 60/右 40、上下顺序正确" |
| 窗口生命周期 | 最小化时**仍泵消息**（否则唤不醒、只能从任务栏关）但**整帧跳过渲染**（0×0 交换链 `Present` 会崩）；`ResizeBuffers` 前先解绑 + `Flush` RTV，失败则重建交换链。回归护栏 = 冒烟测试 [1b]/[1c] |
| 布局持久化 | 关掉 ImGui 自带的 `imgui.ini`（`io.IniFilename=nullptr`），停靠布局 + 窗口几何写进 `[GUI] dock_layout=` / `window=`；ini 改写保注释、保行序、只动自己认识的键，写前留一代 `.bak` |
| 截图 | `powershell -File tools\diag\grab_window.ps1 -ProcessName ecm_gui -Out shot.png`（按窗口类抓图；D3D11 窗口用 `PrintWindow(PW_RENDERFULLCONTENT)`，否则抓到空帧）|
| 界面文案 | `src/gui/localization/*.xml`（UTF-8 无 BOM），`english.xml` 是基线，缺键回退英文；GUI 里 `Language → Reload localization` 热重载 |
| 字体 | 不打包字体：运行时挑系统字体（CJK → `msyh.ttc` 等；拉丁 → `segoeui` 等），字号 `[GUI] font_size = auto` = `15 px × DPI`（150 % 屏 = **22.5 px**，不凑整；也可写小数如 `23.5`，范围 6–96），`[GUI] font_snap = 1` 让字形推进量对齐整像素（拉丁小字更锐）。**画不出当前语言的字体不会被使用**：`[GUI] font` 指定了也先救回系统 CJK 字体，实在没有就切回英文界面（绝不显示 `???`）；运行中在 Language 菜单切语言会**重新挑字体**。ImGui 1.92+ 动态图集：**不要**再传 `GetGlyphRangesChineseFull()`；ImGui 只有灰度抗锯齿（无 ClearType），想更清楚请调大 `font_size` |
| 构建 | **一条命令**：`powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_gui.ps1`（自动找 `vcvars64.bat`、必要时 configure、构建 4 个目标；加 `-Selftest` 顺带自测，`-Clean` 从零重建，`-Reconfigure` 换依赖路径后重配）。**注意**：GUI 的构建目录是 **NMake Makefiles**，所以直接在普通 PowerShell 里跑 `cmake --build build_gui --target ecm_gui` 必定失败（`nmake` 只存在于 VS 开发者环境）—— 要么用这个脚本，要么先 `call vcvars64.bat` |
| 跑测试脚本的参数 | GUI 类脚本传 `-Exe <ecm_gui.exe>`，**driver 类脚本（`test_worker_sections.ps1`）传 `-Exe <ecm_cuda.exe>`** —— 传错不会报错，只会让 `ecm_gui.exe -ini … --worker 1` 开出一个什么都不做的窗口（GUI 现在会对 `--worker` 打警告）。用 `test_gui_all.ps1` 跑就不会踩到：每项都声明了要哪种 exe |
| worker exe 前提 | GUI 用 `<exe> -ini <ini> --worker N` 启动 worker，所以 **worker 可执行文件必须支持 `--worker`**（即 D1/D2 之后的构建）。旧构建会走单跑路径并打印 `No input number on stdin`，GUI 会把它诊断成 `DIAGNOSIS: … older than the D1/D2 driver changes` 并显示在状态栏/日志/表格红 `!`。一条命令自查：`([Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes('<path>\ecm_cuda.exe'))).Contains('--worker')` |
| DPI | GUI 是 per-monitor DPI aware（物理像素）；外部脚本比较窗口坐标时要乘 `GetDpiForWindow()/96`，字号按同一比例走 |
| 开发文档 | [../docs/DEV_ECM_GUI.md](../docs/DEV_ECM_GUI.md)（决策记录、里程碑、验收、坑）；worktodo 段化与 `gpucurves` 推荐见 [../docs/DEV_ECM_WORKTODO.md](../docs/DEV_ECM_WORKTODO.md) |

## ecm_prob/ · ecm_report/ · log_parser/

各自独立，用法见其目录内 `README.md`：

- `ecm_prob/` — 参数化概率/`D_eff` 实测（含 `FindGroupOrder3_example.gp`）
- `ecm_report/` — 进度数据库与图表（`import_ecm.py` / `download_ecm.py`）
- `log_parser/` — `screen.log` 等日志解析
