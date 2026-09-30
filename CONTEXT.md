# CONTEXT.md — 领域术语辞典

本项目在 [GMP-ECM](https://gitlab.inria.fr/zimmerma/ecm) 因子分解框架基础上，实现 ECM Stage-1 在
OpenCL GPU 上的加速。以下术语按领域分组，为跨设计讨论提供精确、无歧义的共享语言。

---

## 因子分解与 ECM

| 术语 | 定义 |
|------|------|
| **N** | 待分解的大整数（target number to factor），通常以 `(2^k-1)` 形式输入。 |
| **ECM** | Elliptic Curve Method，椭圆曲线因子分解法。以随机椭圆曲线 `(x³+Ax²+x) mod N` 反复尝试寻找 N 的非平凡因子。 |
| **Stage 1** | ECM 第一阶段：使用标量 `s` 乘初始点 `P`，若 `s·P` 产生不可逆的 `z` 坐标则找到因子。本文库目前仅实现 Stage-1 的 OpenCL 加速。 |
| **B1** | Stage-1 光滑性界限。`s = ∏_{p≤B1} p^{⌊log_p(B1)⌋}` 是所有 ≤B1 素数的幂乘积。 |
| **s（batch product）** | 光滑标量，其比特序列（MSB→LSB）驱动 Montgomery Ladder 的每一步。用户输入的 `B1` 通过 GMP-ECM 内置的 `s` 生成逻辑转换。 |
| **sigma（σ）** | Suyama 参数化下的曲线种子。每条曲线的 sigma, sigma+1, ..., sigma+curves-1 各自定义一条不同的椭圆曲线。 |
| **curves（gpucurves）** | 单次 GPU 启动并行处理的曲线数量。每条曲线独立地在 GPU 上运行 `kernel_double_add`。 |
| **Factor found in Step 1** | GPU 计算后，Host 用 `z_final` 与 `N` 求 GCD 得到非平凡因子。若结果为 `N` 或 `1` 则表示该曲线无贡献。 |
| **False factor** | 因子恰好等于 `N`，表明 ECM 计算错误（如算子选择不当导致运算走样）而非找到真正因子。 |
| **Trivial factor** | 因子等于 `1` 或 `N`，无分解价值。 |
| **Smoothness** | 若 `#E(F_p)`（曲线在素域 F_p 上的阶）的素因子全部 ≤B1，则该曲线能成功分解。`--go` 通过 `gp` 计算群阶帮助诊断。 |

---

## 多精度算术

| 术语 | 定义 |
|------|------|
| **limb** | 32 位无符号整数（`uint`），多精度数的基本单元。 |
| **limbs** | 一个多精度数的 limb 个数。`limbs = (N_bit_size + carry_bits + 31) / 32`。 |
| **MAX_LIMBS** | 编译时常量，设定内核容器大小（以 limbs 计）。例如 M421 使用 `MAX_LIMBS=16`（512-bit 容器）。 |
| **Container bits** | `MAX_LIMBS × 32`，GPU 内核的"计算盒子"。N 的实际比特数必须 ≤ container_bits - 进位预留。 |
| **CARRY_BITS** | 进位预留比特数（当前为 6），为 Montgomery 乘法的中间进位提供安全余量。 |
| **Exact-Fit Container** | 固定位宽算子的精确匹配容器：分配的 limbs 恰好等于算子所需，无冗余填充。与旧版"Padding Container"（一律 ≥16 limbs）相反。`max_container_limbs` 对 fixed_width 算子即为其 exact-fit container 的 limbs 数。 |

---

## Montgomery 算术

| 术语 | 定义 |
|------|------|
| **Montgomery Domain** | 模 N 运算的一种表示法：值 x 以 `x·R mod N` 形式存储，其中 `R = 2^(32·limbs)`。转换后乘/平方避免了昂贵的除法，代价是输入/输出需要额外的 Montgomery 规约。 |
| **mont_mul** | `mont_mul(out, a, b, N, np0, limbs)` — Montgomery 乘法：`out = a·b·R⁻¹ mod N`。 |
| **mont_sqr** | Montgomery 平方：`out = a²·R⁻¹ mod N`，通常与 mul 共享算子文件。 |
| **add_mod** | `add_mod(r, a, b, N, limbs)` — 模加法：`r = (a+b) mod N`。 |
| **sub_mod** | `sub_mod(r, a, b, N, limbs)` — 模减法：`r = (a-b) mod N`，返回 borrow 指示是否需补 N。 |
| **special_mult** | `special_mult(r, m, N, np0, limbs)` — 单 limb Montgomery 乘法 `r·m·2⁻³² mod N`，用于 Stage-1 ladder 中的特殊位宽乘数（Suyama d）。与 mont_mul 类似也有固定位宽变体（192/256/384/512/768/1024b）和 generic 通用版，由注册表按容器大小自动选择。 |
| **np0** | Montgomery 还原参数：`np0 = -N⁻¹ mod 2³²`，用于 CIOS 迭代中快速选择商。 |
| **CIOS** | Coarsely Integrated Operand Scanning，Montgomery 乘法的一种高效实现模式。每次外层迭代处理一个 limb，内层做乘积-累加-规约。 |
| **Montgomery Ladder** | ECM 的 double-and-add 主循环。按 s 的比特序列逐位处理，恒定时间，不暴露比特值。每次迭代调用 `double_add_v2`。 |

---

## 算子体系

| 术语 | 定义 |
|------|------|
| **算子（Operator）** | `mont_mul`、`mont_sqr`、`add_mod`、`sub_mod`、`special_mult` 中的一个可替换函数。每个算子有一个标识符（id）、OpenCL 函数名（cl_name）、别名数组（aliases）和源文件路径（kernel_path）。 |
| **算子族（Family）** | 五族：`mont_mul`、`mont_sqr`、`add_mod`、`sub_mod`、`special_mult`。`mont_mul` 与 `mont_sqr` 共用同一 `.cl` 文件但导出不同函数名。 |
| **Fixed-Width 算子** | 固定位宽算子，每个算子运行在其 Exact-Fit Container 中（如 `asm_384b` → 12 limbs、`unroll_1024b` → 32 limbs），不可用于其他容器大小。与 Generic 算子互斥。 |
| **Generic 算子** | 不限位宽的通用算子（`priv_opt`、`unroll32`、`generic`），`max_limbs=0` 表示无上限。 |
| **Auto priority** | 自动选择时的优先级（越小越优先）。`-1` 表示仅可通过显式 `--mul`/`--sqr` 等指定。 |
| **min_limbs / max_limbs** | 算子可接受的最小/最大 limbs 数。`0` 表示无限制。`max_limbs=0` 为通用算子（适用于任意容器）。 |
| **max_container_limbs** | 容器大小约束字段：对 fixed_width 算子等于其 Exact-Fit Container 的 limbs 数；对 generic 算子为 `0`（无容器限制）。若当前 limbs > max_container_limbs（且非 0）则拒绝该算子。 |

---

## 路径注册表

| 术语 | 定义 |
|------|------|
| **X-macro** | C 预处理器的单一数据源模式。`ECM_MONT_OPERATORS(X)` 宏被 mul 行宏和 sqr 行宏各展开一次，自动生成两张描述符表。消除手工维护四份相似代码。 |
| **EcmStage1KernelBuildPlan** | 组装一次 OpenCL 内核所需的全部元信息：`limbs`、`tpi`、五个算子的描述符指针（mul/sqr/add/sub/special_mult）、标准化开关等。 |
| **Operator interface（operator_iface.h.cl）** | OpenCL 内核中的宏别名层：`#define mont_mul(...) ECM_STAGE1_MUL_IMPL(...)` 等。Host 注入 `ECM_STAGE1_*_IMPL` 宏，内核代码只调用统一的别名名。 |
| **Kernel assembly** | Host 按固定顺序将多个 `.cl` 文件拼接为完整内核源码的过程：宏注入 → 公共头 → 算子文件 → 接口宏 → 主入口。 |
| **Vendor mask** | `ECM_GPU_AMD`、`ECM_GPU_NVIDIA` 等位掩码，控制算子在不同 GPU 上的可用性。`ECM_GPU_ANY=0xFFFFFFFF` 表示不限。`exclude_mask` 反向排除特定厂商。 |

---

## 实现细节

| 术语 | 定义 |
|------|------|
| **TPI** | Threads Per Instance，每条曲线使用的线程数。必须整除 limbs 且为 2 的幂。环境变量 `ECM_OPENCL_TPI` 可覆盖。 |
| **Cooperative Work-Group（coop_wg）** | 多线程协作，仅用于 4096-bit 容器。`coop_wg > 1` 时加载 `ecm_stage1_coop.cl`，通过 `__local` 内存共享算子中间结果。 |
| **Checkpoint** | 长时间运行的 `s` 处理可中途保存进度到 `.ckpt` 文件，崩溃后可恢复。包含已处理比特数、曲线数据等。 |
| **Batch processing** | 将 `s` 的全部比特拆分为多个批次在 GPU 上递进（如每次 200-1300 比特），避免单次 Kernel 超时或内存不足。 |
| **Kernel source loading** | 通过 `cgbn::opencl::load_ecm_stage1_kernel_file()` 加载 `.cl` 文件，根路径由 `ECM_KERNEL_ROOT` 环境变量或默认项目根决定。 |
| **ECM_STAGE1_KERNEL_REV** | 内核版本号（当前 14），用于构建选项缓存和兼容性检查。 |

---

## GUI 与多 worker（`ecm_gui`）

| 术语 | 定义 |
|------|------|
| **worker** | 一个 stage-1 执行体。GUI 里一个 worker = 一个 `ecm_cuda.exe -ini ecm.ini --worker N` 进程（进程隔离、独立输出窗、独立 Job object）。ini 的 `[Worker #N]` 段与 worktodo 的 `[Worker #N]` 段用**同一编号**指同一个 worker。 |
| **`[Worker #N]` 段** | ini 与 worktodo 共用的分段方式：无段 = worker 1；解析顺序 `段内键 → 全局键 → 内置默认`。不同 GPU 的最佳 `gpucurves` 不同，因此 worktodo 必须分段而不能共享消费。 |
| **`[GUI]`** | `ecm.ini` 里的前端专属段（窗口几何/停靠串、`refresh_hz`、`gpu_poll_ms`、`language`、`priority`、`font_size`/`font`、results 路径）。driver 忽略未知键，所以两者同住一个文件。 |
| **`--worker N`** | driver 开关：同时选中 ini 与 worktodo 的 `[Worker #N]` 段。**无段且无此开关时行为与旧版逐行一致**（向后兼容硬要求）。**GUI 的硬前提**：worker 可执行文件必须支持它（D1/D2 之后的构建）——旧构建会把 `--worker` 当位置参数、走单跑路径并打印 `No input number on stdin`，GUI 会诊断为 `DIAGNOSIS: … older than the D1/D2 driver changes`（`docs/DEV_ECM_GUI.md` §11.0）。 |
| **`ecm_gui`** | 原生图形前端（Dear ImGui docking + Win32/DX11），源码在 **`src/gui/`**（与 `src/core`、`src/cuda` 平级，不是 `tools/` 下的外置工具）。只做监管/展示/配置，**不含任何数论**（不链 GMP/OpenCL/CUDA）。 |
| **进度行** | 非 TTY（管道/重定向）下 driver 按衰减节奏打印的整行状态：CPU 侧 `stage1: [bar] 42.3% 42.3/100 (~1.20 s/curve) elapsed … ETA …`；GPU 侧 `GPU: [bar] 42.3% … +N bits (~1.20 s/curve) … remaining …`。GUI 靠它取百分比 / 速度 / ETA，进度条由 GUI 自绘。 |
| **事件行** | 日志面板真正关心的行：`START:`、`FACTOR FOUND`、`factor[i]=…`、`Checkpoint saved/loaded`、`Resuming from checkpoint`、`ERROR:` 与 `# ERROR <行>`。 |
| **results 双文件** | `results.json.txt`（追加式 JSONL，真相源，每命中一条对象）+ `results.txt`（同一因子被多条曲线命中时**合并成一行**，含 sigma 列表；从 JSONL 派生、可随时重建）。 |
| **`gpucurves` 推荐** | 由 `--gpu-info` 给出的逐档位 `ipb` 与 `sm_count` 算建议批量：生成器的公式是 **`curves = 块/SM × sm_count × ipb`**（块/SM 默认 2，面板可调），且不小于该档位的 `curves_min`（`blocks_min × ipb`）；`blocks_min` = 每 SM 一块，`curves_min` 就是内核警告里"raise -gpucurves to about N"的 N。**不要**填满 register-allowed 槽位（kernel 是 issue bound）。依据与实测见 `docs/DEV_ECM_WORKTODO.md` §5、生成器细节见 `docs/DEV_ECM_GUI.md` §19。 |
| **`--gpu-info`（D4）** | driver 的纯查询开关：打印设备（`device/name/sm_count/cc`）与**逐档位**的 `bits/tpb/tpi/ipb/blocks_per_sm/blocks_min/curves_min/blocks_wave/curves_wave`，`--bits N` 只留内核会选中的那一档。**零副作用**（不启动内核、不写文件、不建 ini），是"GUI/生成器只认一个真相"的接口。OpenCL 构建回答 `not_applicable` 并退出 0。格式与规则见 `docs/DEV_ECM_GUI.md` §11.1。 |
| **`progress_log_seconds`** | 新的 ini 键：**进度行写进 `log_file` 的最小间隔（秒）**；默认 60，`0` = 文件里不写进度行，负数 = 每行都写。**管道/控制台永远每次**（GUI 靠它显示进度），且 `100.0%` 那行总会写进文件。见 `docs/DEV_ECM_GUI.md` §7.2。 |
| **Prime95 交接（`worktodo.add`）** | 任务完成且 `.save` 同步后，driver 把该任务行**逐字节原样**（保留 AID 与已知因子）追加到 `p95_worktodo_path` 同目录的 `worktodo.add`；Prime95 自己并进 `worktodo.txt` 后删除该文件。段号由 `p95_add_workers` 决定（空/`3`/`1,3`/`1-8`/`auto`），段头**只认 `worktodo.txt` 里真实存在的段**。失败**绝不阻塞任务**：行落 `p95_add_pending.txt`，下次成功交付时一起送出；GUI 用红/黄/绿/灰通知条显示（红 = 有 pending）。见 `docs/DEV_ECM_GUI.md` §18。 |
| **生成器（M6 范围 A）** | `ecm_gui` 的"生成器"面板 + `src/gui/worktodo_gen.{h,cpp}`：粘贴 PrimeNet 作业（`ECM=`/`ECM2=`）→ 过滤/去重/改写/排序/存档名校验 → 逐行推荐曲线 → **预览** → **只追加**到 worktodo（追加前重新校验目标文件的大小/mtime）。与 `tools/ecm_worktodo/ecm.py` 在给定同样曲线数时**逐字节一致**。GUI 不含 GMP，位数用算术估算（已知因子按位数和扣减，取下界）。见 `docs/DEV_ECM_GUI.md` §19。 |
| **曲线控件（`draw_metric_plot`）** | `ecm_gui` 里**所有**折线图共用的一个控件（worker 的秒/曲线 + GPU 的占用率/功耗/SM 频率）：圆角渐变卡片 + 3 条网格线 + 面积 + 2 px 折线 + 末尾样本点 + "标题 + 当前值+单位"标题行 + 量程两端灰字 + 虚线参考线（功耗图 = 已执行的 NVML 上限）+ 悬停读数。高度**按字号缩放**（`height` 的单位是 15 px 字号下的像素），曲线带保底 24 px，保证"标题行 / 曲线带 / 最小最大值行"三条带互不重叠。worker 图纵轴是 **秒/曲线（`s/curve`，越小越快）**，不画进度 %。曲线自己会 trace（`plot: <id> … h= line_h= band_top= band_h=`）供脚本断言。见 `docs/DEV_ECM_GUI.md` §8.2。 |
| **`work_manager.ps1`** | **已废弃**：队列/日志/同步编排由 driver 内置队列管理器取代。仅可作测试夹具，不作为生产路径。 |

---

## 其他缩写

| 缩写 | 含义 |
|------|------|
| **CGBN** | CUDA Cooperative Groups BigNum，上游仓库的名称（仓库历史遗留，当前主线为 OpenCL 而非 CUDA）。 |
| **ISA** | Instruction Set Architecture。AMD GPU 汇编（`v_mad_u64_u32` 等）通过 RGA 工具闭环导出与验证。 |
| **ASM** | 以 `asm_` 前缀的算子变体，包含 GPU 特定内联汇编（如 `add_mod_asm_512b`）。AMD 专用。 |
| **GMP** | GNU Multiple Precision Arithmetic Library，Host 端多精度运算。 |
| **GP / Pari** | PARI/GP 计算机代数系统，用于 `--go` 模式计算椭圆曲线群阶以诊断"为何某 sigma 找到/找不到因子"。 |
