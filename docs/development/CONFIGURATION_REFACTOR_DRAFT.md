# ecm.toml 配置重构草案

## 现状与目标

`layout_version=8` 的配置定义有 156 个用户键，分布为 shared 6、Stage1 58、Stage2 69、GUI 19、worker 4。自动生成的完整默认模板约 87 KiB，常用选项与单阶段内核、调优、诊断及旧档导入参数共用这份模板。精简模板即可缩短日常配置。删键需要逐项核对运行行为。

现有结构中，`[stage1]` 的 `device` 与 `[stage2]` 的 `stage2_device` 风格不同，两个阶段的提交文件却都带阶段前缀。`impl` 实际仅供 CPU 使用，`stage1_threads` 只控制 CPU 线程，`gpu_param` 实际选择 GPU Montgomery 曲线参数化。Stage2 的三个 `*_mb` 都以 MiB 计，分别限制不同区域。GUI 的 `priority` 是进程优先级，共享 `priority` 是队列顺序。

现有 [配置定义](../../config/ecm_options.json) 继续作为键、默认值、范围和英中说明的单一来源。先打通版本识别、旧版转换和读写路径，再切换简明模板、专题示例及 GUI 写出。新版配置与 GWNUM 接入一并发布。配置版本独立于 ECM-ELF 程序版本和生成器的 `layout_version`。

## 文件版本与读取合同

- 在首个表头前写 `config_version = 2`。注释和空行可放在版本字段前。没有版本字段或显式写 `config_version = 1` 的文件都按 v1 读取。
- 读取时先定位根级版本字段，再按该版本识别表和键。重复声明、把版本写进业务表、非整数版本和未支持的版本都应给出路径和行号。版本 2 解析失败时直接报告错误。旧二进制回退使用原来没有版本字段的文件副本。
- 新程序读取版本 1 后，将已显式配置的项按版本映射转换到版本 2 的内部表示。保留原始位置、作用域和显式值来源，再按程序职责校验和投影。保持版本 1 的有效默认与继承语义，区分省略、显式空值、显式 `auto` 和显式写入默认值。若未来改变默认值，迁移须保留旧文件实际得到的值和来源语义。
- 读取旧配置自动完成内存迁移。落盘升级使用显式迁移入口，支持预览并输出新版文件。这样 Stage1、Stage2 和 GUI 可以共用同一旧文件，用户也可保留旧二进制所需的副本。GUI 编辑旧文件时按原版本保存，创建新文件或显式升级时写版本 2。

## 表与点号键

版本 2 的顶层业务表为 `shared`、`stage1`、`stage2`、`gui` 和 `worker`。模板使用 `[worker.N]` 单独呈现每个 worker，N 保持 1…1000000。旧的 `[workers.N]` 与 `[workers.N.stage1]` 等经迁移转为对应的版本 2 路径。每个 `[worker.N]` 下可设置启动选项，也可用 `shared.xxx`、`stage1.xxx` 和 `stage2.xxx` 分别覆盖全局设置。

在 `[stage1]` 或 `[stage2]` 中，常用键直接写，专用设置使用一个用途前缀，如 `handoff.enabled`、`opencl.workgroup_size`、`ntt.length`。worker 覆盖在前面加阶段，例如 `stage1.ntt.length`。模板和迁移工具统一输出点号键，按用途把相邻键放在一起。

TOML 将 `[stage1]` 下的 `handoff.enabled = true` 和单独使用 `[stage1.handoff]`、`enabled = true` 解析为同一结构。版本 2 允许这两种合法表示，内部按同一路径处理。若点号键已定义一个子表，再声明同一表头属于重复定义。把 `"handoff.enabled"` 整体加引号会形成一个含点号的字面键。解析时保留路径段和表定义身份，检查重复路径、重复表头及值与子表冲突。GUI 编辑已有文件时沿用该组已使用的合法表示，新建或显式规范化时写点号键。见 [TOML 1.0 表规则](https://toml.io/en/v1.0.0#table)。

## 默认模板与选项级别

在 schema 标注分组和可见级别，由同一生成器产出模板、参考和 GUI 元数据。

- 常用项放在新文件开头。推荐模板只包含队列、曲线与执行设备、Stage2 B2 和两个阶段的结果去向等常用入口，身份及覆盖示例用注释占位。把每项详细定义集中到逐键参考。
- 高级项按用途生成 CPU、NTT、Prime95 交接、Stage2 调优和显存示例。GUI 的高级展开可读写同一 schema。内核参数、预算和诊断继续接受显式配置。
- 旧项从新默认模板去掉。在参考末尾注明替代方案、兼容行为和移除条件。配置含旧项时，GUI 保留原行；迁移工具显示其实际作用和去向。
- 新模板应保留原先的有效默认。对带 `present` 的选项保留注释或省略，例如两阶段提交文件、Stage2 日志及 fold 预算。它们显式写入内置默认值后也可能改变 worker 文件名或环境覆盖行为。模板变更前用生成的旧默认与拟议简明模板比较投影结果，覆盖 worker 1 和 2、环境变量有无及 direct／queue 两种模式。

依据为 [模板生成器](../../tools/gen/generate_ecm_config.py) 的 `toml_template` 和 [配置与解析](../architecture/CONFIGURATION.md)。

## 拟议命名

分区承担阶段前缀，键承担实际含义，统一小写蛇形。相同作用的阶段键使用同名，例如 `[stage1] gpu_device` 和 `[stage2] gpu_device`。涉及 CUDA／OpenCL、CPU／GPU 的选项在名称或点号前缀中体现适用范围。时间用 `_seconds` 或 `_ms`，显存预算用 `_mib`。布尔值示例统一使用 `true`／`false`，可自动选择的策略使用具名枚举。旧值别名由相应迁移规则处理。

| 用途 | 当前配置 | 拟议配置和行为 |
| --- | --- | --- |
| 队列 | `shared.worktodo`、`priority`、`skip` | `shared.worktodo_file`、`queue_order = "file_order|stage1_first|stage2_first"`、`queue_skip = "none|stage1|stage2"`。排序与跳过仍各自生效。 |
| 结果身份 | `shared.user`、`computer` | 保留原名。这两个字段对应提交身份，模板仅含空值。 |
| 计算目标 | `stage1.target`、`curve`、`stage1_backend`、`device` | `stage1.target`、`curve`、`gpu_backend`、`gpu_device`。Stage2 同用 `gpu_backend`、`gpu_device`，设备序号仍由各 GPU 后端解释。 |
| CPU Stage1 | `impl`、`reduction`、`stage1_threads`、`affinity`、`cpu_mont_chain` | `stage1.cpu_implementation`、`cpu_simd_reduction`、`cpu_threads`、`cpu_affinity`、`cpu_montgomery_chain`。显式选择 GWNUM、IFMA、GMP 或 PRAC 时仍检查曲线、ISA 和后端能力。 |
| 曲线与标量 | `gpu_param`、`lcm`、`naf_w`、`exp_cache` | `stage1.gpu_montgomery_param`、`montgomery_lcm_multiplier`、`edwards_naf_window_bits`、`exponent_cache_dir`。`gpu_montgomery_param` 只管曲线参数化；`montgomery_lcm_multiplier` 仍接受 `"1"`、`"12"`，改变它会改变点乘标量。 |
| Stage1 GPU 算术 | `gpu_algorithm`、`tpi`、`wg_size`、`kernel_*`、`ntt_*` | 保留 `stage1.gpu_algorithm`。`[stage1]` 下使用 `opencl.threads_per_integer`、`opencl.workgroup_size` 与内核覆盖，以及 `ntt.convolution`、`ntt.length`、`ntt.bits_per_word`、`ntt.profile`、`ntt.tile` 等。现有 `tpi` 只供 OpenCL，CUDA 自行按位宽选择。 |
| 存档与输出 | `finished`、`tmp_dir`、`ckpt_seconds`、`ckpt_backups`、`stage1_results_file`、`stage1_submission_file`，以及同类 Stage2 项 | 各阶段内使用 `finished_file`、`results_file`、`submission_file`、`log_file`。Stage1 `tmp_dir` 改为 `save_dir`，继续承担其现有临时文件／检查点路径职责。Stage2 的 `save_dir` 保持输入 save 解析职责，`stage2_progress_dir` 改为 `checkpoint_dir`。Stage1 使用 `checkpoint_interval_seconds`、`checkpoint_backups`。迁移须保留路径来源及默认后缀。 |
| Stage1 备份与移交 | `save_sync_dir_1`、`save_sync_dir_2`、`sync_mode`、`auto_stage2*`、`p95_*` | `[stage1]` 下使用 `backup.directory_1`、`backup.directory_2`，`handoff.enabled`、`handoff.worktodo_file`、`handoff.workers`，以及 `prime95.worktodo_file`、`prime95.save_dir`、`prime95.workers`、`prime95.keep_aid`、`prime95.recover_lost`。备份路径与 Prime95 发布路径各自校验。 |
| Stage2 边界 | `stage2_b2`、`shared.b2_ratio`、`stage2_auto_b2`、`stage2_d` | `[stage2]` 使用 `b2`、`auto_b2`、`d`，比例保留在 `[shared]`。显式 B2、比例及 Auto B2 沿用现有优先级。 |
| Stage2 显存和并行 | `stage2_arena_mb`、`stage2_batch_mb`、`stage2_fold_mb`、`stage2_parallel`、`stage2_parallel_workspace` | `[stage2]` 下使用 `memory.arena_mib`、`memory.s4_batch_mib`、`memory.fold_resident_mib`、`cuda.curve_concurrency`、`cuda.workspace_policy`。并行当前默认串行，工作区当前默认自动。两组旧整数 0／1／2 各有独立映射。 |
| Stage2 选型 | `stage2_tune_runtime`、`stage2_tune_min_repeats`、`stage2_tune_runtime_samples`、`stage2_auto_carrier`、`stage2_opencl_*` 等 | `[stage2]` 下使用 `tune.runtime_enabled`、`tune.min_repeats`、`tune.runtime_sample_limit` 等。CUDA 承载及卷积参数使用 `cuda.*`，OpenCL 点策略和专属文件使用 `opencl.*`。读写共享 tune TOML 的两个显式路径先保持独立。 |
| Stage2 诊断 | `stage2_debug_log`、`stage2_debug_log_file`、`stage2_ntt_counts`、`stage2_factor_only`、`stage2_factorize_hits` | `[stage2]` 下使用 `diagnostics.debug_log`、`diagnostics.debug_log_file`、`diagnostics.ntt_length_counts`、`diagnostics.skip_prime_attribution` 和 `diagnostics.decompose_factors`。`diagnostics.skip_prime_attribution = true` 跳过命中素数归因，仍提取和验证因子。旧布尔值直接迁移。 |
| GUI 与 worker | `gui.NumWorkers`、`gui.exe`、`gui.stage2_exe`、`gui.priority`、`workers.N.autostart` | `[gui]` 下使用 `worker_count`、`stage1_exe`、`stage2_exe`、`process_priority`。`[worker.N]` 下的 `autostart` 改用布尔值。Stage1 读取 `worker_count` 作为自动路由后备，改名时同步 headless 路径。 |

上表是第一批建议映射，最终以逐键迁移清单为准。配置名的调整不改 `worktodo` 行、存档、检查点、结果记录及 tune 文件自身格式。

例如，队列内连续运行 Stage1 和 Stage2，worker 2 可单独覆盖设备编号。

```toml
config_version = 2

[shared]
worktodo_file = "worktodo.txt"
b2_ratio = 1000

[stage1]
curve = "mont"
target = "gpu"
gpu_device = 0
handoff.enabled = true

[stage2]
gpu_device = 0

[worker.2]
stage1.gpu_device = 1
stage2.gpu_device = 1
```

## 旧项与容易误合并的功能

- `p95_dir` 只输出提示，可以先移出模板。仍在使用的 Prime95 路径是 `p95_worktodo_path` 加 `p95_sync_dir`。依据为 [Stage1 main](../../src/main.cpp) 的 `p95cfg` 与 `p95_dir` 日志。
- `edwards_stage2_transfer`／`edwards_stage2_worker` 仍能按接收 worker 的有效 worktodo 路由，并只交接 Edwards。一般 `auto_stage2` 还能交接 Montgomery，取队列的规则不同。迁移前为旧 Edwards 路径补专用夹具，或让新交接配置能表达同样的曲线范围和目标路径。依据为 [Stage1 main](../../src/main.cpp) 的 `auto_stage2_routed` 和 `item.stage2`。
- `stage2_progress_file` 是旧单文件进度导入，显式路径还参与新检查点目录的默认推导。应保留兼容读取和新目录推导直到旧任务完成迁移。`stage2_cost_profile` 仍可让 Auto B2 读取旧 `.cprof`，且会影响 tune 文件回退及 pack 选项。它们可移出新模板，清理应分别验收。依据为 [Stage2 入口](../../src/ecm/stage2/cuda/main.cpp) 的 `legacy_progress`、`progress_dir` 和 `cost_profile` 分支。
- 现有 `b2_ratio` 参考声称直接读档也会计算比例。实际代码在入口用比例关闭 Auto B2，比例乘法目前见于队列任务解析。重构前应补 direct／queue 对照用例，确认后分别修正实现或文档。依据为 [Stage2 入口](../../src/ecm/stage2/cuda/main.cpp) 的 `o.auto_b2` 及 [Stage2 队列任务](../../src/ecm/stage2/task.cpp) 的 `cfg.b2_ratio`。
- Stage1 `ntt_tune_file=""` 关闭持久化，Stage2 `stage2_tune_profile=""` 可启用自动查找 `tune.toml`。Stage2 的 `stage1_tune_profile` 又是读取 Stage1 成本锚点的来源。适合共享一个默认文件名，不能直接视为三个同义键。
- GUI v2 已将窗口、布局等界面状态放进按配置身份区分的 sidecar。旧 GUI 仍实际消费业务 TOML 中的 `dock_layout`、`start_tab`、`font_snap` 等。先在 v2 编辑页和新模板降级展示。GPU 锁频授权、进程设置及 headless 路由计数仍属于业务配置。依据为 [GUI v2 的配置行为](../usage/GUI_V2.md) 和 [旧 GUI 入口](../../src/ecm/application/gui/app.cpp)。
- `workers.N.gpucurves` 的值在当前源代码中赋给 GUI worker 状态，但未找到从该成员到进程命令或队列曲线数的后续使用。下一步以 GUI 实际启动夹具核实后再废弃。独立 CLI 的 `-gpucurves` 和队列行内曲线数仍有消费者。

## 兼容实施顺序

1. 在 [配置解析器](../../src/io/config_format.h) 增加根级版本识别和新版路径解析，并保留原始行。现有读取器要求赋值在表内且键只能是裸键，不能直接消费新版示例。先用版本 1 夹具锁定现有行为，再增加版本 2 的表、点号键和冲突测试。
2. 将 `cli_entries()`／`selected_entries()` 的裸键合并改为按规范分区、字段路径和 worker 作用域识别。生成元数据将路径映射到稳定的字段 ID，消费端仍通过该 ID 读取。Stage2 继承、移交接收端及 GUI 预览用完整身份选字段。保持内置默认、全局、worker 和 CLI 的优先级。保留阶段隔离、仅按需校验 Stage2 继承字段，以及 Stage1 在应用覆盖前校验自身显式值的行为。涉及 [配置投影](../../src/io/config.cpp)、[GUI 编辑器](../../src/ecm/application/gui/toml_file.cpp)、[GUI 设置页](../../src/ecm/application/gui_v2/views/pages.cpp) 和独立 CPU Stage2。
3. 在 schema 集中维护版本 1 到 2 的键映射和旧值转换。每项声明旧路径、目标字段、旧类型和转换规则。同一作用域内有多个来源指向同一规范字段时，列出两个位置并报错。不同层级继续按既有优先级覆盖。保留外阶段或暂无法转换的值的原文和来源，按所属程序延后校验。显式导出时，无法在新版中安全保留的项应报告，并要求处理后再输出可用文件。
4. 提供迁移预览和新版文件导出。导出写入同目录的新文件，保留原文件、注释、显式空值及相对路径语义。GUI 普通保存沿用原版本，并保留未编辑行。
5. 确认 GUI、Stage1、CUDA／OpenCL Stage2 和独立 CPU Stage2 都能读两个版本后，切换默认模板、生成参考和新文件写出。验收同一发布包内所有消费者的兼容性，再与 GWNUM 变更一起发布。
6. 用 humanizer 和 humanizer-zh 校订英文与中文说明。保留键名、示例、默认值及兼容条件，删去重复解释。

## 验收

- 用冻结旧配置逐 worker 比较迁移前后的有效值和来源。覆盖 Stage1 与 Stage2 同名字段同时设置、同作用域重复、旧新拼写冲突、覆盖错误、空值和路径显式标记。
- 验证无版本文件和显式版本 1 等效。覆盖非法版本、未来版本、重复版本、表内版本，以及合法点号键和表头表示得到同一结构、混用重定义被拒绝、字面含点键不误当作路径。验证 `[worker.N]` 的阶段覆盖和旧 `[workers.N.stage1]` 等价。
- 在不同 TOML、可执行文件与 CWD 目录实测路径迁移。检查原有日志后缀、结果目标、指数缓存、Stage1 与 Stage2 移交、旧进度导入、Prime95 与备份文件分离。
- 使用无任务队列验证常规和移交启动。另测试 Stage2 比例、显式 B2、Auto B2、两种并行参数映射和 Stage1 的 GWNUM／PRAC 路由。保留阶段隔离和 GUI 无关键可编辑行为。
- 复用 [TOML 夹具](../../tools/test/test_ecm_toml.ps1)、[GUI worker 兼容矩阵](../../tools/test/gui_v2/test_worker_config_compatibility.py) 和 GUI 配置身份测试。验证读取不写回、导出重复执行结果一致、GUI 编辑点号键或子表后文件仍可读取。检查生成器的 `--check`，用原始旧文件验证旧程序仍可启动。

## 0.63.0 本地验证

- TOML 夹具通过，覆盖 611 项布尔别名检查和 v2 的版本、路径及重复定义。迁移工具 4 项测试通过，另用完整旧模板和导出文件对照 12 次实际 worker 启动，覆盖两个 worker、Stage1、OpenCL Stage2 和 CPU Edwards Stage2。
- 用 M127 对照 v1、v2 的 Stage1 存档和向 worker 3 的交接结果。CUDA／OpenCL Stage2 通过空队列启动和未来版本拒绝检查。GUI 通过点号键编辑、配置身份、原版本保存和模拟 worker 验证。
- `ecm.exe` 与 `ecm_cuda.exe` 均在 v1、v2 下通过 GWNUM 31.6 找到 M991 的因子 `8218291649`，保存点一致。
- 旧 `test_auto_stage2_routing.py` 要求完成记录保留相对 save 路径，0.62.15 与本轮均在此断言失败。实际程序写入绝对路径。新交接对拍使用明确的 save 路径。

原始证据保存在 `data/experiments/config_v2/`、`data/experiments/config_migration/` 和 `build_dev/gui_config_v2_probe/`。

## 下一轮配置重构计划（拟议）

### Stage1 到 Stage2 的交接

当前 `handoff.enabled` 的内置默认值为 `false`。设为 `true` 后，worktodo 中完成 Stage1、无因子且有可用存档的 Edwards 和 Montgomery 任务均可进入 ECM-ELF Stage2 队列。此时 `handoff.edwards_only_enabled` 和 `handoff.edwards_only_worker` 不参与选路。通用交接关闭时，Edwards 专用开关只投递 Edwards，接收 worker 为 0 时沿用当前编号。Prime95 交接由 `prime95.worktodo_file` 独立开启，因此与通用交接一起启用会让符合 Prime95 资格的任务投递两次。直接运行 Stage1 仍可产生 save，但这组开关针对 worktodo 交接。依据见 [Stage1 队列与完成处理](../../src/main.cpp)。

现行配置。全部交给 ECM-ELF 时，使用以下设置。可用 `handoff.worktodo_file` 或 `handoff.workers` 指定接收队列或 worker。选择的 Stage2 程序仍需支持该 save。

```toml
[stage1]
handoff.enabled = true
prime95.worktodo_file = ""
```

现行配置。Prime95 接收其支持的 Montgomery，ECM-ELF 接收 Edwards。路径按实际部署填写。

```toml
[stage1]
handoff.enabled = false
handoff.edwards_only_enabled = true
handoff.edwards_only_worker = 0
prime95.worktodo_file = "D:/Prime95/worktodo.txt"
prime95.save_dir = "D:/Prime95/saves"
```

Prime95 接收符合协议的 PARAM0/3，发布前还校验存档和 `sigma ≤ 2^63−1`。PARAM2 也被 Prime95 跳过。全部走 ECM-ELF 的生产队列应使用 Edwards 或 Montgomery PARAM0，启动能消费队列的 CUDA 或 OpenCL Stage2，并给 Edwards 提供有效 B2。独立 CPU Edwards Stage2 当前通过显式 save 路径运行。为 Prime95 生成的任务使用原任务的 B2。依据见 [Prime95 存档校验](../../src/io/save.cpp) 和 [Stage2 输入](../usage/STAGE2.md)。

下一版用 `[stage1] handoff.mode = "off" | "ecm-elf" | "prime95"` 选择唯一的首选接收方，默认 `ecm-elf`。`prime95` 模式下，`handoff.unsupported = "ecm-elf" | "keep-save"` 决定协议不支持时的行为，建议默认 `"ecm-elf"`。Edwards 可因此走本地 Stage2。两侧均无兼容后端时保留 save 并明确报告未交接。`handoff.worktodo_file`、`handoff.workers` 控制本地投递，`prime95.*` 控制外部投递。Prime95 缺少必需路径、文件写入失败或投递状态不明时按原协议报错或保留待处理记录。只有明确的协议不支持才进入后备路由。

在应用 save 前缀后判断构造，发布时按实际批次校验协议资格。每个新批次只生成一个接收方的投递意图，继续按旧意图记录的目标恢复存量任务。旧版读取兼容原来的双投递和 Edwards 专用路由。显式导出新版时，双投递要求指定目标，Edwards 专用跨 worker 路由须保留接收方的有效 worktodo。若新版不能表达某个旧路由，迁移报告具体来源并停在预览。

### Stage2 的 B2 配置

将 `[shared] b2_ratio` 移到 `[stage2] b2_ratio`，worker 对应使用 `stage2.b2_ratio`。CUDA 和 OpenCL 共用的任务入口在读取 save 后按第一条选中记录的 B1 计算 `B2 = B1 × b2_ratio`。当前实现对队列和直接读档均生效，非零时覆盖 CLI、任务行和 TOML 的 B2，并关闭 CUDA Auto B2。队列要求选中记录的 B1 一致。此次迁移保留计算优先级和溢出检查，同步纠正逐键说明、Stage2 文档及 GUI 预览。独立 CPU Edwards Stage2 当前没有 ratio 入口，是否加入由后续任务决定。依据见 [Stage2 任务](../../src/ecm/stage2/task.cpp#L1123) 和 [Stage2 启动](../../src/ecm/stage2/cuda/main.cpp#L532)。

### Stage1 曲线构造

在 `[stage1]` 用 `param = "e" | "m0" | "m2" | "m3"` 替代 `curve` 与 `gpu_montgomery_param`，值与 save、checkpoint 的构造前缀一致。`target` 继续选择 CPU 或 GPU。省略 `param` 时沿用现有 CPU→`m0`、GPU→`m3` 的有效默认。CPU 只支持 `e` 和 `m0`，OpenCL Stage1 只支持 `m3`，CUDA NTT 支持 `e` 和 `m0`。CUDA CGBN 按实际编入的参数化验收。`ECMSTAGE1` 的显式 save 前缀继续为该任务指定构造。

新增 CLI `--param`，保留 `--curve`、`--gpu-param` 和旧 sigma 写法。新选项与 `-sigma <param>:<seed>` 冲突时报出两个来源。旧版运行时按 worker 和 CLI 应用 `target`、`curve` 和旧 param，再得到有效构造。落盘迁移时保持省略值仍可随 `--target` 变化。若旧配置为同一个 worker 显式指定了只能用于 GPU 的构造，而用户还需要用 `--target cpu` 切换到 PARAM0，迁移预览应提示改用 `--target cpu --param m0`。验收单键 worker 覆盖、CPU 忽略旧 GPU 参数和 Edwards 固定标量。内部仍保存独立的曲线族、Montgomery 参数和 sigma，供构造、存档、结果及 checkpoint 校验。依据见 [Stage1 输入归一化](../../src/io/normalize.cpp#L491)。

### 统一 tune 文件路径

用 `[shared] tune_file` 提供 CUDA 两阶段的共同路径。`[worker.N] shared.tune_file` 为同一 worker 的两个阶段一并覆盖。v3 的 TOML 中阶段级路径报错。相对路径以配置目录为基准，默认使用该目录的 `tune.toml`。旧配置同一 worker 的两个阶段若解析到不同文件，迁移需要选定共用路径。

CUDA 两阶段已经可以在一个便携格式文件内读写各自记录。v1/v2 中，Stage1 旧空路径关闭持久化，Stage2 旧空路径允许默认查找 `tune.toml` 或 `stage2_ecm_tune.toml`。v3 显式空字符串为两阶段关闭自动持久化。迁移检查空值、旧成本档案和同目录回退。Stage2 CLI `--tune-profile` 的读取语义与 `--tune-file` 的调优输出语义继续兼容。

`ntt.profile` 是变换档位。OpenCL 点计划使用不同文件格式，可统一命名为 `opencl.tune_file`。Auto B2 用于估算完整 Stage1 成本的 `tune.stage1_profile` 是独立数据，可改名 `tune.stage1_cost_file`。它们继续独立于 CUDA 共用文件。验收两阶段并发更新同一文件、写入失败后的可读性及对方记录保留。依据见 [Stage1 路径投影](../../src/io/normalize.cpp#L240)、[Stage2 路径选择](../../src/ecm/stage2/cuda/main.cpp#L548) 和 [Stage1 成本文件读取](../../src/ecm/stage2/planning/stage1_tune_profile.h)。

### 按位宽选择 CPU Stage1 后端

`cpu_implementation` 保留单值写法 `"auto"`、`"simd"`、`"scalar"`、`"gwnum"`，并接受由逗号分隔的切片规则。空字符串 `""` 等价于 `"auto"`。按实际算术模数位宽匹配，位宽在剥除已知因子后计算。

```toml
[stage1]
cpu_implementation = "simd[:6144], scalar[:4096], gwnum[4096:], auto[]"
```

- `backend[start:stop]` 采用左闭右开的 Python 切片。省略 `start` 时从最小有效位宽 1 开始，省略 `stop` 时无上限。`backend[]`、`backend[:]` 和全角冒号写法 `backend[：]` 等价。解析时忽略规则内的空白。边界使用正十进制整数，反向或空区间、未知后端、非法分隔符、溢出及步长语法报错。
- 允许重叠。按书写顺序检查区间和后端可用性，两项均满足才选择该后端。小于 4096 位时先试 SIMD，再试标量。4096 至 6143 位先试 SIMD，再试 GWNUM。6144 位及以上先试 GWNUM。若前面的候选都未选中，示例最后显式使用内置 `auto`。列表耗尽仍无可用后端时，报告实际位宽、曲线和候选被跳过的原因。空字符串继续等价于单值 `auto`。
- 可用性检查涵盖曲线、ISA、DLL、位宽及 PRAC/归约要求。GWNUM 要求 Montgomery PARAM0 和可用 DLL。候选在计算开始和写入状态前检查，选定后遇到计算、I/O 或 checkpoint 错误按正常错误路径处理。区间策略只供 CPU 路径，现有单值强制选择和 `--impl` 优先级保持兼容。

启动时解析切片。任务归一化后取得实际曲线、模数位宽和批量，调用统一选择逻辑，再核对已选后端和 checkpoint。日志记录实际位宽、跳过原因、命中规则和最终后端。验收覆盖 4095/4096、6143/6144、10240/10241、重叠顺序、省略边界、空字符串和空白、因子剥离后跨区间、无 IFMA 时回退、无 DLL、Edwards 跳过 GWNUM、列表耗尽和续跑状态。依据见 [CPU 后端选择](../../src/ecm/stage1/selection.cpp#L40) 和 [组合校验](../../src/io/normalize.cpp#L300)。

### 共享日志与完成记录

将 `log_file`、`finished_file` 移到 `[shared]`。所有 worker 和两个阶段使用同一 `log_file`，建议默认 `ecm.log`。每行以 `[Worker #3] [2026-10-17 06:37:11]` 形式标识来源，进程启动记录标明阶段。共享 `finished_file` 建议默认 `worktodo.finished.txt`，按原有 `ECMSTAGE1=`、`ECMSTAGE2=` 记录区分阶段。新版共享相对路径基于 TOML 目录，显式空日志路径继续用于关闭文件日志。

在 `[shared]` 增加 `log_file_max_KB = 10000`，类型 `u64`，建议范围 `[1000..1000000]`，以 `1 KB = 1000 bytes` 计。一次完整日志记录写完后超过上限时，将活动文件 `ecm.log` 归档为 `ecm.N.log`，下一条记录继续写入 `ecm.log`。

滚动时在活动文件同目录扫描与其文件名和扩展名对应的数字编号归档，取最大编号加 1。目录没有归档时从 1 开始，例如已有 `ecm.2.log` 和 `ecm.9.log` 时生成 `ecm.10.log`。扫描、改名和写入由同一个活动日志路径的跨进程锁协调。改名使用禁止覆盖的操作，遇到同名竞争时重新取号。归档失败时保留活动日志并报告原因。验收覆盖已有编号空洞、重启、两个进程同时触发、旧编号无覆盖、超长记录和归档失败。

两阶段统一日志追加规则，GUI 读取共享日志时按行首 worker 编号识别归属。共享完成记录在放开当前跨阶段同路径检查前，需让两阶段在同一文件锁内完成查重和写入，并保留 Stage2 任务标记、错误回执和重放规则。继续阻止完成记录与活动队列、日志、结果文件和检查点路径冲突。验收同一 worker 从 Stage1 切换到 Stage2、多个 worker 同时追加、不同程序目录下的路径解析和异常后重放。依据见 [Stage1 路径检查](../../src/main.cpp#L597)、[Stage2 路径检查](../../src/ecm/stage2/task.cpp#L1088)、[Stage1 完成回执](../../src/io/worktodo.cpp#L1056) 和 [Stage2 追加写入](../../src/ecm/stage2/input.cpp#L219)。

v1/v2 继续按各阶段原有路径基准读取。显式迁移先比较各 worker 的有效路径。两阶段路径一致时合成共享项，路径分开时在预览中列出差异及待选共享目标。旧日志和完成记录原地保留，已有回执继续指向原目标。

### 其他整理与实施顺序

1. 建立配置 v3 到内部设置的映射。读取无版本、v1 和 v2 时维持旧优先级和显式值语义。迁移工具预览列出合并、冲突和无法等价映射的项目。保存点、检查点、worktodo 和 tune 数据文件沿用各自现有格式。
2. 先实施单一交接路由和 `b2_ratio` 移位，再接入共享日志、日志滚动和完成记录。用本地队列和离线 Prime95 drop box 验证 Edwards、PARAM0/2/3、宽 sigma、有因子、无可用 save、空队列恢复和发布失败。检查直接读档、队列、worker 和 GUI 的 B2，以及共享路径的并发追加、归档编号和回执重放。
3. 合并 `param` 与 `tune_file`，再接入 CPU 位宽策略。用实际 worker 对照新旧配置，覆盖可选的阶段覆盖、配置目录与程序目录分离、并发 tune 更新和断点恢复。
4. 将 Stage2 的 `baby_backend`、`seed_backend`、`giant_backend`、`reduce_backend` 和点计划相关键按实际 CUDA、OpenCL 消费者归位。现有选型时序和各自的强制覆盖继续独立。`prime95.legacy_directory` 只输出提示，可从新版配置移除。`worker.gpucurves` 先验证是否进入启动命令，确认无消费者后再移除。GUI 的窗口和布局留在 sidecar，旧 GUI 设置由兼容层处理。
5. 保持两阶段的 `gpu_device`、`gpu_backend`、`save_dir` 和结果路径可独立设置。统一控制台日志级别、结果路径和显式覆盖的命名规则，保留阶段各自的默认值和路径基准。旧进度导入、Prime95 回执和备份继续按各自的路径和读写规则处理。
6. 同步 GUI、CLI 帮助、双语逐键参考和两份模板。在隔离目录使用生产配置副本验收。配置文件迁移和新旧 worker 共存测试通过后再安排上线。

### v3 本地验收

- 实际 M127 从 CPU Stage1 交接到 GPU1 OpenCL Stage2，两个阶段向共用完成文件各写一条记录，队列清空。共享日志保留 worker 标记并按最大编号续写。双进程滚动测试验证 48 条记录无遗漏或重复。
- 离线队列通过 10 个交接场景，包括原位与跨队列、Prime95 单路投递、协议回退、旧意图恢复及目标路径冲突。v1/v2 使用相同任务对拍，存档与交接结果一致。
- CPU Stage1 实际验证 7 个后端策略场景，包括 IFMA、GWNUM 可用和缺失、Edwards 跳过 GWNUM、耗尽及 `--impl` 覆盖。CUDA Stage2 直接读档时，worker 的 `b2_ratio` 将 B2 覆盖为 `100 × 40 = 4000`。
- 在新建的 CUDA Stage1 构建中，GPU1 分别用 `param = "m0"` 和 `param = "e"` 完成 M127，存档和共享日志均已生成。GUI 新建 v3、保留旧版编辑、模拟 worker 和配置身份验证通过。旧构建目录增量重链曾出现运行时 ABI 不一致，本轮 CUDA Stage1 实测使用独立的干净构建。
- 生产配置仅做只读预览，旧日志路径存在分歧且两种交接同时开启。迁移要求明确选择共享输出目标和交接模式。分别指定 ECM-ELF 或 Prime95 后均可生成预览。
