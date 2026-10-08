# CUDA Stage2 发布候选使用说明

本轮准备 Windows `sm_89` 单架构候选：`ecm_cuda_stage2.exe`、`gmp-10.dll`、共用配置模板和说明。多架构包及两个可执行文件的合并留待后续。当前算术组合尚无匹配的新 Auto B2 标定数据，使用显式 B2。本轮只做构建，未运行队列、中断或 GPU 回归测试，不代表已完成发布验收。

## 1. 开始使用

将归一化的 Suyama PARAM=0 Stage1 文本 save 放入程序目录，或在 INI 的 `tmp_dir` / `stage2_save_dir` 指定目录。按机器选择 `device`，它是 CUDA 设备编号，与编译架构不同。这个包的配置缺省设备为 0，本轮没有启动 GPU 运算。

直接执行明确的记录范围：

```powershell
.\ecm_cuda_stage2.exe --save D:\saves\m3701_260e6.save --b2 26000000000 --skip-curves 960 --curves 10
```

直接 `--save` 每次按指定范围执行，不读取或更新队列续跑进度。记录从 1 编号，上例选择 961–970；数量不足时只执行可用记录并警告。`--curves 0` 表示所有剩余记录。

队列使用独立文件 `stage2_worktodo.txt`：

```text
ECMSTAGE2=1,2,3701,-1,"m3701_260e6.save",26000000000,960,10
```

```powershell
.\ecm_cuda_stage2.exe --dry-run
.\ecm_cuda_stage2.exe
.\ecm_cuda_stage2.exe --once
```

`--dry-run` 检查第一项，不执行曲线、不写结果或日志、不修改队列和进度。`--once` 只处理一个任务，包含该任务的全部可用曲线；如该项输入有误，标记后退出。通常不指定 `--once`，连续消费队列直到为空或用户停止。

## 2. 共用 ecm.ini 与迁移

默认读取 exe 同目录 `ecm.ini`。模板为 `ecm.ini.example`；已有 INI 不会被打包脚本覆盖。隐式 INI 不存在使用内置默认，显式指定但不可读则报错。

全部键的默认值、可选范围、空值语义及用途见 [配置说明](ECM_INI_REFERENCE.md)，采用 `key=<domain>; default=<default>` 配置行加逐项英中双语说明，先英文、后中文，包括 Stage1、Stage2、GUI 和旧别名。说明采用 undoc 式用途与取值效果描述，直接解释如何调整，长行自动折分。仓库维护时修改 `config/ecm_options.json` 后运行生成器；流程见 [统一配置维护](DEV_ECM_CONFIG_SCHEMA.md)。用户自己的 `ecm.ini` 仍可直接编辑。

Stage1 继续使用 `worktodo`、`finished`、`log_file`。Stage2 不再继承这三个键，改用：

- `stage2_worktodo`：默认 `stage2_worktodo.txt`。
- `stage2_finished`：默认 `stage2_worktodo.finished.txt`。
- `stage2_log_file`：默认 `stage2_screen.log`；留空禁用普通文件日志，控制台仍显示可读输出。
- `stage2_results_file`：默认 `stage2_results.jsonl`，成功曲线的完整结果。
- `stage2_progress_file`：默认为队列路径加 `[_worker].progress`；不应禁用或与其他输出共用。
- `stage2_debug_log`：默认 false；显式开启调试细节。
- `stage2_debug_log_file`：默认 `stage2_debug.log`，独立于普通运行日志。
- `stage2_save_dir`：空值继承共享 `tmp_dir`；`stage2_device` 缺省继承共享 `device`。

未显式配置普通日志、结果或调试日志路径时，worker 2 等自动使用 `_2` 后缀；显式配置的文件名按原值使用，因此多 worker 应在各自段中设置不同文件名。每个队列同时只支持一个消费进程；并行 worker 使用不同队列。

迁移旧 Stage2 专用 INI 时，将旧 `worktodo/finished/log_file` 值复制到对应 `stage2_*` 键。不能将一个活动队列同时交给 Stage1 和 Stage2。运行时拒绝 Stage2 队列/finished 与已解析的 Stage1 对应路径冲突，也拒绝配置、程序、源 save、队列、进度、结果和日志之间的危险路径重叠。

Stage2 INI 相对路径以 INI 目录为基准，CLI 相对路径以当前工作目录为基准；Stage1 保持原有 exe 目录规则。推荐将共用 INI 放在两个 exe 同目录，避免外部 INI 的不同路径基准产生歧义。

普通 `[queue]`、`[gpu]`、`[stage2]` 标题仅供阅读，不能隔离同名键。只有 `[Worker #N]` 定义覆盖范围。Stage1 键名保持原有大小写规则；Stage2 专用解析器不区分键名大小写。CLI > worker 专用键 > 全局键 > 内置默认；B2 为 CLI 非零值 > 任务非零值 > `stage2_b2`。只有未指定非零 B2 时才考虑 Auto B2。

上述普通标题规则仅适用于命令行驱动，GUI 把所有标题当作真实分区。发布模板使用注释分组，保证公共键也能被 GUI 读取；公共键应放在任何分区之前。

## 3. 数量不足与任务错误

请求 10 条、跳过后仅有 3 条时，提示 requested=10/available=3，执行这 3 条。全部成功后记录 completed=3，将原任务行移至 finished 并继续下一项。没有可用记录时提示并按零曲线完成；不等待 save 追加新记录。可用数指本次选择范围内的记录数，最多为请求数，不是整个文件的总记录数。

存档缺失、文本损坏、校验和不匹配、N 不匹配、不可用 X 或任务字段错误：记录原因，将原队列行变为 `# ERROR ...` 注释，在 finished 留下错误原因与原任务，继续下一项。修复后可手动恢复该任务行。文件权限/读写故障、进度状态不一致、CUDA/显存/设备故障及未知内部故障停止进程，保留未完成任务。队列存在输入错误时，即使后续项成功，最终退出码为 1；致命错误为 2，正常完成为 0。

整项所选记录先验证，发现损坏不会将其当作“数量不足”略过。源 save 从不修改。

## 4. 逐曲线续跑与退出

仅队列自动续跑，保存完成曲线数与当前尝试的随机回执，不保存 Stage2 中间计算状态。

1. 启动曲线前，将尝试回执原子写入进度文件。
2. 子进程完成算术检查和因子验证，追加结果及回执，刷新到磁盘。
3. 父进程收到成功退出并确认回执后，原子更新完成曲线数。
4. 如果在 2 与 3 之间中断，重启扫描结果回执，补记完成，避免重复执行这条曲线。

进度绑定任务、队列内容、save 完整 SHA256、选择范围、结果路径及相关计算配置。续跑期间不要编辑队列、追加/替换 save 或改变计算配置。变化时拒绝沿用未完成进度；应恢复原输入，或明确归档进度文件后重新执行。保留该限制可避免同名存档被替换、相邻重复任务及参数变化导致错误跳过。

任务完成时，finished 先写带随机任务 ID 的回执与原行，再推进队列，最后清除进度。重启可识别已写入的 finished 回执。结果文件中未以换行结束的尾行视为可能的中断写入，不用作成功回执。新追加会与残缺尾行分隔。

第一次 Ctrl+C：提示等待当前曲线完成，结果与进度落盘后退出；不再领取新曲线。第二次 Ctrl+C：立即终止子进程及其子进程，保留已有进度；当前曲线没有完整成功回执时下次从头重算。关闭窗口或杀进程同样依赖完成回执恢复，不承诺当前曲线完成。进度和结果不要单独删除其中一份。

## 5. 控制台、普通日志与调试日志

`verbose=true` 默认显示任务摘要、曲线编号、阶段切换、时长和结果；false 保留曲线摘要。长阶段每 30 秒刷新一行运行状态。这是父进程仍在等待的提示，不是 GPU kernel 完成百分比。

阶段包括存档验证、选形与内存规划、baby 点、F 树、Newton 逆、giant/G 树/fold、归一化与下降、块/叶 GCD、最终验证及写入结果。阶段提示的 `previous` 是上一个阶段的墙钟间隔，`elapsed` 是该曲线累计时间；giant/G 树/fold 在批次循环中交织执行，控制台合并显示，详细分项保留在文件。

普通日志追加保存阶段、批次及结束统计；警告和错误也显示在控制台。开启 `stage2_debug_log=true` 后，调试细节写入独立文件；它不会因为 `verbose=true` 自动开启。共用 `debug_log` 可作为兼容开关，Stage2 专用键优先。旧 `stage2_log_level` / `--log-level` 继续作为控制台等级覆盖；旧 debug 等级映射为开启独立调试日志，普通输出最高保持 batches。

普通日志及结果默认不轮转，长期运行由用户归档。子进程的 CUDA 上下文与临时内存在每条曲线结束后释放。

### 5.1 选形和内存摘要

默认 `verbose=true` / `phases` 在曲线开始时显示四行选形和内存规划，统计完成后再显示一行实际占用。`curve` / `quiet` 不增加这些摘要。下面使用所提供日志的数值；末行假设 INI 为 `stage2_arena_mb=0`、`stage2_fold_mb=640`，这两个配置值不能仅从旧日志反推。

```text
real_shape: B1=260000000 B2=2600000000000 S_bits=3275
real_shape: D=1711710 P=155520 baby=155520 giant=1518950 poly_g=10 loops=9
mem: free=7106 MB total=8188 MB reserve=768 MB arena_cap=6338 MB
mem: P=155520 -> largest transform (155521 coeffs) = 3074 MB, tree top (77761 coeffs) = 1537 MB
...
mem: batch=64 arena=3187.815/0 fold=431.897/640
```

这里的 `MB` 沿用引擎标签，实际均为 **MiB = 2^20 bytes**。`giant` 为 giant 点数量，`poly_g` 为 G 树批数，`loops` 为首批之后的 fold 次数；`S_bits` 为实际参与计算的 N 的位宽，去掉已知因子后可小于 worktodo 中的指数。

- `batch`：实际生效的 S4 批次工作缓冲预算，可能来自命令行覆盖；不是已经分配的缓冲大小。
- `arena=a/b`：`a` 为该曲线 NTT arena 的 `full_peak_bytes / 2^20`，保留三位小数；`b` 为当前 worker 合并后的 INI `stage2_arena_mb` 值，未配置时使用内置默认。命令行、环境变量和 Auto B2 不覆盖此处的 `b`，所以 `/0` 表示 INI 要求自动预算，不表示实际占用为零。
- `fold=a/b`：`a` 为实际启用的 fold owner 的 `peak_bytes / 2^20`；回退或未启用为 0。`b` 为 INI `stage2_fold_mb` 值，同样保留配置口径。缺失统计时显示 `n/a`，不会把未知占用写成 0。
- `free/total`：规划时 `cudaMemGetInfo` 的设备全局快照，不是本进程独占的显存统计。`reserve` 是规划预留余量，不是单独分配的一块显存；本例自动 arena 预算为 `7106-768=6338 MiB`。
- `largest transform/tree top`：沿用引擎的单个乘法形状工作缓冲估算，包含三大 NTT 数组与输出槽，不等于 arena 最终峰值，也不是两块始终同时存在的分配。

**`arena_mb` 确实属于 NTT 显存池。**池内包括复用的 A/B/Q 大工作区、输出/小缓冲、NTT 表和 fuse 基础缓冲；不包含独立的 fold owner、giant 坐标、其他树缓冲、全部下降元数据或 CUDA 上下文。`batched_progress.arena_mb` 是当时池内 CUDA payload 的 MiB 值，显示时截去小数；新的 `arena=a/b` 取整条曲线的完整池峰值。发生 `arena_overflow` 时，按次分配的回退缓冲也不属于池内 payload。INI `stage2_arena_mb` 是这个池的预算，不能当成进程总显存上限。

本例池峰值为 `3342666480 bytes = 3187.815 MiB`，由大缓冲 `3072.000`、小缓冲 `5.678`、表 `73.958`、fuse 基础缓冲 `36.179 MiB` 构成。这组分量在此记录中可对应同一次 `full_peak`；其他运行不能把各模块各自的峰值直接相加。fold owner 的 `431.897 MiB` 另计，下降还额外使用 `3.560 MiB` 元数据；`fold` 摘要只报告 owner，与 `real_batched_folddevice` 一致。

实现仅在父进程控制台投影中拆分原有 `real_shape` / `mem_budget` 行，并从原有统计生成结束摘要。普通文件日志仍先写入原始行，调试日志、JSONL、进度回执及其落盘规则均保留。没有额外 GPU 查询、同步或测量 kernel；`batches` 控制台使用简化选形行，其他原有详细统计继续显示。

### 5.2 阅读一次完整曲线日志

以下解释用户提供的 `record=20, sigma=5098253821139211` 单条曲线，B1 为 2.6 亿，B2 为 2.6 万亿。所有时长与占比仅描述这次运行，不是其他设备或规模的性能预测。

1. **领取并读取存档。**`curve_start: 20/960 record=20` 表示本任务第 20 条曲线，对应 save 的第 20 条记录；每条曲线在独立子进程运行。父进程已选取并验证记录，子进程再核对记录指纹。`checksum=verified` 表示存档校验和通过；`normalized_Z=1 stage1_skipped=1` 表示直接使用存档中的普通仿射 X，不重新计算 Stage1。校验和不等于重新证明 Stage1 计算正确。

2. **选择多项式形状和内存预算，约 0.34 秒。**实际余因子为 3275 bits，即 `W=ceil(3275/64)=52` 个 64-bit limb。自动选得 `D=1711710`，`P=phi(D)/2=155520`。baby 索引取 `1<=j<=D/2` 中与 D 互素的 j；giant 上界为 `I=floor(B2/D)+2=1518950`。因此 `ceil(I/P)=10` 个 G 批次，前九批每批 155520 点，末批 119270 点，首批之后执行 9 次 fold。`D` 是步长，`P` 是多项式规模，都不是 NTT 长度。

3. **生成 baby 点并归一化，约 4.68 秒，占整条曲线墙钟 12.1%。**从 Stage1 点 Q 计算 `[j]Q`，得到 baby 的 x 坐标；日志分项为 ladder 4.517 秒、affine 0.158 秒。`degenerate=0` 表示本阶段没有通过不可逆 Z 的 GCD 提前发现因子。两项只是阶段内分项，不应再加到 4.68 秒之上。

4. **构造 baby 的 F 乘积树，约 3.40 秒，占 8.8%。**在模 N 系数环中构造 `F(X)=prod_j(X-x([j]Q))`。有效叶子为 155520，布局补到 `2^18=262144`，有效乘法数为 `155520-1=155519`。NTT 做整数多项式卷积，再由 S4 归约到模 N 系数；不是浮点 FFT。`ntt_seconds=2.252` 是树阶段内部的 NTT 统计，已包含在这个阶段中。

5. **准备 Newton 多项式逆，约 2.45 秒，占 6.3%。**计算反转后的单首 F 的幂级数逆，长度为 `P+1`，供后面所有多项式除 F 取余复用。这里的逆是多项式逆，不是重新求椭圆曲线点；对应精细统计 `inv=2.398` 秒。

6. **giant 点、10 棵 G 树和 9 次 fold，约 22.00 秒，占 56.8%。**使用 seed + xADD chain 分 4 个坐标 chunk 生成 `[iD]Q`；按 P 个点组织 G 树。G 叶子允许保留射影缩放，以减少逐点归一化和搬运。数学上各批表示 `G_k(X)=prod_i(X-x([iD]Q))`；首批初始化 H，其余批次做 `H <- H*G_k mod F`，最终还原累计的可逆缩放因子。精细分项为 giant 1.876 秒、G 树 15.200 秒、fold 4.768 秒；实际 owner 启用且无回退，9 次 fold 对应 27 次完整多项式乘法。进度从 `2/10` 开始是因为首批只初始化 H，不走后面的 fold 进度打印点，并非跳过第一批。

7. **还原缩放并沿 F 树下降，约 5.40 秒，占 14.0%。**Gamma 的设备修正用时 0.017 秒，设备 root 准备用时 0.172 秒；二者是阶段内分项。利用已有 F 逆与缩放下降，在 baby 点上得到等价于 H 求值的叶结果。`scaled_root_device=1`、`scaled_frontier_device=1` 表示相应路径实际启用，`root_inverse_reused=1 root_divisions=0` 表示复用了根逆并避免再次做根除法。18 层下降包含 32 个逻辑乘法批次、311039 对乘法；这些不是 311039 次独立 CUDA kernel 发射。下降上传 F 兄弟多项式共约 1211.335 MiB，最终叶结果回读约 61.699 MiB；`avoided_*` 是相对旧路径省掉的字节估算，不是本次实际传输量。

8. **叶子乘积和 GCD，约 0.13 秒，占 0.3%。**将 155520 个叶结果按 64 个一组累积，共 2430 个块，求块乘积与 N 的 GCD。若命中，再定位块内叶子并可选查找关联素数。本次 `hits=0`，不需要进入命中定位或命名，`name=0`。这说明此曲线没有找到因子，不说明 N 是素数。

9. **完成算术检查、验证因子并写结果，约 0.17 秒。**异步 GMP oracle 在报告成功前被 drain；设备归约自测 2304 个用例、运行中检查 52264 个系数，错误计数均为 0。`full_checks=1` 是某个足够小批次被完整检查的次数，不是整条曲线所有系数都被 GMP 重算。其他优化统计中的 `checked_words=0` 只说明相应额外对照没有开启，不代表全局未做算术检查。没有因子时可选因子分解几乎立即返回，故 `factorization_complete=true` 可以与空因子数组同时出现。

10. **发布与续跑。**子进程先追加结果 JSONL 及成功回执、刷新磁盘，再报告成功；父进程确认回执后更新队列进度，随后领取下一曲线。`curve_done` 表示这条曲线已完成；不是整个 960 曲线任务完成。

#### 计时与占用应如何读

- `previous` 记在下一阶段的开始处。例如 `Build baby-point F tree previous=4.68` 的 4.68 秒属于刚结束的 baby 阶段，F 树本身的 3.40 秒出现在下一条 `Prepare polynomial inverse` 上。
- 整条曲线父进程墙钟为 `38.6996 s`；子进程 `stage2_result.seconds=38.5719 s`；引擎 `stage2_full_wall.total=38.139636 s`。引擎 total 为 `init 8.149359 + main 29.990277`，其中 `shape=0.038077` 已计入 init，不能再次相加。D 扫描、调用前准备、进程和发布等有不同的计时边界。
- `stage2.elapsed=29.99 s` 仅为 main，不能当成完整曲线耗时。`f_tree_incl=8.043 s` 同时包含 baby 和 F 树，也不能与二者重复相加。
- 这次最大模块为 G 树 `15.200 s`，约占完整墙钟 **39.3%**。`t_reduce=4.523 s` 是 CUDA event 测得的设备归约时间，约占 11.7%；`t_reduce_host=9.566 s` 是提交归约的主机路径用时，可包含事件回收与等待，已嵌套在多项式运算中，不能与前者或阶段时间直接相加。`ring_waits=547` 也不能换算成 GPU 空闲时长。
- `ntt_calls` / `poly_muls` 主要数逻辑多项式乘法；`ntt_launches` 数对应批处理入口。单次入口内部还有多轮 NTT、pack、归约等 kernel，不能用这些计数替代全部 kernel 数。
- `host private=6545 MB peak=6827 MB` 是进程主机内存。`device free=2872 MB of 8188 MB` 是下降开始时设备全局余量，不是该进程峰值；arena、fold 与其他模块的独立峰值也不能相加得出进程显存峰值。这份日志没有 GPU busy 时间线，不能据此把剩余时间完全归因于 CPU、传输或 NTT。

## 6. 仓库构建与打包

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_stage2_release.ps1 `
  -Arch sm_89 -SplitCompile 6
```

默认构建目录 `build_cuda_cmake/release_ux_sm89`，打包目录 `dist/cuda-stage2-sm89`。脚本验证 production 算术选择、源文件哈希及 exe 哈希，再复制运行时、模板、说明和构建/包清单。`-SkipBuild` 只用于打包已构建且源码哈希仍匹配的产物。

发布默认使用 vcpkg `x64-windows` GMP，并从同一 prefix 获取头文件、导入库和 DLL，附带 `GMP-COPYRIGHT.txt`。`-Gmp <prefix>` 可显式覆盖；发布不自动回退到本机 Zen3 库。构建清单记录三项 GMP 哈希，打包时逐项核对。旧构建未记录 GMP 身份时需正常重建，不能只换 DLL 或用 `-SkipBuild` 绕过。

也可双击 `tools/build/build_stage2_release.bat`。默认架构为 `60,70,75,86,89,120`，八路 nvcc 内部编译并行；60/70 使用 CUDA 12.6，其余使用 CUDA 13.3，并选择各自支持的 MSVC。按架构依次编译、打包独立程序，输出至 `dist/cuda-stage2/cuda-stage2-sm<arch>` 及同名 ZIP 和 `.sha256`；无需对单架构 Stage2 再拆分。显式 `-Arch sm_89` 只构建该架构并输出 `dist/cuda-stage2-sm89`。ZIP 从干净暂存目录生成，使用模板 INI 和空队列，不包含包目录既有运行数据。

使用 `-Stage1Exe <已验收的ecm_cuda.exe路径>` 可把既有 Stage1 程序一起放入包；该脚本不会替它重编译或宣称已验证其架构。默认包只包含本轮构建的 Stage2；Stage1 多架构构建仍用 `tools/build/build_stage1_release.ps1`。系统需安装对应的 Microsoft C++ x64 运行库以及支持所用 CUDA 工具链的 NVIDIA 驱动。

## 7. 后续 TODO

考虑向后兼容，未来合并两个可执行文件，配置 Stage1 完成后直接继续 Stage2，或交给 Prime95/独立 CUDA Stage2 消费；明确交接确认、续跑归属和重复消费防护。该项不纳入本轮实现。新 Auto B2 成本标定、多架构资格验证和完整发布验收继续独立推进。


## 8. 本轮构建记录与源码入口

2026-10-08首次发布体验候选（配置统一重构前）完成sm_89/split6独立production构建及主机端收尾重链；exe为2,993,152 bytes，SHA256为 `92fd5c6b7efef55bebc8f4e07c77f29025198d2dca31ac808187d428691c91f9`。最终LF源码构建：CUDA编译104.4秒，主机端驱动编译11.5秒；此前也完成一次哈希匹配的主机端单独重链。

同日完成配置统一重构后，Stage2通过 `-HostOnly` 复用匹配的CUDA对象并重新编译、链接：exe为2,995,712 bytes，SHA256为 `8d27c4107a7c6ccfddf83a67c040163cf7de2eef37f7b80ea98dfd161d51a389`。最终主机端编译：驱动9.2秒、表达式3.8秒、队列4.8秒、配置4.3秒。GUI编译/链接通过，exe为1,136,128 bytes。生成器 `--check`、原生CMake配置检查通过，未运行功能或GPU回归。

随后按 undoc 风格为 83 个正式键和 17 个兼容别名补齐用户说明，参考文档与 INI 注释从同一 `description` 生成。当前 Stage2 包再次通过主机端编译/链接，exe为2,995,712 bytes，SHA256为 `69645f9b904c10a0d95e8ee54289873e3a3e8d2bf90911d59b71144159478a89`；编译耗时为驱动7.2秒、表达式3.4秒、队列4.0秒、配置3.6秒。中文首次启动模板以 UTF-8 字节嵌入，未运行功能或GPU回归。

增加控制台选形/内存摘要后再次通过 `-HostOnly` 构建，CUDA 引擎及其对象保持匹配：exe为3,019,264 bytes，SHA256为 `c453fa1ad14449fccdb58fe9d783c94877f67ba2adbae47ef40a9ac8bd96db61`；主机端编译为驱动5.0秒、表达式2.0秒、队列2.7秒、配置2.5秒。仅做编译、静态核对与配置生成一致性检查，未运行功能或GPU回归。§5.2 的时长来自用户提供的既有单曲线日志。

该次单架构包位于 `dist/cuda-stage2-sm89`，完整源/对象/工具链清单在 `build_cuda_cmake/release_ux_sm89/build_manifest.json`；配置定义、生成器、生成清单和头文件纳入源码哈希。当前发布入口默认六架构 `60,70,75,86,89,120`，各包输出至 `dist/cuda-stage2/cuda-stage2-sm<arch>`，显式 `-Arch sm_89` 可保留单架构构建。构建清单记录实际 CUDA 路径/版本与 MSVC；各架构的 `package_manifest.json` 保留在对应构建目录，不进入发布目录或 ZIP。发布用户文档只包含 `ECM_INI_REFERENCE.md`，另保留 LICENSE 和 GMP 版权文件。

- 严格队列读取与save选择：[records](../src/core/ecm_cuda_stage2_main.cpp#L171)、[queue_fields](../src/core/ecm_cuda_stage2_main.cpp#L348)。
- Stage2专用INI解析：[生成Settings与共享解析](../src/core/ecm_cuda_stage2_main.cpp#L316)、[统一配置定义](../config/ecm_options.json)。
- 结果落盘与子进程日志/中断管理：[append](../src/core/ecm_cuda_stage2_main.cpp#L387)、[child_run](../src/core/ecm_cuda_stage2_main.cpp#L439)。
- 控制台摘要投影：[ecm_stage2_console.h](../src/core/ecm_stage2_console.h#L14)；日志先写原行的位置：[转发](../src/core/ecm_cuda_stage2_main.cpp#L507)。
- 曲线结果回执：[curve_worker](../src/core/ecm_cuda_stage2_main.cpp#L629)。
- 队列进度绑定、恢复与提交：[调度循环](../src/core/ecm_cuda_stage2_main.cpp#L920)。
- GPU 选形与初始内存规划：[run_real](../src/cuda/ecm_cuda_stage2.cu#L8221)；G 树/fold/下降/GCD：[run_batched](../src/cuda/ecm_cuda_stage2.cu#L7121)。
- NTT 池与峰值统计：[NttArena](../src/cuda/stage2/ntt_runtime.cuh#L1813)、[print_workspace_stats](../src/cuda/stage2/ntt_runtime.cuh#L1963)。
- 原子进度与回执对账：[ecm_stage2_queue_state.h](../src/core/ecm_stage2_queue_state.h)。
- 可读日志与独立诊断：[ecm_stage2_logging.h](../src/core/ecm_stage2_logging.h)。
- 单/多架构发布打包：[build_stage2_release.ps1](../tools/build/build_stage2_release.ps1)。
