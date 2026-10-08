# 独立 CUDA ECM Stage2 生产入口

`ecm_cuda_stage2.exe` 从已经完成 Stage1 的文本 save 读取曲线，执行 CUDA 多项式 Stage2。它不计算 Stage1，也不会重新乘以 12。`exponent=lcm|choose12` 由生成存档的 Stage1 决定；恢复时原样使用存档中的 Q。

截至2026-10-08，已发布生产基线仍是893：固定PTX3 Goldilocks、xADD6、Mersenne点乘折叠、GPU baby、驻留fold及尺寸策略；其发布与完整A/B见[点折叠与D报告](D:/code/MPA-OpenCl/docs/STAGE2_POINT_FOLD_D_CALIBRATION.md:69)。下面带日期的段落保留历次实现记录，不应把早期的“尚未实现”当作当前状态。

开发引擎保留默认关闭的 `NTT_GIANT_SEED_PAIR=1` 和可选CPU base。当前独立生产源码选择配对GPU seed，缓存 `[D]Q` 并从一个ladder同时产生相邻起点；不可逆base回退原算法。成本尚未重标定，不能套旧Auto B2 profile，详见[giant seed算法、容量与验证](D:/code/MPA-OpenCl/docs/STAGE2_XADD_D_OPTIMIZATION.md:178)。已发布893入口仍最多8192位；当前源码支持16384位，独立CU与日志控制已接入候选，验收和发布边界见末尾专节。

16384位扩展已经覆盖save/队列和规划限制、256-limb点/归约分派、除数constant容量以及旧S5局部数组。代码见[save读取](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:144)、[规划上限](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:8)、[模板分派](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:932)、[除数表](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2001)、[S5分派](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:5457)。8192的ladder launch cap是点数而非位宽，继续保留其watchdog合同；大界生产容量/性能仍需独立验收。

Auto B2已有经验证的4acc/v1窄范围组合；后续全范围验证虽然算术和收益排名通过，但耗时精度失败，没有导出新cprof。[当前发布边界](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_G1_EXACT_TREE.md:302)。

当前驻留下降根已接入独立生产源码，功能、搬运及完整计时总账验收通过，生产整曲线收益尚未稳定；末尾“生产接入与完整收尾计时”记录本阶段状态。各带日期段落是当时的测量快照，发布包893与旧Auto B2成本范围仍保持。

## 编译与直接读档

在仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_ecm_cuda_stage2.ps1 `
  -Build build_cuda_cmake/stage2_candidate -Engine production -Arch sm_89 -SplitCompile 6
```

示例输出在 `build_cuda_cmake\stage2_candidate\`，包括 `ecm_cuda_stage2.exe`、`gmp-10.dll`、源文件与工具链哈希清单 `build_manifest.json`。省略 `-Build` 的历史默认目录仍为 `production_stage2`，实验时显式指定新目录。脚本默认独立production/PTX3/outer0；需要MSVC、CUDA nvcc和仓库内GMP，此脚本不编译Stage1/CGBN，也不依赖OpenCL。

```powershell
.\ecm_cuda_stage2.exe --save D:\saves\m4423_260e6.save --b2 20e11 --device 1
.\ecm_cuda_stage2.exe --save D:\saves\m4423_260e6.save --b2 20e11 --device 1 --skip-curves 10 --curves 2
.\ecm_cuda_stage2.exe --save D:\saves\m4423_260e6.save --b2 20e11 --dry-run
```

`--curves 0` 或省略它：执行跳过后的所有记录；`--skip-curves` 默认 0。记录按非空、非注释行编号，从 1 开始。`--dry-run` 检查配置、所选存档和队列匹配，不启动 CUDA、不写结果、不推进队列。

`--device` 是 CUDA 设备编号。例子使用设备 1，实际部署应按机器选择。默认取 ini 的 `device`，没有配置时为 0。`sm_89` 是当前 RTX 40 系列编译配置，其他设备应重新选择编译架构。

根 CMake 也提供可选目标：配置 `ECM_ENABLE_CUDA=ON`、`ECM_BUILD_CUDA_STAGE2=ON` 后执行 `cmake --build <build> --config Release --target ecm_cuda_stage2`。当前基础驱动的子进程和队列实现只支持 Windows。

## 存档范围与检查

支持仓库 param0 Stage1 写出的单行 `METHOD=ECM` 文本记录，例如：

```text
METHOD=ECM; SIGMA=26; B1=1000; N=(2^128+1); X=0x294b8e0c5e3f95b6d6c218e669c557f9; CHECKSUM=755878334;
```

必需字段：`METHOD=ECM`、`SIGMA`、`B1`、`N`、`X`。`PARAM` 缺省或为 0；N 支持已有表达式解析器的十进制、十六进制及整数表达式。X 为普通整数域的仿射坐标，十进制或 `0x` 十六进制，范围 `[0,N)`。Z 缺省视为 1；显式 Z 只能为 1。

有 `CHECKSUM` 时严格验证：

```text
CHECKSUM = (B1 × sigma × N × X) mod 4294967291
```

没有校验和的兼容文本存档可以读入，启动信息会显示 `checksum=absent`。校验和只检验记录一致性，不能证明 Stage1 算法正确，也不能识别历史归一化错误生成的坐标。

限制：N 为奇数、`3<N`；已发布893最多8192位，当前开发构建最多16384位；sigma 是精确解析的 uint64 且至少 6；`B1≥2`、`B2>B1`。不支持 param2/param3、Edwards 存档、Prime95 二进制存档，以及尚未完成 Stage1 的 `.ckpt`/二进制 checkpoint。发现不支持的字段或不匹配的校验和时停止。

若保存的 X 已暴露 `1<gcd(X,N)<N`，直接记录 `factor_in_saved_X`，无需再次执行 Stage2。X=0 不能作为有效恢复点，会报错。

## ecm.ini

默认读取 **exe 同目录**的 `ecm.ini`。隐式配置不存在时使用内置默认值；显式 `--ini`/`-ini` 指定的配置必须可读。不会自动创建或覆盖 ini。

```ini
[queue]
worktodo=worktodo.txt
finished=worktodo.finished.txt
tmp_dir=saves

[gpu]
device=1

[stage2]
stage2_b2=20e11
stage2_d=0
stage2_batch_mb=64
stage2_arena_mb=0
stage2_results_file=stage2_results.jsonl
```

可复用现有 ini，Stage1 的 method/exponent/gpu_param 等字段不会改变恢复点。基础版本使用共同字段 `worktodo`、`finished`、`tmp_dir`、`log_file`、`device`，以及上述 `stage2_*` 字段。配置中的相对路径以 ini 目录为基准；CLI 的相对路径以当前工作目录为基准。

`[Worker #N]` 覆盖该 worker 的全局配置。ini 和 worktodo 都通过 `--worker N` 选择，默认 1；其他普通分组标题仅用于阅读。

`stage2_d=0` 自动选择 D；显式 D 必须为至少 6 的偶数，最终形状仍由引擎检查。`stage2_arena_mb=0` 自动按可用显存预算，保留引擎默认 768 MiB 余量；非零值设置 NTT arena 容量。**arena 与 batch 是算法缓冲预算，不是整个进程显存的硬上限**；曲线坐标、F 树、下降工作区等另占显存。默认设备 giant leaf 最大额外驻留预算为 512 MiB，超出时沿用引擎的回退路径。

CLI 可覆盖：`--b2`、`--d`、`--device`、`--batch-mb`、`--arena-mb`、`--worktodo`、`--results`、`--log`。B2 优先级为 **CLI > worktodo 非零 B2 > ini stage2_b2**。不猜测 B2；无法获得有效上界时停止。

## worktodo.txt

```text
# N = 2^4423-1，读取 saves\m4423_260e6.save 中前 2 条曲线
ECMSTAGE2=N/A,1,2,4423,-1,m4423_260e6.save,20e11,0,2

[Worker #2]
ECMSTAGE2=1,2,4423,-1,m4423_260e6.save,20e11,2,2
```

```powershell
.\ecm_cuda_stage2.exe --ini .\ecm.ini --worker 1 --dry-run
.\ecm_cuda_stage2.exe --ini .\ecm.ini --worker 1 --once
.\ecm_cuda_stage2.exe --ini .\ecm.ini --worker 1
```

格式：

```text
ECMSTAGE2=[AID,]k,b,n,c,save_name[,B2-or-zero][,skip_curves][,num_curves][,"known_factors"]
```

三个数字后缀按位置从左到右提供，可省略尾部字段：B2 缺省/0 使用 CLI 或 ini；skip 缺省/0 从首条记录开始；num_curves 缺省/0 执行跳过后的全部记录。要指定 skip 或数量而不指定 B2，需要用 0 占住 B2 位置。带引号的已知因子字段可以跟在任意一个数字后缀后，也可直接跟在文件名后。此规则对应 Prime95 [commonc.c:2937](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/commonc.c:2937) 与 [可选字段解析](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/commonc.c:2973)。B2=0 的自动选界在本基础入口中由 CLI/ini 的显式值替代。

例如 `ECMSTAGE2=1,2,3701,-1,"m3701_260e6.save",26000000000,960,10` 解析为 B2=26000000000，跳过记录 1–960，执行记录 **961–970，共 10 条**，与存档文件名中的 B1 无关。

队列 N 使用已有 `k*b^n+c` 与已知因子精确除法实现，必须等于每条所选存档的 N。`"xxx"` 可以被识别为最后的已知因子字段，但不是有效数值因子；实际运行须省略它或替换为真实因子。B1 从记录读取，不依赖文件名；一条任务内所选记录的 B1 必须相同。保存文件的相对路径以 `tmp_dir` 为基准。基本队列入口只接受 `ECMSTAGE2=`，`ECM=`/`ECM2=` 是 Stage1 任务，不能代替 Stage2 存档任务。

没有 worker 标题的任务属于 worker 1。默认连续处理指定 worker 的全部任务，队列为空后退出；`--once` 只处理其第一条任务，不是只处理一条曲线。`--dry-run` 预览其第一条任务。队列模式的曲线数量和跳过数以任务为准，禁止通过 CLI `--curves`/`--skip-curves` 覆盖。

一个 worktodo 文件同时允许一个 Stage2 消费进程，使用 `.stage2.lock` 独占句柄；进程退出后自动释放，空锁文件可以保留。不同队列可分别启动进程。完成任务后复制原任务行到 finished，再移除原任务，保留注释及其他 worker 的任务。运行期间应避免其他程序改写同一个队列；检测到当前任务改变时不会移除新任务。

## 执行、日志和结果

驱动先验证整条任务所选记录、N 与 B2，然后每条曲线启动一个新的同名 exe 子进程。子进程再次核对记录偏移、内容指纹与校验和，并执行：

1. 重建 sigma 的 Suyama a24；直接装入存档 Q=(X:1)，跳过 Stage1 指数生成和 ladder。
2. 原实验引擎的形状选择、Montgomery/归约自检、baby/F 树和缓存准备。
3. giant 树、积累/折叠、scaled 下降、块/叶 GCD 与因子验证。
4. 排空算术检查，成功后写入结果；关闭进程以释放全部 CUDA 及 host 状态。

采用已有优化组合：Mersenne特化、small-prime reuse、device giant seed、批量精确segment inverse、device giant leaf/root、GPU驻留fold、scaled descent、输出窗口/分块，以及异步oracle和批量carry检查。非Mersenne N使用通用模数路径。当前独立production候选固定主要算法选项，冲突的 `NTT_*` 值会拒绝；算法A/B使用 `-Engine development`。下文显式0回退开关属于历史893/开发引擎的接口；生产候选通过预算/分配条件自动回退，具体控制见末节。

GPU fold 的默认开关是 `NTT_FOLD_DEVICE=1`，独立缓冲预算 `NTT_FOLD_DEVICE_MAX_MB=640`（MiB）；设置 `NTT_FOLD_DEVICE=0` 可恢复原 host flat fold。显存不足、形状/后端不兼容或超过预算时自动使用原路径。申请前还预留 workspace 剩余增长和 1 GiB 空间；预算不是全程序显存硬上限。其额外显存为 `8W(9P+8)+48` bytes，W=`ceil(bits(N)/64)`，P=`φ(D)/2`；M4423/P115200 为约 554 MiB，Γ 校正及下降前释放。并发启动不同队列时应为每条活跃曲线分别计入此容量。算法、传输公式、性能和门禁详见 [步骤报告 §33](STAGE2_GPU_CURRENT_PIPELINE.md#33-gpu-驻留-fold算法访存与验证2026-10-04)。

G根直接交接默认 `NTT_GROOT_TO_FOLD=1`，设置0恢复根先读回再上传的路径；要求 GPU fold owner 和 root-only device G-tree 已启用，否则自动使用旧路径。该交接不新增大型缓冲，当前 owner 公式已含16 B输入摘要。日志中的原始G-root FNV仅在 `root_hash_complete=1` 时完整；直接路径应核对 `real_batched_rootfold` 的sum/xor与最终Γ校正叶值。详见 [步骤报告 §34](STAGE2_GPU_CURRENT_PIPELINE.md#34-g-root-直接交接给-gpu-fold2026-10-04)。

检查调度默认 `NTT_S4_ORACLE_ASYNC=1`、`NTT_S4_CARRY_BATCH=1`，可分别设置0回退。oracle使用默认4槽pinned环，最终返回前全部验证；carry合并同形状中间块的诊断读回，覆盖数量保持。M4423实测增加约70 MiB raw pinned staging和1.573 MiB oracle pinned环，显存峰值保持5544 MiB。此额外RAM应计入多曲线预算；不是只看arena上限。详情见步骤报告§35。

NTT tile 默认 `NTT_FUSE_WARP_TAIL=1`，低6层使用warp寄存器交换与常量根约减；显式设0回退原shared实现，实验exe仍默认0。当前收益验证覆盖sm89/GPU1及报告中的M4423负载，其他架构/形状需重新测量；不增加大型buffer。算法、源码行号与8次对照见 [步骤报告§37](D:/code/MPA-OpenCl/docs/STAGE2_GPU_CURRENT_PIPELINE.md:1688)。性能测量前应删除 `NTT_FUSE_TRACE` 环境变量（PowerShell：`Remove-Item Env:NTT_FUSE_TRACE -ErrorAction SilentlyContinue`）；该诊断开关按变量存在性启用，设0或空值仍会逐kernel同步。

控制台与引擎输出受当前候选的 `--log-level` / `stage2_log_level` 控制，默认batches。引擎输出默认在exe/ini目录的 `stage2_screen.log`；worker 2为 `stage2_screen_2.log`。配置中的显式 `log_file` 优先；设空值则让引擎直接输出到控制台。历史893没有日志等级功能。

每条成功曲线追加一个 JSONL 结果，默认 `stage2_results.jsonl` 或 `stage2_results_N.jsonl`，包含状态、save/记录编号与指纹、N、sigma、B1/B2、设备/worker、请求 D、时长、hits、bad_factors 和十进制 factors。自动 D 的实际选择及树哈希保存在完整引擎日志中。

默认仅追查首个命中叶的具体 stage2 素数名字，避免生产形状的诊断扫描耗时过长；所有叶的 GCD/因子提取仍执行。结果中的 factors 是发现的非平凡约数，可能仍是合数，不代表完整分解。`hits=0` 也属于正常成功完成。

当前不提供 Stage2 中途 checkpoint、自动恢复曲线索引、PrimeNet 上报或跨曲线流水线。失败/中断时保留任务及源存档；已经完成的记录可能有结果，再次运行会重做它们并追加新结果。基础版本不保证恰好一次执行。源存档从不改写。

## 实现位置

- 存档文本和校验和解析：[ecm_cuda_stage2_main.cpp:106](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:106)；可选队列字段：[同文件:301](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:301)；配置和调度：[同文件:443](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:443)。
- 独立生产默认值与实现：[ecm_cuda_stage2.cu](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1)；生产API：[同文件:9073](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:9073)；算法配置检查：[同文件:9105](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:9105)。
- 开发驱动：[ecm_cuda_stage2_dev.cu](D:/code/MPA-OpenCl/tools/bench/ecm_cuda_stage2_dev.cu:1)；开发运算引擎：[stage2_tree_gpu.cu](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10896)。
- 已有表达式、ini 和队列工具：`src/core/ecm_expr.cpp`、`ecm_queue_config.cpp`、`ecm_worktodo.cpp`。
- 独立编译脚本：[build_ecm_cuda_stage2.ps1](D:/code/MPA-OpenCl/tools/build/build_ecm_cuda_stage2.ps1:1)。

已发布893包装实验引擎，当前源码已拆出独立production闭包。性能与显存模型见 [当前 Stage2 步骤报告](STAGE2_GPU_CURRENT_PIPELINE.md)，large-N的实际预算应以运行日志为准。

## 本轮验证（2026-10-04）

实际编译生成 sm_89 exe，SHA256 为 `c63249542418de3cc6af2bb93b443ec39eca053a29b95d3df125eefcf31b4860`。编译记录及源码哈希在输出目录；本轮验证只使用 GPU1，没有修改 GPU0 的生产任务。

隔离目录中通过 26 项检查：帮助/直接读档、校验和/param3/二进制/数量不足/非整数上界拒绝、可选后缀的缺省值及 0、带引号文件名/已知因子、worker ini 覆盖、不同 sigma 的各自 Q、队列成功移除与其他 worker 保留、输入和 CUDA 失败时队列保留，已有因子存档直接完成、共享队列工具拒绝移除不匹配任务，以及 M4423 大形状恢复。

- 用户示例去掉占位因子 `"xxx"` 后，dry-run 精确选择记录 961–970，共 10 条，B2=26000000000。原样保留 `"xxx"` 时，数字字段仍解析正确，随后因无效已知因子拒绝执行，队列不变。这里的 M3701 文件只是 **解析用合成记录**，没有声称验证了其 Stage1 或执行了其 GPU Stage2。
- 真正的 frozen CPU Stage1 Q（N=2^128+1，sigma26/B1=1000/B2=1e6/D210）恢复后发现因子 `59649589127497217`，与 CPU 树参考一致；原 save 哈希不变。
- M127 两条独立 CPU 参考生成的 sigma26/27 存档分别执行，装入的 Q 逐条匹配且不同。每条子进程完整完成，结果没有非平凡因子。
- M4423 使用已验证的 choose12 Q，sigma26/B1=1000/B2=2011326186870/D1231230，batch64/arena6300 MiB。Stage2 init=14.808017 s、main=60.308395 s、total=**75.116411 s**，独立驱动曲线墙钟=75.719930 s；这是单次接入验证，不能作为新的优化 A/B。
- M4423 的 Q SHA256=`33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`，最终 115200 叶/8064000 words 哈希=`10619321735931855904`，与原实验基线一致。403 批调用、1979251 对多项式乘法、40218760 归约系数、2400 自检、66139 GMP 样本、4 full checks 均保持，错误数 0。

恢复时 Q 被表示为 `(X:1)`，原实验 Stage1 ladder 输出另一个等价投影表示，因此 **未除去投影比例的 G-root 哈希可能不同**。本次原始 G-root 哈希为 `acd84d2595022e3e`，实验基线为 `105d6128bbf522db`；引擎在 fold 后用 Γ⁻¹ 消除该比例，校正后的最终叶值逐字哈希一致。比较跨入口数值时应核对仿射 Q 与校正后的叶值，不能以原始投影 G-root 哈希判错。

可复查证据：[验证汇总](D:/code/MPA-OpenCl/build_cuda_cmake/_production_stage2/acceptance/summary.json)、[用户示例选择](D:/code/MPA-OpenCl/build_cuda_cmake/_production_stage2/acceptance/user_example.log)、[占位因子拒绝](D:/code/MPA-OpenCl/build_cuda_cmake/_production_stage2/acceptance/user_example_xxx.log)、[M4423 完整日志](D:/code/MPA-OpenCl/build_cuda_cmake/_production_stage2/acceptance/m4423_engine.log)、[JSONL 结果](D:/code/MPA-OpenCl/build_cuda_cmake/_production_stage2/acceptance/m4423_results.jsonl)。测量文件位于 ignored build 目录，不随源码提交。

### 2026-10-04 GPU fold 默认值更新

新版 exe SHA256=`2ba0000aabc6d7609165b8ed8cf69c6b53fd13ae191d95255121c5f3f4fb5d2d`，2740224 bytes；同目录保留 `gmp-10.dll`。基本生产入口21项验收、CUDA失败保留队列、saved-X因子及M4423实际恢复共 **24/0**。用户 worktodo 的 B2/skip/num 选择与已有因子解析重新验证通过；驱动及存档格式本轮未改。

M4423 save 恢复时 owner 启用，14 folds/42 muls，init15.123476/main58.515049/total73.638524 s，最终叶哈希 `10619321735931855904` 与原基线一致。该单次验证用于确认部署接入；性能结论采用实验 exe 的同二进制 ABBA（76.61→74.21 s），不能把两种入口的单次时间直接解释为加速比。[新版生产验收](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_device_20261004/production_accept/summary.json)、[大存档引擎日志](D:/code/MPA-OpenCl/build_cuda_cmake/_fold_device_20261004/production_accept/m4423_engine.log)。

### 2026-10-04 G-root直接交接更新

生产 wrapper 默认 `NTT_GROOT_TO_FOLD=1`，显式0恢复原host根交接；实验 exe 仍缺省0。生产重编译 sm89/CUDA13.3成功（CUDA265.2 s），exe2636800 bytes，SHA256 `08df4c392b24c040ce9892c2591a3de0a6f5a8fc7b5bf5eef980c2932300ec8e`。save/ini/worktodo基本21项加CUDA失败队列保留、saved-X因子和M4423实际恢复，共 **24/0**。用户 B2/skip/num 后缀仍精确选择961–970，字面xxx拒绝，队列不变。

生产 M4423 save：init15.231398/main58.984101/total74.215499 s，15 roots/114352490 words全部直接交接，14 folds/42 muls；最终叶FNV`10619321735931855904`同基线，GMP/carry/NTT计数和bad0保持。这是部署验收单次计时，非新的A/B。save的Q=(X:1)与实验Stage1的等价投影Q使原始根输入摘要分别为sum`3cf2f49cf1972d5d`、xor`3fafa10f6f7f6f62`，与实验摘要不同；Γ⁻¹校正后叶值一致。跨入口不能把原始根摘要当作仿射结果摘要。

[生产验收](D:/code/MPA-OpenCl/build_cuda_cmake/_root_fold_20261004/production_accept/summary.json)、[M4423引擎日志](D:/code/MPA-OpenCl/build_cuda_cmake/_root_fold_20261004/production_accept/m4423_engine.log)。

### 2026-10-04 异步 oracle / carry 合并默认值更新

该轮exe SHA256=`ecee5b977ca52a38c71e284d55f3179e34674d9da24b85d2709ef8fde376b90f`，2636800 bytes；sm89/CUDA13.3，CUDA编译245.1 s。新默认检查调度已经通过生产入口 **24/0**。实际M4423 Stage1 save恢复init15.007673/main57.193577/total72.201250 s，最终叶FNV `10619321735931855904` 与基线相同；oracle1126任务/66139样本全部比较、pending0，carry8241块合并为252次finish，错误0。

性能采用实验exe四组合八次交叉测试：00→11 total73.072038→71.685022 s（−1.898%），main58.6242905→57.379686 s（−2.123%）；本次生产单次时长用于接入验证。两开关可分别设0回退，检查样本/覆盖数保持。相关源码、容量公式与Nsight数据见 [报告§35](D:/code/MPA-OpenCl/docs/STAGE2_GPU_CURRENT_PIPELINE.md:1548)。[生产验收](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_20261004/production_accept/summary.json)、[M4423日志](D:/code/MPA-OpenCl/build_cuda_cmake/_resident_checks_20261004/production_accept/m4423_engine.log)。

同二进制两组ABBA共8次：total均75.086909→74.560906 s（观测−0.70%），main−0.42%，初始化未优化且有波动。确定收益为每曲线减少1.704 GiB主机边界传输，G根主机payload为0；显存观测峰保持5544 MiB。详细公式、测量范围及行号见步骤报告§34。

### 2026-10-04 NTT warp tile默认值更新

该轮工作区 [ecm_cuda_stage2.exe](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe) 为2691072 bytes，SHA256=`905826a8988e2046831e4ecd57406c46a6d8b0db2ac22551b2d6be03d18b588c`；sm89/CUDA13.3，CUDA编译252.8 s。[编译manifest](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/build_manifest.json) 记录共用NTT源码F7C98F0B…ACD7，Stage2树引擎E89B554D…F19未变。已启用warp tile默认1，显式环境变量0仍生效。

生产入口重新验收 **26/0**：基本21项、CUDA失败保留队列、saved-X已有因子、实际M4423 save恢复，以及默认1/显式0的GMP频谱、逆变换和容量切换检查。worktodo B2/skip/num仍选择用户示例961–970；xxx拒绝且队列保留。实际M4423存档恢复init15.776226/main57.036366/total72.812592 s，最终115200叶/8064000字/FNV `10619321735931855904` 与基线一致，oracle1126 jobs/66139样本全部比较，carry8241/252、pending0、错误0。

性能结论采用实验同binary串行ABBA+BAAB共8次，完整Stage2均值 **73.121608→72.528029 s（−.81%）**，main−.925%；独立Systems中tile−5.58%、NTT−2.90%。样本数少且存在波动，不能将生产单次时长作为A/B，也不保证其他形状收益。算法传输量和GPU1峰5544MiB保持。[生产26项](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/production_accept/summary.json)、[M4423日志](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/production_accept/m4423_engine.log)、[性能量化](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/final_quantitative.json)。编译及验收只更新本工作区，GPU0外部生产运行未改。


### 2026-10-04 xADD6 / D重标定 / cooperative NTT v2

该轮[生产exe快照](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/production_f85_before_shape/ecm_cuda_stage2.exe)为4038144 bytes、sm89/CUDA13.3，SHA256 `f85ead72d6e68a5952c5affcd9df1be32002428c08b2376a2f2b396a0e55f7f1`。生产默认NTT_XADD6=1、NTT_D_MODEL=1；分别设0回退。新D系数只在RTX4060 Laptop/M4423/B1=1000/当前驻留与检查配置生效，大界自动D1381380、小界D330330；其他scope回原模型，显式D优先。CPU选D单列d_scan_wall，不含于stage2_full_wall。

最终入口30/0，包含默认/回退、自动D大小界、实际M4423恢复、用户B2/skip/num的961–970选择、xxx拒绝/队列保留和冻结因子。显式D1231230实际save full68.134574 s；自动大界61.729618、小界13.437128 s，均为验收单次。cooperative另一次实际save通过。性能结论使用固定工作量xADD8次A/B（6.26%）及cooperative4次A/B（2.62%），不同D比较另报。

该轮NTT_FUSE_COOP_OUTER缺省0；手动1启用M8实验，旧D拟合会回退。后续尺寸策略与重标定记录见[P3报告](D:/code/MPA-OpenCl/docs/STAGE2_NTT_SHAPE_D_CALIBRATION.md)。所有开关继续保留算术自检/采样与最终drain。原语、D/NTT公式、统计范围及源码见[详细报告](D:/code/MPA-OpenCl/docs/STAGE2_XADD_D_OPTIMIZATION.md)。[最终30项](D:/code/MPA-OpenCl/build_cuda_cmake/_xadd6_20261004/production_accept_v2/summary.json)、[F85编译manifest快照](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/production_f85_before_shape/build_manifest.json)。


### 2026-10-04 按NTT尺寸选择outer / 第二次D标定

最终生产sm89/CUDA13.3编译成功，CUDA593.7 s，exe4042752 bytes、SHA256 `3cf38065e5f88347063f365ac85468475a6624f6e98b52804834d9aef3219f0e`。14项源依赖哈希复核通过，NTT CU/cooperative header与计时实验相同；最终header重构/新D系数/生产wrapper另有生产验证。入口**32/0**：基础21项、CUDA失败队列保留、saved-X已有因子、实际M4423显式D、warp/xADD默认与显式回退、默认模式2自动D大小界，以及outer0恢复原resident模型/outer1强制实验回旧模型。三种outer模式的小界叶摘要一致。第一次验收因准备样本遗漏N/A assignment-id而触发finished文本断言，已按原样本在全新目录重跑32项；源码未因此修改。

生产默认NTT_XADD6=1、NTT_D_MODEL=1、NTT_FUSE_WARP_TAIL=1、NTT_FUSE_COOP_OUTER=2。尺寸策略受设备/t/warp/compact/M限制，未测形状沿用原planner。outer0使用原resident系数，outer1没有匹配经验fit而回§56.1；D模型0恢复旧D选择。显式D仍优先。每条save使用自己的sigma/B1/Q；worktodo用户示例仍选择961–970，xxx拒绝并保留队列，冻结因子59649589127497217正确。

实际save显式D1231230：init/main/full=14.112085/53.712880/67.824965 s；自动大界D1381380：init/main/full=15.391578/48.891094/64.282673 s，CPU选D另0.146617 s；自动小界D330330：init/main/full=3.290443/10.134760/13.425203 s，CPU选D另0.183744 s。叶FNV依次10619321735931855904、4244971527793015097、7549663880496122317保持；这些是单次接入验收，不作为性能A/B。save-Q入口的root摘要与实验自算Stage1入口不同，跨入口不直接比较projective摘要。

产物：[生产exe](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe)、[manifest](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/build_manifest.json)、[32项验收](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/production_accept_final/summary.json)、[来源复核](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/production_provenance.json)。构建/日志及旧F85快照留在ignored目录；源码、门禁、工具和报告提交Git。

固定工作量八次A/B full降2.49%、main降3.25%，NTT arena payload峰少约300MiB；该收益不与P2的2.62%叠加。独立NTT10/0、模型32/0及最终入口32/0通过。公式、scope、拟合误差和来源行号见[P3报告](D:/code/MPA-OpenCl/docs/STAGE2_NTT_SHAPE_D_CALIBRATION.md)。

## 2026-10-05 短归约匹配 D 模型与生产发布

短归约后重新冻结k16..27卷积权重，六条多D曲线加四条同binary锚点拟合profile3；独立B2=1e11留出验证330330每条快于510510，绝对秒数仍低估5.99–11.76%，未回调拟合。大界D1381380、小界D330330，owner512MiB时大界D1141140。真实NTT backend/C++模型与整数features **48/0**，实验planner **22/0**、归约scope **36/0**、shared/warp NTT **14/0**；三个自动D真实曲线均GMP bad0/pending0/clean1。模型不增加GPU数组/传输。

生产33/0、最终实际S4后端选择器30/0；后者为早停决策门禁，不冒充完成曲线。额外修正OLDTAIL/S4_OFF scope，并将同一S4启用快照提前；错误指针初版28/2失败候选保留且未发布。最终E0139328…BE331，4127232 bytes，CUDA571.1 s；15原始依赖及快照核验。生产save固定D1381380四条ABBA：64.647779、56.586517、57.082008、65.380452 s，均值**65.014116→56.834263 s（快12.58%）**，每模式2条、无CI，叶/oracle/因子一致。生产short默认1，显式0回旧归约与resident_shape；实验仍默认0。旧3CF产物已保留。

D138标定均值full56.643870s：init26.59%，baby14.08%、CPUaffine6.10%、F树剩余6.41%；main73.41%，G树23.96%、giant18.28%、下降13.46%、fold10.69%。owner609.086MiB，NTT arena完整payload3341481200B，主workspace3GiB；host输出窗口实际回读2838813040B，不等于所有曲线D2H，没有新NVML全进程峰。具体计算量、内存与传输公式及各阶段误差见[短归约 D 标定与生产报告](D:/code/MPA-OpenCl/docs/STAGE2_SHORT_REDUCTION_D_CALIBRATION.md)。

下一轮优先GPU baby批量规范化。原Systems最大2.784s间隙在X/Z回读后、下次copy提交前，与CPUaffine准备关联；trace无CPU栈，仍为源码推断。giant叶是[-X,Z]，baby F树要求[-X/Z,1]，须增加设备prefix/逆元传播并保留坏Z的GCD/因子语义；先仅回读P个常数再接设备叶frontend。之后复测gl_mod规范化/低层根；多曲线仍需独立状态与workspace lease及RAM/VRAM预算，公平Prime95新对照待做。

## 2026-10-05 GPU baby 实验进度

实验源码新增GPU baby归一化，生产E013二进制保持上一发布状态，该二进制尚不包含新功能。新编译源码可显式设置NTT_BABY_DEVICE=1，默认0；暂时须固定D做性能比较，请求该路径会拒绝旧CPU baby成本模型。八次完整曲线均值57.302417→53.294626秒（快6.99%），host/因子/cache与独立baby/F检查通过。生产默认提升仍需新D和实际save入口验收。详见[GPU baby 归一化报告](D:/code/MPA-OpenCl/docs/STAGE2_GPU_BABY_NORMALIZATION.md)。


## 2026-10-05 GPU baby 生产发布

上述实验进度已由新D和实际save入口验收推进到发布。生产现默认NTT_BABY_DEVICE=1；显式0恢复CPU baby及相应原模型。profile4 resident_baby_v1要求已测short/尺寸策略/设备/N/B1/检查配置；其余scope和预算/诊断回退按实际路径处理，显式D仍优先。

当前发布exe SHA256 `a1910cb47d4a4d567fed441a9f5002f1b0ba1675c376c095035ee01c1456ec6a`，17项源码依赖及快照核验，原E013已备份。生产save/ini/worktodo33/0，新旧scope80/0、30/0，原生planner9/0；owner512未拟合D完整曲线通过。同exe、同save、固定D1381380四次ABBA均值53.569084→50.520226秒（快5.69%，每模式2条，无CI），leaf/oracle/因子和必需检查一致。大界D1381380、小界330330、owner512时1141140，当前首选D保持；新模型反映GPU准备成本而非额外换D提速。源码与详细证据见[GPU baby D标定报告](D:/code/MPA-OpenCl/docs/STAGE2_GPU_BABY_D_CALIBRATION.md)。


## 2026-10-05 固定PTX、点折叠与新D生产状态

当前工作区生产为893F6E90…01F69D，19源码冻结，旧DCF已备份。使用-GlBackend ptx重建：fixedPTX默认NTT_POINT_MERSENNE=1；精确N=2^s−1使用保持Montgomery坐标的乘积折叠/旋转，通用N自动fallback。设0恢复原SOS/REDC；匹配的M4423/RTX4060/check/budget范围下point1选profile6 resident_point_fold_v1、point0选profile5，其他scope沿用保护。runtime/short/fold构建缺省point0。保存点、worktodo/ini接口保持。

完整native入口30/0、发布路径21/0；同Q/B2/D与检查8次串行交叉，旧DCF48.8331495→新39.0421205s，少20.05%，每版本4条/无CI。大界D1381380、小界390390、owner512时1141140、baby128时600600，首选D与固定PTX前一阶段保持。NTT低层单位根在固定PTX下变慢，未接生产。[本轮公式/容量/占比/源码与证据](D:/code/MPA-OpenCl/docs/STAGE2_POINT_FOLD_D_CALIBRATION.md)、[NTT候选负结果](D:/code/MPA-OpenCl/docs/STAGE2_NTT_SMALL_ROOTS_FIXED_PTX.md)。

## 2026-10-05 NTT outer 调度实验选项

生产编译脚本新增 `-OuterUnrollU 0|4`，默认0。4在cooperative outer中展开四组独立蝶形，无新增数据数组或传输，REG/代码体积增加；仅建议在独立实验目录编译并给出显式D。新调度尚未标定，实际宽度4会回退legacy D模型，不能沿用profile6经验速率。

完整卷积k24..27独立重复少0.70–0.78%，但真实保存点八条Stage2均值少0.367%、两个顺序组方向不同，尚未建立稳定整曲线收益，因此生产exe仍893、默认调度0。native算术/实际scope18/0，save/ini/worktodo基本入口21/0。复现、全部样本、资源代价、计算量/访存公式及source-line见[outer ILP报告](D:/code/MPA-OpenCl/docs/STAGE2_NTT_OUTER_ILP.md)。


## 2026-10-05 可选NTT进位/诊断融合

从当前源码独立构建（-GlBackend ptx -OuterUnrollU 0）后，设环境变量NTT_CARRY_CHECK_FUSED=1启用大batch进位/检查融合；未设置默认0。scratch由arena拥有、复用、计入硬预算，公式8ceil(L/256)mB，真实曲线峰4MiB；小调用及预算/分配失败沿用原检查。deferred错误累积与所有必需GMP/oracle检查保留，save/worktodo/ini接口保持。

请求此实验模式会禁用旧经验D profile，请使用显式--d做对照。同binary真实保存点八条预热后交叉full38.2264815→37.9273095s（少0.783%），main少0.940%；每模式4条、无CI。原语/实际NTT/deferred故障/预算回退与native两模式各18/0通过。当前生产exe仍893、实验开关默认0；新NTT成本权重与多D留出完成后再决定默认发布。[公式、source-line、命令、全部样本](D:/code/MPA-OpenCl/docs/STAGE2_NTT_CARRY_CHECK_FUSION.md)。

## 2026-10-05 分离大工作区与fold owner预算的实验工具

`NTT_FOLD_DEVICE_MAX_MB` 可独立限制owner（默认640MiB，0触发预算回退）；现有`--arena-mb`限制缓存/规划口径，包含旧保守table计账，并不等于实际NTT payload或全进程显存上限。生产暂没有单独big allocator硬限。

新增`tools/bench/bench_stage2_budget_scaling.py`，通过真实C++ packing query过滤D形状，并验证actual big_peak；可设置`--big-mb`、`--small-big-mb`、`--owner-mb`、`--small-owner-mb`、`--arena-mb`。`--prepare-only`生成CPU/GMP一致的三位数保存点和规划；`--resume`按已有plan继续，改变预算需新输出目录。工具固定本机GPU1/RTX4060 Laptop，所有曲线保留必需检查，不更改ini/worktodo或生产默认。

本轮52接受曲线/49正式测量审计通过。M4423即使owner回退，D增长时仍alpha约0.54；owner驻留但D固定，alpha可趋近1。同D跨640MiB预算边界，驻留110.95s/回退119.80s（每模式2条，无CI）。报告包含完整矩阵、公式、模块容量、独立整卡/主机commit采样、数据搬运、source行号及可复现命令：[B2/位数/预算实验](D:/code/MPA-OpenCl/docs/STAGE2_B2_MEMORY_BUDGET_SCALING.md)。

## 2026-10-05 Auto B2与tune设计（尚未实现）

已完成Prime95/PRPLL源码调查和[设计报告](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_TUNE_DESIGN.md)。用户确认Auto B2默认最大化总流程相对收益，包含估算的Stage1成本；先提供NTT吞吐量及按位宽/路径的完整成本标定。拟议的`--auto-b2`、`--tune`和新增预算键目前不可使用。当前B2=0仍回到现有配置，最终无有效B2会报错；非零CLI/worktodo/INI优先级保持。

## 2026-10-05 第一轮实现：plan-only与NTT tune

新实验构建已接入`--plan-only`及`--tune ntt`。plan-only使用同一引擎D搜索，读save/队列但不执行曲线、不推进队列；返回模块容量和未标定的时长估计，不能把模块容量相加作为进程峰。

NTT tune要求固定后端构建（例如`-GlBackend ptx`），接受`--length-log2 16:27`、`--tune-repeats 5`、`--tune-memory-mb 1024`、`--tune-file FILE.jsonl`。当前batch1/选定配置，一次iter为两forward加融合product/scale/inverse；不含packing/carry/模N归约/传输。每次核验全部L输出；超预算长度跳过，成功后原子发布JSONL profile，失败保留partial。

arena已改为真实payload计账v2，旧窄范围D标定暂时禁用并回到明确标注的legacy估计。原生产目录与893二进制保持；新功能仅在独立实验构建内，本轮仅编译，无新增运行/门禁/性能结果。`--auto-b2`仍未实现，B2原有语义保持。[构建、命令、source-line、公式与限制](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_TUNE_IMPLEMENTATION.md)。

## 2026-10-05 后续验证与可选复合因子拆解

独立b9af候选已运行12长度NTT tune、plan-only与失败profile保护；20条同save生产基线/候选配对全部通过。仍保留生产893默认，尚未标定大B1/B2完整成本，`--auto-b2`未接入。

候选支持`--factorize-hits [--gp gp.exe] [--factor-timeout 30]`；INI使用`stage2_factorize_hits=1`、`stage2_gp=绝对路径`、`stage2_factor_timeout=30`，CLI显式GP/timeout优先。保留原始`factors`，附加`prime_factors`、逐raw的`factor_analysis`素数/重数/证明/GP日志及`factorization_complete`。GP失败/超时标记unresolved，可离线重试；`seconds`不含可选GP时间，新增`factorization_seconds`单列。该功能默认关闭，需安装PARI/GP。

新增Python/SQLite数据集工具，导入本地梅森数因子表、计算PARAM0群阶/点阶分解和标准单素数B1/B2前沿、导入原生result并在两项界限都改善时更新默认sigma。无因子记录和复合因子的每个proven prime均保留。数据集的B2=0表示Stage1-only，不能直接当原生零B2配置。[完整使用说明、47条真实Stage2证据及限制](D:/code/MPA-OpenCl/docs/STAGE2_FACTOR_DATASET.md)。

## 2026-10-05 因子数据库精简

工作库现在只有mersennes/factors，每因子只保留一个最优sigma及其完整群阶/点阶分解。B1/B2都不增大且至少一项严格减小时替换，不保存其他sigma、运行/导入/分析历史。原生主程序result格式保持，离线ingest流式更新核心记录；旧七表库需migrate_dataset.py一次性迁移。[当前工具说明](D:/code/MPA-OpenCl/tools/ecm_dataset/README.md)。

## 2026-10-05 实验factor-only与成本规划

独立d77f构建支持--factor-only，INI stage2_factor_only=1；跳过可选prime-witness命名，raw因子可能为复合数，配合--factorize-hits可继续拆解。result增加requested_factor_only，hits=0不代表无因子。默认命名与生产893保持。当前Auto B2只有离线工具plan_auto_b2.py在实测scope内规划，原生--auto-b2尚不可使用。[84条曲线验证、Stage1摊销、性能长尾及命令](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_PHASE_CALIBRATION.md)。

## 2026-10-05 原生 Auto B2 接入

独立4acc候选可使用`--auto-b2 --cost-profile FILE`，INI使用`stage2_auto_b2=1`、`stage2_cost_profile=FILE`；仅最终有效B2=0触发自动选择，非零CLI/worktodo/INI保持固定。实际worker联合选择B2/D/owner路径并记录auto_plan，保留原请求；`--plan-only`仅规划、不写result/推进队列。可设置Stage1 batch或每曲线秒数、Stage2 ratio、B2区间、arena与owner预算。

当前运行profile精确绑定GPU1/4acc/PTX3/outer0/accounting2，只发布独立盲测通过的M8191/B1=1000/lcm、30亿～60亿/三D/arena4096/两path范围。2203/4423成本精度未达10%门限，auto拒绝且保留手动运行方式；高B1/泛型N/G1/choose12未覆盖。33调用/213断言与4条auto/manual/queue实际曲线通过；最佳落在下界，非通用生产默认。完整命令及INI样例见[原生 Auto B2 使用与验收](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_NATIVE.md)。

## 2026-10-06 三位宽成本profile与chain对照

最新`docs/data/ecm_auto_b2_multianchor_20261006_gpu1.cprof`适配同4acc二进制，恢复M2203/M4423/M8191、B1=1000/lcm、30亿～60亿/三D/arena4096/两path，六个scope新盲测最大6.99%，原生37调用/255断言及8条实际曲线通过。命令与INI接口保持，只需替换profile路径；高B1/泛型N/G1/choose12仍须扩展。

同binary强制chain在I8327/24977更快，99912仿射点零失配；可以用`NTT_GIANT_CHAIN_MIN=0`配显式B2/D复现实测，但与当前auto成本配置冲突，自动模式会拒绝。默认阈值尚未调整。详情、原始样本及G1/树成本修正方向见[最新标定与阈值实验](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_MULTIANCHOR.md)。

## 2026-10-06 G1/精确树新版候选（尚未发布）

新源码使用cprof v2，支持实测G1区间、精确树调度、按profile绑定chain_min及实测区间并集。独立5f4c候选算术通过，但完整耗时/排名验证失败，暂未生成可用v2 profile；不能将旧4acc/v1 profile用于新二进制。当前通过验收的用法仍是上节4acc+多端点v1组合。离线plan_auto_b2.py新版要求--runtime-profile并调用原生plan-only，使用相同binary/profile格式。

失败数据、边界处理、完整门禁及后续诊断见[Auto B2 G1报告](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_G1_EXACT_TREE.md)。

## 2026-10-06 等待/并发/NTT后续

新版auto额外拒绝CUDA_LAUNCH_BLOCKING=1，避免使用异步标定率预测强制同步配置。同步、pinned读回、双进程及全局tile11均未证明收益；pinned原型撤回，生产tile12及单进程逐curve方式保持。并发实验确认结果一致、采样VRAM/RAM及预算负门禁，不等于生产并发/总显存lease已经实现。Auto B2 v2仍无通过完整门禁的运行profile。

新bench_stage2_variants.py可对照tile/shared radix等现有开关，bench_stage2_concurrency.py用于有预算防护的两点语料试验，analyze_stage2_trace.py重建同时malloc峰与API等待。均不自动更新生产配置/成本profile。详见[最新实测与后续](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_WAIT_NTT.md)。


## 2026-10-06 分 D 标定工具及发布限制

新版离线拟合默认 `--fit-scope per_d`，成本采集可用 `--holdout-all-d` 和 `--extend-study FILE` 补齐所有D的留出点，扩展到新目录并保留原观测。验证工具 `--check-inputs-only` 可先核对build/profile/save身份，执行GPU曲线数为0。最新六组收益排名通过，但27个scope中仍6个未通过完整秒数精度；没有新的v2 cprof，不可将候选JSON直接用于生产auto。

实验 `--reuse-context` 路径已撤回，当前源码不提供该选项；实验exe及脚本在冻结证据中。现有通过验收的4acc/v1组合保持，详见[最新完整结果](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_G1_EXACT_TREE.md)及[上下文实验](D:/code/MPA-OpenCl/docs/STAGE2_CONTEXT_REUSE.md)。


## 2026-10-06 可选CPU等待及Auto网格边界

独立8cee候选可在手动固定B2时用环境变量NTT_CUDA_WAIT_MODE=0/1/2/4（Auto/Spin/Yield/BlockingSync），只绑定实际worker当前设备；未设置保留原等待方式。BlockingSync已验证CPU资源消耗更低，但并非稳定时间加速；没有新增INI键，原auto profile拒绝未标定的非0模式，NTT tune未绑定这个开关。

规划器补齐后续giant分块的chain阈值和关键I平台末端，真实packing计划及1/2/2实际giant分块门禁通过。无新校准cprof、没有晋升生产exe；新binary需完整校准和原生验收。[最新源码行、使用与实验](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_G1_EXACT_TREE.md)。


## 2026-10-07 16384位开发扩展与验收

### 范围和实现

本节是开发候选的功能验收，原发布包893仍保持8192位上限。手动B2支持扩展到16384位；Auto B2的已标定位宽范围保持≤8192，未提供16k成本profile。[自动选择的精确梅森/位宽检查](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:551)继续拒绝未覆盖范围。更宽的算术支持不意味着已标定自动B2或已发布独立生产内核。

1. [共享上限](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:8)定义`max_input_bits=16384`、`max_words=256`，用于实际几何、shape query、save与worktodo入口。奇数、有效X、checksum、B1/B2和param0检查保持；16385位在执行前拒绝。
2. [点/归约模板分派](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:923)增加`<256>`，包含baby、ladder、paired seed、chain、段积和accum。原≤128-limb输入仍分派原容量；没有将所有旧数组扩到256。
3. [设备除数表](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2001)从128到256个64bit words，增加1024B constant payload；[归约形状guard](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2658)按256模板检查`nlimb+1`与`L+nw+2`实际访问范围，局部数组容量`2NW+4`。
4. [Mersenne原语](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2115)的fixture分为128/256模板；部分高limb与整limb都覆盖。点乘保持Montgomery域，radix为`2^(64W)`；梅森位宽不是64倍数时仍需radix旋转，不能直接拿普通余数替代。
5. [旧S5的Horner、subtract和two-minus](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:5351)局部数组改为128/256两档。生产默认的scaled descent策略保持；旧路径用于回归和诊断。S5诊断已读回完整叶值时，[输出叶哈希](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10572)；没有为默认设备路径添加无条件全量D2H。
6. 修复旧S5的P=1断言：形状规划采用`slot_bits=2S+max(1,ceil(log2m))`，验证处原先漏掉`max(1,...)`。旧48b二进制在M4423也会拒绝，证明并非16k特有；[断言](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:6102)现与规划一致，仍执行stride/容量/精确性检查，不改变NTT布局。

[构建脚本](D:/code/MPA-OpenCl/tools/build/build_ecm_cuda_stage2.ps1:13)新增`-SplitCompile 1..64`，缺省1；选项、toolkit、raw依赖与对象SHA写入manifest，HostOnly不能跨该选项复用CUDA对象。本轮用6线程split优化；它不是多条Stage2并行或GPU算法开关。

### 容量、运算和搬运影响

约定模数为N，`S=bit_length(N)`，`W=ceil(S/64)`；P是baby数量，I是giant数量，`m`是一次NTT的共同operand系数数，`Lntt`是实际变换长度。不要混用N与NTT length。

- 点的XZ payload为`16W×点数`B；配对base仍`16W`B，CPU base的接口双向净增仍`8W−8`B，W=256时为2040B。点的位宽扩大没有减少主体传输。
- [giant坐标分块](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9901)仍按256MiB和`16W`B/点向P的整数倍上取整：`C=P ceil(max(P,floor(256MiB/(16W)))/P)`。满256-limb的未对齐容量65536点，向P对齐后可超过256MiB；不是全进程硬限。
- [owner容量](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:23)仍`8W(9P+8)+48`B。W=256时默认640MiB最多容纳P=36407；同P约是W=128的两倍。P=24时为458800B；此值不包含坐标、F/G树、NTT工作区或driver/context。
- packing继续采用`slot_bits=2S+max(1,ceil(log2m))`，`slot_words=ceil(slot_bits/bpw)`，`Lntt=nextpow2(2m×slot_words+1)`，并验证`m×slot_words×(2^bpw−1)^2<q`。NTT三主数组为`24Lntt`B，表和小缓冲另计；位宽翻倍会跨离散长度拐点，不能按统一倍数推算全曲线时间。
- 点模乘的原SOS/REDC主乘累加量近似`2W²`，梅森乘积加折叠近似`W²+O(W)`；W=128→256时平方项约四倍。六模乘xADD、长除法尾部和NTT调度算法均保持，不能把更大范围称为本轮性能提升。
- 256实例的private stack显著大于旧档；cuobjdump的LOCAL=0不是“无local访存”，stack不是整个显存峰。后续需要NCU动态请求和实际分配生命周期，不能把模块自己的peak简单相加。

### 有效保存点和检查合同

[原生验收工具](D:/code/MPA-OpenCl/tools/test/test_stage2_wide_native.py:1)在CPU构造param0/sigma26/B1=20/lcm Stage1，并与独立GMP-ECM输出的完整X比较，随后保存native checksum。五个有效输入是`2^8193−3`、`2^16381−1`、`2^16384−15`、`1019(2^16374−3)`、`2621(2^16372−5)`。前面三个为该测试的单位坐标案例；后两个独立验证giant/base的非单位与proper factor回退。满16384位梅森另用synthetic X=2仅检查算术，不能作为有效生产Stage1 save。

有效单位案例独立从定义计算全部24个baby处的`Π_i(x_baby−x_giant_i) mod N`，按完整W-word叶向量比较FNV；同时GPU完整giant affine对照、seed逐字和段积检查。非单位案例不能要求Z求逆，分别要求已知proper factor1019/2621及CPU/GPU base回退一致。检查日志不是正式性能样本。

最终候选SHA256为`46457e5abd068982b2190a690c5a94d16e92728069dae15a546f6aced62d1939`，sm89/PTX3/outer0/CUDA13.3，26个raw编译依赖及对象来源冻结；最终CUDA编译264.0秒，四host对象7.1/2.9/3.9/3.7秒。初版176.2秒、第二版174.8秒也保留，不将本轮编译波动解释为GPU性能变化。

- [完整原生宽位数门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_wide_20261007/native_gate_r2/summary.json)：**27/0**，其中26条有效Stage1 save调用、1条明确标记的synthetic满limb梅森算术调用。包含原CPU路径、当前默认、GPU/CPU配对，I=2/65/66尾段、old Montgomery tail、scaled及两种旧S5下降。18组单位案例的完整monic叶向量FNV与CPU定义一致，giant affine/seed/段积逐项检查通过。
- 同一工具另过实际16384-bit **INI/worktodo队列1条**：`ECMSTAGE2=1,2,16384,-15,"three.save",13230,1,1`只取第2条并推进finished，完整叶指纹一致；plan-only返回bits=16384/words=256、执行0曲线。16385位save与队列两个负例在执行前拒绝，结果不发布、队列不改。
- 每条保留Montgomery 2048、GMP/oracle以及长除法800个fixture（含129/256-limb、dshift=0/63、quotient修正/借位回补）检查；Mersenne扩展912个fixture通过。计数是每次重复执行的诊断，不能全部相加当成独立数学样本。
- [独立点乘探针](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_wide_20261007/point_probe/summary.json)：8193/16321/16381/16383/16384位×SOS/折叠×三输出别名，共**30次调用通过**；每组64对输入、3步GMP递推，另一次故障注入正确exit1。standalone探针也增加256模板分派，不能仅放宽参数校验。
- [原有默认后端18/0](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_wide_20261007/default_point_gate/summary.json)，另在旧48b与新候选读取完全相同M4423 save/B2/D与S5-no-linear配置，旧断言失败、新完整device/slow叶比较零差异。

[静态资源比较](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_wide_20261007/native_r2/resource_comparison.json)中256模板：ladder REG64/STACK32784B，paired xADD6 REG74/STACK22544B，chain xADD6 REG64/STACK26640B，generic S4 REG46/STACK12336B、梅森S4 REG40/STACK6176B，S5 Horner REG42/STACK14352B。128-limb的18个同名kernel中stack保持，但9个寄存器数改变（含诊断kernel），例如chain6 56→64、paired6 72→74。因此**没有认证旧位宽零性能回退**，生产发布前要做完整曲线对照；不能仅因数学分派未改就宣称机器码/吞吐不变。

[最终审计](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_wide_20261007/final_audit.json)重建全部27项身份、检查raw编译依赖/保存点/log/result SHA、独立叶指纹、两次原始失败、队列/拒绝和primitive/default门禁。原始保存点、工具、源码、失败及资源报告归档在ignored `build_cuda_cmake/_stage2_wide_20261007/evidence.zip`，逐文件SHA见同目录`evidence_manifest.json`；不提交data或build目录。**本阶段正式性能样本0，没有宽位数大B2时间/总VRAM峰值结论，也没有替换发布包893。**

原失败证据保留：初版22项后因S5设备叶值未输出哈希停止，应用exit0；第二版23项后强制通用Newton在P=1断言拒绝。前者补诊断读回后的哈希，后者修正与NTT规划一致的guard；两次均新建输出目录重跑，没有删除失败案例或只按退出码/因子判断正确。

### 复现和下一生产阶段

以下命令属于58db21f当时的构建脚本；重建已测46457e应恢复其冻结26-source闭包和原脚本。当前脚本已区分production/development，按下节命令生成的新binary不是46457e。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_ecm_cuda_stage2.ps1 `
  -Build build_cuda_cmake/reproduce_wide -Arch sm_89 -GlBackend ptx `
  -OuterUnrollU 0 -SplitCompile 6
python tools/test/test_stage2_wide_native.py --prepare-only --output run/wide_fixtures
# runtime验收要求exe旁的frozen_sources_manifest.json和26个原始sources副本；
# raw SHA必须与build_manifest一致，不能把当前改过的源码当成旧binary来源。
python tools/test/test_stage2_wide_native.py --exe build_cuda_cmake/reproduce_wide/ecm_cuda_stage2.exe `
  --fixtures run/wide_fixtures/fixtures.json --output run/wide_gate
```

该宽位数阶段收尾时，[冻结生产包装CU](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_wide_20261007/native_r2/sources/src/cuda/ecm_cuda_stage2.cu:44)仍包含实验树文件。其后独立生产CU及日志控制见下一节；chain策略、大界性能、容量和成本标定继续按实际验收推进。G树/fold/下降和主机准备仍是性能重点，新二进制不能直接复用旧Auto B2 profile。

## 2026-10-07–08 独立生产源码与日志等级候选

### 源码和算法边界

[生产CU](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1)拥有实际Stage2实现；NTT和辅助头文件在 `src/cuda/stage2/`，编译闭包不再包含 `tools/bench/`。开发版改由 [ecm_cuda_stage2_dev.cu](D:/code/MPA-OpenCl/tools/bench/ecm_cuda_stage2_dev.cu:1)包含原实验引擎，原始 `stage2_tree_gpu.cu` 保持原字节。

生产运行路径选择固定PTX3/outer0、tile12/原尺寸策略、xADD6、精确梅森点乘和系数归约、通用N长除法、GPU baby、驻留fold/scaled下降与原检查调度。新增默认配对GPU seed；base非单位时仍回原seed，预算/分配失败仍有CPU baby、host fold等必要回退。C64/32768点chain门槛保留，最终短尾策略尚未重新标定。CPU base在大界没有稳定总墙钟收益，本次保留于开发版。

生产源移除了Stage1重算/prime-power链、未分批Stage2尾部、旧S5 device descent、旧batched descent运行路径、REDC尾部恢复以及probe CLI。独立慢速/GMP参考和算术fixture继续用于诊断；通用模数与非单位回退继续存在。部分NTT诊断模板和未用host辅助仍可进一步整理，不能把本次拆分理解为所有历史代码均已清零。

[配置检查](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:9105)拒绝改变上述主要生产算法的环境变量，如 `NTT_XADD6=0`、`NTT_S5_ON=1`、`NTT_S4_OLDTAIL=1`、`NTT_GIANT_SEED_PAIR=0`、`NTT_GIANT_BASE_CPU=1` 或 `NTT_FUSE_T=11`。owner、arena、batch和gleaf容量继续可调；诊断、故障注入和必需检查保留。实验A/B使用development构建。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_ecm_cuda_stage2.ps1 `
  -Build build_cuda_cmake/stage2_prod_candidate -Engine production -SplitCompile 6
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_ecm_cuda_stage2.ps1 `
  -Build build_cuda_cmake/stage2_dev -Engine development -GlBackend ptx -SplitCompile 6
```

脚本默认production/PTX3/outer0；production拒绝其他后端/outer展开。manifest与HostOnly绑定engine、CUDA source、依赖、split选项、toolkit和对象SHA；不能跨engine复用对象。CMake的Stage2目标同样固定PTX3/outer0。旧构建签名缺少engine时不能通过新的HostOnly检查。

### 日志合同

[日志实现](D:/code/MPA-OpenCl/src/core/ecm_stage2_logging.h:1)按五档过滤正常stdout；CLI为 `--log-level`，INI为 `stage2_log_level`，支持以下名字或对应数字：

- `quiet=0`：正常曲线运行不输出进度。
- `curve=1`：每条曲线开始/完成、结果概要和总时长。
- `phases=2`：增加主要阶段、形状、预算和汇总统计。
- `batches=3`：增加 `batched_progress: batch=n/G`，生产默认。
- `debug=4`：全部诊断，包含内部 `tree_level`、`descent_progress` 和算术覆盖。

CLI覆盖选定Worker INI，再覆盖全局INI，最后使用engine默认。等级传给实际curve子进程，控制台和引擎日志都生效。`--log FILE`仍控制引擎输出位置；错误继续写stderr。plan/tune的JSON是命令数据，在quiet下仍输出；dry-run仍给出所选记录。开发引擎只接受debug，保留原实验日志与开关。

日志等级不控制GMP/oracle/carry检查、结果发布或队列事务。默认batch输出没有内部树层的base/groups行。使用示例：

```ini
stage2_log_level=batches
[Worker #2]
stage2_log_level=curve
```

### 已完成验收与当前发布边界

`native_r2`独立生产候选SHA256=`19ec8a4b9e0f598e08f8ff1ccb81c330edfa55135dad26fbd7054d3651fa491c`，25个raw依赖冻结；CUDA编译117.9秒，开发构建265.8秒，均使用sm89/CUDA13.3/split6。这些是本轮构建记录，不是吞吐对照。exe为3012096B，冻结wide开发基线为5048320B。cuobjdump实际kernel数164/211，生产无旧S5 kernel、无八模乘ladder/chain实例；代码缩小不能直接换算运行加速。通用S4的128/256实例STACK由6192/12336降到3104/6176B、REG46→40，已选chain/ladder资源保持；LOCAL0不代表没有动态local访问。

[独立生产原生验收工具](D:/code/MPA-OpenCl/tools/test/test_stage2_production_native.py:1)首轮 **28/0**：14条正常曲线、实际16384-bit INI/queue一条、plan-only一条、五等级下oracle损坏五条、六个算法冲突和一个非法等级拒绝。使用上节CPU/GMP-ECM一致的有效save，单位案例检查完整24-leaf monic指纹，非单位案例检出1019/2621；含2/66点尾段、owner预算/分配与baby分配回退。quiet输出为空，Worker quiet覆盖全局debug；配置拒绝保持queue且不发布result。五个损坏输入均在各等级检出FATAL，不发布result。

[候选原生证据](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_production_20261007/native_gate_r0/summary.json)、[资源比较](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_production_20261007/resource_comparison.json)。28项小形状使用实际默认门槛，主要走直接ladder，不能冒充默认配对chain验证。另用 [比较工具](D:/code/MPA-OpenCl/tools/bench/bench_stage2_production.py:1) 与冻结46457e开发基线完成 **12/0跨版本门禁**：五个有效宽save各两侧，D30030/P2880/I32768；以及generic16384的I66305完整块＋65点尾段。单位案例逐个比较chain仿射坐标，所有案例的完整叶向量指纹、proper factors及S4检查覆盖一致，非单位base回退保持。

尾段实际采用66240点chain＋65点ladder。初版采集器误要求66305点全部出现于chain诊断，保留原拒绝、原始数据和采集器；修正按实际chunk推导覆盖数量后，恢复已成功完成的baseline原始输出，只继续运行候选尾段。没有改算法、删失败样本或重跑挑选更快结果。此12条包含额外全点CPU比较，不计正式性能样本。[链与短尾原始证据](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_production_20261007/chain_gate/measurements.json)。

补充 [入口/构建隔离工具](D:/code/MPA-OpenCl/tools/test/test_stage2_production_protocol.py:1)：**CPU拒绝6/0、GPU入口4/0**。16385位save/queue在执行前拒绝，队列及result合同保持；production拒绝其他后端/outer4，HostOnly拒绝跨engine。实际default日志验证production=batches、development=debug，CLI debug覆盖Worker quiet并传到curve子进程，固定D210的独立叶指纹保持。quiet tune测L=65536、重复2次，每次检查全部输出，JSON/profile照常发布；这是field卷积协议检查，不是Stage2性能样本。

补充工具的首次CPU检查因继承了PS7模块目录，使Windows PowerShell5.1找不到Get-FileHash；只调整构建检查子进程的PSModulePath后完整重跑。首次GPU队列检查未固定D，实际选择D420/P48却比较D210/P24指纹；工具加显式D210后重跑。原日志/工具保留，生产数学代码和检查强度未因此改变。

### 整曲线对照与容量范围（2026-10-08）

冻结开发基线46457e与独立生产19ec两侧均使用pair1/CPUbase0/C64/chain_min32768/PTX3/outer0、owner640MiB/arena6300MiB，必需检查保持。每种输入先两条预热，再执行完整ABBA＋BAAB；所有样本保留，每侧4条，无置信区间。比较的是源码拆分与生产选择，不把此前pair算法收益再次计入。

- **M4423大界**：同有效sigma26/B1=1000/lcm save、B2=2011326186870、D1381380、I1456028，总均值 **38.89674775→38.883629秒**，减少0.0337%，近似持平。前/后两组分别少0.7793%和慢0.7082%，没有稳定加速证据。NTT模块峰两侧3186.685MiB，owner609.086MiB；这些模块记录不能相加当作进程峰。[全部10条含预热数据](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_production_20261007/whole_ab/measurements.json)。
- **generic8193**：有效sigma26/B1=20 save、D30030/P2880/I32768/B2=983962980，**5.442896→5.48538925秒，慢0.7807%**；NTT模块223.905MiB，owner25.518MiB。
- **M16381**：相同D/P/I/B2、有效B1=20 save，**11.284361→11.305103秒，慢0.1838%**；NTT模块421.875MiB，owner50.641MiB。
- **generic16384**：相同D/P/I/B2、有效B1=20 save，**23.60487375→23.90637225秒，慢1.2773%**；前/后两组均回退，分别1.3589%/1.1958%。NTT模块及owner同上，不能据stack减少宣称吞吐提升。[全部30条含预热数据](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_production_20261007/wide_ab/measurements.json)。

本阶段共 **32条正式计时、8条预热**；12条额外全点诊断、28项原生验收、补充协议与profiler分别记账。所有正式输入的完整叶指纹、factor集合、实际D/I、save/Q记录身份和GMP/S4检查覆盖跨版本一致。宽位数性能范围只覆盖上述D30030形状，没有认证任意B2或更大的16k packing形状，也没有建立相对发布893的净收益结论。

管理员Nsight Systems2026对generic16384同D/P/I候选采集及导出均exit0，仅GPU1；非插桩性能结论采用上面的正式A/B。按malloc/free生命周期重建：tracked设备payload峰 **670834216B（639.757MiB）**，347次分配/347次释放、最终live=0。主机pinned峰37751304B（36.002MiB），末尾仍31989760B全局缓存，由进程退出回收；不能把pinned计为显存或声称所有主机分配显式释放。这些不是包含module/context/driver和local backing的进程完整VRAM或物理RAM峰。[分配与等待审计](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_production_20261007/nsys_wide/audit.json)。

自身GPU事件近似窗口23.8643秒，事件并集22.8013秒，无本进程事件间隙1.0631秒（约4.45%），不是整卡idle比例。该trace含启动自检/收尾：generic S4池7.1454秒、点ladder4.6566秒、chain4.6345秒、paired seed2.4968秒；不能将这些数直接套到M4423大B2的阶段比例。H2D474901264B/688次，D2H271100320B/1084次，GPU copy分别0.03754/0.02240秒；host API仍可等待此前排队工作。静态LOCAL0没有否定实际local访问，本轮没有新增NCU动态计数器结论。

### 审计、发布决定与下一项

[最终独立审计](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_production_20261007/final_audit.json)重建完整矩阵、原始log/result和工具SHA、25/27/26源闭包及对象、实际输入/覆盖、chain与ladder分派、原始拒绝、管理员trace及分配生命周期。源码、保存点、日志、工具和trace在ignored阶段目录归档，逐文件SHA见同目录evidence_manifest.json；不提交data/build目录。

提交时逐文件核对Git blob与raw编译依赖。新生产源/头采用LF；ecm_expr.cpp、ecm_queue_config.cpp、ecm_expr.h固定已有CRLF原始字节，原实验树继续保持混合换行。仅保留字节，不改这些主机函数；不能用Git自动归一化后的文本冒充已测raw来源。

**候选功能验收通过，性能无回退尚未成立；发布893保持。** 最终chain/短尾策略、较大宽形状容量、相对893的净收益、NTT诊断/未用辅助清理及新D/Auto B2成本仍待完成。现有cprof不适配19ec与配对默认。下一项优先证明并实现owner的q/qb别名复用，同时保留generic宽位数回退为后续性能诊断输入；再定位G树/fold/下降的准备、同步和热NTT形状。多曲线仍需私有状态及总RAM/VRAM lease，不能直接并发调用当前全局状态。


## 2026-10-08 Owner临时槽位复用候选

[生产布局](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5184)固定q/qb与G/reverse复用，共享overflow-checked公式 `8W(7P+7)+48 B` 用于实际分配/预算/plan；development可用 `NTT_FOLD_OWNER_REUSE=0/1/2/3` 同binary比较。生产不提供旧布局执行开关，容量/非单位等必要回退保持。详细生命周期、公式、源行、全部样本和复现命令统一维护[预算报告§9](D:/code/MPA-OpenCl/docs/STAGE2_B2_MEMORY_BUDGET_SCALING.md:339)。

M4423/D1381380/P126720/B2≈2.01e12：owner609.09→473.73MiB（−135.35MiB）。640MiB已驻留A/B均值少0.59%，两组方向不同，无稳定加速证据；512MiB原布局回退、新布局驻留，41.3191→38.4979秒（−6.83%），两组同向。16正式样本/4预热，完整叶指纹/因子/检查覆盖保持；收益范围是开发同binary预算边界，不是新生产相对893的发布加速。

候选生产de9830…98cc6a6e，CUDA119.3秒/25raw来源；CPU布局666/0、开发native39/0、独立生产native29/0。实际16384位plan实报reuse3/owner358448B。164个kernel资源记录保持，非周期证明。管理员Systems重建设备malloc峰4457.07→4321.72MiB，486alloc/free、end_live0；主机pinned峰342.26MiB保持。这些是被跟踪payload，不是完整进程VRAM/RAM峰。

当前发布893保持。新trace中NTT tile约7.97秒、无本进程GPU事件约6–7秒；不能全部解释为PCIe，不能用profile间隙差作速度证据。下一项按G树/fold/下降定位host pack/metadata/oracle/同步，并诊断generic16384已知1.28%回退；最终chain、更大宽形状、新D/Auto B2成本与多曲线RAM/VRAM lease继续推进。


## 2026-10-08 泛型 S4 常量除数寻址回退诊断

### 复现与硬件依据

本轮接续独立生产拆分后的generic16384回退，不改变NTT算法、检查频率或owner策略。GPU1/RTX4060 Laptop/sm89/CUDA13.3，使用原CPU/GMP-ECM共同确认的有效泛型save：N=2^16384−15、sigma26/B1=20/lcm，D30030/P2880/I32768/B2=983962980、pair1/CPUbase0/C64/min32768、arena6300/owner640MiB。

先对冻结46457e开发基线与19ec独立生产复现（两侧原owner布局0）：2预热、8正式ABBA+BAAB，完整输出/检查覆盖保持。full均值23.587151→23.92329175秒，慢1.4251%；设备归约6.696–6.697→6.989秒，host归约约0.024–0.028秒。增加的约0.292秒占本批full差额约87%，说明此形状应先检查设备S4，不能继续用“大B2的CPU准备瓶颈”解释所有位宽。

[管理员NCU工具](D:/code/MPA-OpenCl/tools/bench/profile_stage2_reduce_ncu.py:1)以精确NW256/generic mangled名跳过index0/grid1启动自检，捕获index1真实F树归约grid5/block128；实际CSV再次核对唯一kernel、device1及几何。16-pass replay，clock/cache control均none；完整curve仍核验输入、leaf、factor及全部检查覆盖。NCU执行时间仅诊断，不替代无插桩A/B。首版prepare因旧build manifest缺engine字段在wrapper生成前拒绝，原工具保留；改为可选字段后重建采集目录，没有放宽算术检查。

旧开发→独立生产的该launch：REG46→40，执行指令47,640,889→48,478,483（+1.76%）；local load sectors均16,832,378，local store16,845,464→16,833,145，long-scoreboard/issue-active2.306980→2.427157。采集时长8.406720→8.798464ms。更小stack/REG并没有证明更快，也没有证明local动态访存消失。

### 变更及计算量、存储与传输

两侧SASS的除法MAC都展开4项。拆分前除数从constant bank3偏移0读取；独立生产中前置point模式标量使除数偏移8，循环多出ULDC基址、UMOV及UIMAD寻址。仅移动声明顺序的候选4ce74ff未改变这段SASS，已停止其矩阵：保留2预热、2正式及被中断的原始输出，complete=false，不算完整性能结论。

[生产常量结构](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:653)将256-word除数数组放在唯一常量结构首字段，point_mersenne_bits放在其后。数组offset有static_assert；[归约上传](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2411)仍只写nw个word，[point模式上传](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:4097)用offsetof只写标量，避免互相覆盖。生产只保留这个布局，开发源码没有添加第二套算法或运行开关。

令W=ceil(bits(N)/64)、m为归约输入trim后实际limbs，商位数J=max(0,m−W+1)。长除法仍执行W·J次64×64乘减，通常m≈2W，数量仍约W²；加回修正、2-by-1估商及pack均保持。本轮仅消除每个4-limb主循环块的额外寻址，主循环将ULDC/UMOV/UIMAD三条换成USHF一条，约省2·floor(W/4)·J个寻址指令位置/系数；短尾、修正及warp分歧按实际SASS与动态计数器执行。这个近似不能当作CUDA周期公式，不能将指令数直接按峰值吞吐换成曲线时间。

固定最大容量的有效constant payload仍为8·256+4=2052B，结构按8B对齐为2056B；没有增加GPU malloc、局部数组、NTT工作区或主机pinned缓冲。每次N初始化上传8W B除数与4B point模式，接口字节量及同步合同保持。核对164个kernel的REG/STACK/SHARED/LOCAL全部一致。NW128 Mersenne S4的指令序列完全相同；NW128 ladder的57,904条指令仅5处point标量地址0→0x800，不能据此声称周期完全等价。

### 无插桩整曲线 A/B

冻结owner阶段生产de983与结构候选93c592fca16acb622a5d4b40cb03363cadf80a69ffc97306b6fcf378813e0dfc比较；两侧均固定owner reuse3，同检查/日志debug、PTX3/outer0、pair1/CPUbase0/C64/min32768。每case2预热＋8正式ABBA+BAAB，计时期间不编译、profile或分析trace；全部样本保留。新CUDA编译116.6秒/split6，25个raw依赖冻结。

- generic8193：full5.48038525→5.444167秒（−0.6609%）；两组少0.5168%/0.8052%。t_reduce四条各保持1.116→1.085秒（约−2.78%）。
- generic16384：full23.92861075→23.69737775秒（−0.9663%）；两组少1.0381%/0.8944%。t_reduce6.989→6.755–6.756秒（约−3.34%），host计时接近。没有重新对当前候选和46457e做同批正式对照，不用不同批绝对秒数宣称拆分回退已全部消除。
- M16381：full11.416966→11.3746615秒（−0.3705%），t_reduce均0.251秒；它不使用这段长除法，不能将该差额归因于寻址优化。
- M4423大界（原有效B1=1000/lcm save、B2=2011326186870/D1381380/P126720/I1456028/G12）：full38.49004275→38.85724875秒（+0.9540%）。第一组少0.1498%，第二组慢2.0643%；完整范围37.929863..39.469711秒。giant2.64325→2.6445、G树12.66875→12.6795、fold5.60975→5.621秒，下降6.96175→7.2575、inverse1.9335→1.9835秒；这些是包含准备/等待的phase墙钟，非纯kernel。设备S4约1.782–1.783→1.783–1.802秒。不能据阶段接近断言大界无回退，也不删除39.469711秒样本。

此候选矩阵32正式/8预热，均完整leaf指纹、factor、实际D/I、save/Q身份、NTT乘法/归约数量及GMP检查覆盖一致。前述回退复现另8正式/2预热；声明排序的中断矩阵另列，不混进均值。数据位于ignored build_cuda_cmake/_stage2_diag_20261008/ 的repro_generic16384、struct_wide_ab、struct_large_ab。

### 候选NCU、门禁与发布边界

结构候选捕获同一个实际NW256 S4 launch：REG40保持，执行指令降到47,640,770、thread指令438,503,331→431,057,723、local load仍16,832,378 sectors，local store16,847,502，long-scoreboard2.350604；采集8.481216ms。这支持多余constant基址寻址是设备回退的重要来源；没有降低MAC次数或local load数量，剩余串行依赖/local访问仍需优化。三个原始ncu-rep、CSV、命令、源/工具SHA及完整应用输出均保留。

原生生产门禁29/0已完成，含有效宽save、非单位1019/2621、短尾、容量/分配回退、五级oracle故障及queue拒绝。默认chain/完整块＋短尾跨版本门禁12/0完成：五个有效宽save两侧及66,305点完整块＋65点尾段；单位案例逐点仿射比较，非单位保持proper factors与回退。12条额外诊断不计入正式性能样本。

**发布893保持，候选尚未认证完整无回退。** 宽泛型减少寻址的收益已有重复和计数器证据；大界下降/逆元的波动仍须定位，随后推进G树/fold/下降准备与热NTT、最终chain策略、较大16k容量、新D/Auto B2成本和总RAM/VRAM lease。本轮不改变Auto B2门槛或导出新cprof。

本阶段独立审计从全部原始log/result重建40正式样本、10预热、12条chain诊断、29项原生门禁及3次管理员NCU采集，另列声明排序中断矩阵。审计核对对象/25个当前raw依赖与Git暂存字节、输入/完整输出/检查覆盖、采集及收集工具快照、实际kernel选择与三路计数器；证据及逐文件SHA归档于同阶段目录evidence.zip/evidence_manifest.json。build/data继续忽略。复现工具与候选已入阶段提交，发布exe字节保持893。


## 2026-10-08 驻留 H 的 GPU Gamma 校正

### 准备瓶颈与算法

接续上一节的93c592候选，先定位fold结束到scaled descent的准备。独立初始Systems trace（GPU1、原有效M4423/B1=1000/lcm、D1381380/P126720/I1456028/G12）记录自身GPU事件span37.888747秒、并集30.783450秒、无自身事件7.105296秒。最大0.842465秒间隙位于fold收尾到下降首个大上传；日志flat→CPoly桥接0.116445秒、CPU Gamma校正约0.689秒，与它接近。正常trace匹配event的API完成后延迟最大约173微秒，不能解释约7秒间隙；这不排除未捕获的Windows长尾，也不是整卡idle证明。F逆元已从finvflat恢复复用，不是重复计算逆元。

原[CPU校正](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7598)对每个H系数执行GMP乘法/mod N。令f=Gamma⁻¹（已有Ginv，不能再次求逆）、W=ceil(bits(N)/64)、R=2^(64W)：一次在CPU编码fR mod N，随后[新内核](D:/code/MPA-OpenCl/src/cuda/stage2/scale_plain.cuh:6)执行Mont(H_i,fR)=H_i f mod N，输入/输出始终为普通域。radix取实际W，不是模板容量NW，也不是2^bits(N)。沿用已验证的s2g_mont_mul；精确梅森沿用乘积折叠/radix旋转，泛型用SOS/REDC，奇数及canonical系数合同保持。

[FoldDeviceState::scale](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5334)在最后fold完成后、finish读回/释放之前使用result allocation里已闲置的q槽保存W-word标量，H仍在独立source allocation中原地更新。此前NTT gather、oracle独立快照和reverse/subtract按default stream排序，不让S4输入与输出别名。新乘法完成后同步，再走原finish读回、bridge及下降。没有增加NTT乘法、S4归约或改变root输入digest。

生产[调用](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7578)固定请求GPU；owner未活跃（包括G1、预算/分配回退）、空H或f=1时保留原CPU/无操作路径。成功GPU校正后跳过CPU整系数循环。开发保留NTT_GSCALE_DEVICE=0/1，默认0；生产只接受1，不保留CPU/GPU算法选择分支，CPU仅为必要回退。NTT_GSCALE_DEVICE_CHECK=1逐系数检查完整H；TEST_BAD会真正改写一个设备输出word，必须同时打开检查。原语TEST开关只在开发引擎使用。

### 计算、容量与搬运

用C表示活跃H系数数，多轮fold正常C<=P；每系数一条线程、block128，grid=ceil(C/128)，一条GPU校正launch。梅森主乘累加约C·W²+O(CW)，泛型约2C·W²+O(CW)；标量编码另有一次GMP移位/mod N。不能将这些数量直接换成GPU周期或将CPU与GPU的MAC速度等同。

正常路径无额外cudaMalloc或持久NTT/owner数组，owner仍8W(7P+7)+48B。标量H2D额外8W B，H的原最终D2H量8WC B保持；owner传输统计包含标量，逻辑avoided计数不增加。主机临时scalar/hn向量payload16W B，加GMP临时空间；该数不是进程RAM峰。新kernel对H的接口读取/写入各8WC B，模数/标量读取有重复，实际cache/DRAM字节应由计数器测量，不能只用接口payload代替动态访问。

当前sm89/CUDA13.3编译的NW=4/8/16/32/64/128/256七档均REG40，STACK分别112/208/400/784/1552/3088/6160B，即本构建24NW+16B/线程；SHARED/LOCAL均0。STACK仍可形成local访存和driver backing，不能把LOCAL=0或无新增cudaMalloc写成显存零成本，也不能将C·STACK当作进程显存峰。

诊断完整H检查另读回before/after各8WC B、主机两向量payload16WC B；毒化另D2H8B/H2D8B。这些诊断运行不是正式样本。M4423大界C126720/W70，正常只新增560B上传；完整H检查8,870,400 words，额外135.352MiB诊断D2H及相同向量payload。

### 数学、入口和正式计时

开发[门禁/同二进制工具](D:/code/MPA-OpenCl/tools/bench/bench_stage2_gscale.py:1)首轮21/0：五个CPU/GMP共同验证的宽save，两侧完整leaf/factor/NTT-S4覆盖一致；单位例完整monic叶与独立定义一致，非单位1019/2621保持回退。完整H逐word对照GMP，G1/短尾、预算/分配回退、真实设备毒化、不带检查的毒化与非法flag均覆盖。另两条大界诊断完整检查126720个H系数，CPU/GPU完整leaf一致；补充六条32,768点宽形状诊断（gate-wide），校正后的整个H与GMP逐word一致，跨模式完整leaf保持。

原语补测12条运行，六个有效61/509/521/1279/2203/8191位CPU/GMP Stage1保存点，各point模式0/1；与宽输入合起来覆盖全部七档模板容量。每次原语fixture有12个scalar/count组合，因子0/1/N−1/7、count1/2/129，数据含0/1/N−1和确定性随机数。最初509位候选因Stage1 Z不可逆在准备阶段拒绝，GPU曲线0；保留primitive_r0原文件/工具/拒绝记录，r1确定性选合法奇数offset并核对GMP坐标，没有修改算术。

独立生产66bb451cf3be2d4eac9f20306908e253f5f9eeea5457d03f602d90963c5a78aa编译CUDA114.7秒/split6；26raw依赖冻结，不依赖tools/bench。扩展[原生门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_production_native.py:1)用--gscale-check，36/0，包括完整H、宽save、非单位、必要回退、实际INI/queue/plan、五级日志下oracle及Gamma毒化拒绝；生产旧CPU算法覆盖设置及开发原语TEST设置在队列事务之前拒绝。这是检查项数，不是36条完成的GPU曲线。

同二进制开发大界每侧一条预热＋8正式ABBA+BAAB，保持PTX3/outer0、pair1/CPUbase0/C64/min32768、owner640/reuse3/arena6300、factor-only及默认必需检查。full38.63585175→38.318530秒，减少0.8213%，两组分别少1.2353%/0.4017%。GPU校正正式0.018213..0.019313秒，上传560B、额外诊断D2H0。下降均值6.99775→7.326秒、inverse1.96825→2.03975秒，其他phase接近；局部CPU节省没有全部转化成整曲线收益，不叠加前面各轮百分比。

独立生产93c592→66bb451宽输入，每case2预热＋8正式ABBA+BAAB、相同配置/默认检查/debug日志。generic8193 full5.4490465→5.40522475秒（−0.8042%），M16381 11.44266175→11.29162375秒（−1.3200%），generic16384 23.72155075→23.586632秒（−0.5688%）。完整leaf/factor与NTT/S4覆盖保持；这批形状的收益不能外推任意B2/位宽或替代较大16k显存容量验收。

宽输入的两组减少比例分别为generic8193的0.5829%/1.0257%、M16381的1.4215%/1.2180%、generic16384的0.5663%/0.5712%。对应CPU/GPU gscale均值（日志毫秒精度）为0.05125/0.004、0.1485/0.008、0.14875/0.015秒。生产大界另2预热＋8正式ABBA+BAAB：full38.5815745→38.11755575秒（−1.2027%），两组少1.4950%/0.9067%；CPU gscale0.83525秒，GPU约0.018秒。开发同二进制CPU gscale均值0.83725秒；与初始trace的约0.689秒属于不同批次，不用它们混算加速。

开发与生产合计40正式/10预热，全部样本保留；诊断、失败准备及profiler单列，不混入性能均值。开发原语r1为12/0；完善源/辅助工具及GMP保存点身份绑定后，最终工具r2又12/0。首轮gate及正式矩阵使用各自原始collector快照，不把修改后的工具SHA替换进旧记录。当前生产164个既有kernel的REG/STACK/SHARED/LOCAL全部保持，新增7个plain-scale模板；这是资源比较，不是完整SASS或周期等价证明。

### 管理员Systems/Compute与准备空隙

同开发binary的CPU0/GPU1依次独立采集，PTX3/outer0、pair1/CPUbase0/C64/min32768、owner640/reuse3/arena6300、默认检查、CUDA_LAUNCH_BLOCKING=0。两次Systems2026.1.3 capture/export/collector exit0；不在采集中编译或分析另一个trace。按GPU1/kernel/memcpy/memset事件和SQLite实际字节量核对，fold最后16B digest读回结束到下降第一个70,963,200B上传的自身事件间隙0.752876493→0.130642568秒，少0.622234秒；候选其前有一次plain-scale，实际grid990/block128、17.975964ms。CPU/GPU capture日志gscale0.608/0.018秒、bridge0.101604/0.091005秒，与移除该CPU准备的判断吻合。它是定位证据，不是第二套无插桩正式提速。

整个自身事件span38.107019→36.587057秒、并集30.853124→30.864237秒、无自身事件7.253895→5.722820秒（19.04%→15.64%）；其他准备和调度也有变化，不能把整段1.531秒差全部归因Gamma。候选仍有约0.326秒、0.236秒的其他准备间隙。event-sync完成后API内延迟最大0.267/0.140ms；仅覆盖已匹配的正常event，不作为未捕获长尾的结论。

实际H2D4866/7,151,041,091B→4867/7,151,041,651B，恰好多一次560B；D2H两侧6131/3,209,120,312B，D2D40/841,498,560B。两个tracked设备payload峰均4,531,652,368B（4321.720474MiB），各486 alloc/free，end_live0；pinned峰358,886,600B（342.260933MiB）及末端缓存357,255,360B保持。NVML200ms两侧221/212条GPU1记录，采样整卡峰均4989MiB，包含driver/context/private backing；不是连续或任意形状的进程峰保证。没有把NTT/owner/pinned模块峰相加。

管理员Compute2026.2.1用扩展[采集器](D:/code/MPA-OpenCl/tools/bench/profile_stage2_reduce_ncu.py:1)的--kind gscale，从已完成同binary矩阵绑定实际NW128、grid990/block128/device1和完整输入/leaf/检查覆盖，16-pass kernel replay、clock/cache control none。实际17.992800ms，REG40/分配40、active warp43.516（90.66%）、issue16.21%。local load/store分别168,566,024/166,345,753 sectors，lg_throttle/issue_active49.2453、long_scoreboard12.6640，GPU DRAM throughput24.50% peak。sector×32仅是累计local访问代理，不是DRAM/PCIe/显存容量；不乘16passes，stall ratio也不是墙钟占比。replay警告备份设备数据到系统内存，采集时间/内存不能用于生产容量认证，完整leaf与覆盖已在collect中核对。

首次NCU prepare因生成的锚定regex含caret被wrapper安全guard拒绝，GPU曲线0；保留ncu_gscale初版工具/拒绝。r1移除不必要caret、保留模板绑定，并再次验证实际kernel/维度/设备，capture/collect正常。没有绕过安全guard或以没有实际kernel的报告通过。

本轮证据在ignored build_cuda_cmake/_stage2_prepare_20261008，开发/生产源码、工具、输入、原始门禁、全部矩阵和trace分别冻结。发布893保持，没有新Auto B2成本文件；候选支持手动B2/16384位，但较大16k、最终短尾、D/Auto成本及整个发布资格仍须推进。

下一轮优先将下降根部的H反转/组包与已缓存finv结合，考察能否在owner释放前完成根乘法，减少当前H读回、flat→CPoly桥接和重新上传；先证明生命周期及完整普通域叶值，不重复求F逆元。再按真实热长度、batch和树层次优化tile/outer。新Gamma内核只占本批约0.05%全曲线，不优先继续微调它；旧pinned/context/global tile11负结果保持，多曲线仍需私有状态与总RAM/VRAM lease。

复现：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_ecm_cuda_stage2.ps1 `
  -Build build_cuda_cmake/gscale_dev -Engine development -GlBackend ptx -SplitCompile 6
python tools/bench/bench_stage2_gscale.py --exe build_cuda_cmake/gscale_dev/ecm_cuda_stage2.exe `
  --fixtures build_cuda_cmake/_stage2_wide_20261007/fixtures_r2/fixtures.json `
  --mode gate --output run/gscale_gate
python tools/bench/bench_stage2_gscale.py --exe build_cuda_cmake/gscale_dev/ecm_cuda_stage2.exe `
  --fixtures build_cuda_cmake/_stage2_wide_20261007/fixtures_r2/fixtures.json `
  --save build_cuda_cmake/_fixed_d_20261005/native_accept/m4423.save `
  --mode timing --output run/gscale_timing
```

生产编译用-Engine production，原生验收用test_stage2_production_native.py --gscale-check；跨生产比较用bench_stage2_production.py。各输出使用新目录；profiler由管理员启动，prepare先绑定完成矩阵，再用生成的capture.ps1采集，--collect-only验证。原始save必须保留自身CPU/GMP来源，不能用任意X代替性能输入。

本阶段[独立最终审计](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_prepare_20261008/final_audit.json)重新解析所有原始result/log、正式顺序与检查覆盖，核对28/26个开发/生产raw依赖、对象、原始工具及拒绝记录、actual profiler kernel、传输和生命周期；complete=true。源码/工具/输入/日志/矩阵/trace归档于同目录evidence.zip及evidence_manifest.json，exe/obj/DLL通过外部SHA绑定，不内嵌。发布893字节保持；阶段结题不等于长期Stage2优化或整个发布资格完成。

## 2026-10-08 驻留 H/finv 的下降根准备（开发候选）

### 路径与合同

接续 b52eea0 的 GPU Gamma 校正，新增开发开关 `NTT_SCALED_ROOT_DEVICE=0/1`，默认0。当前独立生产 CU 和发布893尚未接入这一候选。本轮用同二进制、相同Gamma设备路径比较根准备，不混用两次编译的GPU代码。

令W=ceil(bits(N)/64)、P=deg F、C=H的活跃系数数。下降根仍为 `prefix_P(rev_(P-1)(H) * (rev_P(F))^-1)`，普通域不变，缓存finv已存在，没有再次求逆。只有owner活跃、0<C<=P、scaled descent启用且Gamma已经在GPU校正（或Gamma=1）时使用新路径；G1/根仍需除法、预算/分配/后端回退和CPU Gamma保留原路径。

[反转内核](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7621)按word映射，H不足P个系数时高位补零；输出复用最后fold后闲置的G/reverse槽。随后[FoldDeviceState::scaled_root](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7846)用已有S4ResidentOwner/S4DeviceBatch，从source allocation中反转H和finv前P项直接gather到NTT，乘积prefix P写入独立result allocation中闲置T槽。三项word-offset元数据上传24B，输入/输出不别名、边界及owner租约仍由原S4入口检查。

[收尾选择](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10448)读回P项根状态，再读已有digest、释放owner；正常不再读回H或恢复H/finv的CPoly。完整GMP节点/传统下降检查开启时，额外保留H供oracle使用。根状态交给[descent_scaled](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8029)，跳过原根组包/乘法，随后子层算法保持。根乘法仍计入descent的mul_calls/mul_pairs及t_descent；`scaled_root_device.seconds`包含本次反转、NTT/S4、根读回及诊断（如果开启）。旧real_batched_wall的pre/loop/post边界仍不包含所有下降前准备，不能代替stage2_full_wall，也不能与子阶段直接相加。生产移植前应整理这项统计边界。

### 容量、计算与搬运

没有新增持久CUDA数组，owner仍为8W(7P+7)+48B，生命周期延长到根输出读回完成。NTT根乘法次数、长度和S4输出窗口保持。新增反转读取8CW B、写入8PW B，O(PW)个word映射；resident输出scatter还读/写各8PW B。这些是GPU接口payload，不是DRAM计数或CUDA D2D API字节量。

正常路径原H读回8CW B、原根操作数上传16PW B被移除；根结果8PW B读回两侧均保留。净节省H2D为16PW−24B，D2H为8CW B。根主机向量payload8PW B仍存在，原finvflat在根准备时暂时保留8W(P+1)B、随后释放；H/finv的CPoly恢复、H副本及根A/B主机组包被跳过。不能把这些模块payload简单相加当作进程RAM峰，vector描述符、allocator、pinned cache与其他阶段还有各自生命周期。

`NTT_SCALED_ROOT_CHECK=1`读回完整H和finv，CPU独立反转/补零，再走原NTT/S4根乘法并逐word比较整个根，额外输入诊断D2H为8W(C+P)B且多一次根乘法。小规模另用NTT_SCALED_CHECK对所有节点做独立GMP长除法/三角求解；大规模完整根比较共享既有NTT/S4算术，不能称为全根独立GMP计算。TEST_BAD真实异或一个设备根word，另D2H/H2D各8B，必须带CHECK；所有诊断排除正式计时。

### 验证与固定顺序计时

开发候选SHA256为215f7cfe9fee3183272a94717a42903fdbfd6d897d4d69f32ee6dd3fef218a7c；CUDA编译265.5秒/split6，28个raw依赖和5个对象冻结。追加Auto B2范围检查后复用已验证CUDA对象重编译host；[旧成本配置](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:547)拒绝启用新根准备，没有生成新成本文件。

[同二进制工具](D:/code/MPA-OpenCl/tools/bench/bench_stage2_gscale.py:1)增加--target scaled-root：33项初始门禁，11个有效CPU/GMP Stage1输入覆盖七档模板，包括已知非单位因子1019/2621、短2/66、预算/分配回退、实际根毒化和非法flag；小规模完整scaled节点及独立monic叶值通过。大界另2条完整根比较，检查126720项/8,870,400 words；四种owner mask补查5条，三宽32768点形状补查6条，合计46个引擎检查项。诊断多出的1次S4 launch/1对poly_mul/P个归约系数在工具和最终审计中单独核对；不要求它们与无诊断侧的计数相同。

[反转映射工具](D:/code/MPA-OpenCl/tools/test/test_stage2_scaled_root_reverse.py:1)从实际编译冻结的CU抽取内核体，单独验证216例/593,024 words：W=1/8/9/20/35/70/128/129/256，P=1/2/3/17/129，短/满H、零数据、同一分配内不相交输入输出及保护区。它证明映射/补零边界，不替代完整引擎算术或生产资源检查。

正式矩阵共32样本/8预热，每case2预热后8条ABBA+BAAB，默认必需检查、PTX3/outer0/pair1/CPUbase0/C64/min32768/owner640/reuse3/arena6300/factor-only。完整叶指纹、factor和NTT/S4检查覆盖保持，全部样本保留。M4423/B1=1000/lcm/sigma26、D1381380/P126720/I1456028/G12：full37.73597125→37.28418250秒（本批少1.1972%），两组少0.7785%/1.6149%。descent均值6.9655→6.68625秒（已计入GPU根准备），候选根准备均值0.184037秒；inverse1.921→1.89975、F树6.781→6.752也有变化，不能把整个full差全部归因局部搬运。

宽有效输入B1=20/sigma26、D30030/P2880/I32768：generic8193 full5.40459925→5.39263450秒（少0.2214%，两组0.1824%/0.2603%）；M16381 11.28936775→11.25344300秒（少0.3182%，两组0.0597%/0.5761%）；generic16384 23.50315325→23.47909875秒（少0.1023%，两组0.0631%/0.1415%）。宽形状的收益很小，不外推任意B2/硬件或作为较大16k容量认证。

### 管理员profiler与后续

Systems2026.1.3对照独立于正式矩阵。大界H2D从4867次/7,151,041,651B变为4866次/7,009,115,275B，恰少141,926,376B=16PW−24；D2H从6131次/3,209,120,312B变为6130次/3,138,157,112B，恰少70,963,200B=8CW。D2D两侧40次/841,498,560B保持，新增scatter是kernel，不计为DMA API。H2D/D2H DMA累计时间分别0.543515→0.531658、0.251965→0.246535秒，不能与CPU准备时间混为一项。

两侧tracked设备payload峰均4,531,652,368B（4321.72MiB）、486次alloc/free、end-live0；pinned峰358,886,600B（342.26MiB）保持。200ms NVML分别217/213条GPU1样本，整卡采样峰均4989MiB，含driver/context，不是连续进程峰认证。自身事件span37.434867→36.961554秒、并集30.852007→30.836865秒、无自身事件6.582860→6.124688秒（17.58%→16.57%）；这是诊断采集，不作为另一个正式提速百分比，也不是整卡idle证明。

候选Systems实际反转一次，grid34650/block256、REG18、0.558319ms。Compute2026.2.1选择同一真实kernel/维度/设备，16-pass replay、clock/cache control none，完整输出与参考保持；反转0.606720ms、REG18/分配24、active warp76.61%、DRAM234.51GB/s、local load/store sectors均0。replay备份设备到系统RAM，时间/峰不能替代生产容量。反转只占约0.002% full，不值得优先继续微调。

本次重编译开发kernel218→219；除新增反转REG18/STACK0/SHARED0/LOCAL0，既有block-product NW256 REG56→48、NW64 REG48→56，STACK保持8208/2064。没有修改它们的源码，但不能声称既有资源完全不变；同二进制A/B避免把这些codegen变化计入根算法收益。

证据统一在ignored build_cuda_cmake/_stage2_scaled_root_20261008。原始collector、全部正式/诊断输出、输入、冻结源码/对象身份及profiler分别保存；最终审计final_audit.json和evidence_manifest.json核对原始字节与覆盖。下一步整理准备计时边界并移植到独立生产CU、做生产门禁/对照；再优先热点NTT（本trace tile约7.99秒、两类outer_coop共约4.39秒），而非0.6ms反转。较大16k、最终chain/短尾、新D/Auto成本与并发总RAM/VRAM lease仍未完成，发布893保持。

复现（每次输出使用新目录）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_ecm_cuda_stage2.ps1 `
  -Build build_cuda_cmake/scaled_root_dev -Engine development -GlBackend ptx -SplitCompile 6
python tools/bench/bench_stage2_gscale.py --target scaled-root `
  --exe build_cuda_cmake/scaled_root_dev/ecm_cuda_stage2.exe `
  --fixtures build_cuda_cmake/_stage2_scaled_root_20261008/fixtures.json `
  --save build_cuda_cmake/_fixed_d_20261005/native_accept/m4423.save `
  --mode timing --output run/scaled_root_timing
```

## 2026-10-08 驻留下降根生产接入与完整收尾计时（候选验收）

### 实现与当前证据

生产 [scaled_root_reverse_kernel](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5133) 和 [FoldDeviceState::scaled_root](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5356) 已接入独立CU；沿用上一节的普通域根公式、死G/reverse与T槽、24B元数据和必要回退。生产固定请求驻留根，不提供旧根算法的选择开关；开发保留同二进制0/1对照。生产仍无tools/bench编译依赖，五级日志/错误及队列合同保持。

初版production_r0为ed977be379d1369523aff8a6ded08984bb728f7fab70b0fa3d0fda873939e7c1，CUDA126.0秒/split6；26raw依赖、5对象冻结。初版原生47/0，追加六个CPU/GMP Stage1位宽后完整53/0；小规模完整根和独立GMP scaled节点、非单位1019/2621、短2/66、预算/分配回退、五级实际毒化和queue保护通过。反转映射216/0、593,024 words，取自实际冻结生产CU；原有171个kernel静态REG/STACK/SHARED/LOCAL保持，新增反转REG18/其余0，不能据此声称整个机器码或周期保持。

初版跨生产chain/短尾诊断12/0，包含完整块加65点尾段；该矩阵带完整仿射检查，属于正确性证据。补充小门禁在该诊断期间执行，均非正式性能样本。开发计时回归被主动中断并保留不完整记录，后续改为串行重跑；没有把它纳入已完成门禁。正式大界矩阵在候选预热后被计时总账门禁拒绝，正式样本0，不能发布本阶段提速比例。

### 完整计时边界

`post`从最后fold循环结束后、Gamma与根准备之前开始；新增`real_batched_prepare.post_fold_to_descent`覆盖该准备窗口，根乘法继续记入descent。两个子项有重叠，不应相加成互斥阶段；整曲线仍用stage2_full_wall。

首条M4423候选预热原始日志显示pre2.327＋loop21.156＋post6.728＝30.211秒，而调用方elapsed30.25秒（精确main30.247563）。全叶指纹4244971527793015097及必需算术检查通过；collector按原0.03秒门限拒绝。原计时终点位于run_batched返回之前，漏掉了局部资源析构、最终oracle drain和结果合并等返回窗口；本次oracle累计drain仅0.000739秒，不能解释全部差额。未放宽门限，也未删掉失败预热。

修正为在run_batched返回之前保存post终点，再在调用方同一main终点测量`return_finalize`并加入post；开发的两个调用点及生产调用点一致。该值是整个返回/最终化窗口，不把它全部称为cudaFree时间；无需依赖C++ NRVO。第一修正版production_r1完成M4423复现，逐word检查8,870,400根word通过，return_finalize=0.022120秒，但pre/loop/post仍与调用方相差约0.02秒。源码另有chunk内临时对象在原单块timer结束后析构；第二修正将整个giant循环改为一个连续计时区间，和pre/post共用边界，再重新编译验收。上述带诊断复现不作为性能样本。 增加与stage2_full_wall.main六位小数值比较的0.003秒门禁，初版缺0.036563秒、第一修正版仍缺0.018456秒均被该精确边界拒绝；保留原elapsed两位小数的0.03秒检查。[完整根检查工具](D:/code/MPA-OpenCl/tools/test/test_stage2_production_root.py:1)绑定完成的正式矩阵，逐word对照独立CPU组包后的旧NTT/S4根，核对额外1次乘法与P个归约系数；大规模算术共用NTT/S4，不能称为全根独立GMP证明。

连续区间最终候选production_r3 SHA256=fbfc9d24ae6ec6c8baf420258cfc13d750893d3d2e2999adf9e78d62caf7ab9f，CUDA134.4秒/split6、26raw依赖；development_r2 SHAd4116c5163d39b8bbda69fa7343e6e9b6155cbacc165497af9dfd9d6ae37c555，CUDA282.0秒、28raw依赖，均5对象且原始字节冻结。生产CU新字段曾引入一个CRLF，按既有LF规则规范化后另编final候选，未用修改后的文件冒充旧构建来源。初版与最终版172个kernel的完整cuobjdump SASS输出逐字节相同（每份341,269,914B）；这是计时修正未改变GPU指令的证据，不是host机器码、周期或性能证明。

最终大界诊断完整8,870,400根word通过，pre＋loop＋post=31.024秒，精确main=31.024070秒，差额0.000070秒；return_finalize=0.016387秒。3毫秒精确门禁通过；新增诊断根乘法计数也保持预期。最终生产七档原生53/0、开发回归33/0和反转映射216/0已完成，包含精确计时总账；跨生产chain/短尾12/0也已完成；32正式/8预热矩阵已按串行计划完成，结果见下一节。这条带完整根检查的诊断不作为性能样本。

### 正式计时与完整根补查

最终生产66bb451→fbfc9d24；每形状先baseline/candidate各一条预热，再8正式ABBA+BAAB，合计32正式/8预热。PTX3/outer0/pair1/CPUbase0/C64/min32768/owner640/reuse3/arena6300/factor-only/debug及默认检查保持，计时期间没有编译、profiler或重型trace分析。每版每形状4条正式，每组每版2条；没有置信区间，所有样本保留。

- M4423大界：full均值37.82075925→37.73831500秒，减少0.21799%；两组分别增加0.32797% / 减少0.75310%。
- generic8193：full均值5.40173025→5.40615025秒，增加0.08183%；两组分别减少0.08831% / 增加0.25281%。
- m16381：full均值11.30973900→11.25109200秒，减少0.51855%；两组分别减少0.80979% / 减少0.22699%。
- generic16384：full均值23.56061150→23.53559075秒，减少0.10620%；两组分别减少0.00155% / 减少0.21073%。

大界为M4423/B1=1000/lcm/sigma26、D1381380/P126720/I1456028/G12；三宽为有效B1=20/sigma26保存点、D30030/P2880/I32768。完整叶指纹、因子及NTT/S4工作量与默认检查覆盖保持。大界和generic8193两组方向相反，其他两形状的差异也很小；本阶段没有建立普遍稳定整曲线提速，不能用上一节开发同binary约1.20%替代生产结果。

原始样本和固定顺序分别见[大界矩阵](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_root_prod_20261008/cross_timing_final_r3/measurements.json)和[宽位矩阵](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_root_prod_20261008/cross_timing-wide_final_r3/measurements.json)。完整根工具绑定这两个完成矩阵，4/0，分别检查8,870,400 / 371,520 / 737,280 / 737,280 words，共10,716,480；最终叶值/因子保持，额外一条根乘法、一次S4 launch及P个归约系数逐项核对。它们是诊断，不混入正式计时；大规模root算术仍共用NTT/S4，小规模逐节点另有独立GMP证明。

### 管理员profiler、容量与热NTT形状

Systems2026.1.3两侧顺序capture/export、Compute2026.2.1 capture/collect均exit0；原验收进程真正退出、全部32正式/8预热及4根诊断完成后才启动采集。所有capture完成后再进行离线trace分析，工具/输入/binary/完整输出和检查覆盖均绑定冻结来源。

实际H2D 4867次/7,151,041,651B→4866次/7,009,115,275B，少141,926,376B；D2H 6131次/3,209,120,312B→6130次/3,138,157,112B，少70,963,200B，分别精确对应16PW−24和8CW。D2D两侧40次/841,498,560B保持；reverse/scatter是kernel，不伪记为DMA。H2D累计0.542662→0.531460秒、D2H0.251822→0.246066秒，接口搬运节约并不等于相同数目的CPU等待或整曲线节约。

两侧tracked设备payload峰均4,531,652,368B（4321.720474MiB）、486alloc/free、end_live0；pinned峰358,886,600B（342.260933MiB），末端缓存357,255,360B保持并由进程退出回收。200ms NVML 216/213条GPU1样本，整卡采样峰均4989MiB；模块/driver/context/local backing与其他进程不在malloc payload内，采样也不是连续峰值保证。没有将模块容量相加作为进程峰值。

自身GPU事件span37.596170→37.088031秒、并集30.869870→30.844228秒、无自身事件6.726300→6.243802秒（17.89%→16.84%）。这是诊断采集窗口，不作另一套正式提速或整卡idle证明；正常匹配event-sync的API内完成后延迟最大约0.163/0.169ms，覆盖6023/6006次API，不能解释或排除未捕获的长尾。

Compute实际捕获唯一reverse：device1/grid34650/block256、0.611008ms、REG18/分配24、active warp76.80%、DRAM232.54GB/s、local load/store sectors0。16-pass replay、clock/cache control none，保留系统RAM备份警告；该时间/备份容量不用于生产速度或容量认证。完整leaf/factor和S4覆盖与已完成正式参考保持。

候选Systems tile累计7.991461秒、两类outer_coop合计4.386704秒。新增只读形状统计从原始demangled名字、grid和dynamic shared重建110组，绑定实际冻结launcher：tile的N=gridX·(dynamicSharedMemory/8)，cooperative outer的N=gridX·2^M·V（M8取V16，其余V32），nbatch=gridY；这是有源码依据的几何推导，不声称捕获了kernel参数或树phase。Nsight实际名字含(int)/(bool)注解，采集前用已完成trace的表结构/单行元数据纠正parser并保留原SHA，不改任何正式样本。

- N=2^27/batch1/t12：tile forward74次/1.015524秒，inverse37次/0.685929秒；M8与M7 outer正逆合计2.728562秒。它是重要热形状，但不是全部NTT成本。
- N=2^11/batch990/t11：tile forward2560次/0.511074秒，inverse1280次/0.326817秒；需要独立批量形状门禁，不能只测最大N的单batch。

完整形状记录见同阶段ntt_hot_shapes.json。累计kernel时间包含启动检查/收尾，不能直接归属某棵树或与重叠phase墙钟相加。

独立审计已重建全部原始门禁/样本、真实D/P/I和输入身份、两组统计、计时总账、四个完整根、26/28个raw来源与对象、管理员capture及生命周期。初次审计错误要求开发CPU对照也有逐节点GMP覆盖；保留拒绝与工具，最终严格核验17条实际开启完整GMP节点检查的开发调用，其余CPU对照的checked计数为0，并核对完整leaf与诊断额外乘法。没有削弱算术门禁或重新挑选样本。



### 后续与发布边界

本阶段功能与搬运验收通过，未建立普遍稳定整曲线提速。证据保存在ignored build_cuda_cmake/_stage2_root_prod_20261008；初版、失败预热、主动中断、首次审计拒绝与工具快照保留。[最终独立审计](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_root_prod_20261008/final_audit.json)绑定全部原始证据，evidence.zip/evidence_manifest.json保存来源、输入、日志、矩阵和trace的逐文件SHA；exe/obj/DLL在磁盘通过外部SHA绑定。Git提交前另外核对当前26/28个raw编译来源与暂存blob的逐字节身份。发布893保持，没有新Auto B2成本或较大16k形状认证；阶段验收不等于长期优化目标或完整发布资格完成。

下一阶段先对实际N=2^27/batch1的M7/M8与正逆tile、N=2^11/batch990的tile进行管理员NCU诊断，按带宽/依赖/同步证据选择候选。另一个源码可行候选是将scaled下降frontier继续保留在GPU：两份owner的较小allocation容纳3PW words，可放当前frontier的PW和本层兄弟F输入最多2PW，另一份写下一层；需新增最多24P B metadata、严格预算及必要回退。若第l层真正乘法的A长度总和为M_l、输出系数总和为J_l，则相对当前根路径可省H2D 8W∑M_l减新增metadata、D2H 8W∑J_l；这是生命周期/接口量推导，尚未实现或测量。应由实际下降准备成本决定其相对NTT的优先级。

完成本阶段后优先实际热点NTT。上一阶段大界trace中tile约7.99秒、两类outer_coop共约4.39秒；根反转仅约0.6毫秒。按实际length/batch/正逆变换分布选择候选，不重复全局tile11或仅凭单batch field最快配置推广。较大宽形状应使用精确P=phi(D)/2：D300300给P28800，W256 owner约393.764MiB；D600600给P57600、owner约787.514MiB，不能误当作640MiB以内驻留。上述两档仅为公式计算，尚未实际大形状运行/峰值认证。


## 2026-10-08 接续：实际热NTT诊断完成

驻留根阶段已提交930dc3a。上述后续热形状NCU已完成：8个实际kernel、完整输出/检查与GPU指令身份独立核验通过；直接skip/invocation筛选的异常开销与隔离host range修复均保留。tile active warp96–99%、REG40/local0；M7/M8 outer active warp约33%、shared容量2CTA而REG允许5CTA。下一候选优先收窄outer的V轴保持radix/pass与数学MAC，再评估tile canonical add/sub指令；尚未测量其性能收益。[诊断、全部硬件值和复现工具](D:/code/MPA-OpenCl/docs/STAGE2_NTT_SHAPE_D_CALIBRATION.md:121)。发布893和旧Auto B2成本保持。


## 2026-10-08 接续：outer V收窄开发候选

开发同二进制已加入M7/V16、M8/V8及独立mask；资源API确认shared容量2→5/4CTA，REG48/46与LOCAL0保持。17个独立probe调用、12条原生gate通过；两轮k27完整卷积快6.38%/6.41%。8条正式整曲线37.564333→37.461348秒（少0.274%），两组慢0.094%/快0.640%，不提升生产默认。管理员Systems实际outer4.395095→4.137593秒（少5.859%），但tile增加约0.10秒、无自身事件仍约6秒；传输字节保持。失败编译、跨构建SASS不相同及初次管理员启动失败均保留；候选较大16k容量/硬件occupancy、新D/cprof和生产移植未完成。公式、全部样本、工具、审计与后续选择统一维护[NTT专题](D:/code/MPA-OpenCl/docs/STAGE2_NTT_SHAPE_D_CALIBRATION.md:195)。下一项考察tile算术和GPU下降frontier。
