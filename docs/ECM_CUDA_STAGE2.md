# 独立 CUDA ECM Stage2 生产入口

`ecm_cuda_stage2.exe` 从已经完成 Stage1 的文本 save 读取曲线，执行 CUDA 多项式 Stage2。它不计算 Stage1，也不会重新乘以 12。`exponent=lcm|choose12` 由生成存档的 Stage1 决定；恢复时原样使用存档中的 Q。

截至2026-10-08，已发布生产基线仍是893：固定PTX3 Goldilocks、xADD6、Mersenne点乘折叠、GPU baby、驻留fold及尺寸策略；其发布与完整A/B见[点折叠与D报告](D:/code/MPA-OpenCl/docs/STAGE2_POINT_FOLD_D_CALIBRATION.md:69)。下面带日期的段落保留历次实现记录，不应把早期的“尚未实现”当作当前状态。

开发引擎保留默认关闭的 `NTT_GIANT_SEED_PAIR=1` 和可选CPU base。当前独立生产源码选择配对GPU seed，缓存 `[D]Q` 并从一个ladder同时产生相邻起点；不可逆base回退原算法。成本尚未重标定，不能套旧Auto B2 profile，详见[giant seed算法、容量与验证](D:/code/MPA-OpenCl/docs/STAGE2_XADD_D_OPTIMIZATION.md:178)。已发布893入口仍最多8192位；当前源码支持16384位，独立CU与日志控制已接入候选，验收和发布边界见末尾专节。

16384位扩展已经覆盖save/队列和规划限制、256-limb点/归约分派、除数constant容量以及旧S5局部数组。代码见[save读取](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:144)、[规划上限](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:8)、[模板分派](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:932)、[除数表](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2001)、[S5分派](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:5457)。8192的ladder launch cap是点数而非位宽，继续保留其watchdog合同；大界生产容量/性能仍需独立验收。

Auto B2已有经验证的4acc/v1窄范围组合；后续全范围验证虽然算术和收益排名通过，但耗时精度失败，没有导出新cprof。[当前发布边界](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_G1_EXACT_TREE.md:302)。

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
