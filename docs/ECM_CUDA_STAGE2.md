# 独立 CUDA ECM Stage2 生产入口

`ecm_cuda_stage2.exe` 从已经完成 Stage1 的文本 save 读取曲线，调用当前实验版 CUDA 多项式 Stage2。它不计算 Stage1，也不会重新乘以 12。`exponent=lcm|choose12` 由生成存档的 Stage1 决定；恢复时原样使用存档中的 Q。

## 编译与直接读档

在仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_ecm_cuda_stage2.ps1 -Arch sm_89
```

输出在 `build_cuda_cmake\production_stage2\`，包括 `ecm_cuda_stage2.exe`、`gmp-10.dll`、源文件与工具链哈希清单 `build_manifest.json`。可以把 exe 和 DLL 一起复制到生产目录。需要 MSVC、CUDA nvcc 和仓库内 GMP；此脚本不编译 Stage1/CGBN，也不依赖 OpenCL。

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

限制：N 为奇数、`3<N`、最多 8192 位；sigma 是精确解析的 uint64 且至少 6；`B1≥2`、`B2>B1`。不支持 param2/param3、Edwards 存档、Prime95 二进制存档，以及尚未完成 Stage1 的 `.ckpt`/二进制 checkpoint。发现不支持的字段或不匹配的校验和时停止。

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

采用已有优化默认值：Mersenne 特化、small-prime reuse、device giant seed、批量精确 segment inverse、device giant leaf/root、GPU 驻留 fold、scaled descent、输出窗口/分块，以及异步 oracle 和批量 carry 检查。非 Mersenne N 使用原引擎一般模数路径。显式 `NTT_*` 环境变量可以覆盖默认值，仍保留实验自检与 GMP 采样。普通部署无需设置这些变量。

GPU fold 的默认开关是 `NTT_FOLD_DEVICE=1`，独立缓冲预算 `NTT_FOLD_DEVICE_MAX_MB=640`（MiB）；设置 `NTT_FOLD_DEVICE=0` 可恢复原 host flat fold。显存不足、形状/后端不兼容或超过预算时自动使用原路径。申请前还预留 workspace 剩余增长和 1 GiB 空间；预算不是全程序显存硬上限。其额外显存为 `8W(9P+8)+48` bytes，W=`ceil(bits(N)/64)`，P=`φ(D)/2`；M4423/P115200 为约 554 MiB，Γ 校正及下降前释放。并发启动不同队列时应为每条活跃曲线分别计入此容量。算法、传输公式、性能和门禁详见 [步骤报告 §33](STAGE2_GPU_CURRENT_PIPELINE.md#33-gpu-驻留-fold算法访存与验证2026-10-04)。

G根直接交接默认 `NTT_GROOT_TO_FOLD=1`，设置0恢复根先读回再上传的路径；要求 GPU fold owner 和 root-only device G-tree 已启用，否则自动使用旧路径。该交接不新增大型缓冲，当前 owner 公式已含16 B输入摘要。日志中的原始G-root FNV仅在 `root_hash_complete=1` 时完整；直接路径应核对 `real_batched_rootfold` 的sum/xor与最终Γ校正叶值。详见 [步骤报告 §34](STAGE2_GPU_CURRENT_PIPELINE.md#34-g-root-直接交接给-gpu-fold2026-10-04)。

检查调度默认 `NTT_S4_ORACLE_ASYNC=1`、`NTT_S4_CARRY_BATCH=1`，可分别设置0回退。oracle使用默认4槽pinned环，最终返回前全部验证；carry合并同形状中间块的诊断读回，覆盖数量保持。M4423实测增加约70 MiB raw pinned staging和1.573 MiB oracle pinned环，显存峰值保持5544 MiB。此额外RAM应计入多曲线预算；不是只看arena上限。详情见步骤报告§35。

NTT tile 默认 `NTT_FUSE_WARP_TAIL=1`，低6层使用warp寄存器交换与常量根约减；显式设0回退原shared实现，实验exe仍默认0。当前收益验证覆盖sm89/GPU1及报告中的M4423负载，其他架构/形状需重新测量；不增加大型buffer。算法、源码行号与8次对照见 [步骤报告§37](D:/code/MPA-OpenCl/docs/STAGE2_GPU_CURRENT_PIPELINE.md:1688)。性能测量前应删除 `NTT_FUSE_TRACE` 环境变量（PowerShell：`Remove-Item Env:NTT_FUSE_TRACE -ErrorAction SilentlyContinue`）；该诊断开关按变量存在性启用，设0或空值仍会逐kernel同步。

控制台输出每条记录开始/完成和结果文件路径。完整引擎输出默认在 exe/ini 目录的 `stage2_screen.log`；worker 2 为 `stage2_screen_2.log`。配置中的显式 `log_file` 优先；设空值则让引擎直接输出到控制台。

每条成功曲线追加一个 JSONL 结果，默认 `stage2_results.jsonl` 或 `stage2_results_N.jsonl`，包含状态、save/记录编号与指纹、N、sigma、B1/B2、设备/worker、请求 D、时长、hits、bad_factors 和十进制 factors。自动 D 的实际选择及树哈希保存在完整引擎日志中。

默认仅追查首个命中叶的具体 stage2 素数名字，避免生产形状的诊断扫描耗时过长；所有叶的 GCD/因子提取仍执行。结果中的 factors 是发现的非平凡约数，可能仍是合数，不代表完整分解。`hits=0` 也属于正常成功完成。

当前不提供 Stage2 中途 checkpoint、自动恢复曲线索引、PrimeNet 上报或跨曲线流水线。失败/中断时保留任务及源存档；已经完成的记录可能有结果，再次运行会重做它们并追加新结果。基础版本不保证恰好一次执行。源存档从不改写。

## 实现位置

- 存档文本和校验和解析：[ecm_cuda_stage2_main.cpp:106](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:106)；可选队列字段：[同文件:301](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:301)；配置和调度：[同文件:443](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:443)。
- 生产默认值：[ecm_cuda_stage2.cu:6](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:6)；独立引擎封装：[同文件:38](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:38)。
- 共用运算引擎与 save Q 接口：[stage2_tree_gpu.cu:10533](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10533)，跳过 Stage1 的分支位于 [10799](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10799)。
- 已有表达式、ini 和队列工具：`src/core/ecm_expr.cpp`、`ecm_queue_config.cpp`、`ecm_worktodo.cpp`。
- 独立编译脚本：[build_ecm_cuda_stage2.ps1](D:/code/MPA-OpenCl/tools/build/build_ecm_cuda_stage2.ps1:1)。

本入口只包装现有实验算法。性能与显存模型见 [当前 Stage2 步骤报告](STAGE2_GPU_CURRENT_PIPELINE.md)，生产 large-N 的实际预算应以运行日志为准。

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

当前工作区 [ecm_cuda_stage2.exe](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe) 为2691072 bytes，SHA256=`905826a8988e2046831e4ecd57406c46a6d8b0db2ac22551b2d6be03d18b588c`；sm89/CUDA13.3，CUDA编译252.8 s。[编译manifest](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/build_manifest.json) 记录共用NTT源码F7C98F0B…ACD7，Stage2树引擎E89B554D…F19未变。已启用warp tile默认1，显式环境变量0仍生效。

生产入口重新验收 **26/0**：基本21项、CUDA失败保留队列、saved-X已有因子、实际M4423 save恢复，以及默认1/显式0的GMP频谱、逆变换和容量切换检查。worktodo B2/skip/num仍选择用户示例961–970；xxx拒绝且队列保留。实际M4423存档恢复init15.776226/main57.036366/total72.812592 s，最终115200叶/8064000字/FNV `10619321735931855904` 与基线一致，oracle1126 jobs/66139样本全部比较，carry8241/252、pending0、错误0。

性能结论采用实验同binary串行ABBA+BAAB共8次，完整Stage2均值 **73.121608→72.528029 s（−.81%）**，main−.925%；独立Systems中tile−5.58%、NTT−2.90%。样本数少且存在波动，不能将生产单次时长作为A/B，也不保证其他形状收益。算法传输量和GPU1峰5544MiB保持。[生产26项](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/production_accept/summary.json)、[M4423日志](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/production_accept/m4423_engine.log)、[性能量化](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_warp_20261004/final_quantitative.json)。编译及验收只更新本工作区，GPU0外部生产运行未改。
