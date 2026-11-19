# Edwards 统一存档与 CPU Stage2

CPU 与 GPU Edwards Stage1 均生成 `METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM` 多曲线文本 save。统一的 `ecm_stage2.exe` 默认用 CPU 路径读取完成点，通过 GWNUM/polymult 或 libecm 继续 Stage2。队列中可按 [worker 选择设备](STAGE2.md#选择运行设备)。CPU 队列和直接读档均按 CPU 后端的算术与内存条件接收文本 save。宽位完成曲线实测见 [CPU Stage2 位宽对照](../performance/STAGE2.md#cpu-宽位分区矩阵)。

## 构建与运行

Windows x64 统一版使用 MSVC、CMake、Ninja、CUDA、GMP 和本地 GMP-ECM 源码。CPU 路径随 CUDA 版 `ecm_stage2.exe` 构建，`gwnum.dll` 可自行编译并放在程序旁。只需 CPU 的机器也可用下面的本机构建入口，输出包括 `ecm_stage2.exe`、`gwnum.dll` 和 `gmp-10.dll`。

```powershell
tools/build/build_stage2_cpu_local.ps1
.\build_stage2_cpu_local\ecm_stage2.exe --version
```

该入口使用 `.refactor/p95v3106b01.source/gwnum` 下的 `gwnum64.lib` 和 `polymult64.lib`。若本地尚未生成这两个上游库，先在 x64 Developer PowerShell 运行 `tools/build/build_gwnum_local.ps1 -Stage2`。

统一版用 `--target cpu --version` 查看 CPU worker 版本、加载的 DLL 构建指纹、Stage2 ABI、GWNUM 版本和 CPU 支持的指令集。CPU 专用版直接使用 `--version`。每条执行 GW 算术的曲线在建计划后记录系数 FFT 描述与外层 polymult 指令集，内部 JSONL 对应 `fft-description` 和 `polymult-isa`。DLL 同时导出 Stage1 ABI v1。静态对照用 `tools/build/test/build_cpu_stage2.ps1 -GwPoly -Build build_dev/build_cpu_stage2_static` 单独构建。

本机依赖为 GMP-ECM `7.0.6-dev`，上游提交 `17b5cdf1536b96fce4a7b92832f57f9028fbe570`。构建使用通用 GMP limb 路径及 OpenMP，不依赖上游 MPIR/YASM 汇编工程。实现入口为 [CPU Stage2](../../src/ecm/stage2/edwards/cpu/main.cpp)、[严格存档读取](../../include/ecm/io/save.h) 和 [依赖构建](../../tools/build/internal/ecm_libecm.cmake)。

Ubuntu WSL 2 的 CPU 独立版使用 Linux libecm，支持直接读档和 Edwards worktodo；构建入口与交接示例见 [WSL 2 本机构建](BUILD.md#wsl-2-本机构建)。

典型交接：

```powershell
ecm_cuda.exe --curve ed --target gpu -d 1 -sigma 17 -gpucurves 4 -save ed.save 1000 1000
ecm_stage2.exe --save ed.save --b2 100000000 --results results.jsonl --submission results.json.txt --exponent 127
```

Stage1 的 N 由标准输入提供；上例的 `--exponent 127` 仅适用于保存的 N 确实等于 2^127−1。一般整数可省略 exponent；梅森余因子必须显式提供 `--exponent P` 和可重复的 `--known-factor DECIMAL`，程序验证 `(2^P−1)/∏known-factors = N`，不会按位宽猜指数。输入存档与结果文件不能指向同一个文件。

`--backend auto|ntt|plain|gwpoly` 默认 auto。静态构建或加载到兼容 Stage2 DLL 时，auto 选择 gwpoly。DLL 缺失或只有 Stage1 导出时，显式 gwpoly 报错，auto 沿用 libecm 的 `NTT_SIZE_THRESHOLD=30`，不超过 30 个 GMP limb 使用 NTT，更宽使用 plain。显式选项可覆盖。GW 后端直接调用算术库，由本项目生成点序列、F/G 产品树、Newton 倒数、模 F 折叠和 scaled 下降，不启动 Prime95。

`--threads` 默认 1。gwpoly 在线程间分配点序列与底层产品树，并调用 polymult 的多线程乘法；各线程独立持有 GW 上下文及可能被原地变换的点常量。曲线批次逐条处理。libecm 的此选项仅设置 OpenMP 上限：plain 主路径串行，NTT 只有局部表示转换并行，核心 NTT 并行仍关闭，不能据此宣称四核 Stage2。

`--b2` 使用精确十进制整数，必须大于所选记录 B1。libecm 入口限制为 10^12，gwpoly 的整数规划范围为 10^15；后者不是最大界限的性能资格，当前实际完成的正确性界限为 B2=2.6×10^12、N=M347。`--memory-mib` 默认 512，是规划预算，不限制进程工作集或包含全部运行时开销。GW 以实际 FFT 布局估计 F 树、G 工作区及临时系数，自动选择平滑 D；`--d D` 可固定偶数 D（6…20,000,000），P=φ(D)/2，预算不足拒绝执行。自动选择是静态成本估计，未保证最快。

自动规划在最终 D 比试探轮更小时，会按最终的 P 重新选取满足 GWNUM 安全余量的 FFT 长度，再按该布局复核内存预算。每个 D 先尝试完整 F 树。超出预算时，规划器比较顶部保留层数，保存归一化 baby 根，下降时逐棵重建底层子树。估算分别覆盖初始化、G/H 主循环和下降时同时存活的数组，另外计入线程工作区、多项式 FFT 暂存及分配余量，FFT 安全检查换档后重新计算。`gwpoly-plan` 报告完整或分段布局、单个系数的估计字节和顶部保留分裂次数，`planned_mib` 为布局估值。调优候选与显式 D 均使用同一准入条件。[CPU 规划源码](../../src/ecm/stage2/cpu/gwpoly.cpp)

完整 F 树按层连续保存各首一多项式的低次系数，整棵树的根另存首项 1。分段下降只重建该段根以下的树行。同一父节点下，两个待重建段的根数相等时，右段复用左段的系数行并重新计算系数。Newton 倒数只计算每次扩展的新系数。缩放后的根余式直接作为最终叶数组。每个父节点用一次 polymult 调用共用父变换，两个孩子分别覆盖原行中自己的区间。下降预算计入这一个输出行，并另留两个系数行的 GW 分配余量。最后一对线性因子直接用乘加计算两个叶值。

在能减少至少一个 G 块时，CPU 只遍历原 giant 网格的高位后缀，保留原终点和小素数检查。日志 `gwpoly-range` 给出原起点、当前起点与终点。后缀发现因子或分母／GCD 饱和时，释放工作区，按完整区间重算后再交付因子或报告错误。中断和舍入错误直接上报。[覆盖与因子恢复](../reference/ECM_STAGE2_RELOCATION.md)

### 将 B2 扩展到整块

CPU gwpoly 默认将 B2 扩展到整块。需要精确请求边界时，在 `[stage2]` 中设置 `b2_round_chain = false`，或传入 `--b2-round-chain 0`。CLI 优先于 worker 配置。直接读档和 worktodo 共用实现，DLL 需要 Stage2 ABI 4 提供 D/P 预检，静态 gwpoly 同样支持。能力按 ABI 判断。libecm 的 plain、NTT 和低于 ABI 4 的 DLL 默认使用精确 B2；显式要求它们凑整会报错。

每个已通过内存预检的候选 D 按其 P 和高位后缀单独求实际 B2。若已有 K 个 G 块且尾块不足 P 个点，先将末端向上移尾块缺少的点数，再按新终点复核后缀起点，直到同样的 K 块恰好容纳 K×P 个点。实际 B2 取最后一个中心还能覆盖的最大整数，即 `last×D+D/2−1`，受 CPU 上限 10¹⁵ 和已启用的 AutoB2 上限限制。空网格或原本已满的尾块沿用请求值。显式 D 若无法在上限内补满，会在算术前报错。额外覆盖可以找到请求以外的因子，也会增加 giant 点运算量。CPU 仍需明确的请求 B2，可来自 CLI、配置、工作行或 B2 比率。

日志 `gwpoly-bounds` 记录请求、实际 B2 及 D/P。内部 JSONL 的 `stage2-engine.requested-B2` 和 `actual-B2` 保留二者，顶层 `b2`、公开单曲线结果和完成回执使用实际 B2。队列检查点保留原任务的请求边界。任务末尾的 NF 汇总沿用最后一条计入曲线的实际 B2，逐条界限见内部 JSONL。不同 D 可以得到不同实际界限；tune 每个候选只读取其实际 B2 的样本，兼容已有相同几何的记录。默认凑整与精确界限的性能对照见[四组整块实验](../performance/CPU_GWPOLY_STAGE2.md)。

### 多项式执行与阶段计时

Baby 点按互素剩余类做 Montgomery 差分递推。每线程平均分到至少 512 个根时，步长取 gcd(D, 30)，只启动与该步长互素的剩余类。步长为 2 的布局和较短的切片沿用奇数链。各线程保留自己的种子并把点写到原来的根序位置，随后批量归一化。启用 `--gw-check` 会打印步长，并用独立梯形复核各切片或剩余类的首、中、末三点。

在单线程 AVX-512 上，系数 FFT 长度达到 15360 且原生循环梅森 FFT 的舍入余量足以容纳两组系数乘积和时，通用 H 折叠只归一化 HG 的高段来求商。低段保留在频域，直接传给低项乘减。其他布局在求商前归一化整个 HG。未使用零填充 FFT 的小数组和受影响的缩放布局使用独立乘法、减法，避开 GWNUM 31.6 融合乘减对数据头的越界读取。`--gw-check` 会通过 `deferred_low` 报告实际分支。[H 折叠和 baby 点实测](../performance/CPU_GWPOLY_STAGE2.md#0699-h-折叠和-baby-点)

H 缩放只计算低段乘积，将输出按下降所需的逆序直接写入系数槽。若仅有一个短 G 块，输入只取 H 的有效长度，缩放余式的其余高次系数补零。

Giant 点直接写入 P+1 槽的 G 缓冲区。每段归一化先取分母乘积的负逆元，输出带符号的横坐标。对相邻两项 a、b，就地写入 ab 和 a+b，得到首一二次因子 (X+a)(X+b)。奇数尾保留最后一项。每个建树线程只需一个临时 GW 数。单线程在实际块长至多 20160 根时，按 GW 系数大小选取 2、4、8、16 或 32 根。当 4 根能放入 256 KiB 时，先在每个连续小片内完成低层乘法与归一化，再逐层合并片根。更大的块、多线程、两根一片以及分段 F 树重建使用逐层合并。启用 `--gw-check` 时，`gwpoly-gtree` 打印实际根数和片长。上层父节点覆盖两个子节点所在区间，单独的尾节点留在原位。首个满块用 `G−F` 初始化 H，后续块在 G 和 H 中复用乘积、商及余式的空间。分段 F 树的子树根也采用原地合并。最后一次折叠后释放 G 缓冲区，准入继续按上述临时空间余量估算。[宽位内存与耗时对照](../performance/CPU_GWPOLY_STAGE2.md#giant-点g-树和-h-折叠)

完成常规多项式路径时，DLL 的 `gwpoly-phases` 行分别报告 baby 点、F 建树、倒数、giant 点、G 建树、H 折叠、H 缩放、下降和叶积至整体 GCD 的秒数。`gwpoly-descent` 的 `ftree_rebuild_seconds` 是 descent 中用于分段重建的时间，完整 F 树报告零。阶段图使用完整 NF，读档、发布与进程启动在引擎计时之外。[Prime95 阶段对照](../performance/CPU_GWPOLY_STAGE2.md)

Stage2 ABI 3 起会在曲线内输出规划、baby 点、F 树、倒数、giant 循环、下降和 GCD。设置 `log_level = "phases"` 可查看阶段，`"batches"` 另显示 giant 批次和估算剩余时间。GUI 直接消费同一进度事件，队列模式也写入共享日志。批次进度按本次执行范围计数，命中后重放原范围会重新计数。libecm 仍按曲线报告。

梅森承载在 AVX-512 或 FMA3 上对 baby 和 giant 差分链使用融合乘减。开启 `--gw-check` 时，每个点生成线程用独立的标量梯形算法重算本段首、中、末三个 baby 点，以 GMP 核对射影等式。另用 GMP 直接累乘三个 baby 点对应的全部 giant 差值，与最终叶值比较；结果中的 `checked-values` 统计这一组叶投影。默认计时关闭校验。`--d` 和 `--gw-check` 要求显式 `--backend gwpoly`。`--selftest-gwpoly --threads 4` 独立验证复合环上的多项式算法，无需 save。`--dry-run` 只检查文件和目标元数据，不执行完整算术与内存规划。

```powershell
ecm_stage2.exe --save ed.save --backend gwpoly --threads 4 --memory-mib 4096 --b2 2600000000000 --exponent 347
```

上例要求保存的 N=M347；余因子须另给实际已知因子。未提供 exponent 时调用 GW 一般模数设置接口；提供并通过目标关系检查时，在原 M=2^p−1（p 为原始梅森指数）上执行 GW 乘法，所有求逆、GCD、直接验证和因子仍针对实际 N。FFT 按最大多项式尺寸预留误差余量，运行检查 GWNUM 错误及最大舍入误差≤0.40；超限报错，不生成当前 NF。当前没有自动换大 FFT 重算或曲线内部检查点。

直接运行 `--save` 时，`--skip N --curves K` 选择有效记录范围。Ctrl+C 或停止事件可中断当前曲线；重新启动时用 `--skip` 指定已完成数。提交账本按数学工作去重，逐曲线 JSONL 可能包含重复运行的记录。

## 运行时调优

CPU 的 `gwpoly` 默认从实际任务学习。自动 D 规划先用实际系数 FFT 和 `--memory-mib` 验证默认 D 及最多两个备选 D。先轮流补足各候选的重复次数，再选平均整曲线用时最短的合格候选。样本来自成功完成的无因子曲线。试用候选遇到算术或分配失败时，本进程将它从当前预检结果移除，以自动 D 重试该条曲线，重试不计入观测。

```toml
config_version = 4

[shared]
tune_file = "tune.toml"

[stage2]
tune_runtime_enabled = true
tune_min_repeats = 2
tune_runtime_sample_limit = 10
```

路径默认位于 `ecm.toml` 旁，支持 worker 覆盖。CLI `--tune-profile FILE` 优先，相对路径以启动目录为准。CLI 的 `--tune-min-repeats` 和 `--tune-runtime-samples` 可覆盖重复数和每个候选的样本上限。达到上限后保留已有均值。`--tune-runtime 0` 只读复用合格记录。将 `tune_file` 或 `--tune-profile` 显式置空可以停用读写。指定非零 `--d`、`--dry-run` 或 `--gw-check` 时直接走相应执行路径。

CPU 与 CUDA 可以共用同一个 `tune.toml`。CPU 使用 `[cpu_stage2]` 的 `format = 3`，矩阵上方的 `cpu` 和 `supported` 分别记录型号和支持的指令集。不同的型号或指令集组合各占一张矩阵，附加表为 `[cpu_stage2.host.N]`。CPU 与 CUDA 矩阵共用列宽对齐方式。CPU 每行记录 GWNUM 版本、P/E 线程数、模数模式、`polymult_isa`、`fft_length`、B1/B2、D/P、F 树方案、`planned_mib`、重复次数、整曲线平均耗时、样本标准差、实测位宽区间、程序版本和 UTC 更新时间。计时包含 GW 初始化及清理，排除调优预检、读写文件和结果发布。完整 FFT 描述和分项耗时仍在运行日志中。

默认每个物理核只分配一个逻辑 CPU，先选 Windows 报告的性能核，再选能效核。相同的两组双 P 核都记为 `threads = [2, 0]`。显式 `cpu_affinity` 列表用完其中的独立物理核后，才会使用列表内的 SMT 兄弟线程。出现这类样本时，矩阵增加 `smt_extra` 列。CPU Stage2 设置 `cpu_affinity = "none"` 时由系统调度，暂停调优读写。

不同位宽或预算可复用同一行。程序先用当前模数、D 和预算检查 FFT 布局、误差余量及 F 树方案，再按 FFT 长度、polymult 指令集、F 树方案和规划内存等条件匹配。保存新样本前会校验实际 FFT 描述与本次预检一致。`max_exponent` 是该系数 FFT 在当前多项式余量下的规划上界，普通模数的内部 FFT 指数可能大于实际位宽。`tested_min`、`tested_max` 记录已有样本的位宽，梅森承载时记录承载指数。`planned_mib` 是规划估值。

CPU 与 CUDA 使用同一发布锁和原子替换。原 `format = 2` 矩阵可以直接读取，下一次合格观测会将现有样本转写为新版。更早的 `[[cpu_stage2]]` 样本跳过。运行时调优需要 Stage2 ABI 4。较旧 DLL 仍可执行 Stage2，ABI 3 支持实时进度。可用 `tools/build/build_gwnum_local.ps1 -Stage2` 构建新版 DLL，再放到 `ecm_stage2.exe` 旁。共用 tune 文件时，CPU 和 CUDA worker 均应更新到支持新矩阵的版本。

## worktodo、停止与跨后端续算

CPU 与 CUDA 构建都生成 `ecm_stage2.exe`。并存时放在不同目录。GUI 的 `[gui] stage2_exe` 可填写其中一个程序的路径；留空会搜索工作流和 GUI 目录中的统一名称、兼容名称和唯一匹配的子包。CPU 版接收与 GPU 相同的 worker 编号、队列事件和停止句柄。命令行也可消费同一份队列。

```powershell
.\cpu-stage2\ecm_stage2.exe --config .\ecm.toml --worker 1 --once
```

以下为 v4 配置示例。CPU 算术参数可按 worker 覆盖，命令行优先。

```toml
config_version = 4

[shared]
cpu_affinity = "0-7"

[stage2]
cpu_backend = "auto"
stage2_cpu_threads = 4
cpu_memory_mib = 512

[worker.2]
cpu_affinity = "8-15"
stage2_cpu_threads = 2
```

`stage2_cpu_threads` 控制每条 CPU Stage2 曲线的计算线程数。`cpu_affinity` 划定 worker 的 CPU 集合，也适用于同编号的 Stage1 和 GPU Stage2 worker。见 [亲和性和线程配置](../architecture/CONFIGURATION.md#worker-cpu-资源)。

`ECMSTAGE2=` 从任务校验 N 和已知因子，`ECMSTAGE2N=` 从 save 确定 N。队列按有效 save 记录的 skip 和数量选择曲线，使用共享的 B2 和 b2_ratio 规则。CPU 队列当前支持 Edwards；有效的 Montgomery PARAM0 任务交给 CPU 时会报错并保留队列行。D 的 GPU 配置服务于 GPU 规划，CPU 使用自身规划器，显式固定 D 时使用 `--backend gwpoly --d D`。

两端共用队列锁、输入身份、V5 检查点、finished 回执和发布代码。首次正常停止会让已经开始的曲线完成，提交结果后停止领取。异常终止后从最近一次持久化的完整曲线继续。真因子在曲线提交后发布；任务结束时封存 NF，写 finished、推进 worktodo，最后删除检查点。可选 `results_file` 是逐曲线诊断输出。

换后端时，停止原 worker，保留原 worktodo、完整 save、检查点目录和提交输出，使用相同配置与 worker 编号启动另一版程序。任务身份绑定 AID、实际 N、完整 save 的 SHA-256 和 skip。检查点保存完成位置、已完成的并行槽及待发布结果；下一条曲线按接手后端自己的 D、FFT、树和内存计划计算。GPU 成本调优、显存调度与中途的多项式工作区只属于当前执行进程。[共享队列驱动](../../src/ecm/stage2/queue.cpp) · [检查点格式](../../src/io/stage2_checkpoint.cpp) · [双向验证](../../tools/test/test_cpu_stage2_queue.py)

## 存档与数学边界

CPU/GPU Stage1 的默认文件名均为 `ed_m{位数}_{B1}.save`（遵循配置的名称模板），显式 `-save` 覆盖，`-savea` 原子追加。完整文本发布成功后才删除检查点；写入失败保留任务。旧 CPU 单曲线二进制文件仍可作为备选输出，但不是本流程的正确性交接依据。

记录保存 N、完整 64 位 sigma、B1、规范 X、`POINT|FACTOR` 和数学字段校验。`POINT` 是已完成 `48·lcm(1…B1)` 的 Montgomery 仿射 x；`FACTOR` 的 X 必须是 N 的真因子。分母 GCD=N 的恒等结果没有可继续的点，因此不写 POINT，也不计作 Stage2 完成曲线。无可用记录的空 save 不能提交给 CPU Stage2。

读取器校验模型、版本、标量、重复字段、范围和校验和。Stage2 重构 Edwards 参数，并使用

\[
a_{24}=\frac1{1-d},\qquad A=4a_{24}-2\pmod N.
\]

GW 后端直接使用保存的 x 和 a24；libecm 设置 A、保存的 x 和 `B1done=B1`。两者跳过 Stage1，不再次乘 48。运行时核对 libecm 返回的 Stage1 点及完成界限未改变；libecm 使用 S=1。GW baby 集为 1≤j≤D/2 且 gcd(j,D)=1，giant 为 [mD]Q，求 G 在 baby 横坐标处的乘积。主路径均为快速多项式算法。GW 在 baby 批量归一化成功后复用已覆盖小素数的分母检查，对 (B1,min(B2,D/2)] 内整除 D 的不同素数分别补算梯形和 GCD。默认按请求 B2 发布，启用整块扩展的 gwpoly 按实际 B2 发布。公式及环上适用条件见 [多项式 Stage2 数学](../reference/ECM_STAGE2_MATH.md)。

曲线构造、字典批量求逆或恢复归一化失败必须提取真因子或报错，不能使用零化字典继续算术。SIMD 字典不可逆时转入逐曲线的检查路径。所有输出因子都在实际 N 上验证 `1<f<N` 且 `f|N`；饱和 GCD=N 和算术错误不生成 NF。

## 两份结果文件

`--results` 保存逐曲线内部 JSONL，包含引擎、后端、时间、save 行号、真实构建身份和公开 CRC32。libecm 的内部 `fft-length=0`；gwpoly 记录实际 GW FFT 长度，并在 `stage2-engine` 中记录 init/main、最大舍入误差和额外 GMP 校验数。实际 D/P 紧随 B2。这些内部记录不能直接作为手工提交文件。

`--submission` 另存 Prime95 ECM 字段结构的 `results.json.txt`，使用正的参考 FFT 长度、数字平台号及真实 ECM-ELF 程序名；gwpoly 另外输出实际 `stage2-fft-length`，NF 曲线合并为不可变批次，按提交字段生成 CRC32，省略内部扩展与 `security-code`。TOML 身份、目标关系检查、账本去重和无算术恢复命令统一见 [ECM 结果输出](ECM_RESULTS.md)。已接受的 Montgomery 批次不等于本后端或 Edwards 的服务器验收，详见结果主题。

这里的实际 Stage2 FFT 长度是 GWNUM 每个系数模运算所用的长度，不是外层多项式次数 P，也不是 GPU 打包多项式的 NTT 长度。

## 验证与实测

0.33.0 的 GW 后端数学、多线程与大 B2 门禁见 [test_cpu_gwpoly_stage2.py](../../tools/test/test_cpu_gwpoly_stage2.py)。证据 `data/experiments/cpu_gw_poly_stage2_20261017/validation_05/` 包含 CPU/GPU1 save 交接、独立小界限因子 oracle、复合环的 Horner/余式验证、三处完整 GMP 投影（含 10232/16272 位实际余因子）、损坏与元数据拒绝、真实停止不发布 NF，以及双文件 CRC 与批次检查。未启用 GW 的构建另通过原 CPU 流程回归。正式 4 GHz 测量见 [CPU GW 多项式性能](../performance/CPU_GWPOLY_STAGE2.md)；本轮早期 3.1 GHz 探索耗时不参与该结论。

318–16272 位余因子、B1=50000 的完整 GPU 基准与 Prime95 历史参考，统一见 [Edwards Stage2 基准](../performance/EDWARDS_STAGE2.md)。其中 CPU NTT/plain 使用 4096 MiB 规划预算，各保留 16 个单次参考样本，CPU 采集已停止，不代表完整双后端矩阵。本节保留下述小位宽、独立冻结版本的验证及性能数据。

0.66.0 的双向接续测试使用两条 M127 曲线。CPU 与 CUDA 分别在第一条开始后收到 GUI 停止事件，接手端从第二条开始，finished 各写一次，两条 NF 各合并成一条 `curves=2` 的提交。独立 CPU 任务发布 Stage2 因子 10007，并将紧随其后的 Stage1 行留在 worktodo。运行证据在 `data/experiments/cpu_stage2_queue_066/qualified/`。GUI v2 自动发现 CPU 子包及兼容 CUDA 文件名的运行日志位于同级 `gui_cpu/` 和 `gui_cuda_legacy/`。

正确性脚本为 [test_edwards_cpu_pipeline.py](../../tools/test/test_edwards_cpu_pipeline.py)。冻结证据在 `data/experiments/edwards_full_pipeline_20261016/`：CPU 标量、多线程、AVX512-IFMA 与 GPU1 RTX4060 的 127/347/2203 位保存点对照，独立 Python `48LCM` 点乘，Stage1/Stage2 真因子，已知因子元数据，存档损坏拒绝，默认/追加输出，写入失败队列保持，以及真实 Stage1 停止恢复和 Stage2 中断。GPU 测试仅用设备 1。

结果分离版本的数学回归及双文件验证见 `data/experiments/ecm_submission_20261016/`，范围见 [结果输出验证](ECM_RESULTS.md#验证与边界)。下述性能数据使用独立冻结二进制，不代表新输出与账本的发布开销。

双后端性能由 [bench_edwards_cpu_stage2.py](../../tools/bench/bench_edwards_cpu_stage2.py) 采集。CPU 为 Ryzen AI 9 HX 370，Windows 11，GMP-ECM 7.0.6-dev，单线程、512 MiB 规划预算，输入为 M127/M347/M2203、B1=1000、sigma=17，每格 1 条曲线、1 次预热和 3 个正式样本，交替后端。CPU 功耗、频率和调度未固定或测量。下表为构造曲线至完整 libecm 返回的中位秒数，不包含读档、进程启动及 JSON 发布；完整进程计时另存原始汇总。

| 位宽 | B2 | NTT / s | 非 NTT / s |
| --- | ---: | ---: | ---: |
| 127 | 10^6 | 0.00764 | 0.00705 |
| 127 | 10^8 | 0.07667 | 0.09121 |
| 127 | 10^12 | 16.55754 | 17.83203 |
| 347 | 10^6 | 0.01146 | 0.00893 |
| 347 | 10^8 | 0.14177 | 0.13634 |
| 2203 | 10^6 | 0.07418 | 0.05268 |
| 2203 | 10^8 | 1.24691 | 0.76320 |

证据为 `bench_01/summary.json` 和 `bench_1e12_01/summary.json`，冻结性能二进制 SHA256 为 `a23c83ccb53b62101f7e98b0c8803d274e49a87e2529b77c33156c28029b6222`，保存在 `frozen/`。这是显式 NTT/plain 测量，不包含自动后端选择成本。B2=10^12 的两条路径均完整结束；该界限目前仅对 127 位完成性能资格，不能外推大位宽耗时。此格单进程峰值工作集的中位数分别为 255.85/403.76 MiB，没有相加组件峰值。

![双后端完整曲线时间与进程峰值工作集](../../data/figures/edwards_cpu_stage2_20261016/backends.png)

图由 [plot_edwards_cpu_stage2.py](../../tools/bench/plot_edwards_cpu_stage2.py) 读取上述两份汇总生成；误差线保留三次实测的最小/最大值，不是置信区间。图表、原始实验与汇总均在 Git 忽略的 data 目录，可用脚本再生成。
