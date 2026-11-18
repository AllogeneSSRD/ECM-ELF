# Edwards 统一存档与 CPU Stage2

CPU 与 GPU Edwards Stage1 均生成 `METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM` 多曲线文本 save。独立程序 `ecm_stage2.exe` 读取其中的完成点，以 GWNUM/polymult 或 libecm 继续 Stage2；也可继续使用本仓库的 CUDA Stage2。CPU Stage2 不读取旧 Edwards/Prime95 二进制 `.tmp`，不通过 feeder 交接。

## 构建与运行

Windows x64 使用 MSVC、CMake、Ninja、GMP 和本地 GMP-ECM 源码。常规 PowerShell 可以一并构建 CPU worker 和 Stage2 DLL，输出目录包含同版本 `ecm_stage2.exe`、`gwnum.dll` 和 `gmp-10.dll`。

```powershell
tools/build/build_stage2_cpu_local.ps1
.\build_stage2_cpu_local\ecm_stage2.exe --version
```

该入口使用 `.refactor/p95v3106b01.source/gwnum` 下的 `gwnum64.lib` 和 `polymult64.lib`。若本地尚未生成这两个上游库，先在 x64 Developer PowerShell 运行 `tools/build/build_gwnum_local.ps1 -Stage2`。

`--version` 显示 worker 版本、加载的 DLL 构建指纹、Stage2 ABI、GWNUM 版本和 CPU 支持的指令集。每条执行 GW 算术的曲线在建计划后记录系数 FFT 描述与外层 polymult 指令集，内部 JSONL 对应 `fft-description` 和 `polymult-isa`。DLL 同时导出 Stage1 ABI v1。静态对照用 `tools/build/test/build_cpu_stage2.ps1 -GwPoly -Build build_dev/build_cpu_stage2_static` 单独构建。CPU worker 支持 TOML 结果配置及独立 CLI。

本机依赖为 GMP-ECM `7.0.6-dev`，上游提交 `17b5cdf1536b96fce4a7b92832f57f9028fbe570`。构建使用通用 GMP limb 路径及 OpenMP，不依赖上游 MPIR/YASM 汇编工程。实现入口为 [CPU Stage2](../../src/ecm/stage2/edwards/cpu/main.cpp)、[严格存档读取](../../include/ecm/io/save.h) 和 [依赖构建](../../tools/build/internal/ecm_libecm.cmake)。

典型交接：

```powershell
ecm_cuda.exe --curve ed --target gpu -d 1 -sigma 17 -gpucurves 4 -save ed.save 1000 1000
ecm_stage2.exe --save ed.save --b2 100000000 --results results.jsonl --submission results.json.txt --exponent 127
```

Stage1 的 N 由标准输入提供；上例的 `--exponent 127` 仅适用于保存的 N 确实等于 2^127−1。一般整数可省略 exponent；梅森余因子必须显式提供 `--exponent P` 和可重复的 `--known-factor DECIMAL`，程序验证 `(2^P−1)/∏known-factors = N`，不会按位宽猜指数。输入存档与结果文件不能指向同一个文件。

`--backend auto|ntt|plain|gwpoly` 默认 auto。静态构建或加载到兼容 Stage2 DLL 时，auto 选择 gwpoly。DLL 缺失或只有 Stage1 导出时，显式 gwpoly 报错，auto 沿用 libecm 的 `NTT_SIZE_THRESHOLD=30`，不超过 30 个 GMP limb 使用 NTT，更宽使用 plain。显式选项可覆盖。GW 后端直接调用算术库，由本项目生成点序列、F/G 产品树、Newton 倒数、模 F 折叠和 scaled 下降，不启动 Prime95。

`--threads` 默认 1。gwpoly 在线程间分配点序列与底层产品树，并调用 polymult 的多线程乘法；各线程独立持有 GW 上下文及可能被原地变换的点常量。曲线批次逐条处理。libecm 的此选项仅设置 OpenMP 上限：plain 主路径串行，NTT 只有局部表示转换并行，核心 NTT 并行仍关闭，不能据此宣称四核 Stage2。

`--b2` 使用精确十进制整数，必须大于所选记录 B1。libecm 入口限制为 10^12，gwpoly 的整数规划范围为 10^15；后者不是最大界限的性能资格，当前实际完成的正确性界限为 B2=2.6×10^12、N=M347。`--memory-mib` 默认 512，是规划预算，不限制进程工作集或包含全部运行时开销。GW 以实际 FFT 布局估计 F 树、G 工作区及临时系数，自动选择平滑 D；`--d D` 可固定偶数 D（6…20,000,000），P=φ(D)/2，预算不足拒绝执行。自动选择是静态成本估计，未保证最快。

规划器以 L=⌈log₂P⌉、W=`gwnum_size+64` 估算完整 F 树所需的系数空间和临时空间，共 `(P(L+18)+256)W` 字节。对每个 D 先尝试完整树。超预算时，较大的树可保留顶部三次分裂得到的上层多项式和归一化 baby 根，按最多八个子树分段重建。分段预算以常驻空间加主循环、下降两阶段临时空间的较大值计算。两个方案都超预算的 D 被排除。FFT 因安全余量升档后重新核算。完整树的成本分数为 `6PL+(B2/D+P)(L+8)`，分段树另计 `2PL` 重建成本。日志中的 `gwpoly-plan` 给出树策略和单系数规划字节数，`planned_mib` 为布局估值；进程峰值需独立测量。[CPU 规划源码](../../src/ecm/stage2/cpu/gwpoly.cpp)

Giant 横坐标直接写入 P+1 槽的 G 缓冲区，归一化后相邻两根就地生成首一二次因子。上层父节点覆盖两个子节点所在区间，单独的尾节点留在原位。首个满块用 `G−F` 初始化 H，后续块在 G 和 H 中复用乘积、商及余式的空间。分段 F 树的子树根也采用原地合并。最后一次折叠后释放 G 缓冲区，准入继续按上述临时空间余量估算。[宽位内存与耗时对照](../performance/CPU_GWPOLY_STAGE2.md#giant-点g-树和-h-折叠)

完成常规多项式路径时，DLL 的 `gwpoly-phases` 行分别报告 baby 点、F 建树、倒数、giant 点、G 建树、H 折叠、H 缩放、下降和叶积至整体 GCD 的秒数。分段 F 树在下降中重建的时间计入 descent。阶段图使用完整 NF，读档、发布与进程启动在引擎计时之外。[Prime95 阶段对照](../performance/STAGE2.md#宽位单线程阶段耗时)

`--gw-check` 在完整运行中额外使用 GMP 直接累乘三个 baby 点对应的全部 giant 差值，与最终叶结果比较，适合正确性验证，不计入默认性能测量。`--d` 和 `--gw-check` 要求显式 `--backend gwpoly`。`--selftest-gwpoly --threads 4` 独立验证复合环上的多项式算法，无需 save。`--dry-run` 只检查文件和目标元数据，不执行完整算术与内存规划。

```powershell
ecm_stage2.exe --save ed.save --backend gwpoly --threads 4 --memory-mib 4096 --b2 2600000000000 --exponent 347
```

上例要求保存的 N=M347；余因子须另给实际已知因子。未提供 exponent 时调用 GW 一般模数设置接口；提供并通过目标关系检查时，在原 M=2^p−1（p 为原始梅森指数）上执行 GW 乘法，所有求逆、GCD、直接验证和因子仍针对实际 N。FFT 按最大多项式尺寸预留误差余量，运行检查 GWNUM 错误及最大舍入误差≤0.40；超限报错，不生成当前 NF。当前没有自动换大 FFT 重算或曲线内部检查点。

直接运行 `--save` 时，`--skip N --curves K` 选择有效记录范围。Ctrl+C 或停止事件可中断当前曲线；重新启动时用 `--skip` 指定已完成数。提交账本按数学工作去重，逐曲线 JSONL 可能包含重复运行的记录。

## 运行时调优

CPU 的 `gwpoly` 默认从实际任务学习。自动 D 规划先用实际系数 FFT 和 `--memory-mib` 验证默认 D 及最多两个备选 D。先轮流补足各候选的重复次数，再选平均整曲线用时最短的合格候选。样本来自成功完成的无因子曲线。试用候选遇到算术或分配失败时，本进程将它从当前预检结果移除，以自动 D 重试该条曲线，重试不计入观测。

```toml
[shared]
tune_file = "tune.toml"

[stage2]
tune.runtime_enabled = true
tune.min_repeats = 2
tune.runtime_sample_limit = 10
```

路径默认位于 `ecm.toml` 旁，支持 worker 覆盖。CLI `--tune-profile FILE` 优先，相对路径以启动目录为准。CLI 的 `--tune-min-repeats` 和 `--tune-runtime-samples` 可覆盖重复数和每个候选的样本上限。达到上限后保留已有均值。`--tune-runtime 0` 只读复用合格记录。将 `tune_file` 或 `--tune-profile` 显式置空可以停用读写。指定非零 `--d`、`--dry-run` 或 `--gw-check` 时直接走相应执行路径。

CPU 使用独立的 `[[cpu_stage2]]` 表，CUDA Stage1/Stage2 保留自身的表和矩阵。CPU 记录按处理器、GW 构建、目标位宽、承载指数、B1/B2、线程数、亲和性、内存预算、D/P 和实际 FFT 长度匹配。每个候选保存重复次数、平均整曲线用时、FFT 描述和外层指令集、实际块数与 giant 点数、内存规划估值，以及初始化、主循环、baby 点、F 树、倒数、giant 点、G 树、折叠、缩放、下降、叶积与 GCD 的阶段均值。测量包含 GW 执行的初始化和清理，排除调优预检、读写文件和结果发布。初始化和主循环为外层计时。`estimated_memory_mib` 是规划值。

CPU 与 CUDA 使用同一发布锁写回文件。新版 worker 可加载只支持 Stage2 ABI 1 的 GWNUM DLL，继续按自动或指定 D 执行。运行时调优需要当前构建的 DLL 或静态版本。共享文件含 CPU 表时，CUDA worker 也应使用支持该表的版本。

## worktodo、停止与跨后端续算

CPU 与 CUDA 构建都生成 `ecm_stage2.exe`。并存时放在不同目录。GUI 的 `stage2_exe` 可填写其中一个程序的路径；留空会搜索工作流和 GUI 目录中的统一名称、兼容名称和唯一匹配的子包。CPU 版接收与 GPU 相同的 worker 编号、队列事件和停止句柄。命令行也可消费同一份队列。

```powershell
.\cpu-stage2\ecm_stage2.exe --config .\ecm.toml --worker 1 --once
```

v3 配置示例。CPU 算术参数可按 worker 覆盖，命令行优先。

```toml
[shared]
cpu_affinity = "0-7"

[stage2]
cpu.backend = "auto"
cpu.threads = 4
cpu.memory_mib = 512

[worker.2]
shared.cpu_affinity = "8-15"
stage2.cpu.threads = 2
```

`cpu.threads` 控制每条 CPU Stage2 曲线的计算线程数。`shared.cpu_affinity` 划定 worker 的 CPU 集合，也适用于同编号的 Stage1 和 GPU Stage2 worker。见 [亲和性和线程配置](../architecture/CONFIGURATION.md#worker-cpu-资源)。

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

GW 后端直接使用保存的 x 和 a24；libecm 设置 A、保存的 x 和 `B1done=B1`。两者跳过 Stage1，不再次乘 48。运行时核对 libecm 返回的 Stage1 点及完成界限未改变；libecm 使用 S=1。GW baby 集为 1≤j≤D/2 且 gcd(j,D)=1，giant 为 [mD]Q，求 G 在 baby 横坐标处的乘积。主路径均为快速多项式算法。GW 在 baby 批量归一化成功后复用已覆盖小素数的分母检查，对 (B1,min(B2,D/2)] 内整除 D 的不同素数分别补算梯形和 GCD。两侧规划可能覆盖请求 B2 以外的尾部，JSON 仅报告保守的请求 B2。公式及环上适用条件见 [多项式 Stage2 数学](../reference/ECM_STAGE2_MATH.md)。

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
