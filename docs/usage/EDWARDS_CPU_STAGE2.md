# Edwards 统一存档与 CPU Stage2

CPU 与 GPU Edwards Stage1 均生成 `METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM` 多曲线文本 save。独立程序 `ecm_cpu_stage2.exe` 读取其中的完成点，以 libecm 继续 Stage2；也可继续使用本仓库的 CUDA Stage2。CPU Stage2 不读取旧 Edwards/Prime95 二进制 `.tmp`，不通过 feeder 交接。

## 构建与运行

Windows x64 新功能构建使用 `tools/build/test/build_cpu_stage2.ps1 -Jobs 8`，输出 `build_cpu_stage2/native/ecm_cpu_stage2.exe`。需要 MSVC、CMake/Ninja、现有 GMP 和 `.refactor/gmp-ecm` 源码，不需要 CUDA。源目录可在 CMake 中设置 `ECM_LIBECM_SOURCE_DIR`；根 CMake 也提供默认关闭的 `ECM_BUILD_CPU_STAGE2`。适配器不修改外部源码，分发目录附 GMP DLL 与上游许可文件。支持 INI 中的结果身份和提交文件配置；此程序尚未纳入 CUDA 发布包、GUI 或 Stage2 队列入口。

本机依赖为 GMP-ECM `7.0.6-dev`，上游提交 `17b5cdf1536b96fce4a7b92832f57f9028fbe570`。构建使用通用 GMP limb 路径及 OpenMP，不依赖上游 MPIR/YASM 汇编工程。实现入口为 [CPU Stage2](../../src/core/ecm_cpu_stage2_main.cpp)、[严格存档读取](../../src/core/ecm_edwards_save_reader.h) 和 [依赖构建](../../tools/build/internal/ecm_libecm.cmake)。

典型交接：

```powershell
ecm_cuda.exe --curve ed --target gpu -d 1 -sigma 17 -gpucurves 4 -save ed.save 1000 1000
ecm_cpu_stage2.exe --save ed.save --b2 100000000 --results results.jsonl --submission results.json.txt --exponent 127
```

Stage1 的 N 由标准输入提供；上例的 `--exponent 127` 仅适用于保存的 N 确实等于 2^127−1。一般整数可省略 exponent；梅森余因子必须显式提供 `--exponent P` 和可重复的 `--known-factor DECIMAL`，程序验证 `(2^P−1)/∏known-factors = N`，不会按位宽猜指数。输入存档与结果文件不能指向同一个文件。

`--backend auto|ntt|plain` 默认 auto：沿用依赖源码 `NTT_SIZE_THRESHOLD=30`，不超过 30 个 GMP limb 使用 NTT，更宽使用非 NTT 多项式。当前 64 位 limb 对应约 1920 位分界。这是静态启发式，不能保证每个 B2 上最快；显式选项可覆盖。`--threads` 默认 1，只设置 libecm 的 OpenMP 线程上限。当前普通多项式主路径串行；NTT 有局部表示转换并行，核心 `mpzspv_mul_ntt` 的并行开关仍关闭，不能视为完整多线程 Stage2。曲线也逐条串行处理。性能比较须结合实际 CPU 时间/墙钟核对占用，不能把设置 4 线程等同于四核加速。

`--b2` 使用精确十进制整数，当前入口限制到 10^12，必须大于所选记录的 B1。`--memory-mib` 默认 512，传给 libecm 的内存规划器；它不限制进程工作集，也不包含所有 GMP/运行时开销。较小预算可能增加分块数并显著变慢。`--dry-run` 只校验文件身份和目标元数据，不执行算术或完整内存规划。

`--skip N --curves K` 按有效 save 记录选择范围。每完成一条曲线原子追加一行内部 JSONL；Ctrl+C 或停止事件中断当前曲线时不报告 NF，之前已发布行保留。CPU Stage2 暂无曲线内部恢复或算术跳过：续跑时根据已完成记录数设置 skip，中断的当前曲线从 Stage1 末点重新开始。提交账本会对相同数学工作去重，但重复启动仍可能重算并追加内部行。Stage1 与 Stage2 仍需分别启动。

## 存档与数学边界

CPU/GPU Stage1 的默认文件名均为 `ed_m{位数}_{B1}.save`（遵循配置的名称模板），显式 `-save` 覆盖，`-savea` 原子追加。完整文本发布成功后才删除检查点；写入失败保留任务。旧 CPU 单曲线二进制文件仍可作为备选输出，但不是本流程的正确性交接依据。

记录保存 N、完整 64 位 sigma、B1、规范 X、`POINT|FACTOR` 和数学字段校验。`POINT` 是已完成 `48·lcm(1…B1)` 的 Montgomery 仿射 x；`FACTOR` 的 X 必须是 N 的真因子。分母 GCD=N 的恒等结果没有可继续的点，因此不写 POINT，也不计作 Stage2 完成曲线。无可用记录的空 save 不能提交给 CPU Stage2。

读取器校验模型、版本、标量、重复字段、范围和校验和。Stage2 重构 Edwards 参数，并使用

\[
a_{24}=\frac1{1-d},\qquad A=4a_{24}-2\pmod N.
\]

向 libecm 设置 A、保存的 x 和 `B1done=B1`，跳过 Stage1；不再次乘 48。运行时核对 libecm 返回的 Stage1 点及完成界限未改变。当前多项式使用 S=1，NTT 与非 NTT 都执行完整 Stage2；没有采用按每个素数执行 ladder 的生产算法。libecm 的参数规划可能覆盖超过请求 B2 的尾部，JSON 仅报告保守的请求 B2，不额外宣称未核定的上界。

曲线构造、字典批量求逆或恢复归一化失败必须提取真因子或报错，不能使用零化字典继续算术。SIMD 字典不可逆时转入逐曲线的检查路径。所有输出因子都在实际 N 上验证 `1<f<N` 且 `f|N`；饱和 GCD=N 和算术错误不生成 NF。

## 两份结果文件

`--results` 保存逐曲线内部 JSONL，包含引擎、后端、时间、save 行号、真实构建身份和公开 CRC32。此文件的非 FFT 算术 `fft-length=0` 是内部记录，不能直接作为手工提交文件。

`--submission` 另存 Prime95 ECM 字段结构的 `results.json.txt`，使用正的参考 FFT 长度、数字平台号及真实 ECM-ELF 程序名；NF 曲线合并为不可变批次，省略内部扩展、CRC32 和 `security-code`。INI 身份、目标关系检查、账本去重和无算术恢复命令统一见 [ECM 结果输出](ECM_RESULTS.md)。服务器对新程序身份的接受情况仍待实际提交验证。

## 验证与实测

318–16272 位余因子、B1=50000、4096 MiB 规划预算的双后端 CPU/GPU 对照与 Prime95 历史参考，统一见 [Edwards Stage2 基准](../performance/EDWARDS_STAGE2.md)。本节保留下述小位宽、独立冻结版本的验证及性能数据。

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
