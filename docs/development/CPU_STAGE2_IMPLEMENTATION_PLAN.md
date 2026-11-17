# CPU Stage2 实现与优化计划

先交付 Windows x64 的 Edwards CPU Stage2。用本地编译的 `gwnum.dll` 从已有 Edwards 文本 save 续算，冻结可复现的性能基线，再按 Prime95 的完整多项式路径优化到同一性能量级。Edwards 验收后接入 Montgomery。自研 CPU 多项式算术放在 GWNUM 版可用、正确性与性能基线冻结之后。

首版可用性的范围为独立 CLI，覆盖构建、读档、查找因子、停止与按曲线续跑。CLI 验收后再评估 GUI、worktodo 和曲线内恢复，优先复用相应的仓库协议。第一阶段已在 0.65.0 跑通，下面的起点描述和分阶段任务保留作为开发记录。当前行为、历史结果和数学推导分别见 [CPU Stage2 用法](../usage/EDWARDS_CPU_STAGE2.md)、[GW 多项式性能](../performance/CPU_GWPOLY_STAGE2.md) 和 [Stage2 数学](../reference/ECM_STAGE2_MATH.md)。

0.66.0 已接入 GUI/worktodo。CPU 与 CUDA 使用同一整曲线检查点，双向接续、首次停止排空和结果发布均有真实 worker 覆盖。下一轮分别优化 CPU 和 CUDA，先按相同的 Edwards 末点和几何拆解耗时，再由各自的整曲线实测决定取舍。[双向优化实验](#51-cpu-与-cuda-的双向优化实验) · [操作与适配边界](../usage/EDWARDS_CPU_STAGE2.md#worktodo停止与跨后端续算)

## 1. 复用现有实现

- `ecm_cpu_stage2.exe` 已能从 CPU/CUDA Edwards Stage1 文本 save 读取完成点，验证目标 N，重构曲线参数，调用 `gwpoly` 或 libecm，再写逐曲线结果。`gwpoly` 已实现 Montgomery x-only 点生成、批量求逆、F/G 积树、Newton 倒数、模 F 折叠、循环高段 scaled 下降和 GCD。点、底层树和长多项式分别已有并行路径。保留这些代码作为首轮移植的可执行对照。[CLI 入口](../../src/ecm/stage2/edwards/cpu/main.cpp#L150) · [计算入口](../../src/ecm/stage2/cpu/gwpoly.cpp#L370)
- `-GwPoly` 当前把 `gwnum64.lib` 和 `polymult64.lib` 静态链入 CPU Stage2。`build_gwnum_local.ps1` 生成同名 `gwnum.dll`，0.64.5 的导出只覆盖 Stage1 的 PARAM0 梯形状态。DLL 阶段需要增加 Stage2 能力，并沿用原 Stage1 导出和调用合同。[CPU Stage2 构建](../../tools/build/internal/ecm_cpu_stage2.cmake) · [DLL 构建](../../tools/build/gwnum/CMakeLists.txt) · [Stage1 ABI](../../include/ecm/stage1/gwnum_abi.h)
- 历史 0.33.0 静态 `gwpoly` 的四线程完整耗时为 Prime95 31.4b05 历史四线程的 **1.11–1.81 倍**。三个位宽、两档实际 B2 共六格，`gwpoly` 每格 3 次，Prime95 每格 1 次。设备为 Ryzen AI 9 HX 370，历史记录为固定 4 GHz、约 4 GiB 规划条件，功耗未采样，耗时采用完整 Stage2 口径。宽位高 B2 的 G 树和 fold 是优先剖析对象。两侧运行几何及部分 FFT 长度不同。按性能文档保留这组数字的设备、功耗记录边界和计时口径，重新采样后再判断是否追平。[实测及原始证据位置](../performance/CPU_GWPOLY_STAGE2.md)

## 2. Edwards save 与正确性合同

沿用 `METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM` 的严格读取器和 `--skip` / `--curves` 范围。记录中的 N、sigma、B1、状态、校验及完成点 X 为输入依据。`POINT` 已执行 `48·lcm(1,…,B1)`，X 是 Montgomery 仿射横坐标。按 Edwards sigma 在实际 N 上重构 `a24=1/(1−d)`，将 `Q=(X:1)` 直接交给 x-only Stage2。`FACTOR` 分支按 Stage1 已有因子处理。批次计算前校验所选范围及 B2。提供原梅森指数和已知因子时精确验证余因子关系，再考虑以 `M=2^p−1` 承载乘法。求逆、因子校验和最终 GCD 始终按实际 N 解释。[存档协议](../architecture/SAVE.md#5-edwards-save-version1) · [构造函数](../../src/ecm/curve/edwards.h#L59)

固定 `(N, curve, param, sigma)`、起点和 Stage1 标量合同。用独立有限域点阶或小整数逐素数 oracle 冻结已知因子所需的最小 B1/B2。B1 对照 Stage1 完成点与规定标量。对固定 B1 的 Stage2 样本至少测 `B2_min−1`、`B2_min` 和其后一个界限，记录合数 N 上的实际真因子。另测多因子同时命中、构造退化、归一化分母不可逆和 GCD 饱和。届时以 [数据集](../../tools/ecm_dataset/README.md) 和 [现有 GW 验收脚本](../../tools/test/test_cpu_gwpoly_stage2.py) 中能保持上述身份的样本起步。

规划层记录请求 B2、giant 网格和真实处理区间。覆盖合同包括 `(B1,B2]` 内所有素数，单独处理整除 D 的素数、只能由 `m=0` 表示的小素数、最后一个 D 段和短 G 尾块。历史 `gwpoly` 会多算网格边缘，Prime95 还会将小素数移位和补齐整块，因此测最小 B2 时需要逐素数 oracle。任何由尾部上扩、额外复合倍数或归一化触发的命中都要区分来源，防止用更改 D 或尾块后提前发现的因子改写正式命中界限。实现时在覆盖边界上安排可精确复核的路径，对饱和或来源不明的正命中按候选子块与原区间回放拆分，按固定次序确认因子。完整 NF 路径继续使用快多项式和小素数补算，额外回放只在命中或边界验证需要时启用。保守路径也无法确认时返回可诊断状态。发布前验证 `1<f<N` 且 `f|N`。停止、算术错误、超出预算的内存计划或未解开的 `gcd=N` 禁止写当前曲线 NF。[现有 wheel 与尾块](../../src/ecm/stage2/cpu/gwpoly.cpp#L442) · [移位覆盖与原区间回放](../reference/ECM_STAGE2_RELOCATION.md)

## 3. 里程碑一：本地 DLL 与可用的 Edwards CLI

先恢复共享验证入口。`test_cpu_gwpoly_stage2.py` 仍给当前 worker 传 `--ini`，应改为 `--config` 与有效 TOML；`bench_edwards_stage2_reference.py` 和 `bench_cpu_gwpoly_stage2.py` 的 CPU 调用也要同步。后者还引用搬迁前的 `src/core`、`src/cpu`，应从当前构建清单冻结依赖，并把实际 `gwnum.dll` 与 EXE、GMP DLL 一起复制到采样目录。先在一条小曲线上核对采集到的实际后端、结果及二进制哈希，再启动大 B2 矩阵。[现行 CLI](../../src/ecm/stage2/edwards/cpu/main.cpp#L43) · [GW 采集器](../../tools/bench/bench_cpu_gwpoly_stage2.py#L35)

1. **冻结静态对照。** 用受跟踪的 `tools/build/test/build_cpu_stage2.ps1 -GwPoly` 为当前代码单独构建静态版本，记录 EXE、上游两静态库、编译器与 GMP 身份。用 libecm 显式 `ntt` / `plain`、GMP 直接乘积和小界限逐素数 oracle 对拍。旧静态方案作为本地 A/B 对照，先将同一个算法移至 DLL。
2. **扩展本地构建。** 给 `tools/build/build_gwnum_local.ps1` 增加可选 Stage2 构建能力，在同一个 Windows x64 模块中加入匹配版本的 `polymult64.lib` 和现有 `gwpoly` 主链。保留 Stage1 ABI v1。为 Stage2 增加独立的版本化 C 导出，校验 ABI、结构大小和所需能力。利用现有模块目录加载方式，只从 worker 所在目录加载，明确报出旧 DLL 缺少 Stage2 导出或不兼容的原因。常规构建可独立使用 libecm，GW 版由本地上游源码包构建，分发边界依照 [GWNUM 许可与构建说明](../reference/GWNUM_LICENSE.md)。[现有加载器](../../src/ecm/stage1/montgomery/cpu/gwnum_curve.cpp#L38)
3. **DLL 每曲线调用一次。** 传入调用方拥有的定长小端 32 位字数组 `N/X/a24`、B1/B2、承载指数、D 覆盖项、线程与内存预算，可用停止回调和输出统计结构。返回完成、因子、停止或错误状态，附带固定长度错误文本。DLL 内部保留 GW/多项式句柄、线程、临时数组和 GMP 生命周期。GW/`mpz_t`/C++ 对象、分配器和异常留在各自模块内。EXE 继续负责 save、身份、因子复核和发布。需要中途进度时按块报告，避免按系数跨 DLL 调用。
4. **给用户和采集器显示 CPU 版本。** 为 CPU worker 提供 `--version`，显示 ECM-ELF 程序版本、构建 ID、Stage2 ABI、实际加载的 `gwnum.dll` 与其 GWNUM/polymult 构建身份，同时显示 CPU 型号。启动正常任务时显示实际选择的后端、线程数和内存预算，建计划后记录所选指令集和 CPU 算术路径、每系数 FFT 长度、D/P、块数和最大 roundoff。基准清单冻结实际加载的 DLL SHA256。显式 `gwpoly` 缺 DLL 要报错；`auto` 无可用 DLL 时在 libecm 支持的界限内明确显示 NTT/plain 回退。一次已开始的 GW 曲线失败后，保留原错误与状态供重跑。
5. **验收可用性。** 验证同一 Edwards CPU/CUDA 末点、完整 64 位 sigma、合法与损坏 save、一般模数与梅森余因子、支持/缺失/旧版 DLL、Stage1 旧导出回归、1/4 线程、显式 NTT/plain、内存不足、真实停止后按曲线继续、内部 JSONL 与提交账本。验证 DLL 与静态版在同一安全 FFT 和同一 D 下因子、同几何 GMP 投影、停止及错误行为一致。保留现行 GW 错误检查和 `maxerr≤0.40`。在用法文档给出通过验收的构建、DLL 放置、读档与续跑命令，说明手工 skip 和结果账本行为。[CPU 流程验收](../../tools/test/test_edwards_cpu_pipeline.py) · [GW 验收](../../tools/test/test_cpu_gwpoly_stage2.py)

### 第一阶段验收（0.65.0）

动态版从 CPU 与 GPU1 的 Edwards 48LCM 文本 save 续算，确认 M127 保存点一致。DLL 验证通过 29 项，包括已知因子 10007、1/4 线程数学自检、64 位 sigma、多曲线、跳过记录、损坏记录、实际中断退出 130，以及 M347 在 B2=2.6×10^12 上的三处 GMP 投影。另验证缺少 DLL 或旁置旧 Stage1 DLL 时的 auto 回退与显式错误。两条曲线的实停续跑在 sigma=17 发布后退出 130，按同一 B2 及 `--skip 1` 继续计算 sigma=18，内部结果恰好两行。新 DLL 的 Stage1 ABI v1 通过 16 个独立算术用例。修复了多曲线内部 JSONL 重复写换行，并按当前提交格式核验公开 CRC。用户可用 `--version` 和曲线日志区分 CPU 能力、系数 FFT 与 polymult 指令集。[正确性证据](../../data/experiments/cpu_stage2_dll_065/gates_b/summary.json) · [加载与命中对照](../../data/experiments/cpu_stage2_dll_065/loader_and_factor.json) · [Stage1 ABI](../../data/experiments/cpu_stage2_dll_065/stage1_probe.log)

本轮使用两份完整梅森数存档做固定 D 的 DLL/静态对照。代表性样本与原始条件见 [性能记录](../performance/CPU_GWPOLY_STAGE2.md)。三种精确大余因子的六格 Prime95 比较、逐素数最小界限的扩展验证及 Montgomery 接入继续按后续阶段实施。

## 4. 里程碑二：建立 Prime95 性能基线

先恢复历史三个位宽与两档实际 B2 的六格输入，原指数为 1009、10267、16319，实际 N 为 383、10232、16272 位，Edwards `sigma=26`、`B1=50000`。本次检查时历史屏幕日志、正式结果文件和 Prime95 31.4b05 EXE 可定位，旧冻结目录与 Stage2 计划未在当前 `data/` 找到。用现有准备工具按完整 N 和已知因子重新生成独立证据，重新验证每份 Stage1 完成点，再保存原始输入和 SHA256。Prime95 新对照优先使用与本地 GWNUM 同源的 31.6b01，在隔离目录采样本地 ECM，核对实际选中的 polymult 路径、Edwards 构造、完成点及调整后的 B2。31.4b05 继续提供历史趋势。若标准 Prime95 无法消费本项目 save，可用本地参考程序在 Stage1 转成 Montgomery 后导出规范 `x/a24`，或在计时之外向参考路径注入同一已验证末点。源码、载入方式和 EXE 身份单列。正式 A/B 只使用完成点一致的行。[参考准备工具](../../tools/bench/README_EDWARDS_STAGE2_REFERENCE.md) · [原始发布版](https://www.mersenne.org/download/software/v31/31.06/)

- **算术归因。** 比较静态与 DLL 完整等几何运行，固定 N/Q/a24、B1/B2、承载、D、P、起止 giant 索引、块形状、FFT 档、线程数和内存条件。优先沿用 `--d`，用受跟踪的诊断入口控制其余计划变量，保留安全 FFT 检查。逐层记录 F/G 树、点与归一化、倒数、fold、下降、GCD 和资源释放。Prime95 若无法固定某些变量，只把相同尺寸与工作量的阶段或算子用于归因，正常整曲线仍按实际计划记录。
- **用户性能。** 在同机对比 DLL 自动规划与 Prime95 生产规划，按精确 N、构造、完成点、B1、可比实际 B2、内存设置和线程数配对，逐条记录请求 B2、覆盖端点、移位策略、D/P、FFT、实际 giant 数和块数。两侧有效覆盖不同的样本保留为覆盖/效率对照。默认采 1/4 线程及 4096 MiB 规划条件，六格每档至少 5 次完成采样，轮换顺序，记录固定的亲和性、处理器拓扑、有效频率、功耗、温度、系统及 EXE/DLL 哈希。另加 512 MiB 和不同尾形状做容量验证，随后扩展一般模数和宽位 B2≥10^12。
- **计时与内存。** 主指标为完整单曲线 CPU Stage2 墙钟，包含曲线重构、规划/初始化、计算、GCD 和清理，排除 Stage1、读档、启动与结果发布。Prime95 用 init、main、GCD 三项对齐，补测未计入的清理。另报端到端进程墙钟、CPU 时间、单进程峰值工作集和 private bytes。分阶段计时使用互斥边界，分配账本跟踪同时存活的容量。提前命中、失败和中断另列。计时保留常规误差门禁，GMP `--gw-check` 深度校验单独测。[现有采集器](../../tools/bench/README_CPU_GWPOLY_STAGE2.md)
- **建议验收门槛。** 固定几何的 DLL/静态各格中位耗时比目标 `≤1.03`，测量波动超过差值时补样。四线程正式六格的 DLL/Prime95 中位比几何平均目标 `≤1.10`，单格 `≤1.20`，同时完整覆盖、内存预算口径和正确性门禁均达标。该阈值是计划中对「与 Prime95 相当」的量化定义。1T、内存受限和更大范围各自报告，不从四线程六格外推。原始证据写 `data/experiments/`，汇总写 `data/benchmarks/`。

## 5. 里程碑三：按剖析优化多项式主链

保持平衡积树和 scaled remainder。设 `P=φ(D)/2`，每棵规模为 P 的树用 `O(M(P) log P)` 级别的多项式工作量，`M(P)` 为长度 O(P) 的一次快乘成本。跨块用 `H←H·G mod F` 保持累积状态次数小于 P，最后用同一棵 F 树求叶值。外层乘法规模、每系数 GW FFT 长度及实际变换次数分别计数。[数学与复杂度](../reference/ECM_STAGE2_MATH.md#工作量与空间)

1. **先测 G 树和 fold。** 现有 G 复用双工作区、保存 `polymult` 计划。固定几何上测试 G 首块直接写入 H、连续工作区与更少的系数搬移。对后续满块，试验 F 和逆多项式跨块预变换、商的 high product、`polymult_fma` 低段融合 `T−qF`，以及结果的 FFT 状态衔接。按命中的形状和线程数复用计划，独立比较短尾，记录减少的复制、变换和 live bytes。检查每个新路径与 GMP 余式一致，再测完整曲线。[本仓库余式路径](../../src/ecm/stage2/cpu/gwpoly.cpp#L174) · [Prime95 主循环](../../.refactor/p95v3106b01.source/ecm.cpp#L9454) · [预处理接口](../../.refactor/p95v3106b01.source/gwnum/polymult.h#L181)
2. **其次优化 F 树容量与 scaled 下降。** 对比全量存树、保留选定层并在下降时分段重建。按完整流程收益评价预处理压缩、同父两子节点的 `polymult_several` 共享变换及底层叶值分段合并。先验证真实非均分次数和空节点，真实内存不足时取消或回到已验收计划。F/R/G/H、树层、FFT scratch、线程副本和下降释放都进入生命周期模型；磁盘层缓存留到重算成本有实测依据再评估。[Prime95 下行与重建](../../.refactor/p95v3106b01.source/ecm.cpp#L9538)
3. **优化点生成和线程调度。** 如果阶段计时显示成本突出，比较持久点 worker、块间连续 giant 种子、批量归一化窗口和融合 xDBL/xADD。继续为各线程复制可能被 GW 原地改写的常量。小多项式在树节点间并行，大多项式交给内部 helper，统一线程预算，验证没有嵌套过量调度。以 1T/4T 完整耗时与 CPU 占用判断有效收益。[当前点并行](../../src/ecm/stage2/cpu/gwpoly.cpp#L253) · [并发错误的历史证据](CPU_GW_POLY_STAGE2_DRAFT.md)
4. **按真实算术布局联合选 D、FFT 档和内存策略。** 用现有平滑 D 及 Prime95 同源表的几何作候选，考虑实际 giant 起点/终点、尾块、各层乘法形状、预变换成本、CPU cache、FFT 安全余量、线程数和 `polymult_mem_required`。固定 D 保留用于归因。优先在已覆盖的 CPU/位宽和预算上用完整曲线校准，其他输入用保守静态选择。算术失败若需自动重试，先建立大一档 FFT 的确定重跑和重新预算，完成前保留原曲线未发布状态。[现有 D 选择](../../src/ecm/stage2/cpu/gwpoly.cpp#L348) · [Prime95 成本模型](../../.refactor/p95v3106b01.source/ecm.cpp#L5753)
5. **单独验证 relocation。** 用 Prime95 的 first-missing-prime 策略与仓库已写明的覆盖证明，比较无移位与移动起点后实际减少的 giant 段数。对移动过的素数、B1/B2 边界、整块补齐和多因子饱和做独立对拍。精确正命中回放和覆盖验证通过后再纳入自动规划，分别统计相同 B2 的整曲线成本与额外覆盖。[Prime95 移位及上扩](../reference/PRIME95_STAGE2.md#4-b2-上扩完整-g-多项式与-relocation) · [仓库回归用例](../../tools/test/test_edwards_stage2_relocation.py)

每个变化先单独与 DLL 基线做固定几何对拍，保留失败日志和完整 NF 采样。合入默认前复测自动规划六格，继续保留显式 D 与现有安全计划作为定位手段。若目标有剩余差距，用实测的阶段耗时、等形状变换和内存账本定位下一步。

### 5.1 CPU 与 CUDA 的双向优化实验

**先按负载选实验。** 小 B2 先测建树、下降和线程准备。宽位、大 B2 先测每个完整 G 块都要执行的建树与模 F 折叠。同一 N 的多曲线队列另测准备成本能否摊薄。CPU 的对照是本地 GWNUM DLL，CUDA 的对照是同卡当前生产路径，两端各自计算提速比。

以下重新统计 0.66.0 冻结日志。Windows x64、Ryzen AI 9 HX 370、GWNUM 31.6 DLL `6bbf348cf243d3a4`，Edwards AM、sigma=17、B1=1000、4 线程、512 MiB，未锁频或采样功耗。每格取 5 次普通 NF，排除独立的深度校验。整曲线列是耗时中位数，百分比先按每次阶段秒数除以该次整曲线秒数，再取中位数。点包含 baby、giant 和归一化；树包含 F/G 建树及构造 F 倒数。计时排除进程启动、读档和发布。表中的阶段只覆盖部分整曲线工作。[基线与二进制身份](../performance/CPU_GWPOLY_STAGE2.md) · [逐次日志目录，取两组 `dll_1` 至 `dll_5`](../../data/experiments/cpu_stage2_queue_066/release_isa/)

| 输入 | B2 / D | G 块 | 整曲线 | 点 | 树及倒数 | 折叠 | 下降 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M127 | 10⁹ / 60060 | 3 | 0.6562 s | 18.2% | 32.0% | 7.5% | 12.6% |
| M2203 | 10⁸ / 30030 | 2 | 0.2281 s | 9.0% | 26.7% | 4.2% | 20.6% |

高 B2 的历史 CPU 对照中，10232/16272 位实际余因子分别有 50/85 个 G 块。另有 CUDA 固定 D 的布局实验，改变相同真实叶子的分组，在跨 NTT 长度台阶时改善了 G 树和整曲线，在未跨台阶的对照上出现回退。这两项观察指向同一个待测问题：省下的是一次初始化、每一个满块，还是只在特定乘法长度才有的成本。CPU 侧要记录 GW 系数 FFT 档和外层乘法形状，CUDA 侧记录 NTT 长度和次数。[CPU 历史六格及条件](../performance/CPU_GWPOLY_STAGE2.md#0330-与-prime95-历史对照) · [CUDA 布局 A/B、设备及采样条件](../performance/STAGE2_GTREE_LAYOUT.md)

#### CPU 候选，借用 CUDA 的调度和数据布局

1. **先复用一条曲线内的工作。** CPU 的 `parallel_points` 每次建 GW clone 和线程，`tree_jobs` 会在多个树层调用它；G 树已有两块系数工作区，但每个块仍重建各层次数、偏移和任务表。先分两次 A/B，分别复用同次数 G 块的只读层表、在一条曲线内保留每线程私有的 GW clone。记录 clone 次数与时间、计划命中、各层耗时和峰值工作集。短尾按实际次数建表；保持私有点常量、错误汇总和停止轮询。多曲线再试复用同 D 和布局的只读表。Q、F、倒数和可变 GW 缓冲区归各自曲线；并发预建下一棵 F 树排在有队列吞吐和峰值内存对照之后。[CPU 点与建树调度](../../src/ecm/stage2/cpu/gwpoly.cpp#L265) · [CUDA 保留叶顺序的 wheel 调度](../../src/ecm/stage2/cuda/stage2_baby_wheel_plan.h)
2. **宽位多块先缩短首块，再摊薄后续折叠。** CUDA 用首个 G 根给 H 赋初值；CPU 已对每块复用 F 和倒数，首块仍走通用余式。可先独立试一个等价特例：CPU 的 F、G 都首一，满块同为 P 次，故第一块的规范余式就是 `G−F` 的低 P 项；首块不足 P 次时直接取 G。H 需拥有独立于下一块覆写区的存储。随后单独比较固定 F/倒数预变换与低段乘减，记录每满块变换次数、复制量、预处理秒数和回本块数。CUDA 的齐次 G 含尺度因子，移植首块规则时必须带上实际首项系数。[CPU 余式](../../src/ecm/stage2/cpu/gwpoly.cpp#L186) · [CUDA 首块与后续折叠](../../src/ecm/stage2/cuda/batch.cpp#L624)
3. **下降阶段按形状分组。** CPU 当前分别算两个子窗口并深度优先下降；CUDA 已按同层子次数分组。CPU 先只改同父的两次乘法，试用 `polymult_several` 共享父操作数，再独立比较按层成批和现有递归。固定 F 树，比较子窗口、叶序、整曲线和同时存活的系数数目。仅当点计时占主要成本，再对照 CUDA 的 wheel 分段和 giant 齐次叶方案，分别计数梯形种子、差分加法与求逆。齐次方案要承担非首一乘法和 Gamma 校正的成本。[CPU scaled 下降](../../src/ecm/stage2/cpu/gwpoly.cpp#L207) · [CUDA 下降分组](../../src/ecm/stage2/cuda/remainder.cpp#L993) · [上游多乘接口](../../.refactor/p95v3106b01.source/gwnum/polymult.h#L204)

#### CUDA 候选，用 CPU 的表示和结果检验

1. **多 G 块先测折叠中的主机等待。** 驻留版 `step_loaded` 在有余式时，从设备取回 8 字节的有效长度供下一块使用。CPU 的余式允许用 P 个系数槽保存高位为零的结果。CUDA 可做同样的 A/B：减法写全 P 项，设备缓冲固定保留 P 项，下一块按这个上界乘。比较减少的主机等待、额外零系数的 NTT 工作、每次折叠和完整曲线耗时。覆盖短尾、最高项抵消、零余式及仅一块的输入。诊断模式保留规范化后的逐项 GMP 对照。[CUDA 余式与长度读回](../../src/ecm/stage2/cuda/remainder.cpp#L463) · [驻留折叠校验](../../tools/test/test_stage2_fold_device.py)
2. **在 NF 常见的任务上延后读取叶值。** CUDA 驻留下降结束时会读回全部叶值；驻留块积路径又直接从设备叶子求每 64 叶的乘积。借鉴 CPU 遇到饱和才逐叶细查的做法，GPU 先交付块积给主机 GMP。若某块的 GCD 大于 1（包括等于 N），仅取该块叶子并继续现有因子定位。深度校验和完整摘要仍读所有叶。对照全 NF、早命中、多块命中、饱和与队列早交接，确保最后一个消费者结束才释放设备缓冲。核对 D2H 字节、显存持有时间、单曲线耗时及下一条曲线能否按时启动。已有尾部算子测试可作为附加对照，是否启用以完整队列实测决定。[CUDA 末层读回](../../src/ecm/stage2/cuda/remainder.cpp#L1023) · [块积、早交接和命中定位](../../src/ecm/stage2/cuda/batch.cpp#L935) · [既有 CPU/GPU 尾部对照](../performance/STAGE2_CPU_TAIL.md)
3. **若下降仍占主要时间，再测同父变换复用。** 以 CPU 的同父双窗口实验作数学参照，在现有 CUDA 同形状批乘中核对一对兄弟实际重复打包或变换父向量多少次。只在计数确有重复、变换长度和表示相容的层上试共享。对比 launch 数、正逆变换数、额外显存、整曲线与非均分子节点。CPU 和 GPU 可分别选中不同的树布局或层级阈值。[CUDA 批乘与父向量偏移](../../src/ecm/stage2/cuda/remainder.cpp#L1006) · [G 树布局数学](../reference/ECM_STAGE2_G_TREE_LAYOUT.md)

#### 首轮实验与取舍

0.66.1 的首轮已在 127、2203 和 10232 位输入上完成。CPU 接入了 CUDA 已使用的 baby 分母检查复用；层表、单任务 clone、首块折叠与下降的逐项结果，以及两张 CUDA 卡的长度读回、叶值和同父末层对照见[性能报告](../performance/STAGE2.md#0661-cpu-与-cuda-小规模对照)。跨曲线持久线程、固定 F 预变换和设备内同父 NTT 复用仍是后续候选。

- **先统一观测。** 保留现有外层墙钟，将 CPU 的点拆成 baby/giant 与归一化，将树拆成 F、倒数、满块 G 与尾块 G，再单列 fold、下降、叶积/GCD、清理。沿用 CUDA 的互斥阶段计时；CUDA 事件、CPU 等待和传输量单列归因。先用上表两条 save 复现旧版，再从第 4 节六格中选一条多满块的宽位 save。CPU、CUDA 分开跑，避免竞争主机核心或功率预算。[CUDA 阶段边界](../../src/ecm/stage2/cuda/phase_times.h) · [CPU 基线采集](../../tools/bench/compare_cpu_stage2_gw.py)
- **用同一个数学中间值对拍。** 固定实际 N、同一 Edwards save 记录、B1/B2、D、baby 索引和实际 giant 范围；关闭 relocation 或核实两端覆盖相同。小形状全量检查，大形状抽查固定 baby 索引的 GMP 直接乘积。比较每块的 `H mod F` 时统一到实际 N、普通系数和相同叶序；CUDA 齐次路径先移除同一前缀的可逆 Gamma，首块先取规范余式。非单位分母走因子分支。保存首个不一致的块、索引和系数，并覆盖非二次幂、单满块、满块加一叶、真因子及饱和。继续执行 GW 误差门禁和 CPU↔CUDA 整曲线检查点双向测试。[CPU GMP 验证](../../tools/test/test_cpu_gwpoly_stage2.py) · [CUDA 尺度合同](../../src/ecm/stage2/cuda/batch.cpp#L122) · [队列接续验证](../../tools/test/test_cpu_stage2_queue.py)
- **同后端判定收益。** 每个候选单独开关，在同一机器、同一几何和预算下交错运行基线与候选，各至少 5 条完整 NF，记录配对耗时比、中位数和离散程度。CPU 记录线程、亲和性、实际 FFT/外层 ISA 和进程峰值；GPU 记录型号、驱动、设备显存峰值、主机工作集及传输量。两侧均保存 EXE/DLL 或设备镜像身份、输入哈希、功耗/频率或未采样说明。单曲线之外，用同一多 sigma 队列区分首条和后续曲线，计入停止、发布和资源交接。差值落入重复运行波动时补样；仅算子改善的候选保留实验状态。固定 D 过关后再用自动 D、相邻位宽、另一档 B2 和内存受限输入复核适用范围。CPU 对 Prime95 的独立验收沿用第 4 节；CUDA 按同卡生产版的整曲线耗时和队列吞吐量验收。

## 6. 里程碑四：Montgomery save 续算

先接入 PARAM0。复用现有 Montgomery 文本解析合同和 Suyama 曲线构造，在主机侧生成同样的 `N/X/a24/B1/B2` 输入。`METHOD=ECM` 缺省 `PARAM` 按 PARAM0 解读。对显式 `PARAM=2/3` 保留各自的构造身份并按该参数化重构 `a24`，各自完整验收后再开放。PARAM2 的起点和曲线来自辅助曲线倍点，PARAM3 使用对应的定点参数；同一数值 sigma 在三个参数化中的意义各自独立。现有共享 Stage2 读取器仅接受 Montgomery PARAM0，扩展其他参数前需要单独的协议和生成端实档对照。[存档兼容矩阵](../architecture/SAVE.md#71-当前能力矩阵) · [共享读档](../../src/ecm/stage2/input.cpp#L71) · [参数化公式](../architecture/ECM_PARAMETERIZATIONS.md)

按实际生产者生成的 save 冻结 `(N, curve, param, sigma, B1, X)`，逐族验证完成点、规范 `a24`、已知因子和最小界限，再比较 CPU/CUDA 支持范围内的续算结果。发布、去重、日志和队列身份沿用已有的曲线及 param 字段，覆盖「同 N/sigma/B1、不同 param」的冲突样本。Montgomery 自己采样性能，各族先与同一曲线及末点的参考对照，保留 Edwards 的基线身份。

## 7. 里程碑五：自研 CPU 多项式算术

GWNUM CLI 可用且整曲线基线冻结后，再开始替换算术后端。复用已验证的 F/G/H 几何、输入、覆盖和结果合同，先在固定计划上替换多项式乘法。以 GMP 直接卷积作 oracle，libecm NTT/plain 和本地 GWNUM 作性能参考。首个候选是短乘用 GMP 或 Karatsuba、长乘实验整数 NTT，分别测一般 N 与经过验证的梅森承载。对于 NTT，先证明 digit 卷积的系数上界、选定素数乘积的恢复范围和归约的正确性，再测打包、NTT/CRT、进位、模归约和临时空间。宽位条件下，依据 GWNUM 的差距分解再决定是否实验加权 FFT。已有 CPU 点算术和 Stage1 NTT 探针可用来选择低层对照条件，外层多项式树仍需独立测量。[CPU NTT 探针](CPU_STAGE1_NTT_DRAFT.md) · [现有 NTT 表示](../architecture/NTT.md)

验收顺序为精确卷积及高/低/循环窗口 → Newton 倒数、同几何 fold 与 Horner 对拍 → Edwards 完成点的完整曲线 → Montgomery 各 param → 同机六格性能和内存对照。覆盖非二次幂、尾块、最坏系数、复合环非单位、实测最大位宽、线程切换、取消和饱和回放。原生后端的启用范围按实测确定，先在相邻位宽和两档 B2 上证明完整曲线收益，再调整自动选择。最终用法、架构及性能结论分别回填现有主题，此计划文档保留供后续核对。
