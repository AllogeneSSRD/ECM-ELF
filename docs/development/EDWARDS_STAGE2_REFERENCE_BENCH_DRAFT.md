# Edwards Stage2 CPU/GPU 基准草稿

本临时草稿记录 2026-10-17 的基准工作，完成后保留，待用户处理。实验原始输入、结果、日志、遥测及二进制均在忽略的 `data/experiments/edwards_stage2_reference_20261017/`；图表在 `data/figures/`，汇总在 `data/benchmarks/`。

## 测量约定

用户提供 Prime95 31.4b05 的 `4Core@4GHz` 四份日志：Edwards/Montgomery 各有 B2 请求值 2.6×10^10、2.6×10^11 两档。Edwards 13 个指数、26 个条件，B1=50000；sigma 除 M2003 为 27 外均为 26。Prime95 的 init、main、GCD 为三个互斥父阶段，完整 Stage2 是三者之和，不能仅取 main。

日志里的 M 标签不是实际模数。复用 `tools/log_parser/analyze_prime95_ecm.py` 和 `compare_prime95_gpu.py` 的恢复入口，用正式结果的完成记录校验和关联及 known-factors 精确整除恢复历史 N。原始身份、AID、校验值只留在私有快照，不写入本草稿、正式文档或代码。新测量按同一个 N、Edwards sigma、B1 和 Prime95 的调整后 B2 配对，不按指数/近似位数配对，不线性缩放 B2 耗时。

GPU 使用设备 1 RTX4060 Laptop；普通余因子与显式原始梅森指数承载分别测量。CPU 按用户要求测 NTT 和 plain 两个后端，规划内存 4096 MiB。每格正式样本 3 次，首轮保留，不丢弃预热，轮换实现顺序。CPU 进程亲和性默认限制到日志中同组逻辑 CPU 1/3/5/7。用户进一步确认当前 CPU 仍固定 4GHz，确认写入 `cpu_conditions.json`；未采集有效时钟计数器，WMI 的 2000MHz 不作为运行频率证据。程序不修改系统频率/功耗。

Stage1 输入准备不计入 Stage2；所有实现消费同一份严格校验的 48LCM 完成点。提前找到因子或出错的样本独立保存，不当作完整 NF 成本，不更改 sigma 来改善结论。GPU native 全流程计时与 CPU 构造加 libecm 返回存在边界差异，另存进程墙钟并说明差异。

## CPU 并行能力核对

用户指出当前 CPU Stage2 可能未实现多线程。源码核对：适配器编译时开启 OpenMP 并设置线程上限；普通多项式主路径 `stage2.c`/`listz.c`/`polyeval.c` 没有并行区；NTT 的表示转换有局部 OpenMP，但 `mpzspv.c` 中 `MPZSPV_MUL_NTT_OPENMP=0` 禁用了核心乘法的并行区。曲线按主程序循环串行消费。

`thread_probe_01/` 使用 M2203、B1=1000、B2=10^9、单曲线，NTT/plain 的 1/4 线程各三次、顺序交替。NTT 中位秒数 5.062164/5.059731，plain 为 3.417530/3.430955；总 CPU 时间/进程墙钟均约 1。这是该输入上的占用观测，不证明所有更大输入都不会触发局部并行，也不能把 OpenMP 线程数量等同于有用并行加速。

线程数设置不能标为“4 核并行实现”。用户同意稍后补测 Prime95 单线程后，主矩阵改为 CPU 线程上限 1；此调整在正式测量前记录，先前的 1/4 线程核对独立保留。Prime95 当前四线程实现单列，不把它与单线程 libecm 的差异写成算法本身的加速比。后续补测作为独立数据源，不覆盖现有历史证据。

收集器父环境没有显式 OMP/KMP/GOMP 变量，快照为 `openmp_environment.json`；主矩阵另显式设置 `OMP_WAIT_POLICY=PASSIVE`。同文件后部的 DCT/reciprocal 并行入口用于 `pm1fs2.c`，不能据此认定当前 ECM S=1 主路径已经并行。并行说明、线程探针及末点验证器已提交为 `e13a2ab`。

## 进行中的证据

`matrix_01/` 冻结四份原始屏幕日志、正式结果、三个执行程序及 GMP DLL，13 份共享 Edwards save 均已验证。前十份通过 Python x-only ladder 核对；为减少大输入准备耗时，独立 GMP x-only ladder 与已验证输入对照通过后用于其余点，并通过“错误末点但校验和有效”的拒绝测试。初始慢速准备进程由本任务主动停止，原始日志与已完成验证保留，后续准备没有覆盖原始快照或已验证 save。

计划包含 26×4×3 个正式调用。M503 的两档四路径首个正式调用已通过，不作为丢弃预热。随后按 GPU、CPU 两个测量批次推进，各自轮换实现顺序；每批调用的引擎集合、重复数及脚本身份另存 execution_batches。原生计时是主要性能指标；进程墙钟通过 250 ms 轮询观察退出，包含最多一个轮询周期的检测延迟，不作为小样本精确启动开销。中途子集不标为完整。待完成后把性能、图表、限制与优化参照写入正式性能主题，不把计划值当作已测结果。

GPU 的 156 次正式运行已全部完成，随后继续 CPU 的 NTT/plain 矩阵。CPU 初段与 0.31.0 结果元数据门禁有时间重叠；门禁没有与 GPU 基准重叠，首次临时核绑定验证中断后，常规设置的完整检查通过。验证区间及停顿证据见 `data/experiments/ecm_submission_v031_20261017/native_test_conditions.json` 和 `native_gates_01/02/`。主矩阵没有隔离整机后台负载，汇总时核对重叠样本的波动，需要补测时另存新证据，不覆盖已有行。

阶段审计 `audit_partial_03.json` 已核对 182/312 次调用，GPU 156、CPU 26，失败记录为 0。`preview_03/` 重新生成四幅 PNG/SVG：完整耗时、Prime95 曲线族上下文、GPU init/main、CPU 线程探针；修正了图注与横轴重叠，实图检查通过，全部预览保留 PARTIAL 标签。

CPU 测量期间另推进可选 NTT 长度计数器；GPU 基准不受新二进制影响，仍使用原冻结文件。CUDA 构建和开关性能对照的活动区间保存在 `data/experiments/stage2_ntt_counts_20261017/`。收集器每次完成时记录自身文件身份，运行期间不修改其源码；首次准备未接入 `--ladder-checker` 的问题先在工具说明中如实限定为续准备使用，待活动进程结束后再补接线和验证。
