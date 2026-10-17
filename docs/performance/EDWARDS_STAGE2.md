# Edwards Stage2 CPU/GPU 基准

本页维护 Edwards 大余因子 Stage2 的性能参照。计算入口及后端选择见 [CPU Stage2 使用](../usage/EDWARDS_CPU_STAGE2.md)，GPU 管线见 [Stage2 架构](../architecture/STAGE2.md)。小位宽、B2=10^12 的既有 CPU 测量仍在使用页，使用其独立冻结版本和条件，不混入本矩阵。

## 输入与统计口径

输入来自用户提供的 Prime95 31.4b05 `4Core@4GHz` 四份完整屏幕日志：Edwards/Montgomery 各有请求 B2=2.6×10^10、2.6×10^11 两档，B1=50000。每个模型 13 个目标、26 个条件，每格历史样本 n=1。Montgomery 是独立曲线族的参考，不能当作相同 Edwards 末点的实现对照。

M 指数对应的实际模数是剥离已知因子的余因子：

\[
N=(2^p-1)/\prod_i f_i.
\]

恢复工具利用正式结果的完成记录关联和 known-factors 精确整除核对 N。新测量必须同时匹配 N、Edwards sigma、B1 和日志中的调整后 B2；不按指数标签或近似位宽配对，不按 B2 比例缩放耗时。除 M2003 使用 sigma27 外，Edwards 均为 sigma26。

低档调整后 B2 通常为 27,015,978,990，M6011 为 27,567,389,850；高档 M503–M3001 为 278,982,814,110，M4001–M14401 为 273,142,759,890，M16319 为 264,004,751,010。精确值保留在逐格 CSV 和计划快照中。

相同 B2 表示调用入口声明的覆盖范围一致。各实现仍可能采用不同多项式、分块和尾部覆盖；libecm 只保守报告请求界限。因此这组数据比较完整实现的成本，不代表各实现执行了相同数量的底层运算。

Stage1 使用相同的 CPU 标量 Edwards `48·lcm(1…B1)` 完成点，准备成本不计入 Stage2。13 份 save 均经过独立 Montgomery x-only ladder 验证；生产 Stage1 使用 Edwards 点公式。四条路径消费同一份 save，不再次乘 48。首次调用保留，每格正式样本三次，轮换实现顺序；不丢弃预热。提前找到因子的样本独立保留，不混入完整 NF 时间。

计时边界分别为：

- Prime95：初始化、主计算、最终 GCD 三个互斥阶段之和。日志中的 Stage2 complete Total time 仅为主计算，不能单独作为完整 Stage2。
- GPU：`stage2_full_wall.total=init+main`，包含必要自检、树初始化、主计算及 GCD；不包含 Stage1、驱动入口选型和进程启动。嵌套算子计时用于定位瓶颈，不再加到完整时间中。
- CPU：构造曲线至完整 libecm 返回，包含构造成本；不包含读档、进程启动和 JSON 发布。使用 S=1，显式 NTT/plain；两后端可选择不同多项式规划。

另存进程墙钟；收集器每 250 ms 观察退出，因此它包含最多一个轮询周期的退出检测延迟，不用于精确估计启动开销。图中三次样本显示中位数与最小/最大值，不是置信区间；Prime95 单次历史点没有误差估计。

## 设备、版本与证据

CPU 为 AMD Ryzen AI 9 HX 370、Windows 11。用户确认仍固定 4 GHz；没有采集有效时钟计数器，系统接口报告的 2000 MHz 不作为有效频率证据。本仓库 CPU 线程上限为 1，亲和性限制到逻辑 CPU 1/3/5/7，4096 MiB 是 libecm 规划预算，不是进程峰值或硬限制。Prime95 历史数据为四线程、4 GHz；两者不按线程数换算，也不将差异归因为算法本身。

Prime95 日志中的 available memory 均为 4096 MB，using memory 随输入为 118–3922 MB；这里保留日志的单位和报告含义，不把它当作本工具测得的单进程峰值工作集。没有隔离整机后台负载，本仓库另外保存 CPU 时间/墙钟占用与 GPU 设备遥测。

CPU 依赖为 GMP-ECM 7.0.6-dev，上游提交 `17b5cdf1536b96fce4a7b92832f57f9028fbe570`。普通多项式主路径串行，NTT 的部分表示转换有 OpenMP，核心 `mpzspv_mul_ntt` 的并行开关关闭，曲线逐条串行处理。`--threads 4` 不是四核 Stage2 实现。更大输入是否触发局部并行须结合实际占用判断。

GPU 使用设备 1 RTX4060 Laptop，驱动 610.88；程序不修改频率或功耗。arena6300/fold640/batch256 MiB，D 自动选择，关闭短标定、运行时 tune 与自动承载，分别显式指定普通余因子和原梅森指数承载。两侧 D/P 可能不同，所测收益包含规划差异，不能单独归因于归约算术。其他文档中的 1800 MHz 或更高时钟数据属于各自实验，不直接混合。

冻结 GPU 为 sm89/CUDA13.3.73/MSVC14.51.36231、PTX3/addsub1；三个 EXE SHA256：

- Stage1：`774b7cfb0ff72d5e2afdf941e21bd5a0e28671c78c260d1d5875a5ae6ce74c74`。
- CPU Stage2：`a75565dc040ee76daa6d1962d54aa52ae63f3c342818e73b09d6340ff402d28f`。
- GPU Stage2：`e67d5e9adb20fb96185acc292185b58828a39c0a2a8084de90ff0ed17bb85de4`。

三者共用 GMP Zen3 DLL，SHA256=`969f0e96ac587dedfecf611ca3442fc7211f9218f1a553214c83548dcdbcfd6e`；Prime95 EXE 的身份另保存在计划中。运行时依赖也冻结到实验目录，不按当前构建目录的同名文件推定版本。

证据根目录为 `data/experiments/edwards_stage2_reference_20261017/`。`matrix_01/` 保存原始日志与正式结果私有快照、13 份 save、EXE/DLL、逐调用结果/日志/遥测和计划；`compiled_sources_02/` 保存与 GPU 构建清单匹配的依赖源码。原始账户字段、AID 和结果校验值不进入公共 Git。

相同 EXE 已分别通过 CPU 42 项、GPU 29 项管线回归，证据在 `data/experiments/ecm_submission_20261016/cpu_regression_02/`、`gpu_regression_03/`，涵盖独立末点、因子、损坏拒绝及结果发布。基准另逐调用核对实际目标、后端和完成状态；GPU 必须完整关闭时间账本，GMP 自检及抽检数量为正、bad=0。上述检查是数学与本地流程验证，不代表服务器已经接受第三方结果。

## CPU 线程探针

同一冻结 CPU EXE，M2203、B1=1000、B2=10^9、4096 MiB 规划预算，1/4 线程各三次、顺序交替：NTT 中位时间为 5.062164/5.059731 秒，plain 为 3.417530/3.430955 秒，CPU 总时间/进程墙钟均约 1。这说明该输入基本使用一个核心，不能据此断言所有更大输入都不触发局部并行。CPU 时间/墙钟表示占用，不是有用工作的效率。

证据为 `thread_probe_01/summary.json`；实现入口为 [线程探针](../../tools/bench/bench_cpu_stage2_threads.py)。

## 复现与待验证条件

使用 [精确输入基准工具](../../tools/bench/README_EDWARDS_STAGE2_REFERENCE.md) 准备、执行和绘图；[独立审计](../../tools/test/check_edwards_stage2_reference.py) 核对输入身份、实际后端、冻结文件与调用完整性。汇总和图表输出到忽略的 `data/benchmarks/`、`data/figures/`，原始证据不覆盖。

GPU 普通/承载两条路径的 156 次运行已经完成；CPU 两后端仍在采集。当前冻结审计通过 182/312 次调用，失败记录为 0；这份阶段审计保存在 `audit_partial_03.json`。预览图已检查排版，位于 `data/figures/edwards_stage2_reference_20261017/preview_03/`，对应汇总在 `data/benchmarks/edwards_stage2_reference_20261017/preview_03/`，图上明确标记 PARTIAL。

TODO：完整 CPU 三轮及最终对比图仍待采集完成，不将当前预览标为完整。CPU 初段与结果门禁、NTT 计数器开发/开销对照有后台负载重叠；构建与测试区间另存证据，不丢弃首轮来改善结论。Prime95 单线程数据由用户稍后补测；若它选择不同的调整后 B2，需补充相同 B2 的本仓库测量，再进行同口径比较，不线性换算。
