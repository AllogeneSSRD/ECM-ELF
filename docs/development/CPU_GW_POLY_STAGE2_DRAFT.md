# CPU GWNUM 多项式 Stage2 开发草稿

目标：从现有 Edwards 48LCM Stage1 文本 save 直接继续 Stage2，增加实际多线程的快速多项式后端，并与 Prime95 完整 Stage2 耗时比较。原 libecm NTT/plain 保留为独立正确性与性能参考，不将线程上限当成并行实现。

计划使用本地 Prime95 31.6b01 源目录提供的 x64 GWNUM/polymult 库；本项目实现点序列、批量归一化、产品树、Newton 倒数、G/H 模 F 折叠及 scaled 下降。所有逆元和因子判断针对实际 N，梅森承载仅在精确验证 N 整除原 M 后使用。暂不承诺与 Prime95 相同性能，须经实测确认。

实现先验收随机复合模数下的多项式乘法、倒数、余式和求值，再验证已知 Stage2 因子与独立小界限 oracle、存档与中断语义。性能使用完整 Stage2 边界，固定输入、线程与预算，保留首轮；原始证据在 `data/experiments/cpu_gw_poly_stage2_20261017/`。不恢复已停止的旧 CPU 基准矩阵。

本草稿保留，完成后由用户决定处理。

## 实现与接口

新增 `src/cpu/ecm_stage2_gwpoly.{h,cpp}`，独立算术入口接收实际 N、Stage1 仿射 x、a24、B1/B2、线程与内存规划选项。当前用户入口仍为 Edwards 48LCM 文本 save；不兼容旧二进制 tmp，不启动 Prime95。GW 构建的 auto 选择此后端，显式 NTT/plain 及无 GW 构建保留原 libecm 路径。

实现完整的 baby/giant 点序列、分段批量求逆、保留 F 树、双工作区 G 树、Newton 倒数、截断商/余式折叠，以及反转状态的循环高段 scaled 下降。G 工作区跨层、跨块复用，F 每层连续存储；底层节点和点序列使用独立克隆句柄并行，大乘法使用 polymult helper。主路径不对整个 B2 区间筛素数，只补算不在正 giant 索引内的小素数。

真实 GW FFT 在内部记录中输出，并作为内部 CRC32 第 8 槽；提交格式的 top-level fft-length 继续使用既定参考值，新增实际 stage2-fft-length。D/P 紧随 B2。GW 采用实际 FFT 布局做预算规划，并对所需 polynomial safety margin 预留空间；运行门限保持 maxerr≤0.40，错误或停止不发布 NF。

新增功能版本为 0.33.0。构建入口 `tools/build/test/build_cpu_stage2.ps1 -GwPoly`；依赖为本地 Prime95 31.6b01 目录中的 GWNUM/polymult x64 库，与其静态 CRT 匹配。项目未改写第三方源码。

## 诊断与修复依据

- 首次 helper 崩溃：裸 `polymult_launch_helpers` 分派的是调用方的用户回调，当前没有设置该回调；移除这个调用，实际 polymult 自行调度内部任务。早期崩溃及探针日志保留在实验目录。
- 小模数预算规划：先按真实 FFT 布局探测，不再用猜测的每系数字节数拒绝可行的固定 D。`validation_01/` 的失败证据保留。
- FFT 余量重复扣除：`gwset_polymult_safety_margin` 会从 EXTRA_BITS 中扣除预留，选择时须比较未预留容量，不能再将已扣后的容量与同一需求比较。来源为本地 gwnum.h / gwnum.c；修复减少了不必要的 FFT 膨胀，没有放宽错误门限。
- 大 wheel 并发：M347、B2=2.6×10^12、四线程在 baby 点生成时触发约 0.5 的舍入误差，单线程通过。GW add/subtract 在源 FFT 状态不同的时候会原地转换源操作数，因此共享点常量存在并发写入。各线程复制 Q、step/base 和 a24 后，同一四线程输入通过，三处 GMP 投影一致，maxerr≈0.001709。原始失败在 `large_bounds_01…04/`，单线程对照在 `large_bounds_05/`，修复验证在 `large_bounds_06/`。正式源码已移除全部 DEBUG 探针。

## 验证与频率条件

`validation_04/summary.json` 的 29 项门禁通过，覆盖 1/4 线程复合环的 Newton/余式/Horner/多次 fold、CPU/GPU1 存档交接、64 位 sigma、三处完整 GMP 投影、独立小界限 Stage2 因子 oracle、Stage1 已有因子、损坏和目标元数据拒绝、真实停止事件不生成 NF，以及内部 CRC 和提交批次。大 B2 的成功范围为 M347、B1=1000、B2=2.6×10^12；整数参数上限 10^15 不作为相同规模的性能或正确性实跑结论。

`libecm_regression_01/summary.json` 的 39 项原 CPU 流程门禁通过，确认未启用 GW 构建的 auto/NTT/plain 和保存点、输出、中断行为。构建日志与不同版本证据分别保留，不按当前同名二进制推定原始身份。

用户澄清：本轮 ECM-ELF 早期原型试跑为 3.1 GHz，目前固定 4 GHz；Prime95 `4Core@4GHz` 历史记录仍为 4 GHz。`prototype_perf_*` 和 `affinity_probe_*` 只作优化探索证据，不进入正式性能结论；不按频率比例换算。`large_bounds_01` 跨频率调整，仅作为失败算术证据。澄清另存 `frequency_clarification.json`，不改写原始日志。

正式性能冻结在 `fixed4g_01/`：相同已独立验证的 Stage1 save，三种实际余因子位宽、两档 B2、1/4 线程各三次，保留首轮。收集器保存 EXE/DLL、GW 库、链接的 libecm 静态库、源码/头文件、原计划、脚本和每次进程指标。正式结论与图表集中维护在 [CPU 性能](../performance/CPU_GWPOLY_STAGE2.md)，使用方式见 [CPU Stage2](../usage/EDWARDS_CPU_STAGE2.md)。本草稿不作为性能权威页面。

## 未完成边界

当前曲线内部没有恢复点，错误没有自动扩大 FFT 后重跑；GUI/队列和自动 Stage1→Stage2 尚未集成。需要扩大一般模数、其他 CPU、生产 B1、大位宽 B2≥10^12、低预算及内存失败覆盖。静态 D 成本模型不等同于 Prime95 的完整调优器，性能差距按正式数据保留，不把共享 GW 算术库表述为完整 Prime95 功能或通用性能相等。

最终冻结版本验证：`validation_05/` 的 31 项门禁通过，与 `fixed4g_01/bin/` 的 EXE SHA256 一致，补齐 10232/16272 位实际余因子的三处完整 GMP 投影。正式 4 GHz 采集 36/36 完成、无失败，两个 PNG/SVG 图已核对；性能差距及未完成边界按正式性能主题保留，不将旧 3.1 GHz 原型混入。

同一冻结 GW 构建内的 NTT/plain 显式路径另在 `legacy_in_gw_build_01/` 通过两项 M347 保存点续算检查，覆盖静态 CRT 构建中的原后端共存；无 GW 构建的 39 项回归仍保留其独立身份。
