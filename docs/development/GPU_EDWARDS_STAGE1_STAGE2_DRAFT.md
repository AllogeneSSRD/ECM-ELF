# GPU Edwards Stage1 与 Stage2 开发草稿

目标是让 CUDA Stage1 执行 Atkin–Morain Edwards 曲线，并把其完成点交给 CUDA 多项式 Stage2。当前源码已连接批量 worker、二进制分片、普通整数检查点和 Edwards 身份明确的文本 save；Stage2 接入同曲线的 Montgomery 常数，复用生产 NTT 管线。当前规范入口为 [Stage1 GPU Edwards](../architecture/STAGE1.md#gpu-edwards)和[Stage2 输入](../architecture/STAGE2.md#stage1-save-与承载资格)。本文保留开发轮次与证据，完成后由用户决定如何处理，不自动删除。

## 初步开发：GPU 扩展坐标算术

[cgbn_edwards_point.cuh](../../kernels/cuda/cgbn_edwards_point.cuh)实现 `a=1` 的扩展 Edwards 加法和倍点，使用 CGBN Montgomery 域、规范模数余数及 `(X,Y,Z,T)`。调用方负责提供已构造的曲线参数 `d`、基点、奇数模数，以及比模数多至少一位的容器；点公式不执行模逆。加法和倍点允许输入输出点重叠。

[GPU 探针](../../tools/test/edwards_gpu_point_probe.cu)由 CPU 的 `edwards_atkin_morain` 构造曲线，用 `48·lcm(1…B1)` 生成标量，GPU 逐位执行倍点和条件加法。独立 [构建入口](../../tools/build/test/build_edwards_gpu_point_probe.ps1)生成测试程序。探针检查 `XY=ZT`、Edwards 曲线方程，并将 GPU 结果的 `(Z+Y):(Z−Y)` 与 CPU `edwards_stage1_curve` 的点交叉相乘比较；投影缩放不同不按原始坐标逐字比较。

2026-10-15，在 GPU 1（NVIDIA GeForce RTX 4060 Laptop GPU，sm_89）上构建并运行，5 组各 4 曲线全部匹配，坏计数为 0：

| 目标 N | B1 | 标量位数 | CGBN 容器 |
| --- | ---: | ---: | ---: |
| 2^127−1 | 20 | 34 | 128/TPI4 |
| (2^127−1)(2^89−1) | 20 | 34 | 256/TPI4 |
| 2^347−1 | 1000 | 1444 | 384/TPI4 |
| 2^521−1 | 1000 | 1444 | 768/TPI8 |
| 2^677−1 | 1000 | 1444 | 768/TPI8 |

二进制与源码 SHA256、构建日志、GPU 1 测试输出、设备身份及 CUDA 版本保存在忽略目录 `data/experiments/gpu_edwards_20261015/`。当前验证覆盖点结果，不证明生产 B1 性能、因子命中、检查点或 Stage2 正确性。

## 初步接入计划记录

1. 把 GPU Edwards 点状态纳入 Stage1 的批次与切片调度，复用指数、停止、检查点、错误及 save 发布语义；曲线构造失败和非单位分母须在目标 N 上正确分类与求因子。以 CPU/Prime95 独立末点、重启后末点和实际 Mersenne 任务为门禁，之后才开放 `curve=ed,target=gpu`。
2. 定义带曲线身份、参数和完整校验的 Edwards 交接 save。当前 CPU Prime95 二进制 MIDSTAGE 记录 Qx/Qz，而 CUDA Stage2 只接受 Suyama PARAM0 文本记录；不能把 Edwards sigma 当 Suyama sigma 重新构造曲线。
3. 从 Edwards 曲线参数求出 Stage2 Montgomery 倍点所需常数，独立核对 Prime95 的 `Ad4` 约定与本仓库 xDBL 约定，接入 baby、giant 的点生成。以小素数直接参考、实际 Mersenne 保存点、非单位路径及完整多项式结果交叉验证。
4. 在 GPU 1 上做完整 B1/B2 曲线、因子与无因子回归，验证队列重启和结果持久化，再更新正式使用与性能文档。

数学与 Prime95 来源见 [Edwards 固定参考](../reference/ECM_EDWARDS_MATH.md)；现有 Montgomery Stage2 的输入与运算界限见 [Stage2 管线](../architecture/STAGE2.md)。

## 2026-10-16：批量 worker 与 NTT Stage2 接入

新增 [批量 CUDA 模块](../../src/cuda/ecm_cuda_edwards.cu)、[worker](../../src/core/ecm_edwards_gpu_driver.inl)、[带失败分类的曲线构造](../../src/core/ecm_edwards_curve.h)和[交接格式](../../src/core/ecm_edwards_io.h)。曲线辅助倍点采用 Jacobian 坐标，模逆失败在实际 N 上返回因子或退化结果；不继承 CPU 参考中忽略逆失败的行为。GPU 保持标准扩展坐标，二进制标量位号与 NAF 格式隔离。安全停止和周期检查点沿用原子替换与备份轮换；恢复核对完整任务身份、基点和累加点不变量。

Prime95 本地副本 `ecm.cpp:4409–4412` 明确保存 `Ad4=(dD-dN)/dD=1-d`；本仓库 `s2g_xdbl` 使用 `(A+2)/4`，因此 Edwards 交接传 `1/(1-d)`。Stage2 从目标 N 与 Edwards sigma 重建该值，不调用 Suyama 构造；baby、giant、小素数和多项式路径共用同一常数。Stage1 不可逆归一化得到的真因子使用明确的 FACTOR 记录，Stage2 在进入多项式运算前验证因子。

[批量探针](../../tools/test/edwards_gpu_stage1_probe.cpp)在 GPU1 RTX 4060 Laptop（sm_89）上通过 57 条曲线：M127/B1=20/33 条，M127·M89/B1=20/4 条，M347、M677、M991、M2203、M4003/B1=1000 各 4 条。曲线参数、末点和分母 GCD 均与现有 CPU 实现核对；在第 17 位中止，再以 113 位片长续跑，结果与完整运行逐坐标一致。构建入口为 `tools/build/test/build_edwards_gpu_stage1_probe.ps1`，日志在 `data/experiments/gpu_edwards_pipeline_20261016/probe.log`。

worker 构建入口为 `tools/build/test/build_edwards_gpu_workers.ps1`；[端到端门禁](../../tools/test/test_edwards_gpu_pipeline.py)独立以 Python 整数及 Montgomery ladder 计算完整 48·lcm 参考点，并直接求 Stage2 每个 monic 叶子的乘积。2026-10-16 的 `gate_03/summary.json` 为 complete=true，29 个命令检查通过，包含：

- M127、M347、M677、M991、M2203、M4003：B1=1000，各 4 条 Stage1 曲线；首条接 B2=13230、D=210 的完整 Stage2。每组 24 个 monic 叶值的完整摘要均与 Python 直接乘积一致，返回因子的乘积 GCD 也与参考一致；M4003 的真因子为 16756559。
- N=2^128+1、sigma=17、B1=20：普通与 p=256 承载路径在目标 N 上的 24 个叶值摘要一致，并与 Python 参考一致。
- N=1009·M127：Stage1 真因子记录与 Stage2 读入因子路径通过。N=10007·M127、sigma=17、B1=20、B2=40000：Stage1 无因子，Stage2 找到 10007 的因子分量；所有返回值逐项验证为 N 的真因子。
- M347、B1=100000：真实进程接到继承事件后停止，队列保持未完成；损坏主检查点后从 `.bak.ckpt` 恢复，最终 save 与相同输入的完整运行逐字节一致。恢复所得点继续执行 B2=200000 的 Stage2。
- 方法、标量约定、checksum、sigma 四类损坏均拒绝；原 Suyama PARAM0 路径仍匹配独立完整叶值摘要。Stage2 队列执行完成后再启动，结果文件不增加重复记录。

[容器与发布失败门禁](../../tools/test/test_edwards_gpu_boundaries.py)的 `boundaries_02/summary.json` 为 complete=true，9 个命令检查通过：M1279、M4423、M8191、M9689，B1=20、各 1 曲线，覆盖 2048、8192、17408 容器，与独立参考末点一致；M9689 再完成 B2=13230、D=210 的 24 叶值全摘要核对。sigma=2^64−1 的单曲线正确，跨 2^64 的批次和 Edwards lcm=12 均拒绝。故意令 Stage1 save 目标不可写时，任务仍留在 worktodo，不写 finished。结合前述探针与闭环测试，七种容器均有代表位宽的正确性样本，但尚未穷尽各容器边界。

全部运行使用 GPU1，设备 UUID、驱动、功耗上限查询结果、CUDA 版本、二进制 SHA256 与构建身份保存在 `data/experiments/gpu_edwards_pipeline_20261016/`。早期探针的曲线构造边界处理更新后，复用未改变输入和编译命令的 CUDA 对象，重建主机探针，57 条曲线再次通过；对应 `probe_current.log` 与 `probe_current_manifest.json`，全部登记源码 hash 与当前文件一致。功耗和时钟未锁定，本轮只做正确性验证，正式性能样本数为 0，不给出加速比。`gate_01` 的诊断采集遗漏和 `gate_02` 对 M4003 无因子的错误假定均保留原始日志；最终门禁读取独立诊断文件，并用直接乘积/GCD 判定真实因子，不放宽数学比较。

复现顺序：先运行两个 `tools/build/test/build_edwards_gpu_*.ps1` 构建入口；运行 `edwards_gpu_stage1_probe.exe 1 <新输出目录>`；再运行 `python tools/test/test_edwards_gpu_pipeline.py --stage1 <Stage1 exe> --stage2 <Stage2 exe> --output <新目录> --device 1` 及同参数的 `test_edwards_gpu_boundaries.py`。输出应位于 `data/experiments/`，不能覆盖已有原始证据。Stage1 路由单元回归与配置生成器的 `--check` 也已通过。

当前限制和后续门禁见正式 [GPU Edwards](../architecture/STAGE1.md#gpu-edwards)及 [TODO](../TODO.md)：GPU NAF 与吞吐调优、生产 B1/B2、各容器边界、其他发布架构、GUI 位进度、Edwards Auto B2 成本资格及 Prime95 二进制输入仍需后续工作。本文保留，不自动清理。
