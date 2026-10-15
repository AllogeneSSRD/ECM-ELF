# GPU Edwards Stage1 与 Stage2 开发草稿

目标是让 CUDA Stage1 执行 Atkin–Morain Edwards 曲线，并把其完成点交给 CUDA 多项式 Stage2。当前 `curve=ed,target=gpu` 仍由 [Stage1 选择检查](../../src/core/ecm_stage1_selection.h)拒绝；[Stage2 入口](../../src/core/ecm_cuda_stage2_main.cpp)仍只读 Suyama PARAM0 文本 save。因此当前进度是点算术验证，不是两阶段生产功能。

## 当前开发轮次：GPU 扩展坐标算术

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

## 接入顺序与待验证条件

1. 把 GPU Edwards 点状态纳入 Stage1 的批次与切片调度，复用指数、停止、检查点、错误及 save 发布语义；曲线构造失败和非单位分母须在目标 N 上正确分类与求因子。以 CPU/Prime95 独立末点、重启后末点和实际 Mersenne 任务为门禁，之后才开放 `curve=ed,target=gpu`。
2. 定义带曲线身份、参数和完整校验的 Edwards 交接 save。当前 CPU Prime95 二进制 MIDSTAGE 记录 Qx/Qz，而 CUDA Stage2 只接受 Suyama PARAM0 文本记录；不能把 Edwards sigma 当 Suyama sigma 重新构造曲线。
3. 从 Edwards 曲线参数求出 Stage2 Montgomery 倍点所需常数，独立核对 Prime95 的 `Ad4` 约定与本仓库 xDBL 约定，接入 baby、giant 的点生成。以小素数直接参考、实际 Mersenne 保存点、非单位路径及完整多项式结果交叉验证。
4. 在 GPU 1 上做完整 B1/B2 曲线、因子与无因子回归，验证队列重启和结果持久化，再更新正式使用与性能文档。

数学与 Prime95 来源见 [Edwards 固定参考](../reference/ECM_EDWARDS_MATH.md)；现有 Montgomery Stage2 的输入与运算界限见 [Stage2 管线](../architecture/STAGE2.md)。
