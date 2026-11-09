# Stage1 NTT 优化开发记录

## 0.59.2 逆向 tile 直接加载频谱

将乘积的操作数指针直接传给 Stage1 首个 inverse tile。每块处理一个曲线和乘积，沿用原有变换顺序，平方只读同一个输入。共享的 tile butterfly 与逆向外层调度保持统一，独立的 Stage2 tile 入口仍按原方式读写。Stage1 少启动三个 gather kernel，频域临时数组每曲线减少 `4·L` 个 64 位字。

本地 0.59.1 与 0.59.2 均为 CUDA 13.3、sm89、CGBN 5120 档。RTX 4060 Laptop GPU1，驱动 610.88，频率和功耗设置未调整。用完整梅森数、Montgomery PARAM0、LCM 和 sigma 11 起点，关闭定时检查点，按 AB、BA、AB 交替启动，比较三次设备 event 的中位数。计时覆盖阶梯和提交间隔，排除曲线与标量准备、Graph 建立、末点回读和发布。

- B1=256，4003、8191、16001、32749、65521、131071 位，每档 1、4、8 条曲线。18 条件的改善中位数 4.45%，范围 −1.35% 至 7.85%。32749 位单曲线 48.733 → 49.391 ms。
- B1=4096，32749、65521 位各 1、4 条曲线。三个条件改善 2.28–2.57%；32749 位单曲线 751.689 → 763.386 ms。保留这项 1.56% 退化，后续可按实际批次复查 tile。
- 132 次性能运行逐条比对归一化 X、因子与 save checksum。生产验收另覆盖直接提交、Graph、CGBN 与 Python 对照、构造和末点因子，以及改换计划后的真实断点恢复与 Stage2 队列交接。
- 原始数据与 200 ms 功耗采样在 `data/experiments/stage1_ntt_fused_0592/` 的 `matrix/`、`long/`。复测入口为 [compare_stage1_ntt.py](../../tools/bench/compare_stage1_ntt.py)。
