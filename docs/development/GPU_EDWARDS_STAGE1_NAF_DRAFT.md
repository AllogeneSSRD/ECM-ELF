# GPU Edwards Stage1 窗口优化草稿

本草稿记录当前开发、验证及原始证据位置，完成后保留待用户处理。

## 目标与验收

- GPU1 RTX 4060 Laptop 上测量二进制 Edwards、Montgomery PARAM0/PARAM3 在常用不同位宽的完整 Stage1 性能，冻结优化前版本。
- 接入窗口 NAF，核对完整标量结果、复合数因子、所有容器档、任意分块停止恢复及 Stage2 交接；不能用性能样本替代正确性验证。
- 研究 Prime95 倍点算法，按实测选择有效实现；更新当前主题说明并提交 Git。

## 参考与当前问题

基线为 `fc66536945ca9dc796781236f9df827f66a284e0`，GPU Edwards 使用二进制倍加、256 位切片与普通域检查点。CPU Edwards 已有窗口 NAF，可作独立执行路径的末点核对。

第三方资料为 `.refactor/p95v3106b01.source/undoc.txt:367` 与 `ecm.cpp:2106`。文档列算法 4/5/6，但同版源码 `ecm.cpp:2143` 明确说明算法 6 已废弃，实际执行分支为 5、5a、7、4；不能据旧说明声称当前实现了算法 6。算法 5 源自 [Atnashev/Woltman, ePrint 2021/1061](https://eprint.iacr.org/2021/1061)，其 FFT 合并操作需要与 CGBN 模乘成本分开分析。

## 测量约定

同一 GPU、N、B1、曲线数及固定 sigma 起点；Edwards 标量为 48·lcm，Montgomery 为 lcm，比较的是各曲线族实际 Stage1 工作。不同容器/TPI 逐项记录。完整进程墙钟包含进程启动、指数/曲线准备、GPU、最终 GCD/归一化及保存，不能与 kernel 计时混称。每组先预热，再正反交错重复；保留失败样本。设备功耗/时钟使用原设置，不调整硬件。测量期间不并行运行 GPU1 测试。

TODO：冻结基线与源依赖；完成测量、窗口/倍点选择、正确性门禁和正式说明。

## 2026-10-16：NAF 编码门禁

`src/core/ecm_edwards_naf.h` 以两个 32-bit 位平面保存非零位置和有符号字典索引；非零间隔至少 w，因此索引不会重叠。最高位进位独立纳入列数，不与旧二进制 bitnum 混用。

`tools/build/test/build_edwards_naf_test.ps1` 使用实际 MSVC/GMP 构建后通过 46574 组精确重建：窗口 2..12、1..4095、2 的幂及前后邻值、31/32/33 等 limb 边界与确定性随机大整数。覆盖 603364 个负数位及 126047 个跨 limb 索引。测试还逐项检查数字范围、奇数性、非相邻间隔、最高非零位和加法计数；这只证明编码正确，GPU 执行尚待验证。构建身份在 `build_cuda_cmake/edwards_naf_test/manifest.json`。
