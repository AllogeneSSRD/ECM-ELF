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

## 2026-10-16：GPU NAF 算术门禁

独立 CUDA 模块 `src/cuda/ecm_cuda_edwards_naf.cu` 已通过 GPU1 探针：35 个用例、441 个逻辑曲线组合，其中 NAF 384 条、原二进制对照 57 条。NAF 覆盖窗口 2/3/5/6/8/12、倍点算法 4/5、全部七档容器（256..17408）、部分 block 的 33 曲线、完整 Mersenne 和 M127·M89 复合输入；B1 为 20 或 1000。每条核对现有 CPU Edwards 的投影末点与 GCD、构造参数及扩展点不变量，完成态与第 17 列停止后改用 113 列切片恢复逐坐标一致。B1=1000 的窗口 6 案例实际有 1445 列，覆盖原标量 1444 位之外的 NAF 进位。

证据为 `data/experiments/edwards_naf_20261016/probe_01.log`、`probe_01/manifest.json`。探针 SHA256 为 `0e07c0f366fced6383654d4af5c8d55ee02845127ff297368825cb1c1c7b0a52`，sm89/CUDA13.3，正式性能样本数为 0。首轮构建遇到 CGBN load 不接受 const 指针，已按其接口修正；原始失败日志保留于 `compile_01/naf.log`。

生产驱动与 V1/V2 磁盘恢复仍待门禁，正式性能矩阵尚未完成。优化前基线 exe 已冻结于 `baseline_build/bin/`，SHA256=`3777416c9026027c88d2d56f78e6fbfa28a3886a7d78aa25ce9a500b66a534cb`；编译对象和扩建档位身份见 `baseline_build/identity_v2.json`，原始代码及 CGBN/GMP 头文件在 `sources_v2.zip`。M4003 的三曲线族试跑仅用于核对采集工具，不计正式性能。
