# OpenCL Stage2 点提交与多曲线研究草稿

状态：0.54.2已构建，显式点窗口与默认路径已验证；尚未自动选择chunk或并行曲线数，未更新生产目录。当前结论与复现入口统一见[性能报告](../performance/OPENCL_STAGE2.md#点提交组织与多曲线吞吐)，本文保留研究过程，完成后由用户决定处理。

## 诊断假设与反馈闭环

1. 宽位点SOS／private数组及组织不合适：只改group或内层展开，逐点对比完整X/Z，再看设备事件时间。
2. 单曲线提交点数不足：固定G批和NTT运算，扩大坐标窗口及chunk；NTT计数和每G批原始标量样本不得减少。
3. CPU准备、点生成和同步间隙可重叠：固定总曲线数，比较C1/2/4，分别记录完整吞吐、单曲线延迟、CPU及内存。
4. 多context可能只增加竞争：GPU点进程与CPU点分流分别测试，不能把多进程同时存在当作GPU物理并行。

首个多进程初筛在所有任务退出后立即查询Job，遇到active_processes退出记账延迟；改为有界等待完整进程树排空，原始失败保留在`small_p_screen/`。正式重复使用`small_p_repeat/`，未丢首轮。

## 实施

- PointGenerator允许一次生成多个G批的坐标；仍对每G批首／中／末从原Q、完整iD重算。末尾短批去重方式保持。
- `ECM_OPENCL_POINT_WINDOW=0..65536`，默认0；按已有batch额度估算上限并对齐原G批，不修改G树和fold划分。chunk继续独立限制单条设备命令。
- 参数在点生成前拒绝越界；不可逆种子、饱和分母、因子严格整除与失败排空继续由原路径处理。
- 多曲线工具只生成隔离任务，不并发消费生产worktodo。Windows Job计账覆盖继承子进程，不包含attach之前极短的启动段；峰值commit与NVML设备占用分开。
- 编译hint与睡眠等待只在`tools/experiments/opencl/point_codegen.py`的实验资源副本中测试，未修改生产内核或等待策略。PTX的local提取已限定声明行；旧探针结果中的程序级ld/st计数不能当作动态spill流量。

## 结果与验收

- [x] 小P：2位宽×3模式×C1/4×3组×4曲线，144次完整NF。CPU点吞吐2.61×／2.32×，GPU点1.07×／0.91×。
- [x] 大P：7995-bit、D30030、B2=26000000、最大NTT2²³，12次NF的n=1筛选。GPU点双worker退化，CPU点约1.50×；明确没有形成n=3生产吞吐结论。
- [x] 点窗口：4位宽×2B2×3变体×3遍，72次NF；逐格NTT、GMP checked_words、点数、坐标下载量和输入一致。
- [x] 大P窗口／chunk单变量：同save共9次NF，默认13.375935 s，chunk768为6.899059 s，窗口+chunk为6.903580 s，均为n=3中位数。大P只含一个G批，收益由chunk带来。
- [x] 0.54.2 RTX4060 Laptop设备点、CPU点模式、AMD gfx1150各18项生产门禁：真实正命中、普通／承载NF、预算/checksum/非法参数、队列停止和重排、窗口短批与裁剪；共54项。
- [x] 新窗口门禁初次误从可读控制台查找NTT摘要，实际摘要在文件日志，导致测试StopIteration；修正测试读取位置，保留失败目录`verification_window_gpu1/`，正式结果在`verification_window_gpu1_final/`。
- [x] 源码公开版本0.54.2；构建身份、工具和资源随实验冻结；CUDA默认实现未修改。

正式门禁：`data/experiments/opencl_concurrency_20261029/verification_window_{gpu1_final,cpu_gpu1,amd}/verification.json`。性能身份和条件见正式报告。该范围不等于16384-bit长标量、Intel或生产大B2资格。

## 待推进

- [ ] 单曲线按设备／位宽／标量长度选择chunk与窗口，加入最长命令和全卡资源约束，扩大验证后再改默认。
- [ ] 分段giant chain及退化恢复，继续使用原始标量和全叶哨兵。
- [ ] CPU点分流的有界多曲线调度与全卡总预算；同context跨曲线相同shape批处理需先去除全局重入限制。
- [ ] 长命令低CPU等待与短NTT自适应等待；verdict有界延后不能削弱检查。
- [ ] SOS小展开和NTT tile/outer继续保留为校准候选，不依据单个微基准直接推广。
