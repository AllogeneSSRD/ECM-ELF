# Stage2 tile/outer 与准备同步草稿

状态：联合计划与准备归属分析完成；生产默认策略保持当前设置。

## 范围

- 比较tile10–13与M5–M8，保留生产自动表基线；先密集GMP及完整数组检查，再做固定D曲线。
- 从本进程GPU事件并集的空隙出发，区分CUDA API等待、分配/释放和API外准备；不把嵌套API秒数当作纯拷贝成本。
- GPU1 4060 Laptop保持用户55W/1800MHz上限设置，不调整设备控制；不干扰GPU0外部任务。

原始证据置于 `data/experiments/stage2_tile_sync_20261015/`，汇总和图在根data对应子目录。候选须通过独立GMP、故障注入、完整曲线、必要的管理员剖析；局部吞吐不等于生产收益。

## 实施与验证

- 扫描tile10–13×M5–M8加生产基线，固定CTA512；k20–25/batch1和4，k26–27/batch1，共1428正式样本、108,452,118,528个最终A/B words通过。动态shared函数上限固定到最大tile，避免小tile覆盖大tile启动能力。
- 密集GMP：102项、12,533,760 words及专用bad1故障通过；2592项策略边界、352,321,536 words缓存切换及专用故障通过，原Mont/Goldilocks/coop门禁保留。
- 新增默认关闭NTT_FUSE_JOINT，仅在受支持自动表下选k23 tile11/M6和k25 tile12/M7，包含中段M6。显存描述和arena命中都使用实际tile，旧严格cprof拒绝该实验配置。
- 80条固定D正式曲线的中位数之和55.803204→55.695035秒（0.19%），没有默认部署资格；24条控制/承载复测未稳定复现首轮4.22%/−0.39%。8条完整目标叶子诊断一致。全部原始样本保留，不用复测替换首轮。
- 增加默认关闭NTT_HOST_TRACE与CPU RAII范围；NVTX3随CUDA头文件提供，没有额外运行DLL发布要求。首次包含Windows SDK出现small/scr2宏冲突，以WIN32_LEAN_AND_MEAN/NOMINMAX隔离后编译通过。该失败属于构建问题，未进入任何性能样本。
- 管理员采集完整API、基础CPU范围及扩充CPU范围，共4条本主题Systems曲线；加上原中段参考，空隙口径可交叉核对。扩充剖析构建与正式构建SASS全文hash完全一致；剖析构建另有独立正常曲线输出/工作量核验，秒数不混合。
- GPU窗口空隙0.598152/6.349079秒；API外约0.256秒，其中S4准备45ms、常量初始化32ms、主机树驱动30ms、Mont启动检查20ms、未标记79ms。H2D/D2H真实GPU事件仅约26ms，数秒同步API持续时间主要与GPU执行重叠。
- NTT请求memset写84.3GB、GPU约0.299秒，pack约0.257秒。下一优先方向为完整输出覆盖的pack/zero/gather融合；仍须写零，不能把全部memset时间直接计为收益。tile11/CTA256尚未测量。
- 最终探针构建重新通过原GMP/故障、joint密集GMP/故障和策略缓存切换/故障；负profile mode返回2。生产构建对JOINT/HOST_TRACE的2、−1、bad、1x均返回2；最终范围追踪构建的一组完整短曲线A/B输出与工作量一致。此处17个算子冒烟样本和2条曲线均不追加到1428/80正式样本。最终NVTX收集校验要求捕获到实际CPU范围，已有trace通过。

正式说明为[outer性能](../performance/NTT_OUTER.md)和[准备与同步](../performance/STAGE2_PREPARATION.md)，复现见[工具说明](../../tools/bench/README_NTT_OUTER_PLANS.md)。本草稿保留供用户决定整理，不自动删除。
