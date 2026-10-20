# Stage2 pack / zero 对照工具

范围、公式、硬件及性能结论统一见[准备与同步](../../docs/performance/STAGE2_PREPARATION.md#packzero-融合路径)。独立生产程序默认开启融合，对照工具显式指定两侧0/1；工具不修改显卡时钟或功耗。

## 独立探针

- 构建：`tools/build/test/build_stage2_pack_zero_probe.ps1 -Build <独立目录> -Arch sm_89`。
- `<exe> <device> --check`：GMP位提取oracle检查全部输出及guard，正常663项、38,030,604 words、bad0；另检查1,048,576个快速商余数。原pack与新pack均验证。
- `<exe> <device> --fault`：仅翻转融合结果的一个尾部digit，必须bad1、退出3。
- `<exe> <device> --bench <bits> <P> <batch> <path> <repeats> <iterations>`：path0/1/2为contiguous/gather/reverse；交错两路径，以设备事件计每操作数准备时间，完整数组结果须一致。
- `benchmark_stage2_pack_zero.py --exe <probe> --output <新目录> --device 1`：先执行上述门禁/故障，再运行21个形状，默认6样本、每样本8次。无丢弃预热；两侧均预先上传源数据，计时只包含memset+pack或融合pack/zero。保存二进制/源码身份、原始日志、GPU采样与中位数/MAD。

探针从实际生产头文件调用原/新packer；GMP oracle按系数逐个提取位段，不使用融合kernel的索引除法。完整大数组的路径对照不代替独立oracle。

## 完整曲线与剖析

`benchmark_stage2_graph.py --variant pack-zero --exe <生产Stage2 exe> --cases <cases.json> --output <新目录> --device 1 --repeats 4`。

复用固定D工具，Graph/MID_COOP/JOINT关闭，只改变NTT_S4_PACK_ZERO=0/1。每条曲线核对GMP、因子、rootfold、数学NTT长度分布及S4分块，且要求on侧实际覆盖digit；输入、曲线和预算字段见[完整Stage2工具](README_NTT_OUTER_PLANS.md#完整-stage2)。`--diagnostic`另核对完整叶子指纹，其运行不进入性能样本。

pack-zero对照可使用其他生产设备；中段M6/joint独立对照仍限制为已测4060LP。case的`d=0`使用当前自动D选择；两侧的算术结果、NTT分布、分块、自检、GMP抽查和S4 launch数量仍须一致，不用改变D制造局部收益。

case可含`environment`对象，对两侧共同设置严格0/1的`NTT_SHORT_GRAPH`、`NTT_FUSE_MID_COOP`、`NTT_FUSE_JOINT`、`NTT_WORKSPACE_REUSE_BQ`、`NTT_ARENA_WORKSPACE_POOL`，分别验证组合；不改变pack-zero的A/B开关。中段及joint表仍受设备适用性检查约束。`arena_mb`、`fold_mb`可测试预算回退，`batch_mb`控制分块预算。不同设备顺序测试，避免CPU准备或其他GPU工作相互干扰。

`--telemetry-ms 500`为默认GPU采样间隔；0关闭本工具的周期NVML采样，用于辨认监测干扰。关闭时gpu.csv为空，报告不声明连续频率证据；锁频条件须另行记录。command.json记录NTT/CUDA环境，源脚本版本与采样设置保存在identity.json。采样关闭后仍可能有系统或其他程序的等待，不能自动归因于监测工具。

正式运行完成后，使用`profile_stage2_graph.py --reference <已完成command.json> --output <新目录> --kind systems --all-cuda-apis --host-trace --run-admin`采集同一曲线。运行前应确认没有本任务的GPU计时并发。采集器还会核对融合开关与实际覆盖量，剖析秒数不混入A/B。

`analyze_stage2_cuda_gaps.py --trace <trace.sqlite> --device 1 --output data/benchmarks/<名称>/gaps.json`按互斥区间分解GPU空隙。API等待、GPU kernel/copy/memset累计时间及空隙交集不可相加。

## 汇总与图

`analyze_stage2_pack_zero.py --operator <探针summary.json> --stage2 <完整曲线summary.json> --output data/benchmarks/<名称> --figures data/figures`。

生成准备成本及完整矩阵图、operator/stage2 CSV与汇总JSON。矩阵总计是各格中位数之和；不同轮次、诊断、剖析和复测保留独立样本，不相互覆盖。

可同时指定`--systems-off <默认采集目录> --systems-on <融合采集目录>`，核对二进制、输入、路径及算术/工作量，生成`profile_comparison.json`。其中pack+memset为诊断事件累计时间，与正式曲线分开；GPU间隙仍由独立的互斥区间工具计算。

`analyze_stage2_pack_zero_matrix.py --study "GPU1-main=<完整矩阵目录>" --study "GPU0-fixed=<完整矩阵目录>" --output data/benchmarks/<新目录> --figures data/figures --prefix <唯一图名>`生成多矩阵CSV/JSON、各格收益条图和实际N位宽/B2热图。普通和承载分开，频率/功率从高负载GPU采样提取；单独复测不覆盖原始矩阵，不混入其总计。模块`full_peak`核对相同，不作为进程显存峰。

## 配置接线验证

配置键为`stage2_pack_zero`，CLI为`--pack-zero 0|1`，完整默认值和覆盖规则见[INI参考](../../docs/ECM_TOML_REFERENCE.md#stage2_pack_zero)。

`tools/test/test_stage2_pack_zero_config.py --exe <新生产exe> --save <M503的318-bit余因子save> --device 1 --output data/experiments/<新目录>`使用B1=20、sigma26有效save，检查默认值、共享/worker INI、环境、CLI、旧cprof隐式关闭和非法输入。有效项执行完整曲线，核对设置实际传入曲线子进程并通过GMP与完成回执；非法项须在曲线前拒绝。此项验证不加入性能矩阵。
