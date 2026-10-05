# Auto B2 / tune 第一轮实现：共享规划、payload 计账与域卷积 tune

2026-10-05，接续 [设计报告](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_TUNE_DESIGN.md)。本轮实现 P0 的基础部分及 P1 最小可用接口；Auto B2、跨位宽完整成本标定和全进程活跃显存模型仍需后续推进。

## 1. 本轮提供的功能

- `--plan-only`：读取并校验已有 save/队列输入，调用当前引擎的同一 D 搜索；返回机器可读的几何、模块容量和明确标注的时间估计。初始化 CUDA 上下文、查询显存，不执行曲线或推进队列。
- `--tune ntt`：测量固定 Goldilocks 后端在 k=16..27 的域卷积吞吐量。每个 iteration 为两次 forward 和一次带 product/scale 的 inverse。当前 batch=1、当前选定配置，不是所有配置的自动搜索。
- arena accounting v2：FuseCtx、数组和 carry scratch 的预算统一为实际请求的 CUDA payload；分配准入、淘汰减账和统计使用一致公式。
- 保存 profile 的二进制/构建清单 SHA256、GPU UUID、SM、CUDA runtime/driver、实际调度与所有原始 event 样本。profile 是 JSONL，便于增量写入和后续读取。

**本轮仅编译，不运行 CLI 测试、GPU tune、Stage2 曲线或门禁；没有新增吞吐量或加速数据。** 编译成功也不能替代算术、内存边界和性能验证。独立实验构建不覆盖原生产目录。

## 2. 构建与命令

从仓库根目录编译固定 PTX 后端：

```powershell
tools/build/build_ecm_cuda_stage2.ps1 `
  -Build build_cuda_cmake/_auto_b2_tune_20261005/native `
  -Arch sm_89 -GlBackend ptx -Rebuild
```

实验输出路径为 [ecm_cuda_stage2.exe](D:/code/MPA-OpenCl/build_cuda_cmake/_auto_b2_tune_20261005/native/ecm_cuda_stage2.exe)，旁边为 `build_manifest.json` 与 GMP DLL。builder 新增三个 header 的依赖，编译后再次核验原始依赖 SHA；有变化则拒绝记录为有效构建。

本轮已成功编译固定PTX3 / sm89 / outer0：CUDA310.5s，main4.2s，其余三组件2.5/3.1/2.9s，链接成功。二进制SHA256为 `386d2e1bf578f24e32f03d89366750f549c1cadf769a466560a1da36cea87f44`。包含builder在内的23份原始依赖已核验并冻结于 [sources目录](D:/code/MPA-OpenCl/build_cuda_cmake/_auto_b2_tune_20261005/native/sources)，清单见 [frozen_sources_manifest.json](D:/code/MPA-OpenCl/build_cuda_cmake/_auto_b2_tune_20261005/native/frozen_sources_manifest.json)。NTT源LF、Stage2树源CRLF及最后单独CR保持。

以下是使用命令，**本轮未执行**：

```powershell
$exe = 'D:/code/MPA-OpenCl/build_cuda_cmake/_auto_b2_tune_20261005/native/ecm_cuda_stage2.exe'

# 固定 D 的保存点规划；不执行曲线。
& $exe --save build_cuda_cmake/_budget_scaling_20261005/study_v3/m4423.save `
  --b2 800000000000 --d 1021020 --device 1 --arena-mb 6300 --plan-only

# 默认 repeats=5；一次预热不计入 median。
# 3072MiB 是本次 tune 持有的数组/表/scratch payload 预算。
& $exe --tune ntt --device 1 --length-log2 16:27 `
  --tune-repeats 5 --tune-memory-mb 3072 --tune-file stage2_tune.jsonl
```

`--length-log2` 接受单个 k 或 `FIRST:LAST`，当前范围16..27；repeats为1..1000；默认 tune 预算1024MiB，因此某些大长度会被跳过。runtime 后端构建与同步 trace 模式不接受 tune；使用 `-GlBackend ptx` 等固定后端并关闭 `NTT_FUSE_TRACE`。

`--tune` 与 save/worktodo/B2/D/曲线选择/plan-only 等模式冲突时直接报错。它不需要存在 worktodo。设备来自显式 `--device` 或现有配置；本文命令显式使用GPU1。

## 3. 共享规划接口及其边界

共享整数几何位于 [ecm_stage2_geometry.h](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:22)，由真实 `ntt_shape_query` 回调提供 length 与 output_slots，未另写浮点版 packing 公式。

```text
P = phi(D)/2
W = ceil(S/64)
I = floor(B2/D)+2
G = ceil(I/P)
owner_bytes = 8W(9P+8)+48
fold_big_bytes = 24 L_fold
arena_estimate_bytes = 8[(3L_fold+out_fold)+2(3L_tree+out_tree)]
```

共享函数检查加法/乘法溢出。owner 的实际分配准入也调用同一公式。`L_fold` 查询 P+1；`tree_length` 查询 P/2+1，是原保守规划公式的名义 tree shape，**不是所有 partial/unbalanced tree 节点的实际 length 清单**。后续完整 shape 直方图应逐操作计数。

[引擎规划接入](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10681) 保留现有 `real_run_words` 的规划数值，同时输出 `ecm_stage2::Plan`；[公开 API](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:54) 调用同一 `run_real(...,d_plan_only=true)`，在 baby 列表生成及曲线运算前退出。

结果类型为 `stage2_plan`，包含 B1/B2/D/P/I/G、S/W、fold/tree length、owner/baby payload、arena估计、free显存与预算，以及 model/calibrated/time estimate。

- `owner_budget_fits` 仅比较 owner 单项预算，不包括实际 headroom/backend/分配结果。
- `arena_estimate_fits` 仅说明原规划估计在当前 cap 内。
- `residency_guaranteed=false`，`process_peak_estimated=false`。各模块字段不能相加当作进程峰值。
- 显式 D 超过规划预算时可返回带 false 的诊断计划；该输出不等于许可实际分配成功。
- `--plan-only` 和 `--dry-run` 不同：dry-run只读输入；plan-only会初始化CUDA并查询当前资源。

当前 B2 仍必须来自非零 CLI/worktodo/INI，并大于对应 save 的 B1。没有实现 `--auto-b2` 或 B2=0 自动搜索。

## 4. arena payload v2 的具体修复

原 FuseCtx 准入使用 `8[L+L/2+固定pass预留]`，统计/淘汰却按实际 tables/base 计算。现在先 [fuse_describe](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1500) 生成原有调度，**不分配内存或生成表**；fuse_init复用该描述再分配。NTT kernel、pass序列和表生成算法未改。

按同一描述计算：

```text
M_base = 8[2·2^t + scratch_words + radix_scratch_words]
M_table = 8·Σ_forward/inverse_passes[coarse_words(pass)+2^M(pass)]
M_new_FuseCtx = M_base + M_table
```

[fuse_planned_table_words](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1534) 同时供准入与已缓存表统计使用。淘汰只减真正释放的 table words，base 仍持有且仍计账。

大数组三块各 `8L·batch`；小数组是 out_slots 和两个 verdict words 每slice。移除旧 big/small 每entry额外16B的估算预留，因为其并不是这些 CUDA 分配的 payload。carry诊断 scratch保留原真实字节计账。

数组容量乘法增加溢出拒绝，cap 比较使用减法形式；grow先释放旧容量再计新容量，未引入两个大容量同时存活。已有内置workspace检查中的旧预留表达式同步更新，未增加或运行测试。

日志新增 [ntt_arena_accounting](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1964)：`version=2 payload_bytes=... cap_bytes=...`。其 bytes 应表示 arena 持有的 CUDA payload；仍不包括 per-call回退分配、owner、G树/坐标、CUDA上下文等。cache cap仍不等于全进程硬显存上限。

payload是cudaMalloc请求的容量，不包括分配器对齐/驱动额外占用；实际free显存仍以CUDA查询为准。

这是缓存准入策略的改变，可能影响缓存命中/驻留/性能。因此 [旧 rates 兼容保护](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10782) 暂时拒绝已冻结的profile5/6等窄范围标定，日志给出 `cache_payload_v2_unmeasured`，D排名回到明确标记的 `legacy_56_1`。旧秒数也不能用于此次Auto B2成本保证；新标定完成后再接入新的profile版本。

## 5. tune 的测量合同

[测量实现](D:/code/MPA-OpenCl/src/cuda/ecm_stage2_tune.cuh:39) 直接调用引擎的 `ntt_forward_fused` / `ntt_inverse_fused`：

1. 初始化设备，记录UUID/SM/runtime/driver/固定后端。
2. 对每个合法k，从同一FuseCtx描述计算两块外部数组、表及scratch payload。
3. 预算同时受用户指定 MiB 与实时free显存减768MiB余量约束；超限记录 `skipped_memory`，不运行该长度。
4. 每个长度建立独立arena，记录数组/表建立时间 `setup_seconds`。CUDA上下文初始化发生在这项计时之外。
5. 每次初始化两组稀疏输入，然后在events内执行一次域卷积；一次预热后收集指定次数。
6. events外核验全部L个输出：前5个系数由独立GMP域卷积参考给出，其余必须为0；预热和每个计时样本都核验。
7. 输出原始seconds、median/min/max、`conv_iter_per_s=1/median_seconds`、预热/验证费用及实际配置；释放本长度资源后继续。

稀疏输入沿用既有卷积fixture的工作合同，NTT仍处理整个L。它不等于真实ECM packing数据分布的完整成本。当前没有批量b>1、单独forward/inverse吞吐、参数搜索、设备资源利用率采样或完整多项式归约计时。

`setup_seconds`、`verification_seconds` 与 `warmup_seconds` 分开；`conv_iter_per_s` 只来自event窗口，不包含packing、carry、模N归约、host传输或曲线启动。不得把它直接替代完整Stage2 seconds。

所有长度被跳过、验证失败、CUDA失败或输出写入失败均不能发布可用profile。并发任务可能在规划后抢占显存；这时 CUDA 分配失败会使本次tune失败，保留partial，不伪造一个测量结果。

## 6. Profile 的持久化

[CLI writer](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:507) 写入 `FILE.partial.PID`，每行flush：

- `profile`：schema=1、unit、binary SHA256、build manifest SHA256、请求区间/重复次数。
- `device`：设备/后端/预算/accounting版本。
- `sample`：measured 或 skipped_memory，包含长度、计时、payload/实际调度等。
- `complete`：measured/skipped数量和usable标记。

验证失败会追加 failed_verification，并停止。至少有一项测量成功且全部实际执行样本验证通过才返回成功。writer再次核验exe/manifest指纹，关闭文件后用Windows原子替换发布正式profile；失败保留partial，原正式文件不替换。

[SHA256](D:/code/MPA-OpenCl/src/core/ecm_stage2_fingerprint.h:12) 使用Windows CNG动态加载，没有增加链接依赖。记录清单摘要不代表已经逐项验证运行目录的源文件；binary SHA是本次实际运行版本的主要身份。下一阶段读取profile时还需核验范围、后端和device，不能只按文件名加载。

目标文件限制为.jsonl，并避免覆盖当前exe/config/队列/result/log；比较同时考虑Windows大小写及已有文件别名。profile的JSONL是首版实现选择，替代设计中的单个嵌套JSON对象，便于中断恢复和逐行审计。

## 7. 尚需完成的工作

1. P0完整阶段活跃显存清单、总显存/独立big硬限与运行时租约；当前仅统一payload基础和几何查询。
2. P1按实际调用直方图补batch、较小length、配置候选与可恢复增量tune；当前只测选定配置。
3. P2真实多项式操作及2203/4423/8191bits驻留/回退完整phase标定、Stage1每曲线摊销成本。
4. P3根据用户确认的K/(T1+T2)目标联合搜索B2/D/path，接入profile读取、CLI/INI/worktodo及按scope反馈。

在这些工作和必要验证完成前，原生产893版本仍是现有生产基线；本轮代码/二进制属于独立实验实现，不宣称性能提升或已通过新算术门禁。
