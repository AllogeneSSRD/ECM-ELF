# Stage1 易用性与统一批次检查点草稿

当前开发目标：Edwards 配置 Prime95 转移路径时自动忽略；CPU/GPU 均提供可捕获的曲线内进度；Stage1 使用共用结果模型生成内部 JSONL 与提交 JSON；Edwards、Montgomery 使用单个批次检查点。

用户确认检查点名称严格为 `.e_s1_m127_1000.ckpt`、`.m3_s1_bit512_1000.ckpt` 这种形式，不追加曲线序号。k·b^n+c 任务使用原始 n，其余整数使用实际位宽；完整模数和批次身份在文件内部校验。同名但身份不同的检查点不得覆盖。

CPU 各条曲线保留独立进度，存入一个带校验和的容器。周期写入和停止写入使用原子替换及备份轮换。GPU 使用本身的批次状态格式，共用命名与发布生命周期。

结果发布在队列推进前完成。仅完成 Stage1 的结果不得冒充已完成所请求 B2 的 Stage2 结果。提交文件仍为项目既定的 `results.json.txt`，内部文件独立保存。测试仅在本地，不发送服务器。

用户确认 Stage1 NF 仅写内部 JSONL 与 save，提交文件仅新增 Stage1 因子。全新路径整批 NF 时创建空提交文件和归属账本；Stage1 不封存同一账本中 Stage2 尚未发布的 NF。

## 当前实现

版本为 0.34.0。Edwards 与 Montgomery 的 CPU/CUDA 路径均使用 `ECM_STAGE1_BATCH_V1` 容器；其中保留各后端原生载荷。CPU Edwards 另保留完成结果，输出失败后可以直接重新发布；SIMD 回调的模数上下文由每个批次独立持有。可选 resident/PRAC 也使用相同容器和停止协议。

内部序列化与 CRC32 集中在 `ecm_internal_result.h`，提交数学字段使用 `ecm_result_submission.h`。Stage1 为每条曲线持久化去重回执，批次结束后统一导出，避免逐曲线反复生成全部结果；Stage2 默认仍按原有节奏发布。GPU sigma 在整批准备完毕后持久化，曲线计算开始前已经固定。

Edwards、Montgomery 的最终文本 save 共用原子发布与数学记录去重。Prime95 转移路径对 Edwards/GPU PARAM2 不生效；过时 CLI `--p95-dir` 也明确提示忽略。GUI 缓存默认独立为 `gui_results.jsonl`，拒绝占用 CLI 归属文件。共享配置身份只有一份定义，由 `shared_with` 生成 Stage1/Stage2 绑定。

Stage1 构建身份从链接后生成的 sidecar 读取，不放入 CUDA 共用编译参数。最终主机刷新前后的 CUDA 对象哈希一致；构建时发现的 BOM 辅助脚本共享临时文件竞争已改用每次调用的独立名称。

## 验证与证据

本次为正确性和文件生命周期验证，不形成正式性能比较。GPU 测试固定使用 GPU1 RTX 4060 Laptop；CPU 同时覆盖 scalar 与本机可用的 AVX512-IFMA SIMD。最终软件版本、二进制 SHA256、测试结果位置记录在 `data/experiments/stage1_usability_20261017/validation_final_manifest.json`；较早的验证清单和原始证据保留原样。

- `local_validation_04/`：全位宽 local 二进制通过 27 组验证。覆盖 Edwards CPU/GPU NAF/binary、Montgomery CPU/GPU 0/2/3、8191-bit 任意整数、k·b^n+c、停止恢复、损坏主档回退、批次独占、目标/count/sigma 冲突拒绝、双文件发布失败恢复、save/JSON 去重及 worker 2 私有身份。跨阶段验证确认 Stage1 NF 不写提交回执，Stage2 完成后才发布 NF，因子重读不重复计入，Stage1 不封存未发布的 Stage2 NF。停止测试要求已经产生实际算术进度，故障注入前后检查点另存为 `.bin`。
- `resident_validation_02/`、`prac_validation_02/`：限定 1024-bit、显式启用可选内核的测试构建，分别验证冷启动、停止、备份恢复及最终双文件输出。默认 local 构建不包含这些可选内核。
- `montgomery_endpoints_02/`：恢复后的 Montgomery 0/2/3 及 resident/PRAC 批次中，30 条 NF 末点与本地原生 GMP-ECM 完全一致；另 10 条因子记录验证为实际模数的非平凡因子。参考程序身份见验证清单。
- `batch64_validation_02/`：最终 local 程序的 64 曲线 GPU NF 批次及相同工作重放；内部和 save 始终各 64 行且字节一致，提交为空、账本中没有 NF 回执，CLI Prime95 路径未被访问。独立复核最终完整验收产生的 147 条 Stage1 内部 CRC32。
- `submission_final_04/`：22 项 CPU/CUDA Stage2 双结果与私有 Prime95 正式样本字段结构的离线回归。较早的 `submission_regression_02/` 另保留 59 条 CPU 内部 CRC32 与数学目标关系的独立校验。
- `converter_regression_01/`：23 项转换器回归。GUI 自检 36 项、结果存储 44 项、日志解析 76 项通过，日志保留在本证据目录及对应构建日志。

正式构建入口为 `tools/build/build_stage1_local.ps1`，使用 sm_89、完整位宽、8 路并行 CUDA 编译、无 PTX；输出 `build_cuda_cmake/ecm_cuda.exe`。常规开发测试使用 `tools/build/test/build_edwards_gpu_workers.ps1`；结果公共模块另验证 CPU Stage2 与 CUDA Stage2 host-only 重建。

跨阶段因子验证另发现 Stage1 因子的旧回执键包含未执行 B2，已修正为与 B2 无关；从 Stage1 交给 Stage2 后因子行由重复的 8 条恢复为 4 条，并通过旧账本导出去重回归。证据为 `stage1_factor_dedup_before/`、`stage1_factor_dedup_after/` 与 `factor_submission_regression_03/`（22 项 CPU/CUDA 回归）。`final_factor_pipeline_01/` 使用当前 local 二进制实际执行 Stage1，再由 CPU Stage2 重读，提交文件始终为 4 条因子且字节一致。

提交前隐私检查覆盖 worker 后缀结果文件、统一检查点和私有账本；`privacy_guard_01/` 使用独立 Git index 与既有公开 blob 验证拒绝暂存，不向实际索引或 Git 对象写入私有内容。

## 限制与 TODO

本次运行验收为 Windows x64；POSIX 文件锁和平台身份分支仍需对应系统的编译、设备与停止验证。长一般整数采用完整 N 的本地 JSON 扩展，不据此声称满足 PrimeNet 消息容量。现有独立旧检查点不自动导入统一批次容器；当前格式与限制以正式 Stage1 使用文档为准。

草稿保留，由用户决定后续清理；正式说明位于 `docs/usage/STAGE1.md`、`docs/usage/ECM_RESULTS.md` 及生成的 INI 参考。
