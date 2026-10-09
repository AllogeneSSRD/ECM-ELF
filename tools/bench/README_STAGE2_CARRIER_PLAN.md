# Stage2 承载与内存工具

数学、容量公式与当前组件保证集中在 [内存](../../docs/architecture/MEMORY.md)、[Stage2](../../docs/architecture/STAGE2.md)；性能数据见 [Stage2性能](../../docs/performance/STAGE2.md)。本页说明工具用途和输入。

所有输出使用新 `data/` 子目录；统计写 `data/benchmarks/`，图写 `data/figures/`。失败结果不覆盖，冻结证据不改路径/内容；用户显式参数应遵循同一位置规则。

## 研究与准备

- `analyze_stage2_carrier_plan.py`：计算 N/M packing、D/P、预算与并存下界。`--bits`、`--carrier-bits`、`--b2`、`--d`、预算和 buffers 参数；输出 JSON，不启动曲线，`measured=false`。
- `plot_stage2_carrier_plan.py`：读取分析 JSON，生成 PNG/SVG；可读取位宽扫描统计。
- `prepare_stage2_carrier_inputs.py`：准备明确 target/carrier 的参考输入和 save，不把合成宽位数算术覆盖当作生产性能输入。

## 完整曲线与分析

- `bench_stage2_carrier.py`：固定输入、同二进制因素对照，`--mode timing|check`；`--comparison` 选择支持的承载/布局/预算因素。用 `--help` 查看当前因素名及开关。
- `analyze_stage2_carrier_bench.py`：校验完整计时矩阵、输入和工具身份，汇总均值/标准差/阶段/容量并作图。
- `analyze_stage2_memory_ledger.py`：分析实际分配/释放、live、peak、阶段和最终 live=0；所有权/别名必须符合 ledger。
- `test_stage2_memory_ledger.py`：运行 ledger 核对；额外诊断不计入正式速度样本。

计时需预热、交错顺序和重复；check 含叶投影/请求/台账等额外工作。正式计时保持强制算术检查，不加入额外 fixture。完整输出/GMP/factor 对照与容量/性能结论分别通过。

## 原生计划与请求

生产 `--plan-only` 查询真实 device/packing，不运行曲线、不推进队列。`request_program.version=2` 包含实际 operand 长度、batch、phase、截断、input 来源和 tree_leaves；算术签名与来源/边界签名单独核对。

`request_program` 目前只描述多 G 的条件驻留满次数 fold。G1局部 inverse/root division 等返回不支持原因；不能将组件有效误读成完整准入成功。

- `test_stage2_request_program.py`：计划请求与实际执行签名、树组及输入来源核对。
- `test_stage2_workspace_plan.py`：多 N/P、pool/keyed、复用、carry、紧预算和原生 shape/descriptor 计划核对，0实际曲线。

## NTT 组件事件

- `test_stage2_ntt_memory.py`：CPU allocator/密集请求回归，legacy 总量接口。
- `test_stage2_ntt_events.py --output <new-dir>`：从生产 fuse/arena 提取分配语句，CUDA 分配替换为 CPU opaque ledger，核对逐项 site/顺序/字节/live；需要 MSVC C++17，不调用 GPU。
- `verify_stage2_ntt_events.py`：独立整数事件程序核对原生 plan JSON，供 workspace runner 使用。

NTT JSON version2 提供 exact_allocation_events、fuse_layouts、物理分配/释放与 grouped 计数。计数是成功前缀，不含未来 close 或 refusal 后的 per-call fallback。CPU allocator不证明 CUDA物理分配/free或真实驻留。

当前事件矩阵173 cases、47,936 events、168,166 assertions，故意修改释放顺序的模型被拒绝。重复压缩依赖完整保留状态，observer 不生成所有跳过重复事件。

## S4 与 giant

- `test_stage2_s4_memory.py`：raw/output/pack/reducer/selftest/tree metadata CPU事件核对；错误自检窗口模型应拒绝。
- `test_stage2_s4_program.py`：真实 descriptor、请求路由和树租约；可选 `--exe`、`--fixtures`、`--large-save`、`--large-reference` 执行原生门禁。
- giant 模型通过原生 `giant_memory` 输出核对保留 seed、满/尾 chunk、单位/驻留路线和 compact products；相关 runner 以实际参数帮助为准。

S4 三个组件边界和 peak 不包含 NTT、giant、fold/frontier；giant 模型同样只保证本组件。各自峰值不能相加。`NTT_REQUEST_AUDIT`/owned ledger 只用于诊断，不混入正式 timing。

## 验证与证据位置

运行前冻结 exe、DLL、source closure、save、工具与 controls。CPU ledger、native plan、短/大 GPU算术和性能是不同范围，失败必须如实保留。重新构建/修改源码后不能把当前文件哈希冒充旧二进制来源。

已完成事件/路由证据在 `data/stage2_ntt_events_20261009/` 与 `data/stage2_s4_program_20261009/`。它们属于组件规划门禁，没有新增大规模速度结论。完整联合执行器、driver free、回退和 D/Auto B2接入见 [TODO](../../docs/TODO.md)。
