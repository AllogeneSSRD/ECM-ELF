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

### 调优工作量与阶段特征

`analyze_stage2_tune_workload.py`在CPU上读取完整ECM tune的原始计划和正式回执，不启动CUDA。参数为`--evidence <原始调优目录>`、`--profile <对应完整ECM TOML>`、`--output <新目录>`，可选`--ntt-profile <NTT TOML>`。输出目录使用`data/experiments/`。

输出`workload.toml`按阶段、NTT长度和每次调用的slice数保留逻辑多项式乘法数、物理调用数、输出系数数、N与N·log₂N工作量。中间G批次按repeat直接累计，不展开全部批次；分块遵循原始plan的packing形状、batch预算、buffer数和chunk上限，独立核对每个阶段原生groups/pairs/chunks总数。适用范围为当前多G、条件驻留、完整次数fold请求程序；不描述G1或owner回退。

每次正式曲线的init/main及其他阶段计时以配对数组保留，核对init+main=引擎total。新样本声明`phase_accounting="exclusive_engine_v1"`，逐次核对十个互斥阶段之和、init/main及原始回执与发布数组；旧阶段计时保持`legacy_overlapping`，不补造新阶段。父程序发布的worker墙钟及worker−engine残差也按次保留，检查样本、中位数及MAD。详细边界见[完整Stage2 tune](../../docs/architecture/AUTO_B2.md#完整-stage2-tune)。所有统计均不相加独立中位数。文件必须覆盖对应性能profile的全部样本，缺少回执或不一致不能发布成功输出。原始文件、分析工具的路径和SHA只记录在另一个`evidence.json`，性能TOML不记录这些身份信息。

可选NTT文件必须完成且通过算术检查，设备UUID、SM、CUDA和固定后端条件一致。格式1只提供已测batch=1的单slice秒数及“逻辑pairs×单slice秒数”串行参照，因缺少归约mask和环境策略，不能获得策略资格。格式2按`(length,slices)`精确匹配批量中位数，参照为“physical calls×批量中位数”；同时统计覆盖pairs/calls及缺失batch bins，不对未测形状插值。归约mask和完整命名环境一致时可标记`ntt_policy_qualified=true`，但`ranking_qualified=false`、`ntt_feature_is_time_prediction=false`仍保持。两种参照都不含packing/carry/S4归约、点运算、树准备、传输、自检和冷启动，不是完整曲线或阶段预测，不将其相加到含NTT的计时。

`test_stage2_tune_ntt_batches.py --exe <生产exe> --save <已验证save> --device <n> --carrier <p> --output <新目录>`执行GPU验证：小长度、不同slice常数项、非2次幂批量、65535 slices、等级覆盖、显式参数覆盖、内存跳过、JSONL与失败发布保护，再对普通/承载两种路径执行完整Stage2及mandatory GMP检查。输入保存点应先独立验证。输出冻结源码和依赖身份，证据放入忽略目录，不混入性能TOML。它不证明跨位宽、跨B1的收益排名。

工作量分析可重复`--ntt-profile <文件>`汇集策略一致、实测形状不重复的文件；每请求阶段另给覆盖计数，缺项阶段不输出完整参照秒数。`benchmark_stage2_ntt_workload.py --exe <生产exe> --profile <完整ECM.toml> --evidence <原始ECM目录> [--ntt-profile <已有NTT.toml> ...] --device <n> --repeats <n> --memory-mb <MiB> --output <新目录>`按长度补测实际缺失slices，拒绝不匹配的基准、计划或策略。可重复`--additional-plan <原生plan.jsonl>`加入同scope组的留出形状，不加载其曲线成本；内存不足明确失败，不能把skip算作覆盖。

`analyze_stage2_tune_components.py --profile <完整ECM.toml> --workload <workload.toml> [--ntt-profile <文件> ...] --output <新目录>`拟合实验NTT/配对阶段模型；格式4自动读取内嵌NTT。可配合`--holdout-result <已有验收JSON> --query-plans <对应原生计划目录>`作回顾对照。回顾结果不代替新的独立验收，输出模型TOML不进入生产选型。`test_stage2_tune_components.py`核对配对守恒、合成参数恢复、无效输入拒绝和不合格组；数学公式与适用条件见[组合模型](../../docs/architecture/AUTO_B2.md#ntt-与配对阶段组合)。

`validate_stage2_tune_components.py --exe <生产exe> --save <已验证save> --models <models.toml> --profile <完整ECM.toml> --evidence <训练证据目录> --ntt-profile <文件> [重复] --device <n> --holdout-b2 <未测B2...> --output <新目录>`先补齐查询所需NTT形状，冻结全部预测，再测至少四候选的完整曲线。默认每候选1次预热、3次正式重复，NTT形状21次重复；保持8%时间/5%排名门限。失败不改写训练模型或删去候选。

`test_stage2_tune_components_native.py --profile <完整ECM.toml> --plans <case_*.plan.jsonl所在目录> --ntt-profile <文件> [重复] [--query-plan <独立计划.jsonl> ...] --output <新目录>`编译CPU fixture，比较原生请求形状、配对固定项、非负拟合、留一误差和查询预测与独立Python参考；核对原生NTT导入、格式4往返及与生产相同的Estimator。所有需要的NTT形状须精确覆盖且策略一致；同时检查缺项、端点、预算错配、未核验及高误差组拒绝。输出默认由调用者指定到`data/experiments/`，冻结源闭包、输入、fixture和DLL；不查询CUDA设备、不执行曲线。

生产命令`--tune ecm --tune-merge <ECM.toml> --tune-ntt-profile <NTT.toml> [重复] --tune-file <新配置.toml>`将性能数据整理为格式4；测量模式也支持NTT导入。合并不查询CUDA设备，不接受重复NTT实测形状，失败保留旧目标。`test_stage2_tune_merge_runtime.py`可重复`--ntt-profile`检查实际CLI的内嵌导出、再合并及失败保护。格式4主路径的完整曲线验收用`validate_stage2_tune_selection.py`，额外提供`--training-plans <冻结训练目录>`，由独立Python参考重建组合模型；精确形状缺项时核对完整曲线模型回退。

`test_stage2_tune_component_auto_runtime.py --exe <生产exe> --profile <格式4完整ECM.toml> --save <已验证save> --training-plans <冻结训练目录> --device <n> --output <新目录>`核对Auto B2的INI入口、显式普通/承载及D锁定、精确点优先和缺少NTT形状时的分块模型回退，再运行一条未锁定D/承载的完整曲线。默认未测B2=12.5e9可由`--b2`覆盖；输入需有合格普通/承载模型及完整训练形状。T1=3是固定决策测试输入，不是Stage1实测或独立全流程收益验收；不把该单条曲线作为重复性能样本。

`validate_stage2_tune_auto_profit.py --exe <生产exe> --profile <完整ECM.toml> --stage1-profile <实测Stage1.toml> --save <已验证save> --device <n> --stage1-batch <C> --output <新目录>`独立验证有限候选的全流程收益。可选`--stage1-exponent lcm|choose12`、`--ratio <R>`、`--repeats <n>`（默认3，至少2）。格式4另给`--training-plans <冻结训练目录>`；`--holdout-b2 <未测B2...>`把每档应用于所有适用D/算术组，不删除原锚点。全部候选预热后按正/反顺序交错正式重复，自动选中点始终通过未锁定B2/D/承载的入口执行。逐候选时间误差≤8%、有限收益损失≤5%；进程墙钟另列，NVML只读采样，输入/工具/EXE/DLL及源闭包冻结。它不证明连续B2范围的全局最佳值。

`test_stage2_tune_profit_reference.py --profile <格式4完整ECM.toml> --training-plans <冻结训练目录> --holdout-dir <已完成四候选验收目录> --output <新目录>`在CPU上核对收益工具与固定B2工具共用的独立参考。覆盖原生已冻结预测的重算、精确点优先、旧格式回退和缺失/重复/策略错误训练计划拒绝，不运行CUDA或编译；不代替新的完整曲线收益验收。

`test_stage2_tune_workload.py --output <新目录> [--evidence <原始调优目录>]`执行CPU门禁：手算满/尾分块、chunk上限、万亿次repeat压缩、错误协议拒绝及已保存原生plan的独立密集请求重放。它不证明GPU真实调用计数或NTT批处理吞吐；后者需要运行审计和独立完整曲线留出验证。

当前事件矩阵173 cases、47,936 events、168,166 assertions，故意修改释放顺序的模型被拒绝。重复压缩依赖完整保留状态，observer 不生成所有跳过重复事件。

## S4 与 giant

- `test_stage2_s4_memory.py`：raw/output/pack/reducer/selftest/tree metadata CPU事件核对；错误自检窗口模型应拒绝。
- `test_stage2_s4_program.py`：真实 descriptor、请求路由和树租约；可选 `--exe`、`--fixtures`、`--large-save`、`--large-reference` 执行原生门禁。
- giant 模型通过原生 `giant_memory` 输出核对保留 seed、满/尾 chunk、单位/驻留路线和 compact products；相关 runner 以实际参数帮助为准。

S4 三个组件边界和 peak 不包含 NTT、giant、fold/frontier；giant 模型同样只保证本组件。各自峰值不能相加。`NTT_REQUEST_AUDIT`/owned ledger 只用于诊断，不混入正式 timing。

## 验证与证据位置

完整ECM tune另保存`chunk_routes_v1`分块工作量。`stage2_tune_route_cost.py`是独立Python参考，按整数分块和标量位数计算chain/ladder点数、分块数和ladder迭代量；NumPy回归核对原生`giant_route_cost_v2`成本。旧`linear_giant_points_v1`文件可读但只用于精确实测点。新模型不预测未测ladder分支、正ladder迭代量范围之外或纯ladder组；其他资格及8%/5%独立验收门限见[Auto B2](../../docs/architecture/AUTO_B2.md#b2区间成本预测)。

`test_stage2_giant_work.py --fixture <fixture.exe> --prediction-dir <预测测试目录> --output <新目录>`逐点独立核对工作量与原生输出，并比较NumPy和原生成本；这是CPU模型验证，不替代GPU完整曲线验证。

`test_stage2_tune_route_policy_runtime.py --exe <生产exe> --profile <完整tune.toml> --save <save> --device <id> --output <新目录>`用plan-only验证实际分块策略：有效数据接受；合成修改容量、chain阈值或强制ladder后，固定B2拒绝复用该选型，Auto B2拒绝无匹配候选。它不执行曲线，不代替时间预测验收。实际运行的分块策略与实时显存准入分别检查。

`test_stage2_tune_tail_grid.py --fixture <fixture.exe> --valid-profile <CPU有效profile> --output <新目录>`独立枚举整数I，核对自适应样本的范围、G2/G树上限、去重和chain/ladder来源，并检查等级1…10默认目录的容量。它不运行GPU或证明时间拟合精度。

`test_stage2_tune_tail_runtime.py --exe <生产exe> --save <save> --device <id> --output <新目录>`运行完整tune，默认等级3、D60060/120120、重复3次；`--carrier <p>`加入合法承载对照。`--level <3..10>`、`--ds <D,...>`、`--b2 <B2,...>`、`--tail-samples <0..16>`、`--max-batches <n>`覆盖范围。显式B2默认关闭补样，尾点数量另行开启。工具核对CLI拒绝、计划/回执/性能样本对应及分块来源；独立时间/排名验收另用`validate_stage2_tune_selection.py`，其8%/5%门限不变。保存点应事先独立验证；发现因子或算术问题立即保留失败证据。

`analyze_stage2_tune_sampling.py --profile <完整tune.toml> [更多文件] --b2 <查询B2...> --output <新目录>`只做CPU采样诊断：保留全部锚点，按组比较现行四特征模型和探索性chain分块特征的非负拟合、留一误差及逐点残差。查询结果来自独立NumPy参考，报告不是独立GPU时间/排名验收，也不会改写性能配置或生产模型。路径与输入哈希只写入忽略目录中的诊断JSON。探索性特征可能加重过拟合，不能按训练残差更小直接启用。

`test_stage2_tune_sampling.py --fixture <fixture.exe> --profile <完整tune.toml> [更多文件] --b2 <查询B2...> --output <新目录>`将上述查询逐组与原生预测比较，并检查路线错配、无效时间、重复I及错误计时单位拒绝。这是CPU参考一致性检查，不执行曲线。

运行前冻结 exe、DLL、source closure、save、工具与 controls。CPU ledger、native plan、短/大 GPU算术和性能是不同范围，失败必须如实保留。重新构建/修改源码后不能把当前文件哈希冒充旧二进制来源。

已完成事件/路由证据在 `data/stage2_ntt_events_20261009/` 与 `data/stage2_s4_program_20261009/`。它们属于组件规划门禁，没有新增大规模速度结论。完整联合执行器、driver free、回退和 D/Auto B2接入见 [TODO](../../docs/TODO.md)。
