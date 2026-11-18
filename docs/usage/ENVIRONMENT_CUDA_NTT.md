# CUDA Stage2 NTT 与多项式环境变量

在启动 Stage2 进程前设置变量。PowerShell 示例见[环境变量用法](ENVIRONMENT.md)。本页的默认值对应生产 CUDA Stage2，使用 `tools/build` 的 `-Engine development` 构建进行算法对照时，默认值和允许范围可能不同。

表中开关用 `0` 关闭、`1` 开启。标为「生产固定」的变量可省略，显式设置时必须等于表中的值，否则启动被拒绝。诊断、故障注入和替代算法请在独立测试进程中使用。预算中的 MB 和 KB 分别按 MiB 和 KiB 计算。

## NTT 变换与进位

记 $k=\log_2 L$，其中 $L$ 是一次 NTT 的长度。自动 outer 选型还受具体 GPU、tile、warp 尾段与紧凑工作区的约束。[默认值与生产校验](../../src/ecm/stage2/cuda/options.cpp) · [生产编译限制](../../src/ecm/stage2/cuda/context.h) · [变换计划与工作区](../../src/ecm/cuda/ntt/runtime.cpp) · [卷积系数检查](../../src/ecm/cuda/ntt/shape.cpp)

| 变量 | 默认 | 用途 |
| --- | ---: | --- |
| `NTT_FUSE_T` | 12 | tile 层数上限，受长度和共享内存限制。生产固定。 |
| `NTT_FUSE_M` | 4 | 标量 outer 单趟合并层数，开发入口钳制到 1–4。生产固定。 |
| `NTT_FUSE_COOP_OUTER` | 2 | 0 标量，1 强制尝试协作 outer，2 使用设备自动表。生产固定。 |
| `NTT_FUSE_COOP_M` | 自动 | 强制协作模式的 outer 层数，钳制到 5–8。生产入口拒绝设置此变量。 |
| `NTT_FUSE_WARP_TAIL` | 1 | 使用 warp shuffle 执行 tile 尾层。生产固定。 |
| `NTT_FUSE_COMPACT_SCRATCH` | 1 | 使用紧凑 outer twiddle 工作区。生产固定。 |
| `NTT_FUSE_MID_COOP` | 0 | 为符合自动表条件的 RTX 4060 Laptop 在 `k=21–23` 采用 M6。 |
| `NTT_FUSE_JOINT` | 0 | 含中段 M6，并为受支持的 `k=23` 选 tile11/M6、`k=25` 选 tile12/M7。 |
| `NTT_GL_SHORT_REDUCE` | 1 | 选 Goldilocks 短归约。生产固定。 |
| `NTT_GL_PTX_REDUCE` | 1 | 在短归约上选择 PTX 变体。生产编译为 PTX3。开发构建可用其他模式，环境请求与固定编译模式不符时退出。 |
| `NTT_GL_SHIFT_SCALE` | 0 | 对可用位移实现的逆变换缩放乘数试用 shift 实现。生产固定。 |
| `NTT_CARRY_RESIDUAL_MODE` | 0 | 0 共享 64 位原子汇总，1 先在 warp 和线程块内紧凑汇总。 |
| `NTT_CARRY_ROUNDS` | 自动 | 只允许提高按形状算出的进位高度归约轮数。 |
| `NTT_CARRY_CHECK_FUSED` | 0 | 合并进位写回及一致性检查，额外申请临时设备内存。生产固定。 |
| `NTT_CARRY_CHECK_ALLOC_FAIL` | 0 | 模拟上述融合工作区分配失败，验证回退。 |
| `NTT_MAXCOEFF_LIMBS` | 8000000 | GMP 最大卷积系数实测的输入长度上限。0 跳过实测，系数界仍逐调用检查。 |

## 短卷积、Graph 与显存

短卷积命中要求 1024 或 2048 点、warp tile、两份对齐且不重叠的缓冲区，且当前调用无需中间调试输出。Graph 另外要求可复用的 arena 工作区和缓存的变换表。[运行时分派](../../src/ecm/cuda/ntt/runtime.cpp) · [Graph 键和准入](../../src/ecm/cuda/ntt/graph.cpp) · [Graph 策略](../../src/ecm/cuda/ntt/runtime.h) · [arena 预算](../../src/ecm/stage2/cuda/plan.cpp)

| 变量 | 默认 | 用途 |
| --- | ---: | --- |
| `NTT_STAGE2_SHORT_CONV` | 0 | 0 原卷积，1 短档寄存器逆向，2 融合 A/B 前向、乘法和逆向，模式 2 要求无 outer。未满足条件时使用原路径。 |
| `NTT_SHORT_GRAPH` | 0 | 缓存短长度卷积的 CUDA Graph 提交。 |
| `NTT_SHORT_GRAPH_MAX_K` | 19 | Graph 每条变换的最大 `k`。 |
| `NTT_SHORT_GRAPH_MAX_WORDS` | 4194304 | Graph 单次 `L × batch` 的上限。 |
| `NTT_SHORT_GRAPH_MIN_CALLS` | 4 | 同一执行形状达到此调用次数才建立 Graph。最小为 1。 |
| `NTT_SHORT_GRAPH_ENTRIES` | 64 | Graph LRU 最大条目数，钳制到 1–256。 |
| `NTT_ARENA_RESERVE_MB` | 768 | 估算 arena 前为其他用途保留的显存。空闲量不超过保留量时采用空闲量的一半。 |
| `NTT_ARENA_CAP_KB` | 自动 | arena 缓存上限。CLI 或 TOML 的 arena MB 非零时由其换算并覆盖。建议清除变量恢复自动，显式 0 会取消执行时缓存上限。 |
| `NTT_ARENA_WORKSPACE_POOL` | 1 | 共用可增长的工作区。关闭后按形状单独持有。 |
| `NTT_WORKSPACE_REUSE_BQ` | 0 | 在工作区池且允许复用的调用中共用 B/Q 缓冲区，将三块大缓冲区减到两块。 |
| `NTT_PHASE_TRIM_RAW`、`NTT_PHASE_TRIM_OUTPUT` | 0 | 在阶段交接时分别释放 raw 暂存和输出缓冲区。 |
| `NTT_OWNER_TRIM_FUSE` | 0 | 常驻 fold/前沿 owner 准入时尝试清理融合 NTT 缓存。 |
| `NTT_MEMORY_LEDGER` | 0 | 记录代码中受跟踪的设备分配及阶段快照。统计范围取决于分配点的插桩。 |

阶段工作区行为见 [batch.cpp](../../src/ecm/stage2/cuda/batch.cpp) 和 [remainder.cpp](../../src/ecm/stage2/cuda/remainder.cpp)。设备分配统计见 [device_memory_ledger.h](../../src/ecm/cuda/runtime/device_memory_ledger.h)。

## 多项式分块、卷积与归约

同一项有正式 CLI 或 TOML 设置时按对应入口的优先级解析。特别是 `NTT_S4_WEIGHTED` 由 `--weighted-ntt` 写入进程环境，设置变量后直接启动 CLI 会被覆盖。[多项式乘法分块](../../src/ecm/stage2/cuda/product.cpp) · [S4 归约与抽查](../../src/ecm/stage2/cuda/s4.cpp) · [CLI/TOML 解析](../../src/ecm/stage2/cuda/host_policy.cpp) · [主程序入口](../../src/ecm/stage2/cuda/main.cpp)

| 变量 | 默认 | 用途 |
| --- | ---: | --- |
| `NTT_S3_COMPACT_PRODUCTS` | 1 | 使用独立的紧凑乘积缓冲区。 |
| `NTT_S4_BATCH_MB` | 64 | 多项式乘法分块预算，由 Stage2 CLI 或 TOML 解析出的 batch MB 写入。 |
| `NTT_S4_CHUNK_MAX` | 0 | 额外限制单块的乘法组数。0 使用预算选出的值。 |
| `NTT_S4_WORKSPACE_BUDGET` | 0 | 按实际工作区字节数决定乘法分块。 |
| `NTT_S4_PACK_ZERO` | 1 | 生产启动器用 `--pack-zero` 或 TOML `pack_zero` 的有效值配置内部引擎。旧成本表要求关闭。 |
| `NTT_S4_WEIGHTED` | 0 | 启用加权 NTT，要求 CUDA 生产引擎和 pack/补零融合。使用 `--weighted-ntt 1` 或 TOML `weighted_ntt`。 |
| `NTT_S4_HALF_INVERSE` | 0 | 在加权 NTT 且 `k ≥ 24` 时尝试半长逆变换计划。tile 或趟数等条件不满足则回退。 |
| `NTT_S4_PACK_DIRECT`、`NTT_S4_FLAT_DIRECT` | 1 | 启用设备直接打包及 flat 输入直通。生产固定。 |
| `NTT_S4_HOSTPACK` | 0 | 使用旧 CPU 打包路径作对照。生产固定。 |
| `NTT_S4_OUTPUT_WINDOW`、`NTT_S4_CHUNK_OUTPUT` | 1 | 只写当前所需系数窗口，按 chunk 暂存输出。生产固定。 |
| `NTT_S4_FINAL_READBACK` | 0 | 延后到末端集中读回输出。生产固定。 |
| `NTT_S4_KEEPHOUT` | 0 | 在已有归约 hook 时也装配并读回普通卷积输出，供对照。 |
| `NTT_S4_ASYNC`、`NTT_S4_DEFER_CARRY` | 1 | 启用异步 S4 与延后读回进位检查。 |
| `NTT_S4_CARRY_BATCH` | 1 | 合并多组进位检查读回。生产固定。 |
| `NTT_S4_MERSENNE` | 1 | 对符合条件的梅森承载使用折叠归约。生产固定。 |
| `NTT_S4_COOP_REDUCE` | 0 | 0 标量，32 warp 协作。CLI 或 TOML 的归约后端优先，执行还受模数与位宽限制。 |
| `NTT_S4_COOP_TPB` | 128 | 协作归约每块线程数，允许 64、96、128，`--reduce-threads` 优先。 |
| `NTT_S4_COOP_KERNEL` | 1 | 协作归约变体 1–4，3/4 为显式实验，`--reduce-kernel` 优先。0 由 CLI 解释为采用当前默认。 |
| `NTT_S4_ORACLE_ASYNC`、`NTT_S4_ORACLE_PACK` | 1 | 启用异步 GMP 对照与打包抽样。前者生产固定。 |
| `NTT_S4_ORACLE_RING` | 4 | 异步对照环的槽位数，钳制到 1–8。 |
| `NTT_S4_ORACLE_PIN_KB` | 0 | 对照快照的页锁定缓冲区分配量子，允许 0、64、256。0 使用精确容量。 |
| `NTT_S4_ORACLE_PIN_POOL` | 0 | 1 让原始数据和输出快照共用分配 owner，0 分开分配。 |
| `NTT_S4_SAMPLE` | 96 | 每次 GMP 对照的采样上限，设为 0 跳过这项抽样。 |
| `NTT_S4_CHECK_EVERY` | 8 | 周期检查的调用间隔，须为正数。 |
| `NTT_S4_OFF`、`NTT_S4_OLDTAIL`、`NTT_S5_ON` | 0 | 开发对照分别禁用 S4、使用旧 S4 归约尾部、启用旧 S5 descent。生产固定。 |

## Giant 多项式树、fold 与下降阶段

[树实现](../../src/ecm/stage2/cuda/tree.cpp) · [fold 与常驻前沿](../../src/ecm/stage2/cuda/remainder.cpp) · [显存和 resident 规划](../../src/ecm/stage2/cuda/plan.cpp) · [批次执行](../../src/ecm/stage2/cuda/batch.cpp)

| 变量 | 默认 | 用途 |
| --- | ---: | --- |
| `NTT_GFINV_SEG_EXACT`、`NTT_GFINV_BATCH` | 1 | 使用精确分段积和统一批量求逆。生产固定。 |
| `NTT_SMALL_PRIME_REUSE` | 1 | 在满足缓存身份时复用同一曲线的小素数处理结果。生产固定。 |
| `NTT_S4_GROOT_ONLY` | 1 | G-tree 只保留根结果。生产固定。 |
| `NTT_GROOT_DEVICE`、`NTT_GROOT_COMPACT_RAW`、`NTT_GROOT_LEAF_STAGING` | 1 | 使用设备 G-root、紧凑 raw 缓冲和分段 leaf 暂存。生产固定。 |
| `NTT_GROOT_LEAF_CHUNK` | 0 | 限制一次暂存的 G-tree 叶数，0 使用工作区计算出的大小。 |
| `NTT_DEVICE_GLEAF`、`NTT_GROOT_TO_FOLD` | 1 | 在满足显存与形状准入时保留 G-tree 叶节点，并把设备 G-root 直接交给 fold。生产固定。 |
| `NTT_DEVICE_GLEAF_MAX_MB` | 512 | 设备 G-tree 叶节点的显存预算。 |
| `NTT_FOLD_FLAT`、`NTT_FOLD_DEVICE` | 1 | 使用 flat fold，并允许设备常驻累加器。生产固定。 |
| `NTT_FOLD_DEVICE_MAX_MB` | 640 | fold 常驻 owner 的显存上限。显式 CLI 或 TOML 预算优先。 |
| `NTT_FOLD_OWNER_REUSE` | 3 | fold owner 缓冲区复用方案。0 不复用，1 复用 q/qb，2 复用 G/reverse，3 两者都复用。生产固定 3。 |
| `NTT_SCALED_DESCENT`、`NTT_GSCALE_DEVICE`、`NTT_SCALED_ROOT_DEVICE`、`NTT_SCALED_FRONTIER_DEVICE` | 1 | scaled descent、设备 Gamma 校正、scaled root 准备、前沿常驻。生产固定，执行仍受准入和回退条件约束。 |
| `NTT_SCALED_FRONTIER_MAX_MB` | 继承 fold 预算 | 限制设备下降前沿工作区，实际预算取该值与 fold 上限的较小值。 |
| `NTT_STAGE2_RESIDENT_PRODUCTS` | 1 | 前沿设备常驻时保留 product 叶，供后续 GCD 路径使用。 |

G 树布局通过 `--groot-layout` 或 TOML `groot_balanced` 配置。固定参数 A/B 用法见 [Stage2 使用说明](STAGE2.md#固定参数g树实验)。

## 正确性检查、故障注入与日志

下表的布尔诊断默认关闭。带 `_TEST_BAD` 和 `_ALLOC_FAIL` 的变量用于验证报错或回退。故障注入应和表内对应测试一起使用。调试输出按当前日志等级过滤，设备同步式 trace 会改变计时。[fixture 分派](../../src/ecm/stage2/cuda/engine.cpp) · [多项式对照](../../src/ecm/stage2/cuda/checks.cpp) · [NTT 对照](../../src/ecm/cuda/ntt/checks.cpp)

| 变量 | 用途 |
| --- | --- |
| `NTT_HOST_TRACE` | 1 启用 NVTX3 主机范围，缺少 NVTX3 头文件的构建会拒绝。 |
| `NTT_HOST_BREAK`、`NTT_FUSE_TRACE` | 前者分解 NTT 主机时间，后者在变换各步同步并输出 trace，也会使 Graph 缓存跳过调用。两者检查变量是否存在，设为 0 仍会启用。 |
| `NTT_FUSE_LIFETIME_TEST`、`NTT_FUSE_WARP_TEST` | 检查融合表生命周期和容量，后者扩展其 warp 用例。 |
| `NTT_FUSE_COOP_TEST`、`NTT_FUSE_COOP_BAD` | 协作 outer 对照及 fixture 故障注入。 |
| `NTT_ARENA_WORKSPACE_TEST` | 检查 arena 工作区跨批复用及树容量模型。 |
| `NTT_REQUEST_AUDIT` | 输出算子请求、容量及租约审计。 |
| `NTT_S3_PRODUCTS_TEST`、`NTT_S3_PRODUCTS_TEST_BAD` | 检查紧凑乘积及故障注入。 |
| `NTT_S4_FLAT_TEST`、`NTT_S4_FINAL_READBACK_TEST`、`NTT_S4_OUTPUT_WINDOW_TEST` | 检查 flat 输入、集中末端读回和输出窗口。 |
| `NTT_S4_CARRY_TRACE`、`NTT_S4_CARRY_TEST_BAD`、`NTT_S4_ORACLE_TEST_BAD` | 输出进位结果摘要，分别人为破坏进位结果和 GMP 对照值。 |
| `NTT_S4_MERSENNE_TEST`、`NTT_S4_MERSENNE_TEST_BAD` | 梅森折叠归约 fixture 及故障注入。 |
| `NTT_S4_GROOT_TEST`、`NTT_GROOT_DEVICE_TEST`、`NTT_GROOT_DEVICE_CHECK`、`NTT_GROOT_DEVICE_TEST_BAD` | 分别检查根生命周期、运行设备树 fixture、与 GMP 核对设备树，以及破坏设备根。 |
| `NTT_DEVICE_GLEAF_CHECK`、`NTT_DEVICE_GLEAF_TEST_BAD` | 核对设备叶输出及注入异常。 |
| `NTT_GROOT_TO_FOLD_CHECK`、`NTT_GROOT_TO_FOLD_TEST_BAD` | 核对设备根直接交接的结果及注入异常。 |
| `NTT_FOLD_FLAT_TEST`、`NTT_FOLD_FLAT_TEST_BAD` | flat fold fixture 及故障注入。 |
| `NTT_FOLD_DEVICE_TEST`、`NTT_FOLD_DEVICE_CHECK`、`NTT_FOLD_DEVICE_TEST_BAD`、`NTT_FOLD_DEVICE_ALLOC_FAIL` | 设备 fold fixture、GMP 核对、坏值注入和模拟分配失败。 |
| `NTT_GFINV_SEG_TEST`、`NTT_GFINV_SEG_CHECK`、`NTT_GFINV_SEG_TEST_BAD` | 分段积 fixture、执行中核对和故障注入。 |
| `NTT_GFINV_BATCH_TEST`、`NTT_GFINV_BATCH_TEST_BAD` | 批求逆 fixture 及故障注入。 |
| `NTT_SMALL_PRIME_CHECK`、`NTT_SMALL_PRIME_TEST_BAD`、`NTT_SMALL_PRIME_CACHE_STALE` | 核对缓存值、注入坏值、模拟缓存身份不匹配。 |
| `NTT_SCALED_TEST`、`NTT_SCALED_CHECK`、`NTT_S4_DESCENT_CHECK` | scaled descent fixture、scaled 检查、对全部叶分别计算两种 descent 并比对。 |
| `NTT_GSCALE_DEVICE_CHECK`、`NTT_SCALED_ROOT_CHECK`、`NTT_SCALED_FRONTIER_CHECK` | 分别核对设备 Gamma 校正、根和前沿。 |
| `NTT_GSCALE_DEVICE_TEST_BAD`、`NTT_SCALED_ROOT_TEST_BAD`、`NTT_SCALED_FRONTIER_TEST_BAD` | 对上述三处注入坏值。生产校验要求同时显式开启各自的 `_CHECK`。 |
| `NTT_SCALED_FRONTIER_ALLOC_FAIL` | 模拟设备下降前沿分配失败。 |
| `NTT_S5_SELFTEST_N` | 覆盖旧归约自测的 stride，至少为单系数槽宽。默认由用例数和槽宽算出。 |
| `NTT_STAGE1_Q_DUMP`、`NTT_TARGET_LEAF_HASH` | 输出完整 Stage1 末点的仿射 x 坐标、目标叶值摘要。 |
| `NTT_REAL_F_DUMP` | 设为文件路径，将实际 F 多项式、点和输入参数写入文件。使用可写的实验路径。 |
| `NTT_NAME_HITS`、`NTT_NAME_MAX`、`NTT_NAME_BUDGET_BLOCKS` | 控制命中候选的素数回查。默认开启，每批最多回查 1 个命中叶，`NTT_NAME_MAX=0` 取消叶数上限。块预算默认根据约 419 万次素性测试推算，也可显式覆盖。GCD 所得因子独立保留。 |

## 开发构建的附加变量

旧开发引擎的入口见 [worker.cu](../../tools/experiments/cuda/stage2_tree/worker.cu)，对应算法、fixture 与输出见 [engine.cuh](../../tools/experiments/cuda/stage2_tree/engine.cuh)。

| 变量 | 默认 | 用途 |
| --- | ---: | --- |
| `NTT_ARENA_FRACTION` | 未设 | 旧独立实验入口按当前空闲显存的给定比例覆盖 arena。 |
| `NTT_ARENA_S2` | 0 | 旧 S2 实验增加持久乘法 arena。 |
| `NTT_S4_DESCENT_TRACE` | 0 | 输出旧下降路径逐层的状态。 |
| `NTT_GSCALE_DEVICE_TEST`、`NTT_SCALED_FRONTIER_TEST` | 0 | 设备 Gamma 和前沿专项 fixture。生产固定为 0。 |
| `NTT_SCALED_ROOT_TEST` | 0 | 生产配置保留此校验项并固定为 0。当前开发引擎的根专项检测使用 `NTT_SCALED_ROOT_CHECK`。 |
| `NTT_S5_CHUNK`、`NTT_S5_NO_LINEAR` | 4096、0 | 限制旧 S5 前沿 chunk，最少 2 并按系数预算再缩小。开关让一次线性除法也进入通用 Newton 除法。 |
| `NTT_S5_REDDUMP`、`NTT_S5_DIGDUMP`、`NTT_S5_DIVDUMP`、`NTT_S5_FRONTIER_DUMP` | 0 | 分别输出旧 S5 归约、digit、除法和前沿中间量。`NTT_S5_REDDUMP` 在生产固定为 0。 |
| `NTT_S5_CANON_CHECK`、`NTT_S5_LEVEL_CHECK` | 0 | 检查旧 S5 的 digit 规范性和逐层结果。 |
| `NTT_STAGE1_EXTRA` | 1 | 旧开发引擎自行执行内置 Stage1 时的附加标量，只接受 1 或 12。读取 Stage1 save 时无效。 |

`NTT_S4_SHORT_BATCH_TUNE` 和 `NTT_S4_SHORT_BATCH_HASH` 由规划器从 `--short-ntt-batch-tune` 和已选样本生成，用于记录调参身份。单独设置变量不会启用短档批量调参。`NTT_MEMORY_AUDIT` 是运行环境匹配时忽略的历史名称，当前显存记录使用 `NTT_MEMORY_LEDGER`。[规划与环境匹配](../../src/ecm/stage2/planning/planning.cpp)
