# CPU/GMP 群阶、整数分解与点阶模块

本模块位于 `src/ecm/core/group_order/`，独立使用 GMP 与 C++17。0.51.0 将其接入 Stage1 `--go` 与 Stage2 可选因子再分解，作为 GP 不可用时的有界回退；数据集脚本仍保持原有 GP 行为。开发研究与冻结实施记录保留在[临时草稿](../development/LOCAL_GROUP_ORDER_DRAFT.md)。

数据集正确性和性能结果另有[HTML 阅读版](../web/cpu-group-order-validation-20261023.html)，数字以草稿中的验收记录和冻结证据为依据。

## 能力与结果语义

公共入口为 `group_order.h` 中的 `ecm_group_order` 命名空间：

- `factor_integer`：对正整数执行试除、完全幂拆分与 Pollard–Brent rho 递归分解；1 返回空分解，0 或负数无效。预算不足时返回已找到的因子和剩余 `cofactor`，二者乘积保持等于原输入。
- `construct_curve`：按素因子 p、曲线族、Montgomery param 和完整 uint64 sigma 重建正确模型与初始点。支持 Montgomery param 0/1/2/3 和 Atkin–Morain Edwards。Edwards 要求 param=-1，Montgomery param 0 要求 sigma≥6，其他构造仍需通过分母与退化检查。
- `analyze` / `analyze_curve`：未提供倍数时独立计算群阶，再分解并求原始点阶；提供 `known_multiple` 时只验证和分解该消去倍数，再求点阶，不声明该倍数等于完整群阶。可提供已有分解以隔离整数分解成本，模块重新核对乘积、重复项、指数及因子素性分类。
- `multiply` / `point_on_curve`：用于独立验证的短 Weierstrass 群运算。低层标量乘要求调用方提供合法素域与点；高层分析入口负责检查。

模型内部统一为短 Weierstrass。Montgomery 的 B 系数保留在映射中，避免切换二次扭曲；Edwards 复用 `ecm_edwards_curve.h` 的构造，再映射原始点。点阶不包含 choose12 或 48LCM 的额外标量；调用方若需有效点阶，必须按实际执行标量另行处理。

`group_order_computed=true` 只表示本地完成群阶计数。输入消去倍数保存在 `multiple`，不会填入 `group_order`；后者未计算时为 0。点阶的剥离检查为 `[m]P=O`，并对最终每个不同素因子 q 检查 `[m/q]P≠O`，也支持无穷远点的阶为 1。

### 素性分类与证明范围

`Primality` 区分 `composite`、`probable` 和 `proven`。不超过 uint64 的整数先经 GMP 筛选，再做前 12 个素数底数的强伪素数检验；该范围内通过者标为 proven。所用范围依据为 [Sorenson–Webster 的十二底数结果](https://arxiv.org/abs/1509.00864)。更大整数通过 GMP 概率检验后只标为 probable，当前模块没有大整数素性证明器。

`status=complete` 表示本次数值运算完成，并不等同于全部素性已证明；必须另外读取 `all_proven`。`point_minimality_checked` 表示已执行逐因子最小性检查，当模数或分解因子仅为 probable 时，数学结论也以对应素性成立为条件。已有数据集的 proven 标记和 GP 证据属于验证端的外部依据，不会被模块伪装为本地产生的证明。

群阶计数内部必须用已证明的消去倍数分解来取得点阶除数；发现该分解含仅 probable 的因子时，不据此签发唯一群阶，返回资源限制。模数自身的分类仍单独保留。模块不会把外部声称的 `PrimePower.primality` 直接当作证明。

## 完整群阶算法和预算

小域逐 x 计算 Legendre 符号并加上无穷远点。超过枚举门限后，在精确 Hasse 区间上使用 BSGS 找到点阶倍数，完成分解与精确点阶剥离，累积曲线及二次扭曲两侧的点阶 LCM。结合 `#E+#twist=2p+2` 的整除约束，通过广义 CRT 在 Hasse 区间内唯一确定群阶。它能够从低阶初始点继续抽取点，不以一次随机点消去作为群阶证明。

每次分析共享一个墙钟预算，rho 迭代数也在调用内累计。默认：2000 ms、4000000 次 rho 迭代、200000 个 baby 表项、24 个抽样点、试除到 1000、枚举 p≤4096、输入不超过 4096 bit。实现允许的 baby 表项上限为 10000000、试除门限上限为 1000000、输入 bit 上限为 65536；这些上限不代表已验证的实用范围。baby 表项预算是条目数上限，不是严格字节内存配额。

预算检查是协作式的，不会中断一次 GMP 操作；小量超时偏差属于当前行为。整数分解返回 `partial` 和余因子；高层计数或点阶阶段达到上限返回 `resource_limit`。结果中已经完成的字段可以保留，必须先检查 status 再使用最终点阶。

38～49 位十进制 Edwards 参考未采用本地完整群阶计数，当前 BSGS 表预算会明确拒绝。没有 SEA、p−1/ECM/QS 整数分解补充后端、大整数素性证明或成功 Stage2 剩余阶恢复入口。

## 主程序的 GP 优先与回退

共享协调层为 `service/diagnostics/order_analysis.h/.cpp` 的 `ecm_diagnostics::Session`。它按每个请求验证数学身份，优先启动指定 GP；缺失、启动失败、超时、异常退出、缺少协议或输出超过 2 MiB 时，在 auto 模式尝试 CPU/GMP。同一任务会记住已确认不可用的 GP，后续命中不重复等待。非法曲线、复合模数、违反 Hasse 界或分解乘积等数学错误单独返回，不伪装为可接受结果；单个错误曲线不会改变后续合法曲线的 GP 优先策略。

GP 使用直接进程调用和独立临时脚本，输出写入有大小限制的临时捕获文件；轮询进程和输出大小，覆盖等待、输出和截止时间，超时回收子进程。Windows 与 POSIX 路径均不经 shell 拼接执行；当前主程序接入验收在 Windows 完成，POSIX 进程路径尚未运行验证。

Stage1 `--go` 默认 `--go-backend auto`；可指定 `gp` 或 `native`。`--go-gp-timeout 1..600` 控制 GP 截止秒数，默认 30。三种模式的诊断未完成都不改变主任务退出码。因子输出与存档、内部结果和提交文件由原任务流程完成后，才进入 CLI 诊断。现有 GPU Edwards 对 `--go` 的限制保留；可进入诊断的路线使用实际曲线族及 `TaskResult` 保存的 uint64 sigma，不能重新猜测另一条曲线。

`go` 与 `go_factor` 仍为群阶及 B1 范围内分解；旧 `go_min_bounds` 保留群阶口径。新增 `go_status` 明确后端、状态、素性证明资格、回退原因；`go_point_order`、`go_point_factor` 和 `go_point_bounds` 仅在点阶分析完成后输出。点阶界限按实际 1LCM、12LCM 或 48LCM 调整，并输出标准单素数 Stage2 的非支配前沿；B2=0 表示仅 Stage1。旧群阶字段和新点阶字段不能互换。

Stage2 的 `stage2_factorize_hits` / `--factorize-hits` 开启后，同样 GP 优先，使用 `stage2_gp` / `--gp`，GP 超时沿用 `stage2_factor_timeout` / `--factor-timeout`。原始已验证因子先写提交文件，再运行附加分析；诊断异常被隔离，原始 `factors`、曲线状态和队列成功回执不变。诊断日志目录不可写不影响计算与发布。

Stage2 结果中的 `factor_analysis` 保留 `raw`，新增 `backend`、`gp_failure`、`remaining_cofactor` 和逐因子的 `primality`。完整且所有因子已证素才允许 `factorization_complete=true`；probable 项仍保留在带 `proven=false` 的 `prime_powers`，不放入已证素的 `prime_factors`。部分分解保持素数幂乘积与余因子的乘积等于原始 raw。诊断信息位于内部 JSONL 的 `stage2-engine` 对象内，不改变提交文件中的原始命中因子。

本地回退每例最多 2 秒，每个 Stage1 调用或 Stage2 曲线 worker 累计最多 10 秒，并保留核心模块的 rho、表项与输入位数上限。这是协作式预算，不能中断单次 GMP 运算；完整群阶超出预算时明确未完成。它不提供大素域完整计数或本地大整数素性证明，也不借用测试数据库中的群阶冒充运行时已知信息。

主程序回归构建入口为 `tools/build/test/build_go_fallback.ps1`（CPU 路线）、`build_stage1_fallback_cuda.ps1`、`build_go_fallback_fixture.ps1` 和 `build_stage2_fallback.ps1`。`tools/test/test_go_fallback.py --target cpu|gpu` 检查真实命中与提交文件，CPU 还比较因子 save；`test_stage2_gp_fallback.py` 检查队列回执及附加诊断。Stage2 测试通过合成 saved-X 因子进入真实队列和发布流程，不代表额外验证 GPU Stage2 算术。两个 host-only 构建仍执行原有 CUDA 对象身份检查。

## 独立构建和验证

Windows 测试构建仅需 MSVC x64 与 GMP，不查找 CUDA：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/test/build_group_order.ps1
build_cpu_group_order/group_order_probe.exe --self-test
python tools/test/test_cpu_group_order.py --output data/experiments/cpu_group_order_20261023/new_run
```

构建脚本支持 `-Build` 和 `-GmpRoot`，生成 exe、GMP DLL 和带源码、编译器、构建参数、二进制摘要的 `manifest.json`。验证脚本的输出目录必须不存在；可传入 `--probe`、`--db`、`--reference`、`--gp`、`--factor-budget-ms`、`--count-cases` 和 `--benchmark-repeats`。原始证据在 Git 忽略的 `data/experiments/`，不写工作数据库。

也可直接构建独立静态库与自测，不配置仓库主工程：

```powershell
cmake -S src/ecm/core/group_order -B build_cpu_group_order_cmake
cmake --build build_cpu_group_order_cmake --config Release
ctest --test-dir build_cpu_group_order_cmake -C Release --output-on-failure
```

GMP 安装前缀可用 `ECM_GROUP_ORDER_GMP_ROOT` 指定。独立 CMake 输出未生成数据集验证所需的 manifest；需要完整数据集身份记录时使用测试构建脚本。独立入口继续可用；主程序由正式共享源码清单编译核心模块及诊断协调层。

`group_order_probe --stream` 是测试协议：逐行 TSV 输入、逐行 JSON 输出。`FACTOR` 输入整数和预算；`ANALYZE` 输入曲线身份、可选倍数与分解及各预算。字段顺序由 `tools/test/test_cpu_group_order.py` 的 `command` 和探针 `request` 共同维护；它不作为生产 CLI 合同。

## 验证与性能口径

验证覆盖默认库所有 2671 条已有群阶的 Montgomery 记录，以及 17 条 Edwards 冻结参考：每例独立做有预算整数分解，再用已有完整分解检查原始点阶，防止把因子分解超时误判为点阶错误。完整群阶使用按模数 bit 数分层的子集；超出表预算明确计入未完成。另用 GP 构造的 35 组输入检查四种 Montgomery 参数化、Edwards、不同平方根分支、完整 uint64 sigma 和退化情况。

性能数据必须区分以下边界：

- 单例内部时间：同一常驻测试进程内，从模块入口到返回；输入解析、进程启动和测试端日志不计入。全库数据为每个不同输入测一次的耗时分布，不是同一输入的重复测量。
- 配对批次：冻结相同的 24 条输入，CPU/GMP 和 GP 分别每批启动一个进程，串行执行，交替先后，至少 3 轮；包括启动、输入输出以及该工作负载的运算。分别测完整群阶分析、纯整数分解、提供已知分解的点阶。
- 内部 `construction_ms`、`count_ms`、`factor_ms`、`point_ms` 描述高层阶段；`count_ms` 已包含计数内部对抽样点的分解和阶剥离。`total_ms` 包含这些阶段，不能与其相加。嵌套 `multiple_factorization.statistics` 也不能重复计入总时间。

单线程共享桌面，CPU 频率未固定，功耗未测；结果作为该二进制和输入的基线，不声明普适加速。精确结果和当前性能数据见[实施草稿中的验收记录](../development/LOCAL_GROUP_ORDER_DRAFT.md#cpu-gmp-独立模块实施验收)。
