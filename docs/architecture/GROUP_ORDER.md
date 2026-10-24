# 群阶、点阶与整数分解诊断

共享诊断层位于 `src/ecm/service/diagnostics/`，供 Stage1、Stage2 与 GUI v2 使用。默认优先本地 PARI DLL，备选外部 GP；纯 CPU/GMP 算法保留为显式选项。`src/ecm/core/group_order/` 是独立 GMP 数学核心，负责曲线构造、验证和可选的有界分析。数据集脚本仍保持独立 GP 调用。共享接入的开发与验收记录见[临时草稿](../development/SHARED_PARI_DIAGNOSTICS_DRAFT.md)。

数据集正确性和性能结果另有[HTML 阅读版](../web/cpu-group-order-validation-20261023.html)，数字以草稿中的验收记录和冻结证据为依据。

## 能力与结果语义

本节与下文“完整群阶算法和预算”描述独立的纯 GMP 核心；默认共享 PARI 路线见“共享后端选择与进程隔离”。公共入口为 `group_order.h` 中的 `ecm_group_order` 命名空间：

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

38～49 位十进制 Edwards 参考未采用本地完整群阶计数，当前 BSGS 表预算会明确拒绝。纯GMP核心没有SEA、p−1/ECM/QS整数分解补充后端、大整数素性证明或成功Stage2剩余阶恢复入口。共享层的本地PARI库是独立扩展路线，见下节；不改变本核心的算法或证明资格。

## 共享后端选择与进程隔离

`ecm_diagnostics::Session` 的默认 `Auto` 顺序为 **PARI DLL → 外部 GP**。显式 `pari`（兼容别名 `native`）只运行 DLL，`gp` 只运行外部程序，`gmp` 才运行原有有界 GMP 算法。默认路线不自动选择 GMP；缺失依赖、超时、异常退出、协议错误或超过 2 MiB 输出上限均保留原因。PARI 已得到的群阶、因子和余因子在 GP 不可用时继续保留；不能将未完成结果声明为完整分析。曲线构造错误、合数模数等数学输入错误返回无效，不改变后续合法请求的后端策略。会话记住不可用的 GP，避免后续命中重复等待。

PARI 适配器位于 `service/diagnostics/pari_native.h/.cpp`，仅在同目录的 `ecm_diagnostics_host.exe` 加载 `libpari.dll` 与 `pari-data/seadata`。主 GUI、Stage1 和 Stage2 均不加载 PARI DLL；它们通过有界临时请求与输出文件调用独立进程。一个请求对应一个 PARI 进程，不复用全局栈，不要求 GP 解释器或脚本。Windows Job 负责取消、超时及异常回收，并对每个 PARI 进程设置 512 MiB 内存上限；PARI 栈初值 128 MiB。该 ABI 当前仅验证 Win64 PARI 2.17.3、64 位字长、单线程版本；其他平台默认可尝试 GP，不冒充 DLL 已获资格。

曲线及原始点由既有 GMP 构造映射到短 Weierstrass，然后调用 PARI 的 `ellinit`、`ellcard`、`factorint`、`isprime` 和 `ellorder`。因子分解后逐项实际证素；接收方重新检查 Hasse 界、分解乘积、点阶整除、消去与最小性。当前 DLL 群阶分析的素域模数范围为 ≤192 bit，整数分解输入范围为 ≤4096 bit。整数分解先做 50 ms / 8192 次 rho 的便宜预分解；这及 GMP 曲线构造、验证是基础运算，不等于自动切换到 GMP 群阶后端。

PARI 单次进程最多 5 秒，会话的累计本地预算默认 10 秒；GUI 整次本地会话设为 5 秒。输出采用逐行检查点，超时后只接收最后一条完整、校验通过的记录。GP 默认最多 30 秒、128 MiB 栈；它与 DLL 均不经 shell 执行。自动转 GP 的请求可能超过 5 秒；这不是任意 192 bit 输入必然完成的承诺。显式 GMP 沿用每次 2 秒、会话累计 10 秒的协作式预算，不能中断单次 GMP 运算。

### Stage1 与 Stage2

Stage1 `--go` 默认 `--go-backend auto`，可指定 `pari` / `native`、`gp` 或 `gmp`。`--go-gp-timeout 1..600` 仅控制备选 GP 的超时秒数。诊断失败不改变主任务退出码：因子输出、存档、内部结果和提交文件完成后才进入诊断。现有 GPU Edwards 对 `--go` 的限制保留；使用实际曲线族与保存的 uint64 sigma。

`go`、`go_factor` 与 `go_min_bounds` 仍采用群阶口径；`go_point_order`、`go_point_factor` 和 `go_point_bounds` 只在点阶分析完成后输出。点阶界限按实际 1LCM、12LCM 或 48LCM 调整。`go_status` 保留实际后端、状态、证明资格、`pari_failure` 和 `gp_failure`；群阶与点阶字段不能互换。

Stage2 的 `stage2_factorize_hits` / `--factorize-hits` 开启后使用相同 Auto 策略执行整数分解，不自动执行整套群阶分析。`stage2_gp` / `--gp` 指定备选外部程序，`stage2_factor_timeout` / `--factor-timeout` 只控制 GP 的超时。原始已验证因子先发布，诊断异常不改变原始 `factors`、曲线成功状态和队列回执。内部 `stage2-engine.factor_analysis` 保留 raw、后端、PARI/GP 失败原因、素数幂、证明资格与余因子；仅完整且全部已证素时 `factorization_complete=true`。分量乘积与余因子之积必须等于 raw，诊断日志不可写也不影响发布。

构建、依赖分发和源码许可见[构建说明](../usage/BUILD.md#共享-pari-诊断依赖)。后端协议验收入口为 `tools/test/test_shared_pari.py`；生产调用验证入口为 Stage1 的实际命中及 `test_stage2_factorize_frontend.py`。后者以合成 saved-X 因子进入真实前端和发布流程，不代表额外验证 GPU Stage2 算术。

## GUI v2 分析入口

GUI 的 Auto 和本地模式由 `ecm_group_order_ui.exe` 调用同一个 `Session`，与 worker 共用 DLL 引擎、后端策略和验证；主界面保持 GMP/PARI 无依赖。显式 PARI/GP 仍可直接启动外部 GP，因此 GP-only 部署不要求本地助手。Auto 在 GUI 助手缺失但 GP 可用时也允许直接尝试 GP；不会因此切换 GMP。显式 GMP 由 GUI 助手运行原数学核心。

合数输入先做有界分解，再按每个不同素因子分析同一曲线身份；幂次只表示输入中重复次数，不声明得到模合数环或扩域的群阶。便宜分解得到的小素分量优先分析，避免难余因子阻塞已有结果；保留各分量完成状态、证明资格及余因子。底层 `Session::group` 仍要求素数模数 >3，特征 2/3 或退化构造只影响对应分量。

GUI 对本地分析设置累计 5 秒预算，对整个 Auto 异步请求设置 45 秒截止，取消/退出回收整棵子进程。界面与交付文件见[GUI使用说明](../usage/GUI_V2.md#群阶分解--group-order-factor)。当前共享层的 ≤192 bit 验证、实际 worker 身份及限制在[共享接入草稿](../development/SHARED_PARI_DIAGNOSTICS_DRAFT.md)记录；下列旧样本只对应明确冻结的二进制，不能直接充当重构后性能数据。

### GUI本地192 bit扩展的验证范围

冻结证据为`data/experiments/gui_v2/native_order_192_0517_01/qualification-02/`。16组输入包含用户的82 bit因子`3436193553215096358758503`、sigma=1955774944、param0，以及127/160/192 bit各一个素数、sigma=26的五种构造（Montgomery param0/1/2/3、Edwards）。192 bit素数为`6277101735386680763835789423207666416083908700390324961279`。所有组都完成群阶、整数分解、原始点阶、逐因子最小性与素性证明，并与单独运行的GP值逐字段一致；原始输入和响应保留在该目录，不使用已有群阶数据库。

设备AMD Ryzen AI 9 HX 370，Windows x64，PARI 2.17.3 Win64单线程，MSVC测试构建，GMP 6.3.0。测试将本地PATH限制为System32且故意指定不存在的GP路径。计时从助手进程创建到退出，包含初始化、请求读取、计数/分解/证素、GMP点阶复核和检查点写入，不含测试端写请求、解码JSON；串行共享桌面，频率未固定，功耗未测。用户输入重复3次的中位数161.650 ms、最大163.232 ms；上述192 bit素数的param0重复3次中位数806.753 ms、最大811.317 ms。16组各一次中最慢为该192 bit素数的param3，1225.443 ms。没有把超时、部分结果或概率素性作为完成样本；这些结果不代表任意192 bit群阶分解或其他机器的5秒保证。

助手SHA256为`e74bd8f55bbc1b4fc68d75443d71b72322394531e0490071716abaaed7c20dc6`，PARI DLL为`50420f175ce1753870d17188e1330f8a58b5d46ac1468c5d32a6edeb2aa3dccd`；构建输入和GUI身份见同目录根部的构建记录及汇总。GP仅作为功能对照，未据此进行GP/本地性能比较。故意挂起的独立夹具另验证约5秒结束和已完成分量保留；512 MiB为作业限额，不是测得的进程峰值内存。

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
