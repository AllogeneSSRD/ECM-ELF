# Stage2 环境变量

以下配置供点生成、主机等待和队列调度对照使用。设置前先检查 [Stage2 使用说明](STAGE2.md) 和 [TOML 参考](../ECM_TOML_REFERENCE.zh-CN.md) 中是否已有对应选项。环境变量的设置、清除和子进程继承见 [环境变量入口](ENVIRONMENT.md)。卷积、S4 归约、多项式树及其诊断见 [NTT 与多项式环境变量](ENVIRONMENT_CUDA_NTT.md)。

## CUDA Baby 点

| 环境变量 | 未设置时 | 取值和效果 |
|---|---|---|
| `NTT_BABY_COOP` | 后端 `auto` 从标量开始 | `0` 固定每点一线程，`32` 请求 32 线程协作。设备 Baby 路径和实际算术位宽 8193–16384 bits 都符合时才可使用协作。 |
| `NTT_BABY_WHEEL` | 自动 | `0` 用独立 ladder，`1` 请求分段递推。自动请求要求 SM89、至少 31 个 64-bit 算术 limb，且 Baby 点数在 17280–103680。种子数还须至多为点数的四分之一。显式请求跳过这些自动范围，仍检查形状和空间。公共输入非单位或准入失败时全部回退到独立 ladder，单段检查发现非单位时只重算该段。 |
| `NTT_BABY_WHEEL_STEP` | `0` | `0` 自动在 2310、210、30、6、2 中找满足条件的步长。显式步长须是整除 D 的偶数且不大于 30030，否则分段递推回退。 |
| `NTT_BABY_WHEEL_WALK` | `32` | 每段最多递推的点数。有效范围 `2..64`，超出时分段递推回退。 |
| `NTT_BABY_DEVICE_MAX_MB` | `512` | 设备 Baby 临时缓冲上限，单位 MiB。取此上限和留出 64 MiB 后的设备空闲显存的较小值。容量不足则回退；`0` 可用于验证预算回退。 |
| `NTT_LADDER_CAP` | `8192` | 独立 ladder 每次提交的点数上限。非正整数恢复 8192，超过 2²⁴ 时钳制到 2²⁴。 |

`--baby-backend`、显式 `baby_backend` 和环境变量依次优先。生产引擎的 `NTT_BABY_DEVICE` 固定为 `1`，使用的是带预算回退的设备 Baby 路径。设置 `log_level="debug"` 后，`baby_wheel` 日志给出是否启用、回退原因及重放计数。以下示例在同一 save、B2、D 和设备上强制独立 ladder，以便对照自动决策。

```powershell
$previous = $env:NTT_BABY_WHEEL
$env:NTT_BABY_WHEEL = '0'
try {
    .\ecm_cuda_stage2.exe --save .\curve.save --b2 26000000000 --d 210210 --device 1 --log-level debug
} finally {
    $env:NTT_BABY_WHEEL = $previous
}
```

[点后端解析](../../src/ecm/stage2/cuda/host_policy.cpp#L44) · [递推与显存准入](../../src/ecm/stage2/cuda/points.cpp#L688) · [Baby 路径与回退](../../src/ecm/stage2/cuda/points.cpp#L829) · [设备资格](../../src/ecm/stage2/cuda/plan.cpp#L410) · [ladder 上限](../../src/ecm/stage2/cuda/options.cpp#L283)

## CUDA Giant 点

| 环境变量 | 未设置时 | 取值和效果 |
|---|---|---|
| `NTT_GIANT_COOP` | 后端 `auto` 从标量开始 | `0` 标量，`16` 或 `32` 请求相应线程数协作执行 giant 链。仅在实际算术 8193–16384 bits 且走 chain 时生效。 |
| `NTT_GIANT_SEED_COOP` | 后端 `auto` 从标量开始 | `0` 标量，`32` 请求以协作后端生成相邻 seed 和设备端的 `[D]Q`。适用位宽和 chain 要求同上。 |
| `NTT_POINT_SQUARE` | `1` | `0\|1`。点算术关闭或启用专用平方，前者走通用乘法。 |
| `NTT_GIANT_POINT_BUDGET_KB` | `262144` | giant 坐标块的 X/Z 预算，单位 KiB，须为正整数。点块按整 P 个点组成，默认向上取整。该预算只计坐标。 |
| `NTT_GIANT_CHUNK_FLOOR` | `0` | `1` 使点块向下取整，最少仍为一整批 P 个点，故一批本身可能超出预算。 |
| `NTT_GIANT_LADDER` | `0` | `1` 强制 giant 逐点 ladder，用于和链递推对照。 |
| `NTT_GIANT_TAIL_BALANCE` | `0` | 完整区间内调整最后两块的整 P 批次。生产 CLI 会写入 `giant_tail_balance`／`--giant-tail-balance` 的有效值，启动前设置此环境变量不会覆盖它。 |
| `NTT_GIANT_CHAIN_BLOCK` | `64` | 每条链的点数。生产允许 `32\|64`，CLI 会用 `giant_chain`／`--giant-chain` 的结果重写。开发引擎允许 `4..1048576`，超限钳制。 |
| `NTT_GIANT_CHAIN_MIN` | `32768` | chain 的最低点块规模。生产显式值只能为 32768，合格的尾部策略可调整有效门槛。开发引擎可以改门槛，`0` 允许小块直接尝试 chain。 |
| `NTT_GIANT_CHAIN_SMALL_BLOCK`、`NTT_GIANT_CHAIN_SMALL_MAX` | `0`、`8192` | 短块专用链长及最大点数。生产要求前者为 0。开发引擎中，非零链长钳制到 4–64，后者上限为 1048576。 |

`--giant-backend`／`--seed-backend`、对应 TOML、对应协作环境变量依次优先。承载路径以承载模数的实际算术位宽判断资格。链形状和尾部策略见 [Stage2 点块与链长](STAGE2.md#cuda点块与链长选项)。[点后端与链策略解析](../../src/ecm/stage2/cuda/host_policy.cpp#L44) · [点块预算](../../src/ecm/stage2/planning/giant_chunk.h#L14) · [运行时链选择](../../src/ecm/stage2/cuda/batch.cpp#L236)

以下值由生产引擎固定。显式传入不同值会在配置检查时报错。开发引擎可用这些变量对照算法，构建入口为 [`build_stage2_dev.ps1`](../../tools/build/build_stage2_dev.ps1)。

| 环境变量 | 生产固定值 | 作用 |
|---|---|---|
| `NTT_BABY_DEVICE` | `1` | 请求设备 Baby 点生成和分组归一化。 |
| `NTT_GIANT_SEED_DEVICE` | `1` | 在设备生成 giant 链种子。 |
| `NTT_GIANT_SEED_PAIR` | `1` | 从 `[D]Q` 出发，在同一次 ladder 中计算相邻 seed。 |
| `NTT_GIANT_BASE_CPU` | `0` | 生产在设备计算 `[D]Q`。开发对照可设 `1` 切换到 CPU。 |
| `NTT_POINT_MERSENNE` | 显式值须为 `1` | 允许符合资格的点算术使用梅森归约。采用固定 PTX 模式 3 的构建默认设置为 1，实际启用仍取决于模数和内核。 |
| `NTT_XADD6` | `1` | 使用六乘版本的差分加点。开发引擎可设 `0` 对照通用公式。 |

生产 `NTT_D_MODEL` 默认 `1`，它只记录 CUDA 引擎内旧 D 成本模型的请求状态，当前生产实现始终报告 `enabled=0`。需要固定 D 时使用 `--d` 或 TOML 的 `d`。旧测量成本文件的环境校验可拒绝不同的点后端、等待、诊断设置，并把 `NTT_D_MODEL` 设置为 0。[生产固定项](../../src/ecm/stage2/cuda/options.cpp#L410) · [编译期点算术策略](../../src/ecm/stage2/cuda/context.h#L49) · [D 模型状态](../../src/ecm/stage2/cuda/plan.cpp#L88) · [旧成本文件条件](../../src/ecm/stage2/planning/planning.cpp#L2457) · [开发引擎读取](../../tools/experiments/cuda/stage2_tree/engine.cuh#L398)

### 点生成检查与故障注入

以下默认关闭，以 `1` 启用，适用于隔离测试。故障注入项会故意触发错误或回退，需单独使用测试存档和性能文件。

| 环境变量 | 作用 |
|---|---|
| `NTT_BABY_DEVICE_TEST` | 启动时运行设备 Baby 的合成数据检查。 |
| `NTT_BABY_DEVICE_CHECK` | 对实际设备 Baby 结果用独立 ladder 和 GMP 仿射归一化逐点比较。 |
| `NTT_BABY_DEVICE_TEST_BAD` | 翻转一项 Baby 输出位。与 `NTT_BABY_DEVICE_CHECK=1` 搭配验证检查能捕获损坏。 |
| `NTT_BABY_DEVICE_ALLOC_FAIL` | 模拟设备 Baby 缓冲准入失败，回退到原有生成路径。 |
| `NTT_BABY_WHEEL_ALLOC_FAIL` | 模拟分段递推缓冲分配失败，回退到设备独立 ladder。 |
| `NTT_GIANT_CHAIN_CHECK` | 对实际 giant 链结果再次运行独立 ladder，比较归一化 x。 |
| `NTT_GIANT_SEED_CHECK` | 对设备 giant 链 seed 做主机对照。 |
| `NTT_XADD6_TEST` | 启动时对 128 组普通域和 Montgomery 域输入进行差分加点对照。 |
| `NTT_XADD6_TEST_BAD` | 配合 `NTT_XADD6_TEST=1` 损坏候选输出，用于验证测试能报错。 |

[设备 Baby 对照](../../src/ecm/stage2/cuda/points.cpp#L931) · [Giant seed 对照](../../src/ecm/stage2/cuda/points.cpp#L395) · [Giant 链对照](../../src/ecm/stage2/cuda/batch.cpp#L328) · [xADD 对照](../../src/ecm/stage2/cuda/checks.cpp#L845)

## CUDA 主机等待与进度

| 环境变量 | 未设置时 | 取值和效果 |
|---|---|---|
| `NTT_CUDA_WAIT_MODE` | CUDA 的上下文默认调度 | `0` 自动，`1` 忙等，`2` 让出 CPU，`4` 阻塞。设置在 CUDA 设备上下文。 |
| `NTT_HOST_EVENT_WAIT` | `0` | `0\|1`。`1` 为明确的事件等待启用有界退避轮询，仍可回到原生事件同步。 |
| `NTT_HOST_READBACK_WAIT` | `0` | `0\|1`。`1` 在同步设备到主机拷贝前通过额外事件等待，适合单独测量读回策略。 |
| `NTT_S4_TIMING_FIFO` | `1` | `0\|1`。使用 FIFO 回收 S4 计时事件，独立于上述等待模式。 |
| `NTT_NO_PROGRESS` | `0` | `1` 关闭 CUDA 引擎的 S4 批进度，其他阶段及结果的显示仍由日志级别决定。 |

显式设置 `--low-cpu-wait` 或 `low_cpu_wait` 会重写前三项。启用时写入 `4 / 1 / 0`，禁用时写入 `0 / 0 / 0`。要单项对照，两处都应省略。调试输出包括 `stage2_host_wait_stats`。由 CUDA 驱动处理的外部变量 `CUDA_LAUNCH_BLOCKING` 也参与旧成本文件校验，旧文件要求未设置或 `0`。[等待策略优先级](../../src/ecm/stage2/cuda/host_policy.cpp#L135) · [事件等待实现](../../src/ecm/stage2/cuda/host_wait.h#L12) · [进度开关](../../src/ecm/stage2/cuda/engine.cpp#L988)

## CUDA 队列调度

| 环境变量 | 未设置时 | 取值和效果 |
|---|---|---|
| `ECM_STAGE2_QUEUE_PIPELINE` | 符合小位宽条件时为 `2` | `0` 关闭，`1` 无运行时调优时允许提前准备下一曲线，`2` 支持运行时调优反馈。常规双槽且启用素数命名时自动上限为 2048 bits，其余为 4096 bits。仍要求同一执行窗口有多条曲线，且没有未恢复的持久化预留、relocation、tune 子进程或外置 phase gate。 |
| `ECM_STAGE2_EARLY_GPU_HANDOFF` | `1` | `0\|1`。允许合格的 `--factor-only` 流水线在原曲线 CPU 收尾前交接 GPU 槽位。要求关闭运行时调优并具备 GPU 完成事件。 |
| `ECM_STAGE2_QUEUE_PHASE_MASK` | `57` | `0..127` 的十进制掩码，控制双槽中哪些阶段使用同一互斥量。也是并发性能样本的匹配条件。 |
| `ECM_STAGE2_GIANT_GATE` | 按批互斥 | `loop` 请求将整个 giant/G 树循环作为一段加锁，要求掩码同时含 8 和 16。其他值继续按批加锁。 |
| `ECM_STAGE2_QUEUE_D_SEARCH` | `0` | `1` 让双槽 `auto` 评估邻近 D。仅在 D 未固定且未强制使用租约模式时适用。 |
| `ECM_STAGE2_PHASE_GATE` | 无外置互斥量 | Windows 实验接口，格式为 `掩码:组名`。同组进程共享一个互斥量；组名 1–48 字符，只接受字母、数字、`-` 和 `_`。非零掩码与独立 tune 命令互斥，并关闭曲线运行时学习及启动短校准。手动设置会阻止队列自动双槽和预启动。 |
| `ECM_STAGE2_PHASE_TRACE` | `0` | `0\|1`。在 debug 日志输出 `stage2_phase_window`，记录起止时间、等待时间和是否加锁，涵盖未加锁阶段。启用 NVTX 跟踪时也会开启阶段标记。 |
| `ECM_STAGE2_FTREE_LEASE` | `0` | `0\|1`，有 phase gate 组名时允许复用同组大工作区租约。生产 worker 会根据 `cuda.workspace_policy` 的固定队列路线写入或清除该变量，队列对照使用该 TOML 选项。 |

掩码位依次为 Baby `1`、F 树 `2`、逆多项式 `4`、Giant `8`、G 树及 fold `16`、下降 `32`、GCD `64`。例如 `57 = 1+8+16+32`。正常队列测试通过 `ECM_STAGE2_QUEUE_PHASE_MASK` 调整掩码，由程序生成子进程的组名。[双槽用法](STAGE2.md#cuda-双曲线调度) · [流水线默认条件](../../src/ecm/stage2/process.cpp#L35) · [队列准入](../../src/ecm/stage2/task.cpp#L665) · [邻近 D 条件](../../src/ecm/stage2/task.cpp#L895) · [阶段与租约实现](../../src/ecm/stage2/cuda/phase_gate.h#L16)

## 原生 OpenCL Stage2

仅影响 `ecm_opencl_stage2.exe`。生产启动器会把 `opencl.point_plan`、非零的 `opencl.point_chunk`／`point_group`／`point_window`、关闭的 `opencl.short_point_batch` 和实际生效的调优配置传入环境。因此这些 TOML 值可覆盖同名环境设置。选择 `opencl.point_plan="auto"` 后才进入 `ECM_OPENCL_POINTS` 的显式分支。显式点参数和树实验开关跳过自动点计划比较。AMD 的目标大于 512 bits 时使用 CPU 点，仍在 OpenCL 上运行多项式阶段。

| 环境变量 | 未设置时 | 取值和效果 |
|---|---|---|
| `ECM_OPENCL_POINTS` | 随设备和形状自动选择 | `cpu\|device` 固定点生成位置，保持多项式执行位置和设备资格检查。 |
| `ECM_OPENCL_POINT_CHUNK` | 按厂商及目标 N 的 limb 数选择 | `1..4096`，每条设备点命令的点数上限。显式值锁定手动计划。 |
| `ECM_OPENCL_POINT_GROUP` | NVIDIA `32`，其他 `0` | 控制本地工作组大小。`0` 由运行时选择，正值不得超过设备和点内核的限制。 |
| `ECM_OPENCL_POINT_WINDOW` | 自动选择 | `0..65536`，显式 0 保留原 G 批窗口，正数请求跨批坐标窗口。 |
| `ECM_OPENCL_POINT_AUTO` | `1` | `0\|1`。对无手动 chunk/group 且符合资格的 NVIDIA 设备点，依据命令耗时升档。 |
| `ECM_OPENCL_WIDE_AUTO` | `1` | `0\|1`。将自动点批量策略扩展到 8193–16384 bits。 |
| `ECM_OPENCL_GIANT_BASE` | `1` | `0\|1`。复用 `[D]Q`，将 giant 点的标量从 iD 缩成 i。 |
| `ECM_OPENCL_GIANT_CHAIN` | 仅在合格的短标量计划或相应点计划中启用 | `0\|1`，手动比较独立 ladder 和设备递推。递推需要设备点和 `ECM_OPENCL_GIANT_BASE=1`。 |
| `ECM_OPENCL_SHORT_POINT_BATCH` | `1` | `0\|1`，允许合格 RTX 4060 Laptop 短标量形状合并 Baby 提交并使用 giant 递推。 |
| `ECM_OPENCL_CPU_CHAINS` | 合格的宽位或密集 CPU 形状开启 | `0\|1`，在 GMP 点路径上启用分段递推。 |
| `ECM_OPENCL_GIANT_CPU_THREADS` | 合格的 HX 370／RTX 4060 Laptop 密集 GMP 形状为 4，其他为 1 | `1..4`。在 GMP giant 路径上并行生成与归一化；手动指定后最小并行窗口降至 2 点。 |
| `ECM_OPENCL_BATCH_GATHERS` | RTX 4060 Laptop 且 P≥32768 时为 1 | `0\|1`，成批提交多项式 gather。 |
| `ECM_OPENCL_FULL_G_BATCH` | 合格形状为 1 | `0\|1`，在设备拼接到 P 个根再建 G 树，末批取实际点数。自动条件为 RTX 4060 Laptop 普通路径、31 个算术 limb、P=80640、点块 45100 且预算至少 5120 MiB。 |
| `ECM_OPENCL_QUADRATIC_G_LEAVES` | 合格形状为 1 | `0\|1`，用相邻根直接组成二次叶。自动条件为 RTX 4060 Laptop 普通路径、点块至少 4096，且（算术 limb 数，P）为（31，80640）或（192，13440）。 |
| `ECM_OPENCL_MONIC_G_LEVELS` | 上述二次叶自动条件且算术 limb 数为 192 时为 1 | `0\|1`，当 G 树子节点同次且可缩短 NTT 时省去首一项。 |
| `ECM_OPENCL_CACHE_PROGRAMS` | `1` | `0\|1`，在同一 OpenCL context 按源代码和构建参数复用 program。 |

以下变量由生产启动器从配置写入曲线子进程。生产使用 TOML 或 CLI 设置相应值。

| 环境变量 | 生产来源及用途 |
|---|---|
| `ECM_OPENCL_CONFIG_PLAN` | `opencl.point_plan`，传递 `auto`、`cpu`、`device` 或 `chain`。 |
| `ECM_OPENCL_RUNTIME_PROFILE` | `opencl.tune_file` 解析后的本机路径，仅在运行时学习启用、点计划为 `auto` 且已指定独立 TOML 文件时传递。 |
| `ECM_OPENCL_RUNTIME_REPEATS` | `tune.min_repeats`，候选需要的实测曲线数，默认 2。内部采样器接受 1–1000。 |
| `ECM_OPENCL_RUNTIME_BUDGET` | `tune.runtime_samples`，候选总样本预算，采样器无上游设置时取 10，接受 0–1000。预算至少容纳两组完整对照才发起采样。 |

[OpenCL Stage2 用法](STAGE2.md#原生-opencl-入口) · [配置传递](../../src/ecm/stage2/cuda/main.cpp#L561) · [点计划条件](../../src/ecm/stage2/opencl/engine.cpp#L102) · [点生成及选型](../../src/ecm/stage2/opencl/engine.cpp#L193) · [多项式开关](../../src/ecm/stage2/opencl/engine.cpp#L419) · [学习预算](../../src/ecm/stage2/opencl/point_tune.h#L40)
