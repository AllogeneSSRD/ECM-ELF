# PRAC 种子缓存开发草稿

目标是让离线工具搜索并保存每个素数的赢家，文件头可直接报告 ladder 和 PRAC 的 D/A/W，运行时重放赢家以重建原执行描述符。

- 假设每个素数的最优种子只依赖素数、候选窗口和成本模型。素数序号和重复次数由 $B_1$ 与 LCM 模式重建。
- 本实现沿用 [GMP-ECM LucasChainGenerator](../../.refactor/ecm/LucasChainGenerator/README) 的离线预计算思路。本项目搜索 Prime95 的种子窗口，保存种子，执行器保持原 PRAC 规则。第三方工具使用其自己的链枚举与编码。

源码依据、文件合同和实测数据见 [PRAC 架构](../architecture/PRAC.md)。原始记录位于 `data/experiments/prac-cache-v2-20261114/`。

## 交接状态

本轮 PRAC 功能沿 0.63 版本线更新为 `0.63.4`。下列本地验证使用此前的 0.63.3 构建，具体版本及二进制身份见各组实测记录。

| 改动位置 | 当前实现 |
| --- | --- |
| [共享计划器](../../src/ecm/stage1/montgomery/prac/ecm_prac_plan.cpp) | 120 B v2 头记录两套 D/A/W，载荷每个素数一个 32 位种子。筛法重建素数和重复数，逐种子重放验证成本及描述符哈希。v1 的 7 档加载后迁移，身份仍使用原 64 B 逻辑头 |
| [CPU IFMA 入口](../../src/ecm/stage1/montgomery/cpu/workflow.cpp) 和 [CUDA 入口](../../src/ecm/stage1/montgomery/cuda/prac/runtime.cu) | PRAC 可选 `ECM_PRAC_SEARCH=1..50`，默认 7。CUDA 从请求快照读取。日志包含权重和运算次数，执行描述符布局保持 16 B |
| [离线工具](../../tools/gen/prepare_prac_cache.cpp) | 增加 `--search` 与 `--summary`，旧生成命令继续可用。报告命令读取并验证文件头和文件长度 |
| [CUDA 窗口验证](../../tools/test/test_cuda_prac_windows.py) 和 [PRAC 规则验证](../../tools/test/test_prac_single_add.py) | 支持从 v1 或 v2 计划取样。v2 按素数序号重建样本描述符 |
| [使用说明](../usage/STAGE1.md#预计算-prac-种子) 和 [交互页](../web/prac-search.html) | 补充离线生成、选档、读头和迁移用法 |

## 预生成与实测

仓库根 `cache/` 已生成并回读 12 份 50 档 v2 种子文件。覆盖 (B_1=50{,}000, 100{,}000, 1{,}000{,}000, 11{,}000{,}000, 44{,}000{,}000, 260{,}000{,}000) 的 LCM 倍数 1 和 12，12 份种子文件共 141,417,312 B。每种 LCM 模式还生成 4 份达到写缓存门槛的标量文件。生成与回读日志为 `precompute-t1.log` 和 `precompute-t12.log`。

最大样本的种子文件由 227,133,824 B 降为 56,783,560 B。50 档、LCM=1 的文件汇总如下。

| 方法 | D | A | W |
| --- | ---: | ---: | ---: |
| Ladder | 375,102,575 | 375,102,574 | 3,751,025,745 |
| PRAC | 63,605,892 | 505,182,143 | 3,349,122,318 |

首次生成 2.6 亿、LCM=1 的 50 档计划记录为 177.102 s，当时主机还有其他任务，此值用于记录本次预计算，缺少独立性能对照。旧三次热读记录保存在 `replay-r1.log` 至 `replay-r3.log`。接续复测在同机以 5 组交替热读配对比较 7 档 v1/v2，另对既有 50 档 v2 分段测量。条件、范围与二进制身份见 [PRAC 架构](../architecture/PRAC.md#热缓存读取与重放耗时)，原始样本在 `continuation/timing.csv` 和 `continuation/timing-summary.json`。

GMP-ECM 上游 README 的分钟和小时是整张 Lucas 链表的离线生成时间。这里的候选种子窗口与上游按长度枚举差分链的搜索规模不同，见 [参考源码与耗时分析](../reference/ECM_MONTGOMERY_MATH.md#大-b₁-的离线生成时间)。

## 已通过的验证

- 用旧工具写出 v1，在 (B_1=2,100,100000) 的两种 LCM 模式验证 v1→v2。种子载荷与旧描述符提取的种子逐字节一致，7 档身份和旧头哈希一致。
- 按同一输入分别用 1、7、50 档生成，和仓库 Python 计划器核对全部种子与 D/A/W。核对头校验、报告命令、载荷破损后重建、非法搜索宽度拒绝，以及 v1/v2 窗口记录一致。临时验证脚本为 `verify.py`。
- 0.63.3 已构建 CPU/OpenCL 主程序、离线工具及启用 PRAC 的 CUDA Stage1 完整目标（CUDA 13.3、sm_89、384 位档、12 个设备对象）。在独立目录以正常 Windows 路径权限重试，CPU 的 `weakly_canonical` 和先前 CUDA/MSVC 的 `0xC0000005` 均未重现。构建日志为 `continuation/build-cpu.log` 和 `continuation/cuda384-final.log`。
- CPU AVX512-IFMA 和 RTX 4060 Laptop CUDA 各运行 8 条 Montgomery PARAM0 曲线（$N=2^{127}-1$，$B_1=1000$，$B_2=0$，LCM 倍数 12，sigma=26～33）。ladder、PRAC 7 档、PRAC 50 档及 50 档热命中的 save 末点和校验字段均相等。两种后端也在 $B_1=100000$ 命中先前预计算的 50 档文件，再次与 ladder 逐曲线对齐。
- CPU 0.62.15 在 $B_1=100000$、CUDA 0.53.6 在 $B_1=10000$ 创建真实的 7 档中途检查点。0.63.3 分别用 v1 自动迁移、只保留 v2 的副本恢复，均继续原进度并对齐 8 条 ladder 末点。改用 50 档时，CUDA 拒绝恢复并保留原件，CPU 跳过旧的未完成状态并从头重算。证据为 `continuation/runtime-cpu-complete/result.json` 与 `continuation/runtime-gpu-complete/result.json`，同目录保存输入、日志及二进制 SHA-256。
- 现有 `test_cuda_p1_prac.py` 的 16 项通过，覆盖两种 LCM 模式、ladder/resident/PRAC 末点、断点恢复、损坏拒绝及重建。结果见 `continuation/standard-cuda-prac/result.json`。

## 接续优先项

本地 384 位档的完整目标、末点、检查点和热缓存测量已完成。全位宽、多架构发行包及冷存储读取待复跑。草稿和原始记录保留。

## Lucas 链转种子与离线选种

已完成逆推现有 PRAC 规则的原型，外加有下界剪枝的扩大窗口搜索。三份 v2 实验缓存保存在 `data/experiments/prac-lucas-compare-20261114/joint-*`，以现有共享计划器回读，并在十万档通过 CPU IFMA 与 CUDA 末点对照。逆推、候选扫描与 D/A/W 见 [PRAC 架构](../architecture/PRAC.md#保留种子格式的离线选链实验)。

剩余决策是是否把离线原型做成正式可选生成模式。实验缓存沿用 v2 格式，使用独立目录区分搜索策略，运行时根据实际描述符计算计划身份。若需要把任意 Lucas 获胜链全部纳入执行，仍需定义能表示这些链的执行指令和检查点身份，并验证 CUDA 活跃点与寄存器开销。

## CPU 与 RTX 4070 Ti 执行对照

0.63.3 的原生 v2 计划已用于 M4423、$B_1=100{,}000$ 的 ladder、PRAC 7 档及 50 档对照。计量边界和实测汇总见 [CPU IFMA 与 RTX 4070 Ti 的首次建表和主计算](../architecture/PRAC.md#cpu-ifma-与-rtx-4070-ti-的首次建表和主计算)。原始数据、4608 位构建清单和运行脚本保存在 `data/experiments/prac-execution-4070ti-20261114/`。
