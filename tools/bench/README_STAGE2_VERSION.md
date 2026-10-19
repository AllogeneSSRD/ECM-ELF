# 数字版本 Stage2 全范围复测

`bench_stage2_version.py` 复用已验收的 Edwards / Prime95 精确输入矩阵，测 GPU1 的普通余因子与梅森承载。默认 13 个输入、两档实际 B2、两条 GPU 路径、各三次，共 156 次。CPU 和 Montgomery 不重跑。当前结果见 [Stage2 性能](../../docs/performance/STAGE2.md)，输入与历史参考见 [参考页](../../docs/performance/EDWARDS_STAGE2.md)。

## 构建与测量

使用正式 local 构建入口，测量前完成编译；不要同时运行本任务的 GPU 诊断：

以下示例记录已完成的 0.38.4 研究。重建该版本须使用对应版本源码；用当前源码开展新研究时，应同时更新目录与 `--version`，以程序实际报告的版本为准。

```powershell
tools/build/build_stage2_local.ps1 -Build build_cuda_cmake/local_stage2_0_38_4 `
  -Arch sm_89 -Engine production -SplitCompile 8
python tools/bench/bench_stage2_version.py `
  --exe build_cuda_cmake/local_stage2_0_38_4/ecm_cuda_stage2.exe `
  --reference data/experiments/edwards_stage2_reference_20261017/matrix_01 `
  --version 0.38.4 --output data/experiments/stage2_v0_38_4_20261018/full_matrix
```

新输出目录必须不存在；参考目录须含已完成准备的 `plan.json`、13 份已独立验证的 save、26 个 Edwards Prime95 条件。工具复制 EXE/GMP、输入及采集器，核对精确余因子整除、save 状态和校验。程序在每次结果中报告的数字版本必须与 `--version` 一致。新增研究元数据和图注使用版本号，历史未记录数字版本的程序明确标注未知，不补造版本。

设备固定为 GPU1，操作者须确认它仍为 RTX 4060 Laptop。工具记录实际设备名称，不改变时钟、功耗、优先级或 GPU0 的工作。预算为 arena6300/fold owner640/batch256 MiB；D 自动选择，短标定/runtime tune/自动承载/relocation/NTT 计数关闭。每轮遍历全部精确条件，偶数轮交换普通/承载顺序；每次为独立进程，没有丢弃的预热或首轮。所有调用保留正常算术检查。

每次调用前后查询选中 GPU UUID 的计算进程：只允许本输出目录中的程序；其他 GPU 的任务不影响该检查。外部任务或无法识别的进程会中止采集，查询快照和已有数据保留。`--allow-shared-device` 仅用于明确接受共享设备的独立研究，记录在协议中。前后快照不证明调用期间每一瞬间都没有其他进程。

主计时为 `stage2_full_wall.total=init+main`，包括引擎内必需检查和 GCD，排除 Stage1、进程启动、外层选 D 及计时前预检。进程墙钟另存，250 ms 轮询包含最多一周期的退出观察延迟。NVML 观测整张选中显卡，不代表进程峰值或纯内核功耗。

## 续跑与审计

```powershell
python tools/bench/bench_stage2_version.py --resume --version 0.38.4 `
  --output data/experiments/stage2_v0_38_4_20261018/full_matrix
python tools/test/check_stage2_version.py `
  --study data/experiments/stage2_v0_38_4_20261018/full_matrix `
  --output data/experiments/stage2_v0_38_4_20261018/full_matrix/audit.json
```

计划在每条接受后原子保存。`--max-runs N` 可在完成 N 条新调用后正常停下，样本保留且矩阵仍未完成；续跑跳过已接受 ID。采集器改变、已有失败、存在未入账调用日志时拒绝续跑，须先调查并保留证据，不能覆盖原文件后补成成功样本。改变版本、输入或协议应使用新目录。

正式审计要求 156 个唯一 ID、52 格各三次，重新核对输入/save、结果版本和身份、实际 B2/D/P、φ(D)/2、完整区间点数与 G 批数、NF 状态、原始计时守恒、强制 GMP 检查、分项记录和调用设置。提前出因子的运行保留，但不满足完整 NF 时间矩阵。该审计不声称提供每个叶值的独立 oracle；正常 GMP 抽样与独立末点验证各自有明确范围。

获准只跑一次的筛查可在采集时指定`--repeats 1`，审计须显式指定`--expected-repeats 1`；仍要求完整52格、全部身份/数学检查和已声明的前后进程快照。审计默认仍为三次，不把一次结果当作三次中位数或波动估计。不同样本数的D搜索版本对照见[有界搜索验证](README_STAGE2_D_CATALOG.md#有界搜索实现验证)。

## 汇总、绘图与公开副本

```powershell
python tools/bench/plot_stage2_version.py `
  --study data/experiments/stage2_v0_38_4_20261018/full_matrix `
  --output data/benchmarks/stage2_v0_38_4_20261018/final `
  --figures data/figures/stage2_v0_38_4_20261018/final `
  --publish docs/performance/attachments
```

需要 Matplotlib 和 NumPy。新统计/图目录必须不存在；绘图先严格审计当前矩阵，逐格核对旧 Edwards 参考的身份，再生成 CSV/JSON 与 PNG/SVG。绝对耗时和 NTT 模块显存图以相同数据同时生成 `-log`、`-linear` 两版（含 PNG/SVG），线性纵轴从0开始；百分比图保持线性坐标。当前 GPU 使用中位数及 min/max，版本比较阴影组合两侧观测范围，不是置信区间。init/main 按逐调用比例取均值；NTT 模块峰单列，不加组件峰、不当作进程峰。

同一工具重绘已验收历史参考：Prime95 31.4b05 全部两曲线族、32 条 libecm 样本、CPU GW 0.33.0 的 36 次实测，以及 Montgomery 55 W 及两种频率设置下的三份 9 位宽×3 B2×3 次矩阵。不将缺测 CPU 单元插值，不按频率、B2 或线程数归一化。历史参考路径集中在绘图脚本顶部，原始研究不改写。`--references-only` 可在等待 GPU 时单独重绘这些参考，不能生成当前版本完整性能结论。

`--publish` 只复制选定 PNG 到指定 attachments 目录；原始结果、账户字段、运行校验值、EXE、遥测和全部数据只在忽略的 data 目录。公开提交只包含工具、文档和审查过的图。

## 正常运行时选型矩阵

当前矩阵使用 `build_cuda_cmake/stage2_d_search_0_39_3/` 的 ECM-ELF 0.39.3，保留 0.38.4 完整默认矩阵作参考。`bench_stage2_runtime_matrix.py` 冻结指定 EXE/GMP、13 份已独立验证的原始 Stage1 点。每个实际 N/B2 条件提供一条 **上限 48 曲线的连续 worktodo 队列**，共 26 条队列；当当前最快合格参数组完成第 3 个入账观察时，提前结束该条件。相同保存点重放仅测量耗时和选型，不增加独立 ECM 覆盖。

```powershell
python tools/bench/bench_stage2_runtime_matrix.py `
  --baseline data/experiments/stage2_v0_38_4_20261018/full_matrix `
  --exe build_cuda_cmake/stage2_d_search_0_39_3/ecm_cuda_stage2.exe `
  --version 0.39.3 `
  --import-study data/experiments/stage2_runtime_v0_39_3_20261019/matrix_01 `
  --output data/experiments/stage2_runtime_v0_39_3_20261019/adaptive_01
```

D/P 和普通/承载全由正常运行时选型决定；不执行单独 tune，不锁定 D 或算术。每组采样上限保持默认 10，启用准备缓存。队列提供并验证原指数、已知因子和精确 N；48 条计划用于保留正常的队列剩余量判断和父子进程准备复用，外部采集器按第 3 次观察提前停止。先运行最小位宽低 B2，随后按位宽和 B2 降序，优先核对最大位宽高 B2。

停止条件要求完整形状相同、当前调用不是探测对、更新后的均值仍不高于其他已准入重复候选；探测过程中的两个观察不会触发停止。采集器用继承的原生停止事件请求退出，保留已经开始的至多一条额外曲线、未完成 worktodo 和检查点，不修改原始结果。此规则是用户指定的采集启发式，不表示候选已穷尽：程序可能在更多重复之后开放更大的搜索配额。

本轮保留原规则的三条完整队列，以及用户要求立即停止的 14328-bit 高 B2 队列中 42 次完整 NF，共 186 次调用；后续条件采用提前停止。`--import-study` 只接受记录了用户批准停止的原始研究，并核对同一冻结程序、输入和数学结果。中断时未完成的一次调用只保留日志，不计入 NF 或成绩；该队列未落盘的遥测和精确进程墙钟明确标为不可用，完整曲线计时与算术检查仍逐条验证。原目录与日志不改写。全新研究省略 `--import-study`，从空白性能文件开始。不同条件实际执行次数和被选行样本数分别报告。

逐调用保留完整引擎计时、`curve_done.wall`、实际 D/P/承载、选择和搜索理由、GMP 检查；逐队列保留整卡遥测、进程墙钟和性能文件快照。调用前后查询 GPU1 外部进程，发现竞争即保留证据并拒绝该轮；边界快照不能证明每一瞬间绝无竞争。GPU0 用户工作不作更改。预算保持 arena6300/fold640/batch256 MiB，短标定、relocation、NTT 计数关闭。

成绩采用最终 tune 的最佳合格实测决策：全部队列完成后通过生产 worktodo 的只规划入口读取，关闭新采样、不锁定参数。程序须返回 `runtime_measured_fastest`、至少两次合格观察，均值为已准入重复候选中的最小者。记录实际样本数、均值、样本标准差及原始 min/max；这些观察也参与选型，**不冒充独立的选型后 n=3 验证**，也不证明全部 D 的全局最优。默认普通/承载仍使用原 n=3 中位数，图注区分统计口径。

```powershell
python tools/bench/bench_stage2_runtime_matrix.py --resume `
  --output data/experiments/stage2_runtime_v0_39_3_20261019/adaptive_01
python tools/test/check_stage2_runtime_matrix.py `
  --study data/experiments/stage2_runtime_v0_39_3_20261019/adaptive_01 `
  --output data/experiments/stage2_runtime_v0_39_3_20261019/adaptive_01/audit.json
python tools/bench/plot_stage2_version.py `
  --study data/experiments/stage2_v0_38_4_20261018/full_matrix `
  --runtime-study data/experiments/stage2_runtime_v0_39_3_20261019/adaptive_01 `
  --output data/benchmarks/stage2_runtime_v0_39_3_20261019/final `
  --figures data/figures/stage2_runtime_v0_39_3_20261019/final `
  --publish docs/performance/attachments
```

审计要求完整 26 个条件、每条实际执行的唯一完整 NF，核对提前停止的触发观察、原生退出确认及队列剩余状态，同时核对点重放、输入、版本、原始计时、数学检查及覆盖；用 JSONL 的完整精度观察独立重算各组均值/方差，六位小数文本计时另按舍入精度对照。保留队列另与导入前的原始计划逐项核对。最终只规划输出须与被选性能行和原始观察对应。

`stage2_ecm_tune.toml` 为活动文件，最终完全相同的 `stage2_ecm_tune_frozen.toml` 可通过 `stage2_tune_profile` / `--tune-profile` 复用。它们留在忽略的 data 目录，不公开设备身份或运行去重标识。学习队列的引擎、队列和进程时间分别报告，不与最优成绩重复相加；计时边界见 [准备结果的复用](../../docs/architecture/AUTO_B2.md#准备结果的复用)。
