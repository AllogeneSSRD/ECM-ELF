# Stage2 D 候选表工具

生产、tune及Python规划工具的唯一自动D来源是 [ecm_stage2_d_catalog.h](../../src/core/ecm_stage2_d_catalog.h)。当前1094项D按数值升序、去重维护，P=φ(D)/2统一派生；`default_plateau`标记既有保守平台保护的14项资格范围，不构成独立表。

当前运行与搜索规则见 [Auto B2 / tune](../../docs/architecture/AUTO_B2.md)，来源研究、冻结证据和待验收事项见 [D候选表草稿](../../docs/development/STAGE2_D_CATALOG_DRAFT.md)。Prime95数学依据见 [固定参考](../../docs/reference/PRIME95_STAGE2.md#3-d-候选表的含义与适配条件)。

## 完整379419项枚举

[本机参考文本](../../data/experiments/stage2_d_catalog_implementation_20261011/D_47smooth_379419.txt)包含379419条D/P数据，按D升序排列。列为D、P=φ(D)/2、是否满足偶数D≥6。7,198,077 bytes，SHA256为`fb616e036324e25734632b8d4b5121b004f463c7aa2126db20bdea21847d68e1`；同目录JSON保留数量和生成工具身份。

枚举域为≤200000000的47-smooth D、非零P，包含奇数；282395项满足portable偶数条件。上限包含D=200000000，严格小于上限时少1项。1094项共享表来自多来源扩展池，其中4项含53或59，不能解释为参考枚举的纯子集。参考文本最后一列是偶数D≥6的筛选标记；头文件中的布尔字段则是`default_plateau`策略标记，false不表示奇数。数学说明见 [D参数数学](../../docs/reference/ECM_STAGE2_D_PARAMETERS.md)，精确来源计数见 [当前生成规则](../../docs/architecture/AUTO_B2.md#来源筛选与规模)。它用于检查覆盖和来源，不在生产或tune中重新枚举。文件保存在Git忽略目录，分发源码时不包含本机数据。

从仓库根目录再生成，需要Python 3.11+，目标文件和同名JSON必须不存在：

```powershell
python tools/bench/stage2_d_catalog.py --export-reference data/experiments/my_d_catalog/D_47smooth.txt
```

`--d-limit`可改变参考域上限，默认200000000；改变上限后项数不一定为379419。此命令不执行GPU任务，不修改候选头文件。

`stage2_d_catalog.candidates()`读取共享头文件，提供D/P给规划、预算分析和相关工具；`reference_candidates()`只为全枚举与离线扩充研究服务。新增自动候选应审查后修改共享头文件，不能借参考枚举或性能文件静默改变生产候选域。

## 离线覆盖分析

`analyze_stage2_d_catalog.py`检查当前共享表，解析Prime95有效表和既有实测几何，生成候选扩充审查产物及P→NTT边界图。它不运行GPU曲线，不修改生产EXE、INI、队列或性能文件，也不自动部署提议表。

从仓库根目录运行，需要Python 3.11+和matplotlib：

```powershell
python tools/bench/analyze_stage2_d_catalog.py `
  --profile build_cuda_cmake/production_stage2/stage2_ecm_tune.toml `
  --output data/experiments/stage2_d_catalog_my_run `
  --figures data/figures/stage2_d_catalog_my_run
```

输出目录必须不存在，重复研究换目录以保留已有证据。默认上游`.refactor/p95v3106b01.source/ecm.cpp`，可用`--prime95-source`覆盖；`--d-limit`默认200000000。省略`--figures`时图前缀为`data/figures/<输出目录名>`。

### 输入与计算

- 从`src/core/ecm_stage2_d_catalog.h`读取当前D，另读取格式7 TOML `width.*.table.*.matrix`中的已测D；表外已测D在提议中保留，不直接改变运行域。
- 解析本地Prime95两处`poly_D_data`的条件布局，只采用有效声明，核对`numrels=φ(D)/2`。布局不符合预期时停止。
- 枚举偶数47-smooth参考候选，按同P取最大D作为边界代表；它不是其他同P参数被严格支配的证明。
- 并入共享表、实测D、Prime95表及2幂扩展；在52个选定位宽、fold长度2¹⁴～2²⁷及填充树边界附近取P邻项。
- 按默认packing整数精确性约束计算fold形状；完整树、inverse、分块、执行路径和联合显存由原生计划另行决定。

纯数学依赖为`stage2_d_catalog.py`的候选/φ、`calibrate_stage2_d.py`的φ/shape；不会调用它们的计时入口。

### 输出

- `analysis.json`：源码提交、输入SHA256、来源计数、上游表、选定S、边界及候选全集。
- `catalogue.csv`：D、P和来源标记，一项可有多种来源。
- `catalogue_proposal.h`：D/P审查提议，运行不引用此文件；部署时只把审查通过的D及策略标记纳入共享头文件。
- `sources/`：本次输入与工具冻结副本。
- `<图前缀>_fold_ntt_steps.png/.svg`：P与fold长度的关系，不是运行时间或进程显存峰值。

候选数量不等于完整曲线测量数量，长度相符不证明整条曲线显存适配。默认性能保护保持配对中位数/MAD门限和现场原生准入。各输入域的性能证据见[Stage2性能](../../docs/performance/STAGE2.md)；离线边界与规划查询不能代替完整曲线计时。

## 持续搜索的离线结构分析

`analyze_stage2_d_search.py`读取当前共享表，计算几何包络、同P冗余、P非单调性及不同S/B2下的粗区间规模，保存几何支配却需要更多fold的反例。它只调用现有Python整数packing复刻，不运行CUDA、完整曲线、测试或原生资源准入，也不修改生产表。

运行 `python tools/bench/analyze_stage2_d_search.py --output data/experiments/<新目录>`。默认S为512/1939/4667/7995/12011/16319，B2为260000000/26000000000/2600000000000，owner为640 MiB；可用`--bits`、`--b2`、`--owner-mb`覆盖。输出目录必须不存在。

输出`analysis.json`、`catalog.csv`和`sources/`；图默认写入`data/figures/<目录名>_geometry.png/.svg`，`--figures`可指定前缀。图中fold代理为`(G−1)L log₂L`，忽略尾fold缩短及其他阶段，不是Stage2时间预测。原理及待实现策略见[D搜索研究草稿](../../docs/development/STAGE2_D_SEARCH_DRAFT.md)。

### 有界搜索实现验证

`tools/build/test/build_stage2_d_search.ps1`在独立目录构建主机选型改动，复用并由正式构建器逐项核验原CUDA对象/依赖；默认基于`build_cuda_cmake/local_stage2_0_38_4`，可用`-BaseBuild`、`-Build`覆盖。

`tools/test/test_stage2_d_search.py --output data/experiments/<新目录>`执行共享搜索器CPU门禁。`test_stage2_d_search_gpu.py --exe <EXE> --save <已核验save> --output data/experiments/<新目录>`在GPU1重放保存点，检查自动D/承载、多尺度探索、后续轮次、采样与显式锁；`--exponent`、`--b2`、`--curves`可覆盖。它保存冻结EXE/DLL、构建输入和日志，属于功能验证，不代表新增ECM覆盖或性能提升。

`test_stage2_d_search_default_plans.py --exe <EXE> --reference <完整52格基准目录> --output data/experiments/<新目录>`逐格比较无tune的D/P，要求`curves_executed=0`。计划一致仍需结合已有完整曲线采集器验证性能。

完整曲线使用`bench_stage2_version.py`。`compare_stage2_d_search.py`分别审计两份完整矩阵并核对精确N/save、B1、sigma、B2和算术路径，输出逐格CSV、统计JSON及可选图。默认旧矩阵每格三次、新筛查每格一次，图注和统计保留两侧样本数；可用`--baseline-repeats`、`--candidate-repeats`显式修改，不从数据自动降低期望次数。

```powershell
python tools/bench/compare_stage2_d_search.py `
  --baseline data/experiments/stage2_v0_38_4_20261018/full_matrix `
  --candidate data/experiments/stage2_d_search_implementation_20261019/default_matrix_serial02 `
  --output data/benchmarks/stage2_d_search_0_39_0/default01 `
  --figures data/figures/stage2_d_search_0_39_0/default01
```

主指标为52格引擎完整时间的逐格中位数之和；n=1一侧就是原值。总耗时不增加才满足矩阵数值门槛；同时变慢超过5%和0.25秒的格另行交替A/B复测，可用`--retest-relative`、`--retest-seconds`指定预先约定的筛查门限。该门限只生成复测清单，不删除样本、不自动判定回退来源，也不替代矩阵总耗时要求。图中正值表示变慢，阴影为观测范围组合，不是置信区间。不同会话的温度和后台负载仍可能不同，不能由该图直接归因到D搜索改动。

`bench_stage2_d_default_pair.py --baseline <旧完整n=3矩阵> --candidate <新EXE> --candidate-version <版本> --output data/experiments/<新目录> --repeats 1`复用上述采集器，在同一会话按52个相同条件交替旧/新版本，合计104次。相邻格反转版本顺序，不预热；启动前校验旧EXE的原始结果指纹与新构建输入。每个版本目录保持标准矩阵格式，可分别用`check_stage2_version.py --expected-repeats 1`审计，再用同一比较工具指定两侧均为1。`pair.json`保留总调度与两侧身份；`--resume`只续跑已记录成功调用后的剩余格，遇到失败或未记录调用须先调查。配对结果补充历史n=3参考，不覆盖历史会话和原始失败样本。

比较该交替实验时，额外传`compare_stage2_d_search.py --paired-study <交替实验根目录>`；工具要求两侧恰为该目录的`baseline/`、`candidate/`，核对完整状态、EXE身份、版本、次数及保留调用顺序，图注改为同会话相邻版本。一格一次仍只作筛查，不估计方差；没有该选项时沿用独立历史会话口径。编排器的纯CPU检查为`tools/test/test_stage2_d_default_pair.py`，比较和实验身份检查为`test_stage2_d_search_comparison.py`，均需新的`--output data/experiments/<目录>`，不调用CUDA。

### 同入口的准备缓存对照

`bench_stage2_d_prepare.py --exe <EXE> --profile <已有运行统计的TOML> --reference <完整52格矩阵> --output data/experiments/<新目录>`从已核验输入选择实际余因子，在GPU1重放12条相同点的worktodo队列。`--exponent`、`--b2`、`--curves`覆盖任务；`--order carrier,uncached,on,on,uncached,carrier`比较默认承载、同一实测文件且关闭准备缓存、同一文件且开启缓存。`normal`为默认普通，`off`为保留已有实测选择但关闭继续采样，`on`开启正常运行采样；`--prepare-cache 0|1`控制其余模式的内部诊断开关。

默认要求性能文件未改写、已有10次观察、无试跑且所有实测文件组选择相同D/算术。需要研究学习阶段时显式`--allow-learning`，报告标为非冻结控制，不能与稳定状态混淆。逐队列保存全部曲线，包括首条冷启动及最后三条短队列；数学检查、精确N/B1/B2/sigma和完成进度仍核对。`report.json`分别给出引擎之和、`curve_done.wall`之和、外层进程观察墙钟及准备分项。前两项不包含退出轮询量化误差，原生规划时间嵌套在候选/默认准备内，不能再重复相加。重放仅用于性能/功能观察，不计作新增ECM覆盖；原始EXE、save及profile不改写。

`plot_stage2_d_prepare.py --study <稳定study> [--study <另一study>] --output data/benchmarks/<新目录> --figures data/figures/<新目录>`核对冻结EXE/编译输入、原始NF结果、计时闭合、性能文件身份及调用前后共享状态，再生成每曲线均摊时间的引擎/外层堆积图。黑点为完整队列；橙色是队列总时间减引擎，不是独立CPU核计时。只接受完整、冻结、无试跑的控制，不将学习数据混入；设备/预算检查对应此采集器的GPU1、arena6300/fold640/batch256 MiB。输出审计JSON、绘图脚本副本和PNG/SVG；已有目录不覆盖。图中位宽、B1/B2、版本与频率观测取自输入，首轮与尾部都保留。

`tools/test/test_stage2_d_prepare_plot.py --study <完整稳定study> --output data/experiments/<新目录>`在只读输入替身上检查错误记录拒绝，覆盖失败/学习状态、计时和几何不一致、非有限耗时；不改写原始实验，也不调用GPU。

## 压缩候选表提议生成

`compress_stage2_d_catalog.py`从指定的冻结候选头文件提取字面量，以保护集合、几何包络、NTT/树边界及默认owner布局的预算边界产生提议。不会改写生产头文件。`--protect-profile`仅采用干净且有正式重复的实测记录；`--protect-analysis`要求完整、零失败的矩阵。它们可重复指定。

```powershell
python tools/bench/compress_stage2_d_catalog.py `
  --input-header data/experiments/frozen_catalogue/sources/src/core/ecm_stage2_d_catalog.h `
  --protect-profile build_cuda_cmake/production_stage2/stage2_ecm_tune.toml `
  --protect-analysis data/benchmarks/my_verified_matrix_analysis.json `
  --output data/experiments/my_catalogue_compression
```

默认检查52个S、owner预算48/128/256/640/1024 MiB，可用`--bits`、`--fold-budgets-mb`覆盖。输出`ecm_stage2_d_catalog.h`提议、`retained.csv`、`removed.csv`、来源冻结副本和`analysis.json`。后者记录保护项、每个边界的P代表、输入/生成物SHA256及覆盖限制。表外保护项会报告为`excluded_protected`，不能忽略；自动候选域的扩充仍需审查。

按D/P筛选是覆盖启发式；完整时间还取决于尾树、点生成、NTT效率和联合显存。部署提议后须通过原生准入和完整矩阵，不允许以表大小或模型成本宣布性能验收。

## 交替性能矩阵

`bench_stage2_d_catalog.py`复用位宽扫描的collector，对两份冻结study交替运行相同9份余因子save和三档B2。两份study可用`bench_stage2_n_scaling.py --prepare-only --reference-study ... --cofactors-only`准备；源码、EXE、DLL和输入必须已冻结且哈希匹配。启动短校准关闭，普通余因子路径、D=0；GPU设置只读。

```powershell
python tools/bench/bench_stage2_d_catalog.py `
  --baseline-study data/experiments/catalogue_full_study `
  --candidate-study data/experiments/catalogue_compact_study `
  --output data/experiments/catalogue_paired_matrix --device 0 --repeats 1
```

`--profile FILE.toml`让两侧使用相同冻结tune文件，不训练或回写。省略时比较默认选型。`--exponents`、`--b2`可限制独立复测子矩阵；`--repeats`控制正式重复，9份输入的预热另存。`--resume`只续跑缺少记录的调用；已有未记录日志需先调查，不能覆盖。

两侧输出在`baseline/`和`candidate/`，格式与位宽扫描一致。用现有`analyze_stage2_n_scaling.py`核对原始哈希、输入和完整曲线，再以`--baseline-analysis`生成比较；`plot_stage2_n_scaling.py`读取这些审计结果绘图。主指标为`stage2_full_wall.total`，选型和进程时间另列。保留所有正式样本；明显回退另开复测目录，不删除初次样本。

## 原生选型与预算核对

`audit_stage2_d_catalog.py`对同两份冻结study比较默认和冻结tune的完整矩阵计划，并查询4个代表指数、5种arena/fold预算组合。每次要求`curves_executed=0`；不会训练或回写profile，不把查询用时作为曲线性能。

```powershell
python tools/bench/audit_stage2_d_catalog.py `
  --baseline-study data/experiments/catalogue_paired_matrix/baseline `
  --candidate-study data/experiments/catalogue_paired_matrix/candidate `
  --profile build_cuda_cmake/production_stage2/stage2_ecm_tune.toml `
  --output data/experiments/catalogue_native_audit --device 0
```

输出冻结工具、profile、逐次命令/日志与`report.json`；核对退出状态、D/P/I/G、请求程序、选择记录和内存路径。出现差异时保留证据并返回失败，先解释差异，再决定是否进入完整计时。相同规划不证明该预算下运行速度相同或不会OOM。
