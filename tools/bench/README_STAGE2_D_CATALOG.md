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
