# Stage2 D 候选表工具

生产、tune及Python规划工具的唯一自动D来源是 [ecm_stage2_d_catalog.h](../../src/core/ecm_stage2_d_catalog.h)。当前2080项D按数值升序、去重维护，P=φ(D)/2统一派生；`default_plateau`标记既有保守平台保护的14项资格范围，不构成独立表。

当前运行与搜索规则见 [Auto B2 / tune](../../docs/architecture/AUTO_B2.md)，来源研究、冻结证据和待验收事项见 [D候选表草稿](../../docs/development/STAGE2_D_CATALOG_DRAFT.md)。Prime95数学依据见 [固定参考](../../docs/reference/PRIME95_STAGE2.md#3-d-候选表的含义与适配条件)。

## 完整379419项枚举

[本机参考文本](../../data/experiments/stage2_d_catalog_implementation_20261011/D_47smooth_379419.txt)包含379419条D/P数据，按D升序排列。列为D、P=φ(D)/2、是否满足偶数D≥6。7,198,077 bytes，SHA256为`fb616e036324e25734632b8d4b5121b004f463c7aa2126db20bdea21847d68e1`；同目录JSON保留数量和生成工具身份。

枚举域为≤200000000的47-smooth D、非零P，包含奇数；282395项满足portable偶数条件。上限包含D=200000000，严格小于上限时少1项。2080项共享表是多来源并集，其中14项含53或59，不能解释为参考枚举的纯子集。参考文本最后一列是偶数D≥6的筛选标记；头文件中的布尔字段则是`default_plateau`策略标记，false不表示奇数。数学说明见 [D参数数学](../../docs/reference/ECM_STAGE2_D_PARAMETERS.md)，精确来源计数见 [当前生成规则](../../docs/architecture/AUTO_B2.md#来源筛选与规模)。它用于检查覆盖和来源，不在生产或tune中重新枚举。文件保存在Git忽略目录，分发源码时不包含本机数据。

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

候选数量不等于完整曲线测量数量，长度相符不证明整条曲线显存适配。默认性能保护保持配对中位数/MAD门限和现场原生准入；当前新候选域、区间重启仍待完整矩阵性能验收。前期2080项草案/24项原生计划的冻结身份见草稿，它们不是当前二进制的运行验证。
