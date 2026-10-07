# ECM Stage2 因子数据集工具

这些脚本用于建立和维护梅森数因子测试集，为 Auto B2 提供可复查的正确性数据：

1. 从本地因子表导入 `M=2^exponent−1` 及已知因子。
2. 对指定因子和 sigma，计算 PARAM0 曲线的完整群阶、初始点阶及分解。
3. 推导标准单素数 Stage2 模型下的 B1/B2 界限。
4. 生成有效 Stage1 保存点，调用生产程序进行正对照和目标负对照。
5. 导入主程序 result，拆解复合因子，仅在界限更优时更新保留的 sigma。

完整数学合同、实验结果和源码位置见 [实现报告](D:/code/MPA-OpenCl/docs/STAGE2_FACTOR_DATASET.md)。这些工具尚不负责自动选择生产 B2。

## 1. 环境与文件位置

- Python 3，仅使用标准库；SQLite 不需要单独安装或启动数据库服务。
- PARI/GP：计算群阶、点阶和拆解因子时需要。显式 `--gp PATH` 优先；省略该选项或只传 `--gp` 时，从环境变量 PATH 查找 `gp.exe` / `gp`，与Stage1一致。
- 生产测试另需 CUDA Stage1 和 Stage2 程序及可用 GPU。当前 runner 固定使用 **GPU1**，不接受 GPU0。

本目录文件：

| 文件 | 作用 |
| --- | --- |
| `ecm_stage2_dataset.sqlite` | 默认工作数据库，由原 `data/` 迁入 |
| `ecm_dataset.py` | 日常 CLI：初始化、分析、导入 result、统计、导出因子 CSV |
| `dataset.py` | 公共库：两表 schema、HTML 解析、GP 调用、即时界限计算和最优 sigma 更新 |
| `migrate_dataset.py` | 从旧七表迁移到两表，保留已选最优 sigma并回收空间 |
| `param0_order.gp` | Suyama PARAM0 模型、群阶/点阶分解、素性及最小点阶检查 |
| `prepare_dataset.py` | 按指数、因子位数、sigma 批量计算阶，准备实验候选 |
| `scan_sigma.py` | 对指定exponent/sigma闭区间扫描目录因子，只保存最优sigma |
| `run_production_dataset.py` | 先准备实验计划，再运行实际 Stage1/Stage2 并记录生产证据 |
| `export_dataset.py` | 只读导出两张核心表、因子 CSV 和指纹清单 |

以下命令均在 **仓库根目录 `D:\code\MPA-OpenCl`** 执行。默认数据库路径按脚本所在仓库定位，不随当前工作目录变化；用户指定的相对路径则相对于当前工作目录。

```powershell
Set-Location D:\code\MPA-OpenCl
$gp = 'D:\AppData\Pari64-2-17-3\gp.exe'
```

GP已在PATH中时，无需重复指定完整路径；也支持只写 `--gp`：

```powershell
$env:PATH = 'D:\AppData\Pari64-2-17-3;' + $env:PATH
python tools/ecm_dataset/scan_sigma.py --exponent-range 223 --sigma-range 6:26 --max-factor-digits 17 --gp
```

`--gp gp.exe` 等裸命令名同样按PATH解析。显式路径会清理多余引号；显式路径无效时直接报错，不回退到另一个GP。未配置任何可用GP时提示添加其目录到PATH或传入完整路径。这里没有专用的 `GP_PATH` / `GP_BIN` 环境变量约定。

原目录中的历史快照 `data/ecm_stage2_snapshot_20261005/` 保留，用于追溯旧实验；它不是脚本当前写入的数据库。

## 2. 日常使用：初始化、分析和导入

### 初始化已知因子

```powershell
python tools/ecm_dataset/ecm_dataset.py init
python tools/ecm_dataset/ecm_dataset.py summary
```

默认输入 `.refactor/Mersenne_exponent_factor_1-9999.html`。也可指定其他相同格式的本地文件：

```powershell
python tools/ecm_dataset/ecm_dataset.py init --html D:\datasets\Mersenne_exponent_factor_1-9999.html
```

重复导入不会重复添加相同指数/因子。每个目录因子都检查 `2^exponent mod factor = 1`；这项检查不代表已经证明素性。新因子的 sigma、B1/B2、群阶和点阶字段初始为 NULL。

### 分析某个因子的多个 sigma

```powershell
python tools/ecm_dataset/ecm_dataset.py analyze `
  --exponent 223 --factor 196687 --sigma 6 9 26 --gp $gp
```

输出每个候选 sigma 的临时分析及最终保留的最佳记录；不保存其他 sigma。最优比较统一使用普通 `lcm(1..B1)`。指定 `--torsion 12` 可查看 choose12 界限，只改变输出界限，不建立另一份 sigma 记录：

```powershell
python tools/ecm_dataset/ecm_dataset.py analyze `
  --exponent 223 --factor 196687 --sigma 9 --torsion 12 --gp $gp --timeout 60
```

当前保留的 sigma 可以复用已存阶分解；`--retry` 强制重算它。其他 sigma 及错误/超时不缓存，下次调用会重新计算。每条 GP 调用默认超时30秒。当前只支持 PARAM0，sigma 范围为6～2^64−1，原梅森指数范围1～9999。

### 导入 Stage2 的 result JSONL

```powershell
python tools/ecm_dataset/ecm_dataset.py ingest `
  --results run/results.jsonl --gp $gp
```

结果为完整梅森数时，从 `N_hex` 推断指数。若程序处理的是剥离部分因子后的余因子，必须传入原梅森指数（或 result 已含 `mersenne_exponent`）：

```powershell
python tools/ecm_dataset/ecm_dataset.py ingest `
  --results run/cofactor_results.jsonl --exponent 2657 --gp $gp --timeout 60
```

脚本逐行读取 result，验证实际 N 整除原梅森数、raw factor 是 proper divisor、`bad_factors=0`，拆解每个 raw composite，为每个 proven prime 分析当前 sigma。它读取 `factors`，不因 `hits=0` 忽略有效因子；无因子运行直接跳过，不写数据库。

`--torsion` 默认1，设置12可推导该点的 choose12 界限。数据库内保存及比较的 B1/B2 始终采用 lcm 口径，不与运行 result 的 B1/B2 混用。

导入只输出本次计数：`updated`、`unchanged`、`unresolved`、`analysis_errors` 等，不保存时间戳、去重键或原始 result。重复导入可能重复计算已丢弃的 sigma，但不会增加历史行；相等界限不替换。GP 未完成时可修复环境或增加超时后重新导入。

### 使用另一个数据库

`ecm_dataset.py` 的 `--db` 必须放在子命令之前：

```powershell
python tools/ecm_dataset/ecm_dataset.py --db run/custom.sqlite init
python tools/ecm_dataset/ecm_dataset.py --db run/custom.sqlite summary
```

其他脚本也支持 `--db PATH`，但没有子命令。实验可使用独立数据库，避免修改默认语料。

## 3. 批量准备群阶候选

### 按 exponent / sigma 范围扫描

```powershell
python tools/ecm_dataset/scan_sigma.py --exponent-range 200:1000 --sigma-range 6:100 `
  --policy normal --max-factor-digits 17 --gp $gp
```

两个区间的起止值都包含在内；也可传单值，例如 `--exponent-range 223`。允许exponent 1～9999、sigma 6～2^64−1。只扫描数据库已有的因子，先执行 `init` 导入目录。

默认扫描范围内全部因子。建议首次显式限制 `--max-factor-digits 17`，避免大因子的群阶计算耗时过长。`--min-factor-digits` 默认1；`--timeout` 默认30秒，是每个因子/sigma的GP调用超时。自定义数据库用 `--db PATH`。数据库启用 WAL，连接遇到锁最多等待一小时；扫描会在计算前取完待处理因子，避免长时间持有读取游标。

```powershell
python tools/ecm_dataset/scan_sigma.py --db run/custom.sqlite `
  --exponent-range 223:431 --sigma-range 6:26 --min-factor-digits 5 `
  --max-factor-digits 17 --timeout 60 --progress-seconds 5 --gp $gp
```

逐因子、逐sigma升序扫描；`--policy` 可选择：

- `normal`（默认）：比较分数 `B1 × max(B1, B2)`，只在分数严格减小时替换。普通 Stage2 候选即比较 `B1 × B2`；数据库中表示 Stage1-only 的 `B2=0` 按 `B1²` 计。若只有一项减小，还须满足 `B2 < 100000 × B1`。
- `strict`：只有 B1、B2 都严格减小时才替换，不要求比例界限。当前 B2 已为 0 时不会再替换。
- `either`：沿用旧扫描规则，只要任一项严格减小即可替换；两项都减小时不要求比例界限，否则须满足 `B2 < 100000 × B1`。连续使用此规则可能使最终两项都比早期记录大。

首次保存时，三个模式都要求 `B2 < 100000 × B1`。每次改善立即提交，不保存扫描历史或失败候选。控制台输出预计任务数、已处理数量、更新/不变/缓存/失败/超时计数及进度；进度在GP调用完成后输出。所选模式包含在开始事件中。结果可能依赖扫描顺序，不宣称是耗时意义下的全局最优。

Ctrl+C中断时已提交的最优记录仍保留。没有自动扫描进度表；继续时指定剩余范围，或重跑原范围（不增历史行，但丢弃的sigma会重新计算）。脚本不运行GPU。

### 按典型因子位数准备生产实验

```powershell
python tools/ecm_dataset/prepare_dataset.py --output run/orders --gp $gp
```

默认指数：223、431、1367、2657、4933、6977、8171；默认 sigma：6～17及26。每个指数选择接近6、12、17十进制位的目录因子，因子选择上限17位，去重后批量分析。数据库中没有合适因子时不会凭空生成候选。

```powershell
python tools/ecm_dataset/prepare_dataset.py --output run/orders_custom --gp $gp `
  --exponents 223 431 --sigmas 6 9 26 --max-factor-digits 17 --timeout 90
```

`--timeout` 在这个脚本中是 **每个指数 GP 批次** 的超时，不是每个 sigma。输出包含 GP 脚本、stdout/stderr、`preparation.json`；分析也写入数据库。该阶段只做 CPU/GP 准备，不运行 GPU 曲线。

## 4. 使用生产程序进行因子测试

### 先生成计划，再执行

先完成目录导入和批量分析。选择一个新输出目录：

```powershell
python tools/ecm_dataset/run_production_dataset.py --prepare-only --output run/production
python tools/ecm_dataset/run_production_dataset.py --output run/production
```

准备阶段生成 `plan.json`，选取 Stage1 能正常完成且目标留给 Stage2 的候选，计算 CPU 参考保存点。执行阶段实际运行 GPU Stage1，核对 normalized X/checksum，再运行正对照和目标负对照。

默认程序路径：

```text
build_cuda_cmake/ecm_cuda.exe
build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe
```

可用 `--stage1 PATH` / `--stage2 PATH` 指定其他构建；运行时使用私有 INI 和输出目录。输出保存点、日志、result JSONL、`runs.json`、`provenance.json`，运行记录只留在指定实验目录，不写数据库。runner 只使用保留的最优 sigma，若无法形成指定正/负对照则报告无候选，不恢复已丢弃的 sigma。**运行后还需用 `ingest` 导入各条 result，才能更新因子分析和最佳 sigma。**

负对照要求指定的目标因子不出现，不要求所有其他因子都不出现。当前计划使用 `B2_target−4D`，不是对实现最小请求 B2 的精确搜索。

### 允许先剥离 Stage1 已找到的因子

```powershell
python tools/ecm_dataset/run_production_dataset.py --prepare-only --allow-cofactors `
  --exponents 2657 4933 6977 --output run/cofactors
python tools/ecm_dataset/run_production_dataset.py --output run/cofactors
```

runner 会确认实际 Stage1 GCD 与准备结果一致，剥离后针对新的模数重新生成 Stage1 保存点，不会把原 N 的点直接用于余因子。导入余因子 result 时指定原 `--exponent`。

### 原生复合因子拆解与断点继续

对于支持新接口的 Stage2 构建，可以添加：

```powershell
python tools/ecm_dataset/run_production_dataset.py --output run/new_candidate `
  --stage2 build_cuda_cmake/_factor_dataset_20261005/native/ecm_cuda_stage2.exe `
  --factorize-hits --gp $gp
```

该目录必须已有由 `--prepare-only` 生成的计划。原生可选拆解附带素数、重数和 GP 证明状态；离线 `ingest` 仍会验证并进行群阶分析。

中断后以同一计划、runner 和二进制添加 `--resume`。脚本拒绝用不同二进制续跑。改脚本（包括本次默认数据库路径迁移）也会改变准备工具指纹；旧计划请用对应旧脚本继续，或在新目录重新准备，不要修改计划中的SHA绕过检查。

## 5. 数据含义与“更优”的规则

- `digital` 是十进制位数，不是 bit length；大整数在 SQLite 中以 TEXT 保存。
- `group_order` 是整个曲线群阶，`point_order` 是当前 sigma 初始点的精确阶；B1/B2 界限由点阶推导。
- 数据库 `(B1, B2=0)` 表示 **Stage1-only**。这不是原生程序的零 B2 配置/Auto B2 语义，不应直接转为 Stage2 命令。
- 每个因子只保留一个 sigma。普通导入中，新 B1、B2 都不大于旧值且至少一个严格变小时才替换；相等、变差或一优一劣均保留旧记录。`scan_sigma.py` 使用上面选择的扫描模式。首次从候选可行界限中选一对，扫描时先过滤 B2/B1 比例。
- 最优比较使用 lcm；choose12 界限从保留的点阶即时计算，不单独持久化。
- 推导界限针对标准单素数 semismooth 模型。引擎扫描尾部及退化点可能暴露额外因子，实际运行 B1/B2 仅留在程序 result 中，不写入核心数据库。
- 保存完整正/负结果不等于无偏随机曲线样本；不能直接用这批精心选择的曲线估计成功概率或决定生产 Auto B2。

数据库只有两表：

- `mersennes`：指数、表达式、梅森数的十进制位数。
- `factors`：因子、十进制位数、素性状态、最优 sigma、B1/B2、对应群阶/点阶和完整分解。

无来源记录、分析历史、frontier或运行记录。空间随因子数增长，不随扫描曲线数增长。

## 6. 导出和审计

仅导出因子表 CSV：

```powershell
python tools/ecm_dataset/ecm_dataset.py export --output run/factors.csv
```

导出全部表和 SHA256 清单：

```powershell
python tools/ecm_dataset/export_dataset.py --output run/snapshot
```

`export_dataset.py` 只读数据库，要求输出目录不存在或为空。只导出核心两表与CSV，不再提供 `--evidence-root`；旧实验快照可以继续单独查看。

本机独立审计入口：

```powershell
python tools/test/audit_ecm_factor_dataset.py --output run/audit.json --gp $gp
```

审计涵盖核心因子的整除关系、所保留群阶/点阶的分解乘积、Hasse界、独立 x-only 点阶检查和素性。不再读取历史运行表或要求本机 result/log/save 存在。

## 7. 从旧数据库迁移

旧七表数据库不能直接使用新脚本；执行一次：

```powershell
python tools/ecm_dataset/migrate_dataset.py
# 自定义数据库
python tools/ecm_dataset/migrate_dataset.py --db run/legacy.sqlite
```

迁移保留所有梅森数、因子及当前已选最优 sigma 的核心字段；删除 `sources`、`analyses`、`frontier`、`observations`、`production_runs` 及因子中的来源/历史ID。不会从已丢弃的 sigma 中重新排名。事务内逐项核对核心值完全一致，提交后 VACUUM 回收历史数据页面。

迁移前通过SQLite备份一次到 `数据库文件名-before-core-v2.bak`，已有备份不会覆盖。完成后重复执行不会再次备份或删除。数据库查看器应先关闭；旧备份只用于恢复，不会被新脚本读写或继续增长。

默认数据库和该备份沿用仓库忽略规则，不自动加入Git。历史快照保持原样；新导出只包含核心数据。
