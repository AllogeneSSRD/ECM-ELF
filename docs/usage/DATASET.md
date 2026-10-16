# 梅森因子数据集

工具位于 `tools/ecm_dataset/`，默认数据库为该目录的 `ecm_stage2_dataset.sqlite`，不提交 Git。默认路径按脚本位置解析，显式相对路径按工作目录解析。

## 持久数据

只保留 `mersennes` 与 `factors` 两张核心表，当前模型为Montgomery PARAM0：原梅森指数、因子/重数与素性状态，以及每因子的一个 sigma、B1/B2、群阶/点阶及其分解。没有每次扫描、导入或运行的 observations/frontier/history 表。Edwards使用下文独立分析入口，输出带模型身份的JSON/CSV，不写入这两表。

普通 analyze/ingest 保留当前要求的支配替换：新 B1≤旧 B1、新 B2≤旧 B2，且至少一项严格减少。界限相等不替换；失败/未胜出的 sigma 不持久化。所有保存比较使用 lcm，choose12 只是由点阶导出的查看口径。

B2=0 表示该点 Stage1 已足够。成功条件应根据点阶而非仅群阶推导；单素数 Stage2 界限要保留素数幂约束。目录整除检查不等于素性证明。

## 操作入口

| 脚本 | 用途 |
| --- | --- |
| `ecm_dataset.py init/summary/analyze/ingest` | 导入目录、统计、计算阶、导入 Stage2 result |
| `scan_sigma.py` | 按 exponent/sigma 闭区间扫描已有因子 |
| `prepare_dataset.py` | 批量准备因子和 sigma 的实验候选 |
| `run_production_dataset.py` | 先生成计划，再运行 Stage1/Stage2 正负对照 |
| `export_dataset.py` | 只读导出核心表、CSV 和身份 |
| `migrate_dataset.py` | 将受支持旧 schema 转为当前核心数据 |
| `dataset.py` / `param0_order.gp` | 共享存储、界限、GP 模型和分解 |
| `edwards_order.py` / `edwards_order.gp` | Atkin-Morain Edwards群阶、点阶、分解及48LCM界限，独立输出 |

Python 使用标准库；计算群阶/点阶及拆解因子需要 PARI/GP。显式 `--gp <path/name>` 优先；省略或仅传 `--gp` 时从 PATH 查找 gp.exe/gp。显式无效路径报错，不另选程序。

## Edwards群阶与点阶

[独立入口](../../tools/ecm_dataset/edwards_order.py)接收已知素因子p和完整uint64 sigma，按[共享C++构造](../../src/core/ecm_edwards_curve.h)重建Atkin-Morain曲线。GP先证明p素性、排除退化并核对基点，再通过等价Weierstrass模型计算群阶/点阶、完整分解及各因子的素性。Python核对Hasse界、分解乘积与点阶整除关系。数学映射及最小点阶验证见[固定参考](../reference/ECM_EDWARDS_MATH.md#41-在有限素域计算群阶与点阶)。

```powershell
python tools/ecm_dataset/edwards_order.py `
  --prime 3740809188848298874514458320563758983497238991 `
  --sigma 320289107620271 --exponent 10937 `
  --output data/experiments/edwards_order_single

python tools/ecm_dataset/edwards_order.py `
  --fixtures tools/ecm_dataset/fixtures/edwards_mersenne_ca.json `
  --output data/experiments/edwards_order_reference
```

输出目录必须不存在。`--exponent`可选，仅检查p整除 `2^exponent−1`，没有数据库的9999上限；p必须是已知素因子，不能把未分解的复合梅森数作为有限域模数。`--timeout`默认每条曲线120秒，`--stack`默认128M。完整GP脚本可在GP中直接读取后调用 `ecm_edwards_order(p,sigma)`。

每条曲线保留输入脚本、stdout/stderr及摘要；顶层保留GP版本、exe/libpari摘要和脚本/fixture快照，汇总为 `summary.json`、`orders.csv`。`complete`表示所有计算和内部检查完成，外部GO是否匹配单列为 `reported_relation`、`reported_relations` 和 `all_reported_group_orders_match`；不把外部样本差异掩盖成通过。`point_order`是原始基点阶，`order_after_48`为其除以与48的GCD；B1/B2前沿按48LCM口径和单剩余素数推导，B2=0表示仅Stage1即可消去。数学界限不代表运行时整数范围内均可执行。

用户提供的17行mersenne.ca数据在PARI/GP 2.17.3下逐行核对，15行完整群阶一致；原表和配对顺序原样保留在fixture，未将其视作独立下载的站点快照。两处差异为：

- M13883，p=`190998088435897266796075481606147454353399`、sigma=`4674866307349397`：完整群阶是 `190998088435897266796654466340963036240192`，为给定乘积的3倍；分解中应为 `3³`。给定乘积既不满足该p的Hasse界，也不是该初始点的精确阶。
- M15173，sigma=`1038876698`：给定分解精确复现Montgomery PARAM3的群阶 `5347934906960045728053670085517242297419632`。同sigma的Edwards群阶为 `5347934906960045728048506633196338632580320 = 2⁵·5·23·37·39276842736193050294128280208551253177`，不能将该行作为Edwards群阶的正对照。

原始证据为 `data/experiments/edwards_group_order_20261016/reference_03/`，C++构造及独立点阶验证为同根的 `validation_02/`，CLI检查为 `cli_01/`。17组的d和基点坐标全部与实际C++构造一致；24组小素域组合中22组有效并通过独立逐点计数，2组被双方判退化；另验证6种无效输入和5种缺失证据。汇总为 `data/benchmarks/edwards_group_order_20261016/qualification_01/summary.json`，SHA256=`34970b36e80673744736847c3bba4a0f9635e85859afd7a7bc39c29ce731a1f8`，同目录 `orders.csv` 保留17行结果。可使用 `tools/build/test/build_edwards_order_probe.ps1` 构建CPU探针，再运行 `tools/test/test_edwards_group_order.py --reference <完整fixture输出> --probe <探针exe> --output <新证据目录>` 复核。

主程序的 `--go` 尚未接入此Edwards模型；现有Montgomery诊断不能作为Edwards群阶证据。当前入口也不提供Edwards数据库存储、sigma扫描或生产正负对照的自动执行。

## 扫描策略

扫描参数为 `--exponent-range <first:last>`、`--sigma-range <first:last>`，端点包含，可只传单值。exponent 范围 1…9999，sigma 为 6…2⁶⁴−1。只扫描已入库因子；因子十进制位数可由 min/max 选项限制，每次 GP 调用有 timeout。

扫描脚本提供显式策略，与普通导入的支配规则分开：

- `normal`（扫描默认）：按 5000·B1+max(B1,B2) 严格降低选择；若 B1/B2 没有同时严格降低，还要求 B2<100000·B1。
- `strict`：B1、B2 都严格降低才替换。
- `either`：至少一项严格降低；若不是两项都严格降低，还要求 B2<100000·B1。可能依赖扫描顺序，不保证两项单调。

三种扫描策略首次保存均要求比例上限。日常 analyze/ingest 不使用这些扫描策略，继续采用双项不增且至少一项减小。每次改善立即提交，Ctrl+C 保留已提交数据；没有扫描进度表，重跑被丢弃 sigma 可能再次计算。

## Result 与生产验证

ingest 验证实际 N 整除原梅森数、raw factor 是 proper divisor、算术无 bad factor；拆解 raw composite 后，为 proven prime 分析 sigma。余因子结果需明确原 exponent，不能仅从位宽推断。

生产 runner 先准备 CPU 参考保存点，实际 Stage1 的归一化 X/checksum 必须一致，再执行 Stage2。负对照只要求目标因子不出现，不要求无其他因子。runner 当前限制 GPU1，日志/save/结果/冻结身份写到实验目录，运行完成后另调用 ingest 才更新数据库。

详细参数和调用入口见 [工具 README](../../tools/ecm_dataset/README.md)；实际替换依据为 [store_best](../../tools/ecm_dataset/dataset.py#L163)，评分为 [bound_score](../../tools/ecm_dataset/dataset.py#L157)。
