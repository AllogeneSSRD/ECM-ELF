# 梅森因子数据集

工具位于 `tools/ecm_dataset/`，默认数据库为该目录的 `ecm_stage2_dataset.sqlite`，不提交 Git。默认路径按脚本位置解析，显式相对路径按工作目录解析。

## 持久数据

只保留 `mersennes` 与 `factors` 两张核心表：原梅森指数、因子/重数与素性状态，以及每因子的一个 sigma、B1/B2、群阶/点阶及其分解。没有每次扫描、导入或运行的 observations/frontier/history 表。

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

Python 使用标准库；计算群阶/点阶及拆解因子需要 PARI/GP。显式 `--gp <path/name>` 优先；省略或仅传 `--gp` 时从 PATH 查找 gp.exe/gp。显式无效路径报错，不另选程序。

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
