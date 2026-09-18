# ECM Report Tools

从 [PrimeNet ECM Progress](https://www.mersenne.org/report_ecm/) HTML 报告导入 SQLite，并查询 / 可视化梅森数 ECM 进度。

依赖：Python 3.10+ 标准库；绘图还需 `matplotlib`、`numpy`。

默认数据库：`tools/ecm_report/ecm_progress.db`（可用 `--db` 或环境变量 `ECM_DB` 覆盖；`.db` 不进 git）。

## 快速流程

```bash
# 0. （可选）定时下载报告 → html/YYYY-MM-DD.html
python download_ecm.py --once          # 测一次
python download_ecm.py                 # 每 2 小时循环

# 1. 导入报告
python import_ecm.py html/2026-07-29.html
# 或批量导入 html\ 下全部文件（建议加 --force 以便非交互覆盖同日）
import_html.bat --force
# 或
python import_ecm.py report.html

# 2. 各 t-level 最小指数
python query_min_exponents.py

# 3. 两日进度差（默认次新 → 最新）
python query_changes.py

# 4. 密排柱状图（默认指数 1–20000）
python plot_ecm.py
```

## 数据模型（简要）

| 表 | 作用 |
|---|---|
| `exponents` | 主表：是否分解、**最新** `t_level` / `curves`、最近报告时间 |
| `progress_history` | **稀疏**历史：仅在 `t_level`/`curves` 相对上次快照变化时写入一行 |

- `factored_date`：仅在观测到从「无已知因子」迁到「有已知因子」时写入  
- 报告日取自页面 `Current time: YYYY-MM-DD … UTC` → `YYYYMMDD`  
- 某日状态（as-of）= `report_date <= D` 的最近一行；绘图默认读主表即可

---

## `download_ecm.py`

按间隔从 PrimeNet 下载 ECM Progress HTML，保存到 `html/YYYY-MM-DD.html`（日期取自页面 `Current time`；同日覆盖）。

```bash
python download_ecm.py [--lo 1] [--hi 100000] [--interval 2] [--once]
                       [--out-dir PATH] [--timeout 120]
```

| 参数 | 说明 |
|---|---|
| `--lo` / `--hi` | 指数范围（默认 `1`–`100000`），同时用于 `ecmnof_*` 与 `ecm_*` |
| `--interval` | 下载间隔（小时，默认 `2`） |
| `--once` | 只下载一次后退出 |
| `--out-dir` | 输出目录（默认 `tools/ecm_report/html`） |
| `--timeout` | HTTP 超时秒数（默认 `120`） |

请求 URL：

```text
https://www.mersenne.org/report_ecm/?txt=1&ecmnof_lo=LO&ecmnof_hi=HI&ecm_lo=LO&ecm_hi=HI
```

---

## `import_ecm.py`

解析 HTML 中两个梅森 `<pre>` 段，写入 / 更新数据库。  
`progress_history` 为稀疏存储：仅当相对上一次快照的 `t_level`/`curves` 变化时写入；导入前会压缩已有冗余行。

```bash
python import_ecm.py <report.html> [--db PATH] [--force]
```

| 参数 | 说明 |
|---|---|
| `html` | PrimeNet ECM Progress HTML 路径 |
| `--db` | SQLite 路径（默认 `ecm_progress.db`） |
| `--force` | 同日数据已存在时强制覆盖（非交互环境必加） |

同日冲突：TTY 下询问；非交互则退出并提示 `--force`。

批量导入可用同目录 `import_html.bat`（按文件名排序处理 `html\*.html`；可用环境变量 `ECM_HTML_DIR` 改目录）：

```bat
import_html.bat --force
import_html.bat --force --db ecm_progress.db
set ECM_HTML_DIR=D:\path\to\html
import_html.bat --force
```

---

## `query_min_exponents.py`

查询主表：每个 `t_level` 下，有因子 / 无因子的**最小指数**。

```bash
python query_min_exponents.py [--db PATH]
```

| 参数 | 说明 |
|---|---|
| `--db` | SQLite 路径 |

输出示例：

```text
Level	Exponents with known factors	Exponents with no known factors
75	N/A	1277
70	1213	1619
...
```

---

## `query_changes.py`

对比两个日期的进度（**as-of 稀疏历史**）：取各日 `report_date <= D` 的最近快照再比较。

```bash
python query_changes.py [--db PATH] [--from YYYYMMDD] [--to YYYYMMDD] [--all]
```

| 参数 | 说明 |
|---|---|
| `--db` | SQLite 路径 |
| `--from` | 起始日；省略时：若也无 `--to` 则为次新历史日，若仅有 `--to` 则为最早 |
| `--to` | 结束日；省略时为最新历史日 |
| `--all` | 额外列出 as-of 未变化的指数 |

默认：历史事件日中的**次新 → 最新**。

输出块：Changed（as-of 差异）、First seen、History events（窗口内实际写入行）、Newly factored。

---

## `plot_ecm.py`

密排柱状图：范围内每个库中指数占 1 像素宽（无空位）；  
纵轴 `y = (t_level - 5) + 5 × min(1, curves / curves_to_test)`（`t_level` 为当前档）。

```bash
python plot_ecm.py [--lo 1] [--hi 20000] [--db PATH] [--date YYYYMMDD]
                   [--color factored|level]
                   [--plots separate|merged|overlay|all|known|no_known]
                   [--xtick-gap 1000|auto] [--compress N] [--compress-agg avg|max]
                   [--ymin Y] [--ymax Y] [--fig-height INCHES] [--dpi N] [--vgrid]
                   [--out PATH] [--csv PATH]
```

| 参数 | 说明 |
|---|---|
| `--lo` / `--hi` | 指数范围（默认 `1`–`20000`） |
| `--db` | SQLite 路径 |
| `--date` | 可选 as-of 日（稀疏历史）；省略则用主表最新 |
| `--color` | `factored`（默认，按是否分解分色）或 `level`（按 t-level） |
| `--plots` | 输出哪些图（默认 `separate`）；`merged` 合并；`overlay` 见下；`all` 含 overlay；或 `known` / `no_known` |
| `--xtick-gap` | 横轴刻度：整数间隙（默认 `1000`）；`auto` 为像素均匀刻度 |
| `--compress` | 每像素合并 N 个连续指数（默认 `1`） |
| `--compress-agg` | 压缩桶内聚合：`avg`（默认）或 `max` |
| `--ymin` / `--ymax` | Y 轴起止（默认自动）；例如 `--ymin 40` |
| `--fig-height` | 图高度（英寸，默认见脚本）；**像素高度 ≈ `--fig-height` × `--dpi`** |
| `--dpi` | 输出 DPI（默认 `100`；约 1 柱 = 1 像素宽） |
| `--vgrid` | 在 **x 轴刻度位置**画竖虚线（与 `--xtick-gap` / `auto` 标记一致；默认关闭） |
| `--out` | 输出路径或文件名前缀；多图时自动加后缀 |
| `--csv` | 可选，将摘要表写入 CSV |

### `--plots overlay`

同图叠画，**横轴以 known（有因子）密排为基准**：

1. 每个 no_known 映射到**最近 known 像素**（同像素多个取 **max(y)**）  
2. 先画 known 柱 `[0, y_known]`（前景色）  
3. 仅当 `y_no > y_known` 时，在上方画超出段 `[y_known, y_no]`（后景色）  

### 调整 Y 轴起点 / 图高度（代码位置）

优先用命令行；若要改默认值，编辑 `plot_ecm.py`：

| 想改什么 | CLI | 代码位置 |
|---|---|---|
| Y 轴下限 | `--ymin 40` | `plot_bars()` / `plot_overlay()` 内 `ax.set_ylim(...)` |
| Y 轴上限 | `--ymax 75` | 同上 |
| 图高度（像素） | `--fig-height 8 --dpi 100` → 约 800px 高 | 顶部 `DEFAULT_FIG_HEIGHT` / `DEFAULT_DPI` |
| 图宽度 | （由柱数决定） | `fig_w = max(n / dpi, 4.0)` |

终端同时打印按 `t_level` 的摘要：`count / min_exp / max_exp / known / no_known / mean_progress`。

示例：

```bash
# 默认：两张分离图，刻度每 1000
python plot_ecm.py

# known 为 x 基准的叠画 + x 刻度竖线
python plot_ecm.py --plots overlay --vgrid --xtick-gap 1000

# 压缩用 max + overlay
python plot_ecm.py --plots overlay --compress 4 --compress-agg max

# Y 从 40 起，图高约 800 像素
python plot_ecm.py --plots overlay --ymin 40 --fig-height 8 --dpi 100
```

---

## 环境变量

| 变量 | 说明 |
|---|---|
| `ECM_DB` | 默认数据库路径（各脚本的 `--db` 均可再覆盖） |

## 样例文件

- `example/GIMPS ECM Progress - PrimeNet.html` — 导入用样例报告  
- `html/` — `download_ecm.py` 下载目录（gitignore，按日覆盖）  
- `report.html` — 本地工作用报告（可选）
