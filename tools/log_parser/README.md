# ECM 日志解析器 (ECM Log Parser)

一个可视化的 Prime95 / mprime worker 日志解析器。上传日志文件，自动将每一次 ECM 运行解析为一行数据，支持筛选、时间线可视化，并可导出 XLSX / CSV。

正在运行的CPU ECM性能实验可使用独立命令行入口`analyze_prime95_ecm.py`：读取日志/worktodo快照，导出逐曲线、分组和详细阶段CSV/JSON，区分提前因子与未完成记录。无需Flask依赖，见[性能分析说明](README_PRIME95_ECM_BENCH.md)。

## 功能

- **网页上传** `.log` / `.txt` 文件，后端 (Flask + Python) 解析。
- **运行分段**：每一条 `ECM on M...: Edwards/Montgomery curve #N` 开启一次运行，支持`[Worker]`和`[Worker #N]`，直到该worker的下一条`ECM on`（或文件结尾）为止。网页使用以下基础状态；性能命令行另有更严格的完成判定：
  - `complete` — 完整跑完 Stage 1 & Stage 2
  - `stage1-only` — 只完成 Stage 1
  - `interrupted` — Stage 1 未完成
- **提取字段**：Worker、指数、Curve#、s、B1、Actual B2、Worth、Available/Using 内存、Stage 1 时间、Stage 2 init/complete/GCD 时间、S1 FFT、S1 FFT type、S2 FFT、S2 FFT type、开始/结束时间。
  - FFT 追踪：每个 worker 维护一个“当前 FFT”，由 `Using` / `Switching to` / `Switching back to` 三种行更新（FFT 类型支持 `AVX-512`、`FMA3` 等含连字符的写法）。
  - S1 FFT = ECM 行处 worker 的当前 FFT（worker 首次运行某指数、或重启恢复的首次才会打印一次 `Using`；同一指数后续 curve 无 `Using`，会沿用上一次的当前 FFT）。
  - S2 FFT = Stage 1 完成后到 Stage 2 之间的 `Switching to ... FFT length`；若 S1 FFT == S2 FFT，程序不会打印 `Switching to`，此时 S2 FFT/type 自动沿用 S1。
  - S1 与 S2 的 FFT type 可能不同，故分列显示。
- **筛选**：Worker 序号 / 指数 / Curve# / 状态（多选），B1 / Available mem / Using mem（区间），日期范围（按开始时间）。
- **可视化**：汇总统计卡片、按 Worker 分组的 Gantt 时间线（Stage 1 蓝 / Stage 2 橙）、可排序的数据表格。
- **导出**：XLSX（openpyxl）或 CSV，可选“当前筛选”或“全部”。

## 运行

```bash
conda activate web
cd D:\code\MPA-OpenCl\tools\log_parser
pip install -r requirements.txt   # 首次运行
python app.py
```

然后浏览器打开 http://127.0.0.1:8000 （如需换端口：`set PORT=8080` 后再运行）。

## 命令行快速验证解析器

```bash
python parser.py ..\screen_example.log
```

## 文件结构

```
log_parser/
├── app.py              # Flask 服务与导出
├── parser.py           # 日志解析核心 + 列定义
├── analyze_prime95_ecm.py # CPU性能快照与CSV/JSON导出
├── compare_prime95_gpu.py # 恢复精确N并关联历史GPU测量
├── plot_prime95_gpu.py    # CPU/GPU对比PNG/SVG与可选Canvas
├── README_PRIME95_ECM_BENCH.md # 性能统计口径和使用说明
├── requirements.txt
├── templates/index.html
└── static/
    ├── app.js          # 前端逻辑（筛选/表格/Gantt/导出）
    └── style.css
```
