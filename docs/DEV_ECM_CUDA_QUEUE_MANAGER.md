# ECM-CUDA 队列管理器 + `ecm.ini` 开发文档

本文件描述 `ecm_cuda.exe`（以及共享同一 driver 的 OpenCL `ecm.exe`）内置的**工作队列管理器**与 **ini 配置化**改造的代码结构与开发指南。

---

## 1. 概述

`ecm_cuda` 在保留原「单次命令行」模式的基础上，新增了一个内置工作队列管理器，用于替代原先依赖外部 `work_manager.ps1` 的队列/日志/同步编排，并把原先散落在命令行里的运行参数收敛到 ini 配置。

两种运行模式：

| 触发条件 | 模式 |
|----------|------|
| **无位置参数**（即命令行里没有 `<B1>`） | 队列管理器模式：读 `ecm.ini` + `worktodo.txt`，循环处理直到队列为空 |
| **有位置参数** `<B1> [B2]` | 单次 CLI 模式（完全不变，忽略 ini） |

`-ini <path>` 可在队列模式下指定自定义 ini 路径（缺省为 exe 目录下的 `ecm.ini`）。

---

## 2. 模式分发（`src/core/ecm_driver.cpp`）

`main()` 的入口逻辑：

```
ecm_wants_usage()                          # 仅 -h/--help 打印用法
  └─ 解析 argv（含新增的 -ini <path>）
  └─ ecm_install_timestamped_iostreams()   # 时间戳输出
  └─ ecm_enable_console_ansi()             # Windows 控制台启用 ANSI
  └─ if pos.empty()   → run_queue_manager(ini_path)      # 队列模式
  └─ else             → 单次 CLI：读 stdin N → run_stage1_once(...)
```

核心重构是把原来 `main()` 里内联的 stage-1 执行逻辑抽成 `run_stage1_once()`，供 CLI 与队列两种模式复用。

---

## 3. 代码结构

### 新增

- `src/core/ecm_queue_config.{h,cpp}`
  `EcmQueueConfig` 结构、`key = value` ini 解析、默认模板生成（中英双语注释）。
- `src/core/ecm_worktodo.{h,cpp}`
  `ECMSTAGE2` 行解析、B1 提取、N 计算、worktodo 推进（成功移除 / 失败 `# ERROR`）、`.save` 同步。

### 修改

- `src/core/ecm_driver.cpp`
  `run_stage1_once()`、`run_queue_manager()`、`main()` 分发。
- `src/opencl_ecm_log.{h,cpp}`
  `screen.log` 镜像（tee）、进度条颜色、控制台 ANSI 启用。
- `kernels/cuda/cgbn_stage1.cu`
  ASCII 进度条（颜色 / ETA 平滑 / sigma 打印时机）。
- `src/opencl_ecm_stage1.cpp`
  OpenCL 侧 sigma 打印时机（与 CUDA 对齐）。
- `CMakeLists.txt`
  `ecm` 与 `ecm_cuda` 两个目标各增加两个新源文件。

---

## 4. 数据流（队列循环）

```
run_queue_manager()
  1. 解析 exe 目录；定位 ecm.ini（缺失则生成默认模板）
  2. ecm_queue_config_load() → EcmQueueConfig
  3. opencl_ecm_set_work_dir(exe_dir)      # 相对路径统一以 exe 目录为基准
  4. ecm_log_set_progress_color(cfg.progress_color)
  5. 打开 screen.log；ecm_log_set_mirror(logf)  # 时间戳输出 tee 到日志
  6. 启动时全量同步 .save
  7. 循环：
       a. ecm_worktodo_first_line()        读第一条任务行
       b. ecm_parse_stage2_line()          解析 ECMSTAGE2
       c. ecm_extract_b1_from_save_name()  提取 B1
       d. ecm_compute_stage2_n()           计算 N
       e. run_stage1_once()                执行 stage-1（内部 checkpoint 自动续跑）
       f. 成功 → finished 追加 + worktodo 移除
          失败 → worktodo 就地改写为 "# ERROR <原行>"
       g. 增量同步 .save
  8. 队列空 → 退出
```

---

## 5. 关键模块详解

### 5.1 `ecm_queue_config`（ini 配置）

- 格式：`key = value`，`#` 开头为注释，忽略空行；**手写解析，不引入第三方 ini 库**。
- `ecm_queue_config_write_default()` 生成带中英双语注释（英文在前）的默认模板。
- 键清单：

  | 键 | 默认 | 说明 |
  |----|------|------|
  | `worktodo` | `worktodo.txt` | 工作队列文件 |
  | `finished` | `worktodo.finished.txt` | 成功完成的任务追加到此 |
  | `save_sync_dir_1/2` | 空 | `.save` 同步目标目录（空 = 关闭） |
  | `sync_mode` | `incremental` | 每任务同步模式 `incremental \| full` |
  | `log_file` | `screen.log` | 日志文件（空 = 仅 stdout） |
  | `device` | `0` | GPU 设备索引 |
  | `ckpt_seconds` | `600` | GPU 检查点间隔（秒） |
  | `verbose` | `1` | 详细程度 |
  | `tpi` | `8` | 每实例线程数（OpenCL） |
  | `wg_size` | `0` | 显式工作组大小（0 = 自动） |
  | `kernel_mul/sqr/add/sub/special_mult` | 空 | OpenCL 算子路径覆盖（CUDA 忽略） |
  | `sigma` | `0` | 固定 sigma（0 = 随机） |
  | `save_name_pattern` | `m{n}_{b1}.save` | save_name 约定（文档/校验） |
  | `progress_color` | `cyan` | 进度条颜色 |

### 5.2 `ecm_worktodo`（队列解析）

任务行格式（与 `pipeline/ecm.py::to_stage2_line` 一致）：

```
ECMSTAGE2=[<aid>,]<k>,<b>,<n>,<c>,<save_name>,<B2>,<skip_curves>,<curves_to_run>[,"factors"]
```

- `ecm_parse_stage2_line()`：CSV 解析（支持引号与 `""` 转义）；`aid` 为可选的**非整数**首字段。
- `ecm_extract_b1_from_save_name()`：取 `save_name` 最后一个 `_` 到 `.save` 之间的 token，`strtod` 解析（兼容 `110e6`、`1e7`、`1000000`）。
- `ecm_compute_stage2_n()`：`N = k*b^n + c`，再逐个已知因子 `mpz_divexact`（任一不整除即报错）。
- `ecm_worktodo_advance()`：原子重写（tmp + `MoveFileExA` 替换）；成功移除 / 失败就地 `# ERROR`。
- `ecm_sync_save_files()`：Windows 用宽字符 API（`FindFirstFileW`/`CopyFileW`），正确支持 CJK 路径（如 `GIMPS_同步`）。

### 5.3 `run_stage1_once`（共享执行）

CLI 与队列两处复用：分配 `factors/array_found` 并返回给调用方；内部做 `params`、`batch_s`、`ecm_backend_prepare`、`ecm_backend_stage1`、save。

> 注意：**不在**这里打印 sigma。因为 sigma 可能被 checkpoint 覆盖，实际值由后端在 checkpoint 加载后打印（见 5.4）。

### 5.4 进度条与 sigma 打印（`kernels/cuda/cgbn_stage1.cu`）

- `print_progress()`：
  - `newline=false`（交互终端）：原地 `\r` 刷新，并加 ANSI 颜色（`progress_color` 可配）。
  - `newline=true`（重定向/日志）：走 `outputf` 输出**整行**（带时间戳、进 `screen.log`、不带 ANSI）。
- `remaining`：维护**最近 50 个 batch 的速度环形缓冲**，`remaining = 剩余位数 / 平均速度`，不随 checkpoint 续跑起点漂移、也消除单批速度抖动。
- sigma 打印时机：在 checkpoint 加载**之后**打印实际 sigma，并标注 `[restored from checkpoint]` / `[computed]`，避免误导。
- kernel 信息行：`GPU: CGBN<TPI, BITS> kernel, N is <nbits> bits (...)`，TPI 在前。

### 5.5 日志镜像（`src/opencl_ecm_log.cpp`）

- `ecm_log_set_mirror(FILE*)`：所有带时间戳的输出（C++ 流 + `ecm_ts_vfprintf`/`outputf`）额外写入镜像文件。
- `ecm_log_set_progress_color()` / `ecm_log_progress_color_code()` / `ecm_log_progress_color_reset()`：颜色名 → ANSI 码。
- `ecm_enable_console_ansi()`：Windows 上启用 `ENABLE_VIRTUAL_TERMINAL_PROCESSING`。

---

## 6. 关键设计决策

- **finished 只放成功项**；失败任务留在 worktodo 并就地标记 `# ERROR`，不污染 finished。
- **致命错误才中止队列**：GPU 初始化失败、ini 无法生成、worktodo 路径非法；其余任务级错误只跳过当前行。
- **B1 不单独存字段**：由 `save_name` 提取，与 `pipeline/ecm.py` 的生成约定一致。
- **B2 / skip_curves 忽略**：它们是给 prime95 stage-2 消费者的，本程序只做 stage-1。
- **相对路径以 exe 目录为基准**：双击启动行为确定，不依赖 CWD。
- **颜色只进终端不进日志**：避免 ANSI 转义污染 `screen.log`。
- **不引入第三方库**：ini 手写解析、进度条纯 ASCII，保持零额外依赖。

---

## 7. 扩展指南

- **新增 ini 键**：`EcmQueueConfig` 加字段 → `ecm_queue_config_load()` 加分支 → 默认模板加一行 → `run_queue_manager()` 里消费。
- **新增任务行字段**：改 `EcmStage2Task` + `ecm_parse_stage2_line()` 的列索引。
- **新增后端行为**：通过 `include/ecm_backend.h` 的 seam 扩展；队列管理器与后端无关，对 CUDA / OpenCL 通用。
- **进度条样式/颜色**：`print_progress()` 内改条形宽度与格式；颜色集合在 `ecm_log_set_progress_color()` 里扩展。
