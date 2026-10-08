# ECM / ECM-CUDA 图形前端（`ecm_gui`）开发文档

`ecm_gui` 是 `ecm.exe`（OpenCL）与 `ecm_cuda.exe`（CUDA/CGBN）的原生图形前端：管理多个 worker 进程、
每个 worker 一个输出小窗、监控 GPU（占用/频率/功耗），配置全部落在 `ecm.ini` 的 `[GUI]` 与
`[Worker #N]` 段里。本文记录**已定决策**、架构、配套 driver 改动与里程碑验收标准。

状态：**规划已定稿，代码未开始**（决策记录见 §13；改决策请改这份文档，别只改代码）。

---

## 1. 目标与非目标

### 目标

1. **多 worker 管理**：启动/停止/重启/自动恢复，每个 worker 一个进程、一份输出。
2. **小窗输出**：每个 worker 一个可弹出的小窗口，显示该进程的 stdout/stderr，可读、可复制。
3. **GPU 监控**：占用率、SM 时钟、功耗等，按设备聚合，读数有历史曲线。
4. **低占用**：常驻内存预算 **≤ 50 MB**（含 ImGui + 字体图集）：GPU 监控采样、日志解析、进程 I/O 都不阻塞 UI 帧。
5. **简洁有效**：主窗 = worker 表 + GPU 面板 + 日志/详情面板，操作只有"启动/停止/重启/打开目录"这一档。
6. **可本地化**：界面文案外置为 XML，用户可自行翻译并热重载（§10）。

### 非目标（明确不做，避免返工）

| 不做 | 原因 |
|---|---|
| 把 ECM 数学（GMP/CGBN/OpenCL）链进 GUI 进程 | GUI 只是前端；数学在 worker 进程里，崩溃隔离与显存/上下文归属都靠进程边界（§5） |
| GUI 关闭后让 worker 在后台继续跑（detach） | 管道一断就失去退出码与重启控制 → 只剩 tail 日志文件，属于半成品状态（§5.6） |
| 在 GUI 里做 stage-2 / 拉起 `ecm_p95feeder` | v1 范围外；stage-2 交接由 Prime95 侧负责，记入 TODO（§16） |
| Android | 无 ImGui 后端（要自写 SDL/GLES + 触摸 UI），与桌面 UI 代码几乎不可共享 |
| 内置字体分发 | 中文需要 CJK 字体（+10 MB 以上）→ 改为运行时加载系统字体（§10.3） |

---

## 2. 技术选型

| 决策 | 选择 | 理由 |
|---|---|---|
| UI 库 | **Dear ImGui（docking 分支）** | 表格/曲线/停靠/多视口都是现成的；"内嵌可停靠 + 任意面板弹出为独立 OS 窗"一份实现覆盖两种布局偏好 |
| 平台后端 | `imgui_impl_win32` + `imgui_impl_dx11` | Windows 原生、无额外依赖（`d3d11.dll`/`dxgi.dll` 系统自带） |
| 渲染 | Direct3D 11 + 10 Hz 重绘（可配 `refresh_hz`） | 帧率与 worker 无关；空闲时把 `Present` 间隔拉长即可 |
| GPU 监控 | **NVML**（`LoadLibrary("nvml.dll")`，`C:\Windows\System32`） | 实测本机存在（驱动 610.88）；不 spawn `nvidia-smi` 进程；调用微秒级，2 Hz 采样不干扰 CUDA |
| 进程 | `CreateProcessW`（共享一个管道写端、Job object） | §5 |
| 配置 | 单一 `ecm.ini`（`[GUI]` + `[Worker #N]`） | §4，见 prime95 `prime.txt` 的心智模型 |
| 数学依赖 | **零**（GUI 不链 GMP/OpenCL/CUDA） | GUI 只解析文本，不做任何数论判断（§9） |

为什么**不**用 `master` 分支的 ImGui：master 上"Platform Window"恒为一个，标签页只在 docking 分支里产生
（`imgui.h` 内注释原文：*"always only one in 'master' branch"*、*"Tabs are automatically created by the
docking system (when in 'docking' branch)"*）→ 无法实现"面板弹出为独立窗"。

---

## 3. 平台抽象（Windows 先行，Linux 留口）

UI 代码 100% 与平台无关；平台相关只有三处，各抽成一个接口（约 150 行声明 + 一个 `_win32.cpp` 实现）：

| 接口 | 职责 | Windows 实现 | Linux 实现（M7） |
|---|---|---|---|
| `Platform` | 窗口/消息循环/字体文件定位/打开资源管理器 | Win32 + DX11 后端，`SHOpenFolderAndSelectItems` | GLFW + OpenGL3，`xdg-open` |
| `WorkerProc` | spawn/终止/读管道/设置优先级/进程树 | `CreateProcessW` + `CreateIoCompletionPort` + Job object | `posix_spawn` + `pipe` + `killpg` |
| `GpuMonitor` | 设备枚举 + 指标采样 | NVML（`LoadLibrary("nvml.dll")`） | NVML（`libnvidia-ml.so`） |

Android 明确排除；不要为了"以后也许"把抽象做厚——三层足够。

---

## 4. 配置：一份 `ecm.ini` 装下全部

> **全量键表见 `docs/DEV_ECM_INI.md`**（分区：全局 / `[Worker #N]` / `[GUI]`，逐键标注"谁消费、
> 有没有 CLI 等价物、默认值"，并单列"只有 ini、故意没有 CLI"的那一组）。
> 下面是段语义与 GUI 写 ini 的规则。

### 4.1 段语义（与 prime95 一致）

```ini
# ── 无 section = 全局默认（所有 worker 的默认值）────────────────
device = 0
method = gpu
gpucurves = 960
tmp_dir = saves
log_file = screen_<N>.log      # <N> 由 GUI 在未显式设置时补成 screen_1.log / screen_2.log …
NumWorkers = 2                 # 由 GUI 维护（driver 不需要，但读它不报错）

[GUI]
refresh_hz = 10
gpu_poll_ms = 500
language = chineseSimplified
localization_dir = localization
priority = below_normal
exit_confirm = ask             # 关窗口时：ask=弹窗确认（默认）| stop=不弹窗但先写检查点 | kill=立即杀
graceful_stop_ms = 300000      # 退出/停止时等新检查点的上限（毫秒），到点则按"未写检查点"终止
font_size = auto               # auto = 15 px × DPI 缩放（小数也行，如 23.5 / 24）；
                               # 0 或 auto = 自动；范围 6–96
font =                         # 留空 = 按语言自动挑系统字体；也可写完整路径
font_snap = 1                  # 1 = 字形推进量对齐整像素（拉丁小字更锐）；0 = 关（用小数字号时）
results_json = results.json.txt
results_txt = results.txt
window = 120,80,1600,900
dock_layout = <序列化串>        # dock_layout_ver = 3（版本不符会重建默认布局一次）
                               # 注：拖出去的浮动面板几何也在 dock_layout 串里，
                               #     不另设 win_<id>= 键（早期方案，已弃）

[Worker #1]
device = 0                     # 覆盖全局
worktodo = worktodo.txt        # 段感知：只消费本段的行（D2）
name = 4070Ti-110e6            # GUI 专属键，driver 忽略未知键
color = 0.2,0.7,1.0,1.0
autostart = 1

[Worker #2]
device = 1
gpucurves = 384
name = 4060L-260e6
```

解析顺序：`[Worker #N][key]` → 全局 `[key]` → 内置默认值。

### 4.2 已知约束与规则（都是从现有实现实测出来的）

* `src/core/ecm_queue_config.cpp:29-52` 的解析器**完全不看 section**（逐行 `key=val`、`#` 注释、剥 BOM、
  未知键静默忽略）→ **D1 要把它改成 section 感知**；改完后 `[GUI]` 与 `[Worker #N]` 里的未知键一律忽略，
  因此**不需要** `gui_` 之类的前缀（早期方案里那个前缀是为绕开"段盲解析串键"的风险，现在风险消失）。
* 队列模式入口：**无位置参数**即进入队列管理器（`src/core/ecm_driver.cpp:3583`），
  `-ini <path>` 可换 ini 路径；`--go` 是"打印群阶"，**不是**队列开关（3481）。
* worker 切段用 **`--worker N`**（1 起）：`ecm_cuda.exe -ini ecm.ini --worker 3`。同一开关同时选中
  ini 的 `[Worker #N]` 段与 worktodo 的 `[Worker #N]` 段 → GUI 与 driver 语义完全一致。
* 向后兼容（硬要求）：没有 `[Worker #N]` 段、也没有 `--worker` 时，行为必须与现在**逐字节一致**——
  现有 ini 与 `work_manager.ps1` 风格的调用照跑。

### 4.3 GUI 写 ini 的规则

GUI 需要回写 ini（窗口几何、`NumWorkers`、autostart、语言等），所以：

1. **原子改写**：写 `<ini>.tmp` → `MoveFileEx(..., MOVEFILE_REPLACE_EXISTING)`；中途失败原文件不受损。
2. **只改认识的键的 value**：其它行（含注释、空行、未知键、用户自定顺序）**原样搬运并保序**。
3. 新键追加到对应段末尾；段不存在则创建（`[GUI]` 放全局键之后、`[Worker #N]` 之前）。
4. 写回前保留一次备份 `<ini>.bak`（只保留一代）。

> 这条"保留注释与行序"的要求决定了 GUI 侧的 ini 读写**不能**复用 driver 的
> `ecm_queue_config_load()`（它是"取值即丢结构"的一次性解析）→ GUI 自带一个小的结构化 ini 读写器
> （`src/gui/ini_file.{h,cpp}`，~300 行），driver 侧不需要知道它。

---

## 5. 进程模型

### 5.1 一个 worker = 一个进程

```
ecm_gui.exe
 ├── ecm_cuda.exe -ini ecm.ini --worker 1   (device 0)  ─┐ 各自 Job object
 └── ecm_cuda.exe -ini ecm.ini --worker 2   (device 1)  ─┘ + 各自 stdout/stderr 管道
```

理由：崩溃隔离；`-d` 分卡天然互不干扰；GUI 不需要喂 stdin（队列模式从 worktodo 取任务，
`ecm_driver.cpp:3583` 之后才可能读 stdin，队列路径不读）。

### 5.2 spawn 细节

* `CreateProcessW` + `CREATE_NO_WINDOW` + `CREATE_UNICODE_ENVIRONMENT`；**不经 `cmd.exe`**
  （中文路径、`( ) ^ |` 之类的字符不会落到 shell 解析）。
* stdin：给一个空句柄（或 `NUL`）。
* **stdout 与 stderr 指向同一个管道写端**（`hStdError = hStdOutput`）→ GUI 用**一个**读线程、
  按真实写入顺序拿到行，不存在两个管道之间的重排。
* 环境变量继承 GUI 的（`ECM_*` 覆盖仍可用）。

### 5.3 实时性（已验证，无需改 driver）

* `ecm_ts_vfprintf()` 每次调用都 `std::fflush(f)`（`src/opencl_ecm_log.cpp:180`）；
  CUDA 侧进度同样 `fflush(stdout)`（`kernels/cuda/cgbn_stage1.cu:116,1684`）。
* 全仓库没有 `sync_with_stdio(false)` → `std::cout` 与 stdio 同步，`fflush(stdout)` 能把 `std::cout`
  的内容一起带出来。
* pipe 不是 TTY（`stdout_is_tty_local()`，`ecm_driver.cpp:1210`）→ driver 自动切**日志模式**：
  不打 `\r`，按衰减节奏打整行（`emit_progress_line`，1219）→ 日志量天然有界。

### 5.4 Job object 与优先级

* 每个 worker 一个 Job（`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`）：GUI 崩溃/退出时进程树全灭，
  **不会留下占着显存的孤儿**。"停止 worker" 用 `TerminateJobObject()`（能带走孙进程，比 `TerminateProcess` 干净）。
* 优先级 `[GUI] priority = idle|below_normal|normal|above_normal|high`，默认 `below_normal`
  （GUI 与 worker 抢 CPU 时 UI 不掉帧）。

### 5.5 崩溃策略（自动重启 + 熔断）

* 队列空而正常退出（`run_queue_manager` 里 `break` → `return 0`，`ecm_driver.cpp:3155,3265`）→
  显示 `queue empty` 并保持 Stopped，**不重启**。
* 非"队列空"退出 → 等 5 s 重启；**5 分钟内连续崩 ≥3 次** → 停止、标红、冻结最后一次输出（熔断）。
* 为什么敢自动重启：任务行只在成功后才被移除（`ecm_worktodo_advance(..., Remove)`，2973），
  崩溃时任务行**留在文件里**；中途 ckpt 也在（`--ckpt` / `ecm_checkpoint.cpp`）→ 重启即续跑。
* 退出码**不能**当命中判据：单跑模式找到因子也 `return 0`（3827）。命中一律靠文本钩子（§7.2）。

### 5.6 GUI 关闭 → 先问用户，同意后**先写检查点**再终止

用户要求（2026-09-28；并当场纠正了我最初理解错的"静默终止"）：**关窗口必须弹窗提醒**；
用户同意后，**worker 必须先写检查点**，然后才被终止。

实现（`App::request_close` / `begin_graceful_exit` / `tick_graceful_stops` / `draw_exit_modal`）：

1. `WM_CLOSE`（或 File → Quit）→ `App::request_close()`：
   * **没有 worker 在跑** → 直接关；
   * 有 worker → **弹模态框**（"仍有 N 个 worker 在运行，停止（每个先写检查点）并退出？"），
     窗口**不关**。回车 = 停止并退出，Esc = 取消（两者都在应用层处理，脚本可 PostMessage 驱动）；
   * `[GUI] exit_confirm = stop`：不弹框但仍走"先写检查点"的流程（脚本 / 无人值守）；
   * `[GUI] exit_confirm = kill`：立即终止（就是那个被否掉的"静默杀"，只作为显式选项保留）。
2. 同意后进入 **Stopping** 阶段：GUI 以 `<driver 目录>` 下 `.ecm_ckpt_*.dat` 的最新 mtime 为基线，
   等到**更新的**检查点出现才调 `TerminateJobObject`；期间模态窗口按 worker 显示
   `waiting for checkpoint` / `checkpoint written, stopping` / `stopped`，并提供
   **Force quit（不写检查点）** 按钮。等待是有界的：`[GUI] graceful_stop_ms`（默认 300000 = 5 分钟）。
3. trace 里的证据链（`tools/test/test_gui_exit_checkpoint.ps1` 断言的就是这些）：
   ```
   exit: confirmation requested (1 worker(s) running)
   exit: confirmed by the user
   worker 1: graceful stop requested (waiting for a checkpoint, max 90 s)
   worker 1: checkpoint written (.ecm_ckpt_991_7fffffff_ffffffff.dat), safe to stop
   worker 1: stopping (checkpoint written)
   exit: all workers stopped, closing
   ```
   测试还会**独立核对磁盘上**该文件的 mtime 比关闭前更新 —— 不只看 trace 自证。
4. 最长等待取决于 driver 的 `ckpt_seconds`（生产 ini 是 120 s）。如果 driver 从不写检查点，
   等到 `graceful_stop_ms` 后以"未写检查点"终止，并在 trace 里写明
   `stopping (checkpoint NOT written (timeout))`。
5. 单个 worker 的 **Stop** 按钮（含 Stop all）走同一条"先写检查点"的路：
   `worker N: stop requested (waiting for a checkpoint, max N s)`。
6. **顺带修掉一个真实缺陷**：GUI 过去不设置子进程的工作目录，而 driver 的检查点名**只有文件名**
   （`cgbn_stage1.cu: get_checkpoint_filename`）⇒ 从别处启动 GUI 时检查点会落到 GUI 的 CWD 而不是
   driver 目录。这不仅让上面的等待永远看不到检查点，也让 `.save`/指数缓存落错地方。
   现在 `WorkerSpawn::working_dir` = **driver exe 所在目录**，与手工在 driver 目录里跑一致。
7. 无人值守长跑仍建议直接用队列模式（`ecm_cuda.exe -ini ecm.ini --worker N`）或 `tools/` 里的脚本。

### 5.7 同卡重复占用警告

driver 是单进程，无从知道别的 worker 占了哪张卡 → **重复 `device=` 的检测只能在 GUI 侧做**：
无论重复来自全局默认值还是 `[Worker #N]` 覆盖，都在 worker 行上给黄字 + 主窗顶栏一条提示
（"Worker #1 与 #2 都用 device 0"），**不阻止启动**（用户可能故意共享）。同时在 driver 自己的
stdout 里也打一行提示（D5，便于 headless 用户看到）。

---

## 6. 队列（worktodo）

* **单一 `worktodo.txt`，用 `[Worker #N]` 分段**；没有段则视为 worker 1（与 prime95 的 `worktodo.txt` 同构）。
* 为什么不用"共享一个池、多进程抢占"：不同 GPU 的最佳 `gpucurves` 不同（本机 4070 Ti 60 SM vs
  4060 Laptop 24 SM），统一消费会互相拖累；段化还能让每张卡的计划独立演进。
* driver 侧改动 D2：`ecm_worktodo_first_line` / `ecm_worktodo_advance`
  变成**段感知**——只读自己的段、只删自己段的首行，其它段与注释原样保留。
* GUI 侧 v1 **只读**（显示剩余行数、当前任务、`finished` 尾部、命中列表）；编辑/生成留给 M6（§11，`docs/DEV_ECM_WORKTODO.md`）。

---

## 7. 输出面板、日志与状态

### 7.1 布局模型（ImGui docking + viewports）

* **默认布局（首次运行由 `DockBuilder` 建，见 `App::build_default_layout`）**：
  ```
  ┌────────────────────────────┬──────────────────┐
  │ Workers 表 ┆ 详情(标签页)   │ GPU 面板         │
  │        左 60% 宽，高 50%    │   右 40% 宽      │
  ├────────────────────────────┤   高 50%         │
  │ 每 worker 输出（标签页）    ├──────────────────┤
  │        左 60% 宽，高 50%    │ Results 表       │
  └────────────────────────────┴──────────────────┘
  ```
  即 **左列 60% / 右列 40%，每列上下各 50%**：左上 `Workers`（`详情` 与它同节点、默认不是选中页），
  左下 = 每个 worker 一个日志标签页，右上 `GPU`，右下 `Results`。实测矩形（978×544 客户区，
  `--trace` 原样输出，见 `tools/test/_run/layout_probe`）：
  `###workers 91,129 585×255` ≡ `###detail 91,129 585×255`（**同矩形 = 同一停靠节点的标签页**）/
  `###log1 91,387 585×255` / `###gpu 679,129 390×255` / `###results 679,387 390×255`，
  左列 585 / 右列 390 = 60.0 % / 40.0 %（客户区 978 宽）。
* **面板 ID 必须稳定**：所有面板用 `标题 + "###workers"` 这类固定后缀，否则**切换语言**会让标题变化、ImGui 视作不同窗口 → 布局丢失。用户拖动后的布局照旧写进 `[GUI] dock_layout`。
* **布局版本**：`[GUI] dock_layout_ver`（当前 3）。存的是旧版本、或根本还没有 dockspace 节点时，启动会**重建默认布局**一次 —— 这样"面板堆在中间"的历史 ini 会被自动修好，而不是被永久钉住。默认布局改过一次（v2 三列 → v3 四象限），所以老 ini 会被自动升到 v3。
* `--trace` 会打一行每面板矩形（`layout: workers ###workers x=… y=… w=… h=…`）与视口矩形，验收脚本据此断言"都在客户区内、互不重叠"（相等矩形 = 同一节点的标签页）。
* 任意面板可拖出成独立系统窗（`ImGuiConfigFlags_DockingEnable | ViewportsEnable`），
  用户想要 prime95 那种"每 worker 一个小窗"就把它拖出来即可。
* 几何与停靠布局由**我们**写进 `[GUI]`：`window=x,y,w,h` + `dock_layout=<串>`，
  并设 `io.IniFilename = nullptr` 关掉 ImGui 自带的 `imgui.ini`（满足"配置都在 `[GUI]` 下"）。
  浮动出去的面板几何也在 `dock_layout` 串里，因此**没有** `win_<id>=x,y,w,h` 这一族键
  （那是最初的计划，实测不需要，见 §13）。
  注意：prime95 的 `[Windows] W1=0 0 950 250 0 -1 -1 -11 -45` 那种 9 参数格式**不采用**
  （其 GUI 源码未随源码包发布，9 个字段的语义无法确证；我们用 4 数矩形 + 停靠状态串）。

### 7.2 分层日志

* **事件层**（进日志面板）：`START: <行>`、`FACTOR FOUND …`、`factor[i]=…`、
  `Checkpoint saved …`、`Resuming from checkpoint …`、`Checkpoint loaded …`、
  `ERROR:` / `# ERROR <行>`、驱动警告（含 "raise -gpucurves" 之类）。
* **进度层**（不进日志面板，喂状态列与速度历史）：两种前缀、字段相同，直接可解析：

  | 路径 | 行格式（日志模式） | 出处 |
  |---|---|---|
  | CPU（Edwards/Mont） | `stage1: [====>   ] 42.3%  42.3/100 (~1.20 s/curve)  elapsed 51.0s  ETA 69.0s` | `ecm_driver.cpp:1263` |
  | GPU（CUDA/OpenCL） | `GPU: [====>   ] 42.3%  123456, +789 bits (~1.20 s/curve)  elapsed 51.0s  remaining 69.0s` | `cgbn_stage1.cu:102` |

* **ANSI**：driver 会写 `\033[36m`（默认进度色，`opencl_ecm_log.cpp:27`）与
  `"\033[0m\r\n"`（`src/opencl_ecm_stage1.cpp:55`）→ GUI 必须解析成颜色/剥离，否则满屏乱码。
* **打印节奏（2026-09-29 修，回答"ecm_cuda 明明已经是实时进度"）**：进度回调**每批**都会被调用，
  而 `batch_size` 自适应到"一批 ≈100 ms"（`cgbn_stage1.cu`：`batch_time < 80` 就 +10 %、
  `> 120` 就 −10 %）⇒ **交互式终端下本来就是 ~100–200 ms 刷新一次**（`stdout_is_tty()` 为真时走
  `\r` 原地更新 + `fflush`，CPU/Mont 路径走 `indicators` 进度条，同样是每次回调都刷）。
  但重定向到**管道/文件**（GUI 的场景）走的是另一条分支：整行 + 时间戳，节奏**只看批号**
  `emit_progress_line(n)`（含 `n % 10000 == 0`），而**批号会从检查点恢复**
  （`batches_complete = ckpt_header.batches_complete`）⇒ 恢复后的运行 n≈73900 只命中
  `n % 10000 == 0`，每批 ~1 s 就是**约 1.7 小时一行**：GUI 于是一直显示"等待第一行进度"，
  而同一份 exe 在控制台里每批都在刷新（用户实测对比，差异就出在这里）。
  现在**两条路径都加时间兜底**：距上一行 ≥ **200 ms** 就输出（GPU 用 CUDA 计时器 `*gputime`、
  CPU/Mont 用 `stage1_now_ms()`），旧的批号里程碑保留以兼容老日志尾随工具。
  实测：用户那份**恢复运行**的 B1=2.6e8 任务，32 s 内 **135 行 ≈ 4.2 行/s**（≈200 ms 一行 ✓）。
  > 代价：driver 的 `log_file` 也按这个节奏增长（~110 B/行 × 5 行/s ≈ 2 MB/小时）。
* **文件节奏 = `progress_log_seconds`（2026-09-29 加，回答"管道每次、文件每 n 分钟"）**：
  **管道/控制台永远每次**（GUI 靠它更新进度条与 ETA，约 200 ms 一行）；
  **`log_file` 另加一道时间闸门**，由 ini 键 `progress_log_seconds` 控制：

  | 取值 | 文件里的进度行 | 用途 |
  |---|---|---|
  | `60`（默认） | 最多每 60 s 一行 | 长时间任务不再把 `screen.log` 写爆（上面那 2 MB/小时 → ~7 KB/小时） |
  | `0` | 一行都不写 | 只想要事件/结果，不想要进度噪音 |
  | 负数 | 每行都写 | 老行为（需要完整进度轨迹时） |

  无论取值如何，**报告 `100.0%` 的那一行总会写进文件**，所以"任务跑完了"永远能在日志里看到；
  闸门用 `steady_clock`（系统时间跳变不会让日志静默或突然刷一堆行），并且是在**日志镜像层**
  （`opencl_ecm_log.cpp` 的 `ecm_ts_vfprintf`）判定，因此 CPU/GPU 两种进度行形状都覆盖，
  驱动本身不需要知道有没有日志文件。判定 = "行里有 ASCII 条 + `%`"（`GPU: [` / `stage1: [`），
  横幅、结果、警告一律不算进度行。实测（M521/B1=1e5/4 曲线，7.5 s 任务）：
  `0` → 文件 1 行（100 %）、管道 27 行；`1` → 文件 7 行、管道 26 行；`-1` → 文件 = 管道 = 27 行。
  验收：`tools/test/test_progress_cadence.ps1`（19 项）。
* **恢复百分比**：`Resuming from checkpoint: 23.8% complete` / `Checkpoint loaded: … (23.8%)`
  会被 GUI 解析成进度条起点（trace：`progress seed=23.8 (from checkpoint)`），
  所以即使第一行进度还没来，进度条也不是空白。
* 每个 worker 一个环形缓冲（默认 5000 行）+ 自动滚底 + 悬停暂停 + 一键复制 + `显示原始输出` 开关。
* 进度条由 GUI **自己画**（driver 那 40 字符 ASCII 条不显示，只解析它的百分比/ETA）。
* 每 worker 的 `log_file` 默认是 `screen.log` → 多 worker 会互相插花 ✗ ⇒ GUI 在未显式设置时
  补成 `screen_<N>.log`（不覆盖用户显式给定的值）。

### 7.3 Workers 表与详情

**布局（用户反馈后改过，M10）**：一行一个 worker，**worktodo 任务行单独占一行**（跨整表宽度、暗淡显示、
自动换行）——之前任务行是一个表格列，worktodo 行 ~80 字符会把那一列撑到几千像素宽，
把进度条/速度/ETA 挤出面板右边（用户原话："进度条，eta 等均不显示"）。
现在除**进度条列（占剩余宽度）**外每列都是**按示例文本算出的固定宽度**（跟随 DPI 字号），
状态列的 `state_text` 过长时截断并以 tooltip 显示全文（`--trace` 的 `table: workers …` 行给出实测几何，
`fits=1` 表示进度条与 ETA 都在面板内）。

| 列 | 来源 |
|---|---|
| `#` / 状态 | GUI 状态机（Stopped / Starting / Running / Restarting / Error / Queue empty）+ 状态说明（截断 + tooltip）。**本地化键是显式表**（`state_key()`），不再是"从状态名推导"——推导版本把 `Stopped` 变成 `state__stopped`（多一个下划线），键不存在 ⇒ 表里直接显示 `workers.state__stopped`（用户实测 BUG）|
| GPU | 配置里的 `device=` + NVML 设备名（**归属靠配置，不靠 NVML**——NVML 的进程级 util 在 Windows 上不可靠） |
| 名称 | `[Worker #N] name=`（旧 driver 会在后面打红 `!`，见 §11.0）|
| 进度% / 速度 / ETA | 进度行解析（`s/curve`、`elapsed`、`ETA|remaining`），进度条由 GUI 自绘 |
| 命中数 | GUI 记账（命中来自 `factor[i]=` / `FACTOR FOUND`）|
| Actions | 只画**当前可用的那个按钮**（运行中 = Stop，否则 = Start；熔断后 Start 被拒并给出原因）—— 两个按钮并排要多花 ~50 px，而进度条更需要这点宽度 |
| **任务行（第二行）** | `START:` 行的原文（`ECMSTAGE2=…`），整行跨列、换行、悬停看全文。实测（用户窗口 2093×1091、面板 ~1240 px）：`task_wrap_w=1201`、`progress_w=475`、`fits=1` |

**表格边框只用横线**（`BordersInnerH | BordersOuterH`，不再是 `Borders`）：ImGui 把表格边框画在
**单元格内容之后**，所以纵线（"y 轴线"）会横穿那条跨列的任务行。行之间仍有横线分隔，列靠位置与表头对齐。
（这是 ImGui 的限制：没有"按行隐藏边框"的开关，除非把任务行挪到表格外面。）

**Results 表的列顺序与滚动**（用户要求）：先放**标量/有限字段**
（`Factor → bits → Hits → First seen`），再放**列表字段**（`Curves → Sigmas`，列宽固定、超长截断、
悬停显示全文），并开 `ScrollX | ScrollY`（表格高度 = 面板剩余高度），`TableSetupScrollFreeze(4, 1)`
让滑到右边时前 4 列仍可见。

**进度条"没显示"的排查顺序**（用户实测过两次，原因不同）：

1. 先看 trace 有没有 `worker N: progress pct=…`：**有** = GUI 收到了并解析成功（进度行不进事件日志面板，
   这是刻意的；想看原文就勾 `raw output`）。**没有** = driver 还没打第一行（它只在
   `emit_progress_line(batches_complete)` 为真时打：前 3 个 batch 各一行，之后 10/100/1000/10000
   个 batch 各一行，见 `cgbn_stage1.cu:123`）——大 B1 任务的第一个 batch 可能要等一会儿。
2. 详情面板现在会显示 `progress lines <N>` 与最后一次解析到的数值：**N=0 且状态 Running** ⇒ 是
   上面第 1 条的第二种情况，不是 GUI 聋了。
3. 进度条格子本身宽不宽：见 §7.3 上表与 `table: workers …` 的 `fits`/`progress_w`。
4. 运行中但还没有第一行进度时，进度条画的是 `waiting for the first progress line`（i18n 键
   `workers.waiting_progress`），不再是空白条 —— 空白条会被读成"GUI 坏了"。

详情面板（选中行）：**生效配置**（全局 + 段覆盖后的逐键结果，用于排查"我改了 ini 为什么没生效"）、
task 行原文、完整命令行、**秒/曲线（`s/curve`）历史曲线**、事件列表、按钮（Start/Stop/Restart/打开保存目录）。

> **输出面板画的是"秒/曲线"，不是进度 %，也不是"曲线/秒"（2026-09-29 用户要求）**：
> ① 进度 % 的曲线被**删掉**（百分比在同一行已有进度条，再画一条曲线是重复信息）；
> ② 纵轴单位从 `curves/s` 改成 **`s/curve`**（越小越快）—— 用户要的就是这个量，
> 而 `curves/s` 与"快"是反着读的（数值越大越快），容易看反；
> ③ 中文标签 `秒/曲线（越小越快）`，英文 `s/curve (lower is better)`。
> 曲线点的来源与判据见 §8.2，测试断言见 §15（`test_gui_workers.ps1` 的 `[3c]` 段：
> 断言曲线存在、有真实序列（`n≥2`）、标签单位为 `s/curve`、**数值 ~1.2 而不是 `curves/s` 的 ~0.83**，
> 并且**不存在**任何进度 % 曲线）。

---

## 8. GPU 监控

* NVML 采样放**独立线程**，周期 `[GUI] gpu_poll_ms`（默认 500 ms），历史 240 个样本（默认周期下 ≈2 分钟）；
  所有 NVML 调用失败都只在 UI 上降级（"NVML 不可用" + 具体原因），绝不影响 worker 与 UI 帧。
* **实现（M4，已落地）**：`src/gui/gpu_monitor.{h,cpp}` —— **运行时动态加载** NVML
  （`LoadLibrary("nvml.dll")`，失败再试 `%ProgramFiles%\NVIDIA Corporation\NVSMI\nvml.dll`；
  测试可用环境变量 `ECM_GUI_NVML=<dll 路径>` 指到别处，用来验证降级路径），按名字取入口点。
  **不链 NVML 导入库、不打包 `nvml.h`**：`gpu_monitor.cpp` 里自己声明用到的那一小撮 ABI
  （全是 `unsigned int`/`unsigned long long` 的平凡结构体 + 按名的函数），并由 `--gpu-selftest`
  与 `nvidia-smi` 交叉比对来保证声明没错（结构体布局错了会显示垃圾值而不是"看起来合理"）。
* 每张卡的卡片：`util%（含 mem util）`、`功耗 / 功耗上限（含百分比）`、`SM 时钟`、`显存时钟`、
  `温度`、`显存占用/总量`、**节流原因**（bitmask 译成文字，`idle` 单独出现时视为正常不显示），
  以及 util / 功耗 / SM 时钟三条历史曲线。
* `GpuInfo::cores` 是 **CUDA 核数**（`nvmlDeviceGetNumGpuCores`；60-SM 的 Ada 卡报 7680 = 60×128），
  **不是 SM 数** —— SM 数由 driver 的 `--gpu-info`（D4）给出，`gpucurves` 推荐器用那个。
* worker↔GPU 归属仍靠配置（`device=`），不靠 NVML（Windows 上 NVML 的进程级 util 不可靠）。

### 8.1 已知的读数怪癖（实测，写进 §16 让用户心里有数）

| 现象 | 事实 |
|---|---|
| 功耗偶尔跳到 ~590 W | 本机 RTX 4060 Laptop GPU（强制上限 55 W）会间歇性报 **590.01 W**，**`nvidia-smi` 与 NVML 都会**（同一时刻另一个工具可能报 9.6 W）。物理上不可能 ⟹ `--gpu-selftest` 把"超过强制上限 3×"的样本判为**不可信**并跳过比对，面板照原样显示（那是驱动给的数）|
| 忙时交叉比对要允许重试 | 比对的目的是证明**自声明的 NVML ABI/枚举读对了量**（错字段会差一个数量级），不是两点读数逐点相等：机器忙时（GUI 自己在渲染、别的测试在跑 GPU 任务）util/功耗/温度都会有百分比级差异，SM 时钟更是在 210↔2595 MHz 间跳（12 倍）⟹ 时钟只做**量程**断言（100..4000 MHz；显存时钟 10501 会被抓出），其余用观测区间 ±容差，并且整套比对**失败会自动重采一次**，两次都不同意才算失败。实测：空闲 5/5、GPU 满载 5/5 通过 |
| 时钟读数秒级跳变 | 空闲卡在 210 MHz 与 2595 MHz 之间跳（功耗门控）⟹ 与 `nvidia-smi` 的比对必须**交错采样**（先采 NVML、立刻查该卡）并对时钟只做量级校验，否则会误判 |
| GUI 自己占一点 GPU | 界面走 D3D11 + ImGui，窗口本身会让 util 出现几个百分点甚至偶发尖峰 ⟹ `--gpu-selftest` 的 util 容差放到 ±40（不是 ±5）；worker 在跑时面板上的 util 也含 GUI 自身这点开销 |
| **"曲线是一条直线"先看是哪种直线**（用户实测反馈） | 空闲卡的功耗/频率**本来就几乎不动** ⟹ 直线是对的，不是 bug。实测（`gpu: history` 自证行，`test_gui_gpu_curves.ps1`）：<br>• 忙卡（4070 Ti 跑 `ecm_cuda`）：`samples=44 util=17 distinct power=22 distinct clock=21 distinct`，功耗 `9.2..102.5 W`、SM 时钟 `632..2774 MHz`；<br>• 空闲卡（4060 Laptop）：`power=2 distinct (1.5..1.6 W) clock=1 distinct (210..210 MHz)` —— 频率就是一条直线。<br>另一种"假直线"是量程问题：`PlotLines` 传 `0,0` 时 ImGui 自动量程会把 ±0.2 W 的抖动压平 ⟹ 现在按**观测窗口 min/max ±12 %** 手动设量程（span 至少 2 %，见 §8.2），并且 trace 里直接给 `plot=lo..hi` 与 `distinct` 供脚本断言 |

### 8.2 曲线重做：一个共用的"可读"曲线控件（2026-09-29，用户要求"更美观、易读"）

**问题**：原先每张图都是一句 `ImGui::PlotLines(...)`。ImGui 只给一条折线 + 一个外框，于是实际效果是
①没有底色/网格，线的形状靠眼睛猜；②没有纵轴刻度，只能看 `min/max` 两个灰字；③量程要么固定要么
`0,0` 自动（`auto` 会按当前数据抖动，同一张卡在两帧之间纵轴刻度就变了）；④**没有单位**，
`100` 是 100 %、100 W 还是 100 MHz 全靠上下文；⑤"秒/曲线"这类**越小越好**的量没有参照物。

**做法**：新增一个共用控件 `App::draw_metric_plot(const MetricPlot &p, const std::vector<float> &v,
float *used_lo, float *used_hi)`（`src/gui/app.h` / `src/gui/app.cpp`），**替换掉全部 `PlotLines`**
（worker 速度图 + GPU 的 util/功耗/SM 时钟三张图走同一份代码）。参数是一个 `MetricPlot` 结构体：

| 字段 | 作用 |
|---|---|
| `id` | 唯一 id，同时是 trace 的主键与 ImGui 的 id（`gpu0/util`、`worker2/s_per_curve`） |
| `label` / `unit` / `color` | 左上角标题、右上角数值后面的单位、曲线与面积的基色 |
| `height` / `decimals` | 图高（GPU 图 58、worker 图 62，单位是 **15 px 字号**下的像素：实际高度 = 该值 × 字号/15，并保证曲线带 ≥24 px）、数值小数位 |
| `fixed_range` + `lo`/`hi` | 固定量程（util 恒为 `0..100`，百分比不该跟着数据缩放） |
| `ref` / `has_ref` / `ref_label` | 虚线参考线 + 右端小标签（功耗图用**已执行的功耗上限**，见下） |
| `ms_per_sample` | 悬停时把样本下标换算成"多少秒前"，鼠标读数才有意义 |
| `empty_text` | 样本不足 2 个时显示的 `collecting…`（本地化键 `gpu.collecting` / `workers.collecting`），而不是画一条假直线 |

**画出来的东西（自下而上）**：圆角卡片底（`AddRectFilled` + `AddRectFilledMultiColor` 的竖直渐变；
注意 `AddRectFilledMultiColor` **没有圆角参数**，圆角只能由随后的 `AddRect` 给）→ 3 条横向网格线
（`IM_COL32(255,255,255,22)`）→ 面积填充 → 2 px 折线 → 最新样本的圆点 → 标题行（左标题、右 `数值+单位`）
→ 量程两端的最小/最大灰字 → 参考虚线 + 标签 → 鼠标悬停时的竖直引导线 + tooltip（数值、单位、`n` 个样本前）。

**面积填充为什么是"逐列矩形"而不是多边形**：曲线（尤其功耗）是**非凸**的，
`AddConvexPolyFilled` 对非凸多边形会画错；而 ImGui 的 `PathFillConvex` 系列没有非凸版本。
所以面积由**每列一个矩形**拼成（`AddRectFilled`），视觉等价于面积图且对任意形状都正确。

**量程规则**（两张自动量程的图 —— 功耗与 SM 时钟）：取**观测窗口**的 `min/max`，上下各留 **12 %** 余量，
并且**至少 2 % 的跨度下限**（否则一条 1.5↔1.6 W 的空闲曲线会被放大成"剧烈波动"）。
实测（§8.1 那张表）：空闲卡 `power=2 distinct (1.5..1.6 W)`，用了 2 % 下限后曲线才是平的。
**曲线画完后把真实量程回传**（`used_lo`/`used_hi`），`trace_gpu_history()` 报的就是**画出来的**那个量程，
所以 `test_gui_gpu_curves.ps1` 的"量程必须覆盖观测数据"断言测的是真图，而不是一份副本。

**曲线也要能被脚本验证**：每张图按帧比较渲染串，**同一张图前 3 次变化必打，之后每图每 2 s 最多一行**：

```
plot: gpu0/util  label="占用率" n=22 lo=0.00   hi=100.00 last=99.00
plot: gpu0/power label="功耗"   n=22 lo=137.13 hi=167.17 last=143.73 ref=285.00
plot: worker1/s_per_curve label="秒/曲线（越小越快）" n=6 lo=1.20 hi=1.20 last=1.20 h=78 line_h=23 band_top=26 band_h=27
```

限流是**按图**（`std::map<std::string,…>`，键 = `id`）而不是全局：早期版本用一个共享时间戳，
结果只有"第一张图"有 trace，其余全被吃掉（实测）。另外 `gpu: limits dev=N power_limit_w=X`
在**上限变化时**打一行 —— 这样脚本可以把功耗图上的 `ref=` 与 NVML 的真实上限**交叉核对**，
而不是相信那张图。

**布局必须跟着字号走（这一条是本轮最要紧的发现）**：卡片是"标题行 + 曲线带 + 最小/最大值行"，
而 `MetricPlot::height` 给的是**15 px 字号下的数值**。原先按固定像素用，于是用户在 **150 % 缩放**
（字体 22.5 px、行高实测 ~31 px）下，两个文字行**互相重叠**、并且一起压在曲线上——看起来就是
"图很糊、曲线被字盖住"。现在高度按 `font_size_px_/15` 缩放，并给曲线带保底 **24 px**
（`h = max(height × scale, 2×line_h + 8 + 24)`），三条带互不重叠。这条有**机器断言**：
trace 里给 `h= line_h= band_top= band_h=`，`test_gui_workers.ps1` 的 `[3d]` 段断言
`band_top ≥ line_h + 2`、`band_top + band_h ≤ h − line_h`、`band_h ≥ 24`（每张图各 3 条）。

**两个真实数据问题（都是先看到图"不对"才查出来的）**：

1. **一条物理上不可能的尖峰毁掉整张图**：4060 Laptop 会间歇性报 590 W（强制上限 55 W，§8.1 已记录），
   而功耗曲线是按观测窗口自动量程的 ⇒ 量程被拉成 `0..660.61 W`，真实曲线只剩 `1.5..9.4 W`，
   在图上就是贴着底边的一条直线。实测（`plot: gpu1/power`）：修前 `lo=0.00 hi=660.61`，
   修后 `lo=0.68 hi=10.49`。规则与整机功耗合计那条一致：**样本 > 3 × 强制上限就丢弃**，
   只影响曲线，面板文字行照原样显示驱动给的读数。
2. **参考线可能落在量程外**：功耗上限 285 W，而观测窗口只有 `134..168 W` ⇒ 原先的
   `if (ref >= lo && ref <= hi)` 让虚线**永远不出现**（"上限线去哪了？"）。现在越界时把虚线
   **贴到越界的那条边**（上方/下方），标签带一个**自绘的小三角**指向该方向（自绘而不是 `↑` 字形：
   不依赖字体是否覆盖 U+2191），既保住"离上限还很远"这个信息，又不会为了塞进上限而把曲线压平。


---

## 9. 命中因子：`results.json.txt` + `results.txt`

现状（**这是要修的数据丢失点**）：队列路径只把 `factor[i]=<dec>` 与 `FACTOR FOUND aid=… task=…`
打在 stdout（`ecm_driver.cpp:2945,2952`），落盘只有 `finished`（原任务行）与 `.save`；
日志一滚（`screen.log` 虽持久，但可能被人工清理）因子就难追溯。

* **`results.json.txt`（追加式 JSONL，真相源）**：每命中一条独立对象，永不重写 → 崩溃/断电最多丢最后一条。
  ```json
  {"status":"F","worktype":"ECM","exponent":5351,"factors":["12345678901234567891"],"b1":110000000,
   "param":0,"sigma":12345,"curve":37,"stage":1,"device":0,"worker":1,
   "task":"ECMSTAGE2=…","timestamp":"2026-10-05T12:00:00Z"}
  ```
  字段名对齐 prime95 现行 `results.json.txt`（`status/worktype/exponent/factors/b1/b2/…`）。
* **`results.txt`（合并表，人类可读，从 JSONL 派生）**：同一因子被多条曲线命中时**合并成一行**，
  按因子十进制串去重：
  ```
  # ecm_gui results — 同一因子被多条曲线命中时合并；updated 2026-10-05T12:00:00Z
  M5351 has a factor: 12345678901234567891 (ECM curves 37,91,204, B1=110e6, param 0, Sigmas=[12345,23456,34567], hits=3)
  ```
  行首沿用 prime95 `M<exp> has a factor: ` 形状（便于沿用既有 grep/汇总习惯）。
  写入方式：GUI 原子重写（tmp + `MoveFileEx`）；任何时刻都能由 JSONL 重建 ⇒ 不怕重写丢失。
* UI：命中时**行高亮 5 秒** + 顶栏状态行 `factor found: <F> (worker N, M factor(s) total)`；
  Results 面板列出合并后的因子表（因子/位数/曲线/sigma 列表/命中数/首次时间，悬停显示 `results.txt` 那一行），
  并带 **"从 JSONL 重建 results.txt"** 与"打开结果目录"两个按钮。不做因子校验（driver 已做：
  `opencl_ecm_stage1.cpp:471` "invalid factor (N % factor != 0)"，另有 `--verify-gpu`）。

### 9.1 实现细节（M5，已落地）

| 项 | 事实 |
|---|---|
| 文件位置 | `[GUI] results_json=` / `results_txt=`，默认在 **exe 目录**下（`results.json.txt` / `results.txt`）|
| JSONL 字段 | `status`(F) / `worktype`(ECM) / `exponent` / `factors[]` / `b1` + `b1_text` / `n` / `param` / `method` / `curve` / `sigma` / `save` / `stage`(1) / `worker` / `device` / `task` / `timestamp`(UTC) —— 名字对齐 prime95 的 `results.json.txt` |
| 合并键 | 因子十进制串；合并 `curves`（去重、按命中顺序）、`sigmas`（去重）、`hits`（含重复命中）、首/末命中时间 |
| 指数与 B1 来源 | save 名走 `m{n}_{b1}.save` 契约（与 driver 同一规则）；没有 save 名时从任务行的 `k,b,n,c` 取 n |
| 原子性 | JSONL 只追加（崩溃最多丢最后一行，且**截断尾行在重放时被忽略**）；`results.txt` 走 tmp + `MoveFileEx` 替换，随时可由 JSONL 重建 |
| 重建入口 | GUI 按钮"从 JSONL 重建 results.txt"（`ResultsStore::rebuild_from_jsonl`）—— 也用来验证"表是派生的" |
* 依赖 D3：队列包装层的命中行补字段 `factor[i]=<dec> curve=<idx> sigma=<64bit> param=<p> save=<file>`
  —— 否则 GUI 得跨三种不同格式（mont/edwards 的 `  curve %u sigma=%llu -> factor found`、
  OpenCL 的 `GPU: factor found in Step 1 with curve %d (-sigma %d:%u)`、CGBN 的
  `GPU: factor %s found in Step 1 with curve %ld (sigma %d:%lu)`）按 curve 索引拼接 ✗ 脆弱。

---

## 10. 本地化与字体

### 10.1 文件（UTF-8 **无 BOM**，与 Notepad++ 的本地化文件一致）

```
localization/
├── english.xml              # 基线：所有键必须在这里有定义（缺失即回退到这里）
├── chineseSimplified.xml
└── README.md                # 贡献流程：复制 english.xml → 改 name → 提 PR
```

```xml
<?xml version="1.0" encoding="utf-8" ?>
<EcmGui>
  <Native-Langue name="English" filename="english.xml" version="1">
    <Panel id="workers">
      <Item id="start" name="Start"/>
      <Item id="stop"  name="Stop"/>
      <Item id="state_running" name="Running"/>
    </Panel>
    <Panel id="gpu">
      <Item id="util" name="GPU util"/>
      <Item id="power" name="Power"/>
    </Panel>
  </Native-Langue>
</EcmGui>
```

* 只吃 UTF-8（含/不含 BOM 均可）；非 UTF-8 给**明确报错**而不是乱码。
* 缺键/缺 Panel → 回退 `english.xml` 的同名项；UI 里放 "Reload localization" 按钮**热重载**（改完立即可见）。
* 选择语言：`[GUI] language = chineseSimplified`（= 文件名去掉 `.xml`），默认 `english`。

### 10.2 编码约定（实测）

Notepad++ 的 `chineseSimplified.xml` 是 UTF-8 无 BOM（首字节 `3C 3F 78 6D` = `<?xm`，
按 UTF-8 解出 `name="简体中文"`）。我们的文件用同样约定，方便用户拿其它工具的本地化文件照抄结构。

### 10.3 字体（不打包任何字体）

ImGui 内置字体**只有 ASCII**，且默认只有 13 px（高分屏上小到看不清），直接显示中文是方块
⇒ 策略是"**运行时加载系统字体 + 字号按 DPI 自动放大**"：

1. **字号**：`[GUI] font_size`（默认 `auto`）→ **`15 * window_dpi_scale()`，保留小数**。
   DPI 由 `GetDpiForWindow()`（失败退 `GetDeviceCaps(LOGPIXELSY)/96`）拿到，实测 150 % 屏 → **22.5 px**
   （不再四舍五入成 23：1.92 的动态 atlas 就是按实际绘制字号光栅化的，凑整反而会因缩放而发虚）。
   想固定就写 `font_size = 18`（显式像素值优先，不再乘 DPI），也接受小数（`23.5`）；范围 6–96。
2. **字体文件**：`[GUI] font = <路径>` 显式指定；否则按界面语言选：
   需要 CJK（`chineseSimplified` 等）→ `cjk_font_candidates()`（`msyh.ttc` 微软雅黑 / `msyhbd.ttc` /
   `simhei.ttf` / `msjh.ttc` / `simsun.ttc` / `Deng.ttf`）；纯拉丁语言 → `ui_font_candidates()`
   （`segoeui.ttf` / `tahoma.ttf` / `arial.ttf` / `verdana.ttf`）—— **拉丁语言也换掉内置位图字体**，
   否则英文界面同样小。全部失败 → `AddFontDefault()` 但把 `cfg.SizePixels` 设成算出来的字号（至少不小）。
3. **不传 glyph ranges**：本仓库 vendored 的是 docking 分支 `1.93.0 WIP`（commit `64944b45`），
   1.92 起字体 atlas 改成**按需动态增长**，`AddFontFromFileTTF(path, size)` 两个参数即可；
   `GetGlyphRangesChineseFull()` 已经落进 `#ifndef IMGUI_DISABLE_OBSOLETE_FUNCTIONS` 区
   （`third_party/imgui/imgui.h:3910,3941`），既不需要、也不该用 —— 传了反而把 atlas 按 21000 字形预烤。
4. **样式同步缩放**：`style.ScaleAllSizes(dpi_scale)`，否则 23 px 字配 4 px 间距会很挤。
5. **"中文没显示"要分三层查，别只看一层**（本轮实测教训）：

   | 层 | 症状 | 判据 |
   |---|---|---|
   | 本地化没加载 | 界面是英文（或键名） | trace `localization: ... language=… keys=… missing=…`（实测 `keys=69 missing=0`）|
   | 字体没有该字形 / 没烤进 atlas | 显示 `???` 或 tofu 方块 | trace `map=.. baked=..`（见下）|
   | 字体选错了（**运行中切语言**） | 大量 `???` | trace 里 `[startup]` / `[language change]` 两行（见第 8 条）|
   | atlas 里有但后端没画出来 | 方块 / 空白 | **像素**（`tools/test/test_gui_cjk_pixels.ps1`）|

   `--trace` 会打四行自证（实测，150 % 屏 + 中文）：
   ```
   font: C:\WINDOWS\Fonts\msyh.ttc at 22.5 px (dpi x1.50, cjk system font, snap=1), CJK language [startup]
   font: measured latin 'WW'=36x23 cjk 2-glyphs=35x23 cjk_ok=1 map=11 baked=11 negctl=00
   localization: dir=… language=chineseSimplified keys=69 baseline=69 missing=0 cjk=1
   localization: sample workers.title='工作线程'
   ```
   * `map=11` = `ImFont::IsGlyphInFont(U+6587/U+4EF6)` 都为真（TTF 真有这两个字）；
   * `baked=11` = 当前字号下 `ImFontBaked::FindGlyphNoFallback()` 返回**真字形**。
     **宽度不能当判据**：`FindGlyph()` 在缺字时会悄悄换成 U+FFFD fallback，而那个方块**也有宽度**
     （本轮实测：中文两字 35 px、拉丁 `WW` 36 px，看着"正常"，其实可能全是 fallback）。
   * `negctl=00` = U+E123（私用区，任何 UI 字体都不映射）必须报"缺" ——
     它要是报"有"，上面两个判据就什么也证明不了。
   * `sample` 行直接打出一条真实译文的**原始字节**（`工作线程` = U+5DE5 U+4F5C U+7EBF U+7A0B），
     用来区分"界面是英文"和"界面是中文但字是方块"。
6. **像素级验收**：`tools/test/test_gui_cjk_pixels.ps1` 真抓窗口
   （`tools/diag/grab_window.ps1`，`PrintWindow(PW_RENDERFULLCONTENT)`）并用
   `tools/diag/text_ink_probe.ps1` 量第一条文字带（= 菜单栏）里的逐字形格子。实测标定：

   | 情形 | 字体 | 字号 | 格子 | 中位宽 | 不同 ink 值 |
   |---|---|---|---|---|---|
   | 真中文 | `msyh.ttc`（自动） | 40 px | 5 | **30 px**（0.75 em） | **5**（每个字自己一套笔画）|
   | tofu（`[GUI] font` 强制成无中文字体，**修好之前**） | `arial.ttf` | 40 px | 6 | **17 px**（0.43 em） | **2**（同一个 fallback 反复出现）|

   判据是"中位格子宽 ≥ 0.6 em **且** 不同 ink 值 ≥ 3"。注意"中间是不是空心"**不能**当判据 ——
   实测 fallback 方块中间也有笔画（`midInk>0`），真中文和 tofu 在这一点上没区别。
   现在 `[GUI] font` 指向画不出中文的字体时**会被救回来**（见第 7 条），所以那一行数字是
   "修好之前"的标定值，用来证明这套判据当初确实能分辨 tofu。
7. **画不出当前语言 = 不许显示方块**（本轮核心修复）。字体选择集中在
   `main_win32.cpp: apply_ui_font()`，规则：

   | 情形 | 行为 |
   |---|---|
   | `[GUI] font=<路径>`，且该字体画不出该语言 | trace `cannot draw CJK: rescuing with …`，**换成系统 CJK 字体**（用户要的是能看的界面，不是一个具体文件） |
   | 系统里没有任何能画该语言的字体 | trace + 状态栏 `no CJK-capable font found: switched the UI to English (set [GUI] font=<path>)`，**界面切回英文**（永远可读），绝不显示 `???` |
   | 两者都不需要（英文界面 + 拉丁字体） | 正常加载 |

   判据用的是 `ImFont::IsGlyphInFont()`（TTF 里到底有没有这个字），不是"量一下文字宽度"。
   `ECM_GUI_CJK_FONT=<path|none>` 可覆盖查找（`none` = 假装系统里没有 CJK 字体），
   测试用它来跑"完全没有 CJK 字体"这一路。
8. **语言可以在运行中切换 → 字体必须跟着换**（"我这边中文是 ???，你测试却正常"的真正原因）。
   字体是在**启动时**按当时的语言挑的；用户在 Language 菜单里切成中文时，如果启动时是英文界面，
   加载的就是拉丁字体（segoeui），于是所有中文标签都渲成 `???`。
   现在 `App::reload_localization()` 会置 `font_reload_requested_`，帧循环在**两帧之间**
   重新调用 `apply_ui_font()`，trace 里能看见两次决策：
   ```
   font: C:\WINDOWS\Fonts\segoeui.ttf at 40.0 px (dpi x1.50, latin system font, snap=1) [startup]
   language: runtime switch to chineseSimplified (--switch-language)
   font: C:\WINDOWS\Fonts\msyh.ttc at 40.0 px (dpi x1.50, cjk system font, snap=1), CJK language [language change]
   font: measured latin 'WW'=62x40 cjk 2-glyphs=60x40 cjk_ok=1 map=11 baked=11 negctl=00
   ```
   `--switch-language <stem> [--switch-language-after <frames>]` 是**诊断开关**（和 `--trace` 一样
   刻意留在二进制里）：它做的和 Language 菜单完全同一个调用，好让脚本不用点菜单就能验证这条路。
   顺手删掉了 `App::init` 里那个硬编码 `if (language_ == "chineseSimplified" && 找不到字体)`
   的老守卫 —— 策略现在只在一个地方实现（`apply_ui_font`），而且对任何语言都成立。
9. **字号与锐度**：`font_size` 支持小数（150 % 屏自动值是 22.5，不再四舍五入成 23），
   `font_snap = 1`（默认）打开 `ImFontConfig::PixelSnapH`：ImGui 的文字是按**小数坐标**排的，
   推进量落到整像素上会明显更锐（代价：字号是非整数时该关掉它，见 ImGui 注释）。
   另外要清楚：ImGui 用的是**灰度抗锯齿，没有 ClearType 亚像素渲染**，
   所以它永远不会像原生 Windows 控件那样锐 —— 想更清楚就调大 `font_size`。
10. Linux（M7）：用 fontconfig 找 Noto CJK，找不到同样回退英文。

---

## 11. driver 侧配套改动（D1–D5）

### 11.0 硬前提：worker 可执行文件必须支持 `--worker`（本轮实测的"Start 点了没反应"）

GUI 启动 worker 用的命令行是 **`<exe> -ini <ini> --worker N`**（`worker_proc.cpp`）。
**旧版 driver（D1/D2 之前）不认识 `--worker`**，会把它当成位置参数塞进 `pos`，
于是**不走队列管理器**，而是走"单跑一个数"的老路径 → 读 stdin → 失败：

```
ecm driver starting
  mode: cpu-stub, gpucurves=0, ckpt=600s, device=0, group_order=off
No input number on stdin
```

用户看到的现象就是"环境都配好了，点 Start 就失败"，而且因为退出码非 0，GUI 会按崩溃策略
重启 3 次然后熔断（`state Error (crashed 3 times in 300 s)`）。
**这不是配置问题**：同一个 ini 用命令行直接跑（`ecm_cuda.exe` 不带参数 → 队列模式）是好的。

怎么确认手上的 exe 支不支持（一条命令，不用跑）：

```powershell
# 打印 True/False：二进制里有没有 --worker / "Worker #" 这两个字符串
$b=[IO.File]::ReadAllBytes('<path>\ecm_cuda.exe'); $t=[Text.Encoding]::ASCII.GetString($b)
"'--worker'=" + $t.Contains('--worker') + "  'Worker #'=" + $t.Contains('Worker #')
```

实测（2026-09-28）：生产目录里 9/26 那份 `ecm_cuda.exe`（10,887,680 B）**两个都是 False**，
而带 D1–D3 的构建（10,900,992 B）两个都是 True。修法就是**用当前源码重建 ecm_cuda**
（`-DECM_CUDA_ARCHITECTURES=89`、CUDA 13.3 与生产目录一致），或把 [GUI] exe= 指向新构建。

GUI 侧也加了诊断（不再只显示"一直在重启"）：`log_parse.cpp` 认出
`No input number on stdin` 后，`App` 会把原因写进状态栏 + trace，并在 Workers 表的
`Name` 列打个红色 `!`（悬停显示原因）：

```
worker 1: DIAGNOSIS: this worker executable does not understand '--worker' (it prints
"No input number on stdin"): it is older than the D1/D2 driver changes -- rebuild
ecm_cuda from the current source, or point [GUI] exe= at a current build
```

验收：`ecm_gui_log_parse_test`（`old_driver` 分类 5 项断言）+ `tools/test/test_gui_workers.ps1`
第 4 个假 worker 用 `--scenario stale-driver` 复现整条链路（诊断 + 熔断）。

> **进度（2026-09-28）**：**D1、D2 已实现并验收**（`ecm_cuda.exe` 10,899,968 B，12 cubins，0 PTX）。
> 实现文件：`src/core/ecm_queue_config.{h,cpp}`（两阶段解析：先读 `(段,键,值)`，再按
> `[Worker #N] → 全局 → 默认` 合成一份有效键值表喂给原有的映射分支，因此旧键兼容警告与
> "未知键忽略"行为都没变）、`src/core/ecm_worktodo.{h,cpp}`（段感知 + `list_workers`）、
> `src/core/ecm_driver.cpp`（`--worker N`、worker 2..N 的 `screen_<N>.log` 默认值、横幅多一行 `worker :`）。
> 验收：`tools/test/worktodo_sections_test.cpp`（27 项断言，失败非零退出）、
> `tools/test/test_worker_sections.ps1`（24 项端到端检查，全过）。
> 顺带修掉一个**真实缺陷**：队列在 `advance()` 失败时会无限重跑同一行（现在改为打印错误并中止队列，
> 见 `queue_run_one()` 与错误分支的 `break`）。D3–D5 未做。
>
> **M1 已实现并验收（同日）**：见 §14 的 M1 行与 §12 的落地结构（`src/gui/`、`localization/`、
> `ECM_BUILD_GUI`/`ECM_IMGUI_DIR`、`--selftest`、`--trace`、`tools/test/test_gui_smoke.ps1`）。
> `ecm_gui.exe` 676 KB / 无项目代码依赖。
>
> **M2 已实现并验收（同日）**：worker 进程管理（Job object + 单一管道合流 + 崩溃重启/熔断 +
> 优先级 + 重复 device 警告 + 队列状态）已接进 UI，并新增纯函数解析层（ANSI / 进度 / 事件 / 命中）。
> 验收：`--worker-selftest`（无窗口、假 worker）、`ecm_gui_log_parse_test` **43/43**、
> `tools/test/test_gui_workers.ps1` **20/20**（真窗口 + 3 个假 worker：正常/崩溃一次/挂死）、
> `tools/test/test_gui_real_workers.ps1` **23/23**（**真 ecm_cuda + 双卡**，两 worker 各消费自己
> 的 worktodo 段，日志逐条证明"没串"）。§12 的坑表补了 4 条本轮踩到的。
>
> **M3 已实现并验收（同日）**：`log_parse` 的进度字段现在有**真值断言** —— `--worker-selftest`
> 从 **39/39** 扩到 **48/48**（新增：从真实管道里解析 `pct / s-per-curve / ETA / 曲线数 / bits / GPU 形态`、
> 表格用的任务行、命中计数）；UI 侧每 worker 加了**速度历史曲线**（当时是 `ImGui::PlotLines` 画
> `curves/s` **加**进度 %；2026-09-29 重做为共用的 `App::draw_metric_plot`，纵轴改成 `s/curve`
> 并删掉进度 %，见 §8.2 / M14）。
>
> **D3 已实现并验收（同日）**：命中行补 `curve=/sigma=/param=/method=/save=`，**队列与单跑两条路径都补**；
> 为此 `Stage1RunResult` 新增 `sigmas`（逐曲线真实 sigma 数组 —— Edwards 路径每条曲线是随机 sigma，
> 用 `firstsigma+i` 会算错）。验收：`tools/test/test_hit_fields.ps1` **11/11**：
> `factor[0]=1943118631 curve=0 sigma=2504506541 param=3 method=gpu save=m677_1e6.save`（M677/B1=1e6，
> 5 条命中行；断言字段齐全、`curve` 与下标一致、`sigma-curve` 为同一常数、因子真的整除 `2^677-1`）。
> 过程中发现两个**与 GUI 无关的既有问题**，见 §16。D4/D5 未做。
>
> **M4 已实现并验收（同日）**：`src/gui/gpu_monitor.{h,cpp}`（动态加载 NVML + 采样线程 + 历史）
> + GPU 面板（每卡卡片：util/功耗与上限/SM 与显存时钟/温度/显存/节流原因 + 三条曲线）
> + 降级路径（NVML 缺失时面板显示原因、GUI 照常运行）。验收：`ecm_gui.exe --gpu-selftest` **27/27**
> （字段合理性 + 采样线程历史 + **与 `nvidia-smi` 交错交叉比对** + 假 DLL 降级），
> `tools/test/test_gui_gpu.ps1` **15/15**（含"GUI 在 `ECM_GUI_NVML` 指向不存在时仍能启动并报告原因"）。
> 交叉比对抓到两件事：`nvmlDeviceGetNumGpuCores` 是 **CUDA 核数不是 SM 数**（已改名 `cores`），
> 以及本机 4060 Laptop 会间歇报 590 W（两个工具都会，见 §8.1）。
>
> **M5 已实现并验收（同日）**：`src/gui/results.{h,cpp}` + Results 面板 + 命中行高亮/状态行提示。
> 验收：`ecm_gui_results_test` **42/42**（JSONL 追加式、同一因子合并成一行且 curve/sigma 去重、
> 表可从 JSONL 重建且逐字节一致、截断尾行容错、save 名契约与 N 表达式取指数、行格式精确匹配），
> `tools/test/test_gui_results.ps1` **27/27**：真 `ecm_cuda` 跑 M677/B1=1e6，**连跑两轮** →
> JSONL 只增不减、`results.txt` 始终"一个因子一行"、两轮命中数相加等于对象数、
> 已知因子 1943118631 的 sigma 列表随第二轮增长、每行都能由 JSONL 复算出来。

| 编号 | 改动 | 位置 | 验收 |
|---|---|---|---|
| **D1** | ini 解析 section 感知 + `--worker N`（`[Worker #N]` 覆盖全局；无段无开关 = 今天的行为） | `src/core/ecm_queue_config.{h,cpp}`、`src/core/ecm_driver.cpp` | 现有 `ecm.ini` 跑队列模式输出与改前**逐行一致**；带 `[Worker #1] device=1` 时打印的 device 变成 1；`--worker 9` 越界给明确报错 |
| **D2** | worktodo 段感知：只读/只推进自己段 | `src/core/ecm_worktodo.{h,cpp}` | 两段各自消费、互不删行；无段文件行为不变（现有回归脚本 + `ecm_worktodo_test` 仍过） |
| **D3** | 命中行补 `curve= sigma= param= save=` | `src/core/ecm_driver.cpp`（队列包装层命中打印处） | 三种后端各跑一个已知命中用例，行里四个字段齐全且与 `curve i sigma=M` 行一致 |
| **D4** | `--gpu-info [-d N] [--gpu-param 0\|2\|3] [--bits N]`：打印设备与**逐档位**的 `bits/tpb/tpi/ipb/blocks_per_sm/blocks_min/curves_min/blocks_wave/curves_wave` 后退出（仿 `--showkernel` 的"打印即退"风格） | `src/core/ecm_driver.cpp`（打印）+ `kernels/cuda/cgbn_stage1.cu`（档位表与占用率查询）+ `src/cuda/ecm_cuda_backend.cu` / `src/opencl_backend_glue.cpp`（后端钩子） | **已完成（2026-09-29）**：`tools/test/test_gpu_info.ps1` **52 项**（格式全 `key=value`、退出码、档位递增、`ipb == tpb/tpi`、`curves_min == blocks_min*ipb`、`curves_wave == blocks_wave*ipb`、`blocks_min == sm_count`、`--bits N` 选档与运行路径 `CGBN<tpi, bits>` 实测一致、`-d 99`/超大 `--bits` 干净失败、**零副作用**（不建 ini、不写文件）、ini 覆盖顺序、OpenCL 打 `not_applicable` 且退出 0）。 |
| **D5** | 队列模式下若 `NumWorkers > 1` 且本段 `device=` 与更低编号段相同 → 打一行警告 | `src/core/ecm_driver.cpp` | 构造重复 device 的 ini，headless 跑一次即出现该行；不改变退出码 |

### 11.1 D4 的输出格式（契约）

```
gpu_info=1                       # 或 gpu_info=not_applicable（OpenCL 构建）
backend=CUDA/CGBN  device=0  name=NVIDIA GeForce RTX 4070 Ti  sm_count=60  cc=8.9
gpu_param=0  fold=0  carry_bits=6  picked=0  tier_count=31  ini=<路径|->  worker=1
tier bits=128 tpb=128 tpi=4 ipb=32 blocks_per_sm=10 blocks_min=60 curves_min=1920 blocks_wave=600 curves_wave=19200
tier bits=192 ...（升序）
```

* 前 13 行是**一行一个** `key=value`；档位行以**唯一没有 `=` 的 token `tier`** 开头，后面是空格分隔的
  `key=value`。解析器（`src/gui/worktodo_gen.cpp: parse_gpu_info`）就按这两条规则读。
* `--bits N` 时只输出 kernel 会选中的那一档（`bits >= N + carry_bits` 的最小档，与内核选择逻辑同源），
  并置 `picked=1`；没有任何档位放得下则退出码 1。
* `blocks_min` = 每 SM 一个块（低于它内核会警告"some SMs idle"）；`curves_min` = `blocks_min × ipb`
  ——**就是内核警告行让我们"raise -gpucurves to about N"里的那个 N**。
  `blocks_wave`/`curves_wave` = 寄存器允许的块槽位与其曲线数（内核注释里"整数倍可避免半波"的那个量）。
  `fold_*` 只在折叠构建（`-DECM_MERS_FOLD=1`）里出现。
* **零副作用**：`--gpu-info` 会 `cudaSetDevice`（占用率与计算能力相关，必须问对卡），但**不启动内核、
  不读检查点、不写任何文件、不建 ini**；`-ini` 存在时只读它来决定生效的 `device`/`gpu_param`
  （CLI 优先）。因此可以随时在别的 worker 跑着的时候执行。
* 两处**内核警告文案修正**（原本打印的是 block 数 `sm_count`，喂给 `gpucurves` 会欠占用）：
  普通构建改成 `curves = sm_count × ipb` 并说明"整数倍可保持整波"；折叠构建改成
  `2 × sm_count × ipb`。见 `docs/DEV_ECM_WORKTODO.md` §5.4。

---

## 12. 目录与构建

**已落地（M1–M5；**GUI 是 `src` 里的一等公民，不是 `tools/` 下的外置工具**）**：

```
src/gui/
├── CMakeLists.txt          # 目标 ecm_gui（WIN32）+ ecm_gui_imgui（vendored ImGui）
│                           # + ecm_gui_fake_worker / ecm_gui_log_parse_test / ecm_gui_results_test
├── main_win32.cpp          # WinMain + D3D11 设备/交换链 + ImGui 初始化 + 帧循环 + --selftest + --trace
├── app.{h,cpp}             # 应用状态、面板（Workers 表 / GPU / 每 worker 输出 / 详情）、布局持久化、生效配置
├── ini_file.{h,cpp}        # 结构化 ini 读写（保注释/保序/未知键保留 + 原子替换 + 一代 .bak）
├── localization.{h,cpp}    # XML 文案加载 + 回退 + 缺键统计 + 语言枚举
├── log_parse.{h,cpp}       # ANSI 剥离 / 时间戳 / 进度行字段 / 事件与错误分类 / 管道行切分（纯函数）
├── gpu_monitor.{h,cpp}     # NVML 动态加载 + 采样线程 + 历史（可选依赖，缺失即降级）
├── results.{h,cpp}         # results.json.txt（追加式 JSONL）+ results.txt（合并表，可重建）
├── results_test.cpp        # 上述两个文件的单测（42 项断言）
├── gpu_selftest.{h,cpp}    # --gpu-selftest：字段合理性 + 采样线程 + nvidia-smi 交叉比对 + 降级
├── worker_proc.{h,cpp}     # worker 进程：Job object、单一管道读写线程、重启与熔断、优先级
├── worker_selftest.{h,cpp} # --worker-selftest：无窗口驱动上面这套（假 worker 场景）
├── platform.h              # 三层边界声明（Platform / WorkerProc / GpuMonitor）
├── platform_win32.cpp      # 平台实现（exe 目录、路径、系统字体查找 + DPI、资源管理器、命令行解析）
├── fake_worker.cpp         # 假 worker：脚本化输出/崩溃/挂死/派生孙进程（测试夹具）
├── log_parse_test.cpp      # 解析层单测（43 项断言）
└── localization/
    ├── english.xml             # 基线（50+ 条文案）
    ├── chineseSimplified.xml   # 简体中文（0 缺键）
    └── README.md               # 贡献流程 + 编码约定
third_party/imgui/          # vendored docking 分支（见其 README.md）
```

> **目录位置**：GUI 源码在 `src/gui/`，和 `src/core`、`src/cuda` 平级，**不是** `tools/` 下的外置脚本式工具。
> `localization/` 也随源码在 `src/gui/localization/`（早先在仓库根，本轮并入源码树），
> 构建时 POST_BUILD 拷到 exe 旁边（GUI 只认 exe 旁边的这一份）。

CMake：

```cmake
option(ECM_BUILD_GUI "Build the ImGui front-end (Windows only, needs third_party/imgui)" ON)
set(ECM_IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui" CACHE PATH "...")
# NOT ECM_BUILD_GUI      -> "gui: disabled"
# not WIN32              -> "gui: skipped (Windows-only for now)"
# 缺 ${ECM_IMGUI_DIR}/imgui.h -> "gui: skipped" + 提示
# 三者都让 configure 正常结束，且不会出现 ecm_gui 目标（实测过：见 §14 M1 行）
```

### 怎么自己编译（一条命令）

**推荐**（不用先开 VS 开发者命令行）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\dev\build_gui.ps1
# 常用变体
... build_gui.ps1 -Selftest                     # 编译 + 跑 --selftest
... build_gui.ps1 -Clean -Selftest              # 从零重建
... build_gui.ps1 -Reconfigure                  # 换了依赖路径 / 缓存报错时
... build_gui.ps1 -BuildDir build_gui_test      # 换个构建目录
... build_gui.ps1 -Generator "Visual Studio 17 2022"   # 想要 IDE 工程
```

脚本做四件事：找 `vcvars64.bat`（VS 18/17/16，Community 或 BuildTools）→ 必要时 configure
（自动探测 GMP / OpenSSL / vendored ImGui，可用 `-Gmp` / `-OpenSslRoot` / `-ImGuiDir` 覆盖）→
在带开发者环境的 shell 里 `cmake --build` 四个目标 → 报告产物并（可选）跑 `--selftest`。

**为什么直接 `cmake --build build_gui --target ecm_gui` 会失败**：

```
no such file or directory
CMake Error: Generator: build tool execution failed, command was: nmake -f Makefile /nologo ecm_gui
```

本项目的构建目录用的是 **NMake Makefiles** 生成器，而 `nmake` 只存在于 Visual Studio 的开发者
环境里；普通 PowerShell 里没有它。两种解法：

```powershell
# 1) 用脚本（推荐）
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\dev\build_gui.ps1
# 2) 或者自己进开发者环境再手动跑（与脚本等价）
& "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build_gui -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DECM_BUILD_TOOLS=OFF `
      -DGMP_INCLUDE_DIR=third_party/gmp-zen3/dist/include -DGMP_LIBRARY=third_party/gmp-zen3/dist/lib/gmp.lib `
      -DOPENSSL_ROOT_DIR=D:/code/vcpkg/installed/x64-windows
cmake --build build_gui --target ecm_gui
```
（或者直接用“x64 Native Tools Command Prompt for VS”开一个终端。）

**一句话记法**：GUI 的构建目录是 NMake，所以*任何* `cmake --build build_gui` 都必须在一个已经
`vcvars64.bat` 过的 shell 里跑。

`ecm_gui` 只依赖 `imgui` 源码 + 系统库（`d3d11 dxgi d3dcompiler dwmapi shell32` + 未来动态加载的
`nvml.dll`）；**不**链接 `ecm`/`ecm_cuda` 的任何代码或 GMP/OpenCL。产物落在构建根
（`ecm_tool_output_root`），并把 `localization/` 目录拷到 exe 旁边。

运行/自测：

```powershell
cmake --build build_gui --target ecm_gui ecm_gui_fake_worker ecm_gui_log_parse_test ecm_gui_results_test
build_gui\ecm_gui.exe                 # 正常启动
build_gui\ecm_gui.exe --selftest      # 无窗口自测：ini/本地化/字体/ImGui（36 项）
build_gui\ecm_gui.exe --worker-selftest   # 无窗口自测：worker 进程管理（48 项）
build_gui\ecm_gui.exe --gpu-selftest      # 无窗口自测：NVML 监控 + 与 nvidia-smi 交叉比对（27 项）
build_gui\ecm_gui_log_parse_test.exe      # 解析层单测（43 项）
build_gui\ecm_gui_results_test.exe        # results 双文件单测（42 项）
build_gui\ecm_gui.exe -ini <path> --trace
                                      # --trace 写 <exe 目录>\ecm_gui_trace.log：
                                      # start / frame N / WM_CLOSE / loop left / saving / saved
                                      # 窗口生命周期：WM_SIZE / WM_ACTIVATE / WM_SHOWWINDOW / WM_DESTROY
                                      # 字体自证：font: <路径> at 22.5 px (…, snap=1) [startup|language change]
                                      #           + measured … cjk_ok=1 map=11 baked=11 negctl=00
                                      # 本地化自证：localization: … keys=/missing= + 一条真实译文样本
                                      # 布局自证：layout: <面板> ###<id> x=… y=… w=… h=… + viewport pos/size
                                      # GPU 历史自证（每 ~20 个样本一行）：
                                      #   gpu: history dev=0 samples=44 util=17 distinct power=22 distinct
                                      #                   clock=21 distinct flat=0
                                      #   gpu: history dev=0 ranges power=9.2..102.5 plot=9.1..102.6
                                      #                   clock=632..2774 plot=418..2988
                                      # 以及每个 worker 的生命周期：
                                      #   worker 1: autostart pid=1234 cmd=...
                                      #   worker 1: state Running / Restarting / QueueEmpty / Error
                                      #   worker 1: restart #1
                                      #   worker 1: DIAGNOSIS: …（旧 driver 不认识 --worker，见 §11.0）
build_gui\ecm_gui.exe -ini <path> --switch-language chineseSimplified --trace
                                      # 诊断开关：等于在 Language 菜单里切一次语言（脚本用）
                                      # 环境变量 ECM_GUI_CJK_FONT=<path|none> 可覆盖 CJK 字体查找
powershell -File tools\test\test_gui_smoke.ps1         # 真窗口冒烟（57 项）
powershell -File tools\test\test_gui_workers.ps1       # 假 worker 监管 + 表格几何（34 项，含旧 driver 诊断）
powershell -File tools\test\test_gui_gpu.ps1           # NVML 面板 + 降级（15 项）
powershell -File tools\test\test_gui_results.ps1       # results 双文件（27 项，真 ecm_cuda，两轮）
powershell -File tools\test\test_gui_real_workers.ps1  # 真 ecm_cuda + 双卡（23 项；需要 GPU）
powershell -File tools\test\test_gui_cjk_pixels.ps1    # 中文渲染像素验收 + 救回/回退/运行中切语言（31 项）
powershell -File tools\test\test_gui_gpu_curves.ps1    # 真 worker 压卡：功率/频率曲线确实在变（15 项）
powershell -File tools\test\test_gui_exit_checkpoint.ps1  # 退出确认 + 退出前必写检查点（27 项，真驱动）
powershell -File tools\test\test_gui_all.ps1           # 一键跑上面全部（含 headless 自测与单测）
```

`--trace` 是刻意留在发布二进制里的：窗口进程被脚本拉起时，这是唯一能看见"它到底走到哪一步"的办法
（M1 靠它定位关闭时挂死，M2 靠它做无截图的验收）。它用 `_fsopen(..., _SH_DENYWR)` 打开：**写者唯一、
读者可共享**，这样测试脚本能在 GUI 运行时实时读它。

**实现备忘（M1/M2 踩到的坑，别重踩）**：

| 坑 | 事实 |
|---|---|
| ImGui 1.93.0 WIP（docking）的字体 API | 已进入**动态字体**时代：`AddFontFromFileTTF(path, size)` **不要**再传 `GetGlyphRangesChineseFull()`（那些 API 被 obsolete 标记，加上 `IMGUI_DISABLE_OBSOLETE_FUNCTIONS` 会直接编译失败）；也不再需要预先 `GetTexDataAsRGBA32`。好处：不再为 2.1 万汉字预烘焙图集 |
| 关闭时挂死 | 必须在 `ImGui::DestroyContext()` **之前**抓布局（`SaveIniSettingsToMemory()` 依赖上下文）。顺序搞反 = 关闭时静默挂死（进程不退出、无报错） |
| DPI 坐标系 | `ImGui_ImplWin32_EnableDpiAwareness()` 之后 GUI 用**物理像素**；非 DPI-aware 的外部脚本读到的是虚拟化坐标（150% 缩放下 1500 ↔ 1000）。写验收脚本时要乘 `GetDpiForWindow()/96` |
| 关闭窗口的测试方式 | `Process.MainWindowHandle` 在多视口下可能指向 ImGui 的辅助窗；可靠做法是 `EnumWindows` 找本进程 **class = `ecm_gui`** 的窗口再 `PostMessage(WM_CLOSE)` |
| 抓不到 GUI 的 stdout | 给 GUI 子系统进程 `AttachConsole(ATTACH_PARENT_PROCESS)` + `freopen("CONOUT$")` 会**顶掉继承来的管道**，脚本就什么都收不到 ⟹ 只在 `GetStdHandle(STD_OUTPUT_HANDLE)` 无效时才 attach |
| 测试脚本里的 trace 读取 | GUI 运行中 trace 文件是打开的 ⟹ 写端必须 `_SH_DENYWR`（否则读者拿到"文件被占用"），读端用 `FileShare.ReadWrite` 打开 |
| PowerShell 数组字面量里的 `+` | `@('a', 'x = ' + $v, 'b')` 里**逗号优先级高于 `+`** ⟹ 变成两个元素、两行 ini（本轮因此让 `exe = <路径>` 断成两行，GUI 只好回退 PATH 查找并报 `CreateProcessW failed (2)`）。**每个拼接都要加括号** |
| 测试夹具的 save 名 | `ECMSTAGE2=` 行的 B1 是从**最后一个 `_` 到 `.save`** 的 token 抽的 ⟹ `m991_1e4_w1.save` 会被判为"取不到 B1"并标 `# ERROR`（测试用不同指数 `m991_1e4.save` / `m997_1e4.save` 区分两个 worker）|
| GPU 存档落点 | GPU stage-1 的 `.save` 写在 **ecm_cuda 的 exe 目录**（不是 `tmp_dir`），而且**命中因子时本来就不写存档** ⟹ 验收脚本别把"有没有 .save"当成功判据，看日志更可靠 |
| 宿主窗口要用 SetWindowPos 钉住 | 全视口 dockspace 宿主窗口带 `NoMove|NoResize`，而 ImGui 对这类窗口**首帧之后不再接受** `SetNextWindowPos/SetNextWindowSize`；首帧的视口尺寸还可能没稳定 ⟹ 必须在 `Begin` 之后显式 `SetWindowPos/SetWindowSize(vp->WorkPos/WorkSize)`，并把默认布局的构建延后到第 3 帧（实测不这么做面板整体偏移 131,148）|
| 主视口原点是桌面坐标 | 多视口下 `GetMainViewport()->Pos` 是**客户区在桌面坐标系的位置**（本机 `(131,125)`，即窗口边框+标题栏），不是 (0,0) ⟹ 布局断言必须用 `Pos..Pos+Size` 这个矩形，别假设原点为零（我第一版因此误判"面板超出视口"）|
| PowerShell 函数返回值被解包 | 返回**单元素数组**的函数在赋值处会被解包成标量，于是 `$x[0]` 取到的是字符串的**第一个字符**（本轮真踩到：`$lines[0]` 变成 `"M"`，导致"格式不对"的假失败）。调用处一律写 `@(Get-Thing ...)`，别依赖自动解包 |
| **最小化后唤不醒**（用户实测反馈） | 帧循环原来是"先判 `IsIconic` 就 `return`，再 pump 消息" ⟹ 最小化期间**一条消息都不派发**，`WM_SYSCOMMAND/SC_RESTORE` 进不来，窗口只能从任务栏右键关掉。**必须 pump 消息在最前面**，`Iconic` 判断放它后面（`new_frame()` 开头 `PeekMessage`→`TranslateMessage`→`DispatchMessage` 循环） |
| **最小化时还在渲染** = 0xC0000005 | 光把 pump 提前还不够：最小化时交换链是 0×0，仍然 `ImGui::Render()` + `Present()` 会在驱动里炸（退出码 `-1073741819`）。**最小化时整帧跳过**（`g_skip_frame`：`Sleep(10)` + `return`，既不 `NewFrame` 也不渲染）。回归护栏 = 冒烟测试步骤 [1c]："最小化状态下 post `WM_CLOSE`，进程必须退出 0"（消息泵没通就会挂死在那里）|
| `new_frame()` 里不能有"提前 return true" | 曾有一个 `if (resized) return true;` 分支（不调 `NewFrame`）而循环体照样往下跑 `ImGui::Render()` ⟹ 在**没有当前帧**的情况下渲染，直接崩。`new_frame()` 返回 `true` 就**必须**已经建好帧 |
| `ResizeBuffers` 前必须先解绑 | 交换链 resize 时若 back buffer 还被 `OMSetRenderTargets` 绑着，`ResizeBuffers` 返回失败、旧 RTV 又被放掉了 ⟹ `rtv == nullptr` 后每帧空指针或用错尺寸。现在 `cleanup_render_target()` 先 `OMSetRenderTargets(0, nullptr, nullptr)` + `Flush()` 再 `Release()`；`ResizeBuffers` 失败则整条交换链重建；`render_frame()` 开头 `rtv/swap_chain` 为空直接返回 |
| 曲线"是一条直线" | 用 `PlotLines` 的自动量程时，空闲态的功率/频率几乎恒定（同值或 ±0），ImGui 的 `0,0` 自动范围就把差异压成一条线。改为**按当前窗口内观测到的最小/最大值，上下各留 10 % 余量**再手动 `SetupAxisLimits`，这样微小波动也看得见（利用率本身波动大，改前改后都正常）|
| **"中文有没有显示"不能用宽度判断** | `ImFontBaked::FindGlyph()` 在缺字时会**静默换成 U+FFFD fallback**，而那个方块**也有宽度**（实测中文两字 35 px vs 拉丁 `WW` 36 px，看着完全"正常"）。要判断必须用 `ImFont::IsGlyphInFont()`（TTF 里有没有）+ `FindGlyphNoFallback()`（有没有真的烤进当前字号的 atlas）是否为非空，并配一个**私用区码位（U+E123）当反例**证明判据有效 |
| **tofu 的像素特征不是"空心"** | 直觉上以为 fallback 是空框、中心没墨 —— 实测**错**：fallback 中间也有笔画（`midInk>0`）。真正的特征有两条：① 宽度只有 **0.43 em**（真 CJK 是 0.75 em）；② **每个缺字的格子完全一样**（同一套 w/h/ink）。`tools/diag/text_ink_probe.ps1` 就是按这两条写的 |
| 抓图要按"亮度差"找文字 | 深色主题下文字比背景**亮**，"墨 = 暗于阈值"会把整个窗口算成墨（我第一次就写反了，得到"最上面 45 行全是墨"）。`text_ink_probe.ps1` / `row_ink_profile.ps1` 改成先用**全图亮度直方图的众数**当背景，再按 `|lum - bg| > Delta` 判墨；`Delta` 还得够大（140），否则菜单栏底色（`lum≈66` vs 窗口底 `lum≈16`）也算墨 |
| **字体只在启动时挑一次** | 启动时按当时的语言选字体；用户在 Language 菜单里切成中文时若不重新挑，就会拿拉丁字体去画中文 ⇒ 满屏 `???`（这就是"我这边是 ???、你测试却正常"的原因：测试的 ini 里一开始就写着 `language = chineseSimplified`）。现在 `reload_localization()` 置 `font_reload_requested_`，帧循环在**两帧之间**重跑 `apply_ui_font()` |
| **"能不能画这个字"只能问 atlas** | `ImFont::IsGlyphInFont()` + `ImFontBaked::FindGlyphNoFallback()` 才是判据；`CalcTextSize` 量出来的宽度对 fallback 方块同样成立（实测中文两字 35 px、拉丁 `WW` 36 px）。另外 fallback 是 `?` 还是 U+FFFD 取决于字体：内置位图字体没有 U+FFFD，于是显示成 `???` —— 看到 `???` 就该怀疑"字体不支持"而不是"本地化没加载" |
| **`.ps1` 文件没有 BOM 时，中文正则一定匹配不上** | 仓库里的 `.ps1` 都是 UTF-8 **无 BOM**，而 Windows PowerShell 5.1 把无 BOM 的脚本文本按 **ANSI(GBK)** 解码 ⇒ 脚本里写 `-match "工作线程"` 时，模式里的汉字已经乱了，永远不匹配（本轮 `test_gui_cjk_pixels.ps1` 的 [D] 因此假失败）。对策：**别在 .ps1 里放中文模式**，用字符类表达同样的意思（`.ps1` 里的 `[^\x20-\x7E]` = "至少有一个非 ASCII 字符"），或用 `\uXXXX` 转义 |
| **旧 driver 让 Start "看起来没反应"** | 见 §11.0：`--worker` 不被识别 ⇒ 走单跑路径 ⇒ `No input number on stdin` ⇒ 退出 1 ⇒ GUI 重启 3 次熔断。判断：二进制里搜 `--worker` 字符串；GUI 现在会自己诊断（状态栏 + trace `DIAGNOSIS:` + 表格红 `!`） |
| **别从名字推导本地化键** | 状态标签原先用"把 `Stopped` 插下划线变小写"推出 `state_stopped`，但首字母大写会多插一个 `_` ⇒ 键是 `state__stopped`，XML 里没有 ⇒ `Localization::t()` 按约定回退成**字面量** `workers.state__stopped`，用户就在表里看到这串东西（实测 BUG）。现在键是**显式表**（`state_key()`，声明在 `app.h`），且 `--selftest` 逐个状态 × 两种语言断言"能解析、不是回退值" |
| **表格列会被长内容撑宽** | `ImGuiTable` 的固定宽度列只保证"至少这么宽"，内容更长时仍会自己变宽 ⇒ 一个 ~80 字符的 worktodo 列把整表撑到几千像素，后面的进度条/ETA 列整个跑到面板外（用户："进度条，eta 等均不显示"）。做法：长文本移到**表格行的第二行**（跨列 + 换行，`PushClipRect` 放宽裁剪），其余列用示例文本算固定宽度，并让进度条列 `WidthStretch` 吃掉剩余宽度；`--trace` 打出每列实测 x/宽度 + `fits`，测试据此断言 |
| 测试里取 trace 要取**最后**一条 | 每帧都会打的诊断行（例如 `table: workers …`）在日志里有很多条：`[regex]::Match` 只给**第一条**（往往是布局还没建好的最初几帧，数字完全不对）。用 `[regex]::Matches(...)` 再取 `[Count-1]`；同时把这类行**限频**（`frame_counter_ % 60`），否则日志被它刷满 |
| **跑测试脚本时别把 GUI 当 driver 传** | `tools/test/*.ps1` 的参数并不统一：GUI 类脚本要 `-Exe <ecm_gui.exe>`，而 **driver 类脚本（如 `test_worker_sections.ps1`）的 `-Exe` 要的是 `ecm_cuda.exe`**。传错不会报错——`ecm_gui.exe -ini … --worker 1` 只是**开一个什么都不做的窗口**，于是"某个窗口几分钟没动静"（2026-09-29 用户手动关掉的就是它）。现在 ① `test_gui_all.ps1` 每项都显式声明要哪种 exe；② GUI 收到 `--worker` 会打印 `warning: --worker is a driver flag …` 并写进状态栏 |
| **`ImGui::Text` 的格式串与参数必须一一对应** | 这一行把 `%s/%u` 的配对错位了：`%s` 收到一个整数（`clock_mem_mhz`），`vsnprintf` 就把它当**指针**去解引用 ⇒ **整个 GUI 0xC0000005 崩掉**（2026-09-29 实测，用户看到"该内存不能为read"弹窗）。它是**潜伏**的：显存频率为 0 时 MSVC 的 `%s` 打印 `(null)` 不崩，NVML 一开始报非 0 值就立刻崩。**扫一遍**：`tools/diag/check_printf_calls.ps1`（数格式串里的转换符 vs 实参个数，当前 0 处不符）。**首次崩溃怎么定位**：GUI 会为最初几帧打 `draw: <面板> ok`，trace 的最后一条就是崩在哪个面板 —— 现在 `test_gui_smoke.ps1` / `test_gui_gpu.ps1` 都断言"七个阶段全过 + `frame 1 rendered`"，所以"第一帧就崩"再也不能蒙混过关 |

| **PowerShell 5.1 里拿进程退出码只有 `&` 可靠** | `Start-Process -PassThru` 的 `ExitCode` 在本机**总是空**（`WaitForExit()` 之后也一样），而且 `-ArgumentList @()`（空数组）会直接抛异常；`Start-Job` 也**无法**把 `$LASTEXITCODE` 带出 job。结论：跑测试用 `& exe @args *>&1 \| Tee-Object` + `$LASTEXITCODE`；硬超时改成**看门狗 job**（到点只杀 `tools/test/_run` 下的沙箱进程），而不是去杀测试进程 |
| **表格单元里画"跨列"文字的两个坑** | 任务行要跨整表，实测两次才弄对：① 单元自己的裁剪矩形是**那一列**，`PushClipRect(min,max, intersect=true)` 是**求交**，等于没放宽 ⇒ 必须传 `false`；② `PushTextWrapPos(x)` 的 x 是**窗口局部坐标**，传屏幕坐标等于悄悄关掉换行（于是只显示头几个字）。判据：trace 的 `task_wrap_w`（实测 1201 px，未修前约等于首列宽度 28 px）|
| **"固定宽度列"仍会被内容撑宽** | `ImGuiTable` 的 `WidthFixed` 只是"至少这么宽"，内容更长时列还会长 ⇒ 长任务列能把整表撑到几千像素、把后续列顶出面板。做法：长文本移到行的第二行；其余列用样本文本算宽；进度条列 `WidthStretch`；并且**显式给进度条留最小值**（先压缩 Name 列，必要时再压到 64 px 下限），因为"让 ImGui 自己分配"在窄面板上会把进度条挤到 45 px（被测试抓到 `fits=0`）|
| **边框画在内容之后** | 想"跨列一行文字"时，`Borders` 的**纵线会横穿文字**（ImGui 在 `EndTable()` 里最后才画边框，任何"先铺一层背景盖掉"的做法都无效）。要么整个表改用 `BordersInnerH | BordersOuterH`（本项目选择），要么把文字挪到表格外。没有按行关闭边框的 API |
| **driver 的输出是相对 CWD 的** | `get_checkpoint_filename()` 只返回**文件名**（`.ecm_ckpt_<nbits>_<hex>.dat`），`.save`/指数缓存同理 ⇒ 子进程的工作目录决定它们落在哪。GUI 过去不设 `lpCurrentDirectory`，于是从别处启动时这些文件落到 GUI 的 CWD（用户目录里那些 `.save` 就是这么来的）。现在 `WorkerSpawn::working_dir` = driver exe 所在目录。**排查提示**：找不到检查点/存档时，先看 GUI 是"从哪个目录"启动的 |

---

## 13. 决策记录（本轮共识，含被推翻的早期方案）

| # | 决策 | 备注 |
|---|---|---|
| 1 | UI 库 = ImGui；常驻内存 ≤50 MB | 用户选定 |
| 2 | 平台策略 = Windows 先行 + 三层抽象（`Platform`/`WorkerProc`/`GpuMonitor`），Linux 留口，Android 排除 | 用户选定 |
| 3 | GUI 职责 = 监管 + 生成/维护 worker 配置；不内置 worktodo 语义 | 用户选定；M6 再加可视化生成 |
| 4 | 布局 = docking + viewports（内嵌可停靠 + 可弹出） | 用户选定 |
| 5 | **（推翻 #3 早期的"每 worker 一份 driver ini"）** 配置 = 统一 `ecm.ini` + `--worker N` + `[Worker #N]` 覆盖 | Q5/Q6 后定型 |
| 6 | worktodo = 单一文件 + `[Worker #N]` 段（无段 = worker1），为不同 GPU 保留不同 `gpucurves` | 用户选定 |
| 7 | worker 生命周期 = 崩溃自动重启（5 s 退避）+ 5 分钟 3 次熔断；GUI 关闭即停（带确认） | 用户选定 |
| 8 | 重复 `device=` → 警告不阻止（GUI 侧检测 + driver 侧 D5 打印） | 用户选定 |
| 9 | 日志 = 分层（事件进面板，进度喂状态列）+ ANSI 解析 + GUI 自绘进度条 | 用户选定 |
| 10 | 状态数据源 = **纯日志解析**（不做 `--status-file`；若实测脆弱则按 §16 升级） | 用户选定 |
| 11 | results = `results.json.txt`（JSONL 追加）+ `results.txt`（合并表，sigma 列表） | 用户选定 |
| 12 | 语言 = 双语 + 外置 XML（UTF-8 无 BOM）+ 热重载；字体 = **运行时挑系统字体**（CJK 走 `msyh.ttc` 等，拉丁走 `segoeui` 等，都不打包进仓库）+ **字号按 DPI 自动放大**（`[GUI] font_size=auto` = 15 px × 缩放）+ 回退英文 | 用户选定 + 本轮反馈修正（"默认字体过小""中文不显示"） |
| 13 | 布局持久化 = 关掉 `imgui.ini`，写进 `[GUI]`（4 数矩形 + 停靠串；不采用 prime95 的 9 参数格式） | 我方决定，已说明理由 |
| 14 | imgui = **vendored** docking @ `64944b45…`（MIT，随仓库分发，见 `third_party/imgui/README.md`） | 用户下载 + 我方拷贝 |
| 15 | `work_manager.ps1` **已废弃**（被 driver 内置队列取代）；仅可作为测试夹具，不作为生产路径 | 用户确认 |
| 16 | 交接 Prime95 用 **`ECMSTAGE2=` 原样追加**（保留 AID 与已知因子），**不是** feeder 的 `ECM=` 形式；文件级唯一真相 = `p95_worktodo_path` | 用户明确纠正；`commonc.c:2936` 证实 Prime95 认这个关键字 |
| 17 | 交接**命中因子也照样交付**（stage 2 仍需 GCD 并上报，我们无法代它上报） | 用户确认 |
| 18 | 交接功能放**主程序**（`src/core/p95_transfer.cpp`），`ecm_p95feeder` 保持独立、**一行不改** | 用户明确要求 |
| 19 | 失败处理 = 锁 + 原子替换 + pending 文件 + **绝不阻塞任务**；GUI 用**显著颜色通知条**（红/黄/绿/灰），含 pending 也要通知 | 用户明确要求（"包括 pending 都要通知 GUI 并使用显著颜色"） |
| 20 | `progress_log_seconds`：**管道每次、文件每 n 秒**（默认 60，0 = 文件不写进度行），100 % 行总写 | 用户明确要求 |
| 21 | 生成器的推荐曲线 = **`n × sm_count × ipb`**（`n` = 块/SM，默认 2），`ipb` 取该行 N 的档位；GUI **不引 GMP**，位宽用算术估算 | 用户确认；避免复制 kernel 的 TPI 表 |
| 22 | 生成器**"生成预览"与"追加到 worktodo"分成两个按钮**，追加前重新校验 mtime；文件**拖放**只写文档 TODO | 用户明确要求 |
| 23 | 本轮的三个新 ini 键（`progress_log_seconds`、`p95_worktodo_path`、`p95_add_workers`）**只从 ini 读，不开 CLI**；长期方向是"很少用的 CLI 逐步下线、改为 ini-only" | 用户长期意图，见 `docs/DEV_ECM_INI.md` §4 |
| 24 | 折线图统一走共用控件 `App::draw_metric_plot`（圆角卡片 + 网格 + 面积 + 单位 + 参考线 + 悬停读数），**替换掉全部 `ImGui::PlotLines`**；worker 图纵轴改为 **`s/curve`**（越小越快）并**删掉进度 % 曲线**；占用率图固定 `0..100`，功耗/时钟按观测窗口 ±12 % 自动 | 用户明确要求（"删去进度%""曲线/秒 改为秒/曲线""优化所有折线图使其更美观 易读"）；曲线 trace 供脚本断言，见 §8.2 |

---

## 14. 里程碑与验收

每个里程碑的验收都必须是可复现命令 + 明确判据（仓库惯例）。

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M1 骨架** ✅ | `ecm_gui` 目标、Win32+DX11+docking/viewports、布局持久化写回 `[GUI]`、语言系统（XML + 回退 + 热重载）、CJK 系统字体加载、面板骨架（Workers 表 / GPU / 每 worker 输出 / 详情） | **已完成（2026-09-28）**：① `ecm_gui.exe --selftest` **36/36**（ini 往返保注释/保序/未知键保留 + .bak、本地化基线 50 条 + 中文覆盖 + 0 缺键 + 回退、ImGui docking/viewports 标志 + 字体图集）；② `tools/test/test_gui_smoke.ps1` **47/47**（真窗口出现 → `WM_CLOSE` → 退出码 0 → `[GUI] window=`/`dock_layout=` 落盘 → driver 键与注释未动 → 第二次启动几何完全一致）；③ 缺 imgui 时 configure 正常且**无** `ecm_gui` 目标；`-DECM_BUILD_GUI=OFF` → `gui: disabled` 且无目标；④ 目标列表实测含 `ecm_gui`/`ecm_gui_imgui`，`ecm`/`ecm_cuda`/`cgbn_opencl`/`opencl_ecm_entry` 不受影响 |
| **M2 进程与队列** ✅ | ini 读写（保序原子）、worker 生命周期、管道合流、重启熔断、重复 device 警告、队列状态显示 | **已完成（2026-09-28）**：① `--worker-selftest` **48/48**：单一管道抓到全部输出、进度行计数、ANSI 已剥离、START/命中/Checkpoint 事件层、优先级类别生效（`below_normal`=0x4000）、干净退出→`QueueEmpty` 且不重启、崩溃→恰好一次重启（间隔 ≥ 退避）、连续崩 3 次→熔断（重启 2 次后停住、`start()` 拒绝）、`stop()` 经 Job object **连孙进程一起杀掉**、挂死进程 `stop()` 后不重启；② `ecm_gui_log_parse_test` **43/43**；③ `tools/test/test_gui_workers.ps1` **20/20**（真窗口 + 3 个假 worker = 正常/崩溃一次/挂死：autostart、Running→QueueEmpty、崩溃→Restarting→restart #1→QueueEmpty、关闭时停止仍在跑的 worker、无残留进程、ini 键与注释未被破坏）；④ `tools/test/test_gui_real_workers.ps1` **23/23**：**真 `ecm_cuda.exe` + 双卡**（device 0 = M991，device 1 = M997，B1=1e4，8 曲线），两 worker 同时消费**同一个 worktodo 的两个段**：两行都被成功移除、无 `# ERROR`、段头保留、`finished` 两条、两份 `screen_<N>.log` 各自只出现自己的指数与自己的 device（逐条断言"没串"）、关闭后无残留进程 |
| **M3 日志与状态** | ANSI 解析、分层日志、进度解析、表格列、GUI 进度条、详情面板（生效配置） | ① `src/gui/tests/` 里样本行（含两种进度前缀、ANSI、`# ERROR`）解析断言全过；② 真跑时表格 `%`/`s/curve`/ETA 与日志行数值一致 |
| **M4 GPU 监控** ✅ | NVML 动态加载 + 采样线程、每卡卡片（util/功耗与上限/SM 与显存时钟/温度/显存/节流原因 + 三条曲线）、全机合计功耗、优雅降级 | **已完成（2026-09-28）**：① `--gpu-selftest` **27/27**：NVML 加载、每字段合理性、采样线程 ≥2 个时间有序样本、**与 `nvidia-smi` 交错交叉比对**（`-i <卡>` 三次取范围；实测吻合：210/405 MHz、2595/10501 MHz、8.6 vs 8.65 W）、假 DLL 被拒且有原因；② `tools/test/test_gui_gpu.ps1` **15/15**：GUI trace 报 `gpu: nvml ok`、逐卡名称与 `nvidia-smi` 一致（device 数 2/2）、`ECM_GUI_NVML` 指向不存在时 GUI **仍能启动**、trace 报 `gpu: NVML unavailable` 并干净退出 |
| **M5 results** ✅ | JSONL 追加 + 合并表、命中高亮/通知 | **已完成（2026-09-28）**：① `ecm_gui_results_test` **42/42**：同一因子 4 次命中（含重复的 curve+sigma）只产生**一行** `M677 has a factor: 1943118631 (ECM curves 3,4,9, B1=1e6, param 3, gpu, Sigmas=[...], hits=4)`、JSONL 保留全部 4 条、两个因子两行、重建后逐字节一致（只差 `updated` 时间戳）、截断尾行重放被忽略；② `tools/test/test_gui_results.ps1` **27/27**：真 `ecm_cuda` 跑 M677/B1=1e6 **两轮**，JSONL 只增、表仍"一因子一行"、命中数相加一致、已知因子的 sigma 列表增长、每行可由 JSONL 复算；③ 命中时行高亮 + 状态行提示（trace 断言 `results: factor found: …`）|
| **M8 首轮实测反馈修正** ✅ | 用户真机跑出来的 5 条：① 最小化后唤不醒、② 功率/频率曲线是直线、③ 默认字体过小、④ 中文不显示、⑤ GUI 源码应在 `src/` 而不是外置工具目录 | **已完成（2026-09-28）**：① 消息泵提到 `IsIconic` 判断**之前**，并整帧跳过最小化期间的渲染（0×0 交换链 `Present` = 0xC0000005）；验收 = 冒烟测试 **[1b]+[1c]**：最小化→还原后 `iconic=False visible=True` 且进程存活、**最小化状态下** post `WM_CLOSE` 必须退出 0（消息泵没通就挂死）；另加 RTV 解绑 + 交换链重建 + `rtv` 空值护栏，杜绝 resize 后的二次崩溃；② `PlotLines` 自动量程（`0,0`）把恒定读数压成直线 → 改为**观测窗口内 min/max ±10 %**；③ `[GUI] font_size = auto` → `round(15 × DPI)`（150 % 实测 **23 px**），新增 `[GUI] font = <路径>` 覆盖，拉丁语言也换系统字体（不再用 13 px 内置位图字体），`style.ScaleAllSizes(dpi)`；④ CJK 走 `msyh.ttc`/`simhei.ttf`/… 且**不再传 `GetGlyphRangesChineseFull()`**（1.93 docking 是动态图集，那两个 API 已 obsolete）；⑤ 全树 `tools/gui/` → **`src/gui/`**（含 `localization/` 并入源码树、根 `CMakeLists.txt` 改 `add_subdirectory(src/gui)`）。验收 = `test_gui_smoke.ps1` **54/54**，其中 [6] 断言四象限布局（四块面板都在客户区内、互不重叠、左列同为 60 %、右列同为 40 %、`Workers` 在输出标签之上、`GPU` 在 `Results` 之上），[7] 断言字号 ≈ 15×DPI、atlas 里**真有**中文字形（`map=11 baked=11 negctl=00`）且本地化 0 缺键；中文另加**像素级**验收 `test_gui_cjk_pixels.ps1` **16/16**（真字形 0.75 em/5 种 ink vs 反例 tofu 0.43 em/单一 ink，标定过的反例必须判失败）；②的机器判据 = `test_gui_gpu_curves.ps1` **15/15**（真 worker 压卡 40 s：忙卡 `power=22 distinct / clock=21 distinct`、`plot` 区间非退化且覆盖观测值；空闲卡 1~2 个不同值 —— 那种"真·直线"是**正确**的）；全套回归复跑（见 §15）**0 失败** |
| **M6 worktodo 可视化生成** | 见 `docs/DEV_ECM_WORKTODO.md` | **范围 A 已完成（2026-09-29，§19）**：核心（解析/过滤/去重/改写/排序/存档名校验/逐行推荐曲线/段分派/预览/带 mtime 守卫的追加）+ 面板；与 `ecm.py` **逐字节一致** 5 组；`tools/test/test_gui_generator.ps1` **32 项**（含单测 84 项）。未做部分列在 §19.5 与 `DEV_ECM_WORKTODO.md` §9 |
| **M9 第二轮实测反馈修正** ✅ | 用户真机三条：① 字体想用 ini 调大小（且 150 % 下有点糊）、② 他自己跑的时候中文显示 `???`、③ 生产目录点 Start 后 driver 立刻失败（"环境都配好了"）| **已完成（2026-09-28）**：① `[GUI] font_size` 支持小数（150 % 自动值 22.5 不再凑整）+ 新增 `[GUI] font_snap`（`PixelSnapH`，拉丁小字更锐），并把"ImGui 只有灰度抗锯齿、没有 ClearType"写进 §10.3；② 根因是**字体只在启动时按当时的语言挑**，用户从 Language 菜单切中文时用的还是拉丁字体 ⇒ 现在 `reload_localization()` 置标志、帧循环两帧之间重跑 `apply_ui_font()`，并加了"画不出当前语言的字体不许用"的两级兜底（救回系统 CJK 字体 / 切回英文），删掉了 `App::init` 里硬编码 chineseSimplified 的老守卫；③ 根因是**生产目录的 `ecm_cuda.exe` 是 D1/D2 之前的构建**（二进制里没有 `--worker`/`Worker #` 字符串）：GUI 传的 `--worker 1` 被当成位置参数 ⇒ 走单跑路径 ⇒ `No input number on stdin` ⇒ 退出 1 ⇒ 重启 3 次熔断。GUI 现在把这条 stdout 识别成诊断（状态栏 + trace + 表格红 `!`）。验收：`test_gui_cjk_pixels.ps1` **31/31**（含运行中切语言的 trace + 像素证据）、`test_gui_workers.ps1` **23/23**（新增 `--scenario stale-driver` 复现旧 driver 的整条链路）、`ecm_gui_log_parse_test` **49/49**（`old_driver` 分类）；全套回归复跑 **0 失败** |
| **M10 第三轮实测反馈修正** ✅ | 用户真机三条：① Workers 表的 Task 太长、要挪到下一行；② 状态列显示成 `workers.state__stopped`；③ 进度条 / ETA 都不显示 | **已完成（2026-09-28）**：① 任务行从表格列改为**每个 worker 的第二行**（跨整表宽度、换行、暗淡；`PushClipRect` 放宽裁剪），其余列改为按示例文本算的固定宽度、进度条列 `WidthStretch` 吃剩余宽度 ⇒ 长 worktodo 不再挤走别的列；② 根因是**从状态名推导本地化键**（`Stopped` → `state__stopped`，多一个下划线），键不存在 ⇒ `t()` 回退成字面量 ⇒ 表里显示 `workers.state__stopped`；改为 `app.h` 里的**显式键表** `state_key()`，补上缺失的 `state_starting`（en/zh 各一条），并在 `--selftest` 加"每个状态 × 两种语言都能解析"的断言；③ 与①同源：Task 列把 ~80 字符的行撑到几千像素宽，进度条/速度/ETA 被推到面板外。验收：`test_gui_workers.ps1` **31/31**（新增 8 项：读 `table: workers …` 实测几何，断言进度条宽 ≥60 px、结束点在表内、速度列在进度条之后、**ETA 列在面板内**、`fits=1`、任务行确实独立成行）、`test_gui_smoke.ps1` **57/57**（在用户那种 885 px 宽面板下同样 `fits=1`）、`--selftest` **35/35** |
| **M11 第四轮实测反馈修正** ✅ | 用户真机三条：① 测试脚本里进度正常，但在他自己的工作目录跑不显示；② 任务行只显示开头一点点、要占满整行；③ 退出时要静默终止 worker | **已完成（2026-09-28）**：① 用**他的真实任务**（M3571 / B1=2.6e8 / 960 曲线 + 他的检查点与指数缓存）在沙箱里复现：进度行**确实到达并被解析**（trace `worker 1: progress pct=0.0 … eta=88924.6`，driver 日志 8 条 `GPU: [`），所以不是解析问题；可诊断性补三处：详情面板显示 `progress lines <N>`（N=0 + Running ⇒ 是 driver 还没打第一行）、进度条在等第一行时显示 `waiting for the first progress line`（不再空白）、文档写清"进度行不进事件日志面板，要看原文勾 raw output"；② 根因是**两个 ImGui 用法错误**：`PushClipRect(..., intersect=true)` 与单元裁剪求交（等于没放宽），且 `PushTextWrapPos()` 要的是**窗口局部** x（传屏幕坐标 ⇒ 换行被悄悄关掉）⇒ 现在任务行跨列换行，实测 `task_wrap_w=1201 px`（用户窗口尺寸）；顺带给进度条留最小值（先压 Name 列），实测 `progress_w=475 px`、`fits=1`；③ 退出路径改为 `stop_all_quietly()`：`TerminateJobObject` 立即终止 + 单行 `shutdown: terminated N running worker(s)`，不再逐 worker 打 `stopping`、不产生 `Error/Restarting/restart` 记录；并改正旧注释（立即杀进程**不会**让 driver flush 检查点，丢失上界 = `ckpt_seconds`）。验收：`test_gui_workers.ps1` **34/34**（新增：任务行跨列、静默关闭、关闭后无崩溃记录）、`test_gui_smoke.ps1` **57/57**（进度条 ≥90 px + `fits=1`）、`--selftest` **36/36**（动态本地化键守卫） |
| **M12 第五轮实测反馈修正** ✅ | 用户真机四条：① （纠正上轮理解）退出**不能**静默杀 worker —— 要**弹窗提醒**，同意后**先写检查点**再退出；② 要一个"像你测试那样连续跑所有 GUI ps1 脚本"的脚本；③ 任务行会被表格**纵线**遮挡，能否在任务行隐藏 y；④ Results 表重新排序（先标量 `factors/bits/hits`，再列表 `sigmas/curves`）并允许滚动 | **已完成（2026-09-29）**：① `App::request_close` + 确认模态（回车=停止并退出 / Esc=取消）+ Stopping 阶段：以 `<driver 目录>` 的 `.ecm_ckpt_*.dat` mtime 为基线，**等到更新的检查点出现**才终止，期间显示每 worker 状态与 Force quit；`[GUI] exit_confirm = ask\|stop\|kill`、`[GUI] graceful_stop_ms`（默认 5 min）；Stop 按钮走同一条路。顺带修掉**真实缺陷**：GUI 未设子进程工作目录 ⇒ 检查点/`.save` 落到 GUI 的 CWD（现在 = driver exe 目录）。验收 = 新增 `tools/test/test_gui_exit_checkpoint.ps1` **27/27**（真 ecm_cuda + `ckpt_seconds=3`：WM_CLOSE 不关窗→Esc 取消→再 WM_CLOSE→回车确认→`checkpoint written (…)` → `stopping (checkpoint written)` → 退出 0，且**磁盘上 mtime 确实变新**）；② 新增 `tools/test/test_gui_all.ps1`（15 个入口一键跑完；本轮实测 **15/15 项、466 项检查、0 失败**，全程约 4 分钟无人值守）；③ 表格边框改为 `BordersInnerH\|BordersOuterH`（只留横线，纵线不再横穿任务行）；④ Results 列序改为 `Factor/bits/Hits/First seen/Curves/Sigmas` + `ScrollX\|ScrollY` + 冻结前 4 列。另外：GUI 收到 `--worker` 会警告（跑测试时把 GUI 当 driver 传会开出一个空窗口，用户手动关掉的那个就是它） |
| **M7 Linux 移植** | GLFW+OpenGL3、`posix_spawn`、`libnvidia-ml` | 见 §16（暂不排期） |
| **M13 本轮五项（2026-09-29）** ✅ | ① D4 `--gpu-info`（+ 修两处内核警告文案）；② 进度节奏 `progress_log_seconds`（管道每次、文件每 n 秒、100 % 总写）；③ 完成的任务自动交接 Prime95（`worktodo.add` + 通知条）；④ M6 范围 A 生成器（核心 + 面板 + 与 `ecm.py` 逐字节对比）；⑤ 文档（含新 `docs/DEV_ECM_INI.md` 全配置表） | **全部完成**：① `test_gpu_info.ps1` **52 项**（含与运行路径 `CGBN<tpi,bits>` 自证、零副作用、OpenCL `not_applicable`）；② `test_progress_cadence.ps1` **19 项**（实测 `0`→文件 1 行/管道 27 行、`1`→7 行、`-1`→27 行）；③ `test_p95_transfer.ps1` **47 项** + `test_gui_p95_notice.ps1` **23 项**（四色切换、pending 由磁盘驱动、AID/已知因子逐字节保留、锁与 pending 重投）；④ `test_gui_generator.ps1` **25 项**（单测 84 + 与 `ecm.py` 逐字节 5 组 + 逐行推荐独立复算 + 面板 trace）；⑤ `docs/DEV_ECM_INI.md` 新建，`DEV_ECM_GUI.md` 新增 §11.1/§18/§19，`DEV_ECM_WORKTODO.md` §5.3–§9 按实测改写。**随后按用户实测反馈补两处可见性修正**：⑥ 通知条被停靠面板盖住（"四种颜色我实测看不出"）⇒ 预留自己那一行 + 底色带 + 严重度标记 + "未配置"一键打开 ini，并用 `PrintWindow` **像素断言**四色（§18.3.1）；⑦ 生成器块/SM 输入框宽度被 `InputInt` 的步进按钮吃掉（"没有宽度，无法显示数字"）⇒ `step = 0` + 90 px 并把"框宽/可编辑宽"trace 出来给测试断言，同时修掉绝对路径被重复拼接的真 bug、新增 `[GUI] start_tab`（§19.2.1）。最终全套回归 **20/20 项、676 项检查、0 失败**（§15） |
| **M14 曲线重做 + 单位修正（2026-09-29，用户实时反馈）** ✅ | 用户三条：① worker 日志窗口**删去进度 %** 曲线；② 速度曲线**由"曲线/秒"改为"秒/曲线"**；③ **所有折线图更美观、易读** | **完成**：新增共用控件 `App::draw_metric_plot`（`src/gui/app.h`/`app.cpp`）替换掉**全部** `ImGui::PlotLines`：圆角渐变卡片 + 3 条网格线 + 逐列矩形面积 + 2 px 折线 + 末尾样本点 + 标题/数值+单位标题行 + 量程两端灰字 + 虚线参考线（功耗图 = 已执行的 NVML 上限）+ 悬停竖直引导线与"n 个样本前"读数；不足 2 个样本时显示 `collecting…`（i18n）。worker 面板只画 **`s/curve`**（`WorkerView::hist_s_per_curve`，中文标签 `秒/曲线（越小越快）`），进度 % 曲线删除；占用率图固定 `0..100`（不再随数据缩放），功耗/时钟按观测窗口 ±12 %（跨度下限 2 %）自动，**并把真实量程回传**给 `trace_gpu_history`。新增 curve trace（`plot: <id> …`，按图限流：前 3 次变化 + 之后每 2 s）与 `gpu: limits dev=N power_limit_w=X`，让测试能断言"画出来的"曲线与量程而不是相信截图。测试：`test_gui_workers.ps1` 新增 `[3c]` 段（曲线存在、`n≥2`、单位为 `s/curve`、数值 ~1.2 而非 `curves/s` 的 ~0.83、**没有任何进度 % 曲线**）；`test_gui_gpu_curves.ps1` 新增 `[4b]` 段（三张图都有序列、范围包住最新值、util 恒为 0..100、功耗图的 `ref=` 等于 NVML 真实上限）。文档：`DEV_ECM_GUI.md` §8.2 新增（含"面积为什么逐列画""限流为什么按图"两条实测教训）、§7.3 详情面板条目改写、决策 24。**实现过程中又量出三个真问题并修掉**：① **150 % 缩放下卡片三条带重叠**（卡片高度是常量，而文字行高随字号变 ⇒ 标题行与最小/最大值行互相重叠并压住曲线）⇒ 高度按字号缩放 + 曲线带保底 24 px，并新增 `[3d]` 几何断言；② **一条物理上不可能的尖峰毁掉整张功耗图**（`plot: gpu1/power` 修前 `lo=0.00 hi=660.61`、修后 `lo=0.68 hi=10.49`）⇒ 曲线丢弃 >3× 强制上限的样本；③ **参考线落在量程外就永远不显示**（上限 285 W vs 窗口 134..168 W）⇒ 越界时虚线贴边 + 自绘三角指示。另修一个测试自身的缺陷：`grab_window.ps1` 按进程名找窗口，会在操作员自己的 GUI 正在运行时抓到**那个**窗口（实测抓到 1052×629 的生产窗口而不是测试的 1600×1000），像素断言会量错界面还照样通过 ⇒ 新增 `-ProcId`，`test_gui_cjk_pixels.ps1` 改为按 PID 抓。**最终全套回归 20/20 项、742 项检查、0 失败**（§15） |

依赖顺序：**D1+D2 先做**（体量小、决定多 worker 的一切，且能立刻用 headless 队列模式验证不回归），
随后 M1（无依赖，可与 D1/D2 并行），再 M2→M5；M6 消费 D4；M7 最后。
**D1+D2、M1–M5、M8–M11（四轮实测反馈修正）、M12、M13（D4 + 进度节奏 + Prime95 交接 +
M6 范围 A + 文档）、M14（曲线重做 + `s/curve` 单位修正）均已完成并验收。**

---

## 15. 测试策略

* **无窗口自测（已落地）**：三个可脚本化的入口，都是"失败非零退出"：
  * `ecm_gui.exe --selftest` —— ini 往返（保注释/保序/未知键 + `.bak`）、本地化（基线 + 中文 + 回退 + 缺键）、
    ImGui 上下文/docking/viewports、系统字体查找 + DPI 字号（36 项）；
  * `ecm_gui.exe --worker-selftest [--fake <exe>]` —— worker 进程管理全套（48 项，**不需要 GPU**）；
  * `ecm_gui_log_parse_test.exe` —— 解析层（**71 项**，纯函数，样本行逐字来自驱动源码的格式串；
    含本轮新增的 `p95_add:` 通知解析）；
  * `ecm_gui.exe --gpu-selftest` —— NVML 监控，含与 `nvidia-smi` 的**交错交叉比对**与假 DLL 降级（27 项）；
  * `ecm_gui_results_test.exe` —— results 双文件：合并/去重/JSONL 字段/可重建/截断容错（42 项）；
  * `ecm_gui_gen_test.exe` —— **worktodo 生成器**（**84 项**，纯逻辑 + 文件：`--gpu-info` 解析、选档、
    位宽估算、解析、存档名、流水线、mtime 守卫的追加；另有 `--emit` 模式给逐字节对比用）。
* **真窗口冒烟（已落地，括号内为 2026-09-29 22:14 全套实测的项数）**：
  `tools/test/test_gui_smoke.ps1`（**65 项**）、`test_gui_workers.ps1`（**67 项**，含本轮的 `[3c]`
  worker 曲线/单位断言与 [3d] 曲线带几何断言，见 §7.3/§8.2）、	est_gui_real_workers.ps1（23 项，需 GPU）、
  `test_gui_gpu.ps1`（17 项，NVML 面板与降级）、`test_gui_results.ps1`（27 项，真驱动两轮 → 跨运行合并）、
  `test_gui_cjk_pixels.ps1`（**31 项**：像素级中文验收 + 字体救回 + 英文回退 + 运行中切语言，见 §10.3）、
  `test_gui_gpu_curves.ps1`（**28 项**，真 worker 压卡时断言功率/频率曲线**确实在变**，并断言**画出来的**
  三张图（序列长度、量程包住最新值、util 恒 0..100、功耗图的 `ref=` 等于 NVML 真实上限），见 §8.1/§8.2）、
  `test_gui_exit_checkpoint.ps1`（**27 项**：退出确认 + **退出前必写检查点**，见 §5.6）、
  `test_gui_p95_notice.ps1`（**32 项**：Prime95 交接通知条的四色切换，红由磁盘 pending 驱动，见 §18）、
  `test_gui_generator.ps1`（**32 项**：生成器单测 + 与 `ecm.py` **逐字节**对比 + 面板 trace，见 §19）。
* **驱动侧新测试**（也进了同一个套件，`exe = 'driver'`）：
  `test_gpu_info.ps1`（**49 项**，D4，见 §11.1）、`test_progress_cadence.ps1`（**19 项**，见 §7.2）、
  `test_p95_transfer.ps1`（**66 项**，见 §18.4，含 AID 被 PrimeNet 拒收后的自动剥离与台账找回）、
  `test_hit_fields.ps1`（11 项，D3 命中行字段）、`test_worker_sections.ps1`（24 项，D1/D2 段语义）。
* **一键全跑**：`tools/test/test_gui_all.ps1` —— 把上面全部 + headless 自测/单测（**20 个入口**）按顺序跑完，
  逐项打印 `passed/failed` 与耗时，汇总表 + 每项完整日志落在 `tools/test/_run/suite_<时间戳>/`，
  任一失败即非零退出；`-SkipGpu`（不碰显卡）、`-Only '*smoke*'`（挑着跑）、`-List`（只列清单）。
  **最新一套完整回归实测（2026-09-29 22:33，机器上同时跑着用户自己的生产 worker，曲线重做之后）**：
  `20/20 项、742 项检查、0 失败`
  （`selftest 36 / worker-selftest 48 / gpu-selftest 27 / log-parse 71 / results-unit 42 / smoke 65 /
  workers 67 / exit-checkpoint 27 / cjk-pixels 31 / gpu-panel 17 / gpu-curves 28 / real-workers 23 /
  results-e2e 27 / hit-fields 11 / worker-sections 24 / gpu-info 49 / progress-cadence 19 /
  p95-transfer 66 / p95-notice 32 / generator 32`）。
  都用 `EnumWindows` 找 class=`ecm_gui` 的窗口、`PostMessage(WM_CLOSE)` 关闭、读 `--trace` 断言生命周期，
  **不需要截图**；需要 GPU 的用很小的任务（M991/M677 + B1=1e4/1e6，秒级）。
  唯一的例外是 `test_gui_cjk_pixels.ps1`：它**故意**要截图，因为"字有没有真的画出来"只有像素能证明
  （抓图用 `PrintWindow`，被遮挡的窗口也能抓到，不需要人盯着屏幕；**抓图必须按 PID**
  （`grab_window.ps1 -ProcId`）：只按进程名找会抓到操作员自己那个 `ecm_gui`，实测抓回来的是
  1052×629 的生产窗口而不是测试自己的 1600×1000 窗口 —— 那样像素断言量的就是错的界面）。
* **冒烟测试现在覆盖的三类"只能靠肉眼发现"的缺陷**（都源自用户真机反馈，全部改成机器断言）：
  * **[1b]/[1c] 窗口生命周期**：最小化 → 还原（进程存活、窗口可见）；**最小化状态下** post `WM_CLOSE`
    必须退出 0 —— 这一条就是"消息泵被 `IsIconic` 提前 return 挡住"的回归护栏；
  * **[6] 布局几何**：从 `--trace` 的 `layout:` 行读四块面板矩形，断言"都在客户区内 / 没有塌成 0 /
    两两不重叠 / 左列两块同 x 且同宽 / 右列两块同 x 且同宽 / 左列 ≈60 % 右列 ≈40 % / 上下顺序正确"；
  * **[7] 字体**：断言字号 ≈ `15 × DPI`、用的是系统轮廓字体（不是 13 px 内置位图），
    并用 `CalcTextSize("文件")` 实测中文字形宽度（`cjk_ok=1`）—— "字太小""中文是方块"不再靠肉眼回归；
  * **CJK 像素判据的标定（2026-09-29 更新）**：真中文在 `font_size = 40` 下量到 **23–30 px** 的墨迹格宽
  （0.575–0.75 em），而 tofu/英文回退只有 **13–17 px**（0.33–0.43 em）⇒ 通过线取 **0.5 em**（原来是 0.6 em，
  只有 1 px 余量，穷出过一次假失败）。**不要**用 trace 的几何去"精确指定"扫描行带：试过把整条 48 px 菜单栏
  喂给探针，它会把每个汉字拆成偏旁（31 格、中位 13 px）——探针自己的"第一个文本带"规则反而是对的。
* **像素级中文验收**（`test_gui_cjk_pixels.ps1`）：抓真窗口，量菜单栏里每个字形的格子宽度与 ink 值。
    四轮：[A] 自动字体必须出真中文；[B] `[GUI] font` 指向画不出中文的字体 → **必须被救回**
    （trace `cannot draw CJK: rescuing`，像素仍是全宽中文）；[C] `ECM_GUI_CJK_FONT=none`（假装系统无
    CJK 字体）→ **必须切回英文**、绝不出现 `???`；[D] 英文启动 + `--switch-language`（= Language 菜单）
    → 字体必须换、且切换后的像素是全宽中文（这一轮就是用户报的 `???`）。
  * **曲线有数据**（`test_gui_gpu_curves.ps1`）：真的放一个 `ecm_cuda` worker 压卡 40 s，
    断言 `gpu: history` 里忙卡的功率与 SM 时钟各有 ≥5 个不同取值、且 `plot=lo..hi` 覆盖观测区间 ——
    "曲线是直线"这类反馈从此有机器判据，不用盯屏幕。
* **假 worker 夹具（已落地）**：`src/gui/fake_worker.cpp` 支持 `ok` / `crash` / `crash-once` /
  `hang` / `spawn-child` 五种场景 → 进程管理、重启熔断、Job object 杀进程树都不依赖真实算力。
* **解析单测**：样本行内联在 `log_parse_test.cpp` 里（附驱动源码的出处注释）；格式漂移会让测试红，而不是让
  UI 默默显示 `—`。
* **不测数学**：GUI 不做数论，别把 `--verify-gpu` 的活搬到 GUI 测试里；真实验收里"命中"只当事件处理。
* **不回归现有目标**：每次改 CMake 都核对目标列表（`cmake --build <dir> --target help`），
  并确认缺 imgui / `-DECM_BUILD_GUI=OFF` 两条路径都能正常 configure。
* **与生产运行并存**（2026-09-29 实测的教训）：本机可以**同时**跑着用户自己的
  `ecm_gui.exe` + `ecm_cuda.exe`。那时：
  * 进程残留检查必须**按可执行文件路径过滤**（只认本测试启动的那份），否则会把用户的
    生产进程报成"我们的残留"（`real-workers`/`results-e2e`/`p95-notice` 三处已改）；
  * 依赖负载形状的断言（`gpu-curves` 的"时钟要有多个取值"）会**因为另一份任务把卡钉在固定
    boost 档**而失败 —— 这不是面板的缺陷，测试检测到外部 GPU 任务就打印 `[SKIP]` 并说明原因；
  * 时间型断言要按"实际耗时"放宽（`progress-cadence` 的"文件恰好 1 行"在任务超过 60 s 时
    会合理地多出中间行；现在先断言日志文件真的被清空，再断言计数）。
  这三条都是**测试质量**问题，不是功能回归：同样的脚本在"机器空闲"时是严格门限。

---

## 16. TODO / 明确延后

| 项 | 说明 |
|---|---|
| **CLI 单跑 GPU 模式坏了（既有缺陷，与 GUI 无关）** | `echo (2^k-1) \| ecm_cuda.exe -gpu …` 对**所有** param（0/2/3）都在内核里报 `CGBN error occurred: invalid modulus (it must be odd)`（`kernels/cuda/cgbn_stage1.cu:1585`），而同一条任务走**队列模式完全正常**（M991/B1=1e4 命中 3 条曲线、M677/B1=1e6 命中 5 条）。已排除本轮改动：两条模式在同一个 GPU 函数里填 `sigmas`，队列路径通过。**待查**：单跑模式的 batch 组装（第一个 `ecm_cuda` 用例请用队列模式） |
| **`-sigma` 与 param3 的取值范围检查像是反的（既有疑点）** | `-sigma 20001 -gpu-param 3` → 驱动接受，内核报 `invalid modulus`（`d = sigma>>32 = 0`）；`-sigma 85903640887297 -gpu-param 3`（`d = 20001`，内核要的非零 `d`）→ 驱动**拒绝**："`-sigma` value exceeds 2^32-1; the GPU batch path (gpu_param = 3) needs a 32-bit sigma"。两者只能对一个，需要对着 `cgbn_stage1.cu` 的 param3 曲线构造实测后再定（GUI 侧暂不依赖 `-sigma`，走 ini 的随机 sigma） |
| GPU 面板：风扇转速 | M4 已做 util/功耗（含上限与百分比）/SM 与显存时钟/温度/显存/节流原因/全机合计功耗/三条曲线；风扇转速（`nvmlDeviceGetFanSpeed`）留待需要时加（笔记本 dGPU 常常不报风扇） |
| `--status-file` | 若"纯日志解析"在换构建/格式漂移后变脆弱，driver 侧加"每 N 秒原子写一行机器可读状态"，GUI 改为读文件（§13 决策 10 的既定升级路径） |
| stage-2 交接 | **已完成（2026-09-29，§18）**：驱动把完成的任务行原样追加进 Prime95 的 `worktodo.add`，GUI 用红/黄/绿/灰通知条显示状态。仍未做：GUI 里手动改 `worktodo.add`、或反过来从 Prime95 取任务 |
| GUI 内 worktodo 编辑 | **部分完成（2026-09-29，§19，M6 范围 A）**：粘贴/打开文件 → 预览 → 追加到 worktodo；未做：过滤器 UI、`ECM=`/`ECM2=`（Prime95 用）输出、CLI 命令输出、任务编辑/删除、从结果反推（详见 §19.5） |
| **生成器面板的文件拖放（drag & drop）** | 用户已明确"只写进文档 TODO，先不做"：把 `.txt`/`.csv` 拖到生成器面板上应当等价于"打开文件"。实现要点（将来做时照抄即可）：`DragAcceptFiles(hwnd, TRUE)` + 在 `WndProc` 里处理 `WM_DROPFILES`（`DragQueryFileA` 取路径、`DragFinish`），拿到路径后走 `gen_load_file(path)` 那条既有路径；ImGui 只负责画高亮边框。放在 §16 而不是顺手做，是因为它需要动 Win32 消息循环（§3 的抽象边界之外） |
| 生成器：与 `ecm.py` 的差异收尾 | 已知且**故意**的差异见 §19.5（已知因子的整除性只做形状检查、不含 CLI 发射器、一次写多个段）；若要完全等价，把 GMP 引进 GUI 或让驱动提供 `--verify-run`（属于 §22 的"要不要把数论搬进 GUI"决策） |
| Linux（M7） | GLFW+OpenGL3、`posix_spawn` + `killpg`、`libnvidia-ml.so`、fontconfig 找 Noto CJK |
| 中文文档 | 本文档与 `DEV_ECM_WORKTODO.md` 目前是中文；若要让外部贡献者参与本地化，考虑加英文镜像 |

---

## 17. 风险

| 风险 | 缓解 |
|---|---|
| driver 输出格式漂移导致解析失效 | 解析集中在 `log_parse.cpp` 并有样本夹具；失效时 UI 显示 `—` 而不是错值；必要时切 `--status-file`（§16） |
| 双卡不同算力造成"快卡等慢卡"观感 | 段化 worktodo + 每段独立 `gpucurves`（§6）；v1 不做自动均衡（§16 可选） |
| GUI 写 ini 与用户手改冲突 | 原子改写 + 保序 + 只改已知键（§4.3），并保留 `.bak` 一代 |
| ImGui docking 分支非 release | pin 到具体 commit（`64944b45…`），升级流程写在 `third_party/imgui/README.md` |
| NVML 版本/驱动不匹配 | 全部调用容错，失败即降级显示（§8） |

---

## 18. Prime95 交接：完成的任务自动进 `worktodo.add`（2026-09-29）

### 18.1 目标与边界

用户要的是"**我们自己算完 stage 1，Prime95 接着算 stage 2**"，而不用手工搬行
（`ecm_p95feeder` 是另一条路：搬 Edwards 的 `.tmp`，保持独立、**本轮未改一行**）。

通道用 Prime95 官方文档里的投放箱：把行追加到 Prime95 `worktodo.txt` **旁边**的
`worktodo.add`，Prime95 自己找时间并进 `worktodo.txt` 然后**删掉** `worktodo.add`
（`undoc.txt:443-447`；源码 `commonc.c: incorporateWorkToDoAddFile()`）。
这样我们**从不改** `worktodo.txt`，Prime95 自己的记账不受影响。

行必须是 Prime95 认的关键字：**`ECMSTAGE2=` 原样追加**（含 AID 与已知因子串）。
`commonc.c:2936-2938` 明确支持
`ECMSTAGE2=k,b,n,c,filename[,B2-or-zero][,skip_curves][,num_curves][,"known-factors"]`，
并且 AID 前缀正是 Prime95 自己 writer 的写法（`commonc.c:3573`），它靠这个 AID 把最终因子
按正确的 assignment 上报 —— 所以**不能**丢掉、也**不能**改写成 feeder 的 `ECM=` 形式。

### 18.2 键与规则

| 键 | 行为 |
|---|---|
| `p95_worktodo_path` | Prime95 的 `worktodo.txt`；**留空（默认）= 功能关闭**，此时行为与本轮之前完全一致 |
| `p95_add_workers` | `空`=不写段头；`3`；`1,3`；`1-8`；`auto`=读同目录 `prime.txt` 的 `NumWorkers` |

* 段头**只认 `worktodo.txt` 里真实存在的段**（`worktodo.add` 不算证据）：不存在就退回不带段头，
  并在日志/GUI 里给黄色提示（Prime95 对未知段头会把行归到别的线程甚至丢掉，不能赌）。
* 多个候选：按"当前排期最少"选（`worktodo.txt` + `worktodo.add` 的活动行数，段头缺失的行算 worker 1，
  与 Prime95 的初始 `tnum=0` 一致）；并列取编号小的。只有一个候选就不数。
* 写入用锁 `worktodo.add.lock`（`CreateFile` `CREATE_NEW`，100 ms 重试、最多 3 s；
  超过 60 s 的锁文件视为被遗弃并抢占）+ **原子替换**（`MoveFileExA(MOVEFILE_REPLACE_EXISTING)`，
  之前的 `remove()+rename()` 存在"文件短暂不存在"的窗口，而 Prime95 恰好在找这个文件）。
* **失败绝不阻塞任务**：任务照样进 `finished`、照样出队；该行落进
  `<驱动目录>\p95_add_pending.txt`，**下一次成功交付时先把它一起送出**，然后删除 pending。
* **命中因子也照常交付**（用户明确要求）：stage 2 仍需做 GCD 并上报，我们无法代它上报。
* 交付发生在"任务完成 → `.save` 同步完"之后、`processed++` 之前（`queue_run_one()` 里）。

### 18.3 GUI：通知条（显著颜色）

菜单栏下面一条**整宽通知条**，永远可见（用户要求"包括 pending 都要通知 GUI 并使用显著颜色"）：

| 颜色 | 含义 | 文本 |
|---|---|---|
| 灰 + `[-]` | `p95_worktodo_path` 为空 | "Prime95 handoff is OFF：p95_worktodo_path 为空…"，右侧按钮变成 **打开 ecm.ini** |
| 绿 + `[OK]` | 已配置、最近一次交付成功 | "交付成功/就绪" |
| 黄 + `[WARN]` | 交付成功但路由退让（缺段头 / AID 被拒 / 补投丢失件） | 警告 + 驱动给出的原因 |
| 红 + `[FAIL]` | 交付失败 / **pending 文件非空** | "失败：N 行待投递" + 原因 |

每条都配一个**底色带**（红/黄/绿/灰各自的深色底）+ 上面的严重度标记，比只改文字颜色醒目得多。
红条**以磁盘上的 pending 文件为准**（不只是最后一条日志），所以上次会话留下的待投递也会红；
右边常驻两个按钮：**打开 Prime95 目录** / **打开待投递文件**。消息按严重度只显示最重的一条，
其余用 `(+N)` 计数。驱动打印的机器可读行：

```
p95_add: ready workers="1-8" file="…\worktodo.add" pending=2 keep_aid=1 rejected_aids=2
p95_add: aid_rejected aid="2F1990C6A1353CE223CBE33633829E3A"
p95_add: ok worker=2 added=1 pending_delivered=0 recovered=0 aid=kept file="…"
p95_add: warn worker=0 added=1 pending_delivered=0 recovered=0 aid=dropped file="…" note="…"
p95_add: pending worker=1 lines=3 file="…" error="…"
```

值一律带引号（只有 `\"` 与行尾 `\\` 是转义），所以空格路径与自由文本都不歧义；
解析在 `src/gui/log_parse.cpp`（`P95Notice`），通知条状态每帧变化都会 trace
（`p95 notice: level=red parked=2 text=…`），因此有机器判据。

### 18.3.1 让它**真的看得见**（2026-09-29 用户反馈后的修正）

用户反馈："**四种颜色的真实切换，我实测看不出**；在没有配置 `p95_worktodo_path` 时也没有明显提示。"
根因不是颜色，而是**位置**：通知条画在 `WorkPos + 一个行高`（即压到 dockspace 区域上），而面板是停靠窗口，
ImGui 按窗口创建顺序绘制 ⇒ 通知条**一直在面板底下**，等于没画。修法：

1. **预留自己那一行**：`App::draw()` 里先算出 `strip_h`（用上一帧实测的条高，首帧回退 `GetFrameHeight()`），
   dockspace 宿主窗口下移 `strip_h` 并相应减高 ⇒ 通知条与面板**永不重叠**（几何由 trace 断言：
   `bottom <= host_top`）。为什么要自测高度：150 % DPI 下条内容（文字 + 小按钮）比 `GetFrameHeight()`
   高，预留小了窗口会自己向下长 —— 实测预留 31 px、真实 48 px，又叠了 17 px。
2. **每种颜色配一个底色带 + 严重度标记**：`[FAIL]` 红底 / `[WARN]` 黄底 / `[OK]` 绿底 / `[-]` 灰底。
3. **"未配置"给明确动作**：文案直接点名要改的键，右边给一个 **`打开 ecm.ini`** 按钮。
4. **像素级验收**（`test_gui_p95_notice.ps1`）：`PrintWindow(PW_RENDERFULLCONTENT)` 抓**主窗口**
   （多视口模式下同 class 有多个窗口，取客户区最大的那个），按 trace 的条矩形取样一行按色系分类：
   实测灰 `236/236 中性`、绿 `159 绿`、红 `172 红`、黄 `233 黄`。

### 18.2.1 AID（assignment key）：为什么必须检查它（2026-09-29 生产环境实测）

用户在生成环境反复重启都看不到交付生效，追查到根因**在我这版交付逻辑**：原样保留 AID 是错的。

机制（Prime95 源码，`commonc.c:6550-6569`）：

```c
/* If we get an invalid assignment key, then the user probably unreserved */
/* the exponent using the web forms - delete it from our work to do file. */
rc = sendMessage (PRIMENET_ASSIGNMENT_PROGRESS, &pkt2);
if (rc == PRIMENET_ERROR_INVALID_ASSIGNMENT_KEY || rc == PRIMENET_ERROR_WORK_NO_LONGER_NEEDED) {
    rc = deleteWorkToDoLine (tnum, w, TRUE);     /* 整条任务被删掉 */
}
```

* 带 AID 的行 ⇒ Prime95 用这个 key 向 PrimeNet 报进度；**key 一旦失效**（作业已取消/关闭/过期），
  PrimeNet 回 `error 43 / ap: no such assignment key` ⇒ Prime95 **删掉整条任务**，stage 2 永不运行，
  也没有任何结果 —— 我们算完的 stage 1 白丢。
* **不带 AID** 的行 ⇒ `w->assignment_uid[0]` 为空，进度上报整段被跳过（`commonc.c:6554` 的条件），
  Prime95 反而会走 `PRIMENET_REGISTER_ASSIGNMENT`（`commonc.c:6532`）**自己注册一个新 key** ⇒
  工作照跑、结果照记（实测：M5153 那条 AID-less 交付最后报的是 `AID: 7BD17F7BE`）。

生产环境证据链：15:19 交付 M3583（`p95_add: ok worker=1 added=1`，AID `2F1990C6…`）→ Prime95 收下
（`Sending expected completion date for M3583`）→ PrimeNet 回 `no such assignment key … key: 2F1990C6…`
→ 该行从 `worktodo.txt` 消失、`results.txt` 里没有任何 M3583 结果。同一份 `prime.log` 里
`7CF9949553699526D1F4AA791B138423`（更早完成的 M3571）也被拒 ⇒ 同样下场。

现在的行为（三个机制）：

| 机制 | 说明 |
|---|---|
| **识别被拒 key** | 驱动读 Prime95 的 `prime.log` / `results.txt`，只认 `no such assignment key … key: <hex>` 这一种形状（不会把无关的 `key:` 误当 AID）；启动打印 `p95_add: ready … rejected_aids=N`，每个被拒 key 打一行 `p95_add: aid_rejected aid="…"` |
| **交付时自动去掉被拒的 AID** | 该行 AID 在被拒集合里 ⇒ 去掉后再交付，通知 `aid=dropped` + `note="PrimeNet rejected AID …"`（GUI 黄条），Prime95 于是自己注册新作业并把 stage 2 跑完 |
| **丢件自动补投** | 每次交付的行记进 `<驱动目录>\p95_add_sent.txt`（`sent <行>` / `recovered <行>`）。若某行 AID 被拒、且 Prime95 的 `worktodo.txt` + `worktodo.add` 里都已没有它（= 被删了）⇒ 下次交付时**去掉 AID 重新投一次**，通知 `recovered=N`（黄条），且不会重复补投 |

新 ini 键：`p95_keep_aid`（默认 `1` = 保留 AID，符合你原本的要求；`0` = 一律不带）、
`p95_recover_lost`（默认 `1` = 自动补投）。见 `docs/DEV_ECM_INI.md` §1.2。

验收：`tools/test/test_p95_transfer.ps1` 的 [9]/[10]/[10b]/[11] —— 被拒 AID 被去掉、live AID 仍保留、
补投恰好一次、Prime95 仍持有该行时不补投、`p95_keep_aid=0` 时一律不带。

### 18.4 验收

* `tools/test/test_p95_transfer.ps1` **47 项**（真驱动、真任务）：按号路由、缺段头退让、`auto` 读
  `NumWorkers`、区间 `1-2` 选空闲 worker、非法值 `2000`/`abc` 退让、**AID 与已知因子逐字节保留**、
  锁被占用 → 3 s 后落 pending 且**任务仍出队**、下一个任务把 pending 一起送出并清空、超过 60 s 的
  锁被抢占且交付很快（< 2.5 s）、`worktodo.txt` 从未被改动。
* `tools/test/test_gui_p95_notice.ps1` **32 项**（含四色像素判据）：灰/绿/红/黄四种颜色的真实切换（红由磁盘 pending 驱动，
  灰/绿不需要 worker），黄/红发生在**真 worker** 上，关闭 GUI 后不留我们启动的 worker。

---

## 19. worktodo 生成器（M6 范围 A，2026-09-29）

### 19.1 做了什么

面板（`###gen`，与 Workers/Detail 同一个停靠节点的一个标签页）＋纯逻辑核心
`src/gui/worktodo_gen.{h,cpp}`：

```
粘贴/打开文件 → 解析 → 过滤 → 去重 → 改写 → 排序 → 存档名校验 → 逐行算推荐曲线 → 预览 → 追加
```

* 输入：PrimeNet 手动作业（`ECM=` / `ECM2=` 行，可带 AID、`FFT2=`、已知因子串）；
  以 `#` 开头的注释与空行跳过；`ECMSTAGE2=`（我们的输出格式）跳过并计数。
* 逐行推荐曲线：**`curves = 块/SM × SM 数 × ipb`**，其中 `ipb` 来自**该行 N 实际会用的档位**
  （`--bits` 同款选择规则：`bits >= bits(N) + carry_bits` 的最小档），
  `SM 数`/`ipb` 来自 `ecm_cuda.exe --gpu-info`（**D4，现场读，不抄表**）；`块/SM` 默认 **2**（面板可调）。
  推荐值不小于该档位的 `curves_min`（内核自己的"别让 SM 闲着"下限）。
* `bits(N)` 用**纯算术**估算（GUI 不含 GMP）：`log2(k) + n·log2(b)`，`c<0` 且 `k·b^n` 恰为 2 的幂时
  （Mersenne 情形）减 1 —— 否则 `2^521-1` 会被算成 522 位。带已知因子时按"因子位数之和"扣减，
  这是**下界**（可能低 1 位），方向安全：档位偏小 ⇒ `ipb` 偏大 ⇒ 曲线数略多而不是欠占用。
* 段：只输出**有任务**的 `[Worker #N]` 段，段号取自 ini 的 `[Worker #N]`；同一张卡上有多个
  worker 时按行序**轮流分配**；目标卡在面板上选（默认第一个 worker 的卡）。
* 存档名：模板 `m{n}_{b1}.save`（`{n}{k}{b}{c}{b1}{b2}`），并按驱动契约校验
  （必须以 `_<B1>.save` 结尾且 B1 可解析成 > 0 的数），不合格直接报错不生成 —— 因为驱动的
  `ECMSTAGE2=` 路径**从存档名取 B1**。
* **"生成预览"和"追加到 worktodo"是两个独立按钮**：预览只算不写；追加前**重新校验目标文件的
  大小/mtime**（生成预览之后被任何人改过就拒绝并提示重新生成），追加是**只追加**（不截断既有行），
  追加时不覆盖队列文件里正在被消费的行。追加完成后必须重新预览才能再追加（同一条 mtime 规则）。

### 19.2 与 `ecm.py` 的关系

`tools/ecm_worktodo/ecm.py` 是参考实现，两者在"给定同样的曲线数"时必须**逐字节一致**：
字段顺序、CSV 引号规则（`"` 翻倍）、段头与结尾空行、`--sort-by` 的稳定排序与 `idx` 兜底、
去重的胜者规则（B1 → 曲线数 → 有真 AID → 先出现）、已知因子并集，都照抄。

### 19.2.1 面板控件可见性（2026-09-29 用户反馈后的修正）

用户反馈："**生成器的 block/sm 输入框没有宽度，无法显示数字**"。两个原因，都修了：

1. `InputInt` 默认带 `+/-` 步进按钮，而 ImGui **把两个按钮画在给定宽度内部**（每个宽 `GetFrameHeight()`）：
   150 % DPI 下行高 ≈31 px ⇒ 80 px 的框里被吃掉 63 px，只剩十几像素给数字。现在 `InputInt(..., 0, 0)`
   （step = 0 关掉按钮，**必须保持 0**）并把宽度提到 90 px；代码里把两个宽度都 trace 出来
   （`blocks_box_w` 框宽 / `blocks_edit_w` 可编辑宽），测试断言两者 ≥60 px —— 谁把 step 改回非 0，
   测试立刻红，不用等用户报告。
2. 顺带修掉一个**真 bug**：`target=` 把绝对路径又拼了一次工作目录
   （`…\sandbox\D:\…\sandbox\worktodo.txt`）。现在 ini 值解析走 `path_resolve()`：绝对路径原样用，
   相对路径才按驱动目录拼接（`worktodo` 与 `p95_worktodo_path` 都走这条）。
3. **`[GUI] start_tab = workers|detail|gen`**：面板不是被选中的标签时 ImGui 会 `SkipItems`，
   完全不布局、也量不到几何（这正是上面那个宽度一开始量成 176 px 的原因）。这个键既能让你
   打开就落在生成器上，也让测试能真正量到控件；选择必须用 `SetWindowFocus()` 实现——
   `DockNodeUpdate()` 每帧把节点的 NavWindow 写回标签栏（`imgui.cpp:19889`），直接改
   `SelectedTabId` 会被覆盖（实测：节点一直保持 Workers，生成器始终隐藏）。
### 19.3 验收

* `tools/test/test_gui_generator.ps1` **32 项**：
  单元测试 **84 项**（`--gpu-info` 解析、选档、位宽估算、解析、存档名、流水线、mtime 守卫的追加）；
  **与 `ecm.py` 逐字节对比 5 组**（排序 `n` / 排序 `b1` / 去重 / 改写 `set-b1`+`set-b2` / `set-has-na`）；
  逐行推荐曲线与现场 `--gpu-info` 复算一致（实测 `n=101 → 3840`、`n=521/1019 → 1920`，档位不同 ⇒ 曲线数不同）；
  GUI 面板真的画出来并 trace 出状态与目标文件。
* 单测也覆盖了"没有 GPU 报告时拒绝生成推荐值"（OpenCL 构建的 `not_applicable` 路径）。

### 19.4 面板上的文案

`[GUI] language` 决定语言（`localization/{english,chineseSimplified}.xml` 新增 `gen` 面板 22 个键 +
`p95` 面板 7 个键，`--selftest` 会断言两语言 0 缺键）。

### 19.5 范围 A 明确**没做**（写清楚免得当成 bug）

1. **已知因子的整除性只做形状检查**（整数 > 1）：精确整除需要大数运算，GUI 不含 GMP。
   `ecm.py --verify-factors` 仍是唯一做精确校验的地方。
2. **没有 CLI 发射器**（`--emit-cli` 的 sh/ps1/bat）与 **`ECM=`/`ECM2=`（原生 Prime95 worktodo）
   输出**：范围 A 只做 `ECMSTAGE2=` 队列。
3. **过滤器的 UI**：核心已实现 `min_n/max_n/min_curves/max_curves`，面板暂时只暴露
   "去重/排序/存档名/块-SM/目标卡"这几项。
4. **文件拖放**：见 §16 的 TODO 条目（用户明确要求先只写文档）。
5. **非法输入行的策略与 `ecm.py` 不同**：`ecm.py` 遇到坏行直接整体退出；GUI **跳过该行并在
   "被跳过的输入行与警告"里列出来**，其余任务照常生成（一次粘贴 200 行时，一个手误不该让 199 行白费）。