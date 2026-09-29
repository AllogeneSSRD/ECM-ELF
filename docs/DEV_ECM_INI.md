# ecm.ini 全配置参考（分区版）

本文是 **ecm.ini 的完整键表**：每个键属于哪个分区、默认值、谁消费它、有没有 CLI 等价物。
三份来源合并而成，三者永远应该一致：

* `src/core/ecm_queue_config.{h,cpp}` —— 队列模式（`ecm.exe` / `ecm_cuda.exe`）真正解析的键，
  以及缺失 ini 时写出的默认模板；
* `src/gui/app.cpp` —— GUI 读写的 `[GUI]` 键与 GUI 专用的 `[Worker #N]` 键；
* `src/core/ecm_driver.cpp` —— 命令行开关，用来判断"有没有 CLI 等价物"。

约定：

* 键名区分大小写，但**不区分**段头里 `[Worker #N]` 的写法大小写（`[worker #1]` 也可以）。
* 段内键覆盖全局键（第一个段头之前的键是全局默认）。CLI 覆盖 ini。
* `空` 表示留空即关闭该功能。
* `CLI` 列写 `—` 表示**故意没有** CLI 开关：这类键只从 ini 读（详见文末"只有 ini 的键"）。
* 一个键如果被"驱动/队列"消费，指 `run_queue_manager()`；被"GUI"消费指面板/监管层；
  "后端"指 GPU 后端（CUDA = `ecm_cuda.exe`，OpenCL = `ecm.exe`）。

---

## 1. 全局键（第一个 `[Worker #N]` 段头之前）

### 1.1 队列与文件

| 键 | 默认 | 谁消费 | CLI | 说明 |
|---|---|---|---|---|
| `worktodo` | `worktodo.txt` | 驱动/队列 | `--tmp-dir` 不是它 | 任务队列文件；相对路径按**驱动 exe 所在目录**解析。GUI 的生成器也把预览追加到这里。 |
| `finished` | `worktodo.finished.txt` | 驱动/队列 | — | 已完成任务追加到这里的原始行。 |
| `tmp_dir` | `.` | 驱动/队列 | `--tmp-dir` | stage-1 存档目录（`e{n:07d}_c{k}.tmp`）。 |
| `log_file` | `screen.log` | 驱动/队列 | — | 时间戳日志文件；留空=只打屏。多 worker 时若未显式设置，worker 2..N 默认写 `screen_<N>.log`。 |
| `save_sync_dir_1` | 空 | 驱动/队列 | — | 任务完成后把 `.save` 同步过去的目录。 |
| `save_sync_dir_2` | 空 | 驱动/队列 | — | 第二个同步目标，规则同上。 |
| `sync_mode` | `incremental` | 驱动/队列 | — | `incremental`=只同步本任务新增；`full`=整目录比对。 |
| `verbose` | `true` | 驱动/队列 | `-v` | 详细输出。 |
| `progress_color` | `cyan` | 驱动/队列 | — | 进度条颜色：`none|red|green|yellow|blue|magenta|cyan|white|grey`。 |
| `progress_log_seconds` | `60` | 驱动/队列（日志镜像层） | — | **进度行写进 `log_file` 的最小间隔（秒）**：`60`=每分钟一行，`0`=文件里完全不写进度行，负数=每行都写。管道/控制台**不受限制**（GUI 靠它显示进度，约 200 ms 一行），且报告 `100.0%` 的那一行**总会**写进文件。见 `docs/DEV_ECM_GUI.md` §7.2。 |

### 1.2 Prime95 交接（`docs/DEV_ECM_GUI.md` §13）

| 键 | 默认 | 谁消费 | CLI | 说明 |
|---|---|---|---|---|
| `p95_worktodo_path` | 空 | 驱动/队列（+GUI 只读） | — | Prime95 的 `worktodo.txt`。**留空=关闭交接**。开启后：任务完成且 `.save` 同步完，驱动把该任务行**原样**追加到同目录的 `worktodo.add`（保留 AID 与已知因子串），Prime95 自己取走后删除该文件。命中因子的任务也照样交付（stage 2 仍需做 GCD 并上报）。 |
| `p95_add_workers` | 空 | 驱动/队列（+GUI 只读） | — | 写入哪个 `[Worker #N]` 段：空=不写段头（Prime95 归给 worker 1）；`3`；`1,3`；`1-8`；`auto`=读同目录 `prime.txt` 的 `NumWorkers`。`worktodo.txt` 里没有该段则退回不带段头（并在日志/GUI 里给黄色提示）；多个候选时选当前排期最少（`worktodo.txt` + `worktodo.add` 的活动行数）的那个。锁文件 `worktodo.add.lock`，失败行落在 `<驱动目录>\p95_add_pending.txt`，下次成功交付时一起送出。 |
| `p95_dir` | 空 | 驱动/队列 | `--p95-dir`（仅兼容，忽略） | 旧脚本兼容键：stage-1 结果不再写 Prime95 目录，Edwards `.tmp` 交接由独立的 `ecm_p95feeder` 负责。 |

### 1.3 引擎选择（`[method]` 语义，键在全局段）

| 键 | 默认 | 谁消费 | CLI | 说明 |
|---|---|---|---|---|
| `method` | `gpu` | 驱动/队列 | `--method` 与 `-gpu`/`--edwards` | `gpu`=GPU 批量 stage 1（哪个后端由可执行文件决定，见下）；`edwards`=CPU Edwards/Atkin-Morain；`mont`=CPU Suyama-sigma Montgomery（与 gmp-ecm `-param 0` / Prime95 `sigma_type=1` 同族）。 |
| `backend` | `auto` | 驱动/队列（CPU 路径） | `--backend` | `auto|simd|gmp`。 |
| `field` | `auto` | 驱动/队列（CPU 路径） | `--field` | `auto|mersenne|montgomery`。 |
| `stage1_threads` | `0` | 驱动/队列（CPU 路径） | `--stage1-threads` | 0=自动。 |
| `affinity` | 空 | 驱动/队列 | `--affinity` | CPU 亲和性列表。 |
| `save_name_pattern` | `m{n}_{b1}.save` | 驱动/队列 | — | stage-1 存档名模板：`{n}{k}{b}{c}{b1}{b2}`。**驱动从存档名最后一个 `_` 字段取 B1**，所以模板必须让 `_<B1>` 结尾。 |
| `naf_w` | `0` | 驱动/队列（Edwards） | `--naf-w` | NAF 窗口宽度（3..12）。 |
| `exponent` | `lcm` | 驱动/队列 | `--exponent` | `lcm|choose12`。 |
| `exp_cache` | 空 | 驱动/队列 | `--exp-cache` | stage-1 指数缓存目录（默认 exe 目录）。 |
| `sigma` | `0` | 驱动/队列 | `-sigma` | 固定 sigma（0=随机）。 |
| `ckpt_seconds` | `600` | 驱动/队列 | `--ckpt` | 任务中途存档间隔（秒），0=不自动存档（Ctrl+C 仍会存）。 |

### 1.4 GPU 键

| 键 | 默认 | 谁消费 | CLI | 说明 |
|---|---|---|---|---|
| `device` | `0` | 驱动/队列 → 后端 | `-d` | 设备序号（0 起）。 |
| `gpu_param` | `3` | 驱动/队列 → 后端 | `--gpu-param` | 曲线参数化：`0`=Suyama param0（与 `--method mont` 同族，**仅 CUDA**）；`2`=param2 批量-2（**仅 CUDA**）；`3`=gmp-ecm 批量/PARAM=3（默认）。OpenCL 只支持 `3`。 |
| `gpucurves` | `0` | **仅单次运行** | `-gpucurves` | 单次运行（给了 B1/B2 位置参数）时的每批曲线数。**队列模式从不读它**：队列用任务行自带的 `curves_to_run`。`docs/DEV_ECM_WORKTODO.md` §7 已按此改写。 |
| `tpi` | `8` | 后端（OpenCL 专用） | `--tpi` | 每实例线程数；CUDA 的 TPI 由内核档位决定，此键无效。 |
| `wg_size` | `0` | 后端（OpenCL 专用） | `--wg` | 显式 work-group 大小，0=自动。 |
| `kernel_mul` `kernel_sqr` `kernel_add` `kernel_sub` `kernel_special_mult` | 空 | 后端（OpenCL 专用） | `--mul` `--sqr` `--add` `--sub` `--special-mult` | 算子内核路径（id/别名/auto）。CUDA 忽略。 |

---

## 2. `[Worker #N]` 段

段头编号从 **1** 开始（Prime95 的约定；不存在 `[Worker #0]`）。
段内可以出现**上面任何一个全局键**来覆盖它，另外还有下面这些**只在段内有意义**的键。

| 键 | 默认 | 谁消费 | CLI | 说明 |
|---|---|---|---|---|
| `name` | `worker N` | **仅 GUI** | — | 表格里显示的名字。驱动不读。 |
| `autostart` | `0` | **仅 GUI** | — | GUI 启动时自动拉起该 worker。 |
| `extra_args` | 空 | **仅 GUI** | — | 追加到该 worker 命令行末尾的额外参数（空格分隔）。 |
| `device` | 全局 `device` | 驱动/队列 | `-d`（单次运行） | 该 worker 用哪张卡。`test_gui_real_workers.ps1` 用它把两个 worker 钉在两张卡上。 |
| `gpucurves` | 全局 | **仅单次运行** | `-gpucurves` | 同上：队列不看。 |
| `log_file` | `screen_<N>.log`（未显式设置时） | 驱动/队列 | — | 每 worker 独立日志。 |
| `worktodo` | 全局 | 驱动/队列 | — | 该 worker 读哪个队列文件；段头匹配也针对这个文件（`docs/DEV_ECM_WORKTODO.md`）。 |
| `method` / `backend` / `field` / `stage1_threads` / `affinity` / `naf_w` / `exponent` / `sigma` / `ckpt_seconds` / `progress_color` / `progress_log_seconds` / `tpi` / `wg_size` / `kernel_*` / `save_name_pattern` / `tmp_dir` / `save_sync_dir_*` / `sync_mode` | 继承全局 | 驱动/队列 | 见上表 | 段内覆盖全局；`--worker N` 决定读哪一段。 |

---

## 3. `[GUI]` 段（只有 `ecm_gui.exe` 读）

| 键 | 默认 | 谁消费 | CLI | 说明 |
|---|---|---|---|---|
| `NumWorkers` | `1` | GUI | — | 生成几个 worker 行；与每段的 `[Worker #N]` 对应。 |
| `exe` | 空 | GUI | — | worker 可执行文件（`ecm_cuda.exe` / `ecm.exe`）。空=GUI 旁边找，再退回 PATH。驱动不读此键。 |
| `language` | `english` | GUI | `--language`（启动参数） | 界面语言文件主干名（`english` / `chineseSimplified`）。菜单 Language 切换后会写回该键。 |
| `localization_dir` | `<exe>/localization` | GUI | — | 语言 XML 目录。 |
| `font` | 空 | GUI | — | UI 字体路径；空=自动挑（要能画中文）。 |
| `font_size` | `auto` | GUI | — | 字号（px）；`auto`=按 DPI 缩放（150% 下约 22.5 px）。 |
| `font_snap` | `1` | GUI | — | 字形步进对齐整像素（更锐利）。 |
| `refresh_hz` | `10` | GUI | — | 界面刷新率（启动参数只有 `--language` / `--switch-language` / `--switch-language-after` / `--trace` / `--selftest` / `--gpu-selftest` / `--worker-selftest` / `--ini`，没有刷新率开关）。 |
| `gpu_poll_ms` | `500` | GUI | — | NVML 采样间隔（毫秒）。 |
| `priority` | `below_normal` | GUI | — | worker 进程优先级：`idle|below_normal|normal|above_normal|high`。 |
| `window` | `120,80,1500,900` | GUI | — | 主窗口位置与大小，退出时写回。 |
| `dock_layout` | 空 | GUI | — | ImGui 停靠布局转义串（用 `\n` 转义换行）。 |
| `dock_layout_ver` | `4` | GUI | — | 布局版本；低于当前版本会重建默认布局（4 = 增加"生成器"面板）。 |
| `start_tab` | `workers` | GUI | — | 启动时选中左上方共享节点里的哪个标签：`workers`（默认）/ `detail` / `gen`。非选中的标签 ImGui 会跳过布局（量不到控件几何），所以"打开就落在生成器上"也是脚本能测生成器控件的前提。 |
| `exit_confirm` | `ask` | GUI | — | 关窗行为：`ask`=弹确认框→等新检查点→退出；`stop`=不弹框但仍等检查点；`kill`=立即终止。 |
| `graceful_stop_ms` | `300000` | GUI | — | 等待"新检查点落盘"的上限（毫秒），超时才强杀。 |
| `results_json` | `<exe>/results.json.txt` | GUI | — | 命中记录（追加式 JSONL）。 |
| `results_txt` | `<exe>/results.txt` | GUI | — | 由 JSONL 复算出的每因子一行。 |

GUI 只**读**下面两个队列键，用来显示 Prime95 交接通知条（`docs/DEV_ECM_GUI.md` §13）：

| 键 | 谁消费 | 说明 |
|---|---|---|
| `p95_worktodo_path`（全局段） | GUI 只读 | 为空时通知条显示灰色"未配置"。 |
| `p95_add_workers`（全局段） | GUI 只读 | 显示在通知条右侧。 |

---

## 4. 只有 ini、故意没有 CLI 的键

这一组的共同点是"属于运行配置而不是一次运行的参数"，且都和 GUI/长期运行有关：

* `progress_log_seconds` —— 日志文件节奏；
* `p95_worktodo_path`、`p95_add_workers` —— Prime95 交接（GUI 通知条同时显示状态）；
* 整个 `[GUI]` 段（除 `language`/`refresh_hz` 这类启动参数）；
* `[Worker #N]` 的 `name` / `autostart` / `extra_args`。

用户长期意图：**逐步把很少用的 CLI 开关下线，改为 ini-only**。新增功能默认遵守这条；
已有的 CLI 开关保持不变（脚本仍在用），详见 `docs/DEV_ECM_GUI.md` §15。

---

## 5. 缺省 ini 模板

`ecm_queue_config_write_default()` 在 ini 不存在时写出一份带中英文注释的模板，键的顺序与
本文第 1、2 节一致（`progress_log_seconds`、`p95_*` 都在其中）。它**只写一次**：
`--gpu-info` 之类的查询路径只读不建，避免在你没打算跑队列时突然多出一个配置文件。
