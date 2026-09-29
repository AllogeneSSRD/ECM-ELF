# worktodo 段化与可视化生成（`[Worker #N]`）开发文档

本文描述 worktodo（任务队列）的**段化**改造（driver 侧 D2）、以及**可视化生成**（GUI 侧 M6，
含按 GPU 的 SM 数与 N 的位数推荐 `gpucurves`）的规划。GUI 前端本身的整体架构见
[`docs/DEV_ECM_GUI.md`](DEV_ECM_GUI.md)。

状态：**规划已定稿，代码未开始**。

---

## 1. 现状（都是从代码实测的，别猜）

| 事实 | 出处 |
|---|---|
| 支持两种行格式：`ECMSTAGE2=`（给我们的 GPU stage-1）与 `ECM=`/`ECM2=`（Prime95 原生，两种前缀等价） | `src/core/ecm_worktodo.h:5-15` |
| `ECMSTAGE2=[<aid>,]<k>,<b>,<n>,<c>,<save_name>,<B2>,<skip_curves>,<curves_to_run>[,"factors"]`；**B1 不在字段里**，由 `ecm_extract_b1_from_save_name()` 从 `m{n}_{b1}.save` 的名字抽 | `ecm_worktodo.h:7-10,58-61` |
| 队列循环：每轮**重新读文件的首个任务行** → 跑 → 成功后 `advance(Remove)` 移除该行；失败则把该行**原地替换**成 `# ERROR <原行>`（保留成注释供人工检查） | `src/core/ecm_driver.cpp:3153-3157, 2964-2973` |
| 队列空 → `break` → 进程退出（返回 0） | `ecm_driver.cpp:3155, 3265` |
| `finished` 文件只追加**原任务行**（不含因子） | `ecm_driver.cpp:2968` |
| 队列模式下 **读不到位置参数** 就进入（`pos.empty()`）；`--go` 是打印群阶，不是队列开关 | `ecm_driver.cpp:3583, 3481` |
| 原先的编排脚本 `work_manager.ps1` **已废弃**（被 driver 内置队列取代），仅可作为测试夹具，不作为生产路径 | 用户确认；见 `docs/DEV_ECM_CUDA_QUEUE_MANAGER.md` |
| 生成器 `tools/ecm_worktodo/ecm.py` 只生成、从不执行，也从不改写运行中的 Prime95 拥有的文件 | `tools/README.md` §ecm_worktodo |

---

## 2. 为什么段化：不同 GPU 的最佳 `gpucurves` 不同

本机两张卡（4070 Ti 60 SM / 4060 Laptop 24 SM）算力差 ~2.5×，而 GPU stage-1 是**批大小驱动**的
（`BLOCK_COUNT = ceil(curves / IPB)`，见 §5）⇒ 一个"共享池、谁空谁抢"的模型会让两张卡互相拖累，
且无法让每张卡用各自的最优批量。因此：

> **单一 `worktodo.txt` + `[Worker #N]` 分段**：每段的 `curves_to_run`（即 `gpucurves`）由该段所属
> GPU 决定；每个 worker 进程只消费自己那一段。

这与 prime95 的 `worktodo.txt` 同构（它的 `worktodo.txt` 也是用 `[Worker #N]` 分段）：

```ini
[Worker #1]
ECMSTAGE2=N/A,1,2,5351,-1,"m5351_110e6.save",0,0,960,"25689701826359,…"
ECM=N/A,1,2,15749,-1,26062448,2606244800,50,"28726177,…"

[Worker #2]
ECMSTAGE2=N/A,1,2,8287,-1,"m8287_110e6.save",0,0,960,"36877151"
```

---

## 3. 段语义（D2）

### 3.1 规则

1. 段头：`[Worker #N]`，`N` 从 1 起（与 ini 的 `[Worker #N]`、命令行 `--worker N` 编号一致）。
2. **文件开头到第一个段头之间的任务行** = `[Worker #1]` 的行（与 prime95 的"无段即 worker 1"一致）→
   现有无段文件**行为完全不变**。
3. 一个 worker 只读自己段内的任务行；`advance` 只删自己段的首行。
4. 别的段、段头、注释、空行在改写时**原样保留、保持相对顺序**；文件里不属于任何段的行不动。
5. 重复的 `[Worker #N]` 段头：按**出现顺序拼接**视为同一段（不报错——追加式写入容易产生重复段头）。
6. 未知段名（如 `[Worker #99]` 但 `NumWorkers=2`）：保留不动；GUI 在队列面板给一条提示
   （"有 1 段没有对应 worker"）。
7. `# ERROR <原行>` 的替换只发生在**本段**（只影响自己那行）。
8. 本段无任务行而别的段有 → 本 worker 认为"队列空"并退出（`queue empty`），不去别人的段里偷活。

### 3.2 实现要点（已落地）

* 段头语法由 `ecm_worktodo_parse_worker_header(line, &is_bracket_line)` 统一解析（`[Worker #N]`，不区分大小写与空格；
  同一个函数也被 `ecm_queue_config.cpp` 用来切 ini 的 `[Worker #N]` 段 → 两侧语法不会漂移）。
* `ecm_worktodo_first_line(path, line)` / `ecm_worktodo_advance(path, first_line, action)`
  保留为**旧行为入口**，内部转发到新的 3/4 参数版本并传 `worker = 1`（无段文件 = 原来那样）。
* 新增：`ecm_worktodo_first_line(path, worker, line)`、`ecm_worktodo_advance(path, worker, first_line, action)`、
  `ecm_worktodo_list_workers(path, workers, err)`（GUI 用：列出有任务行的段号，便于发现"有段没有对应 worker"）。
* 改写时**按原行**搬运：只有目标行被删除或被替换成 `# ERROR <原行>`，段头/注释/空行/别的段保持原样与顺序。
* 兼容性硬要求：无段文件 + `worker = 1` 时，行为与改前**逐行一致**。
  现有回归：`tools/test/ecm_worktodo_test.cpp`、`tools/test/test_worktodo_pipeline.ps1`（33 项）必须继续全过。

### 3.3 验收

> **状态（2026-09-28）：已实现并通过验收。**
> `tools/test/worktodo_sections_test.cpp`（27 项断言，失败非零退出）+ `tools/test/test_worker_sections.ps1`（24 项端到端检查）。
> 后者不碰 GPU：它用的是**解析失败**的任务行（`ECMSTAGE2=garbage`）。
> 注意别用 `ECMSTAGE2=BOGUS,1,2,5351,-1,"m5351_110e6.save",…` 当"坏行"——解析器会把非整数首字段当成
> 可选 AID，于是它是一条**真实的 M5351 任务**，会把 GPU 跑起来（本轮真踩到一次）。

| 用例 | 判据 |
|---|---|
| 无段文件（现状） | 队列模式输出与改前逐行一致；`ecm_worktodo_test` / `test_worktodo_pipeline.ps1` 全过 |
| 两段文件，两个 worker | 两段各自被消费；任一 worker 崩溃后重启，只从自己段继续；`finished` 里两段的行都能看到 |
| 段内一条坏行 | 该行变成 `# ERROR …` **留在本段**，别的段完全不变（`git diff` 级对比） |
| 段内空、他段非空 | 本 worker 打印 `queue empty` 并退出（不越段消费） |

---

## 4. 消费者矩阵

| 消费者 | 角色 | 备注 |
|---|---|---|
| `ecm.exe` / `ecm_cuda.exe` 队列模式 | **生产路径**（唯一执行者） | D2 之后按 `--worker N` 选段 |
| `ecm_gui` | v1 **只读**（段 → 剩余行数/当前任务/命中）；M6 起可编辑生成 | 编辑时必须走"原子替换 + 备份"，且**拒绝**写正在被 worker 读取的文件的中间态 |
| `tools/ecm_worktodo/ecm.py` | 生成器（分配行 → `ECMSTAGE2=`/`ECM=` 行） | M6 计划给它加 `--emit-worker-sections`（按设备/位数分段输出） |
| `work_manager.ps1` | **已废弃**，仅测试夹具 | 文档与 README 都要标注 |
| Prime95 | stage-2 侧消费者 | 我们**从不**改写它的文件；只写自己的 `worktodo.txt` 与 `.save` |

---

## 5. `gpucurves` 推荐（M6 的核心算法）

### 5.1 事实基础（kernel 侧已实现，别重新发明）

| 事实 | 出处 |
|---|---|
| `BLOCK_COUNT = ceil(curves / (TPB/TPI))`，即 `IPB = TPB / TPI` 是"每块曲线数" | `kernels/cuda/cgbn_stage1.cu:1049, 1288, 1455` |
| `TPI`（每条曲线的线程数）是**运行时**按位数选的，不是编译期常数 | `cgbn_stage1.cu:1190-1353` |
| 判据一：`blocks ≥ sm_count`，否则有 SM 完全没活干（实测 511-bit 档、TPB=256/TPI=4：4096 → 8192 曲线 **+7.6% curve-bits/s**） | `cgbn_stage1.cu:1459-1461, 1500-1504` |
| 判据二：批量是 `sm_count` 的**整数倍**（整波），避免最后一批尾波 | `cgbn_stage1.cu:1470-1471, 1505-1513` |
| 判据三：**折叠域（`ECM_MERS_FOLD`）另需 `blocks ≥ 2×sm_count`**，否则实测比 Montgomery 慢 10–25% | `cgbn_stage1.cu:1480-1497` |
| 反面：**不要**盲目填满 register-allowed block slots —— kernel 是 issue bound，实测 240 blocks（4 blocks/SM）比 120 blocks（2 blocks/SM）略慢（73.41 vs 72.88–72.90 s/curve），功耗墙卡上更亏 | `cgbn_stage1.cu:1463-1471` |
| CPU 路径：`stage1_threads = N` 想跑满需要每任务 ≥ `8N` 曲线（一个任务 = 一个 8 曲线 SIMD 批） | `src/core/ecm_queue_config.cpp:305-309` |

### 5.2 推荐公式

```
blocks_min   = sm_count                      (折叠构建: 2 × sm_count)
blocks_wave  = 向上取整到 sm_count 的倍数      ← 推荐值用这个
curves_min   = blocks_min  × IPB
curves_wave  = blocks_wave × IPB             ← 推荐 gpucurves
```

* `IPB` 与 `sm_count` **必须由 driver 给出**（D4 的 `--gpu-info`），不要在 GUI/生成器里复刻 TPI 选择表
  —— 那是 kernel 的实现细节，复刻必然漂移。
* 推荐值是"`≥1 block/SM` 且整波"，**不是**"填满槽位"（§5.1 最后两条）。

### 5.3 接口（D4）

```
ecm_cuda.exe --gpu-info -d 0 --bits 5120
# 机器可读单行（字段顺序固定、key=value、空格分隔）：
sm_count=60 tpb=256 tpi=… ipb=… blocks_min=60 curves_min=… curves_wave=… [fold_min_blocks=120]
```

* 纯查询、零副作用，仿 `--show-kernels` 的"打印后退出"风格（`ecm_driver.cpp:3578`）。
* `--bits` 缺省 = 用 ini/默认档位；用于"同一张卡不同 N 位数给不同 `gpucurves`"。

### 5.4 已知的不精确处（顺手一起修）

kernel 现有的两条警告把**建议的 gpucurves** 打成了 `sm_count`（或 `2×sm_count`），
而实际要求是 `blocks ≥ sm_count` ⇒ `curves ≥ sm_count × IPB`：

```
"raise -gpucurves to about %ld (a multiple of %d keeps whole waves)", (long)sm_count, (int)IPB
"Raise -gpucurves to about %ld (%d curves/block)", 2L*sm_count, (int)IPB        # fold 分支
```

⇒ D4 里一并把这两行改成打印**曲线数**（`curves_min = blocks_min × IPB`，整波建议值另给），
使"警告里的数字"与"`--gpu-info` 的输出"和"真正该填的值"三者一致。

### 5.5 验收

1. `--gpu-info --bits <b>` 输出里的 `ipb`/`sm_count` 与 kernel 自身警告行里的数值**一致**。
2. 用推荐值（整波）跑一轮真实任务，stdout **不再出现** "raise -gpucurves" / "needs >= 2 resident blocks/SM"。
3. 对照实验（可复现）：同一 N 与 B1 下，`gpucurves = curves_wave` 与 `gpucurves = 2 × curves_wave`
   各跑数次取中位数，确认**没有**因为"填满槽位"而变慢（记录 curve-bits/s）。
4. CPU 路径：`stage1_threads = N` 时检查 `curves ≥ 8N`，不足给提示。

---

## 6. 可视化生成（M6）规划

### 6.1 输入

* Prime95 分配行（`ECM=`/`ECM2=`）或 CSV（`tools/ecm_worktodo/assignment_sample.csv` 的形状）；
* 目标参数：每段（worker）的 `B1`、`device`、`gpucurves`（可由 §5 推荐自动填）、`param`（0/2/3）；
* 现有 `worktodo.txt`（用于**增量追加**而不是整份重写）。

### 6.2 管线（复用 `ecm.py` 已定的规则，别另立一套）

| 规则 | 内容 |
|---|---|
| 去重判据 | `(k,b,n,c)` 按**数值**比较；冲突保留 B1 最大 → curves 最大 → 真 AID 优先 → 先出现者；known factors 取并集 |
| 顺序 | 解析 → 过滤 → 去重 → 重写（`--set-b1/--set-b2/--set-has-na`）→ 排序；过滤必须在去重**之前** |
| save 名契约 | `ECMSTAGE2=` 无 B1 字段，driver 从名字抽 ⇒ 必须匹配 `…_<B1>.save`（工具强校验、非零退出） |
| `B2=0` 语义 | Prime95 里 0 = 自动选 B2，不是"不做 stage 2" |
| 编码 | 读自动识别（UTF-8 有/无 BOM → GBK 兜底）；写 **UTF-8 无 BOM + CRLF** |

### 6.3 分派到段

1. 先按"设备 + 位数档"把任务分组（同段 = 同设备、同 `gpucurves`）。
2. 每组的 `gpucurves` 由 §5 公式给出（同组内不同位数就再拆组）。
3. 段的编号 = ini 段编号（`[Worker #1] device=…` 与 `worktodo [Worker #1]` 必须指同一张卡）。
4. 生成结果先给**预览 + diff**，用户确认后再写；写 = 原子替换 + 备份一代。
5. 拒绝改写正在被读的文件：若目标 `worktodo.txt` 的 `mtime` 在预览后变化 → 重新预览（而不是盲目覆盖）。

### 6.4 验收

* 生成的文件能被 `ecm_worktodo_test` 的解析器接受（跨实现验收，沿用现有 33 项测试的思路）；
* 在一个**临时目录**里用生成结果跑一轮真实队列模式（不碰生产 `worktodo.txt`），检查段不串、`# ERROR` 只落本段；
* 生成的 `gpucurves` 跑起来不再触发 §5.4 的警告。

---

## 7. 与 ini 的对应关系（必须一致）

| ini | worktodo | 含义 |
|---|---|---|
| `NumWorkers = N`（全局） | 段 `[Worker #1..#N]` | worker 数量 |
| `[Worker #k] device=…` | 段 `[Worker #k]` 的行 | 同一张卡、同一份 `gpucurves` |
| `[Worker #k] gpucurves=…` | 该段行里的 `curves_to_run` | 必须一致（不一致时 GUI 给提示，driver 以**行内**字段为准） |
| 无 `[Worker #k]` 段 | 无段行 = worker 1 | 向后兼容 |

---

## 8. TODO

| 项 | 说明 |
|---|---|
| `--emit-worker-sections` | `ecm.py` 支持直接输出分段 worktodo（M6 的 CLI 侧） |
| 自动均衡 | 若将来卡间算力差导致明显空闲，可考虑"段间借调"（需要原子认领，当前明确不做，见 `DEV_ECM_GUI.md` §6） |
| B1 计划推荐 | 按指数规模与 ECM 概率模型推荐 B1/B2（数据源：`tools/ecm_prob`、`tools/stat/ecm_hitrate.ps1`），与 `gpucurves` 推荐并列 |
| stage-2 队列联动 | `ECM=` 行的 B2 与保存目录交接（`ecm_p95feeder`）在 GUI 里可视 |
| `# ERROR` 行回收 | GUI 里一键把 `# ERROR` 行恢复成任务行（含失败原因展示） |
