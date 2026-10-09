# Stage2 使用

`ecm_cuda_stage2.exe` 读取完成的 Stage1 save，运行 CUDA 多项式 ECM Stage2。手动 B2 可处理实际 N 为 2…16384 bits 的受支持输入；Auto B2 另受 profile 资格限制。设备编号是 CUDA 运行时编号，与构建的 sm 架构不同。

## 直接读档

基本参数为 `--save <file> --b2 <integer> --device <id>`。`--skip-curves <n>` 跳过记录，`--curves <n>` 限定数量；具体名字以当前 `--help` 为准。B1、sigma 和 N 取自 save，不重新执行 Stage1。`--factor-only` 省去命中素数命名，保留叶乘积、GCD 与必需检查；`--factorize-hits` 可再用 GP 拆解复合因子。

只接受受支持的、归一化 Suyama PARAM0 文本记录。先验证选择范围中的所有记录，包括 checksum、N、B1、X 和任务匹配；文件损坏不能当作“没有足够曲线”略过。源 save 不修改。校验和通过只说明文件完整。

直接多记录执行不使用队列自动续跑。每条曲线在独立子进程计算，结束后释放 CUDA 上下文与临时资源。

## 队列语法

任务格式：`ECMSTAGE2=[AID,]k,b,n,c,save[,B2-or-zero][,skip_curves][,num_curves][,"known_factors"]`。

数学目标为 k·bⁿ+c，再按已知因子确定实际 N。可选字段顺序固定：B2、跳过数量、执行数量、已知因子。AID 位于 k 之前，末尾 quoted 字段是因子列表。

例如 `ECMSTAGE2=1,2,3701,-1,"m3701_260e6.save",26000000000,960,10,"xxx"` 中，960 是跳过的 save 记录数，10 是请求数量；`xxx` 所在位置必须填可解析的已知因子或空字符串，不能当作任意备注。

整数支持精确十进制及科学记数法，不能经浮点舍入改变 B2。B2 优先级为 CLI 非零值 → 任务非零值 → `stage2_b2`；只有最终 B2=0 且启用 Auto B2 时进行自动选择。

## 数量不足与错误

请求 10 条、跳过后仅有 3 条时，提示 requested/available，执行这 3 条。全部成功后按 completed=3 完成原任务，继续下一项；可用记录为 0 时提示并按零曲线完成，不等待 save 追加。

存档缺失、损坏、checksum/N/X 不匹配或任务字段错误属于输入错误：记录原因，队列原行变为错误注释，finished 保留原任务和原因，继续下一项。文件权限、状态对账、CUDA/设备/显存或未知内部故障属于致命错误：停止，保留未完成任务。退出码为正常 0、存在队列输入错误 1、致命错误 2。

## 自动续跑与退出

仅队列记录完成曲线编号，不保存 Stage2 中间状态。

1. 开始曲线前原子写入尝试回执。
2. 子进程验证结果，追加结果和回执并刷新磁盘。
3. 父进程确认成功回执后原子更新完成数。
4. 中断发生在结果写入与进度更新之间时，重启按结果回执补记，避免重复执行。

进度绑定任务、队列、save 的完整 SHA256、选择范围、输出路径和相关计算配置。续跑期间不要替换/追加 save、编辑队列或改变参数；身份变化会拒绝沿用进度。结果尾部未以换行结束的记录不能作为成功回执。

任务结束先写 finished 回执，再推进队列，最后清除进度。第一次 Ctrl+C 等待当前曲线落盘后退出；第二次立即终止子进程树。强制退出的当前曲线若没有完整回执，下次从头计算。

## INI 与路径

默认读取 exe 同目录 `ecm.ini`。隐式文件缺失采用默认值；显式指定但不可读则报错。全部键见 [INI 参考](../ECM_INI_REFERENCE.md)。

Stage2 使用独立的 `stage2_worktodo`、`stage2_finished`、`stage2_results_file`、`stage2_log_file`、`stage2_progress_file` 和 `stage2_debug_log_file`。save 目录空值继承 `tmp_dir`，设备未指定时继承共享 `device`。多 worker 的显式输出路径须自行区分；未指定的部分输出自动添加 worker 后缀。

Stage2 INI 相对路径基于 INI 目录，CLI 相对路径基于当前工作目录；Stage1 保持其 exe 目录规则。共用 INI 推荐与两个 exe 放在同一目录。只将 `[Worker #N]` 用作命令行覆盖范围；模板的普通分组用注释，GUI 的真实分区规则见 [配置维护](../architecture/CONFIGURATION.md)。

运行时拒绝 Stage1/Stage2 队列冲突，以及 save、程序、配置、队列、进度、结果和日志之间的危险重叠。

## 日志与内存摘要

`verbose` 控制可读的阶段/批次输出；`stage2_debug_log` 控制独立调试文件。警告和错误始终可见。长阶段每 30 秒的父进程状态提示不等于 kernel 完成百分比。

- `curve_start`：任务曲线编号、save 记录编号、sigma、B1 和文件校验状态。
- `stage2_phase`：`previous` 为上阶段墙钟间隔，`elapsed` 为曲线累计时间。
- `real_shape`：B1/B2、算术位宽、D/P、baby/giant 数及 G 批次数。
- `mem`：设备 free/total、reserve 和 arena cap；预计最大变换及树顶形状。
- 曲线结束的 `arena=a/b fold=a/b`：a 为实际完整 NTT 池峰值或启用 fold owner 占用；fold 回退时 a=0。b 保留读取的 INI 值，可能与 CLI/环境覆盖后的有效预算不同。
- `stage2_full_wall.total`：完整引擎 init+main，排除 Stage1 与自动 D 扫描。外层 result/curve_done 还含进程和发布结果等开销。

`arena_mb` 是 NTT 池 payload，不是全部显存。独立 giant、S4、owner、下降元数据、按次回退和驱动资源另计。组件峰值不能直接相加；详细含义见 [内存](../architecture/MEMORY.md)。普通日志和结果默认不轮转。

正常离线计算不需要监听网络。Windows Compute Sanitizer 的注入通信可按目标 exe 名称触发防火墙提示；测试工具在 Windows 子进程使用 named pipes。该行为不表示 Stage2 算法需要网络权限。

## 规划与依据

`--plan-only` 查询设备和算法/组件形状，不运行曲线、不写成功结果、不推进队列。`--dry-run` 检查输入，不承诺实时显存准入。`--carrier-exponent p` 为梅森承载实验，要求保存的目标 N 整除 2ᵖ−1，当前需显式 B2。

- [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp#L172)：`records`；[队列字段](../../src/core/ecm_cuda_stage2_main.cpp#L355)：`queue_fields`。
- [ecm_stage2_queue_state.h](../../src/core/ecm_stage2_queue_state.h)：原子进度和回执对账。
- [ecm_stage2_console.h](../../src/core/ecm_stage2_console.h)：阶段和内存摘要。
- [Stage2 管线](../architecture/STAGE2.md)、[Auto B2](../architecture/AUTO_B2.md)、[构建](BUILD.md)。
