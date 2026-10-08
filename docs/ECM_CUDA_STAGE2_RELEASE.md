# CUDA Stage2 发布候选使用说明

本轮准备 Windows `sm_89` 单架构候选：`ecm_cuda_stage2.exe`、`gmp-10.dll`、共用配置模板和说明。多架构包及两个可执行文件的合并留待后续。当前算术组合尚无匹配的新 Auto B2 标定数据，使用显式 B2。本轮只做构建，未运行队列、中断或 GPU 回归测试，不代表已完成发布验收。

## 1. 开始使用

将归一化的 Suyama PARAM=0 Stage1 文本 save 放入程序目录，或在 INI 的 `tmp_dir` / `stage2_save_dir` 指定目录。按机器选择 `device`，它是 CUDA 设备编号，与编译架构不同。这个包的配置缺省设备为 0，本轮没有启动 GPU 运算。

直接执行明确的记录范围：

```powershell
.\ecm_cuda_stage2.exe --save D:\saves\m3701_260e6.save --b2 26000000000 --skip-curves 960 --curves 10
```

直接 `--save` 每次按指定范围执行，不读取或更新队列续跑进度。记录从 1 编号，上例选择 961–970；数量不足时只执行可用记录并警告。`--curves 0` 表示所有剩余记录。

队列使用独立文件 `stage2_worktodo.txt`：

```text
ECMSTAGE2=1,2,3701,-1,"m3701_260e6.save",26000000000,960,10
```

```powershell
.\ecm_cuda_stage2.exe --dry-run
.\ecm_cuda_stage2.exe
.\ecm_cuda_stage2.exe --once
```

`--dry-run` 检查第一项，不执行曲线、不写结果或日志、不修改队列和进度。`--once` 只处理一个任务，包含该任务的全部可用曲线；如该项输入有误，标记后退出。通常不指定 `--once`，连续消费队列直到为空或用户停止。

## 2. 共用 ecm.ini 与迁移

默认读取 exe 同目录 `ecm.ini`。模板为 `ecm.ini.example`；已有 INI 不会被打包脚本覆盖。隐式 INI 不存在使用内置默认，显式指定但不可读则报错。

全部键的默认值、可选范围、空值语义及用途见 [配置说明](ECM_INI_REFERENCE.md)，采用 `key=<domain>; d=<default>` 配置行加逐项中文说明，包括 Stage1、Stage2、GUI 和旧别名。说明采用 undoc 式用途与取值效果描述，直接解释如何调整。仓库维护时修改 `config/ecm_options.json` 后运行生成器；流程见 [统一配置维护](DEV_ECM_CONFIG_SCHEMA.md)。用户自己的 `ecm.ini` 仍可直接编辑。

Stage1 继续使用 `worktodo`、`finished`、`log_file`。Stage2 不再继承这三个键，改用：

- `stage2_worktodo`：默认 `stage2_worktodo.txt`。
- `stage2_finished`：默认 `stage2_worktodo.finished.txt`。
- `stage2_log_file`：默认 `stage2_screen.log`；留空禁用普通文件日志，控制台仍显示可读输出。
- `stage2_results_file`：默认 `stage2_results.jsonl`，成功曲线的完整结果。
- `stage2_progress_file`：默认为队列路径加 `[_worker].progress`；不应禁用或与其他输出共用。
- `stage2_debug_log`：默认 false；显式开启调试细节。
- `stage2_debug_log_file`：默认 `stage2_debug.log`，独立于普通运行日志。
- `stage2_save_dir`：空值继承共享 `tmp_dir`；`stage2_device` 缺省继承共享 `device`。

未显式配置普通日志、结果或调试日志路径时，worker 2 等自动使用 `_2` 后缀；显式配置的文件名按原值使用，因此多 worker 应在各自段中设置不同文件名。每个队列同时只支持一个消费进程；并行 worker 使用不同队列。

迁移旧 Stage2 专用 INI 时，将旧 `worktodo/finished/log_file` 值复制到对应 `stage2_*` 键。不能将一个活动队列同时交给 Stage1 和 Stage2。运行时拒绝 Stage2 队列/finished 与已解析的 Stage1 对应路径冲突，也拒绝配置、程序、源 save、队列、进度、结果和日志之间的危险路径重叠。

Stage2 INI 相对路径以 INI 目录为基准，CLI 相对路径以当前工作目录为基准；Stage1 保持原有 exe 目录规则。推荐将共用 INI 放在两个 exe 同目录，避免外部 INI 的不同路径基准产生歧义。

普通 `[queue]`、`[gpu]`、`[stage2]` 标题仅供阅读，不能隔离同名键。只有 `[Worker #N]` 定义覆盖范围。Stage1 键名保持原有大小写规则；Stage2 专用解析器不区分键名大小写。CLI > worker 专用键 > 全局键 > 内置默认；B2 为 CLI 非零值 > 任务非零值 > `stage2_b2`。只有未指定非零 B2 时才考虑 Auto B2。

上述普通标题规则仅适用于命令行驱动，GUI 把所有标题当作真实分区。发布模板使用注释分组，保证公共键也能被 GUI 读取；公共键应放在任何分区之前。

## 3. 数量不足与任务错误

请求 10 条、跳过后仅有 3 条时，提示 requested=10/available=3，执行这 3 条。全部成功后记录 completed=3，将原任务行移至 finished 并继续下一项。没有可用记录时提示并按零曲线完成；不等待 save 追加新记录。可用数指本次选择范围内的记录数，最多为请求数，不是整个文件的总记录数。

存档缺失、文本损坏、校验和不匹配、N 不匹配、不可用 X 或任务字段错误：记录原因，将原队列行变为 `# ERROR ...` 注释，在 finished 留下错误原因与原任务，继续下一项。修复后可手动恢复该任务行。文件权限/读写故障、进度状态不一致、CUDA/显存/设备故障及未知内部故障停止进程，保留未完成任务。队列存在输入错误时，即使后续项成功，最终退出码为 1；致命错误为 2，正常完成为 0。

整项所选记录先验证，发现损坏不会将其当作“数量不足”略过。源 save 从不修改。

## 4. 逐曲线续跑与退出

仅队列自动续跑，保存完成曲线数与当前尝试的随机回执，不保存 Stage2 中间计算状态。

1. 启动曲线前，将尝试回执原子写入进度文件。
2. 子进程完成算术检查和因子验证，追加结果及回执，刷新到磁盘。
3. 父进程收到成功退出并确认回执后，原子更新完成曲线数。
4. 如果在 2 与 3 之间中断，重启扫描结果回执，补记完成，避免重复执行这条曲线。

进度绑定任务、队列内容、save 完整 SHA256、选择范围、结果路径及相关计算配置。续跑期间不要编辑队列、追加/替换 save 或改变计算配置。变化时拒绝沿用未完成进度；应恢复原输入，或明确归档进度文件后重新执行。保留该限制可避免同名存档被替换、相邻重复任务及参数变化导致错误跳过。

任务完成时，finished 先写带随机任务 ID 的回执与原行，再推进队列，最后清除进度。重启可识别已写入的 finished 回执。结果文件中未以换行结束的尾行视为可能的中断写入，不用作成功回执。新追加会与残缺尾行分隔。

第一次 Ctrl+C：提示等待当前曲线完成，结果与进度落盘后退出；不再领取新曲线。第二次 Ctrl+C：立即终止子进程及其子进程，保留已有进度；当前曲线没有完整成功回执时下次从头重算。关闭窗口或杀进程同样依赖完成回执恢复，不承诺当前曲线完成。进度和结果不要单独删除其中一份。

## 5. 控制台、普通日志与调试日志

`verbose=true` 默认显示任务摘要、曲线编号、阶段切换、时长和结果；false 保留曲线摘要。长阶段每 30 秒刷新一行运行状态。这是父进程仍在等待的提示，不是 GPU kernel 完成百分比。

阶段包括存档验证、选形与内存规划、baby 点、F 树、Newton 逆、giant/G 树/fold、归一化与下降、块/叶 GCD、最终验证及写入结果。阶段提示的 `previous` 是上一个阶段的墙钟间隔，`elapsed` 是该曲线累计时间；giant/G 树/fold 在批次循环中交织执行，控制台合并显示，详细分项保留在文件。

普通日志追加保存阶段、批次及结束统计；警告和错误也显示在控制台。开启 `stage2_debug_log=true` 后，调试细节写入独立文件；它不会因为 `verbose=true` 自动开启。共用 `debug_log` 可作为兼容开关，Stage2 专用键优先。旧 `stage2_log_level` / `--log-level` 继续作为控制台等级覆盖；旧 debug 等级映射为开启独立调试日志，普通输出最高保持 batches。

普通日志及结果默认不轮转，长期运行由用户归档。子进程的 CUDA 上下文与临时内存在每条曲线结束后释放。

## 6. 仓库构建与打包

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\release_stage2.ps1 `
  -Arch sm_89 -SplitCompile 6
```

默认构建目录 `build_cuda_cmake/release_ux_sm89`，打包目录 `dist/cuda-stage2-sm89`。脚本验证 production 算术选择、源文件哈希及 exe 哈希，再复制运行时、模板、说明和构建/包清单。`-SkipBuild` 只用于打包已构建且源码哈希仍匹配的产物。

使用 `-Stage1Exe <已验收的ecm_cuda.exe路径>` 可把既有 Stage1 程序一起放入包；该脚本不会替它重编译或宣称已验证其架构。默认包只包含本轮构建的 Stage2；Stage1 多架构构建仍用 `tools/build/release_build.ps1`。系统需安装对应的 Microsoft C++ x64 运行库以及支持所用 CUDA 工具链的 NVIDIA 驱动。

## 7. 后续 TODO

考虑向后兼容，未来合并两个可执行文件，配置 Stage1 完成后直接继续 Stage2，或交给 Prime95/独立 CUDA Stage2 消费；明确交接确认、续跑归属和重复消费防护。该项不纳入本轮实现。新 Auto B2 成本标定、多架构资格验证和完整发布验收继续独立推进。


## 8. 本轮构建记录与源码入口

2026-10-08首次发布体验候选（配置统一重构前）完成sm_89/split6独立production构建及主机端收尾重链；exe为2,993,152 bytes，SHA256为 `92fd5c6b7efef55bebc8f4e07c77f29025198d2dca31ac808187d428691c91f9`。最终LF源码构建：CUDA编译104.4秒，主机端驱动编译11.5秒；此前也完成一次哈希匹配的主机端单独重链。

同日完成配置统一重构后，Stage2通过 `-HostOnly` 复用匹配的CUDA对象并重新编译、链接：exe为2,995,712 bytes，SHA256为 `8d27c4107a7c6ccfddf83a67c040163cf7de2eef37f7b80ea98dfd161d51a389`。最终主机端编译：驱动9.2秒、表达式3.8秒、队列4.8秒、配置4.3秒。GUI编译/链接通过，exe为1,136,128 bytes。生成器 `--check`、原生CMake配置检查通过，未运行功能或GPU回归。

随后按 undoc 风格为 83 个正式键和 17 个兼容别名补齐用户说明，参考文档与 INI 注释从同一 `description` 生成。当前 Stage2 包再次通过主机端编译/链接，exe为2,995,712 bytes，SHA256为 `69645f9b904c10a0d95e8ee54289873e3a3e8d2bf90911d59b71144159478a89`；编译耗时为驱动7.2秒、表达式3.4秒、队列4.0秒、配置3.6秒。中文首次启动模板以 UTF-8 字节嵌入，未运行功能或GPU回归。

包位于 `dist/cuda-stage2-sm89`，完整源/对象/工具链清单在 `build_cuda_cmake/release_ux_sm89/build_manifest.json`，包清单为 `package_manifest.json`；配置定义、生成器、生成清单和头文件纳入源码哈希。

- 严格队列读取与save选择：[records](../src/core/ecm_cuda_stage2_main.cpp#L170)、[queue_fields](../src/core/ecm_cuda_stage2_main.cpp#L347)。
- Stage2专用INI解析：[生成Settings与共享解析](../src/core/ecm_cuda_stage2_main.cpp#L315)、[统一配置定义](../config/ecm_options.json)。
- 结果落盘与子进程日志/中断管理：[append与child_run](../src/core/ecm_cuda_stage2_main.cpp#L386)。
- 曲线结果回执：[curve_worker](../src/core/ecm_cuda_stage2_main.cpp#L620)。
- 队列进度绑定、恢复与提交：[调度循环](../src/core/ecm_cuda_stage2_main.cpp#L911)。
- 原子进度与回执对账：[ecm_stage2_queue_state.h](../src/core/ecm_stage2_queue_state.h)。
- 可读日志与独立诊断：[ecm_stage2_logging.h](../src/core/ecm_stage2_logging.h)。
- 单架构发布打包：[release_stage2.ps1](../tools/build/release_stage2.ps1)。
