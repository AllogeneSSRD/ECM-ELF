# GUI Stage2 与任务摘要草稿

## 已确定需求

当前任务不显示完整worktodo。已知因子替换为“已知因子”，存在AID时只显示固定字样“assignment ID”；用户已确认隐藏真实ID。原始文件/日志保留完整数据。GUI加入Stage2运行能力，适度整理界面与输出。

## 实现

- 一实例选择Stage1或Stage2，顶部程序/INI选择只在全部停止后可用。保留Stage1程序配置，Stage2单独路径；同目录或唯一子包可发现，多个包须明确选择。读取现有Stage2-only目录的INI可自动识别该阶段，切换不启动任务。
- Stage2启动使用--ini/--worker、GUI事件与继承停止事件。只传递任务元数据及安全停止，不改算法。GUI专用START事件保持当前任务与计算进程一致；普通命令行未开启该事件时日志不增加任务行。
- 曲线完成数、恢复进度、phase、giant批次分别显示，阶段ETA不混入任务ETA；引擎总时间构建速度估计。GUI结果汇总读取正式结果中的全部因子与当前记录/sigma。
- task_display为显示层解析；异常任务不回显原行。表、提示、输出头部和详情共用摘要。工作程序仍用真实AID和因子校验及计算。
- 安全停止Stage1写检查点，Stage2完成当前曲线；沿用300秒默认等待上限，超时/手动强制退出可能重做未完成曲线。取消等待自动重启，避免停止按钮无效。
- 青绿/紫色进度、后台阶段条、短路径及日志可见行绘制。中英资源同时维护。配置只在ecm_options.json维护并生成。

## 构建与验证边界

GUI使用tools/build/dev/build_gui.ps1，目标只选ecm_gui；Stage2复用已核验的sm89 CUDA对象，重编主机协议。首轮GUI编译发现read_u64的指针引用不能绑定临时指针，改为局部变量。构建日志及最终产物身份留在忽略实验目录。

不运行新增测试或GPU任务。后续需验证真实Stage1/Stage2切换、共享/独立INI、多worker分区、任务摘要、恢复完成数、batch阶段、单/多因子、Stop/超时/强制退出和视觉布局。正式当前规则统一见[GUI说明](../usage/GUI.md)。草稿保留，完成验证后提示用户处理，不自动删除。

生产发布在进程检查时暂缓：生产目录存在旧GUI及Stage1进程。受限环境的初次进程枚举没有显示它们，使用生产可见的只读检查确认PID48684/50792仍在运行。替换操作在任何生产写入前退出；已询问用户安全停止并关闭后继续。新二进制及构建身份先冻结在`data/experiments/gui_stage2_20261014/`，未运行GUI自动化或GPU计时。

最终GUI SHA256为`9e0649ab76f22cf7c4dab3b60b917441314c09a7e9698bd12ea62bc75afed3a0`，配套Stage2为`652f6da7f617b964cca7d99a586a50bf7a28842ec00036ef0d0a3ec41b35224d`。进程结束后先结束残留的受管子进程、将输出管道读至EOF，再判断退出状态，避免丢失最后的因子/队列结束事件；该分支仍列入运行验证。

## 生产目录更新

用户授权安全停止并关闭旧程序后，生产可见的进程检查确认没有该目录的GUI、Stage1或Stage2进程。已将GUI、中英资源和配置说明更新至`D:\code\GIMPS\ecm-win-x86_64-sm89-cuda13.3_param0`，将配套Stage2和配置说明更新至其`cuda-stage2-sm89`子目录，共6个文件；发布哈希与上述最终构建一致。19个现有INI、队列、调优及曲线进度文件逐项哈希不变，GMP DLL与构建配套版本相同且保持原样。

旧文件备份和发布身份位于`data/experiments/gui_stage2_20261014/publication_before/`及`publication.json`。首次替换在Windows接口对空备份路径的转换处中止，未替换生产文件；使用显式备份路径完成替换。未启动GUI或执行GPU曲线，运行及视觉验收仍待验证。当前功能说明仍以[GUI说明](../usage/GUI.md)为准；本草稿保留，待用户决定整理。

## 共用目录与混合worker

用户要求同一生产目录及ecm.ini分别运行Stage1/Stage2 worker，并确认越界条位于Worker管理表格主行。新增worker.stage覆盖GUI默认stage，统一make_worker构建每个worker的协议、程序、队列、路径与停止行为；只重配停止的目标worker，其他worker的进程与统计不重建。界面加入默认阶段、worker数量、两个程序路径和行尾阶段选项。生成器绑定明确目标worker，避免相同GPU的不同阶段混分任务。INI新建键重新投影worker_scope，生效配置及队列预览与原生解析规则一致。

固定列加最小进度列超过面板宽度时启用横向滚动，进度条使用明确宽度、有限比例及一致裁剪；跨列摘要使用独立行高与draw-list文字，不改变第一列宽度。暂未进行窗口复现和视觉验收：本轮按开发约束只进行构建和发布校验，原生窗口自动化不可用。后续需用户核对相同字体、窗口宽度下的总进度条。

生产INI中stage2/GUI位于Worker #2后，其命令行范围仍属于worker 2。计划将共享设置及GUI放在worker前，设置Worker #1 Stage1/GPU0、Worker #2 Stage2/GPU1，autostart均为0。用户补充确认原worktodo的Worker #2任务已完成Stage1，应迁移到Stage2队列的Worker #2，不转回Stage1。保持原任务参数、save文件、进度及实验数据。

首轮编译发现Stage1Values字段为curve_model/execution_target，已修正GUI对规范字段的读取；后续构建及发布身份保存在data/experiments/gui_mixed_workers_20261014/。

混合worker版本编译通过并更新生产目录，GUI SHA256为`7d74e26f78640e87722381c0970266c97fcd197bcf23e59e2c1eba031769db7e`。共8个文件更新，旧文件备份在`data/experiments/gui_mixed_workers_20261014/publication_before/`，发布身份见同目录`publication.json`。Stage1/Stage2计算程序保持原样，Stage2协议二进制仍为`652f6da7f617b964cca7d99a586a50bf7a28842ec00036ef0d0a3ec41b35224d`；没有因本轮GUI改动重编或运行算术内核。

M12163任务从Stage1队列worker 2移至Stage2队列worker 2，原B2=44,000,000,000、skip=0、curves=480及因子参数保持原样。save有480条记录、B1均为22,000,000、N表达式一致且均有checksum字段；这是只读结构检查，没有执行算术/checksum完整验证，原生Stage2仍须逐记录验证。Stage1队列剩余11条任务保持原样；38个save/检查点/调优文件及计算程序哈希不变。原Stage2队列的错误注释保留为证据，不重新激活其任务。worker均为autostart=0。

发布后仍待验证混合进程运行、单worker切换不扰动另一worker、共享队列配置及用户指定大字体窄面板的总进度条；未启动GUI或GPU任务。草稿保留，待用户决定整理。

用户补充调整输出要求：日志窗口START行保留原始worktodo，只在当前任务摘要中替换因子和AID。删除GUI事件接收后的START改写，原生stdout本就输出完整任务，无需改动计算程序。自动滚动选项的英中文案已明确为Auto-scroll to bottom／自动滚动到底部。

该输出调整构建通过并发布至生产GUI，SHA256为`c99ecb62e29470b277d30c0502ef131b6bd0ecc606089e6ee7beef3c210feb2a`；构建、旧程序及发布身份保存在`data/experiments/gui_start_raw_20261014/`。未进行界面或GPU运行验证。
