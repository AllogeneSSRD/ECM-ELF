# GUI v2 侧车配置参考

由 `config/ecm_options.json` 的 `gui_v2_sidecar` 生成。独立 UI 文件不属于业务 TOML；不会写回旧 `[gui]`。

## state.schema

默认值：`1`；类型：`int`。

UI sidecar schema; incompatible versions are rejected.
UI 侧车版本；拒绝不兼容版本。

## ui.theme

默认值：`"dark"`；类型：`string`。

Appearance mode, independent of legacy GUI settings.
外观模式，独立于旧 GUI 设置。

## ui.language

默认值：`"en"`；类型：`string`。

UI language; CJK font failure falls back to English.
界面语言；中文字体不可用时回退英文。

## ui.language_file

默认值：`""`；类型：`string`。

Interface XML filename in localization/v2 beside the executable. Empty migrates the legacy en/zh selection. Companion options.*.xml files are not selectable interface files.
程序旁localization/v2目录中的界面XML文件名；空值按旧en/zh选择迁移。配套options.*.xml不是可选界面文件。

## ui.scale

默认值：`0`；类型：`float`。

Zero selects window DPI; positive values select a fixed interface scale.
零按窗口 DPI 自动计算；正值指定界面比例。

## ui.font_size

默认值：`15`；类型：`int`。

Body and log font size in logical pixels, independently of interface DPI scale. Primary actions and secondary text apply their configured offsets; headings preserve their ratio.
正文与日志字号（逻辑像素），独立于界面 DPI 比例；主要操作与次要说明使用各自偏移，标题保持相对比例。

## ui.primary_font_offset

默认值：`1`；类型：`int`。

Primary action font offset from the body size, in logical pixels; secondary buttons and log text keep the body size.
主要操作字号相对正文的偏移（逻辑像素）；次要按钮和日志正文保持正文字号。

## ui.secondary_font_offset

默认值：`-2`；类型：`int`。

Secondary text font offset from the body size, in logical pixels; the rendered role has a minimum of 8 logical pixels.
次要说明字号相对正文的偏移（逻辑像素）；实际字号最小为 8 逻辑像素。

## ui.worker_chart_columns

默认值：`1`；类型：`int`。

Curve-time chart columns: 1 for list, 2 for two charts per row; narrow workspaces temporarily use one column without changing the saved choice.
曲线耗时图列数：1为列表，2为每行两图；工作区过窄时临时使用单列，不改变保存的选择。

## ui.gpu_history_seconds

默认值：`60`；类型：`int`。

GPU chart time span in seconds, shared by all devices and persisted with UI preferences. Does not change the 500 ms sampling interval.
GPU 图表时间范围（秒），所有设备共用并随界面偏好保存；不改变 500 ms 采样间隔。

## ui.stage2_timing_source

默认值：`"auto"`；类型：`string`。

Stage2 segment duration source: auto overlays previous completed-curve timings on tune data; previous or tune restricts the source. Missing durations are marked unknown.
Stage2区段时长来源：auto按上次完成曲线覆盖tune数据；previous或tune仅使用指定来源，缺项明确标为未知。

## ui.stage2_tune_file

默认值：`""`；类型：`string`。

Optional read-only Stage2 tune summary path. Empty uses the workflow stage2_tune_profile, then tune.toml or stage2_ecm_tune.toml next to the workflow/worker. Relative paths use the workflow directory.
可选的只读Stage2 tune摘要路径。空值使用工作流stage2_tune_profile，再查找工作流/worker旁的tune.toml或stage2_ecm_tune.toml；相对路径基于工作流目录。

## ui.nav_order

默认值：`"01237456"`；类型：`string`。

Stable page IDs 0..7; ID 5 opens Settings. Legacy seven-page orders append Group Order Factor without resetting the layout.
稳定页面编号0..7；5打开设置。旧七页排序追加群阶分解页，不重置布局。

## ui.page

默认值：`0`；类型：`int`。

Selected page: 0 workers, 1 results, 2 generator, 3 probability, 4 GPU, 6 components, 7 Group Order Factor; 5 opens Settings.
页面：0任务、1结果、2生成器、3概率、4 GPU、6组件、7群阶分解；5打开设置。

## ui.worker

默认值：`1`；类型：`int`。

Selected worker partition.
选中的 worker 分区。

## ui.inspector

默认值：`true`；类型：`bool`。

Show the worker inspector on sufficiently wide windows.
宽窗口显示 worker 检查区。

## ui.logs

默认值：`true`；类型：`bool`。

Enable logs in Workers management; automatic hiding on other views does not change this saved preference.
开启Workers管理页日志；其他视图的自动隐藏不改变此保存偏好。

## ui.follow_log

默认值：`true`；类型：`bool`。

Follow new worker output.
跟随新的 worker 输出。

## ui.raw_log

默认值：`false`；类型：`bool`。

Show raw output instead of classified event lines.
显示原始输出而非分类事件。

## ui.dock

默认值：`""`；类型：`string`。

Hex encoded ImGui layout; stored only in this sidecar.
十六进制编码的 ImGui 布局，仅存于本侧车。

## ui.right_pages

默认值：`0`；类型：`int`。

Real pages in the right dock share native horizontal tabs. Closing returns a page to navigation; one real page remains central. Settings bit 5 is cleared when opening old sidecars.
右侧实际页面共用横排标签，关闭后返回导航，至少保留一个实际中央页面；读取旧侧车时移除设置位5。

## ui.right_page

默认值：`4`；类型：`int`。

Last active real page in the right dock. Legacy Settings ID 5 is remapped on load.
右侧停靠区最后激活的实际页面；读取旧设置编号5时改选实际页面。

## ui.right_visible

默认值：`true`；类型：`bool`。

Compatibility field retained for old sidecars. Right-side pages are always displayed; close their tabs to return them to navigation.
保留以读取旧侧车的兼容字段；右侧页面直接显示，关闭标签即可回到导航。

## ui.smooth_scroll

默认值：`true`；类型：`bool`。

Smooth wheel scrolling in main and detached windows.
主窗口与分离窗口使用平滑滚轮滚动。

## ui.toasts

默认值：`true`；类型：`bool`。

Show non-activating action notifications; errors always remain visible.
显示不抢焦点的操作轻提示；错误提示仍显示。

## ui.external_windows

默认值：`false`；类型：`bool`。

Allow detached native panel windows after restart; default keeps docking and floating inside the main window to avoid the observed slow presentation path.
重启后允许原生独立面板窗；默认在主窗口内停靠/浮动，以规避已观察到的慢呈现路径。

## ui.worker_chart_auto_y

默认值：`false`；类型：`bool`。

Fit curve-time Y bounds to finite samples in the current retained window; off keeps a zero baseline.
曲线耗时Y轴按当前保留窗口的有效样本自动缩放；关闭时从0开始。

## ui.logs_keep_visible

默认值：`false`；类型：`bool`。

Keep enabled logs visible across pages; otherwise follow visible Workers management, with a temporary manual override on other views.
已开启日志跨页保持可见；关闭时跟随可见的Workers管理页，其他视图可临时手动打开。

## ui.nav_hide_tab_bar

默认值：`false`；类型：`bool`。

Remember the native Navigation dock Hide tab bar choice across restarts and layout rebuilding.
持久记忆导航停靠窗Hide tab bar选择，布局重建后保留。

## ui.workspace_hide_tab_bar

默认值：`false`；类型：`bool`。

Remember the native Workspace dock Hide tab bar choice across restarts and layout rebuilding.
持久记忆工作区停靠窗Hide tab bar选择，布局重建后保留。

## ui.right_hide_tab_bar

默认值：`false`；类型：`bool`。

Remember the right dock Hide tab bar choice. Multiple native tabs remain visible; the choice resumes when one remains.
持久记忆右侧停靠窗Hide tab bar选择；多个原生标签保持可见，恢复单标签时恢复该选择。

## window.x

默认值：`-2147483648`；类型：`int`。

Window x in physical pixels; INT_MIN selects system placement.
窗口物理像素 x；INT_MIN 由系统放置。

## window.y

默认值：`-2147483648`；类型：`int`。

Window y in physical pixels; INT_MIN selects system placement.
窗口物理像素 y；INT_MIN 由系统放置。

## window.width

默认值：`1280`；类型：`int`。

Restored window width in physical pixels.
恢复窗口的物理像素宽度。

## window.height

默认值：`850`；类型：`int`。

Restored window height in physical pixels.
恢复窗口的物理像素高度。

## window.title_bar

默认值：`false`；类型：`bool`。

Show the system title bar; hidden by default with window controls in the app top bar. Explicit saved choices are retained.
显示系统标题栏；默认隐藏，窗口按钮位于应用顶栏；保留显式保存的选择。

## window.maximized

默认值：`false`；类型：`bool`。

Restore the maximized state.
恢复最大化状态。

## worker.stage1_exe

默认值：`""`；类型：`string`。

Explicit Stage1 worker executable.
显式 Stage1 worker 程序路径。

## worker.stage2_exe

默认值：`""`；类型：`string`。

Explicit Stage2 worker executable.
显式 Stage2 worker 程序路径。

## worker.count

默认值：`0`；类型：`int`。

Number of v2 workers; 0 follows the workflow GUI worker count. Does not modify legacy preferences.
v2 worker 数量；0 跟随工作流中的数量，不改写旧版偏好。

## pet.enabled

默认值：`false`；类型：`bool`。

Show the ELYTerminal-compatible edge companion.
显示兼容 ELYTerminal 行为的窗口边缘宠物。

## pet.animate

默认值：`true`；类型：`bool`。

Animate at the existing GUI cadence; no extra render loop.
按既有 GUI 节奏动画，不另开渲染循环。

## pet.left

默认值：`72`；类型：`float`。

Companion offset from the main window left edge in logical units.
宠物相对主窗口左缘的逻辑单位偏移。

## pet.skin

默认值：`"classic"`；类型：`string`。

ELY appearance. Switches immediately; official battlesuit names are preserved in the selector.
ELY 外观，即时切换；选择项使用装甲的完整英文名称。
