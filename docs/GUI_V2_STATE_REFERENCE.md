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

## ui.scale

默认值：`0`；类型：`float`。

Zero selects window DPI; positive values select a fixed interface scale.
零按窗口 DPI 自动计算；正值指定界面比例。

## ui.page

默认值：`0`；类型：`int`。

Selected central page: workers, results, generator, probability, GPU, settings, components.
中央页面：worker、结果、生成器、概率、GPU、设置、组件。

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

Show the log dock.
显示日志停靠窗。

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

Bit mask of pages moved to the right tool dock; at least one page remains central.
移入右侧工具栏的页面位掩码；至少保留一个中央页面。

## ui.right_page

默认值：`4`；类型：`int`。

Selected right tool page.
当前选中的右侧工具页面。

## ui.right_visible

默认值：`true`；类型：`bool`。

Show moved pages in a dockable right panel.
在可停靠的右侧窗显示移入的页面。

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
