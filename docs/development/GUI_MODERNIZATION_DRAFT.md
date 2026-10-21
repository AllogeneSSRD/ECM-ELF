# 独立 GUI v2：ImGui / Vulkan 现代化规划草稿

状态：P0 已完成，P1–P2 待实施。日期：2026-10-21。原 GUI 保持不变；独立构建 GUI v2，采用 C++ / Dear ImGui / Win32 平台后端 / Vulkan 渲染后端，参考 Ely 组件库统一布局、配色、字体和控件。v2 是前端名称，不表示公共程序版本变为 2.x。

本文件是本主题的权威临时草稿。[HTML 阅读版](../web/gui-v2-vulkan-plan-20261021.html)呈现布局示意、阶段验收和本草稿全文，可离线打开。实施成功后另写 GUI v2 正式说明，旧 GUI 使用说明继续描述旧版；提示用户处理本草稿，不自动删除。

## 实施进度

- P0：已完成，v2 里程碑 0.47.0。旧 GUI 74 个源码、后端、配置和测试文件已冻结；原 exe 单独复制到隔离目录，SHA256 保持 `32f3a9a2905ff39e6f098f0c943dedbaa3bdbea291ef7122417634958b85919b`。旧版公共标识保持 0.46.7，v2 使用独立 `version.h` 记录 0.47.z。
- P0 验证：自检 33、日志 85、生成器 110、结果 26、Controller 44 项/10 场景全部通过。假 worker 输出、工作流与结果输入、英文/中文窗口截图和五次独立空闲采样保存在 [P0 证据](../../data/experiments/gui_v2/p0_20261021_03/baseline.json)。未运行真实计算 worker。
- P0 测量边界：隐藏普通窗口，预热 1 秒后采样 2 秒，界面 10Hz / NVML 500ms；单核 CPU 百分比中位数 1.5625%，working set 中位数 116867072 bytes，private bytes 中位数 117784576 bytes。现场另有 GPU 负载，未隔离功耗策略；这些是冻结输入条件下的原始基线，不构成性能优劣结论，也没有输入延迟/显存/功耗测量。
- P0 原始失败保留：第一轮测试配置误将 worktodo 放入 stage1，改到 shared；第二轮隐藏窗口截图被系统拒绝，改为截图时短暂非激活显示。未改旧实现、测试期望或验证门限。有效证据为 `_03`。
- P1：待实施。先固定专用 Vulkan-Headers / Volk 和同提交 ImGui Vulkan 后端，验证宿主与旧版构建隔离。
- P2：待实施。主题、字体和控件仅进入 v2；P3–P6 不在本批完成范围。
- 草稿与 HTML 由 `tools/gen/render_gui_v2_plan.py` 同步，原始证据保留在忽略目录。各阶段独立提交 `[0.47.z]`。

## 1. 分析边界与基线

- 仓库基线：本次修订检查时 HEAD 为 `8969467`，工作区已有其他开发中的改动；分析包含工作区源码，不能视作该提交的纯净快照。
- 公共源码版本与 `build_gui_local/ecm_gui.exe --version` 均为 `ECM-ELF 0.46.7`。相同版本号不能独立证明源码和二进制完全一致。
- 本次修订核对的旧 GUI SHA256：`32F3A9A2905FF39E6F098F0C943DEDBAA3BDBEA291EF7122417634958B85919B`。这不是将来 v2 的构建身份；P0 仍须冻结实际验收基线。
- 当前 GUI 统一构建入口为 `tools/build/build_gui.bat`，默认输出 `build_gui_local/`，不另设 dev/local/release 三套 GUI 构建。见 `tools/build/README.md`、`tools/build/dev/build_gui.ps1`。
- 初始规划的布局依据源码重建；实际验证与完成范围以本文件“实施进度”为准，未验证的输入法、跨屏和性能仍不能视作通过。
- 本机二进制不当作对外发布版本。本批按 P0–P2 建立新版里程碑，不修改旧版公共版本、不发布安装包。

### 1.1 原 GUI 不变的边界

- 保留 `src/ecm/application/gui/` 的源码、Win32/DX11 渲染路径、`ecm_gui` 目标、`ecm_gui.exe`、`tools/build/build_gui.bat` 及其默认输出 `build_gui_local/`。
- 不向旧目标加入 Vulkan 库、宏、SDK 查找或新样式；不改旧 ImGui 核心、Win32/DX11 后端、默认布局、字体、配置默认值及现有测试期望。
- v2 使用新增 `src/ecm/application/gui_v2/`、`ecm_gui_v2` 目标、`ecm_gui_v2.exe`、独立构建入口和输出目录。根构建系统只增加受 v2 开关保护的接线，不改变旧入口的默认行为。
- 可直接编译复用旧目录中不依赖 UI 的源码；为此建立 v2 专用源码清单/库，不移动、不提取改写旧模块，不改旧目标的链接关系。若 v2 需要不同业务行为，另做 v2 适配，不能偷偷修改被旧版使用的实现。
- v1/v2 UI 状态分别保存。v2 不导入或改写旧 `[gui] dock_layout`、窗口位置、字体、主题及布局版本；v2 不使用旧 App 的保存整个 GUI 设置路径。
- 共用工作流 TOML、任务队列和发布文件是业务数据，不等于共用 UI 状态。v2 只有用户显式编辑计算配置时才写业务字段；保存 v2 外观/窗口状态不能顺带改写旧 `[gui]`。并行查看结果可行，同一 worker/队列组的进程控制必须有单一拥有者，不能让两个 GUI 重复启动同组任务；拥有权检查及旧版并行行为列入 P3 验证。

参考快照：

- `.refactor/Ely-GPUI-Components`：`17f936b98c8c8fbdd5454745ac70c22df8526689`。主要依据当前 `src/theme/`、`src/buttons/`、`src/forms/`、`src/lists/`、`src/shell/`。
- `.refactor/ELYTerminal`：`fe07813fdfa9fd600274066d9fcd88d5f5a3bcc0`。主要参考实际分栏组织、侧栏动作及背景处理，不引入终端引擎。
- 组件库 `frontend/public/shots/` 的深色窗口与浅色路径输入截图已用于辅助观察。图库截图与当前代码可能存在版本差异，具体数值以源码为准。
- 参考目录仅供分析；GPUI/Rust 组件不直接链接到 C++ GUI。只迁移本项目所需的设计规则与交互，保持 ImGui 原生输入、焦点和停靠机制。

## 2. 旧 GUI：作为 v2 行为参照的现状分析

### 2.1 布局

`app.cpp::build_default_layout`（约 627 行）将停靠区分为左 60% / 右 40%，再各上下对半：

- 左上：worker 管理、详情、生成器、概率为同一停靠节点的页签。
- 左下：每个 worker 的输出窗。
- 右上：GPU 监控。
- 右下：Result。

上方另有菜单和 Prime95 状态条，`app.cpp::draw`（426 行）测量并预留顶部空间。所有窗口具有稳定 ID，`on_imgui_ready`（325 行）开启 docking 和多 viewport，布局写回 TOML。

主要矛盾是空间分配：默认主页面只获得停靠区约 30% 面积；11 列 Result 固定在约 20% 面积的右下区域。配置、生成器、概率这种较复杂的页面与 worker 表共享狭小区域。比例是源码几何推算，不是用户使用频率或实测可读性结论。

worker 管理顶部还同时容纳数量、两个程序路径、TOML 路径和全局启动/停止，见 `draw_workers_table`（1741 行）。高频操作与低频配置挤在同一区域。

### 2.2 配色和控件

- `app.cpp` 的日志、worker 状态、Stage1/Stage2 进度、命中高亮、Prime95 提示分别指定颜色；`probability_panel.cpp` 另有独立的背景和曲线常量。
- 未见应用级明暗主题入口或统一的语义样式模块；局部 `PushStyleColor/PushStyleVar` 与绘图颜色共同决定最终外观。
- 当前已经有表格列宽保护、进度与文字同区裁剪、任务摘要独立测高、日志可见行绘制等有效行为，应保留，不因外观改造重新实现业务逻辑。
- v2 的首要任务是统一状态语义、尺寸和控件状态，而不是扩大圆角或给所有面板加卡片。本节发现不授权改造旧 GUI。

### 2.3 字体和 DPI

`main_win32.cpp::apply_ui_font`（391 行）根据语言和配置选取字体：中文优先系统 CJK 字体，英文优先 Segoe UI；自动字号为 `15 × DPI scale`，手工字号为显式像素尺寸，支持小数。中文字体缺失时有救援和回退到英文的处理。

当前主要使用单一默认字体，尚无正文、标题、注释和等宽数据的角色管理。样式 `ScaleAllSizes` 在启动阶段执行；语言切换会重载字体。没有据此确认跨屏 DPI 已损坏，但应用层完整的样式/字体重建及浮动窗跨屏行为需要专项核验，不能直接承诺已支持。

### 2.4 性能和职责边界

- `WorkerController` 拥有 worker 状态和生命周期；`WorkerProc` 拥有原生进程资源。App 借用只读 `WorkerView`，每帧调用 `controller_.tick()`。这条边界适合在不改计算逻辑的情况下改造视图。
- 默认界面刷新 10 Hz、NVML 采样 500 ms，输入消息提前唤醒；未选中的页签跳过内容工作。日志有 4000 行上限，采用 `ImGuiListClipper`。
- Result 可见时每 500 ms 检查来源，变化后同步读取/解析；大文件可能阻塞界面线程。新主题不解决这项限制。若基线测量证明影响交互，另列性能任务，不借机改变发布记录合同。
- 不能把低刷新实现描述成完全事件驱动。若以后跳过绘制，需要继续调度 Controller、退出协调、文件检查和必要的字体纹理处理。

依据：`docs/usage/GUI.md`；`app.cpp::draw_results_panel`（2234 行）；`main_win32.cpp`（约 849–881 行）。

## 3. 从 Ely 提炼的设计规则

### 3.1 颜色：中性背景，状态和数据使用颜色

组件库 `src/theme/palette.rs` 定义成组语义：`bg/surface/sunken/overlay`、`hover/active`、`border/border_strong`、`fg/fg_muted/fg_subtle/fg_disabled`、`accent/on_accent`、`focus/link/selection`、`success/warning/danger/info`。

当前上游深色基础值为背景 `#252525`、表面 `#2c2c2c`、内嵌 `#202020`、文字 `#f3f1f0`、边框 `#3f3f3f`；浅色对应 `#fcfaf7`、`#ffffff`、`#f6f4f1`、`#181613`、`#e1dfdd`。这些是参考值，尚未接入 ECM GUI。

采用以下规则：

- 主按钮使用 Ely 的中性高对比强调色；焦点蓝色不等于所有按钮都涂蓝。
- 工作状态单独映射为运行、等待、错误、停止等语义，必须同时有文字；Stage1/Stage2 和概率曲线的青绿/紫色是数据身份，不等同成功/失败。
- 青绿、紫色可保留现有辨识关系，针对明暗背景分别取值；图例、曲线和悬停数据保持一致。
- Prime95 的未配置、pending、失败、成功均保留原语义。可缩为状态项，失败或积压仍有显著提示，完整说明与打开动作仍可访问。
- 所有应用级颜色通过统一主题读取，包括手绘图表背景；切换浅色时不留下硬编码深色矩形。

### 3.2 尺寸：逻辑单位与文字共同决定实际高度

Ely 的 `src/theme/tokens.rs` 给出：

- 标准控件高 Sm/Md/Lg = 24/28/32；Compact 减 4，Comfortable 加 4。
- 横向内边距 8/12/16；圆角 4/6/8/12；图标常用 14/16/20。
- 表格行高按密度为 28/36/44；侧栏展开 240、折叠 56；标题栏 38。
- 上游字号为 11/12/13/14/16/20/24/32，默认正文 13。不能照搬这些字号并把当前 15px 正文缩小当作优化。

v2 建议先采用与旧版一致的正文自动 15px 和手工字号含义，但在独立 UI 状态中保存，定义正文、辅助说明、标题、等宽数据角色。控件高度取逻辑尺寸与实际字体/内边距所需高度的较大值，不能为套 28px 而裁切中文。第一版提供 Standard 密度，Compact 在中文与缩放通过后再开放。

间距使用少量统一档位，例如 4/8/12/16/24 逻辑单位；图标和间距按 DPI 计算。每次从未缩放基准重新计算样式，避免重复 `ScaleAllSizes` 累乘。

### 3.3 字体与资产

上游 `src/theme/mod.rs` 使用 Inter 和 JetBrains Mono，图标为 Lucide。建议将“外观接近”和“必须换字体”拆开：

1. 先用现有系统字体完成层级、间距和组件，保留 CJK 救援机制。
2. 需要更一致的拉丁文字时，再将 Inter / JetBrains Mono 作为可选随包字体。中文仍需 CJK 合并或明确的分段回退；不能期望 Inter/JetBrains Mono 自动包含中文。
3. 等宽字体优先用于日志、原始 JSON 和数值；导航、表单与帮助文本使用 UI 字体。数据按列右对齐，避免靠空格对齐。
4. 只随包需要的图标；可离线生成有限图集，按 DPI 缓存，不能每帧解析 SVG。不复制整套组件库资产。

移植或分发上游代码/资产时按仓库 LICENSE 保留 Ely MIT、Lucide ISC 和所选字体 OFL 告知。具体清单在资产进入项目时形成；本轮不导入资产。

### 3.4 控件：风格统一，交互继续交给 ImGui

参考 `buttons/button.rs` 的 Primary/Secondary/Outline/Ghost/Subtle/Danger，以及 `forms/path.rs`、`lists/item.rs` 的结构。

第一批只建立项目需要的 Button、IconButton、NavItem、SectionHeader、FormRow、PathField、StatusBadge、Notice、MetricLabel 和进度样式。绘制装饰可自定义，但点击、输入、选择、禁用和键盘焦点优先复用 ImGui 公共控件，保留稳定 ID。

统一检查 normal/hover/pressed/disabled/focus/selected/loading；危险样式用于强制终止等破坏性动作，普通安全停止不自动变成红色主按钮。加载态使用真实任务状态，没有持续任务时不保留无限动画。

不移植 GPUI 的实体树、Flex 布局引擎、事件派发、焦点框架、WebView、终端和宠物；不为这次改造新建通用 GUI 框架。

### 3.5 Vulkan 后端：新增宿主职责与验证门槛

本地 `third_party/imgui/backends/` 目前只有 Win32 和 DX11 文件，没有 Vulkan 后端。本次环境检查没有 `VULKAN_SDK` 变量和 `C:/VulkanSDK/` 目录；这不能证明其他位置没有 SDK，P1 仍需显式查找并记录。当前规划不安装 SDK、不下载依赖。

初始接入固定为：现有 ImGui 核心及 Win32 后端只读复用；将**同一提交** `64944b4520b30772de8dbf0b37d0311746477a32` 的 `imgui_impl_vulkan.h/.cpp` 新增在 v2 专用第三方目录，例如 `third_party/imgui_vulkan/`，记录哈希与许可。不得只取最新 Vulkan 后端配旧核心；如果固定版本不能通过 v2 验收，v2 才引入完整独立的 ImGui 版本目录，旧版仍使用原固定版本。

平台和渲染职责分开：`imgui_impl_win32` 继续处理 HWND、输入、IME、剪贴板和平台窗口；`imgui_impl_vulkan` 绘制 ImGui 数据并处理其纹理。主窗口的 Vulkan instance、设备/队列选择、Win32 surface、swapchain、命令缓冲、同步、呈现和资源生命周期由 v2 宿主管理，不能把换后端理解为只替换一个初始化函数。

P1 至少完成：

- 明确启用 Win32 surface 与 swapchain 所需扩展，按实际 surface 的 graphics/present 能力选择队列与设备。记录渲染设备及选择依据；GUI 渲染 GPU 不与 worker 的 CUDA/OpenCL 设备编号绑定。
- 初始采用传统 render pass / framebuffer 路径，先不强制 dynamic rendering、Vulkan 1.3 或额外大框架；实际最低 API/扩展集合以固定后端与目标设备验证结果为准。
- 按 surface 支持选择格式、图像数量和呈现模式，低占用优先可验证的同步呈现；不能使用官方示例的无限帧模式。帧同步与资源回收有明确拥有权，不把 `vkDeviceWaitIdle` 当作每帧节流。
- 正确处理 resize、零尺寸最小化、`OUT_OF_DATE`、`SUBOPTIMAL`、surface/device lost，避免忙轮询；渲染失败不能跳过 worker 安全停止与退出协调。
- 动态字体和图标纹理按该固定后端的 API 更新、提交和延迟释放；语言/DPI 反复切换期间不能释放 GPU 仍在使用的纹理或 descriptor。
- 开发验收使用 validation layer / debug 输出记录错误；正式运行不要求用户安装开发 SDK 或 validation layers，需检测实际 Vulkan loader/驱动能力。没有可用 Vulkan 时清晰失败，提示使用仍独立存在的原 GUI，不在 v2 内暗中切换 DX11。

多 viewport 必须独立验收。固定 Vulkan 后端依赖 `ImGuiPlatformIO::Platform_CreateVkSurface`，而同提交 Win32 后端没有提供它。v2 需要在自己的 Win32/Vulkan 桥接层根据平台 HWND 调用 `vkCreateWin32SurfaceKHR`，在正确初始化次序注册回调，不能改旧 Win32 后端源码。此回调的初始化接线归 P1：固定 Vulkan 初始化会检查平台后端宣告的多 viewport 能力，即使暂时不开浮动窗，也不能把回调注册拖到 P3。P3 再完整验证弹出窗口、跨屏和销毁/重建；不能把单窗口里程碑称为完整交付。

swapchain 图像数量变化还需核对固定 API 的实际限制：该版本 `ImGui_ImplVulkan_SetMinImageCount` 对变化路径含未支持断言，不能把它当作任意重建都可调用的通用更新器；P1 确定一致的 backend 重建/重新初始化次序，P3 验证全部 viewport 的关联资源。

固定版本后端头部还标明新建 viewport 的闪烁问题；实际严重程度、次窗口呈现模式、跨屏 DPI 与呈现支持均待测。发现问题先定位和验证，不能为了勾选功能直接打开未经验证的多窗口。

官方依据（均固定到相同提交）：[Vulkan 后端头](https://github.com/ocornut/imgui/blob/64944b4520b30772de8dbf0b37d0311746477a32/backends/imgui_impl_vulkan.h)、[Vulkan 多窗口实现](https://github.com/ocornut/imgui/blob/64944b4520b30772de8dbf0b37d0311746477a32/backends/imgui_impl_vulkan.cpp)、[Win32 平台实现](https://github.com/ocornut/imgui/blob/64944b4520b30772de8dbf0b37d0311746477a32/backends/imgui_impl_win32.cpp)、[官方 Vulkan 示例](https://github.com/ocornut/imgui/blob/64944b4520b30772de8dbf0b37d0311746477a32/examples/example_glfw_vulkan/main.cpp)。示例用于理解生命周期，不意味着本项目引入 GLFW。

## 4. v2 目标布局与功能接入

### 4.1 主窗口

默认组织为“左导航 / 中央停靠区 / 右检查区 / 中央底部日志”：

- 左侧导航：工作台、结果、任务生成、ECM 概率；设置和语言入口放底部。建议展开 200、折叠 56 逻辑单位。200 是针对 ECM 的初始候选，不是 Ely 原值 240；最终以中文标签和可用宽度验证。
- 中央主区：v2 使用 `###v2_workers`、`###v2_results`、`###v2_gen`、`###v2_prob` 同一主节点，按需切换。结果和表单在新版获得中央主区域；旧版布局不变。
- 右检查区：`###v2_gpu` 和 `###v2_detail` 为同组页签，建议起始宽度 300 逻辑单位，可调可收起。选中 worker 只更新详情上下文，不自动抢焦点。
- 中央底部：`###v2_logN` 保持独立输出窗与页签，建议初始占中央区域高度 25%，Vulkan 多 viewport 验收后支持弹出。
- 客户区顶部用一条紧凑命令栏展示当前配置和全局启动/安全停止；保留菜单入口，避免迁移时丢失低频动作。Prime95 的异常/待处理提示在顶部测量预留空间。

上述比例与宽度都是设计候选。第一版保留 Windows 系统标题栏；不同时引入无边框、自绘系统按钮和磨砂。

导航只聚焦/重开目标窗口，不能每次点击重建 DockBuilder 布局。浮动窗存在时导航也应聚焦同一实例。页签和导航表达同一选中状态，不制造两套页面内容。

### 4.2 v2 停靠、独立状态与窄窗口

- v2 提供用户手工停靠、分离和跨屏窗口；v2 窗口 ID 不因中文标题改变。旧窗口及其 ID 不修改。
- v2 初次运行直接建立现代默认布局，重置只影响 v2。旧布局留给原程序继续加载，v2 不自动迁移或写回旧布局。
- v2 的启动页、明暗主题、字号/字体角色、窗口位置和停靠状态写入独立侧车文件，初始候选为 v2 输出目录中的 `ecm_gui_v2.ui.toml`；可显式指定其他 UI 状态路径。路径不可写需明确诊断，不悄悄写旧工作流 TOML。
- `--config` 表示工作流 TOML，v2 UI 状态入口与它区分；默认值/范围/英中说明仍在 `config/ecm_options.json` 建立独立 v2 定义并生成，不挪用或改变旧 `gui` 键。具体 schema/键名和生成接线在 P3 确定。
- 窄屏按可用逻辑宽度折叠导航并隐藏默认右区，可通过按钮重开；更窄时表单从两列变单列。保留用户手动隐藏/展开状态，避免每帧按断点反复覆盖。
- 表格保留横向滚动和关键列的最小宽度。缩小窗口不能让进度列、操作按钮或错误提示消失；不强行把 10/11 列挤成不可读文本。
- 底部日志关闭时仍持续消费 worker 输出；切页不能停止 Controller 或退出协调。

### 4.3 在 v2 新建页面，保持业务合同

- worker：保留高密度表格、任务摘要和 Stage2 进度；把程序/TOML 路径与 worker 数量移到配置区域或设置弹窗，常用动作留在命令栏。所有运行中禁用规则继续由原状态决定。
- Result：中央主区容纳筛选、搜索、11 列表格和原始行；来源诊断、复制与选中规则不变。窄屏工具栏换行，不改变筛选和原始 JSON/checksum 字节合同。
- 生成器：以“输入→曲线与输出选项→预览→追加”组织，宽屏可并排输入/预览，窄屏单列；继续保留更改选项使预览失效、目标 worker、mtime 复核、B1/B2 与曲线模型规则。
- 概率：保留后台输入快照、显式计算与反算；统一 FormRow、曲线图背景/图例/坐标样式，不改数值模型和口径。两列表单按可用宽度降为一列。
- GPU：摘要数值＋现有历史曲线，统一单位、标题、未知值和图例；不将 NVML 利用率改称 occupancy，不改变采样频率。
- 日志/详情：UI 与等宽文字分工，错误和阶段色走语义映射；保留 4000 行、裁剪、跟随滚动开关、真实 worker 版本和原始输出。
- 设置：v2 集中路径、worker 数、语言与新版显示配置；将业务字段与 v2 UI 字段分开保存。新增定义经 `config/ecm_options.json` 与生成器接入，旧 GUI 的默认值和 UI 字段不变。生成参考文档不手改。

## 5. 独立工程、依赖与复用边界

所有新版窗口、样式和渲染代码位于新增的 `src/ecm/application/gui_v2/`：

- `main_win32.cpp`：v2 启动参数、原生窗口、消息循环、帧间字体/DPI处理和异常退出协调；不替换旧宿主。
- `render/vulkan_context.*`：instance、物理/逻辑设备、队列、能力诊断。
- `render/vulkan_window.*`：主窗口 surface、swapchain、render pass/framebuffer、帧同步、重建与回收。
- `render/imgui_vulkan_host.*`：ImGui Vulkan 初始化、帧提交、动态纹理，以及 Win32 多 viewport surface 桥接。
- `ui/theme.*`、`ui/fonts.*`、`ui/widgets.*`：Ely 风格语义、尺寸、字体角色和有限控件集合。
- `ui/shell.*`、`views/`：导航、命令栏、通知、默认停靠和各功能页面；新增 v2 App，不调用旧 App 的绘制或保存设置入口。
- `ui_state.*`：独立 v2 UI 状态读写与 schema 版本；不持有 WorkerProc。
- v2 专用第三方目录记录固定 Vulkan 后端文件、来源提交和许可。需要后端修复时先评估 v2 独立版本，禁止给旧目录打补丁。

这些是规划文件名，尚未创建。优先使用 ImGui 公共 API；DockBuilder 内部 API 限于 v2 布局模块。不引入通用渲染引擎或 GPUI 兼容层。

### 5.1 只读复用业务模块

候选为旧目录中的 `worker_controller/worker_proc/log_parse`、`toml_file`、`results`、`worktodo_gen`、`probability_model`、`gpu_monitor`、`localization`、`task_display` 和系统文件/DPI辅助函数。按实际依赖为 v2 目标明确列源，直接编译未修改的原文件；必要的 v2 包装在新目录实现。

旧 `app.cpp`、`probability_panel.cpp` 与 `main_win32.cpp` 是行为/交互参照，不整体链接到 v2，不为新版改写旧 UI。复用代码中的进程路径、默认值和相邻程序探测可能依赖 exe 目录，P3 必须验证 v2 独立输出位置，不假设更换程序名后天然正确。

新主题和 Vulkan 层不反向进入 Controller/Proc 或数学模型。若必须修改共用源才能接通新功能，应先寻找 v2 包装；超出“原 GUI 不变”边界的公共修改需单独列出，不能并入本计划默认范围。

### 5.2 独立构建与可选依赖

- 新增目标 `ecm_gui_v2`、专用 ImGui/Vulkan backend 目标及 v2 业务适配目标；不能把旧 `ecm_gui_imgui` 拉入 v2，否则会同时引入 DX11 后端。
- 新增 `ECM_BUILD_GUI_V2`，默认 OFF；v2 构建入口显式开启。根构建接线不放在旧 `ECM_BUILD_GUI` 成功分支里，旧 GUI 关闭时仍应能单独构建 v2。
- 新增候选入口 `tools/build/build_gui_v2.bat` 与 `tools/build/dev/build_gui_v2.ps1`，默认输出 `build_gui_v2_local/`。它只构建 v2；原 `build_gui.bat` 仍只构建旧版。
- Vulkan SDK/header/import library 检查仅在显式启用 v2 时进行；缺依赖时 v2 给出明确诊断，不能影响旧 GUI、CLI 或没有请求 v2 的 CMake 配置。不强制 GLSL 工具链，是否需要由实际固定后端资源决定。
- v2 接入现有配置生成一致性检查，根目标依赖清单增加 v2，而不改旧目标输入。
- Windows 为首个目标平台。Vulkan 渲染不等于已经跨平台，不在第一版引入 GLFW/SDL 或承诺 Linux/macOS。

### 5.3 并存与回退

v1/v2 使用不同 exe、构建目录、UI 状态、布局版本和资源位置；v2 不覆盖旧版二进制。随时可关闭 v2 并启动原程序，原布局仍可用；这不是在 v2 内动态回退 DX11。

任务、日志和结果仍遵循现有工作流格式。v2 默认不启动真实任务；验收使用隔离配置与假 worker。两版同时打开时，v2 须在启动同一 worker/队列组前验证拥有权，无法确认无其他控制者时拒绝重复启动并给出原因；不能假定只有 v2 写一个锁就能约束未修改的旧版。具体旧版进程识别、配置身份及重启恢复列入 P3。

## 6. 分阶段实施与退出条件

### P0 冻结原版与并存合同

冻结原 GUI exe/源码哈希、原构建与 UI 配置，以及固定任务/结果/日志和操作截图。定义 v2 名称、独立输出/UI路径、只读业务复用清单、功能对照和资源基线。

退出条件：原目录和后端保护清单明确；各项新功能只落新目录；证据写 `data/experiments/gui_v2/<run>/`；不运行用户真实队列。基线记录实际构建身份，不仅记录版本字符串。

### P1 独立 Vulkan 最小宿主

添加 v2 目标和独立入口，补充同提交 Vulkan 后端；完成 Win32 单窗口、设备/呈现队列选择、swapchain、命令/同步、ImGui基本绘制与字体纹理更新。提前注册 Win32 surface 桥接以满足初始化合同，P3再完整验收多窗口。先传统 render pass；记录 SDK、loader、API/扩展、GPU和驱动。

退出条件：启动/关闭、连续缩放、最小化/恢复、无可用 Vulkan、重建及失败清理可验证；开发 validation 输出无未解释错误。v2 开关关闭和无 Vulkan SDK 的条件下，原版配置/构建仍通过。此阶段不宣称完整多窗口或功能就绪。

### P2 v2 主题、字体与最小组件

只在新目录构建明暗主题、正文/辅助/标题/等宽角色、有限图标与按钮/表单/状态样板。采用 Ely 的语义与尺寸，保留中文救援和自动15px的初始可读性。

退出条件：normal/hover/pressed/disabled/focus/selected 可区分；中英切换、100/125/150/200%和动态纹理生命周期通过；样式/字体不累乘，不持续增长显存。原 GUI 不引入这些样式或资源。

### P3 v2 主布局、多窗口与业务接线

新 App 接入未修改的 Controller/Proc 与业务模型，建立导航、中央主区、右检查区和底部日志；添加独立 UI 状态。完成 Win32/Vulkan surface 回调、平台窗口初始化次序、次窗口呈现/重建/销毁、跨屏 DPI和输入。验证 v2 输出目录下的 worker路径解析及两版控制拥有权。

退出条件：浮动窗完整生命周期、窄屏、独立状态重启恢复、通知占位和安全退出通过；v2运行与保存UI后原版UI配置和布局哈希不变；无重复控制同组worker；多viewport闪烁与呈现开销有实际结论。

### P4 在 v2 逐页实现功能等价

顺序：worker/详情/日志 → Result → 生成器 → 概率 → GPU/设置。页面绘制在 v2 重建，业务模型只读复用；每页对照原版合同单独验收。

退出条件：原始结果字节复制、过滤与选中、预览失效/mtime、任务分区、B1/B2与曲线含义、概率快照和监控采样不变。原版页面与既有测试期望不为新版布局修改。

### P5 双版本功能与低占用验收

相同输入、设备、窗口、DPI、刷新与采样条件下配对比较旧DX11与v2 Vulkan，覆盖单窗口/多窗口、空闲/日志/长Result/多worker、最小化恢复和计算并行。

退出条件：v2功能门禁及原版回归通过；原目录/目标/配置隔离证据完整；性能门限在采样前冻结，显著退化须定位，不事后放宽。不得因Vulkan名称声明更低CPU、显存或功耗。

### P6 独立分发准备

沿用 v2 新入口构建，形成独立目录/包，不替换旧版。核对运行时 Vulkan loader/驱动要求、无开发SDK的启动、资源/许可、exe哈希和业务配置定位。新增 v2 正式说明；原 GUI 正式说明仍描述原行为。

不透明主题和系统标题栏作为首版基础。磨砂、自绘标题栏与持续动画单列后续；Vulkan surface 的 alpha/composite、系统合成兼容性需要实测，不能照搬 DXGI 方案。

退出条件：干净分发目录中运行、缺Vulkan清晰失败、缺字体/缺NVML、状态重启恢复与安全退出通过；原版仍可独立启动。公共版本按功能更新规则在实际开发时评估，不把前端 v2 写成公共版本 2.x，也不预占并行开发中的下一版本。本轮只修改规划。

依赖：P0 → P1（Vulkan宿主）→ P2（主题字体）→ P3（完整布局/多窗口/业务）→ P4（页面）→ P5（双版本验收）→ P6（独立分发）。P1/P2为首批实现范围，多窗口和功能等价仍是完整v2交付门槛。

## 7. 验收清单与证据

### 原版不变

- 比较 `src/ecm/application/gui/`、旧 ImGui 核心/Win32/DX11 后端、原构建入口、原UI键默认值及旧测试期望的内容哈希/差异。
- 根构建仅新增可关闭的v2接线；关闭v2或缺SDK时，旧GUI/CLI仍能配置与构建。旧GUI开关关闭、v2开启也能独立构建。
- 运行v2、切换主题/语言/窗口布局并退出后，旧GUI布局/字体/窗口状态不变。旧exe能继续使用原UI状态。
- 原GUI与v2不同时启动同组worker/队列；原配置身份、进程识别和启动竞争需要实际测试，不只检查v2自己的锁。

### Vulkan 专项

主/次surface创建、graphics/present队列支持、swapchain图像数量与格式、render pass/pipeline一致性、帧同步、纹理descriptor/上传/销毁次序；resize、连续创建/关闭viewport、跨屏、最小化零尺寸、OUT_OF_DATE/SUBOPTIMAL及受控失败清理。

开发validation无未解释错误；记录新viewport闪烁及主次窗口呈现模式/空闲帧率。图形设备选择不修改计算设备配置。device/surface失败时仍能执行worker退出协调。没有开发SDK/validation layers的分发环境与无可用loader/驱动两类场景分别测试。

### 业务功能

- Controller/Proc：启动、停止、崩溃/重启、混合阶段、退出确认、checkpoint安全边界及超时终止。
- 显示：任务摘要、GUI与worker版本区分；Stage2任务进度与阶段进度区分；正式结果依据发布文件。
- 生成器：固定输入原始输出对照，B1/B2、e/m0/m2/m3、CPU禁用组合、目标分区、预览失效和外部修改复核。
- Result：F/NF、来源去重、组合筛选/选择清理、JSON/checksum字节复制、截短/替换/半行写入和读取错误。
- 概率：相同输入快照对应相同结果/反算口径，零值/未定义不伪造；不扩大数值模型范围。
- 日志与监控：4000行、可见行绘制、跟随开关；NVML未知值和采样/刷新独立。关闭/切换页面仍推进Controller。

### 可见性、交互与性能

中英、明暗、100/125/150/200%缩放、长路径/标签、空/错误/运行/待停状态，宽/窄布局和跨屏浮动窗。初始客户端逻辑尺寸场景1280×800、1024×700、800×600，按实际最小窗口约束调整并记录对应物理尺寸。键盘Tab/Shift-Tab/Enter/Escape、中文组合输入、焦点和鼠标命中逐项验证，不据此宣称完整无障碍支持。

各性能条件至少5次独立运行，预热后固定时长采样，报告中位数/分位值、设备与驱动/系统/功耗策略、缩放与显示刷新、GUI/worker哈希、输入、样本数、工具和计时边界。分别报告CPU时间、working set/private bytes、GUI显存、绘制时间、输入到呈现延迟；worker吞吐独立计时。不得相加组件峰值或嵌套计时。

证据存 `data/experiments/gui_v2/`；汇总存 `data/benchmarks/`，图存 `data/figures/`。旧DX11与新Vulkan总体验收分开记录后端开销与页面/布局差异，不能把跨版本总差异全部归因于API。

### 测试组织

复用旧GUI无窗口helper、日志/结果/生成器/Controller合同与固定输入。Vulkan/新版UI/独立状态测试放新增 `tools/test/gui_v2/` 或明确的v2测试入口，显式指向 `ecm_gui_v2.exe`，不改旧几何/像素期望适配新版。需要时将可复用测试增加独立v2调用包装，旧调用方式与期望不变。

旧 `test_gui_all.ps1` 等部分脚本默认查找旧 `build_gui/`，保留这些脚本，不把它们直接当v2全套验收。旧版回归和v2验收分别指定二进制、配置、构建目录与证据位置。v2测试helper显式构建，不进入普通交付目标。本轮没有执行这些回归。

## 8. 待验证事项与首批范围

首批实际实现为P0–P2：原版保护基线、独立Vulkan宿主、Ely主题与字体样板。先证明渲染/重建/纹理生命周期，再实现复杂页面。

TODO：Vulkan SDK实际位置及依赖版本；固定后端最低API/扩展；Win32 surface桥接与多viewport闪烁；主次窗口呈现策略；渲染设备选择；旧程序并行控制身份；v2UI侧车schema和权威生成定义；字体/CJK资产；宽窄布局实测；大Result阻塞基线；Vulkan/计算并行的资源与功耗；分发机无SDK验证。

本批实施范围为 P0–P2，实际状态见实施进度。后续 P3–P6 仍是规划；原 GUI 的源码、目标、状态和公共版本保持不变，不将样板程序当作完整前端发布。
