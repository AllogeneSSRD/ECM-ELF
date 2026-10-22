# 独立 GUI v2：ImGui 现代化规划草稿

状态：P0–P4 已完成，P5 待验收（默认DX11的完整性能矩阵未重测），P6 独立候选准备完成；0.49.5 调整导航排序与右侧原生标签、默认隐藏系统标题栏并修复ELY拖动跟随，保留0.49.4的GPU页与悬停设备信息，保留0.49.3的任务表内部纵向滚动与日志/正文字体修复，保留0.49.2的字号、全部worker耗时图、Stage2批次及退出反馈，保留0.49.1的任务表与主题日志，保留默认DX11，可选Vulkan冻结根因待查。日期：2026-10-22。原 GUI 保持不变；独立构建 GUI v2，采用 C++ / Dear ImGui / Win32 平台后端，参考 Ely 组件库统一布局、配色、字体和控件。v2 是前端名称，不表示公共程序版本变为 2.x。初始规划及P0–P6以Vulkan实施；0.47.8经用户授权将DX11设为默认，可选Vulkan不作为稳定路径。

本文件是本主题的权威临时草稿。[HTML 阅读版](../web/gui-v2-vulkan-plan-20261021.html)呈现布局示意、阶段验收和本草稿全文，可离线打开。实施成功后另写 GUI v2 正式说明，旧 GUI 使用说明继续描述旧版；提示用户处理本草稿，不自动删除。

## 实施进度

- P0：已完成，v2 里程碑 0.47.0。旧 GUI 74 个源码、后端、配置和测试文件已冻结；原 exe 单独复制到隔离目录，SHA256 保持 `32f3a9a2905ff39e6f098f0c943dedbaa3bdbea291ef7122417634958b85919b`。旧版公共标识保持 0.46.7，v2 使用独立 `version.h` 记录 0.47.z。
- P0 验证：自检 33、日志 85、生成器 110、结果 26、Controller 44 项/10 场景全部通过。假 worker 输出、工作流与结果输入、英文/中文窗口截图和五次独立空闲采样保存在 [P0 证据](../../data/experiments/gui_v2/p0_20261021_03/baseline.json)。未运行真实计算 worker。
- P0 测量边界：隐藏普通窗口，预热 1 秒后采样 2 秒，界面 10Hz / NVML 500ms；单核 CPU 百分比中位数 1.5625%，working set 中位数 116867072 bytes，private bytes 中位数 117784576 bytes。现场另有 GPU 负载，未隔离功耗策略；这些是冻结输入条件下的原始基线，不构成性能优劣结论，也没有输入延迟/显存/功耗测量。
- P0 原始失败保留：第一轮测试配置误将 worktodo 放入 stage1，改到 shared；第二轮隐藏窗口截图被系统拒绝，改为截图时短暂非激活显示。未改旧实现、测试期望或验证门限。有效证据为 `_03`。
- P1：已完成，v2 里程碑 0.47.1。独立目标、入口与 Win32/Vulkan 宿主位于 `src/ecm/application/gui_v2/`；根开关 `ECM_BUILD_GUI_V2` 默认 OFF，独立构建不依赖旧 GUI、OpenCL 或计算 worker。Vulkan-Headers 1.3.290、Volk SDK 1.3.290.0 与同提交 ImGui Vulkan 后端放专用第三方目录，来源和哈希随源码维护。运行要求 Vulkan 1.1 loader/设备及 swapchain，动态加载，不需要开发 SDK/import library。
- P1 验证：正常运行、缺 loader、device/surface lost 注入、非法参数、真实窗口四次缩放及最小化恢复全部通过；[最终有效证据](../../data/experiments/gui_v2/p1_20261021_04/result.json)，exe SHA256 `33f3ca409e623f5009c1620b7c93de9c60dc00382372b4db30843899ebd463aa`。RTX 4070 Ti，设备 API 1.4.341；使用官方 SDK 1.4.363.0 中解出的 validation layer，未执行 SDK 安装器、未改注册表。validation API 错误/警告均为 0；3 条 general loader 环境警告（重复的 NVIDIA/OBS 隐式层注册）单独计数，完整日志保留。正常启动不启用该开发层。
- P1 隔离：原构建入口在独立 `build_gui_v2_legacy_proof` 构建 DX11 GUI，自检 33 项通过；缺专用 headers 的 v2 配置明确失败。未修改旧 GUI、旧 ImGui 或公共 0.46.7 版本。宿主仅单窗口，无计算任务、UI 状态保存和多 viewport 资格结论。
- P1 诊断：第一次生命周期测试暴露 resize 后访问冲突。最小复现定位到 Vulkan backend shutdown 同时销毁主 viewport 的 Win32 数据；重建时按顺序同时重新初始化两个后端，保留 ImGui context。第二次复测暴露测试读取日志的文件共享问题，改为允许并发只读日志。失败目录 `_01`、`_02` 与最小诊断日志均保留，未改旧代码或降低门限。
- P2：已完成，v2 里程碑 0.47.2。`ui/theme.*` 集中 Ely 明暗语义色及逻辑尺寸；`ui/fonts.*` 建立正文 15、辅助 13、标题 22、等宽 15 逻辑像素角色，使用系统 Segoe UI / 微软雅黑 / Consolas 与 CJK 合并救援。字体缺失回退英文并显示说明。`ui/widgets.*` 提供按钮、有限线条图标、导航选择、SectionHeader、FormRow/路径输入和状态样板；语言切换保持控件 ID，窄窗口换行及滚动，不接业务控制。MIT 设计告知见 `ui/NOTICE.md`，不分发系统字体或 Lucide 资产。
- P2 验证：[最终组合证据](../../data/experiments/gui_v2/p2_20261021_06/result.json)，exe SHA256 `4454d74fad8fe4dd13511729ef81964e4b4e6508979dcb5ae7d84dc378386b8f`。中英 × 明暗 × 100/125/150/200% 共 16 组合，完整预热后 3 轮、48 个采样；字号/间距不累乘，正文和等宽字体均有中文字形，等宽 i/W 宽度一致。同一状态跨轮的显式 Vulkan 分配字节/数量/纹理数完全一致；范围 465408–1222656 bytes，每状态 5 个分配、1 个纹理，正常退出全部归零。该审计不包含驱动/swapchain 全部驻留，不能视为进程总显存或性能优势。
- P2 输入与回退：ImGui 鼠标注入验证按钮，Win32 文本与 Escape/Tab 消息验证输入及可见键盘焦点；缺中文字体的受控模拟正确回退英文。普通模式不启用测试消息。真实桌面鼠标/焦点、IME 与物理跨屏 DPI 仍留 P3；四档比例验证不能替代跨屏资格。真实 HWND 的 P1 六项宿主回归在最终 P2 上再次通过，见 [宿主证据](../../data/experiments/gui_v2/p2_host_20261021_03/result.json)。validation API 错误/警告均为 0，原有 3 条 general 环境警告仍单独记录。
- P2 诊断证据：首轮将首次启动容量与完整切换历史后的缓冲容量比较，发现 1 KiB 差异。逐块审计确认字体纹理同为 262144 bytes，变化来自顶点/索引缓冲（768+256 bytes）；固定完整预热序列后再比较三轮，未放宽相等门限。截图工具改为与宿主一致的物理 DPI 坐标。第三/四轮后台 Win32 鼠标消息受真实桌面鼠标所在窗口影响，改为显式 `--audit-ui` 的 ImGui 输入注入；实际桌面输入不据此宣称通过。原始 `_01`–`_04` 日志和截图保留，`_05` 已通过控件验收；补齐 loader 诊断及原子验证计数后，最终二进制证据为 `_06`。
- P2 构建隔离：根工程 `ECM_BUILD_GUI=OFF`、`ECM_BUILD_GUI_V2=ON` 的独立目标构建通过，日志在 `data/experiments/gui_v2/root-*.log`；专用入口不依赖计算库。默认入口输出 `build_gui_v2_local/`，exe SHA256 `5b1c1eaf9a3b8a84a2ddb20945c5745b04df12207e03e80b8ac0e2453ce63c08`；移除 SDK/validation 环境变量后普通启动、8 帧呈现与退出通过，见 [默认入口证据](../../data/experiments/gui_v2/p2_default_smoke_20261021/result.json)。这是本机系统 runtime 验证，不代替干净分发机验收。补齐 loader 1.4.341 / 请求 API 1.1.0 / instance 扩展诊断，验证回调计数使用原子变量。新增依赖文件指定 LF，保证来源 SHA256 在 Windows checkout 后保持一致。当前使用与限制另见 [GUI v2 开发预览](../usage/GUI_V2.md)。
- P2 文档：HTML 同步显示 3 个已完成阶段、4 个待实施阶段；离线浏览器验证阶段切换、明暗阅读、窄布局和 390px 视口均通过，无脚本错误或横向溢出。验证入口 `tools/test/gui_v2/test_plan_html.cjs`；草稿继续保留，等待用户决定后续整理。
- P3：已完成，v2 里程碑 0.47.3。独立 App 复用原 Controller/Proc/业务模型；导航、工作区、检查区、日志与确认退出接通；侧车采用配置身份隔离的 LocalAppData 路径，显式路径可覆盖。21 个字段由 schema 生成，非法文件保留原字节。原业务 TOML 不因外观保存改写。
- P3 验证：[主集成证据](../../data/experiments/gui_v2/p3_shell_04/result.json)，exe SHA256 `847f981e02ce4e1a65054fb8f5aa895a49addd61988fddb9a8453e09a156183c`。浮动窗跨两块 2560×1600 / 144 DPI 显示器、主 swapchain 三次缩放、浮动窗销毁重建、窄屏、侧车重启、安全停止与退出取消通过。宠物七状态、透明 owned/noactivate 窗口、最小化恢复与鼠标非激活合同通过。不同物理 DPI 与真实 IME 组合输入尚未获得资格，不据合成事件声称通过。
- P3 队列互斥：共用 Stage1/Stage2 队列入口使用进程级文件区间锁，全队列与分区互斥，不同分区可并行，崩溃释放。v2 拒绝不支持能力标记的旧 worker；原 GUI 使用更新后的 worker 方受保护。见 [锁竞争证据](../../data/experiments/gui_v2/p3_ownership_01/result.json)。原 GUI 73 个保护文件及 exe 字节不变；仅获批准的 v2 独立 schema 加法，现有配置投影相同，见 [隔离审计](../../data/experiments/gui_v2/p3_legacy_01/result.json)。
- P3 Vulkan 诊断：次 viewport 在 acquire 前批量转换 swapchain 图像产生 2 条验证错误；最小复现 `p3_viewport_red_01` 留存。v2 专用后端补丁取消提前转换，由 acquire 后 render pass 转换；绿色复现与完整集成均 API 错误/警告 0。原核心及 Win32/DX11 后端不变。主窗口重建保留 backend/字体和次窗口；显示格式改变明确要求重启。
- P3 字体与宠物：缓存中英/等宽字体并共享字体文件内存，16 组合三轮严格资源稳定、退出显式分配归零，见 [字体证据](../../data/experiments/gui_v2/p3_ui_01/result.json)。ELYTerminal MIT 矢量角色离线生成 7×8 atlas；运行时原生分层窗口合成，不引入 Web runtime，默认关闭。严格侧车往返/非法文件保留及宠物几何测试通过（`p3_state_03`）。早期失败样本保留，不以假 worker 自身超时作为安全退出证据。
- P4：已完成，v2 里程碑 0.47.4。七页独立绘制，原 Controller/结果/生成器/概率/NVML 模型只读复用。结果保留原 JSON/checksum 字节和选择清理；生成器目标分区、输入失效、mtime/hash 复核和追加占用接通；概率快照、正反算与三类图/CSV 接通；GPU 500ms 后台采样；设置支持外观、宠物、worker 路径/数量及显式业务编辑。权威 schema 生成双语说明/范围，旧 GUI 偏好禁止通过 v2 编辑。
- P4 验证：`p4_contract_03.log` 验证 worker 2 分区、选项失效、外部队列修改拒绝、CPU m3 拒绝、配置指纹与注释保留、旧 GUI 字段保护和概率快照。旧 schema 的 autostart 是整数，首轮测试误写 bool 的失败样本保留。最终 exe SHA256 `728ce59908e043c2702ee5032ee3154d7605554d7e6164b8464535a5486c20c7`；[七页双语宽窄渲染](../../data/experiments/gui_v2/p4_pages_02/result.json)及 [布局/宠物/安全退出](../../data/experiments/gui_v2/p4_shell_01/result.json) API 验证错误/警告 0。原保护文件和配置既有投影再次通过 `p4_legacy_01`。
- P4 交互补充：结果状态/阶段及概率范围/曲线预设改为点击标签。主窗口和原生浮动窗滚轮短暂 16ms 刷新并指数平滑，距离守恒，结束恢复 100ms；无额外动画线程。Toast 不抢焦点，启动、安全停止及其他操作状态可见。滚轮积分/有限收敛测试通过；真实桌面触控板体验仍需人工体验验收。
- P4 概率修正：旧页面 16 明确是 torsion-only 基线，不能当完整 Edwards D_eff；旧页保持原样。v2 增加 Edwards 经验/相对归一/自定义预设；现有 B1=256、15–25 bit 数据的 Z2xZ8 经验值 26.408122，Edwards/Suyama 相对收益归一到 GMP exp(3.134) 基线为 29.015242。[依据与口径证据](../../data/experiments/gui_v2/p4_probability_01/result.json)。后者仅联合模型启发值，未经独立 Stage2 标定；不把挠群阶 16 或四舍五入 30 写成普适精确常量。论文依据：[ECM using Edwards curves](https://eprint.iacr.org/2008/016.pdf)，经验方法沿用仓库 `tools/ecm_prob/README.md`。
- P5 采样前门限冻结：`tools/test/gui_v2/performance_gates.json` 指定每条件五次、预热 2s/采样 3s、10Hz/500ms、1280×850；单核 CPU 增量≤5pp，private 增量≤192MiB且≤384MiB，宠物额外 CPU≤2pp/内存≤32MiB，最小化 CPU≤1%，帧 CPU/提交 P95≤50ms，消息到提交 P95≤250ms，重复开关 GDI 增量≤2/内存≤8MiB，计算吞吐比≥0.9。尚未完成全部测量；任何未实测项不能记为通过。
- P5：待验收，v2 里程碑 0.47.5。本机功能、互斥、计算及资源回归已执行，仍有一项未稳定通过的最小化 CPU 门禁，不将其改判为通过，不放宽冻结门限。
- P5 功能：[未修改旧版合同与新版合同](../../data/experiments/gui_v2/p5_regression_02/result.json)、[实际 Stage1/Stage2 入口及两版启动顺序](../../data/experiments/gui_v2/p5_consumers_02/result.json)、[标签/原生滚轮/实际启动停止与 Toast](../../data/experiments/gui_v2/p5_interaction_02/result.json)、[多窗口/宠物/状态恢复](../../data/experiments/gui_v2/p5_shell_01/result.json)、[宿主故障回归](../../data/experiments/gui_v2/p5_host_01/result.json)均通过。最终候选 exe SHA256 `ea147acc0aaafccc81130ec70342eb0dbd0fae77308a4be5864808f11131c6b2`；API 验证错误/警告 0。
- P5 真实计算：固定 M991 / B1=10000 / B2=0 / 8 曲线 / Montgomery PARAM0 / sigma 起点10，同一个更新 worker 在旧版和新版各跑五次；十次因子、曲线和 sigma 完全相同。worker 报告 GPU 区间中位数旧 982.059ms、新 1001.390ms，相对吞吐 0.9807≥0.9，见 [证据](../../data/experiments/gui_v2/p5_real_01/result.json)。计时排除前端启动，不等于完整 ECM 吞吐；现场外部 GPU 负载、功耗未控制。初始 Stage1 包装器在成功链接后收集无关目标对象失败，原失败日志保留；实际两个入口独立构建及二进制行为已验证，未把包装器失败记为成功。
- P5 性能诊断：首轮七条件×五对、70 次采样暴露动画 PNG 重复裁切/缩放、结果每帧重建万行集合和长 fence 等待。按 mood/DPI 缓存缩放帧、按输入/文件变化缓存结果筛选；主 fence 非阻塞轮询，继续处理 Controller。原始失败 `p5_performance_01` 留存。[第二轮完整证据](../../data/experiments/gui_v2/p5_performance_02/result.json)仍保留失败标记；同样 70 次采样，长 Result 单核 CPU 中位数从9.375%降到2.083%，动画从6.735%降到2.604%；最坏条件帧 CPU/提交 P95 从456.568ms降到13.332ms，消息到提交 P95=16ms。与前轮比较均为同机短样本、不同二进制/时间，不能扩大成一般性能保证。帧计时与进程 CPU 边界分开；第二轮帧样本从明确的2s预热终点采集。
- P5 资源：第二轮旧版空闲 private 中位数 96,227,328 bytes，新版 219,045,888；新版万行结果245,907,456，均通过增量≤192MiB/绝对≤384MiB门限。新版字体/CJK和布局成本高于旧版，不宣称 Vulkan 更省内存。显式 Vulkan 分配只计本进程 VkDeviceMemory，不是总显存。宠物五次预热后二十次开关 GDI=13→13、private 增加225280 bytes，退出显式 Vulkan 分配归零，见 [资源证据](../../data/experiments/gui_v2/p5_resources_01/result.json)。
- P5 剩余门禁：第二轮最小化单核 CPU 中位数1.031%略高于1%。[五次启用/五次关闭NVML的线程拆分](../../data/experiments/gui_v2/p5_minimized_diagnosis_01/result.json)中，主线程中位数均0；辅助线程出现0–1.56%的短窗口波动，关闭NVML也有波动，不能单独归因NVML。3s计时每个15.625ms CPU量化约0.52pp；聚焦复测中位数较低不覆盖原失败。需要更长区间及驱动线程剖析确认，保留500ms采样合同。总显存、物理输入到显示延迟、受控功耗与独立干净机器仍未获得资格。
- P5 侧车回归修复：新增字段导致重复 [worker] 的失败被严格往返测试捕获；生成器按表分组，重启明确拒绝忽略状态的假通过。schema 源统一LF，生成哈希匹配 Git checkout。未修改旧模型或旧测试期待。
- P6：已完成独立候选准备，里程碑0.47.6；不表示P5门禁或全部硬件资格通过。默认入口 `build_gui_v2_local/ecm_gui_v2.exe`，SHA256 `55d908ed138ebe043b197e9c78120194a73a7e1d00a73c1d3e87a6694c83984e`。独立 GUI-only 包含22个文件（含清单），附GPL项目许可、MIT/Apache第三方告知、资产、构建身份和可重建源码归档；不附计算worker、系统字体、Vulkan loader或MSVC runtime。
- P6 分发：[最终候选ZIP](../../data/experiments/gui_v2/p6_package_06/ECM-ELF-GUI-v2-0.47.6.zip)，SHA256 `23eff0b1f24220e7946d8bab3899c00ba1a80b36d25d229dbbd48542b4e1a7da`。[重新定位/依赖/侧车证据](../../data/experiments/gui_v2/p6_package_test_06/result.json)：中文目录、中文工作流、PATH仅System32、移除SDK/开发层变量后启动；缺loader明确失败，缺中文字体回退、缺宠物资产不崩溃、非法schema/布局保留、重启和业务原字节均通过。PE导入无DX11/OpenCL/WebView2或静态Vulkan import；依赖Windows系统DLL及MSVC runtime。另通过实际不存在NVML DLL的现有覆盖入口验证降级（`p6_missing_nvml_01`），未修改系统DLL。以上是本机干净目录验证，不是独立干净OS资格。
- P6 可靠性：[故障期间活跃worker安全停止](../../data/experiments/gui_v2/p6_failure_shutdown_01/result.json)、[原版/新版合同](../../data/experiments/gui_v2/p6_regression_01/result.json)、[P6多窗口/宠物](../../data/experiments/gui_v2/p6_shell_01/result.json)、[P6标签/滚轮/Toast](../../data/experiments/gui_v2/p6_interaction_01/result.json)通过；API验证错误/警告0，原GUI73个保护文件、配置旧投影与原exe哈希再次不变（`p6_legacy_01`）。16组合/三轮字体严格资源检查在P6候选中通过（`p6_ui_01`）；随后补强非法布局保护及Edwards详情不显示Montgomery param，最终包另有严格状态/分发回归。
- P6 诊断：中文路径原先被窄字符接口误解；只为v2加入UTF-8进程manifest，中文目录中的完整业务读写合同通过，系统代码页和旧GUI不变。非法layout hex原先会被重置后覆盖，`p6_layout_red_01`保留复现；现在装载阶段拒绝且禁止保存。初次包的版本横幅解析、测试环境SystemRoot假设和遗漏忽略目录ImGui核心的源码归档失败全部保留，未放宽检查。
- P6 来源：[最终源码归档独立构建与运行](../../data/experiments/gui_v2/p6_source_build_04/result.json)通过，131个共有源目录文件哈希逐项匹配，归档无Git目录也可构建；不宣称PE逐位可重复。宠物50层/56帧离线重新生成的PNG及元数据与随包版本完全一致（`p6_pet_asset_01`），原始/规范LF源码哈希分别记录，PNG SHA256 `eb0eac5b7a5809d5ea0cf4d1e6dac8e521650671e243648508408cc98bfb1972`。临时草稿保留，待用户决定整理。
- 草稿与 HTML 由 `tools/gen/render_gui_v2_plan.py` 同步，原始证据保留在忽略目录。各阶段独立提交 `[0.47.z]`。


- 适用性优化：已完成，v2 0.47.7。补充外观 / 布局 / 工作流设置分组、双语搜索与 schema 类型/范围帮助、开关式布尔编辑、worker 文件选择。主题/语言/DPI 使用标签；语言菜单即时应用，窗口/表头/常用 UI 提示同步，原始计算日志不翻译。27 个侧车字段统一生成；原配置默认值及算法不变。
- 适用性布局：页面右键或拖到导航下方工具按钮移入右侧，以标签切换；右键、返回按钮或拖回导航返回。至少保留一页在中央；页位/右侧选中项/显示状态与停靠布局保存恢复。移入后自动激活工具窗；窄窗使用紧凑按钮、提示、文字换行与可滚动内容。设置可恢复默认布局，主菜单收拢为视图/语言/通知。
- 适用性图表：原生 ImDrawList 网格、带真实 x 值的多曲线/淡色面积、可隐藏图例、悬停采样读数；无新运行时依赖。概率参数/快照视图分离，计算后显示快照图，输入变动提示重算；bit/digits 使用模型返回的既定 x 单位。GPU 支持15/30/60/120s实际采样时间窗，缺失断线；worker速度图以采样序号为横轴。不会据新图表声明 P5 性能门禁已通过。


- 适用性验证：[最终交互/重启/字体救援](../../data/experiments/gui_v2/usability_ui_04/result.json)通过，exe SHA256 `8337c3e0712d48fce1821c7c0dd05966bce477b1954a791e13876aeee97c21b5`。实际点击搜索、空状态、双向语言切换与快照计算，拖入/拖回、显式保存与重启、明暗快照图、窄窗口和中文字体缺失时无循环重载均通过。`usability_ui_01`暴露工具页被同 dock 检查区遮挡，修复为移入后激活；失败证据保留，后续验证不放宽条件。
- 适用性回归：[全部七页/两语言/宽窄/Vulkan](../../data/experiments/gui_v2/usability_pages_01/result.json)、[原 GUI + v2 合同](../../data/experiments/gui_v2/usability_regression_01/result.json)、[筛选/滚轮/启动停止Toast](../../data/experiments/gui_v2/usability_interaction_01/result.json)及[原文件与原配置投影隔离](../../data/experiments/gui_v2/usability_legacy_01/result.json)通过；Vulkan API验证错误/警告均0。测试指针为显式注入，滚轮走原生窗口消息，不能代替真实IME或不同物理DPI资格；本轮未重做P5完整性能矩阵，原门禁仍待验收。

- 适用性分发：[0.47.7 候选包](../../data/experiments/gui_v2/usability_package_01/ECM-ELF-GUI-v2-0.47.7.zip)，独立 GUI、许可、资产、匹配源码与构建身份；仍不附计算 worker 或系统运行时。草稿继续保留。
- 适用性包验证：[中文目录与依赖/非法侧车检查](../../data/experiments/gui_v2/usability_package_test_01/result.json)通过，21个清单文件逐项校验，另有清单自身；ZIP SHA256 `3d70cae13db874632144a64f5c45306659d1c135769df4aa49f5098045e39be5`。源码归档的[134个共有构建目录文件](../../data/experiments/gui_v2/usability_source_01/result.json)逐项匹配最终构建身份，包含三份新UI头；保守构建目录另外记录本地ImGui README，归档固定核心白名单不包含该说明文件。本轮未重做归档独立编译，不将P6旧版本重建证据当作0.47.7重建证据。离线HTML阶段/390px阅读检查通过，P5仍待验收。


### 0.47.8 问题检查与修复

按用户问题顺序建立验收清单，保留失败样本；不能把驱动恢复、UI单线程阻塞和整机冻结混为一谈。

1. 多屏冻结：最高优先级，已由用户在本轮拖动测试期间确认复现，尚未修复。无计算任务，测试始终在4070 Ti外接屏进行，内屏同时无响应；鼠标指针仍可移动，但应用和任务栏时钟停止更新，Ctrl+Alt+Del没有可见响应，数秒后恢复。冻结期间的点击、拖动和滚动在恢复后体现效果；这不能区分当时已处理输入与恢复后处理积压输入。具体起止时刻及当时的测试子项未知。暂时默认关闭原生独立面板窗，保留主窗口内部停靠；新增28号侧车字段及命令行覆盖，属于未获资格的规避措施，不能宣称整桌面冻结已消除。已停止本机拖动/多平台窗重复复测，继续分析已有日志。
2. 导航：已修复。保留页面名称及完整提示，窄侧栏可横向滚动，默认宽度改为170逻辑单位并受可用空间约束，不再以数字替代名称。切页回到顶部，避免上一页滚动位置隐藏新页信息。
3. ECM概率：已修复。B1/B2、曲线数、范围等输入和计数摘要不显示无意义小数；概率输入、摘要、坐标与悬停使用0–100%。D_eff、比率、非整数解保留精度；模型与CSV仍使用0–1及原快照合同，算法不改。
4. GPU：已修复。核心/显存频率分项卡片及可展开趋势；显存占用率由NVML已用容量/总容量计算，单位MiB，区分于util.memory的读写活跃时间比例。每帧从同一份历史快照读取最新值与图表；时间窗固定为15/30/60/120秒、整数刻度，悬停共享一个实际采样时间并保留一位小数，不重复x值。指标口径见 [NVIDIA说明](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/kw/command)。
5. Ely：已修复客户区边缘定位与相邻帧插值串色。原生GDI+回归在修复前检出78个相邻绿色污染像素，隔离当前帧并镜像边缘采样后为0；缓存策略不变。客户区上下锚点、非激活窗口与睡眠截图通过。上游睡眠标记本来是两个Z，八帧均检查标记范围；PNG仍与原资产逐字节相同，并非重新绘制角色。
6. 设置：已修复。改为外观、导航/布局、工作流、计算设置依次排列的单页流式布局；移除外观/布局/工作流子页，保留显式暂存/保存、侧车独立及搜索筛选。

冻结排查假设优先级更新为：平台窗呈现/交换链同步、图表几何触发的渲染负载、字体纹理上传等待、NVML查询。若呈现路径相关，等待应集中在present/acquire/交换链重建且独立窗条件更严重；若图表相关，无图表对照应明显减少；若字体相关，应与字体上传/重新装载重合；若NVML相关，关闭采样应消失。现有证据只支持优先调查呈现路径，不能认定驱动或桌面合成器根因。`render/call_trace.h` 的可选 `--trace-render` 记录调用进入/返回及UTC与单调时钟映射，不改变Vulkan返回或增加采样线程；逐调用写盘只用于诊断，不用于P5门禁。

呈现证据：`issues_layout_trace_01` 的 `vkQueuePresentKHR` 曾阻塞547/531ms。最终二进制SHA256 `cfbaf09594dd17877bf9f142fd598c8bf76c7cb37a1c3d2fc8df6334afcb8a2f`，RTX4070 Ti、设备API1.4.341、驱动原始版本2559967232、请求API1.1，1400×1000窗口、144 DPI、GPU监控开启，无worker，预热1.5秒，每模式仅一次有界拖动。严格内部窗/独立窗测试均确认工作区实际脱离停靠；CPU边界为NewFrame前至主窗和平台窗绘制完成，不包含屏幕显示时间。独立窗原结果在进程退出前读取日志，漏掉最后一次present=1000ms、整帧=1000.9226ms，原“52.8042ms / responsive=true”不构成通过。修正脚本为进程退出后统计，并从原完整日志离线重算检出该失败，未再启动GUI；原始结果不改写。用户报告的两屏冻结优先于任何CPU/API通过记录。见 `issues_desktop_freeze_report_01/user-report.json`、`call-summary.json` 及 `issues_layout_strict_multi_01/layout.log`。

测试坐标诊断：早期内部单窗测试使用多平台窗坐标转换，硬编码拖动未可靠命中，因此 `issues_layout_single_01`、`issues_layout_single_gpu_01`、`issues_layout_final_01` 不能作布局改善证据。修复审计入口坐标换算后增加“概率已计算/GPU可用/实际脱离停靠”检查；全部失败和旧采样保留。即使严格内部窗一次采样低于3ms，也不足以证明全桌面冻结消失。布局脚本默认内部窗，独立窗须显式选用；当前不再在受影响桌面执行。

系统证据：较早 `issues_0478_diagnosis_01/nvlddmkm-events.xml` 有Graphics FECS Exception: Logging error 0x2，尚未与冻结发生时间关联。补充报告后只读获取最近两小时System/Application事件，未发现新增显示/硬件System事件；Application中的内核报告引用10月5日/9日旧转储，不归为本次故障。无新事件不等于无冻结；证据在 `issues_desktop_freeze_report_01/`。未更改TDR/驱动设置，未运行计算压力任务。

宠物证据与假设更正：早期HTML有重复clipPath ID，现以帧/层命名空间隔离；但inline/isolated SVG之间240个像素只有1–2级抗锯齿差异，不能据此认定其导致睡眠色块。真正原生回归见 `issues_pet_red_01/native-red.log` 与 `issues_regression_03/v2-state.log`（78→0）；唯一ID与每帧两个Z的范围检查见 `issues_pet_atlas_test_03/result.json`，运行时客户区定位/Doze/非激活见 `issues_pet_runtime_02/result.json`。PNG SHA256仍为 `eb0eac5b7a5809d5ea0cf4d1e6dac8e521650671e243648508408cc98bfb1972`；它由固定 `src/pet/rig.rs` 经Python导出7×8 SVG、Node/Playwright离线栅格化为1792×1736透明图集，上游无需有同名文件。

其他回归：原版自检/日志/生成器/结果/Controller及v2状态/业务合同 `issues_regression_03` 全部通过；实际筛选/滚轮/启动停止Toast见 `issues_interaction_02`，右侧页移动、双语、保存重启、设置流式搜索与明暗图表见 `issues_ui_03`，七页双语宽窄见 `issues_pages_03`。API验证错误/警告均0，仅说明调用检查通过；`issues_viewport_02` 的显式独立窗API通过不代表桌面刷新通过。本轮不将P5待验收改为已完成，旧GUI源码、二进制与既有配置投影继续独立审计。

### 0.47.8 DX11默认后端（用户追加授权）

用户补充旧GUI使用DX没有上述冻结，并允许工程量不大时切回DX。v2增加独立 `render/dx11_window.*`，使用与原GUI相同家族的D3D11经典swapchain、匹配且未修改的ImGui DX11后端；`render/render_host.h`只构造所选宿主。默认DX11，`--renderer vulkan`显式诊断；不改ImGui控件/业务模型、旧GUI源码或算法。硬件不支持时仅按旧版方式尝试系统WARP；DX11不初始化Volk/Vulkan设备，Vulkan专用验证/注入/调用跟踪参数在DX11模式明确拒绝。运行时显存审计仍只定义为显式VkDeviceMemory，DX11不把该零值当D3D显存测量。原生独立面板默认仍关闭，但DX11支持显式开启。

最终默认DX11二进制SHA256 `18f81dd93b90fa2ba4849490b45072ed2c8ee182098cb895b170aad5c37bcd03`。`issues_dx11_host_01`确认未传后端参数、实际适配器NVIDIA GeForce RTX4070 Ti、feature level11.0、硬件模式；加载D3D11/DXGI而未加载vulkan-1.dll。三次有限窗口缩放、最小化恢复与正常释放通过。内部/独立面板GPU图表各一次有界拖动：窗口1400×1000物理像素、144DPI、无worker、NVML开启、预热1.5s，统计拖动起至进程退出的CPU帧分别169/245个、最大5.4419/13.7865ms；两者均实际脱离停靠，正常退出。CPU计时不等于显示延迟；这些短样本不用于P5完整门禁。证据 `issues_dx11_layout_single_01` / `issues_dx11_layout_multi_01`。

用户对该轮DX11图表拖动/缩放/最小化反馈“没有观察到冻结”，因此记录为本机这一轮实际画面观察通过；与此前Vulkan冻结分别记录，不能据此认定Vulkan根因已修复，也不能承诺长期或其他驱动稳定。新增 `test_dx11_host.py`记录真实加载模块，公共页面/交互测试显式记录后端，DX11使用独立shutdown/退出检查，不冒充Vulkan API验证。历史Vulkan宿主、字体资源、故障和性能脚本显式选择Vulkan，当前不再自动执行其图形复测。

DX11最终二进制的七页双语宽窄、设置/语言/右侧页移动与快照图表、实际筛选/滚轮/启动停止Toast、上下边缘Doze/非激活、跨屏面板缩放/销毁重建/侧车恢复/安全退出均通过，分别在 `issues_dx11_pages_01`、`issues_dx11_usability_01`、`issues_dx11_interaction_01`、`issues_dx11_pet_runtime_01`、`issues_dx11_shell_01`。未运行真实ECM计算任务，未改变冻结的性能门限；本轮功能验证和用户观察不替代DX11完整性能矩阵、不同物理DPI、IME与干净OS资格。

最终独立候选：[GUI v2 0.47.8 ZIP](../../data/experiments/gui_v2/issues_dx11_package_01/ECM-ELF-GUI-v2-0.47.8.zip)，SHA256 `94f509156b8e2d52968d3383f89498355348fb8547eeec337556250291fe4602`，22个文件含清单，不附计算worker。`issues_dx11_package_test_01/result.json`通过中文重定位目录/PATH仅System32/无开发层环境、字体和宠物缺失、非法侧车/布局保留、重启及原业务字节；PE导入含D3D11而无静态Vulkan/OpenCL/WebView2。缺loader仅在显式Vulkan且立即失败的测试入口验证，没有启动Vulkan设备。包内默认后端、运行要求及Vulkan限制同步。

随包源码在新目录独立构建并默认DX11六帧启动退出成功，见 `issues_dx11_source_build_01/result.json`；139个共有源目录输入逐项匹配主构建身份。仅忽略的上游ImGui说明README未收入源码归档，其源码/头文件/许可均已包含；不宣称PE逐位可重复或独立干净OS资格。旧GUI73个保护文件、原exe哈希及既有配置投影再次通过 `issues_legacy_01`，schema生成物检查通过；当前状态与业务合同助手通过 `issues_dx11_state_01` / `issues_dx11_app_01`。离线HTML七阶段、六个已完成/P5待验收、主题与390px窄屏通过 `issues_html_01`，本次草稿继续保留，不自动清理。

### 0.48.0 ELY 三套装甲皮肤

依据用户例图增加Miss Pink Elf♪、Herrscher of Human: Ego、Hi♪ Love Elf♥三套可切换外观，英文装甲名称经Wiki与官方公告核对。每套重新绘制服饰/发型/头饰，沿用原边缘Q版骨架与七状态×八帧；Classic ELY保留。设置即时选择、侧车持久化、命令行覆盖、未知标识拒绝、缺资源仅关闭宠物、字体符号回退、CMake与分发接线完成。新增功能按仓库惯例提升v2功能版本为0.48.0，原GUI公共标识不变。

[皮肤制作与验收草稿](ELY_SKINS_DRAFT.md)为此子主题的权威临时说明，保留素材/名称出处、配方、失败诊断与证据；[离线皮肤画廊](../web/ely-skins-20261022.html)可切换七种状态、播放/暂停、大小及明暗底色。168帧的头饰和睡眠标记边界、透明图集、真实点击/保存重启/窗口复用及同路径预热后的资源回归通过；早期冷路径增长与失败不改写。完整P5性能矩阵及可选Vulkan资格不因该功能验收而改判。

皮肤PNG从exe旁`assets/`外部读取，不嵌入exe；[安装与替换方式](../usage/GUI_V2.md#安装与替换皮肤)说明复制、重启及新增ID仍须编译的边界。用户已授权先提交本轮代码与文档，新增皮肤图片和SVG继续仅本机保存；忽略规则仅对白名单中的原版图集、元数据、来源/完整MIT许可证和固定源码放行。含本地未跟踪资源的候选包迁移启动和源快照独立重建已通过，证据集中于皮肤草稿。

网页另增三套中空浅色菱形瞳孔变体，提供现有版本切换、七状态和双眼放大。变体仅位于`assets/web/ely_variants/`，保持忽略，不接入程序。原版来源与完整许可证随资源构建复制；本轮仅执行资源目标，exe及原生PNG字节不变。校验与预览证据见皮肤草稿。

### 0.48.1 启动检查、配置发现与GPU精简

上轮皮肤/网页/归属说明已按用户后续授权单独提交为`2005ee7 [0.48.0]`，新增皮肤PNG/SVG保持忽略。本轮修复按用户授权单独整理为[0.48.1]提交；用户新增的基准目录及随后修改的忽略规则不由本轮暂存。

用户报告在生成目录运行Stage2却提示更新Stage1，以及不自动打开同目录`ecm.toml`。`startup_0481_red_01`在隔离目录复现：选中协议正常的测试Stage2，未使用的旧引擎仍被预检拦住；未传`--config`时未发现旁边的配置。初轮另一个断言因Windows路径大小写不同误判，修正为路径大小写不敏感比较，协议成功/拒绝条件未放宽。

排查按三个可证伪方向展开：全引擎预检误拦、入口没有默认配置发现、实际选中引擎缺协议或侧车覆盖路径。读取用户`D:/code/GIMPS/ecm-win-x86_64-sm89-cuda13.3_param0/`的配置与二进制：Stage1无协议标记，Stage2具备协议；配置分别指向同目录两套引擎。Stage1 SHA256为`561b32fa2a872e2d1b725f29d16004609f520d45495aee50246734f4a6ca1518`，Stage2为`1567d1cd813a90ce02bda1a8a4694043cb96cfb343f9c106fc52bbae57013af0`。未更改这些程序或实际队列。

`App::start`只验证本次选中的引擎。既有Controller在替换worker后、启动子进程前发送`StageChanged`通知，v2在该通知内检查实际将启动的路径；不兼容时抛出专用`DispatchBlocked`，由v2 tick边界捕获并提示，阻止该次自动派发，释放已停止worker的GUI租约。手动启动同样受检查，重启等待中的程序若已被替换为不兼容版本则取消重启。未修改共用Controller/旧GUI源码，也未取消进程级队列拥有权锁。

入口解析完参数后，未显式指定配置才查找exe同目录`ecm.toml`，并在UI侧车身份确定前解析该路径；不搜索当前工作目录。显式配置优先，错误路径不偷偷回退；未找到默认配置仍可打开GUI选择文件。

GPU页参考原GUI压缩信息：每设备卡片可折叠，利用率/功耗/温度及频率/显存通常各占一行，窄空间允许自然换行。常驻利用率/显存和功耗图高96逻辑单位，隐藏Y刻度与重复轴标题，保留网格、整数时间刻度及完整单位悬停；频率趋势可展开。ChartFormat的精简选项仅在GPU页启用，概率图保持原坐标与高度。

`startup_0481_green_02/result.json`通过八项入口/分派检查，包括中文便携路径、无关cwd配置不被采用、显式覆盖/失效不回退、默认缺配置仍打开，以及从Stage2自动切Stage1时拒绝不兼容引擎。使用无计算测试worker，不运行ECM。`startup_0481_green_01`保留测试助手没有UTF-8进程manifest导致中文配置参数解码失败的记录；为该v2测试助手补齐已有UTF-8 manifest后，子进程正常结束，阶段切换用例通过，不据“启动PID成功”掩盖崩溃。

`startup_0481_actual_01/result.json`对用户实际Stage2查询得到`ECM_QUEUE_OWNER=1`，再用已占有的隔离队列让GUI预检：未被旧Stage1误拦，仍报告队列已拥有，未启动计算子进程。`startup_0481_contracts_02`通过侧车与业务合同；`startup_0481_legacy_01/result.json`验证旧GUI73个保护文件、旧exe及既有配置投影保持不变。`contracts_01`在受限进程中weakly_canonical访问被拒绝，原失败日志保留；在正常桌面权限运行同一助手通过，未为测试修改正确路径逻辑。

本轮exe SHA256为`88f89aaa20cee0bd5d599903afde7bf678e520c72e252d0e01883ce3a6653c56`，默认DX11。`startup_0481_gpu_01`保存真实本机NVML的中文深色、英文浅色及900px窄窗口截图，已目视核对；RTX4070 Ti/4060 Laptop，144 DPI，现场存在其他GPU负载，不是性能测量。P5矩阵、Vulkan冻结及长期资格仍未改判。

`startup_0481_package_01`生成31文件本地候选包，ZIP SHA256为`0a8dca27ff667848d6fac8b0da1adef30545529ce5d5380c46dcf4dc2ee55f45`；包清单标注基础HEAD为2005ee7且含未提交修复。`startup_0481_package_test_01/result.json`通过30项包内文件指纹、中文目录迁移、三套原生皮肤、缺资源降级、侧车保留/重启和仅System32 PATH检查；计算worker不随包分发。`startup_0481_plan_01`确认HTML七阶段、六完成、P5待验收及390px布局，无脚本错误。原始失败目录保留，临时草稿未删除。

### 0.49.0 任务/日志呈现与可选隐藏标题栏

用户先要求提交上一轮启动修复，已完成`35eb324 [0.48.1]`，未纳入其新增的`/tools/benchmarks/prmers_pm1`忽略规则。随后继续完善GPU页、工作表格与多任务日志，并追加允许隐藏标题栏。本轮含新窗口选项和日志页签，按功能更新使用独立GUI版本0.49.0；该轮未单独提交，后续随0.49.1提交。

GPU使用固定图例，不再生成利用率、显存、功耗或频率曲线的勾选框，也不沿用旧会话的隐藏状态。图内左侧以对应颜色标记曲线，右上角显示当前数值和已知上限，百分比固定100%，功耗使用NVML上限，未知值为`--`；窄区域自动将标题分成两行。保留整数时间刻度、单位悬停和缺失值断线；概率图仍采用原可切换图例与坐标。实现入口为`ui/chart.h::chart`与`views/pages.cpp::draw_gpu`。

工作表改为八列，合并设备/引擎保持信息完整：Worker、状态、设备/引擎、进度、秒/曲线、ETA、完成/命中、操作。TaskDisplay复用老版的目标数摘要，任务行跨列并单独显示Stage2阶段/批次、B1/B2和已知曲线身份；完整源任务可悬停查看。进度展示完成/总曲线及百分比，恢复有标记；没有耗时样本则耗时和ETA保持未知。`ui/worker_presentation.h`只格式化已有数据，不改变任务或算法。`App::draw_workers`使用单元格内外留白、统一32逻辑高度的圆角操作按钮、固定表头及内容高度，避免空表格边线占满工作区。

`App::draw_logs`增加程序页签与各worker稳定ID页签，任务表和日志选择联动；每worker的原始/事件、跟随和子窗口滚动分别保留。现有两个侧车开关继续保存最后操作，作为后续页签默认值，不新增逐worker业务配置。手动上滚暂停自动定位，回到底部后继续；保留长日志裁剪、错误/命中颜色、任务摘要、速度/ETA、文件名和目录入口。

配置来源`config/ecm_options.json`新增独立侧车`window.title_bar`，默认true，共30字段；生成绑定与参考同步。视图菜单与流式设置共用同一个值。隐藏后原生窗口保留缩放边框，顶部空白区通过标准HTCAPTION命中处理拖动，右上角的图形按钮调用原生最小化/最大化/还原及原有WM_CLOSE安全退出。仅在该状态改变时更新WS_CAPTION并请求FRAMECHANGED，不为此更改DX11/Vulkan实现或启用多viewport。旧GUI源码、默认值和二进制保持不变。

`presentation_0490_runtime_01`抓到初始日志选择被程序页签覆盖，保存的失败日志显示log_worker持续为0；改为在遍历页签前固定请求的worker ID，不修改预期。`runtime_02`完整交互通过后，目视发现第一列缺外侧留白且空网格占满面板，补齐PadOuterX并按内容计算高度，同时完善日志任务摘要和耗时单位。`runtime_03/result.json`通过3个无计算worker、4个日志页签的实际点击、各worker原始/跟随选项隔离、表格选择联动、隐藏标题栏样式/拖动命中、最大化/还原、最小化/还原、保留缩放边框、窗口关闭及偏好重启；业务TOML字节不变。该轮exe SHA256为`f45e890ee55b7414cf81a61425a8a58283b0b39e7a39b6a13ef3d4184467aa3b`，后续仅修正标题栏设置的悬停说明。

`presentation_0490_contracts_02`通过30字段侧车往返、旧侧车默认显示标题栏、恢复进度、曲线计数、耗时未知值、ETA及Edwards省略param；`presentation_0490_legacy_01`核对旧GUI73保护文件、旧exe及既有配置投影完全一致。`presentation_0490_startup_01`通过八项0.48.1入口/安全派发回归。`presentation_0490_gpu_01`在3f4f448e…候选上保存NVML真实读数的中英/明暗/窄窗口和两种标题栏截图，已目视核对；该轮之后未改图表实现。现场GPU存在其他负载，这些是布局证据，不是性能数据。

最终exe SHA256为`e0fe880a23c740ebded398a2c49e48ed59de5b081c64ecfe5d7ffe5ca0a0fdab`，构建身份基础HEAD为35eb324并如实记录未提交源码。`presentation_0490_usability_01/result.json`在该exe上通过右侧工具往返、即时双语、明暗概率快照、流式设置搜索、窄窗口、字体恢复以及显式保存/重启；业务配置字节不变。所有原始输出仍在被忽略的`data/experiments/gui_v2/`；P5完整性能矩阵、不同物理DPI、真实跨屏拖动和可选Vulkan冻结资格不因本轮UI检查而改判。

本地[0.49.0候选包](../../data/experiments/gui_v2/presentation_0490_package_01/ECM-ELF-GUI-v2-0.49.0.zip)共31文件，ZIP SHA256为`be02eab59786fad87771a30ec1dd5cec4a1e9020092efcf237503a78afe894b4`。`presentation_0490_package_test_01/result.json`通过30项清单文件指纹、中文目录迁移、四套原生皮肤、缺资源降级、非法侧车保留、保存重启和仅System32 PATH检查；未执行ECM计算，也未启动Vulkan渲染。`presentation_0490_source_01/result.json`核对归档内156项冻结构建输入完全匹配，包含新格式化头和交互测试，不含仅Web使用的菱形瞳孔变体；本轮没有重做归档独立编译。HTML同步展示本轮进度，P5仍待验收；临时草稿保留。

### 0.49.1 任务表与日志可读性修正

用户要求将启停按钮前置、设备/引擎压为单行且保持列宽、任务摘要覆盖网格、表格ETA只显示秒数、日志ETA兼顾秒数与约化时间，并增加单行Stage2参数、速度图和紧凑主题日志。按z+1使用0.49.1，连同上一轮未提交的0.49.0优化一起提交；用户自己的忽略规则仍不纳入。

任务表列顺序为操作、Worker、状态、设备/引擎、任务进度、秒/曲线、ETA、完成/命中。设备列仍为84逻辑单位，缩写为GPU0 / S1或CPU / S2，悬停查看完整信息；全部启停保留在表格上方。新表ID隔离旧列宽记录，避免重排列后继承错误宽度。摘要跨列文字与原始任务提示保持不变，改为在EndTable合并绘图通道并生成边线后，于原子窗口内覆盖不透明背景和文字；裁剪仍受表格正文边界限制。旧截图`presentation_0491_grid_before_01/result.json`确认摘要空白带出现多种网格颜色，后续交互测试对同类区域逐像素检查，避免仅凭文字可见认定修复。

`ui/worker_presentation.h`分别生成秒数ETA和日志双格式ETA，先按秒取整再拆分日/时/分/秒，防止59.8秒余数进位后显示60秒。Stage2形状参数只从现有Stage2Status读取，单行B1/B2/S/D/P缺值为--，不从配置猜测或写死示例值。日志新增每worker耗时图，复用既有历史数组，单位秒/曲线，横轴采样序号；宽窗口与正文并排，窄窗口上下排列；窄且矮时通过工具行速度图按钮展开，当前数值保留在摘要，保留正文空间。仅日志正文覆盖ItemSpacing.y=2，不改变全局控件间距。`ui/log_presentation.h`按已有解析结果区分时间戳、信息、Stage1、Stage2、命中/完成、警告和错误，使用当前主题语义色；显示进度行附加约化ETA，不改写磁盘日志或共享解析器。原始/事件切换、各worker滚动、跟随和安全停止合同保留。

最终exe SHA256为`7241d437ebe2302a5bd70e1512ea91975a7553f04b9a3d5da2d7802f764fd57a`。`presentation_0491_runtime_05`与`presentation_0491_runtime_light_03`分别通过中文深色、英文浅色的三worker/四页签交互，核对首列启停、摘要背景像素一致、ETA两种格式、给定大数参数、6个速度采样、紧凑行距、独立原始/跟随选项、表格联动及隐藏标题栏控制/保存重启。1120×850窗口中增加正文高度至少为两行字体高度的检查，宽窄截图均已目视核对；测试只使用模拟worker，不执行真实ECM计算。`presentation_0491_contracts_03`通过ETA秒取整/跨分时日进位/未知值、Stage2参数缺值和完整示例、主题日志分类，以及既有侧车/业务合同；`presentation_0491_legacy_01`确认旧GUI73文件、旧exe及既有配置投影完全一致。配置生成物检查通过。

失败证据全部保留：`runtime_01`中的模拟进度一次性发出，被Controller按既有合同合并为一个采样，修正为间隔输出后取得6个采样，未改共享Controller；`runtime_02`在最大化后立即读取到上一帧按钮坐标，测试改为等待新坐标再点击还原。中间候选的`runtime_03/04`及`runtime_light_01/02`交互虽通过，窄窗口截图仍显示图表/折叠标题挤出正文，因此不能作为小面板可读性通过证据。最终将速度开关并入工具行、头部间距收为4逻辑单位、正文间距保持2，并增加上述最小正文高度断言。

[0.49.1本地候选包](../../data/experiments/gui_v2/presentation_0491_package_01/ECM-ELF-GUI-v2-0.49.1.zip)共31文件，ZIP SHA256为`31b94df708b6a6e31506e74619acc76dff3750330c624453e7a58344f6375232`。`presentation_0491_package_test_01`通过30项文件指纹、中文工作流迁移、四套原生皮肤、资源缺失降级、非法侧车保留、保存重启与仅System32 PATH；`presentation_0491_source_01`核对157项冻结输入完全匹配，包含两个呈现头和交互测试，不含仅Web的瞳孔变体。本轮未重做归档独立编译，未运行真实计算或Vulkan渲染。包生成于提交前，清单如实标注基础HEAD=35eb324及工作区源码快照，源码字节与本轮构建一致。

P5及跨屏/DPI、Vulkan冻结限制维持原结论。临时草稿继续保留，待用户后续决定整理。

### 0.49.2 字号、全部耗时图与退出反馈

用户要求设置可调整字号、耗时图移到任务工作区下方且不占日志、Stage2日志头补giant batch，以及参考v1优化退出；随后明确耗时图必须按worker顺序全部显示。用户补充要求保持y并增加z，本轮最终版本为0.49.2，当前工作区改动未提交；基础HEAD为65d56ec。workspace_0500命名的本地证据来自更正版本标签前的中间候选，保留目录与原始身份，不改写历史数据。旧GUI源码、二进制、既有业务配置与用户自改的忽略规则保持不变。

`config/ecm_options.json`新增侧车ui.font_size，范围10–30逻辑像素，默认15，字段总数31；生成器同步绑定与配置参考。设置滑条即时调整并可恢复默认，正文/日志及标题、辅助字体保持角色比例；字号与DPI比例独立。`ui::Fonts::push`使用逻辑基准，宿主在帧边界更新字号与主题；只改字号不调用字体文件重载、不清空字体图集。旧侧车缺项默认15，非法字号保留文件且禁止覆盖，独立偏好不写旧GUI font_size。

`App::draw_worker_speed`在任务表下方按Controller顺序绘制所有worker的独立耗时图，标题含worker与Stage，稳定ID避免共享悬停状态；未知耗时保留等待说明，图表多时滚动工作区查看。数据仍为原有hist_s_per_curve，单位秒/曲线、横轴采样序号，不增加采集线程，不因日志页签选择而只显示一项。日志和检查区不再重复绘制耗时图。`ui::stage2_batch`格式化既有批次数据，日志头补阶段/当前曲线、Giant或普通批次条及一位小数阶段剩余秒数；缺批次隐藏进度条，未知或负剩余值显示--，停止状态提示等待当前Stage2曲线完成。

退出复用原Controller状态机和工作流exit_confirm/graceful_stop_ms。`App::draw_exit`补居中确认与单独等待窗口，区分Stage1检查点、Stage2当前曲线及混合任务，显示剩余秒数/逐worker状态/截止后终止说明，支持取消/Esc、确认/Enter和等待期间强退/Esc。v2的request_close与begin_exit对进行中的退出幂等，避免重复关闭重置截止时间；析构清理也复用此入口。宿主在最小化后收到仍需确认/等待的关闭请求时恢复窗口。kill和无运行任务的直接退出行为不变，旧Controller与旧GUI实现不改。

构建exe SHA256为`5d9144b11b759929e00d9a18dc8e59f7bfa9027f4406b3dc8ad438371aeaf486`。`workspace_0500_font_exit_03`通过实际字号滑条15→20、即时生效、退出保存/重启且只载入一次字体文件；通过取消按钮、Esc取消、Enter确认、混合阶段安全停止、最小化恢复、重复关闭保持截止时间、强退按钮、超时和显式kill策略。每分支核对原业务字节及所有模拟worker进程退出。`workspace_0500_contracts_01`通过31字段侧车往返/旧侧车默认值/越界保留、Giant批次2/8与剩余12.0秒、批次夹取/未知时间及既有业务合同；`workspace_0500_legacy_01`核对旧GUI73文件、旧exe与既有配置投影一致。

测试失败原样保留：font_exit_01点击了弹窗首帧尚未居中的旧坐标，之后改为连续三帧控件位置稳定才操作；font_exit_02误将模拟worker标准输出当作前端诊断日志，实际Controller已记录两worker正常停止并退出，改为检查该诊断状态、无超时/强退及进程退出，未改应用实现或降低业务门限。font_exit_03完整分支通过。

`workspace_0500_runtime_02`通过中文深色三worker、四日志页签、两个Stage1历史共12个采样、工作区滚轮往返及全部耗时图；滚动截图已目视核对两张实际曲线和一项Stage2等待说明。日志全宽且无图表，Stage2的Giant批次2/8和剩余12.0秒与原始数据一致。`workspace_0500_light_01`通过英文浅色与宽窄布局，表格/日志联动、隐藏标题栏窗口控制及重启合同继续通过。上述布局验证的exe为5d9144b1…。

随后仅将字号滑条上下限与重置默认值改为引用schema生成值，生成器同时输出数值边界，避免控件维护第二份配置定义；图表与退出实现未变。配置引用修正后的中间exe SHA256为`96e07687a4f52072c9b7ad6533918fc7abca7715d420f4584bfebf1db3cc9252`，`workspace_0500_contracts_02`通过新生成绑定与原合同，字号/退出最终复核见font_exit_04。随后按用户要求仅将版本标签更正为0.49.2，最终exe SHA256为`9fc9711281dfc237507b8cecd93b8f659c250b2192e2bcaae11a7a371c5a98ab`。`workspace_0492_contracts_01`再次通过侧车/字号/批次和既有业务合同；版本更正没有修改字号、图表或退出行为。

[0.49.2候选包](../../data/experiments/gui_v2/workspace_0492_package_01/ECM-ELF-GUI-v2-0.49.2.zip)共31文件，ZIP SHA256为`760d1f18b89f7e1ac45ca5e14d39aeb52919292878eb90a44469f6f1d78ee65a`。`workspace_0492_package_test_01`通过30项文件指纹、中文工作流目录、四套原生皮肤、缺资源降级、非法侧车保留、保存重启及仅System32 PATH；`workspace_0492_source_01`核对157项冻结输入逐字节匹配并包含新交互测试，不含仅Web瞳孔变体。本轮未重做归档独立编译，包为基础HEAD=65d56ec加未提交源码的本地快照；中间workspace_0500包不作为最终分发。

原始数据位于忽略目录data/experiments/gui_v2；无真实ECM或Vulkan渲染，P5和真实跨屏/DPI资格维持待验收，临时草稿保留。

### 0.49.3 表格纵向完整高度与日志字体一致性

用户报告工作区表格缩放后出现内部纵向滚动，日志与UI在同一字号下字面大小不同。按保持y、z+1的要求使用0.49.3，已按用户要求先提交为`8223702 [0.49.3]`，本次GPU更新另行开发。应用旧GUI、业务配置、用户的忽略规则及其run_all测试入口不改。

先构建仅增加审计度量的0.49.2诊断候选，SHA256为`795aef2f08c11f654cc38c1e33ad03c2e97adc53f727a6008a64c3220c23b1d2`。`table_font_0493_before_01/result.json`在6个不自动启动的worker上复现：1920×1200、字号15、100%时表格内部ScrollMaxY=19px；1120×850、字号20、125%时=748px。后者正文与日志的名义字号均25px，但字母M字面高度分别15px与16px，字体分别为微软雅黑和Consolas。字体文件与PushFont链路核对后，不支持重复DPI缩放假设。旧大字号900×700样本的表格已在视口外，ContentSize为0，不能用零滚动值认定通过；后续把该样本窗口扩大并要求全部6行实际提交及真实数据高度不超表格高度，增强验证。

`App::draw_workers`不再减去耗时图预留高度或截到视口剩余空间；表格高度按全部行、摘要换行、留白和横向滚动条计算，移除ScrollY与内部冻结表头。只保留ScrollX；该ImGui实现对仅ScrollX的子窗口明确抑制纵向滚动，额外验证实际末行数据高度，防止仅隐藏滚动条却裁切内容。视口不足时使用外层工作区统一纵向滚动，耗时图仍按worker顺序位于表格下方。`App::draw_logs`把正文角色从Mono改为Body，与UI正文共用字体族、实际字号和DPI；独立的Mono字体角色保留供其他技术显示使用，原始日志字节与语义色不变。

审计模式记录内部ScrollMaxY、数据高度、提交行数、正文/日志字号与字面高度，作为回归入口，不增加普通运行日志。`test_table_typography.py`在真实DX11界面验证6个空闲worker，不启动计算。`table_font_0493_after_01/result.json`通过1920×1200/字号15/100%、1120×850/字号20/125%、1920×1400/字号30/200%三组样本：内部ScrollMaxY均为0；全部6行提交，实际数据高度567/925/2196px分别不超过可用583/943/2212px；正文与日志实际字号均为15/25/60px，M字面高度均为9/15/32px，且使用同一Body字体。此矩阵为侧车缩放和字号覆盖，不等同真实跨屏DPI资格。

最终exe SHA256为`e2230aac5e83ee3818a41d6bad48681b44c4b578e29ee0cfc23fc7cdec22b9e6`。`table_font_0493_runtime_01`通过3个模拟worker的任务表、摘要遮盖网格、秒数ETA、Stage2参数与giant batch、全部worker耗时图、工作区滚动、日志切换/跟随、标题栏切换和窗口退出回归；`table_font_0493_contracts_01`通过应用/侧车合同；`table_font_0493_legacy_01`确认73个旧GUI保护文件逐字节不变，旧exe身份不变。

[0.49.3本地候选包](../../data/experiments/gui_v2/table_font_0493_package_01/ECM-ELF-GUI-v2-0.49.3.zip)共31文件，ZIP SHA256为`b193f23bc96214d8ecea866e7a043a97cf5d59c4b07db860ce482cb162d40a0b`。`table_font_0493_package_test_01`通过30项文件指纹、中文工作流目录、四套原生皮肤、缺资源降级、非法侧车保留、保存重启及仅System32 PATH；`table_font_0493_source_01`确认源码归档157项冻结输入逐字节匹配，包含新回归测试，排除仅Web变体和用户的run_all入口。包为基础HEAD=65d56ec加未提交源码的本地快照，本轮未重做归档独立编译。

正式使用说明与HTML同步；P5、真实跨屏/DPI及可选Vulkan冻结资格维持原结论，无真实ECM或Vulkan渲染，原始失败证据与临时草稿保留。


### 0.49.4 GPU页显示与设备详情

按用户补充要求，先提交已验证的0.49.3（`8223702`），再继续本轮。恢复前核对全部157项冻结构建输入与上一候选一致；GPU改动暂存于忽略目录后单独恢复，没有混入前次提交。0.49.4已按用户“修改前先提交”要求提交为`7d43fed [0.49.4]`，用户的.gitignore修改与run_all入口保持不变。

GPU时间标签移到设置页“GPU 图表时间范围”，以15/30/60/120秒标签选择，默认60秒，所有设备共用并写入独立侧车`ui.gpu_history_seconds`。默认值、合法值及英中说明在`config/ecm_options.json`维护，经生成器同步；缺项使用默认值，非法值保留原侧车且禁止覆盖。GPU页不再显示时间控制。频率图移除独立折叠栏，利用率/显存占用、功耗、核心/显存频率三图直接显示；设备卡片本身仍可折叠。

基本信息与图内右上角使用` | `分隔不同指标，使用两侧留空的` / `分隔当前值与上限/总量，不再出现“上限/max/limit”字样。核心与显存频率各自保留MHz单位，避免把显存频率误作核心频率上限；利用率与显存百分比各自写成当前值/100%。两行概要、96逻辑单位图高、无Y刻度、整数时间轴与一位小数悬停读数保留。

`GpuDetailsCache`为v2独立模块，不修改旧GpuMonitor。后台只读一次NVML驱动版本、驱动支持的CUDA版本及计算能力，CUDA核心数复用监控器的NVML报告。SM数量通过可选CUDA Driver设备属性读取，以物理UUID匹配NVML与CUDA设备，不能按枚举编号假定同一张卡；不创建CUDA上下文或分配计算显存。结果发布为只读缓存，悬停不访问驱动；可选入口缺失、查询失败或UUID无法匹配时对应字段为`--`。悬停依次显示型号、驱动/CUDA、SM规格/数量/CUDA核心数、显存总量、NVML/500ms。CUDA字段指驱动支持版本，不宣称本机Toolkit版本。API依据见[NVML系统查询](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlSystemQueries.html)、[NVML设备查询](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)及[CUDA Driver设备查询](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html)。

`gpu_0494_runtime_01`在真实DX11界面通过中英文两组设置/悬停/重启检查：两张设备各3图默认显示，时间设置15秒与120秒均保存并恢复；两行概要空格及图内分隔规则通过，业务配置字节不变。只读参考为驱动610.88、CUDA支持13.3：RTX4070 Ti为SM8.9/60 SM/7680核心，RTX4060 Laptop为SM8.9/24 SM/3072核心，分别与nvidia-smi和直接驱动查询匹配。此处为设备规格核对，不是计算性能测量；测试没有启动计算worker，不改变已有外部计算负载。

最终exe SHA256为`793f9bb09b120753a5ddaeee3b6c9efef24abdd566158153a9ae42f185fdec3e`。`gpu_0494_contracts_01`通过应用/侧车合同，覆盖旧侧车默认、新字段往返和非法61秒拒绝覆盖；`gpu_0494_legacy_01`确认73个旧GUI保护文件与旧exe身份不变。`gpu_0494_package_test_01`通过30项文件指纹、中文工作流目录、四套原生皮肤、缺资源降级、非法侧车保留、保存重启及仅System32 PATH；不作为独立干净系统资格。

[0.49.4本地候选包](../../data/experiments/gui_v2/gpu_0494_package_01/ECM-ELF-GUI-v2-0.49.4.zip)共31文件，ZIP SHA256为`2036935881cce1cb1a15629dfb64d6b8ffefd1f1648e3fe4152effc37c6d5178`。`gpu_0494_source_01`确认源码归档159项冻结输入逐字节匹配，包含新GPU回归入口，排除仅Web变体和用户的run_all入口。包为基础HEAD=8223702加本轮未提交源码的本地快照，本轮未重做归档独立编译；正式说明与HTML同步。P5、真实跨屏/DPI和可选Vulkan冻结资格维持原结论，无Vulkan渲染，原始证据与临时草稿保留。


### 0.49.5 导航排序、右侧标签与移动跟随

用户要求修改前提交上一版，已先将0.49.4的已验证源码提交为`7d43fed`；核对159项冻结构建输入一致，未提交用户的忽略规则和run_all入口。本轮按保持y、z+1使用0.49.5，当前改动未提交；草稿保留。

导航使用圆角按钮，最小高40、纵向间距8、圆角8逻辑单位，随字号与DPI缩放；悬停和选中使用主题背景。按住约0.32秒后拖到另一项的上/下半部排序，插入线指示落点，快速拖划不启动排序。七页编号的完整排列保存为`ui.nav_order`，右侧页保持其返回顺序；schema维护默认值及排列约束，生成器验证重复、遗漏和非法编号。至少保留一个中央页面的约束不变。

右键页面或长按拖到“右侧标签页 >”可移入右侧。移除Tools总窗口、内部页面按钮、移回按钮和左下返回按钮；每个页面直接作为独立ImGui停靠窗口进入同一右侧节点，多个页面共用横排原生标签并可逐页关闭。关闭后自动取消对应位掩码并激活中央返回页。设置对应区域改称“导航与右侧标签页”，移除总显示开关；`ui.right_visible`仅保留读取旧侧车兼容，旧Tools布局识别后重建为原生页面标签。页面仍允许拖动停靠或浮动，右侧各窗口保留自己的会话滚动位置。

系统标题栏默认隐藏，窗口控制位于应用顶栏右侧，空白区拖动、边缘缩放和安全退出流程保留。已有显式`window.title_bar`选择不强制覆盖；旧侧车缺项使用隐藏默认值，设置和视图菜单可恢复原生标题栏。

ELY跟随按诊断流程先复现：0.49.4的`nav_0495_pet_before_01`在真实Windows SC_MOVE循环中，主窗口位置已改变，普通审计帧数仍为7，宠物位置不变；系统移动循环暂停主循环的假设成立，位置缓存和层级不是该样本原因。`Window::follow_owner`从主窗口位置/尺寸消息直接更新原生透明窗口位置，复用当前像素，不加载素材、不重新缩放、不访问GPU；DPI栅格重绘留给正常tick。最小化时仍隐藏宠物。

最终exe SHA256为`a34f192dadff75763fdb8810297762d8a6008251d4c9ad931b92a2acbed183b9`。`nav_0495_runtime_02`通过长按排序、快速拖划不排序、右侧同节点横排标签、切换/关闭返回与顺序/停靠重启；真实SC_MOVE循环中主循环帧数不增加，连续两次移动后宠物几何立即跟随，证明不依赖松手后的下一帧。该样本只覆盖同一显示器的系统移动循环，不据此声明跨屏/DPI资格。

`nav_0495_presentation_01`通过顶栏命中、最大化/还原、最小化/恢复、窄窗口缩放、自定义关闭、任务表/全部耗时图及日志回归；`nav_0495_usability_03`通过自动DPI下的长按移入、右侧关闭返回、中英文切换、明暗概率快照、流式设置搜索/空状态、显式保存和重启及字体救援。先前runtime_01在停靠重建首帧的临时坐标操作失败；usability_01/02分别点击了尚未滚入可见区、平滑滚动尚未停稳的Save坐标，后续按实际可见位置等待后验证，原始失败证据保留。`nav_0495_contracts_01`通过应用/侧车合同，包含导航顺序往返与非法排列拒绝覆盖、隐藏标题栏默认及已有显式显示选择保留；`nav_0495_legacy_01`确认73项旧GUI保护文件和旧exe身份不变。

[0.49.5本地候选包](../../data/experiments/gui_v2/nav_0495_package_01/ECM-ELF-GUI-v2-0.49.5.zip)共31文件，ZIP SHA256为`4a5500cadf86f02ada36cc9c5e9b733b4a273175754e3fc4f1558b26c9cc0a48`。`nav_0495_package_test_01`通过30项文件指纹、中文工作流目录、四套原生皮肤、缺资源降级、非法侧车保留、保存重启及仅System32 PATH；`nav_0495_source_01`确认源码归档159项冻结输入逐字节匹配，包含新导航测试和更新的可用性入口，排除仅Web变体与用户run_all入口。包为基础HEAD=7d43fed加本轮未提交源码的本地快照；未重做归档独立编译或独立干净OS验证。

正式说明与HTML同步，P5、真实跨屏/DPI及可选Vulkan冻结资格维持待验收；没有运行真实ECM或Vulkan渲染，测试只使用隔离空闲worker或合成worker，不改变用户工作流及外部计算负载。

## 1. 分析边界与基线

### P3–P6 扩展授权与宠物兼容

2026-10-21 用户授权继续 P3–P6，并参考 `.refactor/ELYTerminal/src/pet` 加入宠物。参考宠物为程序自带的分层矢量角色与行为状态机，不是 ChatGPT 宠物包格式。兼容范围包括可关闭的透明非激活独立窗口、主窗口边缘定位/顶部空间不足时转到底边、拖动及抚摸、状态表情、DPI、最小化和退出联动。主 GUI 使用 Vulkan；宠物透明窗口采用 Windows 原生分层位图合成，离线转换的上游矢量资产带 MIT 告知，运行时不引入浏览器/GPUI。

用户另明确批准共用 worker 入口新增进程级队列互斥，以防两版 GUI 重复启动同一队列/worker；旧 GUI 源码、默认值与算法不变。v2 控制要求 worker 支持 `ECM_QUEUE_OWNER=1`；旧二进制不具备该互斥，不能假称受新锁约束。进程生命周期占用与短时队列编辑锁分离，worker=0 的全队列占用与各分区互斥，不同 worker 分区可并行。新增 v2 UI 字段仅位于 `config/ecm_options.json` 的独立侧车定义；现有 owners/options 投影保持原样，现有生成绑定/模板保持原样，仅 schema 源与哈希清单新增受批准内容。

P3 加入侧车/渲染生命周期及宠物平台窗口基础，P4 完成业务页面与宠物状态接线，P5 同时验收宠物关闭/静止/动画的资源边界，P6 独立包携带资产、来源、许可及明确运行条件。所有未实测的物理跨屏、IME、功耗/吞吐条件仍明确列为待验证，不以合成事件代替真实硬件资格。

- 初始分析检查时 HEAD 为 `8969467`，包含当时工作区改动；正式 P0–P2 实施从干净的 `a30c373` 开始，以 P0 保护清单及二进制哈希作为验收身份。
- 公共源码版本与 `build_gui_local/ecm_gui.exe --version` 均为 `ECM-ELF 0.46.7`。相同版本号不能独立证明源码和二进制完全一致。
- 旧 GUI SHA256：`32F3A9A2905FF39E6F098F0C943DEDBAA3BDBEA291EF7122417634958B85919B`，已由 P0 冻结；不代表 v2 构建身份。
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

上游深色基础值为背景 `#252525`、表面 `#2c2c2c`、内嵌 `#202020`、文字 `#f3f1f0`、边框 `#3f3f3f`；浅色对应 `#fcfaf7`、`#ffffff`、`#f6f4f1`、`#181613`、`#e1dfdd`。P2 已接入独立 v2 主题，旧 GUI 配色保持原样。

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

旧 `third_party/imgui/backends/` 仅含 Win32/DX11，保持不变。P1 在 `third_party/imgui_vulkan/` 固定同提交 Vulkan 后端，在 `third_party/gui_v2_vulkan/` 固定 headers/Volk，许可与来源随文件保存。本机无 `VULKAN_SDK` 变量或 `C:/VulkanSDK/`；构建不查找开发 SDK，运行使用系统 loader。开发验收仅从官方 SDK 归档解出 validation layer 到忽略的证据目录，未执行安装器。

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

- 左侧导航：工作台、结果、任务生成、ECM 概率、GPU、设置和组件。当前宽布局占比16%，窄布局11%；可用宽度不足时显示编号并保留完整标签提示。
- 中央主区：稳定窗口 `###v2.workspace` 按独立侧车的 page 切换七个页面。结果和表单获得中央主区域；旧版布局不变。
- 右检查区：`###v2.inspector` 展示选中 worker 详情；GPU 使用中央独立页面。初始宽布局右区占剩余空间29%，可调可关；小于1100逻辑宽度时隐藏。选中 worker 不自动抢焦点。
- 中央底部：`###v2.logs` 切换选中 worker 的事件/原始输出，默认占中央区域高度28%，可手动停靠/弹出。
- 客户区顶部用一条紧凑命令栏展示当前配置和全局启动/安全停止；保留菜单入口，避免迁移时丢失低频动作。Prime95 的异常/待处理提示在顶部测量预留空间。

这些是当前默认布局比例；用户停靠状态另存侧车。保留 Windows 系统标题栏和不透明主题。

导航只切换工作区页面，不重复建立业务实例。默认布局仅首次、显式重置或跨宽窄断点时重建；浮动测试窗不随主 swapchain resize 重建。断点重排当前会重建主布局，这是现存行为。

### 4.2 v2 停靠、独立状态与窄窗口

- v2 提供用户手工停靠、分离和跨屏窗口；v2 窗口 ID 不因中文标题改变。旧窗口及其 ID 不修改。
- v2 初次运行直接建立现代默认布局，重置只影响 v2。旧布局留给原程序继续加载，v2 不自动迁移或写回旧布局。
- v2 的启动页、明暗主题、字号/字体角色、窗口位置和停靠状态写入独立侧车文件，默认在 `%LOCALAPPDATA%/ECM-ELF/gui-v2/<启动配置身份>.toml`；`--ui-state` 可显式指定其他路径。路径不可写需明确诊断，不悄悄写旧工作流 TOML。
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
- ImGui Vulkan 初始化、帧提交和 Win32 surface 桥接集中在 `render/vulkan_window.*`，不额外分出 host 文件；多 viewport 完整生命周期留 P3。
- `ui/theme.*`、`ui/fonts.*`、`ui/widgets.*`：Ely 风格语义、尺寸、字体角色和有限控件集合。
- `app.*`、`views/pages.cpp`：导航、通知、停靠、Controller 适配及各功能页；不调用旧 App 的绘制或保存设置入口。
- `ui_state.*`：独立 v2 UI 状态读写与 schema 版本；不持有 WorkerProc。
- v2 专用第三方目录记录固定 Vulkan 后端文件、来源提交和许可。需要后端修复时先评估 v2 独立版本，禁止给旧目录打补丁。

入口、render、theme/fonts/widgets、App/pages、ui_state 和 pet 模块均已接通。优先使用 ImGui 公共 API；DockBuilder 内部 API 限于 v2 布局模块。不引入通用渲染引擎或 GPUI 兼容层。

### 5.1 只读复用业务模块

候选为旧目录中的 `worker_controller/worker_proc/log_parse`、`toml_file`、`results`、`worktodo_gen`、`probability_model`、`gpu_monitor`、`localization`、`task_display` 和系统文件/DPI辅助函数。按实际依赖为 v2 目标明确列源，直接编译未修改的原文件；必要的 v2 包装在新目录实现。

旧 `app.cpp`、`probability_panel.cpp` 与 `main_win32.cpp` 是行为/交互参照，不整体链接到 v2，不为新版改写旧 UI。复用代码中的进程路径、默认值和相邻程序探测可能依赖 exe 目录，P3 必须验证 v2 独立输出位置，不假设更换程序名后天然正确。

新主题和 Vulkan 层不反向进入 Controller/Proc 或数学模型。若必须修改共用源才能接通新功能，应先寻找 v2 包装；超出“原 GUI 不变”边界的公共修改需单独列出，不能并入本计划默认范围。

### 5.2 独立构建与可选依赖

- 新增目标 `ecm_gui_v2`、专用 ImGui/Vulkan backend 目标及 v2 业务适配目标；不能把旧 `ecm_gui_imgui` 拉入 v2，否则会同时引入 DX11 后端。
- 新增 `ECM_BUILD_GUI_V2`，默认 OFF；v2 构建入口显式开启。根构建接线不放在旧 `ECM_BUILD_GUI` 成功分支里，旧 GUI 关闭时仍应能单独构建 v2。
- 新增候选入口 `tools/build/build_gui_v2.bat` 与 `tools/build/dev/build_gui_v2.ps1`，默认输出 `build_gui_v2_local/`。它只构建 v2；原 `build_gui.bat` 仍只构建旧版。
- 专用 headers/backend 检查仅在启用 v2 时进行；缺依赖时 v2 给出明确诊断，不能影响旧 GUI、CLI。Volk 动态加载系统 Vulkan，不依赖 SDK、import library 或 GLSL 工具链；固定后端已包含所需 shader。
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

退出条件：干净分发目录中运行、缺Vulkan清晰失败、缺字体/缺NVML、状态重启恢复与安全退出通过；原版仍可独立启动。公共版本按功能更新规则在实际开发时评估，不把前端 v2 写成公共版本 2.x，也不预占并行开发中的下一版本。当前是独立 0.47.z 开发与候选分发准备，原 GUI 公共标识仍保持原样。

依赖：P0 → P1（Vulkan宿主）→ P2（主题字体）→ P3（完整布局/多窗口/业务）→ P4（页面）→ P5（双版本验收）→ P6（独立分发）。首批 P0–P2 已完成；后续实现与仍未通过的资格条件以实施进度为准。

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

旧 `test_gui_all.ps1` 等部分脚本默认查找旧 `build_gui/`，保留这些脚本，不把它们直接当 v2 全套验收。旧版回归和 v2 验收分别指定二进制、配置、构建目录与证据位置。v2 测试 helper 显式构建，不进入普通交付目标。P0 已运行的旧合同检查、P1–P2 宿主/组件检查见实施进度；完整业务与性能回归仍属于后续阶段。

## 8. 当前范围与待验证事项

P0–P4 已接通并保留证据，P5 本机回归及性能修复已执行但仍有待验门禁；P6 独立候选准备已完成，不包含完全生产资格声明。当前功能和运行条件见 [GUI v2](../usage/GUI_V2.md)。

TODO：不同物理 DPI/真实 IME，触控板真实体验，多窗口闪烁主观资格，用户选择渲染设备，可选随包字体，长结果与计算并行的完整性能、总显存/受控功耗，以及独立干净机器验证。两块同为 144 DPI 的跨屏测试不替代这些条件。Edwards 联合模型相对归一预设尚无直接 Stage2 标定。

实际状态以实施进度和具名证据为准；旧 GUI 源码/默认值/二进制保持原样。共用 worker 新锁仅约束更新后的计算程序。临时草稿继续保留，完成后由用户决定整理。
