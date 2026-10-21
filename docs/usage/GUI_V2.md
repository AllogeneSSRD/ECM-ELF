# GUI v2

GUI v2 是独立的 Windows / C++ / Dear ImGui / Vulkan 前端，当前开发版本 0.47.4。工作台、结果、生成器、概率、GPU、设置已接通；原 DX11 GUI 的源码、默认值、二进制和构建入口保留。阶段证据与限制见 [实施草稿](../development/GUI_MODERNIZATION_DRAFT.md) 和 [离线 HTML](../web/gui-v2-vulkan-plan-20261021.html)。

## 构建与启动

运行 `tools\build\build_gui_v2.bat`，默认生成 `build_gui_v2_local/ecm_gui_v2.exe`。测试入口 `tools/build/test/build_gui_v2.ps1 -BuildDir <独立目录> -TestHelpers`；helper 不进入默认交付目标。根开关 `ECM_BUILD_GUI_V2` 默认 OFF，与旧 GUI 独立。需要 MSVC/CMake 及仓库固定源码，构建无需开发 Vulkan SDK、CUDA 或浏览器运行时。

运行要求系统 Vulkan 1.1 loader、graphics/present 与 swapchain 驱动；缺失明确失败，可另开原 GUI。常用参数：

- `--config <工作流.toml>`：打开已有业务配置；设置中也可选择。autostart 经过能力与拥有权检查后执行。
- `--ui-state <侧车.toml>`：默认 `%LOCALAPPDATA%/ECM-ELF/gui-v2/<配置身份>.toml`；与业务文件分离，非法侧车保留并禁止覆盖。
- `--theme dark|light`、`--language en|zh`、`--ui-scale 1|1.25|1.5|2`：初始外观；设置可选自动 DPI。
- `--pet` / `--no-pet`、`--pet-static`：宠物及动画覆盖，默认关闭。
- `--no-nvml`：关闭可选的 500ms 后台 GPU 监控。
- `--diagnostics <文件>`、`--version`：诊断路径与独立前端版本。

设置中的 worker 路径、数量为 v2 偏好；数量 0 跟随工作流。相对程序路径优先工作流目录，其次前端目录。外观保存不改旧 `[gui]`。计算设置须显式保存，复核文件指纹并保留未改行/注释；外部修改须重载。运行中禁用编辑，未保存编辑禁止启动。

## 队列、页面与宠物

必须使用更新后的 Stage1/Stage2 worker：`--gui-queue-protocol` 返回 `ECM_QUEUE_OWNER=1`。共用入口的进程级文件区间锁覆盖 worker 生命周期，同一队列的相同分区拒绝重复运行，不同分区可并行；全队列与各分区互斥。崩溃由系统释放锁，持久锁文件不能手工删除绕过占用。原 GUI 必须指向更新 worker 才能参与保护；任意旧计算二进制不遵守新锁，v2 会拒绝启动。算法、曲线、param、sigma 及工作流格式不变。

工作台/检查区区分 Stage2 任务曲线进度与阶段/批次，保留 worker 版本、命令、Prime95 提示、事件/原始日志和安全停止。关闭日志或切页仍推进 Controller。结果支持 F/NF、阶段、来源与搜索，复制原始 JSON/checksum；筛选或文件更新清理失效选择。生成器按目标 worker 分区，输入或选项变化使预览失效，追加前复核文件并锁定队列；CPU 拒绝不支持组合。

概率正反算、图表、CSV 对应显式计算的输入快照。挠群阶 T 不等于 D_eff。Edwards Z2xZ8 的已有 Stage1 经验代表值为 26.4081（B1=256、15–25 bit 点口径）；按 Edwards/Suyama 相对收益归一到 GMP 基线约 29.0，作为联合模型启发预设，未经直接 Stage2 标定。可选择经验口径或自定义，不把 16 当完整 Edwards 预设。数学模型本身和旧概率页不改。依据见 [参数化参考](../reference/ECM_PARAMETERIZATIONS.md) 与 `tools/ecm_prob/README.md`。

常用筛选使用点击标签；滚轮短暂提升刷新频率以平滑滚动，静止恢复 10Hz。Toast 不抢焦点；启动、安全停止、配置/生成器反馈保留详细日志。GPU 未知样本不伪造零值，单位为百分比、MHz、W、摄氏度和 MB。

Ely 宠物兼容固定 MIT 上游角色、七状态、边缘定位、拖动/抚摸、最小化和退出联动；使用原生透明非激活 owned 窗口及离线 atlas，无 GPUI/Web runtime。关闭释放位图，最小化暂停动画。来源/许可见 `third_party/gui_v2_pet/` 和 `assets/gui_v2/pet_ely.json`。

## 验证边界

已验证同为 144 DPI 的两块物理显示器及合成输入。不同物理 DPI、真实 IME、总 GPU 驻留、受控功耗/计算并行与独立干净机器仍需专项资格。系统 Segoe UI / 微软雅黑 / Consolas 不随包分发，缺中文字体退回英文。HDR/显示格式改变导致 render pass 不兼容时要求重启。

`--validation` 要求开发验证层，普通运行不需要。`--frames`、`--audit-ui`、`--audit-app`、`--no-loader`、`--inject-failure`、`--simulate-missing-cjk` 是测试入口。显式 Vulkan 分配不等于进程总显存，不能据 API 名称声明更低占用。验证脚本位于 `tools/test/gui_v2/`，证据存忽略目录 `data/experiments/gui_v2/`。
