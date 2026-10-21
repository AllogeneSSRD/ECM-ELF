# GUI v2 开发预览

GUI v2 是独立的 Windows / C++ / Dear ImGui / Vulkan 前端，当前版本为 0.47.2，完成 P0–P2。现有 DX11 GUI、公共 0.46.7 版本、默认配置和原构建入口保持不变。

当前提供单窗口组件预览：Ely 明暗主题，中英切换，100/125/150/200% 缩放，正文 15、辅助 13、标题 22 逻辑像素及等宽数据角色，按钮、表单、选择、状态与有限线条图标。初始缩放取窗口 DPI；界面比例也可以手工选择。窗口 DPI 改变时重新计算比例。缩放从逻辑基准重新建立样式，语言/比例切换在帧间释放后端纹理再重建字体。

这是组件预览程序，尚未接入任务控制、真实日志/结果、生成器、概率、监控或 UI 状态保存；关闭后外观选择不会保存。样板路径仅供输入预览，不读取或保存工作流，不启动 worker。业务接线、多 viewport、物理跨屏 DPI、IME 和正式分发验收属于后续阶段。

## 构建与运行

沿用本机已有 MSVC/CMake 环境，在仓库运行 `tools\build\build_gui_v2.bat`；默认生成 `build_gui_v2_local\ecm_gui_v2.exe`。原 `tools\build\build_gui.bat` 仍构建旧版。

新功能测试入口为 `tools/build/test/build_gui_v2.ps1`，可用 `-BuildDir build_gui_v2_test` 指定独立目录，用 `-VcVars` 指定 MSVC 环境脚本。入口直接配置 v2 子工程，不需要 OpenCL、GMP、OpenSSL 或 CUDA。根工程也支持 `ECM_BUILD_GUI_V2=ON`，默认 OFF；根工程本身的计算依赖仍按原要求检查。v2 与 `ECM_BUILD_GUI` 开关互相独立。

源码依赖为原目录中固定于 `64944b4520b30772de8dbf0b37d0311746477a32` 的未修改 ImGui 核心/Win32 后端，以及新增的 `third_party/imgui_vulkan/` 与 `third_party/gui_v2_vulkan/`。缺依赖会在配置时给出具体路径。不查找开发 Vulkan SDK、import library 或着色器编译器，也不联网下载构建依赖。

运行要求系统 Vulkan loader 与支持 Vulkan 1.1、graphics/present 队列和 swapchain 的显卡驱动。不满足时明确退出并提示使用原 GUI。不会在 v2 内切换到 DX11。

启动参数：

- `--version`：输出独立 v2 版本后退出。
- `--theme dark|light`、`--language en|zh`、`--ui-scale 1|1.25|1.5|2`：初始外观。
- `--diagnostics <path>`：诊断文件；默认当前目录 `ecm_gui_v2.log`。
- `--validation`：要求已安装或通过 `VK_LAYER_PATH` 提供 Khronos validation layer；缺失明确失败。普通启动不要求该开发层。

正文优先 Segoe UI，中文优先 Microsoft YaHei / SimHei / SimSun；英文正文和等宽 Consolas 合并 CJK 救援。中文字体不可用时回退英文并显示说明。字体由 Windows 字体目录读取，不随程序分发；Ely 设计许可见 [NOTICE](../../src/ecm/application/gui_v2/ui/NOTICE.md)，Vulkan/Volk/ImGui 许可见各第三方目录。

## 验证入口与限制

`tools/test/gui_v2/baseline.py` 冻结/复核旧版，`test_host.py` 验证真实 HWND/Vulkan 窗口生命周期与故障退出，`test_ui.py` 验证中英、明暗、四档缩放、重复纹理重建、等宽/CJK 字形及控件输入。测试显式指定 `--exe`、新的 `--out` 和开发层 `--layers` 目录，原始证据写入 `data/experiments/gui_v2/`。

测试专用参数 `--frames N`、`--no-loader`、`--inject-failure device-lost|surface-lost`、`--simulate-missing-cjk` 用于受控退出和回退验证。故障注入不代表实际拔卡/驱动崩溃测试。

`--audit-ui` 开启测试消息和显式 Vulkan 分配审计：`WM_APP+32` 切换主题/语言/比例，`WM_APP+33` 在 Win32 NewFrame 之后注入 ImGui 鼠标事件，使控件测试不依赖桌面鼠标所在窗口；文本/键盘仍发送 Win32 消息。普通运行不开启这些入口。审计记录应用经 Volk 发出的 `VkDeviceMemory` 分配，覆盖字体纹理及绘制缓冲，不包含驱动和 swapchain 的全部驻留内存，不能据此声称进程总显存或占用优于旧版。

当前阶段的有效证据、阶段退出条件及后续工作见 [实施草稿](../development/GUI_MODERNIZATION_DRAFT.md) 与 [离线 HTML](../web/gui-v2-vulkan-plan-20261021.html)。
