# GUI v2 语言资源

运行时从可执行文件旁的 `localization/v2/` 读取 UTF-8 XML，与当前工作目录无关，不把译文编译进程序。菜单“语言 → 重新载入语言文件”用于应用外部修改，无需重新编译或重启。

- `english.xml`：英文基线，包含全部界面键。
- `chineseSimplified.xml`：随包提供的简体中文；缺项或空值回退到英文基线。
- `options.english.xml`、`options.chineseSimplified.xml`：计算选项说明，由 `config/ecm_options.json` 经 `tools/gen/generate_gui_v2_state.py` 生成，不手改。

格式沿用原 GUI 的 `EcmGui / Native-Langue / Panel / Item`，每个 `Item` 使用 `id` 和 `name` 属性。面板名为 `ui`；默认键使用可读英文命名，例如`action.start`、`worker.progress`、`message.workflow_opened`，没有哈希后缀或数字消息编号。键标识含义，`name`是可编辑译文；翻译时保留键和译文中的`###`控件标识。换行写为`&#10;`，引号、&、<使用XML实体。翻译中的`%s`、`%llu`、`%.1f`等格式参数必须与英文保持类型和顺序一致，`%%`表示格式串内的百分号。

设置中的下拉列表与顶栏语言菜单扫描本目录的实际界面XML，显示文件名时省略末尾`.xml`后缀，不使用固定中英文开关。例如`english.xml`显示为`english`，`custom.name.XML`显示为`custom.name`；文件、选择标识和侧车仍保留完整文件名。复制现有界面文件为其他文件名即可创建新词典；重新打开列表会扫描新增和删除文件。`options.*.xml`不作为界面语言列出；其他非XML文件也不列出。文件名可包含中文，但必须是本目录内的单一文件名，不能使用绝对路径或父目录跳转。

加载会检查重复键、界面项及格式参数，失败时保留上次完整词典与选择并显示错误。译文缺项或空值使用英文基线；所选文件缺失时警告并使用英文缺项回退，启动所选文件损坏时也回退英文。英文基线不可用时显示键名及诊断，不使用内嵌译文。非英文文件使用现有CJK字体路径；字体不可用仍按现有规则切到英文并提示，不分发系统字体。

已创建的旧词典可继续使用原哈希键和`message_数字`键，加载时通过别名转换，不重写用户文件；新词典优先使用默认文件中的可读键。同一个文件同时定义旧键及其对应新键会被当作重复项拒绝，格式校验也在转换后执行。

开发仓库的默认词典、别名与生成头由`tools/gen/generate_gui_v2_localization.py`校验：修改源码默认XML后运行该脚本，使用`--check`检查中英键一致、语义命名、引用及生成物。旧键映射只在`config/gui_v2_localization_aliases.json`维护；它只保存键名，译文始终来自外部XML。直接修改已安装程序旁的语言文件不需要运行生成器。

实际选择保存为独立侧车`ui.language_file`，重启恢复；默认空值按旧`ui.language`的`en` / `zh`迁移。旧字段继续作为字体兼容提示，不是可选文件枚举。命令行`--language en|zh`显式选择随包的两个默认文件，覆盖侧车的本次初始选择。

每个界面文件可另配`options.<界面文件名>`，例如`custom.xml`配套`options.custom.xml`；没有配套文件时计算说明使用英文基线。随包两份`options.*.xml`仍由schema生成。计算程序原始输出保留原文。三套装甲名称、协议/配置键和技术单位不属于译文。

解析适配见 `ui/localization_xml.cpp`，基于旧版解析器的相同格式，使用 Unicode 文件路径；旧 GUI 的源码和语言目录保持独立。构建复制本目录，候选包同时携带 XML。
