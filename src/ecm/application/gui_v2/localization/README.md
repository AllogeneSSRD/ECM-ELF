# GUI v2 语言资源

运行时从可执行文件旁的 `localization/v2/` 读取 UTF-8 XML，与当前工作目录无关，不把译文编译进程序。菜单“语言 → 重新载入语言文件”用于应用外部修改，无需重新编译或重启。

- `english.xml`：英文基线，包含全部界面键。
- `chineseSimplified.xml`：简体中文；缺项或空值回退到英文基线。
- `options.english.xml`、`options.chineseSimplified.xml`：计算选项说明，由 `config/ecm_options.json` 经 `tools/gen/generate_gui_v2_state.py` 生成，不手改。

格式沿用原 GUI 的 `EcmGui / Native-Langue / Panel / Item`，每个 `Item` 使用 `id` 和 `name` 属性。面板名为 `ui`；保留键和 `###` 控件标识。换行写为 `&#10;`，引号、&、< 使用 XML 实体。翻译中的 `%s`、`%llu`、`%.1f` 等格式参数必须与英文保持类型和顺序一致，`%%` 表示格式串内的百分号。

重新载入会检查重复键及格式参数。失败时保留上次完整词典并显示错误；中文文件不可用时回退英文并记录原因。英文基线不可用时显示键名及诊断，不使用内嵌译文。中文字体不可用仍按现有规则切到英文。

当前语言选择仍为 `en` / `zh`；尚未增加任意语言枚举。计算程序原始输出保留原文。三套装甲名称、协议/配置键和技术单位不属于译文。

解析适配见 `ui/localization_xml.cpp`，基于旧版解析器的相同格式，使用 Unicode 文件路径；旧 GUI 的源码和语言目录保持独立。构建复制本目录，候选包同时携带 XML。
