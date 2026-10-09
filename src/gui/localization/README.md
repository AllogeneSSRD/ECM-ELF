# localization/ — ecm_gui 界面文案（欢迎贡献翻译）

`ecm_gui` 的所有可见文案都在这里，**不在代码里**。改完不用重新编译：GUI 菜单
`Language -> Reload localization` 立即生效。

```
localization/
├── english.xml             # 基线：所有键必须在这里有定义（缺键会回退到这里）
├── chineseSimplified.xml   # 简体中文
└── README.md
```

## 加一种语言

1. 复制 `english.xml` → `<你的语言>.xml`（文件名 = 语言标识，例如 `german.xml`、`japanese.xml`）。
2. 把 `<Native-Langue name="English" ...>` 里的 `name` 改成该语言的**自称**（例如 `Deutsch`、`日本語`），
   `filename` 改成你的文件名。
3. 逐个把 `<Item id="..." name="这里"/>` 的 `name` 翻成目标语言。**`id` 不要改**。
4. 编码：**UTF-8**（带不带 BOM 都行，与 Notepad++ 的本地化文件一致）。
   非 UTF-8（比如 GBK）会被 GUI 明确报错，而不是显示乱码。
5. 在 GUI 的 `Language` 菜单里点一次 `Reload localization` 检查效果；
   `[GUI] language = <文件名去掉 .xml>` 会在退出时写进 `ecm.ini`。

## 规则

| 规则 | 说明 |
|---|---|
| `english.xml` 是基线 | 其它语言缺的键**自动回退英文**，所以半成品翻译也能用（不会显示空串） |
| 键名 `panel.id` | 代码里写 `loc.t("workers", "col_state")`；找不到时显示 `workers.col_state` 这种字面量，便于定位 |
| XML 转义 | `&` 写成 `&amp;`、`<` 写成 `&lt;`、`>` 写成 `&gt;`、`"` 写成 `&quot;` |
| 注释可以放心写 | `<!-- ... -->` 会被解析器忽略，可以留翻译笔记 |
| 关于字体 | C++ 侧不打包字体：界面语言需要 CJK 时，运行时加载系统字体（Windows：微软雅黑 `msyh.ttc` 等）。找不到系统字体时 GUI 自动回退英文界面（见 `docs/usage/GUI.md`） |
| 新增文案 | 先在 `english.xml` 里加 `<Item>`，再在代码里用它；漏加会显示 `panel.id` 字面量（自测会报“缺键数量”） |

## 与代码的关系

- 解析器：`src/gui/localization.cpp`（纯文本扫描，不依赖 XML 库；只认 `Panel/Item` 的 `id`/`name` 属性）。
- 自测：`ecm_gui.exe --selftest` 会检查基线可加载、中文文件可加载、无缺键、以及系统 CJK 字体是否找到。
- 界面语言存 `[GUI] language = <file stem>`；`[GUI] localization_dir` 可指向别处（默认是 exe 同级的 `localization/`）。
