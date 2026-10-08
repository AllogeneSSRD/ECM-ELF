# ECM 配置统一维护

## 入口

唯一选项定义为 [config/ecm_options.json](../config/ecm_options.json)。
Stage1、Stage2 和 GUI 的配置默认值、赋值规则、静态限制、别名、模板、符号语法和用户说明均由它派生。

```powershell
# 修改定义后生成；Python 3.10+，仅标准库
python tools/gen/generate_ecm_config.py

# 只检查，不改文件；可直接作为 CI 命令
python tools/gen/generate_ecm_config.py --check

# 普通构建使用原生 CMake 检查，不需要 Python
cmake -P tools/build/internal/check_ecm_config.cmake
```

生成物提交到 Git，程序运行时不读取 JSON、不依赖 Python。生成文件头均有标记，应修改定义后重新生成。

## 生成物

- `src/core/generated/ecm_config_generated.h`：四类配置结构、默认值常量、静态限制、类型赋值分发表和旧键转换。
- `src/core/generated/ecm_ini_template.h`：首次启动 INI 的嵌入内容。
- `config/ecm.ini.default`：首次启动模板的可读副本。
- `config/ecm.ini.example`：发布模板。
- `docs/ECM_INI_REFERENCE.md`：符号配置行与逐项用户说明。
- `config/ecm_config.generated.json`：定义、生成器和生成物的原始 SHA256 清单。

`.gitattributes` 将以上带哈希文件固定为 LF，避免 Windows checkout 自动转换导致误判。

## 选项定义

每条 option 包含 `owner/key/field/type/default/parser/domain/fx/description`。`domain` 是符号取值范围，`fx` 是内部用途标签；`description` 为面向用户的双语说明列表，先英文、后中文，两种语言分别占一个元素。
可增加 `minimum/maximum/unit/empty/zero/implicit/effective_default/present/rules`。

`description` 按 `.refactor/undoc.txt`、`.refactor/undoc zh-Hans.txt` 的说明方式编写：先说明用户能控制什么，再解释各值的实际效果、空值或 0 的行为，以及适用条件或调整代价。避免只写内部模块名或缩写。默认值和范围继续从已有字段生成，不在说明中另建一份键表。

每个正式键与兼容别名必须有非空说明，生成器拒绝缺失说明的条目。说明字符串可用 JSON 的 `\n` 显式换行；生成器保留换行，并按每行 100 个显示列自动折行，中文全角字符计 2 列。英文尽量在词间换行，中文可在字符间换行；语言边界始终另起一行。

参考文档为“配置行 + 英文说明 + 中文说明”，Markdown 用 `<br>` 保留每个物理换行；INI 模板将每行分别生成为 `#` 注释。嵌入 C++ 的模板按 UTF-8 字节生成 ASCII 字符串字面量，避免编译器本地代码页改变中文注释。选项语义或默认值变更时同时维护两种语言。

```json
"description": [
  "English description.\nOptional explicit second line.",
  "中文说明。\n可选的显式第二行。"
]
```

- `default`：C++ 配置结构的初始化值。
- `implicit`：缺省时的派生策略，例如继承设备、从 profile 取时间；此时内部哨兵不表示有效的显式输入。
- `effective_default`：派生的文件名或程序名，由路径接线代码使用生成的常量。
- `present`：生成“用户显式配置过”标记，区分空值和未配置。
- `release_value/first_run_value`：明确记录发布或首次启动模板的有意覆盖；例如 `gpu_param` 的内置 3 与模板 0。
- `template_comment`：生成注释示例，避免把继承哨兵或占位符写成有效配置。
- `aliases`：输入键别名及其转换规则，不重复定义默认值。`fallback_only` 表示正式键存在时忽略该兼容键。

模板中的 worker 日志等动态默认键保持注释，避免显式文件名抑制 worker 后缀。
新增普通选项时修改定义即可生成字段、解析、模板和说明；实现新的业务功能仍需消费该字段。

## 运行时结构

[ecm_ini.h](../src/core/ecm_ini.h) 提供保留原行的统一词法读取、worker 段识别、覆盖合并以及通用类型转换。
`ecm_worktodo_parse_worker_header()` 也委托同一段头识别函数，避免队列与 INI 的 worker 语法漂移。

- Stage1 使用生成的 `Stage1Values/stage1_bindings`，保留旧数值转换和未知键忽略规则。
- Stage2 使用生成的 `Stage2Values/stage2_bindings`，INI 与 CLI 共用精确十进制/科学记数法整数转换。
- GUI 使用同一原行结构，保留注释、顺序、未知键和原子写回；GUI/worker 设置通过生成的赋值规则读取。
- GUI 的实际业务字段和 Stage2 CLI 的成本默认、文件名、预算上限均引用生成常量。

动态校验继续位于消费端：实际设备是否存在、D 的引擎形状、路径冲突、save 与任务匹配、Auto B2 标定资格等。
它们依赖运行环境或算法不变量，不应复制成纯静态 JSON 范围。

## 兼容规则与修正

保持 Stage1 键名大小写敏感、Stage2 专用键不敏感。保留旧别名和调试开关优先级。
CLI 仍只将 `[Worker #N]` 识别为覆盖范围，普通标题不退出该范围；GUI 仍按真实分区读写。
共享模板用注释分组，公共键放在所有分区之前。

同层重复键统一为最后一次出现，worker 层总是覆盖全局层；修复了旧 Stage1 全局重复键可能盖掉 worker 覆盖的问题。
统一读取器接受 `#` 和 `;` 整行注释。非法 GUI 整数恢复默认或夹到定义范围，非有限字号恢复 auto。
GUI 的空值、退出策略、布局版本和文件写回仍由对应界面逻辑消费。

## 构建约束

CMake 的配置阶段及 `ecm_config_check` 构建依赖检查全部哈希；显式构建 Stage1、Stage2、GUI 同样需要通过。
独立 Stage2 构建脚本、发布打包及并行 CUDA 编译入口也先检查。
定义、生成器或生成物任一修改但未重新生成，会报出具体文件并停止。

Stage2 构建清单还记录定义、生成器、原生检查脚本、生成头和生成清单的哈希，保证包内说明可追溯到构建输入。
只有主机端配置代码变化时可以使用已有 `-HostOnly` 机制复用未改变的 CUDA 对象。

本轮构建记录见 [Stage2 发布说明](ECM_CUDA_STAGE2_RELEASE.md)。本次未运行功能或 GPU 回归测试。
