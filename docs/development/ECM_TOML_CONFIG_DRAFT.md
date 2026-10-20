# ECM 配置迁移 TOML：开发草稿

## 目标

将程序和 GUI 的配置文件统一为 `ecm.toml`，停止读取 `ecm.ini`，以 `config/ecm_options.json` 为键、默认值、类型、说明和 worker 覆盖规则的唯一数据源。`worktodo.txt` 自身的 `[Worker #N]` 分区属于队列格式，保持独立。

## 表结构

- `[shared]`：Stage1 与 Stage2 共用的键。
- `[stage1]`、`[stage2]`：各阶段专用键。
- `[gui]`：GUI 全局设置。
- `[workers.N]`：第 N 个 GUI worker 的名称、启动等设置。
- `[workers.N.shared]`、`[workers.N.stage1]`、`[workers.N.stage2]`：允许 worker 覆盖的计算设置。覆盖顺序为内置默认值、对应全局表、对应 worker 表、命令行。

配置参考逐键注明所属表和 worker 覆盖资格。TOML 使用带引号的字符串、布尔值、数字及 `window` 整数数组；拒绝重复键和无效值。默认配置和示例配置按分区排序，每个键保持“`键 = 默认值`、英文注释、中文注释、空一行”的格式。会改变隐式路径或环境变量处理的键暂以注释形式展示，取消注释才表示显式设置。

## 实施与验证

- `src/core/ecm_toml.h` 与生成的 `ecm_toml_schema.h` 共同校验分区、键、类型与重复项，按全局分区、worker 分区、命令行的顺序投影配置。`ecm.ini` 与 `--ini` 已退出主程序接口；Prime95 feeder 独立使用 `feeder.ini`。
- GUI 的 `TomlFile` 读取、保留并原子写回 TOML；字符串、布尔值和窗口数组按 TOML 类型编码。Stage1、CPU Stage2、CUDA Stage2 共享配置文件与 `--config` 命令行入口。配置生成器输出默认、示例、逐键参考、C++ 绑定和哈希清单。
- 默认与示例各有 118 个配置条目，逐项核对“键、英文注释、中文注释、空行”，Python `tomllib` 可解析。`python tools/gen/generate_ecm_config.py --check` 通过。
- `tools/test/test_ecm_toml.ps1` 编译并运行解析、默认值、worker 覆盖和无效输入用例；`test_worker_sections.ps1` 的 24 项通过。GUI 构建及 33 项自检、CPU Stage2 构建、CUDA Stage2 HostOnly 构建、Stage1 CUDA 构建通过。首次 Stage1 构建时，旧 CMake 路径导致临时复制本地头文件；验证后已移除。随后确认正式依赖位于 `third_party/cgbn/include`，并已修正 CMake、发布源文件清单和独立探针路径。
- 改用 `third_party/cgbn/include` 后，CMake 配置成功，发布源文件清单识别 46 个 CGBN 头文件，受影响脚本语法检查通过。新路径的 Stage1 全量重编译未完成：8 路并行时 `nvcc/cl.exe` 多次以 `0xC0000005` 退出，随后单独串行编译 `cgbn_stage1.cu` 仍以相同代码退出。旧可执行文件来自修正路径前的成功构建，不能代替新路径的最终构建验收；需排查本机 CUDA/MSVC 编译器崩溃并重跑 Stage1 构建。
- 后续将问题缩至独立的 CGBN 头文件编译：普通 CUDA 最小文件在受限沙箱中通过，加入 CGBN 头文件则稳定得到 `0xC0000005`；同一最小文件在沙箱外重复两次均通过。CMake 缓存并非该故障的根因。按原有 `third_party/cgbn/include` 配置在沙箱外运行 `build_stage1_local.ps1 -Jobs 4`，21 个 CUDA 单元全部通过并完成主机链接（并行阶段墙钟 1553.7 秒）；输出 `build_cuda_cmake/ecm_cuda.exe` 为 26,102,784 字节，21 个 cubin、0 个 PTX。新二进制运行 `test_worker_sections.ps1` 的 24 项均通过。编译日志与 `timings.json` 保存在忽略的 `build_cuda_cmake/par_nvcc/`，最小复现保存在忽略的 `tools/test/_run/toml_config/`。这是当前执行沙箱与重型 CGBN 编译的交互问题，仓库代码未增加绕过逻辑；后续在该沙箱内重编仍应使用允许编译器正常运行的执行环境。

完成后保留此草稿供用户处理。
