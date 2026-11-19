# Linux x86-64 发布构建草稿

## 构建范围

采用本机 WSL 已验证的 CUDA Stage1、OpenCL/CPU Stage1 和 libecm CPU Edwards Stage2。首个验证 profile 为 Ubuntu 24.04、CUDA 12.9、sm89 SASS 与 compute89 PTX。与 Windows 分别维护发布入口，Linux 使用 CMake 的命令数据库、GNU 依赖文件和文件内容核验。Stage1 继续由原有模块生成器生成分档单元。

## 验证记录

- 小型真实 CMake 项目通过相同时间戳的头文件改动、对象篡改、单元级重新编译和严格跳过构建检查。日志为 `identity-regression-final.log`。
- 完整构建的第二次运行复用 301 个对象，未重编或重链接。核对三个 ELF 的摘要一致。记录为 `warm-and-package.log` 和 `warm-binary-comparison.json`。
- 发布后使用 `--skip-build --skip-package` 复核，同一构建目录的 301 个对象全部通过。记录为 `verify-only.log`。
- 发布检查覆盖 125 个常规模块、36 个 PRAC 模块及最终 CUDA ELF 的 SASS/PTX。解包搬移后，两个 GPU 各通过一次 SASS 和一次强制 PTX JIT，CPU Stage1 通过同一用例。五次 Edwards Stage1→CPU Stage2 交接均找到 10007。归档摘要及运行记录见 `runtime-qualification.json`。
- 以上原始记录位于 `data/experiments/linux_release_20261119/`。本机 WSL 的 OpenCL GPU 资格仍取决于可枚举的 ICD。
- 完整 CUDA Stage2 的 Linux 资格仍涉及 POSIX 曲线进程与队列同步。单曲线探针可以独立验证算术。

正式使用流程见 [Linux x86-64 发布](../usage/BUILD.md#linux-x86-64-发布)。
