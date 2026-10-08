# CUDA Stage1 发布包

发布包包含对应 GPU 架构的 `ecm_cuda.exe`、`gmp-10.dll`、共用 INI 模板、空 `worktodo.txt`、`ECM_INI_REFERENCE.md` 及许可证。构建/拆分/打包清单保留在构建目录，不分发 README 或开发文档。

发布默认使用 vcpkg 通用 x64 GMP，附带 `GMP-COPYRIGHT.txt`。本机 Zen3 GMP 含 BMI2/ADX 等扩展指令，不作为默认发布依赖；显式自定义 GMP 时，以该库真实 CPU 指令基线为准。

## 使用

1. 解压到独立目录。
2. 编辑 `ecm.ini`，设置 GPU、曲线参数、存档目录等。全部配置键见 `ECM_INI_REFERENCE.md`。
3. 把 Stage1 任务放入 `worktodo.txt`，启动 `ecm_cuda.exe`。Stage2 任务使用独立的 `stage2_worktodo.txt`，由 `ecm_cuda_stage2.exe` 消费。

Stage1 存档、检查点和日志位于用户配置的目录。该包的 `ecm.ini` 与独立 Stage2 程序共用键名；两个程序的队列保持独立。默认运行行为以 INI 模板和配置说明为准。

## 架构与构建

默认发布编译包含 `sm_60,70,75,86,89,120`。60/70 使用本机 CUDA 12.6，75/86/89/120 使用 CUDA 13.3，各组使用对应支持的 MSVC；完成后拆分成各架构独立包。指定的 `sm_75` 包保留 compute_75 PTX，可由驱动在更新架构上 JIT；其他包包含各自 SASS。默认构建全部位宽和曲线族，使用通用 Montgomery / TPB128 / 分档寄存器策略。实验 resident/PRAC 默认不编译，需显式 `-EnablePrac`；未编入时请求对应算法会报错。`-Tiers`、PRAC 开关及 CUDA/MSVC 身份记录在各组构建目录的 `stage1_build_manifest.json` 中。

需要相应的 NVIDIA 驱动及 Microsoft Visual C++ x64 运行库。架构清单由构建工具链支持范围决定；构建成功不等于已完成所有架构的运行验收。

仓库一键入口为 `tools/build/build_stage1_release.bat` 或同名 PS1。默认八路并行编译 CUDA TU，随后自动拆分、打包及生成 ZIP 和 `.sha256`。

构建目录分为 `build_cuda_release/cuda12_6` 和 `build_cuda_release/cuda13_3`，避免复用不同编译器的缓存。两组均输出至 `dist/cuda`；nvprune 和 cuobjdump 使用对应组的 CUDA 版本。`-LegacyCudaRoot` 和 `-CudaRoot` 可覆盖默认安装目录。

重新打包会更新程序、模板和说明，但保留包目录已有的 `ecm.ini` 与 `worktodo.txt`。分发 ZIP 总是使用模板配置和空队列，不包含本机运行数据。

## 清单

- 构建目录 `stage1_build_manifest.json`：编译设置、源文件与多架构程序哈希。
- 构建目录 `split_manifest.json`：原始构建身份与各架构程序哈希。
- 构建目录 `package_sm<arch>_manifest.json`：各包静态文件哈希。`tests_run=false` 表示发布脚本未运行测试。
- ZIP 同目录 `.sha256`：归档文件校验值。

Stage1 多架构拆分仅改变设备代码打包。运行时只有在构建包含 PRAC 时才能选择该算法；批量、TPI 等策略仍由当前实现及用户配置决定。
