# 构建与发布

工具位于 `tools/build/`。从任意目录调用 PS1，或双击同名 BAT；BAT 转交参数，无参数运行结束后暂停。默认值由 PS1 维护。

Stage1 CUDA 所需 CGBN 头文件位于 `third_party/cgbn/include`；CMake 通过 `CGBN_INCLUDE_DIR` 指向该目录，需要其他安装位置时可在配置时覆盖。

## 六组入口

独立 CPU Edwards Stage2 使用新功能入口 `tools/build/test/build_cpu_stage2.ps1`，不依赖 CUDA；依赖和输出见 [CPU Stage2](EDWARDS_CPU_STAGE2.md#构建与运行)。它尚未包含在下列 CUDA 发布包中。

| 程序 | 开发 | 本机生产 | 发布生产 |
| --- | --- | --- | --- |
| Stage1 | `build_stage1_dev` | `build_stage1_local` | `build_stage1_release` |
| Stage2 | `build_stage2_dev` | `build_stage2_local` | `build_stage2_release` |

每项均有 `.ps1` 与 `.bat`。Stage1 默认 8 路独立 CUDA TU 并行；Stage2 默认 `SplitCompile=8`，多架构依次构建，每个架构内并行。构建入口不自动运行测试。

本机/开发默认 sm89。Stage1 `-Arch 86`、Stage2 `-Arch sm_86` 可选择其他架构；本机表示单架构构建，不自动检测设备。默认输出到用途隔离的 profile 叶目录，入口打印完整路径：Stage1 为 `build_cuda_local/stage1/sm89/<key>/ecm_cuda.exe`，Stage2 为 `build_cuda_local/stage2/production/sm89/<key>/ecm_cuda_stage2.exe`。开发入口使用 `build_cuda_dev`；Stage2 dev 保留实验引擎默认值，目录为 `stage2/experimental/...`，可显式选生产引擎。发布构建使用 `build_cuda_release/stageN/cuda<版本>/sm<架构组>/<key>`；分发包仍在独立 `dist` 目录。exe、运行时 DLL 和清单保留在各叶目录根。

## 默认策略

Stage1：完整位宽/曲线族、通用 Montgomery、TPB128、分档寄存器策略；resident/PRAC 默认不编译，显式 `-EnablePrac` 才加入。`-Tiers` 可缩小开发档位。折叠域、统一寄存器上限和算术探针不是通用发布默认。

位宽唯一清单为 `include/ecm/cuda/tiers.def`。Edwards binary/NAF 使用32档容器、七组 `kernels/cuda/stage1/edwards/cgbn_edwards_instances_*.cu`；Montgomery PARAM0/2/3 各有七组 `kernels/cuda/stage1/montgomery/instances_param*_g*.cu`，均受 `-Tiers` 筛选。容器边界仍按 Edwards 的一位余量、Montgomery 的六位余量判断，具体限制见 [位宽与资源](../architecture/STAGE1.md#位宽与资源)。PRAC 显式启用时另编译 `ecm_cuda_prac` 对象目标，保留统一构建参数与发布拆分流程。

新功能构建入口 `tools/build/test/build_edwards_gpu_workers.ps1` 可用 `-Stage stage1 -Tiers '' -Jobs 8` 编译完整三曲线族。该 worker 测试入口默认仅1024；检查其他位宽前需显式选档或使用空列表编全档。

`build_edwards_gpu_stage1_probe.ps1` 默认8路并行；`-WorkerBuild <stage1构建目录>` 在源文件、命令、架构及对象摘要全部匹配时，复用该worker的Edwards对象，只编译探针和CPU参考，再链接独立验证程序。构建日志 `par_nvcc/timings.json` 区分队列实际墙钟和各TU同时运行的时间之和，后者不作为串行构建实测或加速比。

独立探针默认编全档，可用 `-Tiers 384` 构建单档；验证时执行 `edwards_gpu_stage1_probe.exe 1 <新输出目录> 384` 检查回退、上限和三种执行模式。多档受限构建使用末尾参数 `--compiled-tiers`，全档完整用例则省略最后参数。探针编译和链接均显式使用 `/MD` 与 worker 一致。短标量 CPU 参考使用 w2，边界输入仅按奇数模数和有效曲线构造筛选，避免把大素数搜索或大 CPU 字典作为算术验收的前置工作；末点、GCD、点不变量和恢复比较均保留。

Stage2：独立生产 CUDA 引擎，固定 Goldilocks PTX 后端、canonical 减法、`AddSubMask=1`、`OuterUnrollU=0`。这里“PTX 后端”是内联模算术，不表示一定携带驱动 JIT 的 compute PTX。开发引擎由独立 wrapper 选择。

## 工具链与架构

两阶段发布默认 `Archs=60,70,75,86,89,120`。

| 架构 | CUDA |
| --- | --- |
| sm60、sm70 | 12.6 |
| sm75、sm86、sm89、sm120 | 13.3 |

需要安装所选 CUDA、相容的 Visual Studio x64 C++ 工具集、CMake、PowerShell 5.1+ 和 GMP。脚本按 CUDA 头文件声明的 MSVC 范围选择受支持工具集，不默认绕过版本检查。通过 `-LegacyCudaRoot`、`-CudaRoot`、`-VcVars` 指定安装入口。

Stage1 分工具链构建多架构程序，再用对应 nvprune/cuobjdump 拆分；默认 sm75 包保留 compute75 PTX，其余包携带各自 SASS。Stage2 逐架构直接构建生产程序。不同 CUDA/MSVC 的缓存目录分离。构建成功不等于每个架构均已完成运行验收。

## GMP 与打包

本机/开发默认使用本地 Zen3 GMP；发布默认使用 vcpkg 通用 x64 GMP，缺失时明确报错，不自动回退到 Zen3。自定义 `-Gmp <prefix>` 必须提供匹配的头文件、导入库、DLL 和许可证。Zen3 库使用 BMI2/ADX 等指令，其 CPU 兼容范围由实际构建决定。

包内只分发程序、GMP DLL、共享 TOML 模板、空队列、`ECM_TOML_REFERENCE.md` 与许可证。不包含 `*manifest.json`、`DEV_*`、README 或其他项目文档。重新打包保留已有 TOML/队列；ZIP 从干净暂存目录生成，不收录 save、日志和运行数据。

Stage1 输出 `dist/cuda/`，Stage2 多架构输出 `dist/cuda-stage2/`；每架构有目录、ZIP 和 SHA256。构建/拆分/打包清单留在构建目录，包括源码、依赖、编译器、架构和程序身份。

## 复用构建

Stage1 默认启用增量复用，按编译命令、响应文件、递归依赖内容与对象 SHA256 判断 CUDA 对象是否可复用；保留时间戳也不能绕过内容校验。`-Incremental` 保留为兼容参数，缺少当前依赖记录的旧对象会重编一次。`-FullRebuild` 删除 CUDA 对象后重编；内部并行编译器的 `-Rebuild` 强制编译选中的单元。主机 CXX 文件仍由 CMake/NMake 管理。`-HostOnly` 要求已配置的同 profile 目录，在配置/主机编译前验证所有 CUDA 对象及原生构建依赖时间戳，任一失效即拒绝；不用于首次配置。`-HostOnly` 与强制重建互斥。旧 `-RelinkOnly` 绕过入口已关闭，使用它会明确报错；主机修改使用正常增量或 `-HostOnly`。

Stage2 默认逐编译单元复用，整体源码/构建签名变化不会直接触发整组重编；每个失效单元打印重编原因。主机 `.cpp` 暂时仍通过 nvcc 编译，普通 CXX 编译器迁移另行实施。链接命令、GMP 导入库、对象摘要和程序摘要全部匹配时跳过链接；旧清单首次建立链接输入记录可能需要一次链接。

Stage2 的 `-HostOnly` 在任何编译前核对完整清单，只复用身份一致的 CUDA 对象，主机部分仍按需编译；架构、后端、工具链、GMP 头、CUDA 依赖或任一 CUDA 对象改变时拒绝。`-HostOnly` 与强制重建的 `-Rebuild` 互斥。普通复用和发布 `-SkipBuild` 同时核对程序、编译单元、依赖和对象；旧 manifest 缺少身份记录时须正常重建。保留构建目录中的 `.ecm.d`、`.ecm-state.json`、源清单和 manifest；不能修改清单或替换 DLL 来绕过核对。

## 用途与 profile 目录

`profile.json` 保存完整 SHA256 及字段，目录使用前 16 位 key。字段包含 role、stage、engine、架构、编译/链接策略、工具链和 GMP 内容身份；Stage1 还包含 CMake 身份、tiers、PRAC 和额外选项。源码 hash、HEAD、程序版本不进入目录 key，源码变化由原有逐 TU 身份处理。不同 profile 使用不同默认叶目录；显式指定的叶目录不匹配时拒绝，不改写 marker。

本机/开发 Stage1 的 `-BuildDir`、Stage2 的 `-Build` 是最终叶目录，不额外拼层；Stage2 内部 `-BuildRoot` 才表示根。Stage1 多工具链 release 的公开 `-BuildDir`、Stage2 多架构 release 的 `-Build` 保持根语义；Stage2 单架构 release 的 `-Build` 为最终叶目录。不能把 local/dev/release 目录互作输出，也不能让包目录与 build 目录相同或互相嵌套。

旧目录不自动移动或复制。显式指定旧目录时，仅对已有正式清单、工具链/GMP/架构/策略相符的目录原位核对并建立 marker；未知非空目录拒绝。`-SkipBuild` 要求已有正确 marker，仍需通过原有构建和打包身份检查。路径消费者需使用入口打印的精确叶目录或显式配置，不选择扫描结果中的“最新”程序。

CPU-only 检查入口为 `test_cuda_build_profile.ps1`、`test_cuda_build_profile_paths.py`。发布静态检查 `test_cuda_p5_release.py` 对默认布局有多个相容 profile 时拒绝歧义，可用 `--build-leaves` JSON 显式指定各 stage/arch 的最终叶目录。该路径检查不等于 release 构建或多架构镜像资格。

## CUDA 源清单与依赖身份

翻译单元唯一清单为 `tools/build/internal/cuda_sources.cmake`：分别维护 Stage1 共用源、CUDA 源、可选 PRAC、Stage2 驱动源、生产主机引擎、算术、NTT 和开发 wrapper。Stage2 生产构建含25个单元：15个 `STAGE2_ENGINE` 主机 CPP、算术 CU、NTT CU、四个 NTT 主机 CPP 和四个驱动 CPP。CMake 直接使用，PowerShell 从该文件导出构建目录内的 `cuda_sources.json`。生产源码闭包拒绝包含 `tools/bench/`、`tools/test/`、`tools/experiments/` 和 `.refactor/`，包括编译器补齐的依赖；开发 wrapper 使用单独源组。实验树、BSGS、独立 NTT、cuFFT 和 giant 变体另有显式实验源组，不加入生产目标。模块职责以 [Stage2 模块边界](../architecture/STAGE2.md#cuda-模块边界) 为准。

CMake 同时导出 `cuda_target_graph.json`，收集目标及其链接的项目对象库/静态库；条件目标按保守全集收集，导入库由工具链与依赖清单另行记录。对象枚举和 CUDA 完全重编按该目标闭包执行。`cuda_build_state.ps1` 扫描本地递归包含，包括 `.inc/.inl/.def`，并用 NVCC 依赖文件补齐宏包含及 SDK 头。每个对象的 `.ecm-state.json` 记录命令、依赖内容指纹与对象摘要；摘要编码在 Windows PowerShell 5.1 和 PowerShell 7 间一致。

源身份验证入口为 `tools/build/test/build_cuda_p0.ps1`，用 `-Stage stage1/stage2/development` 选择构建；CPU 故障注入为 `tools/test/test_cuda_build_identity.ps1`，短曲线双版本对照为 `tools/test/test_cuda_p0_runtime.py`。`tools/build/test/build_cuda_p1.ps1 -Stage tiers` 编译清单的16种筛选/几何配置，`-Stage stage1/stage2` 构建迁移后的程序。结构与冻结算术正文校验使用 `test_cuda_p1_structure.py`；实际注册、跨组曲线与可选运行时分别使用 `test_cuda_p1_registry.py`、`test_cuda_p1_curves.py`、`test_cuda_p1_prac.py`。这些入口不替代完整位宽、多架构或性能验收，证据写入 `data/experiments/`。

`tools/build/dev/` 为开发和 GUI 入口，`test/` 为探针与参考实现构建，`internal/` 为配置检查、并行编译、拆分和打包辅助。详细参数入口见 [构建工具](../../tools/build/README.md)；配置生成合同见 [配置维护](../architecture/CONFIGURATION.md)。

P2 结构验证使用 `test_cuda_p2_structure.py` 对照冻结 Git 基线的设备算术正文并检查模块依赖方向。`build_cuda_p2_arithmetic_probe.ps1 -WorkerBuild <生产目录>` 将独立 GMP 检查器链接到经过身份验证的生产算术对象；`build_ntt_graph_probe.ps1 -WorkerBuild <生产目录>` 同样复用真实 NTT 对象。其他使用生产 NTT 的探针通过 `internal/cuda_ntt_probe.ps1` 消费同一源清单，所有单元保持探针指定的 legacy/per-thread 默认流设置；不再通过实现头或截取生产源码构建。

当前主机清理门禁为 `test_cuda_p4_cleanup.py`，对照 P3 基线核对32个生产模块，只允许有调用点、恒值和作用域证明的有限删除。`test_cuda_p4_isolation.py` 核对实验迁移正文、入口默认值及包含边界，`test_cuda_p4_probe_contracts.py` 核对有限探针接口及独立 oracle。P3 的完整搬迁等价脚本是该阶段的冻结门禁；当前删除后的代码使用 P4 门禁。

`test_cuda_p3_runtime.py` 继续消费冻结旧程序、独立验证的存档和新程序，比较完整叶指纹、稳定结果字段、计划查询、分配/预算回退及各日志级别的损坏拒绝。资源测试通过 `stage2_native_source.py` 读取实际模块的声明和方法，不使用另一份手写资源实现。依赖关系收缩后，构建可能因旧编译依赖记录拒绝收缩后的依赖；保留该记录，使用新构建目录完整编译，再验证复用。

实验完整模块由 `internal/cuda_experiment_probe.ps1` 构建，记录递归源码闭包、命令、对象和程序摘要；入口不再截取 `.cu` 正文。GMP/测试夹具仍位于 `tools/test/`。pack 探针以原 per-thread ABI 独立编译同一生产模块；常规生产对象为 legacy，复用时会拒绝该差异。giant 可通过 `-WorkerBuild` 链接验证过的生产算术对象，其旋转及未注册容量单独编译。host-wait 探针通过显式适配编译真实 Edwards 运行时和设备模块。`build_stage2_d_model_probe.ps1` 默认保留旧实验表，`-Profiles frozen-production` 显式测试冻结的生产候选表。`build_cufft_probe.ps1` 从实验源组构建完整 cuFFT 候选并记录源码、命令、库和对象身份。冻结源码审计由 `internal/cuda_frozen_ntt_probe.ps1` 保留旧快照身份及其完整实现，不能用历史实验脚本的路径推断当前生产闭包。
