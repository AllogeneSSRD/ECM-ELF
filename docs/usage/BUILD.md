# 构建与发布

工具位于 `tools/build/`。从任意目录调用 PS1，或双击 BAT；BAT 转交参数，无参数运行结束后暂停。默认值由 PS1 维护；两个按 CUDA 版本命名的 release BAT 固定传入对应 profile 与 `-Jobs 16`。

Stage1 CUDA 所需 CGBN 头文件位于 `third_party/cgbn/include`；CMake 通过 `CGBN_INCLUDE_DIR` 指向该目录，需要其他安装位置时可在配置时覆盖。

## 构建入口

独立 CPU Edwards Stage2 使用新功能入口 `tools/build/test/build_cpu_stage2.ps1`，不依赖 CUDA；依赖和输出见 [CPU Stage2](EDWARDS_CPU_STAGE2.md#构建与运行)。它尚未包含在下列 CUDA 发布包中。

| 程序 | 开发 | 本机生产 | 发布生产 |
| --- | --- | --- | --- |
| Stage1 | `build_stage1_dev` | `build_stage1_local` | `build_stage1_release` |
| Stage2 | `build_stage2_dev` | `build_stage2_local` | `build_stage2_release` |

每项均有 `.ps1` 与 `.bat`。完整发布可分别使用 `tools/build/build_release_cu11.8.bat`、`build_release_cu13.3.bat`；`build_release.ps1/.bat` 保留双包总入口。两项阶段 release 入口只重建并验证相应阶段，不生成完整包。Stage1 本机/开发默认8路独立 CUDA TU 并行；Stage2 local 默认 `SplitCompile=8`。发布两阶段默认 `Jobs=16`，Stage2 cu13.3 聚合模式使用 `SplitCompile=8`，cu11.8及隔离备选使用1。构建入口不自动运行数学、GPU 或性能测试。

本机/开发默认 sm89。Stage1 `-Arch 86`、Stage2 `-Arch sm_86` 可选择其他架构；本机表示单架构构建，不自动检测设备。local 的 BAT/PowerShell 入口使用固定短目录：Stage1 为 `build_stage1_local/ecm_cuda.exe`，Stage2 为 `build_stage2_local/ecm_cuda_stage2.exe`，GUI 为 `build_gui_local/ecm_gui.exe`。开发入口使用 `build_cuda_dev`；Stage2 dev 保留实验引擎默认值，目录为 `stage2/experimental/...`，可显式选生产引擎。发布构建使用 `build_cuda_release/stageN/cuda<版本>/sm<架构组>/<key>`；完整分发包位于独立 `release/ecm-elf-<版本>-cu<CUDA版本>/` 目录。exe、运行时 DLL 和清单保留在各叶目录根。

## 默认策略

GUI v2 使用独立入口`tools/build/build_gui_v2.bat`，默认生成`build_gui_v2_local`。常规目标包含群阶分析助手及匹配的GMP DLL，无需CUDA；本机候选继续使用本地Zen3/BMI2依赖，未声明通用x64发布资格。`tools/build/package_gui_v2.ps1`是独立GUI候选打包入口，会附带对应GMP源码/许可；与下文计算worker的release策略分开，详见[GUI v2](GUI_V2.md#构建与启动)。

Stage1：完整位宽/曲线族、通用 Montgomery、TPB128、分档寄存器策略；resident/PRAC 默认不编译，显式 `-EnablePrac` 才加入。`-Tiers` 可缩小开发档位。折叠域、统一寄存器上限和算术探针不是通用发布默认。

Stage1 local/dev 默认 `-KernelLayout isolated`，与 release 共用按曲线/param/容器生成的独立编译单元，避免多个大位宽挤在同一组文件形成编译长尾；每个内核及其设备辅助函数仍一起编译。`-KernelLayout grouped` 保留分组备选。local 仍为单架构、压缩、无PTX，不改成发布镜像策略。布局进入构建身份，已有分组缓存不能当成隔离缓存复用；首次切换可指定新的 `-BuildDir build_stage1_local_isolated`，旧目录保留。

位宽唯一清单为 `src/ecm/backend/cuda/dispatch/tiers.def`。Edwards binary/NAF 使用32档容器、七组 `src/ecm/backend/cuda/stage1/edwards/instances/cgbn_edwards_instances_*.cu`；Montgomery PARAM0/2/3 各有七组 `src/ecm/backend/cuda/stage1/montgomery/instances/instances_param*_g*.cu`，均受 `-Tiers` 筛选。容器边界仍按 Edwards 的一位余量、Montgomery 的六位余量判断，具体限制见 [位宽与资源](../architecture/STAGE1.md#位宽与资源)。PRAC 显式启用时另编译 `ecm_cuda_prac` 对象目标；默认完整发布不启用PRAC。

新功能构建入口 `tools/build/test/build_edwards_gpu_workers.ps1` 可用 `-Stage stage1 -Tiers '' -Jobs 8` 编译完整三曲线族。该 worker 测试入口默认仅1024；检查其他位宽前需显式选档或使用空列表编全档。

`build_edwards_gpu_stage1_probe.ps1` 默认8路并行；`-WorkerBuild <stage1构建目录>` 在源文件、命令、架构及对象摘要全部匹配时，复用该worker的Edwards对象，只编译探针和CPU参考，再链接独立验证程序。构建日志 `par_nvcc/timings.json` 区分队列实际墙钟和各TU同时运行的时间之和，后者不作为串行构建实测或加速比。

独立探针默认编全档，可用 `-Tiers 384` 构建单档；验证时执行 `edwards_gpu_stage1_probe.exe 1 <新输出目录> 384` 检查回退、上限和三种执行模式。多档受限构建使用末尾参数 `--compiled-tiers`，全档完整用例则省略最后参数。探针编译和链接均显式使用 `/MD` 与 worker 一致。短标量 CPU 参考使用 w2，边界输入仅按奇数模数和有效曲线构造筛选，避免把大素数搜索或大 CPU 字典作为算术验收的前置工作；末点、GCD、点不变量和恢复比较均保留。

Stage2：独立生产 CUDA 引擎，固定 Goldilocks PTX 后端、canonical 减法、`AddSubMask=1`、`OuterUnrollU=0`。这里“PTX 后端”是内联模算术，不表示一定携带驱动 JIT 的 compute PTX。开发引擎由独立 wrapper 选择。

Stage2 local/release 默认 `-KernelLayout grouped`：设备实现使用算术和NTT两个聚合CU，支持的工具链通过 `SplitCompile` 在单次nvcc内并行优化。完整发布使用 `-Stage2KernelLayout isolated`、阶段发布使用 `-KernelLayout isolated` 可启用176模块隔离备选；该备选仅适用于生产引擎。聚合与隔离使用不同profile和清单，`-SkipBuild` 不接受布局混用。CUDA11.8没有 `--split-compile`，旧包固定1，不传入不支持的选项。

## 工具链与架构

完整发布默认生成两个 CUDA profile；PTX 与 SASS 独立指定，公开 release 入口不再接收旧 `Archs/PtxArch` 参数。

| 分发目录 | CUDA | 唯一 PTX | 唯一 SASS |
| --- | --- | --- | --- |
| ecm-elf-&lt;版本&gt;-cu11.8 | 11.8 | compute_50 | sm_60 |
| ecm-elf-&lt;版本&gt;-cu13.3 | 13.3 | compute_75 | sm_80 |

需要安装所选 CUDA、相容的 Visual Studio x64 C++ 工具集、Git、CMake、Python、PowerShell 5.1+ 和 GMP。完整发布用 `-Cuda118Root`、`-Cuda133Root`、`-VcVars`、`-Gmp` 覆盖安装入口。脚本在构建前核对所有请求工具链的真实版本、虚拟/原生目标及 MSVC 支持范围；缺少任一默认工具链即失败。local/dev 保留各自 `-CudaRoot` 参数。

两阶段直接生成对应 profile 的双目标镜像，不经过六架构裁剪。`tools/build/internal/release_profiles.ps1` 是发布目标的唯一配置。不同 CUDA/MSVC/profile 的缓存目录分离；最终 EXE 的 cuobjdump 镜像检查和逐模块 PTX 清单留在构建目录。构建、兼容 SASS、强制 PTX JIT 和目标实体 GPU 运行是不同的验证条件。

Windows 旧包面向 Maxwell/Pascal：sm50/52 需要 PTX50 JIT；sm61 可使用 sm60 同大版本相容 SASS。452.39 是 CUDA11.x 小版本兼容条件，不能当成11.8 PTX 的最低驱动；522.06是11.8对应的Windows驱动，作为候选测试基线。实际旧卡、旧驱动尚须分别验证，显式加载不能修复不支持的 PTX ISA。[NVIDIA 11.8发布说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html)、[小版本兼容限制](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

## 完整发布与按需加载

release入口先自动同步业务配置和GUI v2派生文件，再解析依赖和构建组件；任一生成步骤失败即停止。`-SkipBuild` 仅检查生成物，不自动更新它们；过期时需先正常构建。无内容变化的生成物不改时间戳。

运行 `tools\build\build_release_cu11.8.bat` 或 `build_release_cu13.3.bat`，分别构建对应 CUDA 的完整包；两者均传入 `-Jobs 16`，并只预检所选 CUDA 工具链。BAT 可继续传入 `-OutDir`、`-SkipBuild` 等参数；需要调整 profile 或并行数时，直接使用 `build_release.ps1 -Profile cu11.8 -Jobs <数量>`。`build_release.ps1/.bat` 默认构建 GUI v2、共享 OpenCL worker 和两套 CUDA Stage1/Stage2，默认并行数也是16。

完整包生成到 `release/ecm-elf-<主程序版本>-cu11.8/`、`release/ecm-elf-<主程序版本>-cu13.3/`，ZIP/SHA256 同层存放，ZIP 名为 `ecm-elf-<主程序版本>-cu<CUDA版本>-win-x64.zip`。ZIP 内的唯一顶层目录保留完整包名，例如 `ecm-elf-0.52.1-cu11.8/`。两个 CUDA 包可先后发布到同一个 `-OutDir`。

`-SkipBuild` 仍核对源码、profile、工具链、依赖、对象和程序身份。`-SkipPackage` 只构建/验证。`-BuildRoot` 默认 `build_cuda_release`；`-OutDir` 默认 `release`，必须与构建目录分离。每个阶段的精确叶目录记录在 `<BuildRoot>/components/cu<版本>.stageN.json`，不按时间扫描“最新程序”。所选包的目录、ZIP 或 SHA256 已存在时均拒绝覆盖，重打包使用新的 `-OutDir`，以保留用户配置和队列。

每个 CUDA 目录内同层包含 OpenCL `ecm.exe`、`ecm_cuda.exe`、`ecm_cuda_stage2.exe`、`ecm_gui_v2.exe`、群阶/诊断助手、唯一通用 `gmp-10.dll`、TOML模板、空队列、语言资源、基础 ELY 四个资源/声明文件、PARI诊断依赖及许可/对应源码。OpenCL附带 `opencl_resources/`、OpenSSL运行库及许可；两套CUDA包共用相同OpenCL产物，OpenCL内核统一由驱动JIT，不按CUDA版本区分。GUI 使用独立版本号；版本文件夹取计算主程序版本。完整包不附原 GUI 或额外宠物皮肤。构建/测试 manifest、验证缓存、原始日志和计时不进入分发目录。

完整入口通过 `build_opencl_local.ps1 -BuildRole release` 在 `<BuildRoot>/opencl-release` 构建独立发布用途的OpenCL worker，使用通用vcpkg GMP并关闭CUDA编译器。可用 `-OpenClRoot`、`-OpenSslRoot` 覆盖头文件/导入库及OpenSSL安装位置。构建前后核对保守源码集合、工具链、依赖和资源身份；`-SkipBuild` 只核对已有产物，不为缺失组件补建。发布程序默认从自身目录查找OpenCL资源，显式 `--kernel-root` 仍优先。OpenCL加载器由用户安装的显卡驱动提供，包内不附厂商驱动。

OpenCL入口输出依赖/编译器、源输入快照、CMake配置、编译链接、依赖部署及最终核对阶段；原生构建输出实时显示，并保存在叶目录的 `build.log`。Stage1在CMake输出设备架构后报告配置耗时和源码扫描，再逐批汇报CUDA复用预检与构建后复查的进度。`par_nvcc/timings.json` 单独记录 `reuse_preflight_seconds` 和 `reuse_postcheck_seconds`；这些是内容身份检查耗时，不算入nvcc编译时间。无须重编CUDA时仍执行内容核验。

cuobjdump提取结果保存在 `<BuildRoot>/components/cuobjdump-cache/`，按对象或EXE内容SHA256、工具路径/摘要以及验证脚本身份区分。未变输入复用提取结果；对象、EXE、工具或验证脚本变化时重新检查相应输入。缓存损坏/缺失自动重建，失败的工具调用不写入缓存；成功条目逐个原子落盘，中断后可复用已完成条目。每次打包仍核对源码、profile、依赖、对象/EXE身份、当前模块成员及PTX/SASS目标，不因缓存命中跳过这些门禁。日志汇报复用和实际检查数量；完整入口加 `-ReinspectCuda` 可强制重新提取所有请求组件的CUDA镜像。

Stage1 local/dev/release默认由 `tiers.def` 派生125个曲线/param/容器模块，另保留3个常规 CUDA 单元；Edwards binary/NAF 共用对应容器。Stage2默认使用两个聚合设备单元，共享原有设备常量；隔离备选由原始显式实例和薄发射器派生176个设备模块，主机常量快照只同步到被调用的模块。生成文件保留于构建目录，其原始源码/生成器均参与身份验证。

CUDA worker 在首次 CUDA 调用前将未设置的 `CUDA_MODULE_LOADING` 默认设为 `LAZY`；用户显式值保留。Stage1 全档枚举使用纯元数据，Montgomery全档`--gpu-info`的`blocks_per_sm=0`表示尚未查询occupancy；指定实际输入位宽后只取得选中档位的内核/occupancy。懒加载需要驱动支持；旧驱动或显式 EAGER 下的加载范围不据此获证。按需 JIT 的最终资格以实际加载模块观察为准，不能只凭 TU 拆分宣布通过。

Stage2聚合模式不提供按容量隔离JIT的保证：算术或NTT镜像可能包含未选容量/变体。静态打包仍检查两个对象、完整设备入口/实例成员及PTX/SASS目标，不再将176个独立镜像作为默认条件。隔离备选继续验证逐模块成员。`test_release_runtime.py` 默认保持原隔离范围门禁；聚合模式必须显式传入 `--allow-grouped-stage2`，报告记录布局，数学及镜像选路验证保留，结果不声明逐容量隔离通过。0.52.1隔离构建的已有JIT数据不能作为聚合模式的运行资格。

## nvcc 编译计时

正式 CMake、独立 Stage2 和维护中的探针构建在实际 nvcc 编译/链接命令附加 `--time`。每次执行写入独立 `<build>/timing/<唯一ID>/nvcc.csv` 及命令/退出码/外层经过时间记录，避免并发追加同一文件；计时路径不进入语义编译指纹。无改动复用不生成虚构耗时，失败记录保留，成功但缺少非空 CSV 会报错。CXX 编译只保留主机/队列计时。并发单元耗时之和不等于队列墙钟，nvcc 内部阶段与外层时间不能重复相加。

上述样本覆盖项目目标的编译和链接；版本查询及CMake内部编译器识别/能力探测不计入目标编译样本。

`tools/build/test/build_cuda_jit_trace.ps1` 编译测试专用 CUPTI 观察 DLL；`tools/test/test_release_runtime.py --help` 指定冻结基线、观察器、设备和证据目录，分别验证数学结果、实际模块及冷/热独立缓存。观察器不随发布包分发。JIT 编译、缓存加载/存储活动与进程墙钟分别记录；缺少独立 JIT 记录不能填零或把完整启动时间称为纯 JIT 时间。强制PTX变量不能替代实际镜像观测；本机现代包的PTX验证方法、对应测试副本及覆盖限制见[发布/JIT验证](../performance/RELEASE_JIT.md)。

发布对象的独立数学探针由 `tools/build/test/build_release_probes.ps1 -Profile cu11.8/cu13.3 -Stage stage1/stage2` 构建。`build_release_ptx_fixture.ps1` 只在独立目录制作并核对 PTX 测试副本，不覆盖发布 EXE。混合队列范围与完整包验证分别使用 `tools/test/test_release_switching.py`、`test_release_bundle.py`；输入和新证据目录须显式指定。

## GMP 与打包

本机/开发默认使用本地 Zen3 GMP；发布默认使用 vcpkg 通用 x64 GMP，缺失时明确报错，不自动回退到 Zen3。自定义 `-Gmp <prefix>` 必须提供匹配的头文件、导入库、DLL 和许可证。Zen3 库使用 BMI2/ADX 等指令，其 CPU 兼容范围由实际构建决定。

完整包的载荷见上节。GMP 发布依赖要求可验证的 vcpkg `share/gmp/vcpkg.spdx.json`、匹配的头文件/导入库/DLL、配方与上游源码；缺少对应材料会在构建前失败。Zen3 不作为发布回退。ZIP 从干净暂存目录生成，不收录 save、日志和运行数据。

对应源码快照从当前Git检出根枚举源码，包含未提交的新源码，排除被忽略的测试输出；固定第三方源码和许可另按明确目录纳入。解压源码重建完整包时，先在解压根执行`git init`，并准备上述工具链、vcpkg GMP和PARI材料；打包拒绝借用解压目录父层的另一个仓库。

发布统一输出到 `release/ecm-elf-<版本>-cu<CUDA版本>/`；分阶段入口只生成构建组件记录。`components/package-manifest.json` 保存本次所选包的目录名、ZIP 名、包内文件、EXE、ZIP及模块报告摘要；它只证明构建和静态镜像/载荷检查，不自动获得运行时资格。`tools/test/test_release_bundle.py --release release --manifest <构建目录>/components/package-manifest.json --output <新证据目录>` 验证当前布局，也保留对已有 schema 1 冻结包的检查能力。

打包首先通过真实生成器读取Stage2只读入口清单，接口检查在耗时的CUDA镜像提取之前完成；只读模式不生成隔离源码。后续输出当前profile、阶段及PTX模块检查进度，区分组件身份验证与cuobjdump镜像检查。

## 复用构建

Stage1 默认启用增量复用，按编译命令、响应文件、递归依赖内容与对象 SHA256 判断 CUDA 对象是否可复用；保留时间戳也不能绕过内容校验。`-Incremental` 保留为兼容参数，缺少当前依赖记录的旧对象会重编一次。`-FullRebuild` 删除 CUDA 对象后重编；内部并行编译器的 `-Rebuild` 强制编译选中的单元。主机 CXX 文件仍由 CMake/NMake 管理。`-HostOnly` 要求已配置的同 profile 目录，在配置/主机编译前验证所有 CUDA 对象及原生构建依赖时间戳，任一失效即拒绝；不用于首次配置。`-HostOnly` 与强制重建互斥。旧 `-RelinkOnly` 绕过入口已关闭，使用它会明确报错；主机修改使用正常增量或 `-HostOnly`。

复用旧的MSVC/CMake构建目录时，需核对关键主机对象的`.obj.d`确实包含头文件依赖。旧编译器探测缓存中的本地化`SHOWINCLUDES`前缀可能与当前输出不匹配，造成依赖文件为空，头文件变化后仍报告构建成功。遇到此情况，先保留证据，备份并移开该目录的C/C++编译器探测缓存和主机对象，再用正式入口重新探测、编译并验证实际程序行为；不编辑依赖指纹或CUDA对象来绕过校验。Stage2使用编译器依赖清单校验内容：新MSVC采用`/sourceDependencies`，14.16采用`/showIncludes`并保留实际SDK/STL/宏包含。

配置读写、sigma 生成、曲线 CPU 参数化、运行选择、逐曲线结果、checkpoint/save、data/result/tune.toml/屏幕的主机实现正文使用普通 CXX。`tools/test/test_host_incremental_boundaries.py` 通过正式 local 脚本验证这些正文修改只重编译主机单元，随后恢复源文件；不会改写对象或清单强行复用。接口、CUDA runtime 或其实际依赖的修改仍可能重编译 CU。验收证据与适用构建见 [统一 Stage1 实施记录](../development/CUDA_STAGE1_UNIFICATION_DRAFT.md#p5-完成完整档位兼容编译边界与固定条件对照)。

Stage2 默认逐编译单元复用，整体源码/构建签名变化不会直接触发整组重编；每个失效单元打印重编原因。主机 `.cpp` 使用选定 MSVC 的普通 C++ 编译，保留原有 `/MT`、C++17、`/O2`、`/EHsc` 和 NTT 宏；CUDA `.cu` 与链接仍使用 nvcc，真实 runtime 调用仍需 CUDA SDK。MSVC的编译器依赖记录将实际 SDK/STL/宏包含纳入同一内容与对象身份校验，不以本地正则扫描替代编译器依赖。链接命令、GMP 导入库、对象摘要和程序摘要全部匹配时跳过链接；旧清单首次建立链接输入记录可能需要一次链接。

Stage2 的 `-HostOnly` 在任何编译前核对完整清单，只复用身份一致的 CUDA 对象，主机部分仍按需编译；架构、后端、工具链、GMP 头、CUDA 依赖或任一 CUDA 对象改变时拒绝。`-HostOnly` 与强制重建的 `-Rebuild` 互斥。普通复用和发布 `-SkipBuild` 同时核对程序、编译单元、依赖和对象；旧 manifest 缺少身份记录时须正常重建。保留构建目录中的 `.ecm.d`、`.ecm-state.json`、源清单和 manifest；不能修改清单或替换 DLL 来绕过核对。

## 用途与 profile 目录

`profile.json` 保存完整 SHA256 及字段。CUDA local 默认使用 `build_stage1_local`、`build_stage2_local` 固定短目录；dev/release 及显式 Stage2 `-BuildRoot` 的 profile 目录使用前 16 位 key。字段包含 role、stage、engine、架构、编译/链接策略、工具链和 GMP 内容身份；Stage1 还包含 CMake 身份、tiers、PRAC 和额外选项。源码 hash、HEAD、程序版本不进入目录 key，源码变化由原有逐 TU 身份处理。固定 local 目录与任何显式叶目录都只容纳一个相容 profile；配置不匹配时拒绝，不改写 marker，也不自动清空目录。需要同时保留不同架构、tiers 或编译策略时，为 Stage1 指定 `-BuildDir`、为 Stage2 指定 `-Build`；Stage2 也可用 `-BuildRoot` 选择带 profile 层级的根。

本机/开发 Stage1 的 `-BuildDir`、Stage2 的 `-Build` 是最终叶目录，不额外拼层；Stage2 内部 `-BuildRoot` 才表示根。公开 release 统一只接收 `-BuildRoot` 根目录，叶目录由完整 profile 派生。不能把 local/dev/release 目录互作输出，也不能让包目录与 build 目录相同或互相嵌套。

旧目录不自动移动或复制。显式指定旧目录时，仅对已有正式清单、工具链/GMP/架构/策略相符的目录原位核对并建立 marker；未知非空目录拒绝。`-SkipBuild` 要求已有正确 marker，仍需通过原有构建和打包身份检查。路径消费者需使用入口打印的精确叶目录或显式配置，不选择扫描结果中的“最新”程序。

CPU-only 身份检查入口为 `test_cuda_build_profile.ps1`、`test_cuda_build_profile_paths.py`；双 profile/生成边界/计时合同使用 `test_release_build_contracts.py`。`test_cuda_p5_release.py`、`test_cuda_p5_images.py`保留为原六架构阶段的显式冻结验收工具，不用于认定当前双 CUDA 完整包。路径检查不等于运行资格。

## CUDA 源清单与依赖身份

翻译单元唯一清单为 `tools/build/internal/cuda_sources.cmake`：分别维护 Stage1 共用源、CUDA 源、可选 PRAC、Stage2 驱动源、生产主机引擎、算术、NTT 和开发 wrapper。Stage2 的 `STAGE2_ENGINE` 包含17个主机 CPP，其中常量快照和模块调度只在隔离宏下提供实现；默认使用算术/NTT两个聚合CU，显式隔离时由设备生成器替换为176个细分单元。实际编译成员以构建目录内 `cuda_sources.json`、模块清单和编译记录为准。生产源码闭包拒绝包含 `tools/bench/`、`tools/test/`、`tools/experiments/` 和 `.refactor/`，包括编译器补齐的依赖；开发 wrapper 使用单独源组。实验树、BSGS、独立 NTT、cuFFT 和 giant 变体另有显式实验源组，不加入生产目标。模块职责以 [Stage2 模块边界](../architecture/STAGE2.md#cuda-模块边界) 为准。

CMake 同时导出 `cuda_target_graph.json`，收集目标及其链接的项目对象库/静态库；条件目标按保守全集收集，导入库由工具链与依赖清单另行记录。对象枚举和 CUDA 完全重编按该目标闭包执行。`cuda_build_state.ps1` 扫描本地递归包含，包括 `.inc/.inl/.def`，并用 NVCC 依赖文件补齐宏包含及 SDK 头。每个对象的 `.ecm-state.json` 记录命令、依赖内容指纹与对象摘要；摘要编码在 Windows PowerShell 5.1 和 PowerShell 7 间一致。

源身份验证入口为 `tools/build/test/build_cuda_p0.ps1`，用 `-Stage stage1/stage2/development` 选择构建；CPU 故障注入为 `tools/test/test_cuda_build_identity.ps1`，短曲线双版本对照为 `tools/test/test_cuda_p0_runtime.py`。`tools/build/test/build_cuda_p1.ps1 -Stage tiers` 编译清单的16种筛选/几何配置，`-Stage stage1/stage2` 构建迁移后的程序。结构与冻结算术正文校验使用 `test_cuda_p1_structure.py`；实际注册、跨组曲线与可选运行时分别使用 `test_cuda_p1_registry.py`、`test_cuda_p1_curves.py`、`test_cuda_p1_prac.py`。这些入口不替代完整位宽、多架构或性能验收，证据写入 `data/experiments/`。

GUI 只有一个 BAT 入口：`tools/build/build_gui.bat`，不分 dev/local/release，默认输出 `build_gui_local/ecm_gui.exe`。参数原样传给现有 `tools/build/dev/build_gui.ps1`；例如 `tools\build\build_gui.bat -Selftest` 构建并自检。无参数调用结束暂停，有参数调用返回实际退出码。

`tools/build/dev/` 为开发辅助与 GUI PowerShell 实现，`test/` 为探针与参考实现构建，`internal/` 为配置检查、并行编译、拆分和打包辅助。详细参数入口见 [构建工具](../../tools/build/README.md)；配置生成合同见 [配置维护](../architecture/CONFIGURATION.md)。

P2 结构验证使用 `test_cuda_p2_structure.py` 对照冻结 Git 基线的设备算术正文并检查模块依赖方向。`build_cuda_p2_arithmetic_probe.ps1 -WorkerBuild <生产目录>` 将独立 GMP 检查器链接到经过身份验证的生产算术对象；`build_ntt_graph_probe.ps1 -WorkerBuild <生产目录>` 同样复用真实 NTT 对象。其他使用生产 NTT 的探针通过 `internal/cuda_ntt_probe.ps1` 消费同一源清单，所有单元保持探针指定的 legacy/per-thread 默认流设置；不再通过实现头或截取生产源码构建。

当前主机清理门禁为 `test_cuda_p4_cleanup.py`，对照 P3 基线核对32个生产模块，只允许有调用点、恒值和作用域证明的有限删除。`test_cuda_p4_isolation.py` 核对实验迁移正文、入口默认值及包含边界，`test_cuda_p4_probe_contracts.py` 核对有限探针接口及独立 oracle。P3 的完整搬迁等价脚本是该阶段的冻结门禁；当前删除后的代码使用 P4 门禁。

`test_cuda_p3_runtime.py` 继续消费冻结旧程序、独立验证的存档和新程序，比较完整叶指纹、稳定结果字段、计划查询、分配/预算回退及各日志级别的损坏拒绝。资源测试通过 `stage2_native_source.py` 读取实际模块的声明和方法，不使用另一份手写资源实现。依赖关系收缩后，构建可能因旧编译依赖记录拒绝收缩后的依赖；保留该记录，使用新构建目录完整编译，再验证复用。

实验完整模块由 `internal/cuda_experiment_probe.ps1` 构建，记录递归源码闭包、命令、对象和程序摘要；入口不再截取 `.cu` 正文。GMP/测试夹具仍位于 `tools/test/`。pack 探针以原 per-thread ABI 独立编译同一生产模块；常规生产对象为 legacy，复用时会拒绝该差异。giant 可通过 `-WorkerBuild` 链接验证过的生产算术对象，其旋转及未注册容量单独编译。host-wait 探针通过显式适配编译真实 Edwards 运行时和设备模块。`build_stage2_d_model_probe.ps1` 默认保留旧实验表，`-Profiles frozen-production` 显式测试冻结的生产候选表。`build_cufft_probe.ps1` 从实验源组构建完整 cuFFT 候选并记录源码、命令、库和对象身份。冻结源码审计由 `internal/cuda_frozen_ntt_probe.ps1` 保留旧快照身份及其完整实现，不能用历史实验脚本的路径推断当前生产闭包。

单任务合同验证入口为 `tools/build/test/build_stage1_task_contract.ps1 -WorkerBuild <Stage1生产目录>`，然后由 `tools/test/test_stage1_task_contract.py --help` 查看运行参数。它以生产清单中的实际任务、checkpoint、日志和运行配置模块构建，读取正式 MSVC/GMP 身份，使用普通 CXX，不调用 nvcc、不添加 CUDA SDK include；编译器依赖记录同时确认未使用 CUDA SDK 头。测试仅替换计算回调，覆盖八种拥有权、停止、错误与锁释放场景，不代替 GPU 运算资格或性能验收，也不表示整个生产程序无需 CUDA。

`tools/test/test_cuda_target_graph.ps1 -WorkerBuild <Stage1生产目录> -Output <证据目录>` 使用真实 CMake 目标收集器做五项 configure 检查，覆盖源目录与目标同名、显式对象库、直接/条件静态链接及 imported 排除。只有链接属性中的目标引用参与依赖图；普通源码路径的目录名不视为目标。

当前包含边界检查为 `tools/test/test_main_flow_boundaries.py --output <新证据目录>`，配合实际编译和运行验证。测试专用 `main_flow_layout.json` 保留历史路径与当前位置的明确映射，生产不读取它；`test_main_flow_layout.py` 是原目录搬迁阶段的冻结对照，不能把后续实现一律按该旧正文验收。本轮目录精简的当前正文、直接相对 include、TU 顺序和 CMake 属性对照及证据见[目录精简草稿](../development/DIRECTORY_LAYOUT_SIMPLIFICATION_DRAFT.md)。旧编译器依赖记录若因搬迁拒绝复用，应保留记录并使用新 local 目录重建，不修改 sidecar 伪装复用。

## OpenCL local

使用 tools/build/build_opencl_local.ps1 构建 ecm.exe，默认叶目录为 build_opencl_local/stage1/&lt;profile-key&gt;。可用 -BuildDir 指定最终叶目录，-OpenClRoot 指定含 include/CL/cl.h 与 OpenCL.lib 的 SDK 根；-Gmp、-OpenSslRoot、-VcVars 可覆盖主机依赖。入口不启用 CUDA 编译器，沿用普通 CXX 的 /MD 优化配置。

输出包含程序、GMP/OpenSSL 运行库、profile.json、build_manifest.json 和 opencl_resources。原有环境或命令选择、PARAM3 支持范围和生产自检保持；资源查找和部署限制见 [OpenCL](../architecture/OPENCL.md#资源与-local-构建)。该入口默认仍为local；完整发布入口用独立release身份调用它并打包，无单独OpenCL dev/release BAT族。

### 编译期间输入稳定性

构建前后核对已知依赖内容。成功编译发现新的系统/宏 include 时纳入新指纹；旧 include 退出新 compiler 清单时，按 live 源根重新读取旧文件哈希，而不是将缺少清单键当作内容变化。旧文件在编译期间被修改或删除、或无法验证仍失败。持久对象记录 schema、命令/工具链/对象哈希和 HostOnly 检查不变。

`tools/test/test_cuda_build_identity.ps1` 覆盖身份变异与跨 PowerShell 指纹；`tools/build/test/test_msvc_header_retirement.ps1` 使用正式主机编译器产生真实旧/新依赖并核对退役行为。两者输出到显式隔离目录，不改生产对象缓存。


## 共享 PARI 诊断依赖

Stage1、Stage2 与 GUI v2 默认采用 PARI DLL → 外部 GP 的诊断顺序；GMP 分析仅显式选择。实际 ECM 运算后端不变。完整本机构建需要准备 `third_party/pari-2.17.3/`；准备命令为 `python tools/build/test/prepare_gui_v2_pari.py --installed <PARI-2.17.3目录> --archive <原始源码tar.gz>`，版本和文件摘要由 provenance 校验。

共享助手的 CMake 入口为 `src/ecm/service/diagnostics/`，生成 `ecm_diagnostics_host.exe`。主工程 Stage1/OpenCL/CMake Stage2 的目标自动依赖它；独立 Stage2 队列构建调用同一目标，使用与 worker 一致的 GMP 安装前缀。测试入口为 `tools/build/test/build_diagnostics.ps1 -Build <独立目录>`；可显式构建 `ecm_diagnostics_probe` 进行协议与故障验收。GUI v2 使用相同目标，保留独立的合数输入协调助手 `ecm_group_order_ui.exe`。

本地输出附带助手、GMP、PARI DLL、`pari-data/seadata`、许可证和 `diagnostics-source.zip`，并生成逐文件及源码摘要的 `diagnostics_build_manifest.json`。Stage1/Stage2 发布打包核对该冻结身份，再将完整依赖纳入 ZIP，支持受限的相对目录路径；不打包用户配置、队列或日志。PARI 原始源码、配置头文件及 SEA 数据声明随包保留；没有 gp.exe。只复制计算 EXE 不构成完整的 DLL 诊断部署。

`-WithoutLocalPari`（共享助手/GUI 测试构建）或 `ECM_DIAGNOSTICS_LOCAL_PARI=OFF` 用于明确的无 DLL 测试，要求新目录；它不启用自动 GMP。Windows 发布候选要求完整 PARI 依赖。运行时 DLL 缺失时 Auto 可尝试用户配置的 GP。当前 Win64 ABI、预算与证明语义见[共享诊断层](../architecture/GROUP_ORDER.md#共享后端选择与进程隔离)。
