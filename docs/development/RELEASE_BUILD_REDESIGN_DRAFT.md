# Release 构建、双 CUDA 分发与按需 JIT 改造草稿

日期：2026-10-23；2026-10-24 用户授权“按照规划实施”。状态：**实施完成，本机验收通过；旧GPU/驱动矩阵待实机验证**。下文保留规划时的现状、需求及决策；实施结果见文末。正式行为同步到相应主题文档，本草稿保留，待用户决定归档或清理，不自动删除。

## 1. 已明确的需求与本轮边界

- 大幅调整现有 release 脚本，以 CUDA 工具链版本作为分发单位。无参数正式发布默认生成两个完整版本：`cu11.8`、`cu13.3`。
- `cu11.8`：CUDA 11.8，嵌入 **compute_50 PTX + sm_60 SASS**，主要面向旧驱动和 GTX 900／10 系。该定位须按设备和镜像路径分别验收，不表示任意旧驱动都支持 PTX JIT。
- `cu13.3`：CUDA 13.3，嵌入 **compute_75 PTX + sm_80 SASS**。
- 输出层级：`release/ecm-elf-<x.y.z>/cu<CUDA版本>/`。每个 CUDA 目录都包含 Stage1、Stage2、GUI 及必要配置和运行资源。
- 用户已选定 **GUI v2**；不在默认完整包中附带原 GUI。
- 用户明确要求：**所有发布组件统一使用 vcpkg 提供的通用 x64 GMP DLL**。Stage1、Stage2、GUI v2 群阶助手使用与该 DLL 匹配的 vcpkg 头文件和导入库；不混入本地 Zen3 GMP。
- 本地编译统一使用 `nvcc --time` 采集编译阶段耗时；运行时另测驱动 JIT 耗时。
- JIT 必须只处理当前任务需要的曲线族、Montgomery 参数化与位宽容器，以及实际执行需要的公共内核。
- 曲线、param、sigma 相同时保持数学行为一致；重组构建和加载边界不能改变因子、群阶或相应 B1/B2 条件。

规划阶段只维护草稿及入口链接；此边界已被2026-10-24的实施授权替代。以下各节保留原规划，未完成验证不能视为发布资格。

## 2. 已核对的现状与差距

| 范围 | 当前实现依据 | 对新方案的影响 |
| --- | --- | --- |
| Stage1 release | `tools/build/build_stage1_release.ps1:21` 默认 Archs=60,70,75,86,89,120，PtxArch=75；按架构小于75分工具链 | 工具链选择须改为显式 CUDA profile，不能由 SASS 数字推断；旧包明确选择 CUDA11.8，而不是沿用现有12.6解析结果 |
| Stage1 编译与拆包 | `tools/build/internal/build_stage1_release_group.ps1`；`split_stage1_release.ps1` | 当前先编多架构，再裁剪对象、临时替换并重链接；新默认流程直接生成两套指定镜像，不需要六架构拆包 |
| Stage1 PTX 参数 | `CMakeLists.txt:344`、`:366`；公开和内部 release 的 PtxArch 校验 | 当前 PTX 依附于 SASS 架构列表；需要独立表达 PTX50/SASS60、PTX75/SASS80，不能顺便增加 sm50 或 sm75 等非目标 SASS |
| Stage2 release | `tools/build/build_stage2_release.ps1`；`internal/package_stage2.ps1` | 当前逐架构构建独立包，仅可选附带 Stage1，没有统一 GUI v2 完整包 |
| Stage2 镜像参数 | `tools/build/internal/cuda_build_profile.ps1:67` 的 `-arch=$Arch` | 改为显式 PTX/SASS 生成策略，并用产物检查确认；不能把单一 -arch 简写当成精确双目标策略 |
| Stage1 设备实例 | `internal/cuda_sources.cmake`；`src/ecm/backend/cuda/dispatch/tiers.def` | Edwards 七组、Montgomery 各参数化七组，一组包含多个位宽；曲线分组不等于按位宽 JIT 隔离 |
| GPU 信息查询 | `src/ecm/backend/cuda/stage1/montgomery/registry.cu:163`、`:204`；`factory.cpp:303` | 当前 tier 查询会取得 kernel 指针，随后可逐档查询 occupancy；须验证这些查询是否提前触发无关模块 JIT |
| 编译计时 | `internal/parallel_nvcc.ps1:177`、`:223`；`build_stage2_local.ps1:160` | 现有外层墙钟仍有价值，但没有统一采集 nvcc 各编译阶段 CSV |
| GUI v2 依赖与载荷 | `src/ecm/application/gui_v2/CMakeLists.txt:110`；`tools/packaging/package_gui_v2.py:19`、`:34` | GUI 自身不链接 GMP，但群阶助手需要 GMP；现有 GUI-only 候选依赖和源码打包引用本地 Zen3，不能原样并入新正式包 |
| 输出和身份校验 | `internal/cuda_build_profile.ps1`、`release_payload.ps1`、`package_archive.ps1` | 保留源码/依赖/对象/程序身份验证和干净暂存打包，将最终分发根改到 release |

现行正式构建说明仍以 [BUILD.md](../usage/BUILD.md) 为准。旧六架构发布尚有未完成验收，不能把它视作本方案已有的兼容性证明，见 [CUDA 模块验收](../performance/CUDA_STRUCTURE_QUALIFICATION.md)。

### 2.1 本轮只读工具链核对

此前只读核对过本机 CUDA12.6.85 和 CUDA13.3.73 的版本、目标列表与 `--time` 帮助。12.6不属于本次最终发布矩阵，该核对不能作为11.8已就绪的证据。**本轮未核对本机11.8是否安装、具体补丁版本或实际编译能力**；实施 P0 必须单独核对11.8的版本、目标、MSVC支持范围和完整编译参数，不将13.3的编译选项原样传给11.8。

## 3. 双 CUDA 发布 profile

两阶段消费同一份发布 profile 定义，避免 Stage1/Stage2 各维护一套目标值。

| 分发目录 | 编译工具链 | 唯一保留的虚拟目标 | 唯一保留的原生目标 |
| --- | --- | --- | --- |
| cu11.8 | CUDA 11.8 | compute_50 PTX | sm_60 SASS |
| cu13.3 | CUDA 13.3 | compute_75 PTX | sm_80 SASS |

这里的50/60/75/80表示目标计算能力，不是 PTX ISA 版本号；实际生成的 PTX ISA 版本还受工具链影响，须与驱动支持一并核对。

设备代码生成策略分别为：

- cu11.8：`--generate-code=arch=compute_50,code=compute_50` 与 `--generate-code=arch=compute_60,code=sm_60`。
- cu13.3：`--generate-code=arch=compute_75,code=compute_75` 与 `--generate-code=arch=compute_80,code=sm_80`。

这是编译参数规划，不是脚本实现。Stage1 可以通过等价的 CMake real/virtual 目标表达，Stage2 通过同一 profile 生成命令；两者最终均以对象和程序中的实际镜像为准。若实现采用外部模块，检查覆盖所有分发设备模块。

关键约束：

1. PTX 和 SASS 是两个独立字段，工具链版本也独立存储。不得为满足旧 PtxArch 校验把 PTX 目标加入 SASS 列表。
2. 不再默认生成 sm60/70/75/86/89/120 六套目录。跨大版本 GPU 的运行可能使用 PTX JIT，不能声称一个 SASS 目标能覆盖所有更高架构。
3. 启动前核对所选工具链的真实版本、虚拟与原生目标支持、MSVC 相容范围。缺少任一默认工具链时，默认完整发布明确失败，不静默省略一个 CUDA 包。
4. 构建 profile、编译命令身份、SkipBuild 校验和包身份增加独立的 PTX/SASS 字段及模块布局身份；不同工具链、阶段和策略的对象严格分离。
5. compute50 下的内建函数、内联 PTX、CGBN 和 Stage2/NTT 指令路径需逐项编译验证。失败时先识别具体限制，不擅自提高用户指定的 PTX 基线，也不改写数学行为绕过失败。
6. 工具链接受目标不等于旧 GPU 已获支持。正式兼容范围还取决于驱动、运行时、设备能力和原生硬件验证；驱动最低版本作为实施 TODO 核对。

### 3.1 旧驱动兼容边界与待验条件

旧包严格保持用户指定的 **PTX50 + SASS60**，不自动增加 SASS50/52/61。显卡与路径分开判定：桌面 GTX900 系 Maxwell 主要为sm52，部分移动版为sm50，必须走本包 PTX50 JIT；GTX10 系 Pascal 主要为sm61，验证使用本包 sm60 的同大版本相容 SASS。sm60 本身的原生运行另需对应设备，不能把sm61相容执行写成sm60原生验收。型号依据见 [NVIDIA 旧GPU表](https://developer.nvidia.com/cuda/gpus/legacy)，镜像规则见 [Pascal 兼容指南](https://docs.nvidia.com/cuda/archive/11.8.0/pascal-compatibility-guide/index.html)。

Windows 驱动规划采用三类条件，不预先承诺最低可用版本：

- **452.39起的小版本兼容条件**：11.x的官方小版本兼容基线不等于本项目整包的最低驱动。仅在有相容 SASS、未调用较新驱动特性的场景中列入候选验证；不能用于保证11.8生成的 PTX 可被旧驱动 JIT。
- **522.06及以上的候选完整验证基线**：522.06是11.8发布时对应的Windows驱动，作为旧包 PTX JIT 和按需加载的首轮测试基线，仍须用具体卡型和实际二进制验证，不能仅凭版本表宣布通过。
- **低于上述 JIT 基线的驱动**：分别验证 SASS 路径和 PTX 路径，读取原始错误并记录是否需要更新驱动。compute50不表示PTX ISA5.0；降低计算能力目标并不消除 PTX 语法版本限制。显式加载也不能修复驱动不认识的 PTX。

上述版本关系见 [CUDA11.8发布说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html)，PTX及新驱动特性限制见 [CUDA小版本兼容](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。当前目标为Windows x64；如扩展Linux，另列其驱动版本，不套用Windows数字。

还需审计运行时/Driver API符号、Graph、内存分配及其他驱动相关特性的真实使用。较新的能力查询或加载API不能以不可用的静态导入使程序在旧驱动上直接启动失败；选择该驱动实际支持的接口或明确拒绝。禁止通过改系统驱动、混放开发机驱动DLL或关闭保护来模拟兼容。

## 4. 入口、输出与完整包

### 4.1 发布入口建议

新增 `tools/build/build_release.ps1/.bat` 作为默认完整发布入口：预检 → 冻结源码和依赖身份 → 构建通用 GUI v2/群阶助手 → 构建 cu11.8 两阶段 → 构建 cu13.3 两阶段 → 检查镜像和载荷 → 打包。每个步骤有独立退出码和日志。

- 无参数生成两个完整包；保留显式选择一个 CUDA profile 的能力，单 profile 输出标记为单独构建，不能冒充默认双版本完成。
- 保留 Stage1/Stage2 分阶段入口供重建和诊断，底层统一消费新 profile；只有完整组件齐备且身份一致时才能生成完整分发包。
- 旧 Archs/PtxArch/按架构拆包参数不作静默转换。实施时明确拒绝不兼容参数并指向新参数；若仍需保留旧流程，必须使用显式独立入口，不与新默认输出混写。
- 默认不自动跑数学、GPU 性能或 JIT 测试；打包本身的身份、镜像和文件完整性校验属于必要步骤。测试脚本显式执行并记录资格。
- 通用 GUI v2 和群阶助手可构建一次、冻结后复制到两套包；发布构建目录与现有 GUI local 候选目录分离，不直接拾取“最新”的本地 exe。

### 4.2 目录建议

```text
release/
  ecm-elf-<x.y.z>/
    cu11.8/
      ecm_cuda.exe                  Stage1
      ecm_cuda_stage2.exe           Stage2
      ecm_gui_v2.exe                GUI v2
      ecm_group_order_ui.exe        GUI v2 群阶助手
      gmp-10.dll                    vcpkg 通用 x64，三个计算组件共用
      ecm.toml / ecm.toml.example
      worktodo.txt
      ECM_TOML_REFERENCE.md
      assets/                   仅包含基础ELY
      localization/v2/
      licenses/                    组件与实际依赖对应的许可和声明
      source/                      按实际分发依赖保留必要的对应源码材料
    cu13.3/                        同样完整且可独立使用
    ecm-elf-<x.y.z>-cu11.8-win-x64.zip
    ecm-elf-<x.y.z>-cu11.8-win-x64.zip.sha256
    ecm-elf-<x.y.z>-cu13.3-win-x64.zip
    ecm-elf-<x.y.z>-cu13.3-win-x64.zip.sha256
```

Stage1、Stage2、GUI 用现有可执行文件名放在 CUDA 目录同层，匹配 GUI v2 当前的相邻 worker 寻址；不额外引入 stage1/stage2/gui 子目录。每份 ZIP 包含对应的 `cu版本/` 顶层目录，可在版本文件夹下解压并保持布局。

版本文件夹取计算主程序的发布版本来源 `include/ecm/platform/program_version.h`。GUI v2 使用独立版本来源 `src/ecm/application/gui_v2/version.h`；记录各组件实际版本和哈希，不要求 GUI 版本与计算版本相等，也不以 GUI banner 命名整套发布。

必要载荷根据生产运行依赖闭包确定。保留 GUI v2 默认 DX11、语言资源、当前附带的宠物资源和群阶助手；不因合包而把可选 Vulkan 诊断路径或本地 PARI/GP 安装变成默认必需项。若按需 JIT 必须引入外部设备模块，则追加明确的运行资源目录，并纳入完整性和移动目录验证。

### 4.3 vcpkg GMP 一致性与打包约束

- 从一个显式解析的 vcpkg 通用 x64 prefix 取得 `include/gmp.h`、`lib/gmp.lib`、`bin/gmp-10.dll` 及对应许可/来源材料，三者身份一起冻结。禁止回退到 `third_party/gmp-zen3` 或从 PATH 临时寻找同名 DLL。
- Stage1、Stage2 和 GUI v2 群阶助手都按上述依赖重新编译；不能只把已链接本地 Zen3 的助手旁边的 DLL 换掉。
- GUI v2 构建入口须能显式接收发布用 GMP prefix，并验证实际配置；GUI 本体继续保持当前的 GMP 隔离方式。依赖的 CRT 和通用 CPU 条件也要核对。
- 每个 CUDA 包内仅放一份同哈希 `gmp-10.dll`。合包前核对所有相关组件的头文件、导入库、运行库身份，不接受按复制顺序覆盖 DLL。
- 现有 GUI-only packager 的 Zen3 许可、对应源码说明和候选要求需按真实 vcpkg 依赖改造后才能复用。正式完整包使用明确载荷清单，不直接拼接其 ZIP。
- 构建/源码/组件/打包 manifest 及计时原始记录留在构建目录或 data 证据目录。许可证、资源来源声明和必要对应源码属于合法载荷，不被“仅分发 TOML 项目说明”的过滤器误删；不随包附带开发草稿、原始日志、剖析文件或个人运行数据。
- 干净暂存生成 ZIP 与 SHA256，不从已有运行目录递归打包。已有版本目录中的用户 ecm.toml/队列不被覆盖；拒绝不明身份的覆盖发布，失败暂存不作为完整发布。
- 默认双版本发布只有在两个包都成功时报告完成；保留失败日志，允许基于完整身份重试。build、暂存和 release 目录相互隔离。

## 5. 本地 nvcc 编译计时

### 5.1 统一采集方式

本方案将“本地编译”解释为开发机上执行的所有仍维护的 nvcc 编译入口，包括 dev/local/release、`tools/build/test/` 及维护中的独立探针/实验构建；共享 helper 统一提供 `--time` 采集。先接入正式入口和本次测试，再按调用点审计逐项补齐其余入口，全部补齐前不宣称“统一计时”已完成。纯 CXX/GUI 编译不伪装成 nvcc 计时。

每次真正执行的 nvcc 命令使用独立 CSV，例如 `<build-leaf>/timing/<run-id>/<单元身份>.csv`。`--time` 会追加文件，故不能让并行进程共享一个 CSV，也不能把新旧构建、不同工具链或重复重试混到同一份记录。nvcc 承担的设备链接/最终链接命令单独记录；纯主机链接保留外层计时。工具版本查询和纯依赖扫描单独分类。

汇总输出保留三个不同边界：

1. nvcc 原始 CSV 中的编译阶段时间及阶段标签，不跨 CUDA 版本硬套私有阶段名称。
2. 每条实际命令的外层经过时间、退出码、源/对象身份、工具链和 CSV 路径；MSVC 主机编译单独归类。
3. 编译队列墙钟、配置/链接/打包墙钟与整个构建墙钟，各自独立。并发命令时间之和、嵌套编译阶段时间不能重复相加成为“总构建耗时”。

增量跳过记录 skipped/reused，不为未执行命令生成本轮阶段耗时。失败 CSV 和日志保留，标记失败或不完整；实际 nvcc 编译成功却缺少应有计时文件时应明确报告采集失败，不能静默填零。

### 5.2 与身份、增量构建和并行度的关系

计时路径、run-id 等仅影响诊断输出的字段不应导致每次构建重编译。建立受限、明确的计时参数规范化：保存完整实际命令，语义指纹仅排除 `--time` 输出路径等已定义字段；架构、宏、编译器、依赖、对象及链接身份仍完整核对，不做宽泛字符串删除。

验证无改动第二次构建不重编 CUDA，不因 CSV 缺失而强制重建旧对象；旧缓存标记为“本轮复用、无新计时”。新版本身份 schema 的接纳规则须与现有严格 SkipBuild/HostOnly 规则相容，不能凭新增计时字段直接接纳未知对象。

维持编译单元间并行与 `--split-compile` 单元内并行的区别。已有 [ptxas 严格句柄检查诊断](PTXAS_SANDBOX_DIAGNOSIS_DRAFT.md) 是本机已知限制；实施采样记录 SplitCompile、工具链和执行环境，必要时用显式 SplitCompile=1 的独立 profile 验证，不把编译失败隐藏为计时缺失，不关闭保护绕过问题。

CUDA11.8单独核对支持的命令选项，尤其不能假定现有Stage2 `--split-compile` 等参数在11.8中可用。对工具链不支持的选项，在构造命令前明确选择受支持的编译策略并记录身份，不先失败再静默删参数重试。两套工具链都必须保留 `--time` 与命令外层墙钟采集。

## 6. 只 JIT 所需曲线与位宽

### 6.1 选择单位与模块边界

“只需要的曲线与位宽”指先在主机上确定曲线族、Montgomery param、实际选中容器和执行方式，再触及相应设备模块。sigma 是运行时曲线身份，不应按每个 sigma 生成一个编译实例。

Stage1 的建议最小隔离单位为 `(curve, param或不适用, container_bits)`，必要时再按 binary/NAF/PRAC 等实际执行方式细分。一个模块可以包含该任务所需的多个准备、执行和收尾内核，但不能包含其他曲线族、其他 Montgomery 参数化或无关位宽实例。

- 继续以 `tiers.def` 为位宽/几何的唯一清单，生成或派生每个实例的构建条目；不手写第二套档位表。
- 将当前按七组划分的实例 TU 拆为按上述单位编译的独立模块。首选保留现有 Runtime API 和内嵌 fatbin 方式，减少对计算接口的影响。
- **TU 拆分只是候选实现，最终以链接后模块和驱动观测为准。** 若设备链接合并、全局变量、函数指针或 runtime 注册仍触发无关模块，则调整引用边界；必要时改用显式按需模块加载，只有证据证明满足要求后才定稿。
- 主机注册表应先用纯元数据选择，选定后才取 kernel/函数属性；避免全局初始化或逐档扫描时取得全部设备函数句柄。
- 正常启动、GUI 探测、`--gpu-info`、occupancy 估算、队列预读和 tune 都纳入检查。无具体任务的 GPU 信息展示不应为所有位宽查询真实 kernel；若确需全档诊断，改为显式操作并说明其开销。
- 保留现有容器选择、Edwards 一位余量、Montgomery 六位余量及受限构建的正确回退。不能为了少加载模块改变输入可用档位或资源判断。

Stage2 的设备算术与 NTT 目前集中在各自 TU，使用的模板/容量划分不等于 Stage1 容器表。实施时先梳理实际调度键：曲线相关点运算按实际曲线与容量隔离；公共模运算/NTT 按任务实际需要的容量、变换形状和执行路径组织，允许必需的公共模块。不能宣称 Stage2 也已按 Stage1 的32档隔离，更不能只验 Stage1 就判定完整发布满足 JIT 要求。

### 6.2 延迟加载与可验证性

使用延迟加载作为具备相应能力的驱动上的正常启动策略之一，在第一次 CUDA 初始化前建立所需设置，并记录实际加载模式。Lazy Loading要求R515及以上用户态驱动支持，Runtime侧的模块级优化在11.8增强；不能假定旧驱动识别 `CUDA_MODULE_LOADING=LAZY` 就实际启用了延迟加载。NVIDIA文档还区分模块与函数加载，且存在managed变量等例外；因此 **LAZY设置本身不是“只编译一个位宽”的证明**，见 [Lazy Loading版本要求](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-c-programming-guide/lazy-loading.html)。

旧驱动须按能力分支验收：

- 支持Lazy Loading时，按实际启用模式验证模块隔离和查询副作用；不得只看环境变量值。
- 不支持Lazy Loading时，不能依赖Runtime自动延迟。若该驱动具备本包所需PTX JIT能力且将纳入支持范围，必须证明显式按任务加载独立模块等方案能满足同一按需JIT硬条件；不能加载一个包含全曲线/全位宽的PTX再声称按需。
- 驱动不能解析11.8 PTX时，Maxwell没有可用的本包SASS，记录为不支持该PTX路径并明确要求升级；不以失败回退方式换算法或暗中增加SASS。Pascal有相容SASS时则独立验证该路径，不强制它走PTX。
- 显式加载候选方案须验证context/stream/Graph、函数生命周期及多worker的正确性，不把加载方式改变引入的计算或恢复错误当作性能问题处理。

测试构建需提供“请求任务 → 选中实例 → 实际触及模块/函数”的可核对记录；结合最终模块内容清单和驱动加载观测，检查未选实例没有进入 JIT。仅凭程序总耗时变短、kernel launch 列表或 CUDA 缓存文件数量不足以证明编译范围。若现有工具无法观测实际边界，先补可观测性，不把推测写成通过。

不要求用户安装 nvcc；发布包内携带所需 PTX/SASS，JIT 由驱动完成。开发机 `nvcc --time` 与驱动 JIT 是两套计时，不能互相替代。

## 7. 实施验证与 JIT 耗时采样

### 7.1 静态与不运行 GPU 的检查

- 验证无参数默认生成两个完整 profile，输出目录、组件名和版本来源正确；单 profile 选择与旧参数错误信息明确。
- 对两阶段所有实际模块检查 PTX `.version`、`.target`、SASS 架构和覆盖范围：cu11.8 只能保留 compute50 对应的 PTX（`.target sm_50`）及 sm60 SASS，cu13.3 只能保留 compute75 对应的 PTX（`.target sm_75`）及 sm80 SASS。逐模块记录实际PTX ISA版本，不从目标能力数字推断。模块数量可多，但每个实际运行入口都必须有对应镜像。
- 检查完整源码/依赖/对象/程序/资源身份、同包 vcpkg GMP 一致性、GUI 资源和空队列；缺组件、错 DLL、过期对象、错版本/profile 等负例必须拒绝。
- 检查计时文件并发隔离、重试隔离、失败保留、命令关联，以及仅改变计时 run-id 不引发 CUDA 重编译。保留严格 HostOnly/SkipBuild 和生产源码闭包限制。
- 改造现有 `test_cuda_p5_release.py`、`test_cuda_p5_images.py` 和 profile 路径检查对六架构的假设；不降低既有数学或身份门限，不改写冻结证据。

### 7.2 图像路径与按需范围

每套包分别执行默认模式、`CUDA_FORCE_PTX_JIT=1`、`CUDA_DISABLE_PTX_JIT=1`，每次使用清洁的子进程环境，清除互相冲突的 FORCE/DISABLE 覆盖。强制 JIT 与 SASS 禁用 JIT 的语义见 [NVIDIA CUDA 环境变量](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)。

- 默认模式确认真实选择；强制 PTX 用于验证两套 PTX 在现有 GPU 上的 JIT；禁用 JIT 只在存在相容 SASS 的设备上要求成功，不相容的测试明确记录预期拒绝。
- 覆盖 Edwards binary/NAF、Montgomery PARAM0/2/3；PRAC 仅在显式启用的构建中另验。完整位宽执行至少一次正确性及模块范围覆盖，边界输入检查容器切换和余量。
- 同进程连续测试：相同任务重复、相同容器换 sigma、同组相邻容器、跨组容器、Montgomery 换 param、切换 Edwards/Montgomery。仅触及新增任务真正需要的实例和公共模块，不做全量预热。
- 分别运行 CPU-only、纯信息查询、GUI v2 生成器/GPU 信息、实际 worker 启动、恢复及 Stage1→Stage2 交接，排查探测路径提前 JIT。
- Stage2 覆盖两种曲线输入、代表性容量、NTT 形状、恢复与 tune 路径；列出必需公共模块和不应触及的变体。多 worker/子进程还要区分每进程加载与驱动磁盘缓存复用。
- 当前可用设备不足以验证原生 sm60/sm80 或旧架构时，分别写明“镜像检查”“兼容 SASS”“强制 PTX JIT”和“未测原生设备”，不互相替代。现代GPU上的强制PTX测试不能代替Maxwell实体GPU或旧驱动测试。

旧包增加设备×驱动×加载能力的显式矩阵，以下均为**待执行要求**：

| 设备与驱动条件 | 默认与禁用JIT | 强制PTX与按需范围 |
| --- | --- | --- |
| Maxwell sm50/sm52，522.06候选基线及较新相容驱动 | 默认应走PTX；禁用JIT必须因无相容SASS而明确失败 | 两阶段数学、按曲线/param/位宽隔离及冷暖缓存实测；同包不出现sm50/52 SASS |
| Pascal sm61，522.06候选基线及较新相容驱动 | 验证实际使用sm60相容SASS；禁用JIT仍可完成必要计算 | 强制PTX另验，不能把默认SASS成功当作PTX已通过 |
| Pascal sm60，匹配的实体设备和驱动 | 单列sm60原生SASS验收；缺设备记未测 | 另验PTX，与sm61相容执行分开 |
| Maxwell/Pascal，452/472等拟覆盖的较旧分支，以及515至522.06之前的可用驱动 | 逐个记录具体驱动；有相容SASS时验证小版本兼容和实际API，不预设整个区间成功 | 不预设PTX成功；验证实际Lazy能力、PTX版本错误和不支持时的提示；可用PTX路径仍须满足按需硬条件 |

样本驱动必须实际支持对应卡型和操作系统。不得为在新GPU上安装不了旧驱动而替换成新驱动后仍标记旧驱动通过；缺少设备/驱动环境的格子保持未测。驱动切换由实施时安排独立测试环境，本规划阶段不改本机驱动。

对支持的PTX组合，比较默认加载、显式LAZY、EAGER诊断以及必要的显式模块加载候选：逐项记录实际模式、模块集合、JIT窗口和结果。EAGER用来揭示全量注册副作用，不能用“该驱动只支持EAGER”为理由放宽正式路径的按需JIT要求。对不支持PTX的组合，预期拒绝仅验证错误处理，不计为按需JIT资格通过。

### 7.3 JIT 计时边界

代表样例覆盖小、中、大位宽和不同曲线/param；两套 CUDA 包、Stage1/Stage2 分开记录。先完成全覆盖的范围与正确性，再对代表样例采样，避免每个探索样例都重复昂贵矩阵。

每个代表配置建议至少5次独立冷缓存进程、5次已有磁盘缓存的新进程，并在单进程内重复相同实例至少3次。报告样本数、中位数和最小/最大值；小样本不宣称稳定 P95。

1. **冷驱动缓存**：使用本次实验专属的新 `CUDA_CACHE_PATH`，记录 cache 开关和容量。每个冷样本独立目录，不删除用户全局 CUDA 缓存。
2. **磁盘缓存命中**：复用同一实验缓存启动新进程，确认没有因路径或身份变化变成新的冷样本。
3. **同进程重复**：重复当前实例；切到新曲线/param/容器时单独标记首次加载与后续命中。

分别记录进程启动至任务准备、CUDA context 初始化、实际模块加载/可能的 JIT、函数取得、首个必要 kernel 完成和整个任务墙钟。若采用显式 Driver API 加载，可读取其 JIT 时间选项；否则使用经过验证的 API 观测/剖析，并把无法单独分离的部分诚实标为“加载及 JIT 窗口”。整段启动时间或冷暖差值包含其他开销，不能直接改名为精确 JIT 时间。

编译 CSV 的阶段语义按对应版本核对，见 [NVCC 11.8 --time](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-compiler-driver-nvcc/index.html#time)、[NVCC 13.3 --time](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-compiler-driver-nvcc/index.html#time)。它们不提供上述运行时驱动 JIT 的替代测量。

每组结果冻结：精确输入 N、曲线/param/sigma、B1/B2、容器/实际位宽、执行方式、GPU/驱动、CPU、功耗/频率条件、工具链、源码与二进制哈希、缓存状态、加载模式、样本数、计时边界、退出状态和证据位置。固定或禁用非必要的自动分析、校准和 tune，涉及这些功能时单列场景，避免把准备开销算成 JIT。

旧驱动样本还记录Windows与驱动精确版本、Runtime/Driver API能力、实际PTX ISA版本、SASS或PTX真实选路、模块加载机制和原始CUDA错误。不同驱动或GPU的缓存不能当作同一暖缓存样本；无法使用较新剖析工具时采用该环境支持的观测方式，缺少可靠观测就保持范围资格待验。SASS路径的启动/加载时间单列，不记作JIT耗时。

必要时使用管理员 Nsight Systems 辅助定位；剖析结果与无剖析采样分开记录。原始命令、日志、trace、计时 CSV 和样本写入 `data/experiments/release_build_redesign/<run-id>/`；汇总写入 `data/benchmarks/`，图表写入 `data/figures/`。不写入 docs/data 等目录。

### 7.4 正确性与发布可用性

- 相同精确 N、曲线、param、sigma、B1/B2 下，对照冻结旧程序与两套新包，比较点/结果、因子合法性、保存/恢复和交接行为；不能仅以“没有崩溃”通过。
- 对适用的已知因子样例复核固定 B1/B2 条件，构建器和模块重组不能改变三大构造参数的含义。
- 在无源码目录、无开发 CUDA Toolkit 路径、带空格的独立目录启动 GUI v2，自动定位同目录两个 worker，并验证群阶助手加载本包 vcpkg DLL。
- 分别核对冷启动体验、缓存命中、稳定运行吞吐和完整构建墙钟；发布体积、编译命令数量与模块数量也一并记录，防止按档位拆分的额外成本被隐藏。
- 按需范围硬条件：未选曲线、param 和无关位宽模块无 JIT；必要公共模块单列。耗时改善不抵消范围或数学错误。
- 绝对 JIT 秒数和性能退化门限尚无本方案基线，列为 TODO；先冻结基线、测量噪声和目标，再确定门限，不预填收益百分比或放宽旧门限。

## 8. 建议实施顺序与完成条件

| 阶段 | 工作 | 进入下一阶段的依据 |
| --- | --- | --- |
| P0：冻结与预检 | 冻结当前程序/输入；核对11.8/13.3、compute50/75及sm60/80闭包、MSVC与命令选项；设计旧驱动矩阵和模块观测办法 | 目标编译可行性及限制明确；SASS/PTX与驱动能力边界明确；计时边界和比较对象冻结 |
| P1：统一计时 | 共享 nvcc --time helper；覆盖正式入口和所有仍维护的本地 nvcc 构建调用点；审计增量身份 | 调用点审计无遗漏；各实际命令有独立 CSV；失败可追溯；无改动构建不因计时路径重编 |
| P2：双 CUDA 构建 | 独立 PTX/SASS 字段，Stage1/Stage2 直接构建两套；脱离默认六架构裁剪逻辑 | 两阶段实际镜像精确符合两个 profile，严格复用与负例通过 |
| P3：按需 JIT | 按曲线/param/容量调整模块与主机选择；处理查询提前触及；覆盖Stage2公共模块及旧驱动有/无Lazy能力分支 | 支持组合的静态模块清单与动态观测一致，范围测试及数学验证通过；不支持组合明确记录 |
| P4：GUI v2 完整包 | GUI/助手改用同套 vcpkg 通用 GMP；统一 release 目录、资源、配置、干净 ZIP | 两个包独立运行，搬移目录和缺组件/错 DLL 负例通过 |
| P5：实测与正式文档 | 全位宽与跨阶段验证；旧GPU/驱动矩阵；冷/暖JIT、构建和运行性能；同步正式说明 | 实测证据齐备，按卡型/驱动/镜像路径给出资格与最低已验版本；未测环境不宣称通过 |

P0–P4 的新测试构建使用 `tools/build/test/`，正式 dev/local/release 行为通过 `tools/build/` 对外。实施中若变更方案，先在本草稿记录理由和验证，再按实际最终行为更新正式说明。

## 9. 正式实施后的文档同步与版本

本轮正式索引仅链接草稿；正式状态文档不提前改写为已实现。

- `docs/usage/BUILD.md`：完整发布入口、两套PTX/SASS、vcpkg GMP、旧驱动的SASS/PTX与加载能力边界、输出、计时和复用合同的权威说明。
- `tools/build/README.md`：精简入口和参数索引，链接 BUILD，不重复维护目标矩阵。
- `docs/architecture/STAGE1.md`、`STAGE2.md`，必要时 `NTT.md`：最终设备模块边界、元数据选择和实际按需加载职责。
- `docs/usage/GUI_V2.md`：完整包内相邻 worker、通用 GMP 群阶助手、资源与独立 GUI-only 候选的当前关系。
- `docs/performance/CUDA_STRUCTURE_QUALIFICATION.md` 或该主题的明确链接：新发布/JIT 实测范围、证据及未测硬件；不重写旧冻结实验来冒充新验收。
- `docs/README.md`：正式实施后按最终状态更新入口；开发草稿保留并提示用户处理。

纯规划不增加版本号。正式实施时按实际影响应用仓库规则：构建优化可酌情不改变主程序版本；若涉及主程序加载行为，至少按实际优化/修复范围评估 z；新增用户功能则评估 y。发布目录读取届时权威版本，不预先锁定下一个版本号。

## 10. 实施 TODO（本轮不视作已验证）

1. 核对11.8安装/补丁、MSVC和命令选项；验证所有Stage1、Stage2、NTT内核的compute50/compute75与sm60/sm80编译闭包，明确指令和运行库限制。
2. 确定链接后实际JIT模块粒度及可靠观测工具；验证Runtime API内嵌独立模块是否足够，以及旧驱动无Lazy能力时是否需显式加载；不以显式加载掩盖PTX ISA不相容。
3. 核对 vcpkg prefix、通用 x64 GMP 来源/构建配置、头/库/DLL/许可/对应源码身份，并让 GUI v2 助手真正消费该依赖。
4. 执行旧GPU×驱动×SASS/PTX×加载能力矩阵，核对两套包的Windows/驱动/CRT条件；分别给出Maxwell JIT、Pascal相容SASS、原生设备与最低已验版本，未测组合保留未测。
5. 完成本地 nvcc 调用点审计与计时 schema，验证并发、路径空格、失败重试、HostOnly、SkipBuild、增量复用。
6. 冻结同条件基线、代表输入和样本数，确定可接受 JIT 延迟、构建墙钟及稳定运行性能门限后采样。
7. 验证 GUI 探测、恢复、自动 Stage2、tune 和多 worker 都不引起无关实例提前 JIT。
8. 实施通过后同步正式主题文档；草稿随实施变更纳入提交，完成时提示用户决定后续处理。

## 11. 实施记录（2026-10-24）

用户授权后开始实施。主程序版本更新到0.52.1，GUI保留独立0.52.0。工具链核对为CUDA11.8.89/MSVC14.16.27023与CUDA13.3.73/MSVC14.51.36231；GMP为vcpkg通用x64-windows的6.3.0#3，冻结头文件、导入库、运行DLL及SPDX，禁止Zen3回退。计算基线0.52.0保存在`data/experiments/release_build_redesign/baseline_20261024/`，不替换其中程序/DLL。

已接入完整发布入口、组件身份记录、精确双镜像、Stage1逐曲线/param/容器模块、Stage2逐设备入口/模板模块、纯元数据枚举、默认LAZY、各维护构建入口nvcc独立计时、基础ELY GUI与通用GMP/对应源码打包。正式说明已同步实现方式，运行资格另以证据判定。

本轮证据根为`data/experiments/release_build_redesign/`，保留失败及重试。CPU合同5项通过（`cpu_contracts_final.log`），原有profile检查和54项构建身份变异检查通过（`profile_final.log`、`identity_final_retry.log`）。两版本Stage2均已完成全176设备模块编译，现代版本的大位宽初轮对照通过；生命周期修复后需用新身份重跑，初轮不作为最终身份资格。

旧工具链退出故障在有/无CUPTI下均以0xC0000374复现。独立带MAP链接副本与异常观察器将退出栈定位到`stage2_memory::ledger`的析构；真实头文件的双TU纯CPU测试同样失败（`ledger_lifecycle_red/result.json`）。改用inline函数中的单一局部静态生命周期后通过（`ledger_lifecycle_green/result.json`）；保留内存账本口径及检查，不取消析构、不关闭验证。两套Stage2通过正式构建入口增量重建。

CUPTI观察器只用于测试。退出时flush导致Windows加载锁挂起的原型已修正为launch API退出时flush；`observer_probe_v3/`同时观察到编译/缓存写入活动与唯一选中Montgomery param0/512模块。该小型构建只证明观测方式可行，不等于完整发布或其他档位通过；单次时间受并发构建影响，不作为性能结论。后续冷/热采样独立于范围观测，分别记录驱动活动与进程墙钟。

完整两套Stage1/Stage2编译及静态镜像清单已完成，四个组件分别严格符合PTX50/SASS60或PTX75/SASS80。发布产物为`release/ecm-elf-0.52.1/cu11.8/`、`cu13.3/`及各自ZIP/SHA256，GUI v2只带基础ELY，全部组件使用同一vcpkg通用GMP。

本机完整Stage1验证通过：旧包314条计算路径及2项元数据查询，现代包471条计算路径及2项查询，覆盖Edwards32档NAF/binary和Montgomery31档param0/2/3。两套独立Edwards GMP探针各131用例、537曲线通过；Stage2算术各1536模数、152064比较无错误。跨阶段、承载、relocation、预算回退、分段恢复与已知1019/2621因子对照通过。

复核时发现Stage2子进程复用模块ID会覆盖旧观察器文件，观察器改为PID/模块ID/事件序号命名后重新执行全容量范围矩阵：14输入、70路径、7130模块事件通过。此最终证据为`stage2_all_capacities_v4/`，之前的数学证据仍保留。现代包强制PTX变量在本机实际选择SASS的样例被负例门禁拒绝；PTX验证使用冻结对象去SASS、逐对象核对PTX原始字节相同的明确测试副本，未替换发布EXE。

源码归档曾误收录忽略目录中的本地测试输出，已保留首轮包为证据，改为Git源码枚举、私有测试目录排除与明确vendor纳入，并随源码保留`.gitignore`。6项CPU合同及最终包门禁通过。最终两个包在独立空格/中文目录、PATH仅System32下完成worker版本、GMP助手及GUI启动验证；实际缺worker和改动DLL负例在私有副本拒绝，原包未变。

80个独立冷/热进程样本完成，原始记录为`runtime_final_sampling/`。Stage1冷JIT中位数11.8为1.3104秒、13.3 PTX副本为0.7272秒；Stage2分别9.6077、9.5718秒；各5个热样本记录缓存活动、无重新编译。进程墙钟、输入、计时边界、设备条件及完整覆盖以正式[发布/JIT验证](../performance/RELEASE_JIT.md)为准；不宣称稳定吞吐收益。

Maxwell/Pascal、sm60/sm80实体GPU和旧驱动本机不可用，继续明确未测，不宣布旧驱动最低已验版本。原现代包在未来GPU自动回退、旧驱动/EAGER加载范围、目标负载的JIT预算和稳定吞吐门限保留待验。草稿随实现纳入提交，完成后由用户处理；正式使用方式见[构建与发布](../usage/BUILD.md)。

## 12. 发布目录与分版本启动入口（2026-10-24）

按用户要求，后续发布直接输出 `release/ecm-elf-<版本>-cu11.8/` 与 `ecm-elf-<版本>-cu13.3/`。ZIP 内唯一顶层目录采用相同完整名称，避免解压后只得到 CUDA 版本目录；主程序版本仍从权威版本头读取，不固定为示例0.52.1。ZIP及SHA256位于release根，各包的目录、ZIP或SHA256已存在时拒绝覆盖；两种profile可以先后发布到同一输出根。现有冻结产物和证据不迁移、不重写。

新增 `tools/build/build_release_cu11.8.bat` 与 `build_release_cu13.3.bat`，分别固定传入对应profile和16个并行任务，保留参数转交、无参数暂停和退出码。总入口及两阶段release PS1默认Jobs同步为16，Stage2仍保持SplitCompile=1。需要其他并行数时直接调用PS1。双包总入口仍可用，不变更本机/开发默认并行数，不修改主程序版本。

打包manifest更新为schema 2并记录实际目录/ZIP名；完整包检查器验证新的压缩成员根目录，同时继续支持schema 1冻结包。`tools/test/test_release_package_layout.py` 使用私有假组件执行真实打包和静态包门禁，覆盖双包、先后单包发布、解压根目录、拒绝覆盖目录/ZIP/SHA256，以及两个BAT实际传参和退出码；测试不编译或运行计算worker。

验证结果：新增布局测试4项、现有 `tools/test/test_release_build_contracts.py` 6项全部通过；三个公开release PS1语法解析无错误，Jobs默认值均为16。本轮未重新编译计算程序或重打包现有发布产物。正式说明已同步到BUILD及构建工具README，草稿保留供用户处理。

## 13. Stage1本机拆分与Stage2聚合默认（2026-10-24）

用户确认Stage2编译不是瓶颈，授权release默认回到与local相同的算术/NTT两个聚合CU，保留176模块拆分为显式备选；同时授权为Stage1本机编译长尾调整代码架构。Stage1 local/dev默认共用release的曲线/param/容器模块生成器，保留 `KernelLayout=grouped`。生成文件仍包含内核及完整设备辅助函数实现；不把单个算术调用链拆成跨对象的设备函数。无需为本次构建策略改动增加主程序版本。

Stage2统一用 `KernelLayout=grouped/isolated` 选择，完整发布入口对应 `Stage2KernelLayout`。默认cu13.3聚合使用nvcc内部SplitCompile=8；cu11.8不支持该选项，保持1；隔离备选保持1。布局进入release profile和manifest，隔离宏、模块生成与常量快照只在显式isolated时启用，SkipBuild严格检查所选布局。打包对聚合模式检查两个设备对象及权威入口/实例成员，对隔离模式检查逐模块成员；元数据枚举不生成隔离源码。

按本次批准变更，Stage2 grouped不再提供按容量隔离JIT保证。运行范围测试必须显式选择 `--allow-grouped-stage2`，报告保存布局；已有0.52.1隔离构建的JIT资格和耗时不能移用于grouped。原冻结包、日志和设备范围证据不变。Stage1按需模块范围要求继续保留。

本次证据写入 `data/experiments/cuda_layout_20261024/`。真实构建初次在受限环境出现CMake空C测试C1902及nvcc主机探测崩溃。最小独立CUDA样例证实相同命令stdout直接写文件返回5、经PIPE采集返回0；`run_compile_jobs.py`改为管道流式落盘，保留退出码和有界内存。Windows stdout句柄回归先观察到FILE类型1导致失败，修复后PIPE类型3成功，并验证非零退出码8保留。CMake空C探测另在正常本机环境重跑，不绕过编译器能力检查。

实际构建验证覆盖Stage1的512、13312、17408容器及Stage2两套CUDA聚合release。此代表性构建不作为全档构建加速比，当前并发负载也不能用于稳定吞吐结论。旧Stage1分组profile不自动重标或覆盖，首次切换使用新BuildDir。

验证结果：Stage1通过正式local入口构建12个CUDA单元，最终EXE有12个cubin、无PTX；同参数HostOnly复用12个单元、CUDA重编数为0（`stage1_build_verified.log`、`stage1_host_only.log`）。复用真实Edwards对象的CPU独立对照9项通过，覆盖三个容器及NAF/binary、边界、分段恢复和自适应重启（`stage1_probe_math.log`）。Montgomery param0/2/3在512、13312两个容器各4曲线，共24条末点与冻结0.52.0基线相同；源码、命令、输入及EXE摘要见私有 `verify_workers.py`、各case目录和 `worker_summary.json`。

Stage2 cu11.8/cu13.3均通过正式release阶段入口及源码/编译器/GMP/对象身份核对。两者各包含112个算术入口和64个NTT入口，静态目录、PTX/SASS及EXE/manifest摘要见 `stage2_cu118_static.json`、`stage2_cu133_static.json` 和对应modules报告。cu13.3独立GMP算术探针1536个模数、152064次比较、错误0；8193位完整worker结果及目标叶指纹与冻结样例一致，2048项Montgomery必需自检、672项S4自检和118项GMP检查无错误，异步检查pending=0。完整worker的debug诊断保存于debug.log；运行检查器同时读取原始stdout与debug文件，保留原数学及指纹门禁，不因日志分流放宽检查。

cu11.8聚合版的同一8193位完整worker也完成，结果及目标叶指纹与冻结样例一致，2048项Montgomery必需自检通过、pending=0，见 `stage2_worker_cu118/summary.json`。该次私有冷缓存运行初始化等待较长；未单独观测驱动JIT窗口，不把整段曲线耗时称为纯JIT，仍保留grouped冷/热预算待验。两套完整worker均在GPU1 RTX4060 Laptop、sm89、610.88驱动验证，设备快照为 `device.csv`；CPU为Ryzen AI9 HX370，未锁定功耗/时钟，期间存在并发构建和其他GPU上下文。该条件不替代旧GPU、旧驱动或原生sm60/sm80资格。

CPU回归为构建/计时合同7项、完整包布局6项、profile隔离33项通过；另有10个PowerShell入口/辅助脚本语法检查通过。隔离备选的生成成员、稳定生成、只读元数据、编译宏、profile分离及打包均有CPU回归。旧GPU/旧驱动、全档构建加速比及grouped的冷/热JIT预算仍待相应条件验证；未重打包原冻结发布目录。草稿保留，由用户决定后续处理。
