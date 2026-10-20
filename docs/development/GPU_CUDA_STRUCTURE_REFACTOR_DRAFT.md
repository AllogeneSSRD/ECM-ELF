# CUDA 结构重构草稿

P5验收单独记录于[CUDA P5验收草稿](CUDA_P5_QUALIFICATION_DRAFT.md)，当前有效覆盖和证据见[CUDA模块验收](../performance/CUDA_STRUCTURE_QUALIFICATION.md)。用户已将P5范围限定为本机LOCAL构建、完整位宽功能与性能，多架构发布暂缓；下文P0～P4各节保留为临时开发记录。

## 范围与约束

按已确认方案分 P0～P5 实施。P0（0.44.1）完成构建源清单、依赖与身份校验，P1（0.44.2）统一位宽清单并拆分 Stage1 主机与设备职责，P2（0.44.3）拆分 Stage2 算术与 NTT 底层模块，P3（0.44.4）拆分 Stage2 主机流程、资源管理和校验。本轮实施 P4：删除已证明不可达的旧分支与无用代码，完成实验消费者隔离；版本只递增修订号至 0.44.5。算法、配置、存档及程序输出语义保持。

P0 基线为 `36171f4`（0.44.0），当时的 TOML、队列及输出行为作为基线。478 个已跟踪 `src/`、`kernels/`、`include/` 文件的原始摘要保存在 `data/experiments/cuda_structure_p0_20261020/baseline.json`。P0 除版本头外未修改这些文件；P1 的迁移与验证另见下节。

## P0 实现

- `tools/build/internal/cuda_sources.cmake` 唯一维护 Stage1 共用源、CUDA 源、可选 PRAC、Stage2 主机源及生产/开发引擎源，保持原编译和链接次序。CMake 直接使用，PowerShell 每次导出到构建目录中的 `cuda_sources.json`，不维护第二份手抄清单。
- CMake 导出真实目标依赖图；对象收集、完全重编与并行编译可覆盖链接的对象/静态模块。未变化的算法编译宏和宿主身份 sidecar 保持。
- `cuda_build_state.ps1` 扫描当前本地递归包含关系，并用 NVCC 编译依赖文件补足宏包含及 SDK 依赖。条件包含按保守全集处理；生产闭包拒绝实验源码。响应文件、命令、依赖内容、对象摘要共同决定复用，不依赖 mtime。
- Stage1 发布身份改为所选目标源闭包，包含 `.inc/.inl/.def` 和链接主机模块的私有包含文件，并冻结 CMake 编译命令与目标图；PRAC 源随显式开关加入，保留既有构建和发布能力。
- Stage2 本机/开发/发布消费相同源清单；HostOnly 校验完整 CUDA 源集、每个编译单元及对象；普通复用与 SkipBuild 同时核对 exe、源成员和对象，不能遗漏新依赖或通过替换对象绕过。
- 保持现有无重名对象的文件名；未来同名翻译单元使用区分目录的对象名。旧 manifest 缺少依赖身份时要求正常重建一次。

## 验证记录

P0 分为源清单与 CPU 故障注入、原生构建、复用/发布校验及短曲线回归四步，前一步通过后进入下一步。所有新证据写入上述实验目录；失败输出保留。

- `identity_01`：测试辅助函数名碰到 PowerShell 的 `Write` 别名，未实际写入夹具，失败保留；已改用独立函数名。
- `identity_02`：暴露 PowerShell 5.1 对 JSON 数组输出的包裹差异，目标闭包筛选失败；修正数组读取后，`identity_03` 的 28 项 CPU 故障注入门禁通过。
- 指纹格式改为 schema 2：UTF-8 长度分帧、路径按 ordinal 排序，避免 Windows PowerShell 5.1 与 PowerShell 7 的 JSON 转义/排序差异。`identity_cross_shell_v2.json` 与 `identity_10/result.json` 核对跨运行时一致性。
- `identity_10`：37 项 CPU 门禁通过，覆盖递归 `.inc/.inl/.def`、保留 mtime 的内容变化、响应文件、宏包含、带空格和 `#` 的依赖路径、被移除的旧编译器依赖、源成员遗漏、命令/对象篡改、多 CUDA 编译单元、生产包含实验源码的拒绝，以及链接主机模块的私有依赖。夹具使用独立目录，不改仓库算法源码。
- `inventory_parity.json`：Stage1 共用、CUDA、PRAC 及 Stage2 源清单与 0.44.0 的成员和次序完全相同。`target_graph/build_02/cuda_target_graph.json` 验证 CMake 对对象库、条件链接和传递静态库的保守收集。
- 原生工具链为 CUDA 13.3、MSVC 14.51.36231，架构 sm89。Stage1 本机/发布验证限制为 `Tiers=384`、PRAC 关闭；未声称完整位宽、旧 CUDA 或全部发布架构通过。`stage1_build_v2.log`、`stage2_build_v2.log`、`development_build_02.log` 分别记录 Stage1、生产 Stage2、开发 wrapper 构建通过。
- `stage1_incremental_02.log`：21 个 CUDA 对象全部复用，实际重编 0 个；`stage2_hostonly_01.log`：校验后复用 CUDA 对象，重编 4 个主机文件；`stage2_reuse_01.log`：源、依赖、对象和 exe 相同，完整复用。`phase_deps_04/result.json`：改写临时 manifest 的 phase 头摘要后，HostOnly 在编译前拒绝，原构建目录未变。
- `runtime_04/summary.json`：冻结的 0.44.0 与最终本机构建在 GPU 1（RTX 4060 Laptop）对照，Stage1 5 组、每组 4 条曲线（Montgomery PARAM0/2/3、Edwards binary/NAF）存档字段一致；Stage2 5 组（Montgomery 普通、Edwards 普通/承载/relocation/主机回退）目标环叶指纹和非计时结果字段一致。输入、完整参数与二进制 SHA256 随各用例保存。只排除版本、源构建身份、时间及相应校验字段；没有性能采样或性能结论。
- `edwards_probe_01.log`：独立 GMP 点运算参考的 3 组用例通过，每组 33 条曲线，包含续算、自适应重启；`relocation_geometry_01.log`：157752 项检查，bad=0。
- `source_freeze_01.json`：478 个已跟踪算法/接口文件中只改版本头，另外 477 个摘要与基线相同。开发 wrapper 额外补上已使用的日志声明头；`development_baseline_02.log` 用原编译参数复现缺声明错误，未改设备算法。
- `release_build_02.log`、`release_skip_split_package_01.log`：通用 x64 GMP 的 Stage1 sm89/384 档发布构建、SkipBuild、21 个 CUDA 对象拆分、对象/exe 还原和隔离 ZIP 打包通过；`release_identity_02.json` 确认拆分还原后身份仍成立，并拒绝对象摘要错误、源摘要遗漏和 exe 摘要错误。首轮手写真实构建故障注入夹具的属性索引错误，说明保留在 `release_identity_fixture_01_failure.txt`，修正后才记为通过。
- `stage2_release_package_01.log`：通用 x64 GMP 的生产 Stage2 sm89 发布构建和隔离 ZIP 打包通过。`stage2_release_skip_01.json`：SkipBuild 身份通过，已有配置与队列保持原摘要，ZIP 使用默认模板且不含内部 manifest。

## 验证边界与保留的失败证据

沙箱内的 MSVC 自检因 PDB 环境报 C1902（`stage1_build_01.log`），原生工具链构建通过。`phase_deps_01`～`03` 是辅助进程的执行策略、模块路径或运行时定位问题，`phase_deps_04` 使用已定位运行时通过；未放宽身份条件。

`runtime_01` 的 PARAM2/3 以 M127 输入运行受限 384 档，触发当前实现精确档位选择后找不到 192 档的限制；改用需要 384 容器的 256 位输入进行双版本对照。`runtime_02` 的原始叶指纹一致，结果比较器把嵌套计时当成稳定字段，修正比较边界后 `runtime_03`、`04` 通过。本轮未修改上述档位选择实现。所有失败输出继续保留。

## 后续

P0/P1 已通过各自限定范围的分步验证；P2 见本文对应章节。保留本草稿并提示用户处理，不自动删除。正式构建说明只记录已验证的当前行为；原始源码快照、失败夹具和日志不更改。

## P1 范围与验证顺序

基线为 `c0c4c74`（0.44.1），源摘要与原二进制冻结在 `data/experiments/cuda_structure_p1_20261020/`。原方案第二批包含：中立位宽清单、Montgomery 七组实例、Edwards 主机循环迁出核目录，以及 PRAC 独立可选目标。不进入 Stage2 算术/NTT 或流程拆分，不清理其旧分支。

先验证清单生成的 full/dev/限制档/别名/实例组与边界；再验证主机/核迁移及实际注册；最后进行独立算术、停止恢复、双版本存档/Stage2 回归和发布构建。每步通过后才继续。Montgomery 的六位余量、Edwards 的一位余量、Stage2 按实际字数选择容量、PRAC 显式启用与全部配置默认保持。

## P1 实现

- 中立接口位于 `include/ecm/cuda/`，`tiers.def` 统一 32 档的位宽、TPI、七组归属、Montgomery/dev 支持、TPB 和旧别名；Stage2 仅消费同文件的七档容量宏，实际 W、Montgomery radix 和分派顺序保持。PRAC 特殊 TPI/寄存器组合与默认网格分开标记，策略实现不变。
- Montgomery 主机入口迁至 `src/cuda/stage1/montgomery.cu`，核发射移入薄包装；PARAM0/2/3 各由七个实例单元编译。Edwards 校验/分派入口使用普通 C++，binary/NAF 的分配、GMP 编解码、事件、切片和回调归同一主机运行时。设备目录只保留算术、模板实例及发射器；缓冲布局与同步/发布顺序保持。
- 可选 PRAC 的设备源码迁至 `kernels/cuda/optional/prac/`，用独立 `ecm_cuda_prac` 对象目标显式链接；主机运行时迁至 `src/cuda/stage1/`。继承实际主目标的宏、编译选项、包含目录、语言标准与架构；不引入 RDC/device link。P0 目标闭包同时收集主目标及 PRAC 对象。
- 公共设备原语在 `kernels/cuda/shared/`；主机兼容 shim 位于 `src/cuda/common/`。CGBN 自身要求 NVCC 主机解析时存在 GMP，唯一适配头 `cgbn_types.cuh` 以主机条件包含满足；项目核接口不传 GMP、回调或检查点对象。探针的引用与 worker 对象复用同步更新。

## P1 验证证据

证据根为 `data/experiments/cuda_structure_p1_20261020/`。CUDA 13.3 / MSVC 14.51.36231 / sm89；运行使用 GPU1（RTX 4060 Laptop），未改变设备功耗/频率。以下均为正确性和构建验证，没有性能采样或提升结论。

- `tiers_04/result.json`：16 种真实 C++ 清单配置通过，覆盖完整/开发/限制/空档、旧别名、七组、PRAC 区间和 TPB 覆盖；检查 Stage1 余量边界及 Stage2 实际字数 1…257 的容量选择。
- `identity_02/result.json`：37 项 CPU 身份故障注入及 Windows PowerShell 5.1 / PowerShell 7 指纹一致性通过。`structure_final_02/result.json`：55 个项目设备闭包文件无反向主机/实验依赖；四段 Montgomery、Edwards、NAF、PRAC 设备算术正文与 `c0c4c74` 去注释/空白后完全相同。
- `step2_build_03.log` / `stage2_build_02.log`：迁移后的默认 Stage1（31 CUDA TU、384 档）及最终生产 Stage2 构建通过。`worker_reuse_01.log`：31 个 CUDA 对象全部复用，重编 0 个；`worker_objects.json`：收集十个 Edwards 对象，包含普通 C++ 前端。`stage2_hostonly_01.log`：身份匹配，程序整体复用。
- `step2_sass_02/result.json`：384 档六个设备函数的 SASS 指令正文与冻结基线相同；仅对函数名中 NVCC 因源路径产生的匿名命名空间文件摘要做规范化，保留模板签名并拒绝规范化重名，未忽略指令差异。
- `wide_build_01.log` / `registry_wide_01/result.json`：384/1024/4096/6144/8192/12288/16384/17408 档跨全部七组构建通过，68 项实际工厂、边界、余量、TPI/TPB 查询通过。`curves_wide_01/result.json`：七个 Montgomery 档各 PARAM0/2/3，共21组、每组2曲线，与冻结的完整 0.44.0 二进制逐字段存档一致；输入位数 W−6，模数、命令、原日志、二进制摘要均随例保存。P0 已证明 0.44.0 与 0.44.1 的相同算法基线。
- `point_small_01.log`：384 档三种 Edwards 模式各33曲线。`point_wide_01.log`：八档各 binary/NAF4/NAF5，共24组独立 GMP 末点、点不变量、固定停止/恢复与自适应重启检查通过。`host_wait_build_01.log` / `host_wait_run_01.log`：迁移后的诊断仍可编译，短调用点不变量通过；原始计时不用于性能结论。
- `prac_build_01.log` / `prac_identity_02.json`：显式 PRAC 构建通过，真实目标图收集42个 CUDA 对象，根/子目标五组编译宏/选项与 CUDA 包含响应文件内容一致，PowerShell 7 验证真实构建身份通过。`prac_runtime_03/result.json`：16组 CPU/GMP 与 ladder/resident/PRAC 完整点、正常恢复、双代损坏拒绝及干净目录重试检查通过；损坏检查点原始字节保留，损坏计划缓存重建后点结果一致。
- `final_runtime_02/summary.json`：最终 Stage1/Stage2 与冻结 0.44.1 比较，5组 Stage1 存档及5组 Stage2 目标环叶指纹和非计时结果字段一致，覆盖 Edwards NAF/binary、普通/承载/relocation/主机回退交接。排除版本、构建身份、计时及关联校验字段，保留其余存档/结果字段比较。
- `dev_build_01.log` / `registry_dev_01/result.json`：384/1024/1280 限档且 `ECM_CUDA_FULL_BUILD=OFF`、`ECM_NO_PARAM2=1` 的真实构建与28项查询通过；Montgomery 停在1024、Edwards 保留1280，PARAM2 明确拒绝。
- `release_build_01.log` / `release_split_package_01.log`：通用 x64 GMP 的 sm89/384 档显式 PRAC 发布构建通过，SkipBuild 身份、42个 CUDA 对象拆分、原对象/exe 还原及隔离 ZIP 打包通过。

提交前清理了新接口头文件末尾的多余空行，`worker_reuse_final.log` / `wide_build_final.log` / `dev_build_final.log` 各只重编1个受影响主机 CUDA 单元；`prac_build_final.log` 重编11个依赖该接口的单元，其余对象通过内容身份复用。设备算术正文校验及上述最终双版本回归重新通过。`final_static_checks.json` 记录11个 PowerShell 脚本语法、生成配置一致性和提交差异空白检查通过。正式发布按最终源码重新构建、拆分和打包通过，结果见 `release_final.log`；`release_restored_identity_final.json` 核对42个原对象/exe 完整还原、真实源身份与根/子目标编译设置，`release_archive_final.json` 核对 ZIP 的8个分发文件且无内部 manifest/运行数据。最终包的16组独立点/恢复/损坏拒绝验证见 `prac_packaged_runtime_final/result.json`。

## P1 失败证据与边界

`tiers_01.log` 保留 Windows PowerShell 丢弃空字符串实参导致的测试夹具失败，比较器接受无实参表示空期望集合后重新验证；未放宽空档必须为空的判断。`step1_build_01/02` 暴露从旧工厂复制的禁用占位定义与新工厂重复，删除重复占位后通过；原始 NVCC 日志均保留。

`step2_build_01` 的 SDK 包含目录被错误同时传给 CUDA，导致保守依赖扫描进入未启用的 RTC 分支；修正为仅普通 C++ 需要的显式 SDK 包含后通过，扫描规则未放宽。`step2_build_02` 触发上游 CGBN 主机 fallback 的 GMP 要求，增加上述窄适配后通过。`step2_sass_01` 仅因 NVCC 路径摘要导致函数名不同，原指令与失败结果保留；比较器的最终边界见上文。

`prac_runtime_01/02` 的夹具沿用旧检查点名称和损坏后自动重建预期，与当前二进制批次轮换合同不符；按现有主/备两代格式改为损坏拒绝并保留证据，重新验证通过，未修改程序恢复行为。首轮 PRAC 编译参数比较把目标各自的响应文件路径误判为包含内容差异；最终逐字比较响应文件内容。

本轮验证了上述选档与 sm89，不宣称完整32档、全部 PRAC 特殊策略或多架构发布均已验收。受限构建 PARAM2/3 精确选档后缺失实例的行为保持。Stage2 主体拆分、旧分支删除及全部探针隔离属于 P2～P4；主机等待诊断仍直接包含私有运行时，保持现有诊断能力，后续改为独立探针接口。P5 的完整矩阵与性能资格未实施。草稿保留，完成后提示用户处理，不自动删除。

## P2 范围与实现

P2 基线为 `5e3ed63`（0.44.2），575 个源码/构建输入摘要及原二进制冻结在 `data/experiments/cuda_structure_p2_20261020/`。本轮版本为 0.44.3，只增加修订号。按算术、NTT、消费者/正式构建的顺序验证，不进入 P3 流程拆分、P4 旧分支清理或 P5 性能资格。

- Stage2 点、baby/giant、S4、pack、系数搬运与缩放移至 `kernels/cuda/stage2/`，薄启动接口位于 `include/ecm/cuda/`。设备常量和上传仍由同一个算术 CU 所有；七档容量与实际 W/radix 不变，实例组合从原二进制实际设备符号核对。
- NTT 的 shape、runtime、Graph、checks 分为四个普通 C++ 模块，设备变换、tile/outer、carry、tune 校验移至一个设备 CU。域原语可供实验探针复用。单一设备 CU 保持运行时后端常量的单一所有权，不增加 RDC 或额外上传。Arena、Graph 字段和默认值、分配/释放顺序、检查、错误回退、同步和计时边界保持。
- 主流程及私有 host 包装改用中立启动接口；共享内存台账迁到 `src/cuda/common/`，保留原 C++17 inline 状态。生产构建共 11 个单元，CMake 与 PowerShell 消费相同算术/NTT 源组。CMake 的 SDK 显式包含只用于普通 C++。
- 删除被迁移的旧实现头，不保留转发壳。giant 探针直接包含纯设备模块；使用生产 NTT 的探针链接显式模块清单，并保持各自的默认流 ABI。Graph 和独立算术探针可验证身份后复用真实生产对象。独立实验树/NTT 引擎保留，开发 wrapper 只增加所需纯校验核依赖；未合并两套引擎。
- 当前模块职责及依赖方向写入 [Stage2](../architecture/STAGE2.md#cuda-模块边界)、[NTT](../architecture/NTT.md#代码入口与依据) 和 [构建](../usage/BUILD.md#cuda-源清单与依赖身份)。CPU 内存模型的现有源码提取测试适配声明/定义分离，产品构建和 GPU 消费者不截取实现源码。NTT 计数测试的旧 INI/结果读取方式按已有 TOML/紧凑结果格式修正，没有修改程序格式或门限。

## P2 验证证据

以下路径相对于 `data/experiments/cuda_structure_p2_20261020/`。CUDA 13.3 / MSVC 14.51.36231，构建及运行验收为 sm89，GPU1 RTX4060 Laptop；未改变功耗或频率。tune 和探针的计时仅为功能输出，不构成性能样本或性能结论。

- `structure_final/result.json`：93 个迁移算术函数正文及完整纯算术头与基线一致，16 个算术/NTT 设备文件无 GMP、检查点或反向主机/实验依赖；主流程和 NTT 主机模块不定义设备核。`host_body_final/result.json`：70 个 NTT 主机函数、24 个 Arena/Graph 方法、两类字段/默认值及 outer 启动帮助函数一致。比较保留字符串、检查、启动参数和同步顺序，仅规范化薄启动接口。
- `arithmetic_gmp_final.log`：直接链接最终生产算术对象，七档容量、114 种通用/梅森模数、实际字数与尾位边界，共 11,286 次独立 GMP 比较，bad=0。`giant_gmp_01.log`：实际 W=3/64/65/128/129/256 的 chain 与独立 GMP 对照通过。`reverse_final/summary.json`：216 个反转、补零和同一分配内保护区用例通过。
- `graph_gmp_final.log`：368 个完整整数卷积及 14,779,392 个 digit 检查通过，8 次 Graph 构建、16 次命中、8 次失效；workspace 的 144 项检查、14 次根表寿命调用和 72 个容量场景通过，最终无泄漏。`graph_fault_01.log` 保留 bad=1、退出码3的预期拒绝。
- `outer_check_01.log`：radix5…8 的 96 个 GMP 场景及缓存模式切换通过，并完成 3024 个主机、200000 个设备算术用例和 100064 个折叠对照。`pack_check_01.log`：663 个布局场景、38,030,604 字及 1,048,576 个除法检查通过；`pack_fault_01.log` 的错误补零被拒绝。生产 NTT 的 outer/pack、Tensor length/concurrency 及独立实验 Tensor 探针均可构建；`tensor_concurrency_check_01.log` 的长度与并发检查通过，live_bytes=0。
- `ntt_memory_01/checks.json`：681,389 项 CPU Arena 检查通过。`ntt_events_02/checks.json`：173 个场景、47,936 个内存事件通过，错误释放顺序被拒绝。`s4_memory_final/checks.json`：299,592 项检查、2,136 个分配事件和 192 个树租约通过；保持 96 窗口强制自检合同。
- `stage2_final_build_03.log` / `cmake_stage2.log`：最终生产 11 个单元分别通过原生入口与真实 CMake 目标构建；后者直接用 MSVC 编译 NTT CPP。`cmake_runtime_final/summary.json` 的5组存档/5组 Stage2 对照同时通过。`development_build_03.log` / `development_hostonly_final.log`：独立开发引擎及最终对象身份复用通过。`hostonly_final.log`：生产三份 CUDA 对象和整体身份匹配，程序完整复用。`identity_committed/result.json`：38 项 CPU 故障注入及 PS5.1/PS7 指纹一致性通过。
- `stage1_build_01.log`：0.44.3 Stage1 的 384 档、31 CUDA 单元构建通过。`packaged_runtime_final/summary.json`：最终 Stage1 与正式 Stage2 包对照冻结 0.44.2，5 组规范存档及5组目标环完整叶/非计时结果字段一致，覆盖普通、Edwards、承载、relocation 和主机回退。
- `wide_inputs/fixtures.json`：五个宽输入的 Stage1 经独立 Python/CPU 与 GMP-ECM 核对。`native_matrix_final_02/summary.json`：8193/16381/16384 位、两类非单位、宽输入主机回退及关闭融合 pack，共7组旧/新生产对照通过；完整目标环叶和稳定结果字段一致，必需自检与真因子校验保留。
- `ntt_counts_final/summary.json`：11 项计数开关、worker/CLI 覆盖、多曲线重置及非法值拒绝通过。`ntt_tune_final_03/summary.json`：长度 2^3…2^16、batch1/3，共28组旧/新 tune 形状通过 GMP 校验；调度、payload、验证字段一致。保留当前 TOML profile 与原始 JSONL 证据布局。
- `release_build_final.log` / `release_skip_final.log`：通用 x64 GMP 的最终 sm89 发布与 SkipBuild 身份核对通过。`release_archive_final.json`：已有 TOML/队列逐字保留，ZIP 的8个分发文件来自干净暂存区，不含用户配置、队列内容、内部 manifest 或运行数据。

## P2 SASS 差异、失败证据与边界

`arithmetic_sass_01/result.json` 和 `sass_final/result.json` 的严格指令比较均返回差异：176 个函数无增删，161 个完全相同，15 个算术函数不同。差异为 mont test 的 NW8/16/32/64/128/256、plain scale 的七档，以及 NW128 的 block product 和 baby product。可见寄存器分配及指令数变化，算术正文完全相同；具体编译器成因未确认。NTT 拆分未新增指令差异。没有放宽比较器、忽略指令或改动算法强求 SASS 相同；这15个函数的实际生产对象均由上述独立 GMP 矩阵覆盖。P2 不保证性能不变，P5 仍须完成性能资格。

首轮算术构建暴露辅助函数顺序，NTT 首轮暴露默认参数重复和 MSVC 启动参数收窄检查；修正声明/定义与显式启动转换后通过，原日志保留。首次身份门禁仍假定单 CU，按新清单核对精确成员；探针构建器的清单层级、CPU 测试声明次序和开发 wrapper 校验核依赖错误均已修正并重跑。实验树文件恢复原 CRLF，仅保留一条必要包含，避免整文件格式改写。

`ntt_counts_01/02` 分别保留旧 INI 参数及旧结果读取断言失败；当前 TOML、结果归一化和全部原门禁通过。宽矩阵首次采集触发 Python 4300 位十进制限制；仅对已限定的宽输入调整采集器后重新完成，未修改程序。tune 采集器原先误认控制台 JSON/单次 complete，按冻结旧版已存在的 TOML + 原始 JSONL、逐形状 complete 合同核对后通过。ZIP 首次校验误把用户现有配置当作归档模板，按现有干净暂存合同核对后通过。原失败日志与样本保留，没有改写原始实验结果。

P3 的主机流程拆分、P4 的剩余实验消费者/旧分支整理和 P5 的多架构、完整矩阵及性能资格未实施。当前验证范围不等于全部 GPU 架构验收。草稿保留并随提交保存，后续处理由用户决定。

提交前清理了新设备头的末尾空白，并精确保留实验树原有混合换行，仅增加一条包含。`structure_commit` / `host_body_commit` 重新通过；`stage2_commit_build.log`、`development_commit_build.log`、`release_commit_build.log`、`cmake_commit_build.log` 更新最终构建身份。`arithmetic_gmp_commit.log` / `graph_gmp_commit.log` 对最终生产对象重新完成上述 GMP/Graph 门禁；`packaged_runtime_commit` / `cmake_runtime_commit` 各重新完成5组存档与5组 Stage2 对照。`hostonly_commit.log`、`release_skip_commit.log`、`release_archive_commit.json` 核对最终复用及打包身份。`sass_commit/result.json` 的176个设备函数与格式清理前逐函数完全相同，和0.44.2的15个差异保持原样。最终二进制、DLL、清单和发布包冻结在 `committed_binaries_02/`；`final_static_checks.json` 记录9个 Python 文件、12个 PowerShell 脚本、配置生成物与暂存空白检查通过。

## P3 范围与实现

基线为 `d19123b`（0.44.3），本轮版本为0.44.4。证据根为 `data/experiments/cuda_structure_p3_20261020/`，`baseline/` 保存源码摘要及 P2 冻结的本机、发布、Stage1 和开发二进制。先核对 P2 可复用身份，再完成整段源码迁移与资源证明；首批构建、SASS 和存档/交接回归通过后，才分离计划查询并收紧模块依赖，最后进行原生扩展、发布及消费者验收。

- 生产引擎拆为15个普通 C++ 单元：context、options、resources、plan、engine、batch、points、product、tree、remainder、s4、factors、checks、diagnostics、tune。删除已完整迁移的主流程 CU 及 baby/frontier/packing 实现头；保留原有函数、检查、可选分支和旧模型，不进入 P4 的死代码清理。权威清单增加 `STAGE2_ENGINE`，生产共25个 TU，其中只有算术与 NTT 两个设备 CU。
- 私有头声明字段和有限模块接口；资源方法移到实现单元，保留字段/默认值、局部 owner 声明、分配、析构和事件顺序。成功结果与 relocation 决策前的异步排空仍在原调用位置，没有新增 CUDA 同步。环境默认值及全部动态快照/共享状态仍由同一 options 单元按原顺序初始化。
- D/预算/请求计算迁到 `prepare_stage2_plan`；公共查询直接使用模数上下文和该规划函数，曲线执行也复用它。`Stage2PlanRuntime` 用原位 optional 保存模型，保持模型缓存到曲线结束的寿命，不增加模型堆分配。纯 giant 分块公式迁入 core，诊断夹具留在 diagnostics。
- 生产 NTT 校准进入 tune.cpp；实验 wrapper 使用自己的实验校准头和既有独立树/NTT。共享的 JSON 转义、plain-scale 帮助函数和纯设备原语保持有限范围，生产闭包不包含实验源码。未改变实验引擎的选项限制或内存模型能力。
- 正式职责、依赖与资源合同改写到 [Stage2](../architecture/STAGE2.md#cuda-模块边界)，同步构建、内存、Auto B2、准备开销和总览的当前入口。CPU 资源夹具读取实际头声明和实现方法；产品构建不截取源码。

## P3 验证证据

以下路径均相对于证据根。CUDA13.3 / MSVC14.51.36231 / sm89；原生运行使用 GPU1 RTX4060 Laptop，未改变功耗、频率或其他进程。Stage1 构建限定384档；本轮不是多架构、完整位宽或性能资格验收。

- `step1_build_04.log`、`step1_runtime/summary.json`、`step1_sass/result.json`：首批25单元构建通过，5组 Stage1 存档及5组 Stage2 交接叶/稳定结果一致；176个设备函数与 P2 的完整 SASS 一致。随后 `step2_build_02.log` 完成规划分离后的新目录全构建。
- `source_complete/result.json`：202个主机函数/方法及38类字段、默认值和构造初始化保持；显式展开规划调用后原语句、字符串、检查和同步一致，全局默认值及快照次序一致。packing、plain-scale、giant planner 和两套校准/清理正文另行逐段比较。95个底层源/接口/NTT 文件的仓库内容身份不变；无反向曲线流程包含。`arithmetic_structure_final/result.json` 的93个设备算术函数门禁通过；`sass_final/result.json` 再次确认176个函数无增删、无指令差异。
- `s4_memory_final/checks.json`：299592项检查、2136个分配事件、192个树租约通过，保留96窗口必需自检。`giant_memory_final/checks.json`：265638项检查、2054个时间线场景、39665个分配事件检查。`owner_memory_final/result.json`：140场景、6580项检查、1960个分配事件；`initial_memory_final/result.json`：50场景、4556项检查和1500项事件检查。错误原生 extent/metadata 变异仍被拒绝。
- `extended_runtime_03/summary.json`：14组双版本完整目标叶指纹、稳定结果及算术检查覆盖一致，包含8193/16381/16384位、通用/梅森、已知真因子、基点非单位、预算0、baby/fold/frontier分配回退、异步环1、阶段回收、两种packing及诊断夹具。单位样本同时核对已有独立 CPU/GMP 单项式叶指纹；真因子逐项验证。timing ledger 只核对闭合，不比较耗时快慢。固定D、自动D、承载、relocation的4组计划查询字段一致且不执行曲线；4种损坏各跨5个日志级别，共20组双版本拒绝，均没有发布成功结果。
- `cmake_stage2_build_02.log`、`build_outputs_verified_02/summary.json`：真实 CMake 目标构建、25单元/2设备 CU 成员及完整目标叶一致。`stage1_version_build.log`：31个 CUDA 对象身份复用，只更新主机入口；`packaged_runtime/summary.json`：0.44.4 Stage1 与通用 GMP 发布 Stage2 对照旧版，5组存档和5组普通/Edwards/承载/relocation/主机回退交接一致。
- `ntt_counts/summary.json`：11个计数开关、覆盖优先级及每曲线清零门禁；`ntt_tune_final/summary.json`：28组长度/批量的独立 GMP 校验、形状、调度与输出字段一致。`development_build.log`：实验默认 add/sub=0 构建通过；`build_outputs_verified_02/summary.json`：旧实验显式 add/sub=1 与新实验默认0各14个 NTT 校准形状均通过 GMP。这里不声称两种编译策略的 SASS 或速度相同。
- `identity_final/result.json`：39项 CPU 构建身份/源闭包故障注入与跨 PowerShell 运行时检查通过。`hostonly_final.log`：精确身份一致，程序完整复用。`release_build.log`、`package_reuse.json` 及发布归档检查记录构建、SkipBuild、已有 TOML/队列保留及干净默认归档；没有向实际工作队列写入任务。

## P3 失败证据、限制与后续

首轮机械迁移暴露相对包含路径、未显式导出的诊断状态、packing 所属模块和 sstream/quote 依赖遗漏，`step1_build_01`～`03` 及 syntax 日志保留；修正声明和模块归属后才进入运行验证。依赖收缩时，`step2_build_01.log` 的旧编译依赖文件使内容未改的旧头在编译后集合中消失，被既有校验拒绝；没有放宽门禁，改用新目录全构建并最终验证复用。首个 CMake 辅助命令使用正斜杠相对路径，被 cmd 当作参数；修正为绝对 Windows 路径后通过。

`extended_runtime_01` 的小存档参数未转为绝对路径，只完成宽样本便在预算用例报找不到存档；修正采集器后重新完成。`extended_runtime_02` 在旧版 M127 上触发 baby 合成夹具 GMP 不匹配。最终分开核对：通用8193位夹具正常通过，M127 上新旧版均以相同 point=1 诊断拒绝。该夹具自行切换 N，活动梅森点常量的刷新/恢复属于待修诊断上下文问题；P3 保持算法和现有拒绝行为，不将诊断失败解释为正常曲线失败。

`build_outputs_smoke` 中的旧实验 worker 已因缺少当前曲线内存模型拒绝工作；后续实验校准首次还暴露仅接受 debug 日志级别的既有限制。`build_outputs_verified_02` 核对新旧 worker 同样拒绝，并在其支持的 debug/纯 NTT 校准路径通过 GMP。生产工作流未依赖该实验能力。以上失败日志与原样本均保留。

P3 不删除尚存的不可达分支、旧候选模型或固定研究资料；剩余实验消费者整理归 P4，完整矩阵、多架构及性能资格归 P5。临时草稿随提交保留，完成后提示用户处理，不自动删除。

提交前的暂存检查发现 factors.cpp、s4.cpp 和 tree.h 末尾多余空行；原字节保存在 `source_before_eof_fix/`，此前验证的二进制保存在 `committed_binaries/`。清理后本机/发布各重编23个主机单元，两个 CUDA 对象严格校验后复用；CMake 同步重建。`eof_source_identity`、`eof_s4_memory`、`eof_giant_memory`、`eof_owner_memory` 对最终源码重新通过。首次整 PE 比较只遇到链接时间戳差异：`eof_binary_identity_02.json` 根据 SDK 的 COFF/IMAGE_DEBUG_DIRECTORY 字段仅规范化两处时间戳，三个最终程序的其余全部字节（指令、常量、数据及 GPU 镜像）完全相同，CUDA 对象摘要也不变，没有忽略指令或其他数据。最终发布包的5组存档/5组 Stage2 对照见 `packaged_runtime_eof`；`release_eof_package.log`、`release_archive_eof.json` 核对最终 SkipBuild 和默认归档，已有配置/队列保持。最终二进制、DLL 和清单冻结在 `committed_binaries_02/`，静态语法与暂存检查见 `final_static_checks.json`，完整矩阵到最终程序的绑定见 `final_runtime_binding.json`。

## P4 范围与删除依据

基线为 `2a2606e`（0.44.4），原程序、清单、对象及输入冻结在 `data/experiments/cuda_structure_p4_20261020/baseline/`。以下路径均以本轮证据根目录为起点。先通过有限源码证明和资源模型，再验证生产二进制，随后迁移实验消费者并重新验证最终版本。

- `batch.cpp` 的 `dev_leaves` 只有一次 false 初始化且地址不逃逸，因此删除 S5 设备叶读回分支；保留被执行分支的原花括号和析构位置。S5 配置、状态字段和兼容输出不删除。
- 私有 `run_real` 只有一个存档入口调用，参数恒为 `run_s2=false, curves=1, d_plan_only=false, plan_result=nullptr`。删除这些私有参数和单次循环，保留原匿名块，使点缓存、owner 及 GMP 清理的先后不变。公共入口和独立计划查询保留。
- D 候选模型有无条件 `calibrated=false`，因此删除不可达的候选选择、成本计算和输出分支。有效的 `legacy_56_1` 公式、D/预算选择、`NTT_D_MODEL` 解析、禁用原因及现有日志字符串保留。旧谓词还会预热实际 NTT 策略缓存，已在相同规划边界保留该调用。
- `read_hex_mpz`、`read_f_dump`、`words_for_bits`、`hex_of_words`、`flat_truncate` 只有定义和私有声明，没有消费者；删除函数、声明、仅供其使用的 `FDump` 及未使用常量。有效的 F 转储写入和诊断开关保留。
- 生产候选速率表从 `src/cuda/stage2/stage2_d_model.cuh` 原字节移到 `tools/experiments/cuda/stage2_tree/d_model_profiles.h`；旧树实验的不同速率表独立保存在 `legacy_d_model_profiles.h`。纯 baby payload 尺寸计算进入核心内存公式，不再为一个有效公式引入整份失效模型。

`source_final/result.json` 严格比较32个保留生产模块，只允许上述有限变换及三段 pack 启动函数迁移；包括所有字面量、CUDA 调用、检查及回退。`arithmetic_source_final/result.json` 核对93个设备算术正文。`contracts_final/result.json` 核对五个探针接口，确认 Stage1、NTT 和独立 oracle 未改。

## P4 实验目录与消费者

27项原始迁移映射在 `tools/test/cuda_p4_migrations.json`；`isolation_final/result.json` 核对正文、入口默认值、宏和138个 CUDA 消费者，三个非法边界样本均被拒绝。

- `tools/experiments/cuda/stage2_tree/` 拥有实验树、开发 wrapper、校准、点算术、baby/giant/scale 候选及冻结 D 表。保留原混合行尾；`.gitattributes` 的逐字节属性迁移到新 engine。实验 plain-scale 使用独立主机适配，只共享允许的纯设备原语和统计/转义帮助函数。
- `stage2_bsgs/` 拥有旧 CGBN BSGS 的五个文件；`ntt_variants/` 拥有独立 NTT、carry、outer、Goldilocks 和 cuFFT 候选；`ntt_tensor/` 拥有 Tensor 算法候选。它们通过实验源组显式构建，不加入25单元生产目标。
- GMP oracle 与测试夹具留在 `tools/test/`，测量入口留在 `tools/bench/`。所有活动 CUDA 消费者停止包含 `.cu`、使用 NO_MAIN 抑制入口或在构建时截取实现正文。冻结审计保留旧快照、原始身份和完整源码，使用独立历史适配，不改写快照。
- pack 探针调用有限启动接口并链接实际算术/NTT 模块，保持其 per-thread ABI。giant 的已注册普通 chain 调用生产算术对象；旋转候选及160/192字控制组由独立 CU 拥有。host-wait 探针通过有限诊断 API 和唯一显式适配调用真实 Edwards 运行时及设备模块。
- 生产闭包检查新增 `tools/test/`，包含编译器间接依赖；`identity_final_02/result.json` 的40项 CPU 身份/故障/跨 PowerShell 门禁通过。

## P4 运行与算术验证

工具链为 CUDA13.3、MSVC14.51.36231、sm89；运行使用 GPU1（RTX4060 Laptop）。本机 GMP 为仓库 Zen3 构建，发布使用 vcpkg 通用 x64 GMP。开发对照显式使用 AddSubMask=0，生产固定为1；host-wait 编译1280档。以下均为正确性或身份检查，不是性能资格。

- `runtime_final/summary.json`：最终0.44.5与冻结0.44.4的14组完整叶/结果、4组计划及20项损坏拒绝一致，覆盖梅森/通用模数到16384位、预算与分配回退、异步 ring、阶段修剪及诊断路径。
- `saves_final/summary.json`：Stage1 的5组、每组4条曲线存档字段一致（PARAM0/2/3、Edwards binary/NAF）；Stage2 的5组续算叶和稳定结果字段一致。Stage1 构建仅384档、PRAC关闭。
- `sass_final/result.json`：最终生产程序的176个设备函数与0.44.4指令完全一致，无增删或变化。`arithmetic_final.log`：真实生产算术对象的114个模数、11286项 GMP 比较，bad=0。`graph_final.log`：368次完整卷积、14779392个 digit、8次构建/16次复用/8次失效，live=0；`graph_fault_final.log` 按原规则检测注入错误并返回3。
- `probe_runtime_03/summary.json`：host-wait 的12组新旧点文件、事件设置及SDK调用数相同；giant 的40项独立 GMP 检查通过，覆盖64/128/160/192/256字与旋转/普通两路；pack 正常及故障门禁通过。giant 的原 oracle 对最多两块检查全部坐标，对中间块保留原抽样边界。
- `initial_memory_final`：50例/4556检查/1500事件，错误范围被拒绝；`owner_memory_final`：140例/6580检查/1960事件，错误元数据范围被拒绝；`s4_memory_final`：299592检查/2136分配事件/192树租约；`giant_memory_final`：265638检查/2054时序/39665事件。均使用实际资源方法，bad=0。
- `experiment_pairs_02/summary.json`：链接冻结旧 worker 对象与隔离后对象，在M383/M521/M1279、实际W=6/9/20（NW8/16/32）比较完整叶及结果，XADD、baby、Gamma/真实 scale、乘积及scaled/慢下降GMP检查通过。实验对象共224个设备函数，移动点原语后7个函数的编译指令变化，未发现函数增删；原始差异保留在 `development_sass/result.json`，这些函数由上述对象对照覆盖，不据此宣称实验SASS或性能完全相同。
- 实验树独立构建通过。`tree_gate.log` 的150项原控制检查通过；其余旧脚本因参考程序缺失/硬编码路径及过期608断言失败，修正测试入口并构建原CPU参考后，`tree_reference_scope.log` 的39项参考检查通过。实际旧/新除法夹具均为800例，未修改数学实现或放宽门限；这不是一次全脚本零失败运行。
- 独立NTT18项、carry的180例/657000字及12配置×18形状×2模式、baby的42例/379552字及9组树真实/回退检查、BSGS自检和42组点GMP检查通过。冻结shape/batch适配以明确标记的Git源夹具构建并通过，batch故障返回3，不声称该夹具是旧发布二进制。
- `fixture_final_gate/summary.json`：最终路径的Tensor正常/故障、tile正常/故障、outer dense/policy及六种冻结候选profile共12项通过；`cufft_gate.log` 原有13项整数/系数GMP门禁通过。

## P4 保留的失败与边界

旧实验对象的指令差异完整保留；没有根据计时选择新算法。早期实验链接因 /MD 与 /MT 混用失败，适配随后按真实worker ABI构建。giant收集器首次误将完整整数检查数视为字数，第二次遗漏原中间块抽样，第三次按未改动oracle的实际计数通过；不曾调整算术或oracle。身份测试首次遗漏夹具目录，修正测试目录创建后40项通过。开发HostOnly因迁移后的carry依赖正确拒绝，随后使用新目录完整构建；所有失败原始证据保留。

正常生产曲线仍不依赖已有M127活动点上下文中的合成baby诊断夹具；P3记录的该夹具限制和实验worker缺少曲线内存模型的拒绝行为继续保留。完整位宽、多架构、全部可选宏矩阵和性能资格属于P5。本草稿随提交保留，完成后提示用户处理，不自动清理；正式现状另已同步到Stage2、NTT、架构总览和构建说明。

## P4 最终构建与交付绑定

`native_final_build.log`、`stage1_build.log`、`experiment_final_build_02.log`、`cmake_stage2_build.log` 和 `release_build.log` 分别确认本机Stage2、受限384档Stage1、独立实验worker、CMake Stage2及通用GMP发布构建通过。`native_reuse_final.log` 的源/工具链/架构、对象与程序复用通过；`release_skip_final.log` 的发布SkipBuild通过，已有TOML和队列摘要保持。`release_archive_final.json` 核对ZIP仅8个允许文件，使用默认配置和空队列。

`probe_native_final/summary.json` 对最终生产算术对象复用的giant探针重新通过12组host-wait、40组giant和2项pack门禁。`experiment_pairs_final/summary.json` 对完整重编后的最终开发对象重新通过三组完整叶/GMP对照。`tune_final/summary.json` 对生产及实验各28种形状、每形状2次校准进行双版本对照，GMP检查、计划形状、调度和稳定输出元数据一致，未用计时作性能结论。`output_final_gate/summary.json` 核对CMake的25单元清单、Montgomery普通及Edwards承载两组完整叶/结果，并复核新旧实验队列入口同样拒绝缺少曲线内存模型的工作。

`packaged_runtime_final/summary.json` 直接运行最终发布包内程序，5组存档及5组Stage2输出对照通过。`powershell_static_final.json`、`python_static_final.json` 记录修改脚本语法；配置生成检查及暂存空白检查通过，配置与两份配置参考生成物无变动。`committed_binaries/identity.json` 冻结五类最终程序、DLL、源码/构建清单及发布ZIP。`final_summary.json` 将全部必需门禁与最终源码、对象和二进制摘要绑定；未重写中间失败或原始实验记录。
