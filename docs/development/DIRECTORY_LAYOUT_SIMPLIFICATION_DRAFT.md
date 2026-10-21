# include/ecm 与 src/ecm 目录精简草稿

日期：2026-10-21。依据：`7585e7a`、源码版本 `0.45.24`。状态：**方案 A 已获授权，D0–D2 完成，当前源码版本 0.45.28；D3 收口验证待完成**。本文件是本轮目录精简的权威草稿，[网页阅读版](../web/directory-layout-plan-20261021.html)呈现同一方案；当前已实现的目录仍以[架构与术语](../architecture/OVERVIEW.md)为准。

## 实施进度与证据

### D0：清单冻结与 GUI 统一入口（已完成，0.45.25）

冻结 529 个 include/src 文件摘要与原始字节、844 个构建/测试/生成引用文件，以及方案 A 的 31 项映射。公开头保持 14 个；原始资源、实例与 PRAC 不变。证据在 [D0 freeze](../../data/experiments/directory_layout_implementation_20261021/d0/freeze.json)，此前主流程资格不改写。

新增唯一 GUI BAT 入口 `tools/build/build_gui.bat`，调用已有 `dev/build_gui.ps1`，不增设 dev/local/release 包装族。默认输出 `build_gui_local/ecm_gui.exe`；参数与退出码原样传递，无参数双击结束暂停。源码软件身份更新为 0.45.25，脚本不按该版本命名。

实际通过 BAT 的 `-Selftest` 调用完成 GUI local 构建，33 项自检全部通过；从仓库外工作目录再次调用同一 BAT，33 项再次通过，重复构建未编译或链接。保留既有 C4244 警告，未改配置默认值或 GUI 行为。日志：[首次构建](../../data/experiments/directory_layout_implementation_20261021/d0/gui.log)、[外部工作目录](../../data/experiments/directory_layout_implementation_20261021/d0/gui_external_cwd.log)。本阶段尚未迁移源码路径，未构建 release。

D0 提交：`22380cc`。D1、D2、D3 按下文顺序继续，验证完成后逐阶段提交。

### D1：普通职责家族精简（已完成，0.45.26）

已迁移 29 个非 backend 文件，删除 7 个空分类目录。配置表达式、兼容执行声明、Stage2 规划与辅助、平台输出、CLI 群阶诊断和共享 Edwards 构造均折回所属家族；include 仍为 14 个公开头。源文件数量保持 515，src 目录由 80 减为 73。

修正生产、测试/生成器、构建源清单、Android 和当前文档的相关引用；历史源码读取映射只更新当前位置，冻结 Git 键继续保留。核对 353 个 C++/CUDA 文件，除允许的版本号和 include 路径外正文不变；资源字节、公开合同、翻译单元顺序与编译策略不变。校验原本直接相对引用的目标也必须保持，避免依赖其他 include 搜索目录的偶然回退。[有效正文及路径审计](../../data/experiments/directory_layout_implementation_20261021/d1/audit_0.45.26_02.json)、[当前包含边界](../../data/experiments/directory_layout_implementation_20261021/d1/boundaries_02/result.json)通过，配置 8 项生成物一致。

正式 local 验证通过：CUDA Stage1（sm89、384/1024 两档，31 个 CU 翻译单元）、生产 CUDA Stage2（35 个翻译单元）、OpenCL、CPU Stage2 libecm，以及 GUI BAT。实际通过 27 项 CPU/GPU/恢复/跨阶段发布案例、Stage2 规划 8 项与任务流程 12 项前后对照、4 曲线停止恢复、23 项结果一致性、OpenCL 计算和 GUI 33 项自检。二进制与匹配清单另行冻结，详见 [D1 结果](../../data/experiments/directory_layout_implementation_20261021/d1/result.json)与[产物身份](../../data/experiments/directory_layout_implementation_20261021/d1/artifacts.json)。未重做全32档、release、多架构、可选 PRAC/GWPoly 或性能资格，不外推既有 0.45.24 的完整资格。

本阶段保留首轮失败：一次性迁移辅助脚本重复清理已空目录；另有 6 个移动头文件共 10 条相对引用，被通用搜索根掩盖而漏改，实际 Stage2 构建及时拒绝。已修正清理脚本与相对引用，并重新审计/构建；最初宽泛搜索根审计 `audit_0.45.26.json` 已明确作废，不作为验收。沙箱 CMake 空程序检测的 C1902，以及结果测试硬链接拒绝，均保留原记录，随后同脚本/参数在沙箱外通过，没有改变算术、编译策略或测试条件。细节见[相对路径修复](../../data/experiments/directory_layout_implementation_20261021/d1/relative_repair.json)。

旧 `test_stage2_planning_extraction.py` 对照早期 44 个函数，D0 与 D1 都有相同的 52 个函数，因此该冻结提取检查在迁移前已不匹配；未修改其断言或当作本轮通过项。[定位证据](../../data/experiments/directory_layout_implementation_20261021/d1/legacy_gate_diagnosis.json)保留。Stage1 closeout 只补充一个已验证的 include 改名，十个实现单元、十个群阶辅助和执行顺序检查通过。D2–D3 继续使用当前冻结正文、包含边界和实际运行验证。

### D1b：local 默认短目录（已完成，0.45.27）

按用户补充要求，CUDA local 的 BAT 与 PowerShell 共用固定短路径：Stage1 为 `build_stage1_local/ecm_cuda.exe`，Stage2 为 `build_stage2_local/ecm_cuda_stage2.exe`；GUI 保持 `build_gui_local/ecm_gui.exe`。构建身份仍写入 `profile.json`，不把不同架构、tiers、引擎或用途自动混入同一个目录。需要并存时显式选择 Stage1 `-BuildDir`、Stage2 `-Build`；Stage2 显式 `-BuildRoot` 仍保留 profile 子目录。dev/release 布局保持现有合同，旧缓存不迁移、不删除。

30 项 profile 检查通过，包含短默认路径、显式根层级、不同用途/阶段拒绝、身份突变、旧缓存准入及包目录隔离。Stage1 BAT 通过显式旧叶目录验证，31 个 CUDA 对象全部复用；GUI BAT 33 项自检通过。Stage2 BAT 已在默认 `build_stage2_local` 实际构建全部 35 个生产翻译单元，8 项规划/曲线子进程前后对照通过。D1 提交为 `2f22d10`；本阶段[结果](../../data/experiments/directory_layout_implementation_20261021/d1b/result.json)、[profile 检查](../../data/experiments/directory_layout_implementation_20261021/d1b/profiles/result.json)与[默认路径构建日志](../../data/experiments/directory_layout_implementation_20261021/d1b/stage2_default.log)保留。Stage1 的固定默认路径由同一解析器的实际 profile 检查验证；本阶段编译通过 BAT 显式复用既有两档验证目录，未在默认目录建立受限两档 profile，以便用户随后直接构建完整默认档位。

### D2：Edwards 私有分类精简（已完成，0.45.28）

checkpoint 与 host-wait 两个私有头已移至 Edwards 家族根，生产主机流程、诊断 Adapter 与检查点测试引用同步调整。方案 A 的 31 个文件路径全部迁移完成，9 个分类目录均删除；src 保持 515 个文件、目录由 80 减为 71，include 仍为 14 个公开头。最大深度仍为 5，最深 51 个文件全部属于实例与 PRAC。[353 文件审计](../../data/experiments/directory_layout_implementation_20261021/d2/audit_0.45.28.json)和[包含边界](../../data/experiments/directory_layout_implementation_20261021/d2/boundaries/result.json)通过。

CUDA Stage1 通过 BAT 在浅层 `build_stage1_local_check` 构建 384/1024 两档，避免在用户完整默认目录写入受限 profile；Stage2 使用默认 `build_stage2_local`，复用两个 CUDA 对象，仅按需重编普通 CXX。GUI、OpenCL 和 CPU Stage2 libecm 也已更新至 0.45.28。27 项曲线/恢复/跨阶段案例、Stage2 停止恢复、GUI 33 项自检通过。12 种 host-wait 组合使用真实 CUDA 1280 档探针验证，事件创建符合合同，输出点逐字节一致；计时字段不作为性能结论。[D2 结果](../../data/experiments/directory_layout_implementation_20261021/d2/result.json)、[host-wait](../../data/experiments/directory_layout_implementation_20261021/d2/host_wait_run/result.json)、[产物身份](../../data/experiments/directory_layout_implementation_20261021/d2/artifacts.json)保留。

检查点探针通过外壳恢复、损坏拒绝、旧 V1 外壳导入、Edwards NAF/Binary 旧载荷读入及新二进制恢复，逐坐标和因子一致。原测试构建入口漏链已存在的 NAF 构造实现；首轮补源误选后端调度文件，触发缺少 CUDA SDK 头，随后定位到真正定义所在的 `core/stage1/ecm_stage1_exp.cpp`，最终链接该普通 CXX 源通过，未添加 SDK 或伪造实现。两个失败轮次原样保留，见[定位记录](../../data/experiments/directory_layout_implementation_20261021/d2/checkpoint_builder_diagnosis.json)与[有效检查点运行](../../data/experiments/directory_layout_implementation_20261021/d2/checkpoint_run_03.log)。

D1b 提交为 `3150abb`。D3 将完成无 CUDA SDK 的主机编译、9 类正文修改零 CU 和重复构建验证，再同步正式目录说明。

## 结论与适用范围

推荐方案 A：保留现有 application、service、core、backend、storage、platform 家族，删除 9 个不承担独立职责的分类目录，调整 31 个文件路径。`include/ecm` 的 14 个公开头保持现有路径。保留真实的曲线家族、实例编译组、PRAC、设备实现与 OpenCL 算子资源目录。

本轮解决路径导航问题。目录层数与 Module 的 Depth 是两件事：去掉一层目录不会自动改善 Interface 或隐藏更多 Implementation。driver、task、factory、QueueWorkflow 的职责与调用关系沿用已完成的主流程重构，不增加新 Adapter、转发头或执行入口。Locality 的收益来自把同一职责的文件放在一起；不声称目录移动带来运行性能或编译速度提升。

如果要求所有实现最多 4 层，再采用方案 B：在 A 的基础上把 `backend/{cpu,cuda,opencl}` 提至 `src/ecm/{cpu,cuda,opencl}`。B 是另一种目标布局，**不列入 A 的默认实施阶段**。

## 实查结果与深度口径

深度从 `include/ecm/` 或 `src/ecm/` 下开始数目录，不计这两个根与文件名。例如 `backend/cuda/stage1/montgomery/instances/x.cu` 为 5 层。

| 范围 | 文件数 | 目录数 | 最大深度 | 说明 |
| --- | --- | --- | --- | --- |
| 规划基线 include/ecm | 14 | 14 | 2 | 4 个头在 1 层，10 个头在 2 层 |
| 规划基线 src/ecm | 515 | 80 | 5 | 339 个 C++/CUDA 相关文件、172 个 OpenCL 资源、4 个 GUI 构建/本地化文件 |
| 方案 A 的 src/ecm | 515 | 71 | 5 | 31 个文件换路径；最深文件由 53 减为 51，剩余全部属于 instances 或 PRAC |
| 方案 B 的 src/ecm | 515 | 70 | 4 | 共 401 个不同文件换路径，包含 A；不是 31 与 372 的简单相加 |

上述是磁盘文件清点，不把 172 个 OpenCL 片段称为 172 个独立 Module，也不把文件数量当成编译单元数量。A、B 均不删除实现文件。

当前 `backend` 包含 CPU 15、CUDA 167、OpenCL 190 个文件，共 372 个。B 的 401 个迁移文件由这 372 个文件与 A 中 29 个非 backend 文件组成。A 的另外 2 个 Edwards 文件已经包含在这 372 个中。

本机[清点与逐文件拟议映射](../../data/experiments/directory_depth_plan_20261021/inventory.json)冻结当前文件摘要、目录深度和 A/B 映射；已验证 Windows 不区分大小写的目标路径无冲突、公开头仍有对应 src 家族。该证据在忽略目录，不替代迁移后的编译验证。

## 方案 A：删除哪些层级

以下 9 项均为明确建议（Strong）。只调整文件归属路径并修正引用，不合并文件、不改变函数、类型、命名空间、链接符号或现有翻译单元。

| 当前相对 src/ecm 的路径 | 拟议路径 | 文件数 | 判断依据 |
| --- | --- | --- | --- |
| application/configuration/legacy/ | application/configuration/ | 2 | ecm_expr.cpp/h 是多个入口正在使用的整数表达式解析器；legacy 不是第二套配置实现 |
| service/execution/legacy/ | service/execution/ | 3 | ecm.h、cgbn_stage1.h、ecm_params_init.cpp 继续组成兼容执行合同；删除分类层，不删除兼容能力 |
| service/planning/legacy/ | service/planning/ | 18 | 当前成本、预测、调优、缓存及 .def 文件直接服务 stage2.cpp；不是已被替代的实现 |
| service/stage2/legacy/ | service/stage2/ | 1 | ecm_stage2_factorize.h 属于当前 Stage2 辅助计算 |
| platform/legacy/ | platform/ | 2 | 当前 Stage2 console/logging 协作，文件名前缀已表达职责 |
| backend/cuda/stage1/edwards/legacy/ | backend/cuda/stage1/edwards/ | 1 | 当前 Edwards checkpoint 编解码/恢复协作；旧格式读取行为继续保留 |
| application/cli/diagnostics/ | application/cli/ | 2 | group_order.cpp/h 是 CLI 的可选群阶诊断，文件名已表达用途 |
| backend/cuda/stage1/edwards/diagnostics/ | backend/cuda/stage1/edwards/ | 1 | host_wait.h 仍是私有诊断 Seam；移动不改变注入 hooks 或实际 CUDA 等待 |
| core/curve/edwards/ | core/curve/ | 1 | ecm_edwards_curve.h 已携带曲线身份，提供跨 CPU/GPU/存档使用的 GMP 构造，无需单文件再套曲线目录 |

Deletion test：删除这些目录标签，行为和使用合同没有需要重新安置的额外内容；复杂度仍在原文件及其所属 Module 内。真正的 Interface 和 Adapter 都保留。这里改善的是导航 Locality，不据此宣称原有 Module 为 Shallow 或已经变为 Deep。

### 关键路径前后对照

```text
service/planning/legacy/ecm_stage2_runtime_cache.h
→ service/planning/ecm_stage2_runtime_cache.h

service/execution/legacy/ecm.h
→ service/execution/ecm.h

core/curve/edwards/ecm_edwards_curve.h
→ core/curve/ecm_edwards_curve.h

backend/cuda/stage1/edwards/diagnostics/host_wait.h
→ backend/cuda/stage1/edwards/host_wait.h
```

`legacy` 目录消失，不意味着具有兼容用途的文件名必须一起改名。例如公开的 `legacy_gpu.h` 和检查点旧格式入口保持原名，避免把目录整理扩大为 Interface 与协议更名。是否改名可另行分析，不能顺手删除历史格式支持。

## 推荐目标目录

以下为 D2 已实现的方案 A 目录。所有公开头完整列出；src 展示职责家族与关键文件，未列出的现有文件留在对应家族。完整逐文件映射见清点证据；D3 验收状态单独记录。

```text
include/ecm/                         # 14 个公开头，保持现有路径
├── application/configuration/runtime.h
├── core/
│   ├── stage1/{scalar.h,selection.h}
│   └── stage2/probability.h
├── service/
│   ├── stage1/task.h
│   ├── work/queue_workflow.h
│   └── execution/legacy_gpu.h
├── storage/
│   ├── queue/stage2_progress.h
│   └── results/publication.h
└── platform/
    ├── fingerprint.h
    ├── logging.h
    ├── program_version.h
    ├── stage1_progress.h
    └── process/curve_process.h

src/ecm/
├── application/
│   ├── cli/                        # driver、输入、queue_source、group_order.cpp/h
│   ├── configuration/              # 配置运行环境、TOML、身份、ecm_expr.cpp/h
│   │   └── generated/
│   ├── cuda_stage2/                # 独立 CUDA Stage2 程序入口
│   ├── cpu_stage2/                 # 独立 CPU Stage2 程序入口
│   ├── p95_feeder/                 # 独立 feeder 程序入口
│   └── gui/
│       └── localization/
├── service/
│   ├── stage1/                     # task、准备、执行分派、执行合同
│   ├── execution/                  # ecm.h、cgbn_stage1.h、ecm_params_init.cpp
│   ├── work/                       # 唯一 QueueWorkflow
│   ├── stage2/                     # 父任务、worker、输入、factorize
│   └── planning/                   # stage2.cpp/h 与当前模型/缓存/调优私有文件
├── core/
│   ├── curve/ecm_edwards_curve.h    # 共享 GMP 曲线构造
│   ├── stage1/                     # 标量、执行选择
│   └── stage2/                     # 几何、候选、树、工作量、模数与 relocation
├── backend/
│   ├── cpu/
│   │   ├── arithmetic/{scalar,ifma}/
│   │   ├── stage1/{edwards,montgomery}/
│   │   └── stage2/gwpoly/
│   ├── cuda/
│   │   ├── factory.cpp
│   │   ├── stage1/
│   │   │   ├── edwards/            # 主机/设备协作、checkpoint、host_wait.h
│   │   │   │   └── instances/
│   │   │   └── montgomery/
│   │   │       ├── instances/
│   │   │       └── prac/
│   │   ├── stage2/
│   │   │   └── device/
│   │   ├── ntt/
│   │   │   └── device/
│   │   ├── arithmetic/{cgbn,goldilocks}/
│   │   ├── planning/
│   │   ├── runtime/
│   │   └── dispatch/
│   └── opencl/
│       ├── factory.cpp
│       ├── stage1/montgomery/
│       ├── arithmetic/
│       ├── runtime/
│       ├── diagnostics/            # 生产自检，不是待删除的 CLI 分类层
│       └── resources/
│           ├── common/
│           ├── add_mod/
│           ├── sub_mod/
│           ├── mont_mul/
│           ├── special_mult/
│           └── diagnostics/        # OpenCL 逻辑资源名继续保留
├── storage/
│   ├── cache/
│   ├── checkpoint/
│   ├── save/
│   ├── queue/
│   ├── transfer/
│   ├── profile/
│   └── results/
│       └── generated/
└── platform/
    ├── process/
    ├── generated/
    ├── ecm_stage2_console.h
    ├── ecm_stage2_logging.h
    └── 现有日志、进度、指纹与文件发布实现
```

## 为什么这些少文件目录仍然保留

### include/ecm 保留 1–2 层

公开头所在目录表达使用者要了解的职责，src 在同路径保留对应实现家族；header-only 模型不必制造同名 .cpp。私有头仍留在 src，不要求复制整棵私有目录到 include，也不要求每个实现文件都有公开头。

例如 `include/ecm/service/work/queue_workflow.h` 对应 `src/ecm/service/work/queue_workflow.cpp`；`include/ecm/platform/process/curve_process.h` 对应 `src/ecm/platform/process/` 的本机进程实现。当前[边界检查](../../tools/test/test_main_flow_boundaries.py)第 36–39 行也核对公开头的对应家族。

把 14 个公开头全部移到 `include/ecm/` 或把 task、queue、backend 合到一个目录，省下的层级有限，却会抹去职责信息并改动所有调用者。当前没有这样的需要。不为实现专用的 CUDA/OpenCL 文件新增公开头。

### 实例、算法、设备与资源目录保留

`instances`：Montgomery 有 22 个文件（21 个实例 .cu 加声明头），Edwards 有 8 个文件（7 个实例 .cu 加声明头）；分组承担位宽/参数实例化与编译职责。[CUDA 源清单](../../tools/build/internal/cuda_sources.cmake)分别列出实例翻译单元。它们属于 Implementation 内部组织，不因压目录而合并 .cu。

`prac`：21 个文件组成可选 Montgomery 算法家族，包含计划、域、描述、窗口、运行时与 11 个 .cu。保留算法归属有助于理解可选路径的验证范围。

`device`：Stage2 的 10 个文件与 NTT 的 7 个文件标明设备实现。普通 CXX、依赖 CUDA runtime 的主机实现、NVCC 编译的设备代码，仍通过现有翻译单元和私有合同隔离。目录只是提示，实际重编译范围由包含关系和源清单决定。

`resources`：172 个 OpenCL 资源按算子划分，相对资源名用于查找和组装；`diagnostics/mont.cl` 也有逻辑名称兼容用途。不能把资源里的 diagnostics 与 CLI 的同名分类一起删除。[OpenCL runtime](../../src/ecm/backend/opencl/runtime/runtime.cpp)第 325、404 行与[根构建](../../CMakeLists.txt)第 102–107 行维护资源根和复制位置。

### 少文件但职责真实的目录保留

`generated` 标明维护来源；`platform/process` 隔离进程句柄、pipe、停止和寿命；`cpu/stage2/gwpoly` 保留可选第三方计算引擎归属；`cpu/arithmetic/{scalar,ifma}` 区分算术实现与指令集要求；`cuda/dispatch` 集中位宽合同；`storage/profile` 负责 CSV 存储；`service/work` 负责队列协议。独立程序入口目录与 GUI localization 也保留。

这些 Module 的删除会把依赖、状态或使用知识散到调用者，具有实际 Locality 与 Leverage。文件少本身不说明 Interface 为 Shallow。A 不调整这些 Seam，也不把真正独立的 Adapter 合并成通用目录。

## 方案 B：需要统一压至 4 层时

推荐强度：Worth exploring。B 在 A 的基础上仅删除 backend 这一层，后端内部的阶段、曲线、算法、实例、设备和资源结构均保留。

```text
src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g0.cu
→ src/ecm/cuda/stage1/montgomery/instances/instances_param0_g0.cu

src/ecm/backend/opencl/resources/mont_mul/...
→ src/ecm/opencl/resources/mont_mul/...

src/ecm/
├── application/
├── service/
├── core/
├── cpu/
├── cuda/
├── opencl/
├── storage/
└── platform/
```

B 的优点是全部后端路径统一减一层，最深从 5 降到 4；保留真实 Adapter 的划分。代价是调整 401 个文件路径，src 顶层从 6 个家族变为 8 个；include 没有公开 backend 头，因此 14 个公开头仍可保留对应家族。B 只是另一种物理组织，运行依赖方向不变，不产生新的 Module Depth 或 Leverage。

额外核对范围包括：CMake 与 CUDA 源清单、私有相对 include、构建脚本指纹和依赖清单、OpenCL 资源源目录与运行时 fallback、Android 的 CMake/Gradle 路径、生成算子脚本、结构测试及网页生成器。运行包中的 `opencl_resources`、资源相对名与资源字节保持原样，不将物理源码目录改名扩大成运行协议变化。

不采用缩写 `application → app`、`configuration → config` 来宣称减层，也不把 `stage1/montgomery` 拼成单个目录名。前者只缩短字符，后者把两项职责挤进名字。保留 Stage1/Stage2 和曲线归属，有利于增加真实支持的后端/曲线组合。

## 实施顺序与验收

方案 A 已获用户授权，以下为本轮实施顺序。沿用现有 local 构建脚本和隔离输出目录，不重新开展已经完成的 P0–P7，也不运行 release。每个实施阶段同步草稿和证据、按 `[0.45.z]` 提交；程序版本随实际源码阶段递增，规划基线 `0.45.24` 保持原身份。

1. **D0：固定清单。** 复核本草稿的 31 项逐文件映射、工作区已有改动和当前 local 配置；记录源/资源摘要、源组成员与当前增量状态。识别引用分类：生产 include、构建/测试/生成器路径、当前文档链接、冻结历史证据。若清单已变化，先更新规划。
2. **D1：精简普通职责家族。** 迁移 29 个非 backend 文件，修正配置、service、platform、共享曲线头与 CLI 引用。保持每个文件/翻译单元独立。先核对现有构建入口与源清单中的路径，再做本阶段 local 编译和相关配置、表达式、规划/队列、曲线/结果冒烟。共享头路径变化仍可能触发 CUDA 编译，属于本阶段的预期成本。
3. **D2：精简 Edwards 私有分类。** 迁移 checkpoint 与 host_wait 两个头，修正主机流程、实际设备协作和诊断探针引用。验证 checkpoint/save 恢复和 host-wait 探针；保留全部实例 .cu、PRAC 和设备源组。若 D1/D2 可在同一 local 验证窗口完成，可安排一次集中的 CUDA 重编译；各阶段仍需有独立验证结论，不承诺总共只编译一次。
4. **D3：收口验证与当前文档。** 比较迁移前后的源组成员在映射后的集合、编译语言与选项；项目 include 无悬空/新循环、公开头不引用私有头、公开家族对应仍成立。使用现有 local 构建覆盖受影响目标，运行 Montgomery/Edwards 算术与实际恢复/队列交接代表案例，再复核普通 CXX 正文修改的 CU 重编译数为 0、恢复后重复构建无多余编译。将正式架构/构建/网页描述改成实际目录；保留此草稿供用户处理。

若选择 B，在 D0 后另设独立的 backend 根迁移阶段，统一改路径并验证资源打包和后端目标；不在 D3 收尾时临时追加。Android 若只完成路径核对而未实际构建，必须明确记录验证限制。

路径移动、改 include、改源清单都可能使现有 CUDA 对象失效，**不能用“非 GPU 行为修改零 CU”要求禁止这次必要重编译**；这条要求针对结构稳定后的普通 CXX 实现正文。实际 CUDA 分配、同步、回读继续依赖 CUDA runtime，不增加脱离 CUDA 的替身实现。

无需为纯路径调整重新追求运行性能提升；若源组、编译选项、生成字节或数学输出发生变化，则先定位原因，不能按目录整理直接验收，也不能沿用旧资格冒充新结果。遇到未预期的 ABI、生成协议、资源名称、算术或恢复行为变化，停止该阶段并说明问题。

## 引用与生成物处理

A 不移动 generated 目录、配置定义或公开头。仍要检查生成输出中的相对 include 是否经过新路径；不能手工编辑生成物。若需要更新生成内容，改[配置生成器](../../tools/gen/generate_ecm_config.py)，重新生成并核对 [generated 清单](../../config/ecm_config.generated.json)，默认值仍只在 `config/ecm_options.json` 维护。

A/B 都应更新当前生效的源路径测试、构建指纹输入和 [CUDA 网页内容](../web/cuda-architecture.content.json)/[生成器](../../tools/gen/generate_cuda_architecture_web.py)所用入口。冻结基线日志、历史摘要和原始失败记录保留原样；旧资料需要阅读映射时补链接，不把历史路径批量替换为当时不存在的新路径。外部 `.refactor` 与 `docs/paper` 不属于本次调整范围。

方案 A 是本轮首选：先获得 9 个无效分类目录的导航收益，保留可独立理解、开发和验证的 Module；只有明确需要最大 4 层，才支付 B 的大范围路径迁移成本。
