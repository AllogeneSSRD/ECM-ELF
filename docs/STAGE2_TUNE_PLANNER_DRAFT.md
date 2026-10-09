# Stage2 tune 与联合选型开发草稿

本文件是开发中的临时草稿。目标完成后提示用户处理；不作为已实现业务规则。

## 完整目标

- 自动 D 在正确的树形与同时存活显存预算下评估较大 D，使用 tune 实测成本排序。
- 对有效 Stage1 save 的目标 N，同时评估普通模数与满足 N∣2ᵖ−1 的承载候选，显式选项优先，无有效匹配数据时保留回退。
- tune 支持等级1—10，逐级增加覆盖、细度与重复次数；除 NTT 外使用梅森素数或已验证无因子的 ECM 输入。性能数据采用可扩展、可读的命名字段，不包含路径或二进制摘要。
- 将实测性能、算术资格、内存准入分别处理；不以吞吐测量放宽算术证明，不以组件峰值相加替代同时存活容量。

## 已完成的格式与 NTT 等级基础

原生驱动默认写 TOML，支持显式 JSONL 兼容输出；两个格式均不保存二进制/manifest摘要。原子发布与运行中身份变动检查保留。NTT等级预设和显式参数覆盖已接入驱动，未更改 CUDA算术和原有 mandatory checks。

等级ℓ：log₂L=16…[20+min(ℓ−1,7)]，重复2ℓ²+1次，另预热1次。该规则当前只覆盖NTT，不代表完整 ECM 调优等级已经完成。

代码入口：`src/core/ecm_stage2_tune_format.h` 的 `effort`/`table`，`src/core/ecm_cuda_stage2_main.cpp` 的 tune 分支；构建脚本纳入新头文件的依赖摘要。

### 验证证据

- `tools/test/test_stage2_tune_format.py` 编译原生序列化 fixture，用 Python TOML reader 验证数值、数组、字符串与分节；测试非法 callback 拒绝、等级1—10与越界等级。证据：`data/experiments/tune_format_20261009/native/`。
- 驱动经 MSVC直接编译、NVCC链接，复用冻结的 CUDA对象；原脚本经 NVCC编译驱动两次出现编译子进程 `0xC0000005`，不能据此声明该完整构建流程通过。原生fixture和直接MSVC驱动编译均通过。辅助命令与对象保留在 `data/experiments/tune_format_20261009/`，旧冻结二进制未修改。
- 当前设备枚举重新为4070 Ti=0、4060 Laptop=1。短正确性验证先在设备0执行，随后在空闲设备1执行 log₂L=16…17、每个3次；TOML解析通过，两个长度 `bad=0`。这只是格式/正确性验证，不用于性能标定或选型排名。证据：`runtime_4060.toml` 与 `runtime_4060.log`。

## 后续实现与验证

### NTT/S4 联合组件进展

同一执行器已交织NTT和S4事件，保留selftest→output/raw→NTT→canonical的顺序，按树租约和阶段边界释放，并要求两边完整状态共同稳定才压缩。组件已接入plan-only，输出联合峰和峰时分项；不作为完整曲线准入、不改变默认D。

合成描述CPU矩阵：1152 cases、455270 assertions；压缩/未压缩联合峰一致，各组件最终容量与原模型一致。原组件回归：NTT681389 checks；S4 299592 checks、2136 events、192 tree leases，bad=0。证据均在 `data/experiments/stage2_workspace_memory_20261009/`。

构建诊断的最小源码对照表明：直接cmd日志文件重定向触发NVCC编译器探测崩溃，管道捕获通过；此前把问题仅归于命令传递的假设不成立。构建脚本改为独立命令文件和管道收集，新增C++/CUDA compile-only回归通过。受限环境完整CUDA编译仍遇到ptxas INVALID_HANDLE，相同参数以常规权限完整构建通过，约105秒；不改GPU设置。固定旧实验exe保持不变。

真实描述的8组native plan通过，覆盖不同D、trim、BQ、keyed、拒绝前缀及G1不支持；新测试入口为 `tools/test/test_stage2_joint_workspace_plan.py`，原workspace-plan测试保留。5872 bits、B1=20、sigma26、B2=2.6e12、D1141140/P103680，arena6300/fold640/batch256：旧静态估计9219.58 MiB，NTT/S4联合组件3588.26 MiB。未包括fold/giant，不能直接作为准入。

一条M503余因子318-bit、D180180、B2=2.6e10曲线通过，自检2016、GMP检查3032、bad=0，hits=0。新exe SHA256=`b1efdba90c099f3d9cf4412ed701369885fd721be882889b5d8fde11caa226cd`，日志/结果在 `data/experiments/stage2_workspace_memory_20261009/`。未宣称性能收益或自动D已改变。

### Fold/frontier 联动进展

`OwnerMemoryState`按生产成功路径逐项记录fold两块大缓冲与map/length/modulus/digest、frontier metadata；inverse与descent边界接入同一工作区执行器。预算前缀拒绝与动态headroom/物理分配拒绝分开，后两者尚未模拟。`resident_workspace_memory`加入owner峰时分项，原NTT/S4查询保留。

提取生产分配/释放语句，140 cases、1960 allocation/free events、6580 checks通过；metadata extent故意改变8 bytes被拒绝。联合压缩1152 cases、946250 checks通过。10个native plan及普通/承载503/fold-budget0三条完整曲线通过，规划布局与运行布局字段一致。原始证据在 `data/experiments/stage2_owner_memory_20261009/`。新exe SHA256=`1d11589c7b1eb673a25f6850f21a7c1e0a2b67f31848e4c169245155f701fe2a`，完整生产构建约94.5秒。

5872-bit、D1141140/P103680、B1=20、sigma26、B2=2.6e12、arena6300/fold640/batch256的NTT/S4/owner联合规划峰4098.979 MiB。它不是实测进程峰，未含giant；自动D仍未按该值放行。下一步将点chunk的生成/保留/销毁和S3容量变化接入共同时间线，并覆盖更早的初始化瞬时量，之后才用于准入。

### Giant 联合时间线进展

`GiantMemoryState`把S3五常量、小素数初始容量、seed/base/segfix、坐标/segment/group和叶值/积按正常成功路径加入统一执行器。S3在混合初始块的首个inverse请求前构造；point chunk生成在fold准入之后、首棵G树之前。chunk跨多棵G树/fold时保留其transient，最后一棵消费完成才释放；压缩仅跳过四组件状态都相同的完整chunk周期，尾部路线变化仍执行。

新增`curve_workspace_memory`查询；旧两种查询保持原scope。模型仍不是完整准入，自动D与承载默认未改变。5872-bit、D1141140、B2=2.6e12、arena6300/fold640/batch256四组件规划峰4406.441 MiB；floor4251.702，BQ3382.441。这些规划值不构成性能证据。

CPU joint矩阵3456 giant cases和1152原cases，共3392047 assertions；超大B2只执行10块。生产S3提取fixture2051 cases、39573逐事件对照通过。14个native plan和三条完整曲线通过；GMP坏计数与坏因子均0，S3四阶段保留容量及closed owned ledger核对通过。fallback下降先申请叶值，在runtime verifier中按实际路径单独核对，不把joint refusal前缀当作回退时间线。

构建96.4秒，exe SHA256=`a90cb1ca1045f1fdc3dd5ae906d949483e091bf68668fd6ae8471615b2ac3beb`。当前4070 Ti设备0忙，验证使用空闲4060 Laptop设备1，无功耗/频率变更。证据目录`data/experiments/stage2_giant_timeline_20261009/`。native查询collector首次引用了不存在的JSON键，修正为现有`I`字段后14例通过，失败输出保留；fallback retained-capacity初次假设叶值在下降后分配与native不符，按当前`!frontier.enabled`路径修正检查，原证据保留。

### 初始化与free快照需求

共享`baby_memory_layout`替代caller和D模型的重复容量计算，显式检查溢出。基础mont selftest在S4之前完成；S4模数之后才生成baby，临时量在F树之前释放。`curve_workspace_memory`升为version2、五个峰时分项；baby预算0返回有效拒绝前缀，增加初始化分配的诊断返回不可用。条件模型未成为完整准入，不改变生产自动D或承载默认。

联动执行器记录baby预留64 MiB、fold/frontier各自未来NTT增长+1 GiB所需的启动free。`required_free_bytes`同时包含联合峰+reserve；`initial_free_snapshot_fits`只是正常成功路径的静态估计。继续保留实际cudaMemGetInfo、cudaMalloc失败及回退，不要求先把全部失败路径精确建模才推进正常路径的tune筛选。

CPU50 cases/1500 native events/4556 checks、错误tree extent+8 mutation拒绝通过；联合初始化3456 cases，总6097043 checks，headroom公式由逐事件边界独立核对。16个native plan和最终binary三条短曲线通过。首个fixture漏了string include导致编译失败，修正后通过；失败输出保留。最终exe SHA256=`4a734f4cab13ae3b4f76d5b65bc3921cd94c826ec427cc30c14323bc018bf466`，完整构建96.6秒，所有编译源hash与当前实现相同。

另一个冻结初始化binary SHA256=`747753d0e09e0b09f181ca04b5a61cf6e42cb8968c50bb7d9f1913cd2102cbe3`完成5872-bit、B1=20、sigma26、B2=2.6e12、D1141140/P103680、arena6300/fold640/batch256曲线；固定1800 MHz，设备1，无设置变更，单条wall96.761609秒。预测/owned账本峰都为4,620,488,216 bytes，mandatory2208、GMP43253、bad=0、hits0。短普通/承载曲线也核对峰一致，fallback不做完整joint峰断言。证据`data/experiments/stage2_initial_timeline_20261009/`，包含每次build对应的frozen sources、查询、原始日志和分配账本。

下一步直接接入完整ECM tune、版本化reader及正常可驻留候选选择；继续保留现有无数据/不适用的兼容路径，不将单条相同峰验证当作所有输入的物理准入证明。

- [ ] 在正常路径联合模型的条件范围内接入候选筛选，保留实时free检查；扩大验证，并补齐动态headroom与非驻留回退模型。
- [ ] 增加有版本的实测数据reader，明确设备、后端、算术路径及测量输入适用范围；路径与二进制信息放实验审计，运行tune数据只保留必要性能与资格字段。
- [ ] 引入无因子完整ECM基准、普通/承载配对测量与D候选网格，将等级映射到完整测量计划。
- [ ] 将选型连接到显式B2主路径和Auto B2；当前静态准入及手动承载均未由本轮改动替代。
- [ ] 先复测5872-bit较大D、7995-bit承载，再覆盖更多位宽/预算与非单位场景；按同设备条件验证候选收益。
- [ ] 实施完整构建、回归与性能验证，更新权威说明。
