# Stage2 内存、生命周期与规划

## 计账口径

MB 日志/预算按 MiB（2²⁰ bytes）换算。设备 free/total 是驱动查询；owned payload 是程序显式拥有的分配字节；模块 peak 是该模块自己的峰。CUDA 上下文、库/驱动开销及其他进程不属于 owned payload。

完整 owned 峰应为 maxₜ[NTT(t)+S4(t)+giant(t)+fold/frontier(t)+其他 owner(t)]。各模块 max 的和只是上界，不能当作实测进程峰。NVML 采样包含背景占用，且可能漏掉短暂峰；释放的逻辑字节也不保证立即等量增加设备 free。

## 预算的作用

| 预算/统计 | 控制或记录 | 不覆盖 |
| --- | --- | --- |
| arena | NTT 工作区、小缓冲、根表、fuse 基础缓存 | 独立 S4、giant、fold/frontier、按次回退、驱动 |
| fold owner | 驻留 F/finv/H/G 与归约临时槽 | NTT、giant、F 子树、下降元数据 |
| batch | 一次多项式请求的分块 payload 策略 | 已保留缓存、单 slice 超预算、完整峰 |
| giant 坐标预算 | X/Z 点块和按 P 对齐策略 | seed/segment/group、其他阶段 |
| reserve | 设备准入预留空间 | 全流程永不 OOM 的证明 |

arena cap 不足时可淘汰缓存或按次分配；owner 预算/headroom 不足时回退到主机路径。回退算术必须一致，代价是传输、组装或重算。只把 cap 调大可能挤掉更有收益的 owner。

## NTT 容量

设 L=2ᵏ、b 为一个物理 chunk 的 slice 数，u 为输出 digit slots：

- pool A/B/Q：8·L·b·v bytes，v=2（适用 B/Q 复用）或 3；保留容量由实际最大请求决定。
- 非 pool 的 keyed A/B/Q：24·L·b bytes，各形状键分别持有。
- digit 输出与结果：8·u·b+16b bytes；输出增长时旧输出先释放，结果缓存仍可能存在。
- carry scratch：达到实际阈值 L·b≥2²⁰ 时为 8·⌈L/256⌉·b；b 为 grid 分割后的 inner batch。

fuse 使用 tile 大小 2ᵗ；outer 第 p 个 pass 的 radix 位数为 mₚ，前面累计位数 aₚ：

- 两个 tile 根表：16·2ᵗ bytes。
- 前向/逆向 pass 表合计：16·Σₚ[L/2^(aₚ+mₚ)+2^mₚ] bytes。
- 基础容量：8·(2^(t+1)+scratch_words+radix_words) bytes。

compact scratch 为实际 pass 需求的最大值，radix 为最大需要 radix；legacy 配置可能保留 L/2+64 scratch 和更大的 radix 容量。零尺寸项不分配。实际值来自 `fuse_describe`，不能只根据 L 猜配置。

### 有序 NTT 模型

`NttMemoryState` 记录每次正常分配/释放的 site、索引、L、slice、字节和完整 NTT live。表按前向 pass 顺序、逆向逆 pass 顺序申请；释放按 native pass/site 顺序，不能假设与申请顺序相同。

cap 淘汰按实际策略处理 carry、cold 表和 keyed buffers，保留 context/base；命中的 context 不必重新缓存已淘汰的表。工作区增长先释放旧容量，冷 trim 可释放完整 context。插入顺序和保留容量是模型状态的一部分。

`ntt_memory` JSON version=2，`exact_allocation_events=true` 表示从实际 fuse 描述得到逐项事件；legacy 汇总 callback 标记 grouped，不能声称逐 CUDA 分配匹配。计数描述成功前缀，发生 cap refusal 时不包含其后的按次回退；计划计数不包含尚未发生的最终 close。

重复请求只有完整保留状态固定后才压缩；压缩计数覆盖重复次数，但 observer 只看实际模拟的事件。不能拼接分别压缩的 NTT/S4 回调来计算联合峰。

## NTT/S4 联合组件

`workspace_memory` 使用同一请求程序交织 NTT/S4 事件：形状常量与自检先于输出/raw 增长，随后执行 NTT 分块，再申请首次 canonical counter。G 树租约包围树请求，在 fold 前释放 metadata；inverse/fold 边界按策略释放 raw/output。重复批次仅在两套完整容量状态都固定时压缩，最终按生产声明顺序释放。

查询返回联合 `peak_bytes` 及该时刻的 `ntt_at_peak`、`s4_at_peak`，这两项属于同一瞬间，可以相加核对联合值。各自的 `ntt_peak_bytes`、`s4_peak_bytes` 仍为不同时间的组件峰，不能相加称为联合峰。`simulated_events` 只计实际模拟事件，压缩跳过的块数另列；它不是整条曲线的分配次数。

该模型当前只覆盖条件驻留路径的 NTT/S4，输出 `process_peak_complete=false`、`admission_model=false`。NTT cap 拒绝时返回有效成功前缀和 `finished=false`，不把后续按次分配回退当作已覆盖。giant、fold/frontier owner、自检之外的其他设备分配、驱动开销及冷 trim 仍未联动；自动 D 尚未根据这个组件结果放行。

CPU验证使用合成形状/表布局，1152组输入检查压缩与逐次执行、各组件最终容量和峰值、逐事件收支、拒绝前缀以及策略不一致，455270项断言通过。它验证模型组合与压缩，不证明生产 fuse 描述或 CUDA 实际分配成功。证据：`data/experiments/stage2_workspace_memory_20261009/cpu_v2/`；入口：`src/core/ecm_stage2_workspace_memory.h` 的 `workspace_memory_plan`。

生产描述查询另覆盖8组：不同D、raw/output trim、两缓冲复用、keyed缓存、32 MiB cap拒绝和G1不支持。输入为M6011余因子5872 bits、B1=20、sigma=26，通常B2=2.6×10¹²；arena6300/fold640/batch256 MiB，设备为4060 Laptop、CUDA13.3、sm89。D=1141140/P=103680的旧静态 `arena_estimate_bytes` 为9219.58 MiB，联合NTT/S4峰为3588.26 MiB；这些是规划计算值，不是实测进程峰，更不包含fold owner或giant。查询中的NTT/S4各自峰与原组件查询一致，拒绝前缀和G1未覆盖状态正确保留。

该轮完整构建通过；新exe SHA256=`b1efdba90c099f3d9cf4412ed701369885fd721be882889b5d8fde11caa226cd`。一条M503余因子318-bit、B1=20、sigma=26、B2=2.6×10¹⁰、D180180曲线验证中，mandatory GMP自检2016 cases和检查3032项均bad=0，结果无hit/bad_factor；它不是完整准入/性能验证。生产查询证据在 `data/experiments/stage2_workspace_memory_20261009/native_plans/results.json`，曲线证据在同目录的 `smoke.log`、`smoke.result.jsonl`。

## S4 容量与树租约

S4 持有 raw A/B、归约输出、临时 pack、模数、按完整 shape 缓存的常量及 canonical counter。每个新形状的 96 窗口自检瞬时申请 8·96·slot_words 和 8·96·W bytes，然后释放；它发生在该次输出/NTT增长前，旧 NTT 缓存仍可能存活。

输入分为 host、tree_raw、fold_owner、frontier_owner。host raw 是 S4 所有；后两项借用其他 owner，不能按数据长度再次计为 S4 分配。

G 树具有 tree_begin/tree_end 租约：raw 容量可保留，树元数据在树结束后、fold 前释放。inverse_done/fold_done 边界按配置释放 raw/输出。S4 模型记录 after_inverse、after_giant、final_payload、peak 和 released_payload；最终释放状态必须为 0。

`s4_memory` 目前只保证条件驻留、直接设备 packing、满次数 fold 的正常路径，不模拟 owner/NTT 准入和真实 driver free。`valid` 不等于完整曲线可驻留。

## Fold 与 frontier

默认复用 q/qb、G/reverse 时，fold owner=8W(7P+7)+48 bytes。两类源/结果槽保持各自所有权；别名只跨已结束的用途，default stream 顺序保证旧读者结束。不同 reuse mask 用共享 `fold_owner_layout` 计算，不复制手工公式。

GPU Gamma 校正与 scaled 根准备复用死槽，不增加持久大数组。进入下降前 owner handoff/释放与 frontier 新分配可能交叠，必须按实际事件计账。frontier 的 logical_peak 只表示算法活跃状态，不包括全量 F 数据、metadata 和驱动；host/pinned staging 单独记录。

完整 F 树普通系数数据近似 O(W·P·log₂P) bytes；实际非空子树次数、padding、索引和主机对象容量须按存储布局核对，不能将 padded P 当作所有层真实系数数。

## Giant 组件

令 c 为保留 seed/点容量，v 为叶值容量，r 为块积容量，f 为 segfix 表容量，h∈{0,1} 表示 base：

S3 workspace=8[5W+(2W+1)c+W(v+r)+W(f+1)·[f>0]+2Wh] bytes。

点 X/Z 坐标=16·Q·W bytes；segment/group 使用各自数量×8W。chain seed 数约 2⌈Q/Cchain⌉+1，ladder 则可能直接保留 Q。尾块切换到 ladder 时可以增长 seed 容量，并保留至末阶段；不能只记录最大满 chunk 的 chain seed。

积缓冲按 r=⌈P/64⌉ 分配，叶值仍为 P。相对于 r=P，节省 8W(P−⌈P/64⌉) bytes。这个末阶段容量收益不必降低发生在 giant 阶段的完整峰。

`giant_memory` 按保留容量和满/尾 chunk 路线预测本组件。非驻留 G 叶并不取消此前 chain 坐标生成峰；小素数处理可能留下初始容量。组件程序最多压缩为满块与尾块，不模拟更早 NTT 拒绝或完整回退。

## 现有查询与验证

plan-only 提供真实 packing、精确非空树组、请求顺序以及 NTT/S4/giant 的条件组件结果；当前输出明确不保证 full process peak、准入和全部 fallback。未来联合规划必须使用共同请求/边界，压缩条件同时检查所有组件状态，并保留实时 free/headroom 查询。

当前 NTT 事件模型的 CPU 账本从生产分配语句生成：173 cases、47,936 events、168,166 assertions，失配 0；故意改变表释放顺序会拒绝。40 个 native plan 查询和 10 条短 GPU 曲线通过组件/路由及算术核对。CPU opaque allocator 不验证 CUDA 物理分配失败、驱动驻留或完整进程峰。

对应冻结二进制 SHA256 为 `55ea67855f8bbf54057293370bac9b6ac04e62953f037682e2137dc756bb1c97`，47 份编译源闭包。原始结果在 `data/stage2_ntt_events_20261009/`；该批没有新增大规模性能结论。

M8011/80111、carrier8011、B1=20、B2=2.6e12、D1381380/P126720、batch256、arena6300 MiB、pool+B/Q 的查询：big=4,294,967,296，digit=7,410,584，table=363,878,560，base=172,170,992 bytes；NTT 成功前缀峰 4,838,427,432 bytes（4614.284 MiB），324 alloc/14 free。它不是同配置实际整曲线峰；三缓冲版本会 cap refusal，较小前缀峰不表示更节省。

## 代码与工具入口

- [ecm_stage2_geometry.h](../../src/core/ecm_stage2_geometry.h#L53)：共享 owner、树与请求容量。
- [ecm_stage2_ntt_memory.h](../../src/core/ecm_stage2_ntt_memory.h#L58)：fuse layout；[状态](../../src/core/ecm_stage2_ntt_memory.h#L110)；[event](../../src/core/ecm_stage2_ntt_memory.h#L144)；[plan](../../src/core/ecm_stage2_ntt_memory.h#L363)。
- [ecm_stage2_s4_memory.h](../../src/core/ecm_stage2_s4_memory.h)：S4 逐项容量；[s4_program_plan](../../src/core/ecm_stage2_s4_program.h#L50)：请求/租约联动。
- [ecm_stage2_giant_memory.h](../../src/core/ecm_stage2_giant_memory.h#L9)：giant 容量与保留状态。
- [plan-only 接入](../../src/cuda/ecm_cuda_stage2.cu#L8762)、[工作区工具说明](../../tools/bench/README_STAGE2_CARRIER_PLAN.md)。
- [性能](../performance/STAGE2.md)、[TODO](../TODO.md)。
