# CUDA 多项式 Stage2 管线

## CUDA 模块边界

`src/ecm/stage2/cuda/` 是生产 Stage2 的普通 C++ 主机模块，私有头声明模块间合同，不包含整段引擎实现。公共入口仍为 `src/ecm/stage2/cuda/api.h`。

- `engine.cpp` 接入一条存档曲线，编排初始化、relocation 试算/重放、检查收尾和结果；`batch.cpp` 执行单次区间的 giant 批循环、fold/下降阶段和因子扫描。
- `plan.cpp` 维护 D 选择、设备/形状/请求/预算查询及计划报告。执行与查询复用 `prepare_stage2_plan`；查询直接使用模数上下文，不进入曲线入口或算术自检。`plan_runtime.h` 保存计划结果及其寿命；纯 giant 坐标分块公式位于 `src/ecm/stage2/planning/giant_chunk.h`。D 扫描继续使用 `legacy_56_1` 公式；`NTT_D_MODEL` 的解析及禁用原因输出保留，冻结的候选速率表只供实验使用。规划仍在原边界预热实际 NTT 设备策略缓存。
- `points.cpp` 维护 baby/giant 生成、分段与分组逆、单位性恢复及 baby 复用缓存。`tree.cpp` 维护 F/G 树布局、实际次数、空位、根选择和子节点释放。
- `product.cpp` 维护普通/批量多项式乘法、输入打包、输出窗口、驻留输入合同与 NTT/S4 计数；不负责树遍历或 GCD。有限的打包启动帮助函数在 `pack_launch.h`，位索引在 `src/ecm/stage2/cuda/device/pack_index.cuh`。`remainder.cpp` 维护 Newton 逆、fold、scaled root/frontier 和下降。
- `s4.cpp` 准备模数/shape 并执行归约 hook；`checks.cpp` 维护必需自检、GMP 抽样、异步快照、carry/归约检查及排空。`factors.cpp` 负责块积读回和目标 N 上的真因子记录。
- `resources.cpp` 实现点、S4、fold/frontier owner 的分配、租约、事件和析构方法；类字段、默认值和接口分别声明在所属模块头中。`context.cpp` 提供共享编码与整数帮助函数。`options.cpp` 在一个翻译单元中先设置生产默认值，再按固定次序构造环境快照及共享状态，保持原来的解析时机。
- `host_wait.cpp` 封装平台休眠资源；`host_wait.h` 统一完成等待和读回前等待，`host_wait_policy.h` 定义可独立验证的有界退避。
- `diagnostics.cpp` 保留可选夹具、GMP 参考、转储、故障注入和崩溃报告；这些诊断仍由现有开关触发。`tune.cpp` 使用生产 NTT 运行时执行校准。

诊断 TODO：`NTT_BABY_DEVICE_TEST` 自建模数的夹具在 M127 的活动梅森点上下文中会报 affine GMP mismatch，通用8193位输入下通过；修复前需核对并验证设备点模数状态的刷新及恢复。正常曲线与必需算术自检不依赖该合成夹具。

主要计算依赖为 engine/batch → points、tree、remainder、factors，tree/remainder → product → NTT/S4 → 中立启动接口。底层模块不包含 `engine.h`，计划模块不依赖曲线执行。资源实现依赖所属 owner 的声明及检查排空接口。可选检查通过有限的 GMP 参考函数调用诊断模块；诊断夹具可调用被测模块，但不能选择或接管生产流程。S4 自检与异步检查通过明确接口复用实际归约和 shape。

`S4ResidentOwner` 持有独立分配，`S4DeviceBatch` 显式携带 owner、范围和元数据，fold/frontier 只借用合同允许的存储。pinned 输入/输出槽及快照事件仍使用原有等待和释放顺序；资源拆分不增加同步。每个成功结果及 relocation 决策前排空检查，reducer 析构时先排空依赖 shape/N 的快照，再释放 shape、设备/主机缓冲和事件。

`src/ecm/stage2/cuda/device/arithmetic_kernels.cu` 维护点与系数算术的显式实例清单。`point.cuh`/`point_mersenne.cuh` 处理点算术，`giant_chain.cuh` 处理相邻 seed 和 chain，`baby_giant.cuh` 处理 baby 归一化，`s4_reduce.cuh` 处理系数归约，`pack.cuh` 处理输入打包，`coefficient_ops.cuh` 与 `plain_scale.cuh` 处理系数搬运和缩放；不使用 RDC。

位宽清单仍由 `src/ecm/cuda/dispatch/tiers.def` 统一维护。Stage2 容量为 4/8/16/32/64/128/256 个 64-bit 字，按实际 W 选择最小可容纳档；循环、Montgomery radix 和梅森尾位使用实际 W/S。local/dev/release默认使用算术与NTT两个聚合编译单元；支持的工具链通过nvcc内部并行优化。生产引擎可显式选择 `KernelLayout=isolated`，由 `tools/build/internal/generate_cuda_modules.py` 从原显式实例、设备定义和薄启动器派生176个独立模块，不改写算术表达式或另建容量表。两种布局通过编译参数和构建profile隔离。

聚合模式沿用各聚合单元的设备常量，不启用隔离宏及主机快照路径，不保证逐容量隔离JIT。隔离备选的每个算术模块持有自己的设备常量；`constants.cpp` 保存按设备区分的主机快照及修订号，更新快照不访问全部设备模块，被调用的薄启动器在修订号/设备变化时才向所属模块上传。NTT的 `module_dispatch.cpp` 只返回选定tile句柄；固定mode3无需广播归约模式。实际按需JIT资格须观察驱动加载模块，构建策略与旧驱动限制见[构建与发布](../usage/BUILD.md#完整发布与按需加载)。

NTT 主机规划、运行时、Graph 和校验位于 `src/ecm/cuda/ntt/`，设备变换和 carry 位于 `src/ecm/cuda/ntt/device/`，域原语位于 `src/ecm/cuda/arithmetic/goldilocks/`。主机持有缓冲、事件、错误检查和计时，设备模块提供薄启动接口；依赖方向为 ECM 编排 → NTT/算术接口 → 设备原语。NTT 不依赖 ECM 树、fold 或 GCD，详见 [NTT 代码入口](NTT.md#代码入口与依据)。

生产源由 `STAGE2_PRODUCTION` 清单选择，其中 `STAGE2_ENGINE` 显式列出18个主机模块。生产闭包拒绝 `tools/bench/`、`tools/test/`、`tools/experiments/` 和 `.refactor/`；编译器发现的间接依赖也受同一检查。

实验实现集中在 `tools/experiments/cuda/`：`stage2_tree/worker.cu` 是开发 wrapper，`main.cu` 是独立树入口，`engine.cuh` 与 `tune.cuh` 维护实验树和校准；该引擎拥有自己的 plain-scale 主机适配，不包含生产 Stage2 私有流程头。`stage2_bsgs/` 保存旧 CGBN BSGS 候选，`ntt_variants/` 保存独立 NTT/carry/outer/cuFFT 候选，`ntt_tensor/` 保存 Tensor 算法候选。它们通过独立源组显式构建，不进入生产目标。baby payload 的纯尺寸公式位于 `src/ecm/stage2/planning/initial_memory.h`，JSON 转义及纯设备原语仍可共享。

`tools/test/` 保存 GMP oracle、夹具和诊断适配，`tools/bench/` 保存测量入口。探针通过中立接口链接实际生产算术/NTT 对象，不包含 `.cu` 或截取实现正文。giant 探针的已注册普通 chain 使用生产对象；旋转候选及160/192字控制组使用独立 `giant_variant.cu`。Edwards host-wait 通过 `src/ecm/stage1/edwards/cuda/host_wait.h` 和唯一显式测试适配连接实际 Stage1 运行时。CPU 内存模型测试的源码提取仅用于 CUDA 测试替身，不参与产品构建。

## 输入与表示

输入为完成的 Suyama PARAM0 或 Atkin–Morain Edwards Stage1 点 Q、目标 N、B1/B2、设备及预算。普通模式在 N 上计算；梅森承载实验在 M=2ᵖ−1 上计算，再在 N 上解释单位性、投影、GCD 和因子。S=bit_length(M)，W=⌈S/64⌉。

### Stage1 save 与承载资格

save提供目标N、sigma、B1和归一化X，不自动推断承载指数。Suyama记录支持PARAM=0、Z=1或省略Z，X须在[0,N)；存在checksum时必须通过校验。Edwards记录要求方法、版本、48LCM标量约定、状态及校验齐全，禁止PARAM字段；格式统一见 [Stage1 Edwards](STAGE1.md#gpu-edwards)。Edwards在目标N上重构曲线，使用 `(A+2)/4=1/(1-d)` 作为统一 baby/giant 点生成常数；Prime95 的 Ad4 是其倒数。该路径直接复用 NTT 多项式计算，不重跑 Stage1。Edwards当前要求显式B2，实验Stage2引擎不消费这种存档。

承载要求N为大于3的奇数、bit_length(N)≤p≤16384，且实际验证N∣2ᵖ−1；显式`--carrier-exponent p`不满足条件时报错。p不要求为素数，也没有按目标位宽自动启用的固定阈值。

完整tune可按成本与联合显存比较普通/合法承载，允许配置跨条件复用及宽度/B2估算，详见[Auto B2/tune](AUTO_B2.md)。合法梅森worktodo结合已知因子和save验证后提供承载指数；仅save不猜指数。显式`--carrier-exponent 0`锁定普通模数。N本身为完整梅森数时，默认归约路径仍可使用梅森快速归约。

承载路径直接接入save的仿射X及Z=1，不重跑Stage1、不再次应用可选的12倍标量。逆元、单位性与最终GCD均针对目标N；承载增加的是Stage2算术位宽，合法性并不保证性能更优。依据：[save解析](../../src/ecm/stage2/input.cpp)、[承载校验](../../src/ecm/stage2/modulus.h#L28)、[坐标接入](../../src/ecm/stage2/cuda/engine.cpp#L110)。

设备点坐标使用 Montgomery 域，NTT digit 在 Goldilocks 域，多项式系数归约后为普通规范整数。不同表示不能直接相乘或判断数学相等。

D 确定 baby 集 J={1≤j≤D/2:gcd(j,D)=1}，P=φ(D)/2。默认 giant 数 I=⌊B2/D⌋+2，分为 G=⌈I/P⌉ 个 G 树批次；可选后缀范围见下文 relocation。补充小素数路径覆盖主树映射不能直接处理的边界。

## 执行步骤

| 步骤 | 工作与结果 | 主要执行位置 |
| --- | --- | --- |
| 1. 读档与规划 | 验证 Q/N/B1/sigma；选择 D，查询 packing 与组件预算 | CPU、设备查询 |
| 2. baby 点 | 求 [j]Q，得到 X/Z；处理不可逆坐标 | GPU 点算术、CPU GCD/逆 |
| 3. baby 归一化 | 设备积树、主机根逆、设备逆传播，生成普通 xⱼ | GPU 为主 |
| 4. F 乘积树 | 叶 X−xⱼ；逐层相乘，保留 F 与下降所需子树 | GPU NTT/S4，主机编排 |
| 5. Newton 逆 | 预计算反转 F 的截断逆 finv，供多次模 F 归约与根准备复用 | GPU 多项式乘法 |
| 6. giant 点块 | 求 [iD]Q；ladder 或相邻 seed+差分 chain；按坐标预算分块 | GPU 为主 |
| 7. giant 归一化/分组 | 精确 16 点段积、单位性分类、分组逆；形成仿射或齐次叶 | GPU、CPU 目标 N 逆/GCD |
| 8. G 乘积树 | 将本批 giant 叶构成 G，多项式根直接交给 fold | GPU NTT/S4 |
| 9. fold | 累计 H←G·H mod F；使用 finv 与长/短乘法，复用驻留临时槽 | GPU；准入失败有主机路径 |
| 10. 缩放与根准备 | 消除齐次缩放 Γ；驻留 H/finv 形成 scaled 下降根 | GPU；回退时 CPU |
| 11. F 树下降 | scaled remainder/middle-product 传播，最终得到 P 个叶值 | GPU NTT/S4、主机元数据 |
| 12. 累积与 GCD | 设备每 64 叶求块积；CPU GCD 提取真因子；relocation 遇到非平凡 GCD 时重放原区间 | GPU 产品、CPU GMP |
| 13. 验证与结果 | 排空算术检查，验证因子；可选命名/GP 分解；写入成功回执 | CPU 与检查收尾 |

Giant、G 树和 fold 交织运行；阶段墙钟不能视为独立并行任务。每曲线 init 包括 baby 集合枚举、baby/F 树和引擎内 NTT/S4 检查，main 包括 inverse、批循环、下降、GCD 和检查收尾。日志中的 shape 是 baby 集合枚举，已包含在 init 中，不能再次求和。引擎时间为 init+main；Stage1、外层 D 选型、计时前的 Montgomery/xADD 预检和进程启动另计。性能比较使用相同边界，进程墙钟单独记录。

## 点生成与归一化

差分 xADD 使用 6 次模乘，固定差点保证 chain 中的相邻点关系。相邻 seed 可同时得到 [iD]Q 与 [(i+1)D]Q，避免重复计算两个独立 ladder。短块、容量不足或非单位场景仍需保留正确回退。

seed 和 chain 的宽位数核在 W>128 时使用 32 线程 CUDA 块，其他宽度仍为 64 线程；每条顺序 chain 仍生成 64 点。此处 CUDA 线程块大小与 chain 点数是两个独立参数。点坐标、Montgomery radix、seed 标量和回退条件保持同一合同。chain 核位于 `src/ecm/stage2/cuda/device/giant_chain.cuh`，正式构建记录其依赖与哈希。

生产路径的通用 chain 下限为每个点块 32768 点；当算术位宽 W>128 个 64-bit limb、驻留 G 叶可用且其坐标预算至少覆盖 32767 点时，有效下限为 8192。这里 W 取实际算术模数，承载模式按承载位宽计算。更短点块或不足的驻留预算仍使用通用路线；`NTT_GIANT_CHAIN_MIN` 的生产配置约束保持不变。共享选择函数为 [giant_chain_minimum](../../src/ecm/stage2/geometry.h)，原生内存规划与 portable tune 使用相同的有效阈值。

新接入的宽尾块先验证 [D]Q 的单位性。非单位基点直接使用逐点 ladder；若差分链生成后的分组乘积在目标 N 上非单位，则恢复此前的逻辑 Gamma，并在原坐标和分段缓冲区内用逐点 ladder 重做整个尾块。该回退保留原有因子集合及非仿射占位叶，避免将表示差异引入结果。已有大点块的路线保持原合同。缓冲区预留公式见 [Giant 组件](MEMORY.md#giant-组件)。

`device_gleaf.chunks` 记录实际准备次数，`retry_chunks` 记录重做次数；分组、检查和传输计数包含重做工作，`coord_peak_bytes`/`group_peak_bytes` 仍为各自同时占用峰。数学上的唯一覆盖由 `real_batched_projective` 的 `leaves=gamma_points` 校验，不能从准备次数推导曲线数。

分块求逆以积换取少量主机逆：可逆集合使用前缀/积树传播，遇到非单位先在目标 N 上判定和求 GCD。不能在承载 M 上的单位性替代 N 上的单位性。Gamma 保存使用齐次叶引入的累计缩放，在最终下降前校正。

## 多项式与 scaled 下降

F(X)=∏ⱼ(X−xⱼ)。giant 叶为齐次形式 ZᵢX−Xᵢ，或等价的已归一化叶；累计 H 与相关缩放共同保持数学结果。G 树非满尾批按真实次数构建，不能用填满后的次数估计所有 NTT。

默认当 W>128 且本次 giant 区间至少包含 4 个完整 G 批次时，驻留 G 根树使用分块空位布局：底部保留密集的二次幂叶块，将最多 128 个块分散在上层完整树的槽位中，减少非二次幂次数导致的高层输入不平衡。其他位宽及较短 Stage2 保留原来的左侧连续布局。启用条件由 `giant_tree_balanced` 统一用于执行器、请求规划及调优工作量；debug日志 `groot_layout` 记录实际选择。

CUDA固定参数实验可用`NTT_GROOT_BALANCED=0|1`覆盖布局。前端要求固定D/B2，关闭自动选型、校准和采样，并隔离旧tune数据；默认不设置此变量。窄位宽的NTT长度下降形状已有局部收益，也存在回退，不能全局启用；条件与数据见[G树性能](../performance/STAGE2_GTREE_LAYOUT.md)。

真实叶数组保持原密集顺序，空位代表次数为 0 的常数 1；没有新增 giant 点或根。F 树及其有序下降保持原布局，主机树回退仍产生相同根。布局、压缩请求分组和 raw 缓冲区容量依据见 [G 树固定资料](../reference/ECM_STAGE2_G_TREE_LAYOUT.md)；请求规划与设备每层排序使用同一形状合同。

G 树不会为了填满 P 而扩大已经选定的 B2。若显式比较完整覆盖边界，先取 `I=floor(B2/D)+2`、`G=ceil(I/P)`，最小扩展值为 `B2'=max(B2,(G*P-2)*D)`；这是增加实际覆盖范围，须以新 B2 运行并报告。非满 G 的空位不能作为已经完成的曲线工作量。

多 G 批次复用 F 和反转逆，不逐批重建。scaled 下降采用反转/截断/middle product 的等价表达，将原余式链传播到各子树；非空子节点使用实际 operand 长度，空节点复制或补零。归一化后的叶投影与直接 Horner/CPU 参考一致是正确性门禁。

## 可选 relocation

`stage2_relocation=1` 或 `--relocation 1` 启用保守后缀算法，默认0，CLI 覆盖 TOML。Edwards 和 PARAM0 共用该路径。设原 giant 终点 I=⌊B2/D⌋+2，m 为不整除 D 的最小素数，A=max(1,⌊I/m⌋)；试算仅生成 i=A,…,I，共 I−A+1 点。原 B1/B2、baby 集、补充小素数路径和终端 guard 覆盖保持不变。证明与 GCD=N 反例见 [relocation 固定资料](../reference/ECM_STAGE2_RELOCATION.md)。

试算经过既有算术检查排空后，只有没有命中且目标 N 上没有非单位时才允许完成 NF。任何真因子、gcd=N、giant/分组/分段求逆非单位都令试算失效：丢弃其因子结果，重放 i=1,…,I，以原算法结果发布。baby 初始化已发现非单位时直接执行原区间，不先做后缀。F 树与 baby 初始化复用；原区间的 inverse、giant/G/fold、下降及 GCD 重新执行。不同区间的叶指纹通常不同，验收比较各自独立参考；重放完成叶必须匹配原算法。原算法只接收真因子块，gcd=N 本身不会发布成因子。

relocation 的点块末尾不足差分 chain 下限时，在容量允许且两个末块都达到有效下限的条件下，从前一块借取整数个 P 点；点数和点块数不变，局部多项式批次与 giant 绝对索引分别计算。无法满足条件时保留既有 ladder 回退。例如 I=149940、D=180180、m=17，后缀141121点；容量69120、chain 下限8192时分成69120/51840/20161。

选 D、Auto B2 和内存准入继续按原完整区间建模，以保留重放条件。计划输出的 `relocation_estimate_scope=original_replay_range` 明确这个范围；时间预测并非后缀专用成本模型。启用时不做旧 ECM 短标定、不写入旧 ECM 性能样本；完整 ECM tune 显式拒绝，纯 NTT 校准仍可使用。现有实时分配和算术检查继续生效。

内部结果记录 `relocation`、`relocation_replayed`、最终 `giant_first/last/points`、试算 `relocation_points/prime` 和 `relocation_probe_seconds/replay_seconds`。后两者是 main 内的子区间，不能再加到完整耗时。重放后的最终 giant 指标为原区间；NTT 次数包含实际试算和重放。提交文件继续报告原 B1/B2，不发布中间 probe 结果。冻结实测与复现入口见 [relocation 草稿](../development/GPU_STAGE2_RELOCATION_DRAFT.md)。

## 工作量

本节 I 表示单次 pass 的实际 giant 点数：默认取 ⌊B2/D⌋+2，relocation 后缀取 ⌊B2/D⌋+3−A，G=⌈I/P⌉。命中时另加原区间 pass，共用的 baby/F 初始化只计一次。

设 Mpoly(a,b,S) 表示一次系数模 M 的多项式乘法成本。完整 F 树有 P−1 次非空逻辑乘法，G 树总计 I−G 次；这不是 kernel launch 数，多个同形乘法可合并成 batch。

每个树层 h=1,2,4…，满配对数 q=⌊P/(2h)⌋，余数 r=P mod 2h；有 q 个 (h+1,h+1) operand 乘法，若 r>h 再有一个 (r−h+1,h+1) 尾乘法。0<r≤h 时复制，不执行 NTT。真实工作应累加每个形状的 Mpoly，而非只用 P log P。

一般近似：F/下降为 O(Mpoly(P) log P)，所有 G 树为 O(G·Mpoly(P) log P)，fold 为 O((G−1)·Mpoly(P))；giant 为 O(I·W²) 的点算术主项，加 seed 与求逆。每次 NTT 的域运算为 O(L log L)，设备系数归约另计。

## 数据与检查

驻留 G-root、fold、scaled root/frontier 减少主机往返，但 F 子树上传、元数据、最终 P·W 字叶读回及 CPU GCD 仍存在。输入源分别为 host、tree_raw、fold_owner、frontier_owner，借用存储不新计 owner。

必需自检、carry 精确性、系数 GMP 抽样、proper divisor 验证和检查排空属于计算合同；factor-only 只省可选命名。性能测量与额外逐节点/台账诊断分开，不能把诊断曲线当作普通速度样本。

完整叶比较使用目标环：承载计算先将叶 v 投影为 v mod N，再与普通模数或其他点生成路线比较。Γ 的逆元只需满足目标 N 上的合同，不同投影坐标可产生不同的 M 代表元；原始 M 叶指纹适合检查同一路线的重复一致性。承载正确性仍须独立核对目标环完整叶、点生成、分段/分组与缩放，因子集合或最终 GCD 相同不能替代这些检查。

## 主机等待与提交回收

S4 每个 shape 保留 8 对 GPU 计时事件。按提交顺序查询并回收已完成事件；最旧事件尚未完成时停止查询其后的事件。满环时只等待、累计并释放最旧一槽，保留其他在途工作。收尾仍完整排空，每对事件只累计一次。`NTT_S4_TIMING_FIFO=0` 供同二进制对照恢复满环全排空；生产默认 1。`ring_waits` 记录满环次数，回收粒度改变后不能仅用次数比较等待成本。

`stage2_low_cpu_wait` 默认 false，命令行 `--low-cpu-wait 0|1` 可覆盖；配置支持 `[stage2]` 及 `[workers.N.stage2]`。显式开启选择 CUDA 设备阻塞调度（`NTT_CUDA_WAIT_MODE=4`）及事件退避等待（`NTT_HOST_EVENT_WAIT=1`），同步读回保留原生路径（`NTT_HOST_READBACK_WAIT=0`）。显式关闭选择自动调度0及原事件等待。父进程通过环境传给曲线及 ECM tune 子进程；未显式设置时保留专家环境覆盖。事件与读回前等待的环境开关均接受严格0/1，默认0。配置语法和优先级见 [TOML参考](../ECM_TOML_REFERENCE.md#stage2_low_cpu_wait)。

事件退避策略先查询完成事件，已完成直接走原同步边界；只有 NotReady 进入休眠重试，其他 CUDA 错误立即返回。请求休眠按 32、64、128、256 µs 退避并封顶。Windows 使用每线程私有高精度 waitable timer，不更改全局定时器分辨率；创建或使用失败则退回原生 CUDA 等待。请求时长不是唤醒延迟保证，系统抖动可能增加完整耗时，因此该模式保留为可选项。

既有阻塞 D2H 的计算等待也可能消耗 CPU。独立的 `NTT_HOST_READBACK_WAIT=1` 实验让选中的生产点生成、树、fold/下降及 NTT 读回在原读回前检查默认 stream：若尚有工作，记录可复用完成事件并按上述策略等待，之后仍执行原 cudaMemcpy。NTT 只调用可选、每线程的读回前回调，不依赖 Stage2 私有模块；引擎用作用域对象安装并恢复该回调。已有 pinned 双缓冲继续等待对应旧槽，不改成等待默认流最后一项。没有新增 GPU 数据数组、分块或 stream，也没有将等待插入 Graph capture。

事件记录位置、GPU算术、自检、carry、GMP样本选择和最终排空保持原合同。预测与休眠不授权提前消费数据或复用缓冲；检查失败不能发布成功。必需自检中的设备同步、未接入的分配/释放及驱动内部背压仍可能消耗 CPU，低 CPU 模式不承诺进程完全休眠。新增元数据为每个 S4 shape 的 9 个 64-bit 提交序号，以及每线程计数器、一个主机定时器和至多一个完成事件；没有新增传输数据量；事件及驱动内部资源未单独测量容量。

调试日志 `stage2_host_wait_stats` 记录调用、立即完成、待完成、查询、休眠、请求休眠总量、回退次数及各类等待墙钟。墙钟与正常阶段计时嵌套，不能再次相加；它也不是 CPU 时间，不能把 requested_sleep_us 当作实际休眠。NTT 独立 tune 的设备事件计时保持原入口，完整 ECM tune/正常观察将上述四个 NTT 环境开关记录为条件；旧严格 cprof 均要求 0。具体性能与覆盖范围见 [主机等待验证](../performance/STAGE2_HOST_WAIT.md)。

代码入口：[等待策略](../../src/ecm/stage2/cuda/host_wait_policy.h)、[完成与读回等待](../../src/ecm/stage2/cuda/host_wait.h)、[平台定时器](../../src/ecm/stage2/cuda/host_wait.cpp)、[S4计时槽](../../src/ecm/stage2/cuda/s4.cpp)、[NTT读回接口](../../src/ecm/cuda/ntt/runtime.h)。第三方原理见 [GPUOWL / PRPLL队列](../reference/GPUOWL_OPENCL_QUEUE.md)。

## 当前规划边界

`request_program`预测满次数驻留fold的乘法顺序、形状、来源和树租约。初始化/NTT/S4/owner/giant按同一时间线计算条件峰和free需求，host fold/frontier可使用受限arena的保守上界。驱动以tune实测或估算选择D及合法承载；单G、baby预算回退、诊断及全部分配失败不在统一保证中。引擎保留实时检查，详见[内存](MEMORY.md)与[Auto B2/tune](AUTO_B2.md)。

## 代码依据

- [生产引擎](../../src/ecm/stage2/cuda/engine.cpp)：`run_real`、relocation 和 [完整计时输出](../../src/ecm/stage2/cuda/engine.cpp#L607)；[批次流程](../../src/ecm/stage2/cuda/batch.cpp)：`run_batched`；[规划](../../src/ecm/stage2/cuda/plan.cpp)：`prepare_stage2_plan`、`ecm_cuda_stage2_plan`。
- [模数上下文](../../src/ecm/stage2/modulus.h#L28)：`configure`，目标/承载资格。
- [请求程序](../../src/ecm/stage2/planning/requests.h#L78)：`request_program`；[F树组](../../src/ecm/stage2/geometry.h#L147)：`tree_multiply_groups`；[G树布局/分组与选择](../../src/ecm/stage2/giant_tree.h)：`giant_tree_balanced`、`giant_tree_layout`、`giant_tree_multiply_groups`。
- [点模乘](../../src/ecm/stage2/cuda/device/point.cuh)、[梅森点算术](../../src/ecm/stage2/cuda/device/point_mersenne.cuh)、[NTT](NTT.md)、[内存](MEMORY.md)。
- 固定资料：[多项式数学](../reference/ECM_STAGE2_MATH.md)、[Prime95 Poly](../reference/PRIME95_STAGE2.md)、[PrMers ECM](../reference/PRMERS_ECM.md)。

## 进度与文件机制的接口

[stage2_checkpoint.h](../../include/ecm/io/stage2_checkpoint.h)公开任务身份和曲线进度数据及读写/回执声明；[stage2_checkpoint.cpp](../../src/io/stage2_checkpoint.cpp)实现 schema、原子落盘、完整结果行与 finished 回执检查。旧身份解码和导入也在该实现内；任务层通过匹配回调提供数学一致性判断。单任务 pending/确认/封存/源行复核由父流程排序，文件模块不拥有另一个任务循环。

[fingerprint.h](../../include/ecm/platform/fingerprint.h)是 SHA256 的普通 C++ 声明；Windows CNG 的动态库与句柄实现仅在 [fingerprint.cpp](../../src/ecm/platform/fingerprint.cpp)。两个模块不依赖 CUDA SDK，构建使用普通主机编译器；实际计算引擎仍保持自己的 CUDA runtime 依赖。

## 主机规划实现

[planning/planning.cpp](../../src/ecm/stage2/planning/planning.cpp)独立编译方案选择、校准、测量和 profile 合并；缓存和作用域状态保留在同一编译单元。配置调用方只投影 B2 默认值与控制台预算，规划实现不包含 TOML/完整 Settings 实现头。实际设备选择和显存快照仍使用 CUDA runtime。共享调用类型位于 `src/ecm/stage2/types.h` 等私有头，不作为外部 API。

## 父任务与曲线子进程

`stage2/cuda/main.cpp` 负责 CLI、配置合并、模式与退出。`stage2/task.cpp` 只有一条父任务循环：校验/恢复 → pending 落盘 → 子进程 → receipt 确认 → done → seal → 源行复核 → finished → 队列移除 → 进度清理。同文件的 curve_worker 保留重读 save/hash、规划、GCD/引擎和持久发布的完整顺序。

`src/ecm/stage2/process.cpp` 管理 Job、管道、进程、停止和 console 投影；Worker/Task 不重复管理这些资源。`src/ecm/stage2/input.cpp` 集中实际输入校验；普通配置 CPP 在发布处读取结果身份。父流程只接受具体路径/队列策略/默认值，无完整 Settings 依赖。私有 Options 仍是 executable 内部调用数据，不是公开计算 API；一个进程只有一个活跃曲线/引擎状态。

## 曲线进程协议

[curve_process.h](../../include/ecm/stage2/curve_process.h)只声明进程请求和同步执行入口。实际协议包括 49 项曲线/规划/发布参数，以及记录位置和 hash、save/日志/结果路径和显示预算；不包含完整 CLI Options 或配置解析类型。任务/校准共用一处协议投影；process 实现管理唯一进程循环和 OS 资源。

输出为退出码（0 成功、3 为不纳入计时的已验证因子、1 失败）、原控制台/日志和可选同步 selection 行回调。回调位于原输出时点、日志写入之前，由任务层验证并接纳 hints；异常按原时点传播。Windows Handle 仅在私有声明中。进程实现可脱离计算引擎、GMP 和 CUDA 独立编译/测试，真实计算仍在独立子进程中完成。
