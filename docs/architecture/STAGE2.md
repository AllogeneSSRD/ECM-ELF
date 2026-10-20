# CUDA 多项式 Stage2 管线

## 输入与表示

输入为完成的 Suyama PARAM0 或 Atkin–Morain Edwards Stage1 点 Q、目标 N、B1/B2、设备及预算。普通模式在 N 上计算；梅森承载实验在 M=2ᵖ−1 上计算，再在 N 上解释单位性、投影、GCD 和因子。S=bit_length(M)，W=⌈S/64⌉。

### Stage1 save 与承载资格

save提供目标N、sigma、B1和归一化X，不自动推断承载指数。Suyama记录支持PARAM=0、Z=1或省略Z，X须在[0,N)；存在checksum时必须通过校验。Edwards记录要求方法、版本、48LCM标量约定、状态及校验齐全，禁止PARAM字段；格式统一见 [Stage1 Edwards](STAGE1.md#gpu-edwards)。Edwards在目标N上重构曲线，使用 `(A+2)/4=1/(1-d)` 作为统一 baby/giant 点生成常数；Prime95 的 Ad4 是其倒数。该路径直接复用 NTT 多项式计算，不重跑 Stage1。Edwards当前要求显式B2，实验Stage2引擎不消费这种存档。

承载要求N为大于3的奇数、bit_length(N)≤p≤16384，且实际验证N∣2ᵖ−1；显式`--carrier-exponent p`不满足条件时报错。p不要求为素数，也没有按目标位宽自动启用的固定阈值。

完整tune可按成本与联合显存比较普通/合法承载，允许配置跨条件复用及宽度/B2估算，详见[Auto B2/tune](AUTO_B2.md)。合法梅森worktodo结合已知因子和save验证后提供承载指数；仅save不猜指数。显式`--carrier-exponent 0`锁定普通模数。N本身为完整梅森数时，默认归约路径仍可使用梅森快速归约。

承载路径直接接入save的仿射X及Z=1，不重跑Stage1、不再次应用可选的12倍标量。逆元、单位性与最终GCD均针对目标N；承载增加的是Stage2算术位宽，合法性并不保证性能更优。依据：[save解析](../../src/core/ecm_cuda_stage2_main.cpp#L123)、[承载校验](../../src/core/ecm_stage2_modulus.h#L28)、[坐标接入](../../src/cuda/ecm_cuda_stage2.cu#L9015)。

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

seed 和 chain 的宽位数核在 W>128 时使用 32 线程 CUDA 块，其他宽度仍为 64 线程；每条顺序 chain 仍生成 64 点。此处 CUDA 线程块大小与 chain 点数是两个独立参数。点坐标、Montgomery radix、seed 标量和回退条件保持同一合同。chain 核位于 `kernels/cuda/stage2_giant_chain.cuh`，正式构建记录其依赖与哈希。

生产路径的通用 chain 下限为每个点块 32768 点；当算术位宽 W>128 个 64-bit limb、驻留 G 叶可用且其坐标预算至少覆盖 32767 点时，有效下限为 8192。这里 W 取实际算术模数，承载模式按承载位宽计算。更短点块或不足的驻留预算仍使用通用路线；`NTT_GIANT_CHAIN_MIN` 的生产配置约束保持不变。共享选择函数为 [giant_chain_minimum](../../src/core/ecm_stage2_geometry.h)，原生内存规划与 portable tune 使用相同的有效阈值。

新接入的宽尾块先验证 [D]Q 的单位性。非单位基点直接使用逐点 ladder；若差分链生成后的分组乘积在目标 N 上非单位，则恢复此前的逻辑 Gamma，并在原坐标和分段缓冲区内用逐点 ladder 重做整个尾块。该回退保留原有因子集合及非仿射占位叶，避免将表示差异引入结果。已有大点块的路线保持原合同。缓冲区预留公式见 [Giant 组件](MEMORY.md#giant-组件)。

`device_gleaf.chunks` 记录实际准备次数，`retry_chunks` 记录重做次数；分组、检查和传输计数包含重做工作，`coord_peak_bytes`/`group_peak_bytes` 仍为各自同时占用峰。数学上的唯一覆盖由 `real_batched_projective` 的 `leaves=gamma_points` 校验，不能从准备次数推导曲线数。

分块求逆以积换取少量主机逆：可逆集合使用前缀/积树传播，遇到非单位先在目标 N 上判定和求 GCD。不能在承载 M 上的单位性替代 N 上的单位性。Gamma 保存使用齐次叶引入的累计缩放，在最终下降前校正。

## 多项式与 scaled 下降

F(X)=∏ⱼ(X−xⱼ)。giant 叶为齐次形式 ZᵢX−Xᵢ，或等价的已归一化叶；累计 H 与相关缩放共同保持数学结果。G 树非满尾批按真实次数构建，不能用填满后的次数估计所有 NTT。

当 W>128 且本次 giant 区间至少包含 4 个完整 G 批次时，驻留 G 根树使用分块空位布局：底部保留密集的二次幂叶块，将最多 128 个块分散在上层完整树的槽位中，减少非二次幂次数导致的高层输入不平衡。其他位宽及较短 Stage2 保留原来的左侧连续布局，避免为少量计算增加形状准备及复制开销。启用条件由 `giant_tree_balanced` 统一用于执行器、请求规划及调优工作量；debug日志 `groot_layout` 记录实际选择。

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

## 当前规划边界

`request_program`预测满次数驻留fold的乘法顺序、形状、来源和树租约。初始化/NTT/S4/owner/giant按同一时间线计算条件峰和free需求，host fold/frontier可使用受限arena的保守上界。驱动以tune实测或估算选择D及合法承载；单G、baby预算回退、诊断及全部分配失败不在统一保证中。引擎保留实时检查，详见[内存](MEMORY.md)与[Auto B2/tune](AUTO_B2.md)。

## 代码依据

- [生产引擎](../../src/cuda/ecm_cuda_stage2.cu)：baby/F、[`run_batched`](../../src/cuda/ecm_cuda_stage2.cu#L7265)、scaled下降、GCD；[完整计时输出](../../src/cuda/ecm_cuda_stage2.cu#L9511)。
- [模数上下文](../../src/core/ecm_stage2_modulus.h#L28)：`configure`，目标/承载资格。
- [请求程序](../../src/core/ecm_stage2_requests.h#L78)：`request_program`；[F树组](../../src/core/ecm_stage2_geometry.h#L147)：`tree_multiply_groups`；[G树布局/分组与选择](../../src/core/ecm_stage2_giant_tree.h)：`giant_tree_balanced`、`giant_tree_layout`、`giant_tree_multiply_groups`。
- [点模乘](../../src/cuda/stage2/stage2_point_mersenne.cuh)、[NTT](NTT.md)、[内存](MEMORY.md)。
- 固定资料：[多项式数学](../reference/ECM_STAGE2_MATH.md)、[Prime95 Poly](../reference/PRIME95_STAGE2.md)、[PrMers ECM](../reference/PRMERS_ECM.md)。
