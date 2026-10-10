# CUDA 多项式 Stage2 管线

## 输入与表示

输入为完成的 Suyama PARAM0 Stage1 点 Q、目标 N、B1/B2、设备及预算。普通模式在 N 上计算；梅森承载实验在 M=2ᵖ−1 上计算，再在 N 上解释单位性、投影、GCD 和因子。S=bit_length(M)，W=⌈S/64⌉。

### Stage1 save 与承载资格

save只提供目标N、sigma、B1和归一化X，不自动推断承载指数。支持PARAM=0、Z=1或省略Z，X须在[0,N)；存在checksum时必须通过校验。承载要求N为大于3的奇数、bit_length(N)≤p≤16384，且实际验证N∣2ᵖ−1；显式`--carrier-exponent p`不满足条件时报错。p不要求为素数，也没有按目标位宽自动启用的固定阈值。

提供完整tune时，可从匹配的普通/合法承载候选按成本与联合显存选择，规则见[实测选型](AUTO_B2.md#实测-d-与承载选择)。显式`--carrier-exponent 0`锁定普通模数；未提供tune或显式p时，不因save来自梅森数自动抬升模数。N本身为完整梅森数时，默认归约路径仍可使用梅森快速归约。

承载路径直接接入save的仿射X及Z=1，不重跑Stage1、不再次应用可选的12倍标量。逆元、单位性与最终GCD均针对目标N；承载增加的是Stage2算术位宽，合法性并不保证性能更优。依据：[save解析](../../src/core/ecm_cuda_stage2_main.cpp#L123)、[承载校验](../../src/core/ecm_stage2_modulus.h#L28)、[坐标接入](../../src/cuda/ecm_cuda_stage2.cu#L8970)。

设备点坐标使用 Montgomery 域，NTT digit 在 Goldilocks 域，多项式系数归约后为普通规范整数。不同表示不能直接相乘或判断数学相等。

D 确定 baby 集 J={1≤j≤D/2:gcd(j,D)=1}，P=φ(D)/2。giant 数 I=⌊B2/D⌋+2，分为 G=⌈I/P⌉ 个 G 树批次。补充小素数路径覆盖主树映射不能直接处理的边界。

## 执行步骤

| 步骤 | 工作与结果 | 主要执行位置 |
| --- | --- | --- |
| 1. 读档与规划 | 验证 Q/N/B1/sigma；选择 D，查询 packing 与组件预算 | CPU、设备查询 |
| 2. baby 点 | 求 [j]Q，得到 X/Z；处理不可逆坐标 | GPU 点算术、CPU GCD/逆 |
| 3. baby 归一化 | 设备积树、主机根逆、设备逆传播，生成普通 xⱼ | GPU 为主 |
| 4. F 积树 | 叶 X−xⱼ；逐层相乘，保留 F 与下降所需子树 | GPU NTT/S4，主机编排 |
| 5. Newton 逆 | 预计算反转 F 的截断逆 finv，供多次模 F 归约与根准备复用 | GPU 多项式乘法 |
| 6. giant 点块 | 求 [iD]Q；ladder 或相邻 seed+差分 chain；按坐标预算分块 | GPU 为主 |
| 7. giant 归一化/分组 | 精确 16 点段积、单位性分类、分组逆；形成仿射或齐次叶 | GPU、CPU 目标 N 逆/GCD |
| 8. G 积树 | 将本批 giant 叶构成 G，多项式根直接交给 fold | GPU NTT/S4 |
| 9. fold | 累计 H←G·H mod F；使用 finv 与长/短乘法，复用驻留临时槽 | GPU；准入失败有主机路径 |
| 10. 缩放与根准备 | 消除齐次缩放 Γ；驻留 H/finv 形成 scaled 下降根 | GPU；回退时 CPU |
| 11. F 树下降 | scaled remainder/middle-product 传播，最终得到 P 个叶值 | GPU NTT/S4、主机元数据 |
| 12. 累积与 GCD | 设备每 64 叶求块积；CPU GCD；饱和块细分 | GPU 产品、CPU GMP |
| 13. 验证与结果 | 排空算术检查，验证因子；可选命名/GP 分解；写入成功回执 | CPU 与检查收尾 |

Giant、G 树和 fold 交织运行；阶段墙钟不能视为独立并行任务。每曲线 init 包括 baby/F 树和必需自检，main 包括 inverse、批循环、下降、GCD 和检查收尾。完整时间为 init+main，Stage1 和自动 D 扫描另计。

## 点生成与归一化

差分 xADD 使用 6 次模乘，固定差点保证 chain 中的相邻点关系。相邻 seed 可同时得到 [iD]Q 与 [(i+1)D]Q，避免重复计算两个独立 ladder。短块、容量不足或非单位场景仍需保留正确回退。

分块求逆以积换取少量主机逆：可逆集合使用前缀/积树传播，遇到非单位先在目标 N 上判定和求 GCD。不能在承载 M 上的单位性替代 N 上的单位性。Gamma 保存使用齐次叶引入的累计缩放，在最终下降前校正。

## 多项式与 scaled 下降

F(X)=∏ⱼ(X−xⱼ)。giant 叶为齐次形式 ZᵢX−Xᵢ，或等价的已归一化叶；累计 H 与相关缩放共同保持数学结果。G 树非满尾批按真实次数构建，不能用填满后的次数估计所有 NTT。

多 G 批次复用 F 和反转逆，不逐批重建。scaled 下降采用反转/截断/middle product 的等价表达，将原余式链传播到各子树；非空子节点使用实际 operand 长度，空节点复制或补零。归一化后的叶投影与直接 Horner/CPU 参考一致是正确性门禁。

## 工作量

设 Mpoly(a,b,S) 表示一次系数模 M 的多项式乘法成本。完整 F 树有 P−1 次非空逻辑乘法，G 树总计 I−G 次；这不是 kernel launch 数，多个同形乘法可合并成 batch。

每个树层 h=1,2,4…，满配对数 q=⌊P/(2h)⌋，余数 r=P mod 2h；有 q 个 (h+1,h+1) operand 乘法，若 r>h 再有一个 (r−h+1,h+1) 尾乘法。0<r≤h 时复制，不执行 NTT。真实工作应累加每个形状的 Mpoly，而非只用 P log P。

一般近似：F/下降为 O(Mpoly(P) log P)，所有 G 树为 O(G·Mpoly(P) log P)，fold 为 O((G−1)·Mpoly(P))；giant 为 O(I·W²) 的点算术主项，加 seed 与求逆。每次 NTT 的域运算为 O(L log L)，设备系数归约另计。

## 数据与检查

驻留 G-root、fold、scaled root/frontier 减少主机往返，但 F 子树上传、元数据、最终 P·W 字叶读回及 CPU GCD 仍存在。输入源分别为 host、tree_raw、fold_owner、frontier_owner，借用存储不新计 owner。

必需自检、carry 精确性、系数 GMP 抽样、proper divisor 验证和检查排空属于计算合同；factor-only 只省可选命名。性能测量与额外逐节点/台账诊断分开，不能把诊断曲线当作普通速度样本。

## 当前规划边界

`request_program`预测满次数驻留fold的乘法顺序、形状、来源和树租约。初始化/NTT/S4/owner/giant按同一时间线预测条件正常路径的峰与free需求；匹配完整Stage2 tune的数据后，驱动据此选择已测D和合法承载。单G批次、诊断和全部分配失败回退不在其统一保证中；引擎保留实时检查和回退，详见[内存](MEMORY.md)与[实测选型](AUTO_B2.md)。

## 代码依据

- [生产引擎](../../src/cuda/ecm_cuda_stage2.cu)：baby/F、[`run_batched`](../../src/cuda/ecm_cuda_stage2.cu#L7267)、scaled下降、GCD；[完整计时输出](../../src/cuda/ecm_cuda_stage2.cu#L9413)。
- [模数上下文](../../src/core/ecm_stage2_modulus.h#L28)：`configure`，目标/承载资格。
- [请求程序](../../src/core/ecm_stage2_requests.h#L76)：`request_program`；[精确树组](../../src/core/ecm_stage2_geometry.h#L93)：`tree_multiply_groups`。
- [点模乘](../../src/cuda/stage2/stage2_point_mersenne.cuh)、[NTT](NTT.md)、[内存](MEMORY.md)。
- 固定资料：[多项式数学](../reference/ECM_STAGE2_MATH.md)、[Prime95 Poly](../reference/PRIME95_STAGE2.md)、[PrMers ECM](../reference/PRMERS_ECM.md)。
