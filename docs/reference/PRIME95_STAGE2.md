# Prime95 Poly Stage2：F/G/H、scaled 下降与访存

固定参考资料。来源：`STAGE2_GPU_COMPARE_PRMERS.md §§27.1–27.2`。保留数学推导、第三方源码分析及其条件；源码行号针对原文研究的本地副本，第三方上游可能已有变化。本文不声明本仓库当前支持、默认值或性能。涉及实验数字时只按原文条件理解；当前实现见 [架构](../architecture/OVERVIEW.md)，当前计时口径见 [性能](../performance/STAGE2.md)。

## 1. Poly Stage 2 在算什么

令 Stage 1 输出点为 Q，选取偶数 D。对 D 的相对素数 residue 生成 baby 点 x_j=x([j]Q)，构造 F(X)=∏(X−x_j)。对每个 giant block，生成点 Y_i=[m_iD]Q，构造 G_b(X)=∏(X−x(Y_i))。然后将每块 giant 信息折叠进 H(X)←G_b(X)H(X) mod F(X)。沿 F 的余式树下降可得到每个 baby 根上的 H(x_j)；它等价于该 baby 点与已处理 giant 点差项的批量乘积（符号/可逆尺度因子按实现约定处理）。最终把叶值合并并对模数 N 求 GCD，以发现因子。

这把 BSGS 的逐项 X_g Z_b−Z_g X_b 命中判定改写成“多项式积—模 F 折叠—多点求值”。D 仍决定 baby 集大小与 giant block 数的折中；算法没有消除所有点运算，而是把大量 prime pair 的交叉项变成平衡树上的批量多项式运算。

## 2. Prime95 源码路径（可直接按行复查）

1. **D、P、B2 与内存成本选择。**numrels 是小于 D/2 的相对素数个数；Poly 实现以它作为多项式长度。每个 giant block 的 section 数向 poly_size 的倍数补齐，保持批形状规则；成本模型估算 F/R/G/H、Ftree 与 polymult scratch 的内存，并加入超出 L2 后的惩罚。代码还搜索更合适的 B2：Poly 下经济区间可以远大于传统 pairing。见 [ecm.cpp:429–436, 731–739, 5756–5810, 6011–6055](../../.refactor/p95v3106b01.source/ecm.cpp)。
2. **F 产品树。**从 nQx 的相对素数点建立线性 monic 因子，先用专用 helper 合并小因子，再逐层两两 polymult。FFT 计划按相近形状保存/重用；每层后批量 unFFT/FFT 系数。见 [ecm.cpp:9095–9212](../../.refactor/p95v3106b01.source/ecm.cpp)。
3. **F 的倒数多项式与预处理。**用 Newton 倍增迭代求 reciprocal 1/F，然后对 F/R 预转置、压缩，以减少内存占用。见 [ecm.cpp:9232–9295](../../.refactor/p95v3106b01.source/ecm.cpp)。
4. **G 产品树。**每个 outer loop 通过 mQ_next_array 生成一块巨点，构造 G 的平衡乘积树。polyG/polyH 共用连续分配，polyGaux 只保留一部分临时结果，注释明确这是用少量辅助空间省掉整块复制。见 [ecm.cpp:9320–9327, 9354–9435](../../.refactor/p95v3106b01.source/ecm.cpp)。
5. **折叠 H←GH mod F。**首个 G block 初始化 H；之后用三段运算：先乘 G·H，乘 1/F 取高位得到商，再用 FMA 减去商乘 F 并只留低位余数。见 [ecm.cpp:9458–9474](../../.refactor/p95v3106b01.source/ecm.cpp)。
6. **Bernstein scaled remainder descent。**先把 H 乘 1/F 形成 scaled remainder，再逐层把父余数对左右子多项式取余，最终落到线性因子/叶值。为了控制峰值内存，Prime95 释放 F/R、切片处理 H，并按可用内存从内存、磁盘保存或重建 Ftree 行。每个父 H 同时对左右两个子树求余时，polymult_several 让两个 child 运算共享共同操作数的变换。见 [ecm.cpp:9538–9578, 9627–9737](../../.refactor/p95v3106b01.source/ecm.cpp)。
7. **合并叶值与 GCD。**helper 线程生成部分积，合并后进入 ECM GCD；在此之前释放大块多项式工作集。见 [ecm.cpp:9760–9880](../../.refactor/p95v3106b01.source/ecm.cpp)。

gwnum/polymult 是 Prime95 的浮点 FFT/实数大整数表示路径，调用方检查 roundoff 和 safety margin；它不是本仓库的精确整数 NTT。Poly 选项包括 monic 输入、只取乘积高/低系数、FMA、保留计划及复用计划；polymult_several 允许一份输入乘多个相关多项式。见 [polymult.h:128–157](../../.refactor/p95v3106b01.source/gwnum/polymult.h)、[polymult.c:4903–5050](../../.refactor/p95v3106b01.source/gwnum/polymult.c)。FFT/Karatsuba 阈值处还留有 “Fix me” 注释，见 [polymult.c:531–539](../../.refactor/p95v3106b01.source/gwnum/polymult.c)。
