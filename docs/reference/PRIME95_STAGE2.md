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

## 3. D 候选表的含义与适配条件

本节核对的副本为 `.refactor/p95v3106b01.source/ecm.cpp`，SHA256=`7e6be864ff0c76420150d36f5a72f182cdfaa21ad07f93f98132c3ae0f133964`。表与源码版本一同解释，不宣称适用于所有Prime95版本。

### 3.1 有效表与 numrels

- `D_data` 小表结构始于 `:433`，含34个pairing候选；其范围与poly表不同。
- `poly_D_data` 在 `:479` 的声明位于 `ALTERNATE_D_DATA_FOR_POLYS_BUILT_FROM_TABLE_BELOW` 分支，含108项；该分支结束于 `:589`。
- 当前有效poly声明在 `:591`～`:680`，88项，D从30至164894730，numrels从4至14100480。逐项核对均有 **numrels=φ(D)/2**。

文件中三处声明合计230条初始化记录，但不能把pairing、条件分支和有效poly表合并成同一次poly搜索的230个候选。当前布局的条件声明也不是额外追加集合；解释其他版本必须重新核对编译分支。当前有效88项全部满足D=2·奇数，即恰有一个2因子；表不是对所有偶数D的均匀枚举。

对偶数D≥6，单位剩余类j与D−j成对，且没有自配的单位类，因此1≤j≤D/2的相对素数个数为φ(D)/2。这个P决定baby数量和多项式次数；D决定巨步间隔。P不是独立于D的自由参数。

欧拉函数给出 **φ(D)=D·∏(1−1/ℓ)**，乘积取D的不同素因子。小素数轮提高D/P=2D/φ(D)；在相同P下，较大的D可减少巨步数量，但生成点的标量、补齐小素数和准备范围还可能变化。

Prime95有效表同时包含素数轮与重复素数因子，例如90090=30030·3、210210=30030·7、810810=30030·27；它不是全部smooth整数的枚举。若D为偶数，φ(2D)=2φ(D)，因此将510510倍增为1021020时，P由46080变92160。这样的2幂扩展合法，是否适合某个执行器由成本和内存决定。

P也不唯一确定D：1111110和1141140均有P103680。不能仅按P去重就认定另一项无用，也不能只按D的距离判定相邻性能形状。

### 3.2 first_missing_prime 与成本搜索

`D_data`结构中的 `first_missing_prime` 是不整除D的第一个素数，`second_missing_prime` 是第二个，见 `:436`。`:797` 选择pairing或poly表；`:813` 遍历表内候选并调用相应成本函数，`:824` 检查numrels资源限制。

Prime95将较小素数移到更大的复合倍数，以减少实际D段数。它根据 `first_missing_prime` 计算B2_start，按B1/C_start等下界裁剪并向D网格对齐，见 `:826`～`:854`；段数由 **(B2_end−B2_start)/D** 决定。续跑时还要避免新D消除已移动素数所需的缺失因子，见 `:898`～`:903`。

所以该表的第一/第二缺失素数、CPU FFT成本、缓存和内存策略属于Prime95自己的运行模型。移植候选几何不等于移植其排名：若另一实现仍从原点生成 ⌊B2/D⌋+2 个giant点，就不能直接计入Prime95的B2_start缩短收益；不能仅因缺失素数更大而认为该D在另一实现更快。

### 3.3 与变换拐点相结合的数学理由

固定B2时，设D≈αP。在一个变换效率近似连续的窄区间，将额外树层和算术位宽成本吸收到近似常数a、b中，初始化随P增长，主循环随巨步数量缩小，可用如下尺度模型：

**T(P)≈[aP+bB2/(αP)]·log P，a,b>0。**

忽略log项的缓慢变化，平衡点约为 **P≈√(bB2/(aα))**。这是局部启发式，不是产品树的准确渐近复杂度；标准产品树通常为O(M(P)log P)，即准线性乘法下还含额外树层的log项。详细工作量和D/P非单调比较见 [D参数数学](ECM_STAGE2_D_PARAMETERS.md#3-d-与-p-怎样共同影响计算量)。尺度模型不能证明某张稀疏表全局最优。

若变换长度由nextpow2确定，P增长会使NTT/FFT突然翻倍；填充树在P越过2的幂时也改变层数，内存不足则可能进入传输更多的路径。各区间内部即使近似平滑，全范围仍可能有多个局部极小值。有效搜索需要比较多个执行区间的代表，而不是假设沿数值D连续下降就能到达全局最佳。

Prime95的88项已配合自身缺失素数移位、CPU FFT、补齐段数、缓存和内存模型使用；源码没有由此给出另一GPU执行器只需88项或200项即可覆盖其最优参数的证明。它适合作为候选几何先验，移植后的覆盖仍须按另一执行器评价。

具体packing、边界计算、当前源码与实测条件属于各执行器的工程状态，见本项目的 [NTT](../architecture/NTT.md)、[Auto B2](../architecture/AUTO_B2.md)。本项目的共享表来源、项数与限制记录在当前实现主题；不把这些工程选择写成Prime95或多项式Stage2的数学要求。
