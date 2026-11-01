# 多项式 ECM Stage2 的位宽复杂度

本文区分数学工作量、固定字长实现及GPU墙钟。公式是按当前算法结构作出的推导，不是对所有ECM实现的统一渐近结论。当前实现、实测拟合及优化顺序见[位宽性能分析](../performance/STAGE2_BIT_SCALING.md)；D/P选择的独立数学依据见[D参数](ECM_STAGE2_D_PARAMETERS.md)。

同一份[位宽报告](../performance/STAGE2_BIT_SCALING.md)集中提供完整符号表、各阶段精确请求计数／内存和传输公式、CGBN与单点短NTT的适用性，以及阶段合并后的当前成本账本；本文保留其通用推导依据。经验指数只在报告标明的输入、设备与计时窗口内成立。

## 1. 约定与复杂度单位

| 符号 | 含义 |
|---|---|
| \(\mathcal N,t\) | 目标模数及其实际位数，\(t=\lfloor\log_2\mathcal N\rfloor+1\) |
| \(s,W\) | 实际算术位数及64-bit limb数，\(W=\lceil s/64\rceil\) |
| \(D,P\) | giant步长及baby数量，\(P=\varphi(D)/2\) |
| \(I,G\) | giant数量及G树数量，当前完整区间为 \(I=\lfloor B2/D\rfloor+2\)、\(G=\lceil I/P\rceil\) |
| \(c,C\) | 每条giant链的点数及一个物理点块的容量；二者不是CUDA块的线程数 |
| \(m,b,L\) | 一次多项式乘法的最大输入系数数、每digit位数及Goldilocks变换长度 |

普通路径\(s=t\)；梅森承载\(s=p\)，实际在\(2^p-1\)环中计算，同时保留对目标\(\mathcal N\)的投影、非单位及因子验证。下述复杂度按64-bit字运算计数，64为固定单位。因此\(O(W^2)\)也可写为关于位数的\(O(s^2)\)，若以整数数值\(\mathcal N\)作变量则是\(O((\log\mathcal N)^2)\)，不能写成关于模数数值的\(O(\mathcal N^2)\)。

当前GPU范围是2…16384 bits；固定64-bit NTT素数和最大长度也有有限范围。公式用于解释此范围内的工作量及可推广的算法结构，不宣称这一固定硬件实现可以处理无限增长的s。

## 2. 点算术：梅森归约不消除完整乘积

一次schoolbook完整乘积有\(W^2\)个64×64位部分积。普通Montgomery模乘随后又进行约\(W^2\)个消元MAC；精确梅森模数可将这一步替换为折叠和旋转：

$$
\begin{aligned}
M_{ordinary}(W)&=2W^2+O(W),\\
M_{mersenne}(W)&=W^2+O(W).
\end{aligned}
$$

这是主导部分积／MAC的计数，不是GPU周期公式；加减、进位、地址计算、私有数组访问和同步另计。两条路径都仍为\(\Theta(W^2)\)。当前实现入口为[point.cuh](../../src/ecm/stage2/cuda/device/point.cuh)140行的`s2g_mont_mul`及[point_mersenne.cuh](../../src/ecm/stage2/cuda/device/point_mersenne.cuh)6行的`s2g_mersenne_mont_reduce`。schoolbook的乘法复杂度也可对照[GMP基例乘法](https://gmplib.org/manual/Basecase-Multiplication)。

当前六模乘xADD包含4次一般乘法、2次平方；xdbl包含3次一般乘法、2次平方。一个普通ladder位步合计7M+4S。关闭专用平方时S调用一般模乘；启用`NTT_POINT_SQUARE=1`时，`point_square.cuh::s2g_mont_sqr`只计算严格上三角、统一倍增后加入对角线，完整乘积部分积数为W(W+1)/2，原归约保持不变：

$$
S_{ordinary}(W)=\tfrac32W^2+O(W),\qquad
S_{mersenne}(W)=\tfrac12W^2+O(W).
$$

这些仍为\(\Theta(W^2)\)的MAC工作，不是周期数或完整Stage2的加速比。加减及保持齐次尺度的halfmod为线性工作。启用范围、默认状态与实测条件由[位宽报告](../performance/STAGE2_BIT_SCALING.md)说明。

其精确性来自二进制基\(B=2^{64}\)下的恒等式

$$
a^2=2\sum_{0\le i<j<W}a_i a_jB^{i+j}
+\sum_{i=0}^{W-1}a_i^2B^{2i}.
$$

严格上三角先形成完整多字乘积，再统一左移一位；不能直接把每个128-bit交叉乘积左移而丢掉第129位。对角线每项占相邻两个limb，按序携带加法进位；乘积完成后才覆盖输出，因而允许输出与输入别名。普通／梅森两条路径共享与一般模乘相同的归约入口。

## 3. Baby、有限giant链与短尾

baby索引集合为\(J=\{1\le j\le D/2:\gcd(j,D)=1\}\)。每个索引目前独立ladder，因此

$$
T_{baby,points}=O\!\left(P\log D\;M_{point}(W)\right).
$$

对完整giant区间，令每个物理点块起点为\(i_a\)、长度为\(n_a\)。每条链先计算相邻seed，再连续差分xADD；保留实际首点索引时，两个主核中的模乘计数为

$$
Q_{seed}=\sum_a\sum_{k=0}^{\lceil n_a/c\rceil-1}
\left(5+11\lfloor\log_2(i_a+kc)\rfloor\right),
$$

$$
Q_{chain}=6\sum_a\sum_{k=0}^{\lceil n_a/c\rceil-1}
\max\bigl(0,\min(c,n_a-kc)-2\bigr).
$$

它们不含[D]Q基点、坐标转换、归一化、段积、自检或其他Stage2阶段。由此得到工作量尺度

$$
T_{giant,points}=O\!\left(
I\left(1+\frac{\log I}{c}\right)M_{point}(W)
\right).
$$

若有\(r_L\)个点走独立ladder，还需加\(O(r_L\log B2\;M_{point}(W))\)。不能把有限c下的全部seed成本删除后，称giant生成严格为线性点数成本；缩短链可能增加算术量，却改善实际并行度。具体语义和正确性边界见[点策略](../architecture/STAGE2.md#完整区间点块策略)。

## 4. 多项式乘法：NTT与系数归约是两项成本

当前Kronecker packing对最大输入m选择

$$
h=2s+\max(1,\lceil\log_2m\rceil),\quad
u=\lceil h/b\rceil,\quad
L=2^{\lceil\log_2(2mu+1)\rceil}.
$$

选定b还须满足精确性界

$$
mu(2^b-1)^2<q,\qquad q=2^{64}-2^{32}+1.
$$

来源为[shape.cpp](../../src/ecm/cuda/ntt/shape.cpp)97行的`choose_cfg`、283行的`ntt_shape_query`。b由实际形状选择，不能在全范围默认为31。给定b后，L相对\(m(s+\log m)/b\)只多幂次取整常数，但实际取整会形成台阶。

一次卷积需要两个正变换和一个逆变换。其NTT工作量为\(O(L\log L)\)，不是\(O(L^2)\)。解包后的系数仍须对ECM模数归约：设系数暂存有U个64-bit limbs，通常\(U=2W+O(\log m/64+1)\)，则

$$
R_{ordinary}=O(W(U-W+1))=O(W^2+W\log m/64),
$$

而在当前输入范围、折叠次数有界时

$$
R_{mersenne}=O(U+W)=O(W+\log m/64).
$$

来源为[stage2_scalar.cuh](../../src/ecm/stage2/cuda/stage2_scalar.cuh)的`s4_div_rem`、`s4_mersenne_rem`及[s4_reduce.cuh](../../src/ecm/stage2/cuda/device/s4_reduce.cuh)。长除法的商位数×除数长度成本可对照[GMP基例除法](https://gmplib.org/manual/Basecase-Division)。

一次多项式乘法的实用参数化上界因此是

$$
C_{poly}(m,s)=O\bigl(L\log L+mR(W)+mW+L\bigr).
$$

第一项是NTT，第二项是系数归约，其余包含打包／carry／输出；不能把整个多项式阶段都叫NTT时间，也不能因NTT准线性就忽略普通系数归约的平方项。

## 5. 完整产品树和下降

对规则二叉产品树，逐层工作可写成

$$
\sum_{a=1}^{\lceil\log_2P\rceil}
O\!\left(\frac{P}{2^a}C_{poly}(2^{a-1}+1,s)\right).
$$

不规则尾树、常数1空节点和截断下降应按真实非空形状累加。常用上界为\(O(C_{poly}(P,s)\log P)\)，可参照[Bernstein的乘法应用综述](https://cr.yp.to/lineartime/multapps-20080503.pdf)；scaled下降的代数合同见[Stage2数学](ECM_STAGE2_MATH.md)。

F树和下降各一遍，G树合计约I/P棵，后续fold约G−1次。将b在一个局部范围视为稳定，可得

$$
T_{poly,all}=O\!\left((P+I)\left[
\frac{s}{b}\log(Ps/b)\log P+R(W)\log P
\right]\right).
$$

这是保守工作上界，fold单独没有完整树的额外层数因子。固定D/P/B2且不改变执行分支时：点算术和普通系数归约的主导项近似平方增长，NTT接近\(s\log s\)但有幂次取整。普通完整流程不能一概称为准线性；承载完整流程也不能一概称为线性，因为点乘仍然平方。

归一化、GCD和其他CPU工作另计为\(T_{host}(P,I,s,t)\)。GMP会随大小选择不同乘法与求逆算法，不能无条件给GCD指定\(O(W^3)\)；参见[GMP算法分派](https://gmplib.org/manual/Multiplication-Algorithms)。

## 6. B2、位宽与固定显存的联合影响

忽略缓慢变化的log项，若D≈rP且没有内存限制，baby与giant可近似平衡为

$$
AP+B\frac{B2}{rP},\qquad
P_{opt}\approx\sqrt{\frac{B\,B2}{A r}}.
$$

这仅给出局部平方根尺度，A/B依赖位宽、算术路径和设备效率。若某个主导驻留预算H限制\(P\le H/(\kappa W)\)，并迫使D≈rP随W反比缩小，则

$$
I\approx\frac{\kappa B2\,W}{rH},\qquad
I M_{point}(W)=O\!\left(\frac{B2\,W^3}{H}\right)
$$

（省略链seed和log因子）。因此**有限显存下的选型变化可以额外引入一个W因子**，即使单个模乘仍是平方。此结论有明确前提，不表示所有高位宽任务都呈立方增长；当前小B2矩阵的高位P近似不变，并不满足这一特定缩P解释。

同理，B2增大而P被固定预算封顶时，I随B2近似线性；不存在全B2范围都适用的\(T\propto\sqrt{B2}\)保证。

## 7. O()与GPU耗时之间仍隔着执行效率

实际耗时应保留如下分解：

$$
T_{wall}\approx\frac{Q_{MAC}}{F_{MAC,eff}}+
\sum_L\frac{Q_{NTT}(L)}{F_{NTT,eff}(L,batch)}+
T_{local/cache}+T_{transfer}+T_{launch/wait}+T_{CPU}.
$$

分母取决于真实并行度和依赖链，不能由峰值算力代替。每线程私有栈、模板容量、chain数／SM、NTT长度／批量台阶及驻留回退都可能使局部拟合指数大于2。单次大于2的经验指数不证明数学算法变为立方或NTT变为平方。

上述加法是工作模型：若\(F_{MAC,eff}\)或\(F_{NTT,eff}\)直接采用包含local、带宽和同步等待的实测kernel吞吐量，相应延迟已在分母中，不能再作为独立时间加一遍。互斥阶段和嵌套设备事件的计量边界必须分别保持。

专用平方降低schoolbook的常数；单层Karatsuba也只是降低二次项常数。只有递归Karatsuba才有\(O(W^{\log_2 3})\)的乘法上界，普通Montgomery归约若仍保持平方，整体模乘仍不会因此变为次平方。相关递推及常数取舍见[GMP Karatsuba说明](https://gmplib.org/manual/Karatsuba-Multiplication)。仓库提供可选schoolbook专用平方，尚未实现Karatsuba点模乘。
