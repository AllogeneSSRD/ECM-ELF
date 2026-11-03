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

同理，B2增大而P被固定预算封顶时，I随B2近似线性；不存在全B2范围都适用的\(T\propto\sqrt{B2}\)保证。封顶不只来自fold owner：NTT主数组及其他同时驻留的数据也限制最大可用长度，精确digit位数变化又会使长度跳变。因此“驻留fold启用”本身不足以保证平方根尺度；还须能在NTT与其他预算内继续增大P。

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

## 8. 协作容器与原Montgomery尺度

实际模数占W个64-bit limbs，而协作容器占\(W_c\ge W\)。记

$$
R_a=2^{64W},\qquad R_c=2^{64W_c},\qquad \delta=64(W_c-W).
$$

不能只把零填充后的输入交给另一radix的Montgomery乘法；\(abR_c^{-1}\)通常不等于所需的\(abR_a^{-1}\)，连续点公式还会累积表示尺度差异。

这里\(W_c\)须对应后端实际Montgomery radix。CGBN允许Env::BITS按32递增，不表示任意该BITS都直接决定Montgomery的R：本地`impl_mpz.cc::mont_mul`按TPI×LIMBS轮消元，CUDA的`core_mont_wmad.cu`也按完整协作组循环。应核对`UNPADDED_BITS`、padding处理与具体接口；当前协作容量均按整组对齐，避免将逻辑BITS与物理radix混为一谈。

对于普通奇数模数，可先求完整乘积并左移\(\delta\)位，再用容器的REDC：

$$
\operatorname{REDC}_{R_c}(ab\,2^\delta)
\equiv ab\,2^\delta R_c^{-1}
\equiv abR_a^{-1}\pmod N.
$$

当\(0\le a,b<N<R_a\)时，\(ab2^\delta<N R_c\)，仍满足宽REDC的输入界；完整分子也小于\(R_c^2\)。补齐只增加宽移位，不需要为每个点反复做输入／输出表示转换。容器本身仍可能增加乘法工作，必须按\(W_c\)而非W计。

对于\(M=2^p-1\)，若容器已有\(v=abR_c^{-1}\bmod M\)，则所需结果为

$$
v2^\delta\bmod M
=\operatorname{rotl}_{p}(v,\delta\bmod p).
$$

该旋转可与梅森折叠的旋转组织合并。这里的旋转是p-bit环上的操作，不能使用完整容器位宽的普通循环移位替代。

还须独立核对后端的规范余数合同。本地WMAD `core_mont_wmad.cu:175`起的Montgomery乘法尾部按radix进位条件减N，可能返回\([N,2N)\)内的同余表示；要求\([0,N)\)的调用方需再比较／条件减N。这不表示该库所有算术入口都使用相同范围。点加法中“一次减N即可”的界通常依赖输入均小于N，不能仅因最终取模同余就忽略规范化。

本地WMAD宽乘入口为`core_mul_wmad.cu:142`，仍是循环部分积；`impl_cuda.cu:526/1032`的平方复用一般乘法。协作改变分工与执行效率，不因此把总工作量变为Karatsuba复杂度。以上两份WMAD文件SHA256分别为`c4b975c54a7a3ae7e695dee9d93ffc4d9be71a2cad7d0fc9e41a7e844803d181`与`f11bd15a1d4d43fbb0bdcde30249cb2acd3cbf5ec409f6c496bc87827627e38a`。

## 9. 论文中的输出归属与寄存器分块

Oancea／Watt的[中等整数GPU加乘论文](https://arxiv.org/html/2405.14642v1)讨论线程块内部的分工：每线程处理若干固定字，独占输出卷积系数，以寄存器累加，再交换进位摘要。其固定长度低半乘积不等于ECM的完整模乘；下述适配推导保留完整高、低半积，不采用论文缺少一般整数恢复条件的单素域FFT参数。详细译述与反例见本机`docs/paper/GPU Implementations for Midsize Integer Addition and Multiplication_技术分析.md`。

令n为一个输入的digit数，完整卷积系数\(C_k=\sum_{i+j=k}a_i b_j\)的项数为

$$
w(k)=\begin{cases}k+1,&0\le k<n,\\2n-1-k,&n\le k<2n-1,\\0,&k=2n-1.\end{cases}
$$

因此可把\(C_k,C_{k+n}\)交给同一执行组，\(0\le k<n\)时总有\(w(k)+w(k+n)=n\)。这是完整乘积的负载均衡配对；直接镜像配对\(C_k,C_{2n-2-k}\)则会把相同项数的两端放在一起，不能均衡工作。平方还须按上三角／对角线权重重新组织，而不是把一般乘法的配对原样照搬。

若输入digit为b bits，含T项的一个未进位系数最多需要约\(2b+\lceil\log_2T\rceil\)bits。64×64产品本身就有128bits，累加多个产品可能需要更多位；不能把一个128-bit累加器当成任意T的无溢出保证。进位合并也须隔离独立整数的段首，不能让前一整数的最高位进入后一整数的最低位。

在固定\(W_c\)个64-bit词的CGBN容器中，使用u个协作线程，每线程输入字数为

$$
q_c=\frac{2W_c}{u}\quad\text{个32-bit字},\qquad
Q_{product}=\Theta((2W_c)^2),\qquad
Q_{thread}\text{约为}\Theta((2W_c)^2/u).
$$

适用条件是容器按线程组整分；\(q_c\)还不包含坐标、宽乘积、累加器、函数调用与进位临时量。总工作量不因u增大变成次平方，单线程分摊量也不等于实际时间。容器零填充的乘积成本相对实际W可放大约\((W_c/W)^2\)；缩小容器和改变u是两个独立变量，须用固定容器的u对照、固定u的容器对照分离。

该思路可用于点链或系数归约，但完整程序还受独立实例数、寄存器驻留、local事务、模归约及同步影响。若只加速giant链，其Amdahl比例应取giant链实际占比，不能使用包括baby、seed、求逆和叶准备在内的整个点组占比。论文采用A100，库分工及测量对象不同，不能直接给出其他GPU的切换阈值或本项目的实测收益。

## 10. 齐次点公式中的表示因子可延后合并

融合中间运算还可以研究Montgomery表示修正。设模数为奇数、\(R=2^{64W}\)，所有符号均表示存储残数。在6次模乘xADD中，令

$$
A=(X_P+Z_P)(X_Q-Z_Q),\qquad B=(X_P-Z_P)(X_Q+Z_Q).
$$

前两次Montgomery乘法给出\(A/R,B/R\)，分别求和／差并除2、平方、乘差分点坐标后，输出为

$$
X'=\frac{Z_D(A+B)^2}{4R^4},\qquad
Z'=\frac{X_D(A-B)^2}{4R^4}\pmod N.
$$

因此可以先以普通模乘计算中间残数，再对两个最终坐标各乘一次\(R^{-4}\)，得到**完全相同的存储值**，不只是仿射意义上的同一点。任意中间值仍须正确模约减和规范化；不能直接丢掉表示因子，也不能丢弃高半乘积。

同理，标准xDBL若使用**普通表示**的系数\(a_{24}\)，先算\(AA=(X+Z)^2,BB=(X-Z)^2,E=AA-BB\)，可得普通中间输出\((AA\,BB,\ E(BB+a_{24}E))\)。两个坐标最后各乘\(R^{-3}\)，等于原Montgomery点公式。此处不能把已乘R的\(a_{24}\)原样用于普通模乘；系数转换必须另计。

在\(N=2^p-1\)上，\(R^{-k}\)是p-bit环右旋\(k(64W)\bmod p\)位。故理论上可把xADD内部6次、xDBL内部多次表示旋转合并到最终两坐标；普通模数下的最终修正可能仍需昂贵运算，不能套用梅森路径的低成本结论。

这是待实现的代数候选，不表示当前GPU后端已延后修正。CPU整数核查覆盖127／8193／10267／12511／16319 bits、普通／梅森、随机及0／N−1边界，共320组；证据为`data/experiments/stage2_point_register_20261101/representation_identity.json`及同目录脚本。它只验证恒等式，不验证GPU进位、寄存器、单位性恢复或完整Stage2性能。


## 11. 精确长除法的 warp 协作组织

本节描述一般的多精度长除法重排，不改变商估计或模数。令基数 $b=2^{64}$，归一化除数为 $N_s$，占 $W$ 词；当前商位的被减窗口占 $W+1$ 词，商估计为 $q<b$。完整乘积可分解为

$$
qN_s=L+bH,\qquad
L=\sum_{i=0}^{W-1}\operatorname{lo}(qn_i)b^i,\quad
H=\sum_{i=0}^{W-1}\operatorname{hi}(qn_i)b^i.
$$

因此可以先减 $L$，再减左移一词的 $H$。每一遍都执行完整 $W+1$ 词减法，不能把两路借位未经处理直接合并。因为 $L+bH=qN_s<b^{W+1}$，这两遍至多发生一次整个窗口的下溢；否则被减总量至少为一个窗口基数，与上界矛盾。两遍下溢标志的逻辑或即为完整乘积的下溢标志。随后沿用商估计所要求的有限次除数加回；不能省去商校正。

词减法 $a_i-v_i-c_i$ 的借位可由两个布尔量描述：生成 $g_i=[a_i<v_i]$，传播 $p_i=[a_i=v_i]$。输入借位的传播只需找到前方最近一个不传播的词，其生成位决定当前借位；若没有这样的词，使用该组的输入借位。一个32线程warp可以用vote位图和最高置位定位求出各词输入借位，再把最后一词的借位传给下一组。末组不足32词时，虚拟高词必须传播借位；加回中的虚拟高词也须传播进位。将其置零而阻断传播会产生错误。

设输入归一化后有 $U$ 个有效词，商位数 $J=\max(0,U-W+1)$。标量每商位含 $W$ 个完整64×64乘积和串行借位依赖；warp重排仍产生相同乘积，工作复杂度仍为 $O(JW)$，常见 $J\approx W$ 时仍为 $O(W^2)$。它缩短的是整数内部的依赖链，并改变余数的存储位置。每个商位还保留商估计、两遍词组前缀和最多两次加回，不能按32线程直接承诺32倍吞吐提升。

对线程块大小 $T$（32的倍数），每块缓存一次除数、每warp保留一个余数，动态共享内存为

$$
M_{\rm shared}=8\left[W+\frac{T}{32}(2W+4)\right]\ \text{bytes}.
$$

词组数为 $\lceil(W+1)/32\rceil$，额外的高词不能遗漏。因此 $W=160,192,224,256$ 等边界会新增一组，不宜只按位宽平方平滑外推。更多并行线程、共享内存容量、系数批量和缓存行为共同决定交叉点。数学重排本身不能证明某设备或完整ECM流程更快；当前实现与测量资格以性能文档为准。

### 合并商乘积后的一遍余数更新

令 $\ell_i=\operatorname{lo}(qn_i)$、$h_i=\operatorname{hi}(qn_i)$，约定越界项为零。先对乘积各字执行精确加法进位：

$$
v_i=(\ell_i+h_{i-1}+k_i)\bmod b,\qquad
k_{i+1}=\left\lfloor\frac{\ell_i+h_{i-1}+k_i}{b}\right\rfloor.
$$

再计算余数窗口 $a_i-v_i-c_i$ 的借位前缀。乘积进位与减法借位是两个不同前缀，不能混为一个标志。乘积加法的生成条件为 $\ell_i+h_{i-1}\ge b$，传播条件为两项之和模 $b$ 等于 $b-1$；与上述借位前缀一样，可用warp位图求各词输入。组间同时保留乘积进位和减法借位。由于 $qN_s<b^{W+1}$，第 $W+1$ 字以上不允许有有效乘积进位。

此重排使每商位对余数窗口的逻辑共享读／写从各 $2(W+1)$ 词降到各 $W+1$ 词。低／高乘积、除数字读、两个前缀以及商校正仍存在，算术工作量仍为 $O(JW)$；它不意味着总访存或耗时减半。具体共享事务还受编译器、bank组织、部分组及加回影响，须由实际内核剖析确认。

### 不变除数的寄存器缓存

同一系数的所有商位使用相同的 $N_s$。合并乘减中反复读取 $n_i$ 和 $n_{i-1}$，可将它们按32词组预装为每线程两个固定索引视图。设模板容量为 $W_c$ 词，缓存组数

$$
K_c=\left\lceil\frac{W_c+1}{32}\right\rceil.
$$

每线程两份64位视图的名义载荷为 $16K_c$ bytes，即 $4K_c$ 个32位寄存器。必须静态展开索引；动态可寻址私有数组可能落入local memory，不能只凭源代码数组大小推断寄存器用量。容量不等于实际 $W$ 时，额外空组仍可能占用寄存器。

按有效除数字计算，未缓存时每商位的两个对齐视图合计读取 $2W$ 个64位词；缓存把这一部分从每系数约 $2JW$ 次共享词读取改为约 $2W$ 次初始读取。商估计仍需最高除数字，商校正可能再次读取整个除数。余数窗口读写、乘积进位／借位前缀、$JW$ 个完整乘积及顺序相关的 $J$ 个商位都没有消失，因此仍是 $O(JW)$ 算术，不能据此承诺总共享流量减少同一比例。

缓存占用与可驻留warp数有取舍，有限组展开还会改变指令数。固定输入的整体对照应把缓存和展开作为联合实现，并分别测量共享事务、寄存器、local访问和完整曲线时间；当前资格见[归约性能](../performance/STAGE2_S4_REDUCTION.md)。

## 12. Carry完整检查的局部汇总

对长度 $L$ 的最终digit数组 $v_i$，基数为 $2^b$，完整检查求

$$
C=\sum_{i=0}^{L-1}[v_i\ge 2^b],\qquad
H=\max_{0\le i<L}\operatorname{bitlength}(v_i),\quad \operatorname{bitlength}(0)=0.
$$

若每块 $T\le1024$ 个线程、每线程至多检查一个digit，块内有 $0\le C_t\le T$、$0\le H_t\le64$，所以局部32位无溢出。先在完整32线程warp内用shuffle归约，再由首warp归约各warp结果，可以保留同一个数学和／最大值。末块越界线程贡献零，仍参与同步。全局计数必须保持64位，并将新的和／最大值合入已有诊断，而不能覆盖之前未处理的失败。

每个batch成员读取 $8L$ bytes，至多有 $\lceil L/T\rceil$ 个全局最大值更新，以及相同数量的非零坏值累加；运算仍为 $O(L)$。两个32项的局部数组需要 $2\cdot32\cdot4=256$ bytes共享内存，无全局scratch。这是依赖和存储组织的变化，不是减少检查范围。共享原子、shuffle和同步的实际成本由硬件与编译结果决定，渐近式不能保证完整Stage2收益。
