# Goldilocks 归约、逆长度缩放与梅森 Montgomery 数学

固定参考资料。来源：`STAGE2_GOLDILOCKS_SHORT_REDUCTION.md §3`, `STAGE2_GOLDILOCKS_PTX_REDUCTION.md 数学部分`, `STAGE2_NTT_INVERSE_SCALE_OPTIMIZATION.md §2`, `STAGE2_POINT_MERSENNE_MONTGOMERY.md §2`。保留数学推导、第三方源码分析及其条件；源码行号针对原文研究的本地副本，第三方上游可能已有变化。本文不声明本仓库当前支持、默认值或性能。涉及实验数字时只按原文条件理解；当前实现见 [架构](../architecture/OVERVIEW.md)，当前计时口径见 [性能](../performance/STAGE2.md)。

## 3. 任意 128 位输入的短归约证明

令 `ε=2^32−1`，`x=lo+hi·2^64`，`hi=h1·2^32+h0`，其中 `0≤lo,hi<2^64`，`0≤h0,h1<2^32`。利用：

```
2^64 ≡ ε (mod q)
2^96 ≡ −1 (mod q)
x ≡ lo − h1 + h0·ε (mod q)
```

先计算 `minus=lo−h1` 的 unsigned64 差。如果借位，实际加了 `2^64`；再减 `ε`，就变成加 `q`，保持同余。检测借位用 `minus>lo`。得到 `a=minus−borrow·ε`。

`b=h0·ε=(h0<<32)−h0` 可直接用 64 位表达，且 `0≤b≤ε²<q`。计算 `sum=a+b`，若进位则补回 `ε`，因为丢弃的 `2^64` 与 `ε` 同余。检测进位用 `sum<a`。

进位时 wrapped sum 至多 `b−1≤ε²−1=2^64−2ε−2`，所以加 `ε` 不会再次进位。无论是否进位，`value` 都是一个 unsigned64 值；由于 `2^64<2q`，最后至多减一次 `q` 即得到 `[0,q)` 的唯一结果。此证明不要求输入是两个 canonical field word 的乘积，覆盖任意 `(lo,hi)`。

原函数源级运行四次 fold，每次保留 carry/borrow 与高半；新函数用高 32 位一次减法、低 32 位一次移位差、两处 carry/borrow 修正和一次规范化。模乘的 `a*b` 与 `__umul64hi(a,b)` 保持。减少的是归约依赖链，不能写成减少所有模乘次数。没有可用 NCU 硬件计数器，因此不报告周期数或实际指令吞吐。


## 显式进位/借位链

NTT 素数 `q=2^64−2^32+1`，`epsilon=2^32−1`。ECM 模数另记 `n_ECM`。令 `hi=h1·2^32+h0`，则任意 unsigned128 输入满足：

```text
lo + hi·2^64 ≡ lo − h1 + h0·epsilon    (mod q)
a = lo − h1 − borrow·epsilon          (unsigned64)
b = h0·epsilon
sum = a+b                            (unsigned64)
value = sum + carry·epsilon
result = value≥q ? value−q : value
```

这是已验证短归约的同一恒等式，借位/进位检测可直接接到 PTX 的 CC 链，避免重新做 64bit 关系比较和选择。

`sub.cc/subc` 提供借位，提取成 0/−1 掩码；减该掩码并传播低 word 借位等价于条件减 epsilon。`b` 由低 word `−h0` 和高 word `h0−[h0≠0]` 组成。合并 a/b 后提取 carry，再条件加 epsilon。最后仅当高 word 是 `0xffffffff`、低 word 非零时减 q。

CC 的 carry/borrow 语义以及不跨函数调用保存的规则来自 [NVIDIA PTX 扩展精度算术文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#extended-precision-arithmetic-instructions-subc)。本实现把整个依赖链放在单个 asm 块内。[sppark 的 CUDA Goldilocks 类型](https://github.com/supranational/sppark/blob/9e5c795/ff/gl64_t.cuh#L130)也是显式进位算术的调研来源；此处使用本仓库 canonical 余数和任意 128bit 输入合同。

与旧短归约相同，进位后 wrapped sum 有上界，补 epsilon 不会再次进位；最终至多减一次 q。函数不依赖输入是两个 canonical word 的乘积。

## 2. 数学与精确性合同

以下 `q=2^64−2^32+1` 是 NTT 素数，ECM 模数另记为 `n_ECM`。设变换长度 `N=2^k`，`1≤k≤32`，输入 `0≤x<q`。

因为 `2^64 ≡ 2^32−1 (mod q)`，且 `(2^32−1)·2^32 ≡ −1`，有：

```text
2^(-k) ≡ 2^(32-k) − 2^(64-k)                 (mod q)
x = h·2^k + r,  h=x>>k,  r=x & (2^k−1)
x·2^(-k) ≡ [h+r·2^(32-k)] − r·2^(64-k)     (mod q)
```

令 `a=h+(r<<(32−k))`、`b=r<<(64−k)`。二者均是 canonical：

- `a < 2^(64−k)+2^32 ≤ 2^63+2^32 < q`；
- `b ≤ 2^64−2^(64−k) ≤ 2^64−2^32 = q−1`。

因此只需一次模减法。若 `a<b`，无符号减法已经加了 `2^64`，再减 `epsilon=2^32−1` 即得到正确余数；结果仍在 `[0,q)`，不需要一般 128bit 归约。

该方法并没有要求原多项式系数、ECM 模数或点坐标可逆。使用它之前的 pointwise `gl_mul(a,b)` 保证 x 为 canonical Goldilocks 余数。

实际 kernel 对 k 和 n_scale 做完整检查：仅当 `1≤k≤32` 且 `n_scale=q−2^(64−k)+2^(32−k)` 时使用移位。自定义 scale、k=0 或范围外继续使用原通用模乘。范围判断先于移位，避免无效移位。


## 2. 数学与计算量

约定 `N=2^s−1`，`W=ceil(s/64)`，`R=2^(64W)`，canonical `0≤a,b<N`，`d=(64W−s) mod s`。原接口返回 `abR⁻¹ mod N`。小于64位时也采用d模s，不假定N的最低字总是全一。

1. 计算完整schoolbook积u=ab，约W²个64bit MAC。
2. 分解u=l+2^s·h。由于u<N²，有l+h<2N。计算v=l+h，一次条件减N即可获得canonical v；s为64倍数时保留末尾进位，部分顶字时保留越过s的位。
3. `2^s≡1 mod N`，所以R⁻¹等价于s位右旋d。输出 `(v>>d) | ((v mod 2^d)<<(s−d))`。两部分位域不重叠；全一表示已在第2步被规范化，右旋继续canonical。d=0直接复制，避免64位无效移位。

这保持了Montgomery表示，不需要重新生成R、a24、Q、Γ、inverse或leaf。普通/Montgomery混合调用（例如Mont(xR,1/z)=x/z）也保持原语义。最终写r延后至a/b已完全读入局部积与规范化out，因此允许r=a或r=b。

通用SOS+REDC约2W² MAC/模乘，新方法W²+O(W)，不是GPU周期公式。M4423的W70/d57，从约9800降至4900 MAC。六模乘xADD仍为6次调用：主导项12W²→6W²；xdbl5次10W²→5W²；一个普通ladder bit的xADD+xdbl为22W²→11W²；chain续点12W²→6W²。这些不包括线性加减/halfmod、初始化与segment修正。

NTT形状、多项式逻辑乘法数和 Kronecker 算法不因点模乘归约原语改变。点加法条数与归约成本是不同优化维度，需分别计数。

实现：[折叠/旋转原语](../../tools/bench/stage2_point_mersenne.cuh#L6)、[当前分派](../../tools/bench/stage2_tree_gpu.cu#L704)、[host精确模数证明与状态重设](../../tools/bench/stage2_tree_gpu.cu#L4137)、[D模型保护](../../tools/bench/stage2_tree_gpu.cu#L10803)。

这里的N是ECM大模数；NTT的Goldilocks素数q及其PTX归约保持。NTT_S4_MERSENNE是此前对Kronecker宽系数做普通域归约的另一条路径，不能代替这里Montgomery结果所需的旋转。
