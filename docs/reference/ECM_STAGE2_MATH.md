# 多项式 ECM Stage2：Newton 逆、fold 与 scaled 下降

固定算法资料，提取自 `STAGE2_GPU_CURRENT_PIPELINE.md` 的公式、`STAGE2_GPU_COMPARE_PRMERS.md` 的 Prime95 分析和 `DEV_GPUOWL_NTT_NOTES.md` 的 scaled 下降推导。以下不规定 CPU/GPU 分工或是否驻留；工程现状见[Stage2 管线](../architecture/STAGE2.md)。

## Baby / giant 判据

选取偶数 D。候选素数 q 写为 `q=iD±j`，baby 点为 `[j]Q`、giant 点为 `[iD]Q`。由于 x(P)=x(−P)，点阶条件可转为两点 x 坐标相遇。

齐次坐标下，相遇因子为 `Xg·Zb−Zg·Xb`。在非奇异模素数 p 曲线上，该值为 0 是对应 x 相遇的判据；分母非单位、无穷远点及被 D 吸收的小素数需要单独处理。

令 `J={1≤j≤D/2:gcd(j,D)=1}`、`P=φ(D)/2`，仿射 baby 根 xⱼ 构成

$$F(X)=\prod_{j\in J}(X-x_j).$$

每个 giant 块构成 Gᵦ，累计 `H←GᵦH mod F`。最终在全部 xⱼ 上求值 H，得到交叉差乘积的等价批量结果。

## 反转与 Newton 逆

对长度 n+1 的系数定义 `rev_n(A)[k]=A[n−k]`。F monic，因此 `rev_P(F)` 的常数项为 1，在合数模环中也可以求形式幂级数逆。

```text
g=1
while precision < required:
    next=min(2*precision,required)
    g=low_next(g*(2-rev_P(F)*g)) mod N
    precision=next
finv=g
```

若 `a·g=1 mod Y^m`，则 `a·g(2−a·g)=1−(1−a·g)^2=1 mod Y^(2m)`，故每轮精度翻倍。中间截断长度是精度的一部分，不能因末尾零系数而任意缩短。

## 多项式模 F 归约

```text
T=G*H
if degree(T)<P:
    H=T
else:
    k=degree(T)-P+1
    qrev=low_k(rev_degree(T)(T)*low_k(finv))
    q=rev_(k-1)(qrev)
    H=low_P(T)-low_P(q*F) mod N
```

常见满次数 fold 有 G·H、逆序商、q·F 三次乘法。只需输出商/余式的指定窗口；截断数学结果不等于底层实现已执行截断 NTT。第一块 G 若次数达到 P，仍须在求值或根准备时正确处理其模 F 意义。

## 齐次缩放

使用 giant 叶 `ZᵢX−Xᵢ`，在 Zᵢ 可逆时有 `Zᵢ(X−xᵢ)`。全部 giant 引入的常数尺度为 `Γ=∏Zᵢ`。

模 F 的乘法和余式传播保留此尺度。需要仿射结果时乘 Γ⁻¹；所有非单位必须先在目标 N 上分类。若算术使用承载 M 且 N∣M，M 上不可逆不自动意味着 N 上不可逆。

## Scaled remainder / middle product

对 monic 节点多项式 M、次数 d，采用普通系数约定：

```text
r=H mod M
S_M=rev_(d-1)(r)/rev_d(M) mod Y^d
```

常数项为 1 的 `rev_d(M)` 可逆。目标孩子次数 a、兄弟次数 b 时：

```text
child_state=coefficients [b,b+a) of
            parent_state*rev_b(sibling_polynomial)
```

兄弟次数为 0 时复制，目标次数为 0 时跳过。在 monic 线性叶 d=1 时，状态中的唯一系数为 H(xⱼ)。这个约定把下降改写为截断/middle product，避免为每个孩子重做完整 Newton 逆与 divmod。

反转起点、截断窗口和状态长度取决于具体定义，不能在不同 scaled 约定间只按数组名称对应。须保持：

1. 根状态与 H mod F 的多点求值等价。
2. 子状态的实际次数与对应子树一致，尾树和空节点按真实次数处理。
3. 叶结果在消除可逆尺度后等于 H(xⱼ)，可用直接 Horner 验证。

Prime95 的具体运算与源码入口见[Poly 源码分析](PRIME95_STAGE2.md)。理论出处为 Bernstein 的 `scaledmod-20040820.pdf`，本地资料在 `docs/paper/Stage2/`。

## 工作量与空间

完整 F 树有 P−1 次非空逻辑乘法；若 I 个 giant 被分为 G 个块，各 G 树合计 I−G 次。逻辑乘法数不等于 kernel 数；满层可以按同形状合批。

以 Mpoly(P) 表示长度 O(P) 的一次多项式乘法成本，积树和下降的常用上界为 `O(Mpoly(P)log P)`，重复 G/fold 另乘块数。真实估计应累加非空尾树形状及截断窗口。

F 树是否保留、压缩、落盘或重建，以及共享操作数变换和 fixed-F 频谱缓存，是内存/重算的不同取舍。它们不改变上述数学目标，也不证明某个 GPU 映射一定更快。
