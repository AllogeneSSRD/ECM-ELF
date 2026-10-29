# GPU Stage1：数学流程与并行成本

固定数学资料，提取自 `ECM_GPU_FLOW.md`，并按 `ECM_Montgomery_STAGE1.md` 的差分坐标约定整理。这里描述算法模型；具体容器、布局、参数化和检查点格式见[当前 Stage1](../architecture/STAGE1.md)。

## 输入与指数

输入为奇数合数 N、光滑界 B1、曲线种子集合及曲线族。令

$$s=\operatorname{lcm}(1,\ldots,B1)=\prod_{\ell\le B1,\ \ell\text{ prime}}\ell^{\lfloor\log_\ell B1\rfloor}.$$

同一 B1 的曲线可以共享 s 的位序列；每条曲线的参数和点状态独立。选择额外乘子 t 时，实际标量为 ts，必须用这个标量解释命中条件。

Suyama PARAM0 与 batch 参数化的 sigma 定义、初始点和常数构造不同，不能互换。Suyama 的 `u=σ²−5, v=4σ, P=(u³:v³)` 及 a24 公式见[Montgomery 数学](ECM_MONTGOMERY_MATH.md)。

## Host / device 数据流

```text
Host: N、B1、种子、曲线族
  ├─ 构造共享标量及位序列
  ├─ 构造各曲线参数、差分点
  └─ 将点/常量转换到选定域
             ↓
Device: 每曲线独立 ladder
  维护 R0=[k]P、R1=[k+1]P，差为 P
  对标量各 bit 执行 xDBL + xADD
             ↓
Host: 解码末点 (X:Z)
  d=gcd(Z,N)
  ├─ 1<d<N：有效因子
  ├─ d=1：可逆，x=X/Z mod N，供后续 Stage2
  └─ d=N：饱和/退化，需要单独处理
```

求逆成功得到的是末点的仿射 x，不是因子。因子必须满足 `1<d<N`；不能把 `d≠N` 当作完整条件。

## Ladder 与切片

```text
R0=O; R1=P
for bit in scalar, from MSB to LSB:
    U=xADD(R0,R1,P)
    if bit==0:
        R0=xDBL(R0); R1=U
    else:
        R1=xDBL(R1); R0=U
```

差分公式需要有效的 P 表示。采用仿射差分优化时先保证 ZP=1；采用齐次差分时乘上相应 XP/ZP。不同分支可用条件交换保持同形执行，但是否实现完整恒定时间还取决于模算术和访存，不能只凭 ladder 名称保证。

标量是公开量，同批曲线共享 bit 分支。切片只改变调度，不改变不变量；续跑必须保留两相邻点、固定差点、曲线常数、域身份和下一 bit 位置。

## 算术与并行模型

记 M 为一次模乘、S 为一次模平方。归一化差分点下，一步 ladder 为 `6M+4S`；齐次差分下为 `7M+4S`，其中 a24 乘法按一般乘法计。特殊小常数或平方专用实现应另计。

设曲线数 C、处理的位数 b、每个整数 ℓ 个 limb。schoolbook/CIOS 主乘累加量约为 `O(C·b·ℓ²)`，模加减约 `O(C·b·ℓ)`。阶段总时间还包括指数/曲线准备、传输、解码、GCD 和输出。

一般 GCD/求逆不能固定写成 O(ℓ³)：复杂度依赖算法与乘法后端。它们每条曲线的次数少，也不代表绝对耗时总能忽略。

协作线程布局可用 `instance=floor(global_thread/TPI)`；每 block 逻辑实例数为 TPB/TPI。更多协作线程减少每线程 limb，却会增加通信、padding，减少 block 中的独立曲线；实际驻留受寄存器、shared、线程等共同限制。

## 可验证的小型 Stage1 示例

取 `N=1139=17×67`，Suyama PARAM0 的 `sigma=6`。构造得到 `A=503 mod N`，起点 x-only 表示为 `(177:156)`，仿射 `x(P)=943 mod N`。为绘制完整坐标，取 `y(P)=1`，选 `B=x(P)^3+A·x(P)^2+x(P)=310 mod N`，得到等价的 Montgomery 扭曲代表 `310y²=x³+503x²+x mod N`。此处的 y 和 B 用于解释 x-only 运算，不是生产状态中的额外输出。

模 17 的曲线为 `4y²=x³+10x²+x`，`P=(8,1)`，点阶 12、群阶 24；模 67 为 `42y²=x³+34x²+x`，`P=(5,1)`，点阶 36、群阶 72。两条曲线均非奇异。点集可枚举，点阶可用独立仿射群运算核对 x-only ladder。

明确采用 `s=lcm(1,…,B1)`，不乘额外常数。B1=3 时 s=6，两个点都不归零；B1=5 时 s=60，仅模 17 的点归零；B1=9 时 s=2520，两个点都归零。用上述初始投影坐标、`R0=O,R1=P`，逐位执行标准差分 ladder，三个末点依次为 `(935:909)`、`(356:187)`、`(528:0)`；末点 GCD 分别为 1、17、1139。投影缩放可以改变坐标数值，点及 GCD 的这些分类在可逆缩放下不变。

该例说明需要检查点阶中的完整素数幂；只在末点检查 GCD 时，更大的 B1 还可能使两个因子同时归零而饱和。教学图能画出两个约化，是因为 N 的分解在算例中预先已知，不能对未知因子的生产 N 任意选小素数来声称等价。可交互验证见[数学演示](../web/gpu-stage1-visualization-20261026.html#demo)。

## 公式与来源

- [Montgomery 曲线、Suyama 与 PRAC](ECM_MONTGOMERY_MATH.md)：完整公式及参考实现入口。
- [参数化与成功率](../architecture/ECM_PARAMETERIZATIONS.md)：光滑性、点阶与群阶。
- [Edwards 数学](ECM_EDWARDS_MATH.md)：另一种点表示及其 Stage1/Stage2 转换。

