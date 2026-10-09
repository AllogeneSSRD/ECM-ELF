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

## 公式与来源

- [Montgomery 曲线、Suyama 与 PRAC](ECM_MONTGOMERY_MATH.md)：完整公式及参考实现入口。
- [参数化与成功率](ECM_PARAMETERIZATIONS.md)：光滑性、点阶与群阶。
- [Edwards 数学](ECM_EDWARDS_MATH.md)：另一种点表示及其 Stage1/Stage2 转换。

