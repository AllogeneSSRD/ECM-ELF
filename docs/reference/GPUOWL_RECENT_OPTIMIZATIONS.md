# gpuowl 近期 FFT 与 NTT 优化分析

研究快照：2026-11-02，本地 `.refactor/gpuowl`，origin 为 `https://github.com/gwoltman/gpuowl.git`，master HEAD 为 `cb560c93fdeeaee07a27b64d3ce1a3e081001577`；目录干净。它的 README 自称 PRPLL，另有 preda/gpuowl 的 upstream，不能把两个仓库的提交混称为同一发行版。本文重点核查 2026-10-24 至 2026-11-02 的相关提交和 HEAD 实现，不声称覆盖未来上游变化。

对照项目为 ECM-ELF 0.57.10、HEAD `9d94e4c7`。本文是固定版本的第三方源码分析，代码位置按该快照阅读，不声明候选已在本仓库实现。离线阅读版见 [HTML](../web/gpuowl-optimizations-20261102.html)。基本域定义、IBDWT 与 CRT 推导仍见 [GPUOWL NTT](GPUOWL_NTT.md)，CUDA 兼容层见 [GPUOWL FFT](GPUOWL_FFT.md)。

**结论：优先借鉴编译器可见的数据布局、短临时量存活期、按形状调优；OpenCL NTT 的多层 outer 融合机会比照搬 GF61 算术更明确。** CUDA 已有 radix4、warp tail、协作 outer、融合 pointwise/scale，不宜再次把这些列为新增收益。

## 1 最近的改动集中在哪里

| 提交 | 已核查的变化 | HEAD 中的状态 |
|---|---|---|
| `9e25a20`、`2beba2a`，10-25/26 | PFA 输入旋转并入全局加载地址，输出旋转并入 LDS 转置写入 | 通过 `MIDDLE_IN_INDEX`、`SHUFFLE_ROW` 保留；in-place 路径仍有寄存器旋转 |
| `c73b172`、`f4f9a2e`，10-26/27 | 展开 GF61 小数组循环，阻止动态选址及过度提升 twiddle 导致 scratch/VGPR 峰值 | PFA≥9 的值选择保护、PFA 列展开与有条件编译器 fence 保留 |
| `2aeb736`、`56ede2a`、`99877d3`，10-26/27，11-01 合并 | GF61 in-place 和 L2 stripe 的地址活跃期缩减 | GF61 的 `zerohack` 分支保留；不是所有类型统一使用 |
| `f653187`，10-27；`21a51c4`，11-01 | GF31 twiddle 循环展开 | 最初局限 tailSquare；HEAD 已在 GF31 `tabMul` 中始终展开，旧注释/旧测量不能当作最终作用范围 |
| `426bdbb`、`049dcbd`，10-28；`dec7080`，11-01 合入 | GF31 unsigned-min 约减；GF61 AMD funnel shift | 前者按设备默认及 tune 选择；后者为 AMD builtin 分支 |
| `cb560c9`，11-02 | GF61 radix4/8 在蝶形前共享补加 M61 的倍数，减少逐输出补加 | 当前 HEAD，重新写有中间范围注释，未附足以量化完整性能的新基准 |
| `280c354`、`cfee70f`，10-26 | 将 256-radix8 纳入 tune；FP64/NTT 探针移到用户指数范围 | 保留；不能用固定 141M 探针代表所有工作规模 |
| `91713d9`，10-30 | PDL 等待增加编译器 `memory` clobber | 保留，属于编译排序正确性，不是新增吞吐结论 |

这里的 PFA 指互素因子分解的 Prime Factor Algorithm；不能把它泛称为任意 radix 融合。上表作者时间与合并时间不同，例如 10 月 26 日的改动在 11 月 1 日合入；分析按内容和 HEAD 状态核对。

## 2 最值得借鉴的是消除 scratch 与控制寄存器存活期

### 2.1 小数组必须让编译器看见固定索引

[`pfaRotate`](../../.refactor/gpuowl/src/cl/fft-middle.cl#L1286) 原来用运行时条件选择 `a[i+s]` 或 `a[i]`。即使循环展开，LLVM 仍可能把它合并为“选择地址后加载”，导致私有数组不能标量化，进入 scratch。

新写法通过 AMD 的 OPAQUE 保持对已加载值的选择，配合 [`pfaMiddleIn`](../../.refactor/gpuowl/src/cl/fft-middle.cl#L1344) 与 `pfaMiddleOut` 的固定循环展开，让小数组保持在 VGPR。关键不是机械添加 pragma，而是确保索引和选择在编译器眼中仍是寄存器值操作。

但全部展开也会过度增加同时存活的列与 twiddle。[`PFA_FENCE`](../../.refactor/gpuowl/src/cl/fft-middle.cl#L1337) 用空的 volatile asm 隐藏值，限制跨列重排和 twiddle 提前加载；twiddle 尽量在要使用时才生成。它是**编译器调度约束，不是线程同步栅栏**，不能替代 barrier 或内存可见性保证。

`f4f9a2e` 的提交说明报告：ROCm 7.2.4，gfx1100/gfx906，PFA middle 10/12/14 原有 176–240 B scratch；展开后 scratch 消失，但不限制列重排会把某些 kernel 从 98 推到 244 VGPR。因此优化目标是“固定小索引且控制活跃集”，不是单纯增加展开程度。

**本仓库的适配位置：**

- CUDA [`outer_fwd_kernel`](../../src/ecm/cuda/ntt/device/tile.cuh#L57) 的 `x[2^M]` 已使用编译期 M 和展开；逆向还有 `tbv[M]`。值得检查高 M 下地址、根和所有 x 是否过早同时存活，而不是假定多加 pragma 一定更快。
- CUDA [`outer_coop_kernel`](../../src/ecm/cuda/ntt/device/outer_coop.cuh#L15) 已把大 radix 轴放在 shared，避免每线程拥有庞大 `x[2^M]`；它不是遗漏了寄存器化。候选应是小 Q 的寄存器局部蝶形、根的就地生成和局部存活期，而非将整块 shared 数组强行搬到寄存器。
- OpenCL [`ecm_ntt_tile`](../../src/ecm/opencl/resources/ntt/transform.cl#L42) 仍通过 local 数组逐层交换，内外循环含运行时方向。可以比较前向/逆向专门化、固定 radix 小寄存器块，减少寻址与同步；同时保留当前 local 实现作为资源受限回退。
- 对 Stage1 CGBN 或 Stage2 协作点算术，可以借鉴相同的临时量管理，但它们不是 PFA，不能直接搬入旋转算法或 OPAQUE 宏。

**收益判断：**仅在现有内核确实有动态索引 scratch 或高寄存器峰值时，才可能出现较大收益。CUDA 已静态展开的路径不应套用 AMD 的两位数比例。scratch 容量、寄存器 spill 和 GPU 累计 local 访存不是同一个指标，须分别查编译资源与实际指令/计数器。

### 2.2 重算少量地址，可能比长期保留地址更便宜

[`fftMiddleInGF61`](../../.refactor/gpuowl/src/cl/fftmiddlein.cl#L499) 的 in-place 路径读写位置相似，编译器原来把许多 64 位地址跨整个 FFT 保留下来。`zerohack=g/131072` 在其使用形状中为零，却让编译器重新计算写地址，降低地址活跃集，避免 scratch。

`2aeb736` 报告某 MIDDLE16 kernel 从 256 VGPR＋100 B scratch 变为 235 VGPR、无 scratch；`99877d3` 在 stripe 路径报告 256＋88 B 变为 233、无 scratch。这说明地址寄存器本身值得分析。

本仓库可参考的方向是“以相对索引代替一组长期存活的 64 位指针、在最后写回时重算地址”。**不应直接复制魔数除法**：本项目 grid 可能超过原形状边界，商不再为零；不同编译器也可能消掉技巧或产生额外指令。尤其不能据此对有别名的输入输出增加错误的 restrict 声明。

## 3 把置换融入已有数据搬运

### 3.1 这次改动消除了什么

PFA 原本在小数组中做运行时旋转，成本约为 PFA·ceil(log₂PFA) 个选择/分量。`9e25a20` 让 [`readMiddleInLine`](../../.refactor/gpuowl/src/cl/middle.cl#L89) 从所需旋转后的行直接加载；索引定义在 [`pfaInIndex`](../../.refactor/gpuowl/src/cl/base.cl#L307)。同一组具有相同旋转，因此没有破坏其原有合并访问。

`2beba2a` 让反向结果直接写入 LDS 转置的目标行。对应 [`SHUFFLE_ROW`](../../.refactor/gpuowl/src/cl/fft-middle.cl#L83) 和输出 middleShuffle，省掉一次寄存器选择网络。变换、索引及结果布局需要成套修改，不能只更改其中一边。

**本仓库的适配：**CUDA [`tile_kernel`](../../src/ecm/cuda/ntt/device/tile.cuh#L158) 和 [`goldilocks_tile.cuh`](../../src/ecm/cuda/arithmetic/goldilocks/goldilocks_tile.cuh) 已通过 DIF/DIT 和 warp-tail 的读写布局消除显式位逆序。因此没有一个尚待删除的全局 bit-reversal pass。可继续检查局部 warp/shared 交换或 pack/gather 能否与已有加载、最终写回合并；不存在同类选择网络的地方就没有这项直接收益。

gpuowl 另有 [`middleShuffleIdx`](../../.refactor/gpuowl/src/cl/fft-middle.cl#L61) 对角式 local 布局，值得作为 bank-conflict 分析范例；它是此快照的配套机制，不据此声明也是本周新增。移植时需按 NVIDIA/AMD、wave32/64、32/64 位元素重新模拟和计数。64 位访问本来需要更多 bank 事务，不能把正常多事务全称为冲突。

**优先级：**有明确独立置换或选择网络时较高；本仓库已合并的 DIF/DIT 路径优先级较低。任何布局变化都必须同步正反变换、点乘谱顺序和根索引。

## 4 对本仓库 OpenCL，更大的机会是减少 outer 全数组遍历

当前 [`forward_passes`](../../src/ecm/opencl/ntt/runtime.cpp#L343) 在 tile 外逐个 radix2 层调用 `global_pass`；逆向也是逐层提交。设备端 [`ecm_ntt_global`](../../src/ecm/opencl/resources/ntt/transform.cl#L19) 每层读写整份数组。tile 内已经局部融合，且 pointwise/scale 已并入 inverse tile，但 outer 尚未具有 CUDA 的多层融合结构。

因此，比率先移植 GF61 汇编更值得做的是：

1. 给 OpenCL 增加小 radix4/radix8 outer 候选，让一个工作组在局部完成多个相邻层。
2. 每线程只持有固定少量元素，按编译资源决定用寄存器还是 local 交换。
3. 借鉴 gpuowl 的固定索引、短存活期、实际形状调优，避免融合后出现 scratch 或 occupancy 悬崖。

例：L=2²⁵、tile_bits=8 时，一次卷积的两正一逆共约 3·(25−8+1)=54 个变换 pass。若每 outer pass 融合至多 4 层，约为 3·(ceil(17/4)+1)=18 个。按每 pass 读写 L 个 8-byte 元素计算，逻辑数组流量少约 **18 GiB/卷积/batch成员**。

这是结构估算，不含根表、pointwise 的另一输入、pack/carry/S4，不等于真实 DRAM 流量；缓存、索引代价和新 kernel 资源会改变实测结果，也不代表完整 Stage2 快三倍。实际 tile_bits 必须用所选计划，示例不替代运行回执。

CUDA 已有 [`outer_coop_kernel`](../../src/ecm/cuda/ntt/device/outer_coop.cuh) 及自动层数表，适合作为项目内语义基线；移植时仍保留 OpenCL 原生资源查询和不同设备的回退，而不是直接复制 CUDA 专用操作。

## 5 新域运算优化的可迁移范围

### 5.1 GF61 延迟约减与补偿合并

最新 `cb560c9` 在 [`fft4Core`](../../.refactor/gpuowl/src/cl/fft4.cl#L176)、`fft8.cl` 先给部分输入共享补加若干 M61，再做蝶形，最终调用约减，减少逐输出的补偿加法。辅助函数在 [`addq`](../../.refactor/gpuowl/src/cl/math.cl#L1408)。其价值在于逐步证明中间范围，再调整补偿的位置。

但 M61=2⁶¹−1 在 64 位机器字中约有 3 位余量，允许多个 M61 倍数的冗余表示。本仓库 q=2⁶⁴−2³²+1 接近机器字上限，2q 已超过 64 位；[`gl_sub_canonical_ptx`](../../src/ecm/cuda/arithmetic/goldilocks/goldilocks_sub.cuh#L5) 明确依赖规范输入，不能直接传入任意冗余残数。

因此可借鉴“把范围证明与蝶形代码一同维护、共享补偿”的方法，不能直接采用 M61 的常数、折叠公式或取消本项目的规范化。若为 Goldilocks 引入冗余表示，需要额外 carry/宽度或有证明的不同表示，并重新比较收益；目前不是低成本的首选项。

### 5.2 AMD 指令选择与 GF31 unsigned-min

[`weakModM61`](../../.refactor/gpuowl/src/cl/math.cl#L1130) 的 `dec7080` 用两个 `alignbit` 形成跨 32 位字的移位，替换较慢的 64 位移位。它是 AMD 专用的同值实现，报告大多约 1% 迭代收益，但 MIDDLE16 的寄存器增加可抵消收益。

[`modM31`](../../.refactor/gpuowl/src/cl/math.cl#L729) 的 unsigned-min 用减法加 min 代替其他条件约减；代码也注明 sm120 上并未显著胜出。M31 的特殊输入条件和范围不能移植到 q64。

本仓库 CUDA 已使用 [`gl_mul`](../../src/ecm/cuda/arithmetic/goldilocks/goldilocks.cuh#L54) 和专用 PTX 归约。更相关的是 OpenCL [`goldilocks.clh`](../../src/ecm/opencl/resources/ntt/goldilocks.clh#L9)：已有 portable `mul_hi` 与 `split_multiply` 两种形式，应该按设备和编译器比较实际指令、carry 链和资源。不要根据高层表达更短就判断更快；不能把 gpuowl 针对有界 61 位乘积省略的进位照搬到完整 64 位 Goldilocks 乘积。

## 6 调优方法比一个全局最优参数更值得借鉴

[`tune.cpp`](../../.refactor/gpuowl/src/tune.cpp#L445) 的 `cfee70f` 改掉了“所有规模都用 141M 指数比较 FP64/NTT”的做法：当用户范围不含 141M 时，在实际范围中间选探针，并使用足够小、能容纳该指数的 favored shape。缓存容量、发射开销与运算类型的优劣会随规模变化。

`280c354` 把 width/height=256 时的 radix8 组合加入计时，同一 7900 XTX 上，NTT 一种形状约快 4.3%，FP64 的另一些形状反而慢。它还记录 quick=10 的单次计时选错配置的例子。因此应比较实际形状和正反方向，不能把某一 radix 视为全设备最优。

HEAD 已将 GF31 `tabMul` 循环始终展开，见 [`fftbase.cl`](../../.refactor/gpuowl/src/cl/fftbase.cl#L1971)。`f653187` 最初的“仅 tailSquareGF31 展开、其他 kernel 不变”及其约 1–1.6% 数据，只支持当时那个补丁，不支持后来扩大作用范围后的所有 kernel。提交历史中的性能数字必须绑定被测版本。

对本仓库应按 device/backend、L、batch、tile/outer、输入驻留、寄存器/local、计时边界分组做算子调优，再以固定 D 的完整曲线确认。现有 [NTT outer 资格](../performance/NTT_OUTER.md)已有更激进联合表仅少约 0.19% 的例子；它说明完整收益受阶段占比限制，不代表将来所有新内核只能取得这个比例。

## 7 不能直接搬来的算法及路径

- **混合 FP32/FP64 与 NTT/CRT。** gpuowl 的任务是巨大梅森整数重复平方，IBDWT 的循环回绕就是目标模数归约。本项目的多项式 Kronecker 槽要求线性卷积和精确系数界，不能去掉补零，也不能用 roundoff 经验门限替代精确性证明。
- **任意 PFA 中间因子。** Goldilocks 域 q−1=2³²·(2³²−1)=2³²·3·5·17·257·65537。它直接支持 3、5、15 等因子，但没有 7、9、11、13、14 阶单位根。gpuowl 的二次扩域可用因子与本项目不同，不能照搬整套 PFA 表。研究 mixed-radix 还要修改 shape、根、逆长度归一化、carry/缓存及 tune，成本高于局部融合。
- **L2 striping。** 可借鉴缓存工作集意识，但只能对依赖闭合的分块串联阶段。NTT 外层会连接不同 stripe，不能任意拆出整条独立小卷积；若利用多个独立 batch，应分别验证缓存命中与提交开销。
- **PDL。** [`dependentLaunchWait`](../../.refactor/gpuowl/src/cl/base.cl#L1170) 的 `memory` clobber 防止编译器把读移到等待前，属于正确性要求。该路径以 CUDA、sm90+、显式 PDL 为条件；现有 sm89 的 4060/4070 路线没有直接收益。

## 8 上游数字应怎样理解

以下均为**提交说明中的作者测量**，没有在本机复现。指标是 PRPLL 每次迭代的微秒，通常包含一轮算法流水线，不是一个 butterfly，也不是 ECM Stage2 时间。提交未为这些样本提供完整功耗/锁频记录和逐次原始日志，不能补填为恒频基准。

| 改动与来源 | 上游报告条件 | 报告结果与限制 |
|---|---|---|
| `f4f9a2e`：PFA 展开＋列约束 | RX7900 XTX；ROCm7.2.4；10000 iterations；8 组随机顺序配对；形状 1:512:12:512 | 1304.0→983.2 μs/iter，约少24.7%；其他若干形状少18.7–26.8%。这是原补丁及指定形状，不代表新 HEAD 的所有设备 |
| `9e25a20`：加载地址吸收旋转 | RX7900 XTX；2000 iterations；3 次交错；1:512:9:512 | 815.2→756.4，约少7.2%；结果 res64 一致，非独立完整结果证明 |
| `2beba2a`：LDS 写入吸收旋转 | RX7900 XTX；2000 iterations；3 次交错；1:512:9:512 | 758.2→691.6，约少8.8%；起点与上一行不同，不能把两个百分比相加 |
| `2aeb736`：in-place 地址存活期 | RX7900 XTX；INPLACE=2；10000 iterations；4 组配对；3:512:16:512 | 899.8→844.4，约少6.0%；该提交另列 1K 形状约少11.8% |
| `dec7080`：AMD funnel shift | RX7900 XTX；10000 iterations；8 组随机配对；在另一乘积优化之上 | 多个 MIDDLE7/8 形状约少1%；MIDDLE16 无稳定收益或回退，显示指令更少仍可能更慢 |

源提交含 POCL/iGPU 的结果一致性及随机主机算术核对说明，应按各自覆盖范围理解。res64 一致不等于本仓库需要的逐系数 GMP、完整 carry 边界或因子合同验证。

## 9 本仓库建议优先级

1. **OpenCL outer 多层融合。** 结构性减少全数组遍历；先做小 radix，不让寄存器/local 需求失控。这是当前源码最明确的未覆盖机会。
2. **CUDA/OpenCL 的小数组标量化与根/地址存活期审计。** 优先查有真实 scratch/local 流量的内核，按方向和形状分别处理；不全面添加 OPAQUE 或无条件展开。
3. **有确切置换成本时合并读写索引，并核查 shared bank 布局。** 已由 DIF/DIT 消除的全局置换不重复计算收益。
4. **按实际工作规模调优。** 小变换与大变换、batch1 与 batch多、普通与承载的完整成本不同，保留反序重复和回退门槛。
5. **域算术只做受证明约束的设备专门化。** GF61/GF31 算术和 mixed-radix 暂作为研究资料，不优先重构 Goldilocks 数学合同。

本仓库 Stage2 点后端已在 0.57.10 中具有协作路径和 auto 资格，见[协作后端选型](../architecture/AUTO_B2.md#协作点后端选型)；不能沿用较早分析的“仅单线程、尚未接入协作”作为当前基线。gpuowl 这些 FFT 改动对点算术主要是存储与编译器方法的启发，不提供直接替换 CGBN 的证据。

新增候选的完整收益只能按实际候选占比 f 与加速 g 估算：时间减少 f·(1−1/g)。未取得本地新计时前，不为 ECM-ELF 填入上游的24%或其他收益数字。

## 10 证据与入口

- 本地提交正文及研究文件哈希：`data/experiments/gpuowl_review_20261102/commits.txt`、`source_identity.json`，由 Git 忽略；仅为来源身份与说明保存，不是 GPU 测试数据。
- 上游：[cb560c9](https://github.com/gwoltman/gpuowl/commit/cb560c9)、[f4f9a2e](https://github.com/gwoltman/gpuowl/commit/f4f9a2e)、[2aeb736](https://github.com/gwoltman/gpuowl/commit/2aeb736)、[9e25a20](https://github.com/gwoltman/gpuowl/commit/9e25a20)、[2beba2a](https://github.com/gwoltman/gpuowl/commit/2beba2a)、[cfee70f](https://github.com/gwoltman/gpuowl/commit/cfee70f)。
- 本项目当前实现：[NTT](../architecture/NTT.md)、[OpenCL](../architecture/OPENCL.md)；现有性能：[NTT outer](../performance/NTT_OUTER.md)、[OpenCL Stage2](../performance/OPENCL_STAGE2.md)。本次仅新增分析文档与阅读版，没有修改实现、运行 GPU 性能实验或变更版本号。
