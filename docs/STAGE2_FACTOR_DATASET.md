# ECM Stage2 因子数据集、复合因子拆解与 Auto B2 准备

日期：2026-10-05。当前工作数据库已按用户确认精简为两表、每因子只保留一个最优sigma；历史运行证据保留在旧快照中。关联设计：[Auto B2 / tune](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_TUNE_DESIGN.md)。本轮完成可运行的数据采集闭环：生产 Stage1 → 保存点 → 生产 Stage2 → 原始因子拆解 → 群阶/点阶 → B1/B2 前沿 → 数据库更新。

## 1. 交付和范围

- [SQLite 数据库](D:/code/MPA-OpenCl/tools/ecm_dataset/ecm_stage2_dataset.sqlite)：Python 标准库即可读写，不需要 MySQL 服务。所有大整数用十进制 TEXT 保存，避免 SQLite 64-bit INTEGER 截断。
- [首轮历史便携快照](D:/code/MPA-OpenCl/data/ecm_stage2_snapshot_20261005/manifest.json)：每张表的 JSONL、因子 CSV、40 条配对测量 CSV、47 条真实 GPU Stage2 结果及日志/保存点原文、NTT 原始样本和审计记录。保留原路径用于追溯；换机器后可直接读取快照中的原文。嵌入文本经Python读取后换行统一为LF；各source SHA对应本机原始文件字节，快照文件另有自身SHA。
- [数据库 CLI](D:/code/MPA-OpenCl/tools/ecm_dataset/ecm_dataset.py:9)：导入因子表、计算群阶、导入原生 result、查询及导出。
- [主程序复合因子处理](D:/code/MPA-OpenCl/src/core/ecm_stage2_factorize.h:51)：可选 PARI/GP 拆解，保留原始因子并附带素因子、重数、证明状态。
- [生产实验工具](D:/code/MPA-OpenCl/tools/ecm_dataset/run_production_dataset.py:63)：实际 GPU Stage1 保存点和 CPU 参考一致后才进入 Stage2，记录二进制/保存点指纹及必需算术检查。

这是 Auto B2 的正确性语料和首轮 NTT 吞吐量数据。`--auto-b2`、跨位宽完整成本模型、K/(T1+T2) 联合搜索尚未接入。所选曲线刻意寻找可验证的因子，不能据此估计随机曲线成功概率。

## 2. 导入因子表与数据库内容

输入 `.refactor/Mersenne_exponent_factor_1-9999.html`，SHA256：

```text
0daf02b3a48fc1c823e2f7d48e2326cdbad71cc8c542c1d4388d04fad1eaa372
```

[解析器](D:/code/MPA-OpenCl/tools/ecm_dataset/dataset.py:57) 只读取 `id="M..."` / `/factor/...` 的数字属性，不执行 HTML。去重后导入 1,154 个指数、3,321 个因子；实际指数范围 11～9973，因子 2～146 十进制位。每个因子都独立验证 `2^exponent mod factor = 1`。这不等于全部目录项已证明为素数，`primality_verified` 在 GP 分析前保持 NULL。

另加入有效的 M256 / 59649589127497217 独立参考记录。首轮历史快照曾有250项分析、485项界限、108条观察和40条正式配对记录。后续用户新增分析后，本次迁移保留1,155个指数、3,322个因子和33个已选最优sigma；其他sigma历史及每次运行记录已从工作数据库删除。

`digital` 表示十进制位数：因子表取 `len(str(factor))`，梅森数表取 `len(str(2^exponent-1))`。不是 bit length；每次运行的实际 modulus bits 另存于测量导出。

[当前schema](D:/code/MPA-OpenCl/tools/ecm_dataset/dataset.py:23)（user_version=2）：

| 表 | 保存内容 |
| --- | --- |
| mersennes | exponent、`2^exponent-1` 表达式、十进制位数 |
| factors | factor、digital、素性状态；一个最优sigma、B1/B2及该sigma对应的完整群阶/点阶与分解，未分析时为NULL |

`sources`、`analyses`、`frontier`、`observations`、`production_runs` 不再存在。来源SHA、时间戳、耗时、原始result、分析错误、去重键和所有非最优sigma均不存入工作数据库。数学候选界限即时计算，不作为独立持久化记录。

## 3. 群阶、点阶和界限的数学合同

### 3.1 复用 PARAM0 模型，增加精确点阶

仓库 [Stage1 的 GP 模型](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:565) 使用 Suyama Montgomery 曲线。新 [param0_order.gp](D:/code/MPA-OpenCl/tools/ecm_dataset/param0_order.gp:4) 使用相同模型：

```text
u = sigma²−5, v = 4·sigma
x = u³/v³
A = (3u+v)(v−u)³/(4u³v)−2
b = x(x(x+A)+1)
b·y² = x³+A·x²+x，初始点 (x,1)
E = [0,bA,0,b²,0]，映射点 P = (b·x,b²)
```

计算 `n = #E(Fp)` 和 `o = ord(P)`，完整分解二者并证明全部分解项为素数。检查 `o | n`、`[o]P=O`、对每个素因子 q 都有 `[o/q]P≠O`。不能直接用群阶的最大素因子替代当前点的要求：点阶可能严格小于群阶。

独立审计用 Python x-only Montgomery ladder 重算后一组检查，不复用 GP 的 `ellorder` 或上述 Weierstrass 映射。250 个分析的 1,182 次标量检查、313 个不同素数的 GP 证明、Hasse 界及分解乘积全部通过。

### 3.2 标准单素数 semismooth 界限

令 `o=∏q^e_q`，Stage1 指数 `k(B1)=lcm(1..B1)`。若使用 `choose12`，指数再乘 12；令 c2=2、c3=1、其他 cq=0，普通 lcm 全部 cq=0。调整后指数 `a_q=max(0,e_q−c_q)`。

Stage1-only 最小界限：

```text
B1_only = max(2, max_q q^a_q)
```

若 Stage2 留下一个素数 ℓ 的一次幂：

```text
B1(ℓ) = max(2, max_q q^(a_q−[q=ℓ]))，省略指数≤0项
B2(ℓ) = ℓ，且 ℓ>B1(ℓ)
```

[bounds](D:/code/MPA-OpenCl/tools/ecm_dataset/dataset.py:108) 保留这些组合及 Stage1-only `(B1_only,0)` 的不可支配前沿。**数据库 B2=0 表示 Stage1-only；不是原生 CLI 的零值/自动配置语义，不可直接当成 Stage2 命令。** 数据库只保存lcm口径的最优记录。`--torsion 12`只改变临时返回的界限，由点阶推导choose12需求，不保存第二份sigma。本轮实际GPU曲线全部为lcm。

这些是标准单素数点阶模型的精确最小界限，描述阶的消去条件；仍需目标因子与其他因子分离，才能得到 proper GCD。它们不是当前多项式引擎“最小请求 B1/B2”的证明。

### 3.3 当前引擎还可暴露退化点，运行界限单独保存

当前几何有 `I=floor(B2/D)+2`，存在扫描尾部超出请求 B2 的情况；还有 [小素数补偿](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9620)、[baby 退化点](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11326)、[giant 归一化失败后的 gcd(Z,N)](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9976)。因此即使 residual order 不是一个素数，也可能在 Stage2 找到因子。

真实例子：M1367 / sigma17 / factor10937，初始点阶 `2760=2³·3·5·23`。B1=2 后 residual order 为1380；D=210 时 `1380/gcd(1380,210)=46`，所以 giant `[46D]Q` 在该因子下是无穷远点。正对照 B2=10663、目标负对照 B2=9823 都覆盖 i=46，并都返回10937；日志均有 giant 退化点。标准模型给出的 `(8,23)` 不排斥这一额外发现。保存点读取路径 [11069](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11069) 没有再次乘12。

实验负对照用 `B2_target−4D`，目标是验证目标因子缺席；不要求所有其他因子缺席。不能用 `B2−1` 证明当前几何的精确阈值，也不能把运行 B1/B2 覆盖写入数学界限字段。

## 4. 生产测试集与结果

测试使用 GPU1 RTX4060 Laptop（UUID `8a67b1f8ef1c3177a822813a7ac2224d`）。外部 GPU0 Stage1 任务未修改。各实例使用私有目录/INI，关闭 Stage1 exponent cache，限定单曲线。CPU 准备先完成，再运行 GPU；编译与 GP 批量群阶准备未混入正式 GPU 计时。

选取 7 个全梅森数位宽，13 个候选因 Stage1 已找到因子而被排除；另用实际 Stage1 找到的 GCD 剥离3个余因子案例，重新生成有效保存点。每个保存点的 normalized X 和 `B1·sigma·N·X mod4294967291` checksum 与独立 CPU 参考一致。

下面为生产基线正对照，全部 D=210；负对照 B2=表中B2−840。每项正对照找到目标因子，负对照未找到该目标；目标可包含于返回的复合因子中。

| 原 M / 实际 bits | 目标因子 | digital | sigma | B1 | B2 | 原生 seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 223 / 223 | 1466449 | 7 | 17 | 3 | 122323 | 0.486067 |
| 431 / 431 | 4642152737 | 10 | 26 | 29 | 952811 | 1.031933 |
| 1367 / 1367 | 2561759 | 7 | 17 | 2 | 10663 | 0.311874 |
| 2657 / 2657 | 148793 | 6 | 7 | 2 | 1777 | 0.519128 |
| 4933 / 4933 | 169745595529 | 12 | 11 | 673 | 12511 | 0.718378 |
| 6977 / 6977 | 880699495783 | 12 | 12 | 83 | 23557 | 1.268735 |
| 8171 / 8171 | 1089700903 | 10 | 16 | 17 | 667697 | 2.834135 |
| 2657 / 2607 | 77028063857760263 | 17 | 9 | 1543 | 1134149 | 1.835783 |
| 4933 / 4919 | 90043391515951 | 14 | 13 | 569 | 414893 | 1.679887 |
| 6977 / 6944 | 32078239749697 | 14 | 10 | 251 | 3167 | 1.267876 |

三个余因子案例分别先剥离实际 Stage1 GCD `1764769250254687`、`29599`、`8762386393`。它们不是把原 M 的保存点直接用于另一个模数。完整 N_hex、保存点、命令和剥离记录在历史快照/实验目录中；工作数据库仅保留最优sigma及其阶分解。

正式配对矩阵：生产基线20条 + 新候选20条 = **40/40通过**，所有 raw factors 集合逐项一致；新候选的46个 raw factor 均拆解完成。47条便携真实 Stage2 证据还包含4条早期真实曲线、1条原生复合拆解及2条数据库更新回归。CPU合成 fixture 另列，不混入实际GPU数据集。

必需检查包括 `gmp_selftest_bad=0`、`gmp_check_bad=0`、`pending=0`、`clean=1`。全套证明在 [audit_final](D:/code/MPA-OpenCl/data/ecm_stage2_snapshot_20261005/artifacts.jsonl)；本机可重跑 [audit_ecm_factor_dataset.py](D:/code/MPA-OpenCl/tools/test/audit_ecm_factor_dataset.py:17)。

时间口径：result `seconds` 在可选 GP 拆解之前截止；`factorization_seconds` 单列。引擎 `stage2_full_wall` 的 init/main/total 与进程/驱动秒数不同。CSV 的 `runner_seconds_including_io_and_hashes` 包括进程创建、结果处理、日志写入和指纹核验。这批每形状单次的短曲线不是性能回归结论，也不足以标定大 B1/大B2成本。

## 5. 主程序拆解复合因子

新独立候选支持：

```powershell
ecm_cuda_stage2.exe --save stage1.save --b2 3271 --d 30 --device 1 `
  --factorize-hits --gp D:\AppData\Pari64-2-17-3\gp.exe --factor-timeout 30
```

INI 对应键：`stage2_factorize_hits=1`、`stage2_gp=...`、`stage2_factor_timeout=30`。GP默认用PATH中的 `gp.exe`；建议INI使用绝对路径。超时范围1～600秒，按每个raw factor计。CLI显式GP/timeout优先于INI。缺省不启用该可选CPU工作。

[worker](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:477) 保留旧 `factors`，附加 `param:0`、`factorization_seconds`、`factorization_complete`、去重 `prime_factors`、`factor_analysis` 中逐raw的素数/重数/GP日志。先检查proper divisor，再调用GP factor/isprime，最后GMP重构验证乘积。GP不存在、退出失败或超时保留raw结果并标记 `unresolved`；离线脚本可再次处理。

真实例子：M223 / sigma7 / B1=5 / B2=3271 / D30，关闭因子命名 `NTT_NAME_HITS=0` 后返回 `288431454463` 和 `18287`：

```text
288431454463 = 196687 × 1466449
prime_factors = [1466449, 18287, 196687]
```

所以 importer 必须读取 `factors`，不能用 `hits=0` 推断无因子。另用CPU合成保存点6223=7²×127验证重数和CLI缺失GP覆盖INI的失败恢复，共2/2通过；该fixture不是有效的完成Stage1点，不进入生产曲线语料。

## 6. 从 result 更新更优 sigma

[ingest](D:/code/MPA-OpenCl/tools/ecm_dataset/dataset.py) 逐行读取原生result，验证实际N、PARAM0和proper divisor关系；全M可推断指数，余因子需指定原指数并验证整除关系。拆解raw composite后，为每个proven prime计算候选sigma的群阶/点阶。

[store_best](D:/code/MPA-OpenCl/tools/ecm_dataset/dataset.py:138) 只有新B1、B2都不大于旧值且至少一个严格减小时才替换。相等、变差或一优一劣均保留旧sigma，不存其他候选。比较和替换在同一IMMEDIATE事务中串行化。首次初始化按该候选的B1、B2升序选一对；不代表Auto B2的时间收益排名。

重复导入不增加历史行，不记录导入时间戳或去重键；可能重新计算此前丢弃的sigma。无因子直接跳过；GP分解/分析失败只在本次计数或返回值报告，之后重新导入可重试。仅当前保留的sigma可复用已存阶分解，`--retry`强制重算。最优比较统一使用lcm。

真实原生回归：M223因子196687，先导入sigma6得到 `(B1,B2)=(29,563)`，后导入sigma9得到 `(9,227)`；默认sigma确实更新为9，旧版重复导入第二条新增0/重复1（历史证据）；新版输出updated=0/unchanged，无导入历史表。独立测试库避免既有更优记录掩盖更新。其群阶/点阶分别为 `195924 / 32654` 与 `196128 / 4086`；原生结果、两次数据库状态及去重结果均已归档。

## 7. 可复用命令

在仓库根目录运行，Python只依赖标准库；GP路径按本机调整。

```powershell
python tools/ecm_dataset/ecm_dataset.py init
python tools/ecm_dataset/ecm_dataset.py analyze --exponent 223 --factor 196687 --sigma 6 9 --gp D:\AppData\Pari64-2-17-3\gp.exe
python tools/ecm_dataset/ecm_dataset.py ingest --results run/results.jsonl --exponent 223 --gp D:\AppData\Pari64-2-17-3\gp.exe
python tools/ecm_dataset/ecm_dataset.py summary
python tools/ecm_dataset/ecm_dataset.py export --output run/factors.csv
python tools/ecm_dataset/prepare_dataset.py --output run/orders
python tools/ecm_dataset/run_production_dataset.py --prepare-only --output run/production
python tools/ecm_dataset/run_production_dataset.py --output run/production
python tools/ecm_dataset/run_production_dataset.py --prepare-only --allow-cofactors --exponents 2657 4933 6977 --output run/cofactors
python tools/ecm_dataset/run_production_dataset.py --output run/cofactors
python tools/test/audit_ecm_factor_dataset.py --output run/audit.json
python tools/ecm_dataset/export_dataset.py --output run/snapshot
```

所有CLI的 `--db PATH` 位于subcommand之前；生产runner参数无subcommand。改预算、sigma计划或二进制请使用新输出目录。`--resume`只在runner/plan指纹与二进制一致时继续。runner限制本机GPU1；迁移设备需明确修改并重新记录身份。候选对照需复制基线保存点/plan，再显式 `--stage2 PATH --factorize-hits --gp PATH --resume`。

## 8. NTT tune 的首批可用数据

新候选执行 `--tune ntt --device 1 --length-log2 16:27 --tune-repeats 3 --tune-memory-mb 3072`。12个长度全部测量，0跳过/失败，36个计时样本及12个预热均验证全部L输出，共1,073,479,680个输出word检查。一次iter严格是两forward+融合product/scale/inverse，batch1；不含packing/carry/模N归约/传输，不能直接用1/iter/s估算完整Stage2。

| log2 L | median ms | 卷积iter/s | owned payload MiB |
| --- | --- | --- | --- |
| 16 | 0.140288 | 7128.19 | 1.157 |
| 17 | 0.160768 | 6220.14 | 3.625 |
| 18 | 0.211968 | 4717.69 | 5.625 |
| 19 | 0.373760 | 2675.51 | 9.626 |
| 20 | 0.656384 | 1523.50 | 17.626 |
| 21 | 1.401856 | 713.34 | 36.126 |
| 22 | 3.445760 | 290.21 | 71.126 |
| 23 | 8.701952 | 114.92 | 141.126 |
| 24 | 13.403136 | 74.61 | 262.127 |
| 25 | 26.726400 | 37.42 | 536.131 |
| 26 | 53.460991 | 18.71 | 1048.132 |
| 27 | 111.750145 | 8.95 | 2072.133 |

数据仅限本机sm89/PTX3/outer0选定配置，三样本无置信区间。k16有可见噪声；不同k的配置也有变化，不能简单把该表解释成固定算法的平滑比例。raw events、调度和SHA随快照保存。

另外验证plan-only：M223/B2=122323/D210返回P24/I584/G25且 `curves_executed=0`，不产生result，不改变保存点。tune全部超预算的1MiB例子返回非零，保留partial且不覆盖旧可用profile。该plan仍标记 `calibrated=false`，本例legacy估计36.54s远大于实跑短曲线时间，不能用它自动选择生产B2。

## 9. 构建身份、验证与下一阶段

```text
生产Stage1  5ff1f58a3a072fb37b7ef6e35d3ac2de5488304d6acc5d4f32b6555cb2c96e6e
生产Stage2  893f6e907c17803ed90b09b98ffbf6b85e08b7deb1335a9d6c6efd1f8a01f69d
新候选      b9afdb49ce858bf4cc0811c836e6b275b7ce0b01eee3235f94ca37aaa6890cda
PARI/GP     e8d7ddfb09c5c61bd58f0d645f6cd664253c4eb1f9b532aaa6bd191625b9a3cf
首轮SQLite  ace4c63cf0a1927bb6d1d49d51205fceb732305f50dd20e607ed39e81fb61c18
```

新候选在 `build_cuda_cmake/_factor_dataset_20261005/native/ecm_cuda_stage2.exe`，sm89/PTX3/outer0、24个原始构建依赖冻结；生产893二进制保持。 [HostOnly](D:/code/MPA-OpenCl/tools/build/build_stage2_local.ps1:45) 在CUDA依赖/架构/后端/toolkit和已有object指纹一致时仅重编host并重链接，不是无条件复用旧CUDA对象。

本轮验证：数据库单元5/5；真实更新/去重3/3；CPU前端fixture2/2；plan/失败profile保护2/2；正式GPU矩阵40/40；12长度NTT全部正确；独立数据库/阶/保存点/原生拆解审计通过。

下一阶段按原设计P2推进：按位宽、batch、NTT调用直方图和实际驻留路径测完整操作成本，加Stage1每曲线摊销；用较大B1/B2验证，再联合搜索B2/D/path最大化相对收益K/(T1+T2)。当前有正确性语料与域卷积速度，仍缺无偏收益模型数据、完整phase标定以及实时显存租约，不能给Auto B2一个已验证的通用默认倍率。

## 10. 两表精简迁移（2026-10-05）

用户确认每个因子仅保留一个最优sigma；替换要求B1/B2均不增大且至少一项严格减小。本次迁移前工作库有255分析/495界限，保留其已选33条sigma及全部1155个指数/3322个因子，逐项比较核心列一致后提交，删除其余历史表并VACUUM。

工作数据库由1,167,360B（1140KiB）降到294,912B（288KiB），减少74.7%。一次性旧库备份为 `tools/ecm_dataset/ecm_stage2_dataset.sqlite-before-core-v2.bak`；不会继续写入。历史实验快照不改动。迁移完成时当前库SHA256：`ac9b6bb886e0194a87eb13338b1a7fa4e082ff77f823197b1b53e93cbed39c9e`。

`migrate_dataset.py`提供旧库显式迁移及重复执行保护；CLI统计/分析/导入/导出、批量准备、production runner、独立核心阶审计与现有回归脚本均已适配两表。runner不再向库写运行记录，导出取消--evidence-root，仅输出两表。旧plan的runner指纹发生变化，需对应旧脚本或重新准备。

本轮没有重跑GP算术审计或GPU曲线；保留上文历史验证结果的时间与范围，不把它们当作本轮重构测试。新的使用说明见 [README](D:/code/MPA-OpenCl/tools/ecm_dataset/README.md)。

### 指定区间扫描最优sigma

新增 [scan_sigma.py](D:/code/MPA-OpenCl/tools/ecm_dataset/scan_sigma.py)：`--exponent-range 200:1000 --sigma-range 6:100 --max-factor-digits 17 --gp PATH`。区间为闭区间，也接受单个整数。对数据库中符合范围/因子位数过滤的因子，依次计算候选并沿用同一两项界限改善规则即时提交最佳记录。默认不限制因子位数，超时按每个因子/sigma计；进度和错误计数只输出控制台，不建扫描/历史表。中断已提交记录保留，继续通过指定剩余区间，不新增进度持久化。
