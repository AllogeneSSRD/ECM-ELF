# Stage1 param0：私有 xADD 输出 Z 复用实验

私有xADD内部临时bn由两个减为一个，新旧七个候选entry完整机器码和资源相同，原生GPU输出/文件语义门禁通过。本轮没有新增吞吐改善；下一项检查Montgomery常数传播。

## 1. 范围与版本

基线 `d126e60`，二进制与上一轮批量实验相同：exe SHA256 `d1a5b6476c17c95762283fd0d30205e5175b3d37ccf49198fa2a99c2476d7370`，候选对象 SHA `1319552d7cae465bc1a90f9c37aa2a12cfe64f3434908a4f67bd4ca24e6b3c7d`。构建前核对既有只读快照11项输入、40项源码/工具/实际CGBN SHA，以及两份完整SASS清单。

本轮只修改[私有 ADD](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cuh:9)及[独立 Montgomery/别名模型](D:/code/MPA-OpenCl/tools/test/test_prac_disjoint_add.py:78)。single-compact 的TPI16 MODE12/13/14和显式TPI32 MODE12/13使用该ADD；旧single-add MODE10/11仍使用公共alias-safe ADD。默认TPI、TPB128、容器、计划与窗口/生产切片语义不改。

GPU1为RTX4060 Laptop、24SM、sm89。实际编译命令为compute_89/sm_89；不能单独以CMakeCache的通用CMAKE_CUDA_ARCHITECTURES字段判断最终对象架构。上一轮该机N4423/4608/TPI16/single-compact/cap168/C2304/50ms是已测最佳显式配置，短时投影4.960886/129.855046 s/curve，不能改写为本轮实测。

## 2. 计算、存活区间与别名约束

旧布局内部有 `t,u` 两个bn，输出 `ox` 提前保存V，`oz`在末尾由u复制。新布局内部仅有t，输出oz从第一次差值开始保存U，并承担其后减法、平方及最终乘法：

1. `t=X1+Z1`，`oz=X2−Z2`，`oz=Mont(oz,t)`。
2. `t=X1−Z1`，`ox=X2+Z2`，`ox=Mont(ox,t)`。
3. `t=oz+ox`，`oz=oz−ox`；分别Montgomery平方。
4. `ox=Mont(Zd,t)`，`oz=Mont(Xd,oz)`。

所有加减、乘法和平方保留归一化，每次ADD仍4M+2S，DBL仍3M+2S。按CGBN当前平方路径的模乘等价量，计划工作量仍 `W=6A+5D`，batch为 `C*W`。本轮不减少依赖MAC，不以删去最终cgbn_set直接推断动态指令减少。

严格要求两个输出彼此分离，并与全部输入（含差分点与模数）分离。调用点[共享DBL链](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cuh:65)使用新建tx/tz；公共ADD仍支持覆盖输入，不可以替换为该私有实现。

源码少一个bn对应4608容器576逻辑bytes；TPI16每线程9个limb，显式TPI32每线程5个limb、160分组槽位。该源码差值不是实际寄存器、stack或显存节省；应以编译资源及完整机器码判断。

显式曲线data仍 `7*C*4608/8` bytes，固定窗口seed同样大小，普通前缀无window seed。GPU计划仍 `16P` bytes，切片边界逻辑量仍 `6*C*4608/8*launches`，没有改变host/device传输与显存数组容量。

## 3. 假设与反馈方式

1. 若局部U扩大实际存活集合，输出Z复用应降低寄存器/stack/spill或指令搬运。
2. 若编译器已合并U与最终Z，完整SASS与资源应相同。
3. 若提前写输出扰动调度，资源或吞吐可能回退。

先重编单一候选TU并逐函数比较完整SASS（含指令和调度编码），再决定生产B1采样。静态指令数量相同不等于完整机器码相同；exe SHA不同也不等于算术内核发生性能变化。

## 4. CPU验证

六组模数/容器：101/16、1000036000099/64、M127/160、M2203/2560、M4423/4608、M8191/9216，每组6个边界+128个固定随机向量，共804组。独立模拟新布局每次早写输出，新旧布局结果、4M+2S及所有归一化修正计数均一致。

公共alias-safe ADD六种输出安排共4824项通过。故意把私有输出覆盖差分点时，新旧布局均786组错误，证明不能放宽契约。四个较大梅森配置的Montgomery −1平方raw REDC≥N，继续需要归一化。新模型为独立整数/存储别名模拟，不是原生GPU证明。

链角色模型17066条数学输入链、102396模型求值、264538规则步骤通过，四规则及终结覆盖。链模型仍使用已有disjoint算术公式；本轮新布局的早写输出由新增Montgomery模型检查，不将旧链模型说成额外804个原生输出复用验证。

## 5. 构建与静态结果

使用tools/build/parallel_nvcc.ps1只重编 `cgbn_stage1_prac_single_add.cu`，实测286.6秒，CMake随后仅链接，没有重编其他CUDA或host TU。新exe SHA256为 `e1f547f2c203679be3f0ddf762c04b85e5f2d643fdc9dbd7ce80bedee169c048`，候选对象SHA为 `b2d90fba5efd349a15dc0489a32ea1ee228177a413531a51a8b76b04481bf89b`，私有body SHA为 `93dbf512e0abf3737177a3b43e9c3417c2bbfcca9e06d76fe291574fd029eee1`。

七个候选entry的**完整函数文本（含指令及调度编码）与资源均相同**。这支持编译器已将U与输出Z合并的假设，不证明整个可执行文件每一部分完全一致。正常TPI16、host、dispatcher对象，以及公共算术头、compile_commands和CMakeCache SHA与构建前快照相同。

| TPI / MODE | 完整SASS比较行数 | 静态指令 | text bytes | 实际寄存器 | stack bytes | ptxas spill store/load bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 / 10（旧single-add） | 72307 | 36152 | 578432 | 170 | 48 | 0 / 0 |
| 16 / 11（旧single-add cap168） | 71923 | 35960 | 575360 | 168 | 96 | 56 / 44 |
| 16 / 12（single-compact natural） | 55507 | 27752 | 444032 | 161 | 48 | 0 / 0 |
| 16 / 13（single-compact cap168） | 55571 | 27784 | 444544 | 162 | 48 | 0 / 0 |
| 16 / 14（single-compact cap128） | 56499 | 28248 | 451968 | 128 | 208 | 808 / 972 |
| 32 / 12（single-compact natural） | 29059 | 14528 | 232448 | 111 | 64 | 0 / 0 |
| 32 / 13（single-compact cap168） | 28147 | 14072 | 225152 | 109 | 64 | 0 / 0 |

上表前后相同，text与指令数是静态跨度和静态条数，不是退休指令、周期或I-cache工作集；ptxas spill字节不是累计流量或显存容量。SASS比较仅规范化行尾和函数尾部空行，没有仅比较opcode数量。

因此本轮没有新增生产性能矩阵或NCU采集，不宣称吞吐改善。上一轮16/C2304/cap168/50ms的投影和资源数据仍属于上一轮计时/采集，不改写成新二进制实测。

## 6. 原生GPU与文件语义

五组显式策略的完整Q/save门禁均完成，每组27类案例，含lcm/choose12、64-bit sigma、正常checkpoint恢复、损坏checkpoint/计划重建、proper factor/退化点/目标边界。下表总Q包含ladder/resident/public边界对照；候选列只数实际修改候选的主案例与恢复，不将总数全部归因于新ADD。

| TPI / 寄存器策略 / 主批量 | 总Q | 候选主案例与恢复Q |
| --- | ---: | ---: |
| 16 / 255 / 8 | 184 | 56 |
| 16 / 168 / 384 | 3568 | 1184 |
| 16 / 128 / 384 | 3568 | 1184 |
| 32 / 255 / 8 | 184 | 56 |
| 32 / 168 / 192 | 1840 | 608 |
| 合计 | **9344** | **3088** |

子乘积窗口188结果、1504 Q、16恢复Q、24非法配置拒绝均通过。生产范围只核对B1=10m/260m的prefix/middle/tail、count32、chunk7/32，逐点与独立CPU Montgomery ladder比较；不完成生产大B1整条曲线，不把小批量C8窗口投影用于生产吞吐。

完整X/Z CSV检查：140项同TPI跨策略逐字节，其中54项跨切片；49项跨TPI plain XZ逐字节；全部188个新窗口与冻结基线同输入的完整CSV也逐字节一致。这些检查覆盖同一批窗口，有重叠，不相加作为独立测试数量。CPU子乘积oracle缓存184 miss/1320 hit，每份GPU输出仍独立比较。

计时矩阵和NCU不重新采集，故本轮结论是源码复用成立、机器码不变、原生输出/文件语义门禁通过；没有新增吞吐收益结论。

## 7. 采用与下一项

本轮否定了“源码少一个临时bn即可继续降寄存器”的预测。保留明确的私有输出复用和独立别名模型，不改变默认/最佳策略推荐；是否进入生产仍依赖已验证的显式候选范围。长期目标仍是减少Stage1耗时和提高曲线吞吐。

### 7.1 下一项：Montgomery常数传播

优先尝试显式、严格host guard的梅森模数候选，从真实算术核心检查 `np0=1` 与已知N limb能否减少实际指令/寄存器。继续保持Montgomery R、归一化、导出与checkpoint数据语义；不同N或cofactor不能静默使用固定N4423常量。

证据与边界：

- [np0生成](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:330)计算 `−N^−1 mod 2^32`；对于 `N=2^k−1,k>=32`，N低word为0xffffffff，因此np0=1。
- 当前实际sm89后端是WMAD。[q生成](D:/code/MPA-OpenCl/cgbn/include/cgbn/core/core_mont_wmad.cu:75)及第二半循环第114行使用 `shuffle(accumulator)*np0`。两个q乘法每组两word，源码求值次数为每线程 `S=TPI*L`、每curve `TPI*S`、完整计划 `TPI*S*W`；不是退休SASS数量或周期，也不据此直接预估速度。
- [CGBN load](D:/code/MPA-OpenCl/cgbn/include/cgbn/impl_cuda.cu:1376)的真实索引为 `lane*LIMBS+limb`，每线程持有连续word。不能使用 `limb*TPI+lane` 生成常量。
- N4423的word0..137为0xffffffff，word138为0x7f，word139及以上为0。4608/TPI16的L=9：lane0..14的9个word全为0xffffffff；lane15为 `[ffffffff,ffffffff,ffffffff,7f,0,0,0,0,0]`。TPI32/L=5则lane27含末尾非零word，lane28..31为零，并保留144..159的padding零值。该映射必须逐word与通用load对照。
- 已做整数shape核对：M2203/2560/TPI16、M4423/4608/TPI16和TPI32、M8191/9216/TPI32逐word重建准确，np0均为1；同位宽的N−2不满足精确梅森guard。这是下一项的布局证据，尚未实现常量模数CUDA内核。
- 初始化常量时可按lane设置值；CGBN运算中的shuffle要求同组线程正确参与，不能把后续完整算术按lane分成不一致分支。
- [内联PTX MAD](D:/code/MPA-OpenCl/cgbn/include/cgbn/arith/asm.cu:68)使用volatile asm及寄存器约束。传入常量并不自动证明Q*N链变为更少机器指令；先检查新编译结果，再决定短时生产A/B。

若采用新域/新容器则必须另建checkpoint契约，本项计划先保持原域与容器。最初以GPU1/N4423/TPI16/C1536及C2304作同机反序对照；需要TPI32性能比较时仍用一半曲线匹配提交grid。

## 8. 复现

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/parallel_nvcc.ps1 `
  -BuildDir build_cuda_cmake/prac -Only 'cgbn_stage1_prac_single_add\.cu$' -Jobs 6
python tools/test/test_prac_disjoint_add.py --output docs/data/outputz_model.json
python tools/test/test_prac_single_add.py --output docs/data/outputz_roles.json
python tools/test/test_cuda_prac.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --bits 4423 --tpi 16 --registers 168 --variant single-compact `
  --curves 384 --target-ms 50 --device 1 --output docs/data/outputz_fullq
python tools/test/test_cuda_prac_windows.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --single-compact --single-compact-tpi32 --single-compact-reg128 --production `
  --production-counts 32 --production-chunks 7 32 --device 1 --output docs/data/outputz_windows
python tools/test/audit_cuda_prac_windows.py --input docs/data/outputz_windows/summary.json `
  --output docs/data/outputz_windows/bitwise_audit.json
```

复现需要空的新输出目录。原始证据置于已排除的docs/data，不提交运行数据。

## 9. 原始证据

- `stage1_add_outputz_before_20261007.json`：基线commit、既有只读快照、11项输入和40项源/工具/CGBN SHA。
- `stage1_add_outputz_build_20261007.log`：单TU286.6秒及最终仅链接；ptxas日志在build_cuda_cmake/prac/par_nvcc。
- `stage1_add_outputz_sass{16,32}_20261007/`、`stage1_add_outputz_compare{16,32}_20261007.json`：完整SASS与资源逐函数比较、导出SHA。
- `stage1_add_outputz_montgomery_20261007.json`、`stage1_add_outputz_roles_20261007.json`：CPU模型与原始计数。
- `stage1_add_outputz_t{16,32}_reg{255,168,128}_q_gate_20261007/`：实际五种完整Q配置；TPI32不支持cap128，不存在该组合目录。
- `stage1_add_outputz_windows_q_gate_20261007/`、`stage1_add_outputz_gates_audit_20261007.json`：完整门禁、窗口/恢复/拒绝结果与同TPI逐字节审计。
- `stage1_add_outputz_before_after_xz_20261007.json`、`stage1_add_outputz_cross_tpi_xz_20261007.json`：188项前后版本、49项跨TPI原始完整XZ文件SHA。
- `stage1_mersenne_constant_shape_20261007.json`：下一项的CGBN索引/np0/梅森guard整数shape证据，非已实现CUDA候选。
- `stage1_add_outputz_final_audit_20261007.json`：源码变化范围、静态证据、原始输出SHA与门禁汇总核对。
