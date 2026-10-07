# Stage1 param0：输出不别名的 xADD 实验（2026-10-07）

## 1. 结论与实验身份

把单点 PRAC 链的输出 `T.x` 用作 xADD 中间值 V，数学上可将函数内部临时大整数从三个降到两个。本机编译器已经完成相同的寄存器复用：新旧 MODE12/13 的**完整 SASS 指令及调度编码完全相同**，寄存器、stack、spill、静态 text 都未改变。本轮不宣称吞吐改善，也没有重新执行性能矩阵或 NCU 重放。

仅演进显式 `single-compact` 候选，默认算法、TPI、TPB、寄存器策略、切片目标保持当前配置。旧 `single-add` MODE10/11 完整 SASS 同样不变。

- 基线 commit：`8eda457`，三临时量 ADD + compact DBL，称为 v1。
- v1 exe SHA256：`6e07c486405627c6c1d901d6224cc96aa3f4942d895ff789a26746ccd5073d41`。
- 本轮输出不别名 ADD，称为 v2；沿用 `ECM_PRAC_VARIANT=single-compact`，没有新增运行时枚举。
- v2 exe SHA256：`bcfa5aa8b6b5da23f218619b2a897241a2676fc897dd5fc310d0e6a29c7b45d0`。
- 专用 body SHA256：`654b5fb3a9fa0468d83bdb73dbfe670f0d433980ae20ae99d6ae44c66d74bd9f`。
- 公共算术头 SHA256：`67dc71128495378991dd750d73f7d8226ebedd255e267abd435dc6331a03e0a6`，未改动。

同一选项跨二进制代表不同候选版本，复现必须同时记录 SHA；本报告的数据不与此前版本混合。

## 2. 改动、约束及计算量

源码：[私有 ADD 与调用点](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cuh:9)，本轮提交中调用点在第67行。专用 TU 位于 [候选 dispatch](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cu:10)。后续修改可能移动行号，应结合本轮 commit 读取。

原 ADD：`t,u,v` 三个内部 bn；本轮：`t,u`，并提前写独立输出 `ox` 保存 V。步骤为：

1. `t=X1+Z1`，`u=X2−Z2`，`u=u*t`。
2. `t=X1−Z1`，`ox=X2+Z2`，`ox=ox*t`。
3. `t=u+ox`，`u=u−ox`，分别平方。
4. `ox=Zd*t`，`u=Xd*u`，`oz=u`。

所有加减、Montgomery 乘法/平方仍做原有归一化，保持每次 xADD **4M+2S**，DBL **3M+2S**。CGBN 此实现的平方使用同类 Montgomery 乘法路径，因此按模乘等价量 ADD=6、DBL=5。给定计划 A 次 ADD、D 次 DBL，工作量仍为 `6A+5D`；没有减少依赖模乘。

严格约束：`ox`、`oz` 必须彼此不同，且均不与点坐标、差分点、模数等任一输入别名。唯一调用点使用新建 `tx,tz`，与 A/B/C 和 N 的固定 bn 分离；没有动态 bn 指针、取地址或设备调用 ABI。公共 alias-safe `prac_add` 保持原实现，`single-add` 继续使用公共 ADD。

容器4608/TPI16，每个完整 bn 的逻辑值为576 bytes，平均每线程9个32-bit limb。本轮减少的是**源码内部临时变量**，不能把576 bytes或9 registers直接解释为实际显存/寄存器节省。编译结果显示实际分配没有减少。

设备曲线、seed、计划、窗口输出及切片边界大小都未改变；没有新增 host/device 传输，也没有减少 PCIe 数据量。此前显式数组和切片公式见 [v1报告§2](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_SINGLE_COMPACT_20261006.md:15)。

## 3. 假设与静态反馈

实验前的预测：

1. 若第三临时 bn 扩大了实际存活集合，输出复用应降低寄存器/溢出，争取168分配档。
2. 若编译器已经将 V 与输出合并，机器码应完全不变。
3. 若提前写输出延长存活区间，寄存器或 spill 可能增加。

结果支持第2项。四个候选 entry 的完整函数文本比较包括地址、指令编码及单独的调度编码行，修剪函数尾部空行；没有只比较 opcode 或指令总数。

- MODE10：72,306行一致；36,152条指令，578,432 bytes text；170 registers，48 bytes stack，spill store/load=0/0。
- MODE11：71,922行一致；35,960条指令，575,360 bytes text；168 registers，96 bytes stack，spill=56/44 bytes。
- MODE12：72,258行一致；36,128条指令，578,048 bytes text；170 registers，48 bytes stack，spill=0/0。
- MODE13：72,002行一致；36,000条指令，576,000 bytes text；168 registers，88 bytes stack，spill=48/40 bytes。

未受影响的正常 TPI16、host、dispatcher 三个对象文件 SHA 与构建前快照相同。这里只证明列出的函数/对象，不扩大为整个程序全部机器码的证明。可执行文件 SHA 不同不能说明算术 kernel 有变化。

资源记录是静态 ptxas 数据，text 是静态指令跨度，不是动态执行条数或指令缓存工作集。GPU1 TPB128 下自然策略仍170实际使用/176硬件分配、最多2 blocks/SM；cap168仍3。后一驻留结论来自 v1 的 NCU，v2机器码一致，本轮未新采集 NCU。

本地证据：`docs/data/stage1_disjoint_add_sass_20261007/{all.sass,resources.txt,summary.json,identity.json}`。构建前快照：`build_cuda_cmake/prac/before_disjoint_add_20261007/`。

## 4. 验证

### 4.1 CPU 模型

[Montgomery/别名模型](D:/code/MPA-OpenCl/tools/test/test_prac_disjoint_add.py:1)：六个模数/容器（101/16、1000036000099/64、M127/160、M2203/2560、M4423/4608、M8191/9216），每组6个边界+128个固定随机输入，共804个正例；公共 ADD 的六种输出别名安排共4,824项通过。故意让新 ADD 输出覆盖差分点，每组131/134项错误，共786项，证明不能推广其别名契约。

正例共3,216乘法、1,608平方、1,166加法修正、1,180减法修正和16次 REDC 修正。四个≥127bit的梅森配置，Montgomery 表示的−1平方均有 raw REDC≥N；即便 R≫N，也不能据此省略归一化。

[链角色模型](D:/code/MPA-OpenCl/tools/test/test_prac_single_add.py:1)：8,533个prime/d组合×两个sigma=17,066数学输入链；baseline、single-add、v1、v2四模型共68,264求值。全部中间/终结 XZ、ADD/DBL计数和独立 ladder 对照一致；共264,538规则步骤，四类规则和终结均覆盖。

CPU 模型不是原生 GPU 正确性的证明。证据：`stage1_disjoint_add_montgomery_20261007.json`、`stage1_disjoint_add_roles_20261007.json`。

### 4.2 原生 GPU 与文件语义

同一 v2 SHA、GPU1/N4423/TPI16：

- natural/register255：184条完整 Q/save 对照；cap168/target50ms/C384：3,568条完整 Q/save 对照，两个脚本各27类案例正常结束。
- 恢复窗口门禁：1,352条窗口 Q、8条 checkpoint 恢复 Q、20个非法配置拒绝。
- B1=10m/260m：prefix/middle/tail，count32，chunk7/32；以独立CPU子乘积 oracle逐点核验。oracle缓存184 miss/1,168 hit，每份 GPU 输出仍比较。
- 只读完整 X/Z CSV 逐字节审计128项，包括48项跨切片比较；后者包含在128中，不重复相加。

原始窗口门禁169条结果全部完成，`passed=true`；跨策略字节审计`passed=true`。这些小批次的日志投影不作为生产吞吐数据。

本地目录：`stage1_disjoint_add_natural_q_gate_20261007`、`stage1_disjoint_add_cap168_q_gate_20261007`、`stage1_disjoint_add_windows_q_gate_20261007`，均在忽略的 `docs/data/` 中。

## 5. 构建及复现

只重编专用 `.cu`：实测197.8秒，随后 CMake 仅链接。上一轮迁移 body 后，单 TU 迭代时间首次得到独立实测；此次没有重编其他六个 CUDA TU。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/parallel_nvcc.ps1 `
  -BuildDir build_cuda_cmake/prac -Only 'cgbn_stage1_prac_single_add\.cu$' -Jobs 6

python tools/test/test_prac_disjoint_add.py --output docs/data/disjoint_model.json
python tools/test/test_prac_single_add.py --output docs/data/disjoint_roles.json
python tools/test/test_cuda_prac.py --bits 4423 --tpi 16 --registers 255 `
  --variant single-compact --device 1 --output docs/data/disjoint_q_natural
python tools/test/test_cuda_prac.py --bits 4423 --tpi 16 --registers 168 `
  --variant single-compact --target-ms 50 --curves 384 --device 1 `
  --output docs/data/disjoint_q_cap168
python tools/test/test_cuda_prac_windows.py --single-add --single-compact --production `
  --production-counts 32 --production-chunks 7 32 --device 1 --output docs/data/disjoint_windows
python tools/test/audit_cuda_prac_windows.py --input docs/data/disjoint_windows/summary.json `
  --output docs/data/disjoint_windows/bitwise_audit.json
```

## 6. 下一项及采用决定

保留为私有显式实验候选，记录“源码临时量减少、机器码完全相同”，不提升默认/最佳吞吐推荐。此前最佳 baseline cap168/50ms 的5.060764/132.429740 s/curve，是 **v1批次的短时投影**，没有改写为 v2 实测。

下一项尝试把 `finish` 与 `rule` 合并为终结哨兵，缩短横跨长 ADD 的标量存活区间。先检查完整 SASS、寄存器及 spill；若机器码改变且有合理收益路径，再执行生产规模的短片/长片和普通前缀 A/B。保持公共算术、计划、归一化和默认配置不变，每次仅改一个因素。
