# Stage2 Tensor NTT 长度扫描草稿

## 目标与范围

在当前生产Goldilocks64/PTX归约及outer计划上，比较CUDA tile、64次unsigned-byte MMA原型及40次signed-byte MMA候选。先完成GMP精确校验，再扫描完整变换与三变换卷积的长度/batch，区分tile局部收益和完整NTT收益。保持生产默认及功耗/频率设置；只有独立算子收益和固定D完整曲线证据满足后，才考虑生产部署。

## 候选

把DFT16分解为偶/奇输入的两个DFT8，加一次根乘与合并蝶形。Goldilocks标准8次根的平衡整数代表为±1、±2²⁴、±2⁴⁸及±(2⁴⁰−2⁸)，仅用byte位置0、1、3、5、6，每位系数为−1/0/1。与8个unsigned输入byte平面组合，共40次s8×u8 MMA。根矩阵采用两个8×8对角块；输入按偶/奇重排，输出以r16ʲ重组为原DFT16频谱。

每个输出的signed和绝对值小于2¹¹⁵；按base256逐对角传播有符号carry并补齐128-bit符号。unsigned fold后，负数补加2³²，消去补码引入的2¹²⁸偏置。必须同时覆盖负和、canonical边界、两方向、pointwise/scale融合及真实tile置换，roundtrip单独通过不作为精确性证明。

## 验收和证据

使用GPU1 RTX4060 Laptop、1800MHz设置；不修改设备控制，记录实际设备UUID、频率/功耗遥测和二进制/源码身份。完整变换使用当前生产outer/tile实现作为CUDA对照，候选仅替换tile；计时外填充输入并比较全部输出。分别记录forward、inverse(pointwise与1/L)和卷积，包含launch的CUDA事件计时与host墙钟，不计分配/根生成/校验。

临时开发过程、失败证据和待确认事项保留本草稿，原始数据位于`data/experiments/`；完成后提炼到正式架构/性能说明并提示用户整理草稿。

## 实现与测量结果

完成Tensor40：5个signed root plane、8个unsigned输入plane、signed115-bit上界、128-bit负和修正，以及两个DFT8合并回DFT16。最终SASS为40条IMMA，对照64条；但整个forward/inverse函数的静态指令数增加64/56，寄存器分别62/59，对照61/62。减少MMA条数没有获得稳定完整卷积收益。

新增直接包含生产NTT运行时的完整算子探针，以及冻结身份、GMP门禁、故意损坏、逐输出核对、native/Graph、batch/CTA和图表工具。受限执行启动cl失败的日志保留；使用完整本机环境构建成功，工具链保持CUDA13.3.73/MSVC14.51.36231。最终EXE SHA256=`ef669d8fb7558752e81b342a73012fa93464061fb5a1a9fa2bac3c736e2f6264`，构建13.52秒。

独立GMP门禁10,805,760个word无误，故意翻转后恰检出1错误。正式六组扫描122个shape/operation组、4026个样本，累计60,953,519,616个A/B word检查无误，全部探针持有buffer及FuseBase在结束时live=0。初始开发扫描另存`initial/`，不进入正式汇总。CTA256为独立GMP门禁，CTA128/512为与生产CUDA的逐字比较。

全部扫描保持用户1800MHz上限/默认55W，未改变任何显卡设置。利用率≥50%的SM实测1455–1800MHz、中位1665MHz，说明不能将本轮称为恒定1800MHz测试。默认功耗条件下的时间可用于本轮交错比较，不能直接与旧固定2400MHz数据或硬件周期互换。原始空闲NVML异常功耗样本保留。

正式结论写入`docs/performance/NTT_TENSOR.md`，算法推导放`docs/reference/TENSOR_NTT_MATH.md`，构建/复现入口放`tools/bench/README_NTT_TENSOR_LENGTHS.md`。生产源码、INI、tune表与D模型没有修改，未发布实验程序或运行完整Stage2曲线。Graph候选仍需完整请求生命周期和固定D曲线验收；本草稿保留，完成后请用户决定整理方式。
