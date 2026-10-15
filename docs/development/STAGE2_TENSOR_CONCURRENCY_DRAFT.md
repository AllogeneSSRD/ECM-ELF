# 单卡CUDA/Tensor NTT并发草稿

## 授权与目标

用户要求测试单卡Tensor Core与CUDA并行效率，并按Tensor待推进列表继续优化。保持GPU1默认55W/1800MHz上限；GPU0现有任务保持运行。比较同一context中的纯CUDA、纯Tensor40、混合serial/parallel及窄重构候选；采用私有输入与只读缓存根表，不引入生产并发共享状态。

## 设计

两条nonblocking工作流及一条控制流用事件fork/join；同一时刻两任务有独立A/B，各自含两次forward和一次pointwise/scale融合inverse。控制组保留串行、单任务和Graph/native提交。主要指标为全部任务的GPU makespan与同数学工作量的Conv/s，不把两个kernel时间相加当墙钟；以最优纯CUDA控制为比较对象，不能只和较慢的混合串行比较。

新候选利用2⁶⁴≡2³²−1及2⁹⁶≡−1，将40次MMA的signed byte对角和直接折到U+2³²V。设S=c₀+2³²c₁+2⁶⁴c₂+2⁹⁶c₃，则U=c₀−c₂−c₃、V=c₁+c₂。将V分成low32和signed high32后，只需canonical(U+high32(V)·(2³²−1))与low32(V)·2³²相加。所有中间量按signed64边界核对，避免对负值左移；不复原128-bit和，不逐byte传播carry。是否减少指令/寄存器与完整NTT时间须实测。

## 验收

完整GMP forward/inverse/卷积及负和、q−1、边界输入，独立故障注入；并发完整输出与独立GMP小形状核对。正式每样本全部A/B与生产CUDA逐字比较，记录所有样本与释放；采集Systems确认事件重叠，NCU不以逐kernel重放证明并发。只在完整Stage2固定D对照支持后考虑生产部署。

原始证据放data/experiments，正式当前结论更新Tensor性能主题，数学证明更新固定资料；本草稿保留供用户整理。

## 实施与候选记录

1. 首个直接U/V累加候选通过完整GMP，但forward由62增至66 registers且部分形状更慢。冻结证据在initial，不作为最终候选。
2. 96-bit低项加signed高项候选避免构造128-bit补码。未约束版本仍有66-register forward，CTA512退到1块；保留n96_initial及graph矩阵。
3. 独立512线程/2块的N96r64入口将正逆均限制到64 registers，无stack/spill。CUDA和Tensor40正逆tile的SASS指令序列与约束前逐函数一致。Tensor40→N96r64静态指令1504→1416、1360→1288，MMA仍40。
4. 最终EXE SHA256 ebc499d64d4433e1390782e360d36ffe207f65d6f68c99d18813723f35df45ac。GMP14407680 word、并发小形状1113600 word无误；故障注入bad=1。Graph19形状/native5形状、13scene、各11次，共3432正式样本、53758735360个最终A/B word无误。
5. Systems4个范围确认实际重叠，串行控制0%，中等并行约91.26%，生产常见大batch约17.16%。NCU使用application-range，CC/CT40/CN96r64各9次应用重放，全部输出检查通过。范围报告GPU为4060 Laptop/24SM，时钟/缓存控制均none；不把插桩秒数混入正式计时。
6. NCU参数与筛选失败记录保留：graph-profiling graph不与range合用，profile-from-start也不能用于range；本机devices=1范围筛选产生空报告。改由应用选择GPU、按报告名称/SM核对，并使用app-range后采集成功。最初CC仅保留最后一次输出，另采cc_checked补齐全部9次；最终汇总只使用完整三组。

## 当前结论与交付

最终19个Graph形状均未超过最优纯CUDA同工作量控制。N96r64相对Tensor40多数差异小于2%，减少静态指令和修复资源拐点未形成生产部署证据。k21–23 outer组织和准备/同步边界仍更值得优先；进一步Tensor需削减整数重构/根乘等共享资源开销。

正式说明已更新Tensor性能、数学固定资料、架构与TODO，复现入口为tools/bench/README_NTT_TENSOR_CONCURRENCY.md。未修改生产CUDA内核、INI、D/tune选型或外部生产目录；未将负收益候选接入完整Stage2。草稿保留，完成后请用户决定整理。
