# CGBN专用平方开发草稿

目标是在4060 Laptop验证共享协作平方，供CUDA Stage1 Montgomery／Edwards及Stage2协作点路径评估。当前生产默认不变，不改变曲线身份、表示或checkpoint。

## 设计与验收

上三角＋对角线减少完整乘积的部分积；普通Montgomery归约先复用CGBN。每线程配对低列k与高列k+n，平衡三角形工作量；共享输入每32词偏移一词，避免固定步长的bank别名。96位列累加器、本地列进位、跨组双半区前缀分开验证。容器有padding时，完整乘积仍按公开BITS拆分，而Montgomery保留CGBN的UNPADDED_BITS对应radix。

先做完整宽平方与模平方的GMP对照：0／1／N−1、全1、交替位、单bit、随机、padding、TPI及别名。对照三种模平方：原融合mont_sqr、通用宽平方＋原REDC、专用宽平方＋原REDC。分别记录批量吞吐、依赖递推、寄存器／local／shared；算子收益不当作完整Stage1收益。

小批量和原生批量通过后，再决定接入Stage1实验路径。每一步保留失败、首次样本、构建输入及二进制身份。GPU0不测试，不更改任一卡设置。临时草稿保留，完成后由用户处理。

## 实验实施与保留边界

完成了三种独立组织：共享输入的对称列累加、寄存器内有限深度分块、三个半宽平方的有限深度Karatsuba。共同探针直接编译真实头文件，不从生产源提取临时内核；构建脚本记录对象、依赖、编译器和二进制身份。

对称列累加先使用两个累加器分别接收对角线与交叉项，随后改为单累加器、记录前缀、后缀作差；8192/TPI16原生批量的模平方从约5.136 ms降到3.678 ms，仍慢于原融合CGBN的约1.65 ms。这些早期构建使用MAX_ROTATION4，不能与后续MAX_ROTATION1构建直接合并。管理员NCU还观察到该列内核的大量共享bank冲突；这只能说明具体映射问题，不能据此否定所有三角形平方。

有限深度Karatsuba复用三个半宽平方，并通过绝对差避免放大求和容器。8192/TPI16、MAX_ROTATION1、384整数×32次递推，原融合Montgomery约1.641 ms，Karatsuba深度1／2为2.108／2.611 ms，GMP通过但性能不合格。该方案与列方案留在`tools/experiments/cuda/cgbn_square/`，不进入生产依赖。

共享原语选择有限深度分块，位于`src/ecm/cuda/arithmetic/cgbn/square.cuh`，请求深度2，TPI8实际深度1。它使用两个半线程组并行平方及完整线程组交叉乘法，配套原REDC；普通Montgomery仍回退性能验收，未接入Stage1或Stage2默认。当前结果、图表和后续条件以[性能说明](../performance/CGBN_SQUARE.md)为准。

## 微基准修正

初版宽平方递推只反馈低半，末次才导出高半；编译器可消除未使用的中间高半。保留`split_*`、`kara_*`及`final_*`内的早期原始记录，但不再用其中的宽平方数字证明完整乘积性能。普通／梅森模平方每一步均消费两半，因此其结果有效。

修正后的`raw_*`将高、低半XOR后反馈，GMP镜像同样递推；额外高半故障注入必须被检测到。两侧都包含线性XOR成本，仍不称为零附加成本的纯宽平方。正式图表只采用该修正后的宽平方及有效模平方样本，不混用两类宽平方计时。

已保留初次构建失败日志：GMP头包含、scratch尺寸常量的host/device注解，以及非依赖类型在host解析阶段的高半翻bit表达式。后者改用公开CGBN置位／XOR接口。没有通过放宽算术门限处理失败；故障注入的非零退出是预期检查结果。

用户确认Compute Sanitizer运行期间弹出`cgbn_square_probe.exe`防火墙提示。工具支持TCP与Windows命名管道，时间点与本地工具通信一致；未取得当时socket事件，不能把推断写成确证。采集器增加sanitizer入口，Windows显式使用命名管道，memcheck／synccheck复核均0错误；未配置防火墙规则。保留一次性检查脚本的context-manager接口错误目录，不计作GPU算术失败。

## 交付与后续

研究范围为GPU1 4060 Laptop，所有原始样本保存在`data/experiments/cgbn_square_20261102/`；未调整GPU设置，未测试GPU0。正式状态和数学分别写入`docs/performance/CGBN_SQUARE.md`及`docs/reference/CGBN_SQUARE.md`。本草稿保留试验取舍及失败边界，完成后提示用户处理，不自动删除。
