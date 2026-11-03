# 整数NTT／CGBN交叉点测量

`bench_cgbn_ntt_multiply.py`对同一组GPU驻留整数比较完整乘积及普通Montgomery模乘；`analyze_cgbn_ntt_multiply.py`审计独立复测、汇总和绘图。设备设置只读，使用已有进程白名单。当前数据和适用范围以`docs/performance/CGBN_NTT_MULTIPLY.md`为准。

## 构建

正式实验入口为`tools/build/test/build_cgbn_ntt_multiply_probe.ps1`，Bits选择实验容器，Tpi为8／16／32。先独立编译一套生产NTT模块，例如`-Build build_dev/mul8192_16 -Bits 8192 -Tpi 16`；后续位宽用`-ModuleBuild build_dev/mul8192_16`复用。生产可执行文件的支持范围不因探针扩大而变化。

模块复用校验源码依赖、模块清单、固定NTT ABI、公共defines、GMP及对象哈希；位宽／TPI是仅用于本探针的LocalDefines。本探针使用legacy default stream；ABI schema 1匹配所选stream与架构，并约定GL3／add-sub1／outer-unroll0安排，修改这些编译约定时应更新schema。也可显式给出完整且仍通过依赖验证的WorkerBuild；过期构建会被拒绝，不能用旧exe替代当前对象。

构建目录先由共享构建辅助生成内部`.gitignore`。编译与正式性能采集串行。所有构建身份、对象与源文件进入manifest；采集器把exe、DLL、清单、依赖及自身快照冻结到输出目录。

## 采集

采集器必填`--build DIR... --output NEW_DIR --suite NAME --process-policy APPROVED_JSON`。默认GPU1；仅使用指定UUID的进程门禁，不操作其他显卡。输出目录须不存在，失败证据原样保留。

- `smoke`：最小可用性检查，不作性能资格。
- `check`：随机／边界、单整数与尾组、一次及多次递推、直接／Graph、短tile及高半故障注入。
- `screen`：相同位宽／批量／运算比较CGBN TPI与NTT Graph tile。默认批量1／192／1536。原始首样本保留，试选不作为最终收益。
- `final --selection SCREEN/summary.json`：冻结每格的筛选优选；三个独立进程轮次、每进程5个event样本。直接NTT复用Graph选中的布局，用于观察提交开销，不称为独立调过的直接布局。
- `sanitizer --sanitizer PATH --sanitizer-tool memcheck|synccheck`：检查结果不进入正常性能统计。Windows仅给工具子进程指定命名管道，不修改防火墙或全局环境。

完整乘积递推将高低半XOR反馈，最后输出两半及反馈状态；模乘维持原R并返回唯一规范代表。NTT包含紧凑打包、两次forward、含pointwise的inverse、精确进位、解包，以及模乘时的原CGBN REDC。虽然依赖递推保持第二操作数不变，NTT每次仍重新变换两个操作数，未启用固定操作数频谱缓存；该表只评价这一通用乘法安排。

事件排除分配、表／Graph构建、初始输入拷贝、最终回读及GMP；这些准备成本另记，`graph_seconds`已包含在`setup_seconds`，不能重复相加。每个event样本后逐词比较GMP，检查输入不可变、保护区、反馈及NTT超出2s位的零尾；这些验算不进入event区间。

## 汇总

分析器接收`--evidence STUDY --output SUMMARY --figures FIGURES`。补充位宽可用`--extra STUDY...`，每个额外目录需有完成的`check_matrix`和独立`final`；不改写旧原始记录。图中阴影为三个进程中位数的最小／最大值，不是置信区间。单整数延迟、固定批量吞吐、完整乘积和模乘分别报告；不以单一阈值外推未测位宽或生产曲线。
