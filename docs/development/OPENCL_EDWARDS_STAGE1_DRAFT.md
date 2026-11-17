# OpenCL Edwards Stage1 开发草稿

入口目标是 1024 bit 内的小模数。旧 PARAM3 私有路径可作为 OpenCL 调度参照。PARAM3 与 Edwards 应各自固定后端之间的 $N,B_1,\sigma$ 和批量口径。Edwards 还需固定 NAF 窗口。

2026 年 4–6 月的小位宽成绩以原版 GMP-ECM 的 GPU PARAM3 为参照。当时开发使用 OpenCL。此轮以 ECM-ELF 自有的 CUDA CGBN 为参照。据维护者的历史比较，CGBN 此后有约 1.5–2 倍提速，足以将两条路径的交叉点向更低位宽推移。该幅度是历史背景，本轮没有重测原版 GMP-ECM。

[旧吞吐工作簿](../paper/OpenCL吞吐量.xlsx) 的 `ecm` 页用标量列数乘曲线数，再除以记录的批次毫秒数。RTX 4060 Laptop 的 OpenCL 列在 M241、M421 的实测峰值均出现在 12,288 条，M1009 在 6,144 条。新实验保留 768 条作为低批量基线，扩到 49,152 条。按 GPU、曲线族、位宽和后端各自选样，可以区分吞吐平台与等批量条件的差别。

第一轮用独立的 OpenCL C 实现建立基线。设备仍一工作项一曲线，通用 CIOS 支持满 limb 模数。复用共享 NAF 编码，GPU 建表缓存 $dT$，切片只在必要时下载 XYZT。主机过程、落盘和数学边界见 [OpenCL 架构](../architecture/OPENCL.md#stage1-曲线执行)。

验收程序 [test_opencl_edwards_stage1.py](../../tools/test/test_opencl_edwards_stage1.py) 用 CPU 端点与独立 Montgomery ladder 校验存档，覆盖满容器高 limb、NAF 窗口、两种倍点公式、真因子和内存降窗。可选 `--resume` 验证中断恢复与整批输出。

RTX 4060 Laptop、RTX 4070 Ti 与 AMD gfx1150 的设备验收覆盖全部七个容器。AMD PAL 3640.0 从 512 bit 起曾在 `clBuildProgram` 中异常退出，禁用宽位 limb 循环的强制展开后通过 512、768、1024 bit 对拍。两张 NVIDIA 卡的中断测试确认恢复后文本存档逐条一致。队列验收在 AMD 上生成 8 条 Edwards 保存点及对应 `ECMSTAGE2=` 交接行。另一份 OpenCL save 由 CUDA Stage2 以 B1=300、B2=2000 实际续算并发布 NF。日志按设备保存在 `data/experiments/opencl_edwards_stage1/`。

首轮用 768 条做等批量对照，原始样本在 `data/experiments/opencl_edwards_stage1/bench_*/`。随后按两张卡各自的吞吐平台扩大批量，在 383／384／385 bit 两侧补测，并用更长的 B1 复核边界。[双卡小位宽对照](../performance/STAGE1.md#opencl-小位宽对照) 记录当前口径。后续剖析可在 384／385 bit 并列采集寄存器、private memory 和 local load/store。若溢出随位宽跃升，先试 CIOS 静态下标和缩短倍点临时变量的存活区间。若依赖链占主导，比较 `mul_hi` 的 32 位部分积与专用平方。同一套点公式和 NAF 协议继续用于对拍。
