# 原生 OpenCL Stage2 生产接入草稿

## 目标与范围

保留多项式 ECM Stage2 算法，将原生 OpenCL 算子接到现有保存点校验、父子进程、队列收据、曲线进度、日志与结果发布流程。构建独立 `ecm_opencl_stage2.exe`，默认 CUDA 路径不切换。

生产后端采用 OpenCL 点生成、CPU 256点批量归一化、OpenCL F/G 树及 Newton/fold/scaled 下降、CPU 块乘积/GCD。点按真实 target N 运算；显式梅森承载只用于多项式算术，输出投影回 target N 后做GCD。宽位AMD点模块的已知资格限制保留，使用CPU点生成配合OpenCL多项式。

每次曲线执行设备乘法启动检查；GPU点取样对照GMP；三个baby位置累计全giant范围的独立GMP乘积并与下降结果对照；因子整除验证及最终设备排空成功后才发布。不可逆分母按点检查，不能用伪造的仿射零根发布成功。饱和且未提取到真因子时返回失败、保留队列。

## 接入边界

- 共用生产入口位于 `src/ecm/stage2/cuda/main.cpp`，通过编译期后端标志选择引擎；该路径名不表示OpenCL依赖CUDA。
- `src/ecm/stage2/opencl/engine.cpp` 实现既有引擎ABI；符号中的CUDA前缀是兼容链接接口。`backend.h` 控制结果中的引擎身份。
- 不读取CUDA tune或旧成本模型。首版使用共享D候选表与独立的基本成本/owned-payload准入模型；显式B2，支持D=0和固定D。
- OpenCL 1.2无可移植空闲显存查询，日志不能把global memory当作free。arena作为owned buffers硬预算，由runtime独立记账和门禁；估计不是进程显存峰。
- OpenCL设备编号按ICD枚举，不能直接当作CUDA编号；当前机器有重复NVIDIA平台，需要显式核对设备名称。
- CUDA Auto B2/tune、自动承载比较、relocation、Graph等尚不接入OpenCL；允许显式验证后的承载指数。

## 验证与证据

实验根目录：`data/experiments/opencl_production_20261029/`。构建、初期配置错误、独立烟测与验证分别保留，不能将失败记录混成性能样本。

`check_opencl_stage2_worker.py` 检查正因子、NF、直接梅森/余因子承载、非法承载、预算、checksum、基本D、plan-only、队列停止与重排续跑。性能工具 `bench_opencl_cuda_stage2.py` 使用4个既有余因子保存点、两档B2、固定D2310，每侧3次，交替后端顺序，不丢首轮；汇总器核对二进制与日志身份。

原始日志和派生数据保存在忽略目录，最终条件、输入、二进制身份、比例和限制写入正式性能文档。

## 验收状态

- [x] 当前源码0.54.0：CUDA完整isolated构建与HostOnly复核，原生OpenCL独立构建。
- [x] 最终OpenCL在NVIDIA/AMD各13项生产门禁；两后端对同一Stage2样例均提取因子257。
- [x] 4位宽×2档B2×2后端×3次，共48条完整NF；保留首轮和波动范围。
- [x] AMD 1939/7995-bit真实Edwards保存点，CPU点生成配合OpenCL多项式，完整NF通过。
- [x] 根CMake正式OpenCL目标独立构建、因子257和无CUDA运行库链接验证。
- [x] 同步架构、使用与性能文档；原始证据、源码快照与派生数据保留在忽略目录。
- [ ] 本草稿保留供用户决定归档或清理，不自动删除。

性能结论以 [正式报告](../performance/OPENCL_STAGE2.md) 为准。当前OpenCL没有性能优势：引擎中位数之比5.35～27.42，完整进程之比1.19～18.10；因此保持显式后端入口，没有替换CUDA默认。下一轮先隔离点生成与G/fold计时，再推进点组织和短NTT提交。
