# Stage2 共用列序与行来源开发草稿

授权：保留当前 `ecm_stage2_tune_portable.h` 中用户调整的列序，集中为C++/Python共用定义；版本号和最后更新时间移至matrix行；程序版本仅作来源记录，升级不清空性能缓存。

实现：共用 `src/core/ecm_stage2_tune_columns.def` 在C++编译时展开，Python通过 `tools/bench/stage2_tune_matrix.py` 直接读取。保留用户已调整的顺序，耗时/标准差/次数位于D后面；两个来源列排在已有优先列之后，剩余可选字段按字母排序，列宽对齐保留。local构建依赖闭包包含该定义，变更后会重编译。

格式8增加行内 `program_version`、`updated_at`，兼容旧format 8的条件级版本。读取时先继承旧行来源，再移除条件中的版本并合并等价条件；独立统计按真实次数加权合并，重复/子集去重，部分重叠拒绝。条件、统计身份、上限和选型资格均排除程序版本，仍核对设备、环境和执行形状。format 8以前的中位数结构仍不兼容，不伪装成均值迁移。

聚合实测行保留最近已知采样时间对应的源版本，模型行记录估算生成来源；时间为UTC九位小数。较低版本也可能更晚运行，不能按版本大小猜测先后。未知时间留空，双方均未知时保留已有来源；读取、格式化、重复导入和未实际纳入统计的结果不更新时间。正常观察使用原 `runtime_completed_ns`，旧JSONL可由该字段恢复时间。逐次来源继续保留在JSONL或冻结证据。

验证完成，证据根目录为 `data/experiments/stage2_tune_provenance_20261018/`：

- `cpu_final01/`、`portable_final01/`：方差参考、快照去重、部分重叠拒绝、样本上限、跨版本复用、设备隔离、NTT参数行、20行矩阵、10级网格、105组邻域和坏文件拒绝。
- `provenance_final01/`：原生/Python共用列序与读写一致；跨版本统计、时间选源、旧条件迁移、未知时间及去重幂等。
- `reuse_final01/`：GPU1直接复用0.37.1的10次普通路线观察，执行一次NF后累计为11次；不重校准，只读不写文件，承载旧行版本及未知时间保留。
- `gpu_final02/`及其审计：M1009的383位余因子，B1=50000、B2=10⁸、D=30030，15条队列NF，普通10次/承载2次两行；11次实测选型均未选择显著被支配路线，GMP抽查18836，算术错误0。另含锁定、停采仍复用、Montgomery Auto B2规划和因子命中排除。
- `portable_gpu_final02/`：22项正式ECM/NTT/短校准/Auto B2/合并与计划门禁。
- `import_legacy01/`、`import_mixed_final01/`：旧JSONL恢复及跨两版24次观察合并，重复输入不增计数；普通20次、承载4次仍在同一条件下两行，均值/方差独立核对，坏检查回执拒绝。
- `sort_final02/`：4条NTT行排序、数值保持、来源保留及格式化幂等。

这是功能验证，计时边界为引擎init+main；功耗/时钟未按性能协议冻结，不据此声称加速。保存点重复使用，不是新增独立ECM覆盖。源JSONL、save和旧profile均未改写。

版本0.37.2。正式local构建为 `build_cuda_cmake/local_stage2_0_37_2/ecm_cuda_stage2.exe`，sm89、PTX3/addsub1、CUDA13.3.73、MSVC14.51.36231、GMP Zen3、SplitCompile=1；85项编译输入经哈希核验，CUDA对象与0.37.1相同。EXE SHA256为 `5714f405d4bde1853666e94c5add94bdf31cbb01e6792e0ba3e3f7f8d81a9243`。冻结汇总见 `final_acceptance.json`。

当前功能说明已同步至 `docs/architecture/AUTO_B2.md`，配置说明经生成器同步。本草稿已完成，按仓库规则保留，等待用户处理；不自动删除。
