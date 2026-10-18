# Stage2 共用行排序开发草稿

授权：将matrix行排序规则集中为C++、Python共用定义，ECM顺序为condition → b2 → d → carrier_exponent → b1 → source。

实现：`src/core/ecm_stage2_tune_rows.def`定义ECM与NTT的键及类型，均稳定升序；NTT保持length → batch → tile，tile缺失按0处理。生产序列化、Python导入和NTT格式化工具直接消费同一规则。行序与列序定义独立，计时、来源、统计与格式8复用规则保持有效。

验证完成，证据在 `data/experiments/stage2_tune_row_order_20261018/`：`portable01/`和`runtime01/`通过现有CPU门禁；`provenance01/`用9行不同参数和来源核对指定优先级、同键稳定性、原生/Python输出及统计不变，并保留跨版本来源、未知时间和旧条件迁移检查；`ntt01/`核对4条NTT行数值排序、来源保留和格式化幂等；`import01/`通过24次既有观察的重复导入/合并及独立统计检查。不执行新的GPU曲线，不声称性能收益。

版本0.37.3，local构建为 `build_cuda_cmake/local_stage2_0_37_3/ecm_cuda_stage2.exe`，sm89、PTX3/addsub1、CUDA13.3.73、MSVC14.51.36231、GMP Zen3、SplitCompile=1；86项编译输入经哈希核验，CUDA对象与0.37.2一致。EXE SHA256为`eaa4822db03d9b74f6dbc577a079f5e1afc44f41f1042cc45352dc7be78bb81a`；冻结汇总见`final_acceptance.json`。

正式说明已更新至 `docs/architecture/AUTO_B2.md`。本草稿已完成，按仓库规则保留，等待用户处理，不自动删除。
