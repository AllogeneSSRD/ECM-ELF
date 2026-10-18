# Stage2 NTT 摘要信息开发草稿

授权：补充NTT table信息，前四列为log2_length、length、iter_per_s、mean_seconds，恢复max_radix_bits、passes_forward等执行参数，行排序不变。用户另确认恢复上一提交的共享行排序实现；原始工作区回退保存在忽略证据目录。

实现范围：ECM与NTT各自的列优先级集中在同一columns.def，NTT稳定行序仍为length、batch、tile。iter_per_s按batch/mean_seconds计算，batch_iter_per_s为1/mean_seconds，合并统计后重新计算。max_radix_bits保留实际值；passes_forward保留实际报告，旧数据有合法tile和radix时按最少外层趟数推导，否则留空，不猜默认执行参数。

验证完成，证据根目录为 `data/experiments/stage2_ntt_metadata_20261018/`：

- `portable_final01/`、`runtime_final01/`、`provenance_final01/`通过现有CPU门禁及共享ECM行列序检查，保留旧版本来源复用、统计合并、未知时间和去重规则。
- `legacy_ntt01/`对旧0.37.2的4条NTT记录补齐展示列，原统计、执行参数及来源逐项保持；C++/Python列序一致；长度对数、两种吞吐率和趟数的损坏记录均拒绝；排序和格式化幂等通过。
- `gpu01/`在GPU1 RTX4060 Laptop上测L=2¹⁰至2¹⁴、batch=1/4、预算256 MiB，各参数先测3次再独立测4次，10行共70次正式观察。实际max_radix_bits和passes_forward与底层回执一致，passes覆盖1/2；重复导入第二批不增计数，合并后每行7次，均值/样本标准差与70条原始时间独立核对，吞吐率按合并均值重算。GPU时钟/功耗未固定，不作速度或性能提升结论。
- `import_final01/`核对既有24次生产观察离线导入与重复合并，计数和统计保持。

版本0.37.4，正式local为 `build_cuda_cmake/local_stage2_0_37_4/ecm_cuda_stage2.exe`，sm89、PTX3/addsub1、CUDA13.3.73、MSVC14.51.36231、GMP Zen3、SplitCompile=1；86项编译输入经哈希核验，CUDA对象与0.37.3相同。EXE SHA256为 `4458a337e554afb663ddadbe5c0d7174e8685b05dadee779caea8e14709b0ddb`，冻结汇总见 `final_acceptance.json`。

正式主题 `docs/architecture/AUTO_B2.md` 已同步更新。原始回执、试跑与工作区回退证据均保留；本草稿已完成，按仓库规则保留，等待用户处理，不自动删除。
