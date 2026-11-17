# Stage1 NTT 已知因子开发草稿

任务保留实际余因子 N 和原始梅森元数据。NTT 算术宽度单独解析为原始 p，配置使用默认 1 的 `ntt.allow_known_factors`。

主机构造和求逆使用 N，GPU 的多精度加减乘方使用 M。原生 Edwards GPU 字典的根逆元使用 N，后续乘积树仍使用 M。完成点回到 N 上归一化，因子仅对 N 验证。检查点把 N 与 p 同时纳入身份。

验证入口为 `tools/test/test_stage1_ntt_known_factors.py`，覆盖线性/加权、Edwards x-only/NAF、主机/GPU 字典、PARAM0、开关关闭、短尾续跑和 Stage2 交接。0.64.4 在 GPU1 验收通过。15899 位目标使用 M15919 承载，末点与 CGBN 及独立参考一致。M65 去掉已知 31 后发布新因子 8191。61 位目标使用 M244 承载的 130 曲线续跑与连续运行一致。原始资料保存在忽略的 `data/experiments/stage1_ntt_known_factors/final_0644/`。
