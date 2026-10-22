# Classic ELY：来源与生成说明

适用资源：本目录的 `pet_ely.png` 和 `pet_ely.json`，即程序中的 Classic ELY。这份说明和 `ELYTerminal-LICENSE.txt` 应与原版资源一同保留、分发；许可证全文保留上游版权与许可声明。

- 上游仓库：[ELY-Open-Source/ELYTerminal](https://github.com/ELY-Open-Source/ELYTerminal)。
- 固定提交：`fe07813fdfa9fd600274066d9fcd88d5f5a3bcc0`。
- 原始矢量角色：[src/pet/rig.rs](https://github.com/ELY-Open-Source/ELYTerminal/blob/fe07813fdfa9fd600274066d9fcd88d5f5a3bcc0/src/pet/rig.rs)。
- 上游版权：Copyright (c) 2026 ELYTerminal contributors。
- 上游许可：MIT，全文见本目录 `ELYTerminal-LICENSE.txt`；仓库固定副本为 `third_party/gui_v2_pet/LICENSE`。

`pet_ely.png` 是 ECM-ELF 离线生成的动画图集，不是声称上游已经提供的同名PNG。固定源码副本为 `third_party/gui_v2_pet/rig.rs`，仅将换行规范为LF；通过 `tools/gen/generate_gui_v2_pet.py` 提取矢量层与离散动画，再用 `tools/gen/render_gui_v2_pet.cjs` 离线栅格化。结果为1792×1736透明图集、7状态×8帧；运行时无需Rust、GPUI或浏览器。

上游原始源码SHA256为 `1ec32d05c9b9c9e9b171bad911c161956daabca25572829afd73d261a919cf88`，LF规范源码SHA256为 `c34e9274cf77d8011eee331d91f3ea7869a619f9fc72266a85ff7a9ad6a9cca6`，生成PNG SHA256为 `eb0eac5b7a5809d5ea0cf4d1e6dac8e521650671e243648508408cc98bfb1972`。几何与哈希同时记录在 `pet_ely.json`。

此说明针对固定MIT矢量角色及其生成物；新增造型的来源单独记录。Git白名单只放行原版PNG、元数据、本说明、许可证和固定上游源码，其余本地皮肤资源仍被忽略。
