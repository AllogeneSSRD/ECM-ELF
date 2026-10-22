# ELY 三套装甲皮肤开发草稿

日期：2026-10-22。GUI v2 功能版本0.48.0。三套图集、原生热切换、资源回归及本地分发检查已完成；用户已授权将本轮代码、说明与测试整理为0.48.0提交；新增皮肤图片/SVG继续仅保存在本机。本文件是皮肤制作和验收的临时权威草稿，完成后保留，由用户决定整理；GUI总体进度见 [现代化草稿](GUI_MODERNIZATION_DRAFT.md)。

## 范围与命名

用户要求依据 `.refactor/爱莉希雅例图/` 制作三套ELY皮肤，并从网络Wiki核对正确英文装甲名称用于命名。使用以下名称，音乐符号和心形保留在界面；文件与配置使用ASCII标识：

- **Miss Pink Elf♪**：粉色妖精小姐♪，对应“物爱”；`miss_pink_elf`。[Wiki](https://honkaiimpact3.fandom.com/wiki/Miss_Pink_Elf%E2%99%AA)。
- **Herrscher of Human: Ego**：真我·人之律者；`herrscher_of_human_ego`。[Wiki](https://honkaiimpact3.fandom.com/wiki/Herrscher_of_Human%3A_Ego)。
- **Hi♪ Love Elf♥**：嗨♪爱愿妖精♥；`hi_love_elf`。[Wiki](https://honkaiimpact3.fandom.com/wiki/Hi%E2%99%AA_Love_Elf%E2%99%A5)；[HoYoverse公告](https://www.hoyoverse.com/ru-ru/news/160583)同样使用此英文名。

命名核对日期为2026-10-22。英文全名保存在 `assets/gui_v2/skins/manifest.json`，生成C++注册表，不手工维护另一份界面翻译。中文名称为同一装甲的名称；不使用素材文件里的简称作为正式英文名。

## 美术与动画

采用现有窗口边缘Q版比例，依据七张用户素材重新绘制服装、头饰和侧分发型，复用固定MIT ELYTerminal矢量角色的脸部表情与动作规则。不是整身游戏模型或贴图提取；参考图不进入运行包，不要求生成时访问 `.refactor/`。

- Miss Pink Elf♪：黑金羽饰、蓝色宝石、冰蓝眼睛、紫色披肩与黑色手套。
- Herrscher of Human: Ego：白色翼冠、薄纱、紫色晶瓣、金色交织结饰与白手套。
- Hi♪ Love Elf♥：白色贝雷帽、玫瑰与红色切面爱心、薄荷色缎带、白色发束与玫瑰红服装。

`shared.svg` 提供共用发型和精灵耳，三个装甲SVG提供各自服饰/头饰层及运动骨骼，manifest提供名称、配色与来源。生成器将它们合成进原骨架，按原七状态生成待机、工作、等待、异常、完成、睡觉、眨眼，每状态8帧。每套透明RGBA图集1792×1736，单帧224×248，运行时112×124逻辑单位；原版Classic ELY继续保留且默认值不变。

资源输出位于 `assets/gui_v2/pet_ely_<id>.png`，同名JSON记录来源、固定骨架哈希、离线配方输入和PNG SHA256。最终图集哈希：

- Miss Pink Elf♪：`4a9141d6ab06d6e0360e067f5b0536a42740a1020281fdccab8eaa41ed677c2e`。
- Herrscher of Human: Ego：`a43f13ad459ccfe003c088fca6de59cf996918a3ab7c91ddc3cb269fe76471a9`。
- Hi♪ Love Elf♥：`014653a65c6e62d1d6c48d93db27b9d3ff4e7daa0a93480414553aafacfa39d7`。

来源与生成方式另见 [资产说明](../../assets/gui_v2/skins/README.md)。SVG是新皮肤的可编辑源；PNG是可复现生成物。浏览器只用于离线栅格化和预览，不进入原生运行时。

## 原生接入

配置新增独立侧车 `pet.skin`，默认 `classic`，仅接受四个注册标识；不写业务TOML或旧GUI偏好。`config/ecm_options.json`是默认值与英中说明的来源，生成29个侧车字段。未知配置标识使非法侧车保留且禁止覆盖，命令行未知标识在启动前拒绝，不允许拼接任意图片路径。

“设置 → ELY皮肤”保留完整英文名，点击立即选择；正常退出或显式保存界面偏好记住选择。命令行 `--pet --pet-skin <id>` 可覆盖初始选择。选择皮肤不强制打开原本关闭的宠物。生成注册表 `generated/pet_skins.h`为名称和文件映射提供同一来源。

`pet/window.cpp`仅加载当前图集。切换时校验新图集、替换原图集并使旧帧缓存失效，保留透明非激活HWND、几何位置和交互；不会持有三套完整动画缓存。缺失或非法图集提示具体文件并关闭宠物，不终止GUI。主窗口仍默认DX11；Vulkan、原GUI与计算算法不在本轮变更范围。

### 本地资源与安装方式

用户最初要求本轮功能不提交，后续授权先提交上轮改动再修复启动问题；新增皮肤图片和SVG继续仅本机保存，不强制添加。原版皮肤及来源说明按后续授权排除在忽略规则之外，精确白名单见下文。运行时PNG从exe旁`assets/`读取，不嵌入二进制；已登记皮肤复制安装或同名替换无需编译。安装步骤、固定布局、重启要求及新增ID仍需编译的边界，统一见 [GUI使用说明：安装与替换皮肤](../usage/GUI_V2.md#安装与替换皮肤)。当前构建仍依赖本地资产及指纹，不能据外部加载声明无资源的干净克隆可直接构建。

本地候选包的源快照使用构建时固定的源目录清单，并补齐未跟踪的生成器/指纹文件，避免为打包而暂存或提交。源快照明确标注含未提交改动，HEAD仅表示基础提交；包及原始证据均保存在忽略目录，不上传。

## 验证与诊断记录

`skins_0480_assets_01/result.json`验证三套共168帧、每帧独立渐变ID、PNG尺寸/透明RGBA、名称和哈希，以及八个睡眠姿势的两个Z与头饰完整边界。早期离线输出在 `skins_0480_svg_01`，最终可复现配方输出在 `_02`；图像字节相同，元数据更新为最终配方身份。原版PNG不覆盖。

`skins_0480_preview_02`验证三款对比HTML、七种状态、播放暂停、1×/1.5×/2×、明暗和透明底纹、390px窄屏；无页面脚本错误。初次预览测试对0/-0使用JavaScript严格等号导致错误断言，修正测试的数值口径，原失败保留。预览页是 [ELY皮肤画廊](../web/ely-skins-20261022.html)，不写客户端设置。

实际界面首轮 `skins_0480_runtime_01`发现英文名字中的♪/♥缺字。追加系统Segoe UI Symbol回退并在字体角色间共享同一份输入文件；未随包分发字体。第二轮符号检查通过、四种外观各七状态可显示、同一HWND保留，但12次热切换的private bytes增长约14MiB，超过8MiB资源回归门限。该失败继续保留，不据GDI句柄稳定声称没有资源问题。

逐轮采样 `_03` 显示首次“直接从新图集跳到Wink行”的序列增加12853248 bytes，后两轮约持平、GDI始终11；原预热是顺序播放0–6行，与测量的直接Wink换装不一致。释放旧解码图集再打开新图集，避免两张完整图集同时存活，窗口保留已提交的位图。`_04`仍出现约13MiB首轮增长，后两轮约持平；释放顺序调整不能单独解释或证明消除这项增长。未确认是具体哪个GDI+/堆/驱动缓存，不写成精确根因。

最终测试固定两轮与测量相同的换装路径，并等待新图集绘制后的界面记录再采样，冷路径成本单独记入原始数据；没有改8MiB/2句柄的稳态门限。`skins_0480_runtime_05/result.json`通过：四种皮肤各七状态、实际按钮点击、同一HWND、符号字形、正常保存与重启、缺少选中资源时关闭宠物且GUI继续、任意路径拒绝、业务TOML字节不变。RTX4070 Ti、默认DX11、144DPI、静态帧，无计算worker。最终exe SHA256 `1e0ca18b1d58b5d179bf6ea48b7f23356d2c45f7a98e31ebbbb81ef06e5d8652`；两轮同路径预热增长212992 bytes，其后三轮/12次切换private增长0、GDI=11→11。它是一次本机序列的稳态证据，不覆盖早期冷路径增长，也不作为全部性能或长期无泄漏保证。

状态/业务合同助手见 `skins_0480_contracts_01`；非法skin保留原侧车且禁止覆盖、旧侧车默认Classic、图集隔离采样等通过。设置中英切换、右侧迁移、明暗快照、保存重启、搜索与缺中文字体回退通过 `skins_0480_usability_01`。旧GUI73个保护文件、旧exe哈希、业务配置既有投影通过 `skins_0480_legacy_01`。生成器和CMake同时校验新资源/配方指纹，原版PNG保持不变。

原始日志、失败样本和截图保存在忽略目录 `data/experiments/gui_v2/skins_0480_*`，不删除或覆盖。完整P5性能矩阵、长期使用、不同物理DPI与干净OS资格不由本轮短测试替代。

### 本地候选与重建检查

`skins_0480_package_01`生成0.48.0独立候选目录及ZIP，含29个文件，exe仍为上述`1e0ca18b…`；ZIP SHA256为`957a043c8dfbcbc7faff6e821609cc393b04c046cbae1568dfe77ae492c95554`。包仅保存在`data/experiments/gui_v2/`，未上传；包含三套PNG及来源说明，不包含用户原始例图。包清单明确含未提交改动和本地未跟踪资源，基础HEAD不表示本轮已提交。

`skins_0480_package_test_01/result.json`通过28项包内文件哈希核对及中文目录迁移启动、三套外部皮肤、缺图降级、侧车保留/重启、仅System32 PATH等检查。Vulkan缺loader检查在设备初始化前退出，不运行已知冻结的Vulkan呈现路径。此项是在当前已安装驱动/runtime的同一机器上完成，不代表独立干净OS资格。

`skins_0480_source_build_01/result.json`核对源快照153项构建目录输入，与冻结清单逐文件一致；解压后在不依赖Git元数据的目录独立构建成功，默认DX11运行8帧正常退出。重建exe SHA256为`bb0af8eca6306db8b415b9176dab94ccb53a29993d019c767376b58d422161a2`，不同构建路径的二进制不声明字节复现。首次临时日志断言错误期待`renderer=dx11`，实际日志为`backend=DX11`；复用现有`renderer_probe`读取同一份成功进程日志后通过，未修改程序或重跑掩盖证据。

安装步骤已写入使用说明和皮肤HTML；`skins_0480_preview_03`再次检查七状态、缩放、明暗与390px窄屏无脚本错误。总体规划HTML在`skins_0480_plan_01`验证P0–P4/P6完成、P5仍待验收，与原门禁状态一致。

### 原版归属说明与 Web 菱形变体

用户补充要求保留原版皮肤的来源与版权说明，并排除在Git忽略规则之外。核对本地上游`ELYTerminal/LICENSE`与固定副本完全一致，版权行为`Copyright (c) 2026 ELYTerminal contributors`，MIT全文不改写。新增`assets/gui_v2/CLASSIC-ELY-NOTICE.md`记录上游仓库、固定提交、原始/规范源码哈希、生成流程及PNG身份；`ELYTerminal-LICENSE.txt`为上游许可证的逐字节副本。这不表示MIT要求使用Git，而是按用户要求让原版及其说明可继续随仓库保留。

`.gitignore`仅对白名单中的原版PNG/JSON、上述来源/许可文件，以及`third_party/gui_v2_pet/`中的LICENSE、README、rig.rs放行。其他第三方本地目录、新装甲PNG/SVG和Web资源保持忽略，不暂存、不提交。CMake资源目标及打包器随原版资源复制两份说明文件；本轮仅运行`ecm_gui_v2_assets`，实际复制成功，未编译或启动原生程序。原版PNG、现有三套原生PNG、注册表/配方指纹以及exe（`1e0ca18b…`）保持不变，故不增加程序版本。

三套Web变体位于`assets/web/ely_variants/`，不在原生构建目录清单、皮肤注册表、配置允许值、CMake图集目标或程序包中。生成器`tools/gen/generate_ely_diamond_preview.py`与共用`diamond-pupils.svg`仅覆盖现有`pupil-crystal-l/r`两层，服装、虹膜、高光、骨骼及可见性沿用。用户进一步明确菱形须为中空浅色轮廓；最终采用`fill=none`、虹膜浅色轮廓，中心保留底色，无实心填充或切面。早期深色实心方案及截图保留在`diamond_0480_svg_02`和`diamond_0480_preview_02`作为被用户纠正的草稿，不作为最终样式。

最终SVG输入位于`diamond_0480_svg_03`，输出为三张1792×1736、56帧透明图集。PNG SHA256分别为：

- Miss Pink Elf♪：`ca5826a55956b4b34b23313ae7e6363b98736d582321139e1fbe6be3897beec5`。
- Herrscher of Human: Ego：`4baac506fc2ce3d27fa9884cfbc06215b4e985883728fc020e5db95f56c4fe8f`。
- Hi♪ Love Elf♥：`f70ce969553cf87d09f3222c700b6be7da4af8230795a62e29dccd43da768215`。

`diamond_0480_assets_03/result.json`核对全部168个SVG姿势的非瞳孔层与既有配方逐层一致，Done/Doze不显示瞳孔，Wink只显示一侧；两眼中心像素等于原虹膜底色，确认中空。另核对源指纹、PNG格式、Git白名单/忽略样本、许可证逐字节一致、原生图集/exe未变及构建目录不含菱形变体。

早期像素差分断言在`diamond_0480_assets_02`失败：独立栅格化的少量非眼边缘像素也有差异，故不能声称所有非眼PNG像素逐字节相同。`diagnosis.json`记录全部非瞳孔SVG层及属性实际完全一致；最终校验精确比较层几何、可见性和瞳孔中心，不为像素断言放宽数值门限，也不声明具体栅格器差异根因。早期深色版本的像素检查为`diamond_0480_assets_01`，与最终样式分开记录。

`diamond_0480_preview_03/result.json`通过3套×2版本、6张图集实际加载、七状态、播放暂停、缩放、明暗、底纹、放大细节和390px窄屏，无脚本错误；最终截图已目视检查。初轮页面测试发现`[data-eyes]`选择器也选中状态标记所在body，已将事件/选中状态范围限定为`button[data-eyes]`，失败目录`diamond_0480_preview_01`保留。网页默认展示中空轮廓并明确“仅Web演示，未接入程序”，图集URL带内容哈希以避免继续显示缓存旧图。
