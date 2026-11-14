# GWNUM 31.6 许可与分发边界

核对对象是 Prime95 31.6 build 1 源码归档 `p95v3106b01.source.zip`。解包后的 `gwnum/gwnum.h:58` 声明库版本为 31.6，同一头文件第 593 行的 `gwnum_is_gpl()` 返回 0。GWNUM 专用条款在 `gwnum/readme.txt:86–111`。归档根目录的 `license.txt` 是 Prime95 应用 EULA，并注明应用含有 Entropia 的分布式计算软件。判断 GWNUM 库的权限以其专用条款和实际构建所含代码的声明为依据。GIMPS 网站的参与规则也适用于获取任务或上传结果等参与行为。[官方 31.6 归档](https://www.mersenne.org/download/software/v31/31.06/) · [GIMPS 源码说明](https://www.mersenne.org/download/) · [GIMPS 参与和 EULA 条款](https://www.mersenne.org/legal/)

## 上游授予的权限

- **使用和修改**。GWNUM 专用条款允许使用、修改，以及以源码或二进制形式再分发原版或修改版。本地用 ECM 寻找合数的因子属于这一使用许可的范围。
- **源码再分发**。保留 GWNUM 的版权声明、限制条款和免责声明。直接修改 FFT 或汇编源码也遵守这些条款，并保留相关文件各自的声明。可以单独维护本项目的适配代码及构建步骤，由用户从官方归档取得 GWNUM。
- **二进制再分发**。专用条款也授予此权限。其版权声明保留条款明确写的是源码。二进制仍有梅森素数发现条款，实际编入的第三方代码可能另有随包通知要求。随 DLL 一并保留 GWNUM 完整许可及适用的第三方声明，可向接收者交代这些条款。
- **梅森素数发现**。若用 GWNUM 找到梅森素数，条款约定 GIMPS 为发现方并适用其奖金规则。ECM 报告的因子和未找到因子都不构成发现梅森素数。GIMPS 下载页也明确区分了用源码找梅森素数和其他使用。[GIMPS 源码说明](https://www.mersenne.org/download/) · [GIMPS 奖励规则](https://www.mersenne.org/legal/)

## Windows x64 库的第三方声明

上游 `gwnum/compil64` 将 C/C++ 对象和 MASM 兼容汇编对象放入 `gwnum64.lib`。库中的 `mult.asm`、`gianthlp.asm` 等文件保留 Mersenne Research 的版权。若公开分发 DLL 或重编源码，应按实际链接及构建输入核对以下声明：

- `gwnum/gwnum.c:48` 包含 `mt19937ar.c`。该文件头要求源码保留版权、条件和免责声明，二进制分发须在随附材料中复现，并限制以贡献者名称作背书。
- `gwnum/giants.c:24` 包含 Takuya Ooura 的 `fftsg.c`。文件头及作者页面允许使用、修改和再分发，修改时保留对原软件包的出处。另保留 `giants.c` 和 `giants.h` 所载 Perfectly Scientific, Inc. 版权。[Ooura FFT 原始发布页](https://www.kurims.kyoto-u.ac.jp/~ooura/fft.html#license)
- `gwnum/gwdbldbl.cpp` 通过 `dd.cc` 纳入 QD 双倍精度实现。`qd/readme-qd` 要求源码及二进制分发保留版权、条件和免责声明，另有禁止背书条款。
- `gwnum/gwbench.c:90` 包含 SQLite amalgamation。官方交付的 SQLite 库源码属于公有领域。[SQLite 版权说明](https://www.sqlite.org/copyright.html)
- x64 汇编入口用到 `unravel.mac`，其中包含 `ksamd64.inc`，继而包含带 Microsoft 版权声明的 `kxamd64.inc` 和 `macamd64.inc`。这两个文件头只有版权信息。若要把这些文件复制到本项目或作为独立源码包再发布，需核对其来源和授权范围。保留其声明，按实际构建判断是否还涉及 SDK 或运行库的再分发条款。

## 与 ECM-ELF 一起分发

ECM-ELF 根目录的 [LICENSE](../../LICENSE) 是 GPLv3。GWNUM 的梅森素数发现条款会影响接收者对库的某一类使用。若把主程序和插件作为一个受 GPLv3 约束的组合分发，需要判断该条款是否构成 GPLv3 §7 和 §10 的附加限制。GWNUM 专用条款没有授予将 GWNUM 直接改按 GPLv3 分发的权利。用 DLL、可选加载、C ABI 或另放一个 ZIP 都不能单独解决这个问题。FSF 对同进程插件的解释会考虑调用方式以及数据交换的紧密程度。GMP 6.3.0 可按 LGPLv3 或 GPLv2 等条件使用，这是另一项依赖的许可，与 GWNUM 条款分别处理。[GPLv3 原文](https://www.gnu.org/licenses/gpl-3.0.html) · [FSF 插件说明](https://www.gnu.org/licenses/gpl-faq.html#GPLPlugins) · [GMP 6.3.0 许可](https://gmplib.org/manual/Copying)

GMP-ECM 提供了本地集成的先例。它的安装说明要求用户从 GIMPS 取得源包并自行构建 GWNUM，再通过 `--with-gwnum` 接入。GMP-ECM 7.0.5 的 `main.c` 在链接了自报为非 GPL 的 GWNUM 时打印许可不兼容的分发提示。[GMP-ECM 安装说明](https://github.com/sethtroisi/gmp-ecm/blob/main/INSTALL-ecm) · [GMP-ECM 维护者的源码覆盖率报告](https://members.loria.fr/PZimmermann/ecmnet/cov/ecm/main.c.gcov.html)

| 方式 | 分发边界 |
| --- | --- |
| 用户本地构建和运行 | 用户从官方 31.6 归档获得 GWNUM，本地生成 DLL 并由 ECM-ELF 调用。保留上游条款。本地使用和向第三方交付链接组合是两件事。 |
| 常规 ECM-ELF 发布 | 主程序随 GPLv3 源码和相应构建材料发布，提供可独立工作的运行时加载接口及本地构建说明。GWNUM 头文件、静态库和链接所得的 DLL 由用户自行取得或生成。 |
| 发布组合或专用插件二进制 | 优先向 GWNUM 权利方取得适用的 GPL 兼容授权。另一条路径是在确认仍满足 GWNUM 及所含第三方许可后，由相关 ECM-ELF 代码的权利人对该库授予明确的 GPLv3 链接例外。应确认例外覆盖实际链接的所有相关 GPL 代码，然后按授权提供源码、构建材料及第三方通知。 |
| 分发真正独立的程序 | 如有单独交付的需要，可评估以通用文件或简单命令行协议运行的独立计算程序。是否属于 GPLv3 §5 的聚合仍取决于程序实际边界及交换内容。 |

[GPLv3 链接例外与聚合说明](https://www.gnu.org/licenses/gpl-faq.html#GPLIncompatibleLibs)
