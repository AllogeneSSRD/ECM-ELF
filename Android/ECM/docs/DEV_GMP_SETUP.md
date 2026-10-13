# Android ECM：GMP 安装与构建配置

ECM GPU Stage-1 在主机侧需要 GMP（`mpz_t`：解析 N、计算 `batch_s`、Montgomery 与因子回读）。未链接 GMP 时 APK 仍可打开 UI，但运行 ECM 只会显示构建说明。

本工程 **arm64-v8a** 对应 vcpkg triplet **`arm64-android`**。

---

## 1. 前置条件

| 组件 | 说明 |
|------|------|
| Android NDK | Android Studio → SDK Manager → NDK（建议与下文 `ANDROID_NDK_HOME` 同一版本） |
| vcpkg | 已 `bootstrap-vcpkg`（Windows：`bootstrap-vcpkg.bat`） |
| 主机 C++ 工具链 | Visual Studio「使用 C++ 的桌面开发」（vcpkg 编译 GMP 需要） |

本仓库 ECM 模块路径：`Android/ECM/`。

---

## 2. 安装 arm64-android GMP（vcpkg）

### 2.1 设置 NDK 环境变量

PowerShell（当前会话）：

```powershell
$env:ANDROID_NDK_HOME = "D:\AppData\Android\Sdk\ndk\28.2.13676358"
```

长期生效：Windows 用户环境变量 `ANDROID_NDK_HOME` 指向 NDK 根目录（含 `build/cmake/android.toolchain.cmake`）。

> 建议与 Android Studio 实际使用的 NDK 版本一致。可在 SDK 的 `ndk/` 下列出已安装版本。

### 2.2 安装 GMP

```powershell
D:\code\vcpkg\vcpkg.exe install gmp:arm64-android
```

首次构建会下载 msys2 等工具并运行 autotools 配置，可能需要数分钟。正常日志包括：

```
-- Generating configure for arm64-android
-- Configuring arm64-android-dbg
...
```

### 2.3 验证安装结果

```powershell
Test-Path "D:\code\vcpkg\installed\arm64-android\include\gmp.h"
Test-Path "D:\code\vcpkg\installed\arm64-android\lib\libgmp.a"
```

两条均应为 `True`。目录布局须为：

```
<prefix>/
  include/gmp.h
  lib/libgmp.a
```

vcpkg 的 `<prefix>` 一般为：`<VCPKG_ROOT>/installed/arm64-android`。

---

## 3. Gradle / CMake 配置（已接入）

`app/build.gradle.kts` 会读取 **`ecm.android.gmp.root`**，并传入 CMake：

```
-DECM_ANDROID_GMP_ROOT=<prefix>
```

CMake 在找到 `include/gmp.h` 时定义 `ECM_HAVE_GMP` 并链接 Stage-1 源文件（见 `app/src/main/cpp/CMakeLists.txt`）。

### 3.1 配置路径（推荐）

在 **`Android/ECM/local.properties`**（gitignore，本机专用）增加一行：

```properties
ecm.android.gmp.root=D\:\\code\\vcpkg\\installed\\arm64-android
```

可参考仓库中的 `local.properties.example`。

### 3.2 备选：gradle.properties

在 `gradle.properties` 取消注释（会进入版本库，适合团队统一路径）：

```properties
ecm.android.gmp.root=D:/code/vcpkg/installed/arm64-android
```

优先级：`local.properties` → `gradle.properties`。

---

## 4. 编译与验证

### 4.1 命令行构建（需 JAVA_HOME）

若 `gradlew` 报 `JAVA_HOME is not set`，任选其一：

1. Android Studio：**Settings → Build, Execution, Deployment → Build Tools → Gradle → Gradle JDK** 选 **Embedded JDK**，然后在 Studio 内 **Build → Rebuild Project**（推荐）。
2. 命令行：设置 `JAVA_HOME` 为 Studio 自带 JBR，例如：
   ```powershell
   $env:JAVA_HOME = "D:\AppData\Android\Android Studio\jbr"   # 按本机路径修改
   cd D:\code\MPA-OpenCl\Android\ECM
   .\gradlew.bat clean assembleDebug
   ```
3. 在 `gradle.properties` 增加（路径按本机修改）：
   ```properties
   org.gradle.java.home=D\:\\AppData\\Android\\Android Studio\\jbr
   ```

### 4.2 构建 APK

```powershell
cd D:\code\MPA-OpenCl\Android\ECM
.\gradlew.bat clean assembleDebug
```

或在 Android Studio：

1. **File → Sync Project with Gradle Files**
2. **Build → Clean Project**
3. **Build → Rebuild Project**

若曾无 GMP 配置编译过，请删除 `app/.cxx` 目录后再 Rebuild（清除旧 CMake 缓存）。

### 4.3 确认 GMP 已链入

**Gradle 同步**时 Build 窗口应出现：

```
ECM: linking GMP from ecm.android.gmp.root=D:/code/vcpkg/installed/arm64-android
```

**CMake 配置**（Build Output 中搜索 `ECM Android`）应出现：

```
ECM Android: ECM_ANDROID_GMP_ROOT='D:/code/vcpkg/installed/arm64-android'
ECM Android: GMP found at D:/code/vcpkg/installed/arm64-android
```

若出现 `ECM Android: GMP not found` 或 Gradle 打印 `ecm.android.gmp.root not set`：

1. 确认 `local.properties` 或 `gradle.properties` 已设置 `ecm.android.gmp.root`
2. **Sync Project with Gradle Files** 后 **Clean + Rebuild**
3. 删除 `Android/ECM/app/.cxx` 后重编
4. 本工程已关闭 `org.gradle.configuration-cache`，避免旧配置缓存忽略新路径

### 4.4 设备上冒烟测试

| 字段 | 值 |
|------|-----|
| N | `(2^421-1)` |
| B1 / B2 | `100` / `0` |
| gpucurves | `64` |
| device | `0` |
| verbose | 勾选 |

成功时不应再出现 “ECM factorization is not linked”；日志中应有 `Parsed N bit-size`、`opencl_ecm_stage1 returned` 等。

桌面对照：

```powershell
echo '(2^421-1)' | .\build\Debug\ecm.exe -v -d 0 --curve mont --target gpu -gpucurves 64 100 0
```

---

## 5. Windows 桌面 `ecm.exe`（与 Android 独立）

根目录 `CMakeLists.txt` 编译桌面目标（`ecm`、`opencl_ecm_entry` 等）需要 **x64-windows** GMP，与 **`gmp:arm64-android`** 无关。

```powershell
D:\code\vcpkg\vcpkg.exe install gmp:x64-windows
```

CMake 自动探测顺序：

1. `-DECM_WINDOWS_GMP_ROOT=<prefix>`
2. 环境变量 `VCPKG_ROOT` → `installed/x64-windows`
3. 回退 `D:/code/vcpkg/installed/x64-windows`

配置日志应含：`GMP: include=... lib=...`。若仅安装过 arm64-android，桌面编译会报 `无法打开包括文件: gmp.h` — 需再装 **x64-windows** triplet。

---

## 6. 常见问题

| 现象 | 处理 |
|------|------|
| vcpkg：`Could not find android ndk` | 设置 `ANDROID_NDK_HOME` 后重试 `vcpkg install` |
| GMP 配置阶段失败 | 换 NDK 版本（如 27.x / 28.x）；更新 vcpkg：`git pull` + `bootstrap-vcpkg.bat` |
| APK 仍显示 stub 提示 | Sync Gradle → 删 `app/.cxx` → Rebuild；Build 日志须含 `GMP found`；勿只 Run 不重编 native |
| `gradlew` 报 JAVA_HOME | 用 Studio Rebuild，或设置 `org.gradle.java.home` / `JAVA_HOME` 指向 JBR |
| Gradle 未打印 `linking GMP` | 检查 `gradle.properties` / `local.properties` 中 `ecm.android.gmp.root` |
| 仅有 `gmp:x64-windows` | 需单独安装 **`gmp:arm64-android`**，桌面 triplet 不能用于 Android |

---

## 7. 相关文档与代码

| 资源 | 说明 |
|------|------|
| [README_ECM_FACTORIZATION.md](../README_ECM_FACTORIZATION.md) | UI 与桌面 CLI 参数对照、架构概览 |
| `app/src/main/cpp/CMakeLists.txt` | `ECM_ANDROID_GMP_ROOT` 检测与源文件列表 |
| `app/src/main/cpp/ecm_android_run.cpp` | JNI 入口；无 GMP 时 stub |
| `syncEcmStage1Kernels`（`app/build.gradle.kts`） | 构建前同步 `ecm_stage1.cl`、`mont.cl`（self-test）等到 assets |

---

## 8. 安装完成后的检查清单

- [ ] `vcpkg install gmp:arm64-android` 成功结束
- [ ] `include/gmp.h` 与 `lib/libgmp.a` 存在
- [ ] `local.properties` 已设置 `ecm.android.gmp.root`
- [ ] Rebuild 后 CMake 日志含 `GMP found`
- [ ] 真机/模拟器运行 `(2^421-1)` 测试通过
