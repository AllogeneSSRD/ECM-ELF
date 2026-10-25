# ECM OpenCL Android 开发文档

在真机上探测 GPU OpenCL，并运行与桌面版同源的 **ECM add/sub** 微基准（`opencl_ecm_addsub` 内核族）。

## 环境要求

| 项目 | 要求 |
|------|------|
| Android Studio | 近期稳定版，NDK r26+ |
| 设备 | **arm64-v8a** 真机，厂商提供 `libOpenCL.so`（常见 Adreno / Mali） |
| minSdk | 31（`uses-native-library` 加载系统 OpenCL） |
| 仓库位置 | 本模块位于 `MPA-OpenCl/Android/ECM`，构建时需能访问上级目录中的 OpenCL 内核与 manifest |

## 基本用法

1. 用 Android Studio 打开 **`Android/ECM`**（不是仓库根目录）。
2. 确认 **`jniLibs/` 内没有** 从手机 `adb pull` 的 `libOpenCL.so`（16 KB 页设备会因对齐错误崩溃）。
3. 连接真机，选择 **arm64-v8a** 构建并 Run。
4. 应用启动后会自动做一次 **设备探测**；结果在底部等宽文本区。

### 界面功能

| 区域 | 说明 |
|------|------|
| **设备性能**（顶部卡片） | 独立后台线程约每 1.5s 刷新：GPU 占用/频率优先读 **高通系统属性**（`vendor.gpu.*`，不受 SELinux 限制），再回退 sysfs；显示数据源 |
| 设备探测 | 枚举平台/设备、读写 buffer 冒烟测试 |
| 简短测试 | GPU 名称 + buffer ping |
| **ECM add/sub 微基准** | 32-bit limb 全路径；**24-bit limb** 仅 `fused` + `fused_unroll` |
| **ECM mont mul/sqr** | 与 `opencl_ecm_montsqr.exe` 同源（WG mode, tpi=4）；**不含 AMD asm** |
| Limb add-mod 微基准 | 独立的小型 16/24/32-bit 单内核测试（非 ECM 完整路径） |

### add/sub 四个参数

与 Windows 命令行一一对应：

```text
opencl_ecm_addsub.exe [--bits <bits>] <kernel_iterations> <instances> <launch_repeats>
```

| UI 字段 | 含义 | 默认（手机友好） | 桌面常见示例 |
|---------|------|----------------|--------------|
| bits | 模数位宽；32-bit limb 须为 32 的倍数；24-bit limb 须为 24 的倍数 | 512 / 504 | 512 / 504 |
| kernel_iterations | 单次 launch 内核内循环次数 | 1000 | 10000 |
| instances | 并行实例数（global size） | 64 | 128 |
| launch_repeats | 重复 enqueue 次数 | 3 | 3 |

总运算量：`instances × kernel_iterations × launch_repeats`。

### mont mul/sqr

对应桌面：

```text
opencl_ecm_montsqr.exe --bits 512 <kernel_iterations> <instances> <launch_repeats>
```

默认 **WG mode**、`tpi=4`（与桌面默认一致）。512-bit 会跑完整路径列表（`priv`/`priv_opt`、`unroll_only_512*`、`fips512*`、`local_only_512`、`opt2_512_local`、`unroll32/64`、`mont_*_wg` 等）；**跳过** `*_asm` 与 4096 专用路径。

输出开头有 **`--- planned 512-bit mont paths ---`**（mul≈15、sqr≈14 条）；未跑的路径会打印 **`skipped (...)`** 原因（limb 不匹配、`clCreateKernel` 失败、enqueue 失败）。末尾 **`--- summary ---`** 统计 ran/skipped。

首次编译 `mont_priv*.cl` 体积大，手机上可能需要 **1–3 分钟**。二次运行走 **OpenCL 二进制缓存**（与桌面 `CGBN_OPENCL_CACHE` 同算法）。

### OpenCL 编译缓存

与 Windows `cgbn::opencl::build_program_from_source` 一致：

- 缓存目录：`/sdcard/Android/data/com.example.ecm/opencl_cache/opencl_{fnv1a64}.bin`（外部应用目录，便于文件管理器 / adb 查看）
- 缓存键：GPU 名称/厂商/驱动版本 + `build_opts` + 完整拼接源码
- 命中时：`clCreateProgramWithBinary` + `clBuildProgram`（通常远快于全量编译）
- 输出含 `compile: cache hit ... ms` 与 `cache: ...` 路径

桌面可通过 `CGBN_OPENCL_CACHE_DIR` / `CGBN_OPENCL_CACHE_DISABLE` 控制；Android 默认启用（`nativeInitAssets` 传入 `AppStoragePaths.openClCacheRoot()`）。

部分手机 GPU 驱动**不支持导出** OpenCL program binary（`CL_PROGRAM_BINARIES` 恒失败）。此时自动改用 **live program cache**：在同一 App 进程内保留已编译的 `cl_program` + 持久 `cl_context`，第二次跑相同 bench 跳过 `clBuildProgram`（**杀进程后仍需重编译**）。

**检查手机上的缓存目录：**

1. App 内点 **OpenCL 探测**：输出顶部有 `cache_dir:` 与 `.bin` 文件列表。
2. 任意 bench 输出含 `cache_enabled:`、`cache_key:`、`cache save:` 或 `compile: cache hit`。
3. adb（debug 包）：

```bash
adb shell run-as com.example.ecm ls -la code_cache/opencl_cache/
adb logcat ECM-OpenCL:I *:S
```

启动 App 后 logcat 应立刻出现 `OpenCL cache root: /data/user/0/com.example.ecm/code_cache`；首次编译后有 `OpenCL cache save:`。

输出中 **ms** 为固定小数；**ops/s ≥ 1e6** 时使用科学计数法（与 Windows `opencl_ecm_addsub` 一致，例如 `1.2224e+07 ops/s`）。

**24-bit limb（Adreno）**：点「24-bit limb（fused + unroll）」；`bits` 建议 **504**（21 limb）或 **288**（12 limb，与 384@32 公平对比）。每个 `uint` 低 24 位为 limb。

**为何 ECM bench 可能看不出 CLPeak 的 3× 优势：**

1. **口径不同**：CLPeak 24-bit 是单元素、内核内 hot loop；ECM bench 默认每次 enqueue 从 global 重载全部 limb。
2. **limb 数不等**：同 bit 宽度下 24-bit limb 更多（384@24=16 limb，384@32=12 limb，多约 33% 运算）。
3. **OpenCL 无 `add24`**：仅 `mul24`/`mad24` 保证 24-bit 路径；加法依赖编译器对「值域 &lt; 2²⁴」的推断，显式 `& 0xFFFFFF` 会阻碍优化。

**改进路径（已实现）**：

- 24-bit：去掉 hot path mask；`fused_hot` / `fused_unroll_hot` 内核内 `inner=kernel_iterations`。
- 32-bit：同样提供 `fused_hot`、`fused_unroll_auto_hot`（及 sub 对应 hot）；冷路径仍为每次 enqueue 重载 global。
- 公平 limb 数：**288-bit@24** vs **384-bit@32**（12 limb）；hot 段 ops/s 可与 CLPeak 口径对照。

## 与 Windows 的复用关系

```text
MPA-OpenCl/
├── cgbn/backends/opencl/kernels/          ← 复用：OpenCL 内核源码
│   ├── ecm_addsub_bench.cl
│   └── mp_addsub/generated/*.cl
├── include/opencl_ecm_addsub_manifest.h   ← 复用：内核列表与构建清单 API
├── tools/bench/opencl_ecm_addsub_manifest.cpp ← 复用：manifest 实现（CMake 链入 libecm.so）
└── tools/bench/opencl_ecm_addsub_bench.cpp    ← 桌面专用（GMP + cgbn::opencl），Android 不链接
```

### 从 Windows 复用的部分

| 资源 | 用途 |
|------|------|
| `ecm_addsub_bench.cl` | 基础 add/sub/mod 内核与 bench entry（32-bit limb） |
| `mp_addsub/limb24_addsub.cl` | 24-bit limb `fused` / `fused_unroll`（Adreno 24-bit ALU） |
| `mp_addsub/generated/*.cl` | `fused_unroll`、`fused_unroll_auto`、priv 等展开实现 |
| `opencl_ecm_addsub_manifest.*` | 决定编译拼接哪些 `.cl`、bench 跑哪些 kernel 名 |
| 内核算法与路径命名 | `fused_unroll`、`fused_unroll_auto`、`legacy`、`mask` 等与桌面一致 |

Gradle 任务 **`syncAddsubKernels`** 在每次 `preBuild` 时把上述内核复制到：

`app/src/main/assets/kernels/cgbn/backends/opencl/kernels/`

也可手动执行：

```bash
./gradlew :app:syncAddsubKernels
```

### Android 独立实现的部分

| 组件 | 说明 |
|------|------|
| `app/src/main/cpp/opencl_loader.*` | `dlopen` 系统 `libOpenCL.so`，不依赖 Khronos 安装版 ICD |
| `app/src/main/cpp/opencl_probe.cpp` | 探测与简短测试 |
| `app/src/main/cpp/opencl_bench.cpp` | 16/24/32-bit limb 微基准（教学用，非 ECM manifest） |
| `app/src/main/cpp/opencl_addsub_bench.cpp` | ECM add/sub bench 宿主逻辑：读 assets、编译、计时、输出 |
| `app/src/main/cpp/kernel_assets.cpp` | 从 APK `AssetManager` 加载 `.cl` 文本 |
| `MainActivity.kt` + `activity_main.xml` | UI、参数输入、后台线程调 JNI |
| `AndroidManifest.xml` | `uses-native-library libOpenCL.so`；不打包 vendor `.so` |

与桌面的主要差异：

- **无 GMP**：正确性用主机端多精度 limb 例程校验 `ecm_mp_add_mod_fused`。
- **无 AMD GPU asm 路径**：`asm_enabled=false`，不编译 `mp_addsub/asm_*.cl`（手机为 Adreno/Mali）。
- **OpenCL 运行时**：动态加载 + 自声明 `cl_*` 常量；桌面用 `cgbn::opencl` + 链接 `OpenCL.lib`。
- **无** `opencl_ecm_montsqr` / mont mul/sqr bench（仍在桌面 `opencl_ecm_montsqr.exe`）。

## 工程结构（简要）

```text
Android/ECM/
├── app/
│   ├── build.gradle.kts      # syncAddsubKernels、abiFilters、禁止打包 libOpenCL
│   ├── src/main/
│   │   ├── AndroidManifest.xml
│   │   ├── assets/kernels/...   # 构建时同步的内核（勿手改，会被覆盖）
│   │   ├── cpp/
│   │   │   ├── CMakeLists.txt   # 链接 manifest.cpp，MPA_ROOT 指仓库根
│   │   │   ├── native-lib.cpp   # JNI
│   │   │   └── opencl_addsub_bench.cpp
│   │   ├── java/.../MainActivity.kt
│   │   └── res/layout/activity_main.xml
└── README.md                 # 本文档
```

## 构建注意

- **不要** 将 `adb pull` 的 `libOpenCL.so` 放入 `jniLibs`。
- 修改仓库内 `cgbn/.../kernels` 后需重新构建 APK（或先跑 `syncAddsubKernels`）。
- 大位宽（如 4096-bit）首次 OpenCL 编译可能需数十秒至数分钟，属正常现象。
- 日志：`adb logcat -s ECM-OpenCL`

## GPU 占用 / 频率（Adreno）

读取顺序：

1. **sysfs**（`kgsl/gpubusy`、`gpuclk`）优先——Adreno 642 高负载时准确；空闲时可能读不到（显示 `—`）。
2. **系统属性白名单**（仅 `vendor.gpu.freq` 等显式列表，**不**扫描全表，避免 MIUI `persist.sys.computility.gpulevel` 误当占用率）。
3. 当前频率不可读时，尝试显示 **max_gpuclk** 为 `— (max xxx MHz)`。

8 Elite (830) 上 `getprop` 无 `vendor.gpu.freq`，且 sysfs 常被 SELinux 拒绝，占用/频率可能长期为 `—`，属机型限制。

面板会显示 `GPU 数据源: 占用=系统属性, 频率=系统属性` 等来源标签。

若仍为 `—`，在电脑上查找本机属性名并反馈或自行加入 `sys_gpu_stats.cpp` 的 `kFreqProps`：

```bash
adb shell getprop | grep -i gpu
adb logcat -s ECM-GPUStats
```

## 输出示例

```text
=== ECM add/sub microbench ===
512-bit, kernel_iterations=10000, instances=128, launch_repeats=1
device: QUALCOMM Adreno(TM) 830
compile: 227.663 ms
verify ecm_mp_add_mod_fused: PASS

--- mp_add_mod ---
fused_unroll: 104.709 ms, 1.2224e+07 ops/s
...

RESULT: PASS
```

## 相关桌面工具

| 可执行文件 | 作用 |
|------------|------|
| `build/Debug/opencl_ecm_addsub.exe` | 完整 ECM add/sub bench（GMP 校验、ASM、LPT 等） |
| `build/Debug/opencl_ecm_montsqr.exe` | Montgomery mul/sqr bench（Android 未移植） |
| `build/Debug/ecm.exe` | 完整 ECM stage1 驱动 |

更上层的 Android 总览见 [`../README.md`](../README.md)。
