# Android ECM 分解（Stage-1）

## UI 与桌面命令对应

| 桌面 `ecm.exe` | Android UI |
|----------------|------------|
| stdin `N` 表达式 | **N 表达式** / 预设下拉 |
| `-gpu` | **运行 ECM (-gpu)** 按钮（固定启用 GPU） |
| `-gpucurves N` | **gpucurves** |
| `B1` `B2` 位置参数 | **B1** / **B2** |
| `-d index` | **设备序号** |
| `-v` | **详细输出** 勾选 |
| `--ckpt sec` | 高级 → **checkpoint 秒**（0=禁用） |
| `-sigma value` | 高级 → **sigma**（空=随机） |
| `--mul/--sqr/--add/--sub` | 高级 → 内核路径（可选） |

示例桌面命令：

```powershell
echo '(2^421-1)' | .\build\Debug\ecm.exe -v -d 0 -gpu -gpucurves 64 2000 0
```

Android 默认字段即上述参数；点 **运行 ECM** 即可。

## 启用完整原生分解（需 GMP）

当前 APK 在未链接 GMP 时仍可打开 UI，运行会提示构建说明。

**完整步骤见：[docs/DEV_GMP_SETUP.md](docs/DEV_GMP_SETUP.md)**（vcpkg 安装、`local.properties` 配置、编译验证与排错）。

简要流程：

1. 设置 `ANDROID_NDK_HOME`，执行 `vcpkg install gmp:arm64-android`
2. 在 `local.properties` 增加（路径按本机修改）：
   ```properties
   ecm.android.gmp.root=D\:\\code\\vcpkg\\installed\\arm64-android
   ```
3. Rebuild APK。Gradle 会自动传入 `-DECM_ANDROID_GMP_ROOT=...`；`ecm_stage1.cl` 等由 `syncEcmStage1Kernels` 同步到 assets。

## 架构说明

- `ecm_android_run.cpp`：解析 N、计算 batch_s、调用 `opencl_ecm_stage1`
- `opencl_android_shim.cpp`：将 `cgbn_stage1` 的直接 `cl*` 调用桥接到动态加载的 OpenCL
- `impl_opencl.cpp`：在 `__ANDROID__` 上通过 assets 加载 `ecm_stage1.cl`
- 微基准（Mont / AddSub）仍使用原有 bench 路径，与 ECM 分解独立
