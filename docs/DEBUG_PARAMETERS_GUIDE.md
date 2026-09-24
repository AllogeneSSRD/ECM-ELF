# cgbn_ecm_stage1 参数调试指南

## 修改位置
[ecm/cudawrapper.c](ecm/cudawrapper.c) - gpu_ecm() 函数，第 ~387 行

## 调试输出说明

添加了**调用前和调用后**的完整参数打印，使用 `OUTPUT_VERBOSE` 宏输出。

### 关键改进

- ✅ **调用前**：打印输入参数值（而非仅地址）
- ✅ **调用后**：打印 GPU 执行结果
- ✅ **因子内容**：十六进制显示发现的因子（前 8 条曲线）
- ✅ **可读性**：秒数、比特数等易于理解的格式

### 运行方式

启用详细调试输出（需要加 `-v` 参数）：
```bash
echo '(2^991-1)' | ./ecm -v -gpu -gpucurves 32 11e6 0
```

### 输出样例 - 调用前

```
=== cgbn_ecm_stage1 BEFORE CALL ===
n (bit-size: 959): 0x7fffffffffffffffffffffffffffffffffffffff...
batch_s (bit-size: 15869673 bits)
  Value: 0x2... (huge product of primes)
nb_curves: 32
firstsigma_ui (initial): 1234567890 (0x499602d2)
gpu_checkpoint_interval_ms: 600000 (600.0 seconds)
verbose: 1
Allocated arrays: factors=0x7fff0000, array_found=0x7fff0800
-----------------------------------
```

### 输出样例 - 调用后

```
=== cgbn_ecm_stage1 AFTER CALL ===
Return value (youpi): 0
GPU execution time: 2345.67 ms
firstsigma_ui (after): 1234567890 (0x499602d2)
Factors found (first 8 curves):
  [Curve 0] Found: 0x3d... (status=1, 128 bits)
  [Curve 1] No factor found
  [Curve 2] Found: 0x7f... (status=1, 512 bits)
  [Curve 3] No factor found
  ... and 28 more curves
-----------------------------------
```

### 参数详解 - 调用前

| 参数 | 类型 | 说明 | 调试用途 |
|------|------|------|---------|
| **n** | `mpz_t` | 要分解的数（十六进制） | 确认输入数值正确 |
| **batch_s** | `mpz_t` | 批积分（十六进制 + 比特数） | 验证 s 的计算（位数很重要） |
| **nb_curves** | `uint` | 并行曲线数 | 确认 GPU 曲线配置 |
| **firstsigma_ui** | `uint32_t` | 首个 sigma 参数值 | 校验曲线参数生成 |
| **checkpoint_interval** | `ulong` | 检查点保存间隔 | 验证恢复机制配置 |
| **verbose** | `int` | 详细输出级别 | 调试信息过滤 |
| **factors / array_found** | 指针 | 输出数组内存地址 | 验证内存分配 |

### 参数详解 - 调用后

| 参数 | 说明 | 调试用途 |
|------|------|---------|
| **Return value (youpi)** | 函数返回值 | 0 = 成功，非 0 = 错误代码 |
| **GPU execution time** | GPU 实际执行时间（毫秒） | 性能测量、性能优化 |
| **firstsigma_ui (after)** | sigma 值是否被修改 | 如与之前不同，表示曲线被替换 |
| **Factors found** | 前 8 条曲线的因子结果 | 十六进制值 + 比特数 + 状态 |

## 调试信息解读

### 调用前检查列表

✅ **n 值验证**
```
n (bit-size: 959): 0x7fff...
```
- 比特数与预期相符（(2^991-1)/(8218291649) ≈ 959 位）
- 十六进制以高位数字开头（0x7f, 0xff）

✅ **batch_s 验证**
```
batch_s (bit-size: 15869673 bits)
  Value: 0x2...
```
- 比特数应该是 B1 的约 log(B1) 倍
- 例如：B1=11e6 → 约 1500 万位

✅ **配置参数验证**
```
nb_curves: 32              # 2^5 = 32 条曲线（推荐 16-256）
firstsigma_ui: 1234567890  # 初始椭圆曲线参数
checkpoint_interval: 600000 ms = 600 seconds  # 10 分钟
```

### 调用后检查列表

✅ **返回值**
```
Return value (youpi): 0    # 0 = 成功，非 0 = 错误
```

✅ **性能指标**
```
GPU execution time: 2345.67 ms  # 大约 2.3 秒
```
- 用于评估 GPU 性能
- 预期：32 条曲线 @ 100-200ms per curve

✅ **因子结果**
```
[Curve 0] Found: 0x3d... (status=1, 128 bits)
[Curve 2] Found: 0x7f... (status=1, 512 bits)
[Curve 1] No factor found
```
- status=1 表示找到因子
- 显示十六进制值和比特数

## 常见问题排查

### 问题 1: batch_s 计算异常

症状：
```
batch_s (bit-size: 1000000 bits)  # 异常大或异常小
```

诊断：
- B1 > 1e8：bit-size 可能 > 2000 万
- B1 < 1e4：bit-size 可能 < 100 万

解决方案：
```bash
# 验证 B1 参数
echo '(2^991-1)' | ./ecm -v -gpu -gpucurves 32 <B1_VALUE> 0
```

### 问题 2: 返回值非零（GPU 错误）

症状：
```
Return value (youpi): -1  # 非零表示错误
```

排查步骤：
1. 检查 n 值是否正确（比特数、十六进制格式）
2. 检查 batch_s 是否计算正确
3. 检查 nb_curves 是否合理（1-256）
4. 查看 OpenCL 编译错误

### 问题 3: GPU 执行时间异常

症状：
```
GPU execution time: 0.01 ms  # 太快
GPU execution time: 60000 ms  # 太慢
```

分析：
- **< 1 ms**：GPU 未执行或内核立即返回
- **> 10s**：性能问题或内存泄漏

解决方案：
- 检查 OpenCL 内核是否正确编译
- 验证 GPU 驱动程序
- 检查内存分配大小

## 与 gmp-ecm 原始版本对比

原始版本（gmp-ecm）的 gpu_ecm() 没有这些调试输出。

新增调试信息的用处：
1. **验证参数传递**：确保 ecm_driver 正确计算 batch_s
2. **性能诊断**：查看曲线数和批积分的平衡
3. **问题复现**：记录完整的参数配置便于复现问题
4. **集成测试**：在 CI/CD 中验证参数正确性

## 输出示例（完整流程）

### 样例 1：正常运行

**输入**：
```bash
echo '(2^991-1)' | ./ecm -v -gpu -gpucurves 32 11e6 0
```

**调用前输出**：
```
=== cgbn_ecm_stage1 BEFORE CALL ===
n (bit-size: 991): 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff...
batch_s (bit-size: 15869673 bits)
  Value: 0x2f4d... (batch product computed)
nb_curves: 32
firstsigma_ui (initial): 2147483647 (0x7fffffff)
gpu_checkpoint_interval_ms: 0 (0.0 seconds)
verbose: 1
Allocated arrays: factors=0x55500000, array_found=0x55501000
-----------------------------------
```

**调用后输出**：
```
=== cgbn_ecm_stage1 AFTER CALL ===
Return value (youpi): 0
GPU execution time: 1234.56 ms
firstsigma_ui (after): 2147483647 (0x7fffffff)
Factors found (first 8 curves):
  [Curve 0] Found: 0x3d7c88b02d (status=1, 37 bits)
  [Curve 1] No factor found
  [Curve 2] Found: 0x5a3f... (status=1, 128 bits)
  [Curve 3] No factor found
  [Curve 4] No factor found
  [Curve 5] No factor found
  [Curve 6] No factor found
  [Curve 7] No factor found
  ... and 24 more curves
-----------------------------------
```

### 样例 2：无因子发现

**特征**：
```
=== cgbn_ecm_stage1 AFTER CALL ===
Return value (youpi): 0
GPU execution time: 2000.00 ms
Factors found (first 8 curves):
  [Curve 0] No factor found
  [Curve 1] No factor found
  ... (所有都是 No factor found)
```

**含义**：
- GPU 正常运行（返回值为 0）
- 没有找到因子（需要增加 B1 或尝试 Stage 2）

### 样例 3：GPU 执行失败

**特征**：
```
=== cgbn_ecm_stage1 AFTER CALL ===
Return value (youpi): -1
GPU execution time: 0.00 ms
```

**排查**：
1. 检查 OpenCL 初始化
2. 验证 GPU 驱动
3. 查看内核编译错误

## 后续步骤

### 如何使用调试信息

1. **记录基准数据**
   ```bash
   # 运行测试表达式并保存输出
   echo '(2^991-1)*17' | ./ecm -v -gpu -gpucurves 32 1e4 0 > debug_output.txt
   ```

2. **对比预期值**
   - n 的比特数
   - batch_s 的比特数（应约为 B1 的 log(B1) 倍）
   - GPU 时间（与硬件性能对比）

3. **调试工作流**
   - 获取 BEFORE 和 AFTER 的完整输出
   - 检查是否有错误代码（youpi ≠ 0）
   - 验证因子是否正确（使用 gmp 验证 n % factor == 0）

4. **性能优化**
   - 监控 GPU execution time
   - 调整 nb_curves（太小 < 16 或太大 > 256）
   - 检查 batch_s 大小与内存分配的平衡

### 相关命令参考

```bash
# 基础测试（简单表达式）
echo '15' | ./ecm -v -gpu -gpucurves 4 1e4 0

# 中等复杂度（大表达式）
echo '(2^991-1)' | ./ecm -v -gpu -gpucurves 32 11e6 0

# 高负载测试（更大的表达式）
echo '(2^1279-1)' | ./ecm -v -gpu -gpucurves 64 1e7 0

# 禁用检查点（加快测试）
echo '(2^256-1)' | ./ecm -v -gpu -gpucurves 16 1e5 0

# 启用检查点（每 10 分钟保存）
echo '(2^2048-1)' | ./ecm -v -gpu --ckpt 600 -gpucurves 128 1e8 0
```

---

**相关文件**:
- [ecm/cudawrapper.c](ecm/cudawrapper.c#L388) - 调试代码位置
- [src/ecm_driver.cpp](../src/ecm_driver.cpp) - 表达式解析和 batch_s 计算
- [ECM_GPU_FLOW.md](ECM_GPU_FLOW.md) - GPU 流程详解
