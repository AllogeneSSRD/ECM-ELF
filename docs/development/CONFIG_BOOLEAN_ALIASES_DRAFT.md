# 布尔配置数值别名检查草稿

提交目标版本0.56.4。正式规则见[配置定义与解析](../architecture/CONFIGURATION.md)，逐键范围由[ecm_options.json](../../config/ecm_options.json)生成。本草稿保留开发验证，完成后由用户决定是否清理。

## 范围与实现

schema声明为bool或bool|string的键接受整数0/1，与false/true等价；三态键保留字符串auto。全局、worker覆盖、Stage1/Stage2和GUI共用相同读取器。普通整数、字符串等配置类型的校验不放宽。整数2、−1和浮点0.0/1.0等不是布尔别名。

读取时仅转换业务值，保留原始行、注释和诊断右值。GUI编辑器对数值和字面量使用同一语义，未修改值保持原始数字拼写；实际编辑后保存为布尔字面量。非法编辑值不能被静默转换为false。默认模板仍生成true/false，生成说明与错误范围同时列出数值别名。

## 验证

- `tools/test/test_ecm_toml.ps1`连接实际生成绑定及GUI编辑器，533项schema驱动别名检查通过，覆盖全部布尔键、全局/worker、严格/计算读取、符号整数零/一和非法值。另检查三态、worker优先级、原行保持、GUI编辑及逐行错误排版。
- 原始拒绝0的复现保留于`build_dev/config_bool_alias_0_56_4/red/`，修正后的检查位于同目录`green/`，补充普通类型隔离和非法全局值不能被worker覆盖掩盖的最终检查位于`final/`。
- `tools/test/test_ecm_config_bool_domains.py`三项检查通过；配置和GUI生成器的过期检查通过。
- 正式本机Stage2构建44单元前／后校验通过；版本0.56.4，`build_stage2_local/ecm_cuda_stage2.exe`的SHA256为`e2f8891da7d717f7df8c32d41974c4710aeb3aad693b4fe2158e39fe5d3e0dfd`，198项来源哈希核对通过。
- 最终二进制通过14项公开CLI/TOML配置检查和真实隔离队列子进程auto覆盖，包含数值0/1、非法整数2、浮点1.0及CLI优先级。证据为`data/experiments/config_bool_alias_20261101/public_config_0_56_4/`。没有改动GPU设置、生产配置或队列，无需重跑算术性能矩阵。
