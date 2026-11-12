# CPU Stage1 加权 NTT 探针草稿

诊断入口为 [cpu_stage1_weighted_ntt.cpp](../../tools/bench/cpu_stage1_weighted_ntt.cpp)。首个原型支持完整 M6011 的 320 点与 M12511 的 640 点，使用同一个小于 \(2^{50}\) 的素数域。点位按 \(e_j=\lceil pj/L\rceil\) 划分。对每个循环卷积系数逐项使用真实 digit 上界和 1／2 权重求和，确保单域能无歧义恢复。初始化核对原根和加权根。

PFA 把 320／640 点变换分成五路 64／128 点和五点变换。内核用 IFMA52 实现常数模乘和频域逐点 Montgomery 乘法。逆长度、逆权重及退出 Montgomery 域合并成常数乘法。乘积系数依次向更高位进位，越过 p 位的进位回到首 digit。GMP 对拍包含零、单位元、接近模数的输入和随机输入，依赖链的末点另与 GWNUM 核对。

```powershell
tools/build/test/build_cpu_stage1_ntt_probe.ps1
build_dev/cpu_stage1_weighted_ntt_probe/cpu_stage1_weighted_ntt.exe
```

探针要求 AVX-512F/DQ/IFMA 和本地 `gwnum64.lib`，Windows 下固定逻辑 CPU 1。使用两个可重用的整条算术链作为计时入口。见 [算术耗时与边界](../performance/STAGE1.md#cpu-加权-ntt-算术探针)。

后续验证先分解 PFA 打包、蝶形、逐点乘法和进位耗时。将频谱保留与融合乘加接入点公式前，逐式证明扩大的系数范围。M12511 的 640 点单域最大系数已超过域模数的一半，任意双乘积加减需要调整形状、增大系数余量或改用其他算术路径。完整点步和单核多曲线吞吐作为后续验证。
