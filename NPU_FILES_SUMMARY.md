# NPU 测试文件总览

为 NPU 适配创建的测试文件汇总。

## 📁 创建的文件

### 1. `test_npu_ops.py` - 完整测试套件 ⭐

**路径**: `/Users/humphrey/Documents/github/Liger-Kernel/test_npu_ops.py`

**说明**: 全面的 NPU 算子测试套件，包含详细的验证逻辑

**特点**:
- ✅ 9 个核心算子完整测试
- ✅ Forward 和 Backward 都有测试
- ✅ 与 PyTorch 原生实现对比验证
- ✅ 详细的误差报告
- ✅ 异常处理和错误提示

**测试的算子**:
1. SwiGLU - SiLU 门控激活
2. GeGLU - GELU 门控激活
3. RMSNorm - RMS 归一化
4. LayerNorm - 层归一化
5. RoPE - 旋转位置编码
6. Softmax - Softmax 激活
7. CrossEntropy - 交叉熵损失
8. KL Divergence - KL 散度
9. JSD - Jensen-Shannon 散度

**使用**:
```bash
python test_npu_ops.py
```

---

### 2. `test_npu_quick.py` - 快速验证脚本 ⚡

**路径**: `/Users/humphrey/Documents/github/Liger-Kernel/test_npu_quick.py`

**说明**: 简化版快速测试，用于快速验证算子是否能运行

**特点**:
- ✅ 简洁的测试逻辑
- ✅ 快速执行
- ✅ 一目了然的结果
- ✅ 适合 CI/CD

**使用**:
```bash
python test_npu_quick.py
```

**输出示例**:
```
==================================================
NPU 快速测试 (设备: npu)
==================================================
✓ SwiGLU              通过
✓ GeGLU               通过
✓ RMSNorm             通过
...
==================================================
结果: 9/9 通过
==================================================
```

---

### 3. `NPU_TEST_README.md` - 使用文档 📖

**路径**: `/Users/humphrey/Documents/github/Liger-Kernel/NPU_TEST_README.md`

**说明**: 详细的使用说明和调试指南

**内容包括**:
- 测试算子列表
- 使用方法
- 配置修改指南
- 输出示例
- 测试逻辑说明
- 调试建议
- 常见问题解答

---

## 🚀 快速开始

### 第一步：修改设备名

在两个测试文件中，将 `DEVICE = "npu"` 改为你的实际 NPU 设备名：

```python
# test_npu_ops.py 第 9 行
DEVICE = "npu"  # 改成你的设备，如 "npu:0", "ascend" 等

# test_npu_quick.py 第 8 行  
DEVICE = "npu"  # 改成你的设备，如 "npu:0", "ascend" 等
```

### 第二步：运行快速测试

```bash
# 先运行快速测试，看看哪些能跑通
python test_npu_quick.py
```

### 第三步：运行完整测试

```bash
# 然后运行完整测试，查看详细误差
python test_npu_ops.py
```

---

## 📊 测试覆盖范围

| 算子类型 | 算子名称 | Forward | Backward | 验证方式 |
|---------|---------|---------|----------|---------|
| 激活函数 | SwiGLU | ✅ | ✅ | vs PyTorch |
| 激活函数 | GeGLU | ✅ | ✅ | vs PyTorch |
| 归一化 | RMSNorm | ✅ | ✅ | vs PyTorch |
| 归一化 | LayerNorm | ✅ | ✅ | vs PyTorch |
| 位置编码 | RoPE | ✅ | ✅ | 形状验证 |
| 激活函数 | Softmax | ✅ | ✅ | vs PyTorch |
| 损失函数 | CrossEntropy | ✅ | ✅ | vs PyTorch |
| 损失函数 | KL Divergence | ✅ | ✅ | vs PyTorch |
| 损失函数 | JSD | ✅ | ✅ | 非负性验证 |

---

## 🔧 配置选项

### 在 `test_npu_ops.py` 中修改：

```python
# 第 9 行：设备名
DEVICE = "npu"

# 第 10 行：数据类型
DTYPE = torch.float32  # 或 torch.float16, torch.bfloat16

# 第 11 行：误差容差
TOLERANCE = 1e-5  # float32: 1e-5, float16: 1e-3
```

---

## 💡 使用建议

### 1. 首次测试流程

```bash
# 1. 先用快速测试看整体情况
python test_npu_quick.py

# 2. 如果有失败，用完整测试查看详细错误
python test_npu_ops.py

# 3. 针对失败的算子单独调试
python -c "from test_npu_ops import test_swiglu; test_swiglu()"
```

### 2. 不同精度测试

```python
# 修改 DTYPE 和 TOLERANCE
# float32 (高精度)
DTYPE = torch.float32
TOLERANCE = 1e-5

# float16 (低精度，更快)
DTYPE = torch.float16
TOLERANCE = 1e-3
```

### 3. 单独测试某个算子

```python
# 在 Python 中导入
from test_npu_ops import test_swiglu, test_rms_norm

# 测试 SwiGLU
test_swiglu()

# 测试 RMSNorm
test_rms_norm()
```

---

## ⚠️ 已知问题和限制

1. **Triton 依赖**: 所有算子都使用 Triton，如果 NPU 不支持 Triton，需要适配
2. **精度差异**: 不同硬件的浮点运算可能有微小差异
3. **内存限制**: 某些测试使用较大的张量，可能需要调整 batch size

---

## 📝 测试输出说明

### 成功输出
```
✓ SwiGLU Forward: 误差=1.192093e-07
✓ SwiGLU Backward: 误差=2.384186e-07
```

### 失败输出
```
✗ SwiGLU Forward: 误差=1.000000e+00
✗ SwiGLU 测试失败: RuntimeError: CUDA error...
```

---

## 🐛 调试技巧

### 1. 查看具体错误

```python
import traceback
try:
    test_swiglu()
except Exception as e:
    traceback.print_exc()
```

### 2. 减小张量大小

```python
# 在测试函数中修改 shape
shape = (1, 2, 64)  # 改小一点
```

### 3. 关闭梯度计算

```python
with torch.no_grad():
    # 只测试 forward
    pass
```

---

## 📞 获取帮助

遇到问题时，请提供：
1. NPU 型号和驱动版本
2. PyTorch 版本 (`torch.__version__`)
3. 完整错误日志
4. 运行的命令
5. 失败的算子名称

---

## 🎯 下一步

1. ✅ 运行测试，确认哪些算子能正常工作
2. ✅ 记录失败的算子
3. ✅ 分析失败原因（Triton 支持、精度问题等）
4. ✅ 根据需要进行针对性适配
5. ✅ 完成后可以集成到正式的测试套件中

---

## 版本信息

- 创建日期: 2025-01-06
- 适用 Liger-Kernel 版本: main branch
- Python 版本要求: >= 3.8
- PyTorch 版本要求: >= 2.0（需支持 NPU）

