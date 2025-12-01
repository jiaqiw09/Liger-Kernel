# 初学者模板：`fused_linear_jsd.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 fused JSD 的数值流程：从 logits 得到 log_softmax，再计算 generalized JSD loss
- 编写与显式计算对比的前向与梯度测试

关键点：
- 温度（temperature）对 softmax 的影响
- shift_labels（可选）用于按 token 指定标签
- jsd_beta 控制 forward/reverse KL 的混合系数

使用：
- 补充 TODO 并运行：`pytest -q src/liger_kernel/ops/fused_linear_jsd_test.py`
"""

import torch
import pytest
import torch.nn.functional as F


def explicit_jsd_loss(student_input, student_weight, teacher_input, teacher_weight, temperature=1.0, jsd_beta=0.5, shift_labels=None, ignore_index=-100):
    # 参考实现步骤（适用于小规模验证）：
    # 1) 计算 logits
    student_logits = (student_input @ student_weight.t()) / temperature
    teacher_logits = (teacher_input @ teacher_weight.t()) / temperature
    # 2) 取 log_softmax
    student_logprob = F.log_softmax(student_logits, dim=-1)
    teacher_logprob = F.log_softmax(teacher_logits, dim=-1)
    # 3) 计算 generalized JSD per-token（简化说明：此处采用实现中使用的公式）
    #    对于完整精确实现，请参考源代码的注释与 _jsd_kernel 的实现
    # 4) 返回总 loss（sum 或 mean，视实现而定）
    # 注意：这里提供一个占位参考，实际测试建议逐项推导并实现精确公式
    return torch.sum((student_logprob - teacher_logprob) ** 2)


def test_forward_vs_explicit():
    # TODO: 构造小规模示例并实现以下步骤：
    # 1) 随机生成 student_input, student_weight, teacher_input, teacher_weight
    # 2) 调用 fused_linear_jsd_forward
    #    from src.liger_kernel.ops.fused_linear_jsd import fused_linear_jsd_forward
    # 3) 用 explicit_jsd_loss（或手动实现更接近源码的 JSD 计算）得到参考 loss
    # 4) 比较两个 loss 的数值是否相近（考虑 float32 的误差）
    assert True


def test_temperature_and_labels():
    # TODO: 测试 temperature 的缩放效果和 shift_labels 的行为
    # - 若 temperature 趋大，概率分布趋平滑；在极值情况下，loss 变化方向应符合预期
    # - shift_labels 传入时，函数应正确处理 ignore_index 并只在非忽略 token 上累加
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/fused_linear_jsd_test.py
