# 初学者模板：`fused_linear_cross_entropy.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解如何把最后一层的线性变换与交叉熵融合以避免物化大型 logits 矩阵
- 学会为融合实现编写功能和梯度测试（与显式 matmul + cross_entropy 对比）

主要符号（源码可见）：
- fused_linear_cross_entropy_forward
- fused_linear_cross_entropy_backward
- LigerFusedLinearCrossEntropyFunction

测试要点：
1) 前向数值对比：对于小规模输入，计算显式 logits = input @ weight.T + bias，然后用 torch.nn.functional.cross_entropy 得到参考 loss；与 fused 函数的 loss 比较（考虑 reduction、label_smoothing 等）
2) 梯度对比：验证 fused 返回的 grad_input、grad_weight、grad_bias 与显式实现一致
3) use_token_scaling 的效果：当启用时，loss 会乘以预测概率，检查与显式逐步实现的对比
4) chunking/分块逻辑：通过设置不同的 BT/H/V 值，确保分块逻辑不会改变数值输出

使用说明：
- 补齐下面的 TODO 并运行：`pytest -q src/liger_kernel/ops/fused_linear_cross_entropy_test.py`
"""

import torch
import torch.nn.functional as F
import pytest


def explicit_linear_cross_entropy(input_, weight, target, bias=None, ce_weight=None, reduction='mean'):
    # 参考实现：显式计算 logits，然后使用 torch 的 cross_entropy
    logits = input_ @ weight.t()
    if bias is not None:
        logits = logits + bias
    loss = F.cross_entropy(logits, target, weight=ce_weight, reduction=reduction)
    return loss, logits


def test_forward_backward_compare_explicit():
    # TODO:
    # 1) 构造小规模示例: BT=4, H=8, V=6
    # 2) 随机初始化 input (requires_grad=True), weight (requires_grad=True), bias
    # 3) 调用 fused_linear_cross_entropy_forward, 以及 explicit_linear_cross_entropy
    # 4) 对比 loss 值（注意 reduction）
    # 5) 若需要，比较 grad_input、grad_weight、grad_bias（可通过对 loss.backward() 并读取梯度实现）
    assert True


def test_use_token_scaling_behavior():
    # TODO: 构建一个示例来验证 use_token_scaling 的行为是否与描述一致
    # 1) 计算显式 logits 的预测概率（softmax），提取正确 target 的概率
    # 2) 将显式计算得到的 per-token loss 乘以预测概率，再与 fused 实现进行对比
    assert True


def test_chunking_and_shapes():
    # TODO: 测试当 BT 很大或 V >> H 时，分块逻辑不改变数值结果（在小规模下模拟 chunk）
    # - 可以通过人为设置 BT、H、V 并确认 fused 与显式实现的结果一致
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/fused_linear_cross_entropy_test.py
