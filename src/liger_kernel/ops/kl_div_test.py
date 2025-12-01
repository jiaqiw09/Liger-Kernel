# 初学者模板：`kl_div.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 KLDiv 的两种目标表示（`log_target=True/False`）以及不同 reduction 模式的语义
- 为前向与后向实现编写数值对比测试（与 PyTorch 的实现或手工公式对比）

测试要点：
- 对比不同 reduction：'none'、'sum'、'mean'、'batchmean'
- 测试 log_target=True 与 log_target=False 的行为
- 后向：验证 `kldiv_backward_triton` 的输出与手动计算的导数一致（例如对 loss 关于 input 的偏导）

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/kl_div_test.py`
"""

import torch
import pytest
import torch.nn.functional as F


def manual_kldiv(y_pred_log, y_true, log_target=False, reduction='batchmean', eps=1e-10):
    # y_pred_log: log Q
    # y_true: either P (prob) or log P depending on log_target
    if not log_target:
        # loss = target * (log(target + eps) - input)
        loss_mat = y_true * (torch.log(torch.clamp(y_true, min=eps)) - y_pred_log)
    else:
        # loss = exp(target) * (target - input)
        loss_mat = torch.exp(y_true) * (y_true - y_pred_log)

    BT, V = loss_mat.shape
    if reduction == 'none':
        return loss_mat
    elif reduction == 'sum':
        return loss_mat.sum(dim=1)
    elif reduction == 'mean':
        return loss_mat.sum() / (BT * V)
    elif reduction == 'batchmean':
        return loss_mat.sum() / BT
    else:
        raise ValueError('unknown reduction')


def test_forward_against_manual():
    # TODO:
    # 1) 构造 small-scale logits (y_pred_log) 和 y_true（作为 prob 或 log prob）
    # 2) 对于每种 reduction 与 log_target 组合，调用 kldiv_forward_triton 并与 manual_kldiv 比较
    assert True


def test_backward_shape_and_scaling():
    # TODO: 验证 kldiv_backward_triton 对 grad_output 的缩放逻辑以及 reduction 导致的最终缩放
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/kl_div_test.py
