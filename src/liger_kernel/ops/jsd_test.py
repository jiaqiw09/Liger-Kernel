# 初学者模板：`jsd.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 generalized JSD 的三种情形（beta=0, beta=1, 0<beta<1）的数值表达与梯度
- 编写小规模手工实现并与 `jsd_forward` / `_jsd_kernel` 对比

测试要点：
- beta=0（forward KL）与 beta=1（reverse KL）的特殊形式
- 通用 beta 的实现（需要正确处理数值稳定性：logsumexp、shift）
- has_label/ignore_index 逻辑：当 label 为 ignore_index 时应跳过
- backward：验证 `jsd_backward` 对 grad_output 的缩放行为

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/jsd_test.py`
"""

import torch
import pytest
import torch.nn.functional as F


def manual_jsd_forward(X_log, Y_log, beta, ignore_index=None, labels=None):
    # X_log, Y_log: shape (BT, V) in logspace
    # labels: optional (BT,) specifying an index to ignore per row
    # 返回 sum(loss), dX (shape same as inputs)
    BT, V = X_log.shape
    loss = torch.zeros(BT, V, dtype=torch.float32, device=X_log.device)
    dX = torch.zeros_like(X_log)
    for i in range(BT):
        # 如果有 labels 且等于 ignore_index，跳过
        if labels is not None and labels[i].item() == ignore_index:
            continue
        X = X_log[i]
        Y = Y_log[i]
        if beta == 0.0:
            # forward KL: sum( exp(Y) * (Y - X) )
            Y_prob = torch.exp(Y)
            l = Y_prob * (Y - X)
            dx = -Y_prob
        elif beta == 1.0:
            X_prob = torch.exp(X)
            l = X_prob * (X - Y)
            dx = l + X_prob
        else:
            max_val = torch.max(torch.stack([X, Y]), dim=0).values
            Q = torch.exp(X - max_val) * torch.exp(max_val)
            P = torch.exp(Y - max_val) * torch.exp(max_val)
            beta_P = beta * P
            one_minus_beta_Q = (1 - beta) * Q
            M = beta_P + one_minus_beta_Q
            log_M = torch.log(M)
            l = beta_P * Y + one_minus_beta_Q * X - M * log_M
            dx = one_minus_beta_Q * (X - log_M)
        loss[i] = l
        dX[i] = dx
    return torch.sum(loss), dX


def test_jsd_forward_compare_manual():
    # TODO:
    # 1) 构造小规模 X_log, Y_log（例如 BT=2, V=4），beta 分别取 0.0, 1.0, 0.5
    # 2) 调用 jsd_forward（from src.liger_kernel.ops.jsd import jsd_forward）和 manual_jsd_forward
    # 3) 对比 loss 和 dX（注意 dtype）
    assert True


def test_backward_scaling():
    # TODO: 验证 jsd_backward 在 grad_output = 1.0 时直接返回 dX，在其他情况按 grad_output 缩放
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/jsd_test.py
