# 初学者模板：`sparsemax.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 验证 sparsemax 的前向与后向实现（与排序/阈值法的手动实现对比）
- 理解支持集（support）和 tau 的计算方式

测试要点：
- 前向：实现手动 sparsemax（对每行排序、求累加和并找 tau），比较 `_sparsemax_forward` 的输出
- 后向：在支持集上分发梯度并减去平均值，比较 `_sparsemax_backward` 的输出

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/sparsemax_test.py`
"""

import torch
import pytest


def manual_sparsemax(x, dim=-1):
    # 基于论文/标准实现的逐行 sparsemax
    # Steps:
    # 1) sort x descending
    # 2) compute cumsum of sorted x
    # 3) find k = max {j | sorted_x_j * j > (cumsum_j - 1)}
    # 4) tau = (cumsum_k - 1) / k
    # 5) y = max(x - tau, 0)
    x_flat = x.clone()
    original_shape = x_flat.shape
    x2 = x_flat.view(-1, x_flat.size(-1)).float()
    n_rows, n_cols = x2.shape
    out = torch.zeros_like(x2)
    for i in range(n_rows):
        row = x2[i]
        zs = torch.sort(row, descending=True).values
        cssv = torch.cumsum(zs, dim=0)
        r = torch.arange(1, n_cols + 1, device=row.device, dtype=row.dtype)
        t = (cssv - 1) / r
        support = (zs > t)
        if support.any():
            k = support.sum()
            tau = (cssv[k - 1] - 1) / k
        else:
            k = 1
            tau = (cssv[0] - 1) / 1
        y = torch.clamp(row - tau, min=0.0)
        out[i] = y
    return out.view(*original_shape), out


def manual_sparsemax_backward(grad_out, out):
    # grad_out: same shape as out
    # out: sparsemax output to determine support
    grad = grad_out.clone()
    grad_flat = grad.view(-1, grad.shape[-1])
    out_flat = out.view(-1, out.shape[-1])
    dx = torch.zeros_like(grad_flat)
    for i in range(grad_flat.shape[0]):
        g = grad_flat[i]
        o = out_flat[i]
        supp = o > 0
        supp_cnt = supp.sum().item()
        if supp_cnt == 0:
            dx[i] = 0
            continue
        g_sum = (g * o).sum().item()
        # In kernel they compute go_sum and divide by supp_cnt
        mean = g[supp].sum() / max(supp_cnt, 1)
        dx_row = torch.zeros_like(g)
        dx_row[supp] = g[supp] - mean
        dx[i] = dx_row
    return dx.view_as(grad)


def test_forward_matches_manual():
    # TODO:
    # 1) 构造 small-scale输入 x (e.g., shape=(3,5))
    # 2) 调用 _sparsemax_forward 并与 manual_sparsemax 比较输出 y
    assert True


def test_backward_matches_manual():
    # TODO:
    # 1) 使用 manual_sparsemax 得到 out_flat
    # 2) 随机生成 grad_out 并分别使用 _sparsemax_backward 与 manual_sparsemax_backward 比较
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/sparsemax_test.py
