# 初学者模板：`fused_add_rms_norm.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 `fused_add_rms_norm_forward` 的数学过程：
  1. S = X + R
  2. rstd = 1 / sqrt(mean(S^2) + eps)  （按行计算）
  3. Y = (S * rstd) * (offset + W)
- 学会如何为不同 `casting_mode`（'llama'、'gemma'、'none'）、`in_place` 行为和返回的 `RSTD` 编写测试

使用方法：
- 按 TODO 提示补充测试并运行：`pytest -q src/liger_kernel/ops/fused_add_rms_norm_test.py`
"""

import torch
import pytest


def naive_fused_add_rms_norm(X, R, W, eps, offset=0.0):
    # X, R: (n_rows, n_cols)
    S = X + R
    mean_square = torch.mean(S * S, dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(mean_square + eps)
    Y = (S * rstd) * (offset + W)
    # 返回 Y, S, rstd（rstd 为每行的标量）
    return Y, S, rstd.squeeze(-1)


def test_forward_matches_naive():
    # TODO: 实现以下步骤：
    # 1) 构造小规模输入，例如 n_rows=2, n_cols=8，dtype=float32
    # 2) 随机生成 X, R, W（W shape=(n_cols,)），设定 eps, offset
    # 3) 调用被测函数：from src.liger_kernel.ops.fused_add_rms_norm import fused_add_rms_norm_forward
    # 4) 调用 naive_fused_add_rms_norm 并对比 Y、S、RSTD（注意 dtype/casting_mode 对数值的影响）
    assert True  # TODO: 替换为实际断言


def test_backward_gradcheck():
    # TODO: 使用 gradcheck 验证 autograd 函数 LigerFusedAddRMSNormFunction 的 backward
    # 注意：gradcheck 需要 double 精度输入，并设置 requires_grad=True
    # 可分别对不同 casting_mode 做检查（在调试时可使用 'none' 以减少 dtype 影响）
    assert True


def test_casting_modes_and_in_place():
    # TODO: 为三种 casting_mode ('llama','gemma','none') 各写一个小测试
    # - 检查输出类型、RSTD 的 dtype（源码中对 llama/gemma 会使用 fp32 存储 rstd）
    # - 测试 in_place=True 与 in_place=False 的差异（主要是函数返回值形状是否一致与 dY 是否被复用）
    assert True


# 调试提示：
# - 若在 float16 上出现数值不稳定（NaN/Inf），在小规模样例上用 float32/float64 验证数值正确性，再回到 float16 做近似比较。
# - rstd 的缓存行为对性能重要，但测试关注数值正确性即可。

# 运行命令：
# pytest -q src/liger_kernel/ops/fused_add_rms_norm_test.py
