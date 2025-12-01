# 初学者模板：`qwen2vl_mrope.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 Qwen2VL 的 M-RoPE 变换以及左右半区（real/imag 或 interleaved 存储）上的复数旋转
- 为前向/后向实现编写小规模数值对比与逆变换验证

测试要点：
1) 构造小规模 q/k（例如 batch=1, seq=2, n_q_head=1, head_dim=4），构造 cos/sin tensors
2) 手动实现 M-RoPE 的变换（基于源码中注释：
   new_left = left * cos - right * sin
   new_right = right * cos + left * sin
   注意大小与 tiling 的影响，先在无 padding 情况下验证
3) 调用 qwen2vl_mrope_forward 并与手动实现比较输出
4) 对于 backward，调用 qwen2vl_mrope_backward 或者在 autograd 中应用 LigerQwen2VLMRopeFunction 的 backward 进行验证：
   BACKWARD_PASS 的变换项应与前向的逆变换一致（符号变化）

实现建议：
- 先实现一个纯 Python 的 small-scale transform，并在不同的 mrope_section 设置下比对
- 注意：源码对 q/k 做了 `transpose(1,2)` 以确保物理内存布局，测试时构造输入时要匹配接口的期望 shape

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/qwen2vl_mrope_test.py`
"""

import torch
import pytest


def manual_mrope_apply(q, cos, sin, mrope_section):
    # q shape expected in the API: (bsz, n_q_head, seq_len, head_dim)
    # cos/sin shape: (3, bsz, seq_len, head_dim)
    # For small-scale correctness验证，我们只实现最基本的变换： split half and apply cos/sin
    bsz, n_q_head, seq_len, head_dim = q.shape
    q_out = q.clone()
    half = head_dim // 2
    for b in range(bsz):
        for t in range(seq_len):
            cos_row = cos[0, b, t, :half]  # 简化使用第一个 section
            sin_row = sin[0, b, t, :half]
            for h in range(n_q_head):
                for d in range(half):
                    l = q[b, h, t, d * 2]
                    r = q[b, h, t, d * 2 + 1]
                    new_l = l * cos_row[d] - r * sin_row[d]
                    new_r = r * cos_row[d] + l * sin_row[d]
                    q_out[b, h, t, d * 2] = new_l
                    q_out[b, h, t, d * 2 + 1] = new_r
    return q_out


def test_forward_manual_small():
    # TODO: 构造小规模输入并比较 qwen2vl_mrope_forward 与 manual_mrope_apply
    # 1) 随机生成 q/k, cos, sin（shape 按注释）
    # 2) 调用 qwen2vl_mrope_forward
    # 3) 手工计算并比对
    assert True


def test_backward_inverse_property():
    # TODO: 验证 backward 的变换（在 dq/dk 上使用 imag/sign 或 BACKWARD_PASS）与前向变换的逆操作一致
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/qwen2vl_mrope_test.py
