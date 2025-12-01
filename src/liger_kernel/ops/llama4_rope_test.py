# 初学者模板：`llama4_rope.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 RoPE（Rotary Positional Embeddings）的核心操作：对 q/k 的每对 real/imag 部分做复数乘法
- 为 `_prepare_freqs`、`llama4_rope_forward` 编写小规模数值对比测试

测试要点：
1) `_prepare_freqs`：验证对 complex 输入或拼接 real/imag 的输入的处理逻辑（broadcast、裁剪）
2) `llama4_rope_forward`：构造小规模 q/k 与 freqs_cis（complex 或拼接形式），比较 kernel 输出与按公式实现的结果
3) 反向验证：利用 `imag_sign=-1.0` 对 dq/dk 执行同样操作应近似逆变换

实现建议（手动/显式实现思路）：
- RoPE 的核心（按 head_dim 一半储存为 real/imag）：
  对于每个 position t 和维度 d：
    new_real = real * cos(theta_td) - imag * sin(theta_td)
    new_imag = real * sin(theta_td) + imag * cos(theta_td)
  如果你用 complex 类型，可以用 (a+ib) * (c+id) 的复数乘法实现。

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/llama4_rope_test.py`
"""

import torch
import pytest


def manual_rope_apply(q, freqs_real, freqs_imag, imag_sign=1.0):
    # q shape: (batch, seq, n_heads, head_dim)
    # freqs_real/freqs_imag shape: (seq, head_dim_half)
    batch, seq, n_heads, head_dim = q.shape
    half = head_dim // 2
    q_out = q.clone()
    for b in range(batch):
        for t in range(seq):
            fr = freqs_real[t]
            fi = freqs_imag[t] * imag_sign
            for h in range(n_heads):
                # split real/imag stored interleaved as [r0, i0, r1, i1, ...]
                for d in range(half):
                    r = q[b, t, h, d * 2]
                    im = q[b, t, h, d * 2 + 1]
                    new_r = r * fr[d] - im * fi[d]
                    new_i = r * fi[d] + im * fr[d]
                    q_out[b, t, h, d * 2] = new_r
                    q_out[b, t, h, d * 2 + 1] = new_i
    return q_out


def test_prepare_freqs_and_forward():
    # TODO:
    # 1) 构造 freqs_cis 三种形态：complex tensor, concat(real, imag), and single-row broadcast case
    # 2) 验证 `_prepare_freqs` 对这些输入的处理行为（从 src.liger_kernel.ops.llama4_rope import _prepare_freqs）
    # 3) 构造小规模 q/k（例如 batch=1, seq=3, n_heads=1, head_dim=4），调用 llama4_rope_forward
    # 4) 用 manual_rope_apply 比较 q_out, k_out
    assert True


def test_backward_imag_sign_inverse():
    # TODO: 验证 backward 用 imag_sign=-1 应近似逆向前向（即对 dq/dk 做 inverse 操作）
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/llama4_rope_test.py
