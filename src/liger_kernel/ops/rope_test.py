# 初学者模板：`rope.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 RoPE 的变换（基于 cos/sin 的左/右半区旋转）并验证 forward/backward 的正确性
- 验证 cos 的批次维度（1 或 bsz）处理逻辑

测试要点：
- 手动实现小规模的 RoPE（按 interleaved 存储 [r0,i0,r1,i1,...] 的方式），对比 `rope_forward` 的输出
- 验证 `rope_backward`（或 BACKWARD_PASS=True）的逆变换属性

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/rope_test.py`
"""

import torch
import pytest


def manual_rope(q, cos, sin, cos_bs=1):
    # q: (bsz, n_head, seq_len, head_dim)
    bsz, n_head, seq_len, head_dim = q.shape
    half = head_dim // 2
    out = q.clone()
    for b in range(bsz):
        for t in range(seq_len):
            if cos_bs == 1:
                cos_row = cos[0, t]
                sin_row = sin[0, t]
            else:
                cos_row = cos[b, t]
                sin_row = sin[b, t]
            for h in range(n_head):
                for d in range(half):
                    r = q[b, h, t, d * 2]
                    im = q[b, h, t, d * 2 + 1]
                    new_r = r * cos_row[d] - im * sin_row[d]
                    new_i = im * cos_row[d] + r * sin_row[d]
                    out[b, h, t, d * 2] = new_r
                    out[b, h, t, d * 2 + 1] = new_i
    return out


def test_rope_forward_matches_manual():
    # TODO:
    # 1) 构造 small-scale q, k, cos, sin（cos_bs=1 与 cos_bs=bsz 两种情形）
    # 2) 调用 rope_forward(from src.liger_kernel.ops.rope import rope_forward)
    # 3) 用 manual_rope 比较两个输出
    assert True


def test_rope_backward_inverse():
    # TODO: 验证 rope_backward 在 dq/dk 上的操作为前向变换的逆（符号变化），使用 small-scale 示例
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/rope_test.py
