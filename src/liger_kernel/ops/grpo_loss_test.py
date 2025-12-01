# 初学者模板：`grpo_loss.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 GRPO 损失的前向与后向计算流程（包括 selective log-softmax 与 clipping 逻辑）
- 为关键组件（`_selective_log_softmax_kernel`, `_grpo_loss_fwd_kernel`, `_grpo_loss_bwd_kernel`）编写小规模数值对比测试

测试要点：
1) `fused_selective_log_softmax` 的正确性：对比手工实现的 selective log-softmax（仅处理序列中感兴趣的索引）
2) GRPO 前向：用小规模 logits 构造手工实现的 forward（参照源码注释）并与 `GrpoLossFunction.forward` 的结果比较
3) GRPO 后向：通过有限差分验证 backward 输出的 dlogits 是否符合手动差分

使用：
- 补齐 TODO 后运行：`pytest -q src/liger_kernel/ops/grpo_loss_test.py`
"""

import torch
import pytest
import math


def manual_selective_log_softmax(logits, input_ids, temperature=0.9, mask=None):
    # logits: (B, L+1, N)
    # input_ids: (B, L+1) or (B, L) depending on caller; in fused_selective_log_softmax they use last L positions
    # 对每个 (b, l) 只计算 logits[b, l, :]
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    input_ids = input_ids[:, -L:]
    if mask is not None:
        mask = mask[:, -L:]
    out = torch.zeros(B, L, dtype=torch.float32, device=logits.device)
    for b in range(B):
        for l in range(L):
            if mask is not None and mask[b, l] == 0:
                out[b, l] = 0.0
                continue
            idx = input_ids[b, l]
            row = logits[b, l] / temperature
            lse = torch.logsumexp(row, dim=-1)
            out[b, l] = row[idx] - lse
    return out


def test_selective_log_softmax():
    # TODO: 构造小规模示例并比较 fused_selective_log_softmax 与 manual_selective_log_softmax
    # 1) 随机生成 logits: (B, L+1, N)（例如 B=2, L=3, N=5）
    # 2) 随机生成 input_ids、mask
    # 3) 从 src.liger_kernel.ops.grpo_loss import fused_selective_log_softmax
    # 4) 比较两者输出
    assert True


def test_grpo_forward_backward_small():
    # TODO: 在非常小的规模下实现 GRPO 前向的手工计算并与 GrpoLossFunction.forward 对比
    # 然后通过有限差分验证 backward 的 dlogits（数值微小扰动）
    # 注意：核心步骤包括计算 logp, coef_1, coef_2, per_token_loss, 是否 clip, 以及 kl 项（如果 beta != 0）
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/grpo_loss_test.py
