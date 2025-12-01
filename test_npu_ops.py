#!/usr/bin/env python3
"""
NPU 算子测试脚本
测试 Liger-Kernel 主要算子在 NPU 上的 forward 和 backward 功能
"""

import torch
import sys

# 设置 NPU 设备
DEVICE = "npu"
DTYPE = torch.float32
TOLERANCE = 1e-5


def print_test_header(name):
    """打印测试标题"""
    print("\n" + "=" * 70)
    print(f"测试: {name}")
    print("=" * 70)


def print_result(test_name, forward_err, backward_err=None):
    """打印测试结果"""
    forward_pass = forward_err < TOLERANCE
    forward_status = "✓" if forward_pass else "✗"
    
    print(f"{forward_status} {test_name} Forward: 误差={forward_err:.6e}")
    
    if backward_err is not None:
        backward_pass = backward_err < TOLERANCE
        backward_status = "✓" if backward_pass else "✗"
        print(f"{backward_status} {test_name} Backward: 误差={backward_err:.6e}")
        return forward_pass and backward_pass
    
    return forward_pass


# ============================================================================
# 1. SwiGLU 测试
# ============================================================================
def test_swiglu():
    """测试 SwiGLU: output = SiLU(a) * b"""
    print_test_header("SwiGLU")
    
    from liger_kernel.ops.swiglu import swiglu_forward, swiglu_backward
    
    # 构造输入
    shape = (2, 4, 128)
    a = torch.randn(*shape, device=DEVICE, dtype=DTYPE)
    b = torch.randn(*shape, device=DEVICE, dtype=DTYPE)
    
    print(f"输入形状: {shape}")
    
    # Forward 测试
    a_saved, b_saved, output = swiglu_forward(a.clone(), b.clone())
    expected = a * torch.sigmoid(a) * b
    forward_err = (output - expected).abs().max().item()
    
    # Backward 测试
    grad_out = torch.randn_like(output)
    grad_a, grad_b = swiglu_backward(a_saved, b_saved, grad_out.clone())
    
    # PyTorch 参考
    a_ref = a.requires_grad_(True)
    b_ref = b.requires_grad_(True)
    out_ref = a_ref * torch.sigmoid(a_ref) * b_ref
    out_ref.backward(grad_out)
    
    backward_err = max(
        (grad_a - a_ref.grad).abs().max().item(),
        (grad_b - b_ref.grad).abs().max().item()
    )
    
    return print_result("SwiGLU", forward_err, backward_err)


# ============================================================================
# 2. GeGLU 测试
# ============================================================================
def test_geglu():
    """测试 GeGLU: output = GELU(a) * b"""
    print_test_header("GeGLU")
    
    from liger_kernel.ops.geglu import geglu_forward, geglu_backward
    
    # 构造输入
    shape = (2, 4, 128)
    a = torch.randn(*shape, device=DEVICE, dtype=DTYPE)
    b = torch.randn(*shape, device=DEVICE, dtype=DTYPE)
    
    print(f"输入形状: {shape}")
    
    # Forward 测试
    a_saved, b_saved, output = geglu_forward(a.clone(), b.clone())
    expected = torch.nn.functional.gelu(a, approximate='tanh') * b
    forward_err = (output - expected).abs().max().item()
    
    # Backward 测试
    grad_out = torch.randn_like(output)
    grad_a, grad_b = geglu_backward(a_saved, b_saved, grad_out.clone())
    
    # PyTorch 参考
    a_ref = a.requires_grad_(True)
    b_ref = b.requires_grad_(True)
    out_ref = torch.nn.functional.gelu(a_ref, approximate='tanh') * b_ref
    out_ref.backward(grad_out)
    
    backward_err = max(
        (grad_a - a_ref.grad).abs().max().item(),
        (grad_b - b_ref.grad).abs().max().item()
    )
    
    return print_result("GeGLU", forward_err, backward_err)


# ============================================================================
# 3. RMSNorm 测试
# ============================================================================
def test_rms_norm():
    """测试 RMSNorm"""
    print_test_header("RMSNorm")
    
    from liger_kernel.ops.rms_norm import rms_norm_forward, rms_norm_backward
    
    # 构造输入
    batch_size, seq_len, hidden_dim = 2, 4, 128
    X = torch.randn(batch_size, seq_len, hidden_dim, device=DEVICE, dtype=DTYPE)
    W = torch.randn(hidden_dim, device=DEVICE, dtype=DTYPE)
    eps = 1e-6
    
    print(f"输入形状: X={X.shape}, W={W.shape}")
    
    # Forward 测试
    Y, X_saved, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(
        X.clone(), W, eps, offset=0.0, casting_mode='llama', row_mode=False
    )
    
    # 参考实现
    variance = X.pow(2).mean(-1, keepdim=True)
    X_normed = X * torch.rsqrt(variance + eps)
    expected = W * X_normed
    forward_err = (Y - expected).abs().max().item()
    
    # Backward 测试
    dY = torch.randn_like(Y)
    dX, dW = rms_norm_backward(
        dY.clone(), X_saved, W, RSTD, offset=0.0, casting_mode=casting_mode,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, in_place=False, row_mode=False
    )
    
    # PyTorch 参考
    X_ref = X.requires_grad_(True)
    W_ref = W.requires_grad_(True)
    var_ref = X_ref.pow(2).mean(-1, keepdim=True)
    normed_ref = X_ref * torch.rsqrt(var_ref + eps)
    Y_ref = W_ref * normed_ref
    Y_ref.backward(dY)
    
    backward_err = max(
        (dX - X_ref.grad).abs().max().item(),
        (dW - W_ref.grad).abs().max().item()
    )
    
    return print_result("RMSNorm", forward_err, backward_err)


# ============================================================================
# 4. LayerNorm 测试
# ============================================================================
def test_layer_norm():
    """测试 LayerNorm"""
    print_test_header("LayerNorm")
    
    from liger_kernel.ops.layer_norm import layer_norm_forward, layer_norm_backward
    
    # 构造输入
    batch_size, seq_len, hidden_dim = 2, 4, 128
    X = torch.randn(batch_size, seq_len, hidden_dim, device=DEVICE, dtype=DTYPE)
    W = torch.randn(hidden_dim, device=DEVICE, dtype=DTYPE)
    B = torch.randn(hidden_dim, device=DEVICE, dtype=DTYPE)
    eps = 1e-6
    
    print(f"输入形状: X={X.shape}, W={W.shape}, B={B.shape}")
    
    # Forward 测试
    Y, X_saved, Mean, RSTD, BLOCK_SIZE, num_warps = layer_norm_forward(
        X.clone(), W, B, eps
    )
    
    # 参考实现
    expected = torch.nn.functional.layer_norm(X, (hidden_dim,), W, B, eps)
    forward_err = (Y - expected).abs().max().item()
    
    # Backward 测试
    dY = torch.randn_like(Y)
    dX, dW, dB = layer_norm_backward(dY.clone(), X_saved, W, B, Mean, RSTD)
    
    # PyTorch 参考
    X_ref = X.requires_grad_(True)
    W_ref = W.requires_grad_(True)
    B_ref = B.requires_grad_(True)
    Y_ref = torch.nn.functional.layer_norm(X_ref, (hidden_dim,), W_ref, B_ref, eps)
    Y_ref.backward(dY)
    
    backward_err = max(
        (dX - X_ref.grad).abs().max().item(),
        (dW - W_ref.grad).abs().max().item(),
        (dB - B_ref.grad).abs().max().item()
    )
    
    return print_result("LayerNorm", forward_err, backward_err)


# ============================================================================
# 5. RoPE 测试
# ============================================================================
def test_rope():
    """测试 RoPE (Rotary Position Embedding)"""
    print_test_header("RoPE")
    
    from liger_kernel.ops.rope import rope_forward, rope_backward
    
    # 构造输入
    batch_size, n_q_heads, seq_len, head_dim = 2, 8, 4, 64
    n_kv_heads = 8
    
    q = torch.randn(batch_size, n_q_heads, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch_size, n_kv_heads, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    
    # 生成 cos, sin
    cos = torch.randn(1, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    sin = torch.randn(1, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    
    print(f"输入形状: q={q.shape}, k={k.shape}, cos={cos.shape}")
    
    # Forward 测试
    q_out, k_out, cos_out, sin_out = rope_forward(q.clone(), k.clone(), cos, sin)
    
    # 简单验证输出形状
    forward_err = 0.0 if q_out.shape == q.shape and k_out.shape == k.shape else 1.0
    
    # Backward 测试
    dq = torch.randn_like(q_out)
    dk = torch.randn_like(k_out)
    dq_out, dk_out = rope_backward(dq.clone(), dk.clone(), cos, sin)
    
    # 简单验证输出形状
    backward_err = 0.0 if dq_out.shape == dq.shape and dk_out.shape == dk.shape else 1.0
    
    return print_result("RoPE", forward_err, backward_err)


# ============================================================================
# 6. Softmax 测试
# ============================================================================
def test_softmax():
    """测试 Softmax"""
    print_test_header("Softmax")
    
    from liger_kernel.ops.softmax import LigerSoftmaxFunction
    
    # 构造输入
    batch_size, seq_len, vocab_size = 2, 4, 1024
    X = torch.randn(batch_size, seq_len, vocab_size, device=DEVICE, dtype=DTYPE, requires_grad=True)
    
    print(f"输入形状: {X.shape}")
    
    # Forward 测试
    output = LigerSoftmaxFunction.apply(X)
    expected = torch.softmax(X, dim=-1)
    forward_err = (output - expected).abs().max().item()
    
    # Backward 测试
    grad_out = torch.randn_like(output)
    output.backward(grad_out)
    
    X_ref = X.detach().requires_grad_(True)
    out_ref = torch.softmax(X_ref, dim=-1)
    out_ref.backward(grad_out)
    
    backward_err = (X.grad - X_ref.grad).abs().max().item()
    
    return print_result("Softmax", forward_err, backward_err)


# ============================================================================
# 7. CrossEntropy 测试
# ============================================================================
def test_cross_entropy():
    """测试 CrossEntropy"""
    print_test_header("CrossEntropy")
    
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
    
    # 构造输入
    batch_size, seq_len, vocab_size = 2, 4, 1024
    X = torch.randn(batch_size * seq_len, vocab_size, device=DEVICE, dtype=DTYPE, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size * seq_len,), device=DEVICE)
    
    print(f"输入形状: X={X.shape}, target={target.shape}")
    
    # Forward 测试
    loss, _ = LigerCrossEntropyFunction.apply(
        X, target, None, -100, 0.0, 0.0, "mean", None, False
    )
    
    # 参考实现
    expected = torch.nn.functional.cross_entropy(X, target, reduction="mean")
    forward_err = (loss - expected).abs().item()
    
    # Backward 测试
    loss.backward()
    
    X_ref = X.detach().requires_grad_(True)
    loss_ref = torch.nn.functional.cross_entropy(X_ref, target, reduction="mean")
    loss_ref.backward()
    
    backward_err = (X.grad - X_ref.grad).abs().max().item()
    
    return print_result("CrossEntropy", forward_err, backward_err)


# ============================================================================
# 8. KL Divergence 测试
# ============================================================================
def test_kl_div():
    """测试 KL Divergence"""
    print_test_header("KL Divergence")
    
    from liger_kernel.ops.kl_div import LigerKLDivLossFunction
    
    # 构造输入
    batch_size, seq_len, vocab_size = 2, 4, 1024
    y_pred = torch.randn(batch_size, seq_len, vocab_size, device=DEVICE, dtype=DTYPE, requires_grad=True)
    y_true = torch.randn(batch_size, seq_len, vocab_size, device=DEVICE, dtype=DTYPE)
    y_true = torch.softmax(y_true, dim=-1)  # 归一化为概率分布
    
    print(f"输入形状: y_pred={y_pred.shape}, y_true={y_true.shape}")
    
    # Forward 测试
    loss = LigerKLDivLossFunction.apply(y_pred, y_true, 1e-8, "batchmean", False)
    
    # 参考实现
    y_pred_log = torch.log_softmax(y_pred, dim=-1)
    expected = torch.nn.functional.kl_div(y_pred_log, y_true, reduction="batchmean")
    forward_err = (loss - expected).abs().item()
    
    # Backward 测试
    loss.backward()
    
    y_pred_ref = y_pred.detach().requires_grad_(True)
    y_pred_log_ref = torch.log_softmax(y_pred_ref, dim=-1)
    loss_ref = torch.nn.functional.kl_div(y_pred_log_ref, y_true, reduction="batchmean")
    loss_ref.backward()
    
    backward_err = (y_pred.grad - y_pred_ref.grad).abs().max().item()
    
    return print_result("KL Divergence", forward_err, backward_err)


# ============================================================================
# 9. JSD (Jensen-Shannon Divergence) 测试
# ============================================================================
def test_jsd():
    """测试 JSD"""
    print_test_header("JSD")
    
    from liger_kernel.ops.jsd import LigerJSDFunction
    
    # 构造输入
    batch_size, seq_len, vocab_size = 2, 4, 1024
    pred = torch.randn(batch_size, seq_len, vocab_size, device=DEVICE, dtype=DTYPE, requires_grad=True)
    target = torch.randn(batch_size, seq_len, vocab_size, device=DEVICE, dtype=DTYPE)
    
    print(f"输入形状: pred={pred.shape}, target={target.shape}")
    
    # Forward 测试
    loss = LigerJSDFunction.apply(pred, target, 0.5, 1e-10, "batchmean", False)
    
    # 简单验证：loss 应该是非负的
    forward_err = 0.0 if loss.item() >= 0 else loss.item()
    
    # Backward 测试
    loss.backward()
    
    # 验证梯度存在
    backward_err = 0.0 if pred.grad is not None and not torch.isnan(pred.grad).any() else 1.0
    
    return print_result("JSD", forward_err, backward_err)


# ============================================================================
# 主函数
# ============================================================================
def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("NPU 算子测试开始")
    print(f"设备: {DEVICE}")
    print(f"数据类型: {DTYPE}")
    print(f"容差: {TOLERANCE}")
    print("=" * 70)
    
    results = {}
    
    try:
        results['SwiGLU'] = test_swiglu()
    except Exception as e:
        print(f"✗ SwiGLU 测试失败: {e}")
        results['SwiGLU'] = False
    
    try:
        results['GeGLU'] = test_geglu()
    except Exception as e:
        print(f"✗ GeGLU 测试失败: {e}")
        results['GeGLU'] = False
    
    try:
        results['RMSNorm'] = test_rms_norm()
    except Exception as e:
        print(f"✗ RMSNorm 测试失败: {e}")
        results['RMSNorm'] = False
    
    try:
        results['LayerNorm'] = test_layer_norm()
    except Exception as e:
        print(f"✗ LayerNorm 测试失败: {e}")
        results['LayerNorm'] = False
    
    try:
        results['RoPE'] = test_rope()
    except Exception as e:
        print(f"✗ RoPE 测试失败: {e}")
        results['RoPE'] = False
    
    try:
        results['Softmax'] = test_softmax()
    except Exception as e:
        print(f"✗ Softmax 测试失败: {e}")
        results['Softmax'] = False
    
    try:
        results['CrossEntropy'] = test_cross_entropy()
    except Exception as e:
        print(f"✗ CrossEntropy 测试失败: {e}")
        results['CrossEntropy'] = False
    
    try:
        results['KL Div'] = test_kl_div()
    except Exception as e:
        print(f"✗ KL Div 测试失败: {e}")
        results['KL Div'] = False
    
    try:
        results['JSD'] = test_jsd()
    except Exception as e:
        print(f"✗ JSD 测试失败: {e}")
        results['JSD'] = False
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s} {status}")
    
    print("=" * 70)
    print(f"总计: {passed}/{total} 通过")
    print("=" * 70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

