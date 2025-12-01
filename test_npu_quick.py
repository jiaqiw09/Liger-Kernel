#!/usr/bin/env python3
"""
NPU 快速测试脚本 - 简化版
快速验证核心算子是否能在 NPU 上运行
"""

import torch

DEVICE = "npu"  # 修改为你的 NPU 设备名


def quick_test(name, test_func):
    """快速测试包装器"""
    try:
        test_func()
        print(f"✓ {name:20s} 通过")
        return True
    except Exception as e:
        print(f"✗ {name:20s} 失败: {e}")
        return False


def test_swiglu():
    from liger_kernel.ops.swiglu import swiglu_forward, swiglu_backward
    a = torch.randn(2, 4, 128, device=DEVICE)
    b = torch.randn(2, 4, 128, device=DEVICE)
    a_saved, b_saved, output = swiglu_forward(a.clone(), b.clone())
    grad_out = torch.randn_like(output)
    grad_a, grad_b = swiglu_backward(a_saved, b_saved, grad_out)
    assert output.shape == a.shape and grad_a.shape == a.shape


def test_geglu():
    from liger_kernel.ops.geglu import geglu_forward, geglu_backward
    a = torch.randn(2, 4, 128, device=DEVICE)
    b = torch.randn(2, 4, 128, device=DEVICE)
    a_saved, b_saved, output = geglu_forward(a.clone(), b.clone())
    grad_out = torch.randn_like(output)
    grad_a, grad_b = geglu_backward(a_saved, b_saved, grad_out)
    assert output.shape == a.shape and grad_a.shape == a.shape


def test_rms_norm():
    from liger_kernel.ops.rms_norm import rms_norm_forward, rms_norm_backward
    X = torch.randn(2, 4, 128, device=DEVICE)
    W = torch.randn(128, device=DEVICE)
    Y, X_saved, RSTD, BS, nw, cm = rms_norm_forward(X.clone(), W, 1e-6, 0.0, 'llama', False)
    dY = torch.randn_like(Y)
    dX, dW = rms_norm_backward(dY, X_saved, W, RSTD, 0.0, cm, BS, nw, False, False)
    assert Y.shape == X.shape and dX.shape == X.shape


def test_layer_norm():
    from liger_kernel.ops.layer_norm import layer_norm_forward, layer_norm_backward
    X = torch.randn(2, 4, 128, device=DEVICE)
    W = torch.randn(128, device=DEVICE)
    B = torch.randn(128, device=DEVICE)
    Y, X_saved, Mean, RSTD, BS, nw = layer_norm_forward(X.clone(), W, B, 1e-6)
    dY = torch.randn_like(Y)
    dX, dW, dB = layer_norm_backward(dY, X_saved, W, B, Mean, RSTD)
    assert Y.shape == X.shape and dX.shape == X.shape


def test_rope():
    from liger_kernel.ops.rope import rope_forward, rope_backward
    q = torch.randn(2, 8, 4, 64, device=DEVICE)
    k = torch.randn(2, 8, 4, 64, device=DEVICE)
    cos = torch.randn(1, 4, 64, device=DEVICE)
    sin = torch.randn(1, 4, 64, device=DEVICE)
    q_out, k_out, _, _ = rope_forward(q.clone(), k.clone(), cos, sin)
    dq = torch.randn_like(q_out)
    dk = torch.randn_like(k_out)
    dq_out, dk_out = rope_backward(dq, dk, cos, sin)
    assert q_out.shape == q.shape and dq_out.shape == q.shape


def test_softmax():
    from liger_kernel.ops.softmax import LigerSoftmaxFunction
    X = torch.randn(2, 4, 1024, device=DEVICE, requires_grad=True)
    output = LigerSoftmaxFunction.apply(X)
    output.backward(torch.randn_like(output))
    assert output.shape == X.shape and X.grad is not None


def test_cross_entropy():
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
    X = torch.randn(8, 1024, device=DEVICE, requires_grad=True)
    target = torch.randint(0, 1024, (8,), device=DEVICE)
    loss, _ = LigerCrossEntropyFunction.apply(X, target, None, -100, 0.0, 0.0, "mean", None, False)
    loss.backward()
    assert loss.dim() == 0 and X.grad is not None


def test_kl_div():
    from liger_kernel.ops.kl_div import LigerKLDivLossFunction
    y_pred = torch.randn(2, 4, 1024, device=DEVICE, requires_grad=True)
    y_true = torch.softmax(torch.randn(2, 4, 1024, device=DEVICE), dim=-1)
    loss = LigerKLDivLossFunction.apply(y_pred, y_true, 1e-8, "batchmean", False)
    loss.backward()
    assert loss.dim() == 0 and y_pred.grad is not None


def test_jsd():
    from liger_kernel.ops.jsd import LigerJSDFunction
    pred = torch.randn(2, 4, 1024, device=DEVICE, requires_grad=True)
    target = torch.randn(2, 4, 1024, device=DEVICE)
    loss = LigerJSDFunction.apply(pred, target, 0.5, 1e-10, "batchmean", False)
    loss.backward()
    assert loss.dim() == 0 and pred.grad is not None


def main():
    print("=" * 50)
    print(f"NPU 快速测试 (设备: {DEVICE})")
    print("=" * 50)
    
    tests = [
        ("SwiGLU", test_swiglu),
        ("GeGLU", test_geglu),
        ("RMSNorm", test_rms_norm),
        ("LayerNorm", test_layer_norm),
        ("RoPE", test_rope),
        ("Softmax", test_softmax),
        ("CrossEntropy", test_cross_entropy),
        ("KL Div", test_kl_div),
        ("JSD", test_jsd),
    ]
    
    results = [quick_test(name, func) for name, func in tests]
    
    print("=" * 50)
    print(f"结果: {sum(results)}/{len(results)} 通过")
    print("=" * 50)


if __name__ == "__main__":
    main()

