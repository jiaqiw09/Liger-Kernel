"""
教学测试模板：`swiglu`

说明：
- 本文件为初学者编写的测试与实现指导模板，目标是帮助你逐步实现 `swiglu.py` 中的前向/后向算子。
- 模板中不包含 Triton 内核或高性能实现的完整代码，只给出详细中文步骤、参考的朴素实现思路、以及 pytest 风格的测试骨架。

使用说明：
- 在实现过程中，按照“实现步骤”逐步完成函数，然后去掉对应的 TODO 注释并运行 pytest。
- 推荐先在 CPU / float64 上实现并验证数值正确性，再迁移到 Triton/GPU 优化实现。

实现目标概述：
- `swiglu_forward(a, b)`：计算 SiLU(a) * b（SiLU 即 x * sigmoid(x)），输入 `a,b` 形状相同，可能有 batch 维度。返回包含输入视图与输出 `c`。
- `swiglu_backward(a, b, dc)`：给定对输出的梯度 `dc`，计算并返回对 `a` 和 `b` 的梯度（注意 `a` 参与 SiLU 的非线性变换，需要正确链式求导）。

实现步骤（逐步完成）：
1) 阅读 `src/liger_kernel/ops/swiglu.py`，理解前向/后向的 API：
   - `swiglu_forward(a, b)`：本模块把 `a,b` reshape 为二维 `(n_rows, n_cols)`，调用 Triton kernel 并返回 `a, b, c`。
   - `swiglu_backward(a, b, dc)`：把 `dc` reshape，同样调用 Triton kernel 计算 `da, db` 并返回视图后的 `a, b`。
2) 先实现一个朴素 CPU 参考实现函数 `swiglu_naive_forward(a, b)`：
   - 输入：两个相同形状的 Tensor `a, b`（torch.Tensor），dtype 建议用 `torch.float64` 以便数值对比。
   - 计算：`c = (a * torch.sigmoid(a)) * b`。
   - 返回 `c`。
3) 实现 `swiglu_naive_backward(a, b, dc)`（手工推导）：
   - 记 s = sigmoid(a)，si = a * s
   - c = si * b
   - dc/db = si
   - dc/da = dc * (d(a*s)/da) * b，其中 d(a*s)/da = s + a * s * (1 - s) = s + si * (1 - s)
   - 因此：
       db = dc * si
       da = dc * (s + si * (1 - s)) * b
   - 返回 da, db
4) 在本文件中先实现上述两个朴素函数（仅用于单元测试和数值参考），并在 pytest 中对比 Triton 实现（当可用）或验证自洽性（forward/backward 的数值一致性）。
5) 梯度检验（可选）：使用 `torch.autograd.gradcheck` 或数值差分验证 `swiglu` 的 backward 是否正确：
   - `gradcheck` 要求函数输入为 `double` 且 `requires_grad=True`。
   - 如果 Triton kernel 暂不可用，可将朴素实现包装为 `torch.autograd.Function` 的 `forward/backward` 并用 `gradcheck` 验证。

测试骨架（pytest）：
 - 包含以下测试：
   1. `test_swiglu_forward_naive_vs_reference`：在随机小张量上对比朴素实现和文件中的 `swiglu_forward` 输出（若在 CPU 上运行，直接比较数值；若 Triton/GPU 环境，可放在 GPU 上测试）。
   2. `test_swiglu_backward_naive_vs_numerical`：用数值差分对朴素后向实现做校验，或使用 `gradcheck` 验证 `backward` 的正确性。
   3. `test_swiglu_shapes_and_dtypes`：检查不同形状（包括 1D、2D、batch 多维）和不同 dtype（float32/float64）下函数是否保留形状并返回合适的 dtype。
   4. `test_swiglu_edge_cases`：测试零向量、大数、极小/极大输入值（关注数值稳定性和 overflow/underflow 问题）。

调试与数值稳定性建议：
- sigmoid 在极大或极小输入上会饱和，若需要提高稳定性可以在计算时使用 `torch.sigmoid` 的数值稳定实现或在 CPU 上先检查。
- 在 float32 下，某些边界样例（例如非常大的正/负 a）会导致梯度接近 0 或 NaN，先在 float64 下跑通过再回到 float32。
- 如果 Triton kernel 使用了 cast 或 to(dtype) 操作，注意保持前向保存的中间量（例如 cast 后的 a）与 backward 的计算一致。

示例（模板代码）:
```
import torch
import pytest

from liger_kernel.ops import swiglu as swiglu_impl


def swiglu_naive_forward(a, b):
    # TODO: 用高精度（float64）实现朴素前向
    # c = (a * torch.sigmoid(a)) * b
    raise NotImplementedError("请按照模板说明实现 swiglu_naive_forward")


def swiglu_naive_backward(a, b, dc):
    # TODO: 实现朴素后向计算并返回 da, db
    raise NotImplementedError("请按照模板说明实现 swiglu_naive_backward")


def test_swiglu_forward_naive_vs_reference():
    # 1) 生成小随机张量
    a = torch.randn(4, 8, dtype=torch.float64, requires_grad=False)
    b = torch.randn_like(a)

    # 2) 计算朴素实现
    # c_naive = swiglu_naive_forward(a, b)

    # 3) 计算库中实现（注意：若库函数需要 cuda，请先移动到相应设备）
    # a_in, b_in, c_impl = swiglu_impl.swiglu_forward(a, b)

    # 4) 对比：使用高精度或相对误差断言
    # assert torch.allclose(c_naive, c_impl, atol=1e-6, rtol=1e-5)
    raise NotImplementedError("把上面的 TODO 逐条实现并移除 NotImplementedError")


def test_swiglu_backward_naive_vs_numerical():
    # 参考实现数值微分或 gradcheck
    # 1) 准备 double 输入并设置 requires_grad
    # 2) 使用 torch.autograd.gradcheck 或手写数值差分校验 da/db
    raise NotImplementedError("实现 gradcheck 或数值差分验证")


def test_swiglu_shapes_and_dtypes():
    # 验证多种形状和 dtype 的兼容性
    raise NotImplementedError("实现形状/类型测试")


def test_swiglu_edge_cases():
    # 测试极端输入以及数值稳定性
    raise NotImplementedError("实现边界情况测试")

```

运行测试：
- 在项目根目录运行：`pytest -q src/liger_kernel/ops/swiglu_test.py -q`
- 若在 GPU/Triton 环境并希望测试 Triton 实现，请确保 Triton 可用并且在测试中把输入 `.to('cuda')`。

下一步建议：
- 完成上述朴素实现并运行测试；在通过数值验证后，把朴素实现替换为或对照 `src/liger_kernel/ops/swiglu.py` 中的 Triton 调用进行调试。
- 若你希望，我可以继续把模板中的 TODO 替换为可运行代码并尝试用本地环境运行部分测试（受限于本地 Triton / GPU 可用性）。
