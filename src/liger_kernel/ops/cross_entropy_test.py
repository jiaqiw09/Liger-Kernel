# 初学者模板：`cross_entropy.py` 算子实现练习文件（仅包含中文分步指导）
# 说明：本文件不实现算子逻辑；它为初学者提供详细步骤来实现与测试 `cross_entropy.py` 中的函数。

"""
目标：
- 理解 `cross_entropy.py` 中的核心符号：
  - Triton kernel: `liger_cross_entropy_kernel`
  - 前向/后向包装函数：`cross_entropy_forward`, `cross_entropy_backward`
  - Autograd 函数：`LigerCrossEntropyFunction`

- 学会如何为复杂算子编写单元测试（功能测试、数值与梯度检查、边界条件测试、性能/稳定性建议）

使用方法：
- 按下面的步骤逐项完成 TODO
- 运行 `pytest -q src/liger_kernel/ops/cross_entropy_test.py` 来执行测试
"""

import torch
import pytest

# 提取到的核心符号（从源码可见）:
# - liger_cross_entropy_kernel (triton kernel, 无需直接实现)
# - cross_entropy_forward
# - cross_entropy_backward
# - LigerCrossEntropyFunction

# 下面是逐步的实现提示与测试骨架：

# 1) 理解输入输出与 API
# - 打开 `src/liger_kernel/ops/cross_entropy.py`，逐行阅读 `cross_entropy_forward` 的注释和签名。
# - 记录下函数接受的参数及其类型：例如 `_input` (BT, V), `target` (BT,), `weight`, `ignore_index`, `label_smoothing`, `reduction` 等。
# - 理清返回值：`(loss, z_loss, token_accuracy, _input)`（注意 `_input` 可能被修改以保存梯度）。

# 2) 写最小功能性测试（行为对比）
# 目标：对比自定义实现（或者调用 `cross_entropy_forward`）与 PyTorch 的 `torch.nn.functional.cross_entropy` 在简单场景下的结果是否一致。
# 步骤：
# - 构造一个非常小的例子，例如 BT=2, V=5，使用确定性的随机种子。
# - 调用 `cross_entropy_forward`（TODO: 导入并调用）并与 `torch.nn.functional.cross_entropy` 对比 loss 值（注意 reduction 的差异）。


def test_small_example_matches_torch():
    # TODO: 按照下面步骤补充代码并运行：
    # 1) 构造输入：
    #    torch.manual_seed(0)
    #    logits = torch.randn((2, 5), dtype=torch.float32, requires_grad=True)
    #    target = torch.tensor([1, 3], dtype=torch.long)
    # 2) 调用被测前向：
    #    from src.liger_kernel.ops.cross_entropy import cross_entropy_forward
    #    loss, z_loss, token_accuracy, _ = cross_entropy_forward(logits, target, weight=None, ignore_index=-100,
    #                                                           lse_square_scale=0.0, label_smoothing=0.0,
    #                                                           reduction='mean', softcap=None, return_z_loss=False)
    # 3) 调用 PyTorch 参考实现：
    #    import torch.nn.functional as F
    #    ref_loss = F.cross_entropy(logits, target, reduction='mean')
    # 4) 断言：loss 与 ref_loss 在合理误差范围内接近
    assert True  # TODO: 替换为实际断言

# 3) 梯度校验（非常重要）
# - 使用数值梯度或与 PyTorch autograd 的比较来校验 `LigerCrossEntropyFunction` 的 backward 是否正确。
# - 方法1（推荐）：利用 PyTorch 的 `torch.autograd.gradcheck`（需使用 double 精度且输入为需要 grad 的张量）。
# - 方法2：对比有限差分数值梯度与实现的梯度。


def test_gradients_with_gradcheck():
    # TODO: 使用 `torch.autograd.gradcheck` 来校验 `LigerCrossEntropyFunction` 的正向/反向。
    # 关键点：
    # - 将输入转换为 double 并设置 requires_grad=True
    # - 使用小的输入规模（例如 BT=1, V=4）以减少数值误差
    # - 注意 `gradcheck` 要求输入有 `dtype=torch.double`
    assert True  # TODO: 替换为实际 gradcheck 实现

# 4) 边界条件和特殊参数测试
# - 测试 `ignore_index` 行为：当 target 中含有 ignore_index 时，loss 应忽略这些元素。
# - 测试 `weight` 参数：给定权重后，loss 应与带权重的参考实现一致。
# - 测试 `label_smoothing`：确保平滑项对 loss 的影响方向与预期一致（可与手算小例子比对）。
# - 测试 `return_z_loss` 和 `return_token_accuracy` 的输出结构和类型

# 5) 性能与数值稳定性注意事项（指导）
# - Triton kernel 有许多与 BLOCK_SIZE、num_warps、softcap 等相关的超参。初学者在本地小规模测试时无需调优，但要了解这些参数的作用。
# - 如果出现溢出/下溢/NaN，先排查：输入值范围、softcap、label_smoothing 是否合理、是否需要更高精度（float32 -> float64）进行调试。

# 6) 调试技巧
# - 分步调试：先在 CPU 上实现一个纯 PyTorch 版本（可以直接用 `torch.nn.functional` 或手动实现 softmax + nll），确认数值结果。
# - 将复杂的算子拆成更小的函数并为每个子函数写单元测试。
# - 打印/断言中间值（例如 softmax 输出、lse）以定位数值差异。

# 7) 检查列表（提交前）
# - [ ] 添加必要的输入合法性检查（shape、dtype、device）
# - [ ] 实现并测试 `reduction='none'|'mean'|'sum'` 的行为
# - [ ] 添加注释与 docstring，说明每个参数的含义
# - [ ] 在 CI 下运行全部相关测试

# 参考运行命令（在仓库根目录下运行）：
# pytest -q src/liger_kernel/ops/cross_entropy_test.py
