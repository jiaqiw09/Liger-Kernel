# 初学者模板：`__init__.py` 对应的练习文件
# 说明：不要在这里实现实际逻辑。这个模板为初学者提供如何阅读包初始化、组织导出符号和编写基本测试的分步指导。

"""
目标：
- 理解包的导出符号（`__all__`、从子模块导入等）
- 学会为包或模块级别的导出编写简单的单元测试

文件位置：`src/liger_kernel/ops/__init__.py`

使用说明：
- 按步骤完成下面的 TODO，逐步实现测试用例并运行 `pytest` 来验证。
"""

# 1) 阅读原始 `__init__.py`
# - 打开 `src/liger_kernel/ops/__init__.py`，查看是否有 `__all__`、`from .foo import bar` 等导入语句。
# - 记录包在被 `import src.liger_kernel.ops` 时应该暴露哪些名字。

# 2) 在这里写测试用例结构（不实现功能，只写测试指导）
#    目标：确认包导入时不会抛异常，并且预期的符号存在。

import importlib
import pytest


def test_import_package():
    """步骤说明：
    1) 动作：尝试导入 `src.liger_kernel.ops`。
    2) 期望：导入成功，不抛出异常。
    3) TODO：如果 `__init__` 要初始化某些注册表，可以在这里断言初始状态。
    """
    # TODO: 运行导入并检查是否会报错
    pkg = importlib.import_module('src.liger_kernel.ops')
    assert pkg is not None


def test_exported_symbols():
    """步骤说明：
    1) 动作：根据 `__init__.py` 中的导出声明（例如 `__all__`）检查符号是否存在。
    2) 如何做：列出期望符号，然后 `hasattr(pkg, name)`。
    3) TODO：把 `expected` 列表替换为从 `__init__.py` 中提取的符号名。
    """
    pkg = importlib.import_module('src.liger_kernel.ops')
    expected = [
        # TODO: 填写例如 'softmax', 'layer_norm' 等符号名
    ]
    for name in expected:
        # TODO: 如果某些符号是从子模块导入的，确认路径是否正确
        assert hasattr(pkg, name), f"缺少导出符号: {name}"

# 3) 运行与验证
# - 在终端中运行： `pytest -q src/liger_kernel/ops/__init___test.py`
# - 若测试失败：打开 `__init__.py`，理解导入机制并调整测试或 `__init__`。

# 4) 提示与扩展练习
# - 练习：为包中某个算子（如 `softmax`）写更详细的导入/别名测试。
# - 思考：如果包通过延迟导入（lazy import）导出符号，如何在测试中处理？
