# __init__.py 文件是导入模块时执行的第 1 个文件，即模块的入口文件

is_simple_core = False

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable
else:
    from dezero.core import Variable
    from dezero.core import Paramter
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    from dezero.core import Config
    from dezero.layers import Layer
    from dezero.models import Model
    from dezero.models import MLP
    from dezero.datasets import Dataset
    from dezero.dataloaders import DataLoader

# 执行 Variable 实例的运算符重载
setup_variable()
