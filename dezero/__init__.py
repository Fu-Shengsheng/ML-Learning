# __init__.py 文件是导入模块时执行的第 1 个文件，即模块的入口文件

from dezero.core_simple import Variable
from dezero.core_simple import Function
from dezero.core_simple import using_config
from dezero.core_simple import no_grad
from dezero.core_simple import as_array
from dezero.core_simple import as_variable
from dezero.core_simple import setup_variable

# 执行 Variable 实例的运算符重载
setup_variable()
