import contextlib

# 控制是否启用反向传播
# 当进行推理验证阶段时，不需要启用，不保留计算之间的连接关系，以节省内存空间
class Config:
    enable_backprop = True

# 借助 @contextlib.contextmanager 装饰器创建一个判断上下文的函数
# 该函数搭配 with 使用，进入 with 时预处理被调用，离开 with 时后处理被调用
@contextlib.contextmanager
def using_config(name, value):
    # 预处理
    old_val = getattr(Config, name)
    setattr(Config, name, value)
    try:
        # 程序暂停
        yield
    finally:
        # 后处理
        setattr(Config, name, old_val)

def no_grad():
    return using_config('enable_backprop', False)