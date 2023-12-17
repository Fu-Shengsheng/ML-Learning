import numpy as np

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self
    
    def update(self):
        # 将梯度为None之外的参数汇总到列表
        params = [p for p in self.target.params() if p.grad is not None]
        # 预处理
        for f in self.hooks:
            f(params)
        
        # 更新参数
        for param in params:
            self.update_one(param)
    
    def update_one(self, param):
        raise NotImplementedError()
    
    def add_hook(self, f):
        self.hooks.append(f)

# Stochastic Gradient Descent 随机梯度下降优化器
class SGD(Optimizer):
    def __init__(self, lr = 0.01):
        super().__init__()
        self.lr = lr
    
    def update_one(self, param):
        param.data -= self.lr * param.grad.data

# Momentum 动量优化器
# 令参数在 梯度 方向受力，使参数更新加速
class MomentumSGD(Optimizer):
    def __init__(self, lr = 0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}
    
    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v