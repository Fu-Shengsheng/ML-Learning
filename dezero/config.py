# 控制是否启用反向传播
# 当进行推理验证阶段时，不需要启用，不保留计算之间的连接关系，以节省内存空间
class Config:
    enable_backprop = True