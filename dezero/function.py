from variable import Variable

class Function:
    # call 的声明使类的实例可以当做可调用对象使用
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        # 输出变量保存创造者信息
        output.set_creator(self)
        # 保存输入值
        self.input = input
        # 保存输出值
        self.output = output
        return output
    
    # 前向传播（计算结果）
    def forward(self, x):
        # 抛出异常，表示这个方法应该通过继承实现
        raise NotImplementedError()
    
    # 反向传播（计算导数）
    def backward(self, gy):
        raise NotImplementedError()