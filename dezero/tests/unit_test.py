import unittest
import numpy as np
import os
import sys
sys.path.append(os.pardir)
from function import square, exp
from variable import Variable
from utils import numerical_diff

# 单元测试
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        excepted = np.array(4.0)
        self.assertEqual(y.data, excepted)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        excepted = np.array(6.0)
        self.assertEqual(x.grad, excepted)

    def test_gradient_check(self):
        # 生成随机输入值
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        # allclose 判断两个数据是否接近
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
