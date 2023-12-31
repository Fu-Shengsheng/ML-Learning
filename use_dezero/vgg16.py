if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero
import dezero.utils
from PIL import Image
from dezero.models import VGG16

model = VGG16(pretrained=True)

# 随机生成一张假数据
x = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 进行预测
y = model(x)
print(y)

# 打印计算图
model.plot(x)


url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)

x = VGG16.preprocess(img)
print('x.shape after preprocess: ', x.shape)
# 增加用于小批量处理的维度
x = x[np.newaxis]
print('x.shape after newaxis: ', x.shape)

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)

print('y.shape after predict: ', y.shape)
predict_id = np.argmax(y.data)

model.plot(x, to_file='vgg.pdf')
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])



