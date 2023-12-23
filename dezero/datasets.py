import numpy as np

class Dataset:
    # transform 和 target_transform 是数据集（数据和标签）的预处理函数
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train

        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    # 定义了通过方括号访问元素时的操作，如x[0]
    def __getitem__(self, index):
        # 只支持 index 是标量的情况
        assert np.isscalar(index)
        # 调用预处理函数
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])
    
    # 使用 len 函数时返回数据集的长度，如 len(x)
    def __len__(self):
        return len(self.data)
    
    def prepare(self):
        pass

class BigData(Dataset):
    def __getitem__(self, index):
        x = np.load('data/{}.npy'.format(index))
        t = np.load('data/{}.npy'.format(index))
        return x, t
    def __len__():
        return 1000000

# 螺旋数据集
class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)

# Toy datasets
# =============================================================================
def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int_)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t