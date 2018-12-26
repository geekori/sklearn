'''

将多张图像文件合成一个图像

'''

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# instances：图像总数
def plot_digits(instances,images_per_row=10):
    size = 28
    # 获取每行图像数，如果图像总数比每行图像数少，那么直接取图像总数
    images_per_row = min(len(instances),images_per_row)
    # 将所有的图像数组转换为2维数组（28*28）
    images = [instance.reshape(size,size) for instance in instances]

    # 计算行数（122 - 1 // 10） + 1 = 13
    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images= []
    # 计算需要设置为0的数组元素个数
    n_empty = n_rows * images_per_row - len(instances)
    # 填充需要设置为0的数组位置
    images.append(np.zeros((size,size * n_empty)))

    for row in range(n_rows):
        rimages = images[row * images_per_row:(row + 1) * images_per_row]
        # 将每行10个图像变成一个由10个数字组成的图像
        row_images.append(np.concatenate(rimages,axis=1))
    # 将多行图形合成一个图像
    image = np.concatenate(row_images,axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')

matfn = '../data/mnist-original.mat'

data = sio.loadmat(matfn)
X = np.rot90(data['data'])
Y = np.rot90(data['label'])
plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600],X[13000:30600:600],X[30600:50000:590]]
plot_digits(example_images,images_per_row=10)
plt.show()







