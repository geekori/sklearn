'''

使用fetch_mldata函数获得MNIST数据源

'''

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

X = mnist.data     # 数据
Y = mnist.target   # 标签

print(X.shape)   # 28 * 28

digit = X[12846] # 2
digit_image = digit.reshape(28,28)

print(Y[12846])

import matplotlib.pyplot as plt
import matplotlib
plt.imshow(digit_image,cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis('off')
plt.show()

from sklearn.datasets import  get_data_home

print(get_data_home())


