'''
使用sklearn内置的图像数据集

'''

from sklearn import datasets

digits = datasets.load_digits()
print(digits)

# data：用于训练或测试的图像数据
# target：图像数据的标签
X,Y = digits['data'],digits['target']
print(X)
print(Y)
print(X.shape)   # 8 * 8

digit = X[21]
digit_image = digit.reshape(8,8)  # 将一维数组变成二维数组
print(Y[21])
import matplotlib.pyplot as plt
import matplotlib
plt.imshow(digit_image,cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis('off')
plt.show()





