# 批量梯度下降

# 为什么叫批量梯度下降：因为可以批量计算theta的值


'''

迭代1000次

'''

import numpy as np

# 产生100行1列的[0,2)的随机数
X = 2 * np.random.rand(100,1)
# 产生带噪音的y
y = 4 + 3 * X + np.random.randn(100,1)
# 将X变成了(100,2)
Xb = np.c_[np.ones((100,1)),X]

# 需要指定3个值
eta = 0.1   # 学习率（步长）
iteration_number = 1000  # 迭代次数
m = 100

# 指定最初的theta值
theta = np.random.randn(2,1)  # 产生theta0和theta1

for iteration in range(iteration_number):
    # 计算梯度向量   dot：点积
    gradients = 2/m * Xb.T.dot(Xb.dot(theta) - y)
    theta = theta - eta * gradients

# theta0 = 4  theta1 = 3

print(theta)

'''
[[3.94216595]
 [3.06018728]]
 
 y = 3.94216595 + 3.06018728 * X 
 
'''

