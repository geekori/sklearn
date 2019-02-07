# 小批量梯度下降算法
import numpy as np
import matplotlib.pyplot as plt
# 产生100行1列的[0,1)的随机数
X = 2 * np.random.rand(100, 1)
# randn:标准正态分布（噪音）
y = 4 + 3 * X + np.random.randn(100, 1)
# 将X变成(100,2)的矩阵，第1列都是1，每个实例的第1个值是1，x0永远是1
X_b = np.c_[np.ones((100, 1)), X]


'''
[[0]
 [2]]
'''
X_new = np.array([[0], [2]])

'''
[[1. 0.]
 [1. 2.]]
'''
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance


m = len(X_b)
np.random.seed(45)
theta = np.random.randn(2,1)  # random initialization

n_iterations = 50
minibatch_size = 20

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

t = 0

for n in range(n_iterations):
    # 返回随机索引[0,m)3
    shuffled_indices = np.random.permutation(m)
    # 打乱训练集
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
print(theta)

