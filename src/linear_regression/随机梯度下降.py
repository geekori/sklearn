# 随机梯度下降

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

iteration_number = 50

# 在迭代的过程中，让学习率（步长）逐渐减少

t0,t1 = 2, 20
# 通过t的不断增加，让函数返回值远离0.1
def learning_schedule(t):
    return t0 / (t + t1)

for n in range(iteration_number):
    for i in range(m):
        if n ==0 and i <20:
            y_predict = X_new_b.dot(theta)
            style = "b-" if i > 0 else "r--"
            plt.plot(X_new,y_predict,style)
        # 随机取索引
        random_index = np.random.randint(m)

        # 得到随机样本X
        x_random = X_b[random_index:random_index+1]

        # 得到随机样本Y
        y_random = y[random_index:random_index+1]

        gradients = 2 * x_random.T.dot(x_random.dot(theta) - y_random)
        # 让学习率逐渐减小
        eta = learning_schedule(n * m + i)
        print(eta)
        theta = theta - eta * gradients

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('随机梯度下降',fontsize=16)
plt.plot(X,y, "c.")
plt.xlabel("x",fontsize=18)
plt.ylabel("y",rotation =0,fontsize=18)
plt.axis([0,2,0,15])
print(theta)
plt.show()


# 使用sklearn API实现随机梯度下降算法
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50,penalty=None,eta0=0.1,random_state=45)
sgd_reg.fit(X,y.ravel())

print(sgd_reg.intercept_,sgd_reg.coef_)
