# 批量梯度下降

# 为什么叫批量梯度下降：因为可以批量计算theta的值


'''

迭代1000次

'''

import numpy as np
import matplotlib.pyplot as plt
# 产生100行1列的[0,2)的随机数
X = 2 * np.random.rand(100,1)
# 产生带噪音的y
y = 4 + 3 * X + np.random.randn(100,1)
# 将X变成了(100,2)


Xb = np.c_[np.ones((100,1)),X]

X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)),X_new]
'''
[[1,0]
  [1,2]
]
'''
def draw(theta,eta):
    # 得到样本数
    m = len(Xb)
    # 绘制样本离散点
    plt.plot(X,y,"c.")
    # 设置迭代次数
    iteration_number = 1000
    # 进行迭代
    for iteration in range(iteration_number):
        # 对前20次迭代绘制直线
        if iteration < 20:
            y_predict = X_new_b.dot(theta)  # 计算出当前theta值的预测结果（直线上的两个点）
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new,y_predict,style)
        gradients = 2 / m * Xb.T.dot(Xb.dot(theta) - y)
        theta = theta - eta * gradients

    plt.xlabel("x",fontsize=18)
    plt.axis([0,2,0,15])
    plt.title(r"$\eta = {}$".format(eta),fontsize=18)
    print(theta)

np.random.seed(45)
theta = np.random.randn(2,1)  # 产生初始的theta值
plt.figure(figsize=(10,5))
# 1行  3列   当前位置索引：1
plt.subplot(131)
draw(theta,eta=0.01)
'''
[[4.0229343]
 [3.0276842]]
'''
plt.ylabel("y",rotation = 0,fontsize=18)

plt.subplot(132)
draw(theta,eta=0.1)
'''
[[4.0638058 ]
 [2.99408138]]

'''
plt.subplot(133)
draw(theta,eta=0.5)  # 没有办法得到正确的theta0和theta1
'''
[[-2.21775593e+93]
 [-2.69748213e+93]]
'''
plt.show()


