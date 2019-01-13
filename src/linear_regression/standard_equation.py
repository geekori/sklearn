# 使用标准方程进行线性回归拟合

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 产生一些线性测试数据
# y = 4 + 3x
X = 2 * np.random.rand(100,1)  # 产生的是x1
y = 4 + 3 * X + np.random.randn(100,1)  # 加上噪声
# theta0 = 4   theta1 = 3
plt.plot(X,y,'b.')
plt.xlabel('$x_1$',fontsize=16)
plt.ylabel('$y$',rotation=0,fontsize=16)
plt.axis([0,2,0,15])
plt.show()

# 使用标准方程进行线性回归拟合
Xa = np.c_[np.ones((100,1)),X]
# inv：求逆矩阵
theta_best = np.linalg.inv(Xa.T.dot(Xa)).dot(Xa.T).dot(y)
print(theta_best)

# 开始预测
X_new = np.array([[0],[1.1],[2.1]])
X_new_a = np.c_[np.ones((3,1)),X_new]

print(X_new_a)

y_predict = X_new_a.dot(theta_best)
print(y_predict)

# 绘制模型的预测结果
plt.plot(X_new,y_predict,'r-',linewidth=2,label='Predictions')
plt.plot(X,y,'b.')  # 绘制训练数据的点
plt.xlabel('$x_1$',fontsize=16)
plt.ylabel('$y$',rotation=0,fontsize=16)
plt.legend(loc='upper left', fontsize=14)
plt.axis([0,2,0,15])

plt.show()


# 使用线性回归模型进行预测
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
print(lin_reg.intercept_,lin_reg.coef_)