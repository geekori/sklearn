'''

用线性模型拟合非线性数据


y = 3 * x^2 + 2*x + 10

将非线性的方程变成线性的方程

x^2
x

从一元二次方程变成了二元一次方程

将x扩展成[x,x^2]

用线性模型拟合非线性数据的方式
1. 降阶
2. 扩展X


'''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(45)

m = 100
X = 6 * np.random.rand(m,1) - 3
#print(X)
# 一元二次方程
y = 0.8 * X**2 + 0.4 * X + 1.4 + np.random.randn(m,1)

plt.plot(X,y,'b.')
plt.xlabel("x",fontsize=18)
plt.ylabel("y",fontsize=18,rotation=0)
plt.axis([-3,3,0,10])
plt.show()

# 扩展X和降阶
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# degree：方程的阶数   include_bias：是否包含偏差列，默认是True
poly_features = PolynomialFeatures(degree=2,include_bias=False)
X_poly = poly_features.fit_transform(X)
#print(X_poly)

# 使用线性模型拟合非线性数据

lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
# [1.59723271] [[0.38165307 0.74859477]]
# theta0：1.59723271   标准值：1.4
# theta1：0.38165307   标准值：0.4
# theta2：0.74859477   标准值：0.8
print(lin_reg.intercept_,lin_reg.coef_)


# 均匀分布
X_new = np.linspace(-3,3,100).reshape(100,1)

X_new_poly = poly_features.transform(X_new)

# 用多项式模型进行预测
y_new = lin_reg.predict(X_new_poly)

plt.plot(X,y,"b.")
plt.plot(X_new,y_new,"r-",linewidth=3,label="Predictions")
plt.xlabel("x",fontsize=18)
plt.ylabel("y",fontsize=18,rotation=0)
plt.axis([-3,3,0,10])
plt.legend(loc="upper left",fontsize=14)
plt.show()