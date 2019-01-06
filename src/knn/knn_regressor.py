'''

使用k-邻近算法进行预测

预测：
1. 找到k个距离预测点最近的点
2. 从k个点得到了k个值，计算k个值的平均数（或加权平均数），计算结果就是预测点的预测值

'''

import matplotlib.pyplot as plt
import numpy as np
# 生成训练样本的数量
dot_num = 60
# 产生二维的数组[[a],[b],[c]]，60个区间在[0,1)的均匀分布的浮点数，
# [0, 10)
X = 10 * np.random.rand(dot_num,1)
print(X)
# 生成扁平的一维数组
y = np.cos(X).ravel()
print(y)

# 为标签数据添加一些噪声
y += 0.2 * np.random.rand(dot_num) - 0.1
print(y)

# 训练模型
from sklearn.neighbors import KNeighborsRegressor
k = 5
knn = KNeighborsRegressor(k)
knn.fit(X,y)
# 开始预测
print('预测cos(1.23)的值：',knn.predict([[1.23]]))

print('cos(1.23)的真实值：',np.cos(1.23))

# 计算拟合曲线针对训练样本的拟合准确性
print(knn.score(X,y))

'''
绘制拟合曲线

'''
T = np.linspace(0,10,2000)[:,np.newaxis]
print(T)

a = np.array([1,2,3,4])
print(a)
print(a[:,np.newaxis])

y_pred = knn.predict(T)

print(y_pred)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 绘制拟合曲线
plt.figure(figsize=(16,10),dpi=144)
plt.scatter(X,y,c='b',label='真实数据',s=100)  # 绘制训练样本
# 绘制拟合曲线
plt.plot(T,y_pred,c='r',label='预测数据',lw=2)
plt.legend()
plt.title('k-邻近算法预测（k=%i)' % k)
plt.show()



