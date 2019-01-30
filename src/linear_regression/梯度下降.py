# 梯度下降

# 预测

from sklearn.linear_model import SGDRegressor
import numpy as np
# 产生1000行，3列，从0到1的浮点数
X = np.random.rand(1000,3)
print(X)
Y = np.random.rand(1,1000)
print(Y)

reg = SGDRegressor()
reg.fit(X,Y[0])  # 开始训练
print(reg.predict([[0.55,0.65,0.54]]))

# 分类
from sklearn.linear_model import SGDClassifier
X = np.random.rand(1000,2)
Y = (X[:,0] + X[:,1] > 0.5)
clf = SGDClassifier()
clf.fit(X,Y)
print(clf.predict([[0.4,0.3]]))
