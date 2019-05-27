'''

SVM线性回归
'''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
m = 50
X = 2 * np.random.rand(m,1)  # 参数一维的0到2之间的50个随机数

y = (4+3*X + np.random.randn(m,1)).ravel()

from sklearn.svm import LinearSVR

# 在间隔内添加更多的实例并不会影响模型的预测结果，所以这个模型被称为epsilon不敏感
svm_reg1 = LinearSVR(epsilon=1.5,random_state=42)
svm_reg2 = LinearSVR(epsilon=0.5,random_state=42)

svm_reg1.fit(X,y)
svm_reg2.fit(X,y)

print(svm_reg1.predict([[0.31]]))  # 1.5
print(svm_reg2.predict([[0.31]]))  # 0.5




