'''
SVM非线性回归

'''

import numpy as np

np.random.seed(42)
m = 100
X = 2 * np.random.rand(m,1) - 1
y = (0.3 + 0.2 * X + 0.4 * X**2 + np.random.randn(m,1)/10).ravel()

from sklearn.svm import SVR
svm_poly_reg1 = SVR(kernel="poly",degree=2,C=100,epsilon=0.1)
svm_poly_reg2 = SVR(kernel="poly",degree=2,C=0.01,epsilon=0.1)
svm_poly_reg1.fit(X,y)
svm_poly_reg2.fit(X,y)

print(svm_poly_reg1.predict([[0.34]]))
print(svm_poly_reg2.predict([[0.34]]))
