'''

支持向量机：线性SVM分类

决策边界

位于街道边缘的实例被称为支持向量


sklearn提供了一个SVC类，有一个超参数C用于控制这个平衡
C越小，街道越宽

iris：安德森鸢尾花卉数据集，英文全称Anderson's Iris data set

iris包含了150个样本。每一行是一个样本。

样本数据共有4列
data
花萼长度   花萼宽度  花瓣长度   花瓣宽度

target
0  1



'''

from sklearn.svm import SVC
from sklearn import datasets

iris = datasets.load_iris()
print(iris["data"].shape)

# 取最后两列
X = iris["data"][:,(2,3)]
y = iris["target"]

value01 = (y == 0) | (y == 1)
print(value01.shape)

X = X[value01]
y = y[value01]
print(X.shape)

svm_clf = SVC(kernel="linear",C=float("inf"))
svm_clf.fit(X,y)

print(svm_clf.predict([[5.6,1.8]]))
print(svm_clf.predict([[1.6,1.8]]))

