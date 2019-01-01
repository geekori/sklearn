
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
matfn = '../data/mnist-original.mat'
data =sio.loadmat(matfn)
X = np.rot90(data['data'])
Y = np.rot90(data['label'])
digit = X[12814]  # 9

'''
多标签分类

单标签分类

多标签
1.  属于哪一类数字
2.  是否属于奇数

Android

1. 移动领域
2. Java

'''

X_train, X_test, y_train, y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]

y_train_7 = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)

y_multilabel = np.c_[y_train_7,y_train_odd]

print(y_multilabel)

knn_clf = KNeighborsClassifier()  # K临近支持多标签分类
#knn_clf.fit(X_train,y_multilabel)
print('-----------')
#print(knn_clf.predict([digit]))  #[[True,True]]
y_multilabel = np.c_[y_train,y_train_odd]
knn_clf.fit(X_train,y_multilabel)
print(y_multilabel)
print(knn_clf.predict([digit]))  # [[9. 1.]]


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
y_train_knn_pred = cross_val_predict(knn_clf,X_train,y_multilabel,cv=3)
print(f1_score(y_multilabel,y_train_knn_pred,average='macro'))

'''
macro：宏精确率

真实值：A A A C B C A B B C
预测值：A A C B A C A C B C

PA = 正确预测为A类的样本个数 / 预测为A类的样本个数 = 3 / 4 = 75%


'''

