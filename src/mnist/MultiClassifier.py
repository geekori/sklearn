
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt

matfn = '../data/mnist-original.mat'
data = sio.loadmat(matfn)
X = np.rot90(data['data'])
Y = np.rot90(data['label'])
digit = X[12814]  # 9

X_train, X_test, y_train, y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=315)

'''
多类别分类器

1. 直接使用支持多类别分类的分类器，如随机森林分类器、朴素贝叶斯分类器
2. 建立多个二元分类器，然后不断调用多次二元分类器进行多类别分类
（1）OvA（一对多）
0 到 9 

0-检测器   1-检测器  2-检测器  ...   9-检测器

每个检测器会得到一个决策分数，选择最高的就是最终的分类结果

一对多策略（OvA，one-versus-the-rest）

（2）OvO(一对一）
0-1检测器   0-2检测器、  0-3检测器、 1-2检测器

N * (N-1) / 2 

N = 10，需要创建45个检测器


如何选择OvA和OvO

有一些算法（支持向量机分类器）在数据量大时表现很糟糕，就应该选择OvO。对于大多数二元分类器，OvA还是最好的选择

sklearn中的API会根据类别自动使用OvA进行多类别分类，SVM（支持向量机分类器）分类器会使用OvO策略。


'''

# 训练10个二元分类器
sgd_clf.fit(X_train,y_train)
print(Y[1281])
print(sgd_clf.predict([X[1281]]))  # 8

digit_scores = sgd_clf.decision_function([X[1281]])
print(digit_scores)
print(np.argmax(digit_scores))
print(sgd_clf.classes_)

# 使用OvO策略

from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5,random_state=315))
ovo_clf.fit(X_train,y_train)
print(ovo_clf.predict([X[1281]]))

print(len(ovo_clf.estimators_))  # 45

# 随机森林

from sklearn.ensemble import  RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=315)
forest_clf.fit(X_train,y_train)
print(forest_clf.predict([X[1281]]))

print(forest_clf.predict_proba([X[1281]]))

# 使用k折叠交叉验证评估多类别分类模型

from sklearn.model_selection import cross_val_score

# [0.87226916 0.85755    0.85242786]
print(cross_val_score(sgd_clf,X_train,y_train,cv=3,scoring="accuracy"))
# [0.93845923 0.93315    0.93569035]
print(cross_val_score(forest_clf,X_train,y_train,cv=3,scoring="accuracy"))


#  通过对特征值进行转换提高分类效果

from sklearn.preprocessing import StandardScaler

X = np.array([
    [1,-1,2],
    [2, 0,0],
    [0,1,-1]
])
# 转换为均值为0，方差为1的正态分布
ss = StandardScaler()

print(X)

scaler = ss.fit(X)
'''
[[ 0.         -1.22474487  1.33630621]
 [ 1.22474487  0.         -0.26726124]
 [-1.22474487  1.22474487 -1.06904497]]

'''
transform = scaler.transform(X)
print(transform)
print(transform.mean(axis=0))
print(transform.std(axis=0))


X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(X_train_scaled)
# [0.90796381 0.9016     0.90043507]
print(cross_val_score(sgd_clf,X_train_scaled,y_train,cv=3,scoring="accuracy"))



