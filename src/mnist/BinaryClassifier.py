'''

训练二元分类器

'''

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matfn = '../data/mnist-original.mat'

data = sio.loadmat(matfn)
X = np.rot90(data['data'])
Y = np.rot90(data['label'])

digit = X[1287]  # 8
digit_image = digit.reshape(28,28)
plt.imshow(digit_image,cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis('off')
plt.show()

X_train,X_test,y_train,y_test = X[:60000],X[60000:],Y[:60000],Y[60000:]

y_train_9 = (y_train == 9) # 只有等于9的设为True，其他的全是False
y_test_9 = (y_test == 9)

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(random_state = 315)
sgd.fit(X_train,y_train_9)

print(sgd.predict([X[12814],X[1287]]))

##### 使用K-fold交叉验证法评估分类器模型的性能 ######

'''
K-fold（K折叠）交叉验证法

将一个非常大的数据集分成K份

K = 5

a b c d e

a：test   b c d e train
b: test   a c d e train

得到5个分数

最后算一下5个分数的平均分，在选择多个模型时，平均分谁高就选谁


'''

from sklearn.model_selection import cross_val_score
# cv：折叠数
# https://scikit-learn.org/stable/modules/model_evaluation.html
print(cross_val_score(sgd,X_train,y_train_9,cv=3,scoring="accuracy"))

print((0.85240738+0.9264+0.92869643)/3)  # 90%

# 70000张图片大概只有10%是9，就算瞎猜，也有90%的可能猜中某张图片不是9

from sklearn.base import BaseEstimator

class Never9Classifier(BaseEstimator):
    def fit(self,X,y=None):
        pass

    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)

never9 = Never9Classifier()

print(cross_val_score(never9,X_train,y_train_9,cv=3,scoring="accuracy"))
print((0.6521+1+1)/3 ) # 88%

#####使用混淆矩阵评估分类器模型的性能#######

'''
混淆矩阵

[
[ X1,X2]
[ X3,X4]
]

第一行：所有"非9"类图片的分类情况
第二行：所有"9"类图片的分类情况

X3+X4大概是X1 + X2的1/9

X1："非9"类图片被识别正确的，【真负类】 真：识别正确   负：识别为"非9类"
X2："非9"类图片被识别错误的，【假正类】 假：识别错误   正：识别为"9类"

X3："9"类图片被识别错误，【假负类】
X4："9"类图片被识别正确，【真正类】

[
[ 真负类,假正类]
[ 假负类,真正类]
]


from 
'''

# 返回预测结果
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd,X_train,y_train_9,cv=3)
print(y_train_9)
print(y_train_pred)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_9,y_train_pred))
'''
[[48908  4134]
 [ 1716  5242]]
 
真负类：48908
假正类：4134

假负类：1716
真正类：5242




'''
y_train_perfect_predictions = y_train_9
print(confusion_matrix(y_train_9,y_train_perfect_predictions))

##### 精度（precision）、召回率（recall）和F1分数 ############

from sklearn.metrics import precision_score,recall_score

print("精度：",precision_score(y_train_9,y_train_pred))  # 56%
print(5242 / (5242 + 4134))

print("召回率：",recall_score(y_train_9,y_train_pred))   # 75%
print(5242 / (5242 + 1716))

from sklearn.metrics import f1_score
print("F1分数",f1_score(y_train_9,y_train_pred))  # 64%
print(5242 / (5242 + (1716 + 4134)/2))

'''
过滤适合于儿童观看的视频
无害视频：True
有害视频：False

低召回率、高准确率（也就是说，将有害视频误识别为无害识别的概率低）

小偷：True
非小偷：False
用于根据行为识别商城中的小偷，希望漏网的越少越好，希望高召回率（99%）
精度，40%， 可能误报会多一些，最后需要加入一个人工审核的流程


'''



