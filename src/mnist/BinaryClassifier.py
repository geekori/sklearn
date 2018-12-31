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
# 默认的阈值是0
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

'''
调整阈值获得特定的精度和召回率





'''
print('------调整阈值获得特定的精度和召回率-------')
print(Y[12814],Y[1854])  # 9  8

# 计算实例的分数
y_scores = sgd.decision_function([X[12814],X[1854]])
y_digit_pred = (y_scores)
threshold = 0 # 阈值
y_digit_pred = (y_scores > threshold)
print("阈值是0的分类结果：", y_digit_pred)

threshold = 50000
y_digit_pred = (y_scores > threshold)
print("阈值是50000的分类结果：", y_digit_pred)

y_scores = cross_val_predict(sgd,X_train,y_train_9,cv=3,method="decision_function")
print(y_scores)

from sklearn.metrics import precision_recall_curve

precisions,recalls,thresholds = precision_recall_curve(y_train_9, y_scores)

def plot_curve_precision_recall_threshold(precisions,recalls,thresholds):
    # 显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示负号
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 绘制精度和阈值的关系曲线
    plt.plot(thresholds, precisions[:-1],"b-",label="精度",linewidth = 2)
    # 绘制召回率和阈值的关系曲线
    plt.plot(thresholds,recalls[:-1],"r--",label="召回率",linewidth=2)

    plt.xlabel("阈值", fontsize=14)

    plt.legend(loc="upper left",fontsize=14)
    # 设置Y轴的坐标刻度
    plt.ylim([0,1])
    plt.xlim([-700000,700000])

plt.figure(figsize=(10,5))
plot_curve_precision_recall_threshold(precisions,recalls,thresholds)
plt.show()

# 调整阈值，让精度达到90%

threshold = 300000
y_digit_pred_90 = (y_scores > threshold)
# 得到精度
print(precision_score(y_train_9,y_digit_pred_90))
# 得到召回率
print(recall_score(y_train_9,y_digit_pred_90))

threshold = 60000
y_digit_pred1 = (y_scores > threshold)
# 得到精度
print("精度1：",precision_score(y_train_9,y_digit_pred1))
# 得到召回率
print("召回率1：",recall_score(y_train_9,y_digit_pred1))

def plot_precision_recall(precisions, recalls):
    plt.plot(recalls,precisions,"b-",linewidth=3)
    plt.xlabel("召回率",fontsize=16)
    plt.ylabel("精度",fontsize=16)
    plt.axis([0,1,0,1])

plt.figure(figsize=(8,6))
plot_precision_recall(precisions,recalls)
plt.show()

########ROC曲线###########

'''
ROC（Receiver Operating Characteristic）曲线，接收者操作特征曲线

绘制的是真正类率【灵敏度】【TPR】和假正类率【FPR】

FPR是被错误分为正类的父类实例比率。等于1减去真负类率（TNR），也称为特异度。

ROC绘制的是灵敏度和 1-特异度的关系


'''

from sklearn.metrics import roc_curve
fpr,tpr,thresholds = roc_curve(y_train_9,y_scores)
def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=3,label=label)
    plt.plot([0,1],[0,1],'r--')
    plt.axis([0,1,0,1])
    plt.xlabel('假正类率（FPR）',fontsize=16)
    plt.ylabel('真正类率（TPR）',fontsize=16)

plt.figure(figsize=(8,6))
plot_roc_curve(fpr,tpr)
plt.show()

# 计算ROC曲线的面积

# AUC
from sklearn.metrics import roc_auc_score
# AUC越接近1越好
print("ROC AUC：", roc_auc_score(y_train_9,y_scores))  # 0.93


######比较随机森林分类器和梯度下降（SGD）分类器的ROC曲线####

# predict_proba
'''
返回一个n行k列的数组
第i行第j列上的数值是模型预测的第i个预测样本的标签为j的概率，所以每一行的和应该等于1

n：预测的实例数
k：分类数

1行10列
image1  0:概率1   1：概率2



'''

from sklearn.linear_model import LogisticRegression
x_train_1 = np.array([
    [1,2,3],
    [1,3,4],
    [2,1,2],
    [4,5,6],
    [3,5,3],
    [1,7,2]])
y_train_1 = np.array([0,0,0,1,1,1])

x_test_1 = np.array([
    [2,2,2],
    [3,2,6],
    [1,7,4]
])

clf = LogisticRegression()
clf.fit(x_train_1,y_train_1)
print(clf.predict(x_test_1)) # [1 0 1]

print(clf.predict_proba(x_test_1))
'''
[[0.43348191 0.56651809]
 [0.84401838 0.15598162]
 [0.13147498 0.86852502]]
 
 [2,2,2]  0的概率是0.43348191， 1的概率是0.56651809
'''

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=315)

y_probas_forest = cross_val_predict(forest_clf,X_train,y_train_9,cv=3,method="predict_proba")
print(y_probas_forest)
y_scores_forest = y_probas_forest[:,1]
print(y_probas_forest)
fpr_forest, tpr_forest,thresholds = roc_curve(y_train_9,y_scores_forest)

plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,"b:", linewidth=2,label="SGD")
plot_roc_curve(fpr_forest, tpr_forest,"Random Forest")

plt.legend(loc="lower right", fontsize=16)
plt.show()






