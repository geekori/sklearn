'''

项目实战：用k-邻近算法预测糖尿病


Pregnancies:怀孕的次数
Glucose:血浆葡萄糖浓度
BloodPressure:舒张压(毫米汞柱)
SkinThickness:肱三头肌皮肤褶皱厚度(毫米)
Insulin:两个小时血清胰岛素(μU毫升)
BMI:身体质量指数,体重除以身高的平方
Diabetes Pedigree Function:糖尿病血统指数,糖尿病和家庭遗传相关
Age:年龄。
Outcome:是否为阳性，1：阳性   0：阴性

'''

# 获取训练数据和测试数据

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('data/diabetes.csv')
print('shape: {}'.format(data.shape))

print(data.head())
print(data.groupby('Outcome').size())

X = data.iloc[:,0:8]  # 取特征数据
Y = data.iloc[:,8]    # 取标签数据
print(Y)
# 拆分成训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)

'''

比较和选择分类模型

KNeighborsClassifier
RadiusNeighborsClassifier

'''

from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier

k = 2
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = k)))
models.append(('KNN-weights', KNeighborsClassifier(n_neighbors = k,weights='distance')))
models.append(('KNN-Radius', RadiusNeighborsClassifier(radius = 500.0)))

results = []
for name,model in models:
    # 训练3个模型
    model.fit(X_train,Y_train)
    results.append((name, model.score(X_test,Y_test)))

# 输出每个模型的分数
for i in range(len(results)):
    print('name:{}; score:{}'.format(results[i][0],results[i][1]))

# 使用交叉数据集训练和校验
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

results = []
for name,model in models:
    kfold = KFold(n_splits = 10)
    cv_result = cross_val_score(model,X,Y,cv=kfold)
    results.append((name,cv_result))

for i in range(len(results)):
    print('name:{}; 分数:{}'.format(results[i][0],results[i][1].mean()))

######训练模型与预测糖尿病

knn = KNeighborsClassifier(n_neighbors = k)
# 训练模型
knn.fit(X_train,Y_train)
# 训练集的分数
train_score = knn.score(X_train,Y_train)

# 测试集的分数
test_score = knn.score(X_test,Y_test)
# 训练集分数：0.8289902280130294；测试集分数：0.7077922077922078
print('训练集分数：{}；测试集分数：{}'.format(train_score,test_score))

# 测试集分数远低于训练集，典型的欠拟合特征   训练数据不足以进行分类或预测
# 葡萄糖含量高，有可能是糖尿病，
X_samples=[[1,152,84,21,0,30.8,0.831,32],[1,185,66,29,0,26.6,0.351,31]]

knn.fit(X,Y)
print(knn.predict(X_samples))

plt.rcParams['font.sans-serif'] = ['SimHei']

'''

绘制学习曲线

学习曲线：训练样本数量与学习效果（分数）的函数

作用：可以直观地看出模型是否会随着训练样本数量的增加，让学习效果越来越好，或持平

1000 

5

20%  40%  60%  80%  100%
200  400  600  800  1000  X 

0.78 0.81  0.83 0.87 0.91 Y

np.linspace(0.2,1,5)
'''

from sklearn.model_selection import learning_curve
# 将数据集随机打乱，然后拆分成训练集和测试集
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=315)
train_sizes,train_scores,test_scores = learning_curve(knn, X, Y,cv=cv,train_sizes=np.linspace(0.1,1,20))

print(train_sizes)
print(train_scores)
print(test_scores)

train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)

plt.grid()

plt.plot(train_sizes,train_scores_mean,'o--',color="r",label="训练集分数")
plt.plot(train_sizes,test_scores_mean,'o--',color="b",label="测试集分数")

plt.legend(loc='best')

plt.show()

'''
得到两个结论
1.  该模型并没有随着训练样本数量的增加，让学习效果变得更好。
2.  两条曲线离得过远，表明是欠拟合。

'''


'''
选择相关特征与数据可视化

Glucose（血糖浓度）
BMI（身体质量指数）【肥胖程度】

'''



from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k = 2)

X_new = selector.fit_transform(X, Y)
print(X_new[0:6])

results = []
for name,model in models:
    kfold = KFold(n_splits = 10)
    cv_result = cross_val_score(model,X_new,Y,cv=kfold)
    results.append((name,cv_result))

for i in range(len(results)):
    print('名称：{};交叉验证分数：{}'.format(results[i][0],results[i][1].mean()))

# 画出数据

plt.figure(figsize=(10,6),dpi=200)
plt.ylabel('身体质量指数（肥胖程度）')
plt.xlabel('血糖浓度')
plt.scatter(X_new[Y==0][:,0],X_new[Y==0][:,1],c='r',s=30,marker='o',label='阴性')
plt.scatter(X_new[Y==1][:,0],X_new[Y==1][:,1],c='b',s=30, marker='^',label='阳性')

plt.legend()

plt.show()





