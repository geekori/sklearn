'''

用k-邻近算法进行分类

1. 产生训练集

2. 计算带分类点到训练集样本点的距离

3. 挑出距离分类点最近的k个点（5）

4. 统计所有分类在k个点占有的席位



'''

import  matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs

# 定义3个中心店的坐标

centers = [[-1,2],[3,2],[0,4]]

# 产生训练集（60个点）
'''
n_samples：训练集样本数
centers：指定中心点
cluster_std：方差，值越大，说明产生的点越松散（距离中心点越远）

'''
X,y = make_blobs(n_samples=60,centers=centers,random_state=315,cluster_std=0.65)
print(X)
print(y)

plt.figure(figsize=(16,10),dpi=144)

c = np.array(centers)
# 100： 10*10
plt.scatter(X[:,0],X[:,1],c=y,s=100,cmap='cool');  # 绘制样本（60个点）
plt.scatter(c[0,0],c[0,1],s=200,label='0',marker='^',c='blue') # 绘制中心点1
plt.scatter(c[1,0],c[1,1],s=200,label='1',marker='^',c='red') # 绘制中心点2
plt.scatter(c[2,0],c[2,1],s=200,label='2',marker='^',c='orange') # 绘制中心点3
plt.legend()



# 开始使用k-邻近算法分类

from sklearn.neighbors import KNeighborsClassifier

# 模型训练
k = 5
clf = KNeighborsClassifier(n_neighbors=k)

clf.fit(X,y)

# 进行分类

X_sample = np.array([[1.4,3.5]])
# 开始分类，[2]
print(clf.predict(X_sample))

# 返回距离最近的k个点
neighbors = clf.kneighbors(X_sample,return_distance=False)
print(neighbors)

plt.scatter(X_sample[:,0],X_sample[:,1],marker='*',c='r',s=300)



counter = [0,0,0]
for i in neighbors[0]:
    plt.plot([X[i][0],X_sample[0,0]],[X[i][1],X_sample[0,1]],'k--',linewidth=1)
    counter[y[i]] += 1
plt.show()
print(counter)

