'''
基于高斯RBF核函数的SVM分类器

'''

import numpy as np
import  matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons

X,y = make_moons(n_samples=100,noise=0.15,random_state=42)


rbf_kernel_svm_clf=Pipeline([
    ('scaler',StandardScaler()),
    ('svm_clf',SVC(kernel='rbf',gamma=5,C=0.001))
])


rbf_kernel_svm_clf.fit(X,y)
print(rbf_kernel_svm_clf.predict([[4.1,0.3]]))

gamma1,gamma2= 0.1,5
C1,C2 = 0.001,1000

hyperparams = (gamma1,C1),(gamma1,C2),(gamma2,C1),(gamma2,C2)

svm_clfs = []

# 绘制样本点
def plot_dataset(X,y,axes):
    # 分类为0的蓝色方块
    plt.plot(X[:,0][y==0],X[:,1][y==0],'bs')
    # 分类为1的绿色三角
    plt.plot(X[:,0][y==1],X[:,1][y==1],'g^')
    plt.axis(axes)
    plt.grid(True)
    plt.xlabel('$x_1$',fontsize=18)
    plt.ylabel('$x_2$', fontsize=18,rotation=0)
# 用于绘制分类曲线
def plot_predictions(clf,axes):
    x0s = np.linspace(axes[0],axes[1],100)
    x1s = np.linspace(axes[2],axes[3],100)
    x0,x1 = np.meshgrid(x0s,x1s)
    X = np.c_[x0.ravel(),x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0,x1,y_pred,cmap=plt.cm.brg,alpha=0.2)

for gamma,C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))
    ])
    rbf_kernel_svm_clf.fit(X,y)
    svm_clfs.append(rbf_kernel_svm_clf)

plt.figure(figsize=(11,7))

for i,svm_clf in enumerate(svm_clfs):
    plt.subplot(221 + i)
    plot_predictions(svm_clf,[-1.5,2.5,-1,1.5])
    plot_dataset(X,y,[-1.5,2.5,-1,1.5])
    gamma,C = hyperparams[i]
    plt.title('$\gamma={},C={}$'.format(gamma,C),fontsize=16)
plt.show()
