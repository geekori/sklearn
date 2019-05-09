'''
基于多项式核的SVM分类器

'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline



X,y = make_moons(n_samples=100,noise=0.15,random_state=42)

ploy_kernel_svm_clf = Pipeline([
    ('scaler',StandardScaler()),
    ('svm_clf',SVC(kernel='poly',degree=3,coef0=1,C=5))
])
ploy_kernel_svm_clf.fit(X,y)
ploy10_kernel_svm_clf = Pipeline([
    ('scaler',StandardScaler()),
    ('svm_clf',SVC(kernel='poly',degree=10,coef0=100,C=5))
])
ploy10_kernel_svm_clf.fit(X,y)
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

plt.figure(figsize=(11,5))
plt.subplot(121)
plot_predictions(ploy_kernel_svm_clf,[-1.5,2.5,-1,1.5])
plot_dataset(X,y,[-1.5,2.5,-1,1.5])
plt.title('$d=3,r=1,C=5$',fontsize=18)


plt.subplot(122)
plot_predictions(ploy10_kernel_svm_clf,[-1.5,2.5,-1,1.5])
plot_dataset(X,y,[-1.5,2.5,-1,1.5])
plt.title('$d=10,r=100,C=5$',fontsize=18)
plt.show()

print(ploy_kernel_svm_clf.predict([[-0.75,1],[0.5,-0.5]]))
print(ploy10_kernel_svm_clf.predict([[-0.75,1],[0.5,-0.5]]))
