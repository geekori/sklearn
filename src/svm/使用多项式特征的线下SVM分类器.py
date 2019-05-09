'''

使用多项式特征的线性SVM分类器

'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X,y = make_moons(n_samples=100,noise=0.15,random_state=42)
print(X)
print(y)
# 搭建一个流水线
polynomial_svm_clf = Pipeline([
    ('poly_features',PolynomialFeatures(degree=3)),  # 3阶多项式
    ('scaler',StandardScaler()),
    ('svm_clf',LinearSVC(C=10,loss='hinge',random_state=42))
])

polynomial_svm_clf.fit(X,y)

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
plot_predictions(polynomial_svm_clf,[-1.5,2.5,-1,1.5])
plot_dataset(X,y,[-1.5,2.5,-1,1.5])

plt.show()
print(polynomial_svm_clf.predict([[-0.75,1],[0.5,-0.5]]))



