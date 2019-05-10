'''
高斯RBF的相似特征

RBF（高斯径向基函数）

使用RBF作为相似函数

地标（landmark）



2个地标x1=-2  x2=1

y = 0.3

x = -1
e = 2.71828
x = exp(-0.3 * 1^2) = 0.74
y = exp(-0.3 * 2^2) = 0.3
'''

import numpy as np
import matplotlib.pyplot as plt

X1D = np.linspace(-4,4,9).reshape(-1,1)
print(X1D)

# 用于计算高斯RBF
def gaussian_rbf(x,landmark,gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark,axis=1)**2)

gamma = 0.3

XK= np.c_[gaussian_rbf(X1D,-2,gamma),gaussian_rbf(X1D,1,gamma)]
print(XK)
yk = np.array([0,0,1,1,1,1,1,0,0])
plt.figure(figsize=(11,5))

plt.subplot(121)

plt.grid(True)
plt.axhline(y=0,color='k')
plt.axvline(x=0,color='k')
plt.scatter(x=[-2,1],y=[0,0],s=150,alpha=0.5,c='red')
plt.plot(X1D[:,0][yk==0],np.zeros(4),'bs')
plt.plot(X1D[:,0][yk==1],np.zeros(5),'g^')
plt.gca().get_yaxis().set_ticks([0,0.25,0.5,0.75,1])
plt.axis([-4.5,4.5,-0.1,1.1])

plt.subplot(122)

plt.grid(True)
plt.axhline(y=0,color='k')
plt.axvline(x=0,color='k')

plt.plot(XK[:,0][yk==0],XK[:,1][yk==0],'bs')
plt.plot(XK[:,0][yk==1],XK[:,1][yk==1],'g^')

plt.plot([-0.1,1.1],[0.57,-0.1],'r--',linewidth=5)
plt.axis([-0.1,1.1,-0.1,1.1])

plt.subplots_adjust()
plt.show()