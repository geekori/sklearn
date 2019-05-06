'''
添加特征使数据集线性可分离

'''

import numpy as np
import matplotlib.pyplot as plt
# 产生均匀分布的数组
X1D = np.linspace(-4,4,9).reshape(-1,1)
X2D = np.c_[X1D,X1D**2]

print(X2D)

plt.figure(figsize=(11,5))
# 将线性不可分离的图放在1行1列上
plt.subplot(121)
plt.grid(True)

y = np.array([0,0,1,1,1,1,1,0,0])
print(X1D[:,0][y==0])
plt.plot(X1D[:,0][y==0],np.zeros(4),'gs')
plt.plot(X1D[:,0][y==1],np.zeros(5),'b^')
plt.xlabel(r'$x_1$',fontsize=18)

plt.axis([-4.5,4.5,-0.2,0.2])
plt.subplot(122)
plt.grid(True)
plt.plot(X2D[:,0][y==0],X2D[:,1][y==0],'gs')
plt.plot(X2D[:,0][y==1],X2D[:,1][y==1],'b^')
plt.xlabel(r'$x_1$',fontsize=18,rotation=0)
plt.plot([-4.5,4.5],[6.5,6.5],'r--',linewidth=5)
plt.axis([-4.5,4.5,-1,17])
plt.subplots_adjust(right=1)
plt.show()
