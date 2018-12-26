'''

直接打开mat图像数据文件

'''

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matfn = '../data/mnist-original.mat'

data = sio.loadmat(matfn)
X = np.rot90(data['data'])
Y = np.rot90(data['label'])
print(X.shape)

digit = X[21354]  # 8
digit_image = digit.reshape(28,28)

print(Y[21354])
plt.imshow(digit_image,cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis('off')
plt.show()



