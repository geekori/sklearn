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

