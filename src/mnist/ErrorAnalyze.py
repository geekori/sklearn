import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
matfn = '../data/mnist-original.mat'
data=sio.loadmat(matfn)
X = np.rot90(data['data'])
Y = np.rot90(data['label'])
digit = X[12814]  # 9

X_train, X_test, y_train, y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(X_train)
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))


sgd_clf = SGDClassifier(random_state=315)
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


'''

通过分析错误类型改进分类模型

'''

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)
conf_mx = confusion_matrix(y_train,y_train_pred)
print(conf_mx)

plt.matshow(conf_mx,cmap=plt.cm.gray)
#plt.show()
# 求每一行的和
row_sums = conf_mx.sum(axis=1)
print(row_sums)

conf_mx_error_ratio = conf_mx / row_sums
print(conf_mx_error_ratio)

np.fill_diagonal(conf_mx_error_ratio,0)
print(conf_mx_error_ratio)

plt.matshow(conf_mx_error_ratio,cmap=plt.cm.gray)
plt.show()

'''
1. 数字3经常被错误第分到5类
2. 其他的数字经常被错误第分到8类和9类
3. 数字8和数字9经常会与其他数字混淆
'''

