from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
matfn = '../data/mnist-original.mat'
data=sio.loadmat(matfn)
X = np.rot90(data['data'])
Y = np.rot90(data['label'])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]


# 清除图像噪声

# 为图像添加噪声

noise = np.random.randint(0,100,(len(X_train),784))
x_train_mod = X_train + noise # 生成有噪声的图像
noise = np.random.randint(0,100,(len(X_test),784))
x_test_mod = X_test + noise

y_train_mod = X_train  # 用没有噪声的图像作为标签

zero_index = 5432  # 0

def plot_digit(data):
    image = data.reshape(28,28)
    plt.imshow(image,cmap = matplotlib.cm.binary,interpolation="nearest")
    # 隐藏坐标轴
    plt.axis("off")

plt.subplot(1,2,1);
plot_digit((x_test_mod[zero_index]))
plt.subplot(122);
plot_digit((X_test[zero_index]))

plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train_mod,y_train_mod)
clean_digit = knn_clf.predict([x_test_mod[zero_index]])  # 去除图像上的噪音
plot_digit(clean_digit)
plt.show()
