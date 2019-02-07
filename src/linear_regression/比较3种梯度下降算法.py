import numpy as np
# 产生100行1列的[0,1)的随机数
X = 2 * np.random.rand(100, 1)
# randn:标准正态分布（噪音）
y = 4 + 3 * X + np.random.randn(100, 1)
# 将X变成(100,2)的矩阵，第1列都是1，每个实例的第1个值是1，x0永远是1
X_b = np.c_[np.ones((100, 1)), X]

eta = 0.1   # 学习率（步长）  哎塔
n_iterations = 1000   # 迭代次数
m = 100





import matplotlib
import matplotlib.pyplot as plt

# 批量梯度下降

theta_path_bgd = []
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
theta_path_bgd = []
def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path_bgd is not None:
            theta_path_bgd.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization
plot_gradient_descent(theta, eta=0.1)

# 随机梯度下降

n_epochs = 50
t0, t1 = 2, 20  # learning schedule hyperparameters
# 逐渐降低学习率，如果t = 0，结果就是0.1
def learning_schedule(t):
    return t0 / (t + t1)


#  迭代50次
theta_path_sgd = []
for epoch in range(n_epochs):
    # 每次从样本中随机选择1个样本计算梯度向量

    for i in range(m):

        # 随机取索引
        random_index = np.random.randint(m)
        #  得到随机样本的x
        xi = X_b[random_index:random_index+1]
        # 得到随机样本的y
        yi = y[random_index:random_index+1]
        # 计算随机样本的梯度向量
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        # 让学习率逐渐减小
        eta = learning_schedule(epoch * m + i)
        # 学习率逐渐减小
        theta = theta - eta * gradients
        if theta_path_sgd is not None:
            theta_path_sgd.append(theta)



# 小批量梯度下降
n_iterations = 50
minibatch_size = 20
t = 0
theta_path_mgd = []
for epoch in range(n_iterations):
    # 返回随机索引[0,m)
    shuffled_indices = np.random.permutation(m)
    # 打乱训练集
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    # 取钱20个
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        if theta_path_mgd is not None:
            theta_path_mgd.append(theta)

theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(8,6))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="随机梯度下降")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="小批量梯度下降")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="批量梯度下降")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])
plt.title("3种梯度下降算法比较", fontsize=16)
plt.show()