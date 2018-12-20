'''
训练线性回归模型，并预测幸福指数

2017

国家人均GDP

国家的人们幸福指数

1. 装载数据  ok
2. 整理数据  ok
3. 连接两个数据集 ok
4. 绘制一个散点图 ok
5. 选择模型（线性回归模型），开始训练
6. 开始预测


透视表


步骤

1. 准备训练数据
2. 选择模型
3. 开始训练
4. 进行预测

'''

import pandas as pd
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt

oecd_bli = pd.read_csv('./oecd_bli_2017.csv',thousands=',')
#print(oecd_bli)
# n/a  NaN
gdp_per_capita = pd.read_csv('./gdp_per_capita.csv',thousands=',',na_values='n/a')
#print(gdp_per_capita)


# 整理和合并数据
def prepare_country_data(oecd_bli, gdp_per_capita):
    # 过滤出整体的幸福指数数据
    oecd_bli = oecd_bli[oecd_bli['INEQUALITY'] == 'TOT']
    # 创建数据透视表
    oecd_bli = oecd_bli.pivot(index="Country",columns="Indicator",values="Value")
    #print(oecd_bli)
    # 该列的名字
    gdp_per_capita.rename(columns={"2017":"GDP per capita"}, inplace=True)

    # 将Country列设置为索引
    gdp_per_capita.set_index("Country", inplace=True)
    #print(gdp_per_capita)

    # 将两个数据集合成一个  数据集名和用于连接的字段
    full_country_data = pd.merge(left=oecd_bli,right=gdp_per_capita, left_index=True,right_index=True)
    full_country_data.sort_values(by='GDP per capita',inplace=True)

    return full_country_data[["GDP per capita",'Life satisfaction']]


country_data = prepare_country_data(oecd_bli,gdp_per_capita)


# 开始绘制散点图
X = np.c_[country_data["GDP per capita"]]
print(X)
Y = np.c_[country_data["Life satisfaction"]]
print(Y)

country_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# 选择模型
model1 = sklearn.linear_model.LinearRegression()
model2 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# 训练模型
model1.fit(X,Y)
model2.fit(X,Y)

# 预测
newX = [[54673],[54213],[12345],[7495]]
print(model1.predict(newX))
print(model2.predict(newX))





