'''

查看和可视化数据集

准备训练集和测试集

'''

import numpy as np
import pandas as pd

housing = pd.read_csv('./datasets/housing/housing.csv')

#print(housing)
#  select count(field1) from ... group by
#print(housing['ocean_proximity'].value_counts())
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,20))
#plt.show()
np.random.seed(315)
#print(housing.describe())
#print(housing)
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    # 获取测试集的索引序列
    test_indices = shuffled_indices[:test_set_size]
    # 获取训练集的索引序列
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]



train_set,test_set = split_train_test(housing,0.2)

print(len(train_set),"train")
print(len(test_set),"test")
#print(train_set)

'''
如果每次产生的train和test不同，解决方案
1.  将结果保存起来
2.  设置固定的随机种子

产生了新的问题

如果更新数据集，train和test就会被打乱


crc32

 >= 2 * 32 * 20%：训练集
 < 2 * 32 * 20%：测试集
 
 关键点：找到比较稳定的列作为索引列
'''

from zlib import crc32

def test_set_check1(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32
# 取hash编码的最后一个字节（0-255），256 * 0.2 = 51 if < 51 ：test   else  train
import hashlib
def test_set_check2(identifier,test_ratio,hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
print(test_set_check1(5439,0.2))  # test
print(test_set_check1(5438,0.2))  # train

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id:test_set_check2(id,test_ratio))
    return data.loc[~in_test_set],data.loc[in_test_set]

housing_with_id = housing.reset_index() # 为housing添加一个index索引列

train_set,test_set = split_train_test_by_id(housing_with_id,0.2,'index')
print(train_set)

# 可以使用比较稳定的特征值作为id，如经纬度
housing_with_id["id"] = housing['longitude'] * 1000 + housing['latitude']
#print(housing_with_id)
train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"id")
#print(train_set)

'''
用Scikit-Learn API产生训练集和测试集



'''

from sklearn.model_selection import train_test_split

train_set,test_set = train_test_split(housing,test_size=0.2,random_state=315)
housing['median_income'].hist()
#plt.show()

housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat']<5,5.0,inplace=True)
print(housing['income_cat'].value_counts())
housing['income_cat'].hist()
#plt.show()

'''

分层抽样

国家的男女比例： 男:51%   女：49%  

抽样后，需要男：51%   女：49%



'''

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=315)

for train_index,test_index in split.split(housing,housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_test_set['income_cat'].value_counts()/len(strat_test_set))

print(strat_train_set['income_cat'].value_counts()/len(strat_train_set))

print(housing['income_cat'].value_counts()/len(housing))

train_set,test_set = train_test_split(housing,test_size=0.2,random_state=315)
def income_cat_proportions(data):
    return data['income_cat'].value_counts() / len(data)
compare_props = pd.DataFrame({
    "完整数据集":income_cat_proportions(housing),
    "分层抽样测试集":income_cat_proportions(strat_test_set),
    "随机抽样测试集":income_cat_proportions(test_set),
}).sort_index()

print(compare_props)

# 通过可视化地理数据寻找模式


housing = strat_train_set.copy()
#housing.plot(kind='scatter', x = 'longitude',y='latitude', alpha=0.1)

'''
半径（s:表示每个地区的人口数量），颜色表示房价（c）【红色表示高房价】

'''
housing.plot(kind='scatter',x = 'longitude',y='latitude',alpha=0.4,
             s=housing['population']/100,label='population',figsize=(10,7),
             c='median_house_value',cmap=plt.get_cmap('jet'),colorbar=True)
#plt.show()


# 用两种方法检测属性之间的相关度

'''
1. 标准相关系数
corr函数获取标准相关系统（皮尔逊相关系数）

相关系数的取值范围：-1到1  越接近1，表示越强的正相关，  越接近-1，表示越强的负相关
0：表示两个属性没有任何关系

2. Pandas的scatter_matrix函数


进行相关度分析的目的：为了选取和房价相关度很强的属性来预测房价

'''

corr_matrix = housing.corr()
print('---------其他属性与median_house_value属性的相关度')
# 人均收入与平均房价相关度非常大
print(corr_matrix['median_house_value'].sort_values(ascending=False))
# 人数和房屋数有非常强的正相关，而房屋平均年龄与房屋数有非常强的负相关
print(corr_matrix['total_rooms'].sort_values(ascending=False))


# 2. scatter_matrix函数

from pandas.tools.plotting import scatter_matrix

attributes = ['median_house_value','median_income','total_rooms','housing_median_age']

# 清除可能有问题的数据
# housing = housing[housing['median_house_value'] < 490000]

scatter_matrix(housing[attributes],figsize=(12,8))
#plt.show()

# 实验不同属性的组合

# 每户的房间数
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']

# 每间房的卧室数
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']

# 每户的人数
housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()

print(corr_matrix['median_house_value'].sort_values(ascending=False))

housing.plot(kind='scatter',x='rooms_per_household', y = 'median_house_value',alpha = 0.1)
# 0,5：水平坐标   0,520000：纵向坐标
plt.axis([0,5,0,520000])

#plt.show()

# 数据清理-填补缺失值

from sklearn.impute import SimpleImputer

# 平均数（mean）、中位数（median）、出现比较频繁的值（most_frequent）、常量（constant）

imputer = SimpleImputer(strategy = 'median')
# 将ocean_proximity列从housing数据集删除
housing_num = housing.drop('ocean_proximity',axis=1)
'''  
# 适配数据集
imputer.fit(housing_num)
# 输出每一列的中位数
print(imputer.statistics_)
print(housing_num.median().values)

X = imputer.transform(housing_num)   #  Numpy数组
print(X)

housing_tr = pd.DataFrame(X,columns=housing_num.columns)
print(housing_tr)
'''

X = imputer.fit_transform(housing_num)
print(X)
housing_tr = pd.DataFrame(X,columns=housing_num.columns)
print(housing_tr)

'''
处理文本和分类属性

'''

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
housing_ocean_proximity = housing['ocean_proximity']
print(housing_ocean_proximity)
# 将文本按枚举类型转换为数值（0到4）
housing_ocean_proximity_encoded = encoder.fit_transform(housing_ocean_proximity) # NumPy
print(housing_ocean_proximity_encoded)
# 获取所有的枚举值
print(encoder.classes_)

'''
带来的问题：单纯根据枚举值转换，会让算法认为相邻的值相似度高，这和实际情况有些不同

解决方案：

二进制： 
10000
01000
00100
00010
00001

独热编码
'''

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories = 'auto')
housing_ocean_proximity_encoded1 = encoder.fit_transform(housing_ocean_proximity_encoded.reshape(-1,1))
#print(housing_ocean_proximity_encoded.reshape(-1,1))
# 稀疏矩阵（SciPy）
print(housing_ocean_proximity_encoded1.toarray())


# 通过label_binarize将前面的操作合二为一
from sklearn.preprocessing import label_binarize

housing_ocean_proximity_encoded2 = label_binarize(housing_ocean_proximity,['<1H OCEAN','INLAND','ISLAND','NEAR BAY','NEAR OCEAN'],sparse_output=True)

print(housing_ocean_proximity_encoded2.toarray())

'''
自定义转换器


BaseEstimator
TransformerMixin

鸭子类型（duck typing）

fit：返回转换器实例本身

transform：一般返回NumPy数组

'''
from sklearn.base import BaseEstimator,TransformerMixin

class CustomTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self,X,y=None):
        return self
    # NumPy数组
    def transform(self,X,y=None):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household,population_per_household]

transformer = CustomTransformer(add_bedrooms_per_room=False)
#new_values = transformer.transform(housing.values)
new_values = transformer.fit_transform(housing.values)
print(new_values)

'''
数据转换管道（pipeline）

Pipeline



'''

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('custom',CustomTransformer()),
    ('std_scaler',StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
print('-------housing_num_tr----------')
print(housing_num_tr)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        return X[self.attribute_names].values
num_attribs = list(housing_num)
print(num_attribs)
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer',SimpleImputer(strategy='mean')),
    ('custom',CustomTransformer()),
    ('std_scaler', StandardScaler())

])

cat_attribs = ['ocean_proximity']
cat_pipeline = Pipeline([
    ('selector',DataFrameSelector(cat_attribs)),
    ('cat_encoder',OneHotEncoder(sparse=False))
])


from sklearn.pipeline import FeatureUnion
# 并行
full_pipeline = FeatureUnion(transformer_list = [
    ('num_pipeline',num_pipeline),
    ('cat_pipeline',cat_pipeline)
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)










