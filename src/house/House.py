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
