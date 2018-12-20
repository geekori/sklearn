import os
import tarfile
from urllib import request

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'
HOUSING_PATH = os.path.join('datasets','housing')

if not os.path.isdir(HOUSING_PATH):
    os.makedirs(HOUSING_PATH)

tgz_path = os.path.join(HOUSING_PATH,'housing.tgz')
# 下载tgz文件
request.urlretrieve(HOUSING_URL,tgz_path)

housing_tgz = tarfile.open(tgz_path)

# 解压
housing_tgz.extractall(path=HOUSING_PATH)
housing_tgz.close()


