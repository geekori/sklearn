'''

concatenate函数演示

'''

import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([[5,6]])
print(a)
print(b)

print(np.concatenate((a,b),axis = 0))
'''
[[1 2]
 [3 4]
 [5 6]]
'''

print(np.concatenate((a,b.T),axis = 1))

'''


[[1 2 5]
 [3 4 6]]
'''
print(np.concatenate((a,b),axis = None))

'''

[1 2 3 4 5 6]
'''