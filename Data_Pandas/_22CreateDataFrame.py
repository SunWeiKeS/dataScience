# In[1]
import pandas as pd
import numpy as np

'''
DataFrame基本操作类似Series，依据行列索引
DataFrame类型可以由如下类型创建：
• 二维ndarray对象
• 由一维ndarray、列表、字典、元组或Series构成的字典
• Series类型
• 其他的DataFrame类型
'''
#从二维ndarray对象创建==========================================================
#                                   行 列
d=pd.DataFrame(np.arange(10).reshape(2,5))
print(d)
'''
最左边和最上边是自动索引生成
   0  1  2  3  4
0  0  1  2  3  4
1  5  6  7  8  9
'''
#从一维ndarray对象字典创========================================================

dt={'one':pd.Series([1,2,3],index=['a','b','c']),
    'two':pd.Series([9,8,7,6],index=['a','b','c','d'])}
d=pd.DataFrame(dt)
print(d)
print(pd.DataFrame(dt,index=['b','c','d'],columns=['two','three']))
'''
数据根据行列索引自动补齐
   one  two
a  1.0    9
b  2.0    8
c  3.0    7
d  NaN    6
====================================================
     two three
b    8   NaN
c    7   NaN
d    6   NaN
'''

#从列表类型的字典创建====================================================
#     列    列数据
dl={'one':[1,2,3,4],'two':[9,8,7,6]}
d=pd.DataFrame(dl,index=['a','b','c','d'])
print(d)

