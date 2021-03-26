import pandas as pd
import numpy as np
'''
比较运算法则
比较运算只能比较相同索引的元素，不进行补齐
二维和一维、一维和零维间为广播运算
采用> < >= <= == !=等符号进行的二元运算产生布尔对象
'''
a=pd.DataFrame(np.arange(12).reshape(3,4))
print(a)
b=pd.DataFrame(np.arange(12,0,-1).reshape(3,4))
print(b)

#同维度运算，尺寸一致
print(a>b)
print(a==b)

print('============================================================')
c=pd.Series(np.arange(4))
print(c)
print('============================================================')

#不同维度，广播运算，默认在1轴
print(a>c)
print('============================================================')

print(c>0)














