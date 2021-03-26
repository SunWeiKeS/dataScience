import pandas as pd
import numpy as np

'''
算术运算规则
算术运算根据行列索引，补齐后运算，运算默认产生浮点数
补齐时缺项填充NaN (空值)
二维和一维、一维和零维间为广播运算
采用+ ‐ * /符号进行的二元运算产生新的对象
'''
a=pd.DataFrame(np.arange(12).reshape(3,4))
print(a)
b=pd.DataFrame(np.arange(20).reshape(4,5))
print(b)



# 自动补齐，缺项补NaN
print(a+b)
print(a*b)


'''
方法形式的运算
方法                            说明
.add(d, **argws)              类型间加法运算，可选参数
.sub(d, **argws)              类型间减法运算，可选参数
.mul(d, **argws)              类型间乘法运算，可选参数
.div(d, **argws)              类型间除法运算，可选参数
'''
print('====================================================================')
print(b.add(a,fill_value=100))
print(a.mul(b,fill_value=0))# 也满足如果没有数据就用NaN补齐

print('====================================================================')

c=pd.Series(np.arange(4))
print(c)
print(c-10)
'''
b
    0   1   2   3   4
0   0   1   2   3   4
1   5   6   7   8   9
2  10  11  12  13  14
3  15  16  17  18  19
'''
#不同维度间为广播运算，一维Series默认在轴1参与运算
print(b-c)
'''
b-c
      0     1     2     3   4
0   0.0   0.0   0.0   0.0 NaN
1   5.0   5.0   5.0   5.0 NaN
2  10.0  10.0  10.0  10.0 NaN
3  15.0  15.0  15.0  15.0 NaN
'''
#使用运算方法可以令一维Series参与轴0运算
print(b.sub(c,axis=0))
'''
b-c
    0   1   2   3   4
0   0   1   2   3   4
1   4   5   6   7   8
2   8   9  10  11  12
3  12  13  14  15  16
'''



