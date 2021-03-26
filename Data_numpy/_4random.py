'''
NumPy的random随机函数
'''

'''
rand(d0,d1,..,dn) 根据d0‐dn创建随机数数组，浮点数，[0,1)，均匀分布
randn(d0,d1,..,dn) 根据d0‐dn创建随机数数组，标准正态分布
randint(low,high,[shape]) 根据shape创建随机整数或整数数组，范围是[low, high)
seed(s) 随机数种子，s是给定的种子值 #不给定则生产的因时间而异，给定了则固定了

'''
import numpy as np

a=np.random.rand(3,4,5)
print(a)
a=np.random.randn(3,4,5)
print(a)
a=np.random.randint(100,200,(3,4))
print(a)

np.random.seed(10)
a=np.random.randint(100,200,(3,4))
print(a)


'''
shuffle(a) 根据数组a的第1轴进行随排列，改变数组x
permutation(a) 根据数组a的第1轴产生一个新的乱序数组，不改变数组x
choice(a[,size,replace,p]) 从一维数组a中以概率p抽取元素，形成size形状新数组
replace表示是否可以重用元素，默认为False
'''

'''
uniform(low,high,size) 产生具有均匀分布的数组,low起始值,high结束值,size形状
normal(loc,scale,size) 产生具有正态分布的数组,loc均值,scale标准差,size形状
poisson(lam,size) 产生具有泊松分布的数组,lam随机事件发生率,size形状
'''