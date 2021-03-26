'''
a.tofile(frame, sep='', format='%s')
• frame : 文件、字符串
• sep : 数据分割字符串，如果是空串，写入文件为二进制
• format : 写入数据的格式
'''
import numpy as np
a=np.arange(100).reshape(5,10,2)
print(a)
a.tofile('b.dat',sep=',',format='%d')
print('success')
#a.tofile('c.dat',format='%d')

'''
np.fromfile(frame, dtype=float, count=‐1, sep='')
• frame : 文件、字符串
• dtype : 读取的数据类型
• count : 读入元素个数，‐1表示读入整个文件
• sep : 数据分割字符串，如果是空串，写入文件为二进制
'''
c=np.fromfile('b.dat',dtype=np.int,sep=',')
print(c)
c=np.fromfile('b.dat',dtype=np.int,sep=',').reshape(5,10,2)
print(c)


'''
该方法需要读取时知道存入文件时数组的维度和元素类型
a.tofile()和np.fromfile()需要配合使用
可以通过元数据文件来存储额外信息
'''