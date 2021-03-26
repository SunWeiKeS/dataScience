# In[1]:
import torch
from torch.autograd import Variable # torch 中 Variable 模块
"""
在 Torch 中的 Variable 就是一个存放会变化的值的地理位置. 里面的值会不停的变化. 
就像一个裝鸡蛋的篮子, 鸡蛋数会不停变动. 那谁是里面的鸡蛋呢, 自然就是 Torch 的 Tensor 咯.
 如果用一个 Variable 进行计算, 那返回的也是一个同类型的 Variable.
"""
# 先生鸡蛋
tensor = torch.FloatTensor([[1,2],[3,4]])
# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensor, requires_grad=True)

print(tensor)
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable)
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""
# In[2]:Variable 计算, 梯度

t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2
print(t_out)
print(v_out)    # 7.5
"""
到目前为止, 我们看不出什么不同, 但是时刻记住, Variable 计算时, 
它在背景幕布后面一步步默默地搭建着一个庞大的系统, 叫做计算图, computational graph.
 这个图是用来干嘛的? 原来是将所有的计算步骤 (节点) 都连接起来, 最后进行误差反向传递的时候, 
 一次性将所有 variable 里面的修改幅度 (梯度) 都计算出来, 而 tensor 就没有这个能力啦.
"""
# In[3]: 获取 Variable 里面的数据
"""
直接print(variable)只会输出 Variable 形式的数据,
 在很多时候是用不了的(比如想要用 plt 画图), 所以我们要转换一下, 将它变成 tensor 形式.
"""
print(variable)     #  Variable 形式
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data)    # tensor 形式
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data.numpy())    # numpy 形式
"""
[[ 1.  2.]
 [ 3.  4.]]
"""


