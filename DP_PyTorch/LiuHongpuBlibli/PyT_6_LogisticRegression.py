# In[2]: 1 逻辑回归
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

import torch.nn.functional as F


class LogisticRegressionModle(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModle, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

        # Nx1 =Nx8 * 8x1 * Nx1  (8,1) 输入维度8 输出维度1
        # self.linear = torch.nn.Linear(8, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

# In[2]: 训练
model = LogisticRegressionModle()
# 2构建损失函数和优化器
criterion = torch.nn.BCELoss(size_average=False)  # 可以继承NN构造自己的损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 可以继承NN构造自己的优化器

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # update更新权重

# In[3]: 绘图
# 测试输出
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))  # 变成200行1列的矩阵 类似于reshape
y_t =model(x_t)
y=y_t.data.numpy() # 拿到数组

plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()
