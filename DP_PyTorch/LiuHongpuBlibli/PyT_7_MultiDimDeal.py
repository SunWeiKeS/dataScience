# In[1]: 处理多维特征的输入 无法运行没有数据集
import numpy as np
import torch

# delimiter 分隔符
path=r'G:\# Project\数据集\UsingDataSet\Other_CSV\diabetes.csv'
xy_data = np.loadtxt(path, delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy_data[:, :-1])  # 包左不包右
y_data = torch.from_numpy(xy_data[:, [-1]])


# In[2]: 建立模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model=Model()

# In[3]: 构造损失函数和优化器
criterion = torch.nn.BCELoss(size_average=False)  # 可以继承NN构造自己的损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 可以继承NN构造自己的优化器

# In[4]: 训练
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # backward
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播

    # update
    optimizer.step()  # update更新权重