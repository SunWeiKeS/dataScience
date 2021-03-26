# In[1]: 1 实现线性回归
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 1,1是指维度

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
# 2构建损失函数和优化器
criterion = torch.nn.MSELoss(size_average=False)  # size_average 是否给每一个批量求均值
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # update更新权重

# 打印输出
print('w= ',model.linear.weight.item())
print('b= ',model.linear.bias.item())

# 测试模型
x_test=torch.Tensor([[4.0]])
y_test= model(x_test)
print('y_pred= ',y_test.data)