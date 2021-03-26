
# In[4]:
"""
在构建神经网络时，我们经常考虑将计算分为几层，其中一些层具有可学习的参数 ，这些参数将在学习过程中进行优化
- 在PyTorch中，该nn程序包达到了相同的目的。
该nn 包定义了一组Modules，它们大致等效于神经网络层。
模块接收输入张量并计算输出张量，但也可以保持内部状态，例如包含可学习参数的张量。
该nn软件包还定义了一组有用的损失函数，这些函数通常在训练神经网络时使用。
"""
import torch
N, D_in, H, D_out = 64, 1000, 100, 10
x=torch.randn(N,D_in)
y=torch.randn(N,D_out)

"""
Use the nn package to define our model as a sequence of layers.nn.Sequential
is a Module which contains other Modules, and applies them in sequence to
produce its output. Each Linear Module computes output from input using a
linear function, and holds internal Tensors for its weight and bias.
"""
model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out)
)
# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn=torch.nn.MSELoss(reduction='sum')

learning_rate=1e-4
for t in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)

    if t%100==99:
        print(t,loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    loss.backward()

     # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate*param.grad
# In[5]:
"""
- 我们已经通过手动更改持有可学习参数的张量来更新模型的权重
即使用torch.no_grad() 或.data避免在autograd中跟踪历史记录。
- 对于像随机梯度下降这样的简单优化算法来说，这并不是一个沉重的负担，
实践中，我们经常使用更复杂的优化器（例如AdaGrad，RMSProp，Adam等）来训练神经网络。
- optim 提供了常用优化算法的实现。

我们将nn像以前一样使用包来定义模型，但是将使用optim包提供的Adam算法来优化模型
"""
import torch

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate=1e-4
optimizer= torch.optim.Adam(model.parameters(),lr=learning_rate)
for i in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()















