# In[1]:
"""
Torch 中提供了一种帮你整理你的数据结构的好东西, 叫做 DataLoader,
我们能用它来包装自己的数据, 进行批训练. 而且批训练可以有很多种途径,


DataLoader 是 torch 给你用来包装你的数据的工具.
所以你要讲自己的 (numpy array 或其他) 数据形式装换成 Tensor,
然后再放进这个包装器中. 使用 DataLoader 有什么好处呢? 就是他们帮你有效地迭代数据,
"""
import torch
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5
# BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

def show_batch():
    for epoch in range(3):   # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())
show_batch()
"""
可以看出, 每步都导出了5个数据进行学习. 然后每个 epoch 的导出数据都是先打乱了以后再导出.

真正方便的还不是这点. 如果我们改变一下 BATCH_SIZE = 8, 这样我们就知道, step=0 会导出8个数据,
但是, step=1 时数据库中的数据不够 8个, 这时怎么办呢:
    这时, 在 step=1 就只给你返回这个 epoch 中剩下的数据就好了.
"""

