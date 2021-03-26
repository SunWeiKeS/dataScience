import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
"""
这份 GPU 的代码是依据之前这份CNN的代码修改的. 
大概修改的地方包括将数据的形式变成 GPU 能读的形式, 然后将 CNN 也变成 GPU 能读的形式. 做法就是在后面加上 .cuda(), 很简单.
"""
# torch.manual_seed(1)

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST,)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# !!!!!!!! 修改 test data 形式 !!!!!!!!! #
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
test_y = test_data.test_labels[:2000].cuda()

# 再来把我们的 CNN 参数也变成 GPU 兼容形式.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

cnn = CNN()

# !!!!!!!! 转换 cnn 去 CUDA !!!!!!!!! #
cnn.cuda()      # Moves all model parameters and buffers to the GPU.

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):

        # !!!!!!!! Change in here !!!!!!!!! #
        b_x = x.cuda()    # Tensor on GPU
        b_y = y.cuda()    # Tensor on GPU

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)

            # !!!!!!!! Change in here !!!!!!!!! #
            pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU

            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)


test_output = cnn(test_x[:10])


# !!!!!!!! 这里有修改 !!!!!!!!! #
pred_y = torch.max(test_output, 1)[1].cuda().data # move the computation in GPU

print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

"""
如果你有些计算还是需要在 CPU 上进行的话呢, 比如 plt 的可视化, 我们需要将这些计算或者数据转移至 CPU.
cpu_data = gpu_data.cpu()

"""