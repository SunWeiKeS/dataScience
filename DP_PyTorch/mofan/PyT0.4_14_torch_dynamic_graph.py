import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# 听说过 Torch 的人都听说了 torch 是动态的, 那他的动态到底是什么呢? 我们用一个 RNN 的例子来展示一下动态计算到底长什么样.
# torch.manual_seed(1)    # reproducible

"""
对比静态动态, 我们就得知道谁是静态的. 在流行的神经网络模块中, Tensorflow 就是最典型的静态计算模块. 
下图是一种我在强化学习教程中的 Tensorflow 计算图. 也就是说, 大部分时候, 用 Tensorflow 是先搭建好这样一个计算系统, 
一旦搭建好了, 就不能改动了 (也有例外, 比如dynamic_rnn(), 但是总体来说他还是运用了一个静态思维), 
所有的计算都会在这种图中流动, 当然很多情况, 这样就够了, 我们不需要改动什么结构. 
不动结构当然可以提高效率. 但是一旦计算流程不是静态的, 计算图要变动. 
最典型的例子就是 RNN, 有时候 RNN 的 time step 不会一样, 或者在 training 和 testing 的时候, batch_size 和 time_step 也不一样,
 这时, Tensorflow 就头疼了, Tensorflow 的人也头疼了. 哈哈, 如果用一个动态计算图的 Torch, 我们就好理解多了, 写起来也简单多了.
"""
# Hyper Parameters
INPUT_SIZE = 1          # rnn input size / image width
LR = 0.02               # learning rate


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []                                   # this is where you can find torch is dynamic
        for time_step in range(r_out.size(1)):      # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()                                # the target label is not one-hotted

h_state = None   # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()   # continuously plot

######################## 前面代码都一样, 下面开始不同 #########################

################ 那节内容的代码结构 (静态 time step) ##########
# for step in range(60):
#     start, end = step * np.pi, (step+1)*np.pi   # time steps
#     # use sin predicts cos
#     steps = np.linspace(start, end, 10, dtype=np.float32)

################ 这节内容修改代码 (动态 time step) ###########################
step = 0
for i in range(60):
    dynamic_steps = np.random.randint(1, 4)  # has random time steps
    start, end = step * np.pi, (step + dynamic_steps) * np.pi  # different time steps length
    step += dynamic_steps

    # use sin predicts cos
    steps = np.linspace(start, end, 10 * dynamic_steps, dtype=np.float32)

#######################  这下面又一样了 ###########################
    print(len(steps))       # print how many time step feed to RNN

    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])    # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)   # rnn output
    # !! next step is important !!
    h_state = h_state.data        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)         # cross entropy loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()

"""
有人会说了, Tensorflow 也有类似的功能呀, 比如说 dynamic_rnn(). 
对的, 没错, 不过大家是否想过,
如果我在 Tensorflow 当中定义一个 input 的 placeholder, 这个 placeholder 将会有 (batch, time step, input size) 这几个维度, 
batch 好说, 随便什么大小都可以, 可是 time step 可是固定的呀, 这可不好改, 或者说改起来很麻烦. 
那 PyTorch 中又可以变 batch 又可以变 time step, 这不是很方便吗. 这就体现了动态神经网络的好处.

经过这样的折腾, torch 还能 handle 住, 已经很不容易啦. 
所以当你想要处理这些动态计算图的时候, Torch 还是你首选的神经网络模块.
"""