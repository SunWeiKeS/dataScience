"""
需要指定比一系列现有模块更复杂的模型。在这些情况下，您可以通过子类化nn.Module和定义forward来定义自己的模块
"""
# In[1]: TowLayerNet
import torch
class TowLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        """
        在构造函数中，我们实例化两个nn.Linear模块并将其分配为成员变量。
        """
        super(TowLayerNet, self).__init__()
        self.linear1 =torch.nn.Linear(D_in,H)
        self.linear2 =torch.nn.Linear(H,D_out)

    def forward(self,x):
        """
        在前向函数中，我们接受输入数据的张量，并且必须返回输出数据的张量。
        我们可以使用构造函数中定义的模块以及张量中的任意运算符。
        """
        h_relu=self.linear1(x).clamp(min=0)
        y_pred=self.linear2(h_relu)

        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x=torch.randn(N,D_in)
y=torch.randn(N,D_out)

model= TowLayerNet(D_in,H,D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.

criterion =torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)

for t in range(500):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    if t % 100 == 99:
        print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# In[2]: LeNet  LeCun等人（1998年）的 LeNet 体系结构
import torch
from  torch import nn
import torch.nn.functional as F

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 =nn.Conv2d(1,6,3)
        self.conv2 =nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 =nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,int(x.nelement()/x.shape[0]))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

model=LeNet().to(device=device) #将模型放到gpu上



