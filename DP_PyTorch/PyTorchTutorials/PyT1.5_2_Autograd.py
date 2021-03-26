# In[1]: 自动求梯度 反向传播
"""

- PyTorch中所有的神经网络都来自于autograd包
 > - autograd.Variable 这是这个包中最核心的类。 它包装了一个Tensor，
并且几乎支持所有的定义在其上的操作。
 > - 一旦你完成张量计算之后就可以调用.backward()函数,它会帮你把所有的梯度计算好.
 > - 通过Variable的.data属性可以获取到张量.
 > - 通过Variabe的.grad属性可以获取到梯度.
>
![autograd](https://pic4.zhimg.com/v2-08e0530dfd6879ff2bee56cfc5cc5073_b.png)

- 对于实现自动求梯度还有一个很重要的类就是autograd.Function.
 > - Variable跟Function一起构建了非循环图,完成了前向传播的计算.
 > - 每个通过Function函数计算得到的变量都有一个.grad_fn属性.
 > - 用户自己定义的变量(不是通过函数计算得到的)的.grad_fn值为空.
 > - 如果想计算某个变量的梯度,可以调用.backward()函数:
   1. 当变量是标量的时候不需要指定任何参数.
   2. 当变量不是标量的时候,需要指定一个跟该变量同样大小的张量grad_output用来存放计算好的梯度.
"""
import torch
from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad = True)
y = x + 2
#  经查，发现creator属性名称已经改为grad_fn，很多文档还未进行修改
# grad_fn attribute
# 这里的x是属于用户自己定义的,而y属于函数产生的,所以y有grad_fn属性,而x没有.
print (x.grad_fn)
print (y.grad_fn)

# y 是作为一个操作的结果创建的因此y有一个creator
z = y * y * 3
out = z.mean()
print(z, out)

# 现在我们来使用反向传播,该过程可自动求出所有梯度
out.backward()
print(x.grad)
# out.backward()和操作out.backward(torch.Tensor([1.0]))是等价的
# 在此处输出 d(out)/dx
# 这里的out为标量,所以直接调用backward()函数即可.
# 一定要注意当out为数组时,用先定义一样大小的Tensor例如grad_output执行.backgrad(grad_output)语句.

"""
计算方式：

$o=\frac{1}{4}\sum_{i}z_{i},$
$z_{i}=3(x_{i}+2)^{2},$
$z_{i}|_{x_{i}=1}=27,$
$\frac{\partial o}{\partial x_{i}}=\frac{3}{2}(x_{i}+2), $
$\frac{\partial o}{\partial x_{i}}|_{x_{i}=1}=\frac{9}{2}=4.5$
"""
# In[2]: 定义正向和反向传播
"""
- 与numpy不同，PyTorch张量可以利用GPU加速其数字计算。
要在GPU上运行PyTorch Tensor，只需将其转换为新的数据类型。
- 在这里，我们使用PyTorch张量使两层网络适合随机数据。
像上面的numpy示例一样，我们需要手动实现通过网络的正向和反向传递：
"""
import torch

dtype= torch.float
device= torch.device("cuda:0")
# device = torch.device("cpu")

# 判断是否支持GPU加速
# torch.cuda.is_available()
N,D_in,H,D_out=64,1000,100,10

# Create random input and output data
x =torch.randn(N,D_in,device=device,dtype=dtype)
y= torch.randn(N,D_out,device=device,dtype=dtype)

# randomly initialize weights
w1 =torch.randn(D_in,H,device=device,dtype=dtype)
w2 =torch.randn(H,D_out,device=device,dtype=dtype)
learning_rate=1e-6
for t in range(500):
    # Forward pass: compute predicted y
    """
    torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等，
        比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
    torch.mm(a, b)是矩阵a和b矩阵相乘，
        比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
    """
    h=x.mm(w1)

    """
    torch.clamp(input, min, max, out=None) → Tensor
    将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    """
    h_relu=h.clamp(min=0)
    y_pred=h_relu.mm(w2)

    # Compute and print loss
    # 一个元素张量可以用item得到元素值，
    # print(x)和print(x.item())值是不一样的，一个是打印张量，一个是打印元素：
    loss =(y_pred-y).pow(2).sum().item()
    if t% 100==99:
        print(t,loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred =2.0 *(y_pred-y)
    # .t() 意义就是将Tensor进行转置
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
# In[3]:
"""
在PyTorch中，我们可以通过定义torch.autograd.Function
>- 实现forward and backward函数的子类来轻松定义自己的 autograd 运算符。
- 然后，我们可以通过构造实例并像调用函数一样调用新的autograd运算符，并传递包含输入数据的张量。

##### 在此示例中，我们定义了自己的自定义autograd函数来执行ReLU非线性，并使用它来实现我们的两层网络：
"""
import torch

class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
dtype=torch.float
device=torch.device("cpu")
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
learning_rate = 1e-6

for t in range(500):
    # to apply  our function, we use function.apply ,method.we alias this as relu
    relu=MyReLU.apply

    # Forward pass: compute predicted y using operations;
    # we compute ReLU using our custom autograd operation.
    y_pred=relu(x.mm(w1)).mm(w2)

    # compute and print loss
    loss=(y_pred -y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
