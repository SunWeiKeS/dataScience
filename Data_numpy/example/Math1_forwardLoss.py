import numpy as np
import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [2, 4, 6]


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_list = []
mse_list = []
for w in np.arange(0.0, 4, 0.1):
    print(u"w= %s" % w)
    l_sum=0
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    for x_val,y_val in zip(x_data,y_data):
        y_pred_val=forward(x_val)
        loss_val=loss(x_val,y_val)
        l_sum +=loss_val
        print('\t', x_val,y_val,y_pred_val,loss_val)
    print('MSE= ',l_sum/len(y_data))
    w_list.append(w)
    mse_list.append(l_sum/len(y_data))

# Draw the graph
plt.plot(w_list, mse_list)
plt.xlabel("loss")
plt.ylabel("w")
plt.show()

