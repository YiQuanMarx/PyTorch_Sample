import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,):
        super(Net,self).__init__()
        se










num_time_steps=50
start=np.random.randint(3,size=1)[0]
# 等距离采样
time_steps=np.linspace(start,start+10,num_time_steps)
# print(time_steps)
data=np.sin(time_steps)
# 代码将一个数组 data 重新塑形（reshape）成一个新的形状，具体地说是将其变成了一个二维数组，其中行数是 num_time_steps，列数是1。
data=data.reshape((num_time_steps,1))
# print(data)
# data=data.reshape((50,1))
x=torch.tensor(data[:-1]).float().view(1,num_time_steps-1,1)
print(x)