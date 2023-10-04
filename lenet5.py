import torch
from torch import nn
from torch.nn import functional as F 

class Lenet5(nn.Module):
    # describe the structure of the network
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit=nn.Sequential(
            # x:[batch,3,32,32] => [batch,6,28,28] 
            # Tip 1
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0), 
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            ) 
        # -> [batch,16,5,5]

        # flatten class, which is not in pytorch
        self.fc_unit=nn.Sequential( 
                nn.Linear(16*5*5,120),
                nn.ReLU(),
                nn.Linear(120,84),
                nn.ReLU(),
                nn.Linear(84,10),
            )
        
        # classification task using Cross Entropy Loss
        self.criteon=nn.CrossEntropyLoss()
        
        # tmp=torch.randn(2,3,32,32)
        # out=self.conv_unit(tmp)
        # print('conv out:',out.shape)

    # define the forward propagation
    def forward(self,x):
        """
        :param x: [b,3,32,32]
        :return:
        """
        # [b,3,32,32] => [b,16,5,5]
        x=self.conv_unit(x)
        batch_size=x.size(0)
        x=x.view(batch_size,16*5*5)
        logits=self.fc_unit(x)
         
        # loss=self.criteon(logits,y)
        # [b,10]
        return logits


def main():
    net=Lenet5()

if __name__ == '__main__':
    main()

'''
Tip 1: 卷积神经网络的输入输出维度变化：
这段代码定义了一个卷积神经网络的一部分，称为 `self.conv_unit`，它由两个卷积层和两个平均池化层组成。

初始维度是 `[batch, 3, 32, 32]`，它的各个维度的含义如下：

- `batch`: 批次大小，表示同时处理的样本数量。
- `3`: 表示输入图像的通道数，即彩色图像的通道数 (R,G,B)。
- `32`: 输入图像的高度。
- `32`: 输入图像的宽度。

经过每一层后，输入的维度会发生变化：

1. 第一个卷积层 `nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)`：
   - 输入通道数从 `3` 变成了 `6`。
   - 由于没有 padding (`padding=0`)，所以输入图像的高度和宽度会减少。输出的特征图的高度和宽度变为 `28`。

2. 第一个平均池化层 `nn.AvgPool2d(kernel_size=2, stride=2, padding=0)`：
   - 使用 2x2 的平均池化，将特征图的高度和宽度减半。
   - 输出的特征图的高度和宽度变为 `14`。

3. 第二个卷积层 `nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)`：
   - 输入通道数从 `6` 变成了 `16`。
   - 由于没有 padding (`padding=0`)，所以输入图像的高度和宽度会减少。输出的特征图的高度和宽度变为 `10`。

4. 第二个平均池化层 `nn.AvgPool2d(kernel_size=2, stride=2, padding=0)`：
   - 再次使用 2x2 的平均池化，将特征图的高度和宽度减半。
   - 输出的特征图的高度和宽度变为 `5`。

最终，经过这四个层的处理后，特征图的维度变为 `[batch, 16, 5, 5]`。这表示在网络的这一部分处理下，每个样本被映射为一个 `16` 通道、高度和宽度为 `5` 的特征图。
'''