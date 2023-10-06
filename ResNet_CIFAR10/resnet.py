import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        # Tip2
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2= nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        if ch_in!=ch_out:
            # [b,ch_in,h,w] => [b,ch_out,h,w]
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )
        else:
            self.extra=nn.Sequential()
 
    def forward(self,x):
        """
        :param x: [b,ch,h,w]
        :return:
        """
        # Tip 3
        # element-wise add: [b,ch_in,h,w] with [b,ch_out,h,w]
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.extra(x)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=3,padding=1),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b,64,h,w] => [b,128,h,w]
        self.blk1=ResidualBlock(64,128,stride=2)
        # [b,128,h,w] => [b,256,h,w]
        self.blk2=ResidualBlock(128,256,stride=2)
        # [b,256,h,w] => [b,512,h,w]
        self.blk3=ResidualBlock(256,512,stride=2)
        # [b,512,h,w] => [b,512,h,w]
        self.blk4=ResidualBlock(512,512,stride=1)
        self.outlayer=nn.Linear(512*1*1,10)

    def forward(self,x):
      # print('input:',x.shape)
      x=F.relu(self.conv1(x))
      # print('after conv0:',x.shape)
      # [b,64,h,w] => [b,1024,h,w]
      x=self.blk1(x)
      # print('after conv1:',x.shape)
      x=self.blk2(x)
      x=self.blk3(x)
      # print('after conv2:',x.shape)
      x=self.blk4(x)
      # print('after conv3:',x.shape)
      # [b,512,h,w] => [b,512,1,1]
      x=F.adaptive_avg_pool2d(x,[1,1])
      # print('after pooling:',x.shape)
      # Tip 5
      x=x.view(x.size(0),-1)
      # print('after view:',x.shape)
      x=self.outlayer(x)
      return x
    
def main():
      # change h and w
      blk=ResidualBlock(64,128,stride=4)
      tmp=torch.randn(2,64,32,32)
      out=blk(tmp)
      # print('block:',out.shape)

      x=torch.randn(2,3,32,32)
      model=ResNet18()
      out=model(x)
      # print('resnet:',out.shape)

if __name__ == '__main__':
    main()




"""
Tip 2: 批归一化（Batch Normalization，简称BN）的作用：

理解批归一化（Batch Normalization，简称BN）的作用需要先了解一些深度学习训练中的一些问题：

1. **内部协变量偏移（Internal Covariate Shift）**：

   在深度网络中，每一层的输入分布会随着训练的进行而发生变化。这种现象称为内部协变量偏移。这使得后续的层需要不断地适应前一层的分布变化。

2. **加速训练速度**：

   BN 可以使得每一层的输入保持在一个稳定的分布上，从而加速训练的速度。这是因为网络不再需要等待前一层的参数稳定下来。

3. **防止梯度消失/爆炸**：

   BN 能够使得网络更加稳定，避免了梯度消失或爆炸的问题，尤其对于非常深的网络尤为重要。

4. **降低对初始化的依赖**：

   BN 使得对网络的初始化不再那么敏感，可以使用更大的学习率。

5. **提高泛化能力**：

   BN 有一定的正则化效果，可以提高网络的泛化能力。

批归一化的具体做法是在每一个小批量数据中，对每个特征的所有样本进行归一化，使得其均值接近于0，方差接近于1。这样可以保持每一层输入的稳定分布。

在神经网络中，批归一化通常在激活函数之前应用，即：

\[z = w \cdot x + b \]
\[ \text{BN}(z) = \gamma \cdot \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta\]

其中：

- \(z\) 是输入的线性变换。
- \(\mu\) 是当前批次的均值。
- \(\sigma\) 是当前批次的标准差。
- \(\gamma\) 是缩放因子。
- \(\beta\) 是平移因子。
- \(\epsilon\) 是一个小的常数，防止除零错误。

总的来说，批归一化有助于训练深度神经网络更加高效稳定，加速了模型的收敛速度，提升了模型的泛化能力。
"""

"""
Tip 3: 残差块（Residual Block）的维度变化
让我们一步一步分析这些操作如何改变输入张量的维度：

假设输入张量的维度为 `[b, ch, h, w]`，其中：

- `b` 表示批量大小（batch size），
- `ch` 表示通道数（channels），
- `h` 表示高度（height），
- `w` 表示宽度（width）。

1. `self.conv1(x)` 操作：
   - 输入维度：`[b, ch_in, h, w]`
   - 输出维度（假设卷积层将输入变换为 `ch_out` 个输出通道）：`[b, ch_out, h, w]`

2. `self.bn1(...)` 操作：
   - 输入维度：`[b, ch_out, h, w]`
   - 输出维度（批归一化不改变张量的形状）：`[b, ch_out, h, w]`

3. `F.relu(...)` 操作：
   - 输入维度：`[b, ch_out, h, w]`
   - 输出维度（ReLU 不改变张量形状）：`[b, ch_out, h, w]`

综合上述操作，整个指令 `x=F.relu(self.bn1(self.conv1(x)))` 的作用是将输入张量 `x` 经过一层卷积、批归一化和 ReLU 激活函数处理，然后得到与输入相同维度的输出。

总结一下，这个指令不会改变张量的维度，它只是对张量的值进行了相应的变换和激活。
"""

"""
Tip 4: 残差块（Residual Block）的设计思路:
让我再次尝试用一个更具体的例子来解释：

假设我们有一个网络，它的任务是将黑白照片变成彩色照片。我们用 \(F(x)\) 来表示网络学到的映射，其中 \(x\) 是黑白照片，\(F(x)\) 是彩色照片。

现在，假设我们给网络一个黑白照片 \(x\)，网络将其转换成了一张彩色照片 \(F(x)\)。然而，\(F(x)\) 与我们希望得到的最终彩色照片之间可能仍然存在一些差异，因为网络可能无法完美地还原所有的细节。

这时，我们引入了残差学习的思想：

1. 我们将原始的黑白照片 \(x\) 与网络生成的彩色照片 \(F(x)\) 相加。

2. 这个相加的结果 \(F(x) + x\) 就相当于在彩色照片 \(F(x)\) 的基础上，进行了一些微小的调整。

3. 这些微小的调整可能包括恢复一些细节、修正一些颜色偏差等。

通过这种方式，网络可以学会如何对 \(F(x)\) 进行一些微调，使得最终的输出 \(F(x) + x\) 更接近于我们希望得到的彩色照片。

换句话说，残差学习使得网络可以通过微调生成的结果，从而更好地逼近我们的目标。

希望这个例子可以让你更清晰地理解残差学习的概念。如果还有疑问，请随时提问！
"""

"""
Tip 5: `x.view(x.size(0), -1)` 的作用:

`x.view(x.size(0), -1)` 是在对输入张量 `x` 进行维度变换操作。具体来说，它将 `x` 从一个多维张量变换成了一个二维张量。

让我们通过一个例子来说明：

假设 `x` 的维度为 `[batch_size, channels, height, width]`，也就是一个四维张量。

```python
import torch

x = torch.randn(2, 3, 32, 32)  # 假设 x 的维度为 [2, 3, 32, 32]
```

现在，我们想将 `x` 变成一个二维张量，保留批大小（batch size）并将其余的维度合并在一起。这时，我们可以使用 `view` 操作：

```python
x = x.view(x.size(0), -1)
```

- `x.size(0)` 获取了 `x` 张量的第一个维度，也就是批大小。在上面的例子中，批大小为2。

- `-1` 表示 PyTorch 应该自动计算该维度的大小，以保证总的元素个数不变。在这里，它将自动计算合适的大小，以使得张量的总元素个数保持不变。

在上述例子中，`x` 将从一个四维张量变成一个二维张量，其形状为 `[2, 3072]`（因为 \(3 \times 32 \times 32 = 3072\)）。

这种变换通常在神经网络中的全连接层之前使用，因为全连接层通常接受二维输入。
"""