import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lenet5 import Lenet5
from resnet import ResNet18

def main():
    # one pic one time
    cifar_train = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ]))
    # multiple pic one time
    cifar_train = DataLoader(cifar_train, batch_size=32, shuffle=True)

    cifar_test = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ]))
    cifar_test = DataLoader(cifar_test, batch_size=32, shuffle=True)
    
    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    device=torch.device('cuda')
    # model=Lenet5().to(device)
    model=ResNet18().to(device)
    # model.load_state_dict(torch.load('model_9.pth'))
    criterion=nn.CrossEntropyLoss().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    # print(model)

    for epoch in range(4):
        model.train()
        # 加载保存的模型参数
        for batch_idx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            # [b, 3, 32, 32]
            # [b]
            logits=model(x) 
            loss=criterion(logits,label)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(epoch, 'loss:', loss.item())

        
        # test
        # Tip 6: model
        model.eval()
        with torch.no_grad():
            total_correct=0
            total_num=0
            for x,label in cifar_test:
                x,label=x.to(device),label.to(device)
                # [b,10]
                logits=model(x)
                # [b]
                pred=logits.argmax(dim=1)
                total_correct+=torch.eq(pred,label).float().sum().item()
                total_num+=x.size(0)
                
            acc=total_correct/total_num
            print('test acc:',torch.eq(pred,label).float().mean().item())
        
        # 在训练完成后保存模型参数
        torch.save(model.state_dict(), f'model_{epoch+9}.pth')




if __name__ == '__main__':
    main()

'''
Tip 6: 保存模型参数

在你提供的代码中，模型参数存储在`model`对象中。这里是一些说明：

1. `model` 是一个神经网络模型，它包含了各种层和参数。

2. `model.train()` 和 `model.eval()` 分别将模型设置为训练模式和评估模式。在训练模式下，模型会计算梯度以进行反向传播和参数更新。在评估模式下，模型不会计算梯度，这通常用于在测试集上进行性能评估。

3. 训练过程中，通过反向传播算法，模型的参数会被更新以最小化损失函数。

4. 你可以使用 `model.state_dict()` 将模型的参数以字典的形式获取到。这个字典包含了模型的所有可训练参数。

如果你想要将模型的参数保存到文件中，你可以使用以下代码：

```python
torch.save(model.state_dict(), 'model.pth')
```

这会将模型的参数保存到名为`model.pth`的文件中。如果需要重新加载模型，可以使用以下代码：

```python
model.load_state_dict(torch.load('model.pth'))
```

这会将之前保存的参数加载到模型中。

请注意，模型的参数是在训练过程中进行更新的，所以保存的参数将是训练后的状态。如果你想要保存中间训练状态的参数，可以在适当的时候进行保存。
'''