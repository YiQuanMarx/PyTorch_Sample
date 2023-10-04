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

    for epoch in range(10):
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