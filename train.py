import os
import sys

import torch
from sympy.combinatorics import Subset
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Subset

from model.Resnet import resnet
from model.VGG_19 import VGG
from model.SE_ResNet import se_resnet
from model.MobileNet import Mobile
import matplotlib.pyplot as plt


def main():
    #选择训练设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #进行数据预处理
    transform ={
        'train':transforms.Compose([
            transforms.Pad(4),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.RandomCrop(32, padding=4),
        ])
            ,
        'val':transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

    #载入数据集
    train_dataset=datasets.CIFAR10(root="dataset/train",train=True,transform=transform['train'],download=True)
    test_dataset=datasets.CIFAR10(root="dataset/test",train=False,transform=transform['train'],download=True)

    # 确定您想要的训练集和验证集的大小，例如 20%
    val_size = int(0.2 * len(test_dataset))
    train_size = int(0.4 * len(train_dataset))

    # 生成随机索引
    val_indices = torch.randperm(len(test_dataset)).tolist()
    train_indices = torch.randperm(len(train_dataset)).tolist()

    # 使用前 20% 的索引来创建子集
    val_dataset = Subset(test_dataset, val_indices[:val_size])
    train_dataset = Subset(train_dataset, train_indices[:train_size])

    # 设置超参数
    lr, epochs,batch_size=0.01,80,32
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)
    val_loader=DataLoader(val_dataset,shuffle=True,num_workers=0,drop_last=True)

    #生成模型对象
    # net = resnet(in_channels=3).to(device)
    # conv_arch = ((2, 64), (2, 128), (3, 256),(3, 512),(3, 512))
    # # net=VGG(conv_arch,3,10,True).to(device)
    # net=se_resnet(in_channels=3).to(device)
    net=Mobile().to(device)

    #损失函数定定义 优化器定义
    loss_function=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=lr)

    # 绘图横坐标和纵坐标列表
    epochs_list = []
    train_loss_list = []

    #设置训练指标
    best_acc = 0.0# 初始化验证集上最好的准确率，以便后面用该指标筛选模型最优参数。

    for epoch in range(epochs):
        net.train()#设置为训练模式 梯度会进行更新
        num_acc = torch.zeros(1).to(device)
        sample_num = 0
        train_bar=tqdm(train_loader, file=sys.stdout, ncols=100)
        for data in train_bar:
            images,labels=data
            sample_num+=images.shape[0]
            optimizer.zero_grad()
            output=net(images.to(device))
            predict_class=torch.max(output,dim=1)[1]
            num_acc+=torch.eq(labels.to(device),predict_class).sum()

            loss=loss_function(output,labels.to(device))#求损失
            loss.backward()
            optimizer.step()
            # print statistics
            train_acc=(num_acc.item()/sample_num)*100
            # .desc是进度条tqdm中的成员变量，作用是描述信息
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} train_acc: {:.3f}% ".format(epoch + 1, epochs, loss, train_acc)

        train_loss_list.append(loss.item())
        epochs_list.append(epoch + 1)
        net.eval()
        with torch.no_grad():
            num_val=0
            val_acc_num=0
            for data in val_loader:
                images,labels=data
                num_val += images.shape[0]
                output=net(images.to(device))
                predict_class=torch.max(output,dim=1)[1]
                val_acc_num+=torch.eq(predict_class,labels.to(device)).item()
                val_acc=(val_acc_num/num_val)*100

        if val_acc>=best_acc:
            best_acc=val_acc
            torch.save(net.state_dict(), "result/MobileNet.pth")
        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' % (epoch + 1, loss, train_acc, val_acc))
        train_acc=0.0
    print('Finished Training')



    # 绘制图像
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_loss_list, marker='o', color='b', linestyle='-', label='Train Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()





