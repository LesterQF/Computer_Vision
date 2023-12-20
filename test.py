import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.Resnet import resnet
from model.SE_ResNet import se_resnet
from model.VGG_19 import VGG
from model.MobileNet import Mobile


def main():
    # 设备选择(CUDA OR CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据路径
    data_path = 'dataset'
    assert os.path.exists(data_path), "{} does not exist".format(data_path)

    # 数据预处理
    data_transform = {
        'test': transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

    # 提取数据
    test_dataset = datasets.CIFAR10(root="dataset/test",train=False,transform=data_transform['test'],download=False)
    test_num = len(test_dataset)
    print("The number of validate data is: ", test_num)

    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

    # net = se_resnet(in_channels=3)
    # net=resnet(in_channels=3)
    # net = resnet(in_channels=3).to(device)
    # conv_arch = ((2, 64), (2, 128), (3, 256),(3, 512),(3, 512))
    # net=VGG(conv_arch,3,10,True).to(device)
    net=Mobile()
    net.to(device)

    # 设置模型参数文件的路径
    pth_file_path = 'result/MobileNet.pth'

    # 初始化变量
    all_y_true = []  # 收集所有批次的真实标签
    all_y_pred = []  # 收集所有批次的预测标签
    loss_function = nn.CrossEntropyLoss()

    # 检查文件是否存在
    if os.path.isfile(pth_file_path):
        # 加载模型参数
        net.load_state_dict(torch.load(pth_file_path))
        print("Model parameters loaded successfully from {}".format(pth_file_path))

        # 测试集准确率
        net.eval()
        acc_num = 0.0
        sample_num = 0
        with torch.no_grad():
            for data, labels in test_loader:
                outputs = net(data.to(device))
                pred_class = torch.max(outputs, dim=1)[1]
                acc_num += torch.eq(pred_class, labels.to(device)).sum().item()
                sample_num += data.size(0)
                loss = loss_function(outputs, labels.to(device))
                all_y_pred.extend(pred_class.cpu().numpy())
                all_y_true.extend(labels.cpu().numpy())

            test_acc = (acc_num / sample_num) * 100
            print("Test dataset: The accuracy is {:.2f}% and the loss is {:.4f}".format(test_acc, loss))
    else:
        print("Model parameters file not found at {}".format(pth_file_path))
        return  # 如果文件不存在，则结束函数

    print("Finished")


if __name__ == '__main__':
    main()