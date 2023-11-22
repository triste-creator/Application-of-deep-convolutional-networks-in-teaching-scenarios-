# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__  # 生成对象的类，或者生成类的元类的名称。
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# # 验证模型在测试集上的正确率
# def validate(model, dataset, batch_size):
#     val_loader = data.DataLoader(dataset, batch_size)
#     device = torch.device("cuda")
#     result, num = 0.0, 0
#     model.to(device)
#     with torch.no_grad():  # 不需要梯度
#         for images, labels in val_loader:
#             images.to(device)
#             labels.to(device)
#             pred = model.forward(images)  #
#             pred = np.argmax(pred.data.numpy(), axis=1)  # 获取模型预测概率最大的神经元分类作为预测结果
#             labels = labels.data.numpy()
#             result += np.sum((pred == labels))  # 模型预测分类与实际测试集分类相等则正确数量+1
#             num += len(images)
#         acc = result / num
#         return acc


# 重写Dataset类
class CNNDataset(data.Dataset):
    # 初始化
    def __init__(self, root):
        super(CNNDataset, self).__init__()
        self.root = root
        df_path = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    # 读取某幅图片，item为索引号
    def __getitem__(self, item):
        # 图像数据用于训练，需为tensor类型，label用numpy或list均可
        face = cv2.imread(self.root + '\\' + self.path[item])
        # 读取单通道灰度图
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化
        face_hist = cv2.equalizeHist(face_gray)

        face_normalized = face_hist.reshape(1, 48, 48) / 255.0
        face_tensor = torch.from_numpy(face_normalized)
        face_tensor = face_tensor.type('torch.FloatTensor')
        label = self.label[item]
        return face_tensor, label

    # 获取数据集样本个数
    def __len__(self):
        return self.path.shape[0]


class FaceCNN(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(FaceCNN, self).__init__()
        # 第一次卷积、池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.BatchNorm2d(num_features=64),  # 归一化
            nn.RReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大值池化
        )
        # 第二次卷积、池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 第三次卷积、池化
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 参数初始化
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 三次卷积池化后，全连接层进行扁平化处理
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y


def get_variable(x):
    return x.cuda()


def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    """
    模型训练
    """
    # Tensorboard可视化
    writer = SummaryWriter("logs3")
    # 载入数据并分割batch
    train_loader = DataLoader(train_dataset, batch_size)
    val_loader = DataLoader(val_dataset, batch_size)
    # 获取测试集长度
    num = len(val_dataset)
    print("模型测试所使用测试集长度为" + str(num))
    # 选择gpu训练，构建模型\
    device = torch.device("cuda")
    model = FaceCNN()
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        loss_rate = 0
        # 损失函数可视化
        board_loss_rate = 0
        # 轮次
        loss_num = 1
        # 模型训练
        model = model.to(device)  # 开始训练
        for images, labels in train_loader:
            # 将input转为cuda.tensor类型，统一Input和Output的数据类型一致
            images = images.to(device)
            labels = labels.to(device)
            board_loss_rate += loss_rate
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            output = model.forward(images)
            # 误差计算
            loss_rate = loss_function(output, labels)
            # 误差的反向传播
            loss_rate.backward()
            # 更新参数
            optimizer.step()
        # 打印每轮的损失
        print('经过 {} 轮训练后 , 损失速率是 : '.format(epoch + 1), loss_rate.item())  # 经过几轮（epochs）后，损失速率是多少
        writer.add_scalar("CrossEntropyLoss_rate", loss_rate, epoch + 1)  # 可视化损失函数
        loss_num = loss_num + 1
        writer.add_graph(model, input_to_model=images)

        # 验证模型准确率
        total_test_loss = 0  # 验证测试集上准确率
        total_acc = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if torch.cuda.is_available():
                    labels = labels.cuda()
                    images = images.cuda()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                total_test_loss = total_test_loss + loss.item()
                acc = (outputs.argmax(1) == labels).sum()
                total_acc = total_acc + acc
        # print("整体测试集上的Loss:{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_acc / num))
        # writer.add_scalar("模型在测试集中损失速率", total_test_loss, epoch + 1)
        writer.add_scalar("模型在测试集中准度", total_acc / num, epoch + 1)
        loss_num = loss_num + 1
    writer.close()
    return model


def main():
    # 数据集实例化(创建数据集)
    train_dataset = CNNDataset(root='cnn_train')
    val_dataset = CNNDataset(root='cnn_val')
    # 超参数可自行指定
    model = train(train_dataset, val_dataset, batch_size=128, epochs=300, learning_rate=0.1, wt_decay=0)
    torch.save(model, 'model_net.pkl')


if __name__ == '__main__':
    main()
