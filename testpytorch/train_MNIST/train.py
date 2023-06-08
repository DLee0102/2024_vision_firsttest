import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import argparse
from tensorboardX import SummaryWriter

# 定义命令行输入超参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        required=False,
        help="folder for saving predictions")
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        required=False,
        help="folder for saving predictions")
    return parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


# 定义残差网络模型
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32)
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(2, 2, 0)
        
    def forward(self, x):
        # Input: 512 * 1 * 28 * 28
        x1 = self.conv0(x)  # 512 * 16 * 28 * 28
        x1 = self.relu(x1)  # 512 * 16 * 28 * 28
        x1 = self.mp(x1)  # 512 * 16 * 14 * 14
        x2 = self.conv1(x1)  # 512 * 16 * 14 * 14
        x2 = self.relu(x1+x2)  # 512 * 16 * 14 * 14

        x2 = self.conv2(x2)  # 512 * 32 * 14 * 14
        x2 = x2.flatten(1)  # 1 * (512 * (32 * 14 * 14))
        x2 = self.fc_layer(x2)  # 512 * 1 * 10

        return x2

# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    # 将模型切换到训练模式
    model.train()
    loss_list = []
    # tqdm产生进度条
    for data, target in tqdm(train_loader):
        # 获取一个batch的数据和标签
        data, target = data.to(device), target.to(device)

        # 每个循环要梯度归零
        optimizer.zero_grad()
        # 输入模型
        output = model(data)
        # 数据和标签输入损失函数
        loss = criterion(output, target)
        # 反向传播
        loss.backward()
        # 梯度下降
        optimizer.step()

        loss_list.append(loss.item())
    # 每个epoch打印一次训练日志
    print('Train Epoch: {} Loss: {:.6f}'.format(
        epoch, sum(loss_list) / len(loss_list)))
    return sum(loss_list) / len(loss_list)

# 定义测试函数
def test(model, device, test_loader):
    # 将模型切换到测试模式
    model.eval()
    test_loss =0
    correct = 0
    testloss_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = criterion(output, target)
            testloss_list.append(test_loss.item())

            # 找到概率最大的下标
            pred = output.max(1, keepdim = True)[1]
            print(pred.shape)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss = sum(testloss_list) / len(testloss_list)
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100.* correct / len(test_loader.dataset)
            ))
    
    return test_loss, 100.* correct / len(test_loader.dataset)


if __name__ == '__main__':
    args = parse_args()

    # 下载训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train = True, download = True,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1037,), (0.3081,))
                ])),
    batch_size = args.batch_size, shuffle = True)

    # 测试集
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train = False, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])),
    batch_size = args.batch_size, shuffle = True)


    # 创建模型
    model = ResNet().to(DEVICE)
    # 定义Adam优化器
    optimizer = optim.Adam(model.parameters())
    # 定义损失函数（交叉熵）
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter()

    best_testloss = 100000
    # 训练
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model,  DEVICE, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, DEVICE, test_loader)
        if test_loss < best_testloss:
            print(f"[ Test | {epoch + 1:03d}/{args.epochs:03d} ] Cross_Enloss = {test_loss:.5f} -> best")
            print(f'Best model found at epoch {epoch+1}, saving model')
            torch.save(model.state_dict(), f'./model/model_best.ckpt')
            best_testloss = test_loss

        writer.add_scalar("CrossEn_Loss/train", train_loss, epoch)
        writer.add_scalar("CrossEn_Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)