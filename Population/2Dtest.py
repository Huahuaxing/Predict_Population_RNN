# batch_first = True , x:[batch_size, seq_len, input_size]   h：[batch_size, num_layers, hidden_size]
# batch_first = False, x:[input_size, seq_len, batch_size]   h:[num_layers, batch_size, hidden_size]

import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl

# 设置全局变量
num_time_steps = 16
seq_len = 5         # 用过去五年的数据预测下一年
input_size = 1      # 二维数据（年份，总人口）
hidden_size = 10
output_size = 1
num_layers = 1
lr = 0.01
num_epochs = 3000

# 读取excel文件
file_path = 'population_data.xlsx'
df = pd.read_excel(file_path)
years = df.iloc[:, 0].values                                        # 将第一列数据提取出来，转化为numpy数组
population = df.iloc[:, 1].values                                   # 将第二列数据提取出来，转化为numpy数组

# 对数据进行均值归一化
data_mean = np.mean(population)                                     # 计算平均值
data_std = np.std(population)                                       # 计算标准差
population_normallized = (population - data_mean) / data_std        # 数据归一化

# 创建输入和输出序列
x_data = []
y_data = []

for i in range(len(population_normallized) - seq_len):
    x_data.append(population_normallized[i:i+seq_len])
    y_data.append(population_normallized[i+seq_len])

x_data = np.array(x_data).reshape(-1, seq_len, input_size)
y_data = np.array(y_data).reshape(-1, output_size)


# 定义rnn类
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)         # 预防梯度消失
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        # [b, seq, h]
        out = out.view(-1, hidden_size)
        out = self.linear(out)      # [seq, h] => [seq, 2]
        out = out.unsqueeze(dim=0)  # => [1, seq, h]
        return out, hidden_prev


# 训练模型
def train_rnn(x_data, y_data):                                    # 训练函数，传入数据data
    model = Net(input_size, hidden_size, num_layers)    # 创建一个类的实例，即一个模型
    print('model:\n', model)                            # 打印模型检验
    criterion = nn.MSELoss()                            # 定义损失函数为均方误差函数(MSE)
    optimizer = optim.Adam(model.parameters(), lr)      # 定义优化器为Adam优化算法，传入模型的参数和定义好的学习率
    # 初始化隐藏状态
    hidden_prev = torch.zeros(num_layers, 1, hidden_size)        # 定义隐藏输入h，层数，批次，维度
    num_loss = []                                       # 定义一个空列表存放每次迭代产生的损失值

    for epoch in range(num_epochs):
        for i in range(len(x_data)):
            x = torch.tensor(x_data[i:i + 1]).float()
            y = torch.tensor(y_data[i:i + 1]).float()
            output, hidden_prev = model(x, hidden_prev)
            hidden_prev = hidden_prev.detach()

            loss = criterion(output.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print('iteration:{} loss:{}'.format(epoch, loss.item()))
            num_loss.append(loss.item())

    print(num_loss)
    # 绘制损失函数
    plt.plot(num_loss, 'r')
    plt.xlabel('训练次数')
    plt.ylabel('loss')
    plt.title('RNN损失函数下降曲线')
    plt.show()

    return hidden_prev, model

# def rnn_pre(model, data, hidden_prev):


if __name__ == "__main__":
    hidden_prev, model = train_rnn(x_data, y_data)
