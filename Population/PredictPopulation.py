# batch_first = False, x:[seq_len, batch_size, input_size]   h:[num_layers, batch_size, hidden_size]
# batch_first = True , x:[batch_size, seq_len, input_size]   h：[batch_size, num_layers, hidden_size]
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 设置全局变量
input_size = 1
batch_size = 1
hidden_size = 10
output_size = 1
num_layers = 1
num_epochs = 3000
lr = 0.003
seq_len = 49


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):            # init为类的构造函数用于初始化实例的属性
        super(Net, self).__init__()                                     # 调用父类的构造函数确保父类被正确初始化
        self.rnn = nn.RNN(                                              # 初始化rnn层，输入特征维度，隐藏层特征维度，rnn层数，
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,                                      # 表示输入数据的形状为[batch_size,seq_len,input_size]
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)                     # 预防梯度消失
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):                                  # rnn前向传播函数
        out, hidden_prev = self.rnn(x, hidden_prev)                     # 调用rnn模型输出传入x1,h1,输出x2,h2进行循环
        # 将rnn的输出从三维转化为二维，以便下面的线性层处理，[seq_size, batch_size, hidden_size] => [seq_len * batch_size, hidden_size]
        out = out.view(-1, hidden_size)
        # 全连接层期望输入数据为二维张量，其形状是[batch_size * seq_len, hidden_size]
        out = self.linear(out)                                          # 应用线性层，输出形状为[batch_size * seq_len, output_size]
        out = out.unsqueeze(dim=0)                                      # 形状变为[1, batch_size * seq_len, output_size]
        return out, hidden_prev


def main():

    # 数据
    # 读取excel文件
    file_path = 'population_data.xlsx'
    df = pd.read_excel(file_path)
    years = df.iloc[:, 0].values  # 将第一列数据提取出来，转化为numpy数组
    population = df.iloc[:, 1].values  # 将第二列数据提取出来，转化为numpy数组

    # 对数据进行均值归一化
    data_mean = np.mean(population)  # 计算平均值
    data_std = np.std(population)  # 计算标准差
    population_normallized = (population - data_mean) / data_std  # 数据归一化

    # 创建输入和输出序列[49, 1, 1]
    x_data = torch.tensor(population_normallized[:-1], dtype=torch.float32).view(seq_len, batch_size, input_size)
    y_data = torch.tensor(population_normallized[1:], dtype=torch.float32).view(seq_len, batch_size, input_size)
    # 初始化模型，损失函数和优化器
    model = Net(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_loss = []               # 损失值列表

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        h0 = torch.zeros(num_layers, batch_size, hidden_size)
        output, hn = model(x_data, h0)  # 导入训练数据运行模型
        loss = criterion(output.view(-1, output_size), y_data.view(-1, output_size))  # 计算损失值
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch:{epoch}, Loss:{loss}')
            num_loss.append(loss.item())

    # 绘制损失函数
    plt.rcParams['font.sans-serif'] = ['SimHei']            # 指定包含中文字符的字体
    plt.plot(num_loss, 'r')
    plt.xlabel('训练次数')
    plt.ylabel('loss')
    plt.title('RNN损失函数下降曲线')
    plt.savefig('RNN损失函数下降曲线')
    plt.show()

    # 使用训练好的模型进行预测
    model.eval()          # 将模型设置为评估模式，该模式下模型不会进行梯度计算，二十仅仅使用前向传播来计算损失，训练完成后使用
    pre_data = y_data.clone()
    predictions = []

    h0 = torch.zeros(num_layers, batch_size, hidden_size)

    # 预测接下来五年的人口数据
    last_year = years[-1]

    for _ in range(5):
        output, h0 = model(pre_data[-1:], h0)                           # 对y_data进行切片，得到[1, batch_size, input_size]
        pre_data = torch.cat((pre_data, output[-1:]), dim=0)    # 切片后进行拼接，拼回去
        population = output.item() * data_std + data_mean               # 进行数据的逆归一化
        last_year += 1
        predictions.append("年份：{}".format(last_year))
        predictions.append("人口数：{}".format(population))

    print(predictions)


if __name__ == '__main__':
    main()
