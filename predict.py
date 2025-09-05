# 导入数据，初始化RNN模型，取前四十五年的数据对后五年的数据进行预测，进行模型训练，最后预测
# 用vscode写代码运行前一定要先保存，没有pycharm好用

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

input_size = 1          # 导入数据的特征维度
hidden_size = 10        # 隐藏层的特征维度
output_size = 1         # 导出数据的特征维度
batch_size = 1          # 同时一次性处理的批次
seq_len = 45            # 导入数据的长度，也是rnn模型的时间步
num_layers = 1          # 隐藏层的层数
lr = 0.001              # 学习率
num_epochs = 2000       # 训练次数

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):    # 初始化类的实例
        super(Net, self).__init__()                             # 初始化父类
        self.rnn = nn.RNN(                                      # 创建一个rnn模型，导入参数
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        # 前向传播函数
        out, hidden_prev = self.rnn(x, hidden_prev)             # 调用模型输入数据得到输出
        # rnn模型输出的out是三维的张量，需要转化为二维数据，再导入线性层处理
        out = out.view(-1, hidden_size)  # 处理后[seq_size, batch_size, input_size]转变为[seq_size*batch_size, input_size]
        
        out = self.linear(out)          # 线性层处理后形状为[seq_len*batch_size, output_size]
        out = out.unsqueeze(dim=0)      # 恢复三维张量[1, seq_len*batch_size, output_size]
        return out, hidden_prev
    

# 读取excel表格并输出为数组
def get_data():
    df = pd.read_excel('population.xlsx')       # 将excel读取成DataFrame形式
    population = df.iloc[:, 1].values           # 将第二列数据提取出来，变成numpy数组

    # 对数据进行均值归一化
    data_mean = np.mean(population)
    data_std = np.std(population)
    population_normallized = (population - data_mean) / data_std

    # 创建输入序列张量，格式为[45, 1, 1]，最后五年的数据用来验证
    x_data = torch.tensor(population_normallized[:-5], dtype=torch.float32).view(seq_len, batch_size, input_size)
    # 创建验证序列张量，格式与x_data相同
    y_data = torch.tensor(population_normallized[1:-4], dtype=torch.float32).view(seq_len, batch_size, input_size)

    return x_data, y_data, data_mean, data_std


# 模型训练函数
def train(x_data, y_data):
    num_loss = []
    model = Net(input_size, hidden_size, num_layers)        # 生成一个Net类的实例
    criterion = nn.MSELoss()                                # 定义损失函数计算器
    optimizer = optim.Adam(model.parameters(), lr=lr)       # 定义优化器
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        h0 = torch.zeros(num_layers, batch_size, hidden_size)       # 定义初始隐藏张量
        output, hn = model(x_data, h0)
        loss = criterion(output.view(-1, output_size), y_data.view(-1, output_size))    # 计算损失值
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch:{epoch}, Loss{loss}')
            num_loss.append(loss.item())

    model.eval()

    # 绘制损失函数
    plt.rcParams['font.sans-serif'] = ['SimHei']    # 指定包含中文字符的字体
    plt.plot(num_loss, 'r')
    plt.xlabel('训练次数')
    plt.ylabel('损失值')
    plt.title('RNN损失函数下降曲线')
    plt.savefig('RNN损失函数下降曲线')



def predict(y_data, data_mean, data_std):
    model = Net(input_size, hidden_size, num_layers)

    prediction = []

    # 预测下五年的数据
    for i in range(5):

        h0 = torch.zeros(num_layers, batch_size, hidden_size)

        output, h0 = model(y_data[-1:], h0)  # x_data[45, 1, 1]，预测时只使用数据的最后一项进行输入，即[1, 1, 1]

        y_data = torch.cat((y_data, output[-1:]), dim=0)            # 将预测的下一年数据接到y_data的最后面，当作下一次预测的输入

        # output是一个张量, data_std是一个numpy数组
        population = output.item() * data_std + data_mean             # 对数据进行反归一化
        prediction.append(population)

    return prediction


def main():
    x_data, y_data, data_mean, data_std = get_data()

    train(x_data, y_data)

    population = predict(y_data, data_mean, data_std)

    print(population)


if __name__ == '__main__':
    # 预测完成，发现数据比较偏离真实情况，可以适当调整加大学习率，训练次数，有利于结果的正确性，拜拜！ 
    main()