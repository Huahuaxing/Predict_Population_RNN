import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):            # init为类的构造函数用于初始化实例的属性
        super(Net, self).__init__()                                     # 调用父类的构造函数确保父类被正确初始化
        self.rnn = nn.RNN(                                              # 初始化rnn层，输入特征维度，隐藏层特征维度，rnn层数，
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,                                           # 表示输入数据的形状为[batch_size,seq_len,input_size]
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


def main():
    # 创建数据数组
    data = [0, 1, 2, 3]

    # 设置全局变量
    num_time_steps = 16
    seq_len = len(data)  # 用过去五年的数据预测下一年
    input_size = 1       #
    hidden_size = 10
    output_size = 1
    num_layers = 1
    lr = 0.01


if __name__ == '__main__':
    main()
