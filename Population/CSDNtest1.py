# 在 PyTorch 的 RNN 模块中，默认情况下使用 tanh 作为激活函数。当你调用 nn.RNN 时，每个时间步的计算会包括应用 tanh 函数。
# pytorch张量的形状为(长度, 批次, 维度）
import torch
import torch.nn as nn

test = torch.tensor([[[1.0]],
                     [[2.0]],
                     [[3.0]],
                     [[4.0]],
                     [[5.0]]])
input_size2 = 1
seq_len2 = 5

# rnn模块
input_size = 100    # 输入数据编码的维度
hidden_size = 20    # 隐含层维度
num_layer = 4       # 隐含层层数

rnn = nn.RNN(input_size=input_size2, hidden_size=hidden_size, num_layers=num_layer)
print('rnn：', rnn)

# 输入的表示
seq_len = 10    # 输入长度
batch_size = 1  # 批次大小

x = torch.zeros(seq_len, batch_size, input_size)        # 输入数据，长度、批次、维度
h0 = torch.zeros(num_layer, batch_size, hidden_size)    # 初始隐藏状态 隐含层层数、批次、隐含层维度
# print(x)
# print(x.shape)

out, h = rnn(test, h0)     # 输出数据

# 打印输出和隐藏状态的尺寸
# print('out.shape：', out.shape)      # out：rnn在每个时间步的输出，长度，批次，隐含层维度
# print('h.shape：', h.shape)          # h：最后一个时间步的隐藏状态，隐含层层数，批次，隐含层维度
print(out)
