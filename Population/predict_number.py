import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 超参数
seq_len = 5
input_size = 1
hidden_size = 20
num_layers = 2
batch_size = 1
num_epochs = 1000

# 标准化数据
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
data_mean = data.mean()
data_std = data.std()
data = (data - data_mean) / data_std
data = torch.tensor(data).view(-1, 1, 1).float()

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 输出层将隐藏状态映射到预测值

    def forward(self, x, h):
        out, (h_n, c_n) = self.lstm(x, h)
        out = self.fc(out)
        return out, (h_n, c_n)

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(num_epochs):
    for i in range(len(data) - seq_len):
        input_seq = data[i:i + seq_len].view(1, seq_len, input_size)  # 形状调整为 (batch_size, seq_len, input_size)
        target_seq = data[i + 1:i + 1 + seq_len].view(1, seq_len, input_size)  # 形状调整为 (batch_size, seq_len, input_size)

        h0 = torch.zeros(num_layers, batch_size, hidden_size)
        c0 = torch.zeros(num_layers, batch_size, hidden_size)

        output, _ = model(input_seq, (h0, c0))

        loss = criterion(output, target_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 使用模型进行预测
model.eval()
with torch.no_grad():
    test_input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    test_input = (test_input - data_mean) / data_std
    test_input = test_input.view(1, seq_len, input_size).float()

    h0 = torch.zeros(num_layers, batch_size, hidden_size)
    c0 = torch.zeros(num_layers, batch_size, hidden_size)

    predictions = []
    input_seq = test_input
    for _ in range(5):
        output, (h0, c0) = model(input_seq, (h0, c0))
        next_pred = output[0, -1, 0].item()  # 取出最后一个时间步的预测值
        predictions.append(next_pred)

        next_input = torch.tensor([[next_pred]])
        input_seq = torch.cat((input_seq[:, 1:, :], next_input.unsqueeze(0)), dim=1)

    # 将预测值反标准化
    predictions = np.array(predictions)
    predictions = predictions * data_std + data_mean

    print("Predicted next 5 values: ", predictions)
