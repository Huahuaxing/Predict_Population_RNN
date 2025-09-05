# 此代码由gpt给出
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# 读取Excel数据并重命名列
data = pd.read_excel('E:/Project/Population/population1.xlsx')
data.columns = ['Year', 'Population']

# 数据预处理
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Population']].values.astype(float))


# 创建自定义Dataset
class PopulationDataset(Dataset):
    def __init__(self, data, seq_length=1):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 创建数据集和数据加载器
seq_length = 1
dataset = PopulationDataset(data_scaled, seq_length)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


# 定义RNN模型
class PopulationRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super(PopulationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# 初始化模型、损失函数和优化器
model = PopulationRNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 进行预测
model.eval()
predictions = []
input_seq = torch.tensor(data_scaled[-seq_length:], dtype=torch.float32).unsqueeze(0)  # 最后的一个序列

with torch.no_grad():
    for _ in range(10):  # 预测未来10年的数据
        predicted = model(input_seq)
        predictions.append(predicted.item())
        input_seq = torch.cat((input_seq[:, :, :], predicted.unsqueeze(0)), dim=1)  # 将维度调整为(1, seq_length, 1)

# 将预测结果逆缩放回原始范围
predictions = scaler.inverse_transform([[p] for p in predictions])

# 将预测结果与年份结合
years = list(range(2018, 2028))  # 从2018年到2027年，因为数据已经包含到2017年
predictions_with_years = list(zip(years, predictions))

# 打印预测结果
print("Predicted populations:")
for year, population in predictions_with_years:
    print(f"Year: {year}, Predicted Population: {population[0]}")

# 注意：这个例子假设数据是按年份排序的，并且没有做复杂的验证和测试分割，适合用于简单的演示目的。
