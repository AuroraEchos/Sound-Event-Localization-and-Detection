import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        裁剪掉最后的chomp_size个元素
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Residual block with dilation
        :param n_inputs: 输入通道数
        :param n_outputs: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param dilation: 膨胀系数
        :param padding: 填充
        :param dropout: dropout概率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1后，输出的size是(Batch, n_outputs, L_out)，L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1
        self.chomp1 = Chomp1d(padding) # 裁剪掉最后padding个元素, 保持输出的size和输入一致
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv2后，输出的size是(Batch, n_outputs, L_out)，L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1
        self.chomp2 = Chomp1d(padding) # 裁剪掉最后padding个元素, 保持输出的size和输入一致
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        初始化权重    
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: (Batch, n_inputs, L_in)
        :return: (Batch, n_outputs, L_out)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        :param num_inputs: 输入通道数
        :param num_channels: 每个TemporalBlock的输出通道数
        :param kernel_size: 卷积核大小
        :param dropout: dropout概率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            print(f"level_{i}: in_channels={in_channels}, out_channels={out_channels}, dilation_size={dilation_size}, kernel_size={kernel_size}, padding={(kernel_size-1) * dilation_size}")
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: (Batch, n_inputs, L_in)
        :return: (Batch, n_outputs, L_out)
        """
        return self.network(x)


# 1、裁剪模块
# 裁剪掉最后的chomp_size个元素

# 2、残差网络模块
# 一个残差网络模块包含两个卷积层，每个卷积层后面跟一个裁剪模块、ReLU激活函数、dropout层

# 3、TCN模块
# 一个TCN模块包含多个残差网络模块


# 进行一个简单的序列预测测试
# 1、构建一个TCN模型
# 2、构建一个简单的序列数据
# 3、训练模型
# 4、预测序列
""" import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam

# 1、构建一个TCN模型
num_inputs = 1
num_channels = [16, 16, 16, 16, 16, 16, 16, 16]
kernel_size = 2
dropout = 0.2
model = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
print(model)

# 2、构建一个简单的序列数据
seq_len = 100
x = np.linspace(0, 10, seq_len)
y = np.sin(x)


# 3、训练模型
x = torch.tensor(y[np.newaxis, np.newaxis, :], dtype=torch.float32)
y = torch.tensor(y[np.newaxis, np.newaxis, :], dtype=torch.float32)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)
model.train()
for i in range(1000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(f"Epoch {i}, Loss: {loss.item()}")
    loss.backward()
    optimizer.step()

# 打印原始序列和预测序列
model.eval()
y_pred = model(x)
plt.plot(x[0, 0, :].numpy(), label='original')
plt.plot(y_pred[0, 0, :].detach().numpy(), label='predict')
plt.legend()
plt.show() """



