import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import math
from math import sqrt
import time
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from matplotlib.font_manager import FontProperties
from pyod.models.cof import COF
from sklearn.preprocessing import StandardScaler
import openpyxl
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 设置中文字体
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12)

# 调用GPU，若无使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# 读取数据
df = pd.read_excel('第一次故障跑程序.xlsx', engine='openpyxl', header=0)

# 使用最后10列作为特征，第二列作为目标 读取数据左闭右开
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

# 计算训练集大小（整个数据集的50%）
train_size = int(len(X) * 0.5)

# 分割数据为训练集和测试集
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X, y

# 数据归一化（必须先用fit_transform(trainData)，之后再transform(testData)）
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 1).to(device)

# 创建TensorDataset和DataLoader，分批输入数据，防止数据量过大，GPU内存不够用
batch_size = 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the ViT model
class ViT(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_layers, output_dim, dropout=0.1):
        super(ViT, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, x):
        x = self.input_embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1])
        return x

# Model parameters
input_dim = X_train_scaled.shape[1]
hidden_dim = 64
nhead = 4
num_layers = 2
output_dim = 1

model = ViT(input_dim=input_dim, hidden_dim=hidden_dim, nhead=nhead, num_layers=num_layers, output_dim=output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 500
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluation
model.eval()
y_pred = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        y_pred.extend(outputs.cpu().numpy())
y_pred = np.array(y_pred)

# 反归一化
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test_scaled)

# 计算评价指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R2: {r2:.4f}')


# 训练集评估
train_pred = []
with torch.no_grad():
    for inputs, targets in train_loader:
        outputs = model(inputs)
        train_pred.extend(outputs.cpu().numpy())
train_pred = np.array(train_pred)
train_pred = scaler_y.inverse_transform(train_pred)
y_train = scaler_y.inverse_transform(y_train_scaled)

train_r2 = abs(r2_score(y_train, train_pred))
print(f'Train R2: {train_r2:.4f}')

# 绘制残差图
# 绘制残差图，将各点用线连接
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Time Series')
plt.title('Residual')
plt.xlabel('Sample points')
plt.ylabel('Bearing vibration/mm')
plt.savefig("LSTM-2.png", dpi=1080)  # 保存图片，分辨率为1080
plt.show()





# 绘图
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.savefig("VIT-1.png", dpi=1080)  # 保存图片，分辨率为600
plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch

# 假设 calculate_rmse 函数和模型预测部分的代码已经有了

# 定义 calculate_rmse 函数
def calculate_rmse(y_true, y_pred, window_size=30):
    window_rmse = [sqrt(mean_squared_error(y_true[i:i+window_size], y_pred[i:i+window_size]))
                   for i in range(len(y_true) - window_size + 1)]
    return window_rmse

# 计算训练集的窗口RMSE值
train_predictions = []
model.eval()
with torch.no_grad():
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        train_predictions.append(outputs.cpu())
train_predictions = torch.cat(train_predictions, dim=0)
train_predictions_np = train_predictions.numpy()
train_predictions_inv = scaler_y.inverse_transform(train_predictions_np.reshape(-1, 1))
y_train_inv = scaler_y.inverse_transform(y_train_scaled)

train_window_rmse = calculate_rmse(y_train_inv, train_predictions_inv)
threshold = max(train_window_rmse)
print(f'threshold: {threshold:.4f}')

# 计算测试集的窗口RMSE值
test_predictions = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        test_predictions.append(outputs.cpu())
test_predictions = torch.cat(test_predictions, dim=0)
test_predictions_np = test_predictions.numpy()
test_predictions_inv = scaler_y.inverse_transform(test_predictions_np.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test_scaled)

test_window_rmse = calculate_rmse(y_test_inv, test_predictions_inv)

# 绘制RMSE曲线
plt.figure(figsize=(14, 7))
plt.plot(test_window_rmse, label='Test Window RMSE')
plt.axhline(y=threshold, color='r', linestyle='--', label='Max Train RMSE')

plt.xlabel('Window Index')
plt.ylabel('RMSE')
plt.title('Test Window RMSE and Max Train RMSE')
plt.legend()
plt.savefig("VIT-3.png", dpi=1080)  # 保存图片，分辨率为600
plt.show()