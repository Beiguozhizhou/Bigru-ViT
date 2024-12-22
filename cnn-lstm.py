import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
from scipy.signal import savgol_filter
from matplotlib.font_manager import FontProperties
from pyod.models.cof import COF


# 设置中文字体
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12)
plt.rcParams['font.family'] = 'Times New Roman'
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
X_test, y_test = X[train_size:], y[train_size:]

# 数据归一化（必须先用fit_transform(trainData)，之后再transform(testData)）
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device).unsqueeze(1)  # Shape: [batch, 1, features]
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 1).to(device)

# 创建TensorDataset和DataLoader，分批输入数据，防止数据量过大，GPU内存不够用
batch_size = 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义CNN-LSTM模型
class CNN_LSTM(nn.Module):
    def __init__(self, cnn_params, lstm_params):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=cnn_params['in_channels'],
            out_channels=cnn_params['out_channels'],
            kernel_size=cnn_params['kernel_size'],
            padding=cnn_params['padding']
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=cnn_params['pool_size'])

        self.lstm = nn.LSTM(
            input_size=cnn_params['out_channels'],
            hidden_size=lstm_params['hidden_size'],
            num_layers=lstm_params['num_layers'],
            batch_first=True,
            bidirectional=lstm_params['bidirectional']
        )

        lstm_output_size = lstm_params['hidden_size'] * (2 if lstm_params['bidirectional'] else 1)
        self.fc = nn.Linear(lstm_output_size, 1)

    def forward(self, x):
        x = self.conv1(x)  # [batch, in_channels, features] -> [batch, out_channels, features]
        x = self.relu(x)
        x = self.maxpool(x)  # [batch, out_channels, features//pool_size]
        x = x.permute(0, 2, 1)  # [batch, features//pool_size, out_channels] for LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_length, hidden_size * num_directions]
        lstm_out = lstm_out[:, -1, :]  # Last time step
        out = self.fc(lstm_out)
        return out


# 超参数
cnn_params = {
    'in_channels': 1,
    'out_channels': 64,
    'kernel_size': 2,
    'padding': 1,
    'pool_size': 2
}

lstm_params = {
    'hidden_size': 128,
    'num_layers': 2,
    'bidirectional': True
}

# 创建CNN-LSTM模型
cnn_lstm_model = CNN_LSTM(cnn_params, lstm_params).to(device)

# Training parameters
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=learning_rate)
num_epochs = 1300


# 定义训练函数（可选，便于管理代码）
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True) as t:
            for X_batch, y_batch in t:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                t.set_postfix(loss=running_loss / len(train_loader))

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


# =======================
# 添加训练时间测量
# =======================

print("\n开始训练模型...")
start_time_train = time.time()  # 记录训练开始时间

# 训练模型
train_model(cnn_lstm_model, train_loader, criterion, optimizer, num_epochs)

end_time_train = time.time()  # 记录训练结束时间
training_time = end_time_train - start_time_train  # 计算训练时间
print(f"\n训练完成，总训练时间: {training_time:.2f} 秒")

# =======================
# 预测（测试）时间已经在原代码中测量
# =======================

# 在测试集上进行预测
cnn_lstm_model.eval()
predictions = []
start_time_test = time.time()  # 记录预测开始时间
with tqdm(test_loader, desc='Predicting', leave=True) as t:
    with torch.no_grad():
        for X_batch, _ in t:
            X_batch = X_batch.to(device)
            outputs = cnn_lstm_model(X_batch)
            predictions.append(outputs.cpu().numpy())  # 修改了这里，保存为numpy格式
end_time_test = time.time()  # 记录预测结束时间
prediction_time = end_time_test - start_time_test  # 计算预测时间
print(f'\n预测完成，总预测时间: {prediction_time:.2f} 秒')

# 将预测结果和真实结果反归一化
predictions_np = np.concatenate(predictions, axis=0)  # 合并所有预测结果
y_test_np = y_test_tensor.cpu().numpy()
predictions_inv = scaler_y.inverse_transform(predictions_np.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))  # 更正这里的反归一化

# 打印反归一化后的真实值和预测值
print('真实值 (反归一化后):', y_test_inv[:10].flatten())  # 打印前10个真实值
print('预测值 (反归一化后):', predictions_inv[:10].flatten())  # 打印前10个预测值


# 计算评估指标的函数，添加注释增强可读性
def calculate_metrics(y_true, y_pred):
    """
    计算多个评估指标，包括平均绝对误差（MAE）、均方误差（MSE）、
    均方根误差（RMSE）、平均绝对百分比误差（MAPE）和决定系数（R2）。
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    # 处理MAPE计算中分母可能为0的情况，避免运行时错误
    mask = y_true!= 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


# 计算并打印误差指标
metrics = calculate_metrics(y_test_inv, predictions_inv)
print(f"MAE: {metrics['MAE']:.4f}")
print(f"MSE: {metrics['MSE']:.4f}")
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"MAPE: {metrics['MAPE']:.4f}")
print(f"R2: {metrics['R2']:.4f}")


# 训练集评估
train_pred = []
cnn_lstm_model.eval()
with torch.no_grad():
    for inputs, targets in train_loader:
        outputs = cnn_lstm_model(inputs)
        train_pred.extend(outputs.cpu().numpy())
train_pred = np.array(train_pred)
train_pred = scaler_y.inverse_transform(train_pred)
y_train_inv = scaler_y.inverse_transform(y_train_scaled)

train_r2 = abs(r2_score(y_train_inv, train_pred))
print(f'Train R2: {train_r2:.4f}')

# 绘制残差图

Residuals1 = (y_test_inv - predictions_inv).flatten()
data_length = len(Residuals1)


def create_sliding_window(data, window_size):
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i + window_size])
    return np.array(X)


# 设置滑动窗口大小
window_size = 50

X_window = create_sliding_window(Residuals1, window_size)

# 将滑动窗口生成的三维数据展平为二维数据
n_samples = X_window.shape[0]
X_flat = X_window.reshape(n_samples, -1)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)



# 定义calculate_rmse函数，添加注释说明参数含义
def calculate_rmse(y_true, y_pred, window_size=30):
    """
    计算给定窗口大小下的均方根误差（RMSE）序列。
    :param y_true: 真实值数据
    :param y_pred: 预测值数据
    :param window_size: 滑动窗口大小，默认为30
    :return: 每个窗口对应的RMSE值列表
    """
    window_rmse = [sqrt(mean_squared_error(y_true[i:i + window_size], y_pred[i:i + window_size]))
                   for i in range(len(y_true) - window_size + 1)]
    return window_rmse


# 计算训练集的窗口RMSE值
train_predictions = []
cnn_lstm_model.eval()
with torch.no_grad():
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        outputs = cnn_lstm_model(X_batch)
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
cnn_lstm_model.eval()
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = cnn_lstm_model(X_batch)
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
plt.xlim(left=0)
plt.ylim(bottom=0)
# plt.title('Test Window RMSE and Max Train RMSE')
plt.legend()
plt.savefig("cnn-lstm-rmse.svg", format='svg')  # 保存为矢量图
plt.show()

# 绘制预测与真实值对比图
plt.figure(figsize=(14, 7))
plt.plot(y_test_inv, label='True Values', color='blue')
plt.plot(predictions_inv, label='Predictions', color='red')
plt.xlabel('Index', fontproperties=font)
plt.ylabel('Value', fontproperties=font)
plt.xlim(left=0)
# plt.title('Predictions vs True Values', fontproperties=font)
plt.legend(prop=font)
plt.savefig("cnn-lstm-predictions.svg", format='svg')  # 保存为矢量图
plt.show()

# 将反归一化的预测值和真实值保存到Excel文件中
df_results = pd.DataFrame({'True Values': y_test_inv.flatten(), 'Predictions': predictions_inv.flatten()})
df_results.to_excel('1predictions_vs_true_values.xlsx', index=False)



# 计算置信区间
def bootstrap_confidence_interval(y_true, y_pred, n_bootstraps=1000, ci=95):
    rng = np.random.default_rng()
    indices = np.arange(len(y_true))
    metrics_list = {'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'R2': []}

    for _ in range(n_bootstraps):
        sample_indices = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(y_true[sample_indices])) < 2:
            # Skip samples with no variation
            continue
        sample_y_true = y_true[sample_indices]
        sample_y_pred = y_pred[sample_indices]
        sample_metrics = calculate_metrics(sample_y_true, sample_y_pred)
        for key in metrics_list.keys():
            metrics_list[key].append(sample_metrics[key])

    confidence_intervals = {}
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    for key, values in metrics_list.items():
        lower = np.percentile(values, lower_percentile)
        upper = np.percentile(values, upper_percentile)
        confidence_intervals[key] = (lower, upper)

    return confidence_intervals


# 计算置信区间
confidence_intervals = bootstrap_confidence_interval(y_test_inv.flatten(), predictions_inv.flatten())

# 打印评估指标和置信区间
print("\nEvaluation Metrics with 95% Confidence Intervals:")
for metric, value in metrics.items():
    ci_lower, ci_upper = confidence_intervals.get(metric, (np.nan, np.nan))
    if metric == 'MAPE':
        print(f"{metric}: {value:.2f}% (95% CI: {ci_lower:.2f}% - {ci_upper:.2f}%)")
    else:
        print(f"{metric}: {value:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")

# 如果需要将置信区间保存或进一步使用，可以 add code here
