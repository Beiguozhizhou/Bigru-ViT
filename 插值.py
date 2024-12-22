import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取Excel文件
filepath = r'第一次故障P12.xlsx'  # 文件路径
data = pd.read_excel(filepath)

# 2. 数据预处理
print("原始数据预览：")
print(data.head())

# 将第一列设为时间索引
data.rename(columns={data.columns[0]: 'Time'}, inplace=True)  # 重命名时间列为 'Time'
data['Time'] = pd.to_datetime(data['Time'])  # 将时间列转换为日期时间格式
data = data.set_index('Time')  # 将时间列设为索引

# 3. 重新采样到1分钟间隔，并使用线性插值
data_resampled = data.resample('1T').interpolate(method='linear')  # 使用线性插值方法

# 4. 可视化结果
plt.figure(figsize=(10, 6))
for column in data.columns:
    plt.plot(data.index, data[column], 'o', label=f'Original Data - {column}')  # 原始数据点
    plt.plot(data_resampled.index, data_resampled[column], '-', label=f'Linear Interpolation - {column}')  # 插值结果

plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Linear Interpolation: Resampling Data to 1-Minute Interval')
plt.legend()
plt.grid()
plt.show()

# 5. 保存处理后的数据
output_path = r'处理后的数据_线性插值.xlsx'
data_resampled.to_excel(output_path)
print(f"\n处理后的数据已保存到：{output_path}")
