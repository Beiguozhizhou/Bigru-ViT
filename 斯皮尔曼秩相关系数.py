import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12)
plt.rcParams['font.family'] = 'Times New Roman'
# 加载数据
filepath = r'第一次故障.xlsx'
data = pd.read_excel(filepath)

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 将第一列设为目标变量
target_variable = df.iloc[:, 0]

# 计算目标变量与其余变量的斯皮尔曼秩相关系数
spearman_corr = df.corr(method='spearman')

# 提取目标变量与其他变量的相关系数
target_corr = spearman_corr.iloc[0, 1:]

# 创建渐变色调的调色板
colors = sns.color_palette("coolwarm", n_colors=len(target_corr))

# 创建条形图进行可视化
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x=target_corr.index, y=target_corr.values, palette=colors)

# 在每个条形图上方显示具体的相关系数数值，并根据条件改变字体颜色
for index, value in enumerate(target_corr.values):
    color = 'red' if value >= 0.74 else 'black'
    plt.text(index, value + np.sign(value) * 0.02, f'{value:.2f}', ha='center', color=color,fontproperties=font)

# 修改横轴标签字体颜色
labels = bar_plot.get_xticklabels()
for label, value in zip(labels, target_corr.values):
    if value >= 0.74:
        label.set_color('red')

bar_plot.set_xticklabels(labels)

# 去掉上边框和右边框
sns.despine(top=True, right=True)

# 添加标签和标题
plt.title('Spearman rank correlation coefficient with N1',fontproperties=font)
plt.xticks(rotation=90,fontproperties=font)
plt.ylabel('Spearman Rank Correlation Coefficient',fontproperties=font)

# 显示图表
plt.tight_layout()
plt.show()
