# 读取csv文件

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filepath = r'第一次故障.xlsx'
data = pd.read_excel(filepath)
df = pd.DataFrame(data)


# 计算出相关系数并输出，这里选择的是皮尔逊相关系数
cor = data.corr(method='pearson')
print(cor)  # 输出相关系数

rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
sns.set(font_scale=0.25,rc=rc)  # 设置字体大小

sns.heatmap(cor,
            annot=True,  # 显示相关系数的数据
            center=0.5,  # 居中
            fmt='.2f',  # 只显示两位小数
            linewidth=0.1,  # 设置每个单元格的距离
            linecolor='blue',  # 设置间距线的颜色
            vmin=0, vmax=1,  # 设置数值最小值和最大值
            xticklabels=True, yticklabels=True,  # 显示x轴和y轴
            square=True,  # 每个方格都是正方形
            cbar=True,  # 绘制颜色条
            cmap='coolwarm_r',  # 设置热力图颜色
            )
plt.savefig("第一次故障相关热力图.png",dpi=1080)#保存图片，分辨率为600
plt.show() #显示图片
