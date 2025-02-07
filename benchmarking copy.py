import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 创建两个示例 DataFrame
data1 = {'ID': ['A', 'B', 'C', 'D'],
         'Value1': [10, 20, 15, 30]}
data2 = {'ID': ['A', 'B', 'C', 'D'],
         'Value2': [25, 35, 10, 20]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# 合并两个 DataFrame，假设 'ID' 是共同的索引
df = pd.merge(df1, df2, how='outer', on='ID')

# 设置柱状图的宽度
bar_width = 0.35

# 设置 X 轴的位置
index = np.arange(len(df))

# 绘制柱状图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制 Value1 的柱状图
bar1 = ax.bar(index, df['Value1'], bar_width, label='Value1', color='b')

# 绘制 Value2 的柱状图，偏移一个柱宽
bar2 = ax.bar(index + bar_width, df['Value2'], bar_width, label='Value2', color='g')

# 添加标签和标题
ax.set_xlabel('ID')
ax.set_ylabel('Values')
ax.set_title('柱状图：Value1 和 Value2 对比')
ax.set_xticks(index + bar_width / 2)  # 调整 X 轴标签的位置
ax.set_xticklabels(df['ID'])
ax.legend()

# 显示图形
plt.tight_layout()
plt.show()
